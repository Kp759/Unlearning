#!/usr/bin/env python3
"""Run four TOFU unlearning settings in one self-contained script.

Settings:
  - full_all_tokens
  - full_selective_tokens
  - emb_lm_all_tokens
  - emb_lm_selective_tokens
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODES = ["full_all_tokens", "full_selective_tokens", "emb_lm_all_tokens", "emb_lm_selective_tokens"]
DEFAULT_MODEL_DIR = PROJECT_DIR / "outputs" / "finetuned_model_3B_instruct"
COMMON_FUNCTION_TOKENS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "for",
    "with", "by", "from", "at", "as", "is", "are", "was", "were", "be", "been", "being", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "them", "his", "her", "their",
    "i", "you", "we", "not", "no", "yes", "do", "does", "did", "has", "have", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "also", "there", "here",
}


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str


@dataclass
class LossResult:
    loss: torch.Tensor
    contributing_tokens: int
    fallback_examples: int


def dtype_from_arg(dtype: str) -> torch.dtype:
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def looks_like_hf_repo_id(path: str) -> bool:
    if path.startswith(("/", "./", "../", "~")):
        return False
    if Path(path).exists():
        return True
    parts = path.split("/")
    if parts[0] in {"outputs", "results", "checkpoints", "data", "scripts"}:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)?", path))


def validate_model_path(model_path: str | None) -> None:
    if not model_path:
        raise FileNotFoundError(
            "Default model path outputs/finetuned_model_3B_instruct does not exist; pass --model-path explicitly."
        )
    if Path(model_path).exists() or looks_like_hf_repo_id(model_path):
        return
    raise FileNotFoundError(
        f"model_path does not exist and does not look like a Hugging Face repo id: {model_path}"
    )


def load_tofu_examples(split: str, limit: int, seed: int) -> List[Example]:
    rows = list(load_dataset("locuslab/TOFU", name=split, split="train"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples: List[Example] = []
    for row in rows[:limit]:
        question = str(row["question"])
        answer = " " + str(row["answer"]).strip()
        examples.append(Example(prompt=f"Question: {question}\nAnswer:", answer=answer))
    return examples


def special_token_ids(tokenizer) -> set[int]:
    return {tid for tid in tokenizer.all_special_ids if tid is not None}


def answer_token_ids(tokenizer, answer: str) -> List[int]:
    return tokenizer.encode(answer, add_special_tokens=False)


def is_bad_selective_token(tokenizer, token_id: int, special_ids: set[int]) -> bool:
    if token_id in special_ids:
        return True
    text = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
    if not text:
        return True
    if all(ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for ch in text):
        return True
    return text in COMMON_FUNCTION_TOKENS


def build_token_sets(tokenizer, forget: Sequence[Example], retain: Sequence[Example], top_k: int) -> Tuple[List[int], List[int]]:
    specials = special_token_ids(tokenizer)
    all_forget = sorted(
        {
            tid
            for ex in forget
            for tid in answer_token_ids(tokenizer, ex.answer)
            if tid not in specials
        }
    )

    forget_df: Counter[int] = Counter()
    retain_df: Counter[int] = Counter()
    for ex in forget:
        forget_df.update(set(tid for tid in answer_token_ids(tokenizer, ex.answer) if tid not in specials))
    for ex in retain:
        retain_df.update(set(tid for tid in answer_token_ids(tokenizer, ex.answer) if tid not in specials))

    n_forget = max(len(forget), 1)
    n_retain = max(len(retain), 1)
    scored: List[Tuple[float, int]] = []
    for tid, df in forget_df.items():
        if is_bad_selective_token(tokenizer, tid, specials):
            continue
        score = (df / n_forget) / ((retain_df.get(tid, 0) / n_retain) + 1e-6)
        scored.append((score, tid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    selective = [tid for _, tid in scored[:top_k]]
    return all_forget, selective


def encode_batch(tokenizer, examples: Sequence[Example], device: torch.device) -> Dict[str, torch.Tensor]:
    eos = tokenizer.eos_token or ""
    input_rows: List[List[int]] = []
    answer_masks: List[List[bool]] = []
    for i, ex in enumerate(examples):
        full = ex.prompt + ex.answer + (eos if eos and not ex.answer.endswith(eos) else "")
        answer_start = len(ex.prompt)
        if getattr(tokenizer, "is_fast", False):
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True)
            ids = enc.input_ids
            offsets = enc.get("offset_mapping")
            mask = [bool(end > answer_start) for _, end in offsets]
        else:
            ids = tokenizer(full, add_special_tokens=True).input_ids
            prompt_len = len(tokenizer(ex.prompt, add_special_tokens=True).input_ids)
            mask = [j >= prompt_len for j in range(len(ids))]
        mask = [m and tid != tokenizer.pad_token_id and tid != tokenizer.bos_token_id for m, tid in zip(mask, ids)]
        if not any(mask):
            raise ValueError(f"Zero answer tokens for TOFU example {i}: prompt={ex.prompt!r}, answer={ex.answer!r}")
        input_rows.append(ids)
        answer_masks.append(mask)

    pad = tokenizer.pad_token_id
    if pad is None:
        raise ValueError("Tokenizer has no pad_token_id")
    max_len = max(len(row) for row in input_rows)
    padded_ids, attention, padded_masks = [], [], []
    for ids, mask in zip(input_rows, answer_masks):
        pad_n = max_len - len(ids)
        padded_ids.append(ids + [pad] * pad_n)
        attention.append([1] * len(ids) + [0] * pad_n)
        padded_masks.append(mask + [False] * pad_n)
    input_ids = torch.tensor(padded_ids, dtype=torch.long, device=device)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
        "labels": input_ids.clone(),
        "answer_mask": torch.tensor(padded_masks, dtype=torch.bool, device=device),
    }


def selected_mask(batch: Dict[str, torch.Tensor], selected_ids: set[int] | None) -> Tuple[torch.Tensor, int]:
    answer_mask = batch["answer_mask"]
    if selected_ids is None:
        return answer_mask, 0
    mask = torch.zeros_like(answer_mask)
    for tid in selected_ids:
        mask |= batch["labels"].eq(tid)
    mask &= answer_mask
    empty = mask.sum(dim=1).eq(0)
    fallback = int(empty.sum().item())
    if fallback:
        mask = mask.clone()
        mask[empty] = answer_mask[empty]
    if mask.sum().item() == 0:
        raise ValueError("Selected-token mask is empty for the whole batch after fallback")
    return mask, fallback


def answer_ce(model, batch: Dict[str, torch.Tensor], selected_ids: set[int] | None) -> LossResult:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask, fallback = selected_mask(batch, selected_ids)
    mask = mask[:, 1:]
    if mask.sum().item() == 0:
        raise ValueError("No answer tokens contribute to CE after shifting")
    logp = F.log_softmax(logits, dim=-1)
    nll = -logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    return LossResult(loss=(nll * mask.float()).sum() / mask.float().sum(), contributing_tokens=int(mask.sum().item()), fallback_examples=fallback)


def retain_kl(student, teacher, batch: Dict[str, torch.Tensor], teacher_batch: Dict[str, torch.Tensor], selected_ids: set[int] | None) -> LossResult:
    student_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    with torch.no_grad():
        teacher_logits = teacher(input_ids=teacher_batch["input_ids"], attention_mask=teacher_batch["attention_mask"]).logits[:, :-1, :].float().to(student_logits.device)
    mask, fallback = selected_mask(batch, selected_ids)
    mask = mask[:, 1:]
    student_lp = F.log_softmax(student_logits, dim=-1)
    teacher_lp = F.log_softmax(teacher_logits, dim=-1)
    teacher_p = teacher_lp.exp()
    tok_kl = (teacher_p * (teacher_lp - student_lp)).sum(dim=-1)
    return LossResult(loss=(tok_kl * mask.float()).sum() / mask.float().sum(), contributing_tokens=int(mask.sum().item()), fallback_examples=fallback)


def freeze_for_mode(model, mode: str) -> Tuple[bool, int, int]:
    in_emb = model.get_input_embeddings()
    out_emb = model.get_output_embeddings()
    tied = in_emb.weight.data_ptr() == out_emb.weight.data_ptr()
    total = sum(p.numel() for p in model.parameters())
    if mode.startswith("full"):
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in in_emb.parameters():
            p.requires_grad_(True)
        for p in out_emb.parameters():
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tied, trainable, total


def unique_trainable_params(model) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))
    return params


def register_row_hooks(model, selected_ids: Sequence[int]) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    in_weight = model.get_input_embeddings().weight
    out_weight = model.get_output_embeddings().weight

    def make_hook(weight: torch.Tensor):
        row_mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
        row_mask[list(selected_ids)] = True
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[~row_mask] = 0
            return grad
        return hook

    handles.append(in_weight.register_hook(make_hook(in_weight)))
    if out_weight.data_ptr() != in_weight.data_ptr():
        handles.append(out_weight.register_hook(make_hook(out_weight)))
    return handles


def sample_batch(examples: Sequence[Example], batch_size: int, rng: random.Random) -> List[Example]:
    return [rng.choice(examples) for _ in range(batch_size)]


def run_eval(model_dir: Path | str, method: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "tofu_eval.py"),
        "--model-dir", str(model_dir),
        "--method", method,
        "--forget-split", args.forget_split,
        "--retain-split", args.retain_split,
        "--output-dir", str(Path(args.output_dir) / "eval"),
        "--max-new-tokens", str(args.eval_max_new_tokens),
        "--seed", str(args.seed),
        "--n-forget-eval", str(args.eval_n_forget),
        "--n-retain-eval", str(args.eval_n_retain),
        "--n-real-authors-eval", str(args.eval_n_real_authors),
        "--n-world-facts-eval", str(args.eval_n_world_facts),
        "--n-perturbed-eval", str(args.eval_n_perturbed),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)


def read_eval_summaries(eval_dir: Path) -> List[Dict[str, object]]:
    rows = []
    for path in sorted(eval_dir.glob("*_summary.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def write_comparison(output_dir: Path) -> None:
    keys = [
        "method", "model_dir", "forget_answer_prob", "retain_answer_prob", "forget_truth_ratio",
        "retain_truth_ratio", "real_authors_truth_ratio", "world_facts_truth_ratio",
        "forget_rougeL_recall", "retain_rougeL_recall",
    ]
    rows = read_eval_summaries(output_dir / "eval")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in keys})
    headers = [
        "method", "model_dir", "forget_answer_prob ↓", "retain_answer_prob ↑", "forget_truth_ratio",
        "retain_truth_ratio", "real_authors_truth_ratio", "world_facts_truth_ratio",
        "forget_rougeL_recall ↓", "retain_rougeL_recall ↑",
    ]
    md = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        md.append("| " + " | ".join(str(row.get(k, "")) for k in keys) + " |")
    md.append("\nMetric directions: lower forget answer probability and forget ROUGE-L recall are better; higher retain answer probability and retain ROUGE-L recall are better; truth ratios are reported as-is.")
    (output_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def save_selected_tokens(path: Path, tokenizer, all_ids: Sequence[int], selective_ids: Sequence[int]) -> None:
    payload = {
        "all_answer_token_ids": list(all_ids),
        "selective_token_ids": list(selective_ids),
        "selective_tokens": [{"id": tid, "text": tokenizer.decode([tid])} for tid in selective_ids],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_one_mode(args: argparse.Namespace, mode: str, tokenizer, forget: Sequence[Example], retain: Sequence[Example], all_ids: Sequence[int], selective_ids: Sequence[int]) -> None:
    mode_dir = Path(args.output_dir) / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = dtype_from_arg(args.dtype) if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, local_files_only=False)
    model.to(device)
    model.config.use_cache = False
    model.train()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    teacher = None
    if args.kl_retain_weight > 0:
        teacher = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, local_files_only=False)
        teacher.to(device)
        teacher.config.use_cache = False
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    tied, trainable, total = freeze_for_mode(model, mode)
    chosen_ids = set(selective_ids) if "selective_tokens" in mode else None
    hooks = register_row_hooks(model, selective_ids) if mode == "emb_lm_selective_tokens" else []
    optimizer = torch.optim.AdamW(unique_trainable_params(model), lr=args.lr, weight_decay=0.0)
    rng = random.Random(args.seed)
    log_path = mode_dir / "train_log.jsonl"
    last: Dict[str, object] = {}

    with log_path.open("w", encoding="utf-8") as logf:
        for step in trange(1, args.steps + 1, desc=mode):
            f_batch = encode_batch(tokenizer, sample_batch(forget, args.batch_size, rng), device)
            r_batch = encode_batch(tokenizer, sample_batch(retain, args.retain_batch_size, rng), device)
            forget_loss = answer_ce(model, f_batch, chosen_ids)
            retain_loss = answer_ce(model, r_batch, chosen_ids)
            retain_kl_loss = LossResult(loss=torch.zeros((), device=device), contributing_tokens=0, fallback_examples=0)
            if teacher is not None:
                teacher_batch = {k: v.to(model_device(teacher)) for k, v in r_batch.items()}
                retain_kl_loss = retain_kl(model, teacher, r_batch, teacher_batch, chosen_ids)
            total_loss = -args.forget_weight * forget_loss.loss + args.retain_weight * retain_loss.loss + args.kl_retain_weight * retain_kl_loss.loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            params = [p for p in unique_trainable_params(model) if p.grad is not None]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip).detach().cpu()) if params else 0.0
            optimizer.step()
            last = {
                "step": step,
                "total_loss": float(total_loss.detach().cpu()),
                "forget_ce": float(forget_loss.loss.detach().cpu()),
                "retain_ce": float(retain_loss.loss.detach().cpu()),
                "retain_kl": float(retain_kl_loss.loss.detach().cpu()),
                "forget_tokens": forget_loss.contributing_tokens,
                "retain_tokens": retain_loss.contributing_tokens,
                "forget_fallback_examples": forget_loss.fallback_examples,
                "retain_fallback_examples": retain_loss.fallback_examples + retain_kl_loss.fallback_examples,
                "grad_norm": grad_norm,
                "mode": mode,
                "n_selected_ids": len(selective_ids) if "selective_tokens" in mode else len(all_ids),
                "tied_embeddings": tied,
            }
            logf.write(json.dumps(last) + "\n")
            logf.flush()

    for handle in hooks:
        handle.remove()
    ckpt_dir = mode_dir / "checkpoint"
    if args.save_model:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
    save_selected_tokens(mode_dir / "selected_tokens.json", tokenizer, all_ids, selective_ids)
    summary = {
        "mode": mode,
        "checkpoint": str(ckpt_dir),
        "n_all_answer_ids": len(all_ids),
        "n_selected_ids": len(selective_ids),
        "tied_embeddings": tied,
        "trainable_params": trainable,
        "total_params": total,
        "last_log": last,
    }
    (mode_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not args.skip_eval and args.save_model:
        run_eval(ckpt_dir, mode, args)


def parse_args() -> argparse.Namespace:
    default_model = str(DEFAULT_MODEL_DIR) if DEFAULT_MODEL_DIR.exists() else None
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=default_model)
    parser.add_argument("--output-dir", default="outputs/tofu_four_settings")
    parser.add_argument("--forget-split", default="forget05")
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--kl-retain-weight", type=float, default=0.0)
    parser.add_argument("--selective-top-k", type=int, default=300)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--eval-n-forget", type=int, default=200)
    parser.add_argument("--eval-n-retain", type=int, default=400)
    parser.add_argument("--eval-n-real-authors", type=int, default=100)
    parser.add_argument("--eval-n-world-facts", type=int, default=117)
    parser.add_argument("--eval-n-perturbed", type=int, default=100)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_model_path(args.model_path)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Re-run with --device cpu to train on CPU.")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}; expected subset of {MODES}")
    set_seed(args.seed)
    out_dir = Path(args.output_dir).resolve()
    args.output_dir = str(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    forget = load_tofu_examples(args.forget_split, args.forget_num, args.seed)
    retain = load_tofu_examples(args.retain_split, args.retain_num, args.seed)
    all_ids, selective_ids = build_token_sets(tokenizer, forget, retain, args.selective_top_k)
    (out_dir / "config_used.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    if not args.skip_eval:
        run_eval(args.model_path, "base", args)
    for mode in modes:
        train_one_mode(args, mode, tokenizer, forget, retain, all_ids, selective_ids)
    if not args.skip_eval:
        write_comparison(out_dir)
    print(f"cat {out_dir / 'comparison.md'}")
    print(f"tail -n 5 {out_dir / 'emb_lm_selective_tokens' / 'train_log.jsonl'}")


if __name__ == "__main__":
    main()
