#!/usr/bin/env python3
"""CE-U unlearning runner for TOFU and MCF across four trainable scopes."""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import urllib.request
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
MCF_URL = "https://memit.baulab.info/data/dsets/multi_counterfact.json"
FUNCTION_TOKENS = {
    "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "for", "with", "by", "from",
    "at", "as", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that", "he",
    "she", "they", "them", "his", "her", "their", "not", "do", "does", "did", "has", "have", "had",
}


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str


@dataclass
class LossResult:
    loss: torch.Tensor
    token_count: int
    fallback_examples: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_arg(dtype: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    if dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def format_tofu_prompt(tokenizer, question: str, prompt_format: str) -> str:
    content = f"Question: {question} Answer:"
    if prompt_format == "plain":
        return f"Question: {question}\nAnswer:"
    if prompt_format == "chat":
        if tokenizer.chat_template is None:
            raise ValueError("--tofu-prompt-format chat requested but tokenizer has no chat_template")
        msg = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    if prompt_format == "auto":
        if tokenizer.chat_template is not None:
            msg = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return f"Question: {question}\nAnswer:"
    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def load_tofu_examples(tokenizer, split: str, limit: int, seed: int, prompt_format: str) -> List[Example]:
    rows = list(load_dataset("locuslab/TOFU", name=split, split="train"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples: List[Example] = []
    for row in rows[:limit]:
        prompt = format_tofu_prompt(tokenizer, row["question"], prompt_format)
        answer = " " + row["answer"].strip()
        examples.append(Example(prompt=prompt, answer=answer))
    return examples


def ensure_mcf(path: Path, cache_path: Path | None, url: str) -> Path:
    if path.exists():
        return path
    if cache_path and cache_path.exists():
        return cache_path
    target = cache_path or path
    target.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, target)
    return target


def format_mcf_prompt(row: dict) -> str:
    rr = row["requested_rewrite"]
    prompt = rr["prompt"]
    subject = rr["subject"]
    return prompt.format(subject) if "{}" in prompt else prompt


def mcf_answer(row: dict, field: str) -> str:
    rr = row["requested_rewrite"]
    key = "target_true" if field == "target_true" else "target_new"
    return " " + rr[key]["str"].strip()


def load_mcf_examples(path: Path, cache_path: Path | None, url: str, limit: int, seed: int, answer_field: str, offset: int = 0) -> List[Example]:
    data_path = ensure_mcf(path, cache_path, url)
    rows = json.loads(data_path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[offset : offset + limit]
    return [Example(prompt=format_mcf_prompt(row), answer=mcf_answer(row, answer_field)) for row in rows]


def bad_token(tokenizer, token_id: int, special_ids: set[int]) -> bool:
    if token_id in special_ids:
        return True
    text = tokenizer.decode([token_id], skip_special_tokens=True).strip().lower()
    if not text:
        return True
    if all(ch in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~" for ch in text):
        return True
    return text in FUNCTION_TOKENS


def answer_ids(tokenizer, answer: str) -> List[int]:
    return tokenizer.encode(answer, add_special_tokens=False)


def build_token_sets(tokenizer, forget: Sequence[Example], retain: Sequence[Example], top_k: int) -> Tuple[List[int], List[int]]:
    specials = set(tokenizer.all_special_ids)
    all_ids = sorted({tid for ex in forget for tid in answer_ids(tokenizer, ex.answer) if tid not in specials})
    f_df: Counter[int] = Counter()
    r_df: Counter[int] = Counter()
    for ex in forget:
        f_df.update(set(tid for tid in answer_ids(tokenizer, ex.answer) if tid not in specials))
    for ex in retain:
        r_df.update(set(tid for tid in answer_ids(tokenizer, ex.answer) if tid not in specials))
    scored: List[Tuple[float, int]] = []
    for tid, df in f_df.items():
        if bad_token(tokenizer, tid, specials):
            continue
        score = (df / max(len(forget), 1)) / ((r_df.get(tid, 0) / max(len(retain), 1)) + 1e-6)
        scored.append((score, tid))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return all_ids, [tid for _, tid in scored[:top_k]]


def encode_batch(tokenizer, examples: Sequence[Example], device: torch.device) -> Dict[str, torch.Tensor]:
    eos = tokenizer.eos_token or ""
    ids_rows: List[List[int]] = []
    label_rows: List[List[int]] = []
    for ex in examples:
        full = ex.prompt + ex.answer + (eos if eos and not ex.answer.endswith(eos) else "")
        if getattr(tokenizer, "is_fast", False):
            enc = tokenizer(full, add_special_tokens=True, return_offsets_mapping=True)
            ids = enc.input_ids
            mask = [end > len(ex.prompt) for _, end in enc.get("offset_mapping")]
        else:
            ids = tokenizer(full, add_special_tokens=True).input_ids
            prompt_len = len(tokenizer(ex.prompt, add_special_tokens=True).input_ids)
            mask = [i >= prompt_len for i in range(len(ids))]
        labels = [tok if keep else -100 for tok, keep in zip(ids, mask)]
        if all(x == -100 for x in labels):
            raise ValueError(f"No answer tokens found for prompt={ex.prompt!r}, answer={ex.answer!r}")
        ids_rows.append(ids)
        label_rows.append(labels)
    pad = tokenizer.pad_token_id
    max_len = max(len(row) for row in ids_rows)
    input_ids, labels, attention = [], [], []
    for ids, labs in zip(ids_rows, label_rows):
        n = max_len - len(ids)
        input_ids.append(ids + [pad] * n)
        labels.append(labs + [-100] * n)
        attention.append([1] * len(ids) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


def shifted_answer_mask(labels: torch.Tensor, ignore_first: bool) -> torch.Tensor:
    mask = labels[:, 1:].ne(-100)
    if ignore_first:
        mask = mask.clone()
        for row in range(mask.shape[0]):
            idx = torch.nonzero(mask[row], as_tuple=False)
            if idx.numel() > 0:
                mask[row, idx[0, 0]] = False
    return mask


def selected_position_mask(labels: torch.Tensor, base_mask: torch.Tensor, selected_ids: set[int] | None) -> Tuple[torch.Tensor, int]:
    if not selected_ids:
        return base_mask, 0
    shift_labels = labels[:, 1:]
    selected = torch.zeros_like(base_mask)
    for tid in selected_ids:
        selected |= shift_labels.eq(tid)
    selected &= base_mask
    empty = selected.sum(dim=1).eq(0)
    fallback = int(empty.sum().item())
    if fallback:
        selected = selected.clone()
        selected[empty] = base_mask[empty]
    if selected.sum().item() == 0:
        raise ValueError("Selected CE-U mask is empty for the whole batch after fallback")
    return selected, fallback


def ceu_forget_loss(model, batch: Dict[str, torch.Tensor], selected_ids: set[int] | None, args: argparse.Namespace) -> LossResult:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
    shift_logits = logits[:, :-1, :]
    shift_labels = batch["labels"][:, 1:]
    base_mask = shifted_answer_mask(batch["labels"], args.ignore_first_answer_token)
    mask, fallback = selected_position_mask(batch["labels"], base_mask, selected_ids if args.selected_forget_only else None)
    flat_logits = shift_logits[mask].float()
    flat_labels = shift_labels[mask].clamp_min(0)
    if flat_labels.numel() == 0:
        raise ValueError("CE-U forget mask has zero tokens")
    z = flat_logits.detach().clone()
    z.scatter_(1, flat_labels.unsqueeze(1), -1e9)
    q = torch.softmax(z / args.kl_temperature, dim=-1).detach()
    logp = torch.log_softmax(flat_logits / args.kl_temperature, dim=-1)
    per_token = -(q * logp).sum(dim=-1)
    return LossResult(loss=per_token.mean(), token_count=int(flat_labels.numel()), fallback_examples=fallback)


def answer_ce_loss(model, batch: Dict[str, torch.Tensor], selected_ids: set[int] | None) -> LossResult:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    base_mask = labels.ne(-100)
    mask, fallback = selected_position_mask(batch["labels"], base_mask, selected_ids)
    nll = -torch.log_softmax(logits, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return LossResult(loss=(nll * mask.float()).sum() / mask.float().sum(), token_count=int(mask.sum().item()), fallback_examples=fallback)


def retain_kl_loss(student, teacher, batch: Dict[str, torch.Tensor], teacher_batch: Dict[str, torch.Tensor], selected_ids: set[int] | None) -> LossResult:
    slogits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    with torch.no_grad():
        tlogits = teacher(input_ids=teacher_batch["input_ids"], attention_mask=teacher_batch["attention_mask"]).logits[:, :-1, :].float().to(slogits.device)
    labels = batch["labels"][:, 1:]
    mask, fallback = selected_position_mask(batch["labels"], labels.ne(-100), selected_ids)
    slp = torch.log_softmax(slogits, dim=-1)
    tlp = torch.log_softmax(tlogits, dim=-1)
    tp = tlp.exp()
    kl = (tp * (tlp - slp)).sum(dim=-1)
    return LossResult(loss=(kl * mask.float()).sum() / mask.float().sum(), token_count=int(mask.sum().item()), fallback_examples=fallback)


def freeze_for_mode(model, mode: str) -> Tuple[bool, int, int]:
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    tied = inp.weight.data_ptr() == out.weight.data_ptr()
    total = sum(p.numel() for p in model.parameters())
    if mode.startswith("full"):
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        for p in inp.parameters():
            p.requires_grad_(True)
        for p in out.parameters():
            p.requires_grad_(True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tied, trainable, total


def unique_params(model) -> List[torch.nn.Parameter]:
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))
    return params


def row_hooks(model, selected_ids: Sequence[int]) -> List[torch.utils.hooks.RemovableHandle]:
    handles: List[torch.utils.hooks.RemovableHandle] = []
    inp = model.get_input_embeddings().weight
    out = model.get_output_embeddings().weight

    def make_hook(weight: torch.Tensor):
        row_mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
        row_mask[list(selected_ids)] = True
        def hook(grad: torch.Tensor) -> torch.Tensor:
            grad = grad.clone()
            grad[~row_mask] = 0
            return grad
        return hook

    handles.append(inp.register_hook(make_hook(inp)))
    if out.data_ptr() != inp.data_ptr():
        handles.append(out.register_hook(make_hook(out)))
    return handles


def make_optimizer(name: str, params: Iterable[torch.nn.Parameter], lr: float):
    params = list(params)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def sample(examples: Sequence[Example], batch_size: int, rng: random.Random) -> List[Example]:
    return [rng.choice(examples) for _ in range(batch_size)]


def run_tofu_eval(checkpoint: Path, method: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "tofu_eval.py"), "--model-dir", str(checkpoint), "--method", method,
        "--forget-split", args.forget_split, "--retain-split", args.retain_split,
        "--output-dir", str(Path(args.output_dir) / "eval"), "--max-new-tokens", str(args.eval_max_new_tokens),
        "--seed", str(args.seed), "--n-forget-eval", str(args.eval_n_forget), "--n-retain-eval", str(args.eval_n_retain),
        "--n-real-authors-eval", str(args.eval_n_real_authors), "--n-world-facts-eval", str(args.eval_n_world_facts),
        "--n-perturbed-eval", str(args.eval_n_perturbed),
    ]
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)


def run_mcf_eval(checkpoint: Path, method: str, args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, str(SCRIPT_DIR / "run_same_mcf_eval.py"), "--model-dirs", f"{method}={checkpoint}",
        "--mcf-path", args.mcf_path, "--out-dir", str(Path(args.output_dir) / "official_eval"),
        "--unlearn-num", str(args.official_unlearn_num), "--retain-num", str(args.official_retain_num),
        "--seed", str(args.seed), "--sample-mode", args.official_sample_mode,
    ]
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)


def save_selected_tokens(path: Path, tokenizer, all_ids: Sequence[int], selected_ids: Sequence[int]) -> None:
    payload = {
        "all_answer_token_ids": list(all_ids),
        "selected_token_ids": list(selected_ids),
        "selected_tokens": [{"id": tid, "text": tokenizer.decode([tid])} for tid in selected_ids],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_mode(args: argparse.Namespace, mode: str, tokenizer, forget: Sequence[Example], retain: Sequence[Example], all_ids: Sequence[int], selected_ids: Sequence[int]) -> None:
    out_dir = Path(args.output_dir) / f"ceu_{mode}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    dtype = dtype_from_arg(args.dtype, args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    model.train()
    teacher = None
    if args.retain_kl_weight > 0:
        ref_path = args.reference_model_path or args.model_path
        teacher = AutoModelForCausalLM.from_pretrained(ref_path, torch_dtype=dtype)
        teacher.to(device)
        teacher.config.use_cache = False
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
    tied, trainable, total = freeze_for_mode(model, mode)
    selected_set = set(selected_ids) if "selective_tokens" in mode else None
    hooks = row_hooks(model, selected_ids) if mode == "emb_lm_selective_tokens" else []
    optimizer = make_optimizer(args.optimizer, unique_params(model), args.lr)
    rng = random.Random(args.seed)
    last_log: Dict[str, object] = {}
    log_path = out_dir / "train_log.jsonl"
    with log_path.open("w", encoding="utf-8") as logf:
        for step in trange(1, args.steps + 1, desc=f"CE-U {mode}"):
            f_batch = encode_batch(tokenizer, sample(forget, args.batch_size, rng), device)
            r_batch = encode_batch(tokenizer, sample(retain, args.retain_batch_size, rng), device)
            ceu = ceu_forget_loss(model, f_batch, selected_set, args)
            retain_ce = answer_ce_loss(model, r_batch, selected_set)
            retain_kl = LossResult(torch.zeros((), device=device), 0, 0)
            if teacher is not None:
                t_batch = {k: v.to(model_device(teacher)) for k, v in r_batch.items()}
                retain_kl = retain_kl_loss(model, teacher, r_batch, t_batch, selected_set)
            total_loss = args.ceu_weight * ceu.loss + args.retain_weight * retain_ce.loss + args.retain_kl_weight * retain_kl.loss
            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            params = [p for p in unique_params(model) if p.grad is not None]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip).detach().cpu()) if params else 0.0
            optimizer.step()
            last_log = {
                "step": step, "dataset": args.dataset, "mode": mode, "total_loss": float(total_loss.detach().cpu()),
                "ceu_forget_loss": float(ceu.loss.detach().cpu()), "retain_ce": float(retain_ce.loss.detach().cpu()),
                "retain_kl": float(retain_kl.loss.detach().cpu()), "forget_tokens": ceu.token_count,
                "retain_tokens": retain_ce.token_count, "forget_fallback_examples": ceu.fallback_examples,
                "retain_fallback_examples": retain_ce.fallback_examples + retain_kl.fallback_examples,
                "grad_norm": grad_norm, "n_selected_ids": len(selected_ids) if selected_set is not None else len(all_ids),
                "tied_embeddings": tied, "lr": args.lr, "ceu_weight": args.ceu_weight,
                "retain_weight": args.retain_weight, "retain_kl_weight": args.retain_kl_weight,
            }
            logf.write(json.dumps(last_log) + "\n")
            logf.flush()
    for handle in hooks:
        handle.remove()
    checkpoint = out_dir / "checkpoint"
    if args.save_model:
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
    save_selected_tokens(out_dir / "selected_tokens.json", tokenizer, all_ids, selected_ids)
    summary = {
        "method": "ceu", "dataset": args.dataset, "mode": mode, "output_dir": str(out_dir),
        "checkpoint": str(checkpoint), "n_forget": len(forget), "n_retain": len(retain),
        "n_selected_ids": len(selected_ids), "tofu_prompt_format": args.tofu_prompt_format,
        "trainable_params": trainable, "total_params": total, "last_log": last_log,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    del model, teacher
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not args.skip_eval and args.save_model:
        method = f"ceu_{mode}"
        if args.dataset == "tofu":
            run_tofu_eval(checkpoint, method, args)
        elif args.official_eval:
            run_mcf_eval(checkpoint, method, args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["tofu", "mcf"], default="tofu")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--reference-model-path", default=None)
    parser.add_argument("--output-dir", default="outputs/ceu_mcf_tofu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-split", default="forget05")
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--mcf-cache-path", default=None)
    parser.add_argument("--mcf-path", default="data/mcf/multi_counterfact.json")
    parser.add_argument("--mcf-url", default=MCF_URL)
    parser.add_argument("--mcf-answer-field", choices=["target_new", "target_true"], default="target_new")
    parser.add_argument("--mode", default=",".join(MODES))
    parser.add_argument("--selective-top-k", type=int, default=300)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--tofu-prompt-format", choices=["auto", "chat", "plain"], default="auto")
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-n-forget", type=int, default=200)
    parser.add_argument("--eval-n-retain", type=int, default=400)
    parser.add_argument("--eval-n-real-authors", type=int, default=100)
    parser.add_argument("--eval-n-world-facts", type=int, default=117)
    parser.add_argument("--eval-n-perturbed", type=int, default=100)
    parser.add_argument("--eval-max-new-tokens", type=int, default=64)
    parser.add_argument("--official-eval", action="store_true")
    parser.add_argument("--official-unlearn-num", type=int, default=50)
    parser.add_argument("--official-retain-num", type=int, default=1000)
    parser.add_argument("--official-sample-mode", choices=["official", "first"], default="official")
    parser.add_argument("--ceu-weight", type=float, default=1.0)
    parser.add_argument("--retain-weight", type=float, default=1.0)
    parser.add_argument("--retain-kl-weight", type=float, default=0.1)
    parser.add_argument("--kl-temperature", type=float, default=1.0)
    parser.add_argument("--ignore-first-answer-token", action="store_true", default=False)
    parser.add_argument("--selected-forget-only", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu to run on CPU.")
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.dataset == "tofu":
        forget = load_tofu_examples(tokenizer, args.forget_split, args.forget_num, args.seed, args.tofu_prompt_format)
        retain = load_tofu_examples(tokenizer, args.retain_split, args.retain_num, args.seed, args.tofu_prompt_format)
        print("[TOFU prompt preview]", repr(forget[0].prompt[:300]))
    else:
        cache_path = Path(args.mcf_cache_path) if args.mcf_cache_path else None
        forget = load_mcf_examples(Path(args.mcf_path), cache_path, args.mcf_url, args.forget_num, args.seed, args.mcf_answer_field)
        retain = load_mcf_examples(Path(args.mcf_path), cache_path, args.mcf_url, args.retain_num, args.seed, args.mcf_answer_field, offset=args.forget_num)
    all_ids, selected_ids = build_token_sets(tokenizer, forget, retain, args.selective_top_k)
    modes = MODES if args.mode == "all" else [m.strip() for m in args.mode.split(",") if m.strip()]
    unknown = [m for m in modes if m not in MODES]
    if unknown:
        raise ValueError(f"Unknown mode(s): {unknown}; expected {MODES} or all")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "config_used.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    for mode in modes:
        train_mode(args, mode, tokenizer, forget, retain, all_ids, selected_ids)
    print(f"cat {Path(args.output_dir) / 'ceu_emb_lm_selective_tokens' / 'train_summary.json'}")
    print(f"tail -n 5 {Path(args.output_dir) / 'ceu_emb_lm_selective_tokens' / 'train_log.jsonl'}")


if __name__ == "__main__":
    main()
