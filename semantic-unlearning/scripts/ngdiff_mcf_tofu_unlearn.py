#!/usr/bin/env python3
"""NGDiff / normalized gradient difference unlearning for MCF and TOFU."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import string
import subprocess
import sys
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
COMMON_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "for",
    "with", "by", "from", "at", "as", "is", "are", "was", "were", "be", "been", "being", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "them", "his", "her", "their",
    "i", "you", "we", "not", "no", "yes", "do", "does", "did", "has", "have", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "also", "there", "here", "who",
    "what", "when", "where", "why", "how", "which", "about", "into", "than", "so", "because",
}


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str
    subject: str = ""
    target_true: str = ""
    target_new: str = ""
    paraphrase_prompts: Optional[List[str]] = None
    source: str = ""


@dataclass
class LossResult:
    loss: torch.Tensor
    contributing_tokens: int
    fallback_examples: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["mcf", "tofu"], default="tofu")
    p.add_argument("--model-path", required=True)
    p.add_argument("--reference-model-path", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forget-split", default="forget05")
    p.add_argument("--retain-split", default="retain95")
    p.add_argument("--forget-num", type=int, default=200)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--mcf-cache-path", default="data/multi_counterfact.json")
    p.add_argument("--mcf-url", default="https://memit.baulab.info/data/dsets/multi_counterfact.json")
    p.add_argument("--mcf-answer-field", choices=["target_true", "target_new"], default="target_true")
    p.add_argument("--mode", choices=["full_all_tokens", "emb_lm_selective_tokens"], default="emb_lm_selective_tokens")
    p.add_argument("--selective-top-k", type=int, default=300)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--retain-batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--retain-kl-weight", type=float, default=0.0)
    p.add_argument("--kl-temperature", type=float, default=1.0)
    p.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--eval-n-forget", type=int, default=200)
    p.add_argument("--eval-n-retain", type=int, default=400)
    p.add_argument("--eval-n-real-authors", type=int, default=100)
    p.add_argument("--eval-n-world-facts", type=int, default=117)
    p.add_argument("--eval-n-perturbed", type=int, default=100)
    p.add_argument("--eval-max-new-tokens", type=int, default=64)
    p.add_argument("--official-eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--official-unlearn-num", type=int, default=50)
    p.add_argument("--official-retain-num", type=int, default=1000)
    p.add_argument("--official-sample-mode", default="official")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_arg(dtype: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    return answer if answer.startswith(" ") else " " + answer


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else PROJECT_DIR / p


def load_tofu(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    forget_rows = list(load_dataset("locuslab/TOFU", name=args.forget_split, split="train"))
    retain_rows = list(load_dataset("locuslab/TOFU", name=args.retain_split, split="train"))
    rng = random.Random(args.seed)
    rng.shuffle(forget_rows)
    rng.shuffle(retain_rows)

    def convert(row: Dict[str, Any]) -> Example:
        return Example(prompt=f"Question: {row['question']}\nAnswer:", answer=normalize_answer(str(row["answer"])), source="tofu")

    return [convert(r) for r in forget_rows[: args.forget_num]], [convert(r) for r in retain_rows[: args.retain_num]]


def format_mcf_prompt(template: str, subject: str) -> str:
    return template.format(subject) if "{}" in template else template


def load_mcf(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    cache = resolve_path(args.mcf_cache_path)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MCF data from {args.mcf_url} to {cache}")
        urllib.request.urlretrieve(args.mcf_url, cache)
    raw = json.loads(cache.read_text(encoding="utf-8"))
    examples: List[Example] = []
    for rec in raw:
        rr = rec.get("requested_rewrite")
        if isinstance(rr, list):
            rr = rr[0]
        if not isinstance(rr, dict):
            raise ValueError("MCF record missing requested_rewrite dict")
        subject = str(rr["subject"])
        target_new = normalize_answer(str(rr["target_new"]["str"]))
        target_true = normalize_answer(str(rr.get("target_true", {}).get("str", "")))
        if args.mcf_answer_field == "target_true" and not target_true.strip():
            raise ValueError("target_true requested but missing in MCF record")
        paraphrases = rec.get("paraphrase_prompts") or rr.get("paraphrase_prompts") or []
        examples.append(Example(
            prompt=format_mcf_prompt(str(rr["prompt"]), subject),
            answer=target_new if args.mcf_answer_field == "target_new" else target_true,
            subject=subject,
            target_true=target_true,
            target_new=target_new,
            paraphrase_prompts=[format_mcf_prompt(str(p), subject) for p in paraphrases],
            source="mcf",
        ))
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    need = args.forget_num + args.retain_num
    if len(examples) < need:
        raise ValueError(f"MCF has {len(examples)} examples, need {need}")
    return examples[: args.forget_num], examples[args.forget_num:need]


def token_ids(tokenizer: AutoTokenizer, text: str) -> List[int]:
    return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]


def bad_selective_token(tokenizer: AutoTokenizer, tid: int) -> bool:
    if tid in set(int(x) for x in tokenizer.all_special_ids if x is not None):
        return True
    text = tokenizer.decode([tid], skip_special_tokens=True).strip().lower()
    if not text:
        return True
    if all(ch in string.punctuation for ch in text):
        return True
    return bool(re.fullmatch(r"[\W_]+", text)) or text in COMMON_FUNCTION_WORDS


def select_tokens(tokenizer: AutoTokenizer, forget: Sequence[Example], retain: Sequence[Example], args: argparse.Namespace) -> Optional[List[int]]:
    if args.mode == "full_all_tokens":
        return None
    forget_df: Counter[int] = Counter()
    retain_df: Counter[int] = Counter()
    for ex in forget:
        forget_df.update(set(token_ids(tokenizer, ex.answer)))
    for ex in retain:
        retain_df.update(set(token_ids(tokenizer, ex.answer)))
    n_forget, n_retain = max(len(forget), 1), max(len(retain), 1)
    scored = [((df / n_forget) / ((retain_df.get(tid, 0) / n_retain) + 1e-6), tid) for tid, df in forget_df.items() if not bad_selective_token(tokenizer, tid)]
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [tid for _, tid in scored[: args.selective_top_k]]


def save_selected_metadata(path: Path, tokenizer: AutoTokenizer, selected: Optional[Sequence[int]], args: argparse.Namespace) -> None:
    payload = {"mode": args.mode, "selective_top_k": args.selective_top_k, "selected_token_ids": list(selected or []), "selected_tokens": [{"id": int(t), "text": tokenizer.decode([int(t)])} for t in (selected or [])]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_batch(tokenizer: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> Dict[str, torch.Tensor]:
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    for ex in examples:
        p_ids = token_ids(tokenizer, ex.prompt)
        a_ids = token_ids(tokenizer, ex.answer) + eos
        if not a_ids:
            raise ValueError("Example has no answer tokens")
        rows.append(p_ids + a_ids)
        labels.append([-100] * len(p_ids) + a_ids)
    max_len = max(len(r) for r in rows)
    input_ids, label_ids, attention = [], [], []
    for row, lab in zip(rows, labels):
        pad_n = max_len - len(row)
        input_ids.append(row + [pad] * pad_n)
        label_ids.append(lab + [-100] * pad_n)
        attention.append([1] * len(row) + [0] * pad_n)
    return {"input_ids": torch.tensor(input_ids, device=device), "labels": torch.tensor(label_ids, device=device), "attention_mask": torch.tensor(attention, device=device)}


def selection_mask(labels: torch.Tensor, selected_ids: Optional[set[int]]) -> Tuple[torch.Tensor, int]:
    answer_mask = labels.ne(-100)
    if selected_ids is None:
        return answer_mask, 0
    selected = torch.zeros_like(answer_mask)
    for tid in selected_ids:
        selected |= labels.eq(tid)
    selected &= answer_mask
    empty = selected.sum(dim=1).eq(0)
    fallback = int(empty.sum().item())
    if fallback:
        selected = selected.clone()
        selected[empty] = answer_mask[empty]
    return selected, fallback


def answer_ce_loss(model: torch.nn.Module, batch: Dict[str, torch.Tensor], selected_ids: Optional[set[int]]) -> LossResult:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:].contiguous()
    mask, fallback = selection_mask(labels, selected_ids)
    if mask.sum().item() == 0:
        raise ValueError("No answer tokens contributed to loss")
    per_tok = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="none").view_as(labels)
    return LossResult((per_tok * mask.float()).sum() / mask.float().sum(), int(mask.sum().item()), fallback)


def retain_kl_loss(student: torch.nn.Module, teacher: torch.nn.Module, batch: Dict[str, torch.Tensor], selected_ids: Optional[set[int]], temperature: float) -> LossResult:
    s_logits = student(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float() / temperature
    with torch.no_grad():
        t_logits = teacher(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float().to(s_logits.device) / temperature
    labels = batch["labels"][:, 1:]
    mask, fallback = selection_mask(labels, selected_ids)
    t_lp = F.log_softmax(t_logits, dim=-1)
    s_lp = F.log_softmax(s_logits, dim=-1)
    kl = (t_lp.exp() * (t_lp - s_lp)).sum(dim=-1) * (temperature ** 2)
    return LossResult((kl * mask.float()).sum() / mask.float().sum().clamp_min(1), int(mask.sum().item()), fallback)


def load_model_and_tokenizer(args: argparse.Namespace) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    dtype = dtype_from_arg(args.dtype) if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.to(device)
    model.train()
    return model, tokenizer, device


def configure_trainable(model: torch.nn.Module, mode: str, selected_ids: Optional[Sequence[int]]) -> Tuple[List[torch.nn.Parameter], bool, List[Any]]:
    hooks: List[Any] = []
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None:
        raise ValueError("Model has no output embeddings/lm_head; emb_lm_selective_tokens requires get_output_embeddings().")
    tied = inp.weight.data_ptr() == out.weight.data_ptr()
    if mode == "full_all_tokens":
        print("WARNING: full_all_tokens NGDiff stores two full gradient copies and may require substantial memory.")
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        if not selected_ids:
            raise ValueError("No selected token ids for emb_lm_selective_tokens")
        for p in model.parameters():
            p.requires_grad_(False)
        inp.weight.requires_grad_(True)
        out.weight.requires_grad_(True)
        def make_hook(weight: torch.Tensor):
            row_mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
            row_mask[list(selected_ids)] = True
            def hook(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                grad[~row_mask] = 0
                return grad
            return hook
        hooks.append(inp.weight.register_hook(make_hook(inp.weight)))
        if not tied:
            hooks.append(out.weight.register_hook(make_hook(out.weight)))
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            params.append(p)
            seen.add(id(p))
    return params, tied, hooks


def grad_norm(grads: Sequence[torch.Tensor]) -> torch.Tensor:
    if not grads:
        return torch.zeros((), device="cpu")
    return torch.sqrt(sum(g.detach().float().pow(2).sum() for g in grads))


def clone_grads(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]


def make_optimizer(name: str, params: Sequence[torch.nn.Parameter], lr: float):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def run_tofu_eval(args: argparse.Namespace, checkpoint: Path) -> None:
    cmd = [sys.executable, str(SCRIPT_DIR / "tofu_eval.py"), "--model-dir", str(checkpoint), "--method", f"ngdiff_{args.mode}", "--forget-split", args.forget_split, "--retain-split", args.retain_split, "--output-dir", str(Path(args.output_dir) / "eval"), "--max-new-tokens", str(args.eval_max_new_tokens), "--seed", str(args.seed), "--n-forget-eval", str(args.eval_n_forget), "--n-retain-eval", str(args.eval_n_retain), "--n-real-authors-eval", str(args.eval_n_real_authors), "--n-world-facts-eval", str(args.eval_n_world_facts), "--n-perturbed-eval", str(args.eval_n_perturbed)]
    subprocess.run(cmd, cwd=str(PROJECT_DIR), check=True)


def run_mcf_official_eval(args: argparse.Namespace, checkpoint: Path) -> None:
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from mcf_zero_unlearn_official_eval import evaluate_model_dir_official, result_to_comparison_row, write_official_comparison  # type: ignore
    except Exception as exc:
        print(f"WARNING: Could not import MCF official eval helpers: {exc}")
        return
    base = evaluate_model_dir_official(args.model_path, args.official_unlearn_num, args.official_retain_num, args.seed, args.official_sample_mode)
    tuned = evaluate_model_dir_official(str(checkpoint), args.official_unlearn_num, args.official_retain_num, args.seed, args.official_sample_mode)
    rows = [result_to_comparison_row("base", base), result_to_comparison_row(f"ngdiff_{args.mode}", tuned)]
    write_official_comparison(rows, Path(args.output_dir) / "official_eval_comparison.md", Path(args.output_dir) / "official_eval_comparison.csv")


def print_final(args: argparse.Namespace) -> None:
    for cmd in [["cat", str(Path(args.output_dir) / "train_summary.json")], ["tail", "-n", "5", str(Path(args.output_dir) / "train_log.jsonl")]]:
        subprocess.run(cmd, check=False)
    if args.dataset == "mcf":
        subprocess.run(["cat", str(Path(args.output_dir) / "official_eval_comparison.md")], check=False)
    else:
        subprocess.run(["cat", str(Path(args.output_dir) / "eval" / "tofu_eco_summary.csv")], check=False)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, device = load_model_and_tokenizer(args)
    forget, retain = load_mcf(args) if args.dataset == "mcf" else load_tofu(args)
    selected = select_tokens(tokenizer, forget, retain, args)
    save_selected_metadata(output_dir / "selected_tokens.json", tokenizer, selected, args)
    selected_set = set(selected) if selected is not None else None
    params, tied, hooks = configure_trainable(model, args.mode, selected)
    optimizer = make_optimizer(args.optimizer, params, args.lr)
    teacher = None
    if args.retain_kl_weight > 0:
        ref = args.reference_model_path or args.model_path
        teacher = AutoModelForCausalLM.from_pretrained(ref, torch_dtype=dtype_from_arg(args.dtype) if device.type == "cuda" else torch.float32).to(device)
        teacher.eval()
        teacher.config.use_cache = False
        for p in teacher.parameters():
            p.requires_grad_(False)
    rng = random.Random(args.seed)
    log_path = output_dir / "train_log.jsonl"
    last: Dict[str, Any] = {}
    eps = 1e-12
    with log_path.open("w", encoding="utf-8") as logf:
        for step in trange(1, args.steps + 1, desc="ngdiff"):
            f_ex = [rng.choice(forget) for _ in range(args.batch_size)]
            r_ex = [rng.choice(retain) for _ in range(args.retain_batch_size)]
            f_batch = build_batch(tokenizer, f_ex, device)
            r_batch = build_batch(tokenizer, r_ex, device)
            optimizer.zero_grad(set_to_none=True)
            retain_ce = answer_ce_loss(model, r_batch, selected_set)
            retain_kl = LossResult(torch.zeros((), device=device), 0, 0)
            retain_obj = retain_ce.loss
            if teacher is not None:
                retain_kl = retain_kl_loss(model, teacher, r_batch, selected_set, args.kl_temperature)
                retain_obj = retain_obj + args.retain_kl_weight * retain_kl.loss
            retain_obj.backward()
            retain_grads = clone_grads(params)
            retain_norm = grad_norm(retain_grads).to(device)
            optimizer.zero_grad(set_to_none=True)
            forget_ce = answer_ce_loss(model, f_batch, selected_set)
            forget_ce.loss.backward()
            forget_grads = clone_grads(params)
            forget_norm = grad_norm(forget_grads).to(device)
            optimizer.zero_grad(set_to_none=True)
            for p, rg, fg in zip(params, retain_grads, forget_grads):
                p.grad = rg / (retain_norm + eps) - fg / (forget_norm + eps)
            total_update_norm = float(torch.nn.utils.clip_grad_norm_(params, args.grad_clip).detach().cpu()) if args.grad_clip and args.grad_clip > 0 else float(grad_norm([p.grad for p in params if p.grad is not None]).detach().cpu())
            optimizer.step()
            last = {"step": step, "dataset": args.dataset, "mode": args.mode, "total_update_grad_norm": total_update_norm, "retain_grad_norm": float(retain_norm.detach().cpu()), "forget_grad_norm": float(forget_norm.detach().cpu()), "forget_ce": float(forget_ce.loss.detach().cpu()), "retain_ce": float(retain_ce.loss.detach().cpu()), "retain_kl": float(retain_kl.loss.detach().cpu()), "forget_tokens": forget_ce.contributing_tokens, "retain_tokens": retain_ce.contributing_tokens, "forget_fallback_examples": forget_ce.fallback_examples, "retain_fallback_examples": retain_ce.fallback_examples + retain_kl.fallback_examples, "n_selected_ids": len(selected or []), "tied_embeddings": tied, "lr": args.lr}
            logf.write(json.dumps(last) + "\n")
            logf.flush()
    for h in hooks:
        h.remove()
    checkpoint = output_dir / "checkpoint"
    if args.save_model:
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
    summary = {**last, "output_dir": str(output_dir), "checkpoint": str(checkpoint) if args.save_model else None, "n_forget": len(forget), "n_retain": len(retain)}
    (output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if not args.skip_eval and args.save_model:
        if args.dataset == "tofu":
            run_tofu_eval(args, checkpoint)
        elif args.official_eval:
            run_mcf_official_eval(args, checkpoint)
    print_final(args)


if __name__ == "__main__":
    main()
