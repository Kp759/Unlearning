#!/usr/bin/env python3
"""CE-U style answer-token unlearning for MCF and TOFU.

This script minimizes a cross-entropy/KL-style objective whose stop-gradient
forget target distribution excludes the gold answer token. It supports full
model training, embedding/lm-head-only training, and selective embedding/lm-head
row updates.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
COMMON_FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "for",
    "with", "by", "from", "at", "as", "is", "are", "was", "were", "be", "been", "being", "it",
    "its", "this", "that", "these", "those", "he", "she", "they", "them", "his", "her", "their",
    "i", "you", "we", "not", "no", "yes", "do", "does", "did", "has", "have", "had", "will",
    "would", "can", "could", "should", "may", "might", "must", "also", "there", "here", "who",
    "what", "when", "where", "why", "how", "which", "than", "into", "about", "over", "under",
}


@dataclass
class Example:
    prompt: str
    answer: str
    subject: str = ""
    target_new: str = ""
    target_true: str = ""
    paraphrase_prompts: Optional[List[str]] = None
    source: str = ""


@dataclass
class LossResult:
    loss: torch.Tensor
    contributing_tokens: int
    fallback_examples: int = 0


@dataclass
class ParamSummary:
    n_trainable_params: int
    n_total_params: int
    trainable_param_percent: float
    trainable_names: List[str]


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def normalize_answer(text: str) -> str:
    text = str(text).strip()
    return text if text.startswith(" ") else " " + text


def token_ids_for_text(tok: AutoTokenizer, text: str) -> List[int]:
    return [int(x) for x in tok(text, add_special_tokens=False)["input_ids"]]


def special_token_ids(tok: AutoTokenizer) -> set[int]:
    ids = set(getattr(tok, "all_special_ids", []) or [])
    ids.update(x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id] if x is not None)
    return {int(x) for x in ids if x is not None}


def answer_with_eos(tok: AutoTokenizer, answer: str) -> str:
    return answer + tok.eos_token if tok.eos_token and not answer.endswith(tok.eos_token) else answer


def format_mcf_prompt(template: str, subject: str) -> str:
    return template.format(subject) if "{}" in template else template


def extract_mcf_rewrite(record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        rr = rr[0] if rr else None
    if not isinstance(rr, dict):
        raise ValueError("MCF record missing dict requested_rewrite field")
    missing = [k for k in ("prompt", "subject", "target_new") if k not in rr]
    if missing:
        raise ValueError(f"MCF requested_rewrite missing required fields: {missing}")
    if not isinstance(rr.get("target_new"), dict) or "str" not in rr["target_new"]:
        raise ValueError("MCF requested_rewrite.target_new.str is required")
    paraphrases = record.get("paraphrase_prompts") or rr.get("paraphrase_prompts") or []
    return rr, [str(p) for p in paraphrases] if isinstance(paraphrases, list) else []


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, path)


def load_mcf(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    cache_path = resolve_path(args.mcf_cache_path)
    download_if_missing(args.mcf_url, cache_path)
    raw = json.loads(cache_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("MCF cache JSON must contain a list of records")
    examples: List[Example] = []
    for rec in raw:
        rr, paras = extract_mcf_rewrite(rec)
        subject = str(rr["subject"])
        target_new = normalize_answer(rr["target_new"]["str"])
        true_raw = rr.get("target_true", {})
        target_true = normalize_answer(true_raw.get("str", "")) if isinstance(true_raw, dict) and true_raw.get("str") else ""
        if args.mcf_answer_field == "target_true" and not target_true:
            raise ValueError("--mcf-answer-field target_true requested but a record lacks target_true.str")
        answer = target_true if args.mcf_answer_field == "target_true" else target_new
        examples.append(Example(
            prompt=format_mcf_prompt(str(rr["prompt"]), subject), answer=answer, subject=subject,
            target_new=target_new, target_true=target_true,
            paraphrase_prompts=[format_mcf_prompt(p, subject) for p in paras], source="mcf"))
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    need = args.forget_num + args.retain_num
    if len(examples) < need:
        raise ValueError(f"MCF contains {len(examples)} examples, need {need}")
    return examples[: args.forget_num], examples[args.forget_num: need]


def load_tofu(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    def convert(row: Dict[str, Any]) -> Example:
        if "question" not in row or "answer" not in row:
            raise ValueError("TOFU rows must contain question and answer")
        return Example(prompt=f"Question: {row['question']}\nAnswer:", answer=normalize_answer(row["answer"]), source="tofu")

    forget = [convert(r) for r in load_dataset("locuslab/TOFU", name=args.forget_split, split="train")]
    retain = [convert(r) for r in load_dataset("locuslab/TOFU", name=args.retain_split, split="train")]
    rng = random.Random(args.seed)
    rng.shuffle(forget); rng.shuffle(retain)
    return forget[: args.forget_num], retain[: args.retain_num]


def load_data(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    return load_mcf(args) if args.dataset == "mcf" else load_tofu(args)


def is_bad_selective_token(tok: AutoTokenizer, token_id: int, specials: set[int]) -> bool:
    if token_id in specials:
        return True
    text = tok.decode([token_id], skip_special_tokens=True).strip().lower()
    return (not text) or bool(PUNCT_RE.fullmatch(text)) or text in COMMON_FUNCTION_WORDS


def select_contrast_tokens(tok: AutoTokenizer, forget: Sequence[Example], retain: Sequence[Example], top_k: int) -> List[int]:
    specials = special_token_ids(tok)
    forget_df: Counter[int] = Counter()
    retain_df: Counter[int] = Counter()
    for ex in forget:
        forget_df.update(set(t for t in token_ids_for_text(tok, ex.answer) if t not in specials))
    for ex in retain:
        retain_df.update(set(t for t in token_ids_for_text(tok, ex.answer) if t not in specials))
    n_forget, n_retain = max(len(forget), 1), max(len(retain), 1)
    scored = []
    for tid, df in forget_df.items():
        if is_bad_selective_token(tok, tid, specials):
            continue
        score = (df / n_forget) / ((retain_df.get(tid, 0) / n_retain) + 1e-6)
        scored.append((score, tid, tok.decode([tid], skip_special_tokens=True)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [tid for _, tid, _ in scored[:top_k]]


def build_batch(tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> Dict[str, torch.Tensor]:
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    for ex in examples:
        prompt_ids = token_ids_for_text(tok, ex.prompt)
        answer_ids = token_ids_for_text(tok, answer_with_eos(tok, ex.answer))
        if not answer_ids:
            raise ValueError(f"Example has zero answer tokens: {ex.prompt[:100]!r}")
        rows.append(prompt_ids + answer_ids)
        labels.append([-100] * len(prompt_ids) + answer_ids)
    max_len = max(len(r) for r in rows)
    input_ids, label_ids, attention = [], [], []
    for row, lab in zip(rows, labels):
        pad_n = max_len - len(row)
        input_ids.append(row + [pad_id] * pad_n)
        label_ids.append(lab + [-100] * pad_n)
        attention.append([1] * len(row) + [0] * pad_n)
    return {"input_ids": torch.tensor(input_ids, device=device, dtype=torch.long),
            "labels": torch.tensor(label_ids, device=device, dtype=torch.long),
            "attention_mask": torch.tensor(attention, device=device, dtype=torch.long)}


def shifted_answer_mask(labels: torch.Tensor, ignore_first: bool) -> torch.Tensor:
    mask = labels.ne(-100)
    if ignore_first:
        for i in range(mask.size(0)):
            idx = torch.nonzero(mask[i], as_tuple=False)
            if idx.numel() > 0:
                mask[i, int(idx[0].item())] = False
    return mask


def restrict_to_selected(labels: torch.Tensor, answer_mask: torch.Tensor, selected_ids: Optional[set[int]]) -> Tuple[torch.Tensor, int]:
    if not selected_ids:
        return answer_mask, 0
    selected = torch.zeros_like(answer_mask)
    selected_tensor = torch.tensor(sorted(selected_ids), device=labels.device, dtype=labels.dtype)
    valid_labels = labels.clamp_min(0)
    selected |= torch.isin(valid_labels, selected_tensor) & answer_mask
    fallback = 0
    for i in range(labels.size(0)):
        if answer_mask[i].any() and not selected[i].any():
            selected[i] = answer_mask[i]
            fallback += 1
    return selected, fallback


def answer_ce_loss(logits: torch.Tensor, labels_full: torch.Tensor, ignore_first: bool = False) -> LossResult:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels_full[:, 1:].contiguous()
    answer_mask = shifted_answer_mask(shift_labels, ignore_first)
    per_token = F.cross_entropy(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100, reduction="none").view_as(shift_labels)
    denom = answer_mask.sum().clamp_min(1)
    return LossResult((per_token * answer_mask.float()).sum() / denom, int(denom.item()), 0)


def ceu_forget_loss(logits: torch.Tensor, labels_full: torch.Tensor, selected_ids: Optional[set[int]], ignore_first: bool, selected_forget_only: bool) -> LossResult:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels_full[:, 1:]
    answer_mask = shifted_answer_mask(shift_labels, ignore_first)
    fallback = 0
    if selected_forget_only and selected_ids:
        answer_mask, fallback = restrict_to_selected(shift_labels, answer_mask, selected_ids)
    safe_labels = shift_labels.clamp_min(0)
    z_masked = shift_logits.detach().float().clone()
    z_masked.scatter_(dim=-1, index=safe_labels.unsqueeze(-1), value=-1e9)
    q = F.softmax(z_masked, dim=-1).detach()
    logp = F.log_softmax(shift_logits.float(), dim=-1)
    per_token = -(q * logp).sum(dim=-1)
    denom = answer_mask.sum().clamp_min(1)
    return LossResult((per_token * answer_mask.float()).sum() / denom, int(denom.item()), fallback)


def retain_kl_loss(model_logits: torch.Tensor, ref_logits: torch.Tensor, labels_full: torch.Tensor, temperature: float) -> torch.Tensor:
    cur = model_logits[:, :-1, :].float() / temperature
    ref = ref_logits[:, :-1, :].float() / temperature
    mask = labels_full[:, 1:].ne(-100)
    ref_p = F.softmax(ref, dim=-1).detach()
    cur_logp = F.log_softmax(cur, dim=-1)
    ref_logp = F.log_softmax(ref, dim=-1).detach()
    kl = (ref_p * (ref_logp - cur_logp)).sum(dim=-1) * (temperature ** 2)
    return (kl * mask.float()).sum() / mask.sum().clamp_min(1)


def sample_batch(examples: Sequence[Example], size: int) -> List[Example]:
    if not examples:
        raise ValueError("Cannot sample from empty examples")
    return [examples[random.randrange(len(examples))] for _ in range(min(size, len(examples)))]


def configure_trainable(model: torch.nn.Module, mode: str, selected_ids: Sequence[int]) -> Tuple[ParamSummary, Dict[str, Any]]:
    total = sum(p.numel() for p in model.parameters())
    info: Dict[str, Any] = {"tied": False, "hooks": []}
    if mode == "full_all_tokens":
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        inp = model.get_input_embeddings(); out = model.get_output_embeddings()
        if inp is None or out is None:
            raise ValueError("Model lacks input or output embeddings")
        in_w, out_w = inp.weight, out.weight
        tied = in_w.data_ptr() == out_w.data_ptr()
        in_w.requires_grad_(True)
        if not tied:
            out_w.requires_grad_(True)
        info.update({"input_weight": in_w, "output_weight": out_w, "tied": tied})
        if mode == "emb_lm_selective_tokens":
            def hook_for(weight: torch.nn.Parameter):
                row_mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
                ids = torch.tensor([i for i in selected_ids if 0 <= i < weight.shape[0]], dtype=torch.long, device=weight.device)
                if ids.numel() > 0:
                    row_mask[ids] = True
                def hook(grad: torch.Tensor) -> torch.Tensor:
                    grad = grad.clone(); grad[~row_mask] = 0; return grad
                return hook
            info["hooks"].append(in_w.register_hook(hook_for(in_w)))
            if not tied:
                info["hooks"].append(out_w.register_hook(hook_for(out_w)))
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return ParamSummary(trainable, total, 100.0 * trainable / max(total, 1), names), info


def unique_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    seen: set[int] = set(); params: List[torch.nn.Parameter] = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p)); params.append(p)
    return params


def make_optimizer(params: Sequence[torch.nn.Parameter], args: argparse.Namespace) -> torch.optim.Optimizer:
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr)
    return torch.optim.AdamW(params, lr=args.lr)


def load_model_and_tokenizer(model_path: str, dtype: torch.dtype, device: str, training: bool):
    if not model_path:
        raise ValueError("--model-path is required")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model.to(torch.device(device))
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    if training:
        model.config.use_cache = False
    return model, tok


def run_tofu_eval(args: argparse.Namespace, checkpoint: Path) -> None:
    eval_script = SCRIPT_DIR / "tofu_eval.py"
    if not eval_script.exists():
        print(f"WARNING: missing {eval_script}; skipping TOFU eval", file=sys.stderr); return
    out_dir = resolve_path(args.output_dir) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(eval_script), "--model-path", str(checkpoint), "--output-dir", str(out_dir),
           "--forget-split", args.forget_split, "--retain-split", args.retain_split,
           "--eval-n-forget", str(args.eval_n_forget), "--eval-n-retain", str(args.eval_n_retain),
           "--eval-n-real-authors", str(args.eval_n_real_authors), "--eval-n-world-facts", str(args.eval_n_world_facts),
           "--eval-n-perturbed", str(args.eval_n_perturbed), "--eval-max-new-tokens", str(args.eval_max_new_tokens)]
    subprocess.run(cmd, check=True)


def run_mcf_official_eval(args: argparse.Namespace, checkpoint: Path) -> None:
    if not args.official_eval:
        return
    try:
        from mcf_zero_unlearn_official_eval import evaluate_model_dir_official, result_to_comparison_row, write_official_comparison
    except Exception as exc:
        print(f"WARNING: cannot import MCF official eval helper: {exc}", file=sys.stderr); return
    out = resolve_path(args.output_dir)
    base = evaluate_model_dir_official(args.model_path, args.official_unlearn_num, args.official_retain_num, args.official_sample_mode)
    ckpt = evaluate_model_dir_official(str(checkpoint), args.official_unlearn_num, args.official_retain_num, args.official_sample_mode)
    rows = [result_to_comparison_row("base", base), result_to_comparison_row("checkpoint", ckpt)]
    write_official_comparison(rows, out / "official_eval_comparison.csv", out / "official_eval_comparison.md")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CE-U answer-token unlearning for MCF and TOFU")
    p.add_argument("--dataset", choices=["mcf", "tofu"], default="tofu")
    p.add_argument("--model-path", required=True)
    p.add_argument("--reference-model-path", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--forget-split", default="forget05"); p.add_argument("--retain-split", default="retain95")
    p.add_argument("--forget-num", type=int, default=200); p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--mcf-cache-path", default="data/multi_counterfact.json")
    p.add_argument("--mcf-url", default="https://memit.baulab.info/data/dsets/multi_counterfact.json")
    p.add_argument("--mcf-answer-field", choices=["target_true", "target_new"], default="target_true")
    p.add_argument("--mode", choices=["full_all_tokens", "emb_lm_all_tokens", "emb_lm_selective_tokens"], default="emb_lm_selective_tokens")
    p.add_argument("--selective-top-k", type=int, default=300)
    p.add_argument("--steps", type=int, default=100); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--retain-batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-6); p.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"); p.add_argument("--device", default="cuda")
    p.add_argument("--ceu-weight", type=float, default=1.0); p.add_argument("--retain-weight", type=float, default=1.0); p.add_argument("--retain-kl-weight", type=float, default=0.1)
    p.add_argument("--kl-temperature", type=float, default=1.0); p.add_argument("--ignore-first-answer-token", action="store_true", default=False); p.add_argument("--selected-forget-only", action="store_true", default=False)
    p.add_argument("--save-model", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--eval-n-forget", type=int, default=200); p.add_argument("--eval-n-retain", type=int, default=400); p.add_argument("--eval-n-real-authors", type=int, default=100)
    p.add_argument("--eval-n-world-facts", type=int, default=117); p.add_argument("--eval-n-perturbed", type=int, default=100); p.add_argument("--eval-max-new-tokens", type=int, default=64)
    p.add_argument("--official-eval", action=argparse.BooleanOptionalAction, default=True); p.add_argument("--official-unlearn-num", type=int, default=50); p.add_argument("--official-retain-num", type=int, default=1000); p.add_argument("--official-sample-mode", default="official")
    return p.parse_args()


def main() -> None:
    args = parse_args(); set_seed(args.seed)
    out_dir = resolve_path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model, tok = load_model_and_tokenizer(args.model_path, torch_dtype(args.dtype), args.device, True)
    forget, retain = load_data(args)
    selected_ids = select_contrast_tokens(tok, forget, retain, args.selective_top_k) if args.mode == "emb_lm_selective_tokens" else []
    write_json(out_dir / "selected_tokens.json", {"token_ids": selected_ids, "tokens": [tok.decode([i], skip_special_tokens=True) for i in selected_ids]})
    summary, train_info = configure_trainable(model, args.mode, selected_ids)
    params = unique_trainable_params(model); opt = make_optimizer(params, args); device = next(model.parameters()).device
    ref_model = None
    if args.retain_kl_weight > 0:
        ref_path = args.reference_model_path or args.model_path
        ref_model, _ = load_model_and_tokenizer(ref_path, torch_dtype(args.dtype), args.device, False)
        ref_model.eval(); [p.requires_grad_(False) for p in ref_model.parameters()]
    selected_set = set(selected_ids) if selected_ids else None
    model.train(); last_row: Dict[str, Any] = {}
    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(1, args.steps + 1), desc="CE-U train"):
            fb = build_batch(tok, sample_batch(forget, args.batch_size), device)
            rb = build_batch(tok, sample_batch(retain, args.retain_batch_size), device)
            opt.zero_grad(set_to_none=True)
            f_logits = model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"]).logits
            ceu = ceu_forget_loss(f_logits, fb["labels"], selected_set, args.ignore_first_answer_token, args.selected_forget_only)
            r_logits = model(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
            retain_ce = answer_ce_loss(r_logits, rb["labels"])
            kl = torch.zeros((), device=device)
            if ref_model is not None:
                with torch.no_grad():
                    ref_logits = ref_model(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"]).logits
                kl = retain_kl_loss(r_logits, ref_logits, rb["labels"], args.kl_temperature)
            total = args.ceu_weight * ceu.loss + args.retain_weight * retain_ce.loss + args.retain_kl_weight * kl
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip) if args.grad_clip and args.grad_clip > 0 else torch.zeros(())
            opt.step()
            last_row = {"step": step, "dataset": args.dataset, "mode": args.mode, "total_loss": float(total.detach().cpu()),
                        "ceu_forget_loss": float(ceu.loss.detach().cpu()), "retain_ce": float(retain_ce.loss.detach().cpu()), "retain_kl": float(kl.detach().cpu()),
                        "forget_tokens": ceu.contributing_tokens, "retain_tokens": retain_ce.contributing_tokens,
                        "forget_fallback_examples": ceu.fallback_examples, "retain_fallback_examples": retain_ce.fallback_examples,
                        "grad_norm": float(grad_norm.detach().cpu()) if torch.is_tensor(grad_norm) else float(grad_norm),
                        "n_selected_ids": len(selected_ids), "tied_embeddings": bool(train_info.get("tied")), "lr": args.lr,
                        "ceu_weight": args.ceu_weight, "retain_weight": args.retain_weight, "retain_kl_weight": args.retain_kl_weight}
            log_f.write(json.dumps(last_row) + "\n"); log_f.flush()
    ckpt = out_dir / "checkpoint"
    if args.save_model:
        model.save_pretrained(ckpt); tok.save_pretrained(ckpt)
    train_summary = {"args": vars(args), "n_forget": len(forget), "n_retain": len(retain), "n_selected_ids": len(selected_ids), "tied_embeddings": bool(train_info.get("tied")), "param_summary": asdict(summary), "last_step": last_row}
    write_json(out_dir / "train_summary.json", train_summary)
    if not args.skip_eval and args.save_model:
        run_mcf_official_eval(args, ckpt) if args.dataset == "mcf" else run_tofu_eval(args, ckpt)
    print(f"cat {out_dir / 'train_summary.json'}")
    print(f"tail -n 5 {out_dir / 'train_log.jsonl'}")
    if args.dataset == "mcf":
        print(f"cat {out_dir / 'official_eval_comparison.md'}")
    else:
        print(f"cat {out_dir / 'eval' / 'tofu_eco_summary.csv'}")


if __name__ == "__main__":
    main()
