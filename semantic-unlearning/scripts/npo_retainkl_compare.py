#!/usr/bin/env python3
"""Standalone NPO + retain CE + retain KL experiment for MCF.

This script intentionally does not import gagd_compare.py.  It trains one or
more semantic-unlearning scopes on a seed split of Multi-CounterFact and writes
per-mode logs plus aggregate comparison files.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MCF_URL = "https://memit.baulab.info/data/dsets/multi_counterfact.json"
MODES = ["full_all_tokens", "full_selective_tokens", "emb_lm_all_tokens", "emb_lm_selective_tokens"]


def dtype_from_arg(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def ensure_mcf(path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Data] {path} missing; downloading {DEFAULT_MCF_URL}")
    urllib.request.urlretrieve(DEFAULT_MCF_URL, path)
    return path


def read_mcf(path: Path, forget_num: int, retain_num: int, seed: int) -> Tuple[List[dict], List[dict]]:
    ensure_mcf(path)
    records = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(records)
    need = forget_num + retain_num
    if len(records) < need:
        raise ValueError(f"MCF has {len(records)} records but {need} are required")
    return records[:forget_num], records[forget_num:need]


def format_prompt(row: dict) -> str:
    rr = row["requested_rewrite"]
    prompt = rr["prompt"]
    subject = rr["subject"]
    try:
        return prompt.format(subject)
    except Exception:
        return prompt.replace("{}", subject)


def answer(row: dict, key: str = "target_new") -> str:
    return row["requested_rewrite"][key]["str"]


def make_examples(rows: Sequence[dict], key: str = "target_new") -> List[Tuple[str, str]]:
    examples: List[Tuple[str, str]] = []
    for row in rows:
        examples.append((format_prompt(row), answer(row, key)))
        for p in row.get("paraphrase_prompts", []) or []:
            examples.append((p, answer(row, key)))
        for p in row.get("neighborhood_prompts", []) or []:
            examples.append((p, answer(row, key)))
    return examples


def collect_selected_token_ids(tokenizer, forget_rows: Sequence[dict]) -> List[int]:
    specials = {x for x in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id] if x is not None}
    selected = set()
    for row in forget_rows:
        rr = row["requested_rewrite"]
        for text in [rr["subject"], rr["target_new"]["str"], rr["target_true"]["str"]]:
            selected.update(tokenizer.encode(text, add_special_tokens=False))
    return sorted(t for t in selected if t not in specials)


def encode_batch(tokenizer, pairs: Sequence[Tuple[str, str]], device: torch.device) -> Dict[str, torch.Tensor]:
    texts, labels = [], []
    for prompt, ans in pairs:
        prefix = prompt.rstrip() + " "
        full = prefix + ans
        full_ids = tokenizer(full, add_special_tokens=True).input_ids
        prefix_ids = tokenizer(prefix, add_special_tokens=True).input_ids
        lab = [-100] * len(full_ids)
        start = min(len(prefix_ids), len(full_ids))
        lab[start:] = full_ids[start:]
        texts.append(full_ids)
        labels.append(lab)
    max_len = max(len(x) for x in texts)
    pad = tokenizer.pad_token_id
    padded, masks, padded_labels = [], [], []
    for ids, lab in zip(texts, labels):
        n = max_len - len(ids)
        padded.append(ids + [pad] * n)
        masks.append([1] * len(ids) + [0] * n)
        padded_labels.append(lab + [-100] * n)
    return {
        "input_ids": torch.tensor(padded, device=device),
        "attention_mask": torch.tensor(masks, device=device),
        "labels": torch.tensor(padded_labels, device=device),
    }


def answer_logp(model, batch: Dict[str, torch.Tensor], reduction: str, allowed_ids: set[int] | None = None) -> torch.Tensor:
    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    valid = labels.ne(-100)
    if allowed_ids is not None:
        allowed = torch.zeros_like(valid)
        for tid in allowed_ids:
            allowed |= labels.eq(tid)
        valid &= allowed
    safe = labels.masked_fill(~valid, 0)
    tok = torch.log_softmax(logits, -1).gather(-1, safe.unsqueeze(-1)).squeeze(-1) * valid.float()
    denom = valid.float().sum(1).clamp_min(1.0)
    return tok.sum(1) if reduction == "sum" else tok.sum(1) / denom


def ce_loss(model, batch: Dict[str, torch.Tensor], allowed_ids: set[int] | None = None) -> torch.Tensor:
    labels = batch["labels"]
    if allowed_ids is not None:
        keep = torch.zeros_like(labels, dtype=torch.bool)
        for tid in allowed_ids:
            keep |= labels.eq(tid)
        labels = labels.masked_fill(~keep, -100)
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=labels).loss


def retain_kl_loss(model, ref, batch: Dict[str, torch.Tensor], ref_batch: Dict[str, torch.Tensor], direction: str, answer_only: bool, allowed_ids: set[int] | None = None) -> torch.Tensor:
    cur_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :].float()
    with torch.no_grad():
        ref_logits = ref(input_ids=ref_batch["input_ids"], attention_mask=ref_batch["attention_mask"]).logits[:, :-1, :].float().to(cur_logits.device)
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100) if answer_only else batch["attention_mask"][:, 1:].bool()
    if allowed_ids is not None:
        selected = torch.zeros_like(mask)
        for tid in allowed_ids:
            selected |= labels.eq(tid)
        mask &= selected
    cur_lp, ref_lp = F.log_softmax(cur_logits, -1), F.log_softmax(ref_logits, -1)
    cur_p, ref_p = cur_lp.exp(), ref_lp.exp()
    tok = (ref_p * (ref_lp - cur_lp)).sum(-1) if direction == "ref_current" else (cur_p * (cur_lp - ref_lp)).sum(-1)
    return (tok * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def freeze_for_mode(model, mode: str) -> None:
    for p in model.parameters():
        p.requires_grad_(mode.startswith("full"))
    if mode.startswith("emb_lm"):
        for p in model.get_input_embeddings().parameters():
            p.requires_grad_(True)
        for p in model.get_output_embeddings().parameters():
            p.requires_grad_(True)


def make_optimizer(name: str, params: Iterable[nn.Parameter], lr: float):
    params = [p for p in params if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(params, lr=lr)
        except Exception as exc:
            print(f"[Warn] adamw8bit unavailable ({exc}); falling back to AdamW")
    return torch.optim.AdamW(params, lr=lr)


def load_ref_model(args, dtype):
    kwargs = {"torch_dtype": dtype}
    if args.reference_device == "auto":
        try:
            return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).to("cuda" if torch.cuda.is_available() else "cpu")
        except torch.cuda.OutOfMemoryError:
            print("[Warn] reference CUDA OOM; falling back to CPU")
            torch.cuda.empty_cache(); gc.collect()
            return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).to("cpu")
    if args.reference_device == "auto":
        kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(args.model_path, **kwargs).to(args.reference_device)


def train_mode(args, mode: str, tokenizer, forget_rows, retain_rows, selected_ids: List[int]) -> Dict[str, object]:
    out_dir = Path(args.output_dir) / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = dtype_from_arg(args.dtype)
    device_map = "auto" if args.device_map == "auto" else None
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype, device_map=device_map)
    if args.device_map == "single":
        model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    ref = load_ref_model(args, dtype)
    ref.config.use_cache = False
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    freeze_for_mode(model, mode)
    model.train()
    opt = make_optimizer(args.optimizer, model.parameters(), args.lr)
    rng = random.Random(args.seed)
    forget_ex = make_examples(forget_rows, "target_new")
    retain_ex = make_examples(retain_rows, "target_new")
    allowed = set(selected_ids) if "selective_tokens" in mode else None
    emb, head = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    keep_emb = emb.detach().clone() if mode == "emb_lm_selective_tokens" else None
    keep_head = head.detach().clone() if mode == "emb_lm_selective_tokens" else None
    log_path = out_dir / "train_log.jsonl"
    last = {}
    with log_path.open("w", encoding="utf-8") as logf:
        for step in trange(1, args.steps + 1, desc=mode):
            fb = [rng.choice(forget_ex) for _ in range(args.batch_size)]
            rb = [rng.choice(retain_ex) for _ in range(args.retain_batch_size)]
            dev = model_device(model)
            f_batch = encode_batch(tokenizer, fb, dev)
            r_batch = encode_batch(tokenizer, rb, dev)
            ref_f_batch = {k: v.to(model_device(ref)) for k, v in f_batch.items()}
            ref_r_batch = {k: v.to(model_device(ref)) for k, v in r_batch.items()}
            cur_logp = answer_logp(model, f_batch, args.npo_logp_reduction, allowed)
            with torch.no_grad():
                ref_logp = answer_logp(ref, ref_f_batch, args.npo_logp_reduction, allowed).to(cur_logp.device)
            npo = (2.0 / args.npo_beta) * F.softplus(args.npo_beta * (cur_logp - ref_logp)).mean()
            rce = ce_loss(model, r_batch, allowed)
            rkl = retain_kl_loss(model, ref, r_batch, ref_r_batch, args.retain_kl_direction, args.retain_kl_answer_only, allowed)
            loss = args.forget_weight * npo + args.retain_weight * rce + args.kl_retain_weight * rkl
            opt.zero_grad(set_to_none=True); loss.backward()
            if mode == "emb_lm_selective_tokens":
                mask = torch.zeros(emb.shape[0], dtype=torch.bool, device=emb.device); mask[selected_ids] = True
                if emb.grad is not None: emb.grad[~mask] = 0
                if head.grad is not None: head.grad[~mask.to(head.device)] = 0
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            opt.step()
            if mode == "emb_lm_selective_tokens":
                with torch.no_grad():
                    mask = torch.zeros(emb.shape[0], dtype=torch.bool, device=emb.device); mask[selected_ids] = True
                    emb[~mask] = keep_emb[~mask]
                    hmask = mask.to(head.device); head[~hmask] = keep_head[~hmask]
            last = {"step": step, "loss": float(loss.detach().cpu()), "npo_loss": float(npo.detach().cpu()), "retain_ce_loss": float(rce.detach().cpu()), "retain_kl_loss": float(rkl.detach().cpu()), "forget_policy_logp": float(cur_logp.mean().detach().cpu()), "forget_ref_logp": float(ref_logp.mean().detach().cpu())}
            logf.write(json.dumps(last) + "\n"); logf.flush()
    metrics = {"mode": mode, "selected_token_ids": len(selected_ids), **last}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if args.save_model:
        ckpt = out_dir / "checkpoint"; ckpt.mkdir(exist_ok=True)
        model.save_pretrained(ckpt); tokenizer.save_pretrained(ckpt)
        metrics["checkpoint"] = str(ckpt)
    del model, ref; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return metrics


def write_comparison(out_dir: Path, rows: List[Dict[str, object]]) -> None:
    keys = ["mode", "loss", "npo_loss", "retain_ce_loss", "retain_kl_loss", "forget_policy_logp", "forget_ref_logp", "selected_token_ids"]
    with (out_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows({k: r.get(k, "") for k in keys} for r in rows)
    md = ["| " + " | ".join(keys) + " |", "| " + " | ".join(["---"] * len(keys)) + " |"]
    for r in rows:
        md.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    (out_dir / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def run_official_eval(args, rows: List[Dict[str, object]]) -> None:
    model_dirs = [f"base={args.model_path}"]
    model_dirs += [f"{r['mode']}={Path(args.output_dir) / r['mode'] / 'checkpoint'}" for r in rows if (Path(args.output_dir) / r['mode'] / 'checkpoint').exists()]
    cmd = [sys.executable, str(SCRIPT_DIR / "run_same_mcf_eval.py"), "--model-dirs", *model_dirs, "--mcf-path", args.mcf_path, "--out-dir", str(Path(args.output_dir) / "same_eval_with_base"), "--unlearn-num", str(args.forget_num), "--retain-num", str(args.retain_num), "--seed", str(args.seed), "--sample-mode", args.official_sample_mode, "--dtype", {"bf16":"bfloat16","fp16":"float16","fp32":"float32"}[args.dtype]]
    if args.skip_ppl: cmd.append("--skip-ppl")
    subprocess.run(cmd, check=True)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--mcf-path", default=str(Path("semantic-unlearning/data/mcf/multi_counterfact.json")))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=MODES + ["all"], default="all")
    p.add_argument("--forget-num", type=int, default=50); p.add_argument("--retain-num", type=int, default=1000); p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=300); p.add_argument("--batch-size", type=int, default=1); p.add_argument("--retain-batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5); p.add_argument("--forget-weight", type=float, default=1.0); p.add_argument("--retain-weight", type=float, default=1.0); p.add_argument("--kl-retain-weight", type=float, default=1.0)
    p.add_argument("--npo-beta", type=float, default=0.1); p.add_argument("--npo-logp-reduction", choices=["sum", "mean"], default="sum"); p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"); p.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "adamw8bit"], default="adamw")
    p.add_argument("--device-map", choices=["single", "auto"], default="single"); p.add_argument("--reference-device", choices=["cuda", "cpu", "auto"], default="auto")
    p.add_argument("--retain-kl-direction", choices=["ref_current", "current_ref"], default="ref_current")
    g = p.add_mutually_exclusive_group(); g.add_argument("--retain-kl-answer-only", dest="retain_kl_answer_only", action="store_true", default=True); g.add_argument("--no-retain-kl-answer-only", dest="retain_kl_answer_only", action="store_false")
    p.add_argument("--gradient-checkpointing", action="store_true"); p.add_argument("--save-model", action="store_true"); p.add_argument("--run-official-mcf-eval", action="store_true")
    p.add_argument("--official-sample-mode", choices=["official", "first"], default="official"); p.add_argument("--skip-ppl", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config_used.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    random.seed(args.seed); torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    forget_rows, retain_rows = read_mcf(Path(args.mcf_path), args.forget_num, args.retain_num, args.seed)
    selected_ids = collect_selected_token_ids(tokenizer, forget_rows)
    print(f"Selected MCF token IDs: {len(selected_ids)}")
    rows = []
    for mode in (MODES if args.mode == "all" else [args.mode]):
        rows.append(train_mode(args, mode, tokenizer, forget_rows, retain_rows, selected_ids))
    write_comparison(out_dir, rows)
    if args.run_official_mcf_eval:
        run_official_eval(args, rows)


if __name__ == "__main__":
    main()
