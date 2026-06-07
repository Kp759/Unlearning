#!/usr/bin/env python3
"""
Full-model answer-only unlearning with non-saturating unlikelihood loss.

Why this script:
  Plain GA/GD minimizes log p(true forget answer). When p(true token) is already
  near 1, the gradient is proportional to (1 - p), so it is tiny.

Fix:
  Use token-level unlikelihood on forget answers:
      L_UL = - mean_t log(1 - p_theta(y_t^f | prefix))
  This has non-vanishing gradient when p_theta(y_t^f) is high.

Objectives:
  ul_gd     : L = wf * UL(forget answer) + wr * CE(retain answer)
  ul_kl     : L = wf * UL(forget answer) + wk * KL(ref || model on retain answer positions)
  ul_gd_kl  : L = wf * UL(forget) + wr * CE(retain) + wk * KL(retain)
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse stable utilities from the current full-model script.
from full_model_unlearn import (
    resolve_dtype,
    encode_answer_only,
    sample_batch,
    ce_loss,
    kl_retain,
    make_optimizer,
)


def log1mexp(log_x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Stable log(1 - exp(log_x)) for log_x <= 0.
    We clamp only to avoid exact log(0) if softmax returns probability 1.0.
    """
    log_x = torch.clamp(log_x, max=-eps)
    cutoff = -0.6931471805599453  # log(0.5)
    return torch.where(
        log_x < cutoff,
        torch.log1p(-torch.exp(log_x)),
        torch.log(-torch.expm1(log_x)),
    )


def unlikelihood_loss(model, batch: Dict[str, torch.Tensor], reduction: str = "mean") -> torch.Tensor:
    """
    Token-level unlikelihood over answer positions only:
        -log(1 - p(correct forget token))

    reduction='mean': average over answer tokens.
    reduction='sum' : sum over answer tokens per example, then average batch.
    """
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].contiguous().float()
    labels = batch["labels"][:, 1:].contiguous()

    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)

    logp = F.log_softmax(logits, dim=-1)
    correct_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)

    tok_loss = -log1mexp(correct_logp)
    tok_loss = tok_loss * valid.float()

    if reduction == "sum":
        # Sum over answer tokens per example, average over batch.
        return tok_loss.sum(dim=1).mean()

    # Mean over all answer tokens in batch.
    return tok_loss.sum() / valid.float().sum().clamp_min(1.0)


def count_trainable_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--objective", choices=["ul_gd", "ul_kl", "ul_gd_kl"], default="ul_gd_kl")

    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--max-length", type=int, default=None)

    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--retain-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=2)

    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", choices=["adamw_torch", "adamw8bit", "paged_adamw8bit"], default="adamw_torch")

    ap.add_argument("--forget-loss-weight", type=float, default=0.1)
    ap.add_argument("--retain-loss-weight", type=float, default=1.0)
    ap.add_argument("--retain-kl-weight", type=float, default=1.0)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--unlikelihood-reduction", choices=["mean", "sum"], default="mean")

    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--max-forget-train-samples", type=int, default=None)
    ap.add_argument("--max-retain-train-samples", type=int, default=800)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    max_length = args.max_length or cfg["model"].get("max_length", 512)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    print("=" * 80)
    print("Full-model unlikelihood unlearning")
    print("=" * 80)
    print(f"model-dir: {args.model_dir}")
    print(f"output-dir: {args.output_dir}")
    print(f"objective: {args.objective}")
    print(f"forget split: {forget_split}")
    print(f"retain split: {retain_split}")
    print(f"dtype: {args.dtype}")
    print(f"lr: {args.lr}")
    print(f"forget weight: {args.forget_loss_weight}")
    print(f"retain CE weight: {args.retain_loss_weight}")
    print(f"retain KL weight: {args.retain_kl_weight}")
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.train()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        print("[Info] gradient checkpointing enabled")

    needs_ref = args.objective in {"ul_kl", "ul_gd_kl"}
    ref = None
    if needs_ref:
        print("[Load] frozen reference model for retain KL")
        ref = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        ref.config.use_cache = False
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    total, trainable = count_trainable_params(model)
    print(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    print("Loading TOFU...")
    forget_rows = list(load_dataset("locuslab/TOFU", name=forget_split, split="train"))
    retain_rows = list(load_dataset("locuslab/TOFU", name=retain_split, split="train"))

    if args.max_forget_train_samples is not None:
        forget_rows = forget_rows[: args.max_forget_train_samples]
    if args.max_retain_train_samples is not None:
        retain_rows = retain_rows[: args.max_retain_train_samples]

    print("Encoding answer-only examples...")
    forget_encoded = [encode_answer_only(tok, r["question"], r["answer"], max_length) for r in forget_rows]
    retain_encoded = [encode_answer_only(tok, r["question"], r["answer"], max_length) for r in retain_rows]

    opt = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    last = {"loss": 0.0, "ul": 0.0, "retain_ce": 0.0, "retain_kl": 0.0}
    pbar = tqdm(range(args.steps), desc=f"full_model_{args.objective}")

    for step in pbar:
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0

        for _ in range(args.grad_accum_steps):
            fb = sample_batch(rng, forget_encoded, args.batch_size, tok.pad_token_id, device)
            rb = sample_batch(rng, retain_encoded, args.retain_batch_size, tok.pad_token_id, device)

            ul = unlikelihood_loss(model, fb, reduction=args.unlikelihood_reduction)
            loss = args.forget_loss_weight * ul
            last["ul"] = float(ul.detach().cpu())

            if args.objective in {"ul_gd", "ul_gd_kl"}:
                r_ce = ce_loss(model, rb)
                loss = loss + args.retain_loss_weight * r_ce
                last["retain_ce"] = float(r_ce.detach().cpu())

            if args.objective in {"ul_kl", "ul_gd_kl"}:
                r_kl = kl_retain(model, ref, rb, temperature=args.kl_temperature)
                loss = loss + args.retain_kl_weight * r_kl
                last["retain_kl"] = float(r_kl.detach().cpu())

            if not torch.isfinite(loss).all().item():
                raise RuntimeError(f"non-finite loss at step {step}: {loss}")

            (loss / args.grad_accum_steps).backward()
            loss_sum += float(loss.detach().cpu())

        if args.grad_clip_norm and args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

        opt.step()
        last["loss"] = loss_sum / args.grad_accum_steps

        if step % 5 == 0 or step == args.steps - 1:
            pbar.set_postfix({
                "loss": f"{last['loss']:.4f}",
                "ul": f"{last['ul']:.4f}",
                "ret_ce": f"{last['retain_ce']:.4f}",
                "ret_kl": f"{last['retain_kl']:.4f}",
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving final model to {out}")
    model.save_pretrained(out)
    tok.save_pretrained(out)

    summary = {
        "method": f"full_model_{args.objective}_answer_only",
        "objective": args.objective,
        "model_dir": args.model_dir,
        "output_dir": args.output_dir,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "retain_batch_size": args.retain_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "forget_loss_weight": args.forget_loss_weight,
        "retain_loss_weight": args.retain_loss_weight,
        "retain_kl_weight": args.retain_kl_weight,
        "unlikelihood_reduction": args.unlikelihood_reduction,
        "gradient_checkpointing": args.gradient_checkpointing,
        "final_logs": last,
    }
    with open(out / "full_model_unlikelihood_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done]")
    print(
        f"python scripts/tofu_eval.py --config {args.config} "
        f"--model-dir {out} --method full_model_{args.objective}_answer_only "
        f"--forget-split {forget_split} --retain-split {retain_split}"
    )


if __name__ == "__main__":
    main()
