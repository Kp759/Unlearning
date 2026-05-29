#!/usr/bin/env python3
"""
Full-model answer-only TOFU unlearning.

Updates the whole model, not selected token rows.

Objectives:
  ga      : L = log p_theta(y_f|x_f)
  ga_gd   : L = log p_theta(y_f|x_f) - lambda log p_theta(y_r|x_r)
  ga_kl   : L = log p_theta(y_f|x_f) + lambda KL(p_ref||p_theta on retain)
  npo_gd  : L = NPO(D_f) + lambda CE(D_r)
  npo_kl  : L = NPO(D_f) + lambda KL(p_ref||p_theta on retain)

All losses are answer-token only because prompt labels are -100.
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


def resolve_dtype(x: str) -> torch.dtype:
    x = str(x).lower()
    if x in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if x in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def encode_answer_only(tok, question: str, answer: str, max_length: int) -> Dict[str, List[int]]:
    prompt_ids = tok.encode(format_prompt(question), add_special_tokens=True)
    answer_text = " " + str(answer).strip()
    if tok.eos_token:
        answer_text += tok.eos_token
    answer_ids = tok.encode(answer_text, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def collate(examples: Sequence[Dict[str, List[int]]], pad_id: int, device: torch.device):
    max_len = max(len(x["input_ids"]) for x in examples)
    input_ids, attention_mask, labels = [], [], []

    for x in examples:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attention_mask.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def sample_batch(rng, encoded, bs: int, pad_id: int, device: torch.device):
    return collate([encoded[rng.randrange(len(encoded))] for _ in range(bs)], pad_id, device)


def answer_logp(model, batch, reduction: str = "mean") -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].contiguous().float()
    labels = batch["labels"][:, 1:].contiguous()

    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)
    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1) * valid.float()

    if reduction == "sum":
        return tok_logp.sum(dim=1)

    denom = valid.float().sum(dim=1).clamp_min(1.0)
    return tok_logp.sum(dim=1) / denom


def ce_loss(model, batch) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss


def kl_retain(model, ref, batch, temperature: float = 1.0) -> torch.Tensor:
    with torch.no_grad():
        ref_logits = ref(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

    cur_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits

    cur = cur_logits[:, :-1, :].contiguous().float() / temperature
    old = ref_logits[:, :-1, :].contiguous().float() / temperature
    labels = batch["labels"][:, 1:].contiguous()
    valid = labels.ne(-100).float()

    cur_logp = F.log_softmax(cur, dim=-1)
    old_p = F.softmax(old, dim=-1)
    token_kl = F.kl_div(cur_logp, old_p, reduction="none").sum(dim=-1)

    return (token_kl * valid).sum() / valid.sum().clamp_min(1.0) * (temperature ** 2)


def npo_loss(model, ref, batch, beta: float, logp_reduction: str):
    cur_logp = answer_logp(model, batch, reduction=logp_reduction)
    with torch.no_grad():
        ref_logp = answer_logp(ref, batch, reduction=logp_reduction)
    diff = cur_logp - ref_logp
    loss = -(2.0 / beta) * F.logsigmoid(-beta * diff).mean()
    return loss, cur_logp.detach().mean(), ref_logp.detach().mean()


def make_optimizer(model, name: str, lr: float, wd: float):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adamw_torch":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    if name in {"adamw8bit", "paged_adamw8bit"}:
        try:
            import bitsandbytes as bnb
        except Exception as e:
            raise RuntimeError("bitsandbytes not available. Use --optimizer adamw_torch.") from e
        return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=wd)

    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--objective", choices=["ga", "ga_gd", "ga_kl", "npo_gd", "npo_kl"], default="ga_gd")

    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--max-length", type=int, default=None)

    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--retain-batch-size", type=int, default=1)
    ap.add_argument("--grad-accum-steps", type=int, default=4)

    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optimizer", choices=["adamw_torch", "adamw8bit", "paged_adamw8bit"], default="adamw_torch")

    ap.add_argument("--forget-loss-weight", type=float, default=1.0)
    ap.add_argument("--retain-loss-weight", type=float, default=1.0)
    ap.add_argument("--npo-beta", type=float, default=1.0)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--logp-reduction", choices=["mean", "sum"], default="mean")

    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--max-forget-train-samples", type=int, default=None)
    ap.add_argument("--max-retain-train-samples", type=int, default=800)
    ap.add_argument("--save-every", type=int, default=0)
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
    print("Full-model answer-only unlearning")
    print("=" * 80)
    print(f"model-dir: {args.model_dir}")
    print(f"output-dir: {args.output_dir}")
    print(f"objective: {args.objective}")
    print(f"forget split: {forget_split}")
    print(f"retain split: {retain_split}")
    print(f"optimizer: {args.optimizer}")
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

    needs_ref = args.objective in {"ga_kl", "npo_gd", "npo_kl"}
    ref = None
    if needs_ref:
        print("[Load] reference model")
        ref = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        ref.config.use_cache = False
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    print("Loading TOFU...")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    forget_rows = list(forget_ds)
    retain_rows = list(retain_ds)

    if args.max_forget_train_samples is not None:
        forget_rows = forget_rows[:args.max_forget_train_samples]
    if args.max_retain_train_samples is not None:
        retain_rows = retain_rows[:args.max_retain_train_samples]

    print("Encoding...")
    forget_encoded = [encode_answer_only(tok, r["question"], r["answer"], max_length) for r in forget_rows]
    retain_encoded = [encode_answer_only(tok, r["question"], r["answer"], max_length) for r in retain_rows]

    opt = make_optimizer(model, args.optimizer, args.lr, args.weight_decay)

    last = {"loss": 0.0, "forget": 0.0, "retain": 0.0, "cur_logp": 0.0, "ref_logp": 0.0}
    pbar = tqdm(range(args.steps), desc=f"full_model_{args.objective}")

    for step in pbar:
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0

        for _ in range(args.grad_accum_steps):
            fb = sample_batch(rng, forget_encoded, args.batch_size, tok.pad_token_id, device)

            rb = None
            if args.objective != "ga":
                rb = sample_batch(rng, retain_encoded, args.retain_batch_size, tok.pad_token_id, device)

            if args.objective == "ga":
                f_lp = answer_logp(model, fb, args.logp_reduction).mean()
                loss = args.forget_loss_weight * f_lp
                last["forget"] = float(f_lp.detach().cpu())

            elif args.objective == "ga_gd":
                f_lp = answer_logp(model, fb, args.logp_reduction).mean()
                r_lp = answer_logp(model, rb, args.logp_reduction).mean()
                loss = args.forget_loss_weight * f_lp - args.retain_loss_weight * r_lp
                last["forget"] = float(f_lp.detach().cpu())
                last["retain"] = float(r_lp.detach().cpu())

            elif args.objective == "ga_kl":
                f_lp = answer_logp(model, fb, args.logp_reduction).mean()
                kl = kl_retain(model, ref, rb, args.kl_temperature)
                loss = args.forget_loss_weight * f_lp + args.retain_loss_weight * kl
                last["forget"] = float(f_lp.detach().cpu())
                last["retain"] = float(kl.detach().cpu())

            elif args.objective == "npo_gd":
                npo, cur, old = npo_loss(model, ref, fb, args.npo_beta, args.logp_reduction)
                r_ce = ce_loss(model, rb)
                loss = args.forget_loss_weight * npo + args.retain_loss_weight * r_ce
                last["forget"] = float(npo.detach().cpu())
                last["retain"] = float(r_ce.detach().cpu())
                last["cur_logp"] = float(cur.detach().cpu())
                last["ref_logp"] = float(old.detach().cpu())

            elif args.objective == "npo_kl":
                npo, cur, old = npo_loss(model, ref, fb, args.npo_beta, args.logp_reduction)
                kl = kl_retain(model, ref, rb, args.kl_temperature)
                loss = args.forget_loss_weight * npo + args.retain_loss_weight * kl
                last["forget"] = float(npo.detach().cpu())
                last["retain"] = float(kl.detach().cpu())
                last["cur_logp"] = float(cur.detach().cpu())
                last["ref_logp"] = float(old.detach().cpu())

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
                "forget": f"{last['forget']:.4f}",
                "retain": f"{last['retain']:.4f}",
                "cur_lp": f"{last['cur_logp']:.4f}",
            })

        if args.save_every and args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt = Path(args.output_dir) / f"checkpoint_step_{step+1}"
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)

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
        "npo_beta": args.npo_beta,
        "kl_temperature": args.kl_temperature,
        "logp_reduction": args.logp_reduction,
        "gradient_checkpointing": args.gradient_checkpointing,
        "final_logs": last,
    }
    with open(out / "full_model_unlearn_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done]")
    print(
        f"python scripts/tofu_eval.py --config {args.config} "
        f"--model-dir {out} --method full_model_{args.objective}_answer_only "
        f"--forget-split {forget_split} --retain-split {retain_split}"
    )


if __name__ == "__main__":
    main()
