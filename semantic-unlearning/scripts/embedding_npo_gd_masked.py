#!/usr/bin/env python3
"""
Answer-token TF-IDF + masked NPO+GD unlearning.

Method:
  answer_tfidf_npo_gd_lmhead_only

Use after building an answer-token TF-IDF token file, e.g.:
  outputs/semantic_tokens_answer_tfidf_medium_safe.json

This script freezes the model and updates only selected input-embedding and/or
LM-head rows. Default is LM-head only.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from embedding_ga_gd_input_output_unlearn import (
    resolve_dtype,
    first_param_device,
    encode_answer_only,
    sample_batch,
    load_token_ids,
    make_row_mask,
    mask_grad_rows,
    finite_or_raise_loss,
    finite_or_raise_grad,
    finite_or_raise_rows,
    anchor_loss,
    clip_rows,
    max_abs_rows,
    force_untie_lm_head_if_needed,
    freeze_except_token_side,
)


def answer_logp(model, batch: Dict[str, torch.Tensor], reduction: str = "mean") -> torch.Tensor:
    """Return per-example answer log probability."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].contiguous().float()
    labels = batch["labels"][:, 1:].contiguous()

    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)

    logp = torch.log_softmax(logits, dim=-1)
    tok_logp = logp.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    tok_logp = tok_logp * valid.float()

    if reduction == "sum":
        return tok_logp.sum(dim=1)
    if reduction == "mean":
        denom = valid.float().sum(dim=1).clamp_min(1.0)
        return tok_logp.sum(dim=1) / denom
    raise ValueError(f"Unknown logp reduction: {reduction}")


def retain_ce_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss


def npo_loss(model, ref_model, batch: Dict[str, torch.Tensor], beta: float, logp_reduction: str):
    """
    NPO forget loss.

    d = log p_theta(y_f|x_f) - log p_ref(y_f|x_f)
    L_NPO = -(2/beta) * log sigmoid(- beta * d)

    Minimizing this lowers forget-answer log probability, but the gradient
    shrinks once the current model is already below the reference model.
    """
    cur_logp = answer_logp(model, batch, reduction=logp_reduction)
    with torch.no_grad():
        ref_logp = answer_logp(ref_model, batch, reduction=logp_reduction)

    diff = cur_logp - ref_logp
    loss = -(2.0 / beta) * F.logsigmoid(-beta * diff).mean()
    return loss, cur_logp.detach().mean(), ref_logp.detach().mean()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--forget-token-json", default="outputs/semantic_tokens_answer_tfidf_medium_safe.json")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=None)

    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--retain-batch-size", type=int, default=2)

    parser.add_argument("--lr-input", type=float, default=1e-5)
    parser.add_argument("--lr-output", type=float, default=1.2e-4)

    parser.add_argument("--npo-beta", type=float, default=1.0)
    parser.add_argument("--logp-reduction", choices=["mean", "sum"], default="mean")
    parser.add_argument("--forget-loss-weight", type=float, default=1.0)
    parser.add_argument("--retain-loss-weight", type=float, default=1.0)

    parser.add_argument("--anchor-lambda-input", type=float, default=0.05)
    parser.add_argument("--anchor-lambda-output", type=float, default=0.06)
    parser.add_argument("--max-delta-norm-input", type=float, default=0.25)
    parser.add_argument("--max-delta-norm-output", type=float, default=0.75)

    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--grad-clip-value", type=float, default=0.05)
    parser.add_argument("--max-row-abs", type=float, default=10.0)

    parser.add_argument("--max-forget-train-samples", type=int, default=None)
    parser.add_argument("--max-retain-train-samples", type=int, default=800)

    parser.add_argument("--force-untie-lm-head", action="store_true")
    parser.add_argument("--update-input-embeddings", action="store_true")
    parser.add_argument("--update-output-lm-head", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.update_input_embeddings and not args.update_output_lm_head:
        args.update_output_lm_head = True

    if args.npo_beta <= 0:
        raise ValueError("--npo-beta must be > 0")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    max_length = args.max_length or cfg["model"].get("max_length", 512)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 80)
    print("Answer-token TF-IDF + masked NPO+GD unlearning")
    print("=" * 80)
    print(f"model-dir: {args.model_dir}")
    print(f"forget-token-json: {args.forget_token_json}")
    print(f"output-dir: {args.output_dir}")
    print(f"forget split: {forget_split}")
    print(f"retain split: {retain_split}")
    print(f"npo beta: {args.npo_beta}")
    print(f"logp reduction: {args.logp_reduction}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[Load] current trainable model")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.config.use_cache = False
    model.train()

    print("[Load] frozen reference model")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    ref_model.config.use_cache = False
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    emb, lm_head, _ = force_untie_lm_head_if_needed(model, args.force_untie_lm_head)
    freeze_except_token_side(model, emb, lm_head, args.update_input_embeddings, args.update_output_lm_head)

    input_device = first_param_device(model)
    emb_device = emb.weight.device
    out_device = lm_head.weight.device
    vocab_size = emb.weight.shape[0]

    forget_token_ids = load_token_ids(Path(args.forget_token_json))
    print(f"selected forget/answer tokens: {len(forget_token_ids)}")

    mask_in = make_row_mask(vocab_size, forget_token_ids, emb_device)
    mask_out = make_row_mask(vocab_size, forget_token_ids, out_device)

    anchor_in_ids = torch.tensor(forget_token_ids, dtype=torch.long, device=emb_device)
    anchor_out_ids = torch.tensor(forget_token_ids, dtype=torch.long, device=out_device)

    emb_anchor = emb.weight.detach()[anchor_in_ids].clone().float()
    out_anchor = lm_head.weight.detach()[anchor_out_ids].clone().float()

    print("Loading TOFU datasets...")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    forget_rows = list(forget_ds)
    retain_rows = list(retain_ds)
    if args.max_forget_train_samples is not None:
        forget_rows = forget_rows[: args.max_forget_train_samples]
    if args.max_retain_train_samples is not None:
        retain_rows = retain_rows[: args.max_retain_train_samples]

    print("Encoding answer-only train examples...")
    forget_encoded = [encode_answer_only(tokenizer, r["question"], r["answer"], max_length) for r in forget_rows]
    retain_encoded = [encode_answer_only(tokenizer, r["question"], r["answer"], max_length) for r in retain_rows]
    print(f"forget train examples: {len(forget_encoded)}")
    print(f"retain train examples: {len(retain_encoded)}")

    param_groups = []
    if args.update_input_embeddings:
        param_groups.append({"params": [emb.weight], "lr": args.lr_input})
    if args.update_output_lm_head and lm_head.weight.data_ptr() != emb.weight.data_ptr():
        param_groups.append({"params": [lm_head.weight], "lr": args.lr_output})
    elif args.update_output_lm_head and lm_head.weight.data_ptr() == emb.weight.data_ptr():
        print("[Warn] lm_head is tied. Use --force-untie-lm-head for separate LM-head editing.")

    if not param_groups:
        raise RuntimeError("No trainable parameters. Check update flags and --force-untie-lm-head.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0, eps=1e-8)

    last_npo = last_retain = last_anchor = last_cur_logp = last_ref_logp = 0.0
    pbar = tqdm(range(args.steps), desc="answer_tfidf_npo_gd_masked")

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        forget_batch = sample_batch(rng, forget_encoded, args.batch_size, tokenizer.pad_token_id, input_device)
        retain_batch = sample_batch(rng, retain_encoded, args.retain_batch_size, tokenizer.pad_token_id, input_device)

        npo, cur_logp, ref_logp = npo_loss(model, ref_model, forget_batch, args.npo_beta, args.logp_reduction)
        finite_or_raise_loss(npo, "npo_loss", step)

        r_loss = retain_ce_loss(model, retain_batch)
        finite_or_raise_loss(r_loss, "retain_ce_loss", step)

        a_loss = torch.tensor(0.0, device=input_device)
        if args.update_input_embeddings:
            a_loss = a_loss + args.anchor_lambda_input * anchor_loss(emb.weight, anchor_in_ids, emb_anchor).to(input_device)
        if args.update_output_lm_head:
            a_loss = a_loss + args.anchor_lambda_output * anchor_loss(lm_head.weight, anchor_out_ids, out_anchor).to(input_device)

        loss = args.forget_loss_weight * npo + args.retain_loss_weight * r_loss + a_loss
        loss.backward()

        if args.update_input_embeddings:
            mask_grad_rows(emb.weight, mask_in)
            finite_or_raise_grad(emb.weight, "input_embeddings.grad", step)
        if args.update_output_lm_head and lm_head.weight.grad is not None:
            mask_grad_rows(lm_head.weight, mask_out)
            finite_or_raise_grad(lm_head.weight, "output_lm_head.grad", step)

        if args.grad_clip_value > 0:
            if emb.weight.grad is not None:
                emb.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
            if lm_head.weight.grad is not None:
                lm_head.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([p for g in param_groups for p in g["params"]], args.grad_clip_norm)

        optimizer.step()

        if args.update_input_embeddings:
            clip_rows(emb.weight, anchor_in_ids, emb_anchor, args.max_delta_norm_input)
            finite_or_raise_rows(emb.weight, anchor_in_ids, "input_embeddings.after_step", step)
        if args.update_output_lm_head:
            clip_rows(lm_head.weight, anchor_out_ids, out_anchor, args.max_delta_norm_output)
            finite_or_raise_rows(lm_head.weight, anchor_out_ids, "output_lm_head.after_step", step)

        if max_abs_rows(emb.weight, anchor_in_ids) > args.max_row_abs:
            raise RuntimeError(f"[Guard] input rows exceeded max-row-abs at step={step}")
        if max_abs_rows(lm_head.weight, anchor_out_ids) > args.max_row_abs:
            raise RuntimeError(f"[Guard] output rows exceeded max-row-abs at step={step}")

        last_npo = float(npo.detach().cpu())
        last_retain = float(r_loss.detach().cpu())
        last_anchor = float(a_loss.detach().cpu())
        last_cur_logp = float(cur_logp.detach().cpu())
        last_ref_logp = float(ref_logp.detach().cpu())

        if step % 10 == 0 or step == args.steps - 1:
            pbar.set_postfix({
                "npo": f"{last_npo:.4f}",
                "ret_ce": f"{last_retain:.4f}",
                "cur_lp": f"{last_cur_logp:.4f}",
                "ref_lp": f"{last_ref_logp:.4f}",
                "anchor": f"{last_anchor:.6f}",
                "out_abs": f"{max_abs_rows(lm_head.weight, anchor_out_ids):.3f}",
            })

    if not torch.isfinite(emb.weight).all().item():
        raise RuntimeError("Final input embedding has NaN/Inf. Not saving.")
    if not torch.isfinite(lm_head.weight).all().item():
        raise RuntimeError("Final output lm_head has NaN/Inf. Not saving.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "method": "answer_tfidf_npo_gd_masked",
        "model_dir": args.model_dir,
        "forget_token_json": args.forget_token_json,
        "output_dir": args.output_dir,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "n_forget_tokens": len(forget_token_ids),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "retain_batch_size": args.retain_batch_size,
        "lr_input": args.lr_input,
        "lr_output": args.lr_output,
        "npo_beta": args.npo_beta,
        "logp_reduction": args.logp_reduction,
        "forget_loss_weight": args.forget_loss_weight,
        "retain_loss_weight": args.retain_loss_weight,
        "anchor_lambda_input": args.anchor_lambda_input,
        "anchor_lambda_output": args.anchor_lambda_output,
        "max_delta_norm_input": args.max_delta_norm_input,
        "max_delta_norm_output": args.max_delta_norm_output,
        "force_untie_lm_head": args.force_untie_lm_head,
        "update_input_embeddings": args.update_input_embeddings,
        "update_output_lm_head": args.update_output_lm_head,
        "final_npo_loss": last_npo,
        "final_retain_ce": last_retain,
        "final_anchor_loss": last_anchor,
        "final_current_forget_logp": last_cur_logp,
        "final_ref_forget_logp": last_ref_logp,
    }
    with open(out_dir / "answer_tfidf_npo_gd_masked_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done]")
    print("Evaluate:")
    print(
        f"python scripts/tofu_eval.py --config {args.config} "
        f"--model-dir {out_dir} "
        f"--method answer_tfidf_npo_gd_masked "
        f"--forget-split {forget_split} --retain-split {retain_split}"
    )


if __name__ == "__main__":
    main()
