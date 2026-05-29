#!/usr/bin/env python3
"""
JSON/TF-IDF + WGA + KL masked embedding/LM-head unlearning.

Method:
  json_tfidf_wga_kl_masked

Core idea:
  - Load forget-relevant token IDs from outputs/semantic_tokens.json.
  - Freeze the full model.
  - Update only selected rows of:
      1. input embeddings
      2. lm_head / output embeddings
  - Forget objective: Weighted Gradient Ascent on forget answers.
  - Retain objective: KL preservation against the original model on retain answers.
"""

import argparse
import json
import random
from pathlib import Path

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


def answer_token_wga_loss(model, batch, gamma: float = 1.0, min_weight: float = 0.0):
    """
    Weighted Gradient Ascent loss on answer tokens only.

    CE = -log p(answer token)
    Plain GA minimizes: -CE

    WGA minimizes:
        - weight * CE

    where:
        weight = p(answer token)^gamma

    This means:
      - if model still strongly predicts forget answer: p high -> strong update
      - if model already forgot: p low -> weak update
    """
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = out.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = batch["labels"][:, 1:].contiguous()
    valid = shift_labels.ne(-100)

    vocab = shift_logits.size(-1)
    ce = F.cross_entropy(
        shift_logits.view(-1, vocab).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    valid_f = valid.float()
    denom = valid_f.sum().clamp_min(1.0)

    with torch.no_grad():
        p_true = torch.exp(-ce).clamp(min=1e-8, max=1.0)
        weights = p_true.pow(gamma)
        if min_weight > 0:
            weights = weights.clamp_min(min_weight)
        weights = weights * valid_f

    wga_loss = -((weights * ce).sum() / denom)
    plain_ce = (ce * valid_f).sum() / denom

    return wga_loss, plain_ce.detach()


def retain_kl_loss(model, ref_model, batch, temperature: float = 1.0):
    """
    KL preservation on answer-token positions only.

    Keeps current model close to original model on retain answers.
    """
    with torch.no_grad():
        ref_logits = ref_model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).logits

    cur_logits = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    ).logits

    shift_cur = cur_logits[:, :-1, :].contiguous().float() / temperature
    shift_ref = ref_logits[:, :-1, :].contiguous().float() / temperature
    shift_labels = batch["labels"][:, 1:].contiguous()
    valid = shift_labels.ne(-100)

    cur_logp = F.log_softmax(shift_cur, dim=-1)
    ref_p = F.softmax(shift_ref, dim=-1)

    token_kl = F.kl_div(cur_logp, ref_p, reduction="none").sum(dim=-1)
    valid_f = valid.float()
    denom = valid_f.sum().clamp_min(1.0)

    return (token_kl * valid_f).sum() / denom * (temperature ** 2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--forget-token-json", default="outputs/semantic_tokens.json")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=None)

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--retain-batch-size", type=int, default=2)

    parser.add_argument("--lr-input", type=float, default=2e-5)
    parser.add_argument("--lr-output", type=float, default=5e-5)

    parser.add_argument("--forget-loss-weight", type=float, default=1.0)
    parser.add_argument("--retain-kl-weight", type=float, default=2.0)
    parser.add_argument("--wga-gamma", type=float, default=1.0)
    parser.add_argument("--wga-min-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=1.0)

    parser.add_argument("--anchor-lambda-input", type=float, default=0.10)
    parser.add_argument("--anchor-lambda-output", type=float, default=0.10)
    parser.add_argument("--max-delta-norm-input", type=float, default=0.20)
    parser.add_argument("--max-delta-norm-output", type=float, default=0.25)

    parser.add_argument("--grad-clip-norm", type=float, default=0.2)
    parser.add_argument("--grad-clip-value", type=float, default=0.02)
    parser.add_argument("--max-row-abs", type=float, default=10.0)

    parser.add_argument("--max-forget-train-samples", type=int, default=None)
    parser.add_argument("--max-retain-train-samples", type=int, default=800)

    parser.add_argument("--force-untie-lm-head", action="store_true")
    parser.add_argument("--update-input-embeddings", action="store_true")
    parser.add_argument("--update-output-lm-head", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not args.update_input_embeddings and not args.update_output_lm_head:
        args.update_input_embeddings = True
        args.update_output_lm_head = True

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    max_length = args.max_length or cfg["model"].get("max_length", 512)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 80)
    print("JSON/TF-IDF + WGA + KL masked embedding/lm_head unlearning")
    print("=" * 80)
    print(f"model-dir: {args.model_dir}")
    print(f"forget-token-json: {args.forget_token_json}")
    print(f"output-dir: {args.output_dir}")
    print(f"forget split: {forget_split}")
    print(f"retain split: {retain_split}")
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

    print("[Load] frozen reference model for retain KL")
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
    freeze_except_token_side(
        model,
        emb,
        lm_head,
        args.update_input_embeddings,
        args.update_output_lm_head,
    )

    input_device = first_param_device(model)
    emb_device = emb.weight.device
    out_device = lm_head.weight.device
    vocab_size = emb.weight.shape[0]

    forget_token_ids = load_token_ids(Path(args.forget_token_json))
    print(f"selected forget tokens: {len(forget_token_ids)}")

    forget_mask_in = make_row_mask(vocab_size, forget_token_ids, emb_device)
    forget_mask_out = make_row_mask(vocab_size, forget_token_ids, out_device)

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
    forget_encoded = [
        encode_answer_only(tokenizer, r["question"], r["answer"], max_length)
        for r in forget_rows
    ]
    retain_encoded = [
        encode_answer_only(tokenizer, r["question"], r["answer"], max_length)
        for r in retain_rows
    ]

    print(f"forget train examples: {len(forget_encoded)}")
    print(f"retain train examples: {len(retain_encoded)}")

    param_groups = []
    if args.update_input_embeddings:
        param_groups.append({"params": [emb.weight], "lr": args.lr_input})
    if args.update_output_lm_head and lm_head.weight.data_ptr() != emb.weight.data_ptr():
        param_groups.append({"params": [lm_head.weight], "lr": args.lr_output})
    elif args.update_output_lm_head and lm_head.weight.data_ptr() == emb.weight.data_ptr():
        print("[Warn] lm_head is tied. Use --force-untie-lm-head for separate output-row editing.")

    if not param_groups:
        raise RuntimeError("No trainable params. Check update flags.")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0, eps=1e-8)

    last_wga = 0.0
    last_fce = 0.0
    last_kl = 0.0
    last_anchor = 0.0

    pbar = tqdm(range(args.steps), desc="json_tfidf_wga_kl_masked")

    for step in pbar:
        optimizer.zero_grad(set_to_none=True)

        forget_batch = sample_batch(
            rng,
            forget_encoded,
            args.batch_size,
            tokenizer.pad_token_id,
            input_device,
        )
        retain_batch = sample_batch(
            rng,
            retain_encoded,
            args.retain_batch_size,
            tokenizer.pad_token_id,
            input_device,
        )

        wga_loss, forget_ce = answer_token_wga_loss(
            model,
            forget_batch,
            gamma=args.wga_gamma,
            min_weight=args.wga_min_weight,
        )
        finite_or_raise_loss(wga_loss, "wga_loss", step)

        kl_loss = retain_kl_loss(
            model,
            ref_model,
            retain_batch,
            temperature=args.kl_temperature,
        )
        finite_or_raise_loss(kl_loss, "retain_kl_loss", step)

        a_loss = torch.tensor(0.0, device=input_device)
        if args.update_input_embeddings:
            a_loss = a_loss + args.anchor_lambda_input * anchor_loss(
                emb.weight,
                anchor_in_ids,
                emb_anchor,
            ).to(input_device)
        if args.update_output_lm_head:
            a_loss = a_loss + args.anchor_lambda_output * anchor_loss(
                lm_head.weight,
                anchor_out_ids,
                out_anchor,
            ).to(input_device)

        loss = (
            args.forget_loss_weight * wga_loss
            + args.retain_kl_weight * kl_loss
            + a_loss
        )

        loss.backward()

        if args.update_input_embeddings:
            mask_grad_rows(emb.weight, forget_mask_in)
            finite_or_raise_grad(emb.weight, "input_embeddings.grad", step)

        if args.update_output_lm_head and lm_head.weight.grad is not None:
            mask_grad_rows(lm_head.weight, forget_mask_out)
            finite_or_raise_grad(lm_head.weight, "output_lm_head.grad", step)

        if args.grad_clip_value > 0:
            if emb.weight.grad is not None:
                emb.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
            if lm_head.weight.grad is not None:
                lm_head.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]],
                args.grad_clip_norm,
            )

        optimizer.step()

        if args.update_input_embeddings:
            clip_rows(
                emb.weight,
                anchor_in_ids,
                emb_anchor,
                args.max_delta_norm_input,
            )
            finite_or_raise_rows(
                emb.weight,
                anchor_in_ids,
                "input_embeddings.after_step",
                step,
            )

        if args.update_output_lm_head:
            clip_rows(
                lm_head.weight,
                anchor_out_ids,
                out_anchor,
                args.max_delta_norm_output,
            )
            finite_or_raise_rows(
                lm_head.weight,
                anchor_out_ids,
                "output_lm_head.after_step",
                step,
            )

        if max_abs_rows(emb.weight, anchor_in_ids) > args.max_row_abs:
            raise RuntimeError(f"[Guard] input rows exceeded max-row-abs at step={step}")
        if max_abs_rows(lm_head.weight, anchor_out_ids) > args.max_row_abs:
            raise RuntimeError(f"[Guard] output rows exceeded max-row-abs at step={step}")

        last_wga = float(wga_loss.detach().cpu())
        last_fce = float(forget_ce.detach().cpu())
        last_kl = float(kl_loss.detach().cpu())
        last_anchor = float(a_loss.detach().cpu())

        if step % 10 == 0 or step == args.steps - 1:
            pbar.set_postfix(
                {
                    "wga": f"{last_wga:.4f}",
                    "f_ce": f"{last_fce:.4f}",
                    "kl": f"{last_kl:.6f}",
                    "anchor": f"{last_anchor:.6f}",
                    "in_abs": f"{max_abs_rows(emb.weight, anchor_in_ids):.3f}",
                    "out_abs": f"{max_abs_rows(lm_head.weight, anchor_out_ids):.3f}",
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "method": "json_tfidf_wga_kl_masked",
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
        "forget_loss_weight": args.forget_loss_weight,
        "retain_kl_weight": args.retain_kl_weight,
        "wga_gamma": args.wga_gamma,
        "wga_min_weight": args.wga_min_weight,
        "kl_temperature": args.kl_temperature,
        "anchor_lambda_input": args.anchor_lambda_input,
        "anchor_lambda_output": args.anchor_lambda_output,
        "max_delta_norm_input": args.max_delta_norm_input,
        "max_delta_norm_output": args.max_delta_norm_output,
        "force_untie_lm_head": args.force_untie_lm_head,
        "update_input_embeddings": args.update_input_embeddings,
        "update_output_lm_head": args.update_output_lm_head,
        "final_wga_loss": last_wga,
        "final_forget_ce": last_fce,
        "final_retain_kl": last_kl,
        "final_anchor_loss": last_anchor,
    }

    with open(out_dir / "json_tfidf_wga_kl_masked_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done]")
    print("Evaluate:")
    print(
        f"python scripts/tofu_eval.py --config {args.config} "
        f"--model-dir {out_dir} "
        f"--method json_tfidf_wga_kl_masked "
        f"--forget-split {forget_split} --retain-split {retain_split}"
    )


if __name__ == "__main__":
    main()