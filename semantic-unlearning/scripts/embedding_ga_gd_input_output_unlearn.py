#!/usr/bin/env python3
"""
Embedding-side GA/GD unlearning for TOFU.

Updates only token-side matrices:
  1) input embeddings: model.get_input_embeddings().weight
  2) output unembedding/lm_head: model.get_output_embeddings().weight

Transformer blocks remain frozen.

Why this script exists:
  Input-embedding-only GA/GD was stable but did not reduce forget_answer_prob.
  Answer probability is controlled directly by output logits, so this script also
  edits lm_head rows for the selected forget/retain tokens.

Recommended use:
  - First build outputs/semantic_tokens.json using JSON-TFIDF filtering.
  - Then run this script from the semantic-unlearning repo root.
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def encode_answer_only(tokenizer, question: str, answer: str, max_length: int) -> Dict[str, List[int]]:
    prompt = format_prompt(question)
    answer_text = " " + str(answer).strip()
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def collate(examples: Sequence[Dict[str, List[int]]], pad_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
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


def sample_batch(rng: random.Random, encoded, batch_size: int, pad_id: int, device: torch.device):
    batch = [encoded[rng.randrange(len(encoded))] for _ in range(batch_size)]
    return collate(batch, pad_id, device)


def lm_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]).loss


def load_token_ids(path: Path) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "token_ids" in data:
        ids = [int(x) for x in data["token_ids"]]
    elif "semantic_tokens" in data:
        ids = [int(x["token_id"]) for x in data["semantic_tokens"]]
    else:
        raise ValueError(f"No token_ids or semantic_tokens found in {path}")
    return sorted(set(ids))


def doc_frequency(dataset, tokenizer, max_samples=None) -> Tuple[Counter, int]:
    df = Counter()
    n = 0
    for i, row in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        text = f"Question: {row['question']} Answer: {row['answer']}"
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            df[int(tid)] += 1
        n += 1
    return df, n


def build_retain_tokens(
    tokenizer,
    forget_ds,
    retain_ds,
    forget_token_set: set,
    top_k: int,
    min_retain_count: int,
    max_forget_ratio: float,
    max_retain_samples=None,
    max_forget_samples=None,
) -> List[int]:
    retain_df, n_retain = doc_frequency(retain_ds, tokenizer, max_retain_samples)
    forget_df, n_forget = doc_frequency(forget_ds, tokenizer, max_forget_samples)

    special_ids = {x for x in [tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.bos_token_id, tokenizer.unk_token_id] if x is not None}
    total_docs = n_retain + n_forget
    scored = []

    for tid, r_count in retain_df.items():
        tid = int(tid)
        if tid in forget_token_set or tid in special_ids:
            continue
        if r_count < min_retain_count:
            continue
        token_str = tokenizer.decode([tid])
        if len(token_str.strip()) < 2:
            continue

        f_count = int(forget_df.get(tid, 0))
        f_ratio = f_count / max(1, n_forget)
        if f_ratio > max_forget_ratio:
            continue

        r_ratio = r_count / max(1, n_retain)
        idf = math.log((total_docs + 1) / (r_count + f_count + 1)) + 1.0
        retain_tfidf = r_ratio * idf
        retain_specificity = (r_ratio + 1e-8) / (f_ratio + 1e-8)
        score = retain_tfidf * math.log1p(retain_specificity)
        scored.append((score, r_count, -f_count, tid))

    scored.sort(reverse=True)
    return [tid for _, _, _, tid in scored[:top_k]]


def make_row_mask(vocab_size: int, token_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    ids = [int(x) for x in set(token_ids) if 0 <= int(x) < vocab_size]
    if ids:
        mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask


def mask_grad_rows(param: torch.nn.Parameter, row_mask: torch.Tensor) -> None:
    if param is None or param.grad is None:
        return
    param.grad[~row_mask] = 0


def finite_or_raise_loss(loss, name: str, step: int):
    if not torch.isfinite(loss).all().item():
        raise RuntimeError(f"[NaNGuard] Non-finite loss at step={step}: {name}={loss}")


def finite_or_raise_grad(param, name: str, step: int):
    if param is None or param.grad is None:
        return
    if not torch.isfinite(param.grad).all().item():
        bad = (~torch.isfinite(param.grad)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite grad at step={step}: {name}, bad_count={bad}")


def finite_or_raise_rows(weight, row_ids, name: str, step: int):
    rows = weight.data[row_ids]
    if not torch.isfinite(rows).all().item():
        bad = (~torch.isfinite(rows)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite rows at step={step}: {name}, bad_count={bad}")


def anchor_loss(weight: torch.Tensor, ids: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(weight[ids].float(), original.float())


@torch.no_grad()
def clip_rows(weight: torch.Tensor, ids: torch.Tensor, original: torch.Tensor, max_norm: float) -> None:
    if max_norm <= 0 or ids.numel() == 0:
        return
    current = weight.data[ids].float()
    delta = current - original.float()
    norms = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-12), max=1.0)
    clipped = original.float() + delta * scale
    weight.data[ids] = clipped.to(weight.dtype)


@torch.no_grad()
def max_abs_rows(weight: torch.Tensor, ids: torch.Tensor) -> float:
    if ids.numel() == 0:
        return 0.0
    return float(weight.data[ids].abs().max().detach().cpu())


def force_untie_lm_head_if_needed(model, force_untie: bool):
    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("Could not find output embeddings / lm_head weight.")

    tied = emb.weight.data_ptr() == lm_head.weight.data_ptr()
    if tied and force_untie:
        old_w = lm_head.weight.detach().clone()
        new_head = nn.Linear(old_w.shape[1], old_w.shape[0], bias=False)
        new_head = new_head.to(device=old_w.device, dtype=old_w.dtype)
        new_head.weight.data.copy_(old_w)
        model.set_output_embeddings(new_head)
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
        lm_head = model.get_output_embeddings()
        tied = False
        print("[Info] Forced lm_head untie: output head now has independent weights.")

    return emb, lm_head, (not tied)


def freeze_except_token_side(model, emb, lm_head, update_input: bool, update_output: bool):
    for p in model.parameters():
        p.requires_grad_(False)
    if update_input:
        emb.weight.requires_grad_(True)
    if update_output:
        lm_head.weight.requires_grad_(True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--forget-token-json", default="outputs/semantic_tokens.json")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=None)

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=4)

    parser.add_argument("--forget-lr-input", type=float, default=5e-6)
    parser.add_argument("--forget-lr-output", type=float, default=2e-5)
    parser.add_argument("--retain-lr-input", type=float, default=2e-6)
    parser.add_argument("--retain-lr-output", type=float, default=5e-6)

    parser.add_argument("--forget-loss-weight", type=float, default=1.0)
    parser.add_argument("--retain-loss-weight", type=float, default=2.0)
    parser.add_argument("--anchor-lambda-input", type=float, default=0.10)
    parser.add_argument("--anchor-lambda-output", type=float, default=0.10)
    parser.add_argument("--max-delta-norm-input", type=float, default=0.20)
    parser.add_argument("--max-delta-norm-output", type=float, default=0.35)

    parser.add_argument("--grad-clip-norm", type=float, default=0.2)
    parser.add_argument("--grad-clip-value", type=float, default=0.02)
    parser.add_argument("--max-row-abs", type=float, default=10.0)
    parser.add_argument("--forget-loss-cap", type=float, default=30.0)

    parser.add_argument("--retain-top-k", type=int, default=7000)
    parser.add_argument("--retain-min-count", type=int, default=10)
    parser.add_argument("--retain-max-forget-ratio", type=float, default=0.004)
    parser.add_argument("--max-retain-token-selection-samples", type=int, default=None)
    parser.add_argument("--max-forget-token-selection-samples", type=int, default=None)

    parser.add_argument("--max-forget-train-samples", type=int, default=None)
    parser.add_argument("--max-retain-train-samples", type=int, default=800)

    parser.add_argument("--forget-steps-per-round", type=int, default=1)
    parser.add_argument("--retain-steps-per-round", type=int, default=1)

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
    print("Input + output embedding-side GA/GD unlearning")
    print("=" * 80)
    print(f"model-dir: {args.model_dir}")
    print(f"forget-token-json: {args.forget_token_json}")
    print(f"output-dir: {args.output_dir}")
    print(f"dtype: {args.dtype}")
    print(f"forget split: {forget_split}")
    print(f"retain split: {retain_split}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=resolve_dtype(args.dtype),
        device_map=args.device_map,
    )
    model.config.use_cache = False
    model.train()

    emb, lm_head, lm_head_separate = force_untie_lm_head_if_needed(model, args.force_untie_lm_head)
    freeze_except_token_side(model, emb, lm_head, args.update_input_embeddings, args.update_output_lm_head)

    input_device = first_param_device(model)
    emb_device = emb.weight.device
    out_device = lm_head.weight.device
    vocab_size = emb.weight.shape[0]

    print(f"input device: {input_device}")
    print(f"input embedding device/dtype: {emb_device} / {emb.weight.dtype}")
    print(f"output lm_head device/dtype: {out_device} / {lm_head.weight.dtype}")
    print(f"embedding shape: {tuple(emb.weight.shape)}")
    print(f"lm_head shape: {tuple(lm_head.weight.shape)}")
    print(f"lm_head separate from input embedding: {lm_head.weight.data_ptr() != emb.weight.data_ptr()}")
    print(f"update input embeddings: {args.update_input_embeddings}")
    print(f"update output lm_head: {args.update_output_lm_head}")

    forget_token_ids = load_token_ids(Path(args.forget_token_json))
    forget_token_set = set(forget_token_ids)
    print(f"forget tokens: {len(forget_token_ids)}")

    print("Loading TOFU datasets...")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    print("Building retain-protection tokens...")
    retain_token_ids = build_retain_tokens(
        tokenizer=tokenizer,
        forget_ds=forget_ds,
        retain_ds=retain_ds,
        forget_token_set=forget_token_set,
        top_k=args.retain_top_k,
        min_retain_count=args.retain_min_count,
        max_forget_ratio=args.retain_max_forget_ratio,
        max_retain_samples=args.max_retain_token_selection_samples,
        max_forget_samples=args.max_forget_token_selection_samples,
    )
    print(f"retain-protection tokens: {len(retain_token_ids)}")

    forget_mask_in = make_row_mask(vocab_size, forget_token_ids, emb_device)
    retain_mask_in = make_row_mask(vocab_size, retain_token_ids, emb_device)
    forget_mask_out = make_row_mask(vocab_size, forget_token_ids, out_device)
    retain_mask_out = make_row_mask(vocab_size, retain_token_ids, out_device)

    anchor_ids = sorted(set(forget_token_ids) | set(retain_token_ids))
    anchor_in_ids = torch.tensor(anchor_ids, dtype=torch.long, device=emb_device)
    anchor_out_ids = torch.tensor(anchor_ids, dtype=torch.long, device=out_device)

    emb_anchor = emb.weight.detach()[anchor_in_ids].clone().float()
    out_anchor = lm_head.weight.detach()[anchor_out_ids].clone().float()

    print(f"anchored rows: {len(anchor_ids)}")
    print(f"initial input max abs rows: {max_abs_rows(emb.weight, anchor_in_ids):.4f}")
    print(f"initial output max abs rows: {max_abs_rows(lm_head.weight, anchor_out_ids):.4f}")

    forget_rows = list(forget_ds)
    retain_rows = list(retain_ds)
    if args.max_forget_train_samples is not None:
        forget_rows = forget_rows[: args.max_forget_train_samples]
    if args.max_retain_train_samples is not None:
        retain_rows = retain_rows[: args.max_retain_train_samples]

    print("Encoding train examples...")
    forget_encoded = [encode_answer_only(tokenizer, r["question"], r["answer"], max_length) for r in forget_rows]
    retain_encoded = [encode_answer_only(tokenizer, r["question"], r["answer"], max_length) for r in retain_rows]
    print(f"forget train examples: {len(forget_encoded)}")
    print(f"retain train examples: {len(retain_encoded)}")

    forget_groups = []
    retain_groups = []
    if args.update_input_embeddings:
        forget_groups.append({"params": [emb.weight], "lr": args.forget_lr_input})
        retain_groups.append({"params": [emb.weight], "lr": args.retain_lr_input})
    if args.update_output_lm_head and lm_head.weight.data_ptr() != emb.weight.data_ptr():
        forget_groups.append({"params": [lm_head.weight], "lr": args.forget_lr_output})
        retain_groups.append({"params": [lm_head.weight], "lr": args.retain_lr_output})
    elif args.update_output_lm_head and lm_head.weight.data_ptr() == emb.weight.data_ptr():
        print("[Warn] lm_head is tied to input embeddings. Use --force-untie-lm-head if you want separate output-row editing.")

    if not forget_groups or not retain_groups:
        raise RuntimeError("No trainable parameter groups. Check update flags.")

    opt_forget = torch.optim.AdamW(forget_groups, weight_decay=0.0, eps=1e-8)
    opt_retain = torch.optim.AdamW(retain_groups, weight_decay=0.0, eps=1e-8)

    pbar = tqdm(range(args.steps), desc="Input+Output GA/GD rows")
    last_f, last_r, last_a = 0.0, 0.0, 0.0

    for step in pbar:
        for _ in range(args.forget_steps_per_round):
            opt_forget.zero_grad(set_to_none=True)
            batch = sample_batch(rng, forget_encoded, args.batch_size, tokenizer.pad_token_id, input_device)
            f_loss = lm_loss(model, batch)
            finite_or_raise_loss(f_loss, "forget_loss", step)
            if float(f_loss.detach().cpu()) > args.forget_loss_cap:
                last_f = float(f_loss.detach().cpu())
                continue

            a_loss = torch.tensor(0.0, device=input_device)
            if args.update_input_embeddings:
                a_loss = a_loss + args.anchor_lambda_input * anchor_loss(emb.weight, anchor_in_ids, emb_anchor).to(input_device)
            if args.update_output_lm_head:
                a_loss = a_loss + args.anchor_lambda_output * anchor_loss(lm_head.weight, anchor_out_ids, out_anchor).to(input_device)

            loss = -args.forget_loss_weight * f_loss + a_loss
            loss.backward()

            if args.update_input_embeddings:
                mask_grad_rows(emb.weight, forget_mask_in)
                finite_or_raise_grad(emb.weight, "input_embeddings.forget_grad", step)
            if args.update_output_lm_head and lm_head.weight.grad is not None:
                mask_grad_rows(lm_head.weight, forget_mask_out)
                finite_or_raise_grad(lm_head.weight, "output_lm_head.forget_grad", step)

            if args.grad_clip_value > 0:
                if emb.weight.grad is not None:
                    emb.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
                if lm_head.weight.grad is not None:
                    lm_head.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for g in forget_groups for p in g["params"]], args.grad_clip_norm)

            opt_forget.step()

            if args.update_input_embeddings:
                clip_rows(emb.weight, anchor_in_ids, emb_anchor, args.max_delta_norm_input)
                finite_or_raise_rows(emb.weight, anchor_in_ids, "input_embeddings.after_forget_step", step)
            if args.update_output_lm_head:
                clip_rows(lm_head.weight, anchor_out_ids, out_anchor, args.max_delta_norm_output)
                finite_or_raise_rows(lm_head.weight, anchor_out_ids, "output_lm_head.after_forget_step", step)

            if max_abs_rows(emb.weight, anchor_in_ids) > args.max_row_abs:
                raise RuntimeError(f"[Guard] input rows exceeded max-row-abs at step={step}")
            if max_abs_rows(lm_head.weight, anchor_out_ids) > args.max_row_abs:
                raise RuntimeError(f"[Guard] output rows exceeded max-row-abs at step={step}")

            last_f = float(f_loss.detach().cpu())
            last_a = float(a_loss.detach().cpu())

        for _ in range(args.retain_steps_per_round):
            opt_retain.zero_grad(set_to_none=True)
            batch = sample_batch(rng, retain_encoded, args.retain_batch_size, tokenizer.pad_token_id, input_device)
            r_loss = lm_loss(model, batch)
            finite_or_raise_loss(r_loss, "retain_loss", step)

            a_loss = torch.tensor(0.0, device=input_device)
            if args.update_input_embeddings:
                a_loss = a_loss + args.anchor_lambda_input * anchor_loss(emb.weight, anchor_in_ids, emb_anchor).to(input_device)
            if args.update_output_lm_head:
                a_loss = a_loss + args.anchor_lambda_output * anchor_loss(lm_head.weight, anchor_out_ids, out_anchor).to(input_device)

            loss = args.retain_loss_weight * r_loss + a_loss
            loss.backward()

            if args.update_input_embeddings:
                mask_grad_rows(emb.weight, retain_mask_in)
                finite_or_raise_grad(emb.weight, "input_embeddings.retain_grad", step)
            if args.update_output_lm_head and lm_head.weight.grad is not None:
                mask_grad_rows(lm_head.weight, retain_mask_out)
                finite_or_raise_grad(lm_head.weight, "output_lm_head.retain_grad", step)

            if args.grad_clip_value > 0:
                if emb.weight.grad is not None:
                    emb.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
                if lm_head.weight.grad is not None:
                    lm_head.weight.grad.clamp_(-args.grad_clip_value, args.grad_clip_value)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_([p for g in retain_groups for p in g["params"]], args.grad_clip_norm)

            opt_retain.step()

            if args.update_input_embeddings:
                clip_rows(emb.weight, anchor_in_ids, emb_anchor, args.max_delta_norm_input)
                finite_or_raise_rows(emb.weight, anchor_in_ids, "input_embeddings.after_retain_step", step)
            if args.update_output_lm_head:
                clip_rows(lm_head.weight, anchor_out_ids, out_anchor, args.max_delta_norm_output)
                finite_or_raise_rows(lm_head.weight, anchor_out_ids, "output_lm_head.after_retain_step", step)

            if max_abs_rows(emb.weight, anchor_in_ids) > args.max_row_abs:
                raise RuntimeError(f"[Guard] input rows exceeded max-row-abs at step={step}")
            if max_abs_rows(lm_head.weight, anchor_out_ids) > args.max_row_abs:
                raise RuntimeError(f"[Guard] output rows exceeded max-row-abs at step={step}")

            last_r = float(r_loss.detach().cpu())
            last_a = float(a_loss.detach().cpu())

        if step % 10 == 0 or step == args.steps - 1:
            pbar.set_postfix({
                "f_loss": f"{last_f:.3f}",
                "r_loss": f"{last_r:.3f}",
                "anchor": f"{last_a:.6f}",
                "in_abs": f"{max_abs_rows(emb.weight, anchor_in_ids):.3f}",
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
        "method": "input_output_embedding_side_ga_gd",
        "model_dir": args.model_dir,
        "forget_token_json": args.forget_token_json,
        "output_dir": args.output_dir,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "n_forget_tokens": len(forget_token_ids),
        "n_retain_tokens": len(retain_token_ids),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "retain_batch_size": args.retain_batch_size,
        "forget_lr_input": args.forget_lr_input,
        "forget_lr_output": args.forget_lr_output,
        "retain_lr_input": args.retain_lr_input,
        "retain_lr_output": args.retain_lr_output,
        "forget_loss_weight": args.forget_loss_weight,
        "retain_loss_weight": args.retain_loss_weight,
        "anchor_lambda_input": args.anchor_lambda_input,
        "anchor_lambda_output": args.anchor_lambda_output,
        "max_delta_norm_input": args.max_delta_norm_input,
        "max_delta_norm_output": args.max_delta_norm_output,
        "force_untie_lm_head": args.force_untie_lm_head,
        "lm_head_separate": lm_head.weight.data_ptr() != emb.weight.data_ptr(),
        "update_input_embeddings": args.update_input_embeddings,
        "update_output_lm_head": args.update_output_lm_head,
    }
    with open(out_dir / "embedding_io_ga_gd_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done]")
    print("Evaluate:")
    print(
        f"python scripts/tofu_eval.py --config {args.config} "
        f"--model-dir {out_dir} "
        f"--method embed_io_ga_gd_forget05 "
        f"--forget-split {forget_split} --retain-split {retain_split}"
    )


if __name__ == "__main__":
    main()
