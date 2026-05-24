#!/usr/bin/env python3
"""
scripts/oarts_apply_directional_lmhead.py

OARTS apply with direction-aware LM-head suppression.

Why this exists:
  Simple lm_head row scaling can accidentally increase answer probability because
  a token logit can be negative; scaling the row toward zero can raise that logit.

This script does:
  1. Optional adaptive input embedding edit, same as oarts_apply.py.
  2. For tokens marked edit_lm_head=True, collect final hidden states from the
     residual forget examples at positions that predict those answer tokens.
  3. Move each LM-head row against its forget hidden-state direction:

        w_t <- w_t - alpha_t * mean_hidden_direction_t

     This directly suppresses the selected answer-token logits on residual
     forget contexts.

Recommended first run:
  python scripts/oarts_apply_directional_lmhead.py \
    --config config/config_3b_instruct_forget05.yaml \
    --tokens-json outputs/semantic_tokens_oarts_residual_v5_targeted_lmhead.json \
    --output-dir outputs/unlearned_model_oarts_v5_directional_mean \
    --input-mode adaptive_mean \
    --lmhead-alpha-scale 0.75 \
    --lmhead-max-delta 4.0
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def encode_answer_only(tokenizer, question: str, answer: str, max_length: int) -> Dict[str, List[int]]:
    prompt = f"Question: {question}\nAnswer:"
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

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def collate_one(example: Dict[str, List[int]], pad_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([example["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([example["attention_mask"]], dtype=torch.long, device=device),
        "labels": torch.tensor([example["labels"]], dtype=torch.long, device=device),
    }


def load_oarts_tokens(path: Path):
    data = load_json(path)
    rows = data.get("semantic_tokens", [])
    if not rows and "token_ids" in data:
        rows = [{"token_id": int(x), "erase_strength": 1.0, "output_strength": 0.0} for x in data["token_ids"]]

    seen = set()
    ids, in_strengths, out_strengths, kept_rows = [], [], [], []
    for r in rows:
        tid = int(r["token_id"])
        if tid in seen:
            continue
        seen.add(tid)
        inp = float(r.get("erase_strength", r.get("input_strength", 1.0)))
        out = float(r.get("output_strength", 0.0))
        if not bool(r.get("edit_lm_head", False)):
            out = 0.0
        ids.append(tid)
        in_strengths.append(max(0.0, min(1.0, inp)))
        out_strengths.append(max(0.0, min(1.0, out)))
        kept_rows.append(dict(r))

    residual_examples = data.get("residual_examples", [])
    return data, ids, in_strengths, out_strengths, kept_rows, residual_examples


def maybe_untie_lm_head(model) -> None:
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None or not hasattr(out, "weight"):
        raise RuntimeError("Model does not expose output embeddings / lm_head")

    if out.weight.data_ptr() != emb.weight.data_ptr():
        print("[DirectionalOARTS] lm_head already untied.")
        return

    print("[DirectionalOARTS] lm_head tied. Untying before output edits.")
    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False)
    new_head = new_head.to(device=old.device, dtype=old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


@torch.no_grad()
def adaptive_input_edit(W: torch.Tensor, ids: List[int], strengths: List[float], mode: str) -> Dict[str, float]:
    if mode == "none":
        return {"old_norm": 0.0, "new_norm": 0.0, "strength_mean": 0.0}

    ids_t = torch.tensor(ids, dtype=torch.long, device=W.device)
    s = torch.tensor(strengths, dtype=torch.float32, device=W.device).clamp(0.0, 1.0).view(-1, 1)
    old = W[ids_t].clone()
    old_norm = torch.linalg.vector_norm(old.float(), dim=1).mean().item()

    if mode == "adaptive_mean":
        vocab = W.shape[0]
        mask = torch.ones(vocab, dtype=torch.bool, device=W.device)
        mask[ids_t] = False
        target_vec = W[mask].float().mean(dim=0).to(W.dtype)
        target = target_vec.unsqueeze(0).expand_as(old)
    elif mode == "adaptive_zero":
        target = torch.zeros_like(old)
    else:
        raise ValueError(f"Unknown input mode: {mode}")

    W[ids_t] = (1.0 - s.to(W.dtype)) * old + s.to(W.dtype) * target
    new_norm = torch.linalg.vector_norm(W[ids_t].float(), dim=1).mean().item()
    return {
        "old_norm": old_norm,
        "new_norm": new_norm,
        "strength_mean": float(s.mean().item()),
        "n_input_tokens": len(ids),
    }


@torch.no_grad()
def collect_forget_hidden_directions(
    model,
    tokenizer,
    examples: List[Dict[str, Any]],
    active_token_ids: List[int],
    max_length: int,
    max_examples: int,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, int]]:
    active = set(int(x) for x in active_token_ids)
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    sums: Dict[int, torch.Tensor] = {}
    counts: Dict[int, int] = {tid: 0 for tid in active}

    use_examples = examples[:max_examples] if max_examples and max_examples > 0 else examples
    print(f"[DirectionalOARTS] Collecting hidden directions from {len(use_examples)} residual examples...")

    for ex in tqdm(use_examples, desc="CollectHidden"):
        q = ex.get("question")
        a = ex.get("answer")
        if q is None or a is None:
            continue

        enc = encode_answer_only(tokenizer, q, a, max_length=max_length)
        batch = collate_one(enc, pad_id, device)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        h = out.hidden_states[-1][0].detach().float()  # [seq, hidden]
        labels = batch["labels"][0].detach().tolist()

        # HF causal LM predicts labels[pos] from hidden[pos-1].
        for pos, target_id in enumerate(labels):
            if target_id == -100 or pos <= 0:
                continue
            target_id = int(target_id)
            if target_id not in active:
                continue
            vec = h[pos - 1]
            if target_id not in sums:
                sums[target_id] = torch.zeros_like(vec)
            sums[target_id] += vec
            counts[target_id] += 1

    directions: Dict[int, torch.Tensor] = {}
    for tid, vec_sum in sums.items():
        c = max(1, counts.get(tid, 0))
        mean = vec_sum / c
        norm = torch.linalg.vector_norm(mean).clamp_min(1e-12)
        directions[tid] = mean / norm

    return directions, counts


@torch.no_grad()
def directional_lmhead_edit(
    W: torch.Tensor,
    rows: List[Dict[str, Any]],
    out_strengths: List[float],
    directions: Dict[int, torch.Tensor],
    counts: Dict[int, int],
    alpha_scale: float,
    max_delta: float,
    min_count: int,
) -> Dict[str, Any]:
    edited = []
    skipped = []

    row_by_tid = {int(r["token_id"]): r for r in rows}
    active_pairs = [(int(r["token_id"]), float(st)) for r, st in zip(rows, out_strengths) if float(st) > 0]

    for tid, strength in active_pairs:
        c = int(counts.get(tid, 0))
        if tid not in directions or c < min_count:
            skipped.append({"token_id": tid, "reason": "no_hidden_direction_or_low_count", "count": c})
            continue

        direction = directions[tid].to(device=W.device, dtype=torch.float32)
        old = W[tid].float().clone()
        old_norm = torch.linalg.vector_norm(old).item()

        # Delta is proportional to row norm and output strength, but clipped.
        delta_norm = float(strength) * float(alpha_scale) * max(old_norm, 1e-6)
        if max_delta and max_delta > 0:
            delta_norm = min(delta_norm, float(max_delta))

        new = old - delta_norm * direction
        W[tid] = new.to(W.dtype)

        r = row_by_tid.get(tid, {})
        edited.append({
            "token_id": tid,
            "token_str": r.get("token_str", ""),
            "count": c,
            "strength": strength,
            "old_norm": old_norm,
            "delta_norm": delta_norm,
        })

    if edited:
        old_norm_mean = sum(x["old_norm"] for x in edited) / len(edited)
        delta_mean = sum(x["delta_norm"] for x in edited) / len(edited)
    else:
        old_norm_mean = 0.0
        delta_mean = 0.0

    return {
        "n_requested": len(active_pairs),
        "n_edited": len(edited),
        "n_skipped": len(skipped),
        "old_norm_mean": old_norm_mean,
        "delta_norm_mean": delta_mean,
        "edited_preview": edited[:100],
        "skipped_preview": skipped[:100],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tokens-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--input-mode", choices=["adaptive_mean", "adaptive_zero", "none"], default="adaptive_mean")
    ap.add_argument("--no-output-edit", action="store_true")
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-hidden-examples", type=int, default=200)
    ap.add_argument("--lmhead-alpha-scale", type=float, default=0.75)
    ap.add_argument("--lmhead-max-delta", type=float, default=4.0)
    ap.add_argument("--lmhead-min-count", type=int, default=1)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")
    max_length = args.max_length or cfg["model"].get("max_length", 512)
    forget_split = args.forget_split or cfg["data"].get("forget_split", "forget05")

    tokens_path = Path(args.tokens_json)
    token_data, ids, in_strengths, out_strengths, rows, residual_examples = load_oarts_tokens(tokens_path)
    active_output_ids = [int(r["token_id"]) for r, st in zip(rows, out_strengths) if float(st) > 0]

    if not residual_examples:
        print("[DirectionalOARTS] No residual_examples in token file; loading full forget split as fallback.")
        residual_examples = list(load_dataset("locuslab/TOFU", name=forget_split, split="train"))

    print("=" * 80)
    print("Directional OARTS apply")
    print("=" * 80)
    print(f"Model:              {model_name}")
    print(f"Tokens:             {tokens_path}")
    print(f"Output dir:         {args.output_dir}")
    print(f"Input mode:         {args.input_mode}")
    print(f"Input tokens:       {len(ids)}")
    print(f"Output tokens:      {len(active_output_ids)}")
    print(f"Residual examples:  {len(residual_examples)}")
    print(f"Alpha scale:        {args.lmhead_alpha_scale}")
    print(f"Max delta:          {args.lmhead_max_delta}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        device_map=args.device_map,
    )
    model.eval()
    if tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id

    summaries: Dict[str, Any] = {}

    if not args.no_output_edit and active_output_ids:
        maybe_untie_lm_head(model)

    if args.input_mode != "none":
        emb = model.get_input_embeddings().weight.data
        summaries["input"] = adaptive_input_edit(emb, ids, in_strengths, args.input_mode)
        print(f"[DirectionalOARTS] Input edit: {summaries['input']}")
        print(f"[DirectionalOARTS] Input finite: {torch.isfinite(emb).all().item()}")

    if not args.no_output_edit and active_output_ids:
        directions, counts = collect_forget_hidden_directions(
            model=model,
            tokenizer=tokenizer,
            examples=residual_examples,
            active_token_ids=active_output_ids,
            max_length=max_length,
            max_examples=args.max_hidden_examples,
        )
        out_w = model.get_output_embeddings().weight.data
        summaries["directional_output"] = directional_lmhead_edit(
            W=out_w,
            rows=rows,
            out_strengths=out_strengths,
            directions=directions,
            counts=counts,
            alpha_scale=args.lmhead_alpha_scale,
            max_delta=args.lmhead_max_delta,
            min_count=args.lmhead_min_count,
        )
        print(f"[DirectionalOARTS] Output edit: {summaries['directional_output']}")
        print(f"[DirectionalOARTS] Output finite: {torch.isfinite(out_w).all().item()}")

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    summary = {
        "method": "Directional_OARTS_apply",
        "model_name": model_name,
        "tokens_json": str(tokens_path),
        "output_dir": str(save_dir),
        "n_input_tokens": len(ids),
        "n_output_tokens": len(active_output_ids),
        "input_mode": args.input_mode,
        "lmhead_alpha_scale": args.lmhead_alpha_scale,
        "lmhead_max_delta": args.lmhead_max_delta,
        "summaries": summaries,
    }
    with open(save_dir / "directional_oarts_apply_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] Saved: {save_dir}")


if __name__ == "__main__":
    main()
