#!/usr/bin/env python3
"""
scripts/apply_token_policy_edits.py

Apply retain-count-aware per-token edit policy.

Expected token fields:
  edit_input: bool
  input_mode: "zero" | "adaptive_mean" | "none"
  edit_lm_head: bool
  output_mode: "zero" | "none"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import yaml
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


def force_untie_lm_head(model) -> None:
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None or not hasattr(out, "weight"):
        raise RuntimeError("No output embeddings/lm_head weight found.")
    if emb.weight.data_ptr() != out.weight.data_ptr():
        print("[PolicyEdit] lm_head already untied.")
        return
    print("[PolicyEdit] lm_head tied. Untying before independent output edits.")
    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False)
    new_head = new_head.to(device=old.device, dtype=old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


@torch.no_grad()
def apply_input_edits(W: torch.Tensor, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    vocab = W.shape[0]
    edit_ids = {int(r["token_id"]) for r in rows if r.get("edit_input", False) and r.get("input_mode") != "none"}
    mask = torch.ones(vocab, dtype=torch.bool, device=W.device)
    if edit_ids:
        ids_t = torch.tensor(sorted(edit_ids), dtype=torch.long, device=W.device)
        mask[ids_t] = False
    retain_mean = W[torch.arange(vocab, device=W.device)[mask]].float().mean(dim=0).to(W.dtype)

    n_zero = 0
    n_mean = 0
    for r in rows:
        if not r.get("edit_input", False):
            continue
        mode = str(r.get("input_mode", "none"))
        tid = int(r["token_id"])
        if not (0 <= tid < vocab):
            continue
        if mode == "zero":
            W[tid] = 0
            n_zero += 1
        elif mode == "adaptive_mean":
            strength = max(0.0, min(1.0, float(r.get("erase_strength", 1.0))))
            old = W[tid].clone()
            W[tid] = (1.0 - strength) * old + strength * retain_mean
            n_mean += 1
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown input_mode: {mode}")
    print(f"[PolicyEdit] input edits: zero={n_zero}, adaptive_mean={n_mean}, finite={torch.isfinite(W).all().item()}")
    return {"n_input_zero": n_zero, "n_input_mean": n_mean}


@torch.no_grad()
def apply_output_edits(W: torch.Tensor, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    vocab = W.shape[0]
    n_zero = 0
    for r in rows:
        if not r.get("edit_lm_head", False):
            continue
        mode = str(r.get("output_mode", "none"))
        tid = int(r["token_id"])
        if not (0 <= tid < vocab):
            continue
        if mode == "zero":
            W[tid] = 0
            n_zero += 1
        elif mode == "none":
            pass
        else:
            raise ValueError(f"Unknown output_mode: {mode}")
    print(f"[PolicyEdit] output/lm_head edits: zero={n_zero}, finite={torch.isfinite(W).all().item()}")
    return {"n_output_zero": n_zero}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--tokens-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")
    data = load_json(Path(args.tokens_json))
    rows = data.get("semantic_tokens", [])
    if not rows:
        raise RuntimeError("No semantic_tokens found in token JSON.")

    print("=" * 80)
    print("Apply retain-count token policy edits")
    print("=" * 80)
    print("model:", model_name)
    print("tokens:", args.tokens_json)
    print("output:", args.output_dir)
    print("n rows:", len(rows))
    bucket_counts = defaultdict(int)
    for r in rows:
        bucket_counts[str(r.get("bucket", "unknown"))] += 1
    print("bucket_counts:", dict(bucket_counts))
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

    force_untie_lm_head(model)
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()

    input_stats = apply_input_edits(emb.weight.data, rows)
    output_stats = apply_output_edits(out.weight.data, rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "method": "retain_count_policy_edits",
        "model_name": model_name,
        "tokens_json": args.tokens_json,
        "output_dir": args.output_dir,
        "n_rows": len(rows),
        "bucket_counts": dict(bucket_counts),
        "input_stats": input_stats,
        "output_stats": output_stats,
    }
    with open(out_dir / "policy_edit_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[Done] saved:", out_dir)


if __name__ == "__main__":
    main()
