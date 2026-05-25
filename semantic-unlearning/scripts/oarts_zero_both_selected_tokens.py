#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
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


def select_token_rows(data: Dict[str, Any], zero_scope: str) -> Tuple[List[int], List[Dict[str, Any]]]:
    rows = data.get("semantic_tokens", [])

    if not rows:
        ids = sorted(set(int(x) for x in data.get("token_ids", [])))
        return ids, [{"token_id": tid} for tid in ids]

    selected = []

    for r in rows:
        if zero_scope == "all":
            keep = True
        elif zero_scope == "stage2_residual":
            keep = bool(r.get("is_residual_answer", False))
        elif zero_scope == "lmhead_marked":
            keep = bool(r.get("edit_lm_head", False)) or float(r.get("output_strength", 0.0)) > 0.0
        elif zero_scope == "zero_retain":
            keep = int(r.get("freq_retain", 999999)) == 0
        elif zero_scope == "zero_retain_or_residual":
            keep = int(r.get("freq_retain", 999999)) == 0 or bool(r.get("is_residual_answer", False))
        else:
            raise ValueError(f"Unknown zero_scope: {zero_scope}")

        if keep:
            selected.append(dict(r))

    ids = sorted(set(int(r["token_id"]) for r in selected))
    return ids, selected


@torch.no_grad()
def zero_rows(W: torch.Tensor, token_ids: List[int], name: str) -> Dict[str, Any]:
    vocab = W.shape[0]
    ids = sorted(set(int(x) for x in token_ids if 0 <= int(x) < vocab))

    if not ids:
        print(f"[ZeroBoth] No valid rows for {name}")
        return {"matrix": name, "n_rows": 0}

    ids_t = torch.tensor(ids, dtype=torch.long, device=W.device)

    before = torch.linalg.vector_norm(W[ids_t].float(), dim=1)
    before_mean = float(before.mean().detach().cpu())
    before_max = float(before.max().detach().cpu())

    W[ids_t] = 0

    after = torch.linalg.vector_norm(W[ids_t].float(), dim=1)
    after_mean = float(after.mean().detach().cpu())
    finite = bool(torch.isfinite(W).all().detach().cpu())

    print(f"[ZeroBoth] {name}: zeroed {len(ids)} rows")
    print(f"[ZeroBoth] {name}: norm mean {before_mean:.6f} -> {after_mean:.6f}; before max {before_max:.6f}")
    print(f"[ZeroBoth] {name}: finite={finite}")

    return {
        "matrix": name,
        "n_rows": len(ids),
        "before_norm_mean": before_mean,
        "before_norm_max": before_max,
        "after_norm_mean": after_mean,
        "finite": finite,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tokens-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument(
        "--zero-scope",
        default="all",
        choices=[
            "all",
            "stage2_residual",
            "lmhead_marked",
            "zero_retain",
            "zero_retain_or_residual",
        ],
    )
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")

    tokens_path = Path(args.tokens_json)
    output_dir = Path(args.output_dir)

    data = load_json(tokens_path)
    token_ids, selected_rows = select_token_rows(data, args.zero_scope)

    print("=" * 80)
    print("Stage1 + Stage2 selected-token hard zero")
    print("=" * 80)
    print(f"Model:       {model_name}")
    print(f"Token file:  {tokens_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Zero scope:  {args.zero_scope}")
    print(f"Selected unique tokens: {len(token_ids)}")
    print("=" * 80)

    if len(token_ids) == 0:
        raise RuntimeError("No tokens selected. Check token JSON and --zero-scope.")

    source_counts = {}
    bucket_counts = {}

    for r in selected_rows:
        src = str(r.get("source", "unknown"))
        bucket = str(r.get("bucket", "unknown"))
        source_counts[src] = source_counts.get(src, 0) + 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    print("[ZeroBoth] Source counts:", source_counts)
    print("[ZeroBoth] Bucket counts:", bucket_counts)

    print("\n[ZeroBoth] First 80 selected tokens:")
    for r in selected_rows[:80]:
        print(
            f"{int(r['token_id']):>8} | {repr(str(r.get('token_str', '')))} | "
            f"src={r.get('source')} bucket={r.get('bucket')} | "
            f"forget={r.get('freq_forget')} retain={r.get('freq_retain')} "
            f"residual={r.get('is_residual_answer')}"
        )

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

    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    if lm_head is None or not hasattr(lm_head, "weight"):
        raise RuntimeError("No lm_head / output embedding weight found.")

    tied = emb.weight.data_ptr() == lm_head.weight.data_ptr()
    print(f"[ZeroBoth] lm_head tied to embedding: {tied}")

    emb_stats = zero_rows(emb.weight.data, token_ids, "input_embeddings")
    lm_stats = zero_rows(lm_head.weight.data, token_ids, "lm_head")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    summary = {
        "method": "stage1_stage2_zero_embeddings_and_lm_head",
        "model_name": model_name,
        "tokens_json": str(tokens_path),
        "output_dir": str(output_dir),
        "zero_scope": args.zero_scope,
        "n_zeroed_tokens": len(token_ids),
        "source_counts": source_counts,
        "bucket_counts": bucket_counts,
        "lm_head_tied": tied,
        "input_stats": emb_stats,
        "lm_head_stats": lm_stats,
    }

    with open(output_dir / "zero_both_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] Saved model to {output_dir}")


if __name__ == "__main__":
    main()
