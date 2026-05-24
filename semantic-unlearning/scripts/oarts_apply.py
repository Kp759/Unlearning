#!/usr/bin/env python3
"""
scripts/oarts_apply.py

Apply OARTS adaptive token edits.

Input token JSON must contain semantic_tokens entries with:
  - token_id
  - erase_strength for input embedding edit
  - output_strength for optional lm_head edit
  - edit_lm_head boolean

Default:
  - edit input embeddings with adaptive_mean
  - edit lm_head only for tokens where output_strength > 0
  - if editing lm_head and it is tied, untie first
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_oarts_tokens(path: Path) -> Tuple[List[int], List[float], List[float], List[Dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = data.get("semantic_tokens", [])
    if not rows and "token_ids" in data:
        rows = [{"token_id": int(x), "erase_strength": 1.0, "output_strength": 0.0} for x in data["token_ids"]]

    seen = set()
    ids: List[int] = []
    in_strengths: List[float] = []
    out_strengths: List[float] = []
    kept_rows: List[Dict] = []

    for r in rows:
        tid = int(r["token_id"])
        if tid in seen:
            continue
        seen.add(tid)
        inp = float(r.get("erase_strength", r.get("input_strength", 1.0)))
        out = float(r.get("output_strength", 0.0))
        if not bool(r.get("edit_lm_head", False)):
            out = 0.0
        inp = max(0.0, min(1.0, inp))
        out = max(0.0, min(1.0, out))
        ids.append(tid)
        in_strengths.append(inp)
        out_strengths.append(out)
        kept_rows.append(dict(r))
    return ids, in_strengths, out_strengths, kept_rows


def maybe_untie_lm_head(model) -> None:
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None or not hasattr(out, "weight"):
        raise RuntimeError("Model does not expose output embeddings / lm_head")
    if out.weight.data_ptr() != emb.weight.data_ptr():
        print("[OARTS] lm_head already untied.")
        return
    print("[OARTS] lm_head tied. Untying before output edits.")
    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False)
    new_head = new_head.to(device=old.device, dtype=old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


@torch.no_grad()
def adaptive_input_edit(W: torch.Tensor, ids: List[int], strengths: List[float], mode: str) -> Dict[str, float]:
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
    return {"old_norm": old_norm, "new_norm": new_norm, "strength_mean": float(s.mean().item())}


@torch.no_grad()
def adaptive_output_edit(W: torch.Tensor, ids: List[int], strengths: List[float], mode: str) -> Dict[str, float]:
    active = [(tid, st) for tid, st in zip(ids, strengths) if st > 0]
    if not active:
        return {"n_output_tokens": 0, "old_norm": 0.0, "new_norm": 0.0, "strength_mean": 0.0}
    out_ids = [int(x[0]) for x in active]
    out_strengths = [float(x[1]) for x in active]
    ids_t = torch.tensor(out_ids, dtype=torch.long, device=W.device)
    s = torch.tensor(out_strengths, dtype=torch.float32, device=W.device).clamp(0.0, 1.0).view(-1, 1)
    old = W[ids_t].clone()
    old_norm = torch.linalg.vector_norm(old.float(), dim=1).mean().item()

    if mode == "scale":
        # output_strength=0.98 => row scale 0.02
        W[ids_t] = (1.0 - s.to(W.dtype)) * old
    elif mode == "adaptive_zero":
        W[ids_t] = (1.0 - s.to(W.dtype)) * old
    elif mode == "adaptive_mean":
        vocab = W.shape[0]
        mask = torch.ones(vocab, dtype=torch.bool, device=W.device)
        mask[ids_t] = False
        target_vec = W[mask].float().mean(dim=0).to(W.dtype)
        target = target_vec.unsqueeze(0).expand_as(old)
        W[ids_t] = (1.0 - s.to(W.dtype)) * old + s.to(W.dtype) * target
    else:
        raise ValueError(f"Unknown output mode: {mode}")

    new_norm = torch.linalg.vector_norm(W[ids_t].float(), dim=1).mean().item()
    return {"n_output_tokens": len(out_ids), "old_norm": old_norm, "new_norm": new_norm, "strength_mean": float(s.mean().item())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--tokens-json", default=None)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--input-mode", choices=["adaptive_mean", "adaptive_zero"], default="adaptive_mean")
    ap.add_argument("--output-mode", choices=["scale", "adaptive_zero", "adaptive_mean"], default="scale")
    ap.add_argument("--no-input-edit", action="store_true")
    ap.add_argument("--no-output-edit", action="store_true")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_root = Path(cfg["output"]["dir"])
    tokens_path = Path(args.tokens_json) if args.tokens_json else out_root / "semantic_tokens_oarts.json"
    model_name = args.model_name or cfg["model"]["name"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")

    ids, in_strengths, out_strengths, rows = load_oarts_tokens(tokens_path)
    output_edit_count = sum(1 for x in out_strengths if x > 0)

    print("=" * 80)
    print("OARTS apply")
    print("=" * 80)
    print(f"Model:            {model_name}")
    print(f"Tokens:           {tokens_path}")
    print(f"Output dir:       {args.output_dir}")
    print(f"Input mode:       {args.input_mode}")
    print(f"Output mode:      {args.output_mode}")
    print(f"Input tokens:     {len(ids)}")
    print(f"Output edit toks: {output_edit_count}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        device_map=args.device_map,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id

    summaries: Dict[str, Dict[str, float]] = {}

    if not args.no_output_edit and output_edit_count > 0:
        maybe_untie_lm_head(model)

    if not args.no_input_edit:
        emb = model.get_input_embeddings().weight.data
        summaries["input"] = adaptive_input_edit(emb, ids, in_strengths, args.input_mode)
        print(f"[OARTS] Input edit: {summaries['input']}")
        print(f"[OARTS] Input finite: {torch.isfinite(emb).all().item()}")

    if not args.no_output_edit and output_edit_count > 0:
        out = model.get_output_embeddings().weight.data
        summaries["output"] = adaptive_output_edit(out, ids, out_strengths, args.output_mode)
        print(f"[OARTS] Output edit: {summaries['output']}")
        print(f"[OARTS] Output finite: {torch.isfinite(out).all().item()}")

    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    summary = {
        "method": "OARTS_apply",
        "model_name": model_name,
        "tokens_json": str(tokens_path),
        "output_dir": str(save_dir),
        "n_tokens": len(ids),
        "n_output_edit_tokens": int(output_edit_count),
        "input_mode": args.input_mode,
        "output_mode": args.output_mode,
        "summaries": summaries,
    }
    with open(save_dir / "oarts_apply_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] Saved: {save_dir}")


if __name__ == "__main__":
    main()
