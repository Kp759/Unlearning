#!/usr/bin/env python3
"""Evaluate only the fixed ZeroUnlearn-compatible Wikidata perplexity text.

This diagnostic intentionally does not load MCF/ZsRE benchmark records or any
held-out prompts. It is safe to use between Stage 1 and Stage 2 when diagnosing
utility degradation, provided the resulting PPL is not used to tune or select
final benchmark checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--wikidata-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--max-input-length", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_path)
    wikidata_dir = Path(args.wikidata_dir)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not wikidata_dir.exists():
        raise FileNotFoundError(wikidata_dir)

    tok = AutoTokenizer.from_pretrained(str(model_path))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    load_kwargs = {"torch_dtype": dtype_from_str(args.dtype)}
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
    if args.device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")

    model.eval()
    model.config.use_cache = False
    device = next(model.parameters()).device
    ppl_text = load_official_ppl_text(wikidata_dir)
    if ppl_text is None:
        raise RuntimeError(f"Could not load PPL text from {wikidata_dir}")

    ppl = official_perplexity(
        model,
        tok,
        ppl_text,
        device,
        max_input_length=args.max_input_length,
    )

    payload = {
        "schema_version": 1,
        "kind": "ppl_only_diagnostic",
        "label": args.label,
        "model_path": str(model_path.resolve()),
        "wikidata_dir": str(wikidata_dir.resolve()),
        "ppl": float(ppl),
        "max_input_length": int(args.max_input_length),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "benchmark_records_loaded": 0,
        "zsre_rephrases_loaded": 0,
        "zsre_locality_loaded": 0,
        "zsre_retain_loaded": 0,
        "selection_or_tuning_use": False,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{args.label} PPL: {ppl:.6f}")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
