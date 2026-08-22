#!/usr/bin/env python3
"""Restore the original Base LM head on a trained H+LoRA checkpoint.

This is a post-training mechanistic diagnostic only.  It leaves the transformer
(including a merged residual LoRA intervention) and input embeddings unchanged,
but replaces the entire output LM-head weight with the original Base head:

    H + LoRA:       logits = W_H h_LoRA
    diagnostic:     logits = W_0 h_LoRA

If forgetting survives after this restoration, the representation intervention
itself carries the unlearning effect rather than relying on the modified SURE-H
reader.  No training or benchmark selection occurs in this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Merged H+LoRA checkpoint")
    p.add_argument("--base-model-path", required=True, help="Original Base checkpoint")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return p.parse_args()


def _dtype(name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def main() -> None:
    a = parse_args()
    dtype = _dtype(a.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, dtype=dtype, device_map="cpu"
    )
    base = AutoModelForCausalLM.from_pretrained(
        a.base_model_path, dtype=dtype, device_map="cpu"
    )
    tok = AutoTokenizer.from_pretrained(a.model_path)

    model_head = model.get_output_embeddings()
    base_head = base.get_output_embeddings()
    if model_head is None or base_head is None:
        raise RuntimeError("both models must expose output embeddings")
    if model_head.weight.shape != base_head.weight.shape:
        raise RuntimeError(
            f"LM-head shape mismatch: {tuple(model_head.weight.shape)} vs "
            f"{tuple(base_head.weight.shape)}"
        )

    input_before = model.get_input_embeddings().weight.detach().clone()
    with torch.no_grad():
        before = model_head.weight.detach().float()
        reference = base_head.weight.detach().float()
        head_delta_norm_before = float((before - reference).norm().cpu())
        head_delta_max_before = float((before - reference).abs().max().cpu())
        model_head.weight.copy_(base_head.weight.to(model_head.weight.dtype))

    after = model_head.weight.detach().float()
    reference = base_head.weight.detach().float()
    head_delta_max_after = float((after - reference).abs().max().cpu())
    input_max_abs_change = float(
        (model.get_input_embeddings().weight.detach() - input_before).abs().max().cpu()
    )
    if head_delta_max_after != 0.0:
        raise RuntimeError("Base LM head was not restored exactly")
    if input_max_abs_change != 0.0:
        raise RuntimeError("input embeddings changed during head restoration")

    # Keep the diagnostic explicitly untied if its parent was untied.  Copying the
    # Base values does not retie storage, so h_LoRA is still read by an exact W0.
    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)

    receipt = {
        "schema_version": 1,
        "operation": "restore_entire_base_lm_head",
        "source_model_path": str(Path(a.model_path).resolve()),
        "base_model_path": str(Path(a.base_model_path).resolve()),
        "output_dir": str(out),
        "head_delta_norm_before": head_delta_norm_before,
        "head_delta_max_before": head_delta_max_before,
        "head_delta_max_after": head_delta_max_after,
        "input_embedding_max_abs_change": input_max_abs_change,
        "training_performed": False,
        "evaluation_data_used": False,
    }
    (out / "restore_base_head_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
