#!/usr/bin/env python3
"""Audit natural-writer forget scores after native BF16 checkpoint reload.

This runs only on the saved training/development examples. It does not open
official MCF paraphrase or neighborhood prompts. The purpose is to separate
export/dtype drift from genuine official-prompt generalization failure.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path

import torch

from static_overlap_core import answer_nll, model_logits
from static_overlap_data import Example


def _load_examples(path):
    allowed = {field.name for field in fields(Example)}
    rows = json.loads(Path(path).read_text())
    return [Example(**{key: row[key] for key in allowed}) for row in rows]


@torch.no_grad()
def score(model, examples, target_probability):
    result = {}
    for split in ("train", "development"):
        current = [
            example
            for example in examples
            if example.split == split and example.role == "forget"
        ]
        if not current:
            raise ValueError(f"Missing forget examples for {split}")
        rows = []
        for example in current:
            nll = float(answer_nll(model_logits(model, example), example))
            rows.append({
                "id": example.id,
                "fact_id": example.fact_id,
                "nll": nll,
                "token_probability": math.exp(-nll),
            })
        probabilities = [row["token_probability"] for row in rows]
        fact_maxima = {}
        for row in rows:
            fact_maxima[row["fact_id"]] = max(
                fact_maxima.get(row["fact_id"], 0.0),
                row["token_probability"],
            )
        failing = sorted(
            fact_id
            for fact_id, probability in fact_maxima.items()
            if probability >= target_probability
        )
        result[split] = {
            "count": len(rows),
            "facts_total": len(fact_maxima),
            "facts_passing_probability_ceiling": len(fact_maxima) - len(failing),
            "failing_fact_ids": failing,
            "mean_token_probability": sum(probabilities) / len(probabilities),
            "max_token_probability": max(probabilities),
            "target_probability": target_probability,
            "target_met_probability_ceiling": max(probabilities) < target_probability,
            "worst_views": sorted(
                rows,
                key=lambda row: (-row["token_probability"], row["id"]),
            )[:10],
        }
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    run = Path(args.run_dir).resolve()
    checkpoint = run / "checkpoint"
    examples_path = run / "natural_writer_examples.json"
    manifest = json.loads((checkpoint / "training_manifest.json").read_text())
    target = float(manifest["settings"]["target_probability"])
    examples = _load_examples(examples_path)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)
    metrics = score(model, examples, target)
    report = {
        "method": manifest["method"],
        "checkpoint": str(checkpoint),
        "dtype": "torch.bfloat16",
        "official_prompts_opened": False,
        "runtime_router": False,
        "private_tokens": False,
        "metrics": metrics,
    }
    out = Path(args.out).resolve() if args.out else run / "bf16_natural_prompt_audit.json"
    if out.exists():
        raise FileExistsError(f"Audit output already exists: {out}")
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
