#!/usr/bin/env python3
"""Eval-DU+ for SURE, step 1: forget facts + an untrained row bank at one layer.

    python -u scripts/prepare_evaldu_association_source.py \
        --model-path outputs/evaldu_plus_v1/ft_mul_chunk \
        --split-manifest outputs/evaldu_plus_v1/seed1/data/split_manifest.json \
        --output-dir outputs/evaldu_plus_v1/seed1/L19/prep --layer 19 --local-files-only

--model-path is the model fine-tuned on FT-Mul-Chunk (it knows the facts).
One row per forget fact with a training prompt; either of the fact's people
makes a prompt eligible for its head (a family fact can be asked from both
sides). fit_linear_router.py trains the heads on the UL prefixes plus
content-free context prefixes; test and chunk probes are never read here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from evaldu_plus_data import DATASET, bank_facts, subject_patterns
from layer_sweep_utils import boundary_norms
from mquake_fact_association_embeddings import BASE_PLAN

ARCHITECTURE = "untrained_association_rows_v1"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--reference-layer", type=int, default=BASE_PLAN["layer"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).resolve()
    split_path = Path(args.split_manifest).resolve()
    split = json.loads(split_path.read_text())
    if split.get("dataset") != DATASET:
        raise ValueError(f"Not a {DATASET} split manifest")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records, facts = bank_facts(split)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                              local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)
    block_count = len(model.model.layers)
    for name in ("layer", "reference_layer"):
        if not 0 <= int(getattr(args, name)) < block_count:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, {block_count - 1}]")
    norms = boundary_norms(model, tokenizer, [r["prefix"] for r in records],
                           sorted({args.layer, args.reference_layer}))
    representation = {
        "layer": int(args.layer), "block_count": block_count,
        "relative_depth": round(args.layer / max(block_count - 1, 1), 4),
        "boundary_norm_median": float(norms[args.layer].median()),
        "boundary_norm_mean": float(norms[args.layer].mean()),
        "reference_layer": int(args.reference_layer),
        "reference_boundary_norm_median": float(norms[args.reference_layer].median()),
        "hidden_source": "raw decoder-block output (pre final norm)",
    }
    artifact = {
        "architecture": ARCHITECTURE,
        "layer": int(args.layer),
        "facts": facts,
        "subject_patterns": subject_patterns(tokenizer, facts),
        "rows": torch.zeros((len(facts), int(model.config.hidden_size)), dtype=torch.float32),
        "dataset": DATASET, "seed": int(split["seed"]), "split": split["split"],
        "forget_facts": len(split["forget"]), "forget_facts_in_bank": len(facts),
        "target_new_used": False, "unknown_or_replacement_target_used": False,
        "training_probe_scope": f"UL-{split['unlearn_data']} prefixes of the forget facts",
        "retain_facts_used_for_training_or_selection": False,
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    manifest = {
        "method": "sure_linear_router_evaldu_plus",
        "architecture": ARCHITECTURE,
        "dataset": DATASET, "seed": int(split["seed"]), "split": split["split"],
        "model_path": str(model_path),
        "split_manifest_path": str(split_path),
        "eval_probes_path": split["eval_probes_path"],
        "eval_probes_sha256": split["eval_probes_sha256"],
        "forget_facts": len(split["forget"]), "forget_facts_in_bank": len(facts),
        "training_records": len(records),
        "plan": {**BASE_PLAN, "layer": int(args.layer),
                 "radius_schedule": [list(x) for x in BASE_PLAN["radius_schedule"]]},
        "facts": facts,
        "router_v2_used": False, "target_new_used": False,
        "layer_representation": representation,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "evaldu_association_source_ready", **representation,
                      "facts_in_bank": len(facts), "training_records": len(records),
                      "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
