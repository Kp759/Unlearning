#!/usr/bin/env python3
"""ZsRE layer sweep, step 1: data + an untrained row bank at one layer.

No Router V2. Writes a run directory that `fit_linear_router.py --run-dir`
accepts: the locked seed-1 ZsRE forget facts (built exactly as the shipped
runner does), an artifact with the layer, facts, subject patterns, all-zero
rows and the metadata the official ZsRE evaluator checks, and a manifest with
the locked case ids.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from layer_sweep_utils import boundary_norms
from static_overlap_fact_association_embeddings import make_subject_patterns
from zsre_fact_association_embeddings import (
    PLAN,
    facts_from_locked_records,
    load_locked_visible_forget,
)

ARCHITECTURE = "untrained_association_rows_v1"


def load_zsre_forget(training_visible, split_manifest_path):
    split_manifest = json.loads(Path(split_manifest_path).read_text())
    if int(split_manifest.get("seed", -1)) != 1:
        raise ValueError("Locked ZsRE split must be seed 1")
    sampling = split_manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != 50:
        raise ValueError("Locked ZsRE split must have 50 forget records")
    roles = split_manifest.get("data_roles", {})
    if roles.get("target_new_visible") is not False:
        raise ValueError("ZsRE split must hide target_new")
    records = load_locked_visible_forget(Path(training_visible))
    if len(records) != 50:
        raise ValueError(f"Expected 50 visible ZsRE forget records, got {len(records)}")
    return split_manifest, sampling, records, facts_from_locked_records(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--reference-layer", type=int, default=PLAN["layer"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).resolve()
    visible_path = Path(args.training_visible).resolve()
    split_path = Path(args.split_manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    split_manifest, sampling, records, facts = load_zsre_forget(visible_path, split_path)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
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

    norms = boundary_norms(model, tokenizer, [f["canonical_prompt"] for f in facts],
                           sorted({args.layer, args.reference_layer}))
    representation = {
        "layer": int(args.layer),
        "block_count": block_count,
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
        "subject_patterns": make_subject_patterns(tokenizer, facts),
        "rows": torch.zeros((len(facts), int(model.config.hidden_size)), dtype=torch.float32),
        # Metadata the official ZsRE evaluator validates (carried through the router fit).
        "dataset": "ZsRE",
        "seed": 1,
        "forget_num": 50,
        "target_new_used": False,
        "unknown_or_replacement_target_used": False,
        "training_probe_scope": "direct requested_rewrite only",
        "official_rephrases_used_for_training_or_selection": False,
        "official_locality_used_for_training_or_selection": False,
        "retain_records_used_for_training_or_selection": False,
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    manifest = {
        "method": "sure_linear_router_layer_sweep_zsre",
        "architecture": ARCHITECTURE,
        "dataset": "ZsRE",
        "seed": 1,
        "forget_num": 50,
        "retain_num_final_evaluation": 1000,
        "model_path": str(model_path),
        "training_visible_path": str(visible_path),
        "split_manifest_path": str(split_path),
        "source_dataset": split_manifest.get("source_dataset"),
        "source_sha256": split_manifest.get("source_sha256"),
        "training_visible_sha256": split_manifest.get("training_visible_sha256"),
        "forget_case_ids": list(sampling.get("forget_case_ids", [])),
        "retain_case_ids_final_evaluation": list(sampling.get("retain_case_ids", [])),
        "plan": {**PLAN, "layer": int(args.layer),
                 "radius_schedule": [list(x) for x in PLAN["radius_schedule"]]},
        "facts": facts,
        "router_v2_used": False,
        "target_new_used": False,
        "layer_representation": representation,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "zsre_association_source_ready", **representation,
                      "facts": len(facts), "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
