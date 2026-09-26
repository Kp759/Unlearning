#!/usr/bin/env python3
"""MQuAKE layer sweep, step 1: data + an untrained row bank at one layer.

No Router V2. Writes a run directory that `fit_linear_router.py --run-dir`
accepts: the locked seed-1 MQuAKE forget associations (deduplicated exactly as
the shipped runner does), an artifact with the layer, facts, subject patterns,
all-zero residual rows and the metadata the official MQuAKE evaluator checks,
and a manifest with the registered split counts. The linear router builds its
training prompts from each association's direct requested_rewrite prompts
(plus content-free context prefixes), as for the shipped MQuAKE linear router.

    python -u scripts/prepare_mquake_association_source.py \
        --model-path <llama> --training-visible <visible.json> \
        --split-manifest <split_manifest.json> \
        --output-dir outputs/<sweep>/L07/prep --layer 7 --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from layer_sweep_utils import boundary_norms
from mquake_fact_association_embeddings import (
    BASE_PLAN,
    build_association_facts,
    load_locked_visible_forget,
)
from static_overlap_fact_association_embeddings import make_subject_patterns

ARCHITECTURE = "untrained_association_rows_v1"


def load_mquake_forget(training_visible, split_manifest_path):
    split_manifest = json.loads(Path(split_manifest_path).read_text())
    if int(split_manifest.get("seed", -1)) != 1:
        raise ValueError("Locked MQuAKE split must be seed 1")
    sampling = split_manifest.get("sampling", {})
    if int(sampling.get("forget_num_instances", -1)) != 50:
        raise ValueError("Locked MQuAKE split must have 50 forget instances")
    if int(sampling.get("retain_num_instances", -1)) != 1000:
        raise ValueError("Locked MQuAKE split must have 1000 retain instances")
    records = load_locked_visible_forget(Path(training_visible))
    if len(records) != int(sampling.get("forget_atomic_fact_count", -1)):
        raise ValueError("Atomic forget-record count does not match the split manifest")
    facts, case_to_fact_id, dedup = build_association_facts(records)
    return split_manifest, sampling, records, facts, case_to_fact_id, dedup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--reference-layer", type=int, default=BASE_PLAN["layer"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).resolve()
    visible_path = Path(args.training_visible).resolve()
    split_path = Path(args.split_manifest).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    split_manifest, sampling, records, facts, case_to_fact_id, dedup = load_mquake_forget(
        visible_path, split_path
    )

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

    prompts = [
        prompt for fact in facts
        for prompt in (fact.get("canonical_prompts") or [fact["canonical_prompt"]])
    ]
    norms = boundary_norms(model, tokenizer, prompts, sorted({args.layer, args.reference_layer}))
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
    case_map = {str(case_id): fact_id for case_id, fact_id in case_to_fact_id.items()}

    artifact = {
        "architecture": ARCHITECTURE,
        "layer": int(args.layer),
        "facts": facts,
        "subject_patterns": make_subject_patterns(tokenizer, facts),
        "rows": torch.zeros((len(facts), int(model.config.hidden_size)), dtype=torch.float32),
        # Metadata the official MQuAKE evaluator validates; carried through
        # fit_linear_router.py into the router artifact.
        "dataset": "MQuAKE-CF-3k-v2",
        "seed": 1,
        "forget_num_instances": 50,
        "forget_atomic_record_count": len(records),
        "unique_forget_association_count": len(facts),
        "association_deduplication": dedup,
        "atomic_case_to_association_id": case_map,
        "target_new_used": False,
        "unknown_or_replacement_target_used": False,
        "training_probe_scope": "direct requested_rewrite only",
        "atomic_questions_used_for_training_or_selection": False,
        "multihop_questions_used_for_training_or_selection": False,
        "retain_records_used_for_training_or_selection": False,
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    manifest = {
        "method": "sure_linear_router_layer_sweep_mquake",
        "architecture": ARCHITECTURE,
        "dataset": "MQuAKE-CF-3k-v2",
        "seed": 1,
        "forget_num_instances": 50,
        "retain_num_instances_final_evaluation": 1000,
        "forget_atomic_record_count": len(records),
        "unique_forget_association_count": len(facts),
        "model_path": str(model_path),
        "training_visible_path": str(visible_path),
        "split_manifest_path": str(split_path),
        "source_dataset": split_manifest.get("source_dataset"),
        "source_revision": split_manifest.get("source_revision"),
        "source_sha256": split_manifest.get("source_sha256"),
        "training_visible_sha256": split_manifest.get("training_visible_sha256"),
        "forget_atomic_case_ids": list(sampling.get("forget_atomic_case_ids", [])),
        "atomic_case_to_association_id": case_map,
        "association_deduplication": dedup,
        "plan": {**BASE_PLAN, "layer": int(args.layer),
                 "radius_schedule": [list(x) for x in BASE_PLAN["radius_schedule"]]},
        "facts": facts,
        "router_v2_used": False,
        "target_new_used": False,
        "layer_representation": representation,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "mquake_association_source_ready", **representation,
                      "atomic_records": len(records), "unique_associations": len(facts),
                      "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
