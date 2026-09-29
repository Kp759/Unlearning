#!/usr/bin/env python3
"""Multi-fact person benchmark, step 1: data + an untrained row bank at one layer.

Same contract as prepare_mquake_association_source.py (the MQuAKE direct
machinery is reused unchanged): the locked forget facts, an artifact with the
layer, facts, subject patterns and all-zero residual rows, and a manifest.
`fit_linear_router.py --run-dir` accepts the output; the router trains on each
forget fact's direct prompt plus content-free context prefixes. Retain facts,
multi-fact sentences and the "{S} {vp}" prompts are never read here.

    python -u scripts/prepare_multifact_association_source.py \
        --model-path <llama> --training-visible <data>/training_visible_forget.json \
        --split-manifest <data>/split_manifest.json \
        --output-dir outputs/multifact_person_v1/seed1/L19/prep --layer 19 --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from layer_sweep_utils import boundary_norms
from mquake_fact_association_embeddings import BASE_PLAN
from multifact_person_data import DATASET, load_multifact_forget
from static_overlap_fact_association_embeddings import make_subject_patterns

ARCHITECTURE = "untrained_association_rows_v1"


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

    split_manifest, sampling, records, facts, case_to_fact_id, dedup = load_multifact_forget(
        visible_path, split_path
    )
    seed = int(split_manifest["seed"])

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

    prompts = [p for fact in facts for p in (fact.get("canonical_prompts") or [fact["canonical_prompt"]])]
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
    eval_probes_path = str(split_manifest.get("eval_probes_path", ""))

    artifact = {
        "architecture": ARCHITECTURE,
        "layer": int(args.layer),
        "facts": facts,
        "subject_patterns": make_subject_patterns(tokenizer, facts),
        "rows": torch.zeros((len(facts), int(model.config.hidden_size)), dtype=torch.float32),
        "dataset": DATASET,
        "seed": seed,
        "forget_num_instances": int(sampling["forget_num_instances"]),
        "forget_atomic_record_count": len(records),
        "unique_forget_association_count": len(facts),
        "association_deduplication": dedup,
        "atomic_case_to_association_id": case_map,
        "target_new_used": False,
        "unknown_or_replacement_target_used": False,
        "training_probe_scope": "direct cloze of the forget facts only",
        "retain_facts_used_for_training_or_selection": False,
        "multi_fact_sentences_used_for_training_or_selection": False,
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    manifest = {
        "method": "sure_linear_router_multifact_person",
        "architecture": ARCHITECTURE,
        "dataset": DATASET,
        "seed": seed,
        "forget_num_instances": int(sampling["forget_num_instances"]),
        "retain_person_count": int(sampling["retain_person_count"]),
        "forget_atomic_record_count": len(records),
        "unique_forget_association_count": len(facts),
        "model_path": str(model_path),
        "training_visible_path": str(visible_path),
        "split_manifest_path": str(split_path),
        "eval_probes_path": eval_probes_path,
        "eval_probes_sha256": split_manifest.get("eval_probes_sha256"),
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
    print(json.dumps({"status": "multifact_association_source_ready", **representation,
                      "forget_facts": len(facts), "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
