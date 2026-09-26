#!/usr/bin/env python3
"""Step 1 of the linear-classifier layer sweep: data + an untrained row bank.

Writes a run directory that `fit_linear_router.py --run-dir` accepts, with no
Router V2 anywhere: the MCF seed-1 forget facts, their training-visible
prompts (the same builder the shipped runner uses), and an artifact holding
the layer, facts, subject patterns and all-zero residual rows. Also records
the boundary-token norm at the layer and at the reference layer (19), which
step 3 uses for norm-matched training.

    python -u scripts/prepare_mcf_association_source.py \
        --model-path <llama> --mcf-path data/multi_counterfact.json \
        --output-dir outputs/<sweep>/L07/prep --layer 7 --local-files-only
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from layer_sweep_utils import boundary_norms
from mcf_sampling import sample_official_mcf_records
from static_overlap_fact_association_embeddings import (
    PLAN,
    build_forget_examples,
    make_subject_patterns,
)
from static_overlap_natural_writer import mcf_facts

ARCHITECTURE = "untrained_association_rows_v1"


def load_mcf_forget_data(tokenizer, mcf_path, forget_num=50, seed=1):
    records = json.loads(Path(mcf_path).read_text())
    forget_records, _ = sample_official_mcf_records(
        records, forget_num=forget_num, retain_num=0, seed=seed, strict=True,
    )
    facts = mcf_facts(forget_records, "forget")
    examples = build_forget_examples(facts, tokenizer, PLAN["max_length"])
    return forget_records, facts, examples


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--reference-layer", type=int, default=PLAN["layer"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    model.requires_grad_(False)
    block_count = len(model.model.layers)
    for name in ("layer", "reference_layer"):
        if not 0 <= int(getattr(args, name)) < block_count:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0, {block_count - 1}]")

    forget_records, facts, examples = load_mcf_forget_data(tokenizer, mcf_path)
    hidden_size = int(model.config.hidden_size)
    train_prompts = [e.prompt for e in examples if e.split == "train"]
    norms = boundary_norms(
        model, tokenizer, train_prompts, sorted({args.layer, args.reference_layer})
    )
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
        "rows": torch.zeros((len(facts), hidden_size), dtype=torch.float32),
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    manifest = {
        "method": "sure_linear_router_layer_sweep",
        "architecture": ARCHITECTURE,
        "model_path": str(model_path),
        "mcf_path": str(mcf_path),
        "sampling": {
            "forget_num": 50, "retain_num": 0, "seed": 1,
            "convention": "ZeroUnlearn/official MCF forget split",
        },
        "plan": {**PLAN, "layer": int(args.layer),
                 "radius_schedule": [list(x) for x in PLAN["radius_schedule"]]},
        "facts": facts,
        "forget_case_ids": [int(r["case_id"]) for r in forget_records],
        "router_v2_used": False,
        "official_paraphrase_fields_used": False,
        "official_neighborhood_fields_used": False,
        "layer_representation": representation,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "association_examples.json").write_text(
        json.dumps([asdict(e) for e in examples], indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "association_source_ready", **representation,
                      "train_views": len(train_prompts),
                      "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
