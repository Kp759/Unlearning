#!/usr/bin/env python3
"""RWKU layer sweep, step 1: data + an untrained row bank at one layer.

No Router V2. Writes a run directory that `fit_linear_router.py --run-dir`
accepts: the RWKU-Batch-50-v1 forget associations of one batch seed (five
people x ten Level-1/Level-2 probes, deduplicated exactly as the shipped RWKU
runner does), an artifact with the layer, facts, RWKU subject patterns
(canonical name + surname), all-zero residual rows, and a manifest with the
fields the RWKU evaluator checks. The linear router builds its training prompts
from each association's own chat-formatted probe plus content-free lead-ins
inside the question (context-prefix families), as for ZsRE/MQuAKE.

Batch seed s selects people s..s+4 of RWKU's first ten targets (cyclic); the
per-person train / held-out partition is frozen and does not depend on s.

    python -u scripts/prepare_rwku_association_source.py \
        --model-path <llama> --seed 2 --output-dir outputs/<sweep>/seed2/L07/prep --layer 7
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from layer_sweep_utils import boundary_norms
from rwku_batch50 import EVALUATION_ONLY_FILES, PROTOCOL_ID, materialize_batch_split
from rwku_fact_association_embeddings import (
    BASE_PLAN,
    build_association_facts,
    make_rwku_subject_patterns,
)

ARCHITECTURE = "untrained_association_rows_v1"


def load_rwku_forget(data_root, batch_seed, tokenizer, split_dir=None, allow_download=True):
    """The batch's 50 training probes and their deduplicated associations."""
    from rwku_batch50 import build_batch_split

    if split_dir is not None:
        split = materialize_batch_split(
            data_root=Path(data_root), output_dir=Path(split_dir),
            batch_seed=int(batch_seed), allow_download=allow_download,
        )
    else:
        split = build_batch_split(
            data_root=Path(data_root), batch_seed=int(batch_seed),
            allow_download=allow_download,
        )
    rows = list(split["forget_train"])
    if len(rows) != 50:
        raise RuntimeError(f"{PROTOCOL_ID} batch seed {batch_seed} must expose 50 training rows")
    facts, record_to_fact_id, dedup = build_association_facts(rows, tokenizer)
    return split, rows, facts, record_to_fact_id, dedup


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-root", default="data/rwku")
    parser.add_argument("--seed", type=int, required=True, help="RWKU batch seed (0-9)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--reference-layer", type=int, default=BASE_PLAN["layer"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)

    model_path = Path(args.model_path).resolve()
    data_root = Path(args.data_root).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, local_files_only=args.local_files_only
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    split_dir = output / "split"
    split, rows, facts, record_to_fact_id, dedup = load_rwku_forget(
        data_root, args.seed, tokenizer, split_dir=split_dir,
        allow_download=not args.no_download,
    )
    split_manifest = split["manifest"]

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
    subjects = [item["subject"] for item in split_manifest["targets"]]
    artifact = {
        "architecture": ARCHITECTURE,
        "layer": int(args.layer),
        "facts": facts,
        "subject_patterns": make_rwku_subject_patterns(tokenizer, facts),
        "rows": torch.zeros((len(facts), int(model.config.hidden_size)), dtype=torch.float32),
        # RWKU metadata; carried through fit_linear_router.py into the router artifact.
        "method": "sure_linear_router_layer_sweep_rwku",
        "dataset": "RWKU",
        "protocol_id": PROTOCOL_ID,
        "seed": int(args.seed),
        "target_seeds": list(split_manifest["target_seeds"]),
        "forget_train_count": len(rows),
        "unique_forget_association_count": len(facts),
        "association_deduplication": {k: v for k, v in dedup.items() if k != "record_to_fact_id"},
        "source_record_to_association_id": record_to_fact_id,
        "heldout_rwku_probes_used_for_training_or_selection": False,
        "neighbor_mia_utility_used_for_training_or_selection": False,
    }
    torch.save(artifact, output / "fact_association_embeddings.pt")
    manifest = {
        "method": "sure_linear_router_layer_sweep_rwku",
        "architecture": ARCHITECTURE,
        "dataset": "RWKU",
        "protocol_id": PROTOCOL_ID,
        "protocol_status": split_manifest["protocol_status"],
        "seed": int(args.seed),
        "target_seeds": list(split_manifest["target_seeds"]),
        "subjects": subjects,
        "forget_train_count": len(rows),
        "unique_forget_association_count": len(facts),
        "model_path": str(model_path),
        "data_root": str(data_root),
        "split_dir": str(split_dir),
        "split_manifest_path": str(split_dir / "split_manifest.json"),
        "rwku_code_revision": split_manifest["rwku_code_revision"],
        "rwku_dataset_revision": split_manifest["rwku_dataset_revision"],
        "association_deduplication": {k: v for k, v in dedup.items() if k != "record_to_fact_id"},
        "source_record_to_association_id": record_to_fact_id,
        "training_visible": [
            "only the 50 RWKU-Batch-50-v1 selected Level-1/Level-2 probes of this batch",
            "subject", "natural query/context", "original sensitive answer",
        ],
        "evaluation_only": [
            "held-out Level-1 and Level-2 probes",
            "held-out Level-2 deterministic paraphrases",
            *list(EVALUATION_ONLY_FILES),
            "Wikidata PPL text",
        ],
        "plan": {**BASE_PLAN, "layer": int(args.layer),
                 "radius_schedule": [list(x) for x in BASE_PLAN["radius_schedule"]]},
        "facts": facts,
        "router_v2_used": False,
        "replacement_target_used": False,
        "layer_representation": representation,
    }
    (output / "association_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"status": "rwku_association_source_ready", **representation,
                      "batch_seed": int(args.seed), "subjects": subjects,
                      "forget_rows": len(rows), "unique_associations": len(facts),
                      "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
