#!/usr/bin/env python3
"""Build a per-seed canonical MCF split with original target_true as sensitive.

The canonical SURE MCF implementation expects the sensitive answer in the
``target_new`` slot.  This adapter preserves the canonical Stage-1/Stage-2
implementation while changing only the semantic mapping used by the locked
training view:

    canonical target_new  <- original target_true   (sensitive)
    canonical target_true <- original target_new    (counterfactual reference)

Only the 50 direct forget requests are written to the training-visible file.
Paraphrases, neighborhoods, generation prompts, all retain records, and PPL
text remain evaluation-only.  Final evaluation must use the original unswapped
MCF source file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from mcf_sampling import sample_official_mcf_records

PROTOCOL = "mcf_sure_canonical_target_true_sensitive_v1"
METRIC_SCHEMA = "mcf_target_true_sensitive_v2"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalize_rr(record: Mapping[str, Any]) -> Mapping[str, Any]:
    rr = record["requested_rewrite"]
    if isinstance(rr, list):
        if len(rr) != 1:
            raise ValueError("Expected one requested_rewrite entry")
        rr = rr[0]
    if not isinstance(rr, Mapping):
        raise ValueError("requested_rewrite must be a mapping")
    return rr


def direct_only_swapped_record(raw: Mapping[str, Any], case_id: int) -> Dict[str, Any]:
    rr = normalize_rr(raw)
    original_new = rr.get("target_new", {}).get("str")
    original_true = rr.get("target_true", {}).get("str")
    if not original_new or not original_true:
        raise ValueError(f"MCF record {case_id} lacks target_new/target_true")
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(rr["prompt"]),
            "subject": str(rr["subject"]),
            # Canonical SURE MCF treats target_new as the sensitive slot.
            "target_new": {"str": str(original_true)},
            "target_true": {"str": str(original_new)},
        },
        "semantic_adapter": {
            "original_sensitive_field": "target_true",
            "original_reference_field": "target_new",
            "canonical_sensitive_slot": "target_new",
            "canonical_reference_slot": "target_true",
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    a = p.parse_args()
    if a.forget_num <= 0 or a.retain_num <= 0:
        raise ValueError("forget-num and retain-num must be positive")

    source = Path(a.mcf_path).resolve()
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(x, dict) for x in raw):
        raise ValueError("MCF source must be a JSON list of objects")

    identity = {id(record): index for index, record in enumerate(raw)}
    forget_raw, retain_raw = sample_official_mcf_records(
        raw,
        forget_num=a.forget_num,
        retain_num=a.retain_num,
        seed=a.seed,
        strict=True,
    )
    forget_ids = [identity[id(record)] for record in forget_raw]
    retain_ids = [identity[id(record)] for record in retain_raw]
    half = len(raw) // 2
    if any(i < half for i in forget_ids):
        raise AssertionError("Forget sample escaped second-half pool")
    if any(i >= half for i in retain_ids):
        raise AssertionError("Retain sample escaped first-half pool")
    if set(forget_ids) & set(retain_ids):
        raise AssertionError("Forget/retain samples overlap")

    visible = [
        direct_only_swapped_record(record, case_id)
        for case_id, record in zip(forget_ids, forget_raw)
    ]
    for record in visible:
        if (
            record["paraphrase_prompts"]
            or record["neighborhood_prompts"]
            or record["generation_prompts"]
        ):
            raise AssertionError("Training-visible MCF exposes held-out probes")

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    visible_path = out / "training_visible_forget.json"
    visible_text = json.dumps(visible, indent=2, ensure_ascii=False) + "\n"
    visible_path.write_text(visible_text, encoding="utf-8")

    manifest = {
        "schema_version": 3,
        "protocol": PROTOCOL,
        "metric_schema": METRIC_SCHEMA,
        "dataset": "mcf",
        "seed": int(a.seed),
        "source_dataset": str(source),
        "training_visible_forget": str(visible_path),
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_sha256": sha256_bytes(visible_text.encode("utf-8")),
        "dataset_size": len(raw),
        "target_semantics": {
            "original_sensitive_field": "target_true",
            "original_reference_field": "target_new",
            "training_sensitive_slot": "target_new",
            "training_reference_slot": "target_true",
            "final_evaluation_uses_original_unswapped_fields": True,
        },
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {
                "start": half,
                "stop_exclusive": len(raw),
                "size": len(raw) - half,
            },
        },
        "sampling": {
            "implementation": "sample_official_mcf_records",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": int(a.forget_num),
            "retain_num": int(a.retain_num),
            "forget_case_ids": [int(x) for x in forget_ids],
            "retain_case_ids": [int(x) for x in retain_ids],
        },
        "data_roles": {
            "stage1_visible": [
                "forget.requested_rewrite.prompt",
                "original forget.requested_rewrite.target_true mapped to canonical target_new",
            ],
            "stage2_visible": [
                "forget.requested_rewrite.prompt",
                "original target_true mapped to canonical target_new",
                "original target_new mapped to canonical target_true",
            ],
            "evaluation_only": [
                "forget.paraphrases",
                "forget.neighborhoods",
                "forget.generation",
                "1000 sampled retain records",
                "PPL text",
            ],
            "same_underlying_forget_facts_at_final_eval": True,
            "prompt_level_holdout": True,
            "final_evaluation_uses_original_source_file": True,
        },
    }
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("target-true-sensitive canonical training set:", visible_path)
    print("split manifest:", manifest_path)
    print("training semantic mapping: original target_true -> canonical target_new (sensitive)")
    print("final evaluation semantic mapping: ORIGINAL UNSWAPPED MCF")
    print(
        f"seed={a.seed}: train/repair={a.forget_num} forget + 0 retain; "
        f"final eval={a.forget_num} forget + {a.retain_num} retain"
    )


if __name__ == "__main__":
    main()
