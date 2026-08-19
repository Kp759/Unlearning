#!/usr/bin/env python3
"""Build fixed-shared SURE MCF split with disjoint retain-train/eval sets.

Training-visible MCF is intentionally symmetric with ZsRE: each forget record
contains only the direct prompt and the sensitive answer.  Original target_true
is mapped into canonical target_new because the shared MCF adapter uses that
slot as sensitive.  Original target_new/reference is NOT exposed to training.

The official 50-forget / 1000-retain-eval sampling is preserved exactly.
A separate retain-train set is sampled deterministically from the remaining
first-half retain pool, guaranteeing zero overlap with official retain eval.
Retain-train exposes only direct prompts; no answer labels or held-out probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping

from mcf_sampling import sample_official_mcf_records
from build_mcf_sure_target_true_canonical_split import normalize_rr

PROTOCOL = "mcf_sure_fixed_shared_retain_v3"
RETAIN_TRAIN_RNG_OFFSET = 104729


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def direct_only_sensitive_record(raw: Mapping[str, Any], case_id: int) -> Dict[str, Any]:
    rr = normalize_rr(raw)
    original_true = rr.get("target_true", {}).get("str")
    if not original_true:
        raise ValueError(f"MCF record {case_id} lacks original target_true")
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(rr["prompt"]),
            "subject": str(rr["subject"]),
            "target_new": {"str": str(original_true)},
        },
        "semantic_adapter": {
            "original_sensitive_field": "target_true",
            "canonical_sensitive_slot": "target_new",
            "reference_answer_exposed_to_training": False,
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def retain_prompt_only_record(raw: Mapping[str, Any], case_id: int) -> Dict[str, Any]:
    rr = normalize_rr(raw)
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(rr["prompt"]),
            "subject": str(rr["subject"]),
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
    p.add_argument("--retain-train-num", type=int, default=1000)
    p.add_argument("--retain-eval-num", type=int, default=1000)
    a = p.parse_args()
    if min(a.forget_num, a.retain_train_num, a.retain_eval_num) <= 0:
        raise ValueError("all split counts must be positive")

    source = Path(a.mcf_path).resolve()
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(x, dict) for x in raw):
        raise ValueError("MCF source must be a JSON list of objects")

    identity = {id(record): index for index, record in enumerate(raw)}
    forget_raw, retain_eval_raw = sample_official_mcf_records(
        raw,
        forget_num=a.forget_num,
        retain_num=a.retain_eval_num,
        seed=a.seed,
        strict=True,
    )
    forget_ids = [identity[id(record)] for record in forget_raw]
    retain_eval_ids = [identity[id(record)] for record in retain_eval_raw]
    half = len(raw) // 2

    retain_eval_set = set(retain_eval_ids)
    remaining_retain_ids = [i for i in range(half) if i not in retain_eval_set]
    if a.retain_train_num > len(remaining_retain_ids):
        raise ValueError("retain-train request exceeds remaining first-half pool")
    rng = random.Random(int(a.seed) + RETAIN_TRAIN_RNG_OFFSET)
    retain_train_ids = rng.sample(remaining_retain_ids, a.retain_train_num)
    retain_train_raw = [raw[i] for i in retain_train_ids]

    if set(retain_train_ids) & retain_eval_set:
        raise AssertionError("retain-train and retain-eval overlap")
    if any(i < half for i in forget_ids):
        raise AssertionError("forget sample escaped second-half pool")
    if any(i >= half for i in retain_eval_ids + retain_train_ids):
        raise AssertionError("retain sample escaped first-half pool")

    forget_visible = [
        direct_only_sensitive_record(record, case_id)
        for case_id, record in zip(forget_ids, forget_raw)
    ]
    retain_visible = [
        retain_prompt_only_record(record, case_id)
        for case_id, record in zip(retain_train_ids, retain_train_raw)
    ]

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    forget_path = out / "training_visible_forget.json"
    retain_path = out / "training_visible_retain.json"
    forget_text = json.dumps(forget_visible, indent=2, ensure_ascii=False) + "\n"
    retain_text = json.dumps(retain_visible, indent=2, ensure_ascii=False) + "\n"
    forget_path.write_text(forget_text, encoding="utf-8")
    retain_path.write_text(retain_text, encoding="utf-8")
    forget_sha = sha256_bytes(forget_text.encode("utf-8"))
    retain_sha = sha256_bytes(retain_text.encode("utf-8"))

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "metric_schema": "mcf_target_true_sensitive_v2",
        "dataset": "mcf",
        "seed": int(a.seed),
        "source_dataset": str(source),
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_forget": str(forget_path),
        "training_visible_forget_sha256": forget_sha,
        "training_visible_sha256": forget_sha,
        "training_visible_retain": str(retain_path),
        "training_visible_retain_sha256": retain_sha,
        "dataset_size": len(raw),
        "target_semantics": {
            "original_sensitive_field": "target_true",
            "original_reference_field": "target_new",
            "training_sensitive_slot": "target_new",
            "reference_answer_exposed_to_training": False,
            "final_evaluation_uses_original_unswapped_fields": True,
        },
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {"start": half, "stop_exclusive": len(raw), "size": len(raw) - half},
        },
        "sampling": {
            "official_forget_and_retain_eval": "sample_official_mcf_records",
            "forget_num": int(a.forget_num),
            "retain_train_num": int(a.retain_train_num),
            "retain_eval_num": int(a.retain_eval_num),
            "forget_case_ids": [int(x) for x in forget_ids],
            "retain_train_case_ids": [int(x) for x in retain_train_ids],
            "retain_eval_case_ids": [int(x) for x in retain_eval_ids],
            "retain_case_ids": [int(x) for x in retain_eval_ids],
            "retain_train_rng_offset": RETAIN_TRAIN_RNG_OFFSET,
            "retain_train_eval_overlap": 0,
        },
        "data_roles": {
            "stage1_visible": ["50 direct forget sensitive answers", "1000 direct retain prompts"],
            "stage2_visible": ["residual direct forget sensitive answers", "same 1000 direct retain prompts"],
            "forget_reference_answer_visible": False,
            "retain_train_answer_labels_visible": False,
            "evaluation_only": ["forget target_new/reference", "forget paraphrases", "forget neighborhoods", "official 1000 retain-eval", "PPL text"],
            "heldout_probes_visible_during_training": False,
            "final_evaluation_uses_original_source_file": True,
        },
    }
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("MCF fixed-shared retain split:", manifest_path)
    print(f"forget={len(forget_visible)} retain_train={len(retain_visible)} retain_eval={len(retain_eval_ids)}")
    print("reference answer exposed to training: False")
    print("retain train/eval overlap: 0")


if __name__ == "__main__":
    main()
