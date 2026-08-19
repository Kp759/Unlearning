#!/usr/bin/env python3
"""Build fixed-shared SURE ZsRE split with disjoint retain-train/eval sets.

The official 50-forget / 1000-retain-eval sampling is preserved exactly.
A separate retain-train set is sampled deterministically from the remaining
first-half retain pool.  Forget training exposes only original target_true as
sensitive; retain training exposes direct prompts only, with no answer labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping

import zsre_zero_unlearn_official_eval as zsre
from build_zsre_zerounlearn_locked_no_neutral_split import direct_only_record

PROTOCOL = "zsre_sure_fixed_shared_retain_v3"
RETAIN_TRAIN_RNG_OFFSET = 104729


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def retain_prompt_only_record(raw: Mapping[str, Any], case_id: int) -> Dict[str, Any]:
    subject = str(raw["subject"])
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(raw["src"]).replace(subject, "{}"),
            "subject": subject,
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zsre-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-train-num", type=int, default=1000)
    p.add_argument("--retain-eval-num", type=int, default=1000)
    p.add_argument("--zsre-url", default=zsre.ZSRE_URL)
    a = p.parse_args()
    if min(a.forget_num, a.retain_train_num, a.retain_eval_num) <= 0:
        raise ValueError("all split counts must be positive")

    source = zsre.download_zsre(Path(a.zsre_path).resolve(), url=a.zsre_url)
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("ZsRE source must be a JSON list of objects")

    forget_pairs, retain_eval_pairs = zsre.sample_official_zsre_raw_records(
        raw,
        forget_num=a.forget_num,
        retain_num=a.retain_eval_num,
        seed=a.seed,
        strict=True,
    )
    forget_ids = [int(index) for index, _ in forget_pairs]
    retain_eval_ids = [int(index) for index, _ in retain_eval_pairs]
    half = len(raw) // 2

    eval_set = set(retain_eval_ids)
    remaining_retain_ids = [i for i in range(half) if i not in eval_set]
    if a.retain_train_num > len(remaining_retain_ids):
        raise ValueError("retain-train request exceeds remaining first-half pool")
    rng = random.Random(int(a.seed) + RETAIN_TRAIN_RNG_OFFSET)
    retain_train_ids = rng.sample(remaining_retain_ids, a.retain_train_num)

    if set(retain_train_ids) & eval_set:
        raise AssertionError("retain-train and retain-eval overlap")
    if any(i < half for i in forget_ids):
        raise AssertionError("forget sample escaped second-half pool")
    if any(i >= half for i in retain_eval_ids + retain_train_ids):
        raise AssertionError("retain sample escaped first-half pool")

    forget_visible = [direct_only_record(record, index) for index, record in forget_pairs]
    retain_visible = [retain_prompt_only_record(raw[index], index) for index in retain_train_ids]

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    forget_path = out / "training_visible_forget.json"
    retain_path = out / "training_visible_retain.json"
    forget_text = json.dumps(forget_visible, ensure_ascii=False, indent=2) + "\n"
    retain_text = json.dumps(retain_visible, ensure_ascii=False, indent=2) + "\n"
    forget_path.write_text(forget_text, encoding="utf-8")
    retain_path.write_text(retain_text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "zsre",
        "seed": int(a.seed),
        "source_dataset": str(source),
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_forget": str(forget_path),
        "training_visible_forget_sha256": sha256_bytes(forget_text.encode("utf-8")),
        "training_visible_retain": str(retain_path),
        "training_visible_retain_sha256": sha256_bytes(retain_text.encode("utf-8")),
        "dataset_size": len(raw),
        "target_semantics": {
            "original_sensitive_field": "target_true",
            "training_sensitive_slot": "target_true",
            "neutral_or_replacement_target_visible": False,
        },
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {"start": half, "stop_exclusive": len(raw), "size": len(raw) - half},
        },
        "sampling": {
            "official_forget_and_retain_eval": "sample_official_zsre_raw_records",
            "forget_num": int(a.forget_num),
            "retain_train_num": int(a.retain_train_num),
            "retain_eval_num": int(a.retain_eval_num),
            "forget_case_ids": forget_ids,
            "retain_train_case_ids": [int(x) for x in retain_train_ids],
            "retain_eval_case_ids": retain_eval_ids,
            "retain_train_rng_offset": RETAIN_TRAIN_RNG_OFFSET,
            "retain_train_eval_overlap": 0,
        },
        "data_roles": {
            "stage1_visible": ["50 direct forget sensitive answers", "1000 direct retain prompts"],
            "stage2_visible": ["residual direct forget sensitive answers", "same 1000 direct retain prompts"],
            "retain_train_answer_labels_visible": False,
            "evaluation_only": ["forget rephrases", "forget locality", "official 1000 retain-eval", "PPL text"],
            "heldout_probes_visible_during_training": False,
        },
    }
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("ZsRE fixed-shared retain split:", manifest_path)
    print(f"forget={len(forget_visible)} retain_train={len(retain_visible)} retain_eval={len(retain_eval_ids)}")
    print("retain train/eval overlap: 0")


if __name__ == "__main__":
    main()
