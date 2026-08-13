#!/usr/bin/env python3
"""Build the exact ZeroUnlearn ZsRE split without any neutral/replacement target.

Sampling is identical to ZeroUnlearn:
  * first half of zsre_mend_eval.json -> retain pool;
  * second half -> forget pool;
  * one random.Random(seed);
  * sample forget first, then retain.

Only the 50 sampled forget direct rewrites are materialized for SURE Stage 1/2.
The training-visible artifact contains the original sensitive target_true only.
It contains no target_new, no Unknown/IDK target, no rephrase/locality prompts,
and no retain records. Final evaluation reopens the unchanged source dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import zsre_zero_unlearn_official_eval as zsre

PROTOCOL = "zsre_zerounlearn_forget_only_locked_no_neutral"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def direct_only_record(raw: Mapping[str, Any], case_id: int) -> Dict[str, Any]:
    required = ("src", "subject", "answers", "rephrase", "loc", "loc_ans")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"ZsRE record {case_id} is missing fields: {missing}")
    answers = raw["answers"]
    if not isinstance(answers, list) or not answers:
        raise ValueError(f"ZsRE record {case_id} has no original answer")
    subject = str(raw["subject"])
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(raw["src"]).replace(subject, "{}"),
            "subject": subject,
            "target_true": {"str": str(answers[0])},
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def assert_locked(records: List[Dict[str, Any]]) -> None:
    for record in records:
        if record.get("paraphrase_prompts"):
            raise AssertionError("training-visible ZsRE exposes a paraphrase")
        if record.get("neighborhood_prompts"):
            raise AssertionError("training-visible ZsRE exposes locality")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, dict):
            raise AssertionError("training-visible ZsRE lacks requested_rewrite")
        if "target_new" in rr:
            raise AssertionError("target_new leaked into no-neutral ZsRE split")
        target_true = rr.get("target_true", {}).get("str")
        if not target_true:
            raise AssertionError("training-visible ZsRE lacks target_true")
        if str(target_true).strip().lower() == "unknown":
            raise AssertionError("literal Unknown appeared as the sensitive target")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--zsre-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--zsre-url", default=zsre.ZSRE_URL)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.retain_num <= 0:
        raise ValueError("forget-num and retain-num must be positive")

    source = zsre.download_zsre(Path(a.zsre_path).resolve(), url=a.zsre_url)
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("ZsRE source must be a JSON list of objects")

    forget_pairs, retain_pairs = zsre.sample_official_zsre_raw_records(
        raw,
        forget_num=a.forget_num,
        retain_num=a.retain_num,
        seed=a.seed,
        strict=True,
    )
    visible = [direct_only_record(record, index) for index, record in forget_pairs]
    assert_locked(visible)

    out = Path(a.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    visible_path = out / "training_visible_forget.json"
    visible_text = json.dumps(visible, ensure_ascii=False, indent=2) + "\n"
    visible_path.write_text(visible_text, encoding="utf-8")

    half = len(raw) // 2
    forget_ids = [int(index) for index, _ in forget_pairs]
    retain_ids = [int(index) for index, _ in retain_pairs]
    if set(forget_ids) & set(retain_ids):
        raise AssertionError("official ZsRE forget/retain samples overlap")
    if any(index < half for index in forget_ids):
        raise AssertionError("forget sample escaped ZeroUnlearn second-half pool")
    if any(index >= half for index in retain_ids):
        raise AssertionError("retain sample escaped ZeroUnlearn first-half pool")

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "source_dataset": str(source),
        "training_visible_forget": str(visible_path),
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_sha256": sha256_bytes(visible_text.encode("utf-8")),
        "dataset_size": len(raw),
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {"start": half, "stop_exclusive": len(raw), "size": len(raw)-half},
        },
        "sampling": {
            "implementation": "sample_official_zsre_raw_records",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": int(a.forget_num),
            "retain_num": int(a.retain_num),
            "forget_case_ids": forget_ids,
            "retain_case_ids": retain_ids,
        },
        "data_roles": {
            "stage1_visible": ["forget.requested_rewrite.target_true"],
            "stage2_visible": ["forget.requested_rewrite.target_true"],
            "evaluation_only": ["forget.rephrase", "forget.locality", "1000 sampled retain records"],
            "target_new_visible": False,
            "neutral_or_replacement_target_visible": False,
            "same_underlying_forget_facts_at_final_eval": True,
            "prompt_level_holdout": True,
            "final_evaluation_uses_original_source_file": True,
        },
    }
    manifest_path = out / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("training-visible no-neutral ZsRE forget set:", visible_path)
    print("split manifest:", manifest_path)
    print(
        f"seed={a.seed}: train/repair={a.forget_num} forget + 0 retain; "
        f"target_new/Unknown=0; final eval={a.forget_num} forget + {a.retain_num} retain"
    )


if __name__ == "__main__":
    main()
