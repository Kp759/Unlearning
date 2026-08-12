#!/usr/bin/env python3
"""Build a ZeroUnlearn-style locked-probe ZsRE split for one seed.

The exact ZeroUnlearn pool/sampling rule is used:

* first half of zsre_mend_eval.json -> retain pool;
* second half -> forget pool;
* sample forget records first, then retain records from the same seeded RNG.

Only the sampled forget records are written to the repair-visible artifact, and
that artifact contains requested_rewrite only.  Rephrase/locality probes and
all sampled retain records remain evaluation-only and are recovered from the
unchanged original ZsRE file during final evaluation.

This is a prompt-level holdout protocol: final forget evaluation uses the same
underlying sampled facts, but their rephrases/locality probes are unseen by
Stage 1 and Stage 2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import zsre_zero_unlearn_official_eval as zsre


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
            "target_new": {"str": zsre.NEUTRAL_TARGET},
            "target_true": {"str": str(answers[0])},
        },
        # Explicit empty fields make accidental probe use fail closed.
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def assert_locked(records: List[Dict[str, Any]]) -> None:
    for record in records:
        if record.get("paraphrase_prompts"):
            raise AssertionError("repair-visible ZsRE exposes a paraphrase")
        if record.get("neighborhood_prompts"):
            raise AssertionError("repair-visible ZsRE exposes a locality probe")
        if record.get("generation_prompts"):
            raise AssertionError("repair-visible ZsRE exposes a generation probe")
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, dict):
            raise AssertionError("repair-visible ZsRE lacks requested_rewrite")
        if rewrite.get("target_new", {}).get("str") != zsre.NEUTRAL_TARGET:
            raise AssertionError("repair-visible ZsRE neutral target changed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zsre-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--zsre-url", default=zsre.ZSRE_URL)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("forget-num and retain-num must be positive")

    source_path = Path(args.zsre_path).resolve()
    source_path = zsre.download_zsre(source_path, url=args.zsre_url)
    source_bytes = source_path.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("ZsRE source must be a JSON list of objects")

    forget_pairs, retain_pairs = zsre.sample_official_zsre_raw_records(
        raw,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        strict=True,
    )
    forget_visible = [direct_only_record(record, index) for index, record in forget_pairs]
    assert_locked(forget_visible)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visible_path = output_dir / "repair_visible_forget.json"
    visible_text = json.dumps(forget_visible, ensure_ascii=False, indent=2) + "\n"
    visible_path.write_text(visible_text, encoding="utf-8")

    half = len(raw) // 2
    forget_indices = [int(index) for index, _ in forget_pairs]
    retain_indices = [int(index) for index, _ in retain_pairs]
    if set(forget_indices) & set(retain_indices):
        raise AssertionError("official ZsRE forget/retain samples overlap")
    if any(index < half for index in forget_indices):
        raise AssertionError("forget sample escaped ZeroUnlearn second-half pool")
    if any(index >= half for index in retain_indices):
        raise AssertionError("retain sample escaped ZeroUnlearn first-half pool")

    manifest = {
        "schema_version": 1,
        "protocol": "zsre_zerounlearn_forget_only_locked_probes",
        "seed": int(args.seed),
        "source_dataset": str(source_path),
        "repair_visible_forget": str(visible_path),
        "source_sha256": sha256_bytes(source_bytes),
        "repair_visible_sha256": sha256_bytes(visible_text.encode("utf-8")),
        "dataset_size": len(raw),
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {
                "start": half,
                "stop_exclusive": len(raw),
                "size": len(raw) - half,
            },
        },
        "sampling": {
            "implementation": "sample_official_zsre_raw_records",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": int(args.forget_num),
            "retain_num": int(args.retain_num),
            "forget_case_ids": forget_indices,
            "retain_case_ids": retain_indices,
        },
        "data_roles": {
            "stage1_visible": ["forget.requested_rewrite"],
            "stage2_visible": ["forget.requested_rewrite"],
            "evaluation_only": [
                "forget.rephrase",
                "forget.locality",
                "1000 sampled retain records",
            ],
            "same_underlying_forget_facts_at_final_eval": True,
            "fact_level_holdout": False,
            "prompt_level_holdout": True,
            "final_evaluation_uses_original_source_file": True,
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"repair-visible ZsRE forget set: {visible_path}")
    print(f"split manifest: {manifest_path}")
    print(
        f"seed={args.seed}: train/repair={args.forget_num} forget + 0 retain; "
        f"final eval={args.forget_num} forget + {args.retain_num} retain"
    )


if __name__ == "__main__":
    main()
