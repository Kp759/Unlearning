#!/usr/bin/env python3
"""Build a strict forget-only MQuAKE split with no neutral/replacement target.

Sampling exactly matches the ZeroUnlearn-style locked protocol already used in
this repository:
- first half: retain pool
- second half: forget pool
- sample forget instances first, then retain instances from one seeded RNG
- flatten requested_rewrite only after instance sampling

The training-visible artifact contains only direct forget facts and their
original sensitive target_true. It intentionally contains no target_new,
Unknown/neutral target, atomic questions, multi-hop questions, or retain data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import mquake_zero_unlearn_official_eval as mquake


PROTOCOL = "mquake_zerounlearn_forget_only_locked_no_neutral"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def direct_only_atomic_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    rewrite = record.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        raise ValueError("flattened MQuAKE record lacks requested_rewrite")
    target_true = rewrite.get("target_true")
    if not isinstance(target_true, Mapping) or not str(target_true.get("str", "")):
        raise ValueError("MQuAKE target_true is missing")

    return {
        "case_id": int(record["case_id"]),
        "mquake_case_id": int(record["mquake_case_id"]),
        "source_index": int(record["source_index"]),
        "rewrite_index": int(record["rewrite_index"]),
        "requested_rewrite": {
            "prompt": str(rewrite["prompt"]),
            "subject": str(rewrite["subject"]),
            "relation_id": rewrite.get("relation_id"),
            "target_true": {"str": str(target_true["str"])},
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
        "attribute_prompts": [],
        "generation_prompts": [],
    }


def assert_locked(records: List[Dict[str, Any]]) -> None:
    forbidden_top_level = {
        "atomic_gen_prompt",
        "multihop_questions",
        "multihop_answer",
        "multihop_answer_alias",
        "multihop_new_answer",
        "multihop_new_answer_alias",
        "questions",
        "question",
    }
    for record in records:
        leaked = forbidden_top_level.intersection(record)
        if leaked:
            raise AssertionError(f"training-visible MQuAKE leaked fields: {sorted(leaked)}")
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise AssertionError("training-visible MQuAKE lacks requested_rewrite")
        if "target_new" in rewrite:
            raise AssertionError("no-neutral protocol must not expose target_new")
        if "question" in rewrite or "mquake_target_new" in rewrite:
            raise AssertionError("training-visible MQuAKE leaked evaluation/editing fields")
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts"):
            raise AssertionError("training-visible MQuAKE exposes held-out probes")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mquake-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--mquake-url", default=mquake.MQUAKE_URL)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("forget-num and retain-num must be positive")

    source_path = mquake.download_mquake(
        Path(args.mquake_path).resolve(), url=args.mquake_url
    )
    source_bytes = source_path.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("MQuAKE source must be a JSON list of objects")
    if len(raw) != 3000:
        raise ValueError(f"expected 3000 MQuAKE-CF-3k-v2 instances, got {len(raw)}")

    forget_pairs, retain_pairs = mquake.sample_zerounlearn_instances(
        raw,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        strict=True,
    )
    flattened_forget = mquake.flatten_sampled_instances(forget_pairs)
    visible = [direct_only_atomic_record(record) for record in flattened_forget]
    assert_locked(visible)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    visible_path = output_dir / "training_visible_forget.json"
    visible_text = json.dumps(visible, ensure_ascii=False, indent=2) + "\n"
    visible_path.write_text(visible_text, encoding="utf-8")

    half = len(raw) // 2
    forget_indices = [int(index) for index, _ in forget_pairs]
    retain_indices = [int(index) for index, _ in retain_pairs]
    if set(forget_indices) & set(retain_indices):
        raise AssertionError("forget/retain samples overlap")
    if any(index < half for index in forget_indices):
        raise AssertionError("forget sample escaped ZeroUnlearn second-half pool")
    if any(index >= half for index in retain_indices):
        raise AssertionError("retain sample escaped ZeroUnlearn first-half pool")

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "source_dataset": str(source_path),
        "source_revision": mquake.MQUAKE_REV,
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_forget": str(visible_path),
        "training_visible_sha256": sha256_bytes(visible_text.encode("utf-8")),
        "dataset_size_instances": len(raw),
        "pool_split": {
            "retain_pool": {"start": 0, "stop_exclusive": half, "size": half},
            "forget_pool": {
                "start": half,
                "stop_exclusive": len(raw),
                "size": len(raw) - half,
            },
        },
        "sampling": {
            "implementation": "sample_zerounlearn_instances",
            "unit": "MQuAKE instance before requested_rewrite flattening",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num_instances": int(args.forget_num),
            "retain_num_instances": int(args.retain_num),
            "forget_source_indices": forget_indices,
            "retain_source_indices": retain_indices,
            "forget_atomic_fact_count": len(visible),
            "forget_atomic_case_ids": [int(row["case_id"]) for row in visible],
        },
        "data_roles": {
            "stage1_visible": ["forget.requested_rewrite prompt/subject/target_true"],
            "stage2_visible": ["same direct forget facts only"],
            "never_visible_to_training": [
                "target_new / neutral replacement target",
                "1000 sampled retain instances",
                "atomic natural-language questions",
                "instance-level multi-hop questions",
                "paraphrase/locality probes",
                "PPL corpus",
            ],
            "same_underlying_forget_instances_at_final_eff_eval": True,
            "final_evaluation_uses_original_source_file": True,
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"training-visible MQuAKE forget facts: {visible_path}")
    print(f"split manifest: {manifest_path}")
    print(
        f"seed={args.seed}: train/repair={args.forget_num} forget instances "
        f"({len(visible)} atomic facts), target_new/Unknown=0, retain=0; "
        f"final eval={args.forget_num} forget + {args.retain_num} retain"
    )


if __name__ == "__main__":
    main()
