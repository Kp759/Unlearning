#!/usr/bin/env python3
"""Build the locked direct-only MCF training view for target-aware SURE v8.

The builder is the only training-side program that reads the original MCF
file.  It writes a deliberately smaller view containing, for each sampled
forget record, only the direct prompt, subject, original ``target_true``, and
original ``target_new``.  Official paraphrases and every other benchmark probe
are absent rather than merely ignored.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import build_sure_minimal_split as shared


PROTOCOL = "sure_mcf_target_aware_direct_only_v8"
TRAINING_FILENAME = "training_visible_target_aware_direct.json"
PROBE_FIELDS = (
    "paraphrase_prompts",
    "neighborhood_prompts",
    "attribute_prompts",
    "generation_prompts",
)


def target_aware_direct_record(raw: Mapping[str, Any], case_id: int) -> Dict[str, Any]:
    rewrite = shared.normalize_mcf_rewrite(raw)
    target_true = str(rewrite.get("target_true", {}).get("str", "")).strip()
    target_new = str(rewrite.get("target_new", {}).get("str", "")).strip()
    if not target_true or not target_new:
        raise ValueError(f"MCF record {case_id} lacks target_true/target_new")
    return {
        "case_id": int(case_id),
        "requested_rewrite": {
            "prompt": str(rewrite["prompt"]),
            "subject": str(rewrite["subject"]),
            "target_true": {"str": target_true},
            "target_new": {"str": target_new},
        },
        "data_role": "target_aware_direct_training_only",
    }


def assert_direct_only_training_view(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise AssertionError("target-aware direct training view is empty")
    seen_case_ids = set()
    for position, record in enumerate(records):
        unexpected = sorted(set(record).intersection(PROBE_FIELDS))
        if unexpected:
            raise AssertionError(
                f"training-visible record {position} contains probe fields: {unexpected}"
            )
        if "requested_rewrite" not in record:
            raise AssertionError(
                f"training-visible record {position} lacks requested_rewrite"
            )
        allowed_record_keys = {"case_id", "requested_rewrite", "data_role"}
        extra_record_keys = sorted(set(record) - allowed_record_keys)
        if extra_record_keys:
            raise AssertionError(
                f"training-visible record {position} has unexpected fields: "
                f"{extra_record_keys}"
            )
        case_id = int(record["case_id"])
        if case_id in seen_case_ids:
            raise AssertionError(f"duplicate target-aware MCF case id: {case_id}")
        seen_case_ids.add(case_id)
        rewrite = record["requested_rewrite"]
        if not isinstance(rewrite, Mapping):
            raise AssertionError(
                f"training-visible record {position} rewrite is not a mapping"
            )
        allowed_rewrite_keys = {"prompt", "subject", "target_true", "target_new"}
        extra_rewrite_keys = sorted(set(rewrite) - allowed_rewrite_keys)
        if extra_rewrite_keys:
            raise AssertionError(
                f"training-visible record {position} rewrite has unexpected fields: "
                f"{extra_rewrite_keys}"
            )
        for field in ("prompt", "subject"):
            if not str(rewrite.get(field, "")).strip():
                raise AssertionError(
                    f"training-visible record {position} lacks {field}"
                )
        for field in ("target_true", "target_new"):
            value = rewrite.get(field)
            if not isinstance(value, Mapping) or not str(value.get("str", "")).strip():
                raise AssertionError(
                    f"training-visible record {position} lacks {field}.str"
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-eval-num", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.forget_num <= 0 or args.retain_eval_num <= 0:
        raise ValueError("forget-num and retain-eval-num must be positive")

    source = Path(args.mcf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MCF dataset not found: {source}")
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("MCF source must be a JSON list of objects")

    forget_pairs, retain_pairs = shared.sample_records(
        "mcf",
        raw,
        forget_num=args.forget_num,
        retain_eval_num=args.retain_eval_num,
        seed=args.seed,
    )
    forget_ids = [int(index) for index, _ in forget_pairs]
    retain_ids = [int(index) for index, _ in retain_pairs]
    half = len(raw) // 2
    if set(forget_ids) & set(retain_ids):
        raise AssertionError("official forget and retain samples overlap")
    if any(index < half for index in forget_ids):
        raise AssertionError("forget sample escaped the official second-half pool")
    if any(index >= half for index in retain_ids):
        raise AssertionError("retain sample escaped the official first-half pool")

    training_records = [
        target_aware_direct_record(record, case_id) for case_id, record in forget_pairs
    ]
    assert_direct_only_training_view(training_records)
    retain_records = [
        shared.prompt_only_retain_record("mcf", record, case_id)
        for case_id, record in retain_pairs
    ]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / TRAINING_FILENAME
    retain_path = output_dir / "evaluation_only_retain_prompts.json"
    training_text = json.dumps(training_records, indent=2, ensure_ascii=False) + "\n"
    retain_text = json.dumps(retain_records, indent=2, ensure_ascii=False) + "\n"
    training_path.write_text(training_text, encoding="utf-8")
    retain_path.write_text(retain_text, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "metric_schema": "mcf_target_true_sensitive_v4_fs",
        "dataset": "mcf",
        "seed": int(args.seed),
        # The hash binds post-training evaluation to the source without giving
        # the learner a source-file path it could follow to held-out probes.
        "source_sha256": shared.sha256_bytes(source_bytes),
        "training_visible_target_aware_direct": str(training_path),
        "training_visible_target_aware_direct_sha256": shared.sha256_bytes(
            training_text.encode("utf-8")
        ),
        "evaluation_only_retain_prompts": str(retain_path),
        "evaluation_only_retain_prompts_sha256": shared.sha256_bytes(
            retain_text.encode("utf-8")
        ),
        "dataset_size": len(raw),
        "sampling": {
            "implementation": "sample_official_mcf_records",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": int(args.forget_num),
            "benchmark_retain_train_num": 0,
            "retain_eval_num": int(args.retain_eval_num),
            "forget_case_ids": forget_ids,
            "retain_eval_case_ids": retain_ids,
            "retain_case_ids": retain_ids,
            "forget_retain_overlap": 0,
        },
        "learner_adapter_contract": {
            "schema_version": 1,
            "direct_prompt_field": "requested_rewrite.prompt",
            "subject_field": "requested_rewrite.subject",
            "sensitive_answer_field": "target_true",
            "reference_answer_field": "target_new",
            "direct_only": True,
            "official_paraphrases_visible_to_learner": False,
            "forbidden_probe_fields": list(PROBE_FIELDS),
        },
        "data_roles": {
            "stage1_visible": [
                "direct forget prompt",
                "original target_true",
                "original target_new",
                "external Wikipedia utility states",
            ],
            "stage2_visible": [
                "the same direct prompts and targets",
                "Stage-1 direct failure positions",
                "external Wikipedia utility states",
            ],
            "checkpoint_selection_visible": [
                "direct FS constraints",
                "direct pairwise margin",
                "Wikipedia utility guards",
                "sparse delta norm",
            ],
            "evaluation_only": [
                "official forget paraphrases and GFS",
                "locality/neighborhood prompts and Spe",
                "official retain sample",
                "PPL text",
            ],
            "GFS_checkpoint_selection": False,
            "benchmark_retain_examples_visible_to_training": 0,
            "heldout_probes_visible_during_training": False,
        },
    }
    manifest_path = output_dir / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("v8 direct-only target-aware training file:", training_path)
    print("post-training retain audit file:", retain_path)
    print("split manifest:", manifest_path)
    print(
        f"dataset=mcf seed={args.seed}: train={len(training_records)} direct "
        f"forget records + 0 benchmark retain + 0 paraphrases; "
        f"post-training retain audit={len(retain_records)}"
    )


if __name__ == "__main__":
    main()
