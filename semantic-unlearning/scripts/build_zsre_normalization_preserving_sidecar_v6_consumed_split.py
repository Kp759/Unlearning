#!/usr/bin/env python3
"""Materialize the direct-only candidate view for consumed ZsRE seed 1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_zsre_zerounlearn_locked_split as locked
import zsre_normalization_preserving_sidecar_v6_core as core
import zsre_zero_unlearn_official_eval as official


PROTOCOL = "zsre_normalization_preserving_sidecar_v6_consumed_direct_only_v1"
TRAINING_FILENAME = "training_visible_direct_only.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zsre-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--zsre-url", default=official.ZSRE_URL)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1:
        parser.error("ZsRE V6 consumed development is restricted to seed 1")
    if args.forget_num != 50 or args.retain_num != 1000:
        parser.error("ZsRE V6 is locked to 50 forget / 1000 retain records")
    return args


def build_split(
    raw: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    forget_num: int,
    retain_num: int,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    forget, retain = official.sample_official_zsre_raw_records(
        list(raw),
        forget_num=int(forget_num),
        retain_num=int(retain_num),
        seed=int(seed),
        strict=True,
    )
    visible = [locked.direct_only_record(record, case_id) for case_id, record in forget]
    locked.assert_locked(visible)
    return (
        visible,
        [int(case_id) for case_id, _record in forget],
        [int(case_id) for case_id, _record in retain],
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.zsre_path).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    source = official.download_zsre(source, url=args.zsre_url)
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise RuntimeError("ZsRE source must be a JSON list of records")
    visible, forget_case_ids, retain_case_ids = build_split(
        raw,
        seed=int(args.seed),
        forget_num=int(args.forget_num),
        retain_num=int(args.retain_num),
    )
    if set(forget_case_ids) & set(retain_case_ids):
        raise RuntimeError("ZsRE V6 forget and retain identities overlap")
    subjects = [
        str(record["requested_rewrite"]["subject"]) for record in visible
    ]
    if len(set(subjects)) != len(subjects):
        raise RuntimeError("ZsRE V6 seed 1 contains ambiguous duplicate subjects")

    output.mkdir(parents=True)
    training_path = output / TRAINING_FILENAME
    training_bytes = (
        json.dumps(visible, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    training_path.write_bytes(training_bytes)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "architecture_protocol": core.PROTOCOL,
        "dataset": "zsre",
        "seed": 1,
        "evaluation_status": "consumed_development_not_blind_not_official",
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_direct_only": str(training_path),
        "training_visible_direct_only_sha256": sha256_bytes(training_bytes),
        "sampling": {
            "implementation": "sample_official_zsre_raw_records",
            "forget_num": 50,
            "retain_num": 1000,
            "forget_case_ids": forget_case_ids,
            "forget_case_ids_sha256": canonical_sha256(forget_case_ids),
            "retain_case_ids_sha256": canonical_sha256(retain_case_ids),
            "forget_retain_overlap": 0,
        },
        "candidate_view": {
            "direct_forget_records": len(visible),
            "paraphrase_prompts_serialized": 0,
            "neighborhood_prompts_serialized": 0,
            "retain_prompts_serialized": 0,
            "ppl_documents_serialized": 0,
            "probe_fields_absent_not_masked": True,
        },
        "collision_policy": {
            "audit_identity": "tokenized_full_scorer_input_and_target_token_id",
            "forget_positive_precedes_exact_preservation_label": True,
            "official_metrics_remain_unmodified": True,
        },
        "splitter_isolated_from_candidate_process": True,
        "candidate_process_evaluation_prompts_seen": 0,
    }
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seed": 1,
                "role": "consumed_development",
                "direct_records": len(visible),
                "training_visible": str(training_path),
                "split_manifest": str(manifest_path),
                "heldout_prompt_text_serialized": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
