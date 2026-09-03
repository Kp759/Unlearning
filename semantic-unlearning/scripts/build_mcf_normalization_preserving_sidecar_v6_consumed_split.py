#!/usr/bin/env python3
"""Materialize direct-only V6 candidate data for consumed MCF seeds 1 or 2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_mcf_sure_target_aware_direct_split as direct
import build_sure_minimal_split as shared
import mcf_normalization_preserving_sidecar_v6_core as core


PROTOCOL = "mcf_normalization_preserving_sidecar_v6_consumed_direct_only_v1"
TRAINING_FILENAME = "training_visible_target_aware_direct.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--frame-lexicon", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed not in (1, 2):
        parser.error("V6 development split is restricted to consumed seeds 1 and 2")
    if args.forget_num != 50 or args.retain_num != 1000:
        parser.error("V6 development is locked to 50 forget / 1000 retain")
    return args


def build_split(
    raw: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    forget_num: int,
    retain_num: int,
) -> tuple[list[dict[str, Any]], list[int], list[int]]:
    forget_pairs, retain_pairs = shared.sample_records(
        "mcf",
        raw,
        forget_num=int(forget_num),
        retain_eval_num=int(retain_num),
        seed=int(seed),
    )
    records = [
        direct.target_aware_direct_record(record, int(case_id))
        for case_id, record in forget_pairs
    ]
    direct.assert_direct_only_training_view(records)
    return (
        records,
        [int(case_id) for case_id, _record in forget_pairs],
        [int(case_id) for case_id, _record in retain_pairs],
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.mcf_path).resolve()
    lexicon_path = Path(args.frame_lexicon).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(output)
    if not source.is_file() or not lexicon_path.is_file():
        raise FileNotFoundError("MCF source or V6 frame lexicon is missing")
    lexicon = json.loads(lexicon_path.read_text(encoding="utf-8"))
    core.validate_frame_lexicon(lexicon)
    source_bytes = source.read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    if source_sha256 != lexicon["derivation"]["source_mcf_sha256"]:
        raise RuntimeError("MCF source differs from V6 frame-lexicon binding")
    raw = json.loads(source_bytes)
    records, forget_case_ids, retain_case_ids = build_split(
        raw,
        seed=int(args.seed),
        forget_num=int(args.forget_num),
        retain_num=int(args.retain_num),
    )
    if set(forget_case_ids) & set(retain_case_ids):
        raise RuntimeError("V6 forget and retain identities overlap")

    output.mkdir(parents=True)
    training_path = output / TRAINING_FILENAME
    training_bytes = (
        json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    training_path.write_bytes(training_bytes)
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "mcf",
        "seed": int(args.seed),
        "evaluation_status": "consumed_development_not_blind_not_official",
        "source_sha256": source_sha256,
        "frame_lexicon_sha256": str(lexicon["lexicon_sha256"]),
        "training_visible_target_aware_direct": str(training_path),
        "training_visible_target_aware_direct_sha256": sha256_bytes(training_bytes),
        "sampling": {
            "implementation": "sample_official_mcf_records",
            "forget_num": int(args.forget_num),
            "retain_eval_num": int(args.retain_num),
            "forget_case_ids": forget_case_ids,
            "forget_case_ids_sha256": canonical_sha256(forget_case_ids),
            "retain_eval_case_ids_sha256": canonical_sha256(retain_case_ids),
            "forget_retain_overlap": 0,
        },
        "candidate_view": {
            "direct_forget_records": len(records),
            "paraphrase_prompts_serialized": 0,
            "neighborhood_prompts_serialized": 0,
            "retain_prompts_serialized": 0,
            "ppl_documents_serialized": 0,
            "probe_fields_absent_not_masked": True,
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
                "seed": int(args.seed),
                "role": "consumed_development",
                "direct_records": len(records),
                "training_visible": str(training_path),
                "split_manifest": str(manifest_path),
                "heldout_prompt_text_serialized": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
