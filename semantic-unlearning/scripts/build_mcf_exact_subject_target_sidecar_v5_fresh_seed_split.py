#!/usr/bin/env python3
"""Materialize only the direct V5 candidate view for one fresh MCF seed.

This isolated minimizer reads the benchmark source solely to reproduce the
official seeded identity.  It serializes the 50 direct forget records and a
hash-bound manifest.  It never serializes paraphrases, neighborhoods, retain
prompt text, or PPL text, so the downstream candidate builder has no interface
through which it can consume those evaluation probes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import build_mcf_sure_target_aware_direct_split as direct
import build_sure_minimal_split as shared
import mcf_exact_subject_target_sidecar_v5_core as core


PROTOCOL = "mcf_exact_subject_target_sidecar_v5_direct_only_fresh_seed_v1"
TRAINING_FILENAME = "training_visible_target_aware_direct.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return sha256_bytes(payload)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--relation-lexicon", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed not in range(2, 11):
        parser.error("fresh V5 confirmation seed must be in [2, 10]")
    if args.forget_num != 50 or args.retain_num != 1000:
        parser.error("fresh V5 confirmation is locked to 50 forget / 1000 retain")
    return args


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


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
    lexicon_path = Path(args.relation_lexicon).resolve()
    output = Path(args.output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"fresh V5 split output already exists: {output}")
    if not source.is_file() or not lexicon_path.is_file():
        raise FileNotFoundError("MCF source or frozen relation lexicon is missing")

    lexicon = load_object(lexicon_path)
    core.validate_relation_lexicon(lexicon)
    source_bytes = source.read_bytes()
    source_sha256 = sha256_bytes(source_bytes)
    expected_source = str(lexicon.get("derivation", {}).get("source_mcf_sha256", ""))
    if source_sha256 != expected_source:
        raise RuntimeError("MCF source differs from the frozen relation lexicon source")
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise RuntimeError("MCF source must be a JSON list of records")

    records, forget_case_ids, retain_case_ids = build_split(
        raw,
        seed=int(args.seed),
        forget_num=int(args.forget_num),
        retain_num=int(args.retain_num),
    )
    if set(forget_case_ids) & set(retain_case_ids):
        raise RuntimeError("fresh V5 forget and retain identities overlap")

    output.mkdir(parents=True)
    training_path = output / TRAINING_FILENAME
    training_text = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    training_path.write_text(training_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "mcf",
        "seed": int(args.seed),
        "source_sha256": source_sha256,
        "training_visible_target_aware_direct": str(training_path),
        "training_visible_target_aware_direct_sha256": sha256_bytes(
            training_text.encode("utf-8")
        ),
        "sampling": {
            "implementation": "sample_official_mcf_records",
            "order": "forget sample first, then retain sample from one seeded RNG",
            "forget_num": int(args.forget_num),
            "retain_eval_num": int(args.retain_num),
            "forget_case_ids": forget_case_ids,
            "forget_case_ids_sha256": canonical_sha256(forget_case_ids),
            "retain_eval_case_ids_sha256": canonical_sha256(retain_case_ids),
            "forget_retain_overlap": 0,
        },
        "candidate_view": {
            "direct_forget_records": len(records),
            "official_paraphrase_prompts_serialized": 0,
            "official_neighborhood_prompts_serialized": 0,
            "official_retain_prompts_serialized": 0,
            "official_ppl_documents_serialized": 0,
            "probe_fields_absent_not_masked": True,
        },
        "splitter_source_accessed": True,
        "splitter_isolated_from_candidate_process": True,
        "candidate_process_official_evaluation_prompts_seen": 0,
    }
    manifest_path = output / "split_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "seed": int(args.seed),
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
