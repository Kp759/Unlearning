#!/usr/bin/env python3
"""Build the direct-only MCF training view for internal rewiring V1.

This is the only V1 program allowed to read the full MultiCounterFact file.
It serializes no paraphrase, neighborhood, retain, PPL, alias, or adversarial
prompt text, and the learner receives no path back to the source dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from mcf_sampling import sample_official_mcf_records


PROTOCOL = "mcf_internal_fact_conditional_embedding_rewiring_v1_direct_only_split"
TRAINING_FILENAME = "training_visible_internal_rewiring_direct.json"
PROBE_FIELDS = (
    "paraphrase_prompts",
    "neighborhood_prompts",
    "attribute_prompts",
    "generation_prompts",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def direct_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    rewrite = raw.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        raise ValueError("MCF record lacks requested_rewrite")
    target_true = rewrite.get("target_true")
    target_new = rewrite.get("target_new")
    if not isinstance(target_true, Mapping) or not isinstance(target_new, Mapping):
        raise ValueError("MCF record lacks target mappings")
    value = {
        "case_id": int(raw["case_id"]),
        "requested_rewrite": {
            "prompt": str(rewrite["prompt"]),
            "subject": str(rewrite["subject"]),
            "relation_id": str(rewrite["relation_id"]),
            "target_true": {"str": str(target_true["str"])},
            "target_new": {"str": str(target_new["str"])},
        },
        "data_role": "internal_rewiring_direct_training_only",
    }
    return value


def assert_direct_only(records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        raise AssertionError("V1 direct-only training view is empty")
    case_ids: set[int] = set()
    for position, record in enumerate(records):
        if set(record).intersection(PROBE_FIELDS):
            raise AssertionError(f"record {position} exposes an evaluation probe")
        if set(record) != {"case_id", "requested_rewrite", "data_role"}:
            raise AssertionError(f"record {position} has unexpected top-level fields")
        case_id = int(record["case_id"])
        if case_id in case_ids:
            raise AssertionError(f"duplicate case id {case_id}")
        case_ids.add(case_id)
        rewrite = record["requested_rewrite"]
        if not isinstance(rewrite, Mapping) or set(rewrite) != {
            "prompt",
            "subject",
            "relation_id",
            "target_true",
            "target_new",
        }:
            raise AssertionError(f"record {position} has an invalid rewrite contract")
        for field in ("prompt", "subject", "relation_id"):
            if not str(rewrite[field]).strip():
                raise AssertionError(f"record {position} lacks {field}")
        for field in ("target_true", "target_new"):
            target = rewrite[field]
            if not isinstance(target, Mapping) or set(target) != {"str"}:
                raise AssertionError(f"record {position} has an invalid {field}")
            if not str(target["str"]).strip():
                raise AssertionError(f"record {position} has an empty {field}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        parser.error(
            "V1 preflight implementation is locked to consumed seed 1 / 50 facts"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    source = Path(args.mcf_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    source_bytes = source.read_bytes()
    raw = json.loads(source_bytes)
    if not isinstance(raw, list) or not all(isinstance(row, dict) for row in raw):
        raise ValueError("MCF source must be a JSON list")
    forget, _unused = sample_official_mcf_records(
        raw, int(args.forget_num), 0, int(args.seed), strict=True
    )
    records = [direct_record(record) for record in forget]
    assert_direct_only(records)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    training_path = output / TRAINING_FILENAME
    training_text = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    training_path.write_text(training_text, encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "mcf",
        "seed": int(args.seed),
        "forget_num": len(records),
        "source_sha256": sha256_bytes(source_bytes),
        "training_visible_file": TRAINING_FILENAME,
        "training_visible_sha256": sha256_bytes(training_text.encode("utf-8")),
        "forget_case_ids": [int(record["case_id"]) for record in records],
        "serialized_prompt_counts": {
            "direct_forget": len(records),
            "official_paraphrase": 0,
            "official_neighborhood": 0,
            "official_retain": 0,
            "official_ppl": 0,
            "alias": 0,
            "adversarial": 0,
        },
        "learner_source_path_available": False,
        "official_evaluation_permitted": False,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "seed": int(args.seed),
                "direct_records": len(records),
                "training_visible": str(training_path),
                "heldout_prompt_text_serialized": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
