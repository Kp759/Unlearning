#!/usr/bin/env python3
"""Build the direct-only MCF split for bi-endpoint nullspace rewiring V2.

This is the only V2 process allowed to read the full MultiCounterFact JSON.
It serializes the 50 forget records plus three disjoint, training-safe direct
record pools used for protection.  Official retain prompt text and every
official paraphrase/neighborhood/PPL prompt remain absent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence

from mcf_sampling import sample_official_mcf_records


PROTOCOL = "mcf_sparse_biendpoint_nullspace_rewiring_v2_direct_only_split"
FILES = {
    "forget": "training_visible_forget_direct.json",
    "protection_fit": "training_visible_protection_fit_direct.json",
    "protection_development": "training_visible_protection_development_direct.json",
    "protection_certification": "training_visible_protection_certification_direct.json",
}
PROBE_FIELDS = (
    "paraphrase_prompts",
    "neighborhood_prompts",
    "attribute_prompts",
    "generation_prompts",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def direct_record(raw: Mapping[str, Any], *, role: str) -> Dict[str, Any]:
    rewrite = raw.get("requested_rewrite")
    if not isinstance(rewrite, Mapping):
        raise ValueError("MCF record lacks requested_rewrite")
    target_true = rewrite.get("target_true")
    target_new = rewrite.get("target_new")
    if not isinstance(target_true, Mapping) or not isinstance(target_new, Mapping):
        raise ValueError("MCF record lacks target mappings")
    return {
        "case_id": int(raw["case_id"]),
        "requested_rewrite": {
            "prompt": str(rewrite["prompt"]),
            "subject": str(rewrite["subject"]),
            "relation_id": str(rewrite["relation_id"]),
            "target_true": {"str": str(target_true["str"])},
            "target_new": {"str": str(target_new["str"])},
        },
        "data_role": str(role),
    }


def assert_direct_only(records: Sequence[Mapping[str, Any]], *, role: str) -> None:
    if not records:
        raise AssertionError(f"empty V2 partition: {role}")
    seen: set[int] = set()
    for position, record in enumerate(records):
        if set(record).intersection(PROBE_FIELDS):
            raise AssertionError(f"{role} record {position} exposes a probe")
        if set(record) != {"case_id", "requested_rewrite", "data_role"}:
            raise AssertionError(f"{role} record {position} has unexpected fields")
        if record["data_role"] != role:
            raise AssertionError(f"{role} record {position} has the wrong role")
        case_id = int(record["case_id"])
        if case_id in seen:
            raise AssertionError(f"duplicate case id {case_id} in {role}")
        seen.add(case_id)
        rewrite = record["requested_rewrite"]
        if not isinstance(rewrite, Mapping) or set(rewrite) != {
            "prompt",
            "subject",
            "relation_id",
            "target_true",
            "target_new",
        }:
            raise AssertionError(f"{role} record {position} has invalid rewrite")
        if "{}" not in str(rewrite["prompt"]):
            raise AssertionError(f"{role} record {position} prompt lacks subject slot")


def build_partitions(
    raw: Sequence[Dict[str, Any]],
    *,
    seed: int,
    forget_num: int,
    official_retain_num: int,
    fit_num: int,
    development_num: int,
    certification_num: int,
) -> tuple[Dict[str, list[Dict[str, Any]]], list[int]]:
    forget_raw, official_retain = sample_official_mcf_records(
        raw, forget_num, official_retain_num, seed, strict=True
    )
    official_ids = {int(record["case_id"]) for record in official_retain}
    first_half = list(raw[: len(raw) // 2])
    available = [
        record for record in first_half if int(record["case_id"]) not in official_ids
    ]
    required = fit_num + development_num + certification_num
    if len(available) < required:
        raise ValueError(
            f"only {len(available)} non-official protection records, need {required}"
        )
    rng = random.Random(int(seed) + 2207)
    selected = rng.sample(available, required)
    cursor = 0
    raw_partitions: Dict[str, Sequence[Dict[str, Any]]] = {
        "forget": forget_raw,
        "protection_fit": selected[cursor : cursor + fit_num],
        "protection_development": selected[
            cursor + fit_num : cursor + fit_num + development_num
        ],
        "protection_certification": selected[
            cursor + fit_num + development_num : required
        ],
    }
    partitions: Dict[str, list[Dict[str, Any]]] = {}
    all_visible_ids: set[int] = set()
    for role, records in raw_partitions.items():
        serialized = [direct_record(record, role=role) for record in records]
        assert_direct_only(serialized, role=role)
        ids = {int(record["case_id"]) for record in serialized}
        if all_visible_ids.intersection(ids):
            raise AssertionError(f"V2 partitions overlap at role {role}")
        if official_ids.intersection(ids):
            raise AssertionError(f"official retain record leaked into {role}")
        all_visible_ids.update(ids)
        partitions[role] = serialized
    return partitions, sorted(official_ids)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--official-retain-num", type=int, default=1000)
    parser.add_argument("--protection-fit-num", type=int, default=2000)
    parser.add_argument("--protection-development-num", type=int, default=500)
    parser.add_argument("--protection-certification-num", type=int, default=1000)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        parser.error("V2 is locked to consumed seed 1 / 50 forget facts")
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
    partitions, official_ids = build_partitions(
        raw,
        seed=int(args.seed),
        forget_num=int(args.forget_num),
        official_retain_num=int(args.official_retain_num),
        fit_num=int(args.protection_fit_num),
        development_num=int(args.protection_development_num),
        certification_num=int(args.protection_certification_num),
    )
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    file_hashes: Dict[str, str] = {}
    role_case_ids: Dict[str, list[int]] = {}
    for role, records in partitions.items():
        text = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
        (output / FILES[role]).write_text(text, encoding="utf-8")
        file_hashes[role] = sha256_bytes(text.encode("utf-8"))
        role_case_ids[role] = [int(record["case_id"]) for record in records]
    manifest = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": "mcf",
        "seed": int(args.seed),
        "source_sha256": sha256_bytes(source_bytes),
        "files": dict(FILES),
        "file_sha256": file_hashes,
        "case_ids": role_case_ids,
        "official_retain_case_ids_only": official_ids,
        "partition_counts": {role: len(rows) for role, rows in partitions.items()},
        "serialized_prompt_counts": {
            "direct_forget": len(partitions["forget"]),
            "direct_protection": sum(
                len(partitions[role]) for role in partitions if role != "forget"
            ),
            "official_paraphrase": 0,
            "official_neighborhood": 0,
            "official_retain": 0,
            "official_ppl": 0,
            "alias": 0,
            "adversarial": 0,
        },
        "partitions_pairwise_disjoint": True,
        "official_retain_text_serialized": False,
        "learner_source_path_available": False,
        "official_evaluation_permitted": False,
    }
    (output / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "seed": int(args.seed),
                "partition_counts": manifest["partition_counts"],
                "official_retain_ids_reserved": len(official_ids),
                "official_prompt_text_serialized": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
