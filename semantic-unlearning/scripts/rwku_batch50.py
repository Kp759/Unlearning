#!/usr/bin/env python3
"""Frozen RWKU Batch-50 split for MCF/ZsRE-style comparison.

Each experimental batch contains five RWKU people and exactly ten forget
training probes per person, for 50 forget examples in total.  The preferred
composition is 8 Level-1 + 2 Level-2 per person, but for unusually small Level-2
pools the selector adapts while always reserving at least one content-distinct
Level-1 and one Level-2 probe for held-out evaluation.

The same 50 examples are evaluated post-unlearning as *efficacy*.  All
remaining exact-content-disjoint Level-1/Level-2 probes for those people are
held out for generalization; paraphrases are derived only from held-out Level-2
rows. Level-3 and other native RWKU probes remain evaluation-only.

Batch seed selects a transparent cyclic five-person window over the registered
ten-target suite. Split seed is frozen at zero and never depends on the batch
seed. Exact duplicate content is de-duplicated for train selection and every
copy of selected content is excluded from held-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from rwku_data import (
    DEFAULT_DATA_ROOT,
    RWKU_CODE_REVISION,
    RWKU_DATASET_REVISION,
    TARGETS_BY_SEED,
    ensure_target_data,
    paraphrase_query,
    record_sha256,
    target_for_seed,
    write_json,
)


PROTOCOL_ID = "RWKU-Batch-50-v1"
SCHEMA_VERSION = "rwku_batch50_v1_manifest"
SPLIT_SEED = 0
TARGETS_PER_BATCH = 5
TRAIN_PER_TARGET = 10
PREFERRED_TRAIN_LEVEL1_PER_TARGET = 8
PREFERRED_TRAIN_LEVEL2_PER_TARGET = 2
MIN_HELDOUT_PER_LEVEL = 1
TOTAL_FORGET_TRAIN = TARGETS_PER_BATCH * TRAIN_PER_TARGET
RETAIN_EVAL_NUM = 1000
REPAIR_RETAIN_NUM = 128

EVALUATION_ONLY_FILES: Tuple[str, ...] = (
    "forget_level3.json",
    "forget_mia.json",
    "retain_mia.json",
    "neighbor_level1.json",
    "neighbor_level2.json",
    "retain_mmlu.json",
    "retain_bbh.json",
    "truthful.json",
    "triviaqa.json",
    "fluency.json",
)


def batch_target_seeds(batch_seed: int) -> Tuple[int, ...]:
    """Return the deterministic five-person cyclic batch for seed 0..9."""

    if not 0 <= int(batch_seed) < len(TARGETS_BY_SEED):
        raise ValueError(f"batch_seed must be in [0,{len(TARGETS_BY_SEED) - 1}]")
    count = len(TARGETS_BY_SEED)
    return tuple((int(batch_seed) + offset) % count for offset in range(TARGETS_PER_BATCH))


def _stable_order_key(*, target_seed: int, level: int, digest: str) -> str:
    payload = f"{PROTOCOL_ID}:{SPLIT_SEED}:{target_seed}:L{level}:{digest}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _unique_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_seed: int,
    level: int,
) -> List[Dict[str, Any]]:
    by_hash: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        digest = record_sha256(row)
        by_hash.setdefault(digest, dict(row))
    return [
        by_hash[digest]
        for digest in sorted(
            by_hash,
            key=lambda digest: _stable_order_key(
                target_seed=target_seed,
                level=level,
                digest=digest,
            ),
        )
    ]


def _training_counts(unique_l1_count: int, unique_l2_count: int) -> Tuple[int, int]:
    """Choose ten train rows while reserving >=1 unique row from each level."""

    available_l1 = unique_l1_count - MIN_HELDOUT_PER_LEVEL
    available_l2 = unique_l2_count - MIN_HELDOUT_PER_LEVEL
    if available_l1 < 1 or available_l2 < 1:
        raise ValueError("Need at least two unique rows in each RWKU level")
    if available_l1 + available_l2 < TRAIN_PER_TARGET:
        raise ValueError(
            "Not enough unique L1/L2 rows to train on ten while reserving "
            "one held-out row from each level"
        )

    train_l2 = min(PREFERRED_TRAIN_LEVEL2_PER_TARGET, available_l2)
    train_l2 = max(1, train_l2)
    train_l1 = TRAIN_PER_TARGET - train_l2
    if train_l1 > available_l1:
        shortfall = train_l1 - available_l1
        train_l1 = available_l1
        train_l2 += shortfall
    if train_l2 > available_l2 or train_l1 + train_l2 != TRAIN_PER_TARGET:
        raise ValueError("Could not allocate an exact ten-example RWKU training budget")
    return train_l1, train_l2


def _annotate(
    row: Mapping[str, Any],
    *,
    target_seed: int,
    subject: str,
    level: int,
    role: str,
) -> Dict[str, Any]:
    value = dict(row)
    value["subject"] = str(value.get("subject") or subject)
    value["level"] = str(level)
    value["rwku_target_seed"] = int(target_seed)
    value["rwku_target_subject"] = subject
    value["batch50_role"] = role
    value["source_record_sha256"] = record_sha256(row)
    return value


def _paraphrase(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value["query"] = paraphrase_query(str(row["query"]))
        value["source_record_sha256"] = str(row["source_record_sha256"])
        value["evaluation_variant"] = "deterministic_surface_paraphrase"
        value["batch50_role"] = "heldout_level2_paraphrase"
        output.append(value)
    return output


def split_target_rows(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_seed: int,
) -> Dict[str, Any]:
    """Select exactly ten train rows while preserving held-out L1 and L2."""

    target = target_for_seed(target_seed)
    required = {"forget_level1.json", "forget_level2.json", *EVALUATION_ONLY_FILES}
    missing = sorted(required - set(datasets))
    if missing:
        raise ValueError(f"{PROTOCOL_ID} missing required datasets: {missing}")

    unique_l1 = _unique_rows(
        datasets["forget_level1.json"], target_seed=target_seed, level=1
    )
    unique_l2 = _unique_rows(
        datasets["forget_level2.json"], target_seed=target_seed, level=2
    )
    train_l1_count, train_l2_count = _training_counts(
        len(unique_l1), len(unique_l2)
    )

    train_l1_raw = unique_l1[:train_l1_count]
    train_l2_raw = unique_l2[:train_l2_count]
    train_source_hashes = {
        record_sha256(row) for row in [*train_l1_raw, *train_l2_raw]
    }

    # Preserve benchmark multiplicity on the held-out side, but remove every
    # duplicate copy of content selected for training.
    heldout_l1_raw = [
        dict(row)
        for row in datasets["forget_level1.json"]
        if record_sha256(row) not in train_source_hashes
    ]
    heldout_l2_raw = [
        dict(row)
        for row in datasets["forget_level2.json"]
        if record_sha256(row) not in train_source_hashes
    ]
    if not heldout_l1_raw or not heldout_l2_raw:
        raise RuntimeError(f"{target.subject}: held-out L1/L2 must both be non-empty")

    train_l1 = [
        _annotate(
            row,
            target_seed=target_seed,
            subject=target.subject,
            level=1,
            role="forget_train_efficacy",
        )
        for row in train_l1_raw
    ]
    train_l2 = [
        _annotate(
            row,
            target_seed=target_seed,
            subject=target.subject,
            level=2,
            role="forget_train_efficacy",
        )
        for row in train_l2_raw
    ]
    heldout_l1 = [
        _annotate(
            row,
            target_seed=target_seed,
            subject=target.subject,
            level=1,
            role="heldout_level1_generalization",
        )
        for row in heldout_l1_raw
    ]
    heldout_l2 = [
        _annotate(
            row,
            target_seed=target_seed,
            subject=target.subject,
            level=2,
            role="heldout_level2_generalization",
        )
        for row in heldout_l2_raw
    ]
    paraphrase = _paraphrase(heldout_l2)

    heldout_hashes = {
        str(row["source_record_sha256"]) for row in [*heldout_l1, *heldout_l2]
    }
    if train_source_hashes & heldout_hashes:
        raise RuntimeError(f"{target.subject}: train/held-out content leakage detected")

    return {
        "target_seed": target_seed,
        "target_directory": target.directory,
        "subject": target.subject,
        "train_level1": train_l1,
        "train_level2": train_l2,
        "train": [*train_l1, *train_l2],
        "heldout_level1": heldout_l1,
        "heldout_level2": heldout_l2,
        "heldout_paraphrase": paraphrase,
        "training_source_hashes": sorted(train_source_hashes),
        "heldout_source_hashes": sorted(heldout_hashes),
        "evaluation_only": {
            filename: [dict(row) for row in datasets[filename]]
            for filename in EVALUATION_ONLY_FILES
        },
        "counts": {
            "source_level1": len(datasets["forget_level1.json"]),
            "source_level2": len(datasets["forget_level2.json"]),
            "unique_level1": len(unique_l1),
            "unique_level2": len(unique_l2),
            "train_level1": len(train_l1),
            "train_level2": len(train_l2),
            "train_total": len(train_l1) + len(train_l2),
            "heldout_level1": len(heldout_l1),
            "heldout_level2": len(heldout_l2),
            "heldout_paraphrase": len(paraphrase),
        },
    }


def build_batch_split(
    *,
    data_root: Path,
    batch_seed: int,
    allow_download: bool,
) -> Dict[str, Any]:
    target_seeds = batch_target_seeds(batch_seed)
    per_target: List[Dict[str, Any]] = []
    source_file_sha256: Dict[str, Dict[str, str]] = {}
    for target_seed in target_seeds:
        target, datasets, hashes = ensure_target_data(
            data_root,
            target_seed,
            allow_download=allow_download,
        )
        split = split_target_rows(datasets, target_seed=target_seed)
        if split["subject"] != target.subject:
            raise RuntimeError("RWKU target identity mismatch")
        per_target.append(split)
        source_file_sha256[str(target_seed)] = dict(hashes)

    forget_train = [row for split in per_target for row in split["train"]]
    heldout_l1 = [row for split in per_target for row in split["heldout_level1"]]
    heldout_l2 = [row for split in per_target for row in split["heldout_level2"]]
    heldout_paraphrase = [
        row for split in per_target for row in split["heldout_paraphrase"]
    ]
    if len(forget_train) != TOTAL_FORGET_TRAIN:
        raise RuntimeError(
            f"{PROTOCOL_ID} must contain exactly {TOTAL_FORGET_TRAIN} forget rows; "
            f"got {len(forget_train)}"
        )

    training_hashes = {
        str(row["source_record_sha256"]) for row in forget_train
    }
    heldout_hashes = {
        str(row["source_record_sha256"]) for row in [*heldout_l1, *heldout_l2]
    }
    if training_hashes & heldout_hashes:
        raise RuntimeError("Batch-level train/held-out content leakage detected")

    manifest: MutableMapping[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "benchmark": "RWKU",
        "protocol_status": "probe_assisted_cross_benchmark_method_extension",
        "rwku_code_revision": RWKU_CODE_REVISION,
        "rwku_dataset_revision": RWKU_DATASET_REVISION,
        "batch_seed": int(batch_seed),
        "split_seed": SPLIT_SEED,
        "target_seeds": list(target_seeds),
        "targets": [
            {
                "target_seed": split["target_seed"],
                "directory": split["target_directory"],
                "subject": split["subject"],
                "counts": split["counts"],
                "training_source_hashes": split["training_source_hashes"],
                "heldout_source_hashes": split["heldout_source_hashes"],
            }
            for split in per_target
        ],
        "forget_budget": {
            "people": TARGETS_PER_BATCH,
            "examples_per_person": TRAIN_PER_TARGET,
            "preferred_level1_per_person": PREFERRED_TRAIN_LEVEL1_PER_TARGET,
            "preferred_level2_per_person": PREFERRED_TRAIN_LEVEL2_PER_TARGET,
            "minimum_heldout_unique_per_level_per_person": MIN_HELDOUT_PER_LEVEL,
            "actual_per_target_level_counts": {
                str(split["target_seed"]): {
                    "level1": split["counts"]["train_level1"],
                    "level2": split["counts"]["train_level2"],
                }
                for split in per_target
            },
            "total": TOTAL_FORGET_TRAIN,
            "same_50_used_for_post_training_efficacy": True,
        },
        "generalization": {
            "heldout_level1_count": len(heldout_l1),
            "heldout_level2_count": len(heldout_l2),
            "heldout_paraphrase_count": len(heldout_paraphrase),
            "paraphrases_derived_only_from_heldout_level2": True,
            "train_content_disjoint": True,
        },
        "retain_budget": {
            "headline_retain_examples": RETAIN_EVAL_NUM,
            "source": "external deterministic MCF retain sample",
            "same_1000_available_to_unlearning_and_post_training_retain_eval": True,
            "additional_disjoint_repair_gate_examples": REPAIR_RETAIN_NUM,
        },
        "native_rwku_evaluation_only_files": list(EVALUATION_ONLY_FILES),
        "source_file_sha256": source_file_sha256,
    }

    return {
        "manifest": dict(manifest),
        "per_target": per_target,
        "forget_train": forget_train,
        "efficacy_forget": [dict(row) for row in forget_train],
        "heldout_level1": heldout_l1,
        "heldout_level2": heldout_l2,
        "heldout_paraphrase": heldout_paraphrase,
    }


def materialize_batch_split(
    *,
    data_root: Path,
    output_dir: Path,
    batch_seed: int,
    allow_download: bool,
) -> Dict[str, Any]:
    split = build_batch_split(
        data_root=data_root,
        batch_seed=batch_seed,
        allow_download=allow_download,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "forget_train_50.json", split["forget_train"])
    write_json(destination / "forget_efficacy_same_50.json", split["efficacy_forget"])
    write_json(destination / "forget_generalization_level1.json", split["heldout_level1"])
    write_json(destination / "forget_generalization_level2.json", split["heldout_level2"])
    write_json(
        destination / "forget_generalization_level2_paraphrase.json",
        split["heldout_paraphrase"],
    )
    write_json(destination / "split_manifest.json", split["manifest"])
    return split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, choices=range(10))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("data/rwku_batch50"))
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    destination = args.output_root / f"batch_seed{args.seed:02d}"
    split = materialize_batch_split(
        data_root=args.data_root,
        output_dir=destination,
        batch_seed=args.seed,
        allow_download=not args.no_download,
    )
    print(json.dumps(split["manifest"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
