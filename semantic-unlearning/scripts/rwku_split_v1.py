#!/usr/bin/env python3
"""Materialize the frozen RWKU-Split-v1 train/evaluation partition.

RWKU is target-centric, so target identity and split randomness must remain
independent.  ``target_seed`` selects the published RWKU target.  The split
itself is frozen at split seed 0 and 50/50, and Level 1 and Level 2 are split
separately so every target keeps both training and held-out supervision.

Protocol:
  * train: 50% of forget_level1 + 50% of forget_level2;
  * held-out cloze eval: remaining forget_level1;
  * held-out direct eval: remaining forget_level2;
  * paraphrase eval: deterministic paraphrases of held-out Level 2 only;
  * forget_level3, MIA, neighbor, and utility files are evaluation-only;
  * positive.json is not part of RWKU-Split-v1 optimization/evaluation.

Exact duplicate records are kept on the same side by the existing
``partition_records`` content-hash grouping.  The manifest records every source
and split hash so downstream training/evaluation can fail closed on leakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

from rwku_data import (
    DEFAULT_DATA_ROOT,
    RWKU_CODE_REVISION,
    RWKU_DATASET_REVISION,
    TARGETS_BY_SEED,
    build_split_manifest,
    ensure_target_data,
    paraphrase_query,
    partition_records,
    record_sha256,
    target_for_seed,
    write_json,
)


SPLIT_ID = "RWKU-Split-v1"
SCHEMA_VERSION = "rwku_split_v1_manifest"
SPLIT_SEED = 0
TRAIN_FRACTION = 0.5

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

MATERIALIZED_FILES: Mapping[str, str] = {
    "forget_train": "forget_train.json",
    "forget_eval_level1": "forget_eval_level1.json",
    "forget_eval_level2": "forget_eval_level2.json",
    "forget_eval_paraphrase": "forget_eval_paraphrase.json",
    "manifest": "split_manifest.json",
}


def _hashes(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    return [record_sha256(row) for row in rows]


def _paraphrase_held_out_level2(
    held_out_level2: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Create evaluation paraphrases only from held-out Level-2 records."""

    output: List[Dict[str, Any]] = []
    for row in held_out_level2:
        adapted = dict(row)
        adapted["query"] = paraphrase_query(str(row["query"]))
        adapted["source_record_sha256"] = record_sha256(row)
        adapted["evaluation_variant"] = "deterministic_surface_paraphrase"
        output.append(adapted)
    return output


def split_target_datasets(
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target_seed: int,
    source_file_sha256: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Build one target's frozen RWKU-Split-v1 partition in memory.

    ``target_seed`` identifies the RWKU person only.  It is deliberately not
    passed to ``partition_records``.  All targets use the same split seed 0.
    """

    target = target_for_seed(target_seed)
    required = {"forget_level1.json", "forget_level2.json", *EVALUATION_ONLY_FILES}
    missing = sorted(required - set(datasets))
    if missing:
        raise ValueError(f"RWKU-Split-v1 is missing required datasets: {missing}")

    train_level1, eval_level1 = partition_records(
        datasets["forget_level1.json"],
        seed=SPLIT_SEED,
        calibration_fraction=TRAIN_FRACTION,
    )
    train_level2, eval_level2 = partition_records(
        datasets["forget_level2.json"],
        seed=SPLIT_SEED,
        calibration_fraction=TRAIN_FRACTION,
    )

    forget_train = [*train_level1, *train_level2]
    eval_paraphrase = _paraphrase_held_out_level2(eval_level2)

    train_hashes = set(_hashes(forget_train))
    eval_level1_hashes = set(_hashes(eval_level1))
    eval_level2_hashes = set(_hashes(eval_level2))
    held_out_hashes = eval_level1_hashes | eval_level2_hashes
    overlap = train_hashes & held_out_hashes
    if overlap:
        raise RuntimeError(
            "RWKU-Split-v1 train and held-out evaluation records overlap: "
            f"{sorted(overlap)}"
        )

    paraphrase_sources = {
        str(row["source_record_sha256"]) for row in eval_paraphrase
    }
    if paraphrase_sources != eval_level2_hashes:
        raise RuntimeError(
            "RWKU-Split-v1 paraphrases must be generated exactly from held-out "
            "Level-2 records"
        )

    level1_manifest = build_split_manifest(train_level1, eval_level1)
    level2_manifest = build_split_manifest(train_level2, eval_level2)

    evaluation_only = {
        filename: {
            "role": "evaluation_only",
            "count": len(datasets[filename]),
            "file_sha256": (
                str(source_file_sha256[filename])
                if source_file_sha256 and filename in source_file_sha256
                else None
            ),
            "gradient_allowed": False,
            "repair_selection_allowed": False,
            "checkpoint_selection_allowed": False,
        }
        for filename in EVALUATION_ONLY_FILES
    }

    manifest: MutableMapping[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "split_id": SPLIT_ID,
        "benchmark": "RWKU",
        "rwku_code_revision": RWKU_CODE_REVISION,
        "rwku_dataset_revision": RWKU_DATASET_REVISION,
        "target_seed": target_seed,
        "target_directory": target.directory,
        "subject": target.subject,
        "split_seed": SPLIT_SEED,
        "train_fraction": TRAIN_FRACTION,
        "split_policy": (
            "forget_level1 and forget_level2 are independently content-hash "
            "partitioned 50/50; duplicate-content groups are indivisible"
        ),
        "training": {
            "sources": ["forget_level1.json", "forget_level2.json"],
            "level1_count": len(train_level1),
            "level2_count": len(train_level2),
            "count": len(forget_train),
            "record_sha256": _hashes(forget_train),
            "gradient_allowed": True,
        },
        "evaluation": {
            "held_out_level1": {
                "role": "memorization_cloze",
                "count": len(eval_level1),
                "record_sha256": _hashes(eval_level1),
            },
            "held_out_level2": {
                "role": "direct_efficacy",
                "count": len(eval_level2),
                "record_sha256": _hashes(eval_level2),
            },
            "held_out_level2_paraphrase": {
                "role": "generalization",
                "count": len(eval_paraphrase),
                "source_record_sha256": [
                    str(row["source_record_sha256"]) for row in eval_paraphrase
                ],
                "derived_from_held_out_level2_only": True,
            },
            "evaluation_only_sources": evaluation_only,
        },
        "level_manifests": {
            "level1": level1_manifest,
            "level2": level2_manifest,
        },
        "disjointness": {
            "train_vs_held_out_record_hashes": True,
            "paraphrase_sources_equal_held_out_level2": True,
        },
        "excluded": {
            "positive.json": (
                "not used by RWKU-Split-v1; retain optimization should come "
                "from the separately declared external retain corpus"
            )
        },
        "source_file_sha256": dict(source_file_sha256 or {}),
    }

    return {
        "target": target,
        "forget_train": forget_train,
        "forget_train_level1": train_level1,
        "forget_train_level2": train_level2,
        "forget_eval_level1": eval_level1,
        "forget_eval_level2": eval_level2,
        "forget_eval_paraphrase": eval_paraphrase,
        "manifest": dict(manifest),
    }


def materialize_target_split(
    *,
    data_root: Path,
    output_root: Path,
    target_seed: int,
    allow_download: bool,
) -> Dict[str, Any]:
    target, datasets, file_hashes = ensure_target_data(
        data_root,
        target_seed,
        allow_download=allow_download,
    )
    split = split_target_datasets(
        datasets,
        target_seed=target_seed,
        source_file_sha256=file_hashes,
    )
    destination = output_root / f"seed{target_seed:02d}_{target.directory}"
    destination.mkdir(parents=True, exist_ok=True)

    write_json(destination / MATERIALIZED_FILES["forget_train"], split["forget_train"])
    write_json(
        destination / MATERIALIZED_FILES["forget_eval_level1"],
        split["forget_eval_level1"],
    )
    write_json(
        destination / MATERIALIZED_FILES["forget_eval_level2"],
        split["forget_eval_level2"],
    )
    write_json(
        destination / MATERIALIZED_FILES["forget_eval_paraphrase"],
        split["forget_eval_paraphrase"],
    )

    manifest = dict(split["manifest"])
    manifest["materialized_files"] = {
        key: value for key, value in MATERIALIZED_FILES.items() if key != "manifest"
    }
    write_json(destination / MATERIALIZED_FILES["manifest"], manifest)
    return {
        "target_seed": target_seed,
        "subject": target.subject,
        "output_dir": str(destination),
        "train_count": len(split["forget_train"]),
        "eval_level1_count": len(split["forget_eval_level1"]),
        "eval_level2_count": len(split["forget_eval_level2"]),
        "eval_paraphrase_count": len(split["forget_eval_paraphrase"]),
    }


def _parse_targets(value: str) -> List[int]:
    if value.strip().lower() == "all":
        return list(range(len(TARGETS_BY_SEED)))
    targets = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not targets:
        raise argparse.ArgumentTypeError("--targets must select at least one target")
    if len(set(targets)) != len(targets):
        raise argparse.ArgumentTypeError("--targets contains duplicates")
    for target_seed in targets:
        target_for_seed(target_seed)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "rwku_split_v1",
    )
    parser.add_argument(
        "--targets",
        type=_parse_targets,
        default=list(range(len(TARGETS_BY_SEED))),
        help="Comma-separated target seeds 0-9, or 'all' (default: all).",
    )
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    summaries = [
        materialize_target_split(
            data_root=args.data_root,
            output_root=args.output_root,
            target_seed=target_seed,
            allow_download=not args.no_download,
        )
        for target_seed in args.targets
    ]
    aggregate = {
        "schema_version": "rwku_split_v1_aggregate",
        "split_id": SPLIT_ID,
        "split_seed": SPLIT_SEED,
        "train_fraction": TRAIN_FRACTION,
        "targets": summaries,
    }
    write_json(args.output_root / "manifest.json", aggregate)
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
