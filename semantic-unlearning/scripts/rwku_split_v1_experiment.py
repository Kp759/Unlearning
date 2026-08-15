#!/usr/bin/env python3
"""Run the existing RWKU methods under the frozen RWKU-Split-v1 protocol.

This entrypoint intentionally reuses ``rwku_experiment.py`` so the model,
Setting 5e, ZeroUnlearn, repair, evaluation, and MCF-retain implementations do
not fork.  It changes only the row-partition contract:

* target seed selects the RWKU person;
* split seed is always 0;
* Level 1 and Level 2 are each split independently at 50/50;
* Level-3/MIA/neighbor/utility data remain evaluator-only;
* reported paraphrases are derived from held-out Level 2 only.

Use this entrypoint, rather than the legacy ``rwku_experiment.py`` command, for
MCF/ZsRE-style train/eval separation experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import rwku_data as DATA
import rwku_experiment as EXP
from rwku_split_v1 import (
    SPLIT_ID,
    SPLIT_SEED,
    TRAIN_FRACTION,
    materialize_target_split,
)


PROTOCOL_LABEL = "rwku_split_v1_probe_assisted"
PROTOCOL_STATUS = "nonofficial_probe_assisted_rwku_split_v1"


def _frozen_partition(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    calibration_fraction: float,
) -> Tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Ignore target seed for partitioning and enforce the frozen v1 split."""

    del seed
    if abs(float(calibration_fraction) - TRAIN_FRACTION) > 1e-12:
        raise ValueError(
            f"{SPLIT_ID} fixes --calibration-fraction at {TRAIN_FRACTION}; "
            f"got {calibration_fraction}"
        )
    return DATA.partition_records(
        records,
        seed=SPLIT_SEED,
        calibration_fraction=TRAIN_FRACTION,
    )


def _annotate_json(path: Path, *, split_manifest_path: Path) -> None:
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object at {path}")
    value["protocol_label"] = PROTOCOL_LABEL
    value["protocol_status"] = PROTOCOL_STATUS
    value["split_id"] = SPLIT_ID
    value["split_seed"] = SPLIT_SEED
    value["train_fraction"] = TRAIN_FRACTION
    value["split_interpretation"] = (
        "Level-1 and Level-2 are independently 50/50 content-hash split; "
        "held-out Level-2 drives direct and paraphrase evaluation; Level-3, "
        "MIA, neighbor, and utility sources are evaluation-only"
    )
    value["split_manifest_path"] = str(split_manifest_path.resolve())
    EXP.write_json(path, value)


def main() -> None:
    # Parse once here for the frozen-protocol preflight. EXP.main() parses the
    # same argv again after the partition hook is installed.
    args = EXP.build_parser().parse_args()
    EXP.validate_args(args)
    if args.training_source is not None:
        raise ValueError(
            f"{SPLIT_ID} is the MCF/ZsRE-style row-split track and cannot be "
            "combined with the staged entity-fact training-source protocols"
        )
    if args.stage != "all":
        raise ValueError(f"{SPLIT_ID} currently requires --stage all")
    if abs(float(args.calibration_fraction) - TRAIN_FRACTION) > 1e-12:
        raise ValueError(
            f"{SPLIT_ID} fixes --calibration-fraction at {TRAIN_FRACTION}"
        )

    target = DATA.target_for_seed(args.seed)
    output_dir = Path(args.output_root) / f"seed{args.seed:02d}_{target.directory}"
    split_dir = output_dir / "rwku_split_v1"
    materialize_target_split(
        data_root=args.data_root,
        output_root=split_dir,
        target_seed=args.seed,
        allow_download=not args.no_download,
    )
    target_split_dir = split_dir / f"seed{args.seed:02d}_{target.directory}"
    split_manifest_path = target_split_dir / "split_manifest.json"

    # The legacy implementation already calls partition_records independently
    # for forget_level1 and forget_level2. Replacing only that function freezes
    # its split seed while preserving all model/method/evaluator code paths.
    EXP.partition_records = _frozen_partition
    EXP.LEGACY_PROTOCOL_STATUS = PROTOCOL_STATUS
    EXP.main()

    _annotate_json(output_dir / "config_used.json", split_manifest_path=split_manifest_path)
    _annotate_json(output_dir / "results.json", split_manifest_path=split_manifest_path)
    print(
        f"{SPLIT_ID} complete for target seed {args.seed} ({target.subject}); "
        f"manifest={split_manifest_path}"
    )


if __name__ == "__main__":
    main()
