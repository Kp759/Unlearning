#!/usr/bin/env python3
"""Fix5b wrapper: quarantine contradictory masked texts before relation learning.

This keeps the Fix5 linear recognition baseline unchanged except for the masked-text
partition contract. If one normalized TARGET_ENTITY input maps to multiple semantic
relation labels, that input is intrinsically ambiguous after masking and is excluded
from fit/calibration/validation rather than relabeled or used as contradictory
supervision. Exact masked overlap across *different* partitions is removed from later
partitions, while same-phase rows with the same semantic label are preserved so that
policy-distinct cases (forbidden vs permitted binding / negative family) remain
available to calibration and validation. All removals are reported in
partition_mask_separation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import mcf_target_relation_classifier_fix5_seed1 as base

Row = base.Row


def separate(parts: Mapping[str, Sequence[Row]]) -> tuple[dict[str, list[Row]], dict[str, Any]]:
    phases = ("fit", "calib", "validation")
    labels: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for phase in phases:
        for row in parts[phase]:
            key = row.masked.casefold()
            labels[key].add(row.relation)
            if len(examples[key]) < 8:
                examples[key].append({
                    "phase": phase,
                    "relation": row.relation,
                    "forbidden": row.forbidden,
                    "family": row.family,
                    "kind": row.kind,
                    "case_id": row.case_id,
                    "masked": row.masked,
                })

    # A masked string with multiple semantic relation labels is impossible
    # supervision for the relation classifier. Quarantine every occurrence.
    conflicts = {key: sorted(vals) for key, vals in labels.items() if len(vals) > 1}
    out: dict[str, list[Row]] = {phase: [] for phase in phases}
    conflict_dropped: Counter[str] = Counter()
    overlap_dropped: Counter[str] = Counter()
    first_partition: dict[str, str] = {}

    for phase in phases:
        for row in parts[phase]:
            key = row.masked.casefold()
            if key in conflicts:
                conflict_dropped[phase] += 1
                continue

            owner = first_partition.get(key)
            if owner is None:
                first_partition[key] = phase
            elif owner != phase:
                # Prevent exact masked-text leakage across semantic partitions.
                overlap_dropped[phase] += 1
                continue

            # IMPORTANT: same-phase duplicates with the same relation survive here.
            # dedup_sem() later collapses them for classifier learning, while
            # dedup_policy() keeps policy-distinct rows (forbidden/kind/candidate).
            out[phase].append(row)

    conflict_examples = []
    for key in sorted(conflicts):
        conflict_examples.append({
            "masked": examples[key][0]["masked"],
            "relations": conflicts[key],
            "occurrences": examples[key],
        })

    report = {
        "masked_label_conflict_unique_n": len(conflicts),
        "masked_label_conflict_rows_dropped": dict(conflict_dropped),
        "masked_label_conflicts": conflict_examples,
        "masked_overlap_dropped": dict(overlap_dropped),
        "unique_masked": {phase: len({r.masked.casefold() for r in rows}) for phase, rows in out.items()},
        "policy": (
            "conflicting masked inputs are quarantined, never relabeled; later exact "
            "cross-partition overlaps are dropped; same-phase same-label rows are "
            "preserved so policy-distinct examples survive"
        ),
    }
    return out, report


# Patch only the partition contract; model, labels, loss, margin calibration, gates,
# and recognition-only constraints remain those of the reviewed Fix5 baseline.
base.separate = separate


if __name__ == "__main__":
    base.main()
