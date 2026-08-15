#!/usr/bin/env python3
"""Aggregate completed TOFU ZeroUnlearn-style locked seeds into JSON/CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


LOCKED_METRICS = {
    "direct_forget_ap": ("forget_direct", "answer_probability_mean"),
    "direct_forget_ap_max": ("forget_direct", "answer_probability_max"),
    "direct_forget_rougeL": ("forget_direct", "rougeL_recall_mean"),
    "paraphrase_forget_ap": ("forget_paraphrase", "answer_probability_mean"),
    "paraphrase_forget_ap_max": ("forget_paraphrase", "answer_probability_max"),
    "paraphrase_forget_rougeL": ("forget_paraphrase", "rougeL_recall_mean"),
    "heldout_direct_ap": ("heldout_direct", "answer_probability_mean"),
    "heldout_direct_rougeL": ("heldout_direct", "rougeL_recall_mean"),
    "heldout_paraphrase_ap": ("heldout_paraphrase", "answer_probability_mean"),
    "heldout_paraphrase_rougeL": ("heldout_paraphrase", "rougeL_recall_mean"),
    "retain1000_ap": ("retain", "answer_probability_mean"),
    "retain1000_rougeL": ("retain", "rougeL_recall_mean"),
}

NATIVE_METRICS = (
    "forget_answer_prob",
    "retain_answer_prob",
    "forget_truth_ratio",
    "retain_truth_ratio",
    "real_authors_truth_ratio",
    "world_facts_truth_ratio",
    "real_authors_normalized_answer_prob",
    "world_facts_normalized_answer_prob",
    "forget_rougeL_recall",
    "retain_rougeL_recall",
    "ks_test_p_value",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="outputs/tofu_zerounlearn_forget_only_locked_3b"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Default: ROOT/aggregate/tofu_zerounlearn_locked_seeds",
    )
    parser.add_argument(
        "--require-all", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def finite_number(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def find_native_summary(seed_root: Path, seed: int) -> Optional[Path]:
    expected = seed_root / "native_eval" / f"tofu_locked_seed{seed}_summary.json"
    if expected.is_file():
        return expected
    candidates = sorted((seed_root / "native_eval").glob("*_summary.json"))
    return candidates[0] if len(candidates) == 1 else None


def extract_seed(root: Path, seed: int) -> Optional[Dict[str, Any]]:
    seed_root = root / f"seed{seed}"
    locked_path = seed_root / "locked_eval.json"
    if not locked_path.is_file():
        return None
    locked = load_json(locked_path)
    summary = locked.get("summary", {})
    groups = summary.get("groups", {}) if isinstance(summary, dict) else {}
    row: Dict[str, Any] = {"seed": seed, "locked_eval": str(locked_path)}
    for metric, (group, key) in LOCKED_METRICS.items():
        value = groups.get(group, {}).get(key) if isinstance(groups.get(group), dict) else None
        row[metric] = finite_number(value)
    row["retain_ap_ratio_to_full_tofu"] = finite_number(
        summary.get("retain_answer_probability_ratio_to_reference")
        if isinstance(summary, dict)
        else None
    )

    native_path = find_native_summary(seed_root, seed)
    row["native_eval"] = str(native_path) if native_path else None
    if native_path:
        native = load_json(native_path)
        for metric in NATIVE_METRICS:
            row[f"native_{metric}"] = finite_number(native.get(metric))
    else:
        for metric in NATIVE_METRICS:
            row[f"native_{metric}"] = None
    return row


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    metric_names = sorted(
        key for key in rows[0] if key not in {"seed", "locked_eval", "native_eval"}
    )
    result: Dict[str, Any] = {}
    for metric in metric_names:
        values = [
            float(row[metric]) for row in rows if finite_number(row.get(metric)) is not None
        ]
        if not values:
            continue
        result[metric] = {
            "n": len(values),
            "mean": statistics.fmean(values),
            "sd_population": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }
    return result


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    for seed in args.seeds:
        row = extract_seed(root, seed)
        if row is None:
            missing.append(seed)
        else:
            rows.append(row)

    if missing and args.require_all:
        raise FileNotFoundError(
            f"missing locked_eval.json for seeds {missing} under {root}"
        )
    if not rows:
        raise FileNotFoundError(f"no completed locked TOFU seeds under {root}")

    if args.output_prefix:
        prefix = Path(args.output_prefix).resolve()
    else:
        prefix = root / "aggregate" / "tofu_zerounlearn_locked_seeds"
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".json")
    write_csv(csv_path, rows)

    payload = {
        "schema_version": 1,
        "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
        "root": str(root),
        "requested_seeds": args.seeds,
        "completed_seeds": [row["seed"] for row in rows],
        "missing_seeds": missing,
        "per_seed": rows,
        "aggregate": aggregate(rows),
        "metric_directions": {
            "direct_forget_ap": "lower",
            "paraphrase_forget_ap": "lower",
            "heldout_direct_ap": "lower",
            "heldout_paraphrase_ap": "lower",
            "retain1000_ap": "higher",
            "retain_ap_ratio_to_full_tofu": "higher",
            "native_real_authors_normalized_answer_prob": "higher",
            "native_world_facts_normalized_answer_prob": "higher",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"completed seeds: {payload['completed_seeds']}")
    if missing:
        print(f"missing seeds: {missing}")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    for metric in (
        "direct_forget_ap",
        "paraphrase_forget_ap",
        "heldout_direct_ap",
        "heldout_paraphrase_ap",
        "retain1000_ap",
        "retain_ap_ratio_to_full_tofu",
    ):
        stats = payload["aggregate"].get(metric)
        if stats:
            print(
                f"{metric}: {stats['mean']:.8g} ± {stats['sd_population']:.8g} "
                f"(n={stats['n']})"
            )


if __name__ == "__main__":
    main()
