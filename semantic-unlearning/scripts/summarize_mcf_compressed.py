#!/usr/bin/env python3
"""Table of MCF compressed-bank runs: metrics vs storage.

    python scripts/summarize_mcf_compressed.py --root outputs/mcf_compressed_v1/L19 \
        --reference outputs/mcf_linear_2x2_seed1_v24/arms/linear_global

One row per <router_{fact,relation}>/<route>_<value mode> run.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path):
    return json.loads(path.read_text()) if path.is_file() else None


def _get(tree, *keys):
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def collect(run_dir, label):
    manifest = _load(run_dir / "association_manifest.json") or {}
    official = _load(run_dir / "official_mcf_eval.json")
    training = _load(run_dir / "training_report.json") or {}
    values = manifest.get("value_storage") or {}
    router = manifest.get("router_storage") or {}
    return {
        "config": label,
        "router_heads": router.get("heads"),
        "value_mode": manifest.get("value_mode", "row-wise (shipped)"),
        "value_per_fact_floats": values.get("per_fact_floats"),
        "value_shared_vectors": values.get("shared_vectors"),
        "value_ratio_to_full": values.get("ratio_to_full_rows"),
        "value_floats_at_100K": _get(values, "extrapolated_total_floats", "100000"),
        "forget_Eff": _get(official, "forget", "Eff"),
        "forget_Gen": _get(official, "forget", "Gen"),
        "forget_Spe": _get(official, "forget", "Spe"),
        "retain_Eff": _get(official, "retain", "Eff"),
        "retain_Gen": _get(official, "retain", "Gen"),
        "PPL": _get(official, "forget_PPL"),
        "display_zero": _get(official, "static_branch_display_zero_check", "passed"),
        "facts_trained": _get(training, "training_coverage", "facts_trained"),
        "stop_reason": training.get("stop_reason"),
        "best_epoch": training.get("best_epoch"),
        "status": "complete" if official else "trained" if training else "missing",
    }


def _fmt(value):
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference", default="")
    args = parser.parse_args(argv)
    root = Path(args.root)
    rows = []
    if args.reference:
        rows.append(collect(Path(args.reference), "shipped L19 (row-wise, per-fact heads)"))
    for router_dir in sorted(root.glob("router_*")):
        if not router_dir.is_dir() or ".incomplete_" in router_dir.name:
            continue
        for run in sorted(p for p in router_dir.iterdir()
                          if p.is_dir() and (p / "association_manifest.json").is_file()
                          and ".incomplete_" not in p.name):
            rows.append(collect(run, f"{router_dir.name}/{run.name}"))
    if not rows:
        raise SystemExit(f"No runs under {root}")
    columns = list(rows[0])
    (root / "compressed_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    with (root / "compressed_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |" for r in rows]
    table = "\n".join(lines) + "\n"
    (root / "compressed_summary.md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
