#!/usr/bin/env python3
"""Collect the MCF layer sweep into one table (JSON + CSV + Markdown).

    python scripts/summarize_mcf_layer_sweep.py \
        --sweep-dir outputs/mcf_layer_sweep_v1 \
        --reference outputs/mcf_linear_2x2_seed1_v24/arms/linear_global

One row per layer, read from each L??/linear_global run:
  official MCF   forget Eff/Gen (lower), forget Spe, retain Eff/Gen, PPL
  router (audit) correct route, false activation, route AUC, runtime mismatches
  geometry       boundary norm, row norm, row/boundary ratio, norm scale
  decomposition  learned-router vs oracle mean answer prob per group (if run)

The reference row is the shipped layer-19 run (gate-trained, unscaled). The
sweep's own layer-19 row is oracle-trained; the two should agree closely, and
that agreement is what licenses reading the other layers as a layer effect.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _get(tree, *keys):
    for key in keys:
        if not isinstance(tree, dict) or key not in tree:
            return None
        tree = tree[key]
    return tree


def collect(run_dir, label):
    run_dir = Path(run_dir)
    manifest = _load(run_dir / "association_manifest.json") or {}
    official = _load(run_dir / "official_mcf_eval.json")
    router = _load(run_dir / "linear_router_report.json")
    decomposition = _load(run_dir / "decomposition" / "router_decomposition.json")
    representation = manifest.get("layer_representation") or {}
    training = _load(run_dir.parent / "rows" / "training_report.json") or {}
    representation = {**representation, **(training.get("layer_representation") or {})}
    audit = _get(router, "route_outcomes_by_split", "audit") or {}
    row = {
        "label": label,
        "layer": _get(manifest, "plan", "layer"),
        "relative_depth": representation.get("relative_depth"),
        "training_route": manifest.get("training_route", "gate"),
        "norm_scale": representation.get("norm_scale", 1.0),
        "forget_Eff": _get(official, "forget", "Eff"),
        "forget_Gen": _get(official, "forget", "Gen"),
        "forget_Spe": _get(official, "forget", "Spe"),
        "retain_Eff": _get(official, "retain", "Eff"),
        "retain_Gen": _get(official, "retain", "Gen"),
        "PPL": _get(official, "forget_PPL"),
        "display_zero": _get(official, "static_branch_display_zero_check", "passed"),
        "audit_correct_route": _get(audit, "correct_route", "rate"),
        "audit_false_activation": _get(
            audit, "false_activation_on_negative_control", "rate"
        ),
        "audit_route_auc": _get(router, "audit_frontier", "linear", "route_auc"),
        "runtime_route_mismatches": _get(router, "runtime_parity", "route_mismatches"),
        "boundary_norm_median": representation.get("boundary_norm_median"),
        "row_norm_median": representation.get("row_norm_median"),
        "row_to_boundary_norm_ratio": representation.get(
            "row_to_boundary_norm_ratio_median"
        ),
        "training_stop_reason": training.get("stop_reason"),
        "status": (
            "complete" if official else "router_only" if router else
            "missing" if not manifest else "rows_only"
        ),
    }
    gap = (decomposition or {}).get("v2_to_oracle_gap") or {}
    for group, values in sorted(gap.items()):
        row[f"router_prob[{group}]"] = values.get("v2_mean_answer_prob")
        row[f"oracle_prob[{group}]"] = values.get("oracle_mean_answer_prob")
    return row


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
    parser.add_argument("--sweep-dir", required=True)
    parser.add_argument("--reference", default="")
    parser.add_argument("--out-prefix", default="")
    args = parser.parse_args(argv)

    sweep = Path(args.sweep_dir)
    rows = []
    if args.reference:
        rows.append(collect(args.reference, "shipped L19 (gate-trained)"))
    for layer_dir in sorted(sweep.glob("L[0-9][0-9]")):
        rows.append(collect(layer_dir / "linear_global", layer_dir.name))
    if not rows:
        raise SystemExit(f"No L?? directories under {sweep}")

    prefix = Path(args.out_prefix) if args.out_prefix else sweep / "layer_sweep_summary"
    columns = list(dict.fromkeys(key for row in rows for key in row))
    prefix.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    headline = [
        "label", "layer", "relative_depth", "forget_Eff", "forget_Gen",
        "forget_Spe", "retain_Eff", "retain_Gen", "PPL", "audit_correct_route",
        "audit_false_activation", "audit_route_auc", "row_to_boundary_norm_ratio",
        "status",
    ]
    lines = [
        "| " + " | ".join(headline) + " |",
        "|" + "---|" * len(headline),
    ]
    lines += ["| " + " | ".join(_fmt(row.get(k)) for k in headline) + " |" for row in rows]
    table = "\n".join(lines) + "\n"
    prefix.with_suffix(".md").write_text(table)
    print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
