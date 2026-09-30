#!/usr/bin/env python3
"""Aggregate a multi-seed layer sweep: mean ± std over seeds, per layer.

    python scripts/summarize_layer_sweep_seeds.py --dataset mcf
    python scripts/summarize_layer_sweep_seeds.py --dataset zsre --modes regular genie
    python scripts/summarize_layer_sweep_seeds.py --dataset rwku --modes regular subject

Reads outputs/<dataset>_multiseed_<mode>_v1/seed*/L??/linear_global with the
single-run collector (summarize_mcf_layer_sweep.collect) and writes, per mode,
<sweep>/multiseed_summary.{md,csv,json}. Only completed runs (official eval
present) enter the statistics; `n` is the number of seeds per cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from summarize_mcf_layer_sweep import collect

METRICS = {
    "mcf": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL",
            "audit_correct_route", "audit_false_activation", "row_to_boundary_norm_ratio"],
    "zsre": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL",
             "forget_paraphrase_route_active", "forget_neighborhood_route_active",
             "audit_correct_route", "audit_false_activation", "row_to_boundary_norm_ratio"],
    "mquake": ["forget_Eff", "forget_AtomicGen", "retain_Eff", "retain_AtomicGen", "PPL",
               "forget_rewrite_route_correct", "forget_atomicgen_route_correct",
               "retain_atomicgen_route_active", "audit_correct_route",
               "audit_false_activation", "row_to_boundary_norm_ratio"],
    # RWKU: recovery %, lower = better forgetting except `neighbor` (locality, higher = better)
    "rwku": ["forget_Eff", "forget_GenL1", "forget_GenL2", "forget_GenPara", "forget_L3",
             "neighbor", "PPL", "same50_route_correct", "heldout_route_active",
             "neighbor_route_active", "audit_correct_route", "audit_false_activation"],
}
# RWKU's base model (all-zero rows) per seed, printed as the first row of each table.
BASE_SWEEP = {"rwku": "rwku_multiseed_base_v1"}


def _stats(values):
    values = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if not values:
        return None, None, 0
    mean = sum(values) / len(values)
    std = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5 if len(values) > 1 else 0.0
    return mean, std, len(values)


def _fmt(mean, std):
    if mean is None:
        return "–"
    return f"{mean:.4g} ± {std:.2g}"


def summarize(sweep, metrics):
    by_layer = defaultdict(list)
    seeds_seen = set()
    for seed_dir in sorted(sweep.glob("seed*")):
        for layer_dir in sorted(seed_dir.glob("L[0-9][0-9]")):
            row = collect(layer_dir / "linear_global", layer_dir.name)
            if row.get("status") != "complete":
                continue
            seeds_seen.add(seed_dir.name)
            row["seed"] = seed_dir.name
            by_layer[layer_dir.name].append(row)
    table = []
    for layer in sorted(by_layer):
        rows = by_layer[layer]
        entry = {"layer": layer, "n": len(rows), "seeds": [r["seed"] for r in rows],
                 "stop_reasons": sorted({str(r.get("training_stop_reason")) for r in rows})}
        for metric in metrics:
            mean, std, n = _stats(r.get(metric) for r in rows)
            entry[metric] = {"mean": mean, "std": std, "n": n}
        table.append(entry)
    return table, sorted(seeds_seen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(METRICS), required=True)
    parser.add_argument("--modes", nargs="+", default=["regular", "genie"])
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--tag-version", default="v1")
    args = parser.parse_args(argv)
    metrics = METRICS[args.dataset]
    for mode in args.modes:
        sweep = Path(args.root) / f"{args.dataset}_multiseed_{mode}_{args.tag_version}"
        if not sweep.is_dir():
            print(f"(skip) {sweep} not found")
            continue
        table, seeds = summarize(sweep, metrics)
        base_dir = Path(args.root) / BASE_SWEEP.get(args.dataset, "__none__")
        if base_dir.is_dir():
            base_rows = [collect(d, d.name) for d in sorted(base_dir.glob("seed*"))
                         if (d / "official_rwku_eval.json").is_file() and d.name in seeds]
            if base_rows:
                entry = {"layer": "base", "n": len(base_rows), "seeds": [r["label"] for r in base_rows],
                         "stop_reasons": ["no unlearning"]}
                for metric in metrics:
                    # zero rows: routing columns describe no edit, so leave them blank
                    skip = "route" in metric or metric.startswith("audit_")
                    mean, std, n = (None, None, 0) if skip else _stats(r.get(metric) for r in base_rows)
                    entry[metric] = {"mean": mean, "std": std, "n": n}
                table.insert(0, entry)
        (sweep / "multiseed_summary.json").write_text(json.dumps(table, indent=2) + "\n")
        with (sweep / "multiseed_summary.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["layer", "n"] + [f"{m}_{s}" for m in metrics for s in ("mean", "std")])
            for e in table:
                writer.writerow([e["layer"], e["n"]] + [e[m][s] for m in metrics for s in ("mean", "std")])
        header = ["layer", "n"] + metrics + ["stop reasons"]
        lines = [f"### {args.dataset.upper()} {mode} (seeds: {', '.join(seeds) or 'none'})", "",
                 "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
        for e in table:
            lines.append("| " + " | ".join(
                [e["layer"], str(e["n"])]
                + [_fmt(e[m]["mean"], e[m]["std"]) for m in metrics]
                + [", ".join(e["stop_reasons"])]
            ) + " |")
        text = "\n".join(lines) + "\n"
        (sweep / "multiseed_summary.md").write_text(text)
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
