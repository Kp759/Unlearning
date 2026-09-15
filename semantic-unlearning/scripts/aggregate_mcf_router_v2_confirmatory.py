#!/usr/bin/env python3
"""Aggregate MCF Router V2 confirmatory seeds 2--10.

Seed 1 is development-only and is not included in confirmatory mean/std.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


METRICS = (
    ("forget_Eff", ("forget", "Eff")),
    ("forget_Gen", ("forget", "Gen")),
    ("forget_Spe", ("forget", "Spe")),
    ("forget_ReleasedAccuracy_Eff", ("forget", "ReleasedAccuracy_Eff")),
    ("forget_ReleasedAccuracy_Gen", ("forget", "ReleasedAccuracy_Gen")),
    ("retain_Eff", ("retain", "Eff")),
    ("retain_Gen", ("retain", "Gen")),
    ("retain_Spe", ("retain", "Spe")),
    ("PPL", ("forget_PPL",)),
)


def get(payload, path):
    value = payload
    for key in path:
        value = value[key]
    return float(value)


def stats(values):
    return {
        "n": len(values),
        "mean": mean(values),
        "std_sample": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def read_seed(root, seed):
    run = root / f"mcf_fact_assoc_router_v2_seed{seed}"
    edited_path = run / "official_mcf_eval.json"
    base_path = run / "official_mcf_base_eval.json"
    train_path = run / "training_report.json"
    for path in (edited_path, base_path, train_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing confirmatory seed {seed} artifact: {path}")
    edited = json.loads(edited_path.read_text())
    base = json.loads(base_path.read_text())
    train = json.loads(train_path.read_text())
    if int(edited.get("seed", -1)) != seed or int(base.get("seed", -1)) != seed:
        raise RuntimeError(f"Seed metadata mismatch in {run}")
    return base, edited, train


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outputs-root", default="outputs")
    p.add_argument(
        "--out-json",
        default="outputs/mcf_fact_assoc_router_v2_confirmatory_2_10_summary.json",
    )
    p.add_argument(
        "--out-csv",
        default="outputs/mcf_fact_assoc_router_v2_confirmatory_2_10_per_seed.csv",
    )
    args = p.parse_args(argv)

    root = Path(args.outputs_root).resolve()
    rows = []
    base_values = {name: [] for name, _ in METRICS}
    edited_values = {name: [] for name, _ in METRICS}
    delta_values = {name: [] for name, _ in METRICS}
    display_zero_passes = 0

    for seed in range(2, 11):
        base, edited, train = read_seed(root, seed)
        row = {
            "seed": seed,
            "stop_reason": train.get("stop_reason"),
            "best_step": train.get("best_step"),
            "display_zero_check": bool(
                edited["static_branch_display_zero_check"]["passed"]
            ),
        }
        display_zero_passes += int(row["display_zero_check"])

        for name, path in METRICS:
            b = get(base, path)
            e = get(edited, path)
            d = e - b
            row[f"base_{name}"] = b
            row[f"v2_{name}"] = e
            row[f"delta_{name}"] = d
            base_values[name].append(b)
            edited_values[name].append(e)
            delta_values[name].append(d)
        rows.append(row)

    development = None
    dev_path = root / "mcf_fact_assoc_router_v2_seed1" / "official_mcf_eval.json"
    if dev_path.is_file():
        dev = json.loads(dev_path.read_text())
        development = {name: get(dev, path) for name, path in METRICS}

    summary = {
        "method": "static_overlap_fact_association_embeddings_router_v2",
        "dataset": "MCF",
        "development_seed": 1,
        "confirmatory_seeds": list(range(2, 11)),
        "confirmatory_n": 9,
        "hyperparameters_retuned_on_confirmatory_seeds": False,
        "display_zero_pass_count": display_zero_passes,
        "display_zero_total": 9,
        "development_seed1": development,
        "base": {name: stats(values) for name, values in base_values.items()},
        "router_v2": {name: stats(values) for name, values in edited_values.items()},
        "paired_delta_v2_minus_base": {
            name: stats(values) for name, values in delta_values.items()
        },
        "per_seed": rows,
    }

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({
        "confirmatory_seeds": summary["confirmatory_seeds"],
        "display_zero": f"{display_zero_passes}/9",
        "router_v2": summary["router_v2"],
        "paired_delta_v2_minus_base": summary["paired_delta_v2_minus_base"],
        "out_json": str(out_json),
        "out_csv": str(out_csv),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
