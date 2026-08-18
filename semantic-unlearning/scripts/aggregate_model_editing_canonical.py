#!/usr/bin/env python3
"""Aggregate canonical model-editing baseline evaluations with sample SD (n-1)."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Dict, List


def mean_sd(values: List[float]) -> Dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("ROME", "MEMIT"), required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--filename", default="official_eval_locked.json")
    a = p.parse_args()

    root = Path(a.root)
    rows = []
    for seed in a.seeds:
        path = root / f"seed{seed}" / a.filename
        result = json.loads(path.read_text(encoding="utf-8"))
        forget = result["forget"]
        rows.append(
            {
                "seed": seed,
                "Eff": float(forget["Eff"]),
                "Gen": float(forget["Gen"]),
                "Spe": float(forget["Spe"]),
                "Spe_success": float(forget["Spe_success"]),
                "PPL": float(result["forget_PPL"]),
            }
        )

    metrics = [key for key in rows[0] if key != "seed"]
    aggregate = {
        "schema_version": 1,
        "method": a.method,
        "dataset": "mcf",
        "std_definition": "sample standard deviation (n-1)",
        "seeds": a.seeds,
        "per_seed": rows,
        "metrics": {
            key: mean_sd([float(row[key]) for row in rows]) for key in metrics
        },
    }
    out_json = root / "aggregate_canonical.json"
    out_json.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# Canonical {a.method} MCF aggregate",
        "",
        "All uncertainty values use sample standard deviation (`n-1`).",
        "",
        "| Metric | Mean ± sample SD |",
        "|---|---:|",
    ]
    for key in metrics:
        item = aggregate["metrics"][key]
        md.append(f"| {key} | {item['mean']:.6f} ± {item['sample_sd']:.6f} |")
    (root / "aggregate_canonical.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
