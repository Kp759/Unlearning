#!/usr/bin/env python3
"""Aggregate canonical SURE final evaluations with sample SD (n-1)."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def mean_sd(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "sample_sd": float("nan")}
    return {
        "mean": float(statistics.mean(values)),
        "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--filename", default="official_eval_locked.json")
    a = p.parse_args()

    root = Path(a.root)
    rows: List[Dict[str, Any]] = []
    for seed in a.seeds:
        path = root / f"seed{seed}" / a.filename
        if not path.exists():
            raise FileNotFoundError(path)
        result = json.loads(path.read_text(encoding="utf-8"))
        forget = result["forget"]
        row = {
            "seed": seed,
            "Eff": float(forget["Eff"]),
            "Gen": float(forget["Gen"]),
            "Spe": float(forget["Spe"]),
            "PPL": float(result["forget_PPL"]),
        }
        if a.dataset == "mcf":
            row["Spe_success"] = float(forget["Spe_success"])
        else:
            retain = result.get("retain", {})
            for key in ("Eff", "Gen", "Spe"):
                if retain.get(key) is not None:
                    row[f"R_{key}"] = float(retain[key])
        rows.append(row)

    metrics = [k for k in rows[0] if k != "seed"]
    aggregate = {
        "schema_version": 1,
        "dataset": a.dataset,
        "std_definition": "sample standard deviation (n-1)",
        "seeds": a.seeds,
        "per_seed": rows,
        "metrics": {key: mean_sd([float(r[key]) for r in rows]) for key in metrics},
    }
    out_json = root / "aggregate_canonical.json"
    out_json.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    md = [
        f"# Canonical SURE {a.dataset.upper()} aggregate",
        "",
        "All uncertainty values use sample standard deviation (`n-1`).",
        "",
        "| Metric | Mean ± sample SD |",
        "|---|---:|",
    ]
    for key in metrics:
        x = aggregate["metrics"][key]
        md.append(f"| {key} | {x['mean']:.6f} ± {x['sample_sd']:.6f} |")
    (root / "aggregate_canonical.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("Wrote", out_json)


if __name__ == "__main__":
    main()
