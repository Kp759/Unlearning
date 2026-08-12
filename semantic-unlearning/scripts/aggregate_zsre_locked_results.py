#!/usr/bin/env python3
"""Aggregate locked ZeroUnlearn-style ZsRE SURE results across seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs/zsre_zerounlearn_forget_only_locked_3b")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--out-prefix", default=None)
    return p.parse_args()


def stats(values: List[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std_population": float(arr.std(ddof=0)),
        "std_sample": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "values": [float(v) for v in arr],
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
        path = root / f"seed{seed}" / "official_eval_locked.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing seed result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        repair_path = root / f"seed{seed}" / "repair_forget_only" / "repair_summary.json"
        repair = (
            json.loads(repair_path.read_text(encoding="utf-8"))
            if repair_path.is_file()
            else {}
        )
        rows.append(
            {
                "seed": int(seed),
                "Eff": float(result["forget"]["Eff"]),
                "Gen": float(result["forget"]["Gen"]),
                "Spe": float(result["forget"]["Spe"]),
                "PPL": (
                    None if result.get("forget_PPL") is None else float(result["forget_PPL"])
                ),
                "retain_Eff": float(result["retain"]["Eff"]),
                "retain_Gen": float(result["retain"]["Gen"]),
                "retain_Spe": float(result["retain"]["Spe"]),
                "selected_lm_head_rows": int(repair.get("selected_lm_head_rows", 0)),
                "selected_scale": repair.get("selected_scale"),
                "delta_norm": repair.get("effective_delta_norm"),
            }
        )

    metric_names = [
        "Eff", "Gen", "Spe", "retain_Eff", "retain_Gen", "retain_Spe"
    ]
    aggregate = {
        "protocol": "zsre_zerounlearn_forget_only_locked_probes",
        "root": str(root),
        "seeds": [int(seed) for seed in args.seeds],
        "per_seed": rows,
        "aggregate": {
            name: stats([row[name] for row in rows]) for name in metric_names
        },
    }
    ppl_values = [row["PPL"] for row in rows if row["PPL"] is not None]
    aggregate["aggregate"]["PPL"] = stats(ppl_values) if ppl_values else None

    prefix = Path(args.out_prefix) if args.out_prefix else root / "aggregate_locked"
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Locked ZeroUnlearn-style ZsRE SURE aggregate",
        "",
        "Stage 1/2 use 50 direct forget requested_rewrite records and zero benchmark retain records. Rephrase/locality probes and 1000 retain records are final-evaluation-only.",
        "",
        "| Metric | Mean | Population SD |",
        "|---|---:|---:|",
    ]
    for name in ["Eff", "Gen", "Spe", "PPL", "retain_Eff", "retain_Gen", "retain_Spe"]:
        item = aggregate["aggregate"].get(name)
        if item is None:
            lines.append(f"| {name} | null | null |")
        else:
            lines.append(
                f"| {name} | {item['mean']:.4f} | {item['std_population']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Per seed",
            "",
            "| Seed | Eff | Gen | Spe | PPL | repair rows | scale |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        ppl = "null" if row["PPL"] is None else f"{row['PPL']:.4f}"
        scale = "null" if row["selected_scale"] is None else str(row["selected_scale"])
        lines.append(
            f"| {row['seed']} | {row['Eff']:.2f} | {row['Gen']:.2f} | "
            f"{row['Spe']:.2f} | {ppl} | {row['selected_lm_head_rows']} | {scale} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    for name in ["Eff", "Gen", "Spe", "PPL"]:
        item = aggregate["aggregate"].get(name)
        if item is not None:
            print(f"{name:4s}: {item['mean']:.4f} ± {item['std_population']:.4f}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    main()
