#!/usr/bin/env python3
"""Aggregate canonical target-true-sensitive MCF results across seeds.

Across-seed uncertainty uses sample standard deviation (n-1), while each seed's
JSON preserves the evaluator's per-record population SD separately.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


METRICS = (
    ("Eff_Pref", ("metrics", "Eff_Pref", "mean"), "↓"),
    ("Gen_Pref", ("metrics", "Gen_Pref", "mean"), "↓"),
    ("Delta_Sensitive_NLL_direct", ("metrics", "Delta_Sensitive_NLL_direct", "mean"), "↑"),
    ("NLL_Separation_direct", ("metrics", "NLL_Separation_direct", "mean"), "↑"),
    ("Delta_Sensitive_NLL_paraphrase", ("metrics", "Delta_Sensitive_NLL_paraphrase", "mean"), "↑"),
    ("NLL_Separation_paraphrase", ("metrics", "NLL_Separation_paraphrase", "mean"), "↑"),
    ("Spe_margin", ("metrics", "Spe_margin", "mean"), "↑"),
    ("Spe_success", ("metrics", "Spe_success", "mean"), "↑"),
    ("PPL", ("metrics", "PPL"), "↓"),
)


def dig(obj: Dict[str, Any], path: tuple[str, ...]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out-prefix", default="target_true_sensitive_aggregate")
    a = p.parse_args()

    root = Path(a.root).resolve()
    rows: List[Dict[str, Any]] = []
    payloads: List[Dict[str, Any]] = []
    for seed in a.seeds:
        path = root / f"seed{seed}" / "target_true_sensitive_eval.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("metric_schema") != "mcf_target_true_sensitive_v2":
            raise RuntimeError(f"Unexpected metric schema in {path}")
        if int(data.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed mismatch in {path}")
        payloads.append(data)
        row: Dict[str, Any] = {"seed": seed}
        for name, metric_path, _ in METRICS:
            row[name] = dig(data, metric_path)
        rows.append(row)

    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "metric_schema": "mcf_target_true_sensitive_v2",
        "dataset": "MCF",
        "method": "SURE-LM-canonical-target-true-sensitive",
        "seeds": list(a.seeds),
        "n_seeds": len(a.seeds),
        "across_seed_sd": "sample_standard_deviation_n_minus_1",
        "metrics": {},
    }
    for name, _, direction in METRICS:
        vals = [float(row[name]) for row in rows if row[name] is not None]
        if len(vals) != len(rows):
            aggregate["metrics"][name] = {
                "mean": None,
                "sample_sd": None,
                "direction": direction,
                "n": len(vals),
            }
            continue
        aggregate["metrics"][name] = {
            "mean": float(statistics.mean(vals)),
            "sample_sd": float(statistics.stdev(vals)) if len(vals) > 1 else None,
            "direction": direction,
            "n": len(vals),
        }

    json_path = root / f"{a.out_prefix}.json"
    csv_path = root / f"{a.out_prefix}.csv"
    md_path = root / f"{a.out_prefix}.md"
    json_path.write_text(json.dumps({"aggregate": aggregate, "per_seed": rows}, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# MCF canonical target-true-sensitive SURE aggregate",
        "",
        "Sensitive = original `target_true`; counterfactual reference = original `target_new`.",
        "Across-seed uncertainty is sample SD (n-1).",
        "",
        "| Metric | Mean | Sample SD | Direction |",
        "|---|---:|---:|:---:|",
    ]
    for name, _, direction in METRICS:
        item = aggregate["metrics"][name]
        mean = "NA" if item["mean"] is None else f"{item['mean']:.6f}"
        sd = "NA" if item["sample_sd"] is None else f"{item['sample_sd']:.6f}"
        lines.append(f"| {name} | {mean} | {sd} | {direction} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(md_path.read_text(encoding="utf-8"))
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
