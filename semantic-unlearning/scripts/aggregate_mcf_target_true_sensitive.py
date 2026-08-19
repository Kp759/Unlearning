#!/usr/bin/env python3
"""Aggregate target-true-sensitive MCF SURE diagnostics across seeds.

Primary paper-facing behavioral metrics are:

  FS  = 100 P[NLL(target_true) > NLL(target_new)] on direct rewrites (higher better)
  GFS = 100 P[NLL(target_true) > NLL(target_new)] on paraphrases (higher better)

These are pairwise forget-success rates, NOT ZeroUnlearn's probability-based
Eff/Gen. Sensitive-preference rates are retained as lower-is-better diagnostics.
The loader accepts both the new v4 FS/GFS schema and legacy v3 files.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence

SCHEMAS = {
    "mcf_target_true_sensitive_v4_fs",
    "mcf_target_true_sensitive_v3_rome",
}

METRICS = (
    ("FS", ("metrics", "FS", "mean"), ("metrics", "Eff", "mean"), "↑"),
    ("GFS", ("metrics", "GFS", "mean"), ("metrics", "Gen", "mean"), "↑"),
    ("SensitivePref_direct", ("metrics", "SensitivePref_direct", "mean"), ("metrics", "Eff_Pref", "mean"), "↓"),
    ("SensitivePref_paraphrase", ("metrics", "SensitivePref_paraphrase", "mean"), ("metrics", "Gen_Pref", "mean"), "↓"),
    ("Delta_Sensitive_NLL_direct", ("metrics", "Delta_Sensitive_NLL_direct", "mean"), None, "↑"),
    ("Delta_Reference_NLL_direct", ("metrics", "Delta_Reference_NLL_direct", "mean"), None, "audit"),
    ("NLL_Separation_direct", ("metrics", "NLL_Separation_direct", "mean"), None, "↑"),
    ("Delta_Sensitive_NLL_paraphrase", ("metrics", "Delta_Sensitive_NLL_paraphrase", "mean"), None, "↑"),
    ("Delta_Reference_NLL_paraphrase", ("metrics", "Delta_Reference_NLL_paraphrase", "mean"), None, "audit"),
    ("NLL_Separation_paraphrase", ("metrics", "NLL_Separation_paraphrase", "mean"), None, "↑"),
    ("Spe_margin", ("metrics", "Spe_margin", "mean"), None, "↑"),
    ("Spe_success", ("metrics", "Spe_success", "mean"), None, "↑"),
    ("PPL", ("metrics", "PPL"), None, "↓/stable"),
)


def dig(obj: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def read_metric(data: Dict[str, Any], primary: Sequence[str], fallback: Sequence[str] | None) -> Any:
    value = dig(data, primary)
    if value is None and fallback is not None:
        value = dig(data, fallback)
    return value


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument("--out-prefix", default="target_true_sensitive_aggregate")
    a = p.parse_args()

    root = Path(a.root).resolve()
    rows: List[Dict[str, Any]] = []
    schemas_seen: set[str] = set()

    for seed in a.seeds:
        path = root / f"seed{seed}" / "target_true_sensitive_eval.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        schema = str(data.get("metric_schema"))
        if schema not in SCHEMAS:
            raise RuntimeError(f"Unexpected metric schema in {path}: {schema}")
        schemas_seen.add(schema)
        if int(data.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed mismatch in {path}")

        row: Dict[str, Any] = {"seed": seed}
        for name, primary, fallback, _ in METRICS:
            row[name] = read_metric(data, primary, fallback)
        rows.append(row)

    aggregate: Dict[str, Any] = {
        "schema_version": 3,
        "metric_schema": "mcf_target_true_sensitive_aggregate_v3_fs",
        "input_schemas": sorted(schemas_seen),
        "dataset": "MCF",
        "method": "SURE-LM-target-true-sensitive",
        "sensitive": "original requested_rewrite.target_true",
        "reference": "original requested_rewrite.target_new",
        "forget_success_rule": "NLL(target_true) > NLL(target_new)",
        "zero_unlearn_probability_eff_gen_computed": False,
        "zero_unlearn_note": (
            "FS/GFS are pairwise success rates and must not be called ZeroUnlearn Eff/Gen. "
            "Probability-based ZeroUnlearn Eff/Gen require a separate aligned evaluator."
        ),
        "seeds": list(a.seeds),
        "n_seeds": len(a.seeds),
        "across_seed_sd": "sample_standard_deviation_n_minus_1",
        "metrics": {},
    }

    for name, _, _, direction in METRICS:
        vals = [float(row[name]) for row in rows if row[name] is not None]
        aggregate["metrics"][name] = {
            "mean": float(statistics.mean(vals)) if vals else None,
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
        "# MCF target-true-sensitive SURE aggregate",
        "",
        "Sensitive = original `target_true`; counterfactual reference = original `target_new`.",
        "`FS`/`GFS` are pairwise behavioral success rates (`NLL(target_true) > NLL(target_new)`), higher is better.",
        "**Do not call FS/GFS ZeroUnlearn Eff/Gen.** ZeroUnlearn's probability-based Eff/Gen are not computed by this evaluator.",
        "Across-seed uncertainty is sample SD (n-1).",
        "",
        "| Metric | Mean | Sample SD | Direction |",
        "|---|---:|---:|:---:|",
    ]
    for name, _, _, direction in METRICS:
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
