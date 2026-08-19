#!/usr/bin/env python3
"""Legacy aggregate for target-true-sensitive MCF pairwise diagnostics.

WARNING: this script does NOT compute ZeroUnlearn's probability-based Eff/Gen.
Older versions incorrectly labeled residual pairwise sensitive-preference rates
as ``Eff``/``Gen``. Those quantities are now named explicitly:

  SensitivePref_direct      = 100 P[NLL(target_true) < NLL(target_new)]
  SensitivePref_paraphrase  = same on held-out paraphrases

The complementary pairwise success rates are FS/GFS. Original MCF target_true
is sensitive; target_new is the counterfactual reference.
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


def dig(obj: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first(data: Dict[str, Any], *paths: Sequence[str]) -> Any:
    for path in paths:
        value = dig(data, path)
        if value is not None:
            return value
    return None


def pop_sd(xs: Sequence[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument(
        "--out-prefix",
        default="zerounlearn_style_target_true_sensitive_aggregate",
        help="Legacy filename only; contents use corrected FS/GFS and SensitivePref names.",
    )
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

        direct_counts = dig(data, ("diagnostics", "direct", "post_counts")) or {}
        para_counts = dig(data, ("diagnostics", "paraphrase", "post_counts")) or {}
        row = {
            "seed": seed,
            "SensitivePref_direct": float(first(
                data,
                ("metrics", "SensitivePref_direct", "mean"),
                ("metrics", "Eff_Pref", "mean"),
            )),
            "SensitivePref_paraphrase": float(first(
                data,
                ("metrics", "SensitivePref_paraphrase", "mean"),
                ("metrics", "Gen_Pref", "mean"),
            )),
            "FS": float(first(data, ("metrics", "FS", "mean"), ("metrics", "Eff", "mean"))),
            "GFS": float(first(data, ("metrics", "GFS", "mean"), ("metrics", "Gen", "mean"))),
            "Spe_margin": float(dig(data, ("metrics", "Spe_margin", "mean"))),
            "Spe_success": float(dig(data, ("metrics", "Spe_success", "mean"))),
            "PPL": float(dig(data, ("metrics", "PPL"))),
            "rewrite_prompt_instances": int(direct_counts.get("prompt_instance_count", 0)),
            "rewrite_sensitive_preferred": int(direct_counts.get("sensitive_preferred_prompt_instances", 0)),
            "rewrite_reference_preferred": int(direct_counts.get("reference_preferred_prompt_instances", 0)),
            "rewrite_ties": int(direct_counts.get("exact_nll_ties", 0)),
            "paraphrase_prompt_instances": int(para_counts.get("prompt_instance_count", 0)),
            "paraphrase_sensitive_preferred": int(para_counts.get("sensitive_preferred_prompt_instances", 0)),
            "paraphrase_reference_preferred": int(para_counts.get("reference_preferred_prompt_instances", 0)),
            "paraphrase_ties": int(para_counts.get("exact_nll_ties", 0)),
        }
        rows.append(row)

    metric_names = (
        "SensitivePref_direct", "SensitivePref_paraphrase", "FS", "GFS",
        "Spe_margin", "Spe_success", "PPL",
    )
    directions = {
        "SensitivePref_direct": "↓",
        "SensitivePref_paraphrase": "↓",
        "FS": "↑",
        "GFS": "↑",
        "Spe_margin": "↑",
        "Spe_success": "↑",
        "PPL": "↓/stable",
    }

    aggregate: Dict[str, Any] = {
        "schema_version": 2,
        "dataset": "MCF",
        "protocol": "target_true_sensitive_pairwise_diagnostics",
        "input_schemas": sorted(schemas_seen),
        "sensitive": "original requested_rewrite.target_true",
        "reference": "original requested_rewrite.target_new",
        "zero_unlearn_probability_eff_gen_computed": False,
        "correction_note": (
            "Older versions of this legacy script mislabeled pairwise sensitive-preference "
            "rates as ZeroUnlearn-style Eff/Gen. Correct names are SensitivePref_direct/"
            "SensitivePref_paraphrase; complementary success metrics are FS/GFS."
        ),
        "metric_semantics": {
            "SensitivePref_direct": "100 * P[NLL(target_true) < NLL(target_new)] on rewrite; lower better",
            "SensitivePref_paraphrase": "same on paraphrases; lower better",
            "FS": "100 * P[NLL(target_true) > NLL(target_new)] on rewrite; higher better",
            "GFS": "same on paraphrases; higher better",
            "Spe_margin": "original MCF neighborhood probability-difference diagnostic; higher better",
            "Spe_success": "original MCF neighborhood true-answer preservation rate; higher better",
            "PPL": "lower/stable is better",
        },
        "seeds": list(a.seeds),
        "n_seeds": len(a.seeds),
        "std_convention_primary": "population standard deviation (ddof=0)",
        "metrics": {},
        "prompt_totals": {
            "rewrite_prompt_instances": sum(r["rewrite_prompt_instances"] for r in rows),
            "rewrite_sensitive_preferred": sum(r["rewrite_sensitive_preferred"] for r in rows),
            "rewrite_reference_preferred": sum(r["rewrite_reference_preferred"] for r in rows),
            "rewrite_ties": sum(r["rewrite_ties"] for r in rows),
            "paraphrase_prompt_instances": sum(r["paraphrase_prompt_instances"] for r in rows),
            "paraphrase_sensitive_preferred": sum(r["paraphrase_sensitive_preferred"] for r in rows),
            "paraphrase_reference_preferred": sum(r["paraphrase_reference_preferred"] for r in rows),
            "paraphrase_ties": sum(r["paraphrase_ties"] for r in rows),
        },
    }
    for name in metric_names:
        vals = [float(r[name]) for r in rows]
        aggregate["metrics"][name] = {
            "mean": float(statistics.mean(vals)),
            "population_sd": pop_sd(vals),
            "direction": directions[name],
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
        "# MCF target-true-sensitive pairwise diagnostics",
        "",
        "Sensitive = original `target_true`; reference = original `target_new`.",
        "**This file does not contain ZeroUnlearn probability-based Eff/Gen.**",
        "",
        "| Metric | Mean ± population SD | Direction |",
        "|---|---:|:---:|",
    ]
    for name in metric_names:
        item = aggregate["metrics"][name]
        lines.append(f"| {name} | **{item['mean']:.4f} ± {item['population_sd']:.4f}** | {directions[name]} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
