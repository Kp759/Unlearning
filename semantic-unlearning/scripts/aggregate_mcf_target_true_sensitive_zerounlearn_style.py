#!/usr/bin/env python3
"""Aggregate target-true-sensitive MCF using the ZeroUnlearn-style directions.

This is the semantic mirror of the registered target-new-sensitive MCF record.
Original MCF ``target_true`` is the sensitive answer and original ``target_new``
is the non-sensitive/reference answer.

To preserve the *meaning* of ZeroUnlearn's lower-is-better Eff/Gen after this
field inversion, Eff/Gen here are the sensitive-preference rates:

    Eff = 100 P[NLL(target_true) < NLL(target_new)]   on rewrite prompts
    Gen = 100 P[NLL(target_true) < NLL(target_new)]   on paraphrases

Thus lower is better: zero means the sensitive original target_true is never
strictly preferred over target_new. Exact NLL ties are reported separately by
the per-seed canonical evaluator and are not counted as sensitive preference.

Spe and PPL are copied from the same original-unswapped MCF final evaluation.
Across-seed uncertainty uses population SD (ddof=0), matching the registered
20260810 ZeroUnlearn-style best-run record.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Sequence


def dig(obj: Dict[str, Any], path: Sequence[str]) -> Any:
    cur: Any = obj
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def pop_sd(xs: Sequence[float]) -> float:
    return float(statistics.pstdev(xs)) if len(xs) > 1 else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", nargs="+", type=int, required=True)
    p.add_argument(
        "--out-prefix",
        default="zerounlearn_style_target_true_sensitive_aggregate",
    )
    a = p.parse_args()

    root = Path(a.root).resolve()
    rows: List[Dict[str, Any]] = []
    for seed in a.seeds:
        path = root / f"seed{seed}" / "target_true_sensitive_eval.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing seed result: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("metric_schema") != "mcf_target_true_sensitive_v3_rome":
            raise RuntimeError(f"Unexpected metric schema in {path}")
        if int(data.get("seed", -1)) != seed:
            raise RuntimeError(f"Seed mismatch in {path}")

        direct_counts = dig(data, ("diagnostics", "direct", "post_counts")) or {}
        para_counts = dig(data, ("diagnostics", "paraphrase", "post_counts")) or {}
        row = {
            "seed": seed,
            # Exact ZeroUnlearn-style semantic analogue after swapping which
            # answer is sensitive: residual sensitive preference, lower better.
            "Eff": float(dig(data, ("metrics", "Eff_Pref", "mean"))),
            "Gen": float(dig(data, ("metrics", "Gen_Pref", "mean"))),
            "Spe": float(dig(data, ("metrics", "Spe_margin", "mean"))),
            "Spe_success": float(dig(data, ("metrics", "Spe_success", "mean"))),
            "PPL": float(dig(data, ("metrics", "PPL"))),
            # Keep canonical complementary ROME success visible for auditing.
            "Canonical_Eff_up": float(dig(data, ("metrics", "Eff", "mean"))),
            "Canonical_Gen_up": float(dig(data, ("metrics", "Gen", "mean"))),
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

    metric_names = ("Eff", "Gen", "Spe", "Spe_success", "PPL", "Canonical_Eff_up", "Canonical_Gen_up")
    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "dataset": "MCF",
        "protocol": "zerounlearn_locked_forget_only_target_true_sensitive_mirror",
        "sensitive": "original requested_rewrite.target_true",
        "reference": "original requested_rewrite.target_new",
        "metric_semantics": {
            "Eff": "100 * P[NLL(target_true) < NLL(target_new)] on rewrite; lower is better",
            "Gen": "100 * P[NLL(target_true) < NLL(target_new)] on paraphrase; lower is better",
            "Spe": "original MCF neighborhood probability-difference score; higher is better",
            "Spe_success": "original MCF neighborhood true-answer preservation rate; higher is better",
            "PPL": "lower/stable is better",
            "Canonical_Eff_up": "100 * P[NLL(target_true) > NLL(target_new)] on rewrite; higher is better",
            "Canonical_Gen_up": "100 * P[NLL(target_true) > NLL(target_new)] on paraphrase; higher is better",
        },
        "seeds": list(a.seeds),
        "n_seeds": len(a.seeds),
        "std_convention_primary": "population standard deviation (ddof=0), matching registered 20260810 record",
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
        "# MCF target-true-sensitive — ZeroUnlearn-style mirrored aggregate",
        "",
        "Sensitive = original `target_true`; non-sensitive/reference = original `target_new`.",
        "Eff/Gen below are residual sensitive-preference rates, so **lower is better**, preserving the semantic meaning of the registered ZeroUnlearn-style track after swapping the sensitive field.",
        "Across-seed uncertainty uses population SD (`ddof=0`).",
        "",
        "| Metric | Mean ± population SD | Direction |",
        "|---|---:|:---:|",
    ]
    directions = {
        "Eff": "↓", "Gen": "↓", "Spe": "↑", "Spe_success": "↑", "PPL": "↓/stable",
        "Canonical_Eff_up": "↑", "Canonical_Gen_up": "↑",
    }
    for name in metric_names:
        item = aggregate["metrics"][name]
        lines.append(f"| {name} | **{item['mean']:.4f} ± {item['population_sd']:.4f}** | {directions[name]} |")
    totals = aggregate["prompt_totals"]
    lines += [
        "",
        "## Prompt-level totals",
        "",
        f"- rewrite prompts: `{totals['rewrite_prompt_instances']}`, sensitive still preferred: `{totals['rewrite_sensitive_preferred']}`, ties: `{totals['rewrite_ties']}`",
        f"- held-out paraphrase prompts: `{totals['paraphrase_prompt_instances']}`, sensitive still preferred: `{totals['paraphrase_sensitive_preferred']}`, ties: `{totals['paraphrase_ties']}`",
        "",
        "Canonical complementary ROME success (`Canonical_Eff_up`/`Canonical_Gen_up`) is retained only to make the direction change fully auditable.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print("JSON:", json_path)
    print("CSV:", csv_path)


if __name__ == "__main__":
    main()
