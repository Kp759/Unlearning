#!/usr/bin/env python3
"""Aggregate locked ZsRE SURE GA/GD results across seeds.

Reads only completed final/post-hoc evaluation artifacts and Stage-2 repair summaries.
This script does not perform model selection and never feeds evaluation metrics back
into training.  It writes per-seed CSV plus mean/sample-SD JSON and Markdown.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List

METRICS = (
    "F-Eff", "F-Gen", "F-Spe",
    "R-Eff", "R-Gen", "R-Spe",
    "PPL",
)


def parse_seed_spec(text: str) -> List[int]:
    out: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def extract_eval(x: Dict[str, Any]) -> Dict[str, float]:
    return {
        "F-Eff": float(x["forget"]["Eff"]),
        "F-Gen": float(x["forget"]["Gen"]),
        "F-Spe": float(x["forget"]["Spe"]),
        "R-Eff": float(x["retain"]["Eff"]),
        "R-Gen": float(x["retain"]["Gen"]),
        "R-Spe": float(x["retain"]["Spe"]),
        "PPL": float(x["forget_PPL"]),
    }


def mean_sd(values: Iterable[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": math.nan, "sd": math.nan}
    return {
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def fmt(x: float) -> str:
    return f"{x:.3f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", default="1-10")
    p.add_argument("--require-all", action="store_true")
    p.add_argument("--output-prefix", default="aggregate_10seeds")
    a = p.parse_args()

    root = Path(a.root).resolve()
    seeds = parse_seed_spec(a.seeds)
    rows: List[Dict[str, Any]] = []
    missing: List[int] = []

    for seed in seeds:
        sroot = root / f"seed{seed}"
        final_path = sroot / "official_eval_locked.json"
        stage1_path = sroot / "official_eval_stage1_posthoc.json"
        repair_path = sroot / "stage2_sensitive_row_repair" / "repair_summary.json"
        if not (final_path.exists() and stage1_path.exists() and repair_path.exists()):
            missing.append(seed)
            continue

        final = extract_eval(json.loads(final_path.read_text()))
        stage1 = extract_eval(json.loads(stage1_path.read_text()))
        repair = json.loads(repair_path.read_text())

        row: Dict[str, Any] = {"seed": seed}
        row.update({f"stage1_{k}": v for k, v in stage1.items()})
        row.update({f"final_{k}": v for k, v in final.items()})
        row.update({
            "active_before": int(repair.get("active_before", -1)),
            "active_after": int(repair.get("active_after", -1)),
            "selected_lm_head_rows": int(repair.get("selected_lm_head_rows", 0)),
            "best_step": int(repair.get("best_step", 0)),
            "selected_scale": float(repair.get("selected_scale", 0.0)),
            "full_delta_norm": float(repair.get("full_delta_norm", 0.0)),
            "Unknown_used": bool(repair.get("Unknown_used", False)),
            "IDK_used": bool(repair.get("IDK_used", False)),
            "target_new_seen": bool(repair.get("target_new_seen", False)),
            "retain_seen": int(repair.get("retain_seen", 0)),
            "rephrases_seen": int(repair.get("rephrases_seen", 0)),
            "locality_seen": int(repair.get("locality_seen", 0)),
            "PPL_seen": bool(repair.get("PPL_seen", False)),
        })
        rows.append(row)

    if missing and a.require_all:
        raise SystemExit(f"Missing/incomplete seeds: {missing}")
    if not rows:
        raise SystemExit("No completed seeds found")

    summary: Dict[str, Any] = {
        "root": str(root),
        "requested_seeds": seeds,
        "completed_seeds": [r["seed"] for r in rows],
        "missing_seeds": missing,
        "n": len(rows),
        "std_definition": "sample standard deviation (n-1)",
        "stage1": {},
        "final": {},
        "repair": {},
    }
    for stage in ("stage1", "final"):
        for metric in METRICS:
            summary[stage][metric] = mean_sd(r[f"{stage}_{metric}"] for r in rows)

    for key in (
        "active_before", "active_after", "selected_lm_head_rows",
        "best_step", "selected_scale", "full_delta_norm",
    ):
        summary["repair"][key] = mean_sd(float(r[key]) for r in rows)

    summary["final"]["Eff_zero_seed_count"] = sum(
        abs(r["final_F-Eff"]) < 1e-12 for r in rows
    )
    summary["protocol_audit"] = {
        "all_Unknown_used_false": all(not r["Unknown_used"] for r in rows),
        "all_IDK_used_false": all(not r["IDK_used"] for r in rows),
        "all_target_new_seen_false": all(not r["target_new_seen"] for r in rows),
        "all_retain_seen_zero": all(r["retain_seen"] == 0 for r in rows),
        "all_rephrases_seen_zero": all(r["rephrases_seen"] == 0 for r in rows),
        "all_locality_seen_zero": all(r["locality_seen"] == 0 for r in rows),
        "all_PPL_seen_false": all(not r["PPL_seen"] for r in rows),
    }

    prefix = root / a.output_prefix
    with (prefix.with_suffix(".csv")).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# ZsRE SURE no-neutral aggressive GA/GD — 10-seed aggregate",
        "",
        f"Completed seeds: {summary['completed_seeds']}",
        f"Missing seeds: {missing}",
        "",
        "| Stage | F-Eff ↓ | F-Gen ↓ | F-Spe ↑ | R-Eff ↑ | R-Gen ↑ | R-Spe ↑ | PPL ↓ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage, label in (("stage1", "Stage 1"), ("final", "Stage 2 / Final")):
        vals = []
        for m in METRICS:
            d = summary[stage][m]
            vals.append(f"{d['mean']:.3f} ± {d['sd']:.3f}")
        lines.append("| " + label + " | " + " | ".join(vals) + " |")
    lines += [
        "",
        f"Eff=0 seeds: {summary['final']['Eff_zero_seed_count']}/{len(rows)}",
        "",
        "## Stage-2 repair",
        "",
        "| Quantity | Mean ± SD |",
        "|---|---:|",
    ]
    for key in (
        "active_before", "active_after", "selected_lm_head_rows",
        "best_step", "selected_scale", "full_delta_norm",
    ):
        d = summary["repair"][key]
        lines.append(f"| {key} | {d['mean']:.3f} ± {d['sd']:.3f} |")
    lines += ["", "## Protocol audit", ""]
    for k, v in summary["protocol_audit"].items():
        lines.append(f"- {k}: {v}")
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")

    print(f"Completed {len(rows)}/{len(seeds)} seeds; missing={missing}")
    print(
        f"{'STAGE':<12}{'F-Eff':>16}{'F-Gen':>16}{'F-Spe':>16}"
        f"{'R-Eff':>16}{'R-Gen':>16}{'R-Spe':>16}{'PPL':>16}"
    )
    for stage, label in (("stage1", "Stage1"), ("final", "Final")):
        chunks = []
        for m in METRICS:
            d = summary[stage][m]
            chunks.append(f"{d['mean']:.3f}±{d['sd']:.3f}")
        print(f"{label:<12}" + "".join(f"{x:>16}" for x in chunks))
    print(f"Eff=0 seeds: {summary['final']['Eff_zero_seed_count']}/{len(rows)}")
    print("Wrote:", prefix.with_suffix(".csv"))
    print("Wrote:", prefix.with_suffix(".json"))
    print("Wrote:", prefix.with_suffix(".md"))


if __name__ == "__main__":
    main()
