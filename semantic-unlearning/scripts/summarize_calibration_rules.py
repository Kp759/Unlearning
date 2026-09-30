#!/usr/bin/env python3
"""Router bias rules side by side, mean ± std over seeds, per dataset and layer.

    python scripts/summarize_calibration_rules.py

Rules (same weights, same rows; only the bias differs):
  shipped        min recall 0.98 on the calibration split (the multiseed runs)
  raw            stage-1 bias, p >= 0.5, no calibration (bias-rule ablation, raw_swap)
  <arm>          recalibrated on the validation set = calibration + audit
                 (outputs/calibration_rules_v1/...: balanced_fact, balanced_prompt, fpr0.1, ...)
Columns: official TEST metrics, then VALIDATION recall and same-subject false
fire (macro = per fact first, then over facts; pooled = over prompts), all
from recalibration.json on the same merged validation prompts.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

from summarize_mcf_layer_sweep import collect

TEST = {
    "mcf": ["forget_Gen", "display_zero", "forget_Spe", "retain_Gen"],
    "zsre": ["forget_Gen", "forget_paraphrase_route_active", "forget_Spe", "retain_Gen"],
    "mquake": ["forget_AtomicGen", "forget_atomicgen_route_correct", "retain_AtomicGen",
               "retain_atomicgen_route_active"],
    "rwku": ["forget_GenL1", "forget_GenL2", "heldout_route_active", "neighbor",
             "neighbor_route_active"],
}
VAL = [("macro_recall", "val recall (macro)"), ("macro_false_fire", "val false fire (macro)"),
       ("recall", "val recall (pooled)"), ("false_fire", "val false fire (pooled)")]
REF = {"mcf": "multiseed_regular_v1", "mquake": "multiseed_regular_v1", "zsre": "multiseed_reworded_v2",
       "rwku": "multiseed_regular_v1"}
RAW = {"mcf": "bias_rule_ablation_v1", "mquake": "bias_rule_ablation_v1",
       "zsre": "bias_rule_ablation_reworded_v2"}


def _num(v):
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.3g} ± {sd:.2g}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs")
    p.add_argument("--out-tag", default="calibration_rules_v1")
    p.add_argument("--datasets", nargs="+", default=["mcf", "zsre", "mquake"])
    p.add_argument("--layers", nargs="+", default=["19", "23"])
    a = p.parse_args(argv)
    root = Path(a.root)

    for ds in a.datasets:
        for layer in a.layers:
            L = f"L{int(layer):02d}"
            table = defaultdict(lambda: defaultdict(list))
            met = defaultdict(list)
            seeds_seen = set()
            for seed_dir in sorted((root / a.out_tag / ds).glob("seed*")):
                arm_dirs = sorted(d for d in (seed_dir / L).glob("*")
                                  if (d / "recalibration.json").is_file())
                if not arm_dirs:
                    continue
                seeds_seen.add(seed_dir.name)
                rec0 = json.loads((arm_dirs[0] / "recalibration.json").read_text())
                val0 = rec0["by_split"]["validation"]
                sources = {
                    "shipped (recall ≥ 0.98, calibration split)":
                        (root / f"{ds}_{REF[ds]}" / seed_dir.name / L / "linear_global", val0["shipped"],
                         rec0["cutoff_t"]["shipped"]),
                }
                if ds in RAW:
                    sources["raw (p ≥ 0.5, no calibration)"] = (
                        root / RAW[ds] / "lbfgs" / ds / seed_dir.name / L / "raw_swap", val0["raw"], 0.0)
                for d in arm_dirs:
                    rec = json.loads((d / "recalibration.json").read_text())
                    label = f"{d.name} (validation = cal + audit)"
                    sources[label] = (d, rec["by_split"]["validation"]["new"], rec["cutoff_t"]["new"])
                    if rec.get("constraint_status"):
                        met[label].append(rec["constraint_status"] == "recall_and_false_fire_met")
                for label, (run, val, t) in sources.items():
                    row = collect(run, label) if run.is_dir() else {}
                    if row.get("status") == "complete":
                        for m in TEST[ds]:
                            table[label][m].append(_num(row.get(m)))
                    for k, _ in VAL:
                        table[label][k].append(_num(val.get(k)))
                    table[label]["cutoff t"].append(_num(t))
            if not seeds_seen:
                continue
            cols = TEST[ds] + [k for k, _ in VAL] + ["cutoff t"]
            names = TEST[ds] + [n for _, n in VAL] + ["cutoff t", "recall AND false-fire targets met (seeds)"]
            print(f"\n### {ds.upper()} {L}  (seeds: {', '.join(sorted(seeds_seen))})\n")
            print("| rule | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1))
            for label, vals in table.items():
                both = f"{sum(met[label])}/{len(met[label])}" if met[label] else "–"
                print(f"| {label} | " + " | ".join(_fmt(vals[c]) for c in cols) + f" | {both} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
