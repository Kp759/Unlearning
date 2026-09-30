#!/usr/bin/env python3
"""Layer-wise table for ONE router bias rule, mean ± std over seeds (like the sweep tables).

    python scripts/summarize_arm_layers.py --arm recall0.98_fact
    python scripts/summarize_arm_layers.py --arm fpr0.1_fact --datasets zsre --layers 19 23

Reads outputs/<out-tag>/<dataset>/seed*/L??/<arm>/{official_<dataset>_eval.json,
recalibration.json}. Official TEST metrics per layer, plus the rule's VALIDATION
recall and same-subject false fire (per-fact averages) and the cutoff it chose.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from summarize_mcf_layer_sweep import collect

TEST = {
    "mcf": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL", "display_zero"],
    "zsre": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL",
             "forget_paraphrase_route_active"],
    "mquake": ["forget_Eff", "forget_AtomicGen", "retain_Eff", "retain_AtomicGen", "PPL",
               "forget_atomicgen_route_correct"],
}
VAL = [("macro_recall", "val recall"), ("macro_false_fire", "val same-subject false fire")]


def _num(v):
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.4g} ± {sd:.2g}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True, help="e.g. recall0.98_fact, fpr0.1_fact, balanced_fact")
    p.add_argument("--root", default="outputs")
    p.add_argument("--out-tag", default="calibration_rules_v1")
    p.add_argument("--datasets", nargs="+", default=["mcf", "zsre", "mquake"])
    p.add_argument("--layers", nargs="+", default=["1", "3", "7", "13", "19", "23", "27"])
    a = p.parse_args(argv)

    for ds in a.datasets:
        head = ["layer", "n"] + TEST[ds] + [n for _, n in VAL] + ["cutoff t"]
        lines = []
        for layer in a.layers:
            L = f"L{int(layer):02d}"
            rows, vals, cuts = [], {k: [] for k, _ in VAL}, []
            for run in sorted(Path(a.root, a.out_tag, ds).glob(f"seed*/{L}/{a.arm}")):
                row = collect(run, a.arm)
                if row.get("status") != "complete":
                    continue
                rows.append(row)
                rec_path = run / "recalibration.json"
                if rec_path.is_file():
                    rec = json.loads(rec_path.read_text())
                    for k, _ in VAL:
                        vals[k].append(_num(rec["by_split"]["validation"]["new"].get(k)))
                    cuts.append(_num(rec["cutoff_t"]["new"]))
            if not rows:
                continue
            cells = [L, str(len(rows))] + [_fmt([_num(r.get(m)) for r in rows]) for m in TEST[ds]]
            cells += [_fmt(vals[k]) for k, _ in VAL] + [_fmt(cuts)]
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            print(f"\n### {ds.upper()} — rule `{a.arm}` (validation = calibration + audit)\n")
            print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
            print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
