#!/usr/bin/env python3
"""Calibrated bias (folded) vs raw bias (plain logistic, p >= 0.5), mean ± std over seeds.

    python scripts/summarize_bias_rule_seeds.py
    python scripts/summarize_bias_rule_seeds.py --roots outputs/bias_rule_ablation_v1 \
        outputs/bias_rule_ablation_reworded_v2 --optimizer lbfgs

Reads <root>/<optimizer>/<dataset>/seed*/L??/comparison.json (run_bias_rule_ablation_one.sh).
Per dataset and layer, two blocks:
  test       official metrics of the folded run and of raw_swap (same rows, raw router)
  validation router recall (correct route) and same-subject false activation on the
             calibration (validation) and audit splits, folded vs raw
Rows whose raw_swap official eval is missing are counted in `missing` and left out.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

TEST = {
    "mcf": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "display_zero"],
    "zsre": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen",
             "forget_paraphrase_route_active", "forget_neighborhood_route_active"],
    "mquake": ["forget_Eff", "forget_AtomicGen", "retain_Eff", "retain_AtomicGen",
               "forget_atomicgen_route_correct", "retain_atomicgen_route_active"],
}
SPLITS = ("calibration", "audit")


def _num(v):
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _fmt(vals, digits=3):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.{digits}g} ± {sd:.2g}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roots", nargs="+",
                   default=["outputs/bias_rule_ablation_v1", "outputs/bias_rule_ablation_reworded_v2"])
    p.add_argument("--optimizer", default="lbfgs")
    p.add_argument("--datasets", nargs="+", default=["mcf", "zsre", "mquake"])
    a = p.parse_args(argv)

    groups = defaultdict(list)
    for root in a.roots:
        for f in sorted(Path(root, a.optimizer).glob("*/seed*/L??/comparison.json")):
            r = json.loads(f.read_text())
            ds = r.get("dataset") or f.parts[-4]
            if ds not in a.datasets:
                continue
            groups[(ds, f.parent.name, Path(root).name)].append((f.parts[-3], r))

    for (ds, layer, root), items in sorted(groups.items()):
        seeds = sorted(s for s, _ in items)
        test = defaultdict(lambda: defaultdict(list))
        val = defaultdict(lambda: defaultdict(list))
        cutoffs, missing = [], 0
        for _, r in items:
            status = r.get("run_status") or {}
            if status.get("raw_swap") != "complete":
                missing += 1
                continue
            for row in r.get("metrics", []):
                if row["metric"] in TEST[ds]:
                    test[row["metric"]]["folded"].append(_num(row.get("folded")))
                    test[row["metric"]]["raw"].append(_num(row.get("raw_swap")))
            router = r.get("router") or {}
            t = router.get("folded_cutoff_t")
            cutoffs.append(_num(t) if not isinstance(t, list) else None)
            for split in SPLITS:
                s = (router.get("by_split") or {}).get(split) or {}
                for key in ("correct_route", "false_activation"):
                    for arm in ("folded", "raw"):
                        val[f"{split} {key}"][arm].append(_num((s.get(key) or {}).get(arm)))
        n = len(items) - missing
        print(f"\n### {ds.upper()} {layer}  ({root}; seeds {', '.join(seeds)}; "
              f"{n} with raw eval{'; ' + str(missing) + ' missing' if missing else ''})")
        print(f"calibrated cutoff t that raw drops: {_fmt(cutoffs)} "
              "(negative = calibration made the router fire MORE than raw)\n")
        print("| metric (test, official) | calibrated bias | raw bias (p ≥ 0.5) |\n|---|---|---|")
        for m in TEST[ds]:
            if m in test:
                print(f"| {m} | {_fmt(test[m]['folded'])} | {_fmt(test[m]['raw'])} |")
        print("\n| router, validation / audit | calibrated bias | raw bias (p ≥ 0.5) |\n|---|---|---|")
        for k in sorted(val):
            print(f"| {k} | {_fmt(val[k]['folded'])} | {_fmt(val[k]['raw'])} |")
    if not groups:
        print("no comparison.json found under", a.roots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
