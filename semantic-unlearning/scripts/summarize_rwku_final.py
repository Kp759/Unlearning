#!/usr/bin/env python3
"""One RWKU table for the paper: base model + every routing/training variant, mean ± std over seeds.

    python scripts/summarize_rwku_final.py                    # layers 19 23
    python scripts/summarize_rwku_final.py --layers 1 3 7 13 19 23 27

Recovery % (share of greedy generations that contain the answer): lower is better
on forget sets (Eff = the 50 trained probes, Gen L1/L2 = held-out questions about
the same people, Para = paraphrased held-out L2, L3 = adversarial); Neighbour is
locality (higher is better). Fire rates: share of held-out forget questions /
neighbour questions on which a row is injected. "rows ok" = rows whose worst
sensitive-token probability reached the 1e-6 training target (of 50).
Only cells with every listed seed complete are shown; `n` says how many.
"""
from __future__ import annotations

import argparse
import math
import statistics as st
from pathlib import Path

from summarize_mcf_layer_sweep import collect

CONFIGS = [
    ("98% recall (calibration split)", "regular", "rwku_multiseed_regular_v1/{seed}/{L}/linear_global"),
    ("98% recall, genie rows", "genie", "rwku_multiseed_genie_v1/{seed}/{L}/linear_global"),
    ("98% recall (merged validation, per fact)", "regular", "calibration_rules_v1/rwku/{seed}/{L}/recall0.98_fact"),
    ("subject gate", "regular", "rwku_multiseed_subject_v1/{seed}/{L}/linear_global"),
    ("subject gate, genie rows", "genie", "rwku_multiseed_subject_genie_v1/{seed}/{L}/linear_global"),
]
COLS = [("forget_Eff", "Eff ↓"), ("forget_GenL1", "Gen L1 ↓"), ("forget_GenL2", "Gen L2 ↓"),
        ("forget_GenPara", "Para ↓"), ("forget_L3", "L3 ↓"), ("neighbor", "Neighbour ↑"),
        ("PPL", "PPL"), ("heldout_route_active", "fire: held-out"),
        ("neighbor_route_active", "fire: neighbour"), ("facts_converged", "rows ok")]


def _num(v):
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _cell(rows, key, pct=False):
    vals = [_num(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    if pct:
        vals = [100 * v for v in vals]
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.1f} ± {sd:.1f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs")
    p.add_argument("--layers", nargs="+", default=["19", "23"])
    p.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    a = p.parse_args(argv)
    root = Path(a.root)
    seeds = [f"seed{s}" for s in a.seeds]

    head = ["config", "layer", "n"] + [name for _, name in COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    base = [collect(root / "rwku_multiseed_base_v1" / s, s) for s in seeds
            if (root / "rwku_multiseed_base_v1" / s / "official_rwku_eval.json").is_file()]
    if base:
        cells = ["base model (no unlearning)", "–", str(len(base))]
        cells += [_cell(base, k) if k in ("forget_Eff", "forget_GenL1", "forget_GenL2", "forget_GenPara",
                                         "forget_L3", "neighbor", "PPL") else "–" for k, _ in COLS]
        lines.append("| " + " | ".join(cells) + " |")
    missing = []
    for label, _, pattern in CONFIGS:
        for layer in a.layers:
            L = f"L{int(layer):02d}"
            rows = []
            for s in seeds:
                run = root / pattern.format(seed=s, L=L)
                row = collect(run, s) if run.is_dir() else {}
                if row.get("status") == "complete":
                    rows.append(row)
                else:
                    missing.append(f"{label} {L} {s}")
            if not rows:
                continue
            cells = [label, L, str(len(rows))]
            for key, _ in COLS:
                cells.append(_cell(rows, key, pct=key.endswith("route_active")))
            lines.append("| " + " | ".join(cells) + " |")
    print("\n### RWKU — final table (recovery %, mean ± std over seeds)\n")
    print("\n".join(lines))
    if missing:
        print(f"\n{len(missing)} runs not complete yet (left out of the means), e.g.: "
              + "; ".join(missing[:6]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
