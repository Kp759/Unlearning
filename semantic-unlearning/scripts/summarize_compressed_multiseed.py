#!/usr/bin/env python3
"""Shared-vector banks vs one-vector-per-fact, mean ± std over seeds.

    python scripts/summarize_compressed_multiseed.py [--datasets mcf zsre] [--layer 19]

Rows per dataset (same router, only the values differ):
  shipped     the multiseed sweep's row-wise rows (outputs/<ds>_<REF>/seed*/L<LL>/linear_global)
  full        one vector per fact, joint in-loop trainer (same trainer as the modes below)
  tied_answer facts with the same answer share one vector
  answer_fixed no stored vector (one scalar on the answer token's direction)
Columns: official test metrics, then value storage (floats per fact, shared vectors)
and how many seeds reached the training target.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from summarize_mcf_layer_sweep import collect

METRICS = {
    "mcf": ["forget_Eff", "forget_Gen", "display_zero", "forget_Spe", "retain_Gen", "PPL"],
    "zsre": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Gen", "PPL"],
    "mquake": ["forget_Eff", "forget_AtomicGen", "retain_AtomicGen", "PPL"],
}
REF = {"mcf": "multiseed_regular_v1", "zsre": "multiseed_reworded_v2",
       "mquake": "multiseed_regular_v1"}
MODE_ORDER = ("full", "tied_answer", "answer_fixed")
TARGET_MET = ("global_train_and_development_target_met", "global_target_met")


def _num(v):
    if isinstance(v, bool):
        return float(v)
    return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    mean = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{mean:.3g} ± {sd:.2g}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs")
    p.add_argument("--out-tag", default="compressed_multiseed_v1")
    p.add_argument("--datasets", nargs="+", default=["mcf", "zsre", "mquake"])
    p.add_argument("--layer", default="19")
    p.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    a = p.parse_args(argv)
    root, L = Path(a.root), f"L{int(a.layer):02d}"

    for ds in a.datasets:
        base = root / a.out_tag / ds
        modes = sorted({d.name for s in a.seeds for d in (base / f"seed{s}" / L).glob("*")
                        if d.is_dir() and ".incomplete_" not in d.name},
                       key=lambda m: (MODE_ORDER.index(m) if m in MODE_ORDER else 99, m))
        if not modes:
            continue
        cols = METRICS[ds]
        print(f"\n### {ds.upper()} {L} (seeds {', '.join(a.seeds)})\n")
        print("| values | seeds done | " + " | ".join(cols)
              + " | floats / fact | shared vectors | target met |")
        print("|" + "---|" * (len(cols) + 5))
        arms = [("shipped (row-wise, per fact)", lambda s: root / f"{ds}_{REF[ds]}" / f"seed{s}" / L / "linear_global")]
        arms += [(m, (lambda s, m=m: base / f"seed{s}" / L / m)) for m in modes]
        for label, path_of in arms:
            vals = {c: [] for c in cols}
            done, met, per_fact, shared = 0, 0, set(), set()
            for s in a.seeds:
                run = path_of(s)
                if not run.is_dir():
                    continue
                row = collect(run, label)
                if row.get("status") != "complete":
                    continue
                done += 1
                for c in cols:
                    vals[c].append(_num(row.get(c)))
                rep = run / "training_report.json"
                if rep.is_file():
                    r = json.loads(rep.read_text())
                    met += r.get("stop_reason") in TARGET_MET
                    storage = r.get("value_storage") or {}
                    per_fact.add(storage.get("per_fact_floats"))
                    if storage.get("shared_vectors") is not None:
                        shared.add(round(float(storage["shared_vectors"]), 1))
            if not done:
                continue
            pf = "/".join(str(x) for x in sorted(per_fact, key=str)) if per_fact else "3072"
            sh = "/".join(f"{x:g}" for x in sorted(shared)) if shared else "0"
            tm = f"{met}/{done}" if label != "shipped (row-wise, per fact)" else "–"
            print(f"| {label} | {done} | " + " | ".join(_fmt(vals[c]) for c in cols)
                  + f" | {pf} | {sh} | {tm} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
