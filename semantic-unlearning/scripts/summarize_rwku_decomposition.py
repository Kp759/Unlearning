#!/usr/bin/env python3
"""RWKU evaluation-time genie table (like the paper's Table 6), mean ± std over seeds.

    python scripts/summarize_rwku_decomposition.py                       # subject gate, L19 L23
    python scripts/summarize_rwku_decomposition.py --tag multiseed_regular_v1 --layers 23

Reads outputs/rwku_<tag>/seed*/L??/linear_global/decomposition/rwku_router_decomposition.json
(from evaluate_rwku_router_decomposition.py / rwku_decomposition.slurm). Columns:
  Base     no row injected
  Router   the run's own router (subject gate or 98% threshold)
  Genie    same-50: the exact trained row is forced (E);
           held-out: every row of the same person is tried, the most suppressive
           one is kept, chosen with the answer (S) -- an upper bound no router can reach
  Random   a random row of the same person (held-out only)
All values are generated-answer recovery %, lower is better.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

GROUPS = [
    ("same50", "Trained same-50", "genie_exact", "E"),
    ("heldout_level1", "Held-out Level-1", "genie_subject", "S"),
    ("heldout_level2", "Held-out Level-2", "genie_subject", "S"),
    ("heldout_paraphrase", "Held-out Level-2 paraphrases", "genie_subject", "S"),
    ("neighbors", "Neighbours (higher = better)", None, ""),
]


def _fmt(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "–"
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{st.mean(vals):.1f} ± {sd:.1f}"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs")
    p.add_argument("--tag", default="multiseed_subject_v1")
    p.add_argument("--layers", nargs="+", default=["19", "23"])
    a = p.parse_args(argv)
    for layer in a.layers:
        L = f"L{int(layer):02d}"
        reports = []
        for f in sorted(Path(a.root, f"rwku_{a.tag}").glob(
                f"seed*/{L}/linear_global/decomposition/rwku_router_decomposition.json")):
            reports.append((f.parts[-5], json.loads(f.read_text())))
        if not reports:
            continue

        def rec(rep, arm, group):
            block = rep["summaries"].get(arm, {}).get(group) or {}
            return block.get("recovery_percent") if block.get("count") else None

        seeds = ", ".join(s for s, _ in reports)
        select = sorted({r.get("genie_select") for _, r in reports})
        print(f"\n### RWKU {a.tag} {L} — evaluation-time genie (seeds: {seeds}; "
              f"genie row chosen by {', '.join(map(str, select))})\n")
        print("| Group | n (per seed) | Base | Router | Genie | Random same-person row |")
        print("|---|---|---|---|---|---|")
        for group, label, genie_arm, tag in GROUPS:
            counts = sorted({(r["summaries"].get("base", {}).get(group) or {}).get("count")
                             for _, r in reports} - {None})
            if not counts:
                continue
            base = [rec(r, "base", group) for _, r in reports]
            router = [rec(r, "v2", group) for _, r in reports]
            genie = [rec(r, genie_arm, group) for _, r in reports] if genie_arm else []
            rand = ([rec(r, "genie_subject_random", group) for _, r in reports]
                    if genie_arm == "genie_subject" else [])
            n = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
            genie_cell = f"{_fmt(genie)} ({tag})" if genie_arm else "= base (abstain)"
            print(f"| {label} | {n} | {_fmt(base)} | {_fmt(router)} | {genie_cell} | {_fmt(rand)} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
