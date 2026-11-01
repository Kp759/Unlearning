#!/usr/bin/env python3
"""Router before / after the two fixes (router_fix.slurm), seed 1, from the diagnostics' metrics.

    python scripts/router_fix_compare.py           # outputs/router_diag_ours_v1 vs .._ours_fix_v1

Per dataset: routing errors split into gate / classifier (router_diag_explain.attribution),
recall, F1, false-positive rate on retain prompts and on prompts with no association, and the
misses per probe group, so the effect on MCF paraphrases and zsRE / MQuAKE questions is visible.
Writes <after root>/COMPARISON.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from router_diag_explain import NAMES, attribution  # noqa: E402


def _load(root, ds, seed):
    p = Path(root) / ds / f"seed{seed}" / "report" / "metrics.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", default="outputs/router_diag_ours_v1")
    ap.add_argument("--after", default="outputs/router_diag_ours_fix_v1")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    md = ["# Router fixes: before vs after (seed 1)", "",
          "Fix 1: unrelated-sentence copies of each fact's positives and negatives. Fix 2: rewrite coverage "
          "floor (third round of plain questions; yes/no judge where the answer-transfer check cannot be used).", "",
          "| dataset | | errors | gate | classifier | recall | F1 | FPR (retain) | FPR (no association) |",
          "|---|---|---|---|---|---|---|---|---|"]
    groups = []
    for ds in ("mcf", "zsre", "mquake", "rwku"):
        rows = {"before": _load(a.before, ds, a.seed), "after": _load(a.after, ds, a.seed)}
        if rows["after"] is None:
            continue
        for label, m in rows.items():
            if m is None:
                continue
            att = attribution(m.get("reasons"))
            al = m["eval"]["by_split"]["all"]
            null = ((m.get("null") or {}).get("deployment_fpr") or {}).get("rate")
            md.append(f"| {NAMES[ds]} | {label} | {sum(att.values())} | {att['gate']} | {att['classifier']} | "
                      f"{_f(al['recall'])} | {_f(al['f1'])} | {_f(al['fpr'])} | {_f(null)} |")
            for g, v in (m["eval"].get("by_group") or {}).items():
                if v.get("fn") or v.get("fp"):
                    groups.append((ds, g, label, v["n"], v["fn"], v["fp"]))
    md += ["", "## Misses and false fires per probe group", "",
           "| dataset | group | | n | missed | false fires |", "|---|---|---|---|---|---|"]
    for ds, g, label, n, fn, fp in sorted(groups, key=lambda x: (x[0], x[1], x[2] != "before")):
        md.append(f"| {NAMES[ds]} | {g} | {label} | {n} | {fn} | {fp} |")
    out = Path(a.after) / "COMPARISON.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
