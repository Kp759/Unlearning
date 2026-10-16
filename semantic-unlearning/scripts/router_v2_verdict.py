#!/usr/bin/env python3
"""Ship-or-not check for the v2 factorized router against the L27 relation-aware router.

v2 replaces exact-match routing only if, on the held-out routing table
(mcf_router_specificity.py), it keeps paraphrase recall >= --min-recall and matches
exact match on locality: neighbourhood fire <= --max-neighbour-fire and near-miss fire
<= --max-near-miss (exact match: 0% on both by construction). Otherwise ship L27
relation-aware and report typo / case robustness as a limitation.

    python scripts/router_v2_verdict.py SPEC.json --baseline "L27 relation-aware" --candidates "v2 S08" "v2 S12"
Exit 0 = some candidate passes (the best one is printed), 1 = none passes.
"""
from __future__ import annotations

import argparse
import json
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec_json")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidates", nargs="+", required=True)
    p.add_argument("--min-recall", type=float, default=0.89)
    p.add_argument("--max-neighbour-fire", type=float, default=0.01)
    p.add_argument("--max-near-miss", type=float, default=0.02)
    a = p.parse_args(argv)
    res = json.load(open(a.spec_json))

    def get(name, key):
        v = res[name]["summary"].get(key)
        return None if v is None else v[0]

    cols = ("paraphrase_recall", "neighborhood_fire", "near_miss_fire", "subject_eval_fire_other",
            "typo_recall", "lowercase_recall")
    print("| router | " + " | ".join(cols) + " | verdict |")
    print("|---" * (len(cols) + 2) + "|")
    passing = []
    for name in [a.baseline] + a.candidates:
        if name not in res:
            print(f"| {name} | (missing) |")
            continue
        vals = {c: get(name, c) for c in cols}
        ok = (vals["paraphrase_recall"] is not None and vals["paraphrase_recall"] >= a.min_recall
              and vals["neighborhood_fire"] <= a.max_neighbour_fire
              and (vals["near_miss_fire"] or 0.0) <= a.max_near_miss)
        verdict = "baseline" if name == a.baseline else ("PASS" if ok else "fail")
        if ok and name != a.baseline:
            passing.append((vals["subject_eval_fire_other"], -vals["paraphrase_recall"], name))
        print(f"| {name} | " + " | ".join("–" if v is None else f"{100 * v:.1f}%" for v in vals.values())
              + f" | {verdict} |")
    rule = (f"recall >= {a.min_recall:.0%}, neighbour fire <= {a.max_neighbour_fire:.0%}, "
            f"near-miss fire <= {a.max_near_miss:.0%}")
    if passing:
        best = min(passing)[2]
        print(f"\nVERDICT: ship v2 ({best}); it passes {rule}.")
        print(f"BEST={best}")
        return 0
    print(f"\nVERDICT: no v2 layer passes ({rule}). Ship {a.baseline}; report typo / case "
          "robustness as a limitation and v1 + v2 as negative results.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
