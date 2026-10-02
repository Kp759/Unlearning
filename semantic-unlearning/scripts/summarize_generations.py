#!/usr/bin/env python3
"""One table over all generation runs: what the model says after unlearning.

    python scripts/summarize_generations.py [--root outputs/generations] [--examples 3]

Reads <root>/<dataset>_seed<S>.jsonl (from generate_after_unlearning.py) and sums
over seeds, per dataset, arm and prompt group:
  answer      the true answer appears in the continuation (forget: want 0, retain: want all)
  abstains    the continuation says "I don't know" / "unknown" / ...
  changed     the continuation differs from the base model's
With --examples N it also prints N forget prompts per dataset with every arm's output.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from generate_after_unlearning import abstains

GROUPS = ("rewrite", "paraphrase", "atomic_gen", "neighborhood", "retain")


def short(text, n=70):
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 1] + "…"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="outputs/generations")
    p.add_argument("--examples", type=int, default=3)
    a = p.parse_args(argv)
    files = sorted(Path(a.root).glob("*_seed*.jsonl"))
    if not files:
        print(f"no generation files under {a.root}")
        return 1
    totals = defaultdict(Counter)        # (ds, label, group) -> counts
    seeds = defaultdict(set)
    examples = defaultdict(list)
    for f in files:
        lines = f.read_text().splitlines()
        meta = json.loads(lines[0])["meta"]
        ds = meta["dataset"]
        seeds[ds].add(meta["seed"])
        for line in lines[1:]:
            r = json.loads(line)
            g = r["group"]
            base = totals[(ds, "base", g)]
            base["prompts"] += 1
            base["answer"] += r["base_has_answer"]
            base["abstains"] += abstains(r["base_output"])
            for label, run in r["runs"].items():
                c = totals[(ds, label, g)]
                c["prompts"] += 1
                c["answer"] += run["has_answer"]
                c["abstains"] += run.get("abstains", False)
                c["changed"] += run["output"].strip() != r["base_output"].strip()
            if g in ("rewrite", "paraphrase", "atomic_gen") and r["base_has_answer"] \
                    and len(examples[ds]) < a.examples:
                examples[ds].append(r)

    for ds in sorted(seeds):
        labels = ["base"] + sorted({k[1] for k in totals if k[0] == ds and k[1] != "base"},
                                   key=lambda x: ("shipped", "joint", "joint_idk", "joint_noidk").index(x)
                                   if x in ("shipped", "joint", "joint_idk", "joint_noidk") else 9)
        print(f"\n### {ds.upper()} (seeds {', '.join(map(str, sorted(seeds[ds])))})\n")
        print("| group | arm | prompts | true answer in output | abstains | output changed vs base |")
        print("|---|---|---|---|---|---|")
        for g in GROUPS:
            for label in labels:
                c = totals.get((ds, label, g))
                if not c:
                    continue
                n = c["prompts"]
                pct = lambda k: f"{c[k]} ({100 * c[k] / n:.0f}%)"
                changed = "–" if label == "base" else pct("changed")
                print(f"| {g} | {label} | {n} | {pct('answer')} | {pct('abstains')} | {changed} |")
        for r in examples[ds]:
            print(f"\n- **[{r['group']}]** {short(r['prompt'], 90)}  (true: {r['answer']})")
            print(f"  - base: {short(r['base_output'])}")
            for label, run in r["runs"].items():
                print(f"  - {label}: {short(run['output'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
