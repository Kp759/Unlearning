#!/usr/bin/env python3
"""Appendix material from baselines_same_subject.py (--groups all four): the per-group table of
every method on the no-association prompts, and an examples table (prompts per group, every
method's greedy answer next to the base model's).

    python scripts/relation_specificity_report.py --dataset mcf --seed 1
    # -> outputs/relation_specificity_v1/<ds>_seed<S>_appendix.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SHORT = {"null_generic": "generic text", "null_ppl_prefix": "perplexity text",
         "null_unseen_subject": "unseen subject", "null_protected_other_relation": "protected subject, other relation"}
ORDER = ["base", "ours", "memit", "zerounlearn", "rome", "ga", "gd", "npo", "rmu"]
LABEL = {"base": "Base", "ours": "SURE (ours)", "memit": "MEMIT", "zerounlearn": "ZeroUnlearn", "rome": "ROME",
         "ga": "GA", "gd": "GD", "npo": "NPO", "rmu": "RMU"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mcf")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default="outputs/relation_specificity_v1")
    ap.add_argument("--probes", default="outputs/router_diag_ours_control_v1/{ds}/seed{seed}/probes_null.json")
    ap.add_argument("--examples", default="Belgium is located in;Ignazio Silone originates from;Luxembourg is affiliated with;"
                                          "Give me a simple recipe for pancakes.;What causes the seasons on Earth?",
                    help="semicolon-separated prompts for the examples table (plus the first unseen-subject prompt)")
    ap.add_argument("--width", type=int, default=90)
    args = ap.parse_args(argv)
    root = Path(args.root)
    src = root / f"{args.dataset}_seed{args.seed}_allgroups.json"
    if not src.is_file():
        src = root / f"{args.dataset}_seed{args.seed}_allgroups.partial.json"
    d = json.loads(src.read_text())
    res, outs = d["results"], d["outputs"]
    probes = json.loads(Path(args.probes.format(ds=args.dataset, seed=args.seed)).read_text())
    methods = [m for m in ORDER if m in res or m == "base"]

    lines = [f"## No-association prompts, {args.dataset} seed {args.seed}: does the unlearned model still give the base "
             f"model's answer?", "",
             "Each prompt is unrelated to the forgotten facts, so the base model's answer is the target. Greedy decoding. "
             "Cells: prompts whose FIRST answer token changed / whose 24-token continuation changed, of the group size. "
             "Last column: mean probability kept for the base model's first answer token (1 = unchanged).", "",
             "| method | " + " | ".join(SHORT[g] for g in SHORT) + " | answer prob. kept |",
             "|---" * (len(SHORT) + 2) + "|"]
    for m in methods:
        if m == "base":
            lines.append("| Base | " + " | ".join("–" for _ in SHORT) + " | 1.00 |")
            continue
        r = res[m]
        bg = r.get("by_group", {})
        cells = []
        for g in SHORT:
            x = bg.get(g)
            cells.append("–" if x is None else f"{x['top1_changed']} / {x['changed']} of {x['of']}")
        lines.append(f"| {LABEL[m]} | " + " | ".join(cells) + f" | {r.get('answer_prob_kept', float('nan')):.2f} |")

    lines += ["", "### Examples (base answer first; a changed first token is marked ✗)", ""]
    wanted = [w.strip() for w in args.examples.split(";") if w.strip()]
    idx = {p.get("prompt"): i for i, p in enumerate(probes) if p.get("prompt")}
    first_unseen = next((p["prompt"] for p in probes if p["group"] == "null_unseen_subject" and p.get("prompt")), None)
    if first_unseen and first_unseen not in wanted:
        wanted.append(first_unseen)
    for w in wanted:
        i = idx.get(w)
        if i is None or "base" not in outs:
            continue
        g = SHORT.get(probes[i]["group"], probes[i]["group"])
        lines += [f"**{w}** ({g})", "", "| method | answer |", "|---|---|"]
        base_first = outs["base"][i].strip().split(" ")[0] if outs["base"][i].strip() else ""
        for m in methods:
            if m not in outs or not outs[m]:
                continue
            a = outs[m][i]
            flag = "" if m == "base" else (" ✗" if (a.strip().split(" ")[0] if a.strip() else "") != base_first else "")
            shown = a.strip().replace("|", "\\|")[:args.width] + ("…" if len(a.strip()) > args.width else "")
            lines.append(f"| {LABEL[m]}{flag} | {shown or '(empty)'} |")
        lines.append("")
    out = root / f"{args.dataset}_seed{args.seed}_appendix.md"
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
