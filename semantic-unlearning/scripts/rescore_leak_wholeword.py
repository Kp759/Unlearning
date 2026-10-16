#!/usr/bin/env python3
"""Re-score held-out generation files with WHOLE-WORD answer matching (no GPU).

The headline leak test is a case-insensitive substring match, which counts 'Croatia' in
'Croatian' and 'French' in 'a French writer' answering a different question equally.
This prints, per run, substring vs whole-word rates for the forget sets (leak) and the
neighbourhood set (answer kept), excluding prompts that already contain the answer.

    python scripts/rescore_leak_wholeword.py                       (all runs in the folder)
    python scripts/rescore_leak_wholeword.py base E_L16_query_logit E_L16_query_logit_v21S8
"""
import json
from pathlib import Path
import re
import sys

HELD = Path("outputs/mcf_heldout_generation_seed1")


def word_in(answer, text):
    return re.search(r"(?<!\w)" + re.escape(answer.strip()) + r"(?!\w)", text, re.IGNORECASE) is not None


def rates(rows, s):
    part = [r for r in rows if r["set"] == s and not r["answer_in_prompt"]]
    if not part:
        return None
    sub = sum(r["answer_in_continuation"] for r in part) / len(part)
    word = sum(word_in(r["answer"], r["continuation"]) for r in part) / len(part)
    return sub, word, len(part)


def main(names):
    files = [HELD / f"{n}.json" for n in names] if names else sorted(HELD.glob("*.json"))
    sets = ("rewrite", "paraphrase", "generation", "neighborhood")
    print("| run | " + " | ".join(f"{s} (substring / whole word)" for s in sets) + " |")
    print("|---" * (len(sets) + 1) + "|")
    for f in files:
        if not f.is_file():
            print(f"| {f.stem} | (missing) |")
            continue
        d = json.loads(f.read_text())
        rows = d.get("examples", [])
        cells = []
        for s in sets:
            r = rates(rows, s)
            cells.append("–" if r is None else f"{100 * r[0]:.1f}% / {100 * r[1]:.1f}% (n={r[2]})")
        print(f"| {d.get('label', f.stem)} | " + " | ".join(cells) + " |")
    print("\nforget sets: leak (lower is better); neighbourhood: answer kept (higher is better)")


if __name__ == "__main__":
    main(sys.argv[1:])
