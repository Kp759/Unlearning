#!/usr/bin/env python3
"""Official MCF metrics (answer-probability v2) of several runs side by side.

    python scripts/compare_official_mcf.py NAME=RUN_DIR [NAME=RUN_DIR ...]
Reads RUN_DIR/official_mcf_eval.json (evaluate_static_overlap_fact_association_embeddings_official.py).
"""
import json
from pathlib import Path
import sys


def main(argv):
    cols = ("Eff", "Gen", "Spe", "ReleasedAccuracy_Eff", "ReleasedAccuracy_Gen")
    print("| run | forget Eff ↓ | forget Gen ↓ | forget Spe ↑ | released acc Eff ↓ | released acc Gen ↓ "
          "| retain Eff | retain Gen | retain Spe | PPL ↓ | runtime router |")
    print("|---" * 11 + "|")
    for arg in argv:
        name, run = arg.split("=", 1)
        path = Path(run) / "official_mcf_eval.json"
        if not path.is_file():
            print(f"| {name} | (no {path}) |")
            continue
        r = json.loads(path.read_text())
        f, t = r["forget"], r["retain"]
        bank = r.get("fact_association_embedding_bank", {})
        router = (bank.get("factorized_router") and "factorized") or "exact-match linear"
        cells = [f"{float(f[c]):.3f}" for c in cols]
        cells += [f"{float(t[c]):.3f}" for c in ("Eff", "Gen", "Spe")]
        ppl = r.get("forget_PPL")
        cells.append("–" if ppl is None else f"{float(ppl):.2f}")
        print(f"| {name} | " + " | ".join(cells) + f" | {router} |")


if __name__ == "__main__":
    main(sys.argv[1:])
