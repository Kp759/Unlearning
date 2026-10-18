#!/usr/bin/env python3
"""Unlearned-model outputs side by side across write layers (CPU, reads saved results only).

    python scripts/show_outputs_layers.py                      (v22 S6, layers 12 16 19, 5 per set)
    python scripts/show_outputs_layers.py --layers 16 19 --n 10 --seed 3
    python scripts/show_outputs_layers.py --sets paraphrase --n 20 > outputs_layers.txt

Per prompt: true answer, base continuation, then each arm's continuation (24 greedy tokens),
tagged routed / LEAK / answer kept. Sections: held-out paraphrases, rewrites, MCF generation
prompts, neighbourhood (other subjects; answer should stay), every leak of any arm, and the
same-subject other-relation probes with the edit forced on (collateral of the vector itself).
"""
import argparse
import json
from pathlib import Path
import random

HELD = Path("outputs/mcf_heldout_generation_seed1")
COLL = Path("outputs/mcf_subject_collateral_seed1")
TITLES = {"paraphrase": "FORGET: held-out paraphrases",
          "rewrite": "FORGET: MCF rewrite prompts (canonical question)",
          "generation": "FORGET: MCF generation prompts (same subject, relation-related)",
          "neighborhood": "LOCALITY: neighbourhood prompts (other subjects; answer SHOULD stay)"}


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def clip(text, n=120):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def tag(r):
    bits = ["routed" if r.get("routed") else "not routed"]
    if r.get("answer_in_continuation") and not r.get("answer_in_prompt"):
        bits.append("answer kept" if r["set"] == "neighborhood" else "LEAK")
    return ", ".join(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="v22")
    ap.add_argument("--subject-layer", type=int, default=6)
    ap.add_argument("--layers", type=int, nargs="+", default=[12, 16, 19])
    ap.add_argument("--sets", nargs="+", default=["paraphrase", "rewrite", "generation", "neighborhood"])
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    arms = {"base": "base"} | {f"L{w}": f"E_L{w}_query_logit_{a.tag}S{a.subject_layer}" for w in a.layers}
    rows = {}
    for label, name in arms.items():
        d = load(HELD / f"{name}.json")
        if d is None:
            print(f"(missing {HELD / name}.json)")
            continue
        rows[label] = {x["id"]: x for x in d["examples"]}
    if "base" not in rows or len(rows) < 2:
        raise SystemExit("need base and at least one arm")
    ref_arm = next(k for k in rows if k != "base")
    ids = list(rows[ref_arm])
    rng = random.Random(a.seed)
    width = max(len(k) for k in rows)

    def show(i):
        ref = rows[ref_arm][i]
        print(f"\nPROMPT [{ref['set']}] {clip(ref['prompt'], 150)}")
        print(f"  true answer: {ref['answer']}")
        for label, r in rows.items():
            if i in r:
                print(f"  {label:>{width}} ({tag(r[i])}): {clip(r[i]['continuation'])}")

    for s in a.sets:
        pool = [i for i in ids if rows[ref_arm][i]["set"] == s]
        if s != "neighborhood":   # show routed ones: those are the prompts the edit acts on
            pool = [i for i in pool if rows[ref_arm][i]["routed"]] or pool
        print("\n" + "=" * 100 + f"\n{TITLES.get(s, s)}  ({min(a.n, len(pool))} random of {len(pool)})")
        for i in rng.sample(pool, min(a.n, len(pool))):
            show(i)

    leaks = sorted({i for k, r in rows.items() if k != "base" for i, x in r.items()
                    if x["set"] != "neighborhood" and x["answer_in_continuation"] and not x["answer_in_prompt"]})
    print("\n" + "=" * 100 + f"\nEVERY LEAK OF ANY ARM ({len(leaks)})")
    for i in leaks:
        show(i)

    coll = {f"L{w}": load(COLL / f"E_L{w}_query_logit_{a.tag}S{a.subject_layer}.json") for w in a.layers}
    coll = {k: v for k, v in coll.items() if v}
    if coll:
        first = next(iter(coll.values()))["rows"]
        pick = rng.sample(range(len(first)), min(a.n, len(first)))
        print("\n" + "=" * 100 + "\nSAME SUBJECT, OTHER RELATION, edit forced on (what the vector does to the "
              f"subject's other facts)  ({len(pick)} random of {len(first)})")
        for k in pick:
            x = first[k]
            print(f"\nPROMPT [{x['relation']}] {x['prompt']}   (forgotten fact: {x['fact']})")
            print(f"  {'base':>{width}}: {clip(x['base'])}")
            for label, c in coll.items():
                print(f"  {label:>{width}}: {clip(c['rows'][k]['forced']['text'])}")


if __name__ == "__main__":
    main()
