#!/usr/bin/env python3
"""Model outputs side by side: base vs exact-match router vs v2.1 factorized router.

Reads existing results only (no GPU):
  outputs/mcf_heldout_generation_seed1/{base,A,B}.json     held-out generation (24 tokens)
  outputs/mcf_subject_collateral_seed1/{A,B}.json          same subject, other relations

    python scripts/show_outputs_e2e.py                     (A = E_L16_query_logit, B = ..._v21S8)
    python scripts/show_outputs_e2e.py --n 6 --a E_L16_query_logit --b E_L16_query_logit_v21S8
Sections: forget paraphrases, MCF generation prompts, neighbourhood prompts (other
subjects), every leak or neighbour fire of B, and same-subject other-relation prompts
where the two routers disagree.
"""
import argparse
import json
from pathlib import Path
import random


def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.is_file() else None


def clip(text, n=110):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + "…"


def tag(row):
    if row is None:
        return "(missing)"
    bits = ["routed" if row.get("routed") else "not routed"]
    if row.get("answer_in_continuation") and not row.get("answer_in_prompt"):
        bits.append("LEAK" if row.get("set") != "neighborhood" else "answer kept")
    return ", ".join(bits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="E_L16_query_logit", help="exact-match router arm")
    ap.add_argument("--b", default="E_L16_query_logit_v21S8", help="v2.1 router arm")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--b-label", default=None, help="display name of --b (default: v2.x from its tag)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import re as _re
    m = _re.search(r"_v(\d)(\d)S\d+$", args.b)
    blabel = args.b_label or (f"v{m.group(1)}.{m.group(2)}" if m else "learned")
    held = Path("outputs/mcf_heldout_generation_seed1")
    coll = Path("outputs/mcf_subject_collateral_seed1")
    runs = {"base": load(held / "base.json"), "exact": load(held / f"{args.a}.json"),
            blabel: load(held / f"{args.b}.json")}
    for name, r in runs.items():
        if r is None:
            print(f"missing held-out file for {name}")
    rows = {name: {x["id"]: x for x in r["examples"]} for name, r in runs.items() if r}
    ids = list(rows.get(blabel, rows.get("exact", {})).keys())
    rng = random.Random(args.seed)

    def show(i):
        ref = next(r[i] for r in rows.values() if i in r)
        print(f"\nPROMPT [{ref['set']}] {clip(ref['prompt'], 140)}")
        print(f"  true answer: {ref['answer']}")
        for name in ("base", "exact", blabel):
            r = rows.get(name, {}).get(i)
            if r is not None:
                print(f"  {name:>5} ({tag(r)}): {clip(r['continuation'])}")

    for s, title in (("paraphrase", "FORGET: held-out paraphrases"),
                     ("generation", "FORGET: MCF generation prompts (same subject, relation-related)"),
                     ("neighborhood", "LOCALITY: neighbourhood prompts (other subjects, same relation; answer SHOULD stay)")):
        pool = [i for i in ids if i.split(":")[1] == s]
        print("\n" + "=" * 100 + f"\n{title}  ({args.n} random of {len(pool)})")
        for i in rng.sample(pool, min(args.n, len(pool))):
            show(i)

    b = rows.get(blabel, {})
    leaks = [i for i, r in b.items() if r["set"] != "neighborhood" and r["answer_in_continuation"]
             and not r["answer_in_prompt"]]
    fires = [i for i, r in b.items() if r["set"] == "neighborhood" and r["routed"]]
    print("\n" + "=" * 100 + f"\n{blabel}: EVERY LEAK ({len(leaks)}) and EVERY NEIGHBOUR FIRE ({len(fires)})")
    for i in leaks + fires:
        show(i)

    ca, cb = load(coll / f"{args.a}.json"), load(coll / f"{args.b}.json")
    if ca and cb:
        pairs = list(zip(ca["rows"], cb["rows"]))
        diff = [(x, y) for x, y in pairs if x["natural"]["fired"] != y["natural"]["fired"]]
        print("\n" + "=" * 100 + "\nSAME SUBJECT, OTHER RELATION (natural routing): routers disagree on "
              f"{len(diff)} of {len(pairs)}; showing {min(args.n * 2, len(diff))}")
        for x, y in rng.sample(diff, min(args.n * 2, len(diff))):
            print(f"\nPROMPT [{x['relation']}] {x['prompt']}   (forgotten fact: {x['fact']})")
            print(f"   base: {clip(x['base'])}")
            print(f"  exact ({'FIRED' if x['natural']['fired'] else 'silent'}): {clip(x['natural']['text'])}")
            print(f"  {blabel:>5} ({'FIRED' if y['natural']['fired'] else 'silent'}): {clip(y['natural']['text'])}")


if __name__ == "__main__":
    main()
