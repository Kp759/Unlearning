#!/usr/bin/env python3
"""Where does MCF's same-subject false fire come from? (CPU only)

Each held-out (calibration + audit) negative control is a forget subject put
into another relation's prompt. This groups them by (forgotten relation ->
donor relation) and reports how often the shipped router fired on each pair,
using the routing recorded in linear_router_dataset.json. If the fires sit on
near-duplicate relation pairs (e.g. citizenship vs country of origin), part of
the "false" fire is the router correctly recognising the same question.

    python scripts/mcf_negative_relation_pairs.py
    python scripts/mcf_negative_relation_pairs.py --sweep outputs/mcf_multiseed_regular_v1 --layer 23 --top 25
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sweep", default="outputs/mcf_multiseed_regular_v1")
    p.add_argument("--layer", default="19")
    p.add_argument("--top", type=int, default=20)
    a = p.parse_args(argv)
    import torch

    L = f"L{int(a.layer):02d}"
    pairs = defaultdict(lambda: [0, 0])          # (fact relation, donor relation) -> [n, fired]
    by_fact_rel = defaultdict(lambda: [0, 0])
    total = [0, 0]
    names = {}
    seeds = 0
    for router in sorted(Path(a.sweep).glob(f"seed*/{L}/router")):
        data = router / "linear_router_dataset.json"
        if not data.exists():
            continue
        seeds += 1
        facts = torch.load(router / "fact_association_embeddings.pt", map_location="cpu",
                           weights_only=False)["facts"]
        relation = {str(f["id"]): str(f.get("relation")) for f in facts}
        for r in json.loads(data.read_text()):
            if r.get("kind") == "positive" or r.get("split") not in ("calibration", "audit"):
                continue
            fired = r.get("linear_routes_to") is not None
            for fid in r.get("negative_for") or []:
                key = (relation.get(str(fid), "?"), str(r.get("donor_relation")))
                pairs[key][0] += 1
                pairs[key][1] += fired
                by_fact_rel[key[0]][0] += 1
                by_fact_rel[key[0]][1] += fired
                total[0] += 1
                total[1] += fired
    if not total[0]:
        print("no held-out negatives found under", a.sweep, L)
        return 0
    try:
        from mcf_shadow_relation_prompts import RELATION_NOUN_PHRASES as names  # labels
    except Exception:
        names = {}
    lab = lambda rel: f"{rel} ({names[rel]})" if rel in names else rel
    print(f"MCF {L}, {seeds} seeds: {total[0]} held-out same-subject negatives, "
          f"{total[1]} fired at the shipped cutoff ({total[1] / total[0]:.1%})\n")
    print(f"Top {a.top} (forgotten relation -> donor relation) pairs by number of fires:\n")
    print("| forgotten relation | donor relation (the negative's question) | negatives | fired | rate |\n|---|---|---|---|---|")
    for (fr, dr), (n, k) in sorted(pairs.items(), key=lambda kv: -kv[1][1])[:a.top]:
        print(f"| {lab(fr)} | {lab(dr)} | {n} | {k} | {k / n:.0%} |")
    print("\nFalse fire by forgotten relation:\n\n| forgotten relation | negatives | fired | rate |\n|---|---|---|---|")
    for fr, (n, k) in sorted(by_fact_rel.items(), key=lambda kv: -kv[1][1] / max(kv[1][0], 1)):
        print(f"| {lab(fr)} | {n} | {k} | {k / n:.0%} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
