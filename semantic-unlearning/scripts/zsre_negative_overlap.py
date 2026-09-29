#!/usr/bin/env python3
"""Are some ZsRE "same-subject" negatives actually the SAME question? (CPU only)

ZsRE has no relation ids (every fact's relation is the same placeholder), so
the router's transplant builder cannot skip donors that ask the same relation:
"Which country is Y located in?" transplanted to subject X becomes a negative
for X's fact "What country is X from?", although it is a paraphrase of it. A
router that generalises better fires on those, and that counts as false fire.

For each router's held-out (calibration + audit) same-subject negatives this
compares the negative's question template (subject, context prefix and stop
words removed) with the fact's own direct question template (content-word
Jaccard; a heuristic LOWER bound: it misses synonyms such as "born" vs
"birthplace", so true same-question contamination is higher) and reports
false fire at the shipped cutoff overall and on the clearly different-relation
negatives only (Jaccard < --max-overlap), using the router's own recorded
routing (`linear_routes_to` in linear_router_dataset.json).

    python scripts/zsre_negative_overlap.py
    python scripts/zsre_negative_overlap.py --examples 15
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from mcf_synthetic_paraphrase_templates import GENERIC_CONTEXT_PREFIXES


STOP = set("""a an the of in on at to for by with from as is are was were be been being do does did
what which who whom whose where when why how that this these those it its his her their
has have had can could would will shall should may might must and or name s""".split())


def template(prompt, subject):
    text = prompt.strip()
    for prefix in GENERIC_CONTEXT_PREFIXES:
        if text.startswith(prefix + " "):
            text = text[len(prefix) + 1:]
            break
    text = re.sub(re.escape(subject), " ", text, flags=re.IGNORECASE)
    return {w for w in re.findall(r"[a-z0-9]+", text.casefold()) if w not in STOP}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--tags", nargs="+", default=["regular_v1", "reworded_v1", "reworded_v2"])
    parser.add_argument("--layers", nargs="+", default=["19", "23"])
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    parser.add_argument("--max-overlap", type=float, default=0.5,
                        help="template Jaccard at/above which a negative counts as a likely "
                             "same-question paraphrase")
    parser.add_argument("--examples", type=int, default=8)
    args = parser.parse_args(argv)
    import torch

    stats = defaultdict(lambda: defaultdict(list))
    examples = []
    for tag in args.tags:
        for layer in args.layers:
            L = f"L{int(layer):02d}"
            for seed in args.seeds:
                router = Path(args.root) / f"zsre_multiseed_{tag}" / f"seed{seed}" / L / "router"
                data = router / "linear_router_dataset.json"
                if not data.exists():
                    continue
                facts = torch.load(router / "fact_association_embeddings.pt", map_location="cpu",
                                   weights_only=False)["facts"]
                by_id = {str(f["id"]): f for f in facts}
                direct = {fid: template(str(f["canonical_prompt"]), str(f["subject"]))
                          for fid, f in by_id.items()}
                rows = [r for r in json.loads(data.read_text())
                        if r.get("kind") != "positive" and r.get("split") in ("calibration", "audit")]
                n = fired = same = same_fired = 0
                for r in rows:
                    for fid in r.get("negative_for") or []:
                        fact = by_id.get(str(fid))
                        if fact is None:
                            continue
                        overlap = jaccard(template(r["prompt"], str(fact["subject"])), direct[str(fid)])
                        fire = r.get("linear_routes_to") is not None
                        n += 1; fired += fire
                        if overlap >= args.max_overlap:
                            same += 1; same_fired += fire
                            if fire and len(examples) < args.examples and tag == args.tags[-1]:
                                examples.append((tag, seed, L, round(overlap, 2),
                                                 fact["canonical_prompt"], r["prompt"]))
                if not n:
                    continue
                s = stats[(tag, L)]
                s["n"].append(n)
                s["likely_same_question"].append(same / n)
                s["ff_all"].append(fired / n)
                s["ff_different"].append((fired - same_fired) / max(1, n - same))
                s["share_of_fires_likely_same"].append(same_fired / max(1, fired))

    fmt = lambda v: f"{st.mean(v) * 100:.1f} ± {(st.stdev(v) if len(v) > 1 else 0) * 100:.1f}"
    head = ["router", "layer", "seeds", "held-out negatives / seed",
            f"likely same question % (Jaccard ≥ {args.max_overlap})",
            "false fire, all %", "false fire, different-relation only %",
            "% of false fires that are likely same question"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for (tag, L), s in sorted(stats.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        print("| " + " | ".join([tag, L, str(len(s["n"])), f"{st.mean(s['n']):.0f}",
                                 fmt(s["likely_same_question"]), fmt(s["ff_all"]),
                                 fmt(s["ff_different"]), fmt(s["share_of_fires_likely_same"])]) + " |")
    if examples:
        print(f"\nFired 'negatives' that look like the same question ({args.tags[-1]}):")
        for tag, seed, L, ov, fact_q, neg in examples:
            print(f"  seed{seed} {L} J={ov}  fact: {fact_q!r}\n{'':20}negative: {neg!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
