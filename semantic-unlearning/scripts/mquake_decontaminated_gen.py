#!/usr/bin/env python3
"""MQuAKE Gen (held-out atomic questions) without the questions the training already saw.

The ours router and the Δe rows train on base-model rewrites of each fact's prompt
(router_rewrites.json -> the router's association_examples.json, read by
train_mcf_jlens_feedback_rows.py). MQuAKE's atomic questions are template questions ("Which
city was Steve Jobs born in?"), so a rewrite can match one word for word: on seed 1, 18 of 127
held-out questions are training positives of their own fact, 12 of them among the rows'
training views. This recomputes the paper's MQuAKE Gen (atomic macro, as
mquake_zero_unlearn_official_eval.summarize_atomic_split) per seed on all questions, on the
questions no training example matches ("clean"), and on the matched ones, from the existing
official_mquake_eval.json files. No model is run.

    python scripts/mquake_decontaminated_gen.py
    python scripts/mquake_decontaminated_gen.py --ours-root outputs/l16_ours_v1 \
        --base-root outputs/l16_paper_v1 --router-tag multiseed_ours_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def norm(text):
    return " ".join(str(text).split()).casefold()


def training_prompts(router_dir):
    """Every prompt in the router's training-visible examples (any fact, any split, any label):
    the pool the router fits on and the Δe rows draw their training views from."""
    path = Path(router_dir) / "association_examples.json"
    if not path.is_file():
        raise FileNotFoundError(f"{path} (the router's training examples)")
    return {norm(e["prompt"]) for e in json.loads(path.read_text()) if e.get("prompt")}


def atomic_rows(eval_json):
    """{atomic case id: (question, [correct per answer token])} from forget_raw."""
    d = json.loads(Path(eval_json).read_text())
    rows = d.get("forget_raw")
    if rows is None:
        raise KeyError(f"{eval_json} has no forget_raw rows")
    out = {}
    for r in rows:
        if r.get("prompt_type") != "atomic_gen":
            continue
        cid = int(r["case_id"])
        q, toks = out.get(cid, (None, []))
        if int(r.get("token_index", 0)) == 0:
            q = r["prompt"]
        toks.append(bool(r["correct"]))
        out[cid] = (q, toks)
    reported = (d.get("forget") or {}).get("AtomicGen")
    if reported is None:
        reported = (d.get("summary") or {}).get("forget_AtomicGen")
    return out, reported


def macro(cases):
    vals = [sum(t) / len(t) for _, t in cases if t]
    return 100.0 * sum(vals) / len(vals) if vals else None


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "–"
    m = sum(xs) / len(xs)
    s = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return f"{m:.2f} ± {s:.2f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ours-root", default="outputs/l16_ours_v1", help="ours / genie runs (<root>/mquake/seed*/)")
    ap.add_argument("--base-root", default="outputs/l16_paper_v1", help="base runs (<root>/mquake/seed*/base)")
    ap.add_argument("--router-tag", default="multiseed_ours_v1",
                    help="routers at outputs/mquake_<tag>/seed<S>/L<layer>/router")
    ap.add_argument("--layer", type=int, default=19)
    ap.add_argument("--out", default=None, help="default: <ours-root>/mquake_decontaminated_gen.md")
    a = ap.parse_args(argv)
    seeds = sorted(p.name for p in Path(a.ours_root, "mquake").glob("seed*"))
    if not seeds:
        raise SystemExit(f"no seeds under {a.ours_root}/mquake")
    per = {m: {"all": [], "clean": [], "seen": []} for m in ("base", "ours", "genie")}
    lines = ["| seed | questions | seen in training | method | Gen, all | Gen, clean | Gen, seen | reported |",
             "|---|---|---|---|---|---|---|---|"]
    for seed in seeds:
        router = Path(f"outputs/mquake_{a.router_tag}/{seed}/L{a.layer:02d}/router")
        seen_prompts = training_prompts(router)
        runs = {"base": Path(a.base_root, "mquake", seed, "base"),
                "ours": Path(a.ours_root, "mquake", seed, "ours"),
                "genie": Path(a.ours_root, "mquake", seed, "genie")}
        for method, run in runs.items():
            path = run / "official_mquake_eval.json"
            if not path.is_file():
                lines.append(f"| {seed} | | | {method} | missing {path} | | | |")
                continue
            cases, reported = atomic_rows(path)
            clean = [c for c in cases.values() if c[0] is not None and norm(c[0]) not in seen_prompts]
            seen = [c for c in cases.values() if c[0] is not None and norm(c[0]) in seen_prompts]
            vals = {"all": macro(cases.values()), "clean": macro(clean), "seen": macro(seen)}
            for k, v in vals.items():
                per[method][k].append(v)
            f = lambda x: "–" if x is None else f"{x:.2f}"
            rep = "–" if reported is None else (f"{100 * reported:.2f}" if reported <= 1 else f"{reported:.2f}")
            lines.append(f"| {seed} | {len(cases)} | {len(seen)} | {method} | {f(vals['all'])} | "
                         f"{f(vals['clean'])} | {f(vals['seen'])} | {rep} |")
    md = ["# MQuAKE Gen without the held-out questions seen in training", "",
          "Gen = held-out atomic-question accuracy (%, atomic macro, the paper's MQuAKE Gen). "
          "Clean: questions no training example of the router (and so of the Δe rows) matches "
          "word for word. Seen: the matched ones. Reported: the evaluator's own AtomicGen (must equal "
          "'all').", "",
          "| method | Gen, all | Gen, clean | Gen, seen |", "|---|---|---|---|"]
    for m in ("base", "ours", "genie"):
        md.append(f"| {m} | {_ms(per[m]['all'])} | {_ms(per[m]['clean'])} | {_ms(per[m]['seen'])} |")
    md += ["", "Per seed:", ""] + lines
    out = Path(a.out or Path(a.ours_root) / "mquake_decontaminated_gen.md")
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
