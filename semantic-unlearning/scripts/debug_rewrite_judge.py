#!/usr/bin/env python3
"""What the same-fact judge (router_rewrites.same_fact_scores) says about real rewrites.

For a few facts of a SURE prep dir: the base model's paraphrase candidates (the router_rewrites
prompts and form checks), each with the judge's P(Yes), its greedy reply, and the verdict under
the absolute rule (P(yes) >= 0.5) and the calibrated rule (router_rewrites.judge_cutoffs: halfway
between the fact's own prompt against itself and against its subject transplanted into another
fact's question). Also prints the drop counters of an existing router_rewrites.stats.json.
Read-only: writes nothing.

    python -u scripts/debug_rewrite_judge.py --prep-dir outputs/zsre_multiseed_ours_v1/seed1/L19/prep \
        --n-facts 6 --device cuda --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import router_rewrites as RR  # noqa: E402
from build_router_rewrites import dataset_of, targets_of  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--n-facts", type=int, default=6)
    p.add_argument("--n-para", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, help="default: bfloat16 on cuda, float32 otherwise")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    prep = Path(args.prep_dir)
    stats_path = prep / "router_rewrites.stats.json"
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        print("existing stats:", json.dumps({k: v for k, v in stats.items()
                                             if k.startswith(("dropped", "candidates", "kept", "meaning"))}))
    source = torch.load(prep / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((prep / "association_manifest.json").read_text())
    facts = list(source["facts"])
    donors = targets_of(facts, dataset_of(facts))
    targets = donors[:args.n_facts]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(manifest["model_path"], dtype=dtype,
                                                 local_files_only=args.local_files_only).to(args.device).eval()

    cutoffs = RR.judge_cutoffs(model, tok, targets, donors)
    asks = [RR.instruct(tok, RR._paraphrase_ask(t, args.n_para, False)) for t in targets]
    outs = RR.generate_texts(model, tok, asks, 40 + 28 * args.n_para)
    rows = []        # (target, label, candidate text, fmt)
    for t, o in zip(targets, outs):
        rows.append((t, "SAME (own prompt)", t.text, t.fmt))
        other = RR.transplant(t, donors)
        if other:
            rows.append((t, "TRANSPLANT (other question)", other, t.fmt))
        for line in RR.lines(o):
            line = line.rstrip(" .…:;,") if t.fmt == "cloze" else line
            ok = RR.well_formed(line, "paraphrase") and RR.has_phrase(line, t.subject) and not RR.has_phrase(line, t.answer)
            rows.append((t, "candidate" if ok else "candidate (fails form)", line, t.fmt))
    pairs = [(RR.judge_side(t.text, t.fmt), RR.judge_side(text, fmt), t.subject) for t, _, text, fmt in rows]
    scores = RR.same_fact_scores(model, tok, pairs)
    replies = RR.generate_texts(model, tok, [RR.instruct(tok, RR.JUDGE.format(a=a, b=b, s=s)) for a, b, s in pairs], 16)
    current = None
    for (t, label, text, _), sc, reply in zip(rows, scores, replies):
        cut = cutoffs[t.id][0]
        if t.id != current:
            current = t.id
            print(f"\n=== {t.id} | subject {t.subject!r} | answer {t.answer!r} | calibrated cutoff {cut:.2f}"
                  f"\n    prompt: {t.text}")
        verdict = f"{'KEEP' if sc >= 0.5 else 'drop'} @0.5, {'KEEP' if sc >= cut else 'drop'} @{cut:.2f}"
        print(f"  [{verdict} | P(yes)={sc:.2f}] {label:27s} {text!r}\n      judge says: {reply!r}")
    cands = [(sc, cutoffs[t.id][0]) for (t, label, _, _), sc in zip(rows, scores) if label == "candidate"]
    same = [sc for (_, label, _, _), sc in zip(rows, scores) if label.startswith("SAME")]
    other = [sc for (_, label, _, _), sc in zip(rows, scores) if label.startswith("TRANSPLANT")]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")  # noqa: E731
    print(f"\nP(yes) mean: SAME {mean(same):.2f} | TRANSPLANT {mean(other):.2f} | candidates "
          f"{mean([c[0] for c in cands]):.2f}. Kept: absolute 0.5 {sum(sc >= 0.5 for sc, _ in cands)}/{len(cands)}, "
          f"calibrated {sum(sc >= cut for sc, cut in cands)}/{len(cands)}")


if __name__ == "__main__":
    main()
