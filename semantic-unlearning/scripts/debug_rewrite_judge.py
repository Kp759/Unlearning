#!/usr/bin/env python3
"""What the rewrite validity checks (router_rewrites.py) say about real rewrites.

For a few facts of a SURE prep dir: the base model's paraphrase candidates (the router_rewrites
prompts and form checks), each with
    transfer   the answer-transfer check (the default): told the fact, does the model give its
               answer to the rewrite? Shown with what the model actually answers.
    yes/no     the old judge's P(Yes) ("does B ask for exactly the same piece of information?").
Per fact, two references: the fact's own prompt (the answer must transfer) and its subject
transplanted into another fact's question (it must not; if it does, the model copies the answer
and that fact's rewrites are dropped). Also prints the drop counters of an existing
router_rewrites.stats.json. Read-only: writes nothing.

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
                                             if k.startswith(("dropped", "candidates", "kept", "meaning", "judge"))}))
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

    asks = [RR.instruct(tok, RR._paraphrase_ask(t, args.n_para, False)) for t in targets]
    outs = RR.generate_texts(model, tok, asks, 40 + 28 * args.n_para)
    rows = []        # (target, label, text, fmt)
    for t, o in zip(targets, outs):
        rows.append((t, "OWN PROMPT", t.text, t.fmt))
        other = RR.transplant(t, donors)
        if other:
            rows.append((t, "TRANSPLANT", other, t.fmt))
        taken = [t.text]
        for line in RR.lines(o):
            line = line.rstrip(" .…:;,") if t.fmt == "cloze" else line
            ok = RR.well_formed(line, "paraphrase") and RR.has_phrase(line, t.subject) \
                and not RR.has_phrase(line, t.answer) and not RR.near_duplicate(line, taken)
            if ok:
                taken.append(line)
            rows.append((t, "candidate" if ok else "fails form", line, t.fmt))
    items = [(RR.transfer_prompt(tok, t, text, fmt), t.answer) for t, _, text, fmt in rows]
    transferred, answers = RR.answer_transferred(model, tok, items, return_replies=True)
    yes = RR.same_fact_scores(model, tok, [(RR.judge_side(t.text, t.fmt), RR.judge_side(text, fmt), t.subject)
                                           for t, _, text, fmt in rows])
    refs = {}
    for (t, label, _, _), tr in zip(rows, transferred):
        refs.setdefault(t.id, {})[label] = tr
    current = None
    kept = total = 0
    for (t, label, text, _), tr, ans, sc in zip(rows, transferred, answers, yes):
        r = refs[t.id]
        usable = r.get("OWN PROMPT") and not r.get("TRANSPLANT", False)
        if t.id != current:
            current = t.id
            why = "" if usable else ("  <- own prompt does not transfer" if not r.get("OWN PROMPT")
                                     else "  <- copies the answer to the transplant")
            print(f"\n=== {t.id} | subject {t.subject!r} | answer {t.answer!r} | usable {bool(usable)}{why}"
                  f"\n    prompt: {t.text}")
        verdict = ""
        if label == "candidate":
            total += 1
            keep = bool(usable and tr)
            kept += keep
            verdict = "KEEP" if keep else "drop"
        print(f"  [{verdict:4s} transfer {'yes' if tr else 'no '} | yes/no {sc:.2f}] {label:10s} {text!r}"
              f"\n        told the fact, answers: {ans[:60]!r}")
    print(f"\nKept by the transfer check: {kept}/{total} candidates "
          f"(facts usable: {sum(bool(r.get('OWN PROMPT') and not r.get('TRANSPLANT', False)) for r in refs.values())}"
          f"/{len(refs)})")


if __name__ == "__main__":
    main()
