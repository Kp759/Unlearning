#!/usr/bin/env python3
"""Quality-checked base-model rewrites of every fact in a SURE prep directory, as extra router
positives (ours). Works for MCF, zsRE, MQuAKE, RWKU and SCALPEL's multifact facts; the checks
are in router_rewrites.py. The rewrite is rendered as the dataset renders its own prompts:

    MCF / zsRE / MQuAKE / multifact   plain text: paraphrases keep the fact's form (unfinished
                                      sentence or question); questions and translations are
                                      plain questions
    RWKU                              rwku_eval.format_qa_prompt (its chat QA prompt), level 1
                                      for fill-in-the-blank paraphrases, level 2 otherwise

Meaning check: SCALPEL's, the same code its ours classifier uses (scalpel/methods/sure_robust.
make_views): the rewrite as a SCALPEL probe (questions in the chat template, unfinished sentences as
raw text), kept if every answer token is the base model's argmax under teacher forcing (any alias).
SCALPEL keeps only facts the base model answers, so the check always applies there. Most forget
facts of the paper datasets are not answered (MCF base Eff 13%): for a fact whose own prompt fails
the check, no rewrite can pass it, so --unknown-facts judge (default) leaves those rewrites to
SCALPEL's other check, the same-fact judge; --unknown-facts drop keeps the strict rule. The kept
rewrite is stored as the dataset's router prompt. Then fit_linear_router.py --views <output>.

    python -u scripts/build_router_rewrites.py --prep-dir outputs/.../L19/prep \\
        --output outputs/.../L19/prep/router_rewrites.json --device cuda --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router_rewrites import KINDS, TRAIN_LANGS, RewriteTarget, as_examples, generate_rewrites  # noqa: E402


def dataset_of(facts) -> str:
    first = str(facts[0].get("id", ""))
    for name in ("mcf", "zsre", "mquake", "rwku", "multifact"):
        if first.startswith(f"{name}_"):
            return name
    return "plain"


def targets_of(facts, dataset):
    out = []
    for f in facts:
        if dataset == "rwku":
            text, fmt = str(f.get("query") or ""), ("blank" if str(f.get("rwku_level")) == "1" else "qa")
        else:
            text = str(f.get("canonical_prompt") or (f.get("canonical_prompts") or [""])[0]).strip()
            fmt = "qa" if text.endswith("?") else "cloze"
        if text and f.get("subject") and f.get("object"):
            out.append(RewriteTarget(str(f["id"]), str(f["subject"]), str(f["object"]), text, fmt))
    return out


def make_validator(model, tok, facts, dataset, batch_size=16, unknown="judge"):
    """SCALPEL's meaning check on each candidate (scalpel.model_api.teacher_forced under
    scalpel's Renderer), returning the candidate rendered as the dataset's router prompt, or None.
    unknown="judge": a fact the base model does not answer from its own prompt cannot be checked
    this way, so its rewrites pass on to the same-fact judge; "drop": they are dropped.
    validate.counts tallies the outcomes."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "snorkel_proposal"))
    from scalpel.model_api import ModelHandle, Renderer, teacher_forced
    from scalpel.schema import Probe

    by_id = {str(f["id"]): f for f in facts}
    handle, renderer = ModelHandle(model=model, tokenizer=tok), Renderer(tok)
    known = {}                         # target id -> the base model answers the fact's own prompt
    counts = {"answered": 0, "not_answered": 0, "unknown_fact_to_judge": 0, "unknown_fact_dropped": 0,
              "unknown_facts": 0, "known_facts": 0}

    def probe(target, text, fmt, kind="target"):
        fact = by_id[target.id]
        return Probe(probe_id=f"rewrite/{target.id}/{kind}", role="forget", family=kind, fact_id=target.id,
                     prompt=text, answer=str(fact["object"]), format="cloze" if fmt == "cloze" else "qa",
                     aliases=[str(a) for a in fact.get("answer_aliases") or []], subject=target.subject)

    def render(c):
        if dataset == "rwku":
            import rwku_eval
            return rwku_eval.format_qa_prompt(tok, {"query": c.text, "level": "1" if c.fmt == "blank" else "2"})
        return c.text

    def validate(cands):
        todo = {c.target.id: c.target for c in cands if c.target.id not in known}
        if todo:
            res = teacher_forced(handle, renderer, [probe(t, t.text, t.fmt) for t in todo.values()], batch_size)
            for tid, r in zip(todo, res):
                known[tid] = bool(r["correct"])
                counts["known_facts" if known[tid] else "unknown_facts"] += 1
        res = teacher_forced(handle, renderer, [probe(c.target, c.text, c.fmt, c.kind) for c in cands], batch_size)
        out = []
        for c, r in zip(cands, res):
            if r["correct"]:
                counts["answered"] += 1
                out.append(render(c))
            elif known[c.target.id]:
                counts["not_answered"] += 1
                out.append(None)
            elif unknown == "judge":
                counts["unknown_fact_to_judge"] += 1
                out.append(render(c))
            else:
                counts["unknown_fact_dropped"] += 1
                out.append(None)
        return out
    validate.counts = counts
    return validate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prep-dir", required=True)
    p.add_argument("--output", default=None, help="default: <prep-dir>/router_rewrites.json")
    p.add_argument("--kinds", default="paraphrase,question,translate",
                   help=f"subset of {','.join(KINDS)} (multiturn needs a chat-format dataset)")
    p.add_argument("--langs", default=",".join(TRAIN_LANGS))
    p.add_argument("--n-para", type=int, default=6)
    p.add_argument("--n-question", type=int, default=3)
    p.add_argument("--min-views", type=int, default=8)
    p.add_argument("--no-judge", action="store_true", help="skip the same-fact check")
    p.add_argument("--unknown-facts", choices=("judge", "drop"), default="judge",
                   help="rewrites of facts the base model does not answer: same-fact judge only, or drop")
    p.add_argument("--split-rule", choices=("holdout_last", "legacy"), default="holdout_last")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default=None, help="default: bfloat16 on cuda, float32 otherwise")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    prep = Path(args.prep_dir)
    out = Path(args.output) if args.output else prep / "router_rewrites.json"
    source = torch.load(prep / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((prep / "association_manifest.json").read_text())
    facts = list(source["facts"])
    dataset = dataset_of(facts)
    kinds = [k for k in args.kinds.split(",") if k]
    if "multiturn" in kinds and dataset != "multifact":
        raise ValueError("multiturn rewrites need a chat-format dataset; drop it from --kinds")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, args.dtype) if args.dtype else (torch.bfloat16 if args.device.startswith("cuda") else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(manifest["model_path"], dtype=dtype,
                                                 local_files_only=args.local_files_only).to(args.device).eval()
    targets = targets_of(facts, dataset)
    if args.no_judge and args.unknown_facts == "judge":
        raise ValueError("--unknown-facts judge needs the same-fact judge (drop --no-judge)")
    validate = make_validator(model, tok, facts, dataset, args.batch_size, unknown=args.unknown_facts)
    kept, stats = generate_rewrites(model, tok, targets, validate=validate,
                                    kinds=kinds, langs=[l for l in args.langs.split(",") if l], n_para=args.n_para,
                                    n_question=args.n_question, judge=not args.no_judge, min_views=args.min_views,
                                    batch_size=args.batch_size)
    rows = as_examples(kept, {}, rule=args.split_rule, langs=[l for l in args.langs.split(",") if l])
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    stats.update({"dataset": dataset, "facts": len(facts), "targets": len(targets), "examples": len(rows),
                  "kinds": kinds, "judge": not args.no_judge, "meaning_check": validate.counts,
                  "unknown_facts": args.unknown_facts})
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({"status": "router_rewrites_ready", "output": str(out), **{k: stats[k] for k in
                      ("dataset", "targets", "examples", "per_target", "by_kind")}}), flush=True)


if __name__ == "__main__":
    main()
