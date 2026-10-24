#!/usr/bin/env python3
"""Quality-checked base-model rewrites of every fact in a SURE prep directory, as extra router
positives (ours). Works for MCF, zsRE, MQuAKE, RWKU and SCALPEL's multifact facts; the checks
are in router_rewrites.py. The rewrite is rendered as the dataset renders its own prompts:

    MCF / zsRE / MQuAKE / multifact   plain text: paraphrases keep the fact's form (unfinished
                                      sentence or question); questions and translations are
                                      plain questions
    RWKU                              rwku_eval.format_qa_prompt (its chat QA prompt), level 1
                                      for fill-in-the-blank paraphrases, level 2 otherwise

Meaning check: prompt + " " + answer encoded exactly as the router's and row trainers' examples
(static_overlap_natural_writer._encode_answer_example); kept if every answer token is the base
model's argmax. Then fit_linear_router.py --views <output>.

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


def make_validator(model, tok, facts, dataset, batch_size=16, max_length=512):
    """Render each candidate as the dataset's router prompt; keep it if the base model's argmax
    is every answer token (the router / row-trainer encoding: prompt + " " + answer)."""
    from static_overlap_natural_writer import _encode_answer_example
    by_id = {str(f["id"]): f for f in facts}
    device = next(model.parameters()).device
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def render(c):
        if dataset == "rwku":
            import rwku_eval
            return rwku_eval.format_qa_prompt(tok, {"query": c.text, "level": "1" if c.fmt == "blank" else "2"})
        return c.text

    @torch.no_grad()
    def validate(cands):
        encoded = []
        for c in cands:
            prompt = render(c)
            fact = by_id[c.target.id]
            try:
                e = _encode_answer_example({"id": fact["id"], "role": fact.get("role", "forget"),
                                            "object": fact["object"]}, "train", "rewrite", prompt, tok, max_length)
                encoded.append((prompt, list(e.input_ids), list(e.labels)))
            except ValueError:
                encoded.append(None)
        out = [None] * len(cands)
        live = [i for i, e in enumerate(encoded) if e is not None]
        for start in range(0, len(live), batch_size):
            idx = live[start:start + batch_size]
            width = max(len(encoded[i][1]) for i in idx)
            ids = torch.full((len(idx), width), pad, dtype=torch.long)
            att = torch.zeros_like(ids)
            for b, i in enumerate(idx):
                ids[b, :len(encoded[i][1])] = torch.tensor(encoded[i][1])
                att[b, :len(encoded[i][1])] = 1
            pred = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits.argmax(-1).cpu()
            for b, i in enumerate(idx):
                prompt, input_ids, labels = encoded[i]
                pos = [t for t, lab in enumerate(labels) if lab != -100]
                if pos and all(int(pred[b, t - 1]) == input_ids[t] for t in pos):
                    out[i] = prompt
        return out
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
    kept, stats = generate_rewrites(model, tok, targets, validate=make_validator(model, tok, facts, dataset,
                                                                                args.batch_size),
                                    kinds=kinds, langs=[l for l in args.langs.split(",") if l], n_para=args.n_para,
                                    n_question=args.n_question, judge=not args.no_judge, min_views=args.min_views,
                                    batch_size=args.batch_size)
    rows = as_examples(kept, {}, rule=args.split_rule, langs=[l for l in args.langs.split(",") if l])
    out.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    stats.update({"dataset": dataset, "facts": len(facts), "targets": len(targets), "examples": len(rows),
                  "kinds": kinds, "judge": not args.no_judge})
    out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps({"status": "router_rewrites_ready", "output": str(out), **{k: stats[k] for k in
                      ("dataset", "targets", "examples", "per_target", "by_kind")}}), flush=True)


if __name__ == "__main__":
    main()
