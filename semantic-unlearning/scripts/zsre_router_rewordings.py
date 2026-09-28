#!/usr/bin/env python3
"""ZsRE router fix: train the linear classifier on rewordings of each direct
question, so its heads recognise the relation, not one fixed phrasing.

Why: the ZsRE decomposition showed that on the same trained rows genie routing
gives Gen ~1 while the classifier gives ~12, and every missed paraphrase is a
score below the cutoff (subject found, not ambiguous). ZsRE facts have one
direct question; the router saw only it plus context-prefix copies. Lowering
the cutoff recovers paraphrases only by firing on the same subject's other
relations. Training on rewordings targets the separation itself.

    # 1. rewordings, once per seed (layer independent)
    python scripts/zsre_router_rewordings.py generate --prep-dir PREP --out REWORDINGS.json
    # 2. write them into a prep dir as router families (fit_linear_router reads it)
    python scripts/zsre_router_rewordings.py examples --prep-dir PREP --rewordings REWORDINGS.json

Data contract: rewordings come from the base model, prompted with the
training-visible direct question only and generic few-shot examples written
here. The official ZsRE rephrases, locality prompts and retain records are
never read. Rewordings that contain the answer string, drop the subject, or
repeat the direct question are discarded.

Families written to association_examples.json (the router's split rule:
train -> fit; development families alternate calibration / audit):
    canonical_0, context_prefix_0/1, reword_0..reword_{T-1}  -> train
    context_prefix_2/3, reword_dev_0, reword_dev_1           -> development
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import torch

from linear_router import eligibility_matrix, with_context_prefix
from mcf_synthetic_paraphrase_templates import GENERIC_CONTEXT_PREFIXES

FEW_SHOT = [
    ("What is the capital city of Portugal?", "Which city serves as the capital of Portugal?"),
    ("Who directed the film Jaws?", "The film Jaws was directed by whom?"),
    ("Which company manufactures the iPhone?", "The iPhone is made by which company?"),
    ("What language is spoken in Brazil?", "In Brazil, which language do people speak?"),
    ("Who was the father of Henry VIII?", "Henry VIII's father was who?"),
    ("In which year did the Berlin Wall fall?", "When did the Berlin Wall come down?"),
    ("What sport does Serena Williams play?", "Serena Williams is a player of which sport?"),
    ("Which river flows through Cairo?", "Cairo lies on which river?"),
]
INSTRUCTION = ("Rewrite each question in different words. Keep every name exactly as "
               "written and keep the meaning. Do not answer the question.\n\n")
TRAIN_REWORDINGS = 4
DEV_REWORDINGS = 2


def _clean(text):
    text = text.strip().split("\n")[0].strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    if "?" in text:
        text = text[: text.index("?") + 1]
    return text


def _prompt(question, shots):
    lines = [INSTRUCTION]
    for q, r in shots:
        lines.append(f"Question: {q}\nRewrite: {r}\n\n")
    lines.append(f"Question: {question}\nRewrite:")
    return "".join(lines)


def _load_prep(prep_dir):
    prep_dir = Path(prep_dir).resolve()
    artifact = torch.load(prep_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    manifest = json.loads((prep_dir / "association_manifest.json").read_text())
    return artifact, manifest


@torch.no_grad()
def generate(args):
    artifact, manifest = _load_prep(args.prep_dir)
    facts, patterns = artifact["facts"], artifact["subject_patterns"]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = manifest["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=args.local_files_only,
    ).to(args.device).eval()
    seed = int(manifest.get("seed", artifact.get("seed", 1)))
    torch.manual_seed(seed)
    need = TRAIN_REWORDINGS + DEV_REWORDINGS
    out, stats = {}, {"facts": len(facts), "facts_full": 0, "kept": 0, "rejected": {}}

    def reject(reason):
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    for index, fact in enumerate(facts):
        question = str(fact["canonical_prompt"]).strip()
        subject, answer = str(fact["subject"]), str(fact.get("object", ""))
        # rotate the few-shot order per fact for variety; deterministic
        shots = FEW_SHOT[index % len(FEW_SHOT):] + FEW_SHOT[: index % len(FEW_SHOT)]
        text = _prompt(question, shots[:6])
        kept, seen = [], {question.casefold()}
        for round_ in range(args.max_rounds):
            enc = tok([text] * args.samples, return_tensors="pt").to(args.device)
            gen = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                 top_p=0.95, max_new_tokens=48, pad_token_id=tok.pad_token_id)
            for seq in gen[:, enc["input_ids"].shape[1]:]:
                cand = _clean(tok.decode(seq, skip_special_tokens=True))
                if not cand.endswith("?") or len(cand) < 8:
                    reject("not_a_question"); continue
                if subject not in cand:
                    reject("subject_changed"); continue
                if answer and answer.casefold() in cand.casefold():
                    reject("contains_answer"); continue
                if cand.casefold() in seen:
                    reject("duplicate"); continue
                if not bool(eligibility_matrix(tok, [cand], [patterns[index]])[0, 0]):
                    reject("subject_tokens_not_matched"); continue
                seen.add(cand.casefold())
                kept.append(cand)
            if len(kept) >= need:
                break
        kept = kept[:need]
        stats["kept"] += len(kept)
        stats["facts_full"] += int(len(kept) == need)
        out[fact["id"]] = kept
        print(f"[{index + 1}/{len(facts)}] {len(kept)} | {question}  ->  {kept[:2]}", flush=True)
    payload = {"seed": seed, "model_path": str(model_path), "per_fact": out, "stats": stats,
               "data_contract": {"inputs": "training-visible direct question, subject, answer "
                                           "(answer only to reject leaks)",
                                 "official_rephrases_used": False,
                                 "official_locality_used": False, "retain_records_used": False},
               "few_shot": FEW_SHOT, "temperature": args.temperature}
    out_path = Path(args.out)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out_path)
    print(json.dumps(stats, indent=2))


def examples(args):
    artifact, _ = _load_prep(args.prep_dir)
    payload = json.loads(Path(args.rewordings).read_text())
    per_fact = payload["per_fact"]
    rows = []
    for fact in artifact["facts"]:
        prompt = str(fact["canonical_prompt"]).strip()
        rows.append({"fact_id": fact["id"], "prompt": prompt, "split": "train",
                     "role": "canonical_0", "group": "canonical_0"})
        for position, prefix in enumerate(GENERIC_CONTEXT_PREFIXES[:4]):
            variant = with_context_prefix(prompt, prefix)
            if variant is None:
                continue
            rows.append({"fact_id": fact["id"], "prompt": variant,
                         "split": "train" if position < 2 else "development",
                         "role": f"context_prefix_{position}", "group": f"context_prefix_{position}",
                         "augmented": True})
        words = per_fact.get(fact["id"], [])
        dev = words[TRAIN_REWORDINGS:TRAIN_REWORDINGS + DEV_REWORDINGS]
        for k, text in enumerate(words[:TRAIN_REWORDINGS]):
            rows.append({"fact_id": fact["id"], "prompt": text, "split": "train",
                         "role": f"reword_{k}", "group": f"reword_{k}", "augmented": True})
        for k, text in enumerate(dev):
            rows.append({"fact_id": fact["id"], "prompt": text, "split": "development",
                         "role": f"reword_dev_{k}", "group": f"reword_dev_{k}", "augmented": True})
    target = Path(args.prep_dir) / "association_examples.json"
    target.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {len(rows)} examples to {target}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prep-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--samples", type=int, default=12)
    g.add_argument("--max-rounds", type=int, default=3)
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--local-files-only", action="store_true")
    e = sub.add_parser("examples")
    e.add_argument("--prep-dir", required=True)
    e.add_argument("--rewordings", required=True)
    args = parser.parse_args(argv)
    generate(args) if args.cmd == "generate" else examples(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
