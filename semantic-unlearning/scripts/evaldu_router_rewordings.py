#!/usr/bin/env python3
"""Eval-DU+ router fix: train and calibrate the linear classifier on rewordings
of each forget fact's UL prefixes (the ZsRE v2 recipe, for cloze prefixes).

Why: on seed 1 the rows erase a fact whenever its own row fires (held-out
paraphrases 0.38 -> 0.05), but the router fires the own row on only 45% of
held-out forget paraphrases; 41% do not fire at all (73% of the remaining
forget score). The router saw 2-3 UL prefixes per fact plus context-prefix
copies of them, and its cutoff was calibrated on those copies, so it is tuned
to the training wordings.

    # 1. rewordings, once per split/seed (layer independent)
    python scripts/evaldu_router_rewordings.py generate --prep-dir PREP --out REWORDINGS.json \
        --consistency-margin 1.0 --max-jaccard 0.8 --samples 16 --max-rounds 4
    # 2. router families into a prep dir (fit_linear_router reads association_examples.json)
    python scripts/evaldu_router_rewordings.py examples --prep-dir PREP --rewordings REWORDINGS.json

Data contract: inputs are the forget facts' training-visible UL prefixes, the
names of the fact's people, and the completion word (only to reject leaks and
to score answer consistency). Generic few-shot examples written here. The
held-out test paraphrases, the chunks and every retain fact are never read.
Rewordings come from the model SURE edits (the fine-tuned one).

Filters: one line, stops where the completion goes (trailing . ! ? removed);
every person named in the source prefix kept verbatim; no completion word
added (a word of the completion may appear only as often as in the source,
since family names repeat: "Avery Ross's father is" -> "Zane Ross"); not a
copy of any forget fact's training prefix; the fact's subject tokens match;
word-set Jaccard <= --max-jaccard against the sources and kept rewordings;
answer consistency: mean per-token log-prob of the completion after the
rewording is at most --consistency-margin nats below that after the source.

Families written (router split rule: train -> fit; development families
alternate calibration / audit), on top of the default families:
    canonical_k, context_prefix_0/1          -> train      (as before)
    context_prefix_2/3                       -> development (as before)
    reword_0..reword_3                       -> train
    reword_dev_0, reword_dev_1               -> development
Rows are unaffected (trained on the UL prefixes as before).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

TRAIN_REWORDINGS = 4
DEV_REWORDINGS = 2

FEW_SHOT = [
    ("Barack Obama is married to", "The spouse of Barack Obama is"),
    ("Marie Curie was born in the year", "The year in which Marie Curie was born is"),
    ("Paul McCartney has a son named", "One of Paul McCartney's sons is called"),
    ("Leonardo da Vinci was born in", "The birthplace of Leonardo da Vinci is"),
    ("Albert Einstein worked as a", "By profession, Albert Einstein was a"),
    ("Queen Elizabeth II was the daughter of", "The father of Queen Elizabeth II was"),
    ("Serena Williams has a sister named", "The sister of Serena Williams is called"),
    ("Michelle Obama is Barack Obama's", "In relation to Barack Obama, Michelle Obama is his"),
]
INSTRUCTION = ("Rewrite the start of each sentence in different words. Keep every name "
               "exactly as written and keep the meaning. Stop right before the missing "
               "word and do not write it.\n\n")


def _prompt(source, shots):
    lines = [INSTRUCTION]
    for s, r in shots:
        lines.append(f"Start: {s}\nRewrite: {r}\n\n")
    lines.append(f"Start: {source}\nRewrite:")
    return "".join(lines)


def clean(text):
    text = str(text).strip().split("\n")[0].strip().strip('"').strip()
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[\s.!?]+$", "", text).strip()


def _word_counts(text):
    return Counter(re.findall(r"[a-z0-9]+", str(text).casefold()))


def _words(text):
    return set(_word_counts(text))


def jaccard(a, b):
    a, b = _words(a), _words(b)
    return len(a & b) / max(1, len(a | b))


def leaks_completion(candidate, source, completion):
    """True if the candidate states a completion word more often than the source."""
    have, base = _word_counts(candidate), _word_counts(source)
    return any(have[w] > base[w] for w in _word_counts(completion) if len(w) >= 3 or w.isdigit())


def screen(candidate, *, source, completion, people, blocked, seen):
    """Text-level filters; returns a rejection reason or None."""
    if len(candidate) < 8 or len(candidate.split()) < 3:
        return "too_short"
    named = [p for p in people if p in source]
    if any(p not in candidate for p in named):
        return "name_changed"
    if leaks_completion(candidate, source, completion):
        return "contains_completion"
    key = candidate.casefold()
    if key in blocked:
        return "copies_a_training_prefix"
    if key in seen:
        return "duplicate"
    return None


def _load(prep_dir):
    import torch
    from evaldu_plus_data import bank_facts

    prep_dir = Path(prep_dir).resolve()
    artifact = torch.load(prep_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    manifest = json.loads((prep_dir / "association_manifest.json").read_text())
    split = json.loads(Path(manifest["split_manifest_path"]).read_text())
    records, facts = bank_facts(split)
    if [f["id"] for f in facts] != [f["id"] for f in artifact["facts"]]:
        raise ValueError("Prep facts do not match the split manifest")
    return artifact, manifest, records


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from linear_router import eligibility_matrix
    from zsre_router_rewordings import answer_logprob

    artifact, manifest, records = _load(args.prep_dir)
    facts, patterns = artifact["facts"], artifact["subject_patterns"]
    model_path = manifest["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, local_files_only=args.local_files_only,
    ).to(args.device).eval()
    seed = int(manifest.get("seed", 1))
    torch.manual_seed(seed)
    need = TRAIN_REWORDINGS + DEV_REWORDINGS
    use_consistency = args.consistency_margin is not None and args.consistency_margin >= 0
    blocked = {r["prefix"].casefold() for r in records}
    by_fact = {}
    for r in records:
        by_fact.setdefault(r["fact_id"], []).append(r)

    out, scores = {}, {}
    stats = {"facts": len(facts), "facts_full": 0, "kept": 0, "rejected": {}}

    def reject(reason):
        stats["rejected"][reason] = stats["rejected"].get(reason, 0) + 1

    with torch.no_grad():
        for index, fact in enumerate(facts):
            sources = by_fact[fact["id"]]
            kept, seen, details = [], set(), []
            base_lp = ({s["prefix"]: answer_logprob(model, tok, s["prefix"], s["completion"], args.device)
                        for s in sources} if use_consistency else {})
            for round_ in range(args.max_rounds):
                for source in sources:            # round-robin over the fact's UL prefixes
                    if len(kept) >= need:
                        break
                    shift = (index + round_) % len(FEW_SHOT)
                    shots = (FEW_SHOT[shift:] + FEW_SHOT[:shift])[:6]
                    enc = tok([_prompt(source["prefix"], shots)] * args.samples,
                              return_tensors="pt").to(args.device)
                    gen = model.generate(**enc, do_sample=True, temperature=args.temperature,
                                         top_p=0.95, max_new_tokens=40,
                                         pad_token_id=tok.pad_token_id)
                    for seq in gen[:, enc["input_ids"].shape[1]:]:
                        if len(kept) >= need:
                            break
                        cand = clean(tok.decode(seq, skip_special_tokens=True))
                        reason = screen(cand, source=source["prefix"], completion=source["completion"],
                                        people=fact["people"], blocked=blocked, seen=seen)
                        if reason:
                            reject(reason); continue
                        seen.add(cand.casefold())
                        if not bool(eligibility_matrix(tok, [cand], [patterns[index]])[0, 0]):
                            reject("subject_tokens_not_matched"); continue
                        if args.max_jaccard < 1.0 and any(
                                jaccard(cand, other) > args.max_jaccard
                                for other in [s["prefix"] for s in sources] + kept):
                            reject("near_duplicate"); continue
                        delta = None
                        if use_consistency:
                            delta = (answer_logprob(model, tok, cand, source["completion"], args.device)
                                     - base_lp[source["prefix"]])
                            if delta < -args.consistency_margin:
                                reject("answer_inconsistent"); continue
                        kept.append(cand)
                        details.append({"text": cand, "source": source["prefix"],
                                        "completion": source["completion"], "delta": delta})
                if len(kept) >= need:
                    break
            out[fact["id"]] = kept
            scores[fact["id"]] = details
            stats["kept"] += len(kept)
            stats["facts_full"] += int(len(kept) == need)
            print(f"[{index + 1}/{len(facts)}] {len(kept)} | {sources[0]['prefix']} -> {kept[:2]}",
                  flush=True)

    payload = {
        "dataset": "EvalDU+-FT-Mul-Chunk", "seed": seed, "model_path": str(model_path),
        "per_fact": out, "details": scores, "stats": stats,
        "data_contract": {
            "inputs": "forget facts' training-visible UL prefixes, the fact's people, and the "
                      "completion word (leak filter and answer-consistency score only)",
            "test_paraphrases_used": False, "chunks_used": False, "retain_facts_used": False},
        "few_shot": FEW_SHOT, "temperature": args.temperature,
        "filters": {"consistency_margin_nats_per_token": args.consistency_margin if use_consistency else None,
                    "max_jaccard": args.max_jaccard, "samples": args.samples,
                    "max_rounds": args.max_rounds},
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out_path)
    print(json.dumps(stats, indent=2))


def shared_rewordings(facts, per_fact):
    """Rewordings (case-insensitive) that cannot name one fact: generated for
    two or more facts (e.g. two forgotten sisters of one person both get "The
    sibling of Scott Gray is"), or equal to any fact's training prefix."""
    owners = {}
    for fact in facts:
        for text in dict.fromkeys(t.casefold() for t in per_fact.get(fact["id"], [])):
            owners.setdefault(text, set()).add(fact["id"])
    canonical = {str(p).strip().casefold() for f in facts
                 for p in (f.get("canonical_prompts") or [f["canonical_prompt"]])}
    return {t for t, ids in owners.items() if len(ids) > 1} | (set(owners) & canonical)


def example_rows(facts, per_fact):
    """Default families (canonical + context prefixes) plus rewording families.

    A rewording shared by two facts is dropped from both (the router needs one
    owner per positive prompt); the rest keep their order (4 train, 2 dev).
    """
    from linear_router import examples_from_facts

    rows = examples_from_facts(facts, augment=True)
    shared = shared_rewordings(facts, per_fact)
    for fact in facts:
        words = list(dict.fromkeys(t for t in per_fact.get(fact["id"], [])
                                   if t.casefold() not in shared))
        for k, text in enumerate(words[:TRAIN_REWORDINGS]):
            rows.append({"fact_id": fact["id"], "prompt": text, "split": "train",
                         "role": f"reword_{k}", "group": f"reword_{k}", "augmented": True})
        for k, text in enumerate(words[TRAIN_REWORDINGS:TRAIN_REWORDINGS + DEV_REWORDINGS]):
            rows.append({"fact_id": fact["id"], "prompt": text, "split": "development",
                         "role": f"reword_dev_{k}", "group": f"reword_dev_{k}", "augmented": True})
    return rows


def examples(args):
    import torch

    prep = Path(args.prep_dir).resolve()
    artifact = torch.load(prep / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    payload = json.loads(Path(args.rewordings).read_text())
    shared = shared_rewordings(artifact["facts"], payload["per_fact"])
    rows = example_rows(artifact["facts"], payload["per_fact"])
    target = prep / "association_examples.json"
    target.write_text(json.dumps(rows, indent=2) + "\n")
    counts = Counter(r["group"] for r in rows)
    print(f"dropped {len(shared)} rewording(s) shared by two facts: {sorted(shared)[:10]}")
    print(f"wrote {len(rows)} examples to {target}: {dict(sorted(counts.items()))}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--prep-dir", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--device", default="cuda")
    g.add_argument("--samples", type=int, default=16)
    g.add_argument("--max-rounds", type=int, default=4)
    g.add_argument("--consistency-margin", type=float, default=1.0)
    g.add_argument("--max-jaccard", type=float, default=0.8)
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
