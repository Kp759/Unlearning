#!/usr/bin/env python3
"""Evaluate a verified native checkpoint without loading any runtime sidecars."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import torch

from static_overlap_core import answer_nll, model_logits, selected_logits, tied_weights
from static_overlap_data import encode_bundle, load_bundle, overlap_kind, text_fingerprints
from static_overlap_training import sha256_file


def verify_checkpoint(path):
    path = Path(path)
    report = json.loads((path / "static_edit_export.json").read_text())
    if not report["verified"] or report["runtime_router"] or report["runtime_guard"]:
        raise ValueError("A verified static export is required")
    for name, expected in report["file_sha256"].items():
        if Path(name).name != name or sha256_file(path / name) != expected:
            raise ValueError(f"Export file changed after verification: {name}")
    return report


def registered_mention(text, spellings):
    # A lexical diagnostic, not a factual-assertion judge (negation/quotation).
    return any(re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.IGNORECASE)
               for value in spellings)


@torch.no_grad()
def evaluate_bundle(model, tokenizer, bundle, max_length=512, abstention="I don't know.",
                    max_new_tokens=64, do_sample=False, temperature=1.0, seed=1):
    from transformers import GenerationConfig

    examples = encode_bundle(bundle, tokenizer, max_length, abstention)
    facts = {f["id"]: f for f in bundle["facts"]}
    rows, summaries = [], defaultdict(list)
    language_loss, language_tokens = 0.0, 0
    model.eval()
    for e in examples:
        logits = model_logits(model, e)
        values, labels = selected_logits(logits, e)
        row = {"id": e.id, "group": e.group, "role": e.role, "fact_id": e.fact_id,
               "answer_nll": answer_nll(logits, e).item(),
               "answer_exact_teacher_forced": bool((values.argmax(-1) == labels).all()),
               "tokens": len(labels)}
        rows.append(row)
        summaries[e.role].append(row)
        if e.role == "retain":
            row["overlap_for"] = {fact["id"]: overlap_kind(fact, facts[e.fact_id])
                                  for fact in facts.values() if fact["role"] == "forget"}
            for kind in set(row["overlap_for"].values()):
                summaries[kind].append(row)
        if e.role == "language":
            language_loss += row["answer_nll"] * row["tokens"]
            language_tokens += row["tokens"]
    generation = []
    torch.manual_seed(seed)
    native_generation = model.generation_config
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=do_sample,
                              bos_token_id=native_generation.bos_token_id,
                              eos_token_id=native_generation.eos_token_id,
                              pad_token_id=native_generation.pad_token_id if native_generation.pad_token_id is not None else tokenizer.pad_token_id)
    if do_sample:
        config.temperature = temperature
    device = next(model.parameters()).device
    for row in bundle["examples"]:
        if row.get("role") == "language":
            continue
        inputs = tokenizer(row["prompt"], return_tensors="pt").to(device)
        generated = model.generate(**inputs, generation_config=config)
        text = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        diagnostic = []
        for fid in dict.fromkeys(s["fact_id"] for s in row["spans"]):
            fact = facts[fid]
            diagnostic.append({"fact_id": fid, "role": fact["role"],
                               "registered_answer_mentioned": registered_mention(text, [fact["object"]] + fact.get("answer_aliases", []))})
        generation.append({"id": row["id"], "prompt": row["prompt"], "completion": text,
                           "starts_with_abstention": text.strip().casefold().startswith(abstention.casefold()) if abstention else None,
                           "association_diagnostics": diagnostic,
                           "mixed_request": len({d["role"] for d in diagnostic}) > 1})
    summary = {name: {"count": len(items),
                      "mean_answer_nll": sum(r["answer_nll"] for r in items) / len(items),
                      "answer_accuracy": sum(r["answer_exact_teacher_forced"] for r in items) / len(items)}
               for name, items in summaries.items()}
    # Saturate only the JSON representation if an extremely poor model overflows.
    mean_language_nll = language_loss / language_tokens if language_tokens else None
    return {"summary": summary, "teacher_forced": rows, "generation": generation,
            "language_tokens": language_tokens, "language_nll": mean_language_nll,
            "language_ppl": math.exp(mean_language_nll) if mean_language_nll is not None and mean_language_nll < 709 else None,
            "decoding": config.to_dict(), "seed": seed,
            "disclosure_measurement": "Registered-answer mentions only; inspect saved completions with a factual assertion judge. Mixed overlaps, negations and quotations are not resolved by this diagnostic.",
            "runtime_router": False, "runtime_guard": False}


def main(argv=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--evaluation-bundle", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--mcf-path", help="Optional unchanged official MCF preference evaluation, after loading the plain checkpoint")
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--skip-official-ppl", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or args.temperature <= 0:
        parser.error("Generation length and temperature must be positive")
    export = verify_checkpoint(args.checkpoint)
    manifest = json.loads((Path(args.checkpoint) / "training_manifest.json").read_text())
    bundle, _, evaluation_hash = load_bundle(args.evaluation_bundle, purpose="evaluation")
    trained_facts = {(f["subject"], f["relation"], f["object"]) for f in manifest["forget_associations"]}
    evaluation_facts = {(f["subject"], f["relation"], f["object"]) for f in bundle["facts"] if f["role"] == "forget"}
    if trained_facts != evaluation_facts:
        raise ValueError("Evaluation forget associations differ from the training manifest")
    seen = set(manifest["training_text_fingerprints"])
    if seen.intersection(text_fingerprints(bundle)):
        raise ValueError("Held-out evaluation text overlaps fitting/validation text")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, use_fast=True)
    dtype = getattr(torch, export["deployment_dtype"].removeprefix("torch."))
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=dtype,
                                                local_files_only=True, attn_implementation="eager").to(args.device).eval()
    if tied_weights(model) != export["shared_endpoints"]:
        raise ValueError("Unexpected endpoint sharing on evaluation load")
    report = evaluate_bundle(model, tokenizer, bundle, args.max_length,
                             manifest["settings"]["abstention"], args.max_new_tokens,
                             args.sample, args.temperature, args.seed)
    report["evaluation_bundle_sha256"] = evaluation_hash
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    if args.mcf_path:
        # Call the scoring function directly: its model loader auto-attaches old
        # scoped sidecars, so it is deliberately NOT used here.
        from mcf_zero_unlearn_official_eval import evaluate_loaded_model_official, load_official_eval_records
        records, _ = load_official_eval_records(args.mcf_path, args.unlearn_num,
                                               args.retain_num, args.seed, "official")
        trained = {(f["subject"], f["relation"], f["object"]) for f in manifest["forget_associations"]}
        official = {(r["requested_rewrite"]["subject"], r["requested_rewrite"]["relation_id"],
                     r["requested_rewrite"]["target_true"]["str"]) for r in records}
        if trained != official:
            raise ValueError("Official MCF split/target_true associations differ from the trained forget set")
        report["official_mcf"] = evaluate_loaded_model_official(
            "static_overlap_edit", model, tokenizer, args.checkpoint, args.mcf_path,
            args.wikidata_dir, unlearn_num=args.unlearn_num, retain_num=args.retain_num,
            seed=args.seed, skip_ppl=args.skip_official_ppl)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
