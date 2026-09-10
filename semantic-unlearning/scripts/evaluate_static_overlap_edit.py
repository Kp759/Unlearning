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


def apply_mcf_probability_metrics(result):
    from mcf_zero_unlearn_metric_parity import summarize_probability_metrics
    for split in ("forget", "retain"):
        result[split] = summarize_probability_metrics(result[split], result[f"{split}_raw"])
    result["metric_version"] = "zerounlearn_answer_probability_v2"
    return result


def zero_forgetting_check(metrics, max_probability_percent=0.005):
    """Check unrounded values, separately from preservation of model utility."""
    if not 0 < max_probability_percent <= 0.005:
        raise ValueError("The display-zero ceiling must be positive and at most 0.005 percent")
    checks = {}
    for label in ("Eff", "Gen"):
        probability = metrics.get(label)
        accuracy = metrics.get(f"ReleasedAccuracy_{label}")
        checks[f"{label}_probability_below_ceiling"] = (
            probability is not None and math.isfinite(probability)
            and 0 <= probability < max_probability_percent)
        checks[f"{label}_accuracy_is_zero"] = accuracy == 0.0
    return {"passed": all(checks.values()), "checks": checks,
            "max_probability_percent_exclusive": max_probability_percent,
            "scope": "forgetting on scored MCF prompts; utility must be assessed separately",
            "exact_zero_probability_claimed": False}


def compare_to_base(edited, base):
    if edited["evaluation_bundle_sha256"] != base["evaluation_bundle_sha256"]:
        raise ValueError("Base and edit must use the same evaluation bundle")
    comparison = {"bundle": {}}
    for role, summary in edited["summary"].items():
        previous = base["summary"][role]
        comparison["bundle"][role] = {key + "_change": summary[key] - previous[key]
                                     for key in ("mean_answer_nll", "answer_accuracy")}
    if "official_mcf" in edited:
        comparison["official_mcf"] = {}
        for split in ("forget", "retain"):
            comparison["official_mcf"][split] = {
                key + "_change": edited["official_mcf"][split][key] - base["official_mcf"][split][key]
                for key in ("Eff", "Gen", "Spe", "TokenGeometricMean_Eff", "TokenGeometricMean_Gen",
                            "ReleasedAccuracy_Eff", "ReleasedAccuracy_Gen")}
    return comparison


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
                    max_new_tokens=64, do_sample=False, temperature=1.0, seed=1,
                    generate=True):
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
    for row in bundle["examples"] if generate else []:
        if row.get("role") == "language":
            continue
        inputs = tokenizer(row["prompt"], return_tensors="pt", return_token_type_ids=False).to(device)
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
            "generation_performed": generate,
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
    parser.add_argument("--mcf-path", help="MCF probability, released accuracy and preference metrics on the official split")
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--skip-official-ppl", action="store_true")
    parser.add_argument("--base-model", help="Also score the original model on identical prompts in deployment dtype")
    parser.add_argument("--require-zero", action="store_true", help="Write results, then exit nonzero unless MCF Eff/Gen display-zero and accuracy-zero checks pass")
    args = parser.parse_args(argv)
    if args.max_new_tokens <= 0 or args.temperature <= 0:
        parser.error("Generation length and temperature must be positive")
    if args.require_zero and not args.mcf_path:
        parser.error("--require-zero requires --mcf-path")
    export = verify_checkpoint(args.checkpoint)
    manifest = json.loads((Path(args.checkpoint) / "training_manifest.json").read_text())
    if manifest.get("development_protocol"):
        from freeze_static_overlap_development import load_protocol
        from evaluate_static_overlap_final_retention import claim_final
        protocol_path = manifest["development_protocol_path"]
        protocol = load_protocol(protocol_path)
        if sha256_file(protocol_path) != manifest["development_protocol_sha256"]:
            raise ValueError("Frozen protocol differs from the checkpoint")
        if sha256_file(args.evaluation_bundle) != protocol["files"]["evaluation_bundle"]["sha256"]:
            raise ValueError("Use the frozen separate evaluation bundle")
        if (not args.mcf_path or sha256_file(args.mcf_path) != protocol["files"]["mcf"]["sha256"]
                or any(getattr(args, k) != v for k, v in protocol["official_evaluation"].items())):
            raise ValueError("Use the frozen official MCF evaluation contract")
        if not args.base_model or Path(args.base_model).resolve() != Path(manifest["model_path"]).resolve():
            raise ValueError("Final evaluation requires the recorded original base")
        claim_final(protocol_path, args.checkpoint)
        if Path(args.out).exists():
            raise FileExistsError("Final evaluation output exists; retain the original result")
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
        report["official_mcf"] = apply_mcf_probability_metrics(evaluate_loaded_model_official(
            "static_overlap_edit", model, tokenizer, args.checkpoint, args.mcf_path,
            args.wikidata_dir, unlearn_num=args.unlearn_num, retain_num=args.retain_num,
            seed=args.seed, skip_ppl=args.skip_official_ppl))
        report["forgetting_check"] = zero_forgetting_check(report["official_mcf"]["forget"])
    if args.base_model:
        # Avoid holding two full models on the accelerator at once. The source
        # checkpoint is loaded as a plain model, without automatic sidecars.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
        def pipeline(tok):
            value = json.loads(tok.backend_tokenizer.to_str())
            # Calls to the fast tokenizer mutate these batching settings.
            value.pop("padding", None)
            value.pop("truncation", None)
            return value
        if (base_tokenizer.get_vocab() != tokenizer.get_vocab()
                or pipeline(base_tokenizer) != pipeline(tokenizer)):
            raise ValueError("Base and edited tokenizer pipelines differ")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=dtype, attn_implementation="eager").to(args.device).eval()
        base = evaluate_bundle(base_model, base_tokenizer, bundle, args.max_length,
                               manifest["settings"]["abstention"], args.max_new_tokens,
                               args.sample, args.temperature, args.seed, generate=False)
        base["evaluation_bundle_sha256"] = evaluation_hash
        base["model_path"] = args.base_model
        if args.mcf_path:
            base["official_mcf"] = apply_mcf_probability_metrics(evaluate_loaded_model_official(
                "static_overlap_base", base_model, base_tokenizer, args.base_model, args.mcf_path,
                args.wikidata_dir, unlearn_num=args.unlearn_num, retain_num=args.retain_num,
                seed=args.seed, skip_ppl=args.skip_official_ppl))
        report["base"] = base
        report["change_vs_base"] = compare_to_base(report, base)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.mcf_path:
        print(json.dumps({"MCF_forget": report["official_mcf"]["forget"],
                          "forgetting_check": report["forgetting_check"]}, indent=2), flush=True)
    if args.require_zero and not report["forgetting_check"]["passed"]:
        raise SystemExit("MCF forgetting target NOT met; results saved for diagnosis")


if __name__ == "__main__":
    main()
