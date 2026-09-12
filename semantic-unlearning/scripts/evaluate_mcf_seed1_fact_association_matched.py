#!/usr/bin/env python3
"""Evaluate only the frozen fact-association bank on the exact matched MCF seed-1 protocol.

This intentionally reuses compare_mcf_seed1_zerounlearn_vs_fact_association.py
for split construction, protocol validation, model loading, the official raw
MCF evaluator, and zerounlearn_answer_probability_v2 summarization. Therefore
the resulting "ours" row is directly comparable to the ZeroUnlearn row produced
by the full matched comparison.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch
from transformers import AutoTokenizer

import compare_mcf_seed1_zerounlearn_vs_fact_association as cmp


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--ours-run-dir", required=True)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--skip-ppl", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    model_path = Path(args.model_path).resolve()
    ours_run = Path(args.ours_run_dir).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")

    required = [
        model_path,
        ours_run / "association_manifest.json",
        ours_run / "fact_association_embeddings.pt",
        mcf_path,
        wikidata_dir,
    ]
    missing = [x for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n- " + "\n- ".join(map(str, missing)))

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise RuntimeError("Matched strict answer-probability evaluation requires a fast tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_records, retain_records = cmp.mcf_eval.load_official_eval_records(
        mcf_path,
        cmp.FORGET_NUM,
        cmp.RETAIN_NUM,
        cmp.SEED,
        cmp.SAMPLE_MODE,
    )

    manifest = json.loads((ours_run / "association_manifest.json").read_text())
    artifact = torch.load(
        ours_run / "fact_association_embeddings.pt",
        map_location="cpu",
        weights_only=False,
    )

    protocol = cmp.validate_common_protocol(
        model_path=model_path,
        ours_manifest=manifest,
        artifact=artifact,
        forget_records=forget_records,
        retain_records=retain_records,
    )
    protocol.update(
        {
            "model_revision": cmp.MODEL_REVISION,
            "mcf_path": str(mcf_path),
            "mcf_sha256": cmp.sha256_file(mcf_path),
            "ours_artifact": str(ours_run / "fact_association_embeddings.pt"),
            "ours_artifact_sha256": cmp.sha256_file(
                ours_run / "fact_association_embeddings.pt"
            ),
            "primary_metric_version": "zerounlearn_answer_probability_v2",
            "final_evaluation_dtype": cmp.FINAL_EVAL_DTYPE,
            "same_raw_evaluator_as_full_comparison": True,
            "same_strict_summarizer_as_full_comparison": True,
            "zero_unlearn_not_executed_in_this_script": True,
        }
    )

    output_dir.mkdir(parents=True)
    (output_dir / "shared_protocol.json").write_text(
        json.dumps(protocol, indent=2, allow_nan=False) + "\n"
    )

    print("=== Evaluating frozen fact-association bank on matched MCF seed 1 ===", flush=True)
    model = cmp.load_base(model_path, dtype=torch.bfloat16)
    model, bank = cmp.load_artifact_into_model(model, artifact)
    model.eval()

    result = cmp.evaluate_one(
        method="FactAssociationBank",
        model=model,
        tokenizer=tokenizer,
        model_dir=ours_run,
        mcf_path=mcf_path,
        wikidata_dir=wikidata_dir,
        skip_ppl=args.skip_ppl,
    )
    result["fact_association_runtime"] = {
        "artifact": str(ours_run / "fact_association_embeddings.pt"),
        "layer": int(artifact["layer"]),
        "facts": len(artifact["facts"]),
        "runtime_counters": bank.counters(),
        "base_weights_edited": False,
        "tokenizer_extended": False,
        "fact_id_injection_used": False,
    }

    row = {"Method": "Ours", **cmp.compact_metrics(result)}

    (output_dir / "ours.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "ours_compact.json").write_text(
        json.dumps(row, indent=2, allow_nan=False) + "\n"
    )

    print(json.dumps(row, indent=2, allow_nan=False), flush=True)
    print(f"Saved full result: {output_dir / 'ours.json'}", flush=True)
    print(f"Saved compact row: {output_dir / 'ours_compact.json'}", flush=True)

    cmp.free_model(model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
