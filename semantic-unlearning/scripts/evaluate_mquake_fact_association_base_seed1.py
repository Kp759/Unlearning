#!/usr/bin/env python3
"""Exact frozen-base comparator for the MQuAKE fact-association seed-1 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import mquake_zero_unlearn_official_eval as mquake
from evaluate_mquake_fact_association_embeddings_official import (
    evaluate_split_fixed_boundary,
)
from frozen_base_fact_association_compat import (
    FrozenBaseAssociationCompatLM,
    NoRouteBatchBank,
)
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
    runtime_aligned_perplexity,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run-dir", required=True)
    p.add_argument("--mquake-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--skip-atomic-gen", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    reference = Path(args.reference_run_dir).resolve()
    manifest = json.loads(
        (reference / "association_manifest.json").read_text()
    )
    if int(manifest.get("seed", -1)) != 1:
        raise ValueError("MQuAKE frozen comparator requires reference seed 1")
    if int(manifest.get("forget_num_instances", -1)) != 50:
        raise ValueError("MQuAKE frozen comparator requires reference forget50")
    if int(
        manifest.get("retain_num_instances_final_evaluation", -1)
    ) != 1000:
        raise ValueError("MQuAKE frozen comparator requires reference retain1000")

    model_path = Path(manifest["model_path"]).resolve()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    forget_records, retain_records = mquake.load_official_eval_records(
        Path(args.mquake_path).resolve(),
        tok,
        forget_num=50,
        retain_num=1000,
        seed=1,
    )
    observed_forget_case_ids = [
        int(record["case_id"]) for record in forget_records
    ]
    expected_forget_case_ids = [
        int(value)
        for value in manifest.get("forget_atomic_case_ids", [])
    ]
    if (
        expected_forget_case_ids
        and observed_forget_case_ids != expected_forget_case_ids
    ):
        raise RuntimeError(
            "MQuAKE base comparator atomic forget records differ from the "
            "architecture seed-1 run"
        )
    if int(manifest.get("forget_atomic_record_count", -1)) != len(
        forget_records
    ):
        raise RuntimeError(
            "MQuAKE reference atomic forget count differs from evaluator split"
        )

    dtype = dtype_from_str(args.dtype)
    raw = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    ).to(args.device).eval()
    raw.requires_grad_(False)
    raw.config.use_cache = False
    bank = NoRouteBatchBank()
    model = FrozenBaseAssociationCompatLM(raw, bank).to(args.device).eval()
    device = next(model.parameters()).device
    llama_like = mquake.is_llama_like(model, tok)
    include_atomic_gen = not args.skip_atomic_gen

    forget_summary, forget_raw, forget_routes = evaluate_split_fixed_boundary(
        model,
        bank,
        tok,
        forget_records,
        device,
        llama_like=llama_like,
        split_name="forget",
        batch_size=args.batch_size,
        include_atomic_gen=include_atomic_gen,
    )
    retain_summary, retain_raw, retain_routes = evaluate_split_fixed_boundary(
        model,
        bank,
        tok,
        retain_records,
        device,
        llama_like=llama_like,
        split_name="retain",
        batch_size=args.batch_size,
        include_atomic_gen=include_atomic_gen,
    )

    legacy_ppl = None
    runtime_ppl = None
    if not args.skip_ppl:
        text = load_official_ppl_text(args.wikidata_dir)
        if text is not None:
            legacy_ppl = official_perplexity(
                model, tok, text, device, max_input_length=100
            )
            runtime = runtime_aligned_perplexity(
                model, tok, text, device, max_input_length=100
            )
            runtime_ppl = runtime["ppl"]

    result = {
        "method": "FrozenBase",
        "dataset": mquake.MQUAKE_FILENAME,
        "dataset_revision": mquake.MQUAKE_REV,
        "seed": 1,
        "forget_num_instances": 50,
        "retain_num_instances": 1000,
        "forget_atomic_record_count": len(forget_records),
        "retain_atomic_record_count": len(retain_records),
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_routes": forget_routes,
        "retain_routes": retain_routes,
        "legacy_PPL": legacy_ppl,
        "runtime_aligned_PPL": runtime_ppl,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
        "comparison_contract": {
            "reference_run_dir": str(reference),
            "same_model_checkpoint": True,
            "same_seed": 1,
            "same_forget_atomic_case_ids_verified": True,
            "same_official_instance_sampling": True,
            "same_atomic_case_expansion": True,
            "same_fixed_boundary_evaluator": True,
            "residual_bank_loaded": False,
            "base_weights_edited": False,
        },
        "route_activity": bank.counters(),
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(
            f"Refusing to overwrite MQuAKE base result: {out}"
        )
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(json.dumps({
        "forget_Eff": forget_summary.get("Eff"),
        "forget_AtomicGen": forget_summary.get("AtomicGen"),
        "retain_Eff": retain_summary.get("Eff"),
        "retain_AtomicGen": retain_summary.get("AtomicGen"),
        "runtime_aligned_PPL": runtime_ppl,
        "legacy_PPL": legacy_ppl,
        "out": str(out),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
