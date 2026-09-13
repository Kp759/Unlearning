#!/usr/bin/env python3
"""Exact frozen-base comparator for the ZsRE fact-association seed-1 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import zsre_zero_unlearn_official_eval as zsre
from evaluate_zsre_fact_association_embeddings_official import (
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
    p.add_argument("--zsre-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    reference = Path(args.reference_run_dir).resolve()
    manifest = json.loads(
        (reference / "association_manifest.json").read_text()
    )
    if int(manifest.get("seed", -1)) != 1:
        raise ValueError("ZsRE frozen comparator requires reference seed 1")
    if int(manifest.get("forget_num", -1)) != 50:
        raise ValueError("ZsRE frozen comparator requires reference forget50")

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

    forget_records, retain_records = zsre.load_official_eval_records(
        Path(args.zsre_path),
        tok,
        forget_num=50,
        retain_num=1000,
        seed=1,
    )
    observed_forget = [int(record["case_id"]) for record in forget_records]
    observed_retain = [int(record["case_id"]) for record in retain_records]
    expected_forget = [
        int(value) for value in manifest.get("forget_case_ids", [])
    ]
    expected_retain = [
        int(value)
        for value in manifest.get("retain_case_ids_final_evaluation", [])
    ]
    if expected_forget and observed_forget != expected_forget:
        raise RuntimeError(
            "ZsRE base comparator forget split differs from the architecture run"
        )
    if expected_retain and observed_retain != expected_retain:
        raise RuntimeError(
            "ZsRE base comparator retain split differs from the architecture run"
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
    llama_like = zsre.is_llama_like(model, tok)

    forget_summary, forget_raw, forget_predictions, forget_routes = (
        evaluate_split_fixed_boundary(
            model,
            bank,
            tok,
            forget_records,
            device,
            llama_like=llama_like,
            split_name="forget",
            batch_size=args.batch_size,
        )
    )
    retain_summary, retain_raw, retain_predictions, retain_routes = (
        evaluate_split_fixed_boundary(
            model,
            bank,
            tok,
            retain_records,
            device,
            llama_like=llama_like,
            split_name="retain",
            batch_size=args.batch_size,
        )
    )

    runtime_ppl = None
    legacy_ppl = None
    if not args.skip_ppl:
        text = load_official_ppl_text(args.wikidata_dir)
        if text is not None:
            legacy_ppl = official_perplexity(
                model, tok, text, device, max_input_length=100
            )
            runtime_ppl = runtime_aligned_perplexity(
                model, tok, text, device, max_input_length=100
            )

    result = {
        "method": "FrozenBase",
        "dataset": "ZsRE",
        "protocol": "ZeroUnlearn official-compatible token accuracy",
        "seed": 1,
        "unlearn_num": 50,
        "retain_num": 1000,
        "forget": forget_summary,
        "retain": retain_summary,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
        "forget_prediction_rows": forget_predictions,
        "retain_prediction_rows": retain_predictions,
        "forget_route_summary": forget_routes,
        "retain_route_summary": retain_routes,
        "runtime_aligned_PPL": runtime_ppl,
        "legacy_PPL": legacy_ppl,
        "comparison_contract": {
            "reference_run_dir": str(reference),
            "same_model_checkpoint": True,
            "same_seed": 1,
            "same_forget_case_ids_verified": True,
            "same_retain_case_ids_verified": True,
            "same_prediction_case_expansion": True,
            "same_fixed_boundary_evaluator": True,
            "residual_bank_loaded": False,
            "base_weights_edited": False,
        },
        "route_activity": bank.counters(),
    }

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite ZsRE base result: {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(json.dumps({
        "forget_Eff": forget_summary["Eff"],
        "forget_Gen": forget_summary["Gen"],
        "forget_Spe": forget_summary["Spe"],
        "retain_Eff": retain_summary["Eff"],
        "retain_Gen": retain_summary["Gen"],
        "retain_Spe": retain_summary["Spe"],
        "runtime_aligned_PPL": (
            None if runtime_ppl is None else runtime_ppl["ppl"]
        ),
        "legacy_PPL": legacy_ppl,
        "out": str(out),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
