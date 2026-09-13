#!/usr/bin/env python3
"""Exact frozen-base comparator for the MCF fact-association seed-1 run."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import torch

from frozen_base_fact_association_compat import (
    FrozenBaseAssociationCompatLM,
    NoRouteBatchBank,
)
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    evaluate_loaded_model_official,
    load_official_eval_records,
)
from mcf_zero_unlearn_metric_parity import summarize_probability_metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-run-dir", required=True)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    reference = Path(args.reference_run_dir).resolve()
    manifest = json.loads(
        (reference / "association_manifest.json").read_text()
    )
    sampling = manifest.get("sampling", {})
    if int(sampling.get("seed", -1)) != 1:
        raise ValueError("MCF frozen comparator requires reference seed 1")
    if int(sampling.get("forget_num", -1)) != 50:
        raise ValueError("MCF frozen comparator requires reference forget50")

    model_path = Path(manifest["model_path"]).resolve()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget_records, _ = load_official_eval_records(
        mcf_path=args.mcf_path,
        unlearn_num=50,
        retain_num=1000,
        seed=1,
        sample_mode="official",
    )
    observed_forget = [int(record["case_id"]) for record in forget_records]
    expected_forget = [
        int(value) for value in manifest.get("forget_case_ids", [])
    ]
    if expected_forget and observed_forget != expected_forget:
        raise RuntimeError(
            "MCF base comparator split differs from the seed-1 architecture run"
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

    result = evaluate_loaded_model_official(
        method="FrozenBase",
        model=model,
        tok=tok,
        model_dir=model_path,
        mcf_path=args.mcf_path,
        wikidata_dir=args.wikidata_dir,
        out_path=None,
        unlearn_num=50,
        retain_num=1000,
        seed=1,
        sample_mode="official",
        skip_ppl=args.skip_ppl,
    )

    legacy_counterfact = {
        split: deepcopy(result[split])
        for split in ("forget", "retain")
    }
    for split in ("forget", "retain"):
        result[split] = summarize_probability_metrics(
            result[split], result[f"{split}_raw"]
        )
    result["legacy_counterfact"] = legacy_counterfact
    result["metric_version"] = "zerounlearn_answer_probability_v2"
    result["comparison_contract"] = {
        "reference_run_dir": str(reference),
        "same_model_checkpoint": True,
        "same_official_sampling": True,
        "same_seed": 1,
        "same_forget_case_ids_verified": True,
        "same_evaluator_core": "evaluate_loaded_model_official",
        "residual_bank_loaded": False,
        "base_weights_edited": False,
    }
    result["route_activity"] = bank.counters()

    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite MCF base result: {out}")
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    print(json.dumps({
        "metric_version": result["metric_version"],
        "forget_Eff": result["forget"]["Eff"],
        "forget_Gen": result["forget"]["Gen"],
        "forget_Spe": result["forget"]["Spe"],
        "forget_ReleasedAccuracy_Eff": result["forget"]["ReleasedAccuracy_Eff"],
        "forget_ReleasedAccuracy_Gen": result["forget"]["ReleasedAccuracy_Gen"],
        "retain_Eff": result["retain"]["Eff"],
        "retain_Gen": result["retain"]["Gen"],
        "retain_Spe": result["retain"]["Spe"],
        "runtime_aligned_PPL": result.get("forget_PPL"),
        "legacy_PPL": result.get("legacy_forget_PPL"),
        "out": str(out),
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
