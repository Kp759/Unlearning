#!/usr/bin/env python3
"""Run ZsRE SURE Stage-2 repair with ZeroUnlearn-style locked data access.

The input is the forget-only Stage-1 checkpoint plus the repair-visible 50
forget records.  Stage 2 sees direct requested_rewrite prompts only.  It never
loads benchmark retain records, rephrases, or locality probes.

The architecture stays the established ZsRE SURE repair:
- freeze transformer and input embeddings;
- safely untie the output LM head if needed;
- identify still-correct direct sensitive-answer token decisions;
- optimize only the ``Unknown`` output row;
- choose a BF16-safe scale using direct rewrite cases only;
- save the frozen repaired checkpoint for a separate final official eval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch

import gagd_active_case_repair as active
import gagd_compare as gagd
import zsre_gagd_setting5e_active_repair as zsre_sure
import zsre_zero_unlearn_official_eval as zsre


METHOD = "zsre_sure_forget_only_locked_active_repair"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--repair-optimizer", choices=["sgd", "adam", "adamw"], default="adamw")
    p.add_argument("--active-logit-margin", type=float, default=0.25)
    p.add_argument("--selection-logit-margin", type=float, default=0.05)
    p.add_argument("--repair-rank", type=int, default=0)
    p.add_argument("--repair-l2-lambda", type=float, default=1e-6)
    p.add_argument("--max-delta-norm", type=float, default=None)
    p.add_argument("--candidate-scales", default=(
        "1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,"
        "0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0"
    ))
    p.add_argument("--stop-when-all-satisfied", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--cache-batch-size", type=int, default=8)
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_locked_records(path: Path, forget_num: int) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("repair-visible ZsRE must be a JSON list")
    if len(data) != forget_num:
        raise ValueError(f"expected {forget_num} forget records, got {len(data)}")
    for row in data:
        if row.get("paraphrase_prompts"):
            raise RuntimeError("Stage 2 received locked ZsRE paraphrases")
        if row.get("neighborhood_prompts"):
            raise RuntimeError("Stage 2 received locked ZsRE locality prompts")
        if row.get("requested_rewrite", {}).get("target_new", {}).get("str") != zsre.NEUTRAL_TARGET:
            raise RuntimeError("Stage 2 neutral target changed from Unknown")
    return data


def validate_args(args: argparse.Namespace) -> None:
    if args.forget_num <= 0:
        raise ValueError("forget-num must be positive")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("repair steps/lr must be positive")
    if args.active_logit_margin < 0 or args.selection_logit_margin < 0:
        raise ValueError("repair margins must be non-negative")
    if args.repair_rank < 0 or args.repair_l2_lambda < 0:
        raise ValueError("repair rank/L2 must be non-negative")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("max delta norm must be finite and non-negative")
    if args.cache_batch_size <= 0:
        raise ValueError("cache batch size must be positive")
    zsre_sure.parse_candidate_scales(args.candidate_scales)


def main() -> None:
    args = parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    visible_path = Path(args.repair_visible_path).resolve()
    records = load_locked_records(visible_path, args.forget_num)
    output_dir = gagd.resolve_output_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    zsre_sure.validate_neutral_target_checkpoint(Path(args.model_path), tok)

    output_layer = active.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    device = gagd.first_device(model)
    llama_like = zsre.is_llama_like(model, tok)
    neutral_token_id = zsre.resolve_neutral_target_token_id(tok)
    original_neutral_row = output_weight[neutral_token_id].detach().clone()

    rewrite_cases = [
        case
        for record in records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]
    rewrite_caches = zsre_sure.cache_prediction_cases(
        model,
        tok,
        rewrite_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache locked ZsRE rewrite tokens",
    )
    active_caches = [
        cache
        for cache in rewrite_caches
        if cache.correct and cache.target_token_id != neutral_token_id
    ]

    zsre_sure.write_jsonl(
        output_dir / "rewrite_tokens_before.jsonl",
        [zsre_sure.cache_report(cache) for cache in rewrite_caches],
    )
    zsre_sure.write_jsonl(
        output_dir / "active_tokens_before.jsonl",
        [zsre_sure.cache_report(cache) for cache in active_caches],
    )

    # Existing optimizer helper expects these fields.  With no protected data,
    # the protected projection is deliberately disabled.
    args.project_away_protected_hidden = False
    delta_rows, repair_logs, optimization = zsre_sure.optimize_neutral_delta(
        active_caches,
        [],
        hidden_size=output_weight.shape[1],
        device=device,
        args=args,
    )
    zsre_sure.write_jsonl(output_dir / "repair_log.jsonl", repair_logs)

    (
        selected_scale,
        scale_reports,
        exact_active_after,
        _exact_protected_after,
        zero_baseline,
    ) = zsre_sure.exact_bf16_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_weight,
        neutral_token_id=neutral_token_id,
        original_neutral_row=original_neutral_row,
        delta_row=delta_rows[0],
        active_cases=rewrite_cases,
        protected_cases=[],
        active_context_cases=rewrite_cases,
        protected_context_cases=[],
        scales=zsre_sure.parse_candidate_scales(args.candidate_scales),
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        minimum_active_margin=args.selection_logit_margin,
    )
    write_json(output_dir / "bf16_exact_scale_sweep_rewrite_only.json", scale_reports)
    write_json(output_dir / "exact_zero_scale_baseline.json", zero_baseline)
    zsre_sure.write_jsonl(
        output_dir / "rewrite_tokens_after.jsonl",
        [zsre_sure.cache_report(cache) for cache in exact_active_after],
    )

    effective_delta = (
        output_weight[neutral_token_id].detach().float()
        - original_neutral_row.detach().float()
    )
    remaining_correct = int(sum(cache.correct for cache in exact_active_after))
    repair_summary: Dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD,
        "protocol_status": "zsre_zerounlearn_forget_only_locked_probes",
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "repair_visible_path": str(visible_path),
        "seed": int(args.seed),
        "forget_records": len(records),
        "retain_records_seen": 0,
        "repair_prompt_scope": "requested_rewrite_only",
        "paraphrases_used_during_repair": False,
        "locality_used_during_repair": False,
        "benchmark_retain_used_during_repair": False,
        "transformer_parameters_trainable": 0,
        "input_embeddings_modified": False,
        "selected_lm_head_rows": 1 if torch.count_nonzero(effective_delta).item() else 0,
        "selected_lm_head_token_ids": [neutral_token_id],
        "selected_lm_head_token": zsre.NEUTRAL_TARGET,
        "effective_delta_shape": [1, int(output_weight.shape[1])],
        "effective_delta_norm": float(effective_delta.norm().cpu()),
        "repair_rank_requested": int(args.repair_rank),
        "active_rewrite_correct_tokens_before": len(active_caches),
        "rewrite_correct_tokens_after_selected_scale": remaining_correct,
        "selected_scale": float(selected_scale),
        "active_logit_margin": float(args.active_logit_margin),
        "selection_logit_margin": float(args.selection_logit_margin),
        "optimization": optimization,
        "selection_uses_heldout_gen_or_spe": False,
    }
    write_json(output_dir / "repair_summary.json", repair_summary)

    config = {
        **vars(args),
        "method": METHOD,
        "protocol": "zsre_zerounlearn_forget_only_locked_probes",
        "repair_prompt_scope": "requested_rewrite_only",
        "benchmark_retain_examples_used_during_repair": 0,
        "paraphrases_used_during_repair": False,
        "locality_used_during_repair": False,
        "neutral_token_id": neutral_token_id,
        "selected_scale": float(selected_scale),
        "selection_uses_heldout_gen_or_spe": False,
    }
    write_json(output_dir / "config_used.json", config)

    zsre_sure.save_checkpoint(model, tok, checkpoint_dir)
    print(f"ZsRE locked active-repair checkpoint: {checkpoint_dir}")
    print(
        f"visible rewrite correct tokens: {len(active_caches)} -> {remaining_correct}; "
        f"selected scale={selected_scale:g}"
    )
    print("Stage 2 data access: 50 direct forget records; 0 retain/rephrase/locality")


if __name__ == "__main__":
    main()
