#!/usr/bin/env python3
"""Run MQuAKE SURE Stage-2 repair with ZeroUnlearn-style locked data access.

The input is the forget-only Stage-1 checkpoint plus the direct atomic rewrite
facts from the sampled 50 forget instances. Stage 2 never loads benchmark
retain instances, atomic natural-language questions, multi-hop questions, or
MQuAKE counterfactual target_new values.

The established SURE repair architecture is retained:
- freeze transformer and input embeddings;
- safely untie the output LM head if needed;
- identify still-correct sensitive-answer token decisions on direct rewrites;
- optimize only the ``Unknown`` output row;
- choose a BF16-safe scale using direct rewrite cases only;
- save the repaired checkpoint for a separate final official evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_gagd_setting5e_active_repair as mquake_sure
import mquake_zero_unlearn_official_eval as mquake
import zsre_gagd_setting5e_active_repair as repair


METHOD = "mquake_sure_forget_only_locked_active_repair"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50, help="MQuAKE instance count")
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


def load_locked_records(
    visible_path: Path,
    manifest_path: Path,
    forget_num: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data = json.loads(Path(visible_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise ValueError("repair-visible MQuAKE must be a JSON list")
    if manifest.get("protocol") != "mquake_zerounlearn_forget_only_locked_probes":
        raise ValueError("unexpected MQuAKE split protocol")
    if int(manifest.get("seed", -1)) != int(seed):
        raise ValueError("MQuAKE split seed does not match repair seed")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num_instances", -1)) != int(forget_num):
        raise ValueError("MQuAKE manifest forget instance count does not match")
    if len(data) != int(sampling.get("forget_atomic_fact_count", -1)):
        raise ValueError("MQuAKE visible atomic fact count does not match manifest")
    if len({int(row["source_index"]) for row in data}) != forget_num:
        raise ValueError("MQuAKE visible facts do not cover the requested forget instances")

    for row in data:
        forbidden = {
            "atomic_gen_prompt",
            "multihop_questions",
            "multihop_answer",
            "multihop_new_answer",
            "question",
            "questions",
        }
        leaked = forbidden.intersection(row)
        if leaked:
            raise RuntimeError(f"Stage 2 received held-out MQuAKE fields: {sorted(leaked)}")
        rewrite = row.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise RuntimeError("Stage 2 MQuAKE record lacks requested_rewrite")
        if "question" in rewrite or "mquake_target_new" in rewrite:
            raise RuntimeError("Stage 2 received held-out MQuAKE rewrite fields")
        if rewrite.get("target_new", {}).get("str") != mquake.NEUTRAL_TARGET:
            raise RuntimeError("Stage 2 neutral target changed from Unknown")
    return data, manifest


def validate_neutral_target_checkpoint(model_path: Path, tok: Any) -> None:
    meta_path = Path(model_path) / "mquake_neutral_target.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"locked MQuAKE Stage-1 checkpoint lacks {meta_path.name}: {model_path}"
        )
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    expected_id = mquake.resolve_neutral_target_token_id(tok)
    if payload.get("neutral_target") != mquake.NEUTRAL_TARGET:
        raise RuntimeError("MQuAKE checkpoint neutral target metadata changed")
    if int(payload.get("neutral_token_id", -1)) != expected_id:
        raise RuntimeError("MQuAKE checkpoint neutral token ID does not match tokenizer")


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
    repair.parse_candidate_scales(args.candidate_scales)


def main() -> None:
    args = parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    visible_path = Path(args.repair_visible_path).resolve()
    manifest_path = Path(args.split_manifest).resolve()
    records, split_manifest = load_locked_records(
        visible_path, manifest_path, args.forget_num, args.seed
    )
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
    validate_neutral_target_checkpoint(Path(args.model_path), tok)

    output_layer = active.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    device = gagd.first_device(model)
    llama_like = mquake.is_llama_like(model, tok)
    neutral_token_id = mquake.resolve_neutral_target_token_id(tok)
    original_neutral_row = output_weight[neutral_token_id].detach().clone()

    rewrite_cases = [
        case
        for record in records
        for case in mquake.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite",),
        )
    ]
    rewrite_caches = repair.cache_prediction_cases(
        model,
        tok,
        rewrite_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache locked MQuAKE rewrite tokens",
    )
    active_caches = [
        cache
        for cache in rewrite_caches
        if cache.correct and cache.target_token_id != neutral_token_id
    ]

    repair.write_jsonl(
        output_dir / "rewrite_tokens_before.jsonl",
        [repair.cache_report(cache) for cache in rewrite_caches],
    )
    repair.write_jsonl(
        output_dir / "active_tokens_before.jsonl",
        [repair.cache_report(cache) for cache in active_caches],
    )

    # With no benchmark-retain/protected data, protected projection is disabled.
    args.project_away_protected_hidden = False
    delta_rows, repair_logs, optimization = repair.optimize_neutral_delta(
        active_caches,
        [],
        hidden_size=output_weight.shape[1],
        device=device,
        args=args,
    )
    repair.write_jsonl(output_dir / "repair_log.jsonl", repair_logs)

    (
        selected_scale,
        scale_reports,
        exact_active_after,
        _exact_protected_after,
        zero_baseline,
    ) = repair.exact_bf16_scale_sweep(
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
        scales=repair.parse_candidate_scales(args.candidate_scales),
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        minimum_active_margin=args.selection_logit_margin,
    )
    write_json(output_dir / "bf16_exact_scale_sweep_rewrite_only.json", scale_reports)
    write_json(output_dir / "exact_zero_scale_baseline.json", zero_baseline)
    repair.write_jsonl(
        output_dir / "rewrite_tokens_after.jsonl",
        [repair.cache_report(cache) for cache in exact_active_after],
    )

    effective_delta = (
        output_weight[neutral_token_id].detach().float()
        - original_neutral_row.detach().float()
    )
    remaining_correct = int(sum(cache.correct for cache in exact_active_after))
    repair_summary: Dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD,
        "protocol_status": "mquake_zerounlearn_forget_only_locked_probes",
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "repair_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(args.seed),
        "forget_instances": int(args.forget_num),
        "forget_atomic_facts": len(records),
        "retain_instances_seen": 0,
        "repair_prompt_scope": "requested_rewrite_only",
        "atomic_questions_used_during_repair": False,
        "multihop_questions_used_during_repair": False,
        "benchmark_retain_used_during_repair": False,
        "benchmark_counterfactual_targets_used_during_repair": False,
        "transformer_parameters_trainable": 0,
        "input_embeddings_modified": False,
        "selected_lm_head_rows": 1 if torch.count_nonzero(effective_delta).item() else 0,
        "selected_lm_head_token_ids": [neutral_token_id],
        "selected_lm_head_token": mquake.NEUTRAL_TARGET,
        "effective_delta_shape": [1, int(output_weight.shape[1])],
        "effective_delta_norm": float(effective_delta.norm().cpu()),
        "repair_rank_requested": int(args.repair_rank),
        "active_rewrite_correct_tokens_before": len(active_caches),
        "rewrite_correct_tokens_after_selected_scale": remaining_correct,
        "selected_scale": float(selected_scale),
        "active_logit_margin": float(args.active_logit_margin),
        "selection_logit_margin": float(args.selection_logit_margin),
        "optimization": optimization,
        "selection_uses_atomic_or_multihop_questions": False,
        "selection_uses_benchmark_retain": False,
        "split_sampling": split_manifest.get("sampling"),
    }
    write_json(output_dir / "repair_summary.json", repair_summary)

    config = {
        **vars(args),
        "method": METHOD,
        "protocol": "mquake_zerounlearn_forget_only_locked_probes",
        "repair_prompt_scope": "requested_rewrite_only",
        "benchmark_retain_instances_used_during_repair": 0,
        "atomic_questions_used_during_repair": False,
        "multihop_questions_used_during_repair": False,
        "benchmark_counterfactual_targets_used_during_repair": False,
        "neutral_token_id": neutral_token_id,
        "selected_scale": float(selected_scale),
        "selection_uses_heldout_questions_or_retain": False,
    }
    write_json(output_dir / "config_used.json", config)

    mquake_sure.save_checkpoint(model, tok, checkpoint_dir)
    print(f"MQuAKE locked active-repair checkpoint: {checkpoint_dir}")
    print(
        f"visible rewrite correct tokens: {len(active_caches)} -> {remaining_correct}; "
        f"selected scale={selected_scale:g}"
    )
    print(
        f"Stage 2 data access: {args.forget_num} forget instances / "
        f"{len(records)} atomic rewrites; 0 retain/questions"
    )


if __name__ == "__main__":
    main()
