#!/usr/bin/env python3
"""Apply an exact BF16-safe active EOS-row repair to a saved ZsRE Setting 5e checkpoint.

Unlike the original combined runner, this staged repair tries every configured
scale after real model materialization and exact token re-evaluation. It accepts
only a nonzero scale that reduces residual forget-token accuracy, introduces no
protected calibration regressions, and passes the official ZsRE utility gates.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import zsre_gagd_setting5e_active_repair as repair
import zsre_zero_unlearn_official_eval as zsre


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting5-checkpoint", required=True)
    parser.add_argument(
        "--output-dir",
        default="outputs/zsre_ultra600_setting5e_active_seed0/seed0/bf16_safe_repair",
    )
    parser.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    parser.add_argument("--zsre-url", default=zsre.ZSRE_URL)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)

    parser.add_argument("--repair-steps", type=int, default=800)
    parser.add_argument("--repair-lr", type=float, default=5e-3)
    parser.add_argument(
        "--repair-optimizer", choices=["sgd", "adam", "adamw"], default="adamw"
    )
    parser.add_argument("--active-logit-margin", type=float, default=0.10)
    parser.add_argument("--repair-rank", type=int, default=64)
    parser.add_argument("--repair-l2-lambda", type=float, default=1e-6)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--retain-calibration-num", type=int, default=128)
    parser.add_argument("--retain-calibration-seed", type=int, default=1729)
    parser.add_argument(
        "--project-away-protected-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate-scales",
        default=(
            "1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,"
            "0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0"
        ),
    )

    parser.add_argument("--utility-drop-tolerance", type=float, default=0.10)
    parser.add_argument("--max-ppl-ratio", type=float, default=1.02)
    parser.add_argument(
        "--strict-utility-gates",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument(
        "--save-selected-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def resolve(path: str) -> Path:
    return gagd.resolve_output_path(path)


def load_model(args: argparse.Namespace) -> Tuple[torch.nn.Module, Any]:
    checkpoint = resolve(args.setting5_checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Setting 5e checkpoint does not exist: {checkpoint}")
    kwargs: Dict[str, Any] = {"dtype": torch_dtype(args.dtype)}
    if args.device_map == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **kwargs)
    if args.device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")
    tok = AutoTokenizer.from_pretrained(str(checkpoint))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.config.use_cache = False
    model.eval()
    return model, tok


def exact_scale_sweep(
    *,
    model: torch.nn.Module,
    tok: Any,
    output_weight: torch.Tensor,
    neutral_token_id: int,
    original_neutral_row: torch.Tensor,
    delta_row: torch.Tensor,
    active_caches: Sequence[repair.TokenLogitCache],
    protected_caches: Sequence[repair.TokenLogitCache],
    scales: Sequence[float],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> Tuple[float, List[Dict[str, Any]], List[repair.TokenLogitCache], List[repair.TokenLogitCache]]:
    reports: List[Dict[str, Any]] = []
    exact_by_scale: Dict[float, Tuple[List[repair.TokenLogitCache], List[repair.TokenLogitCache]]] = {}

    active_cases = [cache.case for cache in active_caches]
    protected_cases = [cache.case for cache in protected_caches]

    for scale in scales:
        repair.materialize_output_row(
            output_weight,
            neutral_token_id,
            original_neutral_row,
            delta_row,
            float(scale),
        )
        exact_active = repair.cache_prediction_cases(
            model,
            tok,
            active_cases,
            neutral_token_id=neutral_token_id,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
            desc=f"exact BF16 active scale={scale:g}",
        )
        exact_protected = repair.cache_prediction_cases(
            model,
            tok,
            protected_cases,
            neutral_token_id=neutral_token_id,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
            desc=f"exact BF16 protected scale={scale:g}",
        )
        active_correct = int(sum(cache.correct for cache in exact_active))
        protected_regressions = int(sum(not cache.correct for cache in exact_protected))
        effective_delta = (
            output_weight[neutral_token_id].detach().float()
            - original_neutral_row.detach().float()
        )
        report = {
            "scale": float(scale),
            "active_correct_tokens": active_correct,
            "active_repaired_tokens": int(len(active_caches) - active_correct),
            "protected_regressions": protected_regressions,
            "materialized_delta_norm": float(effective_delta.norm().cpu()),
            "nonzero_materialized_delta": bool(torch.count_nonzero(effective_delta).item()),
        }
        reports.append(report)
        exact_by_scale[float(scale)] = (exact_active, exact_protected)

    safe = [
        item
        for item in reports
        if item["protected_regressions"] == 0
        and item["scale"] > 0.0
        and item["nonzero_materialized_delta"]
        and item["active_correct_tokens"] < len(active_caches)
    ]
    if safe:
        best = min(
            safe,
            key=lambda item: (
                int(item["active_correct_tokens"]),
                -float(item["scale"]),
                float(item["materialized_delta_norm"]),
            ),
        )
    else:
        best = next(item for item in reports if item["scale"] == 0.0)

    selected_scale = float(best["scale"])
    repair.materialize_output_row(
        output_weight,
        neutral_token_id,
        original_neutral_row,
        delta_row,
        selected_scale,
    )
    exact_active, exact_protected = exact_by_scale[selected_scale]
    return selected_scale, reports, exact_active, exact_protected


def main() -> None:
    args = build_parser().parse_args()
    gagd.set_seed(args.seed)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repair_dir = output_dir / "active_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args)
    device = next(model.parameters()).device
    llama_like = zsre.is_llama_like(model, tok)
    neutral_token_id = int(tok.eos_token_id)

    zsre_path = resolve(args.zsre_path)
    zsre_path = zsre.download_zsre(zsre_path, url=args.zsre_url)
    forget_records, retain_records = zsre.load_official_eval_records(
        zsre_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        zsre_url=args.zsre_url,
    )
    records = (forget_records, retain_records)

    print("Evaluating saved 600-step Setting 5e checkpoint")
    setting5_result = zsre.evaluate_loaded_model_official(
        method="Ultra-aggressive Setting 5e (600 steps)",
        model=model,
        tok=tok,
        model_dir=args.setting5_checkpoint,
        zsre_path=zsre_path,
        wikidata_dir=resolve(args.wikidata_dir),
        out_path=output_dir / "setting5e_official_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=args.skip_ppl,
        zsre_url=args.zsre_url,
        records=records,
    )

    output_layer = active.freeze_model_for_output_repair(model)
    original_neutral_row = output_layer.weight[neutral_token_id].detach().clone()

    forget_active_cases = [
        case
        for record in forget_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite", "paraphrase"),
        )
    ]
    forget_neighborhood_cases = [
        case
        for record in forget_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
            prompt_types=("neighborhood",),
        )
    ]
    calibration_records = repair._sample_retain_records(
        retain_records,
        args.retain_calibration_num,
        args.retain_calibration_seed,
    )
    retain_protected_cases = [
        case
        for record in calibration_records
        for case in zsre.expand_prediction_cases(record, tok, llama_like=llama_like)
    ]

    active_all = repair.cache_prediction_cases(
        model,
        tok,
        forget_active_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache active ZsRE tokens",
    )
    protected_all = repair.cache_prediction_cases(
        model,
        tok,
        forget_neighborhood_cases + retain_protected_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache protected ZsRE tokens",
    )
    active_caches = [
        cache
        for cache in active_all
        if cache.correct and cache.target_token_id != neutral_token_id
    ]
    protected_caches = [
        cache
        for cache in protected_all
        if cache.correct and cache.target_token_id != neutral_token_id
    ]

    print(
        f"Optimizing EOS row {neutral_token_id} ({tok.eos_token!r}): "
        f"active={len(active_caches)}, protected={len(protected_caches)}"
    )
    delta_rows, repair_logs, optimization = repair.optimize_eos_delta(
        active_caches,
        protected_caches,
        hidden_size=output_layer.weight.shape[1],
        device=device,
        args=args,
    )
    repair.write_jsonl(repair_dir / "repair_log.jsonl", repair_logs)

    scales = repair.parse_candidate_scales(args.candidate_scales)
    selected_scale, scale_reports, exact_active, exact_protected = exact_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_layer.weight,
        neutral_token_id=neutral_token_id,
        original_neutral_row=original_neutral_row,
        delta_row=delta_rows[0],
        active_caches=active_caches,
        protected_caches=protected_caches,
        scales=scales,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    gagd.write_json(repair_dir / "bf16_exact_scale_sweep.json", scale_reports)
    repair.write_jsonl(
        repair_dir / "active_tokens_after.jsonl",
        [repair.cache_report(cache) for cache in exact_active],
    )
    repair.write_jsonl(
        repair_dir / "protected_tokens_after.jsonl",
        [repair.cache_report(cache) for cache in exact_protected],
    )

    candidate_result = zsre.evaluate_loaded_model_official(
        method="Setting 5e + BF16-safe active LM-head repair candidate",
        model=model,
        tok=tok,
        model_dir="in-memory:bf16_safe_active_repair",
        zsre_path=zsre_path,
        wikidata_dir=resolve(args.wikidata_dir),
        out_path=repair_dir / "candidate_official_eval.json",
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        batch_size=args.eval_batch_size,
        skip_ppl=args.skip_ppl,
        zsre_url=args.zsre_url,
        records=records,
    )
    gate_report = repair.metric_gate_report(
        setting5_result,
        candidate_result,
        utility_drop_tolerance=args.utility_drop_tolerance,
        max_ppl_ratio=args.max_ppl_ratio,
    )

    active_before = len(active_caches)
    active_after = int(sum(cache.correct for cache in exact_active))
    protected_regressions = int(sum(not cache.correct for cache in exact_protected))
    local_success = bool(
        selected_scale > 0.0
        and active_after < active_before
        and protected_regressions == 0
    )
    accepted = bool(
        local_success and (gate_report["passed"] or not args.strict_utility_gates)
    )

    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        selected_result["method"] = "Setting 5e + BF16-safe active LM-head repair"
        selection_reason = "nonzero_repair_passed_local_and_official_gates"
    else:
        with torch.no_grad():
            output_layer.weight[neutral_token_id].copy_(original_neutral_row)
        selected_result = copy.deepcopy(setting5_result)
        selected_result["method"] = "Ultra-aggressive Setting 5e (repair fallback)"
        selection_reason = (
            "no_nonzero_bf16_safe_improving_scale"
            if not local_success
            else "nonzero_repair_failed_official_metric_gates"
        )

    effective_delta = (
        output_layer.weight[neutral_token_id].detach().float()
        - original_neutral_row.detach().float()
    )
    summary = {
        "method": "zsre_bf16_safe_active_repair",
        "active_tokens_before": active_before,
        "active_tokens_after": active_after if accepted else active_before,
        "active_tokens_repaired": (active_before - active_after) if accepted else 0,
        "protected_tokens": len(protected_caches),
        "protected_regressions_after": protected_regressions if accepted else 0,
        "selected_scale": selected_scale if accepted else 0.0,
        "materialized_delta_norm": float(effective_delta.norm().cpu()),
        "active_repair_applied": accepted,
        "fallback_to_setting5e": not accepted,
        "candidate_accepted": accepted,
        "selection_reason": selection_reason,
        "optimization": optimization,
        "official_metric_gates": gate_report,
    }
    gagd.write_json(repair_dir / "repair_summary.json", summary)

    rows = [
        repair.comparison_row("Ultra-aggressive Setting 5e", setting5_result),
        repair.comparison_row("BF16-safe active candidate", candidate_result),
        repair.comparison_row("Selected", selected_result),
    ]
    repair.write_comparison(output_dir, rows)
    gagd.write_json(
        output_dir / "zsre_bf16_safe_results.json",
        {
            "dataset": "ZsRE",
            "seed": args.seed,
            "repair": summary,
            "setting5e": repair.compact_metrics(setting5_result),
            "candidate": repair.compact_metrics(candidate_result),
            "selected": repair.compact_metrics(selected_result),
        },
    )

    if args.save_selected_checkpoint:
        repair.save_checkpoint(model, tok, output_dir / "selected_checkpoint")

    print(
        "Selected ZsRE result: "
        f"Eff={selected_result['forget']['Eff']}, "
        f"Gen={selected_result['forget']['Gen']}, "
        f"Spe={selected_result['forget']['Spe']}, "
        f"PPL={selected_result.get('forget_PPL')}; "
        f"active_repair_applied={accepted}; selected_scale={summary['selected_scale']}"
    )
    print(f"Comparison: {output_dir / 'comparison.md'}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
