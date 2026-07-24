#!/usr/bin/env python3
"""BF16-safe ZsRE active repair with exact zero-scale baseline comparison.

The previous staged runner counted absolute protected-token errors after
re-batching. A protected token can flip even at scale 0 because BF16 logits are
near-tied and the exact verification pass uses different batch composition from
the initial cache. This runner defines a regression as a token that is correct
in the exact scale-0 BF16 baseline and becomes incorrect at a nonzero scale.

A repair is accepted only when it:
  * uses a genuinely nonzero materialized EOS-row delta;
  * reduces exact active-token correctness relative to scale 0;
  * introduces zero additional protected regressions relative to scale 0; and
  * passes the official ZsRE utility gates.
"""

from __future__ import annotations

import argparse
import copy
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
        default="outputs/zsre_ultra600_setting5e_bf16_safe_repair_v2_seed0",
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
        "--repair-optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adamw",
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


def dtype_for_name(name: str) -> torch.dtype:
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

    kwargs: Dict[str, Any] = {"dtype": dtype_for_name(args.dtype)}
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


def exact_predictions(
    *,
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[zsre.PredictionCase],
    neutral_token_id: int,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    desc: str,
) -> List[repair.TokenLogitCache]:
    return repair.cache_prediction_cases(
        model,
        tok,
        cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=batch_size,
        desc=desc,
    )


def exact_scale_sweep(
    *,
    model: torch.nn.Module,
    tok: Any,
    output_weight: torch.Tensor,
    neutral_token_id: int,
    original_neutral_row: torch.Tensor,
    delta_row: torch.Tensor,
    active_cases: Sequence[zsre.PredictionCase],
    protected_cases: Sequence[zsre.PredictionCase],
    scales: Sequence[float],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> Tuple[
    float,
    List[Dict[str, Any]],
    List[repair.TokenLogitCache],
    List[repair.TokenLogitCache],
    Dict[str, Any],
]:
    exact_by_scale: Dict[
        float,
        Tuple[List[repair.TokenLogitCache], List[repair.TokenLogitCache], float, bool],
    ] = {}

    # Materialize and evaluate every requested scale with the same exact case lists
    # and batch size. This makes scale 0 the numerical BF16 reference condition.
    for scale in scales:
        repair.materialize_output_row(
            output_weight,
            neutral_token_id,
            original_neutral_row,
            delta_row,
            float(scale),
        )
        active_rows = exact_predictions(
            model=model,
            tok=tok,
            cases=active_cases,
            neutral_token_id=neutral_token_id,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
            desc=f"exact BF16 active scale={scale:g}",
        )
        protected_rows = exact_predictions(
            model=model,
            tok=tok,
            cases=protected_cases,
            neutral_token_id=neutral_token_id,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
            desc=f"exact BF16 protected scale={scale:g}",
        )
        effective_delta = (
            output_weight[neutral_token_id].detach().float()
            - original_neutral_row.detach().float()
        )
        delta_norm = float(effective_delta.norm().cpu())
        nonzero = bool(torch.count_nonzero(effective_delta).item())
        exact_by_scale[float(scale)] = (
            active_rows,
            protected_rows,
            delta_norm,
            nonzero,
        )

    if 0.0 not in exact_by_scale:
        raise RuntimeError("Candidate scale sweep must include scale 0.0")

    zero_active, zero_protected, _, _ = exact_by_scale[0.0]
    zero_active_correct = [bool(row.correct) for row in zero_active]
    zero_protected_correct = [bool(row.correct) for row in zero_protected]
    zero_active_count = int(sum(zero_active_correct))
    zero_protected_incorrect = int(len(zero_protected_correct) - sum(zero_protected_correct))

    reports: List[Dict[str, Any]] = []
    for scale in scales:
        active_rows, protected_rows, delta_norm, nonzero = exact_by_scale[float(scale)]
        active_correct = [bool(row.correct) for row in active_rows]
        protected_correct = [bool(row.correct) for row in protected_rows]

        # Only count newly broken tokens that were correct in the exact scale-0
        # baseline. A token already incorrect at scale 0 is numerical baseline
        # instability, not damage caused by a nonzero repair.
        incremental_regressions = int(
            sum(
                baseline_ok and not candidate_ok
                for baseline_ok, candidate_ok in zip(
                    zero_protected_correct,
                    protected_correct,
                )
            )
        )
        recovered_baseline_errors = int(
            sum(
                (not baseline_ok) and candidate_ok
                for baseline_ok, candidate_ok in zip(
                    zero_protected_correct,
                    protected_correct,
                )
            )
        )
        active_count = int(sum(active_correct))
        reports.append(
            {
                "scale": float(scale),
                "active_correct_tokens": active_count,
                "active_repaired_vs_zero": int(zero_active_count - active_count),
                "protected_absolute_incorrect": int(
                    len(protected_correct) - sum(protected_correct)
                ),
                "protected_incremental_regressions_vs_zero": incremental_regressions,
                "protected_zero_scale_incorrect": zero_protected_incorrect,
                "protected_baseline_errors_recovered": recovered_baseline_errors,
                "materialized_delta_norm": delta_norm,
                "nonzero_materialized_delta": nonzero,
            }
        )

    safe = [
        row
        for row in reports
        if row["scale"] > 0.0
        and row["nonzero_materialized_delta"]
        and row["active_repaired_vs_zero"] > 0
        and row["protected_incremental_regressions_vs_zero"] == 0
    ]
    if safe:
        selected = min(
            safe,
            key=lambda row: (
                int(row["active_correct_tokens"]),
                -float(row["scale"]),
                float(row["materialized_delta_norm"]),
            ),
        )
    else:
        selected = next(row for row in reports if row["scale"] == 0.0)

    selected_scale = float(selected["scale"])
    repair.materialize_output_row(
        output_weight,
        neutral_token_id,
        original_neutral_row,
        delta_row,
        selected_scale,
    )
    selected_active, selected_protected, _, _ = exact_by_scale[selected_scale]
    baseline = {
        "active_correct_tokens_at_zero": zero_active_count,
        "protected_correct_tokens_at_zero": int(sum(zero_protected_correct)),
        "protected_incorrect_tokens_at_zero": zero_protected_incorrect,
        "protected_total_tokens": len(zero_protected_correct),
    }
    return selected_scale, reports, selected_active, selected_protected, baseline


def main() -> None:
    args = build_parser().parse_args()
    if args.eval_batch_size <= 0 or args.cache_batch_size <= 0:
        raise ValueError("Evaluation and cache batch sizes must be positive")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("Repair steps and learning rate must be positive")

    gagd.set_seed(args.seed)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repair_dir = output_dir / "active_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args)
    device = next(model.parameters()).device
    llama_like = zsre.is_llama_like(model, tok)
    neutral_token_id = int(tok.eos_token_id)

    zsre_path = zsre.download_zsre(resolve(args.zsre_path), url=args.zsre_url)
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
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
        )
    ]

    active_all = exact_predictions(
        model=model,
        tok=tok,
        cases=forget_active_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache active ZsRE tokens",
    )
    protected_all = exact_predictions(
        model=model,
        tok=tok,
        cases=forget_neighborhood_cases + retain_protected_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache protected ZsRE tokens",
    )
    active_caches = [
        row
        for row in active_all
        if row.correct and row.target_token_id != neutral_token_id
    ]
    protected_caches = [
        row
        for row in protected_all
        if row.correct and row.target_token_id != neutral_token_id
    ]
    active_cases = [row.case for row in active_caches]
    protected_cases = [row.case for row in protected_caches]

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
    (
        selected_scale,
        scale_reports,
        selected_active,
        selected_protected,
        exact_zero_baseline,
    ) = exact_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_layer.weight,
        neutral_token_id=neutral_token_id,
        original_neutral_row=original_neutral_row,
        delta_row=delta_rows[0],
        active_cases=active_cases,
        protected_cases=protected_cases,
        scales=scales,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    gagd.write_json(repair_dir / "bf16_exact_scale_sweep_v2.json", scale_reports)
    gagd.write_json(repair_dir / "exact_zero_scale_baseline.json", exact_zero_baseline)
    repair.write_jsonl(
        repair_dir / "active_tokens_after.jsonl",
        [repair.cache_report(row) for row in selected_active],
    )
    repair.write_jsonl(
        repair_dir / "protected_tokens_after.jsonl",
        [repair.cache_report(row) for row in selected_protected],
    )

    candidate_result = zsre.evaluate_loaded_model_official(
        method="Setting 5e + exact-BF16 active LM-head repair candidate",
        model=model,
        tok=tok,
        model_dir="in-memory:zsre_bf16_safe_v2",
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

    selected_report = next(
        row for row in scale_reports if float(row["scale"]) == selected_scale
    )
    local_success = bool(
        selected_scale > 0.0
        and selected_report["nonzero_materialized_delta"]
        and selected_report["active_repaired_vs_zero"] > 0
        and selected_report["protected_incremental_regressions_vs_zero"] == 0
    )
    accepted = bool(
        local_success and (gate_report["passed"] or not args.strict_utility_gates)
    )

    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        selected_result["method"] = "Setting 5e + BF16-safe active LM-head repair"
        selection_reason = "nonzero_repair_passed_exact_zero_baseline_and_official_gates"
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
        "method": "zsre_bf16_safe_active_repair_v2",
        "active_tokens_cached_before": len(active_caches),
        "active_tokens_zero_scale": exact_zero_baseline[
            "active_correct_tokens_at_zero"
        ],
        "active_tokens_after": int(selected_report["active_correct_tokens"]),
        "active_tokens_repaired_vs_zero": int(
            selected_report["active_repaired_vs_zero"]
        ),
        "protected_tokens_cached": len(protected_caches),
        "protected_zero_scale_incorrect": exact_zero_baseline[
            "protected_incorrect_tokens_at_zero"
        ],
        "protected_incremental_regressions_after": int(
            selected_report["protected_incremental_regressions_vs_zero"]
        ),
        "protected_absolute_incorrect_after": int(
            selected_report["protected_absolute_incorrect"]
        ),
        "selected_scale": selected_scale,
        "materialized_delta_norm": float(effective_delta.norm().cpu()),
        "active_repair_applied": accepted,
        "fallback_to_setting5e": not accepted,
        "candidate_accepted": accepted,
        "selection_reason": selection_reason,
        "official_metric_gates": gate_report,
        "optimization": optimization,
    }
    gagd.write_json(repair_dir / "repair_summary.json", summary)

    if args.save_selected_checkpoint:
        repair.save_checkpoint(model, tok, output_dir / "selected_checkpoint")

    rows = [
        repair.comparison_row("Ultra-aggressive Setting 5e (600 steps)", setting5_result),
        repair.comparison_row("BF16-safe active candidate", candidate_result),
        repair.comparison_row("Selected", selected_result),
    ]
    repair.write_comparison(output_dir, rows)
    gagd.write_json(
        output_dir / "zsre_results.json",
        {
            "seed": args.seed,
            "setting5e": repair.compact_metrics(setting5_result),
            "candidate": repair.compact_metrics(candidate_result),
            "selected": repair.compact_metrics(selected_result),
            "repair": summary,
        },
    )

    print(
        "Selected ZsRE result: "
        f"Eff={selected_result['forget']['Eff']}, "
        f"Gen={selected_result['forget']['Gen']}, "
        f"Spe={selected_result['forget']['Spe']}, "
        f"PPL={selected_result.get('forget_PPL')}; "
        f"active_repair_applied={accepted}; selected_scale={selected_scale}"
    )
    print(f"Comparison: {output_dir / 'comparison.md'}")


if __name__ == "__main__":
    main()
