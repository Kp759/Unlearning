#!/usr/bin/env python3
"""BF16-safe ZsRE active repair with exact zero-scale baseline comparison.

The previous staged runner counted absolute protected-token errors after
re-batching. A protected token can flip even at scale 0 because BF16 logits are
near-tied and the exact verification pass uses different batch composition from
the initial cache. This runner defines a regression as a token that is correct
in the exact scale-0 BF16 baseline and becomes incorrect at a nonzero scale.

A repair is accepted only when it:
  * uses a genuinely nonzero materialized ``Unknown``-row delta;
  * reduces exact active-token correctness relative to scale 0;
  * introduces zero additional protected regressions relative to scale 0; and
  * passes the official ZsRE utility gates.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
from typing import Any, Dict, Tuple

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
    parser.add_argument("--active-logit-margin", type=float, default=0.25)
    parser.add_argument("--selection-logit-margin", type=float, default=0.05)
    parser.add_argument("--repair-rank", type=int, default=0)
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
    parser.add_argument("--target-eff-max", type=float, default=0.0)
    parser.add_argument("--target-gen-max", type=float, default=0.0)
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
        "--save-candidate-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the materialized LM-head repair candidate before any rollback.",
    )
    parser.add_argument(
        "--save-selected-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--fail-if-target-missed",
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
    repair.validate_neutral_target_checkpoint(checkpoint, tok)
    model.config.use_cache = False
    model.eval()
    return model, tok


def main() -> None:
    args = build_parser().parse_args()
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("Forget and retain counts must be positive")
    if args.eval_batch_size <= 0 or args.cache_batch_size <= 0:
        raise ValueError("Evaluation and cache batch sizes must be positive")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("Repair steps and learning rate must be positive")
    if args.active_logit_margin < 0 or args.selection_logit_margin < 0:
        raise ValueError("Active/selection margins must be non-negative")
    if args.repair_rank < 0 or args.repair_l2_lambda < 0:
        raise ValueError("Repair rank/L2 regularization must be non-negative")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("Maximum delta norm must be finite and non-negative")
    if args.retain_calibration_num < 0:
        raise ValueError("Retain calibration count must be non-negative")
    if args.utility_drop_tolerance < 0 or args.max_ppl_ratio < 1:
        raise ValueError("Invalid utility-drop tolerance or PPL ratio")
    if args.target_eff_max < 0 or args.target_gen_max < 0:
        raise ValueError("Target Eff/Gen maxima must be non-negative")
    repair.parse_candidate_scales(args.candidate_scales)

    gagd.set_seed(args.seed)
    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    repair_dir = output_dir / "active_repair"
    repair_dir.mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args)
    device = next(model.parameters()).device
    llama_like = zsre.is_llama_like(model, tok)
    neutral_token_id = zsre.resolve_neutral_target_token_id(tok)

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
    if not 0 <= neutral_token_id < output_layer.weight.shape[0]:
        raise ValueError(
            f"ZsRE neutral token {zsre.NEUTRAL_TARGET!r} is outside the "
            "LM-head vocabulary"
        )
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
    forget_official_cases = [
        case
        for record in forget_records
        for case in zsre.expand_prediction_cases(
            record,
            tok,
            llama_like=llama_like,
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
    official_active_identities = repair.official_correct_case_identities(
        forget_records,
        setting5_result["forget_raw"],
        tok,
        llama_like=llama_like,
        prompt_types=("rewrite", "paraphrase"),
    )
    official_protected_identities = repair.official_correct_case_identities(
        forget_records,
        setting5_result["forget_raw"],
        tok,
        llama_like=llama_like,
        prompt_types=("neighborhood",),
    )
    official_protected_identities.update(
        repair.official_correct_case_identities(
            calibration_records,
            setting5_result["retain_raw"],
            tok,
            llama_like=llama_like,
            prompt_types=("rewrite", "paraphrase", "neighborhood"),
        )
    )

    forget_official_caches = repair.cache_prediction_cases(
        model,
        tok,
        forget_official_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        desc="cache official-order forget ZsRE tokens",
    )
    active_all = [
        row
        for row in forget_official_caches
        if row.case.prompt_type in {"rewrite", "paraphrase"}
    ]
    forget_protected_all = [
        row
        for row in forget_official_caches
        if row.case.prompt_type == "neighborhood"
    ]
    retain_protected_all = repair.cache_prediction_cases(
        model,
        tok,
        retain_protected_cases,
        neutral_token_id=neutral_token_id,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        desc="cache retain-calibration ZsRE tokens",
    )
    protected_all = forget_protected_all + retain_protected_all
    active_caches = [
        row
        for row in active_all
        if row.case.identity in official_active_identities
        and row.target_token_id != neutral_token_id
    ]
    protected_caches = [
        row
        for row in protected_all
        if row.case.identity in official_protected_identities
        and row.target_token_id != neutral_token_id
    ]
    missing_active = official_active_identities - {
        row.case.identity for row in active_all
    }
    missing_protected = official_protected_identities - {
        row.case.identity for row in protected_all
    }
    if missing_active:
        raise RuntimeError(
            "Failed to cache officially correct active ZsRE tokens: "
            f"{sorted(missing_active)[:10]}"
        )
    if missing_protected:
        raise RuntimeError(
            "Failed to cache officially correct protected ZsRE tokens: "
            f"{sorted(missing_protected)[:10]}"
        )

    print(
        f"Optimizing neutral row {neutral_token_id} "
        f"({zsre.NEUTRAL_TARGET!r}): "
        f"active={len(active_caches)}, protected={len(protected_caches)}"
    )
    delta_rows, repair_logs, optimization = repair.optimize_neutral_delta(
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
    ) = repair.exact_bf16_scale_sweep(
        model=model,
        tok=tok,
        output_weight=output_layer.weight,
        neutral_token_id=neutral_token_id,
        original_neutral_row=original_neutral_row,
        delta_row=delta_rows[0],
        active_cases=forget_active_cases,
        protected_cases=[
            row.case
            for row in forget_protected_all
            if row.case.identity in official_protected_identities
            and row.target_token_id != neutral_token_id
        ],
        active_context_cases=forget_official_cases,
        protected_context_cases=forget_official_cases,
        scales=scales,
        device=device,
        llama_like=llama_like,
        batch_size=args.eval_batch_size,
        minimum_active_margin=args.selection_logit_margin,
    )
    gagd.write_json(repair_dir / "bf16_exact_scale_sweep.json", scale_reports)
    gagd.write_json(repair_dir / "exact_zero_scale_baseline.json", exact_zero_baseline)
    official_setting5_active_correct = repair.official_forget_active_correct_tokens(
        setting5_result
    )
    if (
        exact_zero_baseline["active_correct_tokens_at_zero"]
        != official_setting5_active_correct
    ):
        raise RuntimeError(
            "Exact scale-0 active count does not match the official Setting 5e "
            "forget pass despite identical case order and batch size: "
            f"{exact_zero_baseline['active_correct_tokens_at_zero']} != "
            f"{official_setting5_active_correct}"
        )
    if exact_zero_baseline["protected_incorrect_tokens_at_zero"] != 0:
        raise RuntimeError(
            "An officially correct forget-neighborhood token changed at exact "
            "scale 0; refusing a numerically misaligned repair sweep"
        )
    selected_report = next(
        row for row in scale_reports if float(row["scale"]) == selected_scale
    )
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
    official_candidate_active_correct = (
        repair.official_forget_active_correct_tokens(candidate_result)
    )
    if (
        official_candidate_active_correct
        != int(selected_report["active_correct_tokens"])
    ):
        raise RuntimeError(
            "Exact selected-scale active count does not match the final official "
            "candidate evaluation: "
            f"{selected_report['active_correct_tokens']} != "
            f"{official_candidate_active_correct}"
        )
    gate_report = repair.metric_gate_report(
        setting5_result,
        candidate_result,
        utility_drop_tolerance=args.utility_drop_tolerance,
        max_ppl_ratio=args.max_ppl_ratio,
        target_eff_max=args.target_eff_max,
        target_gen_max=args.target_gen_max,
    )

    target_met = bool(
        float(candidate_result["forget"]["Eff"]) <= args.target_eff_max
        and float(candidate_result["forget"]["Gen"]) <= args.target_gen_max
    )
    target_already_met = bool(
        float(setting5_result["forget"]["Eff"]) <= args.target_eff_max
        and float(setting5_result["forget"]["Gen"]) <= args.target_gen_max
    )
    local_success = bool(
        target_already_met
        or (
            selected_scale > 0.0
            and selected_report["nonzero_materialized_delta"]
            and selected_report["active_repaired_vs_zero"] > 0
            and selected_report["protected_incremental_regressions_vs_zero"] == 0
        )
    )
    accepted = bool(
        target_met
        and local_success
        and (gate_report["passed"] or not args.strict_utility_gates)
    )

    # The model currently contains the exact materialized active-repair
    # candidate. Preserve it before a rejected candidate is rolled back.
    if args.save_candidate_checkpoint:
        candidate_checkpoint = output_dir / "active_candidate_checkpoint"
        repair.save_checkpoint(model, tok, candidate_checkpoint)
        gagd.write_json(
            candidate_checkpoint / "candidate_provenance.json",
            {
                "seed": args.seed,
                "source_setting5_checkpoint": str(
                    resolve(args.setting5_checkpoint)
                ),
                "candidate_scale": selected_scale,
                "candidate_accepted": accepted,
                "forget_Eff": candidate_result["forget"]["Eff"],
                "forget_Gen": candidate_result["forget"]["Gen"],
                "forget_Spe": candidate_result["forget"]["Spe"],
                "PPL": candidate_result.get("forget_PPL"),
                "note": (
                    "Raw materialized LM-head repair candidate saved before "
                    "strict-gate rollback."
                ),
            },
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
            "candidate_missed_required_zero_eff_gen_target"
            if not target_met
            else (
                "no_nonzero_bf16_safe_improving_scale"
                if not local_success
                else "nonzero_repair_failed_official_metric_gates"
            )
        )

    effective_delta = (
        output_layer.weight[neutral_token_id].detach().float()
        - original_neutral_row.detach().float()
    )
    summary = {
        "method": "zsre_bf16_safe_active_repair_v2",
        "neutral_target": zsre.NEUTRAL_TARGET,
        "neutral_token_id": neutral_token_id,
        "case_selection_source": "setting5e_official_metric_data",
        "active_tokens_cached_before": len(active_caches),
        "active_tokens_zero_scale": exact_zero_baseline[
            "active_correct_tokens_at_zero"
        ],
        "active_tokens_after_candidate": int(
            selected_report["active_correct_tokens"]
        ),
        "official_batch_alignment_verified": True,
        "active_tokens_repaired_vs_zero": int(
            selected_report["active_repaired_vs_zero"]
        ),
        "optimization_protected_tokens_cached": len(protected_caches),
        "exact_official_forget_protected_tokens": len(selected_protected),
        "protected_zero_scale_incorrect": exact_zero_baseline[
            "protected_incorrect_tokens_at_zero"
        ],
        "protected_incremental_regressions_after": int(
            selected_report["protected_incremental_regressions_vs_zero"]
        ),
        "protected_absolute_incorrect_after": int(
            selected_report["protected_absolute_incorrect"]
        ),
        "candidate_scale": selected_scale,
        "selected_scale": selected_scale if accepted else 0.0,
        "materialized_delta_norm": float(effective_delta.norm().cpu()),
        "active_repair_applied": accepted,
        "fallback_to_setting5e": not accepted,
        "candidate_accepted": accepted,
        "required_target_met": target_met,
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
            "method": "zsre_bf16_safe_active_repair_v2",
            "dataset": "ZsRE",
            "seed": args.seed,
            "forget_num": args.forget_num,
            "retain_num": args.retain_num,
            "zsre_sha256": zsre.file_sha256(zsre_path),
            "setting5e": repair.compact_metrics(setting5_result),
            "active_candidate": repair.compact_metrics(candidate_result),
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
    if args.fail_if_target_missed and not accepted:
        raise RuntimeError(
            "ZsRE BF16-safe repair missed the required target or utility gates. "
            f"Candidate Eff={candidate_result['forget']['Eff']}, "
            f"Gen={candidate_result['forget']['Gen']}, "
            f"Spe={candidate_result['forget']['Spe']}, "
            f"PPL={candidate_result.get('forget_PPL')}; "
            f"reason={selection_reason}."
        )


if __name__ == "__main__":
    main()
