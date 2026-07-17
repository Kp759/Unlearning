#!/usr/bin/env python3
"""Search for a neighborhood-confidence repair above a hard specificity target.

This is a standalone post-processing runner for a protected Setting 5e
checkpoint. It never reruns GA/GD and does not modify input embeddings or the
transformer. Only selected LM-head rows are changed.

The search directly optimizes the official MCF specificity quantity:

    exp(-target_true_nll) - exp(-target_new_nll)

with the same record-level weighting as the official evaluator. Every rewrite
and paraphrase prompt remains protected by its own required margin. Candidate
checkpoints are accepted only when Eff and Gen are zero, Spe reaches the
requested target, no protected prompt fails, and official PPL is no higher
than the allowed ceiling.

Optimizing official neighborhood prompts is benchmark-targeted calibration.
Use a disjoint calibration/evaluation split when reporting held-out results.
"""

from __future__ import annotations

import argparse
import gc
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_active_case_repair as active
import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import (
    evaluate_loaded_model_official,
    is_llama_like,
    load_official_ppl_text,
    official_perplexity,
)


METHOD = "gagd_neighborhood_confidence_repair"


@dataclass(frozen=True)
class CandidateSnapshot:
    stage_target: float
    delta_rows: torch.Tensor
    metrics: Dict[str, Any]
    ppl: float
    gates: Dict[str, Any]


@dataclass(frozen=True)
class PPLDeltaCache:
    hidden: torch.Tensor
    selected_probs: torch.Tensor
    base_token_nll: torch.Tensor
    target_selected_columns: torch.Tensor
    denominator: int


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="Protected Setting 5e checkpoint used as the immutable input.",
    )
    parser.add_argument(
        "--reference-model-path",
        default=None,
        help=(
            "Optional alternate output-layer reference for retain KL. When "
            "omitted, KL is measured against --model-path."
        ),
    )
    parser.add_argument(
        "--experiment-config-path",
        default=None,
        help=(
            "Explicit Setting 5 configuration when it cannot be recovered "
            "from --model-path."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mcf-cache-path", required=True)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--mcf-url", default=gagd.MCF_URL)
    parser.add_argument(
        "--sample-mode",
        choices=["official", "first"],
        default="official",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)

    parser.add_argument("--spe-target", type=float, default=50.0)
    parser.add_argument(
        "--spe-target-schedule",
        default="20,30,40,50",
        help="Comma-separated progressive Spe targets; final target is added.",
    )
    parser.add_argument(
        "--spe-optimization-buffer",
        type=float,
        default=1.0,
        help="Optimize above each reported Spe target to absorb BF16 rounding.",
    )
    parser.add_argument(
        "--neighborhood-tail-floor",
        type=float,
        default=0.10,
        help="Minimum per-record neighborhood probability gap encouraged.",
    )
    parser.add_argument(
        "--target-true-prob-floor",
        type=float,
        default=0.55,
        help="Minimum per-record target-true probability encouraged.",
    )
    parser.add_argument(
        "--row-selection",
        choices=["low_spe", "all"],
        default="low_spe",
        help=(
            "Select target rows from records below the final Spe target or "
            "from every forget record."
        ),
    )

    parser.add_argument("--active-margin", type=float, default=0.10)
    parser.add_argument("--protected-margin-floor", type=float, default=0.0)
    parser.add_argument("--repair-steps-per-stage", type=int, default=100)
    parser.add_argument("--repair-lr", type=float, default=5e-3)
    parser.add_argument(
        "--repair-optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adamw",
    )
    parser.add_argument("--spe-weight", type=float, default=100.0)
    parser.add_argument("--neighborhood-tail-weight", type=float, default=10.0)
    parser.add_argument("--target-true-prob-weight", type=float, default=10.0)
    parser.add_argument("--forget-hinge-weight", type=float, default=100.0)
    parser.add_argument("--retain-kl-mu", type=float, default=1.0)
    parser.add_argument("--delta-l2-lambda", type=float, default=1e-4)
    parser.add_argument(
        "--ppl-nll-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on the exact official-PPL calibration NLL. A positive "
            "value nudges PPL downward while the hard PPL gate prevents "
            "regressions."
        ),
    )
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--repair-rank", type=int, default=32)
    parser.add_argument("--retain-calibration-num", type=int, default=64)
    parser.add_argument("--retain-calibration-seed", type=int, default=1729)
    parser.add_argument("--retain-projection-rank", type=int, default=64)
    parser.add_argument(
        "--project-away-retain-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--margin-batch-size", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=10)

    parser.add_argument(
        "--max-ppl-increase",
        type=float,
        default=0.0,
        help="Maximum candidate PPL minus input-checkpoint PPL.",
    )
    parser.add_argument(
        "--ppl-ceiling",
        type=float,
        default=None,
        help="Optional absolute PPL ceiling, combined with --max-ppl-increase.",
    )
    parser.add_argument(
        "--require-input-zero",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the supplied checkpoint to start at Eff=0 and Gen=0.",
    )
    parser.add_argument(
        "--stop-after-first-qualified",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-best-effort",
        action="store_true",
        help=(
            "Permit saving an Eff=0/Gen=0/PPL-safe candidate below the Spe "
            "target. Disabled by default."
        ),
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--run-official-mcf-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the complete official evaluator before saving.",
    )

    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.forget_num <= 0 or args.retain_num <= 0:
        raise ValueError("--forget-num and --retain-num must be positive")
    if not 0.0 < args.spe_target < 100.0:
        raise ValueError("--spe-target must lie strictly between 0 and 100")
    for name in (
        "spe_optimization_buffer",
        "active_margin",
        "repair_lr",
        "spe_weight",
        "neighborhood_tail_weight",
        "target_true_prob_weight",
        "forget_hinge_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    for name in ("neighborhood_tail_floor", "target_true_prob_floor"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must lie in [0,1]")
    if args.repair_steps_per_stage <= 0 or args.margin_batch_size <= 0:
        raise ValueError("repair steps and margin batch size must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.repair_rank < 0 or args.retain_projection_rank < 0:
        raise ValueError("repair and retain projection ranks must be non-negative")
    if args.retain_calibration_num <= 0:
        raise ValueError("--retain-calibration-num must be positive")
    if args.retain_kl_mu < 0 or args.delta_l2_lambda < 0 or args.ppl_nll_weight < 0:
        raise ValueError("regularization weights must be non-negative")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("--max-delta-norm must be finite and non-negative")
    if not math.isfinite(args.max_ppl_increase):
        raise ValueError("--max-ppl-increase must be finite")
    if args.ppl_ceiling is not None and (
        not math.isfinite(args.ppl_ceiling) or args.ppl_ceiling <= 0
    ):
        raise ValueError("--ppl-ceiling must be finite and positive")


def parse_spe_schedule(text: str, final_target: float) -> List[float]:
    values: List[float] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = float(raw)
        if not 0.0 < value < 100.0:
            raise ValueError("Every scheduled Spe target must lie in (0,100)")
        if value <= final_target:
            values.append(value)
    values.append(float(final_target))
    return sorted(set(values))


def expand_neighborhood_prompt_instances(
    records: Sequence[active.SampledMCFRecord],
) -> List[active.MCFPromptInstance]:
    instances: List[active.MCFPromptInstance] = []
    for record in records:
        prompts = record.raw_record.get("neighborhood_prompts", [])
        instances.extend(
            active.MCFPromptInstance(
                record_index=record.record_index,
                sampled_position=record.sampled_position,
                prompt_type="neighborhood",
                prompt_index=prompt_index,
                prompt=str(prompt),
                target_new=record.target_new,
                target_true=record.target_true,
            )
            for prompt_index, prompt in enumerate(prompts)
        )
    return instances


def position_groups_by_record(
    instances: Sequence[active.MCFPromptInstance],
) -> List[List[int]]:
    grouped: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    order: List[Tuple[int, int]] = []
    for position, instance in enumerate(instances):
        key = (instance.record_index, instance.sampled_position)
        if key not in grouped:
            order.append(key)
        grouped[key].append(position)
    return [grouped[key] for key in order]


def record_mean_tensor(
    values: torch.Tensor,
    position_groups: Sequence[Sequence[int]],
) -> torch.Tensor:
    if not position_groups:
        return values.new_empty((0,))
    return torch.stack(
        [
            values.index_select(
                0,
                torch.tensor(group, dtype=torch.long, device=values.device),
            ).mean()
            for group in position_groups
        ]
    )


def paired_nll_from_delta_caches(
    caches: Sequence[active.RewriteDeltaCache],
    delta_rows: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not caches:
        empty = delta_rows.new_empty((0,))
        return empty, empty
    target_new = torch.stack(
        [
            active.answer_nll_from_delta_cache(cache.target_new, delta_rows)
            for cache in caches
        ]
    )
    target_true = torch.stack(
        [
            active.answer_nll_from_delta_cache(cache.target_true, delta_rows)
            for cache in caches
        ]
    )
    return target_new, target_true


@torch.no_grad()
def score_prompt_instances(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[active.MCFPromptInstance],
    device: torch.device,
    batch_size: int,
    llama_like: bool,
) -> List[Dict[str, Any]]:
    reports: List[Dict[str, Any]] = []
    for start in range(0, len(instances), batch_size):
        chunk = instances[start : start + batch_size]
        target_new_nll, target_true_nll = active.official_prompt_instance_nll_tensors(
            model,
            tok,
            chunk,
            device,
            llama_like,
        )
        for instance, new_nll, true_nll in zip(
            chunk,
            target_new_nll.detach().cpu().tolist(),
            target_true_nll.detach().cpu().tolist(),
        ):
            margin = float(new_nll - true_nll)
            reports.append(
                {
                    "record_index": instance.record_index,
                    "sampled_position": instance.sampled_position,
                    "prompt_type": instance.prompt_type,
                    "prompt_index": instance.prompt_index,
                    "prompt": instance.prompt,
                    "target_new": instance.target_new,
                    "target_true": instance.target_true,
                    "target_new_nll": float(new_nll),
                    "target_true_nll": float(true_nll),
                    "margin": margin,
                    "target_new_probability": math.exp(-float(new_nll)),
                    "target_true_probability": math.exp(-float(true_nll)),
                    "probability_diff_true_minus_new": (
                        math.exp(-float(true_nll)) - math.exp(-float(new_nll))
                    ),
                }
            )
    return reports


def _record_weighted_report_mean(
    reports: Sequence[Dict[str, Any]],
    prompt_type: str,
    value_fn: Any,
) -> float:
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for report in reports:
        if report["prompt_type"] != prompt_type:
            continue
        key = (int(report["record_index"]), int(report["sampled_position"]))
        grouped[key].append(float(value_fn(report)))
    if not grouped:
        raise ValueError(f"No {prompt_type} prompt reports were available")
    record_means = [sum(values) / len(values) for values in grouped.values()]
    return sum(record_means) / len(record_means)


def official_metrics_from_reports(
    forget_reports: Sequence[Dict[str, Any]],
    neighborhood_reports: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    eff_fraction = _record_weighted_report_mean(
        forget_reports,
        "rewrite",
        lambda report: float(report["margin"] < 0.0),
    )
    gen_fraction = _record_weighted_report_mean(
        forget_reports,
        "paraphrase",
        lambda report: float(report["margin"] < 0.0),
    )
    spe_fraction = _record_weighted_report_mean(
        neighborhood_reports,
        "neighborhood",
        lambda report: report["probability_diff_true_minus_new"],
    )
    spe_success_fraction = _record_weighted_report_mean(
        neighborhood_reports,
        "neighborhood",
        lambda report: float(report["margin"] > 0.0),
    )
    return {
        "Eff": 100.0 * eff_fraction,
        "Gen": 100.0 * gen_fraction,
        "Spe": 100.0 * spe_fraction,
        "Spe_success": 100.0 * spe_success_fraction,
        "rewrite_failure_prompt_instances": sum(
            report["prompt_type"] == "rewrite" and report["margin"] < 0.0
            for report in forget_reports
        ),
        "paraphrase_failure_prompt_instances": sum(
            report["prompt_type"] == "paraphrase" and report["margin"] < 0.0
            for report in forget_reports
        ),
    }


def record_spe_by_key(
    neighborhood_reports: Sequence[Dict[str, Any]],
) -> Dict[Tuple[int, int], float]:
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for report in neighborhood_reports:
        key = (int(report["record_index"]), int(report["sampled_position"]))
        grouped[key].append(float(report["probability_diff_true_minus_new"]))
    return {key: sum(values) / len(values) for key, values in grouped.items() if values}


def select_neighborhood_lm_head_rows(
    tok: Any,
    records: Sequence[active.SampledMCFRecord],
    neighborhood_reports: Sequence[Dict[str, Any]],
    spe_target: float,
    row_selection: str,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    if row_selection not in {"low_spe", "all"}:
        raise ValueError(f"Unsupported row selection mode: {row_selection}")
    per_record_spe = record_spe_by_key(neighborhood_reports)
    target_fraction = spe_target / 100.0
    selected_record_keys: List[Tuple[int, int]] = []
    selected_ids: set[int] = set()
    for record in records:
        key = (record.record_index, record.sampled_position)
        if row_selection == "low_spe" and per_record_spe.get(key, float("-inf")) >= (
            target_fraction
        ):
            continue
        selected_record_keys.append(key)
        for answer in (record.target_new, record.target_true):
            selected_ids.update(
                gagd.token_ids_for_text(tok, gagd.normalize_answer(answer))
            )
    selected_ids -= gagd.special_token_ids(tok)
    return sorted(selected_ids), selected_record_keys


def confidence_loss_terms(
    forget_target_new_nll: torch.Tensor,
    forget_target_true_nll: torch.Tensor,
    required_forget_margins: torch.Tensor,
    neighborhood_target_new_nll: torch.Tensor,
    neighborhood_target_true_nll: torch.Tensor,
    neighborhood_position_groups: Sequence[Sequence[int]],
    *,
    spe_target_fraction: float,
    neighborhood_tail_floor: float,
    target_true_prob_floor: float,
) -> Dict[str, torch.Tensor]:
    if not neighborhood_position_groups:
        raise ValueError("Specificity optimization requires neighborhood prompts")
    forget_margins = forget_target_new_nll - forget_target_true_nll
    forget_hinge = active.squared_hinge_loss(
        forget_margins,
        required_forget_margins,
    )

    neighborhood_new_probability = torch.exp(-neighborhood_target_new_nll)
    neighborhood_true_probability = torch.exp(-neighborhood_target_true_nll)
    neighborhood_probability_gap = (
        neighborhood_true_probability - neighborhood_new_probability
    )
    record_spe = record_mean_tensor(
        neighborhood_probability_gap,
        neighborhood_position_groups,
    )
    record_true_probability = record_mean_tensor(
        neighborhood_true_probability,
        neighborhood_position_groups,
    )
    global_spe = record_spe.mean()
    spe_target = global_spe.new_tensor(spe_target_fraction)
    tail_floor = record_spe.new_full(
        record_spe.shape,
        neighborhood_tail_floor,
    )
    true_probability_floor = record_true_probability.new_full(
        record_true_probability.shape,
        target_true_prob_floor,
    )
    return {
        "forget_margins": forget_margins,
        "forget_hinge": forget_hinge,
        "record_spe": record_spe,
        "global_spe_fraction": global_spe,
        "spe_global_hinge": torch.relu(spe_target - global_spe).square(),
        "spe_tail_hinge": torch.relu(tail_floor - record_spe).square().mean(),
        "true_probability_hinge": torch.relu(
            true_probability_floor - record_true_probability
        )
        .square()
        .mean(),
        "minimum_forget_margin_slack": (
            forget_margins
            - required_forget_margins.to(
                device=forget_margins.device,
                dtype=forget_margins.dtype,
            )
        ).min(),
    }


def allowed_ppl(
    input_ppl: float,
    max_ppl_increase: float,
    ppl_ceiling: Optional[float],
) -> float:
    limit = float(input_ppl + max_ppl_increase)
    if ppl_ceiling is not None:
        limit = min(limit, float(ppl_ceiling))
    return limit


def candidate_gates(
    metrics: Dict[str, Any],
    ppl: float,
    *,
    input_ppl: float,
    spe_target: float,
    max_ppl_increase: float,
    ppl_ceiling: Optional[float],
) -> Dict[str, Any]:
    ppl_limit = allowed_ppl(input_ppl, max_ppl_increase, ppl_ceiling)
    eff_zero = (
        float(metrics["Eff"]) == 0.0
        and int(metrics.get("rewrite_failure_prompt_instances", 0)) == 0
    )
    gen_zero = (
        float(metrics["Gen"]) == 0.0
        and int(metrics.get("paraphrase_failure_prompt_instances", 0)) == 0
    )
    spe_met = float(metrics["Spe"]) >= float(spe_target)
    ppl_met = math.isfinite(float(ppl)) and float(ppl) <= ppl_limit + 1e-6
    return {
        "qualified": eff_zero and gen_zero and spe_met and ppl_met,
        "eff_zero": eff_zero,
        "gen_zero": gen_zero,
        "spe_met": spe_met,
        "ppl_met": ppl_met,
        "spe_target": float(spe_target),
        "ppl_limit": ppl_limit,
    }


def candidate_priority(candidate: CandidateSnapshot) -> Tuple[float, float, float]:
    """Prefer lower PPL, then higher Spe, then the smallest sparse delta."""
    return (
        float(candidate.ppl),
        -float(candidate.metrics["Spe"]),
        float(candidate.metrics["selected_lm_head_delta_norm"]),
    )


def best_effort_priority(
    candidate: CandidateSnapshot,
) -> Tuple[float, float, float]:
    """Prefer specificity first when the requested threshold was not reached."""
    return (
        -float(candidate.metrics["Spe"]),
        float(candidate.ppl),
        float(candidate.metrics["selected_lm_head_delta_norm"]),
    )


@torch.no_grad()
def set_selected_lm_head_rows(
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    baseline_rows: torch.Tensor,
    delta_rows: torch.Tensor,
) -> None:
    if len(selected_ids) != baseline_rows.shape[0]:
        raise ValueError("baseline row count does not match selected IDs")
    if baseline_rows.shape != delta_rows.shape:
        raise ValueError("baseline and delta LM-head rows must have equal shape")
    if not selected_ids:
        return
    ids = torch.tensor(
        selected_ids,
        dtype=torch.long,
        device=output_weight.device,
    )
    repaired = baseline_rows.to(
        device=output_weight.device,
        dtype=output_weight.dtype,
    ) + delta_rows.to(
        device=output_weight.device,
        dtype=output_weight.dtype,
    )
    output_weight.index_copy_(0, ids, repaired)


@torch.no_grad()
def build_official_ppl_delta_cache(
    model: torch.nn.Module,
    tok: Any,
    text: str,
    selected_ids: Sequence[int],
    device: torch.device,
    max_input_length: int = 100,
) -> PPLDeltaCache:
    encoded = tok(
        [text],
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    ).to(device)
    output = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
    )
    input_ids = encoded["input_ids"]
    if input_ids.shape[1] < 2:
        raise ValueError(
            "Official PPL calibration text tokenized to fewer than 2 tokens"
        )
    log_probs = F.log_softmax(output.logits[:, :-1, :], dim=-1).squeeze(0)
    hidden = output.hidden_states[-1][:, :-1, :].squeeze(0).float()
    targets = input_ids[:, 1:].squeeze(0)
    selected = torch.tensor(
        selected_ids,
        dtype=torch.long,
        device=log_probs.device,
    )
    selected_lookup = {
        int(token_id): column for column, token_id in enumerate(selected_ids)
    }
    target_selected_columns = torch.tensor(
        [selected_lookup.get(int(token_id), -1) for token_id in targets],
        dtype=torch.long,
        device=log_probs.device,
    )
    return PPLDeltaCache(
        hidden=hidden.detach(),
        selected_probs=log_probs.index_select(-1, selected).exp().float().detach(),
        base_token_nll=(-log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1))
        .float()
        .detach(),
        target_selected_columns=target_selected_columns,
        denominator=int(input_ids.shape[1]),
    )


def official_ppl_nll_from_cache(
    cache: PPLDeltaCache,
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    corrections = cache.hidden @ delta_rows.transpose(0, 1)
    log_shift = active._log_partition_shift(cache.selected_probs, corrections)
    target_correction = corrections.new_zeros(corrections.shape[0])
    selected_mask = cache.target_selected_columns.ge(0)
    if selected_mask.any():
        target_correction[selected_mask] = corrections[
            selected_mask,
            cache.target_selected_columns[selected_mask],
        ]
    return (
        cache.base_token_nll + log_shift - target_correction
    ).sum() / cache.denominator


@torch.no_grad()
def evaluate_sparse_candidate(
    model: torch.nn.Module,
    tok: Any,
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    baseline_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    forget_instances: Sequence[active.MCFPromptInstance],
    neighborhood_instances: Sequence[active.MCFPromptInstance],
    *,
    device: torch.device,
    batch_size: int,
    llama_like: bool,
    ppl_text: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], float]:
    set_selected_lm_head_rows(
        output_weight,
        selected_ids,
        baseline_rows,
        delta_rows,
    )
    try:
        forget_reports = score_prompt_instances(
            model,
            tok,
            forget_instances,
            device,
            batch_size,
            llama_like,
        )
        neighborhood_reports = score_prompt_instances(
            model,
            tok,
            neighborhood_instances,
            device,
            batch_size,
            llama_like,
        )
        metrics = official_metrics_from_reports(
            forget_reports,
            neighborhood_reports,
        )
        ppl = official_perplexity(
            model,
            tok,
            ppl_text,
            device,
            max_input_length=100,
        )
    finally:
        set_selected_lm_head_rows(
            output_weight,
            selected_ids,
            baseline_rows,
            torch.zeros_like(baseline_rows),
        )
    return forget_reports, neighborhood_reports, metrics, float(ppl)


def official_result_metrics(result: Dict[str, Any]) -> Tuple[Dict[str, Any], float]:
    forget = result.get("forget")
    if not isinstance(forget, dict):
        raise ValueError("Official evaluation result is missing its forget summary")
    ppl = result.get("forget_PPL")
    if ppl is None:
        raise ValueError("Official evaluation did not produce PPL")
    return {
        "Eff": float(forget["Eff"]),
        "Gen": float(forget["Gen"]),
        "Spe": float(forget["Spe"]),
        "rewrite_failure_prompt_instances": (0 if float(forget["Eff"]) == 0.0 else 1),
        "paraphrase_failure_prompt_instances": (
            0 if float(forget["Gen"]) == 0.0 else 1
        ),
    }, float(ppl)


def snapshot_to_json(candidate: CandidateSnapshot) -> Dict[str, Any]:
    return {
        "stage_target": candidate.stage_target,
        "metrics": candidate.metrics,
        "ppl": candidate.ppl,
        "gates": candidate.gates,
    }


def _hidden_rows_from_prompt_caches(
    caches: Sequence[active.RewriteDeltaCache],
) -> torch.Tensor:
    rows = [
        answer_cache.hidden
        for cache in caches
        for answer_cache in (cache.target_new, cache.target_true)
        if answer_cache.hidden.numel()
    ]
    if not rows:
        raise ValueError("No answer-token hidden states were cached")
    return torch.cat(rows, dim=0)


def _training_state(
    delta_module: active.SelectedRowDelta,
    forget_caches: Sequence[active.RewriteDeltaCache],
    neighborhood_caches: Sequence[active.RewriteDeltaCache],
    neighborhood_groups: Sequence[Sequence[int]],
    required_margins: torch.Tensor,
    ppl_cache: PPLDeltaCache,
    retain_caches: Sequence[Any],
    *,
    spe_target_fraction: float,
    neighborhood_tail_floor: float,
    target_true_prob_floor: float,
) -> Dict[str, torch.Tensor]:
    delta = delta_module.effective_delta()
    forget_new, forget_true = paired_nll_from_delta_caches(
        forget_caches,
        delta,
    )
    neighborhood_new, neighborhood_true = paired_nll_from_delta_caches(
        neighborhood_caches,
        delta,
    )
    terms = confidence_loss_terms(
        forget_new,
        forget_true,
        required_margins,
        neighborhood_new,
        neighborhood_true,
        neighborhood_groups,
        spe_target_fraction=spe_target_fraction,
        neighborhood_tail_floor=neighborhood_tail_floor,
        target_true_prob_floor=target_true_prob_floor,
    )
    terms["retain_kl"] = active.retain_kl_from_caches(retain_caches, delta)
    terms["delta_l2"] = delta.square().sum()
    terms["ppl_nll"] = official_ppl_nll_from_cache(ppl_cache, delta)
    terms["delta_norm"] = delta.norm()
    return terms


def _loss_from_terms(
    terms: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> torch.Tensor:
    return (
        args.spe_weight * terms["spe_global_hinge"]
        + args.neighborhood_tail_weight * terms["spe_tail_hinge"]
        + args.target_true_prob_weight * terms["true_probability_hinge"]
        + args.forget_hinge_weight * terms["forget_hinge"]
        + args.retain_kl_mu * terms["retain_kl"]
        + args.delta_l2_lambda * terms["delta_l2"]
        + args.ppl_nll_weight * terms["ppl_nll"]
    )


def _stage_log_row(
    stage_target: float,
    step: int,
    terms: Dict[str, torch.Tensor],
    loss: torch.Tensor,
    *,
    norm_before_projection: float,
    norm_after_projection: float,
    norm_projected: bool,
) -> Dict[str, Any]:
    return {
        "stage_target": stage_target,
        "step": step,
        "loss": float(loss.detach().cpu()),
        "estimated_spe": 100.0 * float(terms["global_spe_fraction"].detach().cpu()),
        "forget_hinge": float(terms["forget_hinge"].detach().cpu()),
        "spe_global_hinge": float(terms["spe_global_hinge"].detach().cpu()),
        "spe_tail_hinge": float(terms["spe_tail_hinge"].detach().cpu()),
        "target_true_probability_hinge": float(
            terms["true_probability_hinge"].detach().cpu()
        ),
        "retain_kl": float(terms["retain_kl"].detach().cpu()),
        "ppl_nll": float(terms["ppl_nll"].detach().cpu()),
        "minimum_forget_margin_slack": float(
            terms["minimum_forget_margin_slack"].detach().cpu()
        ),
        "delta_norm_before_projection": norm_before_projection,
        "delta_norm": norm_after_projection,
        "delta_norm_projected": norm_projected,
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    schedule = parse_spe_schedule(args.spe_target_schedule, args.spe_target)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path, source_config, preserved_alphas = active.recover_experiment_config(
        args.model_path,
        args.experiment_config_path,
    )
    active.validate_source_experiment_config(source_config, args)
    config_used: Dict[str, Any] = {
        **vars(args),
        "method": METHOD,
        "source_experiment_config_path": str(config_path),
        "source_experiment_config": source_config,
        "preserved_5e_overlap_alphas": preserved_alphas,
        "spe_target_schedule_resolved": schedule,
        "parameter_scope": "selected_lm_head_rows_only",
        "input_embeddings_frozen": True,
        "transformer_frozen": True,
        "official_prompt_construction": True,
        "official_record_weighting": True,
        "official_ppl_hard_gate": True,
        "benchmark_targeted_neighborhood_calibration": True,
    }
    gagd.write_json(output_dir / "config_used.json", config_used)

    print("Loading deterministic official MCF split")
    forget_records, retain_records = active.load_sampled_mcf_records(args)
    forget_instances = active.expand_prompt_instances(forget_records)
    neighborhood_instances = expand_neighborhood_prompt_instances(forget_records)
    if not neighborhood_instances:
        raise ValueError("The sampled forget records contain no neighborhood prompts")

    print(f"Loading protected checkpoint: {args.model_path}")
    model, tok = gagd.load_model_and_tokenizer(
        active._model_loading_args(args),
        for_training=False,
    )
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_storage_pointer = input_weight.data_ptr()
    input_tensor_version = input_weight._version
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    wikidata_dir = gagd.resolve_output_path(args.wikidata_dir)
    ppl_text = load_official_ppl_text(wikidata_dir)
    if ppl_text is None:
        raise FileNotFoundError(
            f"Official PPL calibration data is required but missing: {wikidata_dir}"
        )

    print("Scoring rewrite, paraphrase, neighborhood, and PPL baselines")
    baseline_forget_reports = score_prompt_instances(
        model,
        tok,
        forget_instances,
        device,
        args.margin_batch_size,
        llama_like,
    )
    baseline_neighborhood_reports = score_prompt_instances(
        model,
        tok,
        neighborhood_instances,
        device,
        args.margin_batch_size,
        llama_like,
    )
    baseline_metrics = official_metrics_from_reports(
        baseline_forget_reports,
        baseline_neighborhood_reports,
    )
    baseline_ppl = float(
        official_perplexity(
            model,
            tok,
            ppl_text,
            device,
            max_input_length=100,
        )
    )
    baseline_metrics["PPL"] = baseline_ppl
    baseline_gates = candidate_gates(
        baseline_metrics,
        baseline_ppl,
        input_ppl=baseline_ppl,
        spe_target=args.spe_target,
        max_ppl_increase=args.max_ppl_increase,
        ppl_ceiling=args.ppl_ceiling,
    )
    gagd.write_json(
        output_dir / "baseline_forget_prompt_instances.json",
        baseline_forget_reports,
    )
    gagd.write_json(
        output_dir / "baseline_neighborhood_prompt_instances.json",
        baseline_neighborhood_reports,
    )
    gagd.write_json(
        output_dir / "baseline_metrics.json",
        {
            "metrics": baseline_metrics,
            "gates": baseline_gates,
        },
    )
    print(
        "Input checkpoint: "
        f"Eff={baseline_metrics['Eff']:.4f}, "
        f"Gen={baseline_metrics['Gen']:.4f}, "
        f"Spe={baseline_metrics['Spe']:.4f}, "
        f"PPL={baseline_ppl:.6f}"
    )
    if args.require_input_zero and (
        baseline_metrics["rewrite_failure_prompt_instances"] > 0
        or baseline_metrics["paraphrase_failure_prompt_instances"] > 0
    ):
        raise RuntimeError(
            "The supplied checkpoint is not an Eff=0/Gen=0 protected input. "
            "Use the exact protected Setting 5e repair or explicitly pass "
            "--no-require-input-zero."
        )

    selected_ids, selected_record_keys = select_neighborhood_lm_head_rows(
        tok,
        forget_records,
        baseline_neighborhood_reports,
        args.spe_target,
        args.row_selection,
    )
    selected_record_key_set = set(selected_record_keys)
    selected_row_report = {
        "row_selection": args.row_selection,
        "spe_target": args.spe_target,
        "selected_parent_record_count": len(selected_record_keys),
        "selected_parent_records": [
            {
                "record_index": record_index,
                "sampled_position": sampled_position,
                "baseline_record_spe": record_spe_by_key(
                    baseline_neighborhood_reports
                ).get((record_index, sampled_position)),
            }
            for record_index, sampled_position in selected_record_keys
        ],
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "selected_lm_head_tokens": {
            str(token_id): tok.decode([token_id]) for token_id in selected_ids
        },
        "selection_includes_target_new_and_target_true_rows": True,
    }
    gagd.write_json(output_dir / "selected_lm_head_rows.json", selected_row_report)
    if not selected_ids and not baseline_gates["qualified"]:
        raise RuntimeError(
            "No eligible LM-head rows were selected, so the requested Spe target "
            "cannot be searched."
        )

    selected_tensor = torch.tensor(
        selected_ids,
        dtype=torch.long,
        device=output_weight.device,
    )
    baseline_rows = (
        output_weight.index_select(0, selected_tensor).detach().clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )
    original_margin_tensor = torch.tensor(
        [float(report["margin"]) for report in baseline_forget_reports],
        dtype=torch.float32,
        device=output_weight.device,
    )
    required_margins = active.build_required_margin_tensor(
        original_margin_tensor,
        active_margin=args.active_margin,
        protected_margin_floor=args.protected_margin_floor,
    )
    neighborhood_groups = position_groups_by_record(neighborhood_instances)

    candidates: List[CandidateSnapshot] = []
    search_log: List[Dict[str, Any]] = []
    if baseline_gates["qualified"]:
        baseline_metrics["selected_lm_head_delta_norm"] = 0.0
        candidates.append(
            CandidateSnapshot(
                stage_target=0.0,
                delta_rows=torch.zeros_like(baseline_rows, device="cpu").float(),
                metrics=dict(baseline_metrics),
                ppl=baseline_ppl,
                gates=baseline_gates,
            )
        )

    if selected_ids and not (
        baseline_gates["qualified"] and args.stop_after_first_qualified
    ):
        print(
            "Caching official-compatible sparse-delta objectives for "
            f"{len(forget_instances)} protected and "
            f"{len(neighborhood_instances)} neighborhood prompt instances"
        )
        forget_caches = active.build_prompt_instance_delta_caches(
            model,
            tok,
            forget_instances,
            selected_ids,
            device,
            args.margin_batch_size,
            llama_like,
        )
        neighborhood_caches = active.build_prompt_instance_delta_caches(
            model,
            tok,
            neighborhood_instances,
            selected_ids,
            device,
            args.margin_batch_size,
            llama_like,
        )
        ppl_cache = build_official_ppl_delta_cache(
            model,
            tok,
            ppl_text,
            selected_ids,
            device,
        )

        retain_calibration_records = active.sample_retain_calibration(
            retain_records,
            args.retain_calibration_num,
            args.retain_calibration_seed,
        )
        reference_output_weight: Optional[torch.Tensor] = None
        reference_output_bias: Optional[torch.Tensor] = None
        if args.reference_model_path and args.retain_kl_mu > 0:
            print(
                "Loading frozen retain-KL output reference: "
                f"{args.reference_model_path}"
            )
            (
                reference_output_weight,
                reference_output_bias,
            ) = active.load_reference_output_layer(
                args.reference_model_path,
                gagd.torch_dtype(args.dtype),
            )
        retain_caches = (
            active.build_retain_kl_caches(
                model,
                reference_output_weight,
                reference_output_bias,
                tok,
                retain_calibration_records,
                selected_ids,
                device,
            )
            if args.retain_kl_mu > 0 or args.project_away_retain_hidden
            else []
        )
        del reference_output_weight, reference_output_bias
        gc.collect()

        retained_basis: Optional[torch.Tensor] = None
        if args.project_away_retain_hidden:
            retained_hidden_rows = [
                cache.hidden for cache in retain_caches if cache.hidden.numel()
            ]
            if not retained_hidden_rows:
                raise ValueError(
                    "Retain hidden projection was requested but no hidden states "
                    "were available"
                )
            retained_basis = active.orthonormal_row_basis(
                torch.cat(retained_hidden_rows, dim=0),
                max_rank=args.retain_projection_rank,
            )
            print(
                f"Projecting away from {retained_basis.shape[0]} deterministic "
                "retain hidden directions"
            )

        direction_basis: Optional[torch.Tensor] = None
        actual_rank = 0
        if args.repair_rank > 0:
            neighborhood_selected_positions = [
                position
                for position, instance in enumerate(neighborhood_instances)
                if (instance.record_index, instance.sampled_position)
                in selected_record_key_set
            ]
            selected_neighborhood_caches = [
                neighborhood_caches[position]
                for position in neighborhood_selected_positions
            ]
            direction_rows = torch.cat(
                [
                    _hidden_rows_from_prompt_caches(selected_neighborhood_caches),
                    _hidden_rows_from_prompt_caches(forget_caches),
                ],
                dim=0,
            )
            direction_rows = active.project_rows_away(
                direction_rows,
                retained_basis,
            )
            direction_basis = active.orthonormal_row_basis(
                direction_rows,
                max_rank=args.repair_rank,
            )
            actual_rank = int(direction_basis.shape[0])
            if actual_rank == 0:
                raise RuntimeError(
                    "All neighborhood repair directions vanished after retain "
                    "projection"
                )
            print(f"Using a rank-{actual_rank} context-local LM-head repair")

        delta_module = active.SelectedRowDelta(
            len(selected_ids),
            output_weight.shape[1],
            direction_basis=direction_basis,
            retained_basis=retained_basis,
            device=output_weight.device,
        )
        optimizer = active.make_repair_optimizer(
            delta_module,
            args.repair_optimizer,
            args.repair_lr,
        )

        global_step = 0
        for stage_target in schedule:
            optimization_target_fraction = min(
                0.999,
                (stage_target + args.spe_optimization_buffer) / 100.0,
            )
            for stage_step in range(1, args.repair_steps_per_stage + 1):
                global_step += 1
                optimizer.zero_grad(set_to_none=True)
                terms = _training_state(
                    delta_module,
                    forget_caches,
                    neighborhood_caches,
                    neighborhood_groups,
                    required_margins,
                    ppl_cache,
                    retain_caches,
                    spe_target_fraction=optimization_target_fraction,
                    neighborhood_tail_floor=args.neighborhood_tail_floor,
                    target_true_prob_floor=args.target_true_prob_floor,
                )
                loss = _loss_from_terms(terms, args)
                loss.backward()
                optimizer.step()
                (
                    norm_before_projection,
                    norm_after_projection,
                    norm_projected,
                ) = active.constrain_effective_delta_norm(
                    delta_module,
                    args.max_delta_norm,
                )
                with torch.no_grad():
                    current_terms = _training_state(
                        delta_module,
                        forget_caches,
                        neighborhood_caches,
                        neighborhood_groups,
                        required_margins,
                        ppl_cache,
                        retain_caches,
                        spe_target_fraction=optimization_target_fraction,
                        neighborhood_tail_floor=args.neighborhood_tail_floor,
                        target_true_prob_floor=args.target_true_prob_floor,
                    )
                    current_loss = _loss_from_terms(current_terms, args)
                    estimated_spe = 100.0 * float(
                        current_terms["global_spe_fraction"].detach().cpu()
                    )
                    forget_satisfied = active.all_margins_satisfied(
                        current_terms["forget_margins"],
                        required_margins,
                    )
                if (
                    stage_step == 1
                    or stage_step % args.log_every == 0
                    or stage_step == args.repair_steps_per_stage
                ):
                    row = _stage_log_row(
                        stage_target,
                        global_step,
                        current_terms,
                        current_loss,
                        norm_before_projection=norm_before_projection,
                        norm_after_projection=norm_after_projection,
                        norm_projected=norm_projected,
                    )
                    search_log.append(row)
                    print(
                        f"stage={stage_target:.1f} step={stage_step} "
                        f"estimated_Spe={estimated_spe:.3f} "
                        f"margin_slack="
                        f"{row['minimum_forget_margin_slack']:.4f} "
                        f"delta_norm={norm_after_projection:.4f}"
                    )
                if (
                    estimated_spe >= 100.0 * optimization_target_fraction
                    and forget_satisfied
                ):
                    break

            delta_rows = delta_module.effective_delta().detach().cpu().float().clone()
            (
                candidate_forget_reports,
                candidate_neighborhood_reports,
                candidate_metrics,
                candidate_ppl,
            ) = evaluate_sparse_candidate(
                model,
                tok,
                output_weight,
                selected_ids,
                baseline_rows,
                delta_rows,
                forget_instances,
                neighborhood_instances,
                device=device,
                batch_size=args.margin_batch_size,
                llama_like=llama_like,
                ppl_text=ppl_text,
            )
            candidate_metrics["PPL"] = candidate_ppl
            candidate_metrics["minimum_forget_margin"] = min(
                (float(report["margin"]) for report in candidate_forget_reports),
                default=None,
            )
            candidate_metrics["minimum_record_spe"] = min(
                record_spe_by_key(candidate_neighborhood_reports).values(),
                default=None,
            )
            candidate_metrics["selected_lm_head_delta_norm"] = float(delta_rows.norm())
            gates = candidate_gates(
                candidate_metrics,
                candidate_ppl,
                input_ppl=baseline_ppl,
                spe_target=args.spe_target,
                max_ppl_increase=args.max_ppl_increase,
                ppl_ceiling=args.ppl_ceiling,
            )
            candidate = CandidateSnapshot(
                stage_target=stage_target,
                delta_rows=delta_rows,
                metrics=candidate_metrics,
                ppl=candidate_ppl,
                gates=gates,
            )
            candidates.append(candidate)
            gagd.write_json(
                output_dir / f"candidate_stage_{stage_target:g}.json",
                {
                    **snapshot_to_json(candidate),
                    "forget_prompt_instances": candidate_forget_reports,
                    "neighborhood_prompt_instances": (candidate_neighborhood_reports),
                },
            )
            active.write_jsonl(output_dir / "repair_log.jsonl", search_log)
            print(
                f"Exact stage candidate: Eff={candidate_metrics['Eff']:.4f}, "
                f"Gen={candidate_metrics['Gen']:.4f}, "
                f"Spe={candidate_metrics['Spe']:.4f}, "
                f"PPL={candidate_ppl:.6f}, "
                f"qualified={gates['qualified']}"
            )
            if gates["qualified"] and args.stop_after_first_qualified:
                break

    gagd.write_json(
        output_dir / "candidate_search.json",
        {
            "baseline_ppl": baseline_ppl,
            "ppl_limit": allowed_ppl(
                baseline_ppl,
                args.max_ppl_increase,
                args.ppl_ceiling,
            ),
            "spe_target": args.spe_target,
            "candidates": [snapshot_to_json(candidate) for candidate in candidates],
        },
    )
    active.write_jsonl(output_dir / "repair_log.jsonl", search_log)

    qualified = [candidate for candidate in candidates if candidate.gates["qualified"]]
    selected_best_effort = False
    if qualified:
        selected_candidate = min(qualified, key=candidate_priority)
    elif args.save_best_effort:
        safe_candidates = [
            candidate
            for candidate in candidates
            if candidate.gates["eff_zero"]
            and candidate.gates["gen_zero"]
            and candidate.gates["ppl_met"]
        ]
        if not safe_candidates:
            raise RuntimeError(
                "No candidate preserved Eff=0, Gen=0, and the PPL ceiling; "
                "nothing was saved."
            )
        selected_candidate = min(safe_candidates, key=best_effort_priority)
        selected_best_effort = True
    else:
        raise RuntimeError(
            f"No candidate reached Eff=0, Gen=0, Spe>={args.spe_target:g}, "
            "and the configured PPL ceiling. No checkpoint was saved. Inspect "
            "candidate_search.json, then expand the stage schedule/rank/steps "
            "instead of weakening the acceptance gates."
        )

    set_selected_lm_head_rows(
        output_weight,
        selected_ids,
        baseline_rows,
        selected_candidate.delta_rows,
    )
    final_forget_reports = score_prompt_instances(
        model,
        tok,
        forget_instances,
        device,
        args.margin_batch_size,
        llama_like,
    )
    final_neighborhood_reports = score_prompt_instances(
        model,
        tok,
        neighborhood_instances,
        device,
        args.margin_batch_size,
        llama_like,
    )
    final_metrics = official_metrics_from_reports(
        final_forget_reports,
        final_neighborhood_reports,
    )
    final_ppl = float(
        official_perplexity(
            model,
            tok,
            ppl_text,
            device,
            max_input_length=100,
        )
    )
    final_metrics["PPL"] = final_ppl
    final_metrics["selected_lm_head_delta_norm"] = float(
        selected_candidate.delta_rows.norm()
    )
    final_gates = candidate_gates(
        final_metrics,
        final_ppl,
        input_ppl=baseline_ppl,
        spe_target=args.spe_target,
        max_ppl_increase=args.max_ppl_increase,
        ppl_ceiling=args.ppl_ceiling,
    )
    final_hard_safe = (
        final_gates["eff_zero"] and final_gates["gen_zero"] and final_gates["ppl_met"]
    )
    if (not selected_best_effort and not final_gates["qualified"]) or (
        selected_best_effort and not final_hard_safe
    ):
        set_selected_lm_head_rows(
            output_weight,
            selected_ids,
            baseline_rows,
            torch.zeros_like(baseline_rows),
        )
        raise RuntimeError(
            "Selected candidate changed when materialized; final hard gates "
            "failed and no checkpoint was saved."
        )

    gagd.write_json(
        output_dir / "final_forget_prompt_instances.json",
        final_forget_reports,
    )
    gagd.write_json(
        output_dir / "final_neighborhood_prompt_instances.json",
        final_neighborhood_reports,
    )

    official_result: Optional[Dict[str, Any]] = None
    official_gates: Optional[Dict[str, Any]] = None
    if args.run_official_mcf_eval:
        print("Running full official MCF evaluation before checkpoint saving")
        official_result = evaluate_loaded_model_official(
            method=METHOD,
            model=model,
            tok=tok,
            model_dir=f"in-memory:{METHOD}",
            mcf_path=gagd.resolve_output_path(args.mcf_cache_path),
            wikidata_dir=wikidata_dir,
            out_path=output_dir / "official_eval.json",
            unlearn_num=args.forget_num,
            retain_num=args.retain_num,
            seed=args.seed,
            sample_mode=args.sample_mode,
            skip_ppl=False,
        )
        official_metrics, official_ppl = official_result_metrics(official_result)
        official_gates = candidate_gates(
            official_metrics,
            official_ppl,
            input_ppl=baseline_ppl,
            spe_target=args.spe_target,
            max_ppl_increase=args.max_ppl_increase,
            ppl_ceiling=args.ppl_ceiling,
        )
        official_hard_safe = (
            official_gates["eff_zero"]
            and official_gates["gen_zero"]
            and official_gates["ppl_met"]
        )
        if (not selected_best_effort and not official_gates["qualified"]) or (
            selected_best_effort and not official_hard_safe
        ):
            set_selected_lm_head_rows(
                output_weight,
                selected_ids,
                baseline_rows,
                torch.zeros_like(baseline_rows),
            )
            raise RuntimeError(
                "The full official evaluator rejected the locally selected "
                "candidate. The input LM head was restored and no checkpoint "
                "was saved."
            )

    if input_weight.data_ptr() != input_storage_pointer:
        raise RuntimeError("Input embedding storage changed during repair")
    if input_weight._version != input_tensor_version:
        raise RuntimeError("Input embedding values changed during repair")
    if input_weight.requires_grad:
        raise RuntimeError("Input embeddings unexpectedly became trainable")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("A model parameter unexpectedly became trainable")

    config_used.update(
        {
            "selected_stage_target": selected_candidate.stage_target,
            "selected_best_effort": selected_best_effort,
            "selected_lm_head_token_ids": selected_ids,
            "selected_lm_head_delta_norm": float(selected_candidate.delta_rows.norm()),
            "input_checkpoint_ppl": baseline_ppl,
            "final_local_metrics": final_metrics,
            "final_local_gates": final_gates,
            "final_official_gates": official_gates,
        }
    )
    gagd.write_json(output_dir / "config_used.json", config_used)
    repair_summary = {
        "method": METHOD,
        "model_path": args.model_path,
        "source_experiment_config_path": str(config_path),
        "preserved_5e_overlap_alphas": preserved_alphas,
        "forget_records": len(forget_records),
        "forget_prompt_instances": len(forget_instances),
        "neighborhood_prompt_instances": len(neighborhood_instances),
        "selected_parent_records": len(selected_record_keys),
        "selected_lm_head_rows": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "selected_lm_head_delta_norm": float(selected_candidate.delta_rows.norm()),
        "selected_stage_target": selected_candidate.stage_target,
        "selected_best_effort": selected_best_effort,
        "baseline": {
            "metrics": baseline_metrics,
            "gates": baseline_gates,
        },
        "final_local": {
            "metrics": final_metrics,
            "gates": final_gates,
        },
        "final_official": (
            {
                "metrics": official_result_metrics(official_result)[0],
                "PPL": official_result_metrics(official_result)[1],
                "gates": official_gates,
            }
            if official_result is not None
            else None
        ),
        "input_embeddings_modified": False,
        "transformer_parameters_trainable": 0,
        "ppl_limit": allowed_ppl(
            baseline_ppl,
            args.max_ppl_increase,
            args.ppl_ceiling,
        ),
        "checkpoint_saved": bool(args.save_model),
    }
    gagd.write_json(output_dir / "repair_summary.json", repair_summary)

    if args.save_model:
        checkpoint_dir = output_dir / "checkpoint"
        active.save_repair_checkpoint(
            model,
            tok,
            checkpoint_dir,
            repair_config=config_used,
        )
        print(f"Saved qualified checkpoint to {checkpoint_dir}")

    print(
        "Selected result: "
        f"Eff={final_metrics['Eff']:.4f}, "
        f"Gen={final_metrics['Gen']:.4f}, "
        f"Spe={final_metrics['Spe']:.4f}, "
        f"PPL={final_ppl:.6f}; outputs in {output_dir}"
    )


if __name__ == "__main__":
    main()
