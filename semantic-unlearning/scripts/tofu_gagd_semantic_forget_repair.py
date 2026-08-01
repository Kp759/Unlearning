#!/usr/bin/env python3
"""Add semantic abstention preference to the existing TOFU active repair.

The underlying Setting 5e repair remains responsible for absolute sensitive-
answer suppression, utility constraints, BF16 materialization, and checkpoint
serialization. This wrapper adds trainable ``Unknown`` answer rows, a per-request
preference margin, semantic candidate ordering, and a post-reload hard gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import gagd_compare as gagd
import tofu_gagd_active_forget_repair as repair
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "tofu_semantic_active_forget_repair"
ORIGINAL_BUILD_PARSER = repair.build_parser
ORIGINAL_VALIDATE_ARGS = repair.validate_args
ORIGINAL_TARGET_TERMS = repair.target_objective_terms
ORIGINAL_REPAIR_METRICS = repair.repair_metrics
ORIGINAL_PARTITION_ROWS = repair.partition_active_forget_row_ids
ORIGINAL_LOAD_INSTANCES = tofu.load_tofu_repair_instances


class _State:
    args: Optional[argparse.Namespace] = None
    baseline_abstention_nll: Optional[torch.Tensor] = None


STATE = _State()


def _abstention_instances(
    forget_instances: Sequence[tofu.TOFUAnswerInstance],
    answer: str,
) -> List[tofu.TOFUAnswerInstance]:
    return [
        tofu.TOFUAnswerInstance(
            **{
                **asdict(instance),
                "split": "abstention",
                "answer": str(answer).strip(),
            }
        )
        for instance in forget_instances
    ]


abstention_instances = _abstention_instances


def _normal_utility_positions(
    instances: Sequence[tofu.TOFUAnswerInstance],
) -> List[int]:
    return [
        index
        for index, instance in enumerate(instances)
        if instance.split != "abstention"
    ]


def _abstention_positions(
    instances: Sequence[tofu.TOFUAnswerInstance],
) -> List[int]:
    return [
        index
        for index, instance in enumerate(instances)
        if instance.split == "abstention"
    ]


def _select(values: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    index = torch.tensor(
        positions,
        dtype=torch.long,
        device=values.device,
    )
    return values.index_select(0, index)


def semantic_preference_terms(
    sensitive_nll: torch.Tensor,
    abstention_nll: torch.Tensor,
    baseline_abstention_nll: torch.Tensor,
    *,
    preference_margin: float,
) -> Dict[str, torch.Tensor]:
    """Compute differentiable abstention-preference and preservation hinges."""
    if sensitive_nll.shape != abstention_nll.shape:
        raise ValueError("Sensitive and abstention NLL tensors must align")
    if abstention_nll.shape != baseline_abstention_nll.shape:
        raise ValueError("Current and baseline abstention NLL tensors must align")

    # log P(Unknown) - log P(sensitive) = NLL_sensitive - NLL_Unknown.
    preference_slack = (
        sensitive_nll - abstention_nll - float(preference_margin)
    )
    preference_error = torch.relu(-preference_slack)
    abstention_degradation = torch.relu(
        abstention_nll - baseline_abstention_nll.to(abstention_nll)
    )
    return {
        "preference_hinge": preference_error.square().mean(),
        "hardest_preference_hinge": preference_error.square().max(),
        "abstention_preservation_hinge": (
            abstention_degradation.square().mean()
        ),
        "preference_slack": preference_slack,
    }


def semantic_metrics(
    sensitive_nll: torch.Tensor,
    abstention_nll: torch.Tensor,
    baseline_abstention_nll: torch.Tensor,
    *,
    preference_margin: float,
    tolerance: float,
) -> Dict[str, Any]:
    gap = sensitive_nll - abstention_nll
    violations = gap < (float(preference_margin) - tolerance)
    sensitive_preferred = sensitive_nll < abstention_nll
    abstention_degraded = abstention_nll > (
        baseline_abstention_nll.to(abstention_nll) + tolerance
    )
    return {
        "preference_margin": float(preference_margin),
        "preference_violation_count": int(violations.sum().detach().cpu()),
        "preference_satisfied_rate": float(
            (~violations).float().mean().detach().cpu()
        ),
        "sensitive_preference_rate": float(
            sensitive_preferred.float().mean().detach().cpu()
        ),
        "minimum_log_probability_preference_margin": float(
            gap.min().detach().cpu()
        ),
        "mean_log_probability_preference_margin": float(
            gap.mean().detach().cpu()
        ),
        "abstention_probability_mean": float(
            torch.exp(-abstention_nll.float()).mean().detach().cpu()
        ),
        "abstention_probability_min": float(
            torch.exp(-abstention_nll.float()).min().detach().cpu()
        ),
        "abstention_degradation_count": int(
            abstention_degraded.sum().detach().cpu()
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = ORIGINAL_BUILD_PARSER()
    parser.description = __doc__
    parser.add_argument("--abstention-answer", default="Unknown")
    parser.add_argument("--preference-margin", type=float, default=1.0)
    parser.add_argument("--preference-hinge-weight", type=float, default=100.0)
    parser.add_argument(
        "--hardest-preference-hinge-weight",
        type=float,
        default=100.0,
    )
    parser.add_argument(
        "--abstention-preservation-weight",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--require-preference-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--semantic-utility-constraint-mode",
        choices=["aggregate", "per-example"],
        default="per-example",
        help=(
            "Utility mode used by semantic candidate optimization and ranking. "
            "The underlying materialization mode may remain aggregate; a "
            "reloaded per-example utility postcheck is applied afterward."
        ),
    )
    original_parse_args = parser.parse_args

    def parse_args(*args: Any, **kwargs: Any) -> argparse.Namespace:
        parsed = original_parse_args(*args, **kwargs)
        STATE.args = parsed
        return parsed

    parser.parse_args = parse_args  # type: ignore[method-assign]
    return parser


def validate_args(args: argparse.Namespace) -> None:
    ORIGINAL_VALIDATE_ARGS(args)
    if not str(args.abstention_answer).strip():
        raise ValueError("--abstention-answer must be non-empty")
    for name in (
        "preference_margin",
        "preference_hinge_weight",
        "hardest_preference_hinge_weight",
        "abstention_preservation_weight",
    ):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )


def load_tofu_repair_instances(
    args: argparse.Namespace,
    tok: Any,
) -> Tuple[
    List[tofu.TOFUAnswerInstance],
    List[tofu.TOFUAnswerInstance],
    List[tofu.TOFUAnswerInstance],
]:
    forget, full_retain, utility = ORIGINAL_LOAD_INSTANCES(args, tok)
    abstention = _abstention_instances(forget, args.abstention_answer)
    return forget, full_retain, [*utility, *abstention]


def partition_active_forget_row_ids(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    active_positions: Sequence[int],
    *,
    max_length: int,
    protected_instances: Sequence[tofu.TOFUAnswerInstance] = (),
) -> Tuple[List[int], List[int]]:
    normal_protected = [
        instance
        for instance in protected_instances
        if instance.split != "abstention"
    ]
    abstention = [
        instance
        for instance in protected_instances
        if instance.split == "abstention"
    ]
    sensitive_ids, sensitive_overlap = ORIGINAL_PARTITION_ROWS(
        tok,
        instances,
        active_positions,
        max_length=max_length,
        protected_instances=normal_protected,
    )
    normal_protected_ids = repair._answer_row_ids(
        tok,
        normal_protected,
        max_length=max_length,
    )
    abstention_ids_all = repair._answer_row_ids(
        tok,
        abstention,
        max_length=max_length,
    )
    abstention_ids = abstention_ids_all - normal_protected_ids
    abstention_overlap = abstention_ids_all & normal_protected_ids
    return (
        sorted(set(sensitive_ids) | abstention_ids),
        sorted(set(sensitive_overlap) | abstention_overlap),
    )


def target_objective_terms(
    current_forget_nll: torch.Tensor,
    required_forget_nll: torch.Tensor,
    current_utility_nll: torch.Tensor,
    required_utility_nll: torch.Tensor,
    *,
    reference_utility_nll: Optional[torch.Tensor] = None,
    utility_instances: Sequence[tofu.TOFUAnswerInstance] = (),
    minimum_utility_probability_ratio: float = 0.9999998,
    utility_constraint_mode: str = "per-example",
) -> Dict[str, torch.Tensor]:
    args = STATE.args
    if args is None:
        raise RuntimeError("Semantic repair arguments were not initialized")
    normal_positions = _normal_utility_positions(utility_instances)
    abstention_positions = _abstention_positions(utility_instances)
    if len(abstention_positions) != current_forget_nll.numel():
        raise ValueError(
            "Semantic repair requires one abstention answer per forget instance"
        )
    normal_instances = [utility_instances[index] for index in normal_positions]
    normal_reference = (
        _select(reference_utility_nll, normal_positions)
        if reference_utility_nll is not None
        else None
    )
    terms = ORIGINAL_TARGET_TERMS(
        current_forget_nll,
        required_forget_nll,
        _select(current_utility_nll, normal_positions),
        _select(required_utility_nll, normal_positions),
        reference_utility_nll=normal_reference,
        utility_instances=normal_instances,
        minimum_utility_probability_ratio=minimum_utility_probability_ratio,
        utility_constraint_mode=args.semantic_utility_constraint_mode,
    )
    current_abstention = _select(current_utility_nll, abstention_positions)
    if STATE.baseline_abstention_nll is None:
        STATE.baseline_abstention_nll = current_abstention.detach().clone()
    semantic = semantic_preference_terms(
        current_forget_nll,
        current_abstention,
        STATE.baseline_abstention_nll,
        preference_margin=args.preference_margin,
    )

    # The existing training loop consumes only these three hinge fields. Fold
    # the semantic components into them with exact weight normalization.
    forget_scale = max(float(args.forget_hinge_weight), 1e-12)
    hardest_scale = max(float(args.hardest_forget_hinge_weight), 1e-12)
    utility_scale = max(float(args.utility_hinge_weight), 1e-12)
    terms["forget_hinge"] = (
        terms["forget_hinge"]
        + float(args.preference_hinge_weight)
        / forget_scale
        * semantic["preference_hinge"]
    )
    terms["hardest_forget_hinge"] = (
        terms["hardest_forget_hinge"]
        + float(args.hardest_preference_hinge_weight)
        / hardest_scale
        * semantic["hardest_preference_hinge"]
    )
    terms["utility_hinge"] = (
        terms["utility_hinge"]
        + float(args.abstention_preservation_weight)
        / utility_scale
        * semantic["abstention_preservation_hinge"]
    )
    terms.update(semantic)
    return terms


def repair_metrics(
    forget_nll: torch.Tensor,
    required_forget_nll: torch.Tensor,
    utility_nll: torch.Tensor,
    required_utility_nll: torch.Tensor,
    utility_instances: Sequence[tofu.TOFUAnswerInstance],
    delta_rows: torch.Tensor,
    *,
    target_nll: float,
    tolerance: float,
    reference_utility_nll: Optional[torch.Tensor] = None,
    minimum_utility_probability_ratio: float = 0.9999998,
    utility_constraint_mode: str = "per-example",
) -> Dict[str, Any]:
    args = STATE.args
    if args is None:
        raise RuntimeError("Semantic repair arguments were not initialized")
    normal_positions = _normal_utility_positions(utility_instances)
    abstention_positions = _abstention_positions(utility_instances)
    normal_instances = [utility_instances[index] for index in normal_positions]
    normal_reference = (
        _select(reference_utility_nll, normal_positions)
        if reference_utility_nll is not None
        else None
    )
    metrics = ORIGINAL_REPAIR_METRICS(
        forget_nll,
        required_forget_nll,
        _select(utility_nll, normal_positions),
        _select(required_utility_nll, normal_positions),
        normal_instances,
        delta_rows,
        target_nll=target_nll,
        tolerance=tolerance,
        reference_utility_nll=normal_reference,
        minimum_utility_probability_ratio=minimum_utility_probability_ratio,
        utility_constraint_mode=args.semantic_utility_constraint_mode,
    )
    current_abstention = _select(utility_nll, abstention_positions)
    if STATE.baseline_abstention_nll is None:
        STATE.baseline_abstention_nll = current_abstention.detach().clone()
    semantic = semantic_metrics(
        forget_nll,
        current_abstention,
        STATE.baseline_abstention_nll,
        preference_margin=args.preference_margin,
        tolerance=tolerance,
    )
    metrics.update(semantic)
    metrics["utility_constraint_violation_count_without_preference"] = int(
        metrics["utility_constraint_violation_count"]
    )
    if args.require_preference_constraints:
        metrics["utility_constraint_violation_count"] += int(
            semantic["preference_violation_count"]
        )
    return metrics


def candidate_priority(metrics: Mapping[str, Any]) -> Tuple[int, int, int, int, float, float]:
    return (
        int(metrics["utility_constraint_violation_count_without_preference"]),
        int(metrics["preference_violation_count"]),
        int(metrics["active_forget_instance_count"]),
        int(metrics["buffered_forget_constraint_unmet_count"]),
        -float(metrics["minimum_log_probability_preference_margin"]),
        float(metrics["selected_lm_head_delta_norm"]),
    )


def _postcheck() -> None:
    args = STATE.args
    if args is None:
        raise RuntimeError("Semantic repair arguments were not initialized")
    output_dir = Path(args.output_dir).resolve()
    checkpoint = output_dir / "checkpoint"
    if not checkpoint.exists() or not args.save_model:
        return

    model, tok = gagd.load_model_and_tokenizer(
        repair._model_args(args, str(checkpoint)),
        for_training=False,
    )
    device = gagd.first_device(model)
    forget, _, utility = ORIGINAL_LOAD_INSTANCES(args, tok)
    abstention = _abstention_instances(forget, args.abstention_answer)
    sensitive_nll = tofu.score_answer_instances(
        model,
        tok,
        forget,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    abstention_nll = tofu.score_answer_instances(
        model,
        tok,
        abstention,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    candidate_utility_nll = tofu.score_answer_instances(
        model,
        tok,
        utility,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    baseline = STATE.baseline_abstention_nll
    if baseline is None:
        raise RuntimeError("Missing baseline abstention NLL for postcheck")
    postcheck = semantic_metrics(
        sensitive_nll,
        abstention_nll,
        baseline.to(abstention_nll),
        preference_margin=args.preference_margin,
        tolerance=args.comparison_tolerance,
    )
    reference_model, reference_tok = gagd.load_model_and_tokenizer(
        repair._model_args(args, args.reference_model_path),
        for_training=False,
    )
    reference_device = gagd.first_device(reference_model)
    reference_utility_nll = tofu.score_answer_instances(
        reference_model,
        reference_tok,
        utility,
        reference_device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach().to(candidate_utility_nll)
    per_example_utility_ratios = torch.exp(
        reference_utility_nll - candidate_utility_nll
    )
    per_example_utility_violations = int(
        (
            per_example_utility_ratios + 1e-12
            < args.min_utility_probability_ratio
        ).sum().cpu()
    )
    preference_passed = bool(
        not args.require_preference_constraints
        or postcheck["preference_violation_count"] == 0
    )
    utility_passed = bool(
        not args.require_utility_constraints
        or per_example_utility_violations == 0
    )
    postcheck.update(
        {
            "method": METHOD,
            "abstention_answer": args.abstention_answer,
            "checkpoint": str(checkpoint),
            "checkpoint_reloaded": True,
            "preference_required": bool(args.require_preference_constraints),
            "preference_passed": preference_passed,
            "per_example_utility_required": bool(
                args.require_utility_constraints
            ),
            "per_example_utility_violation_count": (
                per_example_utility_violations
            ),
            "minimum_per_example_utility_probability_ratio": float(
                per_example_utility_ratios.min().cpu()
            ),
            "mean_per_example_utility_probability_ratio": float(
                per_example_utility_ratios.mean().cpu()
            ),
            "required_minimum_utility_probability_ratio": (
                args.min_utility_probability_ratio
            ),
            "utility_passed": utility_passed,
            "passed": preference_passed and utility_passed,
        }
    )
    (output_dir / "semantic_preference_postcheck.json").write_text(
        json.dumps(postcheck, indent=2) + "\n",
        encoding="utf-8",
    )

    summary_path = output_dir / "repair_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "method": METHOD,
            "abstention_answer": args.abstention_answer,
            "preference_margin": args.preference_margin,
            "semantic_preference_postcheck": postcheck,
            "qualified": bool(summary.get("qualified")) and postcheck["passed"],
        }
    )
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    selected_path = output_dir / "selected_lm_head_rows.json"
    if selected_path.exists():
        selected = json.loads(selected_path.read_text(encoding="utf-8"))
        selected["selection_source"] = (
            "initially active sensitive rows plus abstention rows, excluding "
            "protected retain and utility answer rows"
        )
        selected["semantic_abstention_answer"] = args.abstention_answer
        selected_path.write_text(
            json.dumps(selected, indent=2) + "\n",
            encoding="utf-8",
        )

    del model, reference_model, reference_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not postcheck["passed"] and not args.save_best_effort:
        shutil.rmtree(checkpoint)
        raise RuntimeError(
            "Reloaded semantic checkpoint failed hard postchecks: "
            f"preference_violations={postcheck['preference_violation_count']}, "
            f"utility_violations="
            f"{postcheck['per_example_utility_violation_count']}. The saved "
            "checkpoint was removed."
        )


def main() -> None:
    repair.build_parser = build_parser
    repair.validate_args = validate_args
    repair.target_objective_terms = target_objective_terms
    repair.repair_metrics = repair_metrics
    repair.candidate_priority = candidate_priority
    repair.partition_active_forget_row_ids = partition_active_forget_row_ids
    tofu.load_tofu_repair_instances = load_tofu_repair_instances
    repair.main()
    _postcheck()


if __name__ == "__main__":
    main()
