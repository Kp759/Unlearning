#!/usr/bin/env python3
"""Repair active TOFU forget cases to an absolute answer-probability target.

The input is normally the TOFU Setting 5e checkpoint produced by
``tofu_gagd_setting5e_restore.py``.  This stage never reruns GA/GD.  It freezes
the transformer and input embeddings, unties the LM head when necessary, and
optimizes only LM-head rows used by initially active forget answers.

For every clean TOFU forget example the evaluator reports

    answer_probability = exp(-mean_answer_nll).

The default target is 2e-5.  Initially active examples receive a buffered NLL
floor above ``-log(2e-5)``; initially passing examples are protected so the
failure set cannot migrate.  Deterministic retain, real-author, and world-fact
calibration answers receive per-example NLL ceilings.

No normal checkpoint is saved unless every materialized forget answer meets
the hard probability target. Utility constraints are optimized and reported;
they can optionally be promoted to hard save gates.
"""

from __future__ import annotations

import argparse
import gc
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "tofu_active_forget_repair"


@dataclass(frozen=True)
class CandidateSnapshot:
    step: int
    delta_rows: torch.Tensor
    metrics: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--forget-split",
        choices=sorted(tofu.PAIRED_RETAIN_SPLITS),
        default="forget05",
    )
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--retain-calibration-num", type=int, default=128)
    parser.add_argument("--real-authors-calibration-num", type=int, default=64)
    parser.add_argument("--world-facts-calibration-num", type=int, default=64)
    parser.add_argument("--calibration-seed", type=int, default=2718)

    parser.add_argument(
        "--target-forget-answer-probability",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--target-nll-buffer",
        type=float,
        default=0.25,
        help="Extra NLL required before BF16 materialization.",
    )
    parser.add_argument(
        "--utility-nll-tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum per-example utility NLL increase relative to the input "
            "Setting 5e checkpoint."
        ),
    )
    parser.add_argument("--forget-hinge-weight", type=float, default=100.0)
    parser.add_argument("--utility-hinge-weight", type=float, default=10.0)
    parser.add_argument("--delta-l2-lambda", type=float, default=1e-5)
    parser.add_argument("--repair-steps", type=int, default=500)
    parser.add_argument("--repair-lr", type=float, default=1e-2)
    parser.add_argument(
        "--repair-optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adamw",
    )
    parser.add_argument(
        "--repair-rank",
        type=int,
        default=0,
        help="Zero learns a full selected-row delta; positive values use a rank basis.",
    )
    parser.add_argument("--utility-projection-rank", type=int, default=128)
    parser.add_argument(
        "--basis-max-rows",
        type=int,
        default=512,
        help="Deterministic hidden-row cap before SVD basis construction.",
    )
    parser.add_argument(
        "--project-away-utility-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--comparison-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--materialization-relative-tolerance",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--require-utility-constraints",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Make every utility NLL ceiling a hard save gate. By default the "
            "ceilings remain objective terms and candidate tie-breakers while "
            "the absolute forget target is the hard gate."
        ),
    )
    parser.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-best-effort",
        action="store_true",
        help=(
            "Diagnostic only: save the best candidate even when "
            "the hard forget target is missed."
        ),
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected = tofu.PAIRED_RETAIN_SPLITS[args.forget_split]
    if args.retain_split != expected:
        raise ValueError(
            f"{args.forget_split} must be paired with {expected}, "
            f"not {args.retain_split}"
        )
    if not 0.0 < args.target_forget_answer_probability < 1.0:
        raise ValueError("--target-forget-answer-probability must lie in (0,1)")
    positive = (
        "forget_num",
        "retain_num",
        "retain_calibration_num",
        "real_authors_calibration_num",
        "world_facts_calibration_num",
        "repair_steps",
        "batch_size",
        "max_length",
        "log_every",
        "basis_max_rows",
    )
    for name in positive:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "target_nll_buffer",
        "utility_nll_tolerance",
        "forget_hinge_weight",
        "utility_hinge_weight",
        "delta_l2_lambda",
        "repair_rank",
        "utility_projection_rank",
        "comparison_tolerance",
        "materialization_relative_tolerance",
    )
    for name in nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if not math.isfinite(args.repair_lr) or args.repair_lr <= 0:
        raise ValueError("--repair-lr must be finite and positive")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("--max-delta-norm must be finite and non-negative")


def build_required_forget_nll(
    original_nll: torch.Tensor,
    *,
    target_probability: float,
    target_nll_buffer: float,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    target_nll = -math.log(target_probability)
    buffered_target = target_nll + target_nll_buffer
    target = original_nll.new_full(original_nll.shape, target_nll)
    buffered = original_nll.new_full(original_nll.shape, buffered_target)
    initially_active = original_nll < target
    passing_requirement = torch.minimum(original_nll, buffered).clamp_min(
        target_nll
    )
    required = torch.where(
        initially_active,
        buffered,
        passing_requirement,
    )
    return required, initially_active, target_nll


def active_forget_row_ids(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    active_positions: Sequence[int],
    *,
    max_length: int,
) -> List[int]:
    selected: set[int] = set()
    specials = gagd.special_token_ids(tok)
    for position in active_positions:
        full_ids, prompt_length = tofu.answer_sequence_components(
            tok,
            instances[position],
            max_length,
        )
        selected.update(full_ids[prompt_length:])
    return sorted(selected - specials)


def target_objective_terms(
    current_forget_nll: torch.Tensor,
    required_forget_nll: torch.Tensor,
    current_utility_nll: torch.Tensor,
    required_utility_nll: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if current_forget_nll.shape != required_forget_nll.shape:
        raise ValueError("The objective must contain every forget instance")
    if current_utility_nll.shape != required_utility_nll.shape:
        raise ValueError("The objective must contain every utility instance")
    forget_errors = torch.relu(
        required_forget_nll.to(current_forget_nll) - current_forget_nll
    )
    utility_errors = torch.relu(
        current_utility_nll - required_utility_nll.to(current_utility_nll)
    )
    return {
        "forget_hinge": forget_errors.square().mean(),
        "utility_hinge": utility_errors.square().mean(),
        "forget_slack": (
            current_forget_nll - required_forget_nll.to(current_forget_nll)
        ),
        "utility_slack": (
            required_utility_nll.to(current_utility_nll) - current_utility_nll
        ),
    }


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
) -> Dict[str, Any]:
    target_active = forget_nll < (target_nll - tolerance)
    buffered_unmet = forget_nll < (
        required_forget_nll.to(forget_nll) - tolerance
    )
    utility_violations = utility_nll > (
        required_utility_nll.to(utility_nll) + tolerance
    )
    split_probabilities = tofu.group_answer_probabilities(
        utility_nll,
        utility_instances,
    )
    macro_values = [
        split_probabilities[name]
        for name in tofu.UTILITY_SPLITS
        if name in split_probabilities
    ]
    probabilities = torch.exp(-forget_nll)
    return {
        "active_forget_instance_count": int(target_active.sum().detach().cpu()),
        "buffered_forget_constraint_unmet_count": int(
            buffered_unmet.sum().detach().cpu()
        ),
        "forget_answer_probability_mean": float(
            probabilities.mean().detach().cpu()
        ),
        "forget_answer_probability_max": float(
            probabilities.max().detach().cpu()
        ),
        "minimum_forget_answer_nll": float(forget_nll.min().detach().cpu()),
        "minimum_forget_nll_slack": float(
            (
                forget_nll
                - required_forget_nll.to(forget_nll)
            )
            .min()
            .detach()
            .cpu()
        ),
        "utility_constraint_violation_count": int(
            utility_violations.sum().detach().cpu()
        ),
        "minimum_utility_nll_slack": float(
            (
                required_utility_nll.to(utility_nll)
                - utility_nll
            )
            .min()
            .detach()
            .cpu()
        ),
        "utility_answer_probability_by_split": split_probabilities,
        "utility_macro_answer_probability": (
            sum(macro_values) / len(macro_values)
            if macro_values
            else float("nan")
        ),
        "selected_lm_head_delta_norm": float(delta_rows.norm().detach().cpu()),
    }


def candidate_priority(metrics: Dict[str, Any]) -> Tuple[int, int, int, float, float]:
    return (
        int(metrics["active_forget_instance_count"]),
        int(metrics["buffered_forget_constraint_unmet_count"]),
        int(metrics["utility_constraint_violation_count"]),
        -float(metrics["minimum_forget_answer_nll"]),
        float(metrics["selected_lm_head_delta_norm"]),
    )


def _hidden_rows(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
) -> torch.Tensor:
    rows = [cache.hidden for cache in caches if cache.hidden.numel()]
    if not rows:
        raise ValueError("No answer-token hidden states were cached")
    return torch.cat(rows, dim=0)


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )


def _reports(
    instances: Sequence[tofu.TOFUAnswerInstance],
    nll: torch.Tensor,
    required_nll: torch.Tensor,
    *,
    target_probability: Optional[float] = None,
) -> List[Dict[str, Any]]:
    rows = tofu.instance_reports(instances, nll, required_nll)
    for row in rows:
        if target_probability is not None:
            row["target_answer_probability"] = target_probability
            row["active"] = (
                float(row["answer_probability"]) > target_probability
            )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_config_path, source_config = tofu.discover_source_config(
        args.model_path
    )
    configured_target_nll = -math.log(
        args.target_forget_answer_probability
    )
    config_used = {
        **vars(args),
        "method": METHOD,
        "source_experiment_config_path": (
            str(source_config_path) if source_config_path is not None else None
        ),
        "source_experiment_config": source_config,
        "target_forget_answer_nll": configured_target_nll,
        "buffered_target_forget_answer_nll": (
            configured_target_nll + args.target_nll_buffer
        ),
        "parameter_scope": "initially_active_forget_lm_head_rows_only",
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
    }
    gagd.write_json(output_dir / "config_used.json", config_used)

    tok_for_data = AutoTokenizer.from_pretrained(args.model_path)
    if tok_for_data.pad_token is None:
        tok_for_data.pad_token = tok_for_data.eos_token
    print("Loading deterministic TOFU forget and utility instances")
    forget_instances, utility_instances = tofu.load_tofu_calibration_instances(
        args,
        tok_for_data,
    )
    model, tok = gagd.load_model_and_tokenizer(
        _model_args(args),
        for_training=False,
    )
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = input_weight.data_ptr()
    input_version = input_weight._version
    device = gagd.first_device(model)

    print("Scoring the Setting 5e input checkpoint")
    baseline_forget_nll = tofu.score_answer_instances(
        model,
        tok,
        forget_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    baseline_utility_nll = tofu.score_answer_instances(
        model,
        tok,
        utility_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    (
        required_forget_nll,
        initially_active_mask,
        target_nll,
    ) = build_required_forget_nll(
        baseline_forget_nll,
        target_probability=args.target_forget_answer_probability,
        target_nll_buffer=args.target_nll_buffer,
    )
    required_utility_nll = (
        baseline_utility_nll + args.utility_nll_tolerance
    )
    active_positions = (
        initially_active_mask.nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    selected_ids = active_forget_row_ids(
        tok,
        forget_instances,
        active_positions,
        max_length=args.max_length,
    )
    selected_report = {
        "initially_active_forget_instance_count": len(active_positions),
        "active_positions": active_positions,
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "selected_lm_head_tokens": {
            str(token_id): tok.decode([token_id]) for token_id in selected_ids
        },
        "selection_source": "initially_active_forget_answers_only",
    }
    gagd.write_json(output_dir / "selected_lm_head_rows.json", selected_report)
    if active_positions and not selected_ids:
        raise RuntimeError("Active forget cases produced no eligible LM-head rows")

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
    print(
        f"Caching {len(forget_instances)} forget and "
        f"{len(utility_instances)} utility objectives"
    )
    forget_caches = tofu.build_answer_delta_caches(
        model,
        tok,
        forget_instances,
        selected_ids,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    utility_caches = tofu.build_answer_delta_caches(
        model,
        tok,
        utility_instances,
        selected_ids,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    packed_forget_caches = tofu.pack_answer_delta_caches(forget_caches)
    packed_utility_caches = tofu.pack_answer_delta_caches(utility_caches)
    zero_delta = torch.zeros_like(baseline_rows, dtype=torch.float32)
    baseline_forget_cached = tofu.answer_nlls_from_packed_delta_cache(
        packed_forget_caches,
        zero_delta,
    )
    baseline_utility_cached = tofu.answer_nlls_from_packed_delta_cache(
        packed_utility_caches,
        zero_delta,
    )
    baseline_metrics = repair_metrics(
        baseline_forget_cached,
        required_forget_nll,
        baseline_utility_cached,
        required_utility_nll,
        utility_instances,
        zero_delta,
        target_nll=target_nll,
        tolerance=args.comparison_tolerance,
    )
    gagd.write_json(output_dir / "baseline_local_metrics.json", baseline_metrics)
    gagd.write_json(
        output_dir / "active_cases_before.json",
        [
            row
            for row in _reports(
                forget_instances,
                baseline_forget_nll,
                required_forget_nll,
                target_probability=args.target_forget_answer_probability,
            )
            if row["active"]
        ],
    )
    gagd.write_json(
        output_dir / "forget_instances_before.json",
        _reports(
            forget_instances,
            baseline_forget_nll,
            required_forget_nll,
            target_probability=args.target_forget_answer_probability,
        ),
    )
    gagd.write_json(
        output_dir / "utility_instances_before.json",
        _reports(
            utility_instances,
            baseline_utility_nll,
            required_utility_nll,
        ),
    )

    best = CandidateSnapshot(
        step=0,
        delta_rows=zero_delta.detach().cpu().clone(),
        metrics=baseline_metrics,
    )
    logs: List[Dict[str, Any]] = []
    stopped_early = not active_positions
    steps_completed = 0

    if active_positions:
        active_caches = [forget_caches[position] for position in active_positions]
        direction_basis = None
        if args.repair_rank > 0:
            direction_basis = tofu.limited_hidden_row_basis(
                active_caches,
                max_rank=args.repair_rank,
                max_rows=args.basis_max_rows,
            )
        utility_basis = None
        if (
            args.project_away_utility_hidden
            and args.utility_projection_rank > 0
        ):
            utility_basis = tofu.limited_hidden_row_basis(
                utility_caches,
                max_rank=args.utility_projection_rank,
                max_rows=args.basis_max_rows,
            )
        delta_module = active.SelectedRowDelta(
            len(selected_ids),
            output_weight.shape[1],
            direction_basis=direction_basis,
            retained_basis=utility_basis,
            device=device,
        )
        # Packed caches drive the objective; release the per-example copies
        # after their hidden-state bases have been constructed.
        del active_caches, forget_caches, utility_caches
        optimizer = active.make_repair_optimizer(
            delta_module,
            args.repair_optimizer,
            args.repair_lr,
        )
        for step in range(1, args.repair_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            delta = delta_module.effective_delta()
            current_forget = tofu.answer_nlls_from_packed_delta_cache(
                packed_forget_caches,
                delta,
            )
            current_utility = tofu.answer_nlls_from_packed_delta_cache(
                packed_utility_caches,
                delta,
            )
            terms = target_objective_terms(
                current_forget,
                required_forget_nll,
                current_utility,
                required_utility_nll,
            )
            delta_l2 = delta.square().sum()
            loss = (
                args.forget_hinge_weight * terms["forget_hinge"]
                + args.utility_hinge_weight * terms["utility_hinge"]
                + args.delta_l2_lambda * delta_l2
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite active-repair loss at step {step}"
                )
            loss.backward()
            optimizer.step()
            norm_before, norm_after, norm_projected = (
                active.constrain_effective_delta_norm(
                    delta_module,
                    args.max_delta_norm,
                )
            )
            steps_completed = step
            with torch.no_grad():
                candidate_delta = delta_module.effective_delta()
                current_forget = tofu.answer_nlls_from_packed_delta_cache(
                    packed_forget_caches,
                    candidate_delta,
                )
                current_utility = tofu.answer_nlls_from_packed_delta_cache(
                    packed_utility_caches,
                    candidate_delta,
                )
                metrics = repair_metrics(
                    current_forget,
                    required_forget_nll,
                    current_utility,
                    required_utility_nll,
                    utility_instances,
                    candidate_delta,
                    target_nll=target_nll,
                    tolerance=args.comparison_tolerance,
                )
                if candidate_priority(metrics) < candidate_priority(best.metrics):
                    best = CandidateSnapshot(
                        step=step,
                        delta_rows=candidate_delta.detach().cpu().clone(),
                        metrics=metrics,
                    )
            if step == 1 or step % args.log_every == 0:
                log_row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "forget_hinge": float(
                        terms["forget_hinge"].detach().cpu()
                    ),
                    "utility_hinge": float(
                        terms["utility_hinge"].detach().cpu()
                    ),
                    "delta_l2": float(delta_l2.detach().cpu()),
                    "delta_norm_before_projection": norm_before,
                    "delta_norm": norm_after,
                    "delta_norm_projected": norm_projected,
                    **metrics,
                }
                logs.append(log_row)
                print(
                    f"step={step} active={metrics['active_forget_instance_count']} "
                    f"utility_violations={metrics['utility_constraint_violation_count']} "
                    f"mean_forget_prob={metrics['forget_answer_probability_mean']:.8g} "
                    f"max_forget_prob={metrics['forget_answer_probability_max']:.8g}"
                )
            if (
                args.stop_when_all_satisfied
                and metrics["active_forget_instance_count"] == 0
                and metrics["buffered_forget_constraint_unmet_count"] == 0
                and (
                    not args.require_utility_constraints
                    or metrics["utility_constraint_violation_count"] == 0
                )
            ):
                stopped_early = True
                break

    active.write_jsonl(output_dir / "repair_log.jsonl", logs)
    qualified = (
        best.metrics["active_forget_instance_count"] == 0
        and (
            not args.require_utility_constraints
            or best.metrics["utility_constraint_violation_count"] == 0
        )
    )
    if not qualified and not args.save_best_effort:
        raise RuntimeError(
            "No candidate reached the hard forget-answer "
            f"probability target {args.target_forget_answer_probability}. "
            "No checkpoint was saved."
        )

    best_delta = best.delta_rows.to(device=device, dtype=torch.float32)
    tofu.set_selected_lm_head_rows(
        output_weight,
        selected_ids,
        baseline_rows,
        best_delta,
    )
    after_forget_nll = tofu.score_answer_instances(
        model,
        tok,
        forget_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    after_utility_nll = tofu.score_answer_instances(
        model,
        tok,
        utility_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    after_probabilities = torch.exp(-after_forget_nll)
    allowed_probability = (
        args.target_forget_answer_probability
        * (1.0 + args.materialization_relative_tolerance)
    )
    materialized_active = after_probabilities > allowed_probability
    materialized_utility_violations = after_utility_nll > (
        required_utility_nll + args.comparison_tolerance
    )
    if materialized_active.any() and not args.save_best_effort:
        tofu.set_selected_lm_head_rows(
            output_weight,
            selected_ids,
            baseline_rows,
            torch.zeros_like(best_delta),
        )
        raise RuntimeError(
            "BF16 materialization left "
            f"{int(materialized_active.sum().cpu())} forget cases above "
            f"{allowed_probability}; the edit was reverted."
        )
    if (
        args.require_utility_constraints
        and materialized_utility_violations.any()
    ):
        tofu.set_selected_lm_head_rows(
            output_weight,
            selected_ids,
            baseline_rows,
            torch.zeros_like(best_delta),
        )
        raise RuntimeError(
            "BF16 materialization violated "
            f"{int(materialized_utility_violations.sum().cpu())} utility "
            "constraints; the edit was reverted."
        )

    after_metrics = repair_metrics(
        after_forget_nll,
        required_forget_nll,
        after_utility_nll,
        required_utility_nll,
        utility_instances,
        best_delta,
        target_nll=target_nll,
        tolerance=math.log1p(args.materialization_relative_tolerance),
    )
    forget_after_reports = _reports(
        forget_instances,
        after_forget_nll,
        required_forget_nll,
        target_probability=args.target_forget_answer_probability,
    )
    gagd.write_json(
        output_dir / "active_cases_after.json",
        [row for row in forget_after_reports if row["active"]],
    )
    gagd.write_json(
        output_dir / "forget_instances_after.json",
        forget_after_reports,
    )
    gagd.write_json(
        output_dir / "utility_instances_after.json",
        _reports(
            utility_instances,
            after_utility_nll,
            required_utility_nll,
        ),
    )
    gagd.write_json(output_dir / "candidate_local_metrics.json", after_metrics)

    if (
        model.get_input_embeddings().weight.data_ptr() != input_pointer
        or model.get_input_embeddings().weight._version != input_version
    ):
        raise RuntimeError("Input embeddings changed during active repair")
    summary = {
        "method": METHOD,
        "input_checkpoint": args.model_path,
        "target_forget_answer_probability": (
            args.target_forget_answer_probability
        ),
        "target_forget_answer_nll": target_nll,
        "initially_active_forget_instances": len(active_positions),
        "active_forget_instances_after": int(
            materialized_active.sum().cpu()
        ),
        "selected_lm_head_row_count": len(selected_ids),
        "best_step": best.step,
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "qualified": bool(
            not materialized_active.any()
            and (
                not args.require_utility_constraints
                or not materialized_utility_violations.any()
            )
        ),
        "utility_constraints_required": bool(
            args.require_utility_constraints
        ),
        "utility_constraint_violations_after": int(
            materialized_utility_violations.sum().cpu()
        ),
        "baseline_local_metrics": baseline_metrics,
        "candidate_local_metrics": after_metrics,
        "input_embeddings_unchanged": True,
        "transformer_parameters_frozen": True,
        "only_selected_lm_head_rows_materialized": True,
        "checkpoint_saved": bool(args.save_model),
    }
    gagd.write_json(output_dir / "repair_summary.json", summary)
    if args.save_model:
        active.save_repair_checkpoint(
            model,
            tok,
            output_dir / "checkpoint",
            repair_config=config_used,
        )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        "Active TOFU repair complete: "
        f"mean={float(after_probabilities.mean().cpu()):.8g}, "
        f"max={float(after_probabilities.max().cpu()):.8g}, "
        f"target={args.target_forget_answer_probability:.8g}"
    )


if __name__ == "__main__":
    main()
