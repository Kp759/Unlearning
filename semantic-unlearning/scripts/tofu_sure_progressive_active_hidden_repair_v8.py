#!/usr/bin/env python3
"""SURE-TOFU V8 Stage 2: progressive active-row LM-head repair.

Input is the bounded-top3 Stage-1 checkpoint. Stage 1 is frozen conceptually;
this script starts from its sparse 150-row solution and repairs the final hard
full-answer constraint using only the same 50 training-visible forget QAs.

The key difference from V7 Stage 2 is row selection. We never immediately open
the union of all answer rows from all active QAs. Instead:

1. begin with the Stage-1 sensitive-row union (normally <=150 rows);
2. optimize a Stage-2 delta on those rows against all 50 direct-forget constraints;
3. if some QAs still violate the hard target, promote at most one new content-
   bearing answer row per residual QA;
4. choose the promoted row by exact local sequence-NLL gradient leverage,
   measured in the repair subspace (full hidden space for rank 0, or the
   all-50 forget-hidden basis for rank R>0);
5. restore the exact Stage-1 baseline, re-optimize the whole selected union,
   audit all 50 QAs, and repeat for a fixed number of promotion rounds.

Thus row growth is demand-driven and uses no retain/paraphrase/heldout/PPL
signal. Rank 0 remains the unrestricted ablation; rank 256 is the intended
hidden-direction repair.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_rank0_forget as old
import tofu_sure_rank0_forget_restored as boundary


METHOD = "SURE-TOFU-v8-progressive-active-hidden-repair"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"
DEFAULT_SAFETY_FRACTIONS = (0.002, 0.01, 0.02, 0.05, 0.10)


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty float list")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="bounded-top3 Stage1 checkpoint")
    p.add_argument("--reference-model-path", required=True, help="protected Full-TOFU Base")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--repair-rank", type=int, default=256, help="0=unrestricted; >0=all-50 hidden basis")
    p.add_argument("--repair-steps", type=int, default=10000, help="total optimizer-step budget across all progressive rounds")
    p.add_argument("--max-promotion-rounds", type=int, default=5)
    p.add_argument("--promotions-per-residual-qa", type=int, default=1)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--repair-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--boundary-bisection-steps", type=int, default=30)
    p.add_argument("--materialization-safety-fractions", type=parse_float_list, default=DEFAULT_SAFETY_FRACTIONS)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--comparison-tolerance", type=float, default=1e-6)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def feasible(nll: torch.Tensor, required: torch.Tensor, tolerance: float) -> bool:
    return bool(torch.all(nll >= (required.to(nll) - tolerance)).item())


def repair_priority(metrics: Dict[str, Any]) -> Tuple[int, int, float, float]:
    return (
        int(metrics["active_forget_instance_count"]),
        int(metrics["buffered_forget_constraint_unmet_count"]),
        float(metrics["forget_answer_probability_max"]),
        float(metrics["selected_lm_head_delta_norm"]),
    )


def residual_positions(nll: torch.Tensor, target_nll: float, tolerance: float) -> List[int]:
    return (
        (nll < (target_nll - tolerance))
        .nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def hidden_basis_from_all_50(caches: Sequence[tofu.TOFUAnswerDeltaCache], repair_rank: int) -> torch.Tensor | None:
    if repair_rank == 0:
        return None
    hidden = torch.cat([cache.hidden.float() for cache in caches if cache.hidden.numel()], dim=0)
    basis = active.orthonormal_row_basis(hidden, max_rank=repair_rank)
    if basis.shape[0] == 0:
        raise RuntimeError("all-50 forget hidden states produced zero numerical rank")
    return basis


def load_stage1_sensitive_ids(model_path: Path) -> List[int]:
    report_path = model_path.parent / "sensitive_lm_rows.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"progressive Stage2 requires Stage1 sensitive-row report: {report_path}"
        )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    ids = payload.get("sensitive_token_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError(f"invalid sensitive_token_ids in {report_path}")
    return sorted({int(x) for x in ids})


def document_frequency(tok: Any, instances: Sequence[tofu.TOFUAnswerInstance], max_length: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for instance in instances:
        counts.update(set(locked.answer_token_ids(tok, instance, max_length=max_length)))
    return counts


def exact_row_leverage(
    cache: tofu.TOFUAnswerDeltaCache,
    column: int,
    direction_basis: torch.Tensor | None,
) -> float:
    """Norm of exact sequence-NLL gradient for one LM row in the repair subspace."""
    probs = cache.selected_probs[:, column].float()
    indicator = cache.target_selected_columns.eq(int(column)).float()
    coeff = (probs - indicator) / float(cache.hidden.shape[0])
    gradient = (coeff.unsqueeze(-1) * cache.hidden.float()).sum(dim=0)
    if direction_basis is None:
        value = gradient.norm()
    else:
        value = (gradient @ direction_basis.transpose(0, 1)).norm()
    return float(value.detach().cpu())


def promote_rows(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    all_answer_ids: Sequence[int],
    current_caches: Sequence[tofu.TOFUAnswerDeltaCache],
    residual: Sequence[int],
    selected_ids: Sequence[int],
    direction_basis: torch.Tensor | None,
    *,
    max_length: int,
    promotions_per_qa: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    selected = set(int(x) for x in selected_ids)
    promoted: set[int] = set()
    reports: List[Dict[str, Any]] = []
    lookup = {int(token_id): column for column, token_id in enumerate(all_answer_ids)}
    df = document_frequency(tok, instances, max_length)

    for position in residual:
        answer_ids = locked.answer_token_ids(tok, instances[position], max_length=max_length)
        unique_ids = list(dict.fromkeys(answer_ids))
        candidates = [
            token_id for token_id in unique_ids
            if token_id not in selected
            and token_id not in promoted
            and token_id in lookup
            and locked.is_content_bearing_token(tok, token_id)
        ]
        fallback = False
        if not candidates:
            candidates = [
                token_id for token_id in unique_ids
                if token_id not in selected and token_id not in promoted and token_id in lookup
            ]
            fallback = True

        ranked: List[Tuple[float, int, int, int]] = []
        first_pos = {token_id: index for index, token_id in enumerate(unique_ids)}
        for token_id in candidates:
            leverage = exact_row_leverage(
                current_caches[position], lookup[token_id], direction_basis
            )
            ranked.append((-leverage, int(df[token_id]), int(first_pos[token_id]), int(token_id)))
        ranked.sort()
        chosen = ranked[:promotions_per_qa]
        chosen_rows: List[Dict[str, Any]] = []
        for neg_leverage, freq, _, token_id in chosen:
            promoted.add(token_id)
            chosen_rows.append(
                {
                    "token_id": token_id,
                    "token": locked.decoded_token(tok, token_id),
                    "projected_sequence_nll_gradient_norm": -neg_leverage,
                    "document_frequency_in_50_forget_answers": freq,
                }
            )
        reports.append(
            {
                "position": int(position),
                "source_index": int(instances[position].source_index),
                "candidate_count": len(candidates),
                "fallback_to_noncontent": fallback,
                "chosen": chosen_rows,
            }
        )
    return sorted(promoted), reports


def restore_stage1_rows(
    output_weight: torch.Tensor,
    all_answer_ids: Sequence[int],
    stage1_rows: torch.Tensor,
) -> None:
    zero = torch.zeros_like(stage1_rows, dtype=torch.float32, device=output_weight.device)
    tofu.set_selected_lm_head_rows(output_weight, all_answer_ids, stage1_rows, zero)


def main() -> None:
    a = parse_args()
    if a.forget_num != 50:
        raise ValueError("V8 locked experiment fixes --forget-num=50")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.repair_rank < 0 or a.repair_steps <= 0 or a.max_promotion_rounds < 0:
        raise ValueError("invalid repair rank/step/round controls")
    if a.promotions_per_residual_qa <= 0:
        raise ValueError("promotions-per-residual-qa must be positive")
    if a.repair_lr <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("repair lr/batch-size/max-length must be positive")
    if a.forget_hinge_weight <= 0 or a.hardest_forget_hinge_weight < 0 or a.delta_l2_lambda < 0:
        raise ValueError("invalid repair loss controls")
    if a.boundary_bisection_steps <= 0 or a.comparison_tolerance < 0:
        raise ValueError("invalid bisection/tolerance controls")
    if any(x < 0 or x > 0.5 for x in a.materialization_safety_fractions):
        raise ValueError("materialization safety fractions must lie in [0,0.5]")

    forget_path = Path(a.forget_json).resolve()
    reference_path = Path(a.reference_model_path).resolve()
    model_path = Path(a.model_path).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
    if not reference_path.is_dir() or not model_path.is_dir():
        raise FileNotFoundError("model/reference path missing")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)
    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    data_tok = AutoTokenizer.from_pretrained(str(model_path))
    if data_tok.pad_token is None:
        data_tok.pad_token = data_tok.eos_token
    instances, source_indices = locked.load_forget_instances(forget_path, data_tok, a.forget_num)

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = int(input_weight.data_ptr())
    input_version = int(input_weight._version)
    device = gagd.first_device(model)

    all_positions = list(range(len(instances)))
    all_answer_ids = locked.answer_rows_for_instances(tok, instances, all_positions, max_length=a.max_length)
    stage1_sensitive_ids = load_stage1_sensitive_ids(model_path)
    if not set(stage1_sensitive_ids).issubset(set(all_answer_ids)):
        raise RuntimeError("Stage1 sensitive rows are not a subset of visible answer rows")

    all_ids_tensor = torch.tensor(all_answer_ids, dtype=torch.long, device=output_weight.device)
    stage1_all_rows = output_weight.index_select(0, all_ids_tensor).detach().clone()

    stage1_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    target_nll = -math.log(a.target_forget_answer_probability)
    required_nll = stage1_nll.new_full(stage1_nll.shape, target_nll)
    initial_residual = residual_positions(stage1_nll, target_nll, a.comparison_tolerance)
    stage1_metrics = locked.metrics(
        stage1_nll,
        required_nll,
        torch.zeros((len(stage1_sensitive_ids), int(output_weight.shape[1])), dtype=torch.float32, device=device),
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    write_json(
        root / "forget_instances_before.json",
        locked.report_instances(instances, stage1_nll, required_nll, a.target_forget_answer_probability),
    )

    # Fixed all-50 hidden basis. LM-head-only edits do not change these hidden states.
    all_stage1_caches = tofu.build_answer_delta_caches(
        model, tok, instances, all_answer_ids, device,
        batch_size=a.batch_size, max_length=a.max_length,
    )
    direction_basis = hidden_basis_from_all_50(all_stage1_caches, a.repair_rank)
    actual_rank = 0 if direction_basis is None else int(direction_basis.shape[0])

    total_rounds = a.max_promotion_rounds + 1  # round0 uses Stage1 rows only
    round_step_budget = max(1, int(math.ceil(a.repair_steps / float(total_rounds))))
    selected_ids = sorted(stage1_sensitive_ids)
    round_reports: List[Dict[str, Any]] = []
    promotion_reports: List[Dict[str, Any]] = []
    final_nll = stage1_nll
    final_metrics = stage1_metrics
    final_delta = torch.zeros((len(selected_ids), int(output_weight.shape[1])), dtype=torch.float32, device=device)
    final_selected_safety: float | str | None = None
    final_crossing_step: int | None = None
    final_cached_crossing_metrics: Dict[str, Any] | None = None
    final_materialization_trials: List[Dict[str, Any]] = []
    success = len(initial_residual) == 0

    print(
        f"===== V8 PROGRESSIVE STAGE2 rank={a.repair_rank} actual_rank={actual_rank} "
        f"stage1_rows={len(stage1_sensitive_ids)} initial_active={len(initial_residual)} "
        f"steps_per_round={round_step_budget} max_promotion_rounds={a.max_promotion_rounds} ====="
    )

    for round_index in range(total_rounds):
        if success:
            break

        restore_stage1_rows(output_weight, all_answer_ids, stage1_all_rows)
        selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
        stage1_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()
        caches = tofu.build_answer_delta_caches(
            model, tok, instances, selected_ids, device,
            batch_size=a.batch_size, max_length=a.max_length,
        )
        packed = tofu.pack_answer_delta_caches(caches)
        module = active.SelectedRowDelta(
            len(selected_ids), int(output_weight.shape[1]),
            direction_basis=direction_basis, retained_basis=None, device=device,
        )
        optimizer = active.make_repair_optimizer(module, a.repair_optimizer, a.repair_lr)
        zero = torch.zeros_like(stage1_selected_rows, dtype=torch.float32, device=device)
        previous_delta = zero.detach().clone()
        best_delta = zero.detach().clone()
        best_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero).detach()
        best_metrics = locked.metrics(
            best_nll, required_nll, zero,
            target_nll=target_nll, tolerance=a.comparison_tolerance,
        )
        crossing_low: torch.Tensor | None = None
        crossing_high: torch.Tensor | None = None
        crossing_step: int | None = None
        crossing_metrics: Dict[str, Any] | None = None
        logs: List[Dict[str, Any]] = []

        for step in range(1, round_step_budget + 1):
            optimizer.zero_grad(set_to_none=True)
            delta = module.effective_delta()
            current_nll = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
            errors = torch.relu(required_nll.to(current_nll) - current_nll)
            loss = (
                a.forget_hinge_weight * errors.square().mean()
                + a.hardest_forget_hinge_weight * errors.square().max()
                + a.delta_l2_lambda * delta.square().sum()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite V8 Stage2 loss round={round_index} step={step}")
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                candidate = module.effective_delta().detach().clone()
                candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
                candidate_metrics = locked.metrics(
                    candidate_nll, required_nll, candidate,
                    target_nll=target_nll, tolerance=a.comparison_tolerance,
                )
                if repair_priority(candidate_metrics) < repair_priority(best_metrics):
                    best_delta = candidate.detach().clone()
                    best_nll = candidate_nll.detach().clone()
                    best_metrics = dict(candidate_metrics)

            if step == 1 or step % a.log_every == 0 or step == round_step_budget:
                logs.append(
                    {
                        "round": round_index,
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "forget_hinge": float(errors.square().mean().detach().cpu()),
                        "hardest_forget_hinge": float(errors.square().max().detach().cpu()),
                        "delta_l2": float(delta.square().sum().detach().cpu()),
                        **candidate_metrics,
                    }
                )
            if feasible(candidate_nll, required_nll, a.comparison_tolerance):
                crossing_low = previous_delta.detach().clone()
                crossing_high = candidate.detach().clone()
                crossing_step = step
                crossing_metrics = dict(candidate_metrics)
                break
            previous_delta = candidate.detach().clone()

        del optimizer
        write_jsonl(root / f"round_{round_index:02d}" / "repair_log.jsonl", logs)

        materialization_trials: List[Dict[str, Any]] = []
        selected_safety: float | str | None = None
        if crossing_low is not None and crossing_high is not None:
            candidate_to_materialize: torch.Tensor | None = None
            actual_nll: torch.Tensor | None = None
            actual_metrics: Dict[str, Any] | None = None
            for safety in sorted(set(float(x) for x in a.materialization_safety_fractions)):
                candidate, cached_nll, alpha = boundary.boundary_bisect(
                    packed, crossing_low, crossing_high, required_nll,
                    tolerance=a.comparison_tolerance,
                    iterations=a.boundary_bisection_steps,
                    safety_fraction=safety,
                )
                tofu.set_selected_lm_head_rows(output_weight, selected_ids, stage1_selected_rows, candidate)
                trial_nll = tofu.score_answer_instances(
                    model, tok, instances, device,
                    batch_size=a.batch_size, max_length=a.max_length,
                ).detach().float()
                trial_metrics = locked.metrics(
                    trial_nll, required_nll, candidate.to(device),
                    target_nll=target_nll, tolerance=a.comparison_tolerance,
                )
                passed = feasible(trial_nll, required_nll, a.comparison_tolerance)
                materialization_trials.append(
                    {
                        "safety_fraction": safety,
                        "boundary_alpha": alpha,
                        "cached_minimum_nll_slack": float((cached_nll - required_nll.to(cached_nll)).min().detach().cpu()),
                        "materialized_pass": bool(passed),
                        "materialized_active_count": trial_metrics["active_forget_instance_count"],
                        "materialized_max_probability": trial_metrics["forget_answer_probability_max"],
                        "delta_norm": float(candidate.norm().detach().cpu()),
                    }
                )
                if passed:
                    candidate_to_materialize = candidate.detach().clone()
                    actual_nll = trial_nll
                    actual_metrics = trial_metrics
                    selected_safety = safety
                    break
                restore_stage1_rows(output_weight, all_answer_ids, stage1_all_rows)
            if candidate_to_materialize is None:
                tofu.set_selected_lm_head_rows(output_weight, selected_ids, stage1_selected_rows, crossing_high)
                actual_nll = tofu.score_answer_instances(
                    model, tok, instances, device,
                    batch_size=a.batch_size, max_length=a.max_length,
                ).detach().float()
                actual_metrics = locked.metrics(
                    actual_nll, required_nll, crossing_high.to(device),
                    target_nll=target_nll, tolerance=a.comparison_tolerance,
                )
                candidate_to_materialize = crossing_high.detach().clone()
                selected_safety = "optimizer_crossing_endpoint"
            final_delta = candidate_to_materialize
            final_nll = actual_nll
            final_metrics = actual_metrics
        else:
            tofu.set_selected_lm_head_rows(output_weight, selected_ids, stage1_selected_rows, best_delta)
            final_delta = best_delta.detach().clone()
            final_nll = tofu.score_answer_instances(
                model, tok, instances, device,
                batch_size=a.batch_size, max_length=a.max_length,
            ).detach().float()
            final_metrics = locked.metrics(
                final_nll, required_nll, best_delta.to(device),
                target_nll=target_nll, tolerance=a.comparison_tolerance,
            )

        current_residual = residual_positions(final_nll, target_nll, a.comparison_tolerance)
        round_report = {
            "round": round_index,
            "selected_row_count": len(selected_ids),
            "selected_token_ids": list(selected_ids),
            "optimizer_step_budget": round_step_budget,
            "cached_crossing_step": crossing_step,
            "cached_crossing_metrics": crossing_metrics,
            "best_cached_metrics": best_metrics,
            "materialized_metrics": final_metrics,
            "materialized_residual_positions": current_residual,
            "materialization_trials": materialization_trials,
            "selected_materialization_safety_fraction": selected_safety,
            "incremental_delta_norm_from_stage1": float(final_delta.norm().detach().cpu()),
        }
        round_reports.append(round_report)
        write_json(root / f"round_{round_index:02d}" / "summary.json", round_report)
        print(
            f"V8 round={round_index} rows={len(selected_ids)} "
            f"active={len(current_residual)} max_prob={final_metrics['forget_answer_probability_max']:.8g} "
            f"delta_norm={float(final_delta.norm().detach().cpu()):.6g}"
        )

        if not current_residual:
            success = True
            final_selected_safety = selected_safety
            final_crossing_step = crossing_step
            final_cached_crossing_metrics = crossing_metrics
            final_materialization_trials = materialization_trials
            break

        if round_index >= a.max_promotion_rounds:
            break

        # Promotion leverage is measured at the actually materialized current model.
        current_all_caches = tofu.build_answer_delta_caches(
            model, tok, instances, all_answer_ids, device,
            batch_size=a.batch_size, max_length=a.max_length,
        )
        promoted, per_qa = promote_rows(
            tok, instances, all_answer_ids, current_all_caches,
            current_residual, selected_ids, direction_basis,
            max_length=a.max_length,
            promotions_per_qa=a.promotions_per_residual_qa,
        )
        promotion_payload = {
            "after_round": round_index,
            "residual_count": len(current_residual),
            "residual_positions": current_residual,
            "promoted_unique_row_count": len(promoted),
            "promoted_token_ids": promoted,
            "promoted_tokens": {str(x): locked.decoded_token(tok, x) for x in promoted},
            "per_residual_qa": per_qa,
        }
        promotion_reports.append(promotion_payload)
        write_json(root / f"round_{round_index:02d}" / "promotions.json", promotion_payload)
        if not promoted:
            break
        selected_ids = sorted(set(selected_ids) | set(promoted))

    write_json(root / "promotion_rounds.json", {"rounds": round_reports, "promotions": promotion_reports})

    if not success or not feasible(final_nll, required_nll, a.comparison_tolerance):
        write_json(
            root / "failure.json",
            {
                "status": "FAILED_PROGRESSIVE_ROW_BUDGET",
                "repair_rank_requested": a.repair_rank,
                "repair_rank_actual": actual_rank,
                "max_promotion_rounds": a.max_promotion_rounds,
                "final_selected_row_count": len(selected_ids),
                "final_metrics": final_metrics,
                "rounds": round_reports,
            },
        )
        raise RuntimeError("V8 progressive Stage2 did not reach all-50 feasibility within promotion budget")

    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V8 Stage2 modified input embeddings")

    base_input_rows, base_output_rows = old.load_reference_answer_rows(
        str(reference_path), all_answer_ids, a.dtype
    )
    current_output_rows = output_weight.index_select(0, all_ids_tensor).detach().float().cpu()
    current_input_rows = input_weight.index_select(0, all_ids_tensor.to(input_weight.device)).detach().float().cpu()
    total_answer_delta = float((current_output_rows - base_output_rows.detach().float().cpu()).norm().item())
    input_base_error = float((current_input_rows - base_input_rows.detach().float().cpu()).abs().max().item()) if current_input_rows.numel() else 0.0
    if input_base_error != 0.0:
        raise RuntimeError(f"V8 input embeddings are not exact Base: {input_base_error}")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    write_json(
        root / "forget_instances_after.json",
        locked.report_instances(instances, final_nll, required_nll, a.target_forget_answer_probability),
    )

    summary = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "repair_rank_requested": a.repair_rank,
        "repair_rank_actual": actual_rank,
        "repair_rank_semantics": (
            "unrestricted progressively selected LM-head rows"
            if a.repair_rank == 0
            else "progressively selected LM-head rows restricted to hidden basis from all 50 direct forget QAs"
        ),
        "initially_active_forget_instance_count": len(initial_residual),
        "initially_active_positions": initial_residual,
        "initial_stage1_sensitive_lm_head_row_count": len(stage1_sensitive_ids),
        "selected_active_lm_head_row_count": len(selected_ids),
        "selected_active_lm_head_token_ids": selected_ids,
        "promotion_round_count_used": max(0, len(round_reports) - 1),
        "max_promotion_rounds": a.max_promotion_rounds,
        "promotions_per_residual_qa": a.promotions_per_residual_qa,
        "optimizer_total_step_budget": a.repair_steps,
        "optimizer_step_budget_per_round": round_step_budget,
        "incremental_stage2_delta_norm": float(final_delta.norm().detach().cpu()),
        "total_visible_answer_lm_head_delta_norm_from_base": total_answer_delta,
        "all_visible_answer_input_base_max_abs_error": input_base_error,
        "optimizer_crossing_step": final_crossing_step,
        "cached_crossing_metrics": final_cached_crossing_metrics,
        "selected_materialization_safety_fraction": final_selected_safety,
        "materialization_trials": final_materialization_trials,
        "baseline_stage1_metrics": stage1_metrics,
        "materialized_metrics": final_metrics,
        "basis_source": "all 50 training-visible direct forget answer-position hidden states",
        "row_policy": "Stage1 sensitive rows plus one highest projected-gradient-leverage answer row per residual QA per promotion round",
        "training_data_access": {
            "direct_forget_qas": a.forget_num,
            "retain95": 0,
            "paraphrases": 0,
            "same_author_holdout": 0,
            "real_authors": 0,
            "world_facts": 0,
            "PPL": False,
        },
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "repair_summary.json", summary)
    write_json(
        root / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            **vars(a),
            "forget_json_resolved": str(forget_path),
            "reference_model_path_resolved": str(reference_path),
            "stage1_model_path_resolved": str(model_path),
            "stage2_visible_source_indices": source_indices,
            "data_firewall": "selection, promotion, basis, optimization, stopping use only the same 50 direct forget QAs",
        },
    )

    print("===== SURE-TOFU V8 PROGRESSIVE STAGE2 PASS =====")
    print(
        f"rank={a.repair_rank} actual_rank={actual_rank} stage1_rows={len(stage1_sensitive_ids)} "
        f"final_rows={len(selected_ids)} promotion_rounds={max(0, len(round_reports)-1)} "
        f"stage2_norm={float(final_delta.norm().detach().cpu()):.6g} "
        f"max_prob={final_metrics['forget_answer_probability_max']:.8g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
