#!/usr/bin/env python3
"""SURE-TOFU V7 Stage 2: active-case LM-head repair, rank 0 or forget-hidden rank R.

Input is the V7 sparse LM-head GA/GD checkpoint.  This stage:

1. scores only the same 50 training-visible direct forget QAs;
2. identifies residual active QAs whose answer probability exceeds the hard target;
3. makes editable the union of answer-token LM-head rows from those active QAs;
4. enforces the hard direct-forget constraint on ALL 50 QAs;
5. for --repair-rank 0 learns an unrestricted selected-row delta;
6. for --repair-rank R>0 restricts selected-row deltas to an R-dimensional
   basis built from answer-position hidden states of ALL 50 visible forget QAs;
7. materializes BF16 and performs an all-50 fail-closed audit.

No retain, paraphrase, same-author holdout, real-author, world-fact, or PPL data
is consulted during selection, basis construction, optimization, or stopping.
"""

from __future__ import annotations

import argparse
import json
import math
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


METHOD = "SURE-TOFU-v7-active-case-hidden-repair"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"
DEFAULT_SAFETY_FRACTIONS = (0.002, 0.01, 0.02, 0.05, 0.10)


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty float list")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="V7 Stage1 sparse GA/GD checkpoint")
    p.add_argument("--reference-model-path", required=True, help="Protected Full-TOFU Base")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--repair-rank", type=int, default=256, help="0=unrestricted; >0=all-50 forget-hidden basis rank")
    p.add_argument("--repair-steps", type=int, default=10000)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--repair-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--boundary-bisection-steps", type=int, default=30)
    p.add_argument(
        "--materialization-safety-fractions",
        type=parse_float_list,
        default=DEFAULT_SAFETY_FRACTIONS,
    )
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


def hidden_basis_from_all_50(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    repair_rank: int,
) -> torch.Tensor | None:
    if repair_rank == 0:
        return None
    rows = [cache.hidden.float() for cache in caches if cache.hidden.numel()]
    if not rows:
        raise RuntimeError("all-50 forget caches contain no answer-position hidden states")
    hidden = torch.cat(rows, dim=0)
    basis = active.orthonormal_row_basis(hidden, max_rank=repair_rank)
    if basis.shape[0] == 0:
        raise RuntimeError("all-50 forget hidden states produced a zero-rank basis")
    return basis


def main() -> None:
    a = parse_args()
    if a.forget_num != 50:
        raise ValueError("V7 locked experiment fixes --forget-num=50")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.repair_rank < 0:
        raise ValueError("repair-rank must be >= 0; 0 means unrestricted")
    if a.repair_steps <= 0 or a.repair_lr <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("repair steps/lr/batch-size/max-length must be positive")
    if a.forget_hinge_weight <= 0 or a.hardest_forget_hinge_weight < 0 or a.delta_l2_lambda < 0:
        raise ValueError("invalid repair loss controls")
    if a.boundary_bisection_steps <= 0 or a.comparison_tolerance < 0:
        raise ValueError("invalid bisection/tolerance controls")
    if any(x < 0 or x > 0.5 for x in a.materialization_safety_fractions):
        raise ValueError("materialization safety fractions must lie in [0,0.5]")

    forget_path = Path(a.forget_json).resolve()
    reference_path = Path(a.reference_model_path).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
    if not reference_path.is_dir():
        raise FileNotFoundError(reference_path)

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)
    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    data_tok = AutoTokenizer.from_pretrained(a.model_path)
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

    baseline_nll = tofu.score_answer_instances(
        model,
        tok,
        instances,
        device,
        batch_size=a.batch_size,
        max_length=a.max_length,
    ).detach().float()
    target_nll = -math.log(a.target_forget_answer_probability)
    required_nll = baseline_nll.new_full(baseline_nll.shape, target_nll)
    active_positions = (
        (baseline_nll < (target_nll - a.comparison_tolerance))
        .nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )
    selected_ids = locked.answer_rows_for_instances(
        tok,
        instances,
        active_positions,
        max_length=a.max_length,
    )
    if active_positions and not selected_ids:
        raise RuntimeError("active direct-forget QAs produced no editable answer-token rows")

    zero_for_metrics = torch.zeros(
        (len(selected_ids), int(output_weight.shape[1])),
        dtype=torch.float32,
        device=device,
    )
    baseline_metrics = locked.metrics(
        baseline_nll,
        required_nll,
        zero_for_metrics,
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    write_json(
        root / "forget_instances_before.json",
        locked.report_instances(instances, baseline_nll, required_nll, a.target_forget_answer_probability),
    )

    selected_report = {
        "active_case_definition": "direct forget answer probability > hard target after V7 Stage1",
        "active_forget_instance_count": len(active_positions),
        "active_positions": active_positions,
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "selected_lm_head_tokens": {
            str(token_id): locked.decoded_token(tok, token_id) for token_id in selected_ids
        },
        "row_policy": "union of all answer-token rows from residual active direct-forget QAs",
        "basis_source": "answer-position hidden states from all 50 training-visible direct forget QAs",
        "repair_rank_requested": a.repair_rank,
        "retain_or_heldout_data_consulted": False,
    }
    write_json(root / "selected_active_rows.json", selected_report)

    selected_candidate = zero_for_metrics.detach().clone()
    selected_safety: float | str | None = None
    direction_basis: torch.Tensor | None = None
    optimizer_crossing_step: int | None = None
    cached_crossing_metrics: Dict[str, Any] | None = None
    logs: List[Dict[str, Any]] = []

    if active_positions:
        selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
        baseline_rows = output_weight.index_select(0, selected_tensor).detach().clone()
        caches = tofu.build_answer_delta_caches(
            model,
            tok,
            instances,
            selected_ids,
            device,
            batch_size=a.batch_size,
            max_length=a.max_length,
        )
        packed = tofu.pack_answer_delta_caches(caches)
        direction_basis = hidden_basis_from_all_50(caches, a.repair_rank)
        actual_rank = 0 if direction_basis is None else int(direction_basis.shape[0])
        print(
            f"===== V7 STAGE2 rank={a.repair_rank} actual_rank={actual_rank} "
            f"active_qas={len(active_positions)} active_rows={len(selected_ids)} ====="
        )

        module = active.SelectedRowDelta(
            len(selected_ids),
            int(output_weight.shape[1]),
            direction_basis=direction_basis,
            retained_basis=None,
            device=device,
        )
        optimizer = active.make_repair_optimizer(module, a.repair_optimizer, a.repair_lr)
        zero = torch.zeros_like(baseline_rows, dtype=torch.float32, device=device)
        previous_delta = zero.detach().clone()
        crossing_low: torch.Tensor | None = None
        crossing_high: torch.Tensor | None = None
        best_delta = zero.detach().clone()
        best_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero).detach()
        best_metrics = locked.metrics(
            best_nll,
            required_nll,
            zero,
            target_nll=target_nll,
            tolerance=a.comparison_tolerance,
        )

        for step in range(1, a.repair_steps + 1):
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
                raise FloatingPointError(f"non-finite V7 Stage2 loss at step {step}")
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                candidate = module.effective_delta().detach().clone()
                candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
                candidate_metrics = locked.metrics(
                    candidate_nll,
                    required_nll,
                    candidate,
                    target_nll=target_nll,
                    tolerance=a.comparison_tolerance,
                )
                if repair_priority(candidate_metrics) < repair_priority(best_metrics):
                    best_delta = candidate.detach().clone()
                    best_nll = candidate_nll.detach().clone()
                    best_metrics = dict(candidate_metrics)

            if step == 1 or step % a.log_every == 0:
                logs.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "forget_hinge": float(errors.square().mean().detach().cpu()),
                        "hardest_forget_hinge": float(errors.square().max().detach().cpu()),
                        "delta_l2": float(delta.square().sum().detach().cpu()),
                        **candidate_metrics,
                    }
                )
                print(
                    f"stage2-step={step} active={candidate_metrics['active_forget_instance_count']} "
                    f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g} "
                    f"norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
                )

            if feasible(candidate_nll, required_nll, a.comparison_tolerance):
                crossing_low = previous_delta.detach().clone()
                crossing_high = candidate.detach().clone()
                optimizer_crossing_step = step
                cached_crossing_metrics = dict(candidate_metrics)
                break
            previous_delta = candidate.detach().clone()

        del optimizer
        write_jsonl(root / "repair_log.jsonl", logs)
        if crossing_low is None or crossing_high is None:
            write_json(
                root / "failure.json",
                {
                    "status": "FAILED_NO_CACHED_FEASIBLE_ACTIVE_REPAIR",
                    "repair_rank_requested": a.repair_rank,
                    "repair_rank_actual": actual_rank,
                    "active_forget_instance_count": len(active_positions),
                    "selected_lm_head_row_count": len(selected_ids),
                    "best_metrics": best_metrics,
                },
            )
            raise RuntimeError("V7 active-case repair did not reach cached all-50 feasibility")

        materialization_trials: List[Dict[str, Any]] = []
        final_nll: torch.Tensor | None = None
        final_metrics: Dict[str, Any] | None = None
        for safety in sorted(set(float(x) for x in a.materialization_safety_fractions)):
            candidate, cached_nll, alpha = boundary.boundary_bisect(
                packed,
                crossing_low,
                crossing_high,
                required_nll,
                tolerance=a.comparison_tolerance,
                iterations=a.boundary_bisection_steps,
                safety_fraction=safety,
            )
            tofu.set_selected_lm_head_rows(output_weight, selected_ids, baseline_rows, candidate)
            actual_nll = tofu.score_answer_instances(
                model,
                tok,
                instances,
                device,
                batch_size=a.batch_size,
                max_length=a.max_length,
            ).detach().float()
            actual_metrics = locked.metrics(
                actual_nll,
                required_nll,
                candidate.to(device),
                target_nll=target_nll,
                tolerance=a.comparison_tolerance,
            )
            passed = feasible(actual_nll, required_nll, a.comparison_tolerance)
            materialization_trials.append(
                {
                    "safety_fraction": safety,
                    "boundary_alpha": alpha,
                    "cached_minimum_nll_slack": float(
                        (cached_nll - required_nll.to(cached_nll)).min().detach().cpu()
                    ),
                    "materialized_pass": bool(passed),
                    "materialized_active_count": actual_metrics["active_forget_instance_count"],
                    "materialized_max_probability": actual_metrics["forget_answer_probability_max"],
                    "delta_norm": float(candidate.norm().detach().cpu()),
                }
            )
            if passed:
                selected_candidate = candidate.detach().clone()
                selected_safety = safety
                final_nll = actual_nll
                final_metrics = actual_metrics
                break
            tofu.set_selected_lm_head_rows(
                output_weight,
                selected_ids,
                baseline_rows,
                torch.zeros_like(candidate),
            )

        if final_nll is None or final_metrics is None:
            tofu.set_selected_lm_head_rows(output_weight, selected_ids, baseline_rows, crossing_high)
            actual_nll = tofu.score_answer_instances(
                model,
                tok,
                instances,
                device,
                batch_size=a.batch_size,
                max_length=a.max_length,
            ).detach().float()
            actual_metrics = locked.metrics(
                actual_nll,
                required_nll,
                crossing_high.to(device),
                target_nll=target_nll,
                tolerance=a.comparison_tolerance,
            )
            passed = feasible(actual_nll, required_nll, a.comparison_tolerance)
            materialization_trials.append(
                {
                    "safety_fraction": "optimizer_crossing_endpoint",
                    "boundary_alpha": 1.0,
                    "materialized_pass": bool(passed),
                    "materialized_active_count": actual_metrics["active_forget_instance_count"],
                    "materialized_max_probability": actual_metrics["forget_answer_probability_max"],
                    "delta_norm": float(crossing_high.norm().detach().cpu()),
                }
            )
            if not passed:
                write_json(
                    root / "failure.json",
                    {
                        "status": "FAILED_BF16_MATERIALIZATION",
                        "repair_rank_requested": a.repair_rank,
                        "repair_rank_actual": actual_rank,
                        "materialization_trials": materialization_trials,
                    },
                )
                raise RuntimeError("V7 Stage2 lost all-50 feasibility after BF16 materialization")
            selected_candidate = crossing_high.detach().clone()
            selected_safety = "optimizer_crossing_endpoint"
            final_nll = actual_nll
            final_metrics = actual_metrics
    else:
        actual_rank = 0
        materialization_trials = []
        final_nll = baseline_nll
        final_metrics = baseline_metrics

    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V7 Stage2 modified input embeddings")
    if not feasible(final_nll, required_nll, a.comparison_tolerance):
        raise RuntimeError("V7 final all-50 direct-forget audit failed")

    # Report the TOTAL answer-row displacement from protected Full-TOFU Base,
    # including both Stage1 and Stage2, not merely the incremental repair norm.
    all_answer_ids = locked.answer_rows_for_instances(
        tok,
        instances,
        list(range(len(instances))),
        max_length=a.max_length,
    )
    base_input_rows, base_output_rows = old.load_reference_answer_rows(
        str(reference_path), all_answer_ids, a.dtype
    )
    all_out_ids = torch.tensor(all_answer_ids, dtype=torch.long, device=output_weight.device)
    all_in_ids = all_out_ids.to(input_weight.device)
    current_output_rows = output_weight.index_select(0, all_out_ids).detach().float().cpu()
    current_input_rows = input_weight.index_select(0, all_in_ids).detach().float().cpu()
    total_answer_row_delta_from_base = float(
        (current_output_rows - base_output_rows.detach().float().cpu()).norm().item()
    )
    input_answer_row_base_error = float(
        (current_input_rows - base_input_rows.detach().float().cpu()).abs().max().item()
    ) if current_input_rows.numel() else 0.0
    if input_answer_row_base_error != 0.0:
        raise RuntimeError(
            f"V7 input embeddings are not exact Base on visible answer rows: {input_answer_row_base_error}"
        )

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
            "unrestricted selected active-answer LM-head rows"
            if a.repair_rank == 0
            else "selected active-answer LM-head rows restricted to hidden basis from all 50 direct forget QAs"
        ),
        "initially_active_forget_instance_count": len(active_positions),
        "initially_active_positions": active_positions,
        "selected_active_lm_head_row_count": len(selected_ids),
        "selected_active_lm_head_token_ids": selected_ids,
        "incremental_stage2_delta_norm": float(selected_candidate.norm().detach().cpu()),
        "total_visible_answer_lm_head_delta_norm_from_base": total_answer_row_delta_from_base,
        "all_visible_answer_input_base_max_abs_error": input_answer_row_base_error,
        "optimizer_crossing_step": optimizer_crossing_step,
        "cached_crossing_metrics": cached_crossing_metrics,
        "selected_materialization_safety_fraction": selected_safety,
        "materialization_trials": materialization_trials,
        "baseline_stage1_metrics": baseline_metrics,
        "materialized_metrics": final_metrics,
        "basis_source": "all 50 training-visible direct forget answer-position hidden states",
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
            "reference_model_path_resolved": str(reference_path),
            "forget_json_resolved": str(forget_path),
            "stage2_visible_source_indices": source_indices,
            "parameter_scope": "Stage1 checkpoint plus selected residual-active answer LM-head rows only; transformer/input embeddings frozen",
            "data_firewall": "basis/selection/optimization/stopping use only the same 50 training-visible direct forget QAs",
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print("===== SURE-TOFU V7 STAGE2 PASS =====")
    print(
        f"rank_requested={a.repair_rank} rank_actual={actual_rank} "
        f"active_qas={len(active_positions)} active_rows={len(selected_ids)} "
        f"stage2_norm={float(selected_candidate.norm().detach().cpu()):.6g}"
    )
    print(
        f"final_active={final_metrics['active_forget_instance_count']} "
        f"final_max_prob={final_metrics['forget_answer_probability_max']:.8g} "
        f"total_answer_row_delta_from_base={total_answer_row_delta_from_base:.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
