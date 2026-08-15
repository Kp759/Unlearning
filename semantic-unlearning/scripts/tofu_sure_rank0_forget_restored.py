#!/usr/bin/env python3
"""SURE-TOFU Stage 1B v2: restricted rank-0 after exact non-sensitive restoration.

Corrected semantics:

1. Freeze the direct-forget requirements from the Stage-1A checkpoint.
2. Derive an initial *token-row* sensitive set only from teacher-forced token
   deficits on the currently violating direct QAs.
3. Restore every other visible answer-token row exactly to Full-TOFU Base in
   BOTH the input embedding and detached LM head.
4. Try unrestricted rank-0 LM-head repair on the current sensitive set.
5. Only if that restricted rank-0 problem cannot satisfy all 50 direct
   constraints do we promote additional answer rows from the still-violating
   sequences and retry from the original Stage-1A row snapshots.
6. At the first feasible optimizer crossing, bisect between the last infeasible
   and first feasible deltas and keep a near-boundary solution rather than the
   overshooting optimizer step.
7. Reassert exact Base values for non-sensitive answer rows, materialize the
   sensitive rank-0 delta, and fail closed unless all direct constraints still
   pass.

No retain95, paraphrases, same-author heldout, real-authors, world-facts, PPL,
or final metric is loaded or used for selection.
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_active_forget_repair as native
import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_rank0_forget as old

METHOD = "SURE-TOFU-restricted-rank0-exact-nonsensitive-restore-v2"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Frozen Stage1A checkpoint")
    p.add_argument("--reference-model-path", required=True, help="Original Full-TOFU checkpoint")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--target-nll-buffer", type=float, default=0.25)
    p.add_argument("--forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--repair-steps", type=int, default=10000)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--repair-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--comparison-tolerance", type=float, default=1e-6)
    p.add_argument("--max-promotion-rounds", type=int, default=16)
    p.add_argument("--boundary-bisection-steps", type=int, default=30)
    p.add_argument(
        "--boundary-safety-fraction",
        type=float,
        default=0.002,
        help="Small interpolation fraction beyond the cached feasibility boundary for dtype materialization safety.",
    )
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def feasible_nll(nll: torch.Tensor, required: torch.Tensor, tolerance: float) -> bool:
    req = required.to(device=nll.device, dtype=nll.dtype)
    return bool(torch.all(nll >= req - tolerance).item())


@dataclass
class RepairAttempt:
    feasible: bool
    delta_cpu: torch.Tensor
    nll_cpu: torch.Tensor
    metrics: Dict[str, Any]
    step: int
    optimizer_crossing_step: int | None
    boundary_alpha: float | None
    logs: List[Dict[str, Any]]


@torch.no_grad()
def boundary_bisect(
    packed: tofu.PackedTOFUAnswerDeltaCache,
    low_delta: torch.Tensor,
    high_delta: torch.Tensor,
    required_nll: torch.Tensor,
    *,
    tolerance: float,
    iterations: int,
    safety_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return the smallest feasible interpolation, plus a tiny safety step."""
    if feasible_nll(
        tofu.answer_nlls_from_packed_delta_cache(packed, low_delta),
        required_nll,
        tolerance,
    ):
        nll = tofu.answer_nlls_from_packed_delta_cache(packed, low_delta)
        return low_delta.detach().clone(), nll.detach().clone(), 0.0

    high_nll = tofu.answer_nlls_from_packed_delta_cache(packed, high_delta)
    if not feasible_nll(high_nll, required_nll, tolerance):
        raise ValueError("boundary_bisect requires a feasible high endpoint")

    lo = 0.0
    hi = 1.0
    direction = high_delta - low_delta
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        candidate = low_delta + mid * direction
        candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
        if feasible_nll(candidate_nll, required_nll, tolerance):
            hi = mid
        else:
            lo = mid

    alpha = min(1.0, hi + max(0.0, safety_fraction))
    candidate = low_delta + alpha * direction
    candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
    if not feasible_nll(candidate_nll, required_nll, tolerance):
        candidate = high_delta.detach().clone()
        candidate_nll = high_nll.detach().clone()
        alpha = 1.0
    return candidate.detach().clone(), candidate_nll.detach().clone(), float(alpha)


def run_restricted_rank0(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    selected_ids: Sequence[int],
    required_nll: torch.Tensor,
    target_nll: float,
    args: argparse.Namespace,
    device: torch.device,
) -> RepairAttempt:
    if not selected_ids:
        current_nll = tofu.score_answer_instances(
            model,
            tok,
            instances,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        ).detach().float()
        empty = torch.empty((0, model.get_output_embeddings().weight.shape[1]), dtype=torch.float32)
        metrics = locked.metrics(
            current_nll,
            required_nll,
            empty.to(device),
            target_nll=target_nll,
            tolerance=args.comparison_tolerance,
        )
        return RepairAttempt(
            feasible=feasible_nll(current_nll, required_nll, args.comparison_tolerance),
            delta_cpu=empty,
            nll_cpu=current_nll.cpu(),
            metrics=metrics,
            step=0,
            optimizer_crossing_step=None,
            boundary_alpha=None,
            logs=[],
        )

    caches = tofu.build_answer_delta_caches(
        model,
        tok,
        instances,
        selected_ids,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    packed = tofu.pack_answer_delta_caches(caches)
    output_weight = model.get_output_embeddings().weight
    delta_module = active.SelectedRowDelta(
        len(selected_ids),
        output_weight.shape[1],
        direction_basis=None,
        retained_basis=None,
        device=device,
    )
    optimizer = active.make_repair_optimizer(
        delta_module, args.repair_optimizer, args.repair_lr
    )

    zero_delta = torch.zeros(
        (len(selected_ids), output_weight.shape[1]), dtype=torch.float32, device=device
    )
    zero_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero_delta)
    zero_metrics = locked.metrics(
        zero_nll,
        required_nll,
        zero_delta,
        target_nll=target_nll,
        tolerance=args.comparison_tolerance,
    )
    if feasible_nll(zero_nll, required_nll, args.comparison_tolerance):
        del optimizer
        return RepairAttempt(True, zero_delta.cpu(), zero_nll.cpu(), zero_metrics, 0, None, 0.0, [])

    best_delta = zero_delta.detach().clone()
    best_nll = zero_nll.detach().clone()
    best_metrics = dict(zero_metrics)
    previous_delta = zero_delta.detach().clone()
    previous_nll = zero_nll.detach().clone()
    logs: List[Dict[str, Any]] = []

    for step in range(1, args.repair_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        current_nll = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
        errors = torch.relu(required_nll.to(current_nll) - current_nll)
        loss = (
            args.forget_hinge_weight * errors.square().mean()
            + args.hardest_forget_hinge_weight * errors.square().max()
            + args.delta_l2_lambda * delta.square().sum()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite restricted rank-0 loss at step {step}")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            candidate_delta = delta_module.effective_delta().detach().clone()
            candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate_delta)
            candidate_metrics = locked.metrics(
                candidate_nll,
                required_nll,
                candidate_delta,
                target_nll=target_nll,
                tolerance=args.comparison_tolerance,
            )

            # Track the best infeasible point by the ordinary constraint priority.
            if old.sure_priority(candidate_metrics) < old.sure_priority(best_metrics):
                best_delta = candidate_delta.detach().clone()
                best_nll = candidate_nll.detach().clone()
                best_metrics = dict(candidate_metrics)

            if step == 1 or step % args.log_every == 0:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "forget_hinge": float(errors.square().mean().detach().cpu()),
                    "hardest_forget_hinge": float(errors.square().max().detach().cpu()),
                    "delta_l2": float(delta.square().sum().detach().cpu()),
                    **candidate_metrics,
                }
                logs.append(row)
                print(
                    f"restricted-step={step} active={candidate_metrics['active_forget_instance_count']} "
                    f"buffered={candidate_metrics['buffered_forget_constraint_unmet_count']} "
                    f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g} "
                    f"norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
                )

            if feasible_nll(candidate_nll, required_nll, args.comparison_tolerance):
                boundary_delta, boundary_nll, alpha = boundary_bisect(
                    packed,
                    previous_delta,
                    candidate_delta,
                    required_nll,
                    tolerance=args.comparison_tolerance,
                    iterations=args.boundary_bisection_steps,
                    safety_fraction=args.boundary_safety_fraction,
                )
                boundary_metrics = locked.metrics(
                    boundary_nll,
                    required_nll,
                    boundary_delta,
                    target_nll=target_nll,
                    tolerance=args.comparison_tolerance,
                )
                del optimizer
                return RepairAttempt(
                    True,
                    boundary_delta.cpu(),
                    boundary_nll.cpu(),
                    boundary_metrics,
                    step,
                    step,
                    alpha,
                    logs,
                )

            previous_delta = candidate_delta.detach().clone()
            previous_nll = candidate_nll.detach().clone()

    del optimizer
    return RepairAttempt(
        False,
        best_delta.cpu(),
        best_nll.cpu(),
        best_metrics,
        args.repair_steps,
        None,
        None,
        logs,
    )


@torch.no_grad()
def load_reference_rows(
    model_path: str,
    answer_ids: Sequence[int],
    dtype_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = gagd.torch_dtype(dtype_name)
    reference = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    iw = reference.get_input_embeddings().weight
    ow = reference.get_output_embeddings().weight
    ids_i = torch.tensor(answer_ids, dtype=torch.long, device=iw.device)
    ids_o = ids_i.to(ow.device)
    input_rows = iw.index_select(0, ids_i).detach().cpu().clone()
    output_rows = ow.index_select(0, ids_o).detach().cpu().clone()
    del reference
    gc.collect()
    return input_rows, output_rows


def main() -> None:
    a = parse_args()
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, repair-steps and batch-size must be positive")
    if a.repair_lr <= 0 or a.delta_l2_lambda < 0:
        raise ValueError("invalid optimizer controls")
    if a.max_promotion_rounds <= 0 or a.boundary_bisection_steps <= 0:
        raise ValueError("promotion/bisection controls must be positive")
    if not 0.0 <= a.boundary_safety_fraction <= 0.1:
        raise ValueError("boundary-safety-fraction must lie in [0,0.1]")

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

    tok_data = AutoTokenizer.from_pretrained(a.model_path)
    if tok_data.pad_token is None:
        tok_data.pad_token = tok_data.eos_token
    instances, source_indices = locked.load_forget_instances(
        forget_path, tok_data, a.forget_num
    )

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    input_weight = model.get_input_embeddings().weight
    output_weight = output_embeddings.weight
    device = gagd.first_device(model)

    # Freeze requirements from untouched Stage1A.
    stage1a_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    required_nll, _, target_nll = native.build_required_forget_nll(
        stage1a_nll,
        target_probability=a.target_forget_answer_probability,
        target_nll_buffer=a.target_nll_buffer,
    )
    locked.write_json(
        root / "forget_instances_stage1a_before_row_restore.json",
        locked.report_instances(
            instances, stage1a_nll, required_nll, a.target_forget_answer_probability
        ),
    )

    answer_ids = old.all_answer_rows(tok, instances, max_length=a.max_length)
    if not answer_ids:
        raise RuntimeError("visible direct answers produced no vocabulary rows")
    ids_i = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    ids_o = ids_i.to(output_weight.device)
    stage1a_input_rows = input_weight.index_select(0, ids_i).detach().cpu().clone()
    stage1a_output_rows = output_weight.index_select(0, ids_o).detach().cpu().clone()
    base_input_rows, base_output_rows = load_reference_rows(
        str(reference_path), answer_ids, a.dtype
    )

    all_cache = tofu.build_answer_delta_caches(
        model,
        tok,
        instances,
        answer_ids,
        device,
        batch_size=a.batch_size,
        max_length=a.max_length,
    )
    initial_violating = old.violating_sequence_positions(
        stage1a_nll, required_nll, a.comparison_tolerance
    )
    sensitive_ids: set[int] = set(
        old.sensitive_rows_from_token_deficits(
            all_cache,
            answer_ids,
            required_nll,
            initial_violating,
            tolerance=a.comparison_tolerance,
        )
    )
    if initial_violating and not sensitive_ids:
        sensitive_ids.update(
            old.rows_for_positions(
                tok, instances, initial_violating, max_length=a.max_length
            )
        )

    attempts: List[Dict[str, Any]] = []
    final_attempt: RepairAttempt | None = None
    final_baseline_rows: torch.Tensor | None = None

    for promotion_round in range(a.max_promotion_rounds):
        # Reset the entire visible-answer row state deterministically before each retry.
        old.apply_answer_row_policy(
            input_weight,
            output_weight,
            answer_ids,
            sorted(sensitive_ids),
            stage1a_input_rows,
            stage1a_output_rows,
            base_input_rows,
            base_output_rows,
        )
        selected_ids = sorted(sensitive_ids)
        non_sensitive_ids = sorted(set(answer_ids) - sensitive_ids)
        selected_tensor = torch.tensor(
            selected_ids, dtype=torch.long, device=output_weight.device
        )
        baseline_rows = (
            output_weight.index_select(0, selected_tensor).detach().clone()
            if selected_ids
            else output_weight.new_empty((0, output_weight.shape[1]))
        )

        pre_nll = tofu.score_answer_instances(
            model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
        ).detach().float()
        pre_violating = old.violating_sequence_positions(
            pre_nll, required_nll, a.comparison_tolerance
        )
        print(
            f"promotion-round={promotion_round} sensitive={len(selected_ids)} "
            f"non_sensitive={len(non_sensitive_ids)} pre_rank0_violating={len(pre_violating)}"
        )

        attempt = run_restricted_rank0(
            model,
            tok,
            instances,
            selected_ids,
            required_nll,
            target_nll,
            a,
            device,
        )
        attempt_report = {
            "round": promotion_round,
            "sensitive_row_count": len(selected_ids),
            "non_sensitive_row_count": len(non_sensitive_ids),
            "pre_rank0_violating_sequence_count": len(pre_violating),
            "repair_feasible": attempt.feasible,
            "repair_step": attempt.step,
            "optimizer_crossing_step": attempt.optimizer_crossing_step,
            "boundary_alpha": attempt.boundary_alpha,
            "candidate_delta_norm": float(attempt.delta_cpu.norm().item()),
            "candidate_forget_answer_probability_max": float(
                torch.exp(-attempt.nll_cpu).max().item()
            ),
            "candidate_minimum_nll_slack": float(
                (attempt.nll_cpu - required_nll.detach().cpu()).min().item()
            ),
        }
        attempts.append(attempt_report)

        if attempt.feasible:
            final_attempt = attempt
            final_baseline_rows = baseline_rows
            break

        # Only now promote rows: rank-0 on the current sensitive set actually failed.
        residual_positions = old.violating_sequence_positions(
            attempt.nll_cpu,
            required_nll.detach().cpu(),
            a.comparison_tolerance,
        )
        promotable = set(
            old.rows_for_positions(
                tok, instances, residual_positions, max_length=a.max_length
            )
        ) - sensitive_ids
        if not promotable:
            write_json(root / "promotion_attempts.json", attempts)
            raise RuntimeError(
                "restricted rank-0 failed but no additional non-sensitive answer rows remain to promote"
            )
        sensitive_ids.update(promotable)
        attempt_report["promoted_token_ids"] = sorted(promotable)
        attempt_report["promoted_row_count"] = len(promotable)
    else:
        write_json(root / "promotion_attempts.json", attempts)
        raise RuntimeError("restricted rank-0 exceeded max promotion rounds")

    assert final_attempt is not None
    assert final_baseline_rows is not None
    selected_ids = sorted(sensitive_ids)
    non_sensitive_ids = sorted(set(answer_ids) - sensitive_ids)

    # Rebuild exact pre-rank0 row policy, then materialize only the selected-row delta.
    old.apply_answer_row_policy(
        input_weight,
        output_weight,
        answer_ids,
        selected_ids,
        stage1a_input_rows,
        stage1a_output_rows,
        base_input_rows,
        base_output_rows,
    )
    tofu.set_selected_lm_head_rows(
        output_weight,
        selected_ids,
        final_baseline_rows,
        final_attempt.delta_cpu,
    )

    # Snap non-sensitive rows to Base once more after rank0 materialization.
    id_to_position = {int(token_id): pos for pos, token_id in enumerate(answer_ids)}
    if non_sensitive_ids:
        ns_pos = torch.tensor([id_to_position[x] for x in non_sensitive_ids], dtype=torch.long)
        ns_i = torch.tensor(non_sensitive_ids, dtype=torch.long, device=input_weight.device)
        ns_o = ns_i.to(output_weight.device)
        input_weight.index_copy_(
            0,
            ns_i,
            base_input_rows.index_select(0, ns_pos).to(
                device=input_weight.device, dtype=input_weight.dtype
            ),
        )
        output_weight.index_copy_(
            0,
            ns_o,
            base_output_rows.index_select(0, ns_pos).to(
                device=output_weight.device, dtype=output_weight.dtype
            ),
        )

    materialized_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    materialized_metrics = locked.metrics(
        materialized_nll,
        required_nll,
        final_attempt.delta_cpu.to(device),
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    if not feasible_nll(materialized_nll, required_nll, a.comparison_tolerance):
        write_json(
            root / "repair_summary.json",
            {
                "status": "FAILED_DTYPE_MATERIALIZATION",
                "materialized_metrics": materialized_metrics,
                "promotion_attempts": attempts,
            },
        )
        raise RuntimeError("near-boundary restricted rank-0 lost feasibility after materialization")

    input_restore_error = old.answer_row_restoration_error(
        input_weight, answer_ids, non_sensitive_ids, base_input_rows
    )
    output_restore_error = old.answer_row_restoration_error(
        output_weight, answer_ids, non_sensitive_ids, base_output_rows
    )
    if input_restore_error != 0.0 or output_restore_error != 0.0:
        raise RuntimeError(
            f"non-sensitive rows are not exact Base: input={input_restore_error} output={output_restore_error}"
        )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    locked.write_json(
        root / "forget_instances_after.json",
        locked.report_instances(
            instances, materialized_nll, required_nll, a.target_forget_answer_probability
        ),
    )
    write_json(root / "promotion_attempts.json", attempts)
    write_json(
        root / "answer_row_restoration.json",
        {
            "policy": "restore non-sensitive visible answer rows exactly to Full-TOFU Base before and after restricted rank-0",
            "initial_sensitive_row_count": attempts[0]["sensitive_row_count"],
            "all_visible_answer_row_count": len(answer_ids),
            "sensitive_answer_row_count": len(selected_ids),
            "sensitive_answer_token_ids": selected_ids,
            "non_sensitive_answer_row_count": len(non_sensitive_ids),
            "non_sensitive_answer_token_ids": non_sensitive_ids,
            "promotion_rounds": attempts,
            "promotion_trigger": "only after restricted rank-0 fails to satisfy direct constraints",
            "retain_or_heldout_data_consulted": False,
        },
    )
    summary = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "best_step": final_attempt.step,
        "optimizer_crossing_step": final_attempt.optimizer_crossing_step,
        "boundary_alpha": final_attempt.boundary_alpha,
        "repair_rank_requested": 0,
        "repair_rank_actual": 0,
        "repair_rank_semantics": "unrestricted selected-row delta on sensitive answer rows only",
        "all_visible_answer_row_count": len(answer_ids),
        "sensitive_answer_row_count": len(selected_ids),
        "non_sensitive_answer_row_count": len(non_sensitive_ids),
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "non_sensitive_input_base_max_abs_error": input_restore_error,
        "non_sensitive_output_base_max_abs_error": output_restore_error,
        "materialized_metrics": materialized_metrics,
        "minimum_materialized_nll_slack": float(
            (materialized_nll - required_nll).min().detach().cpu().item()
        ),
        "promotion_attempts": attempts,
        "training_data_access": {
            "direct_forget_qas": a.forget_num,
            "full_tofu_reference_rows": "visible answer token rows only",
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
            "schema_version": 3,
            "method": METHOD,
            "protocol": PROTOCOL,
            "model_path": a.model_path,
            "reference_model_path": str(reference_path),
            "forget_json": str(forget_path),
            "seed": a.seed,
            "forget_num": a.forget_num,
            "target_forget_answer_probability": a.target_forget_answer_probability,
            "target_nll_buffer": a.target_nll_buffer,
            "repair_steps": a.repair_steps,
            "repair_lr": a.repair_lr,
            "repair_optimizer": a.repair_optimizer,
            "repair_rank": 0,
            "delta_l2_lambda": a.delta_l2_lambda,
            "boundary_bisection_steps": a.boundary_bisection_steps,
            "boundary_safety_fraction": a.boundary_safety_fraction,
            "sensitivity_rule": "initial token-NLL deficits; promotion only if restricted rank-0 is infeasible",
            "answer_row_restoration": "non-sensitive visible answer rows exact Full-TOFU Base in input embedding and LM head",
            "stage1b_visible_source_indices": source_indices,
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print(
        "SURE-TOFU Stage1B-v2 PASS "
        f"sensitive={len(selected_ids)} non_sensitive={len(non_sensitive_ids)} "
        f"cross_step={final_attempt.optimizer_crossing_step} alpha={final_attempt.boundary_alpha} "
        f"norm={materialized_metrics['selected_lm_head_delta_norm']:.6g} "
        f"max_prob={materialized_metrics['forget_answer_probability_max']:.8g} "
        f"min_slack={summary['minimum_materialized_nll_slack']:.6g}"
    )
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
