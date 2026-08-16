#!/usr/bin/env python3
"""SURE-TOFU V5: R512 forget-hidden repair followed by sparse active-pair LM repair.

V5 keeps the locked author-balanced TOFU protocol and the V4 parameter policy:
  * Stage1A is already frozen before this script starts.
  * every training-visible answer-token INPUT embedding row is restored exactly
    to the protected Full-TOFU Base;
  * untouched/non-selected visible answer LM-head rows are exact Base;
  * the transformer is frozen and the LM head is untied.

The V5 change is the forgetting repair geometry.

Stage 1B-R512
  1. Select the same initial sparse sensitive rows as V4: the top K
     training-visible answer rows per Stage1A-violating direct QA using the v3
     rare-content-first ranking. No held-out/retain data is visible.
  2. Build a forget-hidden basis from teacher-forced answer-token final hidden
     states of the same 50 direct forget QAs.
  3. Optimize selected LM-head deltas only inside the first 512 numerical
     directions of that forget-hidden basis. If the restricted problem crosses
     feasibility, bisect back to a near-boundary candidate; otherwise retain the
     best training-only restricted candidate for Stage 2.

Stage 2 active-pair repair
  4. Materialize R512 in BF16 and re-score the same 50 direct QAs.
  5. For only residual violating QAs, identify deficient answer-token pairs
     (QA position, token position, vocabulary row) whose token NLL is below the
     sequence requirement. The unique rows from those pairs are the only rows
     eligible for Stage-2 unrestricted repair.
  6. Optimize a fresh unrestricted delta on those active-pair rows while
     enforcing all 50 direct forget constraints. Boundary candidates are
     materialized/audited with deterministic increasing safety fractions.

No retain95, paraphrases, same-author holdout, real-authors, world-facts, PPL,
or locked final metric is loaded or used by this script. Both the R512 and final
checkpoints are saved so the launcher can evaluate the contribution of Stage 2,
then delete checkpoint weights after evaluation.
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
import tofu_gagd_active_forget_repair as native
import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_rank0_forget as old
import tofu_sure_rank0_forget_restored as v2
import tofu_sure_rank0_forget_progressive_v3 as v3
import tofu_sure_rank0_forget_progressive_v4 as v4


METHOD = "SURE-TOFU-R512-active-pair-repair-v5"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"
DEFAULT_SAFETY_FRACTIONS = (0.002, 0.01, 0.02, 0.05, 0.10)


def parse_float_list(text: str) -> Tuple[float, ...]:
    values = tuple(float(x.strip()) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty float list")
    return values


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Frozen Stage1A checkpoint")
    p.add_argument("--reference-model-path", required=True, help="Protected Full-TOFU Base")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--initial-rows-per-example", type=int, default=3)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--target-nll-buffer", type=float, default=0.0)
    p.add_argument("--comparison-tolerance", type=float, default=1e-6)

    p.add_argument("--r512-rank", type=int, default=512)
    p.add_argument("--r512-steps", type=int, default=10000)
    p.add_argument("--r512-lr", type=float, default=5e-3)
    p.add_argument("--r512-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--r512-forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--r512-hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--r512-l2", type=float, default=1e-6)
    p.add_argument("--r512-boundary-bisection-steps", type=int, default=30)
    p.add_argument("--r512-boundary-safety-fraction", type=float, default=0.002)

    p.add_argument("--stage2-steps", type=int, default=10000)
    p.add_argument("--stage2-lr", type=float, default=5e-3)
    p.add_argument("--stage2-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--stage2-forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--stage2-hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--stage2-l2", type=float, default=1e-6)
    p.add_argument("--stage2-boundary-bisection-steps", type=int, default=30)
    p.add_argument(
        "--stage2-materialization-safety-fractions",
        type=parse_float_list,
        default=DEFAULT_SAFETY_FRACTIONS,
    )

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def validate(a: argparse.Namespace) -> None:
    if a.forget_num != 50:
        raise ValueError("V5 locked diagnostic fixes --forget-num=50")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if abs(float(a.target_nll_buffer)) > 1e-12:
        raise ValueError("V5 fixes --target-nll-buffer=0 to match V4")
    for name in (
        "initial_rows_per_example", "r512_rank", "r512_steps", "stage2_steps",
        "batch_size", "max_length", "log_every", "r512_boundary_bisection_steps",
        "stage2_boundary_bisection_steps",
    ):
        if int(getattr(a, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    for name in ("r512_lr", "stage2_lr"):
        if not math.isfinite(float(getattr(a, name))) or float(getattr(a, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be finite and positive")
    for name in (
        "r512_forget_hinge_weight", "r512_hardest_forget_hinge_weight",
        "stage2_forget_hinge_weight", "stage2_hardest_forget_hinge_weight",
    ):
        if float(getattr(a, name)) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive")
    for name in ("r512_l2", "stage2_l2", "comparison_tolerance"):
        if float(getattr(a, name)) < 0:
            raise ValueError(f"--{name.replace('_','-')} must be non-negative")
    if not 0.0 <= a.r512_boundary_safety_fraction <= 0.1:
        raise ValueError("r512 boundary safety fraction must lie in [0,0.1]")
    if any(x < 0 or x > 0.5 for x in a.stage2_materialization_safety_fractions):
        raise ValueError("stage2 materialization safety fractions must lie in [0,0.5]")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def feasible(nll: torch.Tensor, required: torch.Tensor, tolerance: float) -> bool:
    return v2.feasible_nll(nll, required, tolerance)


def metrics(
    nll: torch.Tensor,
    required: torch.Tensor,
    delta: torch.Tensor,
    *,
    target_nll: float,
    tolerance: float,
) -> Dict[str, Any]:
    return locked.metrics(
        nll, required, delta,
        target_nll=target_nll,
        tolerance=tolerance,
    )


def cache_hidden_rows(caches: Sequence[tofu.TOFUAnswerDeltaCache]) -> torch.Tensor:
    rows = [cache.hidden for cache in caches if cache.hidden.numel()]
    if not rows:
        raise RuntimeError("no direct-forget answer-token hidden states were cached")
    return torch.cat(rows, dim=0).float()


def optimize_restricted_r512(
    packed: Any,
    direction_basis: torch.Tensor,
    n_rows: int,
    hidden_size: int,
    required_nll: torch.Tensor,
    target_nll: float,
    a: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    module = active.SelectedRowDelta(
        n_rows,
        hidden_size,
        direction_basis=direction_basis,
        retained_basis=None,
        device=device,
    )
    optimizer = active.make_repair_optimizer(module, a.r512_optimizer, a.r512_lr)
    zero = torch.zeros((n_rows, hidden_size), dtype=torch.float32, device=device)
    zero_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero)
    best_delta = zero.detach().clone()
    best_nll = zero_nll.detach().clone()
    best_metrics = metrics(
        zero_nll, required_nll, zero,
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    previous_delta = zero.detach().clone()
    logs: List[Dict[str, Any]] = []
    crossing_step: int | None = None
    boundary_alpha: float | None = None

    if feasible(zero_nll, required_nll, a.comparison_tolerance):
        del optimizer
        return zero.detach(), zero_nll.detach(), {
            **best_metrics,
            "steps_completed": 0,
            "optimizer_crossing_step": None,
            "boundary_alpha": 0.0,
            "cached_feasible": True,
        }, logs

    for step in range(1, a.r512_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        current = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
        errors = torch.relu(required_nll.to(current) - current)
        loss = (
            a.r512_forget_hinge_weight * errors.square().mean()
            + a.r512_hardest_forget_hinge_weight * errors.square().max()
            + a.r512_l2 * delta.square().sum()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite R512 loss at step {step}")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
            candidate_metrics = metrics(
                candidate_nll, required_nll, candidate,
                target_nll=target_nll,
                tolerance=a.comparison_tolerance,
            )
            if old.sure_priority(candidate_metrics) < old.sure_priority(best_metrics):
                best_delta = candidate.detach().clone()
                best_nll = candidate_nll.detach().clone()
                best_metrics = dict(candidate_metrics)

            if step == 1 or step % a.log_every == 0:
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
                    f"r512-step={step} active={candidate_metrics['active_forget_instance_count']} "
                    f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g} "
                    f"norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
                )

            if feasible(candidate_nll, required_nll, a.comparison_tolerance):
                boundary_delta, boundary_nll, alpha = v2.boundary_bisect(
                    packed,
                    previous_delta,
                    candidate,
                    required_nll,
                    tolerance=a.comparison_tolerance,
                    iterations=a.r512_boundary_bisection_steps,
                    safety_fraction=a.r512_boundary_safety_fraction,
                )
                best_delta = boundary_delta.detach().clone()
                best_nll = boundary_nll.detach().clone()
                best_metrics = metrics(
                    best_nll, required_nll, best_delta,
                    target_nll=target_nll,
                    tolerance=a.comparison_tolerance,
                )
                crossing_step = step
                boundary_alpha = alpha
                break
            previous_delta = candidate.detach().clone()

    del optimizer
    report = {
        **best_metrics,
        "steps_completed": crossing_step or a.r512_steps,
        "optimizer_crossing_step": crossing_step,
        "boundary_alpha": boundary_alpha,
        "cached_feasible": feasible(best_nll, required_nll, a.comparison_tolerance),
    }
    return best_delta.detach(), best_nll.detach(), report, logs


def active_pair_report(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    all_answer_ids: Sequence[int],
    required_nll: torch.Tensor,
    residual_positions: Sequence[int],
    tolerance: float,
    tok: Any,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    unique_rows: set[int] = set()
    pairs: List[Dict[str, Any]] = []
    for qa_position in residual_positions:
        cache = caches[qa_position]
        requirement = required_nll[qa_position].to(
            device=cache.base_token_nll.device,
            dtype=cache.base_token_nll.dtype,
        )
        deficient = cache.base_token_nll < (requirement - tolerance)
        token_positions = deficient.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
        for token_position in token_positions:
            column = int(cache.target_selected_columns[token_position].detach().cpu())
            if column < 0:
                raise RuntimeError("all-answer active-pair cache lost a target vocabulary row")
            token_id = int(all_answer_ids[column])
            unique_rows.add(token_id)
            pairs.append(
                {
                    "qa_position": int(qa_position),
                    "answer_token_position": int(token_position),
                    "token_id": token_id,
                    "token": locked.decoded_token(tok, token_id),
                    "token_nll": float(cache.base_token_nll[token_position].detach().cpu()),
                    "required_sequence_nll": float(requirement.detach().cpu()),
                    "token_nll_deficit": float(
                        (requirement - cache.base_token_nll[token_position]).detach().cpu()
                    ),
                }
            )
    return sorted(unique_rows), pairs


def optimize_unrestricted_stage2(
    packed: Any,
    n_rows: int,
    hidden_size: int,
    required_nll: torch.Tensor,
    target_nll: float,
    a: argparse.Namespace,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[Dict[str, Any]], torch.Tensor, torch.Tensor]:
    module = active.SelectedRowDelta(
        n_rows,
        hidden_size,
        direction_basis=None,
        retained_basis=None,
        device=device,
    )
    optimizer = active.make_repair_optimizer(module, a.stage2_optimizer, a.stage2_lr)
    zero = torch.zeros((n_rows, hidden_size), dtype=torch.float32, device=device)
    zero_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero)
    previous_delta = zero.detach().clone()
    previous_nll = zero_nll.detach().clone()
    best_delta = zero.detach().clone()
    best_nll = zero_nll.detach().clone()
    best_metrics = metrics(
        zero_nll, required_nll, zero,
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    logs: List[Dict[str, Any]] = []
    crossing_low = zero.detach().clone()
    crossing_high = zero.detach().clone()
    crossing_step: int | None = None

    for step in range(1, a.stage2_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        current = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
        errors = torch.relu(required_nll.to(current) - current)
        loss = (
            a.stage2_forget_hinge_weight * errors.square().mean()
            + a.stage2_hardest_forget_hinge_weight * errors.square().max()
            + a.stage2_l2 * delta.square().sum()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage2 loss at step {step}")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
            candidate_metrics = metrics(
                candidate_nll, required_nll, candidate,
                target_nll=target_nll,
                tolerance=a.comparison_tolerance,
            )
            if old.sure_priority(candidate_metrics) < old.sure_priority(best_metrics):
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
                crossing_step = step
                best_delta = candidate.detach().clone()
                best_nll = candidate_nll.detach().clone()
                best_metrics = dict(candidate_metrics)
                break
            previous_delta = candidate.detach().clone()
            previous_nll = candidate_nll.detach().clone()

    del optimizer
    if crossing_step is None:
        raise RuntimeError(
            "V5 Stage2 active-pair unrestricted repair did not reach cached feasibility"
        )
    return (
        best_delta.detach(), best_nll.detach(),
        {**best_metrics, "optimizer_crossing_step": crossing_step},
        logs, crossing_low, crossing_high,
    )


@torch.no_grad()
def set_rows(
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    baseline_rows: torch.Tensor,
    delta: torch.Tensor,
) -> None:
    tofu.set_selected_lm_head_rows(output_weight, row_ids, baseline_rows, delta)


@torch.no_grad()
def all_input_base_error(
    input_weight: torch.Tensor,
    answer_ids: Sequence[int],
    base_input_rows: torch.Tensor,
) -> float:
    ids = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    current = input_weight.index_select(0, ids).detach().float().cpu()
    expected = base_input_rows.detach().float().cpu()
    return float((current - expected).abs().max().item()) if current.numel() else 0.0


def main() -> None:
    a = parse_args()
    validate(a)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    forget_path = Path(a.forget_json).resolve()
    reference_path = Path(a.reference_model_path).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
    if not reference_path.is_dir():
        raise FileNotFoundError(reference_path)

    root = gagd.resolve_output_path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    final_ckpt = root / "checkpoint"
    r512_ckpt = root / "r512" / "checkpoint"

    tok_data = AutoTokenizer.from_pretrained(a.model_path)
    if tok_data.pad_token is None:
        tok_data.pad_token = tok_data.eos_token
    instances, source_indices = locked.load_forget_instances(
        forget_path, tok_data, a.forget_num
    )

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    device = gagd.first_device(model)

    # Freeze the direct-forget requirements from untouched Stage1A, exactly as V4.
    stage1a_nll = tofu.score_answer_instances(
        model, tok, instances, device,
        batch_size=a.batch_size, max_length=a.max_length,
    ).detach().float()
    required_nll, _, target_nll = native.build_required_forget_nll(
        stage1a_nll,
        target_probability=a.target_forget_answer_probability,
        target_nll_buffer=0.0,
    )
    locked.write_json(
        root / "forget_instances_stage1a_before_restore.json",
        locked.report_instances(
            instances, stage1a_nll, required_nll,
            a.target_forget_answer_probability,
        ),
    )

    answer_ids = old.all_answer_rows(tok, instances, max_length=a.max_length)
    if not answer_ids:
        raise RuntimeError("visible direct forget answers produced no answer-token rows")
    ids_i = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    ids_o = ids_i.to(output_weight.device)
    stage1a_input_rows = input_weight.index_select(0, ids_i).detach().cpu().clone()
    stage1a_output_rows = output_weight.index_select(0, ids_o).detach().cpu().clone()
    base_input_rows, base_output_rows = old.load_reference_answer_rows(
        str(reference_path), answer_ids, a.dtype
    )

    rankings, ranking_report = v3.build_progressive_rankings(
        tok, instances, max_length=a.max_length
    )
    initial_violating = old.violating_sequence_positions(
        stage1a_nll, required_nll, a.comparison_tolerance
    )
    sensitive_ids: set[int] = set()
    initial_by_position = v3.add_next_rows(
        sensitive_ids,
        rankings,
        initial_violating,
        a.initial_rows_per_example,
    )
    if initial_violating and not sensitive_ids:
        raise RuntimeError("V5 selector produced no initial sensitive rows")
    selected_ids = sorted(sensitive_ids)
    nonselected_ids = sorted(set(answer_ids) - sensitive_ids)

    # V4 policy: selected output rows keep Stage1A; all other output rows Base;
    # every visible answer input embedding row is forced to Base.
    v4.apply_v4_row_policy(
        input_weight, output_weight, answer_ids, selected_ids,
        stage1a_input_rows, stage1a_output_rows,
        base_input_rows, base_output_rows,
    )
    input_error_before = all_input_base_error(input_weight, answer_ids, base_input_rows)
    nonselected_output_error = old.answer_row_restoration_error(
        output_weight, answer_ids, nonselected_ids, base_output_rows
    )
    if input_error_before != 0.0 or nonselected_output_error != 0.0:
        raise RuntimeError(
            "V5 pre-repair Base restoration failed: "
            f"all_input={input_error_before} nonselected_output={nonselected_output_error}"
        )

    pre_r512_nll = tofu.score_answer_instances(
        model, tok, instances, device,
        batch_size=a.batch_size, max_length=a.max_length,
    ).detach().float()
    pre_r512_violating = old.violating_sequence_positions(
        pre_r512_nll, required_nll, a.comparison_tolerance
    )

    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    r512_baseline_rows = output_weight.index_select(0, selected_tensor).detach().clone()
    r512_caches = tofu.build_answer_delta_caches(
        model, tok, instances, selected_ids, device,
        batch_size=a.batch_size, max_length=a.max_length,
    )
    r512_packed = tofu.pack_answer_delta_caches(r512_caches)
    forget_hidden = cache_hidden_rows(r512_caches)
    r512_basis = active.orthonormal_row_basis(forget_hidden, max_rank=a.r512_rank)
    if r512_basis.shape[0] == 0:
        raise RuntimeError("R512 forget-hidden basis has zero numerical rank")

    print(
        f"===== V5 R512 ===== selected_rows={len(selected_ids)} "
        f"answer_hidden_rows={forget_hidden.shape[0]} requested_rank={a.r512_rank} "
        f"actual_rank={r512_basis.shape[0]} pre_violating={len(pre_r512_violating)}"
    )
    r512_delta, r512_cached_nll, r512_cached_metrics, r512_logs = optimize_restricted_r512(
        r512_packed,
        r512_basis,
        len(selected_ids),
        int(output_weight.shape[1]),
        required_nll,
        target_nll,
        a,
        device,
    )
    write_jsonl(root / "r512" / "repair_log.jsonl", r512_logs)
    set_rows(output_weight, selected_ids, r512_baseline_rows, r512_delta)

    # Output-only edits must leave every visible answer input embedding exact Base.
    input_error_r512 = all_input_base_error(input_weight, answer_ids, base_input_rows)
    if input_error_r512 != 0.0:
        raise RuntimeError(f"R512 modified/restored input embeddings incorrectly: {input_error_r512}")

    r512_materialized_nll = tofu.score_answer_instances(
        model, tok, instances, device,
        batch_size=a.batch_size, max_length=a.max_length,
    ).detach().float()
    r512_materialized_metrics = metrics(
        r512_materialized_nll, required_nll, r512_delta.to(device),
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    r512_residual = old.violating_sequence_positions(
        r512_materialized_nll, required_nll, a.comparison_tolerance
    )

    r512_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(r512_ckpt)
    tok.save_pretrained(r512_ckpt)
    locked.write_json(
        root / "r512" / "forget_instances_after.json",
        locked.report_instances(
            instances, r512_materialized_nll, required_nll,
            a.target_forget_answer_probability,
        ),
    )
    write_json(
        root / "r512" / "summary.json",
        {
            "status": "PASS_TO_STAGE2" if r512_residual else "PASS_ALL_DIRECT",
            "requested_rank": a.r512_rank,
            "actual_rank": int(r512_basis.shape[0]),
            "hidden_size": int(r512_basis.shape[1]),
            "answer_hidden_state_count": int(forget_hidden.shape[0]),
            "selected_lm_head_row_count": len(selected_ids),
            "selected_lm_head_token_ids": selected_ids,
            "pre_r512_violating_sequence_count": len(pre_r512_violating),
            "cached_metrics": r512_cached_metrics,
            "materialized_metrics": r512_materialized_metrics,
            "residual_violating_sequence_count": len(r512_residual),
            "residual_violating_positions": r512_residual,
            "all_visible_answer_input_base_max_abs_error": input_error_r512,
            "benchmark_retain_seen": 0,
            "heldout_or_paraphrase_seen": 0,
            "selection_uses_only_training_visible_direct_forget_qas": True,
            "checkpoint": str(r512_ckpt.resolve()),
        },
    )

    # Build the Stage-2 active-pair set only from residual direct-forget QAs.
    all_answer_cache = tofu.build_answer_delta_caches(
        model, tok, instances, answer_ids, device,
        batch_size=a.batch_size, max_length=a.max_length,
    )
    stage2_ids, pair_rows = active_pair_report(
        all_answer_cache,
        answer_ids,
        required_nll,
        r512_residual,
        a.comparison_tolerance,
        tok,
    )
    if r512_residual and not stage2_ids:
        # Fail closed: a residual sequence must expose at least one answer row.
        stage2_ids = old.rows_for_positions(
            tok, instances, r512_residual, max_length=a.max_length
        )
        pair_rows.append(
            {
                "fallback": True,
                "reason": "residual sequence had no token-level row below sequence requirement",
                "fallback_unique_rows": stage2_ids,
            }
        )

    stage2_summary: Dict[str, Any]
    stage2_delta = torch.empty((0, output_weight.shape[1]), dtype=torch.float32, device=device)
    if not r512_residual:
        stage2_summary = {
            "status": "SKIPPED_R512_ALREADY_FEASIBLE",
            "residual_violating_sequence_count_before": 0,
            "active_pair_count": 0,
            "active_pair_unique_row_count": 0,
            "active_pair_unique_token_ids": [],
            "stage2_delta_norm": 0.0,
            "materialization_safety_fraction": None,
        }
        final_nll = r512_materialized_nll
        final_metrics = r512_materialized_metrics
    else:
        stage2_tensor = torch.tensor(stage2_ids, dtype=torch.long, device=output_weight.device)
        stage2_baseline_rows = output_weight.index_select(0, stage2_tensor).detach().clone()
        stage2_caches = tofu.build_answer_delta_caches(
            model, tok, instances, stage2_ids, device,
            batch_size=a.batch_size, max_length=a.max_length,
        )
        stage2_packed = tofu.pack_answer_delta_caches(stage2_caches)
        (
            stage2_high_delta,
            stage2_cached_nll,
            stage2_cached_metrics,
            stage2_logs,
            crossing_low,
            crossing_high,
        ) = optimize_unrestricted_stage2(
            stage2_packed,
            len(stage2_ids),
            int(output_weight.shape[1]),
            required_nll,
            target_nll,
            a,
            device,
        )
        write_jsonl(root / "stage2_active_pairs" / "repair_log.jsonl", stage2_logs)

        selected_candidate: torch.Tensor | None = None
        selected_fraction: float | None = None
        materialization_trials: List[Dict[str, Any]] = []
        for safety in sorted(set(float(x) for x in a.stage2_materialization_safety_fractions)):
            candidate, candidate_cached_nll, alpha = v2.boundary_bisect(
                stage2_packed,
                crossing_low,
                crossing_high,
                required_nll,
                tolerance=a.comparison_tolerance,
                iterations=a.stage2_boundary_bisection_steps,
                safety_fraction=safety,
            )
            set_rows(output_weight, stage2_ids, stage2_baseline_rows, candidate)
            actual_nll = tofu.score_answer_instances(
                model, tok, instances, device,
                batch_size=a.batch_size, max_length=a.max_length,
            ).detach().float()
            pass_actual = feasible(actual_nll, required_nll, a.comparison_tolerance)
            actual_metrics = metrics(
                actual_nll, required_nll, candidate.to(device),
                target_nll=target_nll,
                tolerance=a.comparison_tolerance,
            )
            materialization_trials.append(
                {
                    "safety_fraction": safety,
                    "boundary_alpha": alpha,
                    "cached_minimum_nll_slack": float(
                        (candidate_cached_nll - required_nll.to(candidate_cached_nll)).min().detach().cpu()
                    ),
                    "materialized_pass": bool(pass_actual),
                    "materialized_active_count": actual_metrics["active_forget_instance_count"],
                    "materialized_max_probability": actual_metrics["forget_answer_probability_max"],
                    "delta_norm": float(candidate.norm().detach().cpu()),
                }
            )
            if pass_actual:
                selected_candidate = candidate.detach().clone()
                selected_fraction = safety
                final_nll = actual_nll
                final_metrics = actual_metrics
                break
            set_rows(
                output_weight,
                stage2_ids,
                stage2_baseline_rows,
                torch.zeros_like(candidate),
            )

        if selected_candidate is None:
            # Last fail-closed attempt: materialize the known cached-feasible optimizer crossing.
            set_rows(output_weight, stage2_ids, stage2_baseline_rows, crossing_high)
            actual_nll = tofu.score_answer_instances(
                model, tok, instances, device,
                batch_size=a.batch_size, max_length=a.max_length,
            ).detach().float()
            actual_metrics = metrics(
                actual_nll, required_nll, crossing_high.to(device),
                target_nll=target_nll,
                tolerance=a.comparison_tolerance,
            )
            materialization_trials.append(
                {
                    "safety_fraction": "optimizer_crossing_endpoint",
                    "boundary_alpha": 1.0,
                    "materialized_pass": feasible(actual_nll, required_nll, a.comparison_tolerance),
                    "materialized_active_count": actual_metrics["active_forget_instance_count"],
                    "materialized_max_probability": actual_metrics["forget_answer_probability_max"],
                    "delta_norm": float(crossing_high.norm().detach().cpu()),
                }
            )
            if not feasible(actual_nll, required_nll, a.comparison_tolerance):
                write_json(
                    root / "stage2_active_pairs" / "failure.json",
                    {
                        "status": "FAILED_BF16_MATERIALIZATION",
                        "active_pairs": pair_rows,
                        "active_pair_unique_token_ids": stage2_ids,
                        "cached_metrics": stage2_cached_metrics,
                        "materialization_trials": materialization_trials,
                    },
                )
                raise RuntimeError("V5 Stage2 lost feasibility after BF16 materialization")
            selected_candidate = crossing_high.detach().clone()
            selected_fraction = -1.0
            final_nll = actual_nll
            final_metrics = actual_metrics

        stage2_delta = selected_candidate
        stage2_summary = {
            "status": "PASS",
            "residual_violating_sequence_count_before": len(r512_residual),
            "residual_violating_positions_before": r512_residual,
            "active_pair_count": sum(1 for row in pair_rows if not row.get("fallback")),
            "active_pair_unique_row_count": len(stage2_ids),
            "active_pair_unique_token_ids": stage2_ids,
            "active_pair_unique_tokens": {
                str(token_id): locked.decoded_token(tok, token_id) for token_id in stage2_ids
            },
            "cached_metrics_at_optimizer_crossing": stage2_cached_metrics,
            "stage2_delta_norm": float(selected_candidate.norm().detach().cpu()),
            "selected_materialization_safety_fraction": selected_fraction,
            "materialization_trials": materialization_trials,
        }

    write_json(
        root / "stage2_active_pairs" / "active_pairs.json",
        {
            "residual_violating_sequence_count_after_r512": len(r512_residual),
            "residual_violating_positions_after_r512": r512_residual,
            "active_pair_count": sum(1 for row in pair_rows if not row.get("fallback")),
            "active_pair_unique_row_count": len(stage2_ids),
            "active_pair_unique_token_ids": stage2_ids,
            "pairs": pair_rows,
            "retain_or_heldout_data_consulted": False,
        },
    )
    write_json(root / "stage2_active_pairs" / "summary.json", stage2_summary)

    final_pass = feasible(final_nll, required_nll, a.comparison_tolerance)
    if not final_pass:
        raise RuntimeError("V5 final direct-forget audit is not feasible")

    # Final restoration audits. Every visible answer input row must still be Base.
    input_error_final = all_input_base_error(input_weight, answer_ids, base_input_rows)
    modified_output_ids = sorted(set(selected_ids) | set(stage2_ids))
    untouched_output_ids = sorted(set(answer_ids) - set(modified_output_ids))
    untouched_output_error = old.answer_row_restoration_error(
        output_weight, answer_ids, untouched_output_ids, base_output_rows
    )
    if input_error_final != 0.0 or untouched_output_error != 0.0:
        raise RuntimeError(
            "V5 final restoration audit failed: "
            f"all_input={input_error_final} untouched_output={untouched_output_error}"
        )

    final_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_ckpt)
    tok.save_pretrained(final_ckpt)
    locked.write_json(
        root / "forget_instances_after.json",
        locked.report_instances(
            instances, final_nll, required_nll,
            a.target_forget_answer_probability,
        ),
    )

    write_json(
        root / "row_selection.json",
        {
            **ranking_report,
            "initial_rows_per_example": a.initial_rows_per_example,
            "initial_violating_positions": initial_violating,
            "initial_selected_by_position": {
                str(k): v for k, v in sorted(initial_by_position.items())
            },
            "r512_selected_unique_row_count": len(selected_ids),
            "r512_selected_token_ids": selected_ids,
            "r512_selected_tokens": {
                str(token_id): locked.decoded_token(tok, token_id) for token_id in selected_ids
            },
            "stage2_active_pair_unique_row_count": len(stage2_ids),
            "stage2_active_pair_token_ids": stage2_ids,
            "retain_or_heldout_data_consulted": False,
        },
    )

    summary = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": a.seed,
        "forget_num": a.forget_num,
        "target_forget_answer_probability": a.target_forget_answer_probability,
        "target_nll_buffer": 0.0,
        "all_visible_answer_row_count": len(answer_ids),
        "all_visible_answer_input_base_max_abs_error": input_error_final,
        "untouched_visible_answer_output_base_max_abs_error": untouched_output_error,
        "r512": {
            "requested_rank": a.r512_rank,
            "actual_rank": int(r512_basis.shape[0]),
            "selected_row_count": len(selected_ids),
            "delta_norm": float(r512_delta.norm().detach().cpu()),
            "cached_metrics": r512_cached_metrics,
            "materialized_metrics": r512_materialized_metrics,
            "residual_violating_sequence_count": len(r512_residual),
        },
        "stage2": stage2_summary,
        "final": {
            "modified_output_row_union_count": len(modified_output_ids),
            "modified_output_token_ids": modified_output_ids,
            "materialized_metrics": final_metrics,
        },
        "training_data_access": {
            "direct_forget_qas": a.forget_num,
            "retain95": 0,
            "paraphrases": 0,
            "same_author_holdout": 0,
            "real_authors": 0,
            "world_facts": 0,
            "PPL": False,
        },
        "r512_checkpoint": str(r512_ckpt.resolve()),
        "checkpoint": str(final_ckpt.resolve()),
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
            "stage1b_visible_source_indices": source_indices,
            "parameter_scope": (
                "all answer input embeddings exact Base; R512 on initial sparse sensitive LM rows; "
                "unrestricted Stage2 only on residual deficient active-pair rows"
            ),
            "data_firewall": (
                "selection/optimization use only the same 50 training-visible direct forget QAs"
            ),
        },
    )

    print("===== SURE-TOFU V5 PASS =====")
    print(
        f"R512 requested={a.r512_rank} actual={r512_basis.shape[0]} "
        f"rows={len(selected_ids)} norm={float(r512_delta.norm().detach().cpu()):.6g} "
        f"residual_qas={len(r512_residual)}"
    )
    print(
        f"Stage2 active_pairs={sum(1 for row in pair_rows if not row.get('fallback'))} "
        f"unique_rows={len(stage2_ids)} norm={float(stage2_delta.norm().detach().cpu()):.6g}"
    )
    print(
        f"Final active={final_metrics['active_forget_instance_count']} "
        f"max_prob={final_metrics['forget_answer_probability_max']:.8g} "
        f"modified_output_rows={len(modified_output_ids)}"
    )
    print("R512 checkpoint:", r512_ckpt)
    print("Final checkpoint:", final_ckpt)


if __name__ == "__main__":
    main()
