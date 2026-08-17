#!/usr/bin/env python3
"""SURE-TOFU V9 Stage 2: minimum-collateral constrained LM-head repair.

Input is the successful bounded-top3 Stage-1 checkpoint.  Stage 1 is not
retrained.  V9 searches for the least Base-disruptive Stage-2 correction that
satisfies the exact full-answer deletion constraint on the same 50
training-visible forget QAs.

The preservation signal is leakage-free: it is the exact same-prompt
non-target KL from the protected Full-TOFU Base distribution on those same 50
visible answer contexts.  No retain, paraphrase, same-author holdout, PPL,
real-author, or world-fact data is loaded or consulted.

For a selected row set S, V9 solves an augmented-Lagrangian problem
approximately of the form

    min  KL(Base_non-target || Current_non-target)
       + fisher_weight * diagonal_Fisher_cost
       + l2 * ||Stage2 correction||^2
       + sum_i alpha_i [tau - NLL_i]_+
       + rho/2 * sum_i [tau - NLL_i]_+^2,

where alpha_i are non-negative dual variables updated only from the 50 visible
direct-forget constraints.  Passed QAs therefore stop receiving direct
forgetting pressure while KL continues to pull the solution toward Base.

Row growth is preservation-aware rather than feasibility-only.  We start from
the Stage-1 sensitive rows.  If deletion is not achieved, or is achieved only
above a same-prompt KL trust budget, we restore the exact Stage-1 checkpoint
and promote a small number of answer rows.  Promotion ranks candidates by
forget-gradient leverage divided by a row-specific diagonal-Fisher collateral
cost in the actual repair subspace.  Rank 1024 is the default because it was
the strongest utility-preserving V8 configuration, but rank remains explicit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_progressive_active_hidden_repair_v8 as v8
import tofu_sure_rank0_forget as old
import tofu_sure_rank0_forget_restored as boundary
from tofu_sure_rowspecific_null_v6_stable import stable_same_prompt_non_target_kl


METHOD = "SURE-TOFU-v9-minimum-collateral-constrained-repair"
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
    p.add_argument("--repair-rank", type=int, default=1024, help="0=unrestricted; >0=all-50 forget-hidden rank")
    p.add_argument("--repair-steps", type=int, default=12000, help="total optimizer-step budget across all rounds")
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--repair-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--fisher-weight", type=float, default=0.05)
    p.add_argument("--fisher-damping", type=float, default=1e-6)
    p.add_argument("--dual-rho", type=float, default=10.0)
    p.add_argument("--dual-lr", type=float, default=0.25)
    p.add_argument("--dual-init", type=float, default=1.0)
    p.add_argument("--dual-max", type=float, default=1000.0)
    p.add_argument("--dual-update-every", type=int, default=1)
    p.add_argument("--post-feasible-steps", type=int, default=250)
    p.add_argument("--max-promotion-rounds", type=int, default=8)
    p.add_argument("--promotion-rows-per-round", type=int, default=25)
    p.add_argument("--promotion-qa-count", type=int, default=20)
    p.add_argument(
        "--same-prompt-kl-budget",
        type=float,
        default=None,
        help="Explicit Base non-target KL trust budget. If omitted, derive from Stage1 KL.",
    )
    p.add_argument("--stage1-kl-budget-multiplier", type=float, default=20.0)
    p.add_argument("--kl-budget-floor", type=float, default=0.02)
    p.add_argument("--boundary-bisection-steps", type=int, default=30)
    p.add_argument("--materialization-safety-fractions", type=parse_float_list, default=DEFAULT_SAFETY_FRACTIONS)
    p.add_argument("--grad-clip", type=float, default=1.0)
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


@torch.no_grad()
def set_rows_exact(output_weight: torch.Tensor, row_ids: Sequence[int], rows: torch.Tensor) -> None:
    if not row_ids:
        return
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    output_weight.index_copy_(
        0,
        ids,
        rows.to(device=output_weight.device, dtype=output_weight.dtype),
    )


def subset_answer_caches(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    all_ids: Sequence[int],
    selected_ids: Sequence[int],
) -> List[tofu.TOFUAnswerDeltaCache]:
    lookup = {int(token_id): column for column, token_id in enumerate(all_ids)}
    columns = [lookup[int(token_id)] for token_id in selected_ids]
    output: List[tofu.TOFUAnswerDeltaCache] = []
    for cache in caches:
        column_tensor = torch.tensor(columns, dtype=torch.long, device=cache.selected_probs.device)
        remap = torch.full(
            (len(all_ids),), -1, dtype=torch.long, device=cache.target_selected_columns.device
        )
        remap[column_tensor.to(remap.device)] = torch.arange(
            len(columns), dtype=torch.long, device=remap.device
        )
        targets = cache.target_selected_columns
        new_targets = torch.full_like(targets, -1)
        mask = targets.ge(0)
        if mask.any():
            new_targets[mask] = remap[targets[mask]]
        output.append(
            tofu.TOFUAnswerDeltaCache(
                base_token_nll=cache.base_token_nll,
                hidden=cache.hidden,
                selected_probs=cache.selected_probs.index_select(1, column_tensor),
                target_selected_columns=new_targets,
            )
        )
    return output


def equal_example_base_non_target_kl(
    base_caches: Sequence[tofu.TOFUAnswerDeltaCache],
    total_delta_from_base: torch.Tensor,
) -> torch.Tensor:
    if not base_caches:
        return total_delta_from_base.new_zeros(())
    return torch.stack(
        [stable_same_prompt_non_target_kl([cache], total_delta_from_base) for cache in base_caches]
    ).mean()


def diagonal_non_target_fisher(
    base_caches: Sequence[tofu.TOFUAnswerDeltaCache],
    direction_basis: torch.Tensor | None,
) -> torch.Tensor:
    """Diagonal Base non-target Fisher in the actual Stage2 coefficient space."""
    if not base_caches:
        raise ValueError("cannot build Fisher geometry from empty caches")
    n_rows = int(base_caches[0].selected_probs.shape[1])
    width = (
        int(base_caches[0].hidden.shape[1])
        if direction_basis is None
        else int(direction_basis.shape[0])
    )
    diagonal = torch.zeros(
        (n_rows, width), dtype=torch.float32, device=base_caches[0].hidden.device
    )
    token_count = 0
    tiny = torch.finfo(torch.float32).tiny
    for cache in base_caches:
        hidden = cache.hidden.float()
        coordinates = hidden if direction_basis is None else hidden @ direction_basis.transpose(0, 1)
        q_target = torch.exp(-cache.base_token_nll.float()).clamp(max=1.0 - 1e-7)
        denom = (1.0 - q_target).clamp_min(tiny).unsqueeze(-1)
        p_non_target = (cache.selected_probs.float() / denom).clamp(min=0.0, max=1.0)
        selected_mask = cache.target_selected_columns.ge(0)
        if selected_mask.any():
            p_non_target = p_non_target.clone()
            rows = selected_mask.nonzero(as_tuple=False).flatten()
            cols = cache.target_selected_columns[selected_mask]
            p_non_target[rows, cols] = 0.0
        bernoulli_var = p_non_target * (1.0 - p_non_target)
        diagonal.add_(bernoulli_var.transpose(0, 1) @ coordinates.square())
        token_count += int(hidden.shape[0])
    if token_count <= 0:
        raise RuntimeError("Base caches contain no answer-token positions")
    return diagonal / float(token_count)


def fisher_quadratic(
    module: active.SelectedRowDelta,
    stage1_delta_from_base: torch.Tensor,
    fisher_diag: torch.Tensor,
    direction_basis: torch.Tensor | None,
) -> torch.Tensor:
    if direction_basis is None:
        correction = module.effective_delta()
        total_coordinates = stage1_delta_from_base + correction
    else:
        if module.coefficients is None:
            raise RuntimeError("ranked repair is missing coefficient parameters")
        stage1_coordinates = stage1_delta_from_base @ direction_basis.transpose(0, 1)
        total_coordinates = stage1_coordinates + module.coefficients
    if total_coordinates.shape != fisher_diag.shape:
        raise RuntimeError("Fisher geometry and repair coordinates have incompatible shapes")
    return (fisher_diag * total_coordinates.square()).sum() / float(max(1, total_coordinates.shape[0]))


def row_gradient_vector(
    cache: tofu.TOFUAnswerDeltaCache,
    column: int,
    direction_basis: torch.Tensor | None,
) -> torch.Tensor:
    probs = cache.selected_probs[:, column].float()
    indicator = cache.target_selected_columns.eq(int(column)).float()
    coeff = (probs - indicator) / float(max(1, cache.hidden.shape[0]))
    gradient = (coeff.unsqueeze(-1) * cache.hidden.float()).sum(dim=0)
    return gradient if direction_basis is None else gradient @ direction_basis.transpose(0, 1)


def fisher_adjusted_score(gradient: torch.Tensor, fisher_diag: torch.Tensor, damping: float) -> float:
    norm = gradient.norm()
    if float(norm.detach().cpu()) == 0.0:
        return 0.0
    unit = gradient / norm
    predicted_cost = torch.sqrt((fisher_diag * unit.square()).sum() + float(damping))
    return float((norm / predicted_cost).detach().cpu())


def document_frequency(tok: Any, instances: Sequence[tofu.TOFUAnswerInstance], max_length: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    for instance in instances:
        counts.update(set(locked.answer_token_ids(tok, instance, max_length=max_length)))
    return counts


def select_fisher_safe_promotions(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    all_answer_ids: Sequence[int],
    current_all_caches: Sequence[tofu.TOFUAnswerDeltaCache],
    all_fisher_diag: torch.Tensor,
    selected_ids: Sequence[int],
    direction_basis: torch.Tensor | None,
    pressure: torch.Tensor,
    *,
    qa_count: int,
    rows_per_round: int,
    max_length: int,
    damping: float,
) -> Tuple[List[int], Dict[str, Any]]:
    selected = set(int(x) for x in selected_ids)
    lookup = {int(token_id): column for column, token_id in enumerate(all_answer_ids)}
    df = document_frequency(tok, instances, max_length)
    pressure_cpu = pressure.detach().float().cpu()
    ordered_qas = sorted(
        range(len(instances)),
        key=lambda i: (-float(pressure_cpu[i]), int(i)),
    )[: min(int(qa_count), len(instances))]

    gradient_sums: Dict[int, torch.Tensor] = {}
    contributors: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    fallback_pool: set[int] = set()
    for position in ordered_qas:
        weight = max(float(pressure_cpu[position]), 1e-8)
        unique_ids = list(dict.fromkeys(locked.answer_token_ids(tok, instances[position], max_length=max_length)))
        for token_id in unique_ids:
            token_id = int(token_id)
            if token_id in selected or token_id not in lookup:
                continue
            fallback_pool.add(token_id)
            if not locked.is_content_bearing_token(tok, token_id):
                continue
            column = lookup[token_id]
            gradient = row_gradient_vector(
                current_all_caches[position], column, direction_basis
            ) * weight
            gradient_sums[token_id] = gradient if token_id not in gradient_sums else gradient_sums[token_id] + gradient
            contributors[token_id].append(
                {
                    "position": int(position),
                    "source_index": int(instances[position].source_index),
                    "pressure": weight,
                }
            )

    used_fallback = False
    if not gradient_sums:
        used_fallback = True
        for position in ordered_qas:
            weight = max(float(pressure_cpu[position]), 1e-8)
            unique_ids = list(dict.fromkeys(locked.answer_token_ids(tok, instances[position], max_length=max_length)))
            for token_id in unique_ids:
                token_id = int(token_id)
                if token_id in selected or token_id not in lookup:
                    continue
                column = lookup[token_id]
                gradient = row_gradient_vector(
                    current_all_caches[position], column, direction_basis
                ) * weight
                gradient_sums[token_id] = gradient if token_id not in gradient_sums else gradient_sums[token_id] + gradient
                contributors[token_id].append(
                    {
                        "position": int(position),
                        "source_index": int(instances[position].source_index),
                        "pressure": weight,
                    }
                )

    ranked: List[Tuple[float, int, int, Dict[str, Any]]] = []
    for token_id, gradient in gradient_sums.items():
        column = lookup[token_id]
        score = fisher_adjusted_score(gradient, all_fisher_diag[column], damping)
        report = {
            "token_id": int(token_id),
            "token": locked.decoded_token(tok, token_id),
            "fisher_adjusted_forget_leverage": score,
            "raw_gradient_norm": float(gradient.norm().detach().cpu()),
            "fisher_diagonal_mean": float(all_fisher_diag[column].mean().detach().cpu()),
            "document_frequency_in_50_forget_answers": int(df[token_id]),
            "contributors": contributors[token_id],
        }
        ranked.append((-score, int(df[token_id]), int(token_id), report))
    ranked.sort()
    chosen_reports = [item[3] for item in ranked[: int(rows_per_round)]]
    chosen_ids = [int(item["token_id"]) for item in chosen_reports]
    return chosen_ids, {
        "pressure_qas": ordered_qas,
        "pressure_values": [float(pressure_cpu[i]) for i in ordered_qas],
        "content_candidates": len(gradient_sums),
        "fallback_to_noncontent": used_fallback,
        "chosen": chosen_reports,
    }


def candidate_priority(metrics: Dict[str, Any], kl_value: float, norm: float) -> Tuple[int, float, float, float]:
    return (
        int(metrics["active_forget_instance_count"]),
        float(metrics["forget_answer_probability_max"]),
        float(kl_value),
        float(norm),
    )


def main() -> None:
    a = parse_args()
    if a.forget_num != 50:
        raise ValueError("V9 locked experiment fixes --forget-num=50")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.repair_rank < 0 or a.repair_steps <= 0 or a.max_promotion_rounds < 0:
        raise ValueError("invalid repair rank/step/round controls")
    if a.repair_lr <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("repair lr/batch-size/max-length must be positive")
    if a.delta_l2_lambda < 0 or a.fisher_weight < 0 or a.fisher_damping <= 0:
        raise ValueError("invalid preservation regularization controls")
    if a.dual_rho <= 0 or a.dual_lr <= 0 or a.dual_init < 0 or a.dual_max <= 0:
        raise ValueError("invalid augmented-Lagrangian controls")
    if a.dual_update_every <= 0 or a.post_feasible_steps < 0:
        raise ValueError("invalid dual/post-feasible controls")
    if a.promotion_rows_per_round <= 0 or a.promotion_qa_count <= 0:
        raise ValueError("promotion controls must be positive")
    if a.stage1_kl_budget_multiplier <= 0 or a.kl_budget_floor < 0:
        raise ValueError("invalid KL budget controls")
    if a.same_prompt_kl_budget is not None and a.same_prompt_kl_budget <= 0:
        raise ValueError("same-prompt-kl-budget must be positive")
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
    stage1_sensitive_ids = v8.load_stage1_sensitive_ids(model_path)
    if not set(stage1_sensitive_ids).issubset(set(all_answer_ids)):
        raise RuntimeError("Stage1 sensitive rows are not a subset of visible answer rows")
    all_lookup = {int(token_id): column for column, token_id in enumerate(all_answer_ids)}
    all_ids_tensor = torch.tensor(all_answer_ids, dtype=torch.long, device=output_weight.device)
    stage1_all_rows = output_weight.index_select(0, all_ids_tensor).detach().clone()

    base_input_rows, base_output_rows_cpu = old.load_reference_answer_rows(
        str(reference_path), all_answer_ids, a.dtype
    )
    base_output_rows = base_output_rows_cpu.to(device=output_weight.device, dtype=torch.float32)
    stage1_all_rows_fp32 = stage1_all_rows.detach().float()
    stage1_delta_all = stage1_all_rows_fp32 - base_output_rows

    stage1_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    target_nll = -math.log(a.target_forget_answer_probability)
    required_nll = stage1_nll.new_full(stage1_nll.shape, target_nll)
    initial_active = v8.residual_positions(stage1_nll, target_nll, a.comparison_tolerance)
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

    # Build exact protected-Base caches once. Stage1 changed only visible answer
    # rows, so restoring all of those rows reconstructs the protected output
    # layer while transformer/input embeddings remain exactly Base.
    set_rows_exact(output_weight, all_answer_ids, base_output_rows)
    base_all_caches = tofu.build_answer_delta_caches(
        model, tok, instances, all_answer_ids, device,
        batch_size=a.batch_size, max_length=a.max_length,
    )
    set_rows_exact(output_weight, all_answer_ids, stage1_all_rows)

    direction_basis = v8.hidden_basis_from_all_50(base_all_caches, a.repair_rank)
    actual_rank = 0 if direction_basis is None else int(direction_basis.shape[0])
    all_fisher_diag = diagonal_non_target_fisher(base_all_caches, direction_basis)

    stage1_base_caches = subset_answer_caches(base_all_caches, all_answer_ids, stage1_sensitive_ids)
    stage1_columns = torch.tensor(
        [all_lookup[token_id] for token_id in stage1_sensitive_ids],
        dtype=torch.long,
        device=device,
    )
    stage1_delta_sensitive = stage1_delta_all.index_select(0, stage1_columns)
    stage1_kl = float(
        equal_example_base_non_target_kl(stage1_base_caches, stage1_delta_sensitive)
        .detach().cpu()
    )
    kl_budget = (
        float(a.same_prompt_kl_budget)
        if a.same_prompt_kl_budget is not None
        else max(float(a.kl_budget_floor), float(a.stage1_kl_budget_multiplier) * stage1_kl)
    )

    total_rounds = a.max_promotion_rounds + 1
    round_step_budget = max(1, int(math.ceil(a.repair_steps / float(total_rounds))))
    selected_ids = sorted(stage1_sensitive_ids)
    dual = torch.full(
        (len(instances),), float(a.dual_init), dtype=torch.float32, device=device
    )
    round_reports: List[Dict[str, Any]] = []
    promotion_reports: List[Dict[str, Any]] = []
    success = len(initial_active) == 0 and stage1_kl <= kl_budget
    final_nll = stage1_nll
    final_metrics = stage1_metrics
    final_kl = stage1_kl
    final_fisher = 0.0
    final_selected_safety: float | str | None = None
    final_correction_all = torch.zeros_like(stage1_delta_all)

    print(
        f"===== V9 MIN-COLLATERAL rank={a.repair_rank} actual_rank={actual_rank} "
        f"stage1_rows={len(stage1_sensitive_ids)} initial_active={len(initial_active)} "
        f"stage1_KL={stage1_kl:.6g} KL_budget={kl_budget:.6g} "
        f"steps_per_round={round_step_budget} ====="
    )

    for round_index in range(total_rounds):
        if success:
            break

        set_rows_exact(output_weight, all_answer_ids, stage1_all_rows)
        selected_columns = torch.tensor(
            [all_lookup[token_id] for token_id in selected_ids],
            dtype=torch.long,
            device=device,
        )
        stage1_selected_rows = stage1_all_rows_fp32.index_select(0, selected_columns)
        base_selected_rows = base_output_rows.index_select(0, selected_columns)
        stage1_delta_selected = stage1_selected_rows - base_selected_rows

        stage1_caches = tofu.build_answer_delta_caches(
            model, tok, instances, selected_ids, device,
            batch_size=a.batch_size, max_length=a.max_length,
        )
        stage1_packed = tofu.pack_answer_delta_caches(stage1_caches)
        base_caches = subset_answer_caches(base_all_caches, all_answer_ids, selected_ids)
        fisher_diag = all_fisher_diag.index_select(0, selected_columns)

        module = active.SelectedRowDelta(
            len(selected_ids), int(output_weight.shape[1]),
            direction_basis=direction_basis, retained_basis=None, device=device,
        )
        optimizer = active.make_repair_optimizer(module, a.repair_optimizer, a.repair_lr)
        best_any_correction = module.effective_delta().detach().clone()
        best_any_nll = tofu.answer_nlls_from_packed_delta_cache(stage1_packed, best_any_correction).detach()
        best_any_kl = float(
            equal_example_base_non_target_kl(
                base_caches, stage1_delta_selected + best_any_correction
            ).detach().cpu()
        )
        best_any_metrics = locked.metrics(
            best_any_nll, required_nll, best_any_correction,
            target_nll=target_nll, tolerance=a.comparison_tolerance,
        )
        best_feasible_correction: torch.Tensor | None = None
        best_feasible_kl = float("inf")
        best_feasible_fisher = float("inf")
        best_feasible_metrics: Dict[str, Any] | None = None
        first_feasible_step: int | None = None
        logs: List[Dict[str, Any]] = []

        for step in range(1, round_step_budget + 1):
            optimizer.zero_grad(set_to_none=True)
            correction = module.effective_delta()
            current_nll = tofu.answer_nlls_from_packed_delta_cache(stage1_packed, correction)
            total_delta = stage1_delta_selected + correction
            kl = equal_example_base_non_target_kl(base_caches, total_delta)
            fisher_cost = fisher_quadratic(
                module, stage1_delta_selected, fisher_diag, direction_basis
            )
            errors = torch.relu(required_nll.to(current_nll) - current_nll)
            lagrangian = (dual * errors).mean()
            augmented = 0.5 * float(a.dual_rho) * errors.square().mean()
            l2 = correction.square().sum()
            loss = (
                kl
                + float(a.fisher_weight) * fisher_cost
                + float(a.delta_l2_lambda) * l2
                + lagrangian
                + augmented
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite V9 loss round={round_index} step={step}"
                )
            loss.backward()
            if a.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), a.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError("non-finite V9 gradient norm")
            optimizer.step()

            if step % a.dual_update_every == 0:
                with torch.no_grad():
                    dual.add_(float(a.dual_lr) * errors.detach()).clamp_(0.0, float(a.dual_max))

            with torch.no_grad():
                candidate = module.effective_delta().detach().clone()
                candidate_nll = tofu.answer_nlls_from_packed_delta_cache(stage1_packed, candidate)
                candidate_total = stage1_delta_selected + candidate
                candidate_kl = float(
                    equal_example_base_non_target_kl(base_caches, candidate_total)
                    .detach().cpu()
                )
                candidate_fisher = float(
                    fisher_quadratic(module, stage1_delta_selected, fisher_diag, direction_basis)
                    .detach().cpu()
                )
                candidate_metrics = locked.metrics(
                    candidate_nll, required_nll, candidate,
                    target_nll=target_nll, tolerance=a.comparison_tolerance,
                )
                candidate_norm = float(candidate.norm().detach().cpu())

                if candidate_priority(candidate_metrics, candidate_kl, candidate_norm) < candidate_priority(
                    best_any_metrics, best_any_kl, float(best_any_correction.norm().detach().cpu())
                ):
                    best_any_correction = candidate
                    best_any_nll = candidate_nll.detach().clone()
                    best_any_kl = candidate_kl
                    best_any_metrics = dict(candidate_metrics)

                is_feasible = v8.feasible(candidate_nll, required_nll, a.comparison_tolerance)
                if is_feasible:
                    if first_feasible_step is None:
                        first_feasible_step = step
                    feasible_key = (candidate_kl, candidate_fisher, candidate_norm)
                    current_key = (
                        best_feasible_kl,
                        best_feasible_fisher,
                        float("inf") if best_feasible_correction is None else float(best_feasible_correction.norm().detach().cpu()),
                    )
                    if feasible_key < current_key:
                        best_feasible_correction = candidate
                        best_feasible_kl = candidate_kl
                        best_feasible_fisher = candidate_fisher
                        best_feasible_metrics = dict(candidate_metrics)

            if step == 1 or step % a.log_every == 0 or step == round_step_budget:
                logs.append(
                    {
                        "round": round_index,
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "base_non_target_kl": float(kl.detach().cpu()),
                        "fisher_quadratic": float(fisher_cost.detach().cpu()),
                        "lagrangian": float(lagrangian.detach().cpu()),
                        "augmented_constraint": float(augmented.detach().cpu()),
                        "delta_l2": float(l2.detach().cpu()),
                        "dual_mean": float(dual.mean().detach().cpu()),
                        "dual_max": float(dual.max().detach().cpu()),
                        **candidate_metrics,
                    }
                )

            if (
                first_feasible_step is not None
                and step >= first_feasible_step + a.post_feasible_steps
            ):
                break

        del optimizer
        write_jsonl(root / f"round_{round_index:02d}" / "repair_log.jsonl", logs)

        chosen_correction = (
            best_feasible_correction
            if best_feasible_correction is not None
            else best_any_correction
        )
        chosen_cached_feasible = best_feasible_correction is not None
        materialization_trials: List[Dict[str, Any]] = []
        selected_safety: float | str | None = None

        # If cached-feasible, contract the correction toward Stage1 until it is
        # just safely feasible. This removes avoidable Stage2 displacement.
        if chosen_cached_feasible:
            candidate_pool: List[Tuple[torch.Tensor, float | str]] = []
            zero = torch.zeros_like(chosen_correction)
            for safety in sorted(set(float(x) for x in a.materialization_safety_fractions)):
                candidate, cached_nll, alpha = boundary.boundary_bisect(
                    stage1_packed,
                    zero,
                    chosen_correction,
                    required_nll,
                    tolerance=a.comparison_tolerance,
                    iterations=a.boundary_bisection_steps,
                    safety_fraction=safety,
                )
                candidate_pool.append((candidate.detach().clone(), safety))
                materialization_trials.append(
                    {
                        "safety_fraction": safety,
                        "boundary_alpha": alpha,
                        "cached_minimum_nll_slack": float(
                            (cached_nll - required_nll.to(cached_nll)).min().detach().cpu()
                        ),
                    }
                )
            candidate_pool.append((chosen_correction.detach().clone(), "best_feasible_endpoint"))
        else:
            candidate_pool = [(chosen_correction.detach().clone(), "best_infeasible_endpoint")]

        materialized_pass = False
        materialized_correction = chosen_correction.detach().clone()
        materialized_nll = best_any_nll
        materialized_metrics = best_any_metrics
        materialized_kl = best_any_kl
        materialized_fisher = float("inf")
        for candidate, safety in candidate_pool:
            set_rows_exact(output_weight, all_answer_ids, stage1_all_rows)
            tofu.set_selected_lm_head_rows(
                output_weight,
                selected_ids,
                stage1_selected_rows,
                candidate,
            )
            actual_nll = tofu.score_answer_instances(
                model, tok, instances, device,
                batch_size=a.batch_size, max_length=a.max_length,
            ).detach().float()
            actual_metrics = locked.metrics(
                actual_nll, required_nll, candidate.to(device),
                target_nll=target_nll, tolerance=a.comparison_tolerance,
            )
            current_selected_rows = output_weight.index_select(
                0, torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
            ).detach().float()
            actual_correction = current_selected_rows - stage1_selected_rows
            actual_total = stage1_delta_selected + actual_correction
            actual_kl = float(
                equal_example_base_non_target_kl(base_caches, actual_total)
                .detach().cpu()
            )
            passed = v8.feasible(actual_nll, required_nll, a.comparison_tolerance)
            trial = {
                "safety_fraction": safety,
                "materialized_pass": bool(passed),
                "materialized_active_count": actual_metrics["active_forget_instance_count"],
                "materialized_max_probability": actual_metrics["forget_answer_probability_max"],
                "base_non_target_kl": actual_kl,
                "kl_budget": kl_budget,
                "within_kl_budget": bool(actual_kl <= kl_budget),
                "stage2_delta_norm": float(actual_correction.norm().detach().cpu()),
            }
            materialization_trials.append(trial)
            if passed:
                materialized_pass = True
                materialized_correction = actual_correction.detach().clone()
                materialized_nll = actual_nll
                materialized_metrics = actual_metrics
                materialized_kl = actual_kl
                selected_safety = safety
                break
            if not chosen_cached_feasible:
                materialized_correction = actual_correction.detach().clone()
                materialized_nll = actual_nll
                materialized_metrics = actual_metrics
                materialized_kl = actual_kl

        # Fisher report at the actual materialized point.
        if direction_basis is None:
            total_coordinates = stage1_delta_selected + materialized_correction
        else:
            total_coordinates = (
                stage1_delta_selected @ direction_basis.transpose(0, 1)
                + materialized_correction @ direction_basis.transpose(0, 1)
            )
        materialized_fisher = float(
            ((fisher_diag * total_coordinates.square()).sum() / float(max(1, len(selected_ids))))
            .detach().cpu()
        )

        within_budget = materialized_pass and materialized_kl <= kl_budget
        current_residual = v8.residual_positions(
            materialized_nll, target_nll, a.comparison_tolerance
        )
        round_report = {
            "round": round_index,
            "selected_row_count": len(selected_ids),
            "selected_token_ids": list(selected_ids),
            "optimizer_step_budget": round_step_budget,
            "first_cached_feasible_step": first_feasible_step,
            "cached_feasible_found": best_feasible_correction is not None,
            "best_cached_feasible_kl": None if best_feasible_correction is None else best_feasible_kl,
            "materialized_pass": materialized_pass,
            "materialized_metrics": materialized_metrics,
            "materialized_residual_positions": current_residual,
            "materialized_base_non_target_kl": materialized_kl,
            "same_prompt_kl_budget": kl_budget,
            "within_kl_budget": within_budget,
            "materialized_fisher_quadratic": materialized_fisher,
            "materialized_stage2_delta_norm": float(materialized_correction.norm().detach().cpu()),
            "selected_materialization_safety_fraction": selected_safety,
            "materialization_trials": materialization_trials,
            "dual_mean": float(dual.mean().detach().cpu()),
            "dual_max": float(dual.max().detach().cpu()),
        }
        round_reports.append(round_report)
        write_json(root / f"round_{round_index:02d}" / "summary.json", round_report)
        print(
            f"V9 round={round_index} rows={len(selected_ids)} "
            f"pass={materialized_pass} active={len(current_residual)} "
            f"max_prob={materialized_metrics['forget_answer_probability_max']:.8g} "
            f"KL={materialized_kl:.6g}/{kl_budget:.6g} "
            f"stage2_norm={float(materialized_correction.norm().detach().cpu()):.6g}"
        )

        if within_budget:
            success = True
            final_nll = materialized_nll
            final_metrics = materialized_metrics
            final_kl = materialized_kl
            final_fisher = materialized_fisher
            final_selected_safety = selected_safety
            final_correction_all.zero_()
            final_correction_all.index_copy_(0, selected_columns, materialized_correction)
            break

        if round_index >= a.max_promotion_rounds:
            break

        # Promotion pressure is exactly the derivative multiplier of the
        # deletion constraints: alpha_i + rho * current_violation_i.
        errors_now = torch.relu(required_nll.to(materialized_nll) - materialized_nll)
        pressure = dual + float(a.dual_rho) * errors_now
        current_all_caches = tofu.build_answer_delta_caches(
            model, tok, instances, all_answer_ids, device,
            batch_size=a.batch_size, max_length=a.max_length,
        )
        promoted, promotion_detail = select_fisher_safe_promotions(
            tok,
            instances,
            all_answer_ids,
            current_all_caches,
            all_fisher_diag,
            selected_ids,
            direction_basis,
            pressure,
            qa_count=a.promotion_qa_count,
            rows_per_round=a.promotion_rows_per_round,
            max_length=a.max_length,
            damping=a.fisher_damping,
        )
        promotion_payload = {
            "after_round": round_index,
            "reason": (
                "deletion_infeasible"
                if not materialized_pass
                else "feasible_but_same_prompt_KL_budget_exceeded"
            ),
            "promoted_unique_row_count": len(promoted),
            "promoted_token_ids": promoted,
            "promoted_tokens": {
                str(token_id): locked.decoded_token(tok, token_id) for token_id in promoted
            },
            **promotion_detail,
        }
        promotion_reports.append(promotion_payload)
        write_json(root / f"round_{round_index:02d}" / "promotions.json", promotion_payload)
        if not promoted:
            break
        selected_ids = sorted(set(selected_ids) | set(promoted))

    write_json(
        root / "promotion_rounds.json",
        {"rounds": round_reports, "promotions": promotion_reports},
    )

    if not success:
        set_rows_exact(output_weight, all_answer_ids, stage1_all_rows)
        write_json(
            root / "failure.json",
            {
                "status": "FAILED_MIN_COLLATERAL_TRUST_REGION",
                "repair_rank_requested": a.repair_rank,
                "repair_rank_actual": actual_rank,
                "stage1_base_non_target_kl": stage1_kl,
                "same_prompt_kl_budget": kl_budget,
                "max_promotion_rounds": a.max_promotion_rounds,
                "final_selected_row_count": len(selected_ids),
                "rounds": round_reports,
                "note": "No retain/heldout metric was used to declare failure or choose rows.",
            },
        )
        raise RuntimeError(
            "V9 could not satisfy all-50 deletion inside the leakage-free same-prompt KL trust budget"
        )

    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V9 Stage2 modified input embeddings")
    if not v8.feasible(final_nll, required_nll, a.comparison_tolerance):
        raise RuntimeError("V9 final all-50 direct-forget audit failed")

    current_output_rows = output_weight.index_select(0, all_ids_tensor).detach().float().cpu()
    current_input_rows = input_weight.index_select(0, all_ids_tensor.to(input_weight.device)).detach().float().cpu()
    base_output_cpu = base_output_rows.detach().float().cpu()
    base_input_cpu = base_input_rows.detach().float().cpu()
    stage1_output_cpu = stage1_all_rows.detach().float().cpu()
    total_answer_delta = float((current_output_rows - base_output_cpu).norm().item())
    incremental_stage2_delta = float((current_output_rows - stage1_output_cpu).norm().item())
    input_base_error = float((current_input_rows - base_input_cpu).abs().max().item()) if current_input_rows.numel() else 0.0
    if input_base_error != 0.0:
        raise RuntimeError(f"V9 input embeddings are not exact Base: {input_base_error}")

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
            "unrestricted progressively selected LM-head correction"
            if a.repair_rank == 0
            else "progressively selected LM-head correction restricted to all-50 forget-hidden basis"
        ),
        "initially_active_forget_instance_count": len(initial_active),
        "initially_active_positions": initial_active,
        "initial_stage1_sensitive_lm_head_row_count": len(stage1_sensitive_ids),
        "selected_active_lm_head_row_count": len(selected_ids),
        "selected_active_lm_head_token_ids": selected_ids,
        "promotion_round_count_used": max(0, len(round_reports) - 1),
        "optimizer_total_step_budget": a.repair_steps,
        "optimizer_step_budget_per_round": round_step_budget,
        "incremental_stage2_delta_norm": incremental_stage2_delta,
        "total_visible_answer_lm_head_delta_norm_from_base": total_answer_delta,
        "all_visible_answer_input_base_max_abs_error": input_base_error,
        "stage1_base_non_target_kl": stage1_kl,
        "same_prompt_non_target_kl_from_base": final_kl,
        "same_prompt_non_target_kl_budget": kl_budget,
        "final_fisher_quadratic": final_fisher,
        "selected_materialization_safety_fraction": final_selected_safety,
        "baseline_stage1_metrics": stage1_metrics,
        "materialized_metrics": final_metrics,
        "row_policy": (
            "Stage1 sensitive rows plus Fisher-adjusted forget-leverage promotions only when deletion "
            "is infeasible or feasible only outside the same-prompt KL trust budget"
        ),
        "objective": (
            "Base same-prompt non-target KL + diagonal-Fisher cost + L2 + per-QA augmented-Lagrangian deletion constraints"
        ),
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
            "derived_same_prompt_kl_budget": kl_budget,
            "data_firewall": (
                "selection, Base-KL preservation, Fisher geometry, dual updates, promotion, optimization, "
                "and stopping use only the same 50 direct forget QAs"
            ),
        },
    )

    print("===== SURE-TOFU V9 MINIMUM-COLLATERAL STAGE2 PASS =====")
    print(
        f"rank={a.repair_rank} actual_rank={actual_rank} "
        f"stage1_rows={len(stage1_sensitive_ids)} final_rows={len(selected_ids)} "
        f"promotion_rounds={max(0, len(round_reports)-1)} "
        f"stage2_norm={incremental_stage2_delta:.6g} total_base_norm={total_answer_delta:.6g} "
        f"KL={final_kl:.6g}/{kl_budget:.6g} "
        f"max_prob={final_metrics['forget_answer_probability_max']:.8g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
