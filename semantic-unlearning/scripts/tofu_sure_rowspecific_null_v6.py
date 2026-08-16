#!/usr/bin/env python3
"""SURE-TOFU V6: Base-anchored row-specific forget subspaces with prompt nulling.

V6 makes the primary LM-head parameter the entire final displacement from the
protected Full-TOFU Base, not an increment on top of unrestricted Stage1A rows.
Each selected vocabulary row gets its own basis from answer positions where
that row is the true target. Components in the same 50 QAs' protected prompt-
hidden span are removed before the row basis is built. The final prompt hidden
position is excluded from the protected span because it predicts the first
answer token and therefore carries legitimate deletion signal.

Stage1A is used only as an initialization source: its selected-row displacement
from Base is projected into each row's safe basis. The primary objective uses
all 50 direct-forget constraints, same-prompt non-target KL to Base, and L2 on
the total displacement from Base. The existing V5 residual active-pair Stage2
is retained as a fail-closed exact-feasibility correction.

No retain95, paraphrase, same-author holdout, real-authors, world-facts, PPL,
or locked final metric is loaded or used for training or candidate selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_active_forget_repair as native
import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_rank0_forget as old
import tofu_sure_rank0_forget_restored as v2
import tofu_sure_rank0_forget_progressive_v3 as v3
import tofu_sure_r512_activepair_v5 as v5


METHOD = "SURE-TOFU-BaseAnchored-RowSpecific-PromptNull-v6"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"
DEFAULT_STAGE2_SAFETY = (0.002, 0.01, 0.02, 0.05, 0.10)


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

    p.add_argument("--primary-steps", type=int, default=10000)
    p.add_argument("--primary-lr", type=float, default=5e-3)
    p.add_argument("--primary-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--same-prompt-kl-weight", type=float, default=10.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--primary-boundary-bisection-steps", type=int, default=30)
    p.add_argument("--primary-boundary-safety-fraction", type=float, default=0.002)
    p.add_argument(
        "--prompt-null-max-rank",
        type=int,
        default=0,
        help="0 uses the full numerical prompt-hidden rank; positive values cap it.",
    )

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
        default=DEFAULT_STAGE2_SAFETY,
    )

    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def validate(a: argparse.Namespace) -> None:
    if a.forget_num != 50:
        raise ValueError("V6 locked diagnostic fixes --forget-num=50")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if abs(float(a.target_nll_buffer)) > 1e-12:
        raise ValueError("V6 fixes --target-nll-buffer=0")
    for name in (
        "initial_rows_per_example", "primary_steps", "stage2_steps", "batch_size",
        "max_length", "log_every", "primary_boundary_bisection_steps",
        "stage2_boundary_bisection_steps",
    ):
        if int(getattr(a, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if a.prompt_null_max_rank < 0:
        raise ValueError("--prompt-null-max-rank must be non-negative")
    for name in ("primary_lr", "stage2_lr"):
        value = float(getattr(a, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in (
        "forget_hinge_weight", "hardest_forget_hinge_weight", "same_prompt_kl_weight",
        "stage2_forget_hinge_weight", "stage2_hardest_forget_hinge_weight",
    ):
        value = float(getattr(a, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    for name in ("delta_l2_lambda", "stage2_l2", "comparison_tolerance"):
        value = float(getattr(a, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if not 0.0 <= a.primary_boundary_safety_fraction <= 0.1:
        raise ValueError("primary boundary safety fraction must lie in [0,0.1]")
    if any(x < 0 or x > 0.5 for x in a.stage2_materialization_safety_fractions):
        raise ValueError("stage2 materialization safety fractions must lie in [0,0.5]")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


@torch.no_grad()
def restore_visible_answer_rows_to_base(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    answer_ids: Sequence[int],
    base_input_rows: torch.Tensor,
    base_output_rows: torch.Tensor,
) -> None:
    ids_i = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    ids_o = ids_i.to(output_weight.device)
    input_weight.index_copy_(
        0, ids_i, base_input_rows.to(device=input_weight.device, dtype=input_weight.dtype)
    )
    output_weight.index_copy_(
        0, ids_o, base_output_rows.to(device=output_weight.device, dtype=output_weight.dtype)
    )


@torch.no_grad()
def protected_prompt_hidden_rows(
    model: nn.Module,
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    device: torch.device,
    *,
    max_length: int,
) -> torch.Tensor:
    """Collect prompt hidden states except the final answer-prediction position."""
    rows: List[torch.Tensor] = []
    for instance in instances:
        encoded = tok(
            instance.prompt,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1][0].float()
        length = int(encoded["attention_mask"][0].sum().item())
        protected_length = max(length - 1, 0)
        if protected_length:
            rows.append(hidden[:protected_length].detach())
        del output
    if not rows:
        hidden_size = model.get_output_embeddings().weight.shape[1]
        return torch.empty((0, hidden_size), dtype=torch.float32, device=device)
    return torch.cat(rows, dim=0)


def row_target_hidden_rows(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    selected_ids: Sequence[int],
) -> List[torch.Tensor]:
    buckets: List[List[torch.Tensor]] = [[] for _ in selected_ids]
    for cache in caches:
        columns = cache.target_selected_columns
        for column in range(len(selected_ids)):
            mask = columns.eq(column)
            if mask.any():
                buckets[column].append(cache.hidden[mask].float())
    hidden_size = caches[0].hidden.shape[-1] if caches else 0
    device = caches[0].hidden.device if caches else torch.device("cpu")
    return [
        torch.cat(parts, dim=0)
        if parts
        else torch.empty((0, hidden_size), dtype=torch.float32, device=device)
        for parts in buckets
    ]


def build_row_specific_bases(
    target_rows_by_selected: Sequence[torch.Tensor],
    prompt_basis: torch.Tensor,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for row_index, rows in enumerate(target_rows_by_selected):
        residual = active.project_rows_away(rows.float(), prompt_basis) if rows.numel() else rows
        basis = (
            active.orthonormal_row_basis(residual)
            if residual.numel()
            else rows.new_empty((0, rows.shape[-1]), dtype=torch.float32)
        )
        raw_norm = float(rows.norm().detach().cpu()) if rows.numel() else 0.0
        residual_norm = float(residual.norm().detach().cpu()) if residual.numel() else 0.0
        bases.append(basis)
        reports.append(
            {
                "selected_row_index": row_index,
                "target_position_count": int(rows.shape[0]),
                "raw_target_hidden_norm": raw_norm,
                "prompt_nulled_hidden_norm": residual_norm,
                "retained_hidden_fraction": residual_norm / raw_norm if raw_norm > 0 else 0.0,
                "row_specific_basis_rank": int(basis.shape[0]),
            }
        )
    return bases, reports


class RowSpecificBaseDelta(nn.Module):
    """Ragged row-specific bases packed into a padded coefficient tensor."""

    def __init__(
        self,
        bases: Sequence[torch.Tensor],
        *,
        initial_delta: torch.Tensor,
        device: torch.device,
    ) -> None:
        super().__init__()
        if not bases:
            raise ValueError("row-specific basis list must be non-empty")
        n_rows = len(bases)
        hidden_size = int(initial_delta.shape[1])
        ranks = [int(basis.shape[0]) for basis in bases]
        max_rank = max(max(ranks, default=0), 1)
        stacked = torch.zeros((n_rows, max_rank, hidden_size), dtype=torch.float32, device=device)
        mask = torch.zeros((n_rows, max_rank), dtype=torch.float32, device=device)
        coefficients = torch.zeros((n_rows, max_rank), dtype=torch.float32, device=device)
        for row_index, basis in enumerate(bases):
            rank = int(basis.shape[0])
            if rank <= 0:
                continue
            b = basis.to(device=device, dtype=torch.float32)
            stacked[row_index, :rank].copy_(b)
            mask[row_index, :rank] = 1.0
            coefficients[row_index, :rank] = (
                initial_delta[row_index].to(device=device, dtype=torch.float32)
                @ b.transpose(0, 1)
            )
        self.register_buffer("bases", stacked)
        self.register_buffer("coefficient_mask", mask)
        self.coefficients = nn.Parameter(coefficients)
        self.ranks = ranks

    def effective_delta(self) -> torch.Tensor:
        return torch.einsum(
            "nr,nrd->nd", self.coefficients * self.coefficient_mask, self.bases
        )


def same_prompt_non_target_kl(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    """Exact KL(Base_non-target || Current_non-target) for selected-row edits."""
    values: List[torch.Tensor] = []
    tiny = torch.finfo(torch.float32).tiny
    for cache in caches:
        corrections = cache.hidden.float() @ delta_rows.transpose(0, 1)
        q_selected = cache.selected_probs.float()
        q_target = torch.exp(-cache.base_token_nll.float()).clamp(max=1.0 - 1e-7)
        z = 1.0 + (q_selected * (torch.exp(corrections) - 1.0)).sum(dim=-1)

        target_multiplier = torch.ones_like(q_target)
        target_correction = torch.zeros_like(q_target)
        selected_mask = cache.target_selected_columns.ge(0)
        if selected_mask.any():
            row_idx = selected_mask.nonzero(as_tuple=False).flatten()
            col_idx = cache.target_selected_columns[selected_mask]
            c = corrections[row_idx, col_idx]
            target_multiplier[selected_mask] = torch.exp(c)
            target_correction[selected_mask] = c

        z_without_target = (z - q_target * target_multiplier).clamp_min(tiny)
        base_without_target = (1.0 - q_target).clamp_min(tiny)
        expected_c_num = (q_selected * corrections).sum(dim=-1)
        if selected_mask.any():
            expected_c_num[selected_mask] = (
                expected_c_num[selected_mask]
                - q_target[selected_mask] * target_correction[selected_mask]
            )
        kl = (
            torch.log(z_without_target)
            - torch.log(base_without_target)
            - expected_c_num / base_without_target
        )
        values.append(kl.clamp_min(0.0))
    return torch.cat(values).mean() if values else delta_rows.new_zeros(())


def primary_priority(
    metric: Dict[str, Any], kl_value: float
) -> Tuple[int, int, float, float, float]:
    return (
        int(metric["active_forget_instance_count"]),
        int(metric["buffered_forget_constraint_unmet_count"]),
        float(metric["forget_answer_probability_max"]),
        float(kl_value),
        float(metric["selected_lm_head_delta_norm"]),
    )


def optimize_primary(
    packed: Any,
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    module: RowSpecificBaseDelta,
    required_nll: torch.Tensor,
    target_nll: float,
    a: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    device = module.coefficients.device
    optimizer = active.make_repair_optimizer(module, a.primary_optimizer, a.primary_lr)
    zero = torch.zeros(
        (len(module.ranks), module.bases.shape[-1]), dtype=torch.float32, device=device
    )
    with torch.no_grad():
        initial_delta = module.effective_delta().detach().clone()
        initial_nll = tofu.answer_nlls_from_packed_delta_cache(packed, initial_delta)
        initial_kl = same_prompt_non_target_kl(caches, initial_delta)
        initial_metrics = v5.metrics(
            initial_nll, required_nll, initial_delta,
            target_nll=target_nll, tolerance=a.comparison_tolerance,
        )

    logs: List[Dict[str, Any]] = []
    best_delta = initial_delta.detach().clone()
    best_nll = initial_nll.detach().clone()
    best_kl = float(initial_kl.detach().cpu())
    best_metrics = dict(initial_metrics)
    crossing_step: int | None = None
    boundary_alpha: float | None = None

    zero_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero)
    if v5.feasible(zero_nll, required_nll, a.comparison_tolerance):
        del optimizer
        zero_kl = same_prompt_non_target_kl(caches, zero)
        zero_metrics = v5.metrics(
            zero_nll, required_nll, zero,
            target_nll=target_nll, tolerance=a.comparison_tolerance,
        )
        return zero, zero_nll, {
            **zero_metrics,
            "same_prompt_non_target_kl": float(zero_kl.detach().cpu()),
            "optimizer_crossing_step": None,
            "boundary_alpha": 0.0,
            "base_already_feasible": True,
        }, logs

    if v5.feasible(initial_nll, required_nll, a.comparison_tolerance):
        boundary_delta, boundary_nll, alpha = v2.boundary_bisect(
            packed, zero, initial_delta, required_nll,
            tolerance=a.comparison_tolerance,
            iterations=a.primary_boundary_bisection_steps,
            safety_fraction=a.primary_boundary_safety_fraction,
        )
        boundary_kl = same_prompt_non_target_kl(caches, boundary_delta)
        boundary_metrics = v5.metrics(
            boundary_nll, required_nll, boundary_delta,
            target_nll=target_nll, tolerance=a.comparison_tolerance,
        )
        del optimizer
        return boundary_delta.detach(), boundary_nll.detach(), {
            **boundary_metrics,
            "same_prompt_non_target_kl": float(boundary_kl.detach().cpu()),
            "optimizer_crossing_step": 0,
            "boundary_alpha": alpha,
            "base_already_feasible": False,
        }, logs

    previous_delta = initial_delta.detach().clone()
    for step in range(1, a.primary_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        current = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
        errors = torch.relu(required_nll.to(current) - current)
        kl = same_prompt_non_target_kl(caches, delta)
        loss = (
            a.forget_hinge_weight * errors.square().mean()
            + a.hardest_forget_hinge_weight * errors.square().max()
            + a.same_prompt_kl_weight * kl
            + a.delta_l2_lambda * delta.square().sum()
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V6 primary loss at step {step}")
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
            candidate_kl = float(same_prompt_non_target_kl(caches, candidate).detach().cpu())
            candidate_metrics = v5.metrics(
                candidate_nll, required_nll, candidate,
                target_nll=target_nll, tolerance=a.comparison_tolerance,
            )
            if primary_priority(candidate_metrics, candidate_kl) < primary_priority(
                best_metrics, best_kl
            ):
                best_delta = candidate.detach().clone()
                best_nll = candidate_nll.detach().clone()
                best_kl = candidate_kl
                best_metrics = dict(candidate_metrics)

            if step == 1 or step % a.log_every == 0:
                logs.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "forget_hinge": float(errors.square().mean().detach().cpu()),
                        "hardest_forget_hinge": float(errors.square().max().detach().cpu()),
                        "same_prompt_non_target_kl": candidate_kl,
                        "delta_l2": float(candidate.square().sum().detach().cpu()),
                        **candidate_metrics,
                    }
                )
                print(
                    f"v6-primary-step={step} active={candidate_metrics['active_forget_instance_count']} "
                    f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g} "
                    f"kl={candidate_kl:.6g} norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
                )

            if v5.feasible(candidate_nll, required_nll, a.comparison_tolerance):
                boundary_delta, boundary_nll, alpha = v2.boundary_bisect(
                    packed, previous_delta, candidate, required_nll,
                    tolerance=a.comparison_tolerance,
                    iterations=a.primary_boundary_bisection_steps,
                    safety_fraction=a.primary_boundary_safety_fraction,
                )
                best_delta = boundary_delta.detach().clone()
                best_nll = boundary_nll.detach().clone()
                best_kl = float(same_prompt_non_target_kl(caches, boundary_delta).detach().cpu())
                best_metrics = v5.metrics(
                    best_nll, required_nll, best_delta,
                    target_nll=target_nll, tolerance=a.comparison_tolerance,
                )
                crossing_step = step
                boundary_alpha = alpha
                break
            previous_delta = candidate.detach().clone()

    del optimizer
    return best_delta.detach(), best_nll.detach(), {
        **best_metrics,
        "same_prompt_non_target_kl": best_kl,
        "optimizer_crossing_step": crossing_step,
        "boundary_alpha": boundary_alpha,
        "base_already_feasible": False,
        "cached_feasible": v5.feasible(best_nll, required_nll, a.comparison_tolerance),
    }, logs


@torch.no_grad()
def current_base_delta_norm(
    output_weight: torch.Tensor,
    answer_ids: Sequence[int],
    modified_ids: Sequence[int],
    base_output_rows: torch.Tensor,
) -> float:
    if not modified_ids:
        return 0.0
    lookup = {int(token_id): pos for pos, token_id in enumerate(answer_ids)}
    positions = torch.tensor([lookup[int(i)] for i in modified_ids], dtype=torch.long)
    ids = torch.tensor(modified_ids, dtype=torch.long, device=output_weight.device)
    current = output_weight.index_select(0, ids).detach().float().cpu()
    base = base_output_rows.index_select(0, positions).detach().float().cpu()
    return float((current - base).norm().item())


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
    primary_ckpt = root / "primary" / "checkpoint"
    final_ckpt = root / "checkpoint"

    tok_data = AutoTokenizer.from_pretrained(a.model_path)
    if tok_data.pad_token is None:
        tok_data.pad_token = tok_data.eos_token
    instances, source_indices = locked.load_forget_instances(forget_path, tok_data, a.forget_num)

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    device = gagd.first_device(model)

    stage1a_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    required_nll, _, target_nll = native.build_required_forget_nll(
        stage1a_nll,
        target_probability=a.target_forget_answer_probability,
        target_nll_buffer=0.0,
    )
    locked.write_json(
        root / "forget_instances_stage1a_before_base_anchor.json",
        locked.report_instances(
            instances, stage1a_nll, required_nll, a.target_forget_answer_probability
        ),
    )

    answer_ids = old.all_answer_rows(tok, instances, max_length=a.max_length)
    if not answer_ids:
        raise RuntimeError("visible direct forget answers produced no answer-token rows")
    ids_o = torch.tensor(answer_ids, dtype=torch.long, device=output_weight.device)
    stage1a_output_rows = output_weight.index_select(0, ids_o).detach().cpu().clone()
    base_input_rows, base_output_rows = old.load_reference_answer_rows(
        str(reference_path), answer_ids, a.dtype
    )

    rankings, ranking_report = v3.build_progressive_rankings(
        tok, instances, max_length=a.max_length
    )
    selection_positions = list(range(len(instances)))
    sensitive_ids: set[int] = set()
    initial_by_position = v3.add_next_rows(
        sensitive_ids, rankings, selection_positions, a.initial_rows_per_example
    )
    if not sensitive_ids:
        raise RuntimeError("V6 selector produced no sensitive rows")
    selected_ids = sorted(sensitive_ids)
    nonselected_ids = sorted(set(answer_ids) - sensitive_ids)

    answer_lookup = {int(token_id): pos for pos, token_id in enumerate(answer_ids)}
    selected_positions = torch.tensor(
        [answer_lookup[token_id] for token_id in selected_ids], dtype=torch.long
    )
    stage1a_selected_rows = stage1a_output_rows.index_select(0, selected_positions).float()
    base_selected_rows_cpu = base_output_rows.index_select(0, selected_positions).float()
    stage1a_selected_delta = stage1a_selected_rows - base_selected_rows_cpu

    restore_visible_answer_rows_to_base(
        input_weight, output_weight, answer_ids, base_input_rows, base_output_rows
    )
    input_error_base = v5.all_input_base_error(input_weight, answer_ids, base_input_rows)
    output_error_base = old.answer_row_restoration_error(
        output_weight, answer_ids, answer_ids, base_output_rows
    )
    if input_error_base != 0.0 or output_error_base != 0.0:
        raise RuntimeError(
            f"V6 Base anchoring failed: input={input_error_base} output={output_error_base}"
        )

    primary_caches = tofu.build_answer_delta_caches(
        model, tok, instances, selected_ids, device,
        batch_size=a.batch_size, max_length=a.max_length,
    )
    primary_packed = tofu.pack_answer_delta_caches(primary_caches)

    prompt_hidden = protected_prompt_hidden_rows(
        model, tok, instances, device, max_length=a.max_length
    )
    prompt_basis = active.orthonormal_row_basis(
        prompt_hidden,
        max_rank=(a.prompt_null_max_rank if a.prompt_null_max_rank > 0 else None),
    )
    target_rows = row_target_hidden_rows(primary_caches, selected_ids)
    row_bases, row_reports = build_row_specific_bases(target_rows, prompt_basis)
    module = RowSpecificBaseDelta(
        row_bases, initial_delta=stage1a_selected_delta.to(device), device=device
    )
    projected_stage1a_delta = module.effective_delta().detach().clone()
    removed_stage1a_component = stage1a_selected_delta.to(device) - projected_stage1a_delta

    ranks = [int(b.shape[0]) for b in row_bases]
    for row_index, (token_id, report) in enumerate(zip(selected_ids, row_reports)):
        report["token_id"] = int(token_id)
        report["token"] = locked.decoded_token(tok, token_id)
        report["stage1a_row_delta_norm"] = float(
            stage1a_selected_delta[row_index].norm().detach().cpu()
        )
        report["projected_stage1a_row_delta_norm"] = float(
            projected_stage1a_delta[row_index].norm().detach().cpu()
        )
        report["removed_stage1a_row_component_norm"] = float(
            removed_stage1a_component[row_index].norm().detach().cpu()
        )

    total_coefficients = int(sum(ranks))
    print(
        "===== SURE-TOFU V6 PRIMARY ===== "
        f"selected_rows={len(selected_ids)} prompt_hidden_rows={prompt_hidden.shape[0]} "
        f"prompt_null_rank={prompt_basis.shape[0]} row_rank_min={min(ranks)} "
        f"row_rank_max={max(ranks)} coefficients={total_coefficients}"
    )
    print(
        f"Stage1A selected delta norm={stage1a_selected_delta.norm().item():.6g} "
        f"projected init norm={projected_stage1a_delta.norm().item():.6g} "
        f"removed component norm={removed_stage1a_component.norm().item():.6g}"
    )

    primary_delta, _primary_cached_nll, primary_cached_summary, primary_logs = optimize_primary(
        primary_packed, primary_caches, module, required_nll, target_nll, a
    )
    write_jsonl(root / "primary" / "repair_log.jsonl", primary_logs)

    base_selected_rows = base_selected_rows_cpu.to(
        device=output_weight.device, dtype=output_weight.dtype
    )
    v5.set_rows(output_weight, selected_ids, base_selected_rows, primary_delta)
    input_error_primary = v5.all_input_base_error(input_weight, answer_ids, base_input_rows)
    if input_error_primary != 0.0:
        raise RuntimeError(f"V6 primary modified input embeddings: {input_error_primary}")

    primary_materialized_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    primary_materialized_metrics = v5.metrics(
        primary_materialized_nll, required_nll, primary_delta.to(device),
        target_nll=target_nll, tolerance=a.comparison_tolerance,
    )
    primary_residual = old.violating_sequence_positions(
        primary_materialized_nll, required_nll, a.comparison_tolerance
    )

    primary_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(primary_ckpt)
    tok.save_pretrained(primary_ckpt)
    locked.write_json(
        root / "primary" / "forget_instances_after.json",
        locked.report_instances(
            instances, primary_materialized_nll, required_nll,
            a.target_forget_answer_probability,
        ),
    )
    write_json(
        root / "primary" / "summary.json",
        {
            "status": "PASS_TO_STAGE2" if primary_residual else "PASS_ALL_DIRECT",
            "selected_lm_head_row_count": len(selected_ids),
            "prompt_hidden_state_count": int(prompt_hidden.shape[0]),
            "prompt_null_basis_rank": int(prompt_basis.shape[0]),
            "prompt_null_excludes_final_prompt_prediction_position": True,
            "row_specific_basis_rank_min": int(min(ranks)),
            "row_specific_basis_rank_max": int(max(ranks)),
            "row_specific_basis_rank_mean": float(sum(ranks) / len(ranks)),
            "row_specific_basis_total_coefficients": total_coefficients,
            "stage1a_selected_rows_delta_norm_from_base": float(stage1a_selected_delta.norm().item()),
            "projected_stage1a_initial_delta_norm_from_base": float(projected_stage1a_delta.norm().detach().cpu()),
            "removed_stage1a_component_norm": float(removed_stage1a_component.norm().detach().cpu()),
            "primary_total_delta_norm_from_base": float(primary_delta.norm().detach().cpu()),
            "cached_summary": primary_cached_summary,
            "materialized_metrics": primary_materialized_metrics,
            "residual_violating_sequence_count": len(primary_residual),
            "residual_violating_positions": primary_residual,
            "all_visible_answer_input_base_max_abs_error": input_error_primary,
            "benchmark_retain_seen": 0,
            "heldout_or_paraphrase_seen": 0,
            "checkpoint": str(primary_ckpt.resolve()),
        },
    )
    write_json(root / "primary" / "row_basis_report.json", row_reports)

    all_answer_cache = tofu.build_answer_delta_caches(
        model, tok, instances, answer_ids, device,
        batch_size=a.batch_size, max_length=a.max_length,
    )
    stage2_ids, pair_rows = v5.active_pair_report(
        all_answer_cache, answer_ids, required_nll, primary_residual,
        a.comparison_tolerance, tok,
    )
    if primary_residual and not stage2_ids:
        stage2_ids = old.rows_for_positions(
            tok, instances, primary_residual, max_length=a.max_length
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
    if not primary_residual:
        stage2_summary = {
            "status": "SKIPPED_PRIMARY_ALREADY_FEASIBLE",
            "residual_violating_sequence_count_before": 0,
            "active_pair_count": 0,
            "active_pair_unique_row_count": 0,
            "active_pair_unique_token_ids": [],
            "stage2_delta_norm": 0.0,
            "selected_materialization_safety_fraction": None,
        }
        final_nll = primary_materialized_nll
        final_metrics = primary_materialized_metrics
    else:
        stage2_tensor = torch.tensor(stage2_ids, dtype=torch.long, device=output_weight.device)
        stage2_baseline_rows = output_weight.index_select(0, stage2_tensor).detach().clone()
        stage2_caches = tofu.build_answer_delta_caches(
            model, tok, instances, stage2_ids, device,
            batch_size=a.batch_size, max_length=a.max_length,
        )
        stage2_packed = tofu.pack_answer_delta_caches(stage2_caches)
        (
            _stage2_high_delta, _stage2_cached_nll, stage2_cached_metrics,
            stage2_logs, crossing_low, crossing_high,
        ) = v5.optimize_unrestricted_stage2(
            stage2_packed, len(stage2_ids), int(output_weight.shape[1]),
            required_nll, target_nll, a, device,
        )
        write_jsonl(root / "stage2_active_pairs" / "repair_log.jsonl", stage2_logs)

        selected_candidate: torch.Tensor | None = None
        selected_fraction: float | None = None
        materialization_trials: List[Dict[str, Any]] = []
        for safety in sorted(set(float(x) for x in a.stage2_materialization_safety_fractions)):
            candidate, candidate_cached_nll, alpha = v2.boundary_bisect(
                stage2_packed, crossing_low, crossing_high, required_nll,
                tolerance=a.comparison_tolerance,
                iterations=a.stage2_boundary_bisection_steps,
                safety_fraction=safety,
            )
            v5.set_rows(output_weight, stage2_ids, stage2_baseline_rows, candidate)
            actual_nll = tofu.score_answer_instances(
                model, tok, instances, device,
                batch_size=a.batch_size, max_length=a.max_length,
            ).detach().float()
            pass_actual = v5.feasible(actual_nll, required_nll, a.comparison_tolerance)
            actual_metrics = v5.metrics(
                actual_nll, required_nll, candidate.to(device),
                target_nll=target_nll, tolerance=a.comparison_tolerance,
            )
            materialization_trials.append(
                {
                    "safety_fraction": safety,
                    "boundary_alpha": alpha,
                    "cached_minimum_nll_slack": float(
                        (candidate_cached_nll - required_nll.to(candidate_cached_nll))
                        .min().detach().cpu()
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
            v5.set_rows(
                output_weight, stage2_ids, stage2_baseline_rows, torch.zeros_like(candidate)
            )

        if selected_candidate is None:
            v5.set_rows(output_weight, stage2_ids, stage2_baseline_rows, crossing_high)
            actual_nll = tofu.score_answer_instances(
                model, tok, instances, device,
                batch_size=a.batch_size, max_length=a.max_length,
            ).detach().float()
            actual_metrics = v5.metrics(
                actual_nll, required_nll, crossing_high.to(device),
                target_nll=target_nll, tolerance=a.comparison_tolerance,
            )
            materialization_trials.append(
                {
                    "safety_fraction": "optimizer_crossing_endpoint",
                    "boundary_alpha": 1.0,
                    "materialized_pass": v5.feasible(
                        actual_nll, required_nll, a.comparison_tolerance
                    ),
                    "materialized_active_count": actual_metrics["active_forget_instance_count"],
                    "materialized_max_probability": actual_metrics["forget_answer_probability_max"],
                    "delta_norm": float(crossing_high.norm().detach().cpu()),
                }
            )
            if not v5.feasible(actual_nll, required_nll, a.comparison_tolerance):
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
                raise RuntimeError("V6 Stage2 lost feasibility after BF16 materialization")
            selected_candidate = crossing_high.detach().clone()
            selected_fraction = -1.0
            final_nll = actual_nll
            final_metrics = actual_metrics

        stage2_delta = selected_candidate
        stage2_summary = {
            "status": "PASS",
            "residual_violating_sequence_count_before": len(primary_residual),
            "residual_violating_positions_before": primary_residual,
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
            "residual_violating_sequence_count_after_primary": len(primary_residual),
            "residual_violating_positions_after_primary": primary_residual,
            "active_pair_count": sum(1 for row in pair_rows if not row.get("fallback")),
            "active_pair_unique_row_count": len(stage2_ids),
            "active_pair_unique_token_ids": stage2_ids,
            "pairs": pair_rows,
            "retain_or_heldout_data_consulted": False,
        },
    )
    write_json(root / "stage2_active_pairs" / "summary.json", stage2_summary)

    if not v5.feasible(final_nll, required_nll, a.comparison_tolerance):
        raise RuntimeError("V6 final direct-forget audit is not feasible")

    input_error_final = v5.all_input_base_error(input_weight, answer_ids, base_input_rows)
    modified_output_ids = sorted(set(selected_ids) | set(stage2_ids))
    untouched_output_ids = sorted(set(answer_ids) - set(modified_output_ids))
    untouched_output_error = old.answer_row_restoration_error(
        output_weight, answer_ids, untouched_output_ids, base_output_rows
    )
    if input_error_final != 0.0 or untouched_output_error != 0.0:
        raise RuntimeError(
            f"V6 final restoration audit failed: all_input={input_error_final} "
            f"untouched_output={untouched_output_error}"
        )

    total_final_delta_norm = current_base_delta_norm(
        output_weight, answer_ids, modified_output_ids, base_output_rows
    )

    final_ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_ckpt)
    tok.save_pretrained(final_ckpt)
    locked.write_json(
        root / "forget_instances_after.json",
        locked.report_instances(
            instances, final_nll, required_nll, a.target_forget_answer_probability
        ),
    )

    write_json(
        root / "row_selection.json",
        {
            **ranking_report,
            "selection_positions": selection_positions,
            "initial_rows_per_example": a.initial_rows_per_example,
            "initial_selected_by_position": {
                str(k): v for k, v in sorted(initial_by_position.items())
            },
            "primary_selected_unique_row_count": len(selected_ids),
            "primary_selected_token_ids": selected_ids,
            "primary_selected_tokens": {
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
        "primary": {
            "selected_row_count": len(selected_ids),
            "prompt_hidden_state_count": int(prompt_hidden.shape[0]),
            "prompt_null_basis_rank": int(prompt_basis.shape[0]),
            "row_specific_basis_rank_min": int(min(ranks)),
            "row_specific_basis_rank_max": int(max(ranks)),
            "row_specific_basis_rank_mean": float(sum(ranks) / len(ranks)),
            "row_specific_basis_total_coefficients": total_coefficients,
            "stage1a_selected_rows_delta_norm_from_base": float(stage1a_selected_delta.norm().item()),
            "projected_stage1a_initial_delta_norm_from_base": float(projected_stage1a_delta.norm().detach().cpu()),
            "removed_stage1a_component_norm": float(removed_stage1a_component.norm().detach().cpu()),
            "total_delta_norm_from_base": float(primary_delta.norm().detach().cpu()),
            "same_prompt_non_target_kl": primary_cached_summary["same_prompt_non_target_kl"],
            "cached_summary": primary_cached_summary,
            "materialized_metrics": primary_materialized_metrics,
            "residual_violating_sequence_count": len(primary_residual),
        },
        "stage2": stage2_summary,
        "final": {
            "modified_output_row_union_count": len(modified_output_ids),
            "modified_output_token_ids": modified_output_ids,
            "total_output_delta_norm_from_base": total_final_delta_norm,
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
        "primary_checkpoint": str(primary_ckpt.resolve()),
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
                "Base-anchored total LM-head displacement restricted to row-specific "
                "target-position bases after prompt-span nulling; unrestricted Stage2 "
                "only on residual deficient active-pair rows"
            ),
            "stage1a_role": (
                "initialization only; selected Stage1A output-row displacement is "
                "projected into the row-specific safe bases"
            ),
            "same_prompt_kl_definition": (
                "KL(Base_non-target || Current_non-target) on the same 50 direct forget "
                "answer contexts, with the true target removed and renormalized"
            ),
            "data_firewall": (
                "selection/optimization use only the same 50 training-visible direct forget QAs"
            ),
        },
    )

    print("===== SURE-TOFU V6 PASS =====")
    print(
        f"Primary rows={len(selected_ids)} prompt_null_rank={prompt_basis.shape[0]} "
        f"row_rank_mean={sum(ranks)/len(ranks):.3f} coefficients={total_coefficients} "
        f"delta_from_base={float(primary_delta.norm().detach().cpu()):.6g} "
        f"KL={primary_cached_summary['same_prompt_non_target_kl']:.6g} "
        f"residual_qas={len(primary_residual)}"
    )
    print(
        f"Stage2 active_pairs={sum(1 for row in pair_rows if not row.get('fallback'))} "
        f"unique_rows={len(stage2_ids)} norm={float(stage2_delta.norm().detach().cpu()):.6g}"
    )
    print(
        f"Final active={final_metrics['active_forget_instance_count']} "
        f"max_prob={final_metrics['forget_answer_probability_max']:.8g} "
        f"total_delta_from_base={total_final_delta_norm:.6g}"
    )
    print("Primary checkpoint:", primary_ckpt)
    print("Final checkpoint:", final_ckpt)


if __name__ == "__main__":
    main()
