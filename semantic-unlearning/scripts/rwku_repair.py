#!/usr/bin/env python3
"""Sparse active-pair LM-head repair for the RWKU experiment.

The repair never edits EOT/EOS.  It finds calibration answer-token positions
that still violate the requested forget margin and may change only their
non-special LM-head rows.  Any answer-token row also used by the unrelated MCF
protection set is excluded.

The transformer and input embeddings stay frozen.  Selected row deltas are
projected away from a broad sample of protected prompt/answer hidden states,
and a materialized-dtype scale sweep enforces three hard protection gates:

* minimum protected-answer probability ratio,
* maximum protected-context selected-logit drift, and
* zero (by default) protected-context top-1 changes.

Scale zero is always present and is selected whenever no effective sparse edit
passes every protection gate.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

import gagd_active_case_repair as active_repair
from rwku_data import record_sha256
from rwku_eval import (
    _normalized_completion,
    _token_ids,
    final_hidden_states,
    format_qa_prompt,
    model_device,
)


@dataclass
class RepairPoint:
    hidden: torch.Tensor
    target_logit: torch.Tensor
    competitor_logit: torch.Tensor
    source_id: str
    token_index: int
    target_token_id: int
    competitor_token_id: int
    baseline_predicted_token_id: int


@dataclass
class AnswerDeltaCache:
    source_id: str
    hidden: torch.Tensor
    target_token_ids: torch.Tensor
    target_selected_indices: torch.Tensor
    repair_mask: torch.Tensor
    base_target_logits: torch.Tensor
    base_selected_logits: torch.Tensor
    base_nonselected_logsumexp: torch.Tensor
    base_nonselected_max: torch.Tensor


@dataclass
class ContextDeltaCache:
    source_id: str
    hidden: torch.Tensor
    base_selected_logits: torch.Tensor
    base_nonselected_max: torch.Tensor
    base_nonselected_token_ids: torch.Tensor
    baseline_predicted_token_ids: torch.Tensor


@dataclass(frozen=True)
class RepairConfig:
    steps: int = 800
    learning_rate: float = 5e-3
    active_margin: float = 0.25
    selection_margin: float = 0.05
    l2_lambda: float = 1e-6
    protected_logit_lambda: float = 1.0
    max_delta_norm: Optional[float] = None
    project_away_protected: bool = True
    protected_projection_rank: int = 256
    protected_contexts_per_example: int = 8
    exclude_protected_answer_rows: bool = True
    min_protected_probability_ratio: float = 0.999
    max_protected_logit_drift: float = 0.05
    max_protected_top1_changes: int = 0
    stop_when_satisfied: bool = True
    candidate_scales: Tuple[float, ...] = (
        1.0,
        0.875,
        0.75,
        0.625,
        0.5,
        0.375,
        0.25,
        0.1875,
        0.125,
        0.09375,
        0.0625,
        0.046875,
        0.03125,
        0.015625,
        0.0078125,
        0.0,
    )


def validate_config(config: RepairConfig) -> None:
    if config.steps <= 0 or config.learning_rate <= 0:
        raise ValueError("repair steps and learning rate must be positive")
    if config.active_margin < 0 or config.selection_margin < 0:
        raise ValueError("repair margins must be non-negative")
    if config.l2_lambda < 0 or config.protected_logit_lambda < 0:
        raise ValueError("repair regularization coefficients must be non-negative")
    if config.max_delta_norm is not None and config.max_delta_norm <= 0:
        raise ValueError("repair max_delta_norm must be positive")
    if config.protected_projection_rank < 0:
        raise ValueError("protected_projection_rank must be non-negative")
    if config.protected_contexts_per_example <= 0:
        raise ValueError("protected_contexts_per_example must be positive")
    if not 0.0 < config.min_protected_probability_ratio <= 1.0:
        raise ValueError("min_protected_probability_ratio must be in (0,1]")
    if config.max_protected_logit_drift < 0:
        raise ValueError("max_protected_logit_drift must be non-negative")
    if config.max_protected_top1_changes < 0:
        raise ValueError("max_protected_top1_changes must be non-negative")
    if not config.candidate_scales:
        raise ValueError("candidate_scales must not be empty")
    if any(
        not math.isfinite(scale) or not 0.0 <= scale <= 1.0
        for scale in config.candidate_scales
    ):
        raise ValueError("candidate scales must be finite values in [0,1]")
    if 0.0 not in config.candidate_scales:
        raise ValueError("candidate scales must include the no-op scale 0")


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _target_competitor(
    logits: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("Expected [positions, vocabulary] logits")
    if target_token_ids.shape != logits.shape[:1]:
        raise ValueError("Target token IDs must align with logit positions")
    top_values, top_ids = torch.topk(logits, k=2, dim=-1)
    target_is_top = top_ids[:, 0].eq(target_token_ids)
    competitor_values = torch.where(
        target_is_top,
        top_values[:, 1],
        top_values[:, 0],
    )
    competitor_ids = torch.where(
        target_is_top,
        top_ids[:, 1],
        top_ids[:, 0],
    )
    return competitor_values, competitor_ids


def _prompt_answer_tensors(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    answer: str,
    *,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    device = model_device(model)
    prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=True)
    answer_ids = _token_ids(
        tokenizer,
        _normalized_completion(answer),
        add_special_tokens=False,
    )
    if not answer_ids:
        return (
            torch.empty((0, 0), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            0,
        )
    allowed_prompt = max(1, max_length - len(answer_ids))
    prompt_ids = prompt_ids[-allowed_prompt:]
    sequence = prompt_ids + answer_ids
    input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
    hidden = final_hidden_states(model, input_ids=input_ids)[0]
    answer_positions = torch.arange(
        len(prompt_ids) - 1,
        len(prompt_ids) + len(answer_ids) - 1,
        device=device,
    )
    targets = torch.tensor(answer_ids, dtype=torch.long, device=device)
    return hidden, answer_positions, targets, len(prompt_ids)


def _logits_for_hidden(
    model: nn.Module,
    hidden: torch.Tensor,
) -> torch.Tensor:
    output_layer = model.get_output_embeddings()
    if output_layer is None:
        raise ValueError("Model does not expose output embeddings")
    native = hidden.to(
        device=output_layer.weight.device,
        dtype=output_layer.weight.dtype,
    )
    return output_layer(native).float()


@torch.no_grad()
def cache_prompt_answers(
    model: nn.Module,
    tokenizer: Any,
    prompt_answer_rows: Sequence[Tuple[str, str, str]],
    *,
    max_length: int = 4096,
) -> List[RepairPoint]:
    """Cache teacher-forced answer positions and their true competitors."""

    points: List[RepairPoint] = []
    for prompt, answer, source_id in prompt_answer_rows:
        hidden_all, positions, target_ids, _ = _prompt_answer_tensors(
            model,
            tokenizer,
            prompt,
            answer,
            max_length=max_length,
        )
        if not target_ids.numel():
            continue
        hidden_native = hidden_all.index_select(0, positions)
        logits = _logits_for_hidden(model, hidden_native)
        competitor_values, competitor_ids = _target_competitor(
            logits,
            target_ids,
        )
        target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
        predicted = logits.argmax(dim=-1)
        hidden = hidden_native.float()
        for token_index, target_token_id in enumerate(target_ids.tolist()):
            points.append(
                RepairPoint(
                    hidden=hidden[token_index].detach(),
                    target_logit=target_logits[token_index].detach(),
                    competitor_logit=competitor_values[token_index].detach(),
                    source_id=source_id,
                    token_index=token_index,
                    target_token_id=int(target_token_id),
                    competitor_token_id=int(competitor_ids[token_index]),
                    baseline_predicted_token_id=int(predicted[token_index]),
                )
            )
    return points


def calibration_prompt_answers(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
) -> List[Tuple[str, str, str]]:
    return [
        (
            format_qa_prompt(tokenizer, row),
            str(row["answer"]),
            str(row.get("view_id") or record_sha256(row)),
        )
        for row in rows
    ]


def protected_prompt_answers(
    examples: Sequence[Any],
) -> List[Tuple[str, str, str]]:
    output: List[Tuple[str, str, str]] = []
    for index, example in enumerate(examples):
        answer = str(getattr(example, "answer", ""))
        prompt = str(getattr(example, "prompt", ""))
        if answer and prompt:
            output.append((prompt, answer, f"retain:{index}"))
    return output


def active_points(
    points: Sequence[RepairPoint],
    *,
    required_margin: float,
) -> List[RepairPoint]:
    return [
        point
        for point in points
        if float(point.competitor_logit - point.target_logit) < required_margin
    ]


def select_active_token_ids(
    points: Sequence[RepairPoint],
    protected_points: Sequence[RepairPoint],
    *,
    special_token_ids: Sequence[int],
    exclude_protected_answer_rows: bool,
) -> Tuple[List[int], List[int]]:
    active_ids = {point.target_token_id for point in points}
    active_ids -= {int(token_id) for token_id in special_token_ids}
    protected_ids = {point.target_token_id for point in protected_points}
    overlap = sorted(active_ids & protected_ids)
    if exclude_protected_answer_rows:
        active_ids -= protected_ids
    return sorted(active_ids), overlap


def repair_excluded_token_ids(tokenizer: Any) -> List[int]:
    """Return every tokenizer special ID, explicitly including Llama EOT."""

    excluded = {
        int(token_id)
        for token_id in (
            getattr(tokenizer, "pad_token_id", None),
            getattr(tokenizer, "eos_token_id", None),
            getattr(tokenizer, "bos_token_id", None),
            getattr(tokenizer, "unk_token_id", None),
        )
        if token_id is not None
    }
    excluded.update(
        int(token_id)
        for token_id in getattr(tokenizer, "all_special_ids", ())
        if token_id is not None
    )
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        eot_id = converter("<|eot_id|>")
        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if (
            isinstance(eot_id, int)
            and eot_id >= 0
            and eot_id != unknown_id
        ):
            excluded.add(int(eot_id))
    return sorted(excluded)


def orthonormal_row_basis(
    rows: torch.Tensor,
    *,
    max_rank: Optional[int] = None,
    tolerance: Optional[float] = None,
) -> Optional[torch.Tensor]:
    if rows.numel() == 0 or max_rank == 0:
        return None
    rows = rows.float()
    _, singular_values, right_vectors = torch.linalg.svd(
        rows,
        full_matrices=False,
    )
    if not singular_values.numel():
        return None
    if tolerance is None:
        tolerance = (
            max(rows.shape)
            * torch.finfo(singular_values.dtype).eps
            * float(singular_values.max().clamp_min(1.0))
        )
    rank = int((singular_values > tolerance).sum().item())
    if max_rank is not None:
        rank = min(rank, max_rank)
    return right_vectors[:rank].contiguous() if rank else None


def project_away(
    rows: torch.Tensor,
    basis: Optional[torch.Tensor],
) -> torch.Tensor:
    if basis is None or not basis.numel():
        return rows
    return rows - (rows @ basis.T) @ basis


def _sample_prompt_context_positions(
    prompt_length: int,
    *,
    maximum: int,
    device: torch.device,
) -> torch.Tensor:
    if prompt_length <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    count = min(maximum, prompt_length)
    if count == 1:
        return torch.tensor(
            [prompt_length - 1],
            dtype=torch.long,
            device=device,
        )
    return torch.linspace(
        0,
        prompt_length - 1,
        steps=count,
        device=device,
    ).round().long().unique(sorted=True)


@torch.no_grad()
def cache_delta_data(
    model: nn.Module,
    tokenizer: Any,
    prompt_answer_rows: Sequence[Tuple[str, str, str]],
    selected_token_ids: Sequence[int],
    *,
    active_margin: float,
    include_contexts: bool,
    contexts_per_example: int,
    max_length: int = 4096,
) -> Tuple[List[AnswerDeltaCache], List[ContextDeltaCache]]:
    if not selected_token_ids:
        return [], []
    selected_ids = torch.tensor(
        list(selected_token_ids),
        dtype=torch.long,
        device=model_device(model),
    )
    selected_index = {
        int(token_id): index
        for index, token_id in enumerate(selected_token_ids)
    }
    answer_caches: List[AnswerDeltaCache] = []
    context_caches: List[ContextDeltaCache] = []
    for prompt, answer, source_id in prompt_answer_rows:
        hidden_all, answer_positions, target_ids, prompt_length = (
            _prompt_answer_tensors(
                model,
                tokenizer,
                prompt,
                answer,
                max_length=max_length,
            )
        )
        if not target_ids.numel():
            continue
        context_positions = (
            _sample_prompt_context_positions(
                prompt_length,
                maximum=contexts_per_example,
                device=hidden_all.device,
            )
            if include_contexts
            else torch.empty(
                (0,),
                dtype=torch.long,
                device=hidden_all.device,
            )
        )
        all_positions = torch.cat(
            [answer_positions, context_positions]
        ).unique(sorted=True)
        all_hidden_native = hidden_all.index_select(0, all_positions)
        all_logits = _logits_for_hidden(model, all_hidden_native)
        position_to_index = {
            int(position): index
            for index, position in enumerate(all_positions.tolist())
        }

        answer_indices = torch.tensor(
            [position_to_index[int(position)] for position in answer_positions],
            dtype=torch.long,
            device=all_logits.device,
        )
        answer_hidden = all_hidden_native.index_select(
            0,
            answer_indices,
        ).float()
        answer_logits = all_logits.index_select(0, answer_indices)
        base_selected = answer_logits.index_select(1, selected_ids)
        nonselected = answer_logits.clone()
        nonselected.index_fill_(1, selected_ids, -torch.inf)
        nonselected_lse = torch.logsumexp(nonselected, dim=-1)
        nonselected_max = nonselected.max(dim=-1).values
        base_target = answer_logits.gather(
            1,
            target_ids[:, None],
        ).squeeze(1)
        target_selected = torch.tensor(
            [
                selected_index.get(int(token_id), -1)
                for token_id in target_ids.tolist()
            ],
            dtype=torch.long,
            device=answer_logits.device,
        )
        selected_other = base_selected.clone()
        has_selected_target = target_selected.ge(0)
        if has_selected_target.any():
            rows = has_selected_target.nonzero(as_tuple=False).flatten()
            selected_other[
                rows,
                target_selected.index_select(0, rows),
            ] = -torch.inf
        baseline_competitor = torch.maximum(
            nonselected_max,
            selected_other.max(dim=-1).values,
        )
        repair_mask = has_selected_target & (
            baseline_competitor - base_target < active_margin
        )
        answer_caches.append(
            AnswerDeltaCache(
                source_id=source_id,
                hidden=answer_hidden.detach(),
                target_token_ids=target_ids.detach(),
                target_selected_indices=target_selected.detach(),
                repair_mask=repair_mask.detach(),
                base_target_logits=base_target.detach(),
                base_selected_logits=base_selected.detach(),
                base_nonselected_logsumexp=nonselected_lse.detach(),
                base_nonselected_max=nonselected_max.detach(),
            )
        )

        if context_positions.numel():
            context_indices = torch.tensor(
                [
                    position_to_index[int(position)]
                    for position in context_positions
                ],
                dtype=torch.long,
                device=all_logits.device,
            )
            context_hidden = all_hidden_native.index_select(
                0,
                context_indices,
            ).float()
            context_logits = all_logits.index_select(0, context_indices)
            context_selected = context_logits.index_select(1, selected_ids)
            context_nonselected = context_logits.clone()
            context_nonselected.index_fill_(1, selected_ids, -torch.inf)
            context_max, context_ids = context_nonselected.max(dim=-1)
            context_caches.append(
                ContextDeltaCache(
                    source_id=source_id,
                    hidden=context_hidden.detach(),
                    base_selected_logits=context_selected.detach(),
                    base_nonselected_max=context_max.detach(),
                    base_nonselected_token_ids=context_ids.detach(),
                    baseline_predicted_token_ids=(
                        context_logits.argmax(dim=-1).detach()
                    ),
                )
            )
    return answer_caches, context_caches


def _candidate_answer_values(
    cache: AnswerDeltaCache,
    delta_rows: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    corrections = cache.hidden @ delta_rows.T
    selected_logits = cache.base_selected_logits + corrections
    log_partition = torch.logaddexp(
        cache.base_nonselected_logsumexp,
        torch.logsumexp(selected_logits, dim=-1),
    )
    indices = cache.target_selected_indices.clamp_min(0)
    target_correction = corrections.gather(1, indices[:, None]).squeeze(1)
    target_correction = torch.where(
        cache.target_selected_indices.ge(0),
        target_correction,
        torch.zeros_like(target_correction),
    )
    target_logits = cache.base_target_logits + target_correction
    return selected_logits, target_logits, log_partition


def answer_nlls(
    caches: Sequence[AnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for cache in caches:
        _, target_logits, log_partition = _candidate_answer_values(
            cache,
            delta_rows,
        )
        values.append((log_partition - target_logits).mean())
    if not values:
        return delta_rows.new_empty((0,))
    return torch.stack(values)


def active_margin_values(
    caches: Sequence[AnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for cache in caches:
        if not cache.repair_mask.any():
            continue
        selected_logits, target_logits, _ = _candidate_answer_values(
            cache,
            delta_rows,
        )
        other_selected = selected_logits.clone()
        rows = cache.repair_mask.nonzero(as_tuple=False).flatten()
        other_selected[
            rows,
            cache.target_selected_indices.index_select(0, rows),
        ] = -torch.inf
        competitor = torch.maximum(
            cache.base_nonselected_max,
            other_selected.max(dim=-1).values,
        )
        values.append(
            (competitor - target_logits).masked_select(cache.repair_mask)
        )
    if not values:
        return delta_rows.new_empty((0,))
    return torch.cat(values)


def active_target_probabilities(
    caches: Sequence[AnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for cache in caches:
        if not cache.repair_mask.any():
            continue
        _, target_logits, log_partition = _candidate_answer_values(
            cache,
            delta_rows,
        )
        values.append(
            torch.exp(target_logits - log_partition).masked_select(
                cache.repair_mask
            )
        )
    if not values:
        return delta_rows.new_empty((0,))
    return torch.cat(values)


def protected_probability_ratio(
    caches: Sequence[AnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> float:
    if not caches:
        return 1.0
    current_nll = answer_nlls(caches, delta_rows).float()
    baseline_nll = answer_nlls(
        caches,
        torch.zeros_like(delta_rows),
    ).float()
    log_count = math.log(len(caches))
    current_log_mean = torch.logsumexp(-current_nll, dim=0) - log_count
    baseline_log_mean = torch.logsumexp(-baseline_nll, dim=0) - log_count
    ratio = torch.exp(current_log_mean - baseline_log_mean)
    if not torch.isfinite(ratio) or ratio <= 0:
        raise FloatingPointError("Protected probability ratio is invalid")
    return float(ratio.detach().cpu())


def _context_hidden(
    caches: Sequence[ContextDeltaCache],
    *,
    hidden_size: int,
    device: torch.device,
) -> torch.Tensor:
    rows = [cache.hidden for cache in caches if cache.hidden.numel()]
    if not rows:
        return torch.empty(
            (0, hidden_size),
            dtype=torch.float32,
            device=device,
        )
    return torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)


def context_diagnostics(
    caches: Sequence[ContextDeltaCache],
    delta_rows: torch.Tensor,
    selected_token_ids: Sequence[int],
) -> Dict[str, Any]:
    if not caches:
        return {
            "protected_context_count": 0,
            "protected_top1_changes": 0,
            "maximum_protected_logit_drift": 0.0,
            "mean_absolute_protected_logit_drift": 0.0,
        }
    selected_ids = torch.tensor(
        list(selected_token_ids),
        dtype=torch.long,
        device=delta_rows.device,
    )
    changes = 0
    drifts: List[torch.Tensor] = []
    context_count = 0
    for cache in caches:
        correction = cache.hidden @ delta_rows.T
        candidate_selected = cache.base_selected_logits + correction
        selected_max, selected_columns = candidate_selected.max(dim=-1)
        selected_predictions = selected_ids.index_select(
            0,
            selected_columns,
        )
        nonselected_max = cache.base_nonselected_max
        nonselected_predictions = cache.base_nonselected_token_ids
        selected_wins = selected_max > nonselected_max
        ties = selected_max.eq(nonselected_max)
        selected_wins = selected_wins | (
            ties & selected_predictions.lt(nonselected_predictions)
        )
        predictions = torch.where(
            selected_wins,
            selected_predictions,
            nonselected_predictions,
        )
        changes += int(
            predictions.ne(cache.baseline_predicted_token_ids).sum().item()
        )
        drifts.append(correction.abs())
        context_count += int(cache.hidden.shape[0])
    drift = torch.cat([row.flatten() for row in drifts])
    return {
        "protected_context_count": context_count,
        "protected_top1_changes": changes,
        "maximum_protected_logit_drift": float(drift.max().detach().cpu()),
        "mean_absolute_protected_logit_drift": float(
            drift.mean().detach().cpu()
        ),
    }


def optimize_delta(
    active_caches: Sequence[AnswerDeltaCache],
    protected_context_caches: Sequence[ContextDeltaCache],
    *,
    n_rows: int,
    hidden_size: int,
    device: torch.device,
    config: RepairConfig,
) -> Tuple[torch.Tensor, List[Dict[str, float]], Dict[str, Any]]:
    validate_config(config)
    if n_rows <= 0:
        raise ValueError("Sparse repair requires at least one selected row")
    protected_hidden = _context_hidden(
        protected_context_caches,
        hidden_size=hidden_size,
        device=device,
    )
    basis = (
        orthonormal_row_basis(
            protected_hidden,
            max_rank=config.protected_projection_rank,
        )
        if config.project_away_protected and protected_hidden.numel()
        else None
    )
    delta_module = active_repair.SelectedRowDelta(
        n_rows,
        hidden_size,
        retained_basis=basis,
        device=device,
    )
    optimizer = torch.optim.AdamW(
        delta_module.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    log: List[Dict[str, float]] = []
    stopped_early = False
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        margins = active_margin_values(active_caches, delta)
        if not margins.numel():
            raise ValueError("No selected active answer-token positions remain")
        hinge = F.relu(config.active_margin - margins).square().mean()
        protected_drift = (
            (protected_hidden @ delta.T).square().mean()
            if protected_hidden.numel()
            else delta.new_zeros(())
        )
        penalty = config.l2_lambda * delta.square().sum()
        loss = (
            hinge
            + config.protected_logit_lambda * protected_drift
            + penalty
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite sparse-repair objective")
        loss.backward()
        optimizer.step()
        if config.max_delta_norm is not None:
            active_repair.constrain_effective_delta_norm(
                delta_module,
                config.max_delta_norm,
            )
        with torch.no_grad():
            effective = delta_module.effective_delta()
            margins = active_margin_values(active_caches, effective)
            unsatisfied = int(
                (margins < config.active_margin).sum().item()
            )
            row = {
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "hinge": float(hinge.detach().cpu()),
                "protected_logit_penalty": float(
                    protected_drift.detach().cpu()
                ),
                "delta_norm": float(effective.norm().cpu()),
                "minimum_active_margin": float(margins.min().cpu()),
                "mean_active_margin": float(margins.mean().cpu()),
                "unsatisfied_active_points": float(unsatisfied),
            }
            log.append(row)
            if config.stop_when_satisfied and unsatisfied == 0:
                stopped_early = True
                break
    delta = delta_module.effective_delta().detach()
    return delta, log, {
        "steps_completed": len(log),
        "stopped_early": stopped_early,
        "active_point_count": int(
            sum(cache.repair_mask.sum().item() for cache in active_caches)
        ),
        "protected_context_count": int(protected_hidden.shape[0]),
        "protected_hidden_rank": 0 if basis is None else int(basis.shape[0]),
        "selected_row_count": n_rows,
        "delta_norm": float(delta.norm().cpu()),
    }


def materialized_delta_rows(
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    candidate = (
        original_rows.float() + float(scale) * delta_rows.float()
    ).to(dtype=original_rows.dtype)
    return candidate.float() - original_rows.float()


@torch.no_grad()
def select_materialized_scale(
    *,
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    active_caches: Sequence[AnswerDeltaCache],
    protected_answer_caches: Sequence[AnswerDeltaCache],
    protected_context_caches: Sequence[ContextDeltaCache],
    selected_token_ids: Sequence[int],
    scales: Sequence[float],
    config: RepairConfig,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Select the strongest effective candidate satisfying every hard gate."""

    reports: List[Dict[str, Any]] = []
    zero_delta = torch.zeros_like(delta_rows)
    baseline_margins = active_margin_values(active_caches, zero_delta)
    for scale in scales:
        materialized = materialized_delta_rows(
            original_rows,
            delta_rows,
            scale,
        )
        margins = active_margin_values(active_caches, materialized)
        probabilities = active_target_probabilities(
            active_caches,
            materialized,
        )
        context = context_diagnostics(
            protected_context_caches,
            materialized,
            selected_token_ids,
        )
        ratio = protected_probability_ratio(
            protected_answer_caches,
            materialized,
        )
        passes = bool(
            ratio + 1e-12 >= config.min_protected_probability_ratio
            and context["maximum_protected_logit_drift"]
            <= config.max_protected_logit_drift + 1e-12
            and context["protected_top1_changes"]
            <= config.max_protected_top1_changes
        )
        reports.append(
            {
                "scale": float(scale),
                "passes_all_protection_gates": passes,
                "protected_answer_probability_ratio": ratio,
                **context,
                "active_unsatisfied": int(
                    (margins < config.selection_margin).sum().item()
                ),
                "active_margin_regressions": int(
                    (margins < baseline_margins - 1e-7).sum().item()
                ),
                "active_satisfied": int(
                    (margins >= config.selection_margin).sum().item()
                ),
                "minimum_active_margin": float(margins.min().detach().cpu()),
                "mean_active_margin": float(margins.mean().detach().cpu()),
                "mean_active_target_probability": float(
                    probabilities.mean().detach().cpu()
                ),
                "materialized_delta_norm": float(
                    materialized.norm().detach().cpu()
                ),
            }
        )
    baseline = next(
        row for row in reports if float(row["scale"]) == 0.0
    )
    effective = [
        row
        for row in reports
        if row["passes_all_protection_gates"]
        and int(row["active_margin_regressions"]) == 0
        and (
            int(row["active_unsatisfied"])
            < int(baseline["active_unsatisfied"])
            or float(row["mean_active_target_probability"])
            < float(baseline["mean_active_target_probability"]) - 1e-8
        )
    ]
    if not effective:
        return 0.0, reports
    best = min(
        effective,
        key=lambda row: (
            int(row["active_unsatisfied"]),
            float(row["mean_active_target_probability"]),
            -float(row["mean_active_margin"]),
            float(row["materialized_delta_norm"]),
        ),
    )
    return float(best["scale"]), reports


@torch.no_grad()
def apply_materialized_delta(
    output_weight: torch.Tensor,
    *,
    token_ids: Sequence[int],
    original_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    scale: float,
) -> None:
    if not token_ids:
        return
    selected = torch.tensor(
        list(token_ids),
        dtype=torch.long,
        device=output_weight.device,
    )
    candidate = (
        original_rows.float() + float(scale) * delta_rows.float()
    ).to(device=output_weight.device, dtype=output_weight.dtype)
    output_weight.index_copy_(0, selected, candidate)


def point_report(point: RepairPoint) -> Dict[str, Any]:
    return {
        "source_id": point.source_id,
        "token_index": point.token_index,
        "target_token_id": point.target_token_id,
        "competitor_token_id": point.competitor_token_id,
        "baseline_predicted_token_id": point.baseline_predicted_token_id,
        "target_logit": float(point.target_logit.cpu()),
        "competitor_logit": float(point.competitor_logit.cpu()),
        "baseline_competitor_minus_target_margin": float(
            (point.competitor_logit - point.target_logit).cpu()
        ),
    }


def _token_text(tokenizer: Any, token_id: int) -> str:
    try:
        return str(tokenizer.decode([token_id]))
    except Exception:
        return f"<token:{token_id}>"


def _no_op_report(
    *,
    tokenizer: Any,
    config: RepairConfig,
    all_active_points: Sequence[RepairPoint],
    overlap_ids: Sequence[int],
    selected_ids: Sequence[int],
    reason: str,
) -> Dict[str, Any]:
    return {
        "method": "sparse_active_pair_lm_head_repair",
        "repair_applied": False,
        "no_op_reason": reason,
        "changed_output_rows": [],
        "selected_scale": 0.0,
        "selected_lm_head_token_ids": list(selected_ids),
        "selected_lm_head_tokens": {
            str(token_id): _token_text(tokenizer, token_id)
            for token_id in selected_ids
        },
        "protected_overlap_token_ids": list(overlap_ids),
        "protected_overlap_tokens": {
            str(token_id): _token_text(tokenizer, token_id)
            for token_id in overlap_ids
        },
        "active_point_count_before_row_filter": len(all_active_points),
        "active_points": [point_report(point) for point in all_active_points],
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "edits_eot_or_eos": False,
        "config": asdict(config),
        "scale_sweep": [],
        "limitations": {
            "multiple_choice": (
                "Letter-scored multiple choice bypasses target-answer LM-head rows."
            ),
            "frozen_base_head_probe": (
                "A frozen-base-head probe intentionally bypasses this repair."
            ),
        },
    }


def hierarchical_repair_outcomes(
    *,
    tokenizer: Any,
    calibration_rows: Sequence[Mapping[str, Any]],
    before_points: Sequence[RepairPoint],
    after_points: Sequence[RepairPoint],
    selected_ids: Sequence[int],
    overlap_ids: Sequence[int],
    special_ids: Sequence[int],
    selected_scale: float,
    config: RepairConfig,
) -> Dict[str, Any]:
    """Report token-, view-, and fact-level support without overclaiming."""

    metadata_by_source: Dict[str, Dict[str, Any]] = {}
    for row in calibration_rows:
        source_id = str(row.get("view_id") or record_sha256(row))
        metadata_by_source[source_id] = {
            "fact_id": str(row.get("fact_id", source_id)),
            "view_id": str(row.get("view_id", source_id)),
            "sensitive_answer_alias": str(row.get("answer", "")),
            "prompt_style": row.get("prompt_style"),
            "boundary_expanding": bool(row.get("boundary_expanding", False)),
        }
    after_by_position = {
        (point.source_id, point.token_index): point for point in after_points
    }
    selected = set(int(value) for value in selected_ids)
    overlap = set(int(value) for value in overlap_ids)
    special = set(int(value) for value in special_ids)
    token_rows: List[Dict[str, Any]] = []
    for point in before_points:
        metadata = metadata_by_source.get(
            point.source_id,
            {
                "fact_id": point.source_id,
                "view_id": point.source_id,
                "sensitive_answer_alias": "",
                "prompt_style": None,
                "boundary_expanding": False,
            },
        )
        after = after_by_position.get((point.source_id, point.token_index))
        before_margin = float(
            (point.competitor_logit - point.target_logit).detach().cpu()
        )
        after_margin = (
            before_margin
            if after is None
            else float(
                (after.competitor_logit - after.target_logit).detach().cpu()
            )
        )
        initially_resolved = before_margin >= config.active_margin
        token_id = int(point.target_token_id)
        if initially_resolved:
            classification = "calibration_resolved_by_setting5e"
            eligible = False
            outcome = "calibration_resolved_by_setting5e"
        elif token_id in special:
            classification = "special_token_pair"
            eligible = False
            outcome = "unresolved_after_repair"
        elif token_id in overlap:
            classification = "shared_protected_answer_pair"
            eligible = False
            outcome = "unresolved_after_repair"
        elif token_id in selected:
            classification = "safe_sparse_head_pair"
            eligible = True
            outcome = (
                "repaired_margin_satisfied"
                if after_margin >= config.selection_margin
                else "unresolved_after_repair"
            )
        else:
            classification = "unsupported_pair"
            eligible = False
            outcome = "unresolved_after_repair"
        token_rows.append(
            {
                **metadata,
                "answer_position": int(point.token_index),
                "tokenizer_token_id": token_id,
                "decoded_token_piece": _token_text(tokenizer, token_id),
                "sensitive_answer_alias": metadata["sensitive_answer_alias"],
                "baseline_target_logit": float(point.target_logit.detach().cpu()),
                "baseline_competitor_logit": float(
                    point.competitor_logit.detach().cpu()
                ),
                "post_setting5_margin": before_margin,
                "post_repair_margin": after_margin,
                "protection_classification": classification,
                "repair_eligibility": eligible,
                "selected_scale": float(selected_scale),
                "token_outcome": outcome,
            }
        )

    tokens_by_view: Dict[str, List[Dict[str, Any]]] = {}
    for row in token_rows:
        tokens_by_view.setdefault(row["view_id"], []).append(row)
    view_rows: List[Dict[str, Any]] = []
    for view_id, rows in sorted(tokens_by_view.items()):
        active = [
            row
            for row in rows
            if row["token_outcome"] != "calibration_resolved_by_setting5e"
        ]
        if not active:
            support = "calibration_resolved_by_setting5e"
            outcome = "calibration_resolved_by_setting5e"
        else:
            eligible = sum(bool(row["repair_eligibility"]) for row in active)
            support = (
                "fully_supported"
                if eligible == len(active)
                else "partially_supported"
                if eligible
                else "unsupported"
            )
            outcome = (
                "calibration_resolved_after_repair"
                if all(
                    row["post_repair_margin"] >= config.selection_margin
                    for row in rows
                )
                else "unresolved_after_repair"
            )
        view_rows.append(
            {
                "fact_id": rows[0]["fact_id"],
                "view_id": view_id,
                "prompt_style": rows[0].get("prompt_style"),
                "boundary_expanding": bool(rows[0].get("boundary_expanding", False)),
                "token_position_count": len(rows),
                "support_outcome": support,
                "view_outcome": outcome,
            }
        )

    views_by_fact: Dict[str, List[Dict[str, Any]]] = {}
    for row in view_rows:
        views_by_fact.setdefault(row["fact_id"], []).append(row)
    fact_rows: List[Dict[str, Any]] = []
    for fact_id, rows in sorted(views_by_fact.items()):
        resolved = sum(
            row["view_outcome"]
            in {
                "calibration_resolved_by_setting5e",
                "calibration_resolved_after_repair",
            }
            for row in rows
        )
        if resolved == len(rows):
            outcome = "all_calibration_views_resolved"
        elif resolved:
            outcome = "partially_resolved"
        elif all(row["support_outcome"] == "unsupported" for row in rows):
            outcome = "unsupported_by_sparse_repair"
        else:
            outcome = "unresolved_after_repair"
        fact_rows.append(
            {
                "fact_id": fact_id,
                "calibration_view_count": len(rows),
                "resolved_calibration_view_count": resolved,
                "fact_outcome": outcome,
            }
        )
    return {
        "token_position_outcomes": token_rows,
        "view_outcomes": view_rows,
        "fact_outcomes": fact_rows,
    }


def run_protected_lm_head_repair(
    model: nn.Module,
    tokenizer: Any,
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    protected_examples: Sequence[Any],
    optimization_protection_examples: Optional[Sequence[Any]] = None,
    config: RepairConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Apply retain-gated sparse answer-row repair.

    The legacy public function name is kept so existing launchers continue to
    work; the report records the new sparse active-pair semantics.
    """

    validate_config(config)
    input_weight = model.get_input_embeddings().weight
    input_pointer = input_weight.data_ptr()
    input_version = input_weight._version
    output_layer = active_repair.freeze_model_for_output_repair(model)
    calibration_pairs = calibration_prompt_answers(
        tokenizer,
        calibration_rows,
    )
    protected_pairs = protected_prompt_answers(protected_examples)
    gate_separated_from_gradient = optimization_protection_examples is not None
    optimization_protection_pairs = protected_prompt_answers(
        protected_examples
        if optimization_protection_examples is None
        else optimization_protection_examples
    )
    calibration_points = cache_prompt_answers(
        model,
        tokenizer,
        calibration_pairs,
    )
    protected_points = cache_prompt_answers(
        model,
        tokenizer,
        protected_pairs,
    )
    optimization_protection_points = cache_prompt_answers(
        model,
        tokenizer,
        optimization_protection_pairs,
    )
    all_active_points = active_points(
        calibration_points,
        required_margin=config.active_margin,
    )
    selected_ids, overlap_ids = select_active_token_ids(
        all_active_points,
        [*protected_points, *optimization_protection_points],
        special_token_ids=repair_excluded_token_ids(tokenizer),
        exclude_protected_answer_rows=config.exclude_protected_answer_rows,
    )

    if not all_active_points:
        report = _no_op_report(
            tokenizer=tokenizer,
            config=config,
            all_active_points=all_active_points,
            overlap_ids=overlap_ids,
            selected_ids=selected_ids,
            reason="all_calibration_answer_positions_already_meet_forget_margin",
        )
    elif not selected_ids:
        report = _no_op_report(
            tokenizer=tokenizer,
            config=config,
            all_active_points=all_active_points,
            overlap_ids=overlap_ids,
            selected_ids=selected_ids,
            reason=(
                "no_active_non_special_answer_row_is_exclusive_to_the_forget_set"
            ),
        )
    else:
        active_caches, _ = cache_delta_data(
            model,
            tokenizer,
            calibration_pairs,
            selected_ids,
            active_margin=config.active_margin,
            include_contexts=False,
            contexts_per_example=config.protected_contexts_per_example,
        )
        protected_answer_caches, protected_context_caches = cache_delta_data(
            model,
            tokenizer,
            protected_pairs,
            selected_ids,
            active_margin=config.active_margin,
            include_contexts=True,
            contexts_per_example=config.protected_contexts_per_example,
        )
        _, optimization_protection_context_caches = cache_delta_data(
            model,
            tokenizer,
            optimization_protection_pairs,
            selected_ids,
            active_margin=config.active_margin,
            include_contexts=True,
            contexts_per_example=config.protected_contexts_per_example,
        )
        supported_active_count = int(
            sum(cache.repair_mask.sum().item() for cache in active_caches)
        )
        if supported_active_count == 0:
            report = _no_op_report(
                tokenizer=tokenizer,
                config=config,
                all_active_points=all_active_points,
                overlap_ids=overlap_ids,
                selected_ids=selected_ids,
                reason="selected_rows_have_no_remaining_active_positions",
            )
        else:
            device = output_layer.weight.device
            selected_tensor = torch.tensor(
                selected_ids,
                dtype=torch.long,
                device=device,
            )
            original_rows = (
                output_layer.weight.index_select(0, selected_tensor)
                .detach()
                .clone()
            )
            delta, optimization_log, optimization_summary = optimize_delta(
                active_caches,
                optimization_protection_context_caches,
                n_rows=len(selected_ids),
                hidden_size=int(output_layer.weight.shape[1]),
                device=device,
                config=config,
            )
            scale, scale_reports = select_materialized_scale(
                original_rows=original_rows,
                delta_rows=delta,
                active_caches=active_caches,
                protected_answer_caches=protected_answer_caches,
                protected_context_caches=protected_context_caches,
                selected_token_ids=selected_ids,
                scales=config.candidate_scales,
                config=config,
            )
            apply_materialized_delta(
                output_layer.weight,
                token_ids=selected_ids,
                original_rows=original_rows,
                delta_rows=delta,
                scale=scale,
            )
            selected = next(
                row
                for row in scale_reports
                if float(row["scale"]) == scale
            )
            supported_ids = set(selected_ids)
            unsupported_points = [
                point
                for point in all_active_points
                if point.target_token_id not in supported_ids
            ]
            report = {
                "method": "sparse_active_pair_lm_head_repair",
                "repair_applied": scale != 0.0,
                "no_op_reason": (
                    None
                    if scale != 0.0
                    else "no_effective_candidate_passed_all_protection_gates"
                ),
                "changed_output_rows": (
                    list(selected_ids) if scale != 0.0 else []
                ),
                "selected_lm_head_token_ids": list(selected_ids),
                "selected_lm_head_tokens": {
                    str(token_id): _token_text(tokenizer, token_id)
                    for token_id in selected_ids
                },
                "protected_overlap_token_ids": list(overlap_ids),
                "protected_overlap_tokens": {
                    str(token_id): _token_text(tokenizer, token_id)
                    for token_id in overlap_ids
                },
                "transformer_frozen": True,
                "input_embeddings_frozen": True,
                "output_head_untied": True,
                "edits_eot_or_eos": False,
                "selection_policy": (
                    "hard protected gates first; best active suppression among "
                    "passing candidates with no calibration-margin regression; "
                    "scale zero fallback"
                ),
                "selected_scale": scale,
                "selected_scale_report": selected,
                "optimization": optimization_summary,
                "config": asdict(config),
                "active_source": (
                    "RWKU calibration level1+level2 active answer-token "
                    "positions only"
                ),
                "protected_source": (
                    "disjoint repair-selection gate answers and prompt contexts"
                ),
                "optimization_protection_source": (
                    "method-visible optimization-protection records"
                    if gate_separated_from_gradient
                    else "legacy protected records reused by the repair objective"
                ),
                "repair_gate_contributed_gradients": not gate_separated_from_gradient,
                "active_point_count_before_row_filter": len(
                    all_active_points
                ),
                "active_point_count_after_row_filter": supported_active_count,
                "unsupported_active_point_count": len(unsupported_points),
                "active_points": [
                    point_report(point) for point in all_active_points
                ],
                "unsupported_active_points": [
                    point_report(point) for point in unsupported_points
                ],
                "scale_sweep": scale_reports,
                "optimization_log": optimization_log,
                "limitations": {
                    "multiple_choice": (
                        "Letter-scored multiple choice bypasses target-answer "
                        "LM-head rows."
                    ),
                    "frozen_base_head_probe": (
                        "A frozen-base-head probe intentionally bypasses this "
                        "repair; lowering it requires representation-level "
                        "unlearning."
                    ),
                },
            }

    after_points = cache_prompt_answers(
        model,
        tokenizer,
        calibration_pairs,
    )
    report["hierarchical_outcomes"] = hierarchical_repair_outcomes(
        tokenizer=tokenizer,
        calibration_rows=calibration_rows,
        before_points=calibration_points,
        after_points=after_points,
        selected_ids=selected_ids,
        overlap_ids=overlap_ids,
        special_ids=repair_excluded_token_ids(tokenizer),
        selected_scale=float(report.get("selected_scale", 0.0)),
        config=config,
    )
    report.setdefault(
        "repair_gate_contributed_gradients",
        not gate_separated_from_gradient,
    )
    report["repair_gate_evaluated_under_no_grad"] = True

    if (
        model.get_input_embeddings().weight.data_ptr() != input_pointer
        or model.get_input_embeddings().weight._version != input_version
    ):
        raise RuntimeError("Input embeddings changed during sparse LM-head repair")
    if output_dir is not None:
        from rwku_eval import write_json

        write_json(Path(output_dir) / "repair_report.json", report)
    return report
