#!/usr/bin/env python3
"""Protected one-row LM-head repair used by the RWKU experiment.

The transformer and input embeddings are frozen.  Only a neutral stop-token
row in the output head is changed.  Calibration target-answer states are the
active constraints; unrelated retained facts are protected.  The learned
direction is projected away from the span of protected hidden states and a
materialized-dtype scale sweep rejects protected top-1 regressions.
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
    competitor_logit: torch.Tensor
    neutral_logit: torch.Tensor
    source_id: str
    token_index: int
    target_token_id: int
    baseline_predicted_token_id: int


@dataclass(frozen=True)
class RepairConfig:
    steps: int = 800
    learning_rate: float = 5e-3
    active_margin: float = 0.25
    selection_margin: float = 0.05
    l2_lambda: float = 1e-6
    max_delta_norm: Optional[float] = None
    project_away_protected: bool = True
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
    if config.l2_lambda < 0:
        raise ValueError("repair L2 coefficient must be non-negative")
    if config.max_delta_norm is not None and config.max_delta_norm <= 0:
        raise ValueError("repair max_delta_norm must be positive")
    if not config.candidate_scales:
        raise ValueError("candidate_scales must not be empty")
    if any(
        not math.isfinite(scale) or not 0.0 <= scale <= 1.0
        for scale in config.candidate_scales
    ):
        raise ValueError("candidate scales must be finite values in [0,1]")
    if 0.0 not in config.candidate_scales:
        raise ValueError("candidate scales must include the no-op scale 0")


def resolve_neutral_token(tokenizer: Any) -> Tuple[str, int]:
    converter = getattr(tokenizer, "convert_tokens_to_ids", None)
    if callable(converter):
        token_id = converter("<|eot_id|>")
        unknown = getattr(tokenizer, "unk_token_id", None)
        if isinstance(token_id, int) and token_id >= 0 and token_id != unknown:
            return "<|eot_id|>", int(token_id)
    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token, str) or eos_token_id is None:
        raise ValueError("Tokenizer has neither a usable EOT nor EOS token")
    return eos_token, int(eos_token_id)


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _competitor(
    logits: torch.Tensor,
    neutral_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.ndim != 2:
        raise ValueError("Expected [positions, vocabulary] logits")
    top_values, top_ids = torch.topk(logits, k=2, dim=-1)
    neutral_is_top = top_ids[:, 0].eq(neutral_token_id)
    competitor_values = torch.where(
        neutral_is_top,
        top_values[:, 1],
        top_values[:, 0],
    )
    competitor_ids = torch.where(
        neutral_is_top,
        top_ids[:, 1],
        top_ids[:, 0],
    )
    return competitor_values, competitor_ids


@torch.no_grad()
def cache_prompt_answers(
    model: nn.Module,
    tokenizer: Any,
    prompt_answer_rows: Sequence[Tuple[str, str, str]],
    *,
    neutral_token_id: int,
    max_length: int = 4096,
) -> List[RepairPoint]:
    """Cache every teacher-forced answer-token state for repair constraints."""

    device = model_device(model)
    points: List[RepairPoint] = []
    for prompt, answer, source_id in prompt_answer_rows:
        prompt_ids = _token_ids(tokenizer, prompt, add_special_tokens=True)
        answer_ids = _token_ids(
            tokenizer,
            _normalized_completion(answer),
            add_special_tokens=False,
        )
        if not answer_ids:
            continue
        allowed_prompt = max(1, max_length - len(answer_ids))
        prompt_ids = prompt_ids[-allowed_prompt:]
        sequence = prompt_ids + answer_ids
        input_ids = torch.tensor([sequence], dtype=torch.long, device=device)
        hidden_all = final_hidden_states(
            model,
            input_ids=input_ids,
        )
        positions = torch.arange(
            len(prompt_ids) - 1,
            len(prompt_ids) + len(answer_ids) - 1,
            device=device,
        )
        hidden_native = hidden_all[0, positions]
        output_layer = model.get_output_embeddings()
        if output_layer is None:
            raise ValueError("Model does not expose output embeddings")

        # Run the LM-head projection in its materialized dtype. The model is
        # normally BF16 on Wulver; converting hidden states to FP32 before this
        # projection causes a Float/BFloat16 matrix-multiplication mismatch.
        hidden_for_head = hidden_native.to(
            device=output_layer.weight.device,
            dtype=output_layer.weight.dtype,
        )
        logits = output_layer(hidden_for_head).float()

        # Keep cached states in FP32 for the repair optimisation.
        hidden = hidden_native.float()
        competitor_values, competitor_ids = _competitor(
            logits,
            neutral_token_id,
        )
        neutral = logits[:, neutral_token_id]
        predicted = logits.argmax(dim=-1)
        for token_index, target_token_id in enumerate(answer_ids):
            points.append(
                RepairPoint(
                    hidden=hidden[token_index].detach(),
                    competitor_logit=competitor_values[token_index].detach(),
                    neutral_logit=neutral[token_index].detach(),
                    source_id=source_id,
                    token_index=token_index,
                    target_token_id=int(target_token_id),
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
            record_sha256(row),
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


def orthonormal_row_basis(
    rows: torch.Tensor,
    *,
    tolerance: Optional[float] = None,
) -> Optional[torch.Tensor]:
    if rows.numel() == 0:
        return None
    _, singular_values, right_vectors = torch.linalg.svd(
        rows.float(),
        full_matrices=False,
    )
    if not singular_values.numel():
        return None
    if tolerance is None:
        tolerance = (
            max(rows.shape)
            * torch.finfo(singular_values.dtype).eps
            * float(singular_values.max())
        )
    rank = int((singular_values > tolerance).sum().item())
    return right_vectors[:rank] if rank else None


def project_away(
    vector: torch.Tensor,
    basis: Optional[torch.Tensor],
) -> torch.Tensor:
    if basis is None or not basis.numel():
        return vector
    return vector - (vector @ basis.T) @ basis


def _stack(
    points: Sequence[RepairPoint],
    field: str,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not points:
        return torch.empty((0,), dtype=torch.float32, device=device)
    return torch.stack([getattr(point, field) for point in points]).to(
        device=device,
        dtype=torch.float32,
    )


def optimize_delta(
    active_points: Sequence[RepairPoint],
    protected_points: Sequence[RepairPoint],
    *,
    hidden_size: int,
    device: torch.device,
    config: RepairConfig,
) -> Tuple[torch.Tensor, List[Dict[str, float]], Dict[str, Any]]:
    validate_config(config)
    if not active_points:
        raise ValueError("Protected repair requires at least one active point")
    active_hidden = _stack(active_points, "hidden", device=device)
    active_competitor = _stack(
        active_points,
        "competitor_logit",
        device=device,
    )
    active_neutral = _stack(active_points, "neutral_logit", device=device)
    protected_hidden = _stack(protected_points, "hidden", device=device)
    basis = (
        orthonormal_row_basis(protected_hidden)
        if config.project_away_protected and protected_hidden.numel()
        else None
    )

    raw_delta = nn.Parameter(
        torch.zeros(hidden_size, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        [raw_delta],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    log: List[Dict[str, float]] = []
    stopped_early = False
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        delta = project_away(raw_delta, basis)
        margins = active_neutral + active_hidden @ delta - active_competitor
        hinge = F.relu(config.active_margin - margins).square().mean()
        penalty = config.l2_lambda * delta.square().sum()
        loss = hinge + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite protected-repair objective")
        loss.backward()
        optimizer.step()
        if config.max_delta_norm is not None:
            with torch.no_grad():
                effective = project_away(raw_delta, basis)
                norm = effective.norm()
                if norm > config.max_delta_norm:
                    raw_delta.mul_(config.max_delta_norm / norm)
        with torch.no_grad():
            effective = project_away(raw_delta, basis)
            margins = (
                active_neutral
                + active_hidden @ effective
                - active_competitor
            )
            unsatisfied = int((margins < config.active_margin).sum().item())
            row = {
                "step": float(step + 1),
                "loss": float(loss.detach().cpu()),
                "hinge": float(hinge.detach().cpu()),
                "delta_norm": float(effective.norm().cpu()),
                "minimum_active_margin": float(margins.min().cpu()),
                "unsatisfied_active_points": float(unsatisfied),
            }
            log.append(row)
            if config.stop_when_satisfied and unsatisfied == 0:
                stopped_early = True
                break
    delta = project_away(raw_delta.detach(), basis).detach()
    return delta, log, {
        "steps_completed": len(log),
        "stopped_early": stopped_early,
        "active_point_count": len(active_points),
        "protected_point_count": len(protected_points),
        "protected_hidden_rank": 0 if basis is None else int(basis.shape[0]),
        "delta_norm": float(delta.norm().cpu()),
    }


def point_margins_for_materialized_row(
    points: Sequence[RepairPoint],
    *,
    original_row: torch.Tensor,
    candidate_row: torch.Tensor,
) -> torch.Tensor:
    if not points:
        return torch.empty((0,), dtype=torch.float32)
    device = original_row.device
    hidden = _stack(points, "hidden", device=device)
    competitor = _stack(points, "competitor_logit", device=device)
    neutral = _stack(points, "neutral_logit", device=device)
    effective_delta = candidate_row.float() - original_row.float()
    return neutral + hidden @ effective_delta - competitor


def select_materialized_scale(
    *,
    original_row: torch.Tensor,
    delta: torch.Tensor,
    active_points: Sequence[RepairPoint],
    protected_points: Sequence[RepairPoint],
    scales: Sequence[float],
    selection_margin: float,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Select in stored row dtype, with protection ahead of efficacy."""

    reports: List[Dict[str, Any]] = []
    for scale in scales:
        candidate = (
            original_row.float() + float(scale) * delta.float()
        ).to(dtype=original_row.dtype).float()
        active_margins = point_margins_for_materialized_row(
            active_points,
            original_row=original_row.float(),
            candidate_row=candidate,
        )
        protected_margins = point_margins_for_materialized_row(
            protected_points,
            original_row=original_row.float(),
            candidate_row=candidate,
        )
        reports.append(
            {
                "scale": float(scale),
                "protected_regressions": int(
                    (protected_margins >= 0.0).sum().item()
                ),
                "active_unsatisfied": int(
                    (active_margins < selection_margin).sum().item()
                ),
                "active_satisfied": int(
                    (active_margins >= selection_margin).sum().item()
                ),
                "minimum_active_margin": (
                    None
                    if not len(active_points)
                    else float(active_margins.min().cpu())
                ),
                "materialized_delta_norm": float(
                    (candidate - original_row.float()).norm().cpu()
                ),
            }
        )
    best = min(
        reports,
        key=lambda row: (
            int(row["protected_regressions"]),
            int(row["active_unsatisfied"]),
            -float(
                row["minimum_active_margin"]
                if row["minimum_active_margin"] is not None
                else 0.0
            ),
            float(row["materialized_delta_norm"]),
        ),
    )
    return float(best["scale"]), reports


@torch.no_grad()
def apply_materialized_delta(
    output_weight: torch.Tensor,
    *,
    token_id: int,
    original_row: torch.Tensor,
    delta: torch.Tensor,
    scale: float,
) -> None:
    candidate = (
        original_row.float() + float(scale) * delta.float()
    ).to(device=output_weight.device, dtype=output_weight.dtype)
    output_weight[token_id].copy_(candidate)


def point_report(point: RepairPoint) -> Dict[str, Any]:
    return {
        "source_id": point.source_id,
        "token_index": point.token_index,
        "target_token_id": point.target_token_id,
        "baseline_predicted_token_id": point.baseline_predicted_token_id,
        "competitor_logit": float(point.competitor_logit.cpu()),
        "neutral_logit": float(point.neutral_logit.cpu()),
        "baseline_neutral_margin": float(
            (point.neutral_logit - point.competitor_logit).cpu()
        ),
    }


def run_protected_lm_head_repair(
    model: nn.Module,
    tokenizer: Any,
    *,
    calibration_rows: Sequence[Mapping[str, Any]],
    protected_examples: Sequence[Any],
    config: RepairConfig,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    validate_config(config)
    output_layer = active_repair.freeze_model_for_output_repair(model)
    neutral_token, neutral_token_id = resolve_neutral_token(tokenizer)
    if not 0 <= neutral_token_id < output_layer.weight.shape[0]:
        raise ValueError("Neutral token ID falls outside the LM-head vocabulary")

    active_points = cache_prompt_answers(
        model,
        tokenizer,
        calibration_prompt_answers(tokenizer, calibration_rows),
        neutral_token_id=neutral_token_id,
    )
    protected_points_all = cache_prompt_answers(
        model,
        tokenizer,
        protected_prompt_answers(protected_examples),
        neutral_token_id=neutral_token_id,
    )
    # A point where the neutral token already wins cannot newly regress.
    protected_points = [
        point
        for point in protected_points_all
        if float(point.neutral_logit) < float(point.competitor_logit)
    ]
    device = model_device(model)
    original_row = (
        output_layer.weight[neutral_token_id].detach().clone()
    )
    delta, optimization_log, optimization_summary = optimize_delta(
        active_points,
        protected_points,
        hidden_size=int(output_layer.weight.shape[1]),
        device=device,
        config=config,
    )
    scale, scale_reports = select_materialized_scale(
        original_row=original_row,
        delta=delta,
        active_points=active_points,
        protected_points=protected_points,
        scales=config.candidate_scales,
        selection_margin=config.selection_margin,
    )
    apply_materialized_delta(
        output_layer.weight,
        token_id=neutral_token_id,
        original_row=original_row,
        delta=delta,
        scale=scale,
    )
    selected = next(
        row for row in scale_reports if float(row["scale"]) == scale
    )
    report: Dict[str, Any] = {
        "method": "protected_lm_head_repair",
        "neutral_token": neutral_token,
        "neutral_token_id": neutral_token_id,
        "changed_output_rows": [neutral_token_id] if scale != 0.0 else [],
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "output_head_untied": True,
        "selected_scale": scale,
        "selected_scale_report": selected,
        "optimization": optimization_summary,
        "config": asdict(config),
        "active_source": "RWKU calibration level1+level2 only",
        "protected_source": "unrelated MCF retain answers",
        "active_points": [point_report(point) for point in active_points],
        "protected_points": [
            point_report(point) for point in protected_points
        ],
        "scale_sweep": scale_reports,
        "optimization_log": optimization_log,
    }
    if output_dir is not None:
        from rwku_eval import write_json

        write_json(Path(output_dir) / "repair_report.json", report)
    return report
