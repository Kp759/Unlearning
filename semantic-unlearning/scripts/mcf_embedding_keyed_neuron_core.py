#!/usr/bin/env python3
"""Pure mechanisms for embedding-keyed sparse SwiGLU conditional suppression.

The end-to-end experiment lives in
``mcf_embedding_keyed_neuron_erasure.py``.  This module contains the small,
auditable pieces that can be tested without loading a language model:

* greedy selection of record-owned neurons whose activations respond to the
  frozen sparse embedding writer but remain quiet on writer-off contexts;
* the historical exact sparse parameterization of existing SwiGLU rows and
  columns, retained for lineage and unit tests;
* V3.5's isolated thresholded residual branch, which reads frozen selected
  gate/up features while leaving the ordinary Base MLP path untouched;
* contextual code responses and detector-gate metrics;
* hard relative-norm projection and materialization/restoration helpers.

No tokenizer expansion, subject-string lookup, retrieval cache, LoRA, or
LM-head edit is implemented here. V3.5 deliberately introduces an internal
activation threshold and explicit sparse residual branch. That branch is a
training-only architecture test and cannot be represented as an ordinary
replacement of the selected model rows/columns; ``materialize`` therefore
refuses it instead of making a false existing-weight claim.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_5_1"


def _as_float_matrix(value: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise ValueError(f"{name} must be a rank-two tensor")
    if not torch.isfinite(value.float()).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().float()


def _quantile(values: torch.Tensor, fraction: float) -> torch.Tensor:
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("quantile input must be a non-empty vector")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("quantile fraction must lie in (0, 1]")
    index = min(values.numel() - 1, max(0, math.ceil(values.numel() * fraction) - 1))
    return values.sort().values[index]


def select_record_owned_neurons(
    writer_activations: Sequence[torch.Tensor],
    writer_off_activations: Sequence[torch.Tensor],
    protected_activations: torch.Tensor,
    *,
    neurons_per_record: int,
    dormant_fraction: float,
    stability_weight: float = 1.0,
    used_neurons: Sequence[int] = (),
    selection_mode: str = "writer_contrastive",
    context_negative_activations: Sequence[torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
) -> Tuple[List[List[int]], List[torch.Tensor], List[Dict[str, Any]]]:
    """Select disjoint neurons with stable writer-induced activation gain.

    ``writer_activations[i]`` and ``writer_off_activations[i]`` are paired
    positive-context activation matrices for record ``i``.  Candidates are
    restricted to the lowest writer-off protected RMS fraction, then ranked by

    ``abs(mean(writer - writer_off)) / (protected_rms + w * delta_std)``.

    ``base_context_contrastive`` is the matched no-writer control.  It never
    observes an embedding delta: candidates are ranked by the difference
    between base-model positive and compositional-negative activation means,
    under the same dormant-neuron, count, layer, and norm budgets.

    The returned signs convert each selected mean displacement to a positive
    code component.  Ownership is disjoint by construction, which makes the
    causal ablations and scaling cost explicit.
    """
    if selection_mode not in {
        "writer_contrastive",
        "base_context_contrastive",
        "dormant_random",
    }:
        raise ValueError(f"unsupported selection mode: {selection_mode!r}")
    if len(writer_activations) != len(writer_off_activations):
        raise ValueError("writer-on and writer-off record groups must match")
    if not writer_activations:
        raise ValueError("at least one record is required")
    if selection_mode == "base_context_contrastive":
        if context_negative_activations is None or len(
            context_negative_activations
        ) != len(writer_activations):
            raise ValueError(
                "base_context_contrastive requires one negative activation "
                "matrix per record"
            )
    if int(neurons_per_record) <= 0:
        raise ValueError("neurons_per_record must be positive")
    protected = _as_float_matrix(protected_activations, "protected_activations")
    intermediate = int(protected.shape[1])
    protected_rms = protected.square().mean(dim=0).sqrt()
    dormant_threshold = _quantile(protected_rms, float(dormant_fraction))
    dormant = protected_rms <= dormant_threshold
    unavailable = torch.zeros(intermediate, dtype=torch.bool)
    for neuron in used_neurons:
        if int(neuron) < 0 or int(neuron) >= intermediate:
            raise ValueError(f"used neuron {neuron} is out of range")
        unavailable[int(neuron)] = True

    ownership: List[List[int]] = []
    signs: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    eps = 1e-8
    for record_index, (writer, writer_off) in enumerate(
        zip(writer_activations, writer_off_activations)
    ):
        writer = _as_float_matrix(writer, f"writer_activations[{record_index}]")
        writer_off = _as_float_matrix(
            writer_off, f"writer_off_activations[{record_index}]"
        )
        if writer.shape != writer_off.shape or writer.shape[1] != intermediate:
            raise ValueError("paired record activations have incompatible shapes")
        if selection_mode == "base_context_contrastive":
            assert context_negative_activations is not None
            negative = _as_float_matrix(
                context_negative_activations[record_index],
                f"context_negative_activations[{record_index}]",
            )
            if negative.shape[1] != intermediate or negative.shape[0] == 0:
                raise ValueError(
                    "base-context negative activations have incompatible shape"
                )
            positive_centered = writer - writer.mean(dim=0, keepdim=True)
            negative_centered = negative - negative.mean(dim=0, keepdim=True)
            mean = writer.mean(dim=0) - negative.mean(dim=0)
            stability = torch.cat([positive_centered, negative_centered], dim=0).std(
                dim=0, unbiased=False
            )
            score_kind = "base_positive_minus_context_negative"
        else:
            displacement = writer - writer_off
            mean = displacement.mean(dim=0)
            stability = displacement.std(dim=0, unbiased=False)
            score_kind = "writer_on_minus_writer_off"
        denominator = protected_rms + float(stability_weight) * stability + eps
        score = mean.abs() / denominator
        eligible = dormant & ~unavailable & torch.isfinite(score)
        candidate_count = int(eligible.sum())
        if candidate_count < int(neurons_per_record):
            raise RuntimeError(
                f"record {record_index} has only {candidate_count} unused dormant "
                f"neurons, needs {neurons_per_record}"
            )
        if selection_mode in {"writer_contrastive", "base_context_contrastive"}:
            masked = score.masked_fill(~eligible, float("-inf"))
            chosen = masked.topk(int(neurons_per_record)).indices.sort().values
        else:
            candidates = eligible.nonzero(as_tuple=False).reshape(-1)
            order = torch.randperm(int(candidates.numel()), generator=generator)
            chosen = (
                candidates.index_select(0, order[: int(neurons_per_record)])
                .sort()
                .values
            )
        unavailable[chosen] = True
        chosen_mean = mean.index_select(0, chosen)
        chosen_signs = torch.where(
            chosen_mean >= 0,
            torch.ones_like(chosen_mean),
            -torch.ones_like(chosen_mean),
        )
        ids = [int(x) for x in chosen.tolist()]
        ownership.append(ids)
        signs.append(chosen_signs)
        reports.append(
            {
                "record_index": record_index,
                "selection_mode": selection_mode,
                "selection_score_kind": score_kind,
                "selected_neurons": ids,
                "selected_scores": [float(x) for x in score.index_select(0, chosen)],
                "selection_contrast_mean": [float(x) for x in chosen_mean],
                "selection_contrast_std": [
                    float(x) for x in stability.index_select(0, chosen)
                ],
                "protected_rms": [
                    float(x) for x in protected_rms.index_select(0, chosen)
                ],
                "dormant_threshold": float(dormant_threshold),
                "candidate_count": candidate_count,
            }
        )
    return ownership, signs, reports


def flatten_ownership(
    ownership: Sequence[Sequence[int]], signs: Sequence[torch.Tensor]
) -> Tuple[List[int], torch.Tensor, List[List[int]]]:
    if len(ownership) != len(signs):
        raise ValueError("ownership and sign groups must match")
    flat: List[int] = []
    flat_signs: List[torch.Tensor] = []
    local_groups: List[List[int]] = []
    seen: set[int] = set()
    for neurons, sign in zip(ownership, signs):
        ids = [int(x) for x in neurons]
        if not ids:
            raise ValueError("every record must own at least one neuron")
        if any(neuron in seen for neuron in ids):
            raise ValueError("record neuron ownership must be disjoint")
        if sign.ndim != 1 or int(sign.numel()) != len(ids):
            raise ValueError("each sign vector must match its neuron group")
        start = len(flat)
        flat.extend(ids)
        flat_signs.append(sign.detach().float())
        local_groups.append(list(range(start, start + len(ids))))
        seen.update(ids)
    return flat, torch.cat(flat_signs), local_groups


def contextual_code_responses(
    edited_activations: torch.Tensor,
    writer_off_baseline: torch.Tensor,
    local_groups: Sequence[Sequence[int]],
    flat_signs: torch.Tensor,
) -> torch.Tensor:
    """Decode record codes from selected-neuron activation displacements."""
    if (
        edited_activations.ndim != 2
        or writer_off_baseline.shape != edited_activations.shape
    ):
        raise ValueError("edited and writer-off activations must be equal matrices")
    if flat_signs.ndim != 1 or flat_signs.numel() != edited_activations.shape[1]:
        raise ValueError("flat signs must cover every selected neuron")
    signed = (edited_activations - writer_off_baseline) * flat_signs.to(
        device=edited_activations.device, dtype=edited_activations.dtype
    )
    columns: List[torch.Tensor] = []
    for group in local_groups:
        index = torch.tensor(
            [int(x) for x in group], dtype=torch.long, device=edited_activations.device
        )
        if index.numel() == 0:
            raise ValueError("code group cannot be empty")
        columns.append(signed.index_select(1, index).mean(dim=1))
    return torch.stack(columns, dim=1)


def signed_group_activations(
    activations: torch.Tensor,
    local_groups: Sequence[Sequence[int]],
    flat_signs: torch.Tensor,
) -> torch.Tensor:
    """Return the actual continuous gate amplitude for each neuron group.

    Unlike ``contextual_code_responses``, this statistic does not subtract a
    prompt-specific Base activation that is unavailable at inference time.
    Training detector locality on this absolute activation makes the audited
    response the quantity that really multiplies the edited down columns.
    """

    if activations.ndim != 2:
        raise ValueError("activations must be a rank-two matrix")
    if flat_signs.ndim != 1 or flat_signs.numel() != activations.shape[1]:
        raise ValueError("flat signs must cover every selected neuron")
    signed = activations * flat_signs.to(
        device=activations.device, dtype=activations.dtype
    )
    columns: List[torch.Tensor] = []
    for group in local_groups:
        index = torch.tensor(
            [int(value) for value in group],
            dtype=torch.long,
            device=activations.device,
        )
        if index.numel() == 0:
            raise ValueError("activation group cannot be empty")
        columns.append(signed.index_select(1, index).mean(dim=1))
    return torch.stack(columns, dim=1)


def detector_objective(
    responses: torch.Tensor,
    owners: torch.Tensor,
    positive: torch.Tensor,
    *,
    positive_target: float,
    off_target_abs_max: float,
    tail_k: int,
    negative_weight: float,
    cross_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return an equal-record, gate-aligned detector objective.

    The training-only detector certificate reduces positive contexts with a
    minimum and off-context responses with a maximum absolute value.  A plain
    prompt mean can therefore look healthy while a single context still
    rejects the record.  This objective mirrors those reductions: each record
    contributes a prompt mean plus the mean of its ``tail_k`` largest squared
    violations.  Negative and cross-record responses are penalized only above
    the registered optimization target instead of being wastefully driven to
    zero.  These targets are deliberately separate from the final detector
    certificate so training can maintain a preregistered safety margin.

    ``responses`` may contain a bounded microbatch of record groups, but every
    represented record receives equal weight regardless of how many contexts
    it owns.  The caller is responsible for accumulating these record means
    over all records before taking one optimizer step.
    """
    if responses.ndim != 2:
        raise ValueError("responses must be [batch, records]")
    batch, records = responses.shape
    if owners.shape != (batch,) or positive.shape != (batch,):
        raise ValueError("owners and positive flags must match response batch")
    if bool((owners < 0).any()) or bool((owners >= records).any()):
        raise ValueError("owner index out of range")
    if batch == 0:
        raise ValueError("detector objective requires at least one response")
    if float(positive_target) < 0 or float(off_target_abs_max) < 0:
        raise ValueError("detector training targets must be non-negative")
    if int(tail_k) <= 0:
        raise ValueError("tail_k must be positive")

    def mean_plus_tail(
        squared_violations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        values = squared_violations.reshape(-1)
        if values.numel() == 0:
            zero = responses.sum() * 0.0
            return zero, zero, zero
        mean = values.mean()
        count = min(int(tail_k), int(values.numel()))
        tail = values.topk(count).values.mean()
        return mean + tail, mean, tail

    rows = torch.arange(batch, device=responses.device)
    owned = responses[rows, owners]
    record_ids = owners.unique(sorted=True)
    write_rows: List[torch.Tensor] = []
    write_mean_rows: List[torch.Tensor] = []
    write_tail_rows: List[torch.Tensor] = []
    negative_rows: List[torch.Tensor] = []
    negative_mean_rows: List[torch.Tensor] = []
    negative_tail_rows: List[torch.Tensor] = []
    cross_rows: List[torch.Tensor] = []
    cross_mean_rows: List[torch.Tensor] = []
    cross_tail_rows: List[torch.Tensor] = []
    consistency_rows: List[torch.Tensor] = []
    for record_id_tensor in record_ids:
        record_id = int(record_id_tensor.item())
        record_mask = owners.eq(record_id)
        record_positive = record_mask & positive.bool()
        record_negative = record_mask & ~positive.bool()
        if not bool(record_positive.any()):
            raise ValueError(f"record {record_id} has no positive detector context")

        write, write_mean, write_tail = mean_plus_tail(
            F.relu(float(positive_target) - owned[record_positive]).square()
        )
        negative, negative_mean, negative_tail = mean_plus_tail(
            F.relu(owned[record_negative].abs() - float(off_target_abs_max)).square()
        )
        write_rows.append(write)
        write_mean_rows.append(write_mean)
        write_tail_rows.append(write_tail)
        negative_rows.append(negative)
        negative_mean_rows.append(negative_mean)
        negative_tail_rows.append(negative_tail)

        if records > 1:
            nonowner_columns = torch.ones(
                records, dtype=torch.bool, device=responses.device
            )
            nonowner_columns[record_id] = False
            cross_values = responses[record_mask][:, nonowner_columns]
            cross, cross_mean, cross_tail = mean_plus_tail(
                F.relu(cross_values.abs() - float(off_target_abs_max)).square()
            )
        else:
            cross = cross_mean = cross_tail = responses.sum() * 0.0
        cross_rows.append(cross)
        cross_mean_rows.append(cross_mean)
        cross_tail_rows.append(cross_tail)

        positive_values = owned[record_positive]
        consistency_rows.append(
            positive_values.var(unbiased=False)
            if positive_values.numel() > 1
            else responses.sum() * 0.0
        )

    write = torch.stack(write_rows).mean()
    negative = torch.stack(negative_rows).mean()
    cross = torch.stack(cross_rows).mean()
    consistency = torch.stack(consistency_rows).mean()
    total = write + float(negative_weight) * negative + float(cross_weight) * cross
    return total, {
        "write": write,
        "write_mean": torch.stack(write_mean_rows).mean(),
        "write_tail": torch.stack(write_tail_rows).mean(),
        "negative": negative,
        "negative_mean": torch.stack(negative_mean_rows).mean(),
        "negative_tail": torch.stack(negative_tail_rows).mean(),
        "cross": cross,
        "cross_mean": torch.stack(cross_mean_rows).mean(),
        "cross_tail": torch.stack(cross_tail_rows).mean(),
        "consistency": consistency,
    }


def detector_writer_off_objective(
    responses: torch.Tensor,
    owners: torch.Tensor,
    *,
    off_target_abs_max: float,
    tail_k: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize only certificate-breaking owned writer-off responses.

    Each record is weighted equally and contributes the mean squared excess
    above ``off_target_abs_max`` plus its worst-``tail_k`` mean.  Harmless
    nonzero activations inside the registered optimization target receive
    exactly zero loss.
    """
    if responses.ndim != 2:
        raise ValueError("responses must be [batch, records]")
    batch, records = responses.shape
    if owners.shape != (batch,):
        raise ValueError("owners must match response batch")
    if batch == 0:
        raise ValueError("writer-off objective requires at least one response")
    if bool((owners < 0).any()) or bool((owners >= records).any()):
        raise ValueError("owner index out of range")
    if float(off_target_abs_max) < 0:
        raise ValueError("off_target_abs_max must be non-negative")
    if int(tail_k) <= 0:
        raise ValueError("tail_k must be positive")

    rows = torch.arange(batch, device=responses.device)
    owned = responses[rows, owners]
    totals: List[torch.Tensor] = []
    means: List[torch.Tensor] = []
    tails: List[torch.Tensor] = []
    for record_id_tensor in owners.unique(sorted=True):
        record_id = int(record_id_tensor.item())
        values = F.relu(
            owned[owners.eq(record_id)].abs() - float(off_target_abs_max)
        ).square()
        mean = values.mean()
        count = min(int(tail_k), int(values.numel()))
        tail = values.topk(count).values.mean()
        totals.append(mean + tail)
        means.append(mean)
        tails.append(tail)
    return torch.stack(totals).mean(), {
        "writer_off_mean": torch.stack(means).mean(),
        "writer_off_tail": torch.stack(tails).mean(),
    }


def mean_plus_tail_squared_loss(
    squared_violations: torch.Tensor,
    *,
    tail_k: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Reduce one record's squared violations by mean plus worst-``k`` mean."""

    values = squared_violations.reshape(-1)
    if values.numel() == 0:
        raise ValueError("tail-aware loss requires at least one value")
    if int(tail_k) <= 0:
        raise ValueError("tail_k must be positive")
    if not torch.isfinite(values).all():
        raise ValueError("tail-aware loss received non-finite values")
    mean = values.mean()
    count = min(int(tail_k), int(values.numel()))
    tail = values.topk(count).values.mean()
    return mean + tail, {"mean": mean, "tail": tail}


def actuator_positive_margin_objective(
    margins: torch.Tensor,
    *,
    margin_floor: float,
    tail_k: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Gate-align a single record's actuator loss to its worst positive margins."""

    if float(margin_floor) < 0:
        raise ValueError("margin_floor must be non-negative")
    return mean_plus_tail_squared_loss(
        F.relu(float(margin_floor) - margins).square(),
        tail_k=int(tail_k),
    )


def paired_on_off_ratios(
    writer_on: torch.Tensor,
    writer_off: torch.Tensor,
    *,
    epsilon: float,
) -> Dict[str, torch.Tensor]:
    """Return stable paired writer-on/off amplitude and leakage ratios.

    The detector certificate constrains absolute activation extrema, but an
    actuator multiplies the remaining writer-off activation by its learned
    down column.  These paired ratios expose that multiplicative selectivity
    directly.  ``epsilon`` is recorded by callers and only prevents division
    by zero; it is not an acceptance tolerance.
    """

    if writer_on.shape != writer_off.shape:
        raise ValueError("writer-on and writer-off tensors must have equal shape")
    if writer_on.numel() == 0:
        raise ValueError("paired ratio audit requires at least one value")
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("ratio epsilon must be finite and positive")
    on = writer_on.detach().float()
    off = writer_off.detach().float()
    if not torch.isfinite(on).all() or not torch.isfinite(off).all():
        raise ValueError("paired ratio audit received non-finite values")
    on_abs = on.abs()
    off_abs = off.abs()
    return {
        "writer_on_abs": on_abs,
        "writer_off_abs": off_abs,
        "writer_on_minus_writer_off": on - off,
        "writer_on_abs_minus_writer_off_abs": on_abs - off_abs,
        "writer_on_to_off_ratio": on_abs / off_abs.clamp_min(float(epsilon)),
        "writer_off_to_on_fraction": off_abs / on_abs.clamp_min(float(epsilon)),
    }


def actuator_cap_sweep_decision(
    cap_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply the preregistered smallest-positive-reachable-cap decision rule."""

    if not cap_rows:
        raise ValueError("actuator cap sweep requires at least one completed cap")
    caps = [float(row["cap"]) for row in cap_rows]
    if any(not math.isfinite(cap) or cap <= 0.0 for cap in caps):
        raise ValueError("actuator sweep caps must be positive and finite")
    if caps != sorted(set(caps)):
        raise ValueError("actuator sweep rows must have strictly increasing caps")
    reachable = [
        float(row["cap"]) for row in cap_rows if bool(row["positive_reachable"])
    ]
    structurally_selective = [
        float(row["cap"])
        for row in cap_rows
        if bool(row["positive_reachable"])
        and bool(row["writer_off_structural_selectivity_passed"])
    ]
    selected = min(reachable) if reachable else None
    selected_structural = (
        min(structurally_selective) if structurally_selective else None
    )
    conclusion = (
        "isolated_threshold_branch_not_positive_reachable_at_registered_cap"
        if selected is None
        else (
            "isolated_positive_reachability_without_writer_off_selectivity"
            if selected_structural is None
            else "isolated_positive_reachability_and_writer_off_selectivity_passed"
        )
    )
    return {
        "selected_smallest_positive_reachable_cap": selected,
        "positive_reachability_passed": selected is not None,
        "smallest_jointly_structurally_selective_cap": selected_structural,
        "structural_selectivity_passed": selected_structural is not None,
        "conclusion": conclusion,
    }


def actuator_reference_regression_objective(
    current_nll: torch.Tensor,
    baseline_nll: torch.Tensor,
    *,
    tolerance: float,
    tail_k: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Penalize a record's reference-NLL regressions only beyond tolerance."""

    if current_nll.shape != baseline_nll.shape:
        raise ValueError("current and baseline reference NLLs must match")
    if float(tolerance) < 0:
        raise ValueError("reference NLL tolerance must be non-negative")
    return mean_plus_tail_squared_loss(
        F.relu(current_nll - baseline_nll - float(tolerance)).square(),
        tail_k=int(tail_k),
    )


def actuator_writer_off_objective(
    current_new_nll: torch.Tensor,
    current_true_nll: torch.Tensor,
    baseline_new_nll: torch.Tensor,
    baseline_true_nll: torch.Tensor,
    *,
    tolerance: float,
    tail_k: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Tail-penalize writer-off NLL drift only outside a registered band."""

    if not (
        current_new_nll.shape
        == current_true_nll.shape
        == baseline_new_nll.shape
        == baseline_true_nll.shape
    ):
        raise ValueError("writer-off current and baseline NLL tensors must match")
    if float(tolerance) < 0:
        raise ValueError("writer-off NLL tolerance must be non-negative")
    violations = torch.cat(
        (
            F.relu(
                (current_new_nll - baseline_new_nll).abs() - float(tolerance)
            ).square(),
            F.relu(
                (current_true_nll - baseline_true_nll).abs() - float(tolerance)
            ).square(),
        )
    )
    return mean_plus_tail_squared_loss(violations, tail_k=int(tail_k))


def actuator_negative_preservation_objective(
    current_new_nll: torch.Tensor,
    current_true_nll: torch.Tensor,
    baseline_new_nll: torch.Tensor,
    baseline_true_nll: torch.Tensor,
    *,
    tail_k: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Equal-record mean-plus-tail preservation for compositional negatives."""

    if not (
        current_new_nll.shape
        == current_true_nll.shape
        == baseline_new_nll.shape
        == baseline_true_nll.shape
    ):
        raise ValueError("negative current and baseline NLL tensors must match")
    violations = torch.cat(
        (
            (current_new_nll - baseline_new_nll).square(),
            (current_true_nll - baseline_true_nll).square(),
        )
    )
    return mean_plus_tail_squared_loss(violations, tail_k=int(tail_k))


def detector_gate_report(
    positive_responses: Sequence[torch.Tensor],
    negative_responses: Sequence[torch.Tensor],
    writer_off_responses: Sequence[torch.Tensor],
    *,
    positive_floor: float,
    off_abs_max: float,
    require_writer_off: bool = True,
    comparison_abs_tolerance: float = 0.0,
) -> Dict[str, Any]:
    if not (
        len(positive_responses) == len(negative_responses) == len(writer_off_responses)
    ):
        raise ValueError("detector response groups must match")
    if float(positive_floor) < 0 or float(off_abs_max) < 0:
        raise ValueError("detector certificate thresholds must be non-negative")
    if float(comparison_abs_tolerance) < 0:
        raise ValueError("comparison_abs_tolerance must be non-negative")
    tolerance = float(comparison_abs_tolerance)
    per_record: List[Dict[str, Any]] = []
    for index, (positive, negative, writer_off) in enumerate(
        zip(positive_responses, negative_responses, writer_off_responses)
    ):
        positive = positive.detach().float().reshape(-1)
        negative = negative.detach().float().reshape(-1)
        writer_off = writer_off.detach().float().reshape(-1)
        if positive.numel() == 0:
            raise ValueError("every detector needs a positive response")
        positive_min = float(positive.min())
        negative_max = float(negative.abs().max()) if negative.numel() else 0.0
        writer_off_max = float(writer_off.abs().max()) if writer_off.numel() else 0.0
        positive_passed = bool(positive_min + tolerance >= float(positive_floor))
        negative_passed = bool(negative_max <= float(off_abs_max) + tolerance)
        writer_off_passed = bool(
            not bool(require_writer_off)
            or writer_off_max <= float(off_abs_max) + tolerance
        )
        passed = bool(positive_passed and negative_passed and writer_off_passed)
        per_record.append(
            {
                "record_index": index,
                "positive_min": positive_min,
                "positive_median": float(positive.median()),
                "positive_max": float(positive.max()),
                "negative_abs_max": negative_max,
                "writer_off_abs_max": writer_off_max,
                "positive_passed": positive_passed,
                "negative_passed": negative_passed,
                "writer_off_passed": writer_off_passed,
                "passed": passed,
            }
        )
    return {
        "criterion": {
            "positive_floor": float(positive_floor),
            "negative_abs_max": float(off_abs_max),
            "writer_off_abs_max": float(off_abs_max),
            "writer_off_required": bool(require_writer_off),
            "comparison_abs_tolerance": tolerance,
            "comparison_policy": {
                "positive": (
                    "positive_min + comparison_abs_tolerance >= positive_floor"
                ),
                "negative": (
                    "observed_negative_abs_max <= negative_abs_max + "
                    "comparison_abs_tolerance"
                ),
                "writer_off": (
                    "observed_writer_off_abs_max <= writer_off_abs_max + "
                    "comparison_abs_tolerance"
                ),
            },
        },
        "passed_records": sum(int(row["passed"]) for row in per_record),
        "total_records": len(per_record),
        "passed": bool(per_record and all(row["passed"] for row in per_record)),
        "failure_counts": {
            "positive": sum(int(not row["positive_passed"]) for row in per_record),
            "negative": sum(int(not row["negative_passed"]) for row in per_record),
            "writer_off": sum(int(not row["writer_off_passed"]) for row in per_record),
        },
        "per_record": per_record,
    }


@dataclass
class SparseNeuronWeights:
    gate_rows: torch.Tensor
    up_rows: torch.Tensor
    down_columns: torch.Tensor

    def cpu(self) -> "SparseNeuronWeights":
        return SparseNeuronWeights(
            self.gate_rows.detach().cpu(),
            self.up_rows.detach().cpu(),
            self.down_columns.detach().cpu(),
        )


def sparse_neuron_weights(
    mlp: nn.Module, neuron_ids: Sequence[int]
) -> SparseNeuronWeights:
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, name):
            raise ValueError(f"MLP lacks required SwiGLU component {name!r}")
    ids = torch.tensor(
        [int(x) for x in neuron_ids],
        dtype=torch.long,
        device=mlp.gate_proj.weight.device,
    )
    if ids.numel() == 0:
        raise ValueError("at least one selected neuron is required")
    intermediate = int(mlp.gate_proj.weight.shape[0])
    if bool((ids < 0).any()) or bool((ids >= intermediate).any()):
        raise ValueError("selected neuron is out of range")
    return SparseNeuronWeights(
        gate_rows=mlp.gate_proj.weight.index_select(0, ids).detach().clone(),
        up_rows=mlp.up_proj.weight.index_select(0, ids).detach().clone(),
        down_columns=mlp.down_proj.weight.index_select(1, ids).detach().clone(),
    )


class SparseSwiGLUNeuronEditor(nn.Module):
    """Sparse SwiGLU feature editor with legacy and isolated residual modes.

    The legacy mode replaces only the selected-neuron contribution:

    ``base_selected_contribution -> edited_selected_contribution``.

    V3.5 instead leaves the Base MLP output untouched, evaluates imported
    detector features as a passive readout, and adds a threshold-gated residual
    through ``down_delta``. Only the legacy mode is materializable into the
    original projection rows/columns.
    """

    def __init__(self, mlp: nn.Module, neuron_ids: Sequence[int]) -> None:
        super().__init__()
        self.neuron_ids = [int(x) for x in neuron_ids]
        base = sparse_neuron_weights(mlp, self.neuron_ids)
        device = mlp.gate_proj.weight.device
        self.register_buffer(
            "base_gate_rows", base.gate_rows.detach().float().to(device)
        )
        self.register_buffer("base_up_rows", base.up_rows.detach().float().to(device))
        self.register_buffer(
            "base_down_columns", base.down_columns.detach().float().to(device)
        )
        self.gate_delta = nn.Parameter(torch.zeros_like(self.base_gate_rows))
        self.up_delta = nn.Parameter(torch.zeros_like(self.base_up_rows))
        self.down_delta = nn.Parameter(torch.zeros_like(self.base_down_columns))
        self.enabled = True
        self.write_enabled = True
        self.capture_activations = False
        self.last_edited_activations: torch.Tensor | None = None
        self.residual_mode = "replace_selected_neuron_contribution"
        self.threshold_gate_off_boundary = 0.0
        self.threshold_gate_on_boundary = 1.0
        self.threshold_local_groups: List[List[int]] = []
        self.register_buffer(
            "threshold_flat_signs",
            torch.empty(0, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "threshold_neuron_owners",
            torch.empty(0, dtype=torch.long, device=device),
        )
        self._act_fn = mlp.act_fn
        self._handle: Any = None

    @property
    def trainable_parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def edited_weights(self) -> SparseNeuronWeights:
        return SparseNeuronWeights(
            self.base_gate_rows + self.gate_delta,
            self.base_up_rows + self.up_delta,
            self.base_down_columns + self.down_delta,
        )

    def selected_activations(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = hidden.float()
        base_gate = F.linear(x, self.base_gate_rows)
        base_up = F.linear(x, self.base_up_rows)
        base_activation = self._act_fn(base_gate) * base_up
        edited_activation = self.edited_selected_activations(x)
        return base_activation, edited_activation

    def edited_selected_activations(self, hidden: torch.Tensor) -> torch.Tensor:
        """Evaluate only the trainable selected gate/up rows on cached input."""
        x = hidden.float()
        edited_gate = F.linear(x, self.base_gate_rows + self.gate_delta)
        edited_up = F.linear(x, self.base_up_rows + self.up_delta)
        return self._act_fn(edited_gate) * edited_up

    def configure_isolated_threshold_residual(
        self,
        local_groups: Sequence[Sequence[int]],
        flat_signs: torch.Tensor,
        *,
        off_boundary: float,
        on_boundary: float,
    ) -> None:
        """Make detector rows a passive readout for an additive residual branch.

        The ordinary MLP output remains untouched.  The edited detector
        activations produce one signed response per record; a clipped linear
        threshold converts the registered off/on certificate gap into a gate in
        ``[0, 1]``.  Only ``gate * activation * down_delta`` is added to the
        residual stream, so an exact-zero ``down_delta`` is an algebraic identity
        regardless of the imported gate/up tensors.
        """

        if not math.isfinite(float(off_boundary)) or not math.isfinite(
            float(on_boundary)
        ):
            raise ValueError("threshold-gate boundaries must be finite")
        if float(on_boundary) <= float(off_boundary):
            raise ValueError("threshold-gate on boundary must exceed off boundary")
        groups = [[int(value) for value in group] for group in local_groups]
        if not groups or any(not group for group in groups):
            raise ValueError("threshold gate requires nonempty record groups")
        flattened = [value for group in groups for value in group]
        expected = list(range(len(self.neuron_ids)))
        if sorted(flattened) != expected or len(set(flattened)) != len(expected):
            raise ValueError(
                "threshold groups must partition every selected neuron exactly once"
            )
        signs = flat_signs.detach().float().reshape(-1)
        if int(signs.numel()) != len(self.neuron_ids):
            raise ValueError("threshold signs must cover every selected neuron")
        if not bool(torch.isfinite(signs).all()) or not bool(
            torch.all(signs.abs().eq(1.0))
        ):
            raise ValueError("threshold signs must be finite values in {-1, +1}")
        owners = torch.empty(len(self.neuron_ids), dtype=torch.long)
        for owner, group in enumerate(groups):
            owners[torch.tensor(group, dtype=torch.long)] = int(owner)
        self.threshold_local_groups = groups
        self.threshold_flat_signs = signs.to(self.base_gate_rows.device)
        self.threshold_neuron_owners = owners.to(self.base_gate_rows.device)
        self.threshold_gate_off_boundary = float(off_boundary)
        self.threshold_gate_on_boundary = float(on_boundary)
        self.residual_mode = "isolated_thresholded_residual"

    def thresholded_group_gates_from_activations(
        self, edited_activation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return signed group responses and their clipped certificate gates."""

        if self.residual_mode != "isolated_thresholded_residual":
            raise RuntimeError("isolated threshold residual mode is not configured")
        if int(edited_activation.shape[-1]) != len(self.neuron_ids):
            raise ValueError("edited activations do not cover the selected neurons")
        leading_shape = edited_activation.shape[:-1]
        flat = edited_activation.reshape(-1, len(self.neuron_ids))
        responses = signed_group_activations(
            flat,
            self.threshold_local_groups,
            self.threshold_flat_signs,
        )
        width = self.threshold_gate_on_boundary - self.threshold_gate_off_boundary
        gates = ((responses - self.threshold_gate_off_boundary) / width).clamp(
            min=0.0, max=1.0
        )
        group_count = len(self.threshold_local_groups)
        return (
            responses.reshape(*leading_shape, group_count),
            gates.reshape(*leading_shape, group_count),
        )

    def isolated_actuator_features_from_activations(
        self, edited_activation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gate each selected activation by its record's thresholded response."""

        responses, gates = self.thresholded_group_gates_from_activations(
            edited_activation
        )
        expanded = gates.index_select(-1, self.threshold_neuron_owners.to(gates.device))
        return edited_activation * expanded, responses, gates

    def isolated_actuator_features(
        self, hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edited_activation = self.edited_selected_activations(hidden)
        return self.isolated_actuator_features_from_activations(edited_activation)

    def _hook(
        self, _module: nn.Module, inputs: Any, output: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled:
            self.last_edited_activations = None
            return output
        hidden = inputs[0]
        if self.residual_mode == "isolated_thresholded_residual":
            edited_activation = self.edited_selected_activations(hidden)
            if self.capture_activations:
                self.last_edited_activations = edited_activation
            else:
                self.last_edited_activations = None
            if not self.write_enabled:
                return output
            (
                actuator_features,
                _responses,
                _gates,
            ) = self.isolated_actuator_features_from_activations(edited_activation)
            correction = F.linear(actuator_features, self.down_delta)
            return output + correction.to(dtype=output.dtype)
        base_activation, edited_activation = self.selected_activations(hidden)
        if self.capture_activations:
            self.last_edited_activations = edited_activation
        else:
            self.last_edited_activations = None
        if not self.write_enabled:
            return output
        edited = self.edited_weights()
        base_contribution = F.linear(base_activation, self.base_down_columns)
        edited_contribution = F.linear(edited_activation, edited.down_columns)
        correction = edited_contribution - base_contribution
        return output + correction.to(dtype=output.dtype)

    def install(self, mlp: nn.Module) -> Any:
        if self._handle is not None:
            raise RuntimeError("sparse neuron editor hook is already installed")
        self._handle = mlp.register_forward_hook(self._hook)
        return self._handle

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    @torch.no_grad()
    def zero_deltas(self) -> None:
        self.gate_delta.zero_()
        self.up_delta.zero_()
        self.down_delta.zero_()

    @torch.no_grad()
    def clamp_relative_(
        self,
        *,
        detector_cap: float,
        actuator_cap: float,
    ) -> Dict[str, float]:
        def clamp_rows(
            delta: torch.Tensor, base: torch.Tensor, cap: float, dim: int
        ) -> torch.Tensor:
            base_norm = base.norm(dim=dim).clamp_min(1e-12)
            delta_norm = delta.norm(dim=dim)
            scale = torch.minimum(
                torch.ones_like(delta_norm),
                float(cap) * base_norm / delta_norm.clamp_min(1e-12),
            )
            if dim == 1:
                delta.mul_(scale.unsqueeze(1))
            elif dim == 0:
                delta.mul_(scale.unsqueeze(0))
            else:
                raise ValueError("unsupported norm dimension")
            return delta_norm / base_norm

        clamp_rows(self.gate_delta, self.base_gate_rows, float(detector_cap), 1)
        clamp_rows(self.up_delta, self.base_up_rows, float(detector_cap), 1)
        clamp_rows(self.down_delta, self.base_down_columns, float(actuator_cap), 0)
        report = self.relative_norm_report()
        report["detector_cap"] = float(detector_cap)
        report["actuator_cap"] = float(actuator_cap)
        return report

    @torch.no_grad()
    def clamp_down_relative_(self, actuator_cap: float) -> Dict[str, float]:
        """Project only actuator columns, leaving a frozen detector bit-exact."""

        base_norm = self.base_down_columns.norm(dim=0).clamp_min(1e-12)
        delta_norm = self.down_delta.norm(dim=0)
        scale = torch.minimum(
            torch.ones_like(delta_norm),
            float(actuator_cap) * base_norm / delta_norm.clamp_min(1e-12),
        )
        self.down_delta.mul_(scale.unsqueeze(0))
        report = self.relative_norm_report()
        report["actuator_cap"] = float(actuator_cap)
        return report

    def relative_norm_report(self) -> Dict[str, float]:
        gate = self.gate_delta.detach().norm(dim=1) / self.base_gate_rows.norm(
            dim=1
        ).clamp_min(1e-12)
        up = self.up_delta.detach().norm(dim=1) / self.base_up_rows.norm(
            dim=1
        ).clamp_min(1e-12)
        down = self.down_delta.detach().norm(dim=0) / self.base_down_columns.norm(
            dim=0
        ).clamp_min(1e-12)
        return {
            "gate_max_relative_norm": float(gate.max()),
            "up_max_relative_norm": float(up.max()),
            "down_max_relative_norm": float(down.max()),
            "gate_mean_relative_norm": float(gate.mean()),
            "up_mean_relative_norm": float(up.mean()),
            "down_mean_relative_norm": float(down.mean()),
        }

    def down_relative_norms(self) -> torch.Tensor:
        """Return one actuator relative-norm ratio per selected down column."""

        return self.down_delta.detach().norm(dim=0) / self.base_down_columns.norm(
            dim=0
        ).clamp_min(1e-12)

    @torch.no_grad()
    def materialize(self, mlp: nn.Module) -> SparseNeuronWeights:
        if self.residual_mode == "isolated_thresholded_residual":
            raise RuntimeError(
                "the isolated thresholded residual is an explicit internal branch "
                "and cannot be represented by replacing existing SwiGLU rows"
            )
        ids = torch.tensor(
            self.neuron_ids, dtype=torch.long, device=mlp.gate_proj.weight.device
        )
        edited = self.edited_weights()
        mlp.gate_proj.weight.index_copy_(
            0, ids, edited.gate_rows.to(mlp.gate_proj.weight.dtype)
        )
        mlp.up_proj.weight.index_copy_(
            0, ids, edited.up_rows.to(mlp.up_proj.weight.dtype)
        )
        mlp.down_proj.weight.index_copy_(
            1, ids, edited.down_columns.to(mlp.down_proj.weight.dtype)
        )
        return sparse_neuron_weights(mlp, self.neuron_ids)


@torch.no_grad()
def replace_sparse_neuron_weights(
    mlp: nn.Module,
    neuron_ids: Sequence[int],
    values: SparseNeuronWeights,
) -> None:
    ids = torch.tensor(
        [int(x) for x in neuron_ids],
        dtype=torch.long,
        device=mlp.gate_proj.weight.device,
    )
    count = int(ids.numel())
    hidden = int(mlp.gate_proj.weight.shape[1])
    if values.gate_rows.shape != (count, hidden):
        raise ValueError("gate row replacement has incompatible shape")
    if values.up_rows.shape != (count, hidden):
        raise ValueError("up row replacement has incompatible shape")
    if values.down_columns.shape != (hidden, count):
        raise ValueError("down column replacement has incompatible shape")
    mlp.gate_proj.weight.index_copy_(
        0, ids, values.gate_rows.to(mlp.gate_proj.weight.dtype)
    )
    mlp.up_proj.weight.index_copy_(0, ids, values.up_rows.to(mlp.up_proj.weight.dtype))
    mlp.down_proj.weight.index_copy_(
        1, ids, values.down_columns.to(mlp.down_proj.weight.dtype)
    )


class ToggleableEmbeddingDelta:
    """Ordinary sparse embedding-row hook with an ablation switch."""

    def __init__(
        self,
        input_layer: nn.Module,
        row_ids: Sequence[int],
        delta: torch.Tensor,
    ) -> None:
        if not hasattr(input_layer, "weight"):
            raise ValueError("input embedding module must expose weight")
        if delta.ndim != 2 or delta.shape[0] != len(row_ids):
            raise ValueError("embedding row ids and delta do not match")
        self.enabled = True
        device = input_layer.weight.device
        vocab = int(input_layer.weight.shape[0])
        self.lookup = torch.full((vocab,), -1, dtype=torch.long, device=device)
        ids = torch.tensor([int(x) for x in row_ids], dtype=torch.long, device=device)
        if ids.numel():
            self.lookup[ids] = torch.arange(ids.numel(), device=device)
        self.ids = ids
        self.delta = delta.detach().float().to(device)
        self.handle = input_layer.register_forward_hook(self._hook)

    def _hook(
        self, _module: nn.Module, inputs: Any, output: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled or self.ids.numel() == 0:
            return output
        token_ids = inputs[0].to(self.lookup.device)
        local = self.lookup[token_ids]
        mask = local.ge(0)
        if not bool(mask.any()):
            return output
        safe = local.clamp_min(0)
        correction = self.delta.index_select(0, safe.reshape(-1)).reshape(
            *safe.shape, self.delta.shape[-1]
        )
        correction = correction * mask.unsqueeze(-1)
        return output + correction.to(device=output.device, dtype=output.dtype)

    def remove(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
