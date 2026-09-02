#!/usr/bin/env python3
"""Pure primitives for MCF frozen-head projected row-partition rewiring V2.3.

V2.3 keeps the LM head frozen and bit-identical.  Only selected input-embedding
rows may change.  The three primitives that are new relative to V2/V2.1/V2.2 are

1. the *answer-contrast* readout direction ``q_i`` (not ``W[target_true]``),
2. the ``(efficacy, potential)`` row diagnostic that decides which rows may be
   edited at all, and
3. a role-aware projection that leaves forget-exclusive rows unconstrained,
   projects shared rows out of their retain-readout subspace, and holds excluded
   rows at exact zero.

Everything here is deterministic and model-free so it can be unit tested without
loading a checkpoint.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import mcf_biendpoint_nullspace_rewiring_v2_core as geometry


PROTOCOL = "mcf_projected_row_partition_embedding_rewiring_v2_3"
REACHABILITY_PROTOCOL = (
    "mcf_projected_row_partition_embedding_rewiring_v2_3_1"
)

FREE = "free"
PROJECTED = "projected"
EXCLUDED = "excluded"
ROLES = (FREE, PROJECTED, EXCLUDED)


def answer_contrast_directions(
    output_weight: torch.Tensor,
    true_ids: torch.Tensor,
    new_ids: torch.Tensor,
    *,
    epsilon: float = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``q_i = normalize(W[new_i] - W[true_i])`` and a validity mask.

    With a frozen LM head the trained margin is exactly

        ``log p(new_i) - log p(true_i) = h . (W[new_i] - W[true_i])``

    because the softmax normalizer cancels in the difference.  ``W[true_i]``
    alone is therefore the wrong axis: it ignores where the reference answer
    row sits and wastes the component the two rows share.

    CounterFact answers frequently share a leading token, so the difference can
    be exactly zero or numerically negligible at some positions.  Those
    positions are masked invalid rather than normalized into noise.  A negative
    id marks a position with no paired counterpart (answers of unequal length).
    """
    if output_weight.ndim != 2:
        raise ValueError("output weight must be a matrix")
    if true_ids.ndim != 1 or new_ids.shape != true_ids.shape:
        raise ValueError("contrast ids must be aligned vectors")
    hidden = int(output_weight.shape[1])
    count = int(true_ids.numel())
    directions = torch.zeros(
        (count, hidden), dtype=torch.float32, device=output_weight.device
    )
    valid = torch.zeros(count, dtype=torch.bool, device=output_weight.device)
    if count == 0:
        return directions, valid
    paired = (true_ids >= 0) & (new_ids >= 0) & (true_ids != new_ids)
    if not bool(paired.any().item()):
        return directions, valid
    positions = torch.nonzero(paired, as_tuple=False).flatten()
    local_true = true_ids.index_select(0, positions).to(output_weight.device)
    local_new = new_ids.index_select(0, positions).to(output_weight.device)
    true_rows = output_weight.index_select(0, local_true).float()
    new_rows = output_weight.index_select(0, local_new).float()
    difference = new_rows - true_rows
    norms = difference.norm(dim=1)
    reference = torch.maximum(true_rows.norm(dim=1), new_rows.norm(dim=1))
    usable = norms > float(epsilon) * reference.clamp_min(1e-20)
    keep = positions[usable]
    if keep.numel() == 0:
        return directions, valid
    directions.index_copy_(
        0, keep, F.normalize(difference[usable], dim=-1, eps=1e-12)
    )
    valid.index_fill_(0, keep, True)
    return directions, valid


def contrast_surgical_penalty(
    hidden: torch.Tensor,
    base_hidden: torch.Tensor,
    directions: torch.Tensor,
    valid: torch.Tensor,
    *,
    sign_margin: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Confine the hidden displacement to ``q_i`` and enforce its sign.

    The first term removes movement unrelated to the trained margin; the second
    is a hinge requiring the surviving component to point toward the reference
    answer.  Invalid positions contribute nothing.
    """
    if hidden.ndim != 2 or base_hidden.shape != hidden.shape:
        raise ValueError("hidden states must be aligned matrices")
    if directions.shape != hidden.shape or valid.shape[0] != hidden.shape[0]:
        raise ValueError("contrast directions are not aligned with hidden states")
    displacement = hidden.float() - base_hidden.to(
        device=hidden.device, dtype=torch.float32
    )
    unit = directions.to(device=hidden.device, dtype=torch.float32)
    aligned = (displacement * unit).sum(dim=-1)
    orthogonal = (displacement - aligned[:, None] * unit).square().sum(dim=-1)
    sign_hinge = F.relu(float(sign_margin) - aligned)
    mask = valid.to(device=hidden.device, dtype=torch.float32)
    denominator = mask.sum().clamp_min(1.0)
    penalty = ((orthogonal + sign_hinge) * mask).sum() / denominator
    counted = int(valid.sum().item())
    return penalty, {
        "valid_cells": counted,
        "total_cells": int(valid.numel()),
        "aligned_min": float(
            aligned[valid].min().detach().cpu() if counted else torch.zeros(())
        ),
        "orthogonal_mean": float(
            orthogonal[valid].mean().detach().cpu() if counted else torch.zeros(())
        ),
    }


def residual_efficacy(
    gradient: torch.Tensor,
    bases: Sequence[torch.Tensor],
    caps: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Per-row surviving forget gradient after retain-subspace projection.

    ``efficacy`` is the fraction of the row's forget gradient that survives
    projection.  ``potential`` is that surviving magnitude scaled by the row's
    own norm cap, i.e. how much margin the row could actually buy before it
    saturates.  A row can have efficacy near one and negligible potential; the
    partition needs both.
    """
    if gradient.ndim != 2 or len(bases) != int(gradient.shape[0]):
        raise ValueError("gradient rows and basis count differ")
    if caps.shape != (gradient.shape[0],):
        raise ValueError("caps must supply one value per row")
    raw = gradient.detach().float().clone()
    residual = raw.clone()
    geometry.project_rowwise_(residual, bases)
    raw_norm = raw.norm(dim=1)
    residual_norm = residual.norm(dim=1)
    efficacy = torch.where(
        raw_norm > 0,
        residual_norm / raw_norm.clamp_min(1e-20),
        torch.zeros_like(raw_norm),
    )
    potential = caps.to(device=raw.device, dtype=torch.float32) * residual_norm
    return {
        "raw_norm": raw_norm.cpu(),
        "residual_norm": residual_norm.cpu(),
        "efficacy": efficacy.cpu(),
        "potential": potential.cpu(),
    }


def role_constrained_gradient(
    gradient: torch.Tensor,
    bases: Sequence[torch.Tensor],
    roles: Sequence[str],
) -> torch.Tensor:
    """Return the locally usable part of a gradient under the row policy.

    The sign is deliberately unchanged: for a scalar margin ``m``, moving a
    row along this result increases the first-order approximation to ``m``.
    This helper applies the exact same exclusion/projection policy used for a
    trained cumulative delta, which prevents the reachability audit from
    crediting directions that optimization will subsequently remove.
    """
    if gradient.ndim != 2 or len(bases) != int(gradient.shape[0]):
        raise ValueError("gradient rows and basis count differ")
    if len(roles) != int(gradient.shape[0]):
        raise ValueError("gradient rows and role count differ")
    constrained = gradient.detach().float().clone()
    apply_role_constraints_(constrained, bases, roles)
    return constrained


def cap_aware_prompt_reachability(
    gradient: torch.Tensor,
    bases: Sequence[torch.Tensor],
    roles: Sequence[str],
    caps: torch.Tensor,
    *,
    base_margin: float,
    required_margin: float,
    tolerance: float = 1e-7,
) -> Dict[str, Any]:
    """First-order per-prompt reachability under every registered row cap.

    With independent L2 caps per physical embedding row, the exact maximum of
    the local linear model is

        ``sum_t cap_t * ||P_t grad_t margin||``.

    ``P_t`` is identity for free rows, the retain-nullspace projector for
    projected rows, and zero for excluded rows.  This is only a local upper
    bound for the nonlinear frozen Transformer, so the runner additionally
    performs a real forward directional sweep before allowing training.
    """
    if gradient.ndim != 2 or caps.shape != (gradient.shape[0],):
        raise ValueError("gradient and row caps are incompatible")
    usable = role_constrained_gradient(gradient, bases, roles)
    norms = usable.norm(dim=1)
    cap_values = caps.detach().to(device=usable.device, dtype=torch.float32)
    row_gains = cap_values * norms
    maximum_improvement = float(row_gains.sum().cpu())
    required_improvement = max(0.0, float(required_margin) - float(base_margin))
    editable_rows = int((norms > float(tolerance)).sum().item())
    predicted_margin = float(base_margin) + maximum_improvement
    passed = bool(
        required_improvement <= float(tolerance)
        or (
            editable_rows > 0
            and maximum_improvement + float(tolerance) >= required_improvement
        )
    )
    return {
        "base_margin": float(base_margin),
        "required_margin": float(required_margin),
        "required_improvement": required_improvement,
        "maximum_first_order_improvement": maximum_improvement,
        "predicted_maximum_margin": predicted_margin,
        "usable_gradient_norm": float(norms.norm().cpu()),
        "usable_gradient_rows": editable_rows,
        "raw_gradient_norm": float(gradient.detach().float().norm().cpu()),
        "passed": passed,
    }


def cap_saturating_margin_direction(
    gradient: torch.Tensor,
    bases: Sequence[torch.Tensor],
    roles: Sequence[str],
    caps: torch.Tensor,
) -> torch.Tensor:
    """Construct the cap-bounded direction maximizing the local margin.

    Each usable row receives its full independent L2 budget.  Multiplying the
    returned tensor by a factor in ``[0, 1]`` gives the registered nonlinear
    directional sweep used after the linear reachability calculation.
    """
    usable = role_constrained_gradient(gradient, bases, roles)
    if caps.shape != (usable.shape[0],):
        raise ValueError("caps must supply one value per row")
    norms = usable.norm(dim=1)
    scales = torch.where(
        norms > 0,
        caps.detach().to(device=usable.device, dtype=torch.float32)
        / norms.clamp_min(1e-20),
        torch.zeros_like(norms),
    )
    direction = usable * scales[:, None]
    apply_role_constraints_(direction, bases, roles)
    return direction


def accept_strict_trust_region_candidate(
    *,
    before_forget: float,
    candidate_forget: float,
    before_constraint_score: float,
    candidate_constraint_score: float,
    minimum_forget_improvement: float,
    tolerance: float = 1e-7,
) -> bool:
    """Accept only feasible candidates with a measurable forget improvement.

    V2.3 delegated this decision to V2.2, whose non-increasing predicate also
    accepted equality.  Projection could therefore erase the useful part of a
    proposal while ``accepted_factor=1`` was still reported.  V2.3.1 makes
    strict progress an invariant.  An infeasible state must improve both the
    constraint score and forgetting; a feasible state must remain feasible.
    """
    required = max(float(minimum_forget_improvement), float(tolerance))
    forget_improved = (
        float(before_forget) - float(candidate_forget) >= required
    )
    if not forget_improved:
        return False
    if float(before_constraint_score) <= 1.0 + float(tolerance):
        return bool(float(candidate_constraint_score) <= 1.0 + float(tolerance))
    return bool(
        float(candidate_constraint_score)
        < float(before_constraint_score) - float(tolerance)
    )


def partition_rows(
    *,
    row_ids: Sequence[int],
    retain_observed: Sequence[bool],
    efficacy: torch.Tensor,
    potential: torch.Tensor,
    frequency: torch.Tensor,
    direct_live_rows: Mapping[int, Sequence[int]],
    efficacy_min: float,
    potential_min: float,
    frequency_max: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Assign every selected row to ``free``/``projected``/``excluded``.

    A row unobserved by the retain bank *after targeted per-row corpus coverage*
    is forget-exclusive: no gradient path from a retain prompt reaches it, so it
    is edited without constraint.  An observed row is projected out of its
    retain-readout subspace, unless it is near-parallel to that subspace, has
    negligible cap-adjusted potential, or is too common to bound by projection
    alone -- those are excluded and held at exact zero.

    Direct-prompt liveness is then enforced: Eff is scored on the direct prompt,
    so a record whose every direct-prompt row was excluded would be unfixable by
    construction.  Such a record reinstates its highest-potential direct row as
    ``projected`` and is reported as liveness-forced higher-collateral.
    """
    count = len(row_ids)
    if len(retain_observed) != count:
        raise ValueError("retain observation flags are not aligned with rows")
    for name, tensor in (
        ("efficacy", efficacy),
        ("potential", potential),
        ("frequency", frequency),
    ):
        if tensor.ndim != 1 or int(tensor.numel()) != count:
            raise ValueError(f"{name} must supply one value per selected row")
    efficacy_cpu = efficacy.detach().float().cpu()
    potential_cpu = potential.detach().float().cpu()
    frequency_cpu = frequency.detach().float().cpu()
    roles: List[str] = []
    reasons: List[str] = []
    for index in range(count):
        if not bool(retain_observed[index]):
            roles.append(FREE)
            reasons.append("forget_exclusive_after_targeted_coverage")
        elif float(efficacy_cpu[index]) < float(efficacy_min):
            roles.append(EXCLUDED)
            reasons.append("near_parallel_to_retain_subspace")
        elif float(potential_cpu[index]) < float(potential_min):
            roles.append(EXCLUDED)
            reasons.append("negligible_cap_adjusted_potential")
        elif float(frequency_cpu[index]) > float(frequency_max):
            roles.append(EXCLUDED)
            reasons.append("common_row_beyond_projection_bound")
        else:
            roles.append(PROJECTED)
            reasons.append("shared_row_projected_out_of_retain_subspace")

    position = {int(value): index for index, value in enumerate(row_ids)}
    forced: List[Dict[str, Any]] = []
    for case_id in sorted(direct_live_rows):
        live = [
            position[int(value)]
            for value in direct_live_rows[case_id]
            if int(value) in position
        ]
        if not live:
            raise ValueError(
                f"case {case_id} has no selected row in its own direct prompt"
            )
        if any(roles[index] != EXCLUDED for index in live):
            continue
        best = max(live, key=lambda index: float(potential_cpu[index]))
        forced.append(
            {
                "case_id": int(case_id),
                "row_index": int(best),
                "token_id": int(row_ids[best]),
                "excluded_reason": reasons[best],
                "efficacy": float(efficacy_cpu[best]),
                "potential": float(potential_cpu[best]),
                "frequency": float(frequency_cpu[best]),
            }
        )
        roles[best] = PROJECTED
        reasons[best] = "direct_prompt_liveness_forced"

    per_row = [
        {
            "row_index": index,
            "token_id": int(row_ids[index]),
            "role": roles[index],
            "reason": reasons[index],
            "retain_observed": bool(retain_observed[index]),
            "efficacy": float(efficacy_cpu[index]),
            "potential": float(potential_cpu[index]),
            "frequency": float(frequency_cpu[index]),
        }
        for index in range(count)
    ]
    report = {
        "rows": count,
        "criterion": {
            "efficacy_min": float(efficacy_min),
            "potential_min": float(potential_min),
            "frequency_max": int(frequency_max),
        },
        "role_counts": {role: sum(value == role for value in roles) for role in ROLES},
        "editable_rows": sum(value != EXCLUDED for value in roles),
        "liveness_forced_records": len(forced),
        "liveness_forced": forced,
        "per_row": per_row,
        "passed": any(value != EXCLUDED for value in roles),
    }
    return roles, report


@torch.no_grad()
def apply_role_constraints_(
    delta: torch.Tensor, bases: Sequence[torch.Tensor], roles: Sequence[str]
) -> None:
    """Project shared rows, zero excluded rows, leave forget-exclusive rows free.

    This is a first-order guarantee only.  The frozen Transformer is nonlinear,
    so V2.3 always follows this with a forward-evaluated acceptance test; the
    projection is a warm start, never a certificate.
    """
    if delta.ndim != 2 or len(roles) != int(delta.shape[0]):
        raise ValueError("role count does not match delta rows")
    if len(bases) != int(delta.shape[0]):
        raise ValueError("basis count does not match delta rows")
    for index, role in enumerate(roles):
        if role == EXCLUDED:
            delta[index].zero_()
        elif role == PROJECTED:
            basis = bases[index]
            if basis.ndim != 2 or basis.shape[1] != delta.shape[1]:
                raise ValueError(f"invalid retain basis for row {index}")
            if basis.shape[0]:
                local = basis.to(device=delta.device, dtype=delta.dtype)
                delta[index].sub_((delta[index] @ local.transpose(0, 1)) @ local)
        elif role != FREE:
            raise ValueError(f"unknown row role: {role}")


def role_compliance_report(
    delta: torch.Tensor,
    bases: Sequence[torch.Tensor],
    roles: Sequence[str],
    *,
    tolerance: float = 1e-5,
) -> Dict[str, Any]:
    """Verify excluded rows stayed zero and projected rows stayed orthogonal."""
    if delta.ndim != 2 or len(roles) != int(delta.shape[0]):
        raise ValueError("role count does not match delta rows")
    values = delta.detach().float()
    excluded_violations: List[int] = []
    projected_residual = 0.0
    projected_violations: List[int] = []
    for index, role in enumerate(roles):
        row = values[index]
        if role == EXCLUDED:
            if float(row.norm().item()) > float(tolerance):
                excluded_violations.append(index)
        elif role == PROJECTED:
            basis = bases[index]
            if basis.shape[0] == 0:
                continue
            local = basis.to(device=row.device, dtype=torch.float32)
            leak = float((row @ local.transpose(0, 1)).norm().item())
            projected_residual = max(projected_residual, leak)
            if leak > float(tolerance):
                projected_violations.append(index)
    return {
        "tolerance": float(tolerance),
        "excluded_nonzero_rows": len(excluded_violations),
        "excluded_violating_rows": excluded_violations[:32],
        "projected_retain_leak_max": projected_residual,
        "projected_violating_rows": projected_violations[:32],
        "passed": not excluded_violations and not projected_violations,
    }


def diagnostic_prediction(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Registered pre-training prediction derived from the row partition.

    The V2.3 claim is that the partition diagnostic predicts when common
    subwords permit or prevent locality.  This records the prediction *before*
    any behavioural result exists, so it can be falsified afterwards rather
    than fitted to the outcome.
    """
    counts = dict(report.get("role_counts", {}))
    editable = int(report.get("editable_rows", 0))
    excluded = int(counts.get(EXCLUDED, 0))
    total = int(report.get("rows", 0))
    forced = int(report.get("liveness_forced_records", 0))
    excluded_fraction = float(excluded) / float(max(total, 1))
    return {
        "rows": total,
        "editable_rows": editable,
        "excluded_fraction": excluded_fraction,
        "liveness_forced_records": forced,
        "predicted_locality_risk": (
            "high" if forced > 0 or excluded_fraction > 0.5 else "low"
        ),
        "prediction_rule": (
            "records forced past exclusion by direct-prompt liveness, and runs "
            "whose excluded fraction exceeds one half, are predicted to carry "
            "the collateral that produced bimodal specificity in the "
            "subject-embedding lineage"
        ),
        "falsifiable_by": (
            "a separate official evaluation showing no association between "
            "liveness-forced records and per-record locality damage"
        ),
    }
