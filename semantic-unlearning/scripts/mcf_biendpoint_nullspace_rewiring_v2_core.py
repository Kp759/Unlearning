#!/usr/bin/env python3
"""Geometry and audit primitives for MCF bi-endpoint rewiring V2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import sure_canonical_core as canonical


PROTOCOL = "mcf_sparse_biendpoint_nullspace_rewiring_v2"


@dataclass(frozen=True)
class EndpointRows:
    input_ids: List[int]
    output_ids: List[int]
    input_owners: Dict[int, List[int]]
    output_roles: Dict[int, Dict[str, List[int]]]


def select_endpoint_rows(
    records: Sequence[Mapping[str, Any]], tok: Any, *, llama_like: bool
) -> EndpointRows:
    """Select complete-subject input rows and evaluated answer output rows.

    A vocabulary row occurs once even when it is shared by several facts or
    answer roles.  Ownership is metadata, never a duplicated parameter.
    """
    special = {
        int(value)
        for value in (
            getattr(tok, "pad_token_id", None),
            getattr(tok, "eos_token_id", None),
            getattr(tok, "bos_token_id", None),
            getattr(tok, "unk_token_id", None),
        )
        if value is not None
    }
    input_owners: Dict[int, set[int]] = {}
    output_roles: Dict[int, Dict[str, set[int]]] = {}
    for position, record in enumerate(records):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise ValueError(f"record {position} lacks requested_rewrite")
        case_id = int(record.get("case_id", position))
        subject_ids_set: set[int] = set()
        subject = str(rr.get("subject", "")).strip()
        for variant in (" " + subject, subject):
            ids = canonical.flat_ids(tok, variant)
            if llama_like and ids:
                ids = ids[1:]
            subject_ids_set.update(int(value) for value in ids)
        subject_ids = sorted(subject_ids_set - special)
        if not subject_ids:
            raise ValueError(f"case {case_id} has no editable complete-subject rows")
        for token_id in subject_ids:
            if token_id in special:
                raise ValueError(
                    f"case {case_id} selected special input row {token_id}"
                )
            input_owners.setdefault(int(token_id), set()).add(case_id)
        for field in ("target_true", "target_new"):
            block = rr.get(field)
            if not isinstance(block, Mapping) or not str(block.get("str", "")).strip():
                raise ValueError(f"case {case_id} lacks {field}.str")
            ids = canonical.answer_token_ids(
                tok, str(block["str"]), llama_like=llama_like
            )
            for token_id in ids:
                if token_id in special:
                    raise ValueError(
                        f"case {case_id} selected special output row {token_id}"
                    )
                roles = output_roles.setdefault(
                    int(token_id), {"target_true": set(), "target_new": set()}
                )
                roles[field].add(case_id)
    return EndpointRows(
        input_ids=sorted(input_owners),
        output_ids=sorted(output_roles),
        input_owners={key: sorted(value) for key, value in input_owners.items()},
        output_roles={
            key: {role: sorted(values) for role, values in roles.items()}
            for key, roles in output_roles.items()
        },
    )


def endpoint_overlap_report(rows: EndpointRows) -> Dict[str, Any]:
    shared_input = {
        str(token_id): owners
        for token_id, owners in rows.input_owners.items()
        if len(owners) > 1
    }
    cross_role_output = {
        str(token_id): roles
        for token_id, roles in rows.output_roles.items()
        if roles["target_true"] and roles["target_new"]
    }
    repeated_output = {
        str(token_id): roles
        for token_id, roles in rows.output_roles.items()
        if len(set(roles["target_true"] + roles["target_new"])) > 1
    }
    return {
        "input_rows": len(rows.input_ids),
        "output_rows": len(rows.output_ids),
        "shared_input_rows": len(shared_input),
        "shared_input_owners": shared_input,
        "cross_role_output_rows": len(cross_role_output),
        "cross_role_output_owners": cross_role_output,
        "multi_fact_output_rows": len(repeated_output),
        "multi_fact_output_owners": repeated_output,
        "one_delta_per_physical_input_row": len(rows.input_ids)
        == len(set(rows.input_ids)),
        "one_delta_per_physical_output_row": len(rows.output_ids)
        == len(set(rows.output_ids)),
    }


def bases_from_row_sketches(
    sketches: torch.Tensor, *, max_rank: int, epsilon: float = 1e-12
) -> List[torch.Tensor]:
    """Convert ``[sketch,row,hidden]`` protected gradients to row bases."""
    if sketches.ndim != 3:
        raise ValueError("row sketches must have [sketch,row,hidden] shape")
    result: List[torch.Tensor] = []
    for row in sketches.transpose(0, 1):
        keep = row.norm(dim=1) > float(epsilon)
        matrix = row[keep].float()
        if matrix.numel() == 0:
            result.append(row.new_empty((0, row.shape[-1]), dtype=torch.float32))
        else:
            result.append(canonical.orthonormal_row_basis(matrix, max_rank=max_rank))
    return result


def common_basis(rows: torch.Tensor, *, max_rank: int) -> torch.Tensor:
    if rows.ndim != 2:
        raise ValueError("protected hidden states must be a matrix")
    return canonical.orthonormal_row_basis(rows.float(), max_rank=max_rank)


def project_vector(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if vector.ndim != 1 or basis.ndim != 2 or basis.shape[1] != vector.shape[0]:
        raise ValueError("incompatible vector/basis shapes")
    if basis.shape[0] == 0:
        return vector
    return vector - (vector @ basis.transpose(0, 1)) @ basis


@torch.no_grad()
def project_rowwise_(delta: torch.Tensor, bases: Sequence[torch.Tensor]) -> None:
    if delta.ndim != 2 or len(bases) != delta.shape[0]:
        raise ValueError("rowwise basis count does not match delta")
    for index, basis in enumerate(bases):
        if basis.ndim != 2 or basis.shape[1] != delta.shape[1]:
            raise ValueError(f"invalid basis for row {index}")
        if basis.shape[0]:
            local = basis.to(device=delta.device, dtype=delta.dtype)
            delta[index].sub_((delta[index] @ local.transpose(0, 1)) @ local)


@torch.no_grad()
def project_common_(delta: torch.Tensor, basis: torch.Tensor) -> None:
    if delta.ndim != 2 or basis.ndim != 2 or basis.shape[1] != delta.shape[1]:
        raise ValueError("common basis is incompatible with delta")
    if basis.shape[0]:
        local = basis.to(device=delta.device, dtype=delta.dtype)
        delta.sub_((delta @ local.transpose(0, 1)) @ local)


def frequency_adjusted_caps(
    base_rows: torch.Tensor,
    counts: torch.Tensor,
    *,
    relative_cap: float,
    alpha: float,
) -> torch.Tensor:
    if base_rows.ndim != 2 or counts.ndim != 1 or counts.shape[0] != base_rows.shape[0]:
        raise ValueError("base rows/counts have incompatible shapes")
    return (
        float(relative_cap)
        * base_rows.float().norm(dim=1)
        / (1.0 + counts.float()).pow(float(alpha))
    )


def relative_caps(base_rows: torch.Tensor, *, relative_cap: float) -> torch.Tensor:
    if base_rows.ndim != 2:
        raise ValueError("base rows must be a matrix")
    return float(relative_cap) * base_rows.float().norm(dim=1)


@torch.no_grad()
def apply_row_caps_(delta: torch.Tensor, caps: torch.Tensor) -> None:
    if delta.ndim != 2 or caps.ndim != 1 or caps.shape[0] != delta.shape[0]:
        raise ValueError("delta/cap shapes are incompatible")
    norms = delta.norm(dim=1)
    scales = torch.minimum(
        torch.ones_like(norms),
        caps.to(device=delta.device, dtype=delta.dtype) / norms.clamp_min(1e-20),
    )
    delta.mul_(scales[:, None])


def cap_report(
    delta: torch.Tensor, caps: torch.Tensor, *, tolerance: float = 1e-6
) -> Dict[str, Any]:
    norms = delta.detach().float().norm(dim=1).cpu()
    caps_cpu = caps.detach().float().cpu()
    ratios = norms / caps_cpu.clamp_min(1e-20)
    return {
        "rows": int(delta.shape[0]),
        "norm_max": float(norms.max().item()) if norms.numel() else 0.0,
        "cap_max": float(caps_cpu.max().item()) if caps_cpu.numel() else 0.0,
        "relative_to_cap_max": float(ratios.max().item()) if ratios.numel() else 0.0,
        "violations": int((norms > caps_cpu + float(tolerance)).sum().item()),
        "passed": bool(torch.all(norms <= caps_cpu + float(tolerance)).item()),
    }


def protection_loss(
    current_logits: torch.Tensor,
    *,
    topk_ids: torch.Tensor,
    base_topk_log_probs: torch.Tensor,
    base_top1_ids: torch.Tensor,
    base_top1_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-prompt top-k KL and Base-top1 log-probability drift."""
    if current_logits.ndim != 2 or topk_ids.ndim != 2:
        raise ValueError("protection logits/ids must be matrices")
    selected = current_logits.float().gather(1, topk_ids)
    current_topk_log_probs = F.log_softmax(selected, dim=1)
    base_probs = base_topk_log_probs.exp()
    kl = (base_probs * (base_topk_log_probs - current_topk_log_probs)).sum(dim=1)
    current_full_log_probs = F.log_softmax(current_logits.float(), dim=1)
    current_top1 = current_full_log_probs.gather(1, base_top1_ids[:, None]).squeeze(1)
    drift = current_top1 - base_top1_log_probs
    return kl, drift


def protection_report(
    kl: torch.Tensor,
    drift: torch.Tensor,
    *,
    kl_mean_max: float,
    kl_absolute_max: float,
    top1_abs_max: float,
) -> Dict[str, Any]:
    if kl.numel() == 0 or drift.numel() == 0:
        raise ValueError("protection certificate cannot be empty")
    kl_cpu = kl.detach().float().cpu()
    drift_cpu = drift.detach().float().abs().cpu()
    mean = float(kl_cpu.mean().item())
    maximum = float(kl_cpu.max().item())
    drift_max = float(drift_cpu.max().item())
    return {
        "prompts": int(kl_cpu.numel()),
        "topk_kl_mean": mean,
        "topk_kl_max": maximum,
        "top1_logprob_abs_max": drift_max,
        "criterion": {
            "topk_kl_mean_max": float(kl_mean_max),
            "topk_kl_absolute_max": float(kl_absolute_max),
            "top1_logprob_abs_max": float(top1_abs_max),
        },
        "passed": bool(
            mean <= float(kl_mean_max)
            and maximum <= float(kl_absolute_max)
            and drift_max <= float(top1_abs_max)
        ),
    }


def projected_gradient_norms(
    input_gradient: torch.Tensor,
    output_gradient: torch.Tensor,
    *,
    input_bases: Sequence[torch.Tensor],
    output_basis: torch.Tensor,
) -> tuple[float, float, float]:
    input_copy = input_gradient.detach().float().clone()
    output_copy = output_gradient.detach().float().clone()
    project_rowwise_(input_copy, input_bases)
    project_common_(output_copy, output_basis)
    input_norm = float(input_copy.norm().item())
    output_norm = float(output_copy.norm().item())
    return (
        input_norm,
        output_norm,
        float((input_copy.square().sum() + output_copy.square().sum()).sqrt().item()),
    )


def select_development_candidate(reports: Sequence[Mapping[str, Any]]) -> int | None:
    """Choose a passing checkpoint by minimum total delta norm, then step."""
    passing = [report for report in reports if bool(report.get("passed"))]
    if not passing:
        return None
    selected = min(
        passing,
        key=lambda report: (float(report["total_delta_norm"]), int(report["step"])),
    )
    return int(selected["step"])
