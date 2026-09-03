#!/usr/bin/env python3
"""Pure primitives for MCF joint forget/retain endpoint rewiring V2.2."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import torch

import mcf_biendpoint_nullspace_rewiring_v2_core as geometry
import sure_canonical_core as canonical


PROTOCOL = "mcf_joint_forget_retain_endpoint_rewiring_v2_2"


def expanded_endpoint_rows(
    records: Sequence[Mapping[str, Any]], tok: Any, *, llama_like: bool
) -> tuple[geometry.EndpointRows, Dict[str, Any]]:
    """Select subject plus complete rendered-question input rows.

    The physical row remains unique.  Expanding beyond subject rows gives the
    frozen Transformer relation-frame capacity while ownership remains fully
    auditable.
    """
    base = geometry.select_endpoint_rows(records, tok, llama_like=llama_like)
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
    owners: Dict[int, set[int]] = {
        int(token_id): set(values) for token_id, values in base.input_owners.items()
    }
    subject_rows = set(base.input_ids)
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        case_id = int(record.get("case_id", position))
        prompt = str(rr["prompt"]).format(str(rr["subject"]))
        ids = canonical.flat_ids(tok, prompt)
        if llama_like and ids:
            ids = ids[1:]
        for token_id in set(int(value) for value in ids) - special:
            owners.setdefault(token_id, set()).add(case_id)
    rows = geometry.EndpointRows(
        input_ids=sorted(owners),
        output_ids=list(base.output_ids),
        input_owners={key: sorted(value) for key, value in owners.items()},
        output_roles=base.output_roles,
    )
    added = set(rows.input_ids) - subject_rows
    return rows, {
        "subject_input_rows": len(subject_rows),
        "rendered_question_input_rows": len(rows.input_ids),
        "relation_frame_rows_added": len(added),
        "added_row_ids": sorted(added),
        "one_delta_per_physical_input_row": len(rows.input_ids)
        == len(set(rows.input_ids)),
    }


@dataclass
class PersistentHardTail:
    capacity: int
    indices: List[int]

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("hard-tail capacity must be positive")
        self.capacity = int(capacity)
        self.indices = []

    def refresh(
        self,
        kl: torch.Tensor,
        drift: torch.Tensor,
        target_drift: torch.Tensor,
        *,
        kl_limit: float,
        drift_limit: float,
        target_drift_limit: float,
        add: int,
    ) -> Dict[str, Any]:
        if kl.ndim != 1 or drift.shape != kl.shape or target_drift.shape != kl.shape:
            raise ValueError("hard-tail metrics must be aligned vectors")
        normalized = torch.maximum(
            kl.float() / float(kl_limit), drift.float().abs() / float(drift_limit)
        )
        normalized = torch.maximum(
            normalized,
            target_drift.float().abs() / float(target_drift_limit),
        )
        take = min(int(add), int(normalized.numel()))
        newest = [
            index
            for index, _ in sorted(
                enumerate(normalized.cpu().tolist()),
                key=lambda item: (-float(item[1]), int(item[0])),
            )[:take]
        ]
        scores = {int(index): float(normalized[index].item()) for index in self.indices}
        for index in newest:
            scores[int(index)] = float(normalized[index].item())
        self.indices = [
            index
            for index, _ in sorted(
                scores.items(), key=lambda item: (-item[1], item[0])
            )[: self.capacity]
        ]
        return {
            "capacity": self.capacity,
            "size": len(self.indices),
            "newest": newest,
            "indices": list(self.indices),
            "global_score_max": float(normalized.max().item()),
            "global_kl_max": float(kl.max().item()),
            "global_top1_abs_max": float(drift.abs().max().item()),
            "global_target_logprob_abs_max": float(target_drift.abs().max().item()),
        }


def compose_active_retain_indices(
    *,
    random_indices: Sequence[int],
    overlap_indices: Sequence[int],
    hard_indices: Sequence[int],
    maximum: int,
) -> List[int]:
    """Hard tail first, then explicit overlaps, then ordinary retain replay."""
    ordered: List[int] = []
    for group in (hard_indices, overlap_indices, random_indices):
        for value in group:
            index = int(value)
            if index not in ordered:
                ordered.append(index)
            if len(ordered) >= int(maximum):
                return ordered
    return ordered


def constraint_score(
    *,
    kl_max: float,
    drift_max: float,
    target_drift_max: float = 0.0,
    kl_limit: float,
    drift_limit: float,
    target_drift_limit: float = 1.0,
) -> float:
    return max(
        float(kl_max) / float(kl_limit),
        float(drift_max) / float(drift_limit),
        float(target_drift_max) / float(target_drift_limit),
    )


def accept_trust_region_candidate(
    *,
    before_forget: float,
    candidate_forget: float,
    before_constraint_score: float,
    candidate_constraint_score: float,
    tolerance: float = 1e-7,
) -> bool:
    """Feasible steps improve forgetting; repair steps improve preservation."""
    if before_constraint_score <= 1.0:
        return bool(
            candidate_constraint_score <= 1.0
            and candidate_forget <= before_forget + float(tolerance)
        )
    return bool(
        candidate_constraint_score < before_constraint_score - float(tolerance)
        and candidate_forget <= before_forget * 1.01 + float(tolerance)
    )


def normalized_row_step(
    gradient: torch.Tensor, caps: torch.Tensor, *, fraction: float
) -> torch.Tensor:
    if gradient.ndim != 2 or caps.shape != (gradient.shape[0],):
        raise ValueError("gradient and row caps are incompatible")
    norms = gradient.float().norm(dim=1)
    budgets = float(fraction) * caps.to(gradient.device, dtype=torch.float32)
    relative_strength = (norms / norms.max().clamp_min(1e-20)).clamp_min(0.0).sqrt()
    scales = torch.where(
        norms > 0,
        budgets * relative_strength / norms.clamp_min(1e-20),
        torch.zeros_like(norms),
    )
    return -gradient.float() * scales[:, None]
