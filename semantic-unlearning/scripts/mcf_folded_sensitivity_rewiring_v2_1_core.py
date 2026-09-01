#!/usr/bin/env python3
"""Deterministic folded-classifier primitives for MCF V2.1."""
from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Dict, List, Mapping, Sequence

import torch


PROTOCOL = "mcf_folded_sensitivity_biendpoint_rewiring_v2_1"


@dataclass(frozen=True)
class FoldedCells:
    hidden: torch.Tensor
    row_indices: torch.Tensor
    signs: torch.Tensor
    case_ids: torch.Tensor
    roles: List[str]


class BalancedBatcher:
    """Deterministic epoch-balanced indices; every item appears once per epoch."""

    def __init__(self, count: int, batch_size: int, seed: int) -> None:
        if count <= 0 or batch_size <= 0:
            raise ValueError("balanced batcher requires positive count and batch size")
        self.count = int(count)
        self.batch_size = min(int(batch_size), self.count)
        self.rng = random.Random(int(seed))
        self.order = list(range(self.count))
        self.cursor = self.count

    def next(self) -> List[int]:
        if self.cursor >= self.count:
            self.rng.shuffle(self.order)
            self.cursor = 0
        end = min(self.cursor + self.batch_size, self.count)
        result = self.order[self.cursor : end]
        self.cursor = end
        if len(result) < self.batch_size:
            self.rng.shuffle(self.order)
            need = self.batch_size - len(result)
            result.extend(self.order[:need])
            self.cursor = need
        return result


def safe_hidden(hidden: torch.Tensor, protected_basis: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 2 or protected_basis.ndim != 2:
        raise ValueError("hidden and protected basis must be matrices")
    if hidden.shape[1] != protected_basis.shape[1]:
        raise ValueError("hidden and protected basis dimensions differ")
    if protected_basis.shape[0] == 0:
        return hidden.float()
    basis = protected_basis.to(device=hidden.device, dtype=torch.float32)
    values = hidden.float()
    return values - (values @ basis.transpose(0, 1)) @ basis


def solve_folded_rows(
    cells: FoldedCells,
    *,
    n_rows: int,
    hidden_size: int,
    protected_basis: torch.Tensor,
    correction_floor: float,
    ridge: float,
    row_caps: torch.Tensor,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Minimum-norm signed contextual classifier for each physical head row."""
    if cells.hidden.ndim != 2 or cells.hidden.shape[1] != hidden_size:
        raise ValueError("folded cell hidden states have the wrong shape")
    if cells.row_indices.numel() != cells.hidden.shape[0]:
        raise ValueError("folded cell rows are not aligned")
    if cells.signs.shape != cells.row_indices.shape:
        raise ValueError("folded cell signs are not aligned")
    if row_caps.shape != (n_rows,):
        raise ValueError("folded row caps have the wrong shape")
    projected = safe_hidden(cells.hidden, protected_basis)
    delta = torch.zeros(
        (n_rows, hidden_size), dtype=torch.float32, device=cells.hidden.device
    )
    per_row: List[Dict[str, Any]] = []
    all_signed: List[torch.Tensor] = []
    for row_index in range(n_rows):
        mask = cells.row_indices.eq(row_index)
        local_hidden = projected[mask]
        local_signs = cells.signs[mask].float()
        if local_hidden.shape[0] == 0:
            per_row.append(
                {
                    "row_index": row_index,
                    "cells": 0,
                    "positive_cells": 0,
                    "negative_cells": 0,
                    "signed_min": None,
                    "cap_ratio": 0.0,
                    "passed": True,
                }
            )
            continue
        targets = local_signs * float(correction_floor)
        gram = local_hidden @ local_hidden.transpose(0, 1)
        gram = gram + float(ridge) * torch.eye(
            gram.shape[0], dtype=gram.dtype, device=gram.device
        )
        try:
            coefficients = torch.linalg.solve(gram, targets)
        except torch.linalg.LinAlgError:
            coefficients = torch.linalg.pinv(gram) @ targets
        vector = local_hidden.transpose(0, 1) @ coefficients
        cap = row_caps[row_index].to(vector)
        norm = vector.norm()
        if norm > cap:
            vector = vector * (cap / norm.clamp_min(1e-20))
        delta[row_index] = vector
        signed = local_signs * (local_hidden @ vector)
        all_signed.append(signed)
        per_row.append(
            {
                "row_index": row_index,
                "cells": int(mask.sum().item()),
                "positive_cells": int((local_signs > 0).sum().item()),
                "negative_cells": int((local_signs < 0).sum().item()),
                "signed_min": float(signed.min().detach().cpu()),
                "signed_median": float(signed.median().detach().cpu()),
                "norm": float(vector.norm().detach().cpu()),
                "cap": float(cap.detach().cpu()),
                "cap_ratio": float(
                    (vector.norm() / cap.clamp_min(1e-20)).detach().cpu()
                ),
                "passed": bool(torch.all(signed >= 0.1).item()),
            }
        )
    signed_all = torch.cat(all_signed) if all_signed else torch.empty(0)
    report = {
        "correction_floor": float(correction_floor),
        "ridge": float(ridge),
        "rows": n_rows,
        "cells": int(cells.hidden.shape[0]),
        "cross_role_rows": sum(
            row["positive_cells"] > 0 and row["negative_cells"] > 0 for row in per_row
        ),
        "signed_min": float(signed_all.min().detach().cpu())
        if signed_all.numel()
        else None,
        "signed_median": float(signed_all.median().detach().cpu())
        if signed_all.numel()
        else None,
        "signed_failures_at_0_1": int((signed_all < 0.1).sum().item()),
        "cap_saturated_rows": sum(
            float(row["cap_ratio"]) >= 1.0 - 1e-6 for row in per_row
        ),
        "delta_norm": float(delta.norm().detach().cpu()),
        "per_row": per_row,
    }
    report["passed"] = report["signed_failures_at_0_1"] == 0
    return delta, report


def _extend_orthonormal_basis(
    basis: torch.Tensor, vectors: torch.Tensor, *, tolerance: float = 1e-6
) -> torch.Tensor:
    """Rebuild an orthonormal row span after adding protected vectors."""
    combined = torch.cat((basis.float(), vectors.float()), dim=0)
    if combined.shape[0] == 0:
        return combined
    orthogonal, triangular = torch.linalg.qr(combined.transpose(0, 1), mode="reduced")
    diagonal = triangular.diagonal().abs()
    scale = diagonal.max().clamp_min(1.0)
    independent = diagonal > float(tolerance) * scale
    if bool(independent.all().item()):
        return orthogonal.transpose(0, 1).contiguous()
    # QR without pivoting can place dependent columns between independent ones;
    # an SVD fallback gives the correct span in that uncommon case.
    _, singular_values, right = torch.linalg.svd(combined, full_matrices=False)
    rank = int(
        (singular_values > float(tolerance) * singular_values.max().clamp_min(1.0))
        .sum()
        .item()
    )
    return right[:rank].contiguous()


def solve_hard_tail_folded_rows(
    cells: FoldedCells,
    *,
    protected_hidden: torch.Tensor,
    n_rows: int,
    hidden_size: int,
    correction_floor: float,
    ridge: float,
    row_caps: torch.Tensor,
    hard_tail_rounds: int,
    hard_tail_per_round: int,
    protected_correction_max: float,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Solve each row while mining its worst protected hidden-state tail.

    Unlike a shared low-rank sketch, every selected output row gets its own
    exact protected nullspace.  The worst still-affected fit states are added
    monotonically, so a later round cannot forget an earlier protection cell.
    """
    if protected_hidden.ndim != 2 or protected_hidden.shape[1] != hidden_size:
        raise ValueError("protected hidden states have the wrong shape")
    if hard_tail_rounds <= 0 or hard_tail_per_round <= 0:
        raise ValueError("hard-tail schedule must be positive")
    protected = protected_hidden.float()
    delta = torch.zeros(
        (n_rows, hidden_size), dtype=torch.float32, device=cells.hidden.device
    )
    per_row: List[Dict[str, Any]] = []
    all_signed: List[torch.Tensor] = []
    for row_index in range(n_rows):
        mask = cells.row_indices.eq(row_index)
        local_hidden = cells.hidden[mask].float()
        local_signs = cells.signs[mask].float()
        if local_hidden.shape[0] == 0:
            per_row.append(
                {
                    "row_index": row_index,
                    "cells": 0,
                    "positive_cells": 0,
                    "negative_cells": 0,
                    "signed_min": None,
                    "protected_basis_rank": 0,
                    "protected_correction_max": 0.0,
                    "protected_correction_failures": 0,
                    "cap_ratio": 0.0,
                    "passed": True,
                }
            )
            continue
        basis = torch.empty(
            (0, hidden_size), dtype=torch.float32, device=local_hidden.device
        )
        selected_protection = torch.zeros(
            protected.shape[0], dtype=torch.bool, device=protected.device
        )
        vector = torch.zeros(
            hidden_size, dtype=torch.float32, device=local_hidden.device
        )
        signed = torch.zeros_like(local_signs)
        protected_corrections = torch.zeros(
            protected.shape[0], dtype=torch.float32, device=protected.device
        )
        rounds_used = 0
        for round_index in range(hard_tail_rounds + 1):
            safe_local = safe_hidden(local_hidden, basis)
            targets = local_signs * float(correction_floor)
            gram = safe_local @ safe_local.transpose(0, 1)
            gram = gram + float(ridge) * torch.eye(
                gram.shape[0], dtype=gram.dtype, device=gram.device
            )
            try:
                coefficients = torch.linalg.solve(gram, targets)
            except torch.linalg.LinAlgError:
                coefficients = torch.linalg.pinv(gram) @ targets
            vector = safe_local.transpose(0, 1) @ coefficients
            vector = safe_hidden(vector.unsqueeze(0), basis).squeeze(0)
            cap = row_caps[row_index].to(vector)
            norm = vector.norm()
            if norm > cap:
                vector = vector * (cap / norm.clamp_min(1e-20))
            signed = local_signs * (local_hidden @ vector)
            protected_corrections = protected @ vector
            violations = protected_corrections.abs() > float(protected_correction_max)
            if not bool(violations.any().item()) or round_index == hard_tail_rounds:
                rounds_used = round_index
                break
            candidates = violations & ~selected_protection
            scores = protected_corrections.abs().masked_fill(~candidates, float("-inf"))
            available = int(candidates.sum().item())
            take = min(hard_tail_per_round, available)
            if take == 0:
                rounds_used = round_index
                break
            indices = torch.topk(scores, k=take).indices
            selected_protection[indices] = True
            basis = _extend_orthonormal_basis(basis, protected.index_select(0, indices))
        delta[row_index] = vector
        all_signed.append(signed)
        cap = row_caps[row_index].to(vector)
        protected_failures = int(
            (protected_corrections.abs() > float(protected_correction_max)).sum().item()
        )
        per_row.append(
            {
                "row_index": row_index,
                "cells": int(mask.sum().item()),
                "positive_cells": int((local_signs > 0).sum().item()),
                "negative_cells": int((local_signs < 0).sum().item()),
                "signed_min": float(signed.min().detach().cpu()),
                "signed_median": float(signed.median().detach().cpu()),
                "protected_basis_rank": int(basis.shape[0]),
                "protected_cells_selected": int(selected_protection.sum().item()),
                "protected_correction_max": float(
                    protected_corrections.abs().max().detach().cpu()
                ),
                "protected_correction_failures": protected_failures,
                "hard_tail_rounds_used": rounds_used,
                "norm": float(vector.norm().detach().cpu()),
                "cap": float(cap.detach().cpu()),
                "cap_ratio": float(
                    (vector.norm() / cap.clamp_min(1e-20)).detach().cpu()
                ),
                "passed": bool(
                    torch.all(signed >= 0.1).item() and protected_failures == 0
                ),
            }
        )
    signed_all = torch.cat(all_signed) if all_signed else torch.empty(0)
    report = {
        "solver": "row_specific_monotone_hard_tail_nullspace",
        "correction_floor": float(correction_floor),
        "ridge": float(ridge),
        "rows": n_rows,
        "cells": int(cells.hidden.shape[0]),
        "protected_hidden_states": int(protected.shape[0]),
        "protected_correction_maximum": float(protected_correction_max),
        "hard_tail_rounds": int(hard_tail_rounds),
        "hard_tail_per_round": int(hard_tail_per_round),
        "cross_role_rows": sum(
            row["positive_cells"] > 0 and row["negative_cells"] > 0 for row in per_row
        ),
        "signed_min": float(signed_all.min().detach().cpu())
        if signed_all.numel()
        else None,
        "signed_failures_at_0_1": int((signed_all < 0.1).sum().item()),
        "protected_correction_failures": sum(
            int(row["protected_correction_failures"]) for row in per_row
        ),
        "protected_correction_global_max": max(
            float(row["protected_correction_max"]) for row in per_row
        ),
        "cap_saturated_rows": sum(
            float(row["cap_ratio"]) >= 1.0 - 1e-6 for row in per_row
        ),
        "delta_norm": float(delta.norm().detach().cpu()),
        "per_row": per_row,
    }
    report["passed"] = bool(
        report["signed_failures_at_0_1"] == 0
        and report["protected_correction_failures"] == 0
    )
    return delta, report


def signed_cell_report(
    cells: FoldedCells, delta: torch.Tensor, *, minimum: float
) -> Dict[str, Any]:
    corrections = (cells.hidden.float() * delta.index_select(0, cells.row_indices)).sum(
        dim=1
    )
    signed = cells.signs.float() * corrections
    failures = signed < float(minimum)
    return {
        "cells": int(signed.numel()),
        "minimum": float(signed.min().detach().cpu()),
        "median": float(signed.median().detach().cpu()),
        "failures": int(failures.sum().item()),
        "criterion": {"minimum": float(minimum), "failures": 0},
        "passed": bool(torch.all(~failures).item()),
    }


def normalized_projected_step(
    gradient: torch.Tensor,
    caps: torch.Tensor,
    *,
    fraction: float,
) -> torch.Tensor:
    if gradient.ndim != 2 or caps.shape != (gradient.shape[0],):
        raise ValueError("gradient/cap shapes are incompatible")
    norms = gradient.float().norm(dim=1)
    budgets = float(fraction) * caps.to(device=gradient.device, dtype=torch.float32)
    scales = budgets / norms.clamp_min(1e-20)
    scales = torch.where(norms > 0, scales, torch.zeros_like(scales))
    return -gradient.float() * scales[:, None]


def targeted_token_prompt_partitions(
    tok: Any,
    prompts: Sequence[str],
    selected_ids: Sequence[int],
    *,
    fit_per_row: int,
    development_per_row: int,
    certification_per_row: int,
) -> tuple[Dict[str, List[str]], Dict[str, Any]]:
    """Greedily allocate disjoint corpus prompts containing selected token rows."""
    selected = set(int(value) for value in selected_ids)
    candidates: Dict[int, List[str]] = {value: [] for value in selected}
    unique_prompts = list(dict.fromkeys(str(prompt) for prompt in prompts))
    for start in range(0, len(unique_prompts), 256):
        batch = unique_prompts[start : start + 256]
        try:
            encoded = tok(batch, add_special_tokens=False)["input_ids"]
            if (
                not isinstance(encoded, Sequence)
                or len(encoded) != len(batch)
                or any(not isinstance(token_ids, Sequence) for token_ids in encoded)
            ):
                raise TypeError(
                    "tokenizer did not return one token sequence per prompt"
                )
        except (TypeError, ValueError):
            encoded = [
                tok(value, add_special_tokens=False)["input_ids"] for value in batch
            ]
        for value, token_ids in zip(batch, encoded):
            ids = set(int(token) for token in token_ids)
            for token_id in sorted(ids.intersection(selected)):
                candidates[token_id].append(value)
    allocated: Dict[str, List[str]] = {
        "fit": [],
        "development": [],
        "certification": [],
    }
    assigned_role: Dict[str, str] = {}
    requirements = (
        ("fit", int(fit_per_row)),
        ("development", int(development_per_row)),
        ("certification", int(certification_per_row)),
    )
    per_row: List[Dict[str, Any]] = []
    for token_id in sorted(selected):
        counts: Dict[str, int] = {}
        for role, required in requirements:
            chosen = [
                prompt
                for prompt in candidates[token_id]
                if assigned_role.get(prompt, role) == role
            ][:required]
            allocated[role].extend(chosen)
            for prompt in chosen:
                assigned_role[prompt] = role
            counts[role] = len(chosen)
        available = len(candidates[token_id])
        per_row.append(
            {
                "token_id": token_id,
                "corpus_occurrences": available,
                "fit": counts["fit"],
                "development": counts["development"],
                "certification": counts["certification"],
                "corpus_absent": available == 0,
                "complete_registered_coverage": all(
                    counts[role] >= required for role, required in requirements
                ),
            }
        )
    for role in allocated:
        allocated[role] = list(dict.fromkeys(allocated[role]))
    report = {
        "selected_rows": len(selected),
        "corpus_absent_rows": sum(row["corpus_absent"] for row in per_row),
        "fully_covered_rows": sum(
            row["complete_registered_coverage"] for row in per_row
        ),
        "partition_prompt_counts": {
            role: len(values) for role, values in allocated.items()
        },
        "partitions_pairwise_disjoint": not (
            set(allocated["fit"]).intersection(allocated["development"])
            or set(allocated["fit"]).intersection(allocated["certification"])
            or set(allocated["development"]).intersection(allocated["certification"])
        ),
        "per_row": per_row,
    }
    return allocated, report


def choose_arm(reports: Sequence[Mapping[str, Any]]) -> float | None:
    passing = [report for report in reports if bool(report.get("passed"))]
    if not passing:
        return None
    selected = min(
        passing,
        key=lambda report: (
            float(report["correction_floor"]),
            float(report["total_delta_norm"]),
        ),
    )
    return float(selected["correction_floor"])
