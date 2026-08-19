#!/usr/bin/env python3
"""Utilities for row-specific, forget-context-projected SURE LM-head deltas.

Each editable sensitive vocabulary row receives its own orthonormal hidden-state
basis built only from direct training-visible forget contexts in which that row
is the teacher-forced sensitive target.  Deltas are parameterized inside that
row-specific basis, so no component orthogonal to the observed forget-context
subspace can be introduced.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import sure_canonical_core as core


def expand_answer_field_cases(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    field: str,
    llama_like: bool,
) -> List[core.SensitivePredictionCase]:
    """Expand one MCF answer field into teacher-forced next-token cases."""
    cases: List[core.SensitivePredictionCase] = []
    for position, record in enumerate(records):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise ValueError(f"Record {position} lacks requested_rewrite")
        subject = str(rr.get("subject", ""))
        prompt_template = str(rr.get("prompt", ""))
        target_block = rr.get(field)
        if not isinstance(target_block, Mapping) or not target_block.get("str"):
            raise ValueError(f"Record {position} lacks {field}.str")
        answer = str(target_block["str"])
        prompt = prompt_template.format(subject)
        tids = core.answer_token_ids(tok, answer, llama_like=llama_like)
        case_id = int(record.get("case_id", position))
        for token_index, token_id in enumerate(tids):
            decoded_prefix = tok.decode(tids[:token_index])
            if llama_like and token_index > 0:
                evaluated_prompt = prompt + " " + decoded_prefix
            else:
                evaluated_prompt = prompt + decoded_prefix
            cases.append(
                core.SensitivePredictionCase(
                    case_id=case_id,
                    record_position=position,
                    token_index=token_index,
                    prompt=evaluated_prompt,
                    target_text=tok.decode([token_id]),
                )
            )
    return cases


def build_row_specific_bases(
    model: nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    *,
    selected_ids: Sequence[int],
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    max_rank: Optional[int],
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Build one orthonormal forget-context basis per selected token id."""
    if not cases:
        raise ValueError("Cannot build context bases from zero cases")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    hidden = core.forward_last_hidden(model, tok, cases, device, batch_size)
    tids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    ).detach()
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for token_id in [int(x) for x in selected_ids]:
        mask = tids.eq(token_id)
        rows = hidden[mask]
        if rows.numel() == 0:
            raise RuntimeError(
                f"Selected sensitive token {token_id} has no forget-context hidden rows"
            )
        basis = core.orthonormal_row_basis(rows, max_rank=max_rank)
        if basis.ndim != 2 or basis.shape[0] <= 0:
            raise RuntimeError(
                f"Sensitive token {token_id} has zero numerical context rank"
            )
        bases.append(basis.detach().float().contiguous())
        reports.append(
            {
                "token_id": token_id,
                "context_count": int(rows.shape[0]),
                "context_rank": int(basis.shape[0]),
                "hidden_size": int(basis.shape[1]),
            }
        )
    return bases, reports


class RowSpecificProjectedDelta(nn.Module):
    """One sparse output-row delta, each constrained to its own fixed basis."""

    def __init__(
        self,
        row_ids: Sequence[int],
        bases: Sequence[torch.Tensor],
        *,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.row_ids = tuple(int(x) for x in row_ids)
        if len(self.row_ids) == 0:
            raise ValueError("RowSpecificProjectedDelta requires at least one row")
        if len(self.row_ids) != len(bases):
            raise ValueError("row_ids and bases must have equal length")
        hidden_size = None
        self._basis_names: List[str] = []
        coeffs: List[nn.Parameter] = []
        for index, basis in enumerate(bases):
            if basis.ndim != 2 or basis.shape[0] <= 0:
                raise ValueError("Every row-specific basis must be non-empty [rank, hidden]")
            if hidden_size is None:
                hidden_size = int(basis.shape[1])
            elif int(basis.shape[1]) != hidden_size:
                raise ValueError("All row-specific bases must share hidden size")
            name = f"basis_{index}"
            self.register_buffer(
                name,
                basis.detach().to(device=device, dtype=torch.float32).contiguous(),
            )
            self._basis_names.append(name)
            coeffs.append(
                nn.Parameter(
                    torch.zeros(
                        int(basis.shape[0]), device=device, dtype=torch.float32
                    )
                )
            )
        self.hidden_size = int(hidden_size or 0)
        self.coefficients = nn.ParameterList(coeffs)

    def basis(self, index: int) -> torch.Tensor:
        return getattr(self, self._basis_names[index])

    def effective_delta(self) -> torch.Tensor:
        rows: List[torch.Tensor] = []
        for index, coeff in enumerate(self.coefficients):
            rows.append(torch.matmul(coeff, self.basis(index)))
        return torch.stack(rows, dim=0)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    @property
    def row_ranks(self) -> List[int]:
        return [int(self.basis(i).shape[0]) for i in range(len(self.row_ids))]


@torch.no_grad()
def project_rows_to_bases(
    rows: torch.Tensor,
    bases: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Project existing row vectors onto matching orthonormal row bases."""
    if rows.ndim != 2 or rows.shape[0] != len(bases):
        raise ValueError("rows must be [n_rows, hidden] and match bases")
    projected: List[torch.Tensor] = []
    for row, basis in zip(rows.float(), bases):
        b = basis.to(device=row.device, dtype=torch.float32)
        if b.ndim != 2 or b.shape[1] != row.shape[0]:
            raise ValueError("basis hidden dimension does not match row")
        projected.append((row @ b.transpose(0, 1)) @ b)
    return torch.stack(projected, dim=0)
