#!/usr/bin/env python3
"""Protected-subspace geometry helpers for two-stage MCF SURE.

This module intentionally consumes only direct training-visible records/cases.
It never opens official paraphrases, neighborhood prompts, benchmark-retain
examples, or PPL text.

Stage 1 geometry:
    H_S  = final hidden states at sensitive prediction positions
    H_NS = preceding-context hidden states from the SAME direct prompts
    B_NS = rowspace(H_NS), rank-capped
    R_S  = H_S - Proj_BNS(H_S)
    B_S  = rowspace(R_S), built row-specifically for each sensitive token

Stage 2 geometry:
    H_P  = Stage-1-success sensitive prediction hidden states
    H_F  = residual-failure sensitive prediction hidden states
    B_P  = rowspace(H_P), rank-capped
    R_F  = H_F - Proj_BP(H_F)
    B_F  = rowspace(R_F), built row-specifically for each repaired token
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from torch import nn

import sure_canonical_core as core


@torch.no_grad()
def collect_preceding_context_hidden(
    model: nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
    *,
    context_window: int = 8,
) -> torch.Tensor:
    """Collect hidden states preceding each final prediction position.

    The final token-position hidden state is intentionally excluded because it
    belongs to H_S.  Only up to ``context_window`` immediately preceding token
    positions are retained per direct teacher-forced case.  A non-positive
    window means use every preceding position.
    """
    rows: List[torch.Tensor] = []
    model.eval()
    for case in cases:
        encoded = tok(str(case.prompt), return_tensors="pt", add_special_tokens=True)
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][0].detach().float()
        if hidden.shape[0] <= 1:
            continue
        preceding = hidden[:-1]
        if context_window > 0:
            preceding = preceding[-int(context_window):]
        if preceding.numel():
            rows.append(preceding.cpu())
    if not rows:
        raise RuntimeError("No preceding-context hidden states were collected")
    return torch.cat(rows, dim=0).contiguous()


def project_away(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Return rows with their orthogonal projection onto ``basis`` removed."""
    x = rows.float()
    if basis.numel() == 0:
        return x
    b = basis.to(device=x.device, dtype=torch.float32)
    return x - (x @ b.transpose(0, 1)) @ b


@torch.no_grad()
def build_stage1_sensitive_bases(
    model: nn.Module,
    tok: Any,
    sensitive_cases: Sequence[core.SensitivePredictionCase],
    selected_ids: Sequence[int],
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    protected_rank: int = 32,
    direction_rank: int | None = 4,
    context_window: int = 8,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], torch.Tensor]:
    """Build row-specific Stage-1 sensitive residual bases."""
    if not sensitive_cases:
        raise ValueError("Stage 1 requires sensitive cases")
    hs = core.forward_last_hidden(model, tok, sensitive_cases, device, batch_size).detach().float()
    tids = core.official_target_ids(
        tok, sensitive_cases, llama_like=llama_like, device=device
    ).detach().cpu()

    h_context = collect_preceding_context_hidden(
        model, tok, sensitive_cases, device, context_window=context_window
    )
    max_p = None if int(protected_rank) == 0 else int(protected_rank)
    b_ns = core.orthonormal_row_basis(h_context, max_rank=max_p).detach().float().cpu()
    if b_ns.ndim != 2 or b_ns.shape[0] == 0:
        raise RuntimeError("Stage-1 protected context basis has zero rank")

    residual = project_away(hs.cpu(), b_ns)
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for tid in [int(x) for x in selected_ids]:
        mask = tids.eq(tid)
        rows = residual[mask]
        raw = hs.cpu()[mask]
        if rows.numel() == 0:
            raise RuntimeError(f"Sensitive token {tid} has no Stage-1 hidden rows")
        raw_energy = float(raw.square().sum().sqrt())
        residual_energy = float(rows.square().sum().sqrt())
        if residual_energy <= 1e-8:
            raise RuntimeError(
                f"Sensitive token {tid} has no direction outside protected context subspace"
            )
        basis = core.orthonormal_row_basis(rows, max_rank=direction_rank)
        if basis.ndim != 2 or basis.shape[0] == 0:
            raise RuntimeError(f"Sensitive token {tid} has zero residual sensitive rank")
        bases.append(basis.detach().float().contiguous())
        reports.append(
            {
                "token_id": tid,
                "sensitive_context_count": int(rows.shape[0]),
                "protected_context_rank": int(b_ns.shape[0]),
                "direction_rank": int(basis.shape[0]),
                "raw_sensitive_energy": raw_energy,
                "residual_sensitive_energy": residual_energy,
                "residual_energy_fraction": residual_energy / max(raw_energy, 1e-12),
                "geometry": "H_S - Proj_BNS(H_S)",
            }
        )
    return bases, reports, b_ns


@torch.no_grad()
def build_stage2_repair_bases(
    model: nn.Module,
    tok: Any,
    sensitive_cases: Sequence[core.SensitivePredictionCase],
    selected_ids: Sequence[int],
    *,
    failed_record_positions: Sequence[int],
    protected_record_positions: Sequence[int],
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    protected_rank: int = 32,
    repair_rank: int | None = 4,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build Stage-2 failure bases after projecting away Stage-1 successes."""
    if not failed_record_positions:
        raise ValueError("Stage 2 requires at least one failed record")
    failed_set = set(int(x) for x in failed_record_positions)
    protected_set = set(int(x) for x in protected_record_positions)

    hs = core.forward_last_hidden(model, tok, sensitive_cases, device, batch_size).detach().float().cpu()
    tids = core.official_target_ids(
        tok, sensitive_cases, llama_like=llama_like, device=device
    ).detach().cpu()
    record_pos = torch.tensor(
        [int(case.record_position) for case in sensitive_cases], dtype=torch.long
    )

    fmask = torch.tensor([int(x) in failed_set for x in record_pos.tolist()], dtype=torch.bool)
    pmask = torch.tensor([int(x) in protected_set for x in record_pos.tolist()], dtype=torch.bool)
    h_f_all = hs[fmask]
    h_p = hs[pmask]
    if h_f_all.numel() == 0:
        raise RuntimeError("No failure hidden states were found")

    if h_p.numel() == 0:
        b_p = torch.empty((0, hs.shape[1]), dtype=torch.float32)
    else:
        max_p = None if int(protected_rank) == 0 else int(protected_rank)
        b_p = core.orthonormal_row_basis(h_p, max_rank=max_p).detach().float().cpu()

    residual_all = project_away(hs, b_p)
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for tid in [int(x) for x in selected_ids]:
        mask = tids.eq(tid) & fmask
        raw = hs[mask]
        rows = residual_all[mask]
        if rows.numel() == 0:
            raise RuntimeError(f"Repair token {tid} has no residual-failure hidden rows")
        raw_energy = float(raw.norm())
        residual_energy = float(rows.norm())
        if residual_energy <= 1e-8:
            raise RuntimeError(
                f"Repair token {tid} is fully contained in protected subspace; safe repair unavailable"
            )
        basis = core.orthonormal_row_basis(rows, max_rank=repair_rank)
        if basis.ndim != 2 or basis.shape[0] == 0:
            raise RuntimeError(f"Repair token {tid} has zero protected-safe repair rank")
        bases.append(basis.detach().float().contiguous())
        reports.append(
            {
                "token_id": tid,
                "failure_context_count": int(rows.shape[0]),
                "protected_rank": int(b_p.shape[0]),
                "repair_rank": int(basis.shape[0]),
                "raw_failure_energy": raw_energy,
                "residual_failure_energy": residual_energy,
                "residual_energy_fraction": residual_energy / max(raw_energy, 1e-12),
                "geometry": "H_F - Proj_BP(H_F)",
            }
        )

    return bases, reports, b_p, h_p, h_f_all


def protected_kl_from_sparse_delta(
    base_logits: torch.Tensor,
    protected_hidden: torch.Tensor,
    selected_ids: Sequence[int],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    """Exact full-vocabulary KL(Base || Edited) for fixed hidden states.

    Only selected LM-head rows change in Stage 2, so edited logits can be
    reconstructed exactly by adding h @ delta_w to those vocabulary columns.
    """
    if base_logits.shape[0] == 0:
        return delta_rows.sum() * 0.0
    logits = base_logits.to(device=delta_rows.device, dtype=torch.float32).clone()
    h = protected_hidden.to(device=delta_rows.device, dtype=torch.float32)
    shifts = h @ delta_rows.transpose(0, 1)
    ids = torch.tensor([int(x) for x in selected_ids], dtype=torch.long, device=logits.device)
    logits[:, ids] = logits[:, ids] + shifts
    base = base_logits.to(device=logits.device, dtype=torch.float32)
    return torch.nn.functional.kl_div(
        torch.log_softmax(logits, dim=-1),
        torch.softmax(base, dim=-1),
        reduction="batchmean",
    )
