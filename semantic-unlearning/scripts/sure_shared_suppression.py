#!/usr/bin/env python3
"""Shared sensitive-token suppression mechanics for fixed SURE architecture.

This module contains no benchmark-specific objective. A benchmark adapter only
supplies the sensitive teacher-forced PredictionCases.

Direct success is identical for every benchmark and requires BOTH:
  1. the best non-sensitive token logit exceeds the sensitive-token logit by
     at least ``required_logit_margin``; and
  2. the sensitive token NLL has increased relative to the frozen Base model by
     at least ``required_nll_increase``.

The second condition prevents an early stop where the sensitive token merely
loses top-1 while its absolute likelihood remains high. It needs no replacement
or neutral answer and therefore applies identically to MCF and ZsRE.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F

import sure_canonical_core as core


def suppression_margins_from_logits(
    logits: torch.Tensor,
    sensitive_ids: torch.Tensor,
) -> torch.Tensor:
    """Return best-non-sensitive minus sensitive logit for every case."""
    if logits.ndim != 2 or sensitive_ids.ndim != 1:
        raise ValueError("Expected logits [batch,vocab] and sensitive ids [batch]")
    if logits.shape[0] != sensitive_ids.shape[0]:
        raise ValueError("Batch dimension mismatch")
    rows = torch.arange(logits.shape[0], device=logits.device)
    sensitive = logits[rows, sensitive_ids].float()
    other = logits.float().clone()
    other[rows, sensitive_ids] = -torch.inf
    best_other = other.max(dim=-1).values
    return best_other - sensitive


def sensitive_token_nlls_from_logits(
    logits: torch.Tensor,
    sensitive_ids: torch.Tensor,
) -> torch.Tensor:
    """Teacher-forced sensitive-token NLL for each PredictionCase."""
    if logits.ndim != 2 or sensitive_ids.ndim != 1:
        raise ValueError("Expected logits [batch,vocab] and sensitive ids [batch]")
    if logits.shape[0] != sensitive_ids.shape[0]:
        raise ValueError("Batch dimension mismatch")
    rows = torch.arange(logits.shape[0], device=logits.device)
    logp = F.log_softmax(logits.float(), dim=-1)
    return -logp[rows, sensitive_ids]


def sensitive_nll_increase_from_logits(
    current_logits: torch.Tensor,
    base_logits: torch.Tensor,
    sensitive_ids: torch.Tensor,
) -> torch.Tensor:
    """Return current sensitive NLL minus frozen-Base sensitive NLL."""
    base = base_logits.to(device=current_logits.device, dtype=torch.float32)
    if current_logits.shape != base.shape:
        raise ValueError("Current and Base logits must have identical shape")
    current_nll = sensitive_token_nlls_from_logits(current_logits, sensitive_ids)
    base_nll = sensitive_token_nlls_from_logits(base, sensitive_ids)
    return current_nll - base_nll


@torch.no_grad()
def evaluate_shared_constraints(
    model: Any,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    base_logits: torch.Tensor,
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Evaluate the two shared direct constraints on all sensitive cases."""
    if not isinstance(base_logits, torch.Tensor) or base_logits.shape[0] != len(cases):
        raise ValueError("Base-logit cache does not align with sensitive cases")
    margin_chunks: List[torch.Tensor] = []
    nll_delta_chunks: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        logits = core.forward_last_logits(model, tok, batch, device)
        tids = core.official_target_ids(
            tok, batch, llama_like=llama_like, device=device
        )
        base_chunk = base_logits[start : start + len(batch)]
        margin_chunks.append(suppression_margins_from_logits(logits, tids).detach())
        nll_delta_chunks.append(
            sensitive_nll_increase_from_logits(logits, base_chunk, tids).detach()
        )
    if not margin_chunks:
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return {"logit_margin": empty, "sensitive_nll_increase": empty}
    return {
        "logit_margin": torch.cat(margin_chunks, dim=0),
        "sensitive_nll_increase": torch.cat(nll_delta_chunks, dim=0),
    }


def failure_mask(
    logit_margins: torch.Tensor,
    sensitive_nll_increase: torch.Tensor,
    *,
    required_logit_margin: float,
    required_nll_increase: float,
) -> torch.Tensor:
    if logit_margins.shape != sensitive_nll_increase.shape:
        raise ValueError("Shared constraint tensors must have identical shape")
    return (
        logit_margins < float(required_logit_margin)
    ) | (
        sensitive_nll_increase < float(required_nll_increase)
    )


def count_failures(
    logit_margins: torch.Tensor,
    sensitive_nll_increase: torch.Tensor,
    *,
    required_logit_margin: float,
    required_nll_increase: float,
) -> int:
    if logit_margins.numel() == 0:
        return 0
    return int(
        failure_mask(
            logit_margins,
            sensitive_nll_increase,
            required_logit_margin=required_logit_margin,
            required_nll_increase=required_nll_increase,
        ).sum().item()
    )
