#!/usr/bin/env python3
"""Shared sensitive-token suppression mechanics for fixed SURE architecture.

This module contains no benchmark-specific objective.  A benchmark adapter only
supplies the sensitive teacher-forced PredictionCases.  Direct success is the
same for every benchmark: at each sensitive token decision, the best
non-sensitive token logit must exceed the sensitive-token logit by at least the
configured margin.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import torch

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


@torch.no_grad()
def evaluate_suppression_margins(
    model: Any,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        logits = core.forward_last_logits(model, tok, batch, device)
        tids = core.official_target_ids(
            tok, batch, llama_like=llama_like, device=device
        )
        chunks.append(suppression_margins_from_logits(logits, tids).detach())
    if not chunks:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.cat(chunks, dim=0)


def count_failures(margins: torch.Tensor, required_margin: float) -> int:
    if margins.numel() == 0:
        return 0
    return int((margins < float(required_margin)).sum().item())
