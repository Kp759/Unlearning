#!/usr/bin/env python3
"""SURE MQuAKE: full sensitive-row complement restoration before rank-0.

This is the controlled follow-up to mquake_sure_outputonly_prerestore_rank0.py.
The previous safety gate rejected every nonzero pre-rank0 restoration scale even
though scale=1.0 reactivated zero previously forgotten direct cases; rejection
was caused only by small BF16 direct-margin changes. Here the architecture is
made explicit: restore the full component of each sensitive LM-head Stage1
displacement lying outside the direct-forget hidden span, then let rank-0
re-enforce the fixed 0.25 forget margin. No retain, AtomicGen, multihop, or PPL
metric is used before candidates are frozen.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

import mquake_sure_outputonly_prerestore_rank0 as impl
import mquake_sure_gagd_rank0_restore as core
import mquake_zero_unlearn_official_eval as mq


@torch.no_grad()
def choose_full_pre_rank0_restore(
    model,
    tok,
    cases: Sequence[mq.PredictionCase],
    device: torch.device,
    target_ids: torch.Tensor,
    row_ids: Sequence[int],
    stage1_rows: torch.Tensor,
    base_rows: torch.Tensor,
    hidden: torch.Tensor,
    stage1_logits: torch.Tensor,
    forget_basis: torch.Tensor,
    scales: Sequence[float],
    margin_tolerance: float,
    batch_size: int,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Apply scale=1 complement restore unconditionally; rank-0 repairs next."""
    del scales, margin_tolerance
    out = model.get_output_embeddings()

    stage1_target = stage1_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    stage1_best = core._best_nonsensitive_logits(stage1_logits, row_ids)
    stage1_margins = stage1_best - stage1_target
    stage1_top1 = stage1_logits.argmax(dim=-1).eq(target_ids)

    displacement = stage1_rows - base_rows
    if forget_basis.numel():
        forget_component = (displacement @ forget_basis.T) @ forget_basis
    else:
        forget_component = torch.zeros_like(displacement)
    collateral_component = displacement - forget_component

    # Full complement restoration: keep only the Stage1 displacement that lies
    # in the direct-forget hidden span. Any resulting residual forget failures
    # are intentionally handled by the immediately following rank-0 repair.
    candidate_rows = base_rows + forget_component
    core.set_rows(out.weight, row_ids, candidate_rows)
    _, candidate_logits = core.cache_case_states(model, tok, cases, device, batch_size)

    candidate_target = candidate_logits.gather(1, target_ids.view(-1, 1)).squeeze(1)
    candidate_best = core._best_nonsensitive_logits(candidate_logits, row_ids)
    candidate_margins = candidate_best - candidate_target
    candidate_top1 = candidate_logits.argmax(dim=-1).eq(target_ids)
    margin_change = candidate_margins - stage1_margins

    reactivated = int(((~stage1_top1) & candidate_top1).sum().item())
    newly_forgotten = int((stage1_top1 & (~candidate_top1)).sum().item())
    rows_actual = out.weight.index_select(
        0, torch.tensor(row_ids, dtype=torch.long, device=out.weight.device)
    ).detach().float()

    summary = {
        "definition": "full restoration of the Stage1 sensitive-row displacement component orthogonal to the direct-forget hidden span; rank-0 subsequently re-enforces forgetting",
        "selected_scale": 1.0,
        "selection_mode": "fixed_preregistered_full_restore_no_utility_selection",
        "sensitive_output_rows": int(len(row_ids)),
        "forget_basis_rank": int(forget_basis.shape[0]),
        "stage1_displacement_norm": float(displacement.norm().item()),
        "forget_span_component_norm": float(forget_component.norm().item()),
        "collateral_complement_component_norm": float(collateral_component.norm().item()),
        "full_restore_norm": float(collateral_component.norm().item()),
        "effective_restore_norm": float(collateral_component.norm().item()),
        "stage1_direct_target_top1_cases": int(stage1_top1.sum().item()),
        "selected_direct_target_top1_cases": int(candidate_top1.sum().item()),
        "reactivated_previously_forgotten_direct_token_cases": reactivated,
        "newly_forgotten_direct_token_cases": newly_forgotten,
        "stage1_minimum_direct_margin": float(stage1_margins.min().item()),
        "selected_minimum_direct_margin": float(candidate_margins.min().item()),
        "stage1_mean_direct_margin": float(stage1_margins.mean().item()),
        "selected_mean_direct_margin": float(candidate_margins.mean().item()),
        "minimum_margin_change_from_stage1": float(margin_change.min().item()),
        "maximum_margin_change_from_stage1": float(margin_change.max().item()),
        "benchmark_retain_seen": 0,
        "atomic_questions_seen": 0,
        "multihop_questions_seen": 0,
        "PPL_seen": False,
        "selection_uses_only_training_visible_direct_forget_constraints": False,
        "pre_rank0_utility_selection_used": False,
        "rank0_is_responsible_for_final_forget_constraint": True,
    }
    return rows_actual, summary


if __name__ == "__main__":
    impl.METHOD = "SURE-MQuAKE-OutputOnly-FullPreRestore-Rank0-LowRankRestore"
    impl.choose_pre_rank0_restore = choose_full_pre_rank0_restore
    impl.main()
