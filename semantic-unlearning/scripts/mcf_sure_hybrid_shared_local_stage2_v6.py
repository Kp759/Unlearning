#!/usr/bin/env python3
"""MCF SURE Stage 2 v6: exact-null hybrid shared + row-local gradient repair.

This keeps the successful v5 mechanics unchanged:
  * direct MCF sequence-margin P/F split;
  * exact protected sequence nullspace plus residual pre-answer context basis;
  * sequence-margin hinge + KL(P) + L2;
  * hard protected-record gate;
  * exact bf16 feasibility check;
  * minimum exact bf16-feasible final scale.

The only architectural change is the repair basis.  v5 used only row-specific
MCF-margin gradient directions, which improved locality but reduced held-out
paraphrase generalization.  v6 adds one deterministic shared failure direction
while keeping the same total per-row rank budget.

For each failed record i, let G_i be the gradient of its differentiable direct
MCF margin with respect to all selected LM-head rows.  Build a rank-1 shared
hidden direction from the dominant rowspace direction of the protected-null
per-record aggregate gradients:

    g_shared,i = (I - P_safe) sum_s G_i[s, :]
    B_shared   = top-1 rowspace({ normalize(g_shared,i) })

For each sensitive row s, remove both the safe space and B_shared from that
row's per-record gradients and retain up to repair_rank-1 local directions:

    G_local,s = (I - P_shared)(I - P_safe) G_s
    B_local,s = rowspace(G_local,s), rank <= repair_rank - 1

The final row basis is

    B_s = rowspace(B_shared U B_local,s), rank <= repair_rank
    Delta w_s = c_s B_s

Thus every row can use one common failure direction for cross-context transfer,
but all remaining capacity stays row-specific and every direction remains in
the exact protected nullspace.  Shared rank is fixed to one by construction;
this script introduces no shared-rank sweep.

No official paraphrases, neighborhoods, benchmark-retain records, or PPL text
are read before the final checkpoint is frozen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import sure_canonical_core as core
import mcf_sure_protected_subspace_stage1 as stage1v2
import mcf_sure_protected_subspace_stage2_mcf_margin as v3
import mcf_sure_exactnull_gradient_stage2_v5 as v5


METHOD = "SURE-MCF-exact-null-hybrid-shared-local-gradient-stage2"
PROTOCOL = "mcf_target_true_exact_null_hybrid_shared_local_gradient_stage2_v6"
SHARED_RANK = 1
HYBRID_BASIS_REPORT: Dict[str, Any] = {}


def _project_away(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return stage1v2.project_away(rows.float(), basis)


def build_hybrid_margin_gradient_bases(
    *,
    caches,
    selected_ids: Sequence[int],
    hidden_size: int,
    safe_basis: torch.Tensor,
    repair_rank: int,
    device: torch.device,
) -> Tuple[List[int], List[torch.Tensor], List[Dict[str, Any]]]:
    """One shared nullspace direction + row-specific residual directions.

    ``repair_rank`` remains the TOTAL per-row rank cap.  Since shared rank is
    fixed to one, each row receives at most ``repair_rank - 1`` local vectors.
    """
    global HYBRID_BASIS_REPORT
    if int(repair_rank) < SHARED_RANK:
        raise ValueError(
            f"repair-rank={repair_rank} is smaller than fixed shared rank {SHARED_RANK}"
        )

    row_gradients: Dict[int, List[torch.Tensor]] = {
        int(tid): [] for tid in selected_ids
    }
    aggregate_residuals: List[torch.Tensor] = []
    aggregate_raw_norms: List[float] = []
    aggregate_residual_norms: List[float] = []

    # Same exact differentiable sequence-margin gradient used in v5, but retain
    # the complete per-record gradient matrix so we can extract its common
    # hidden direction across output rows.
    for cache in caches:
        delta = torch.zeros(
            (len(selected_ids), int(hidden_size)),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        margin = v3.record_margins_from_caches([cache], delta)[0]
        grad = torch.autograd.grad(margin, delta, retain_graph=False)[0].detach().float()

        for row, tid in enumerate(selected_ids):
            g = grad[row]
            if torch.isfinite(g).all() and float(g.norm().detach().cpu()) > 1e-9:
                row_gradients[int(tid)].append(g)

        aggregate = grad.sum(dim=0, keepdim=True)
        raw_norm = float(aggregate.norm().detach().cpu())
        aggregate_raw_norms.append(raw_norm)
        aggregate_residual = _project_away(aggregate, safe_basis)
        residual_norm = float(aggregate_residual.norm().detach().cpu())
        aggregate_residual_norms.append(residual_norm)
        if (
            torch.isfinite(aggregate_residual).all()
            and residual_norm > 1e-9
        ):
            # Equalize failed records so the shared direction represents
            # recurrence across failures rather than whichever record has the
            # largest gradient magnitude.
            aggregate_residuals.append(
                aggregate_residual.squeeze(0) / residual_norm
            )

    if not aggregate_residuals:
        raise RuntimeError(
            "All aggregate MCF-margin gradients vanish in the exact protected nullspace; "
            "no shared failure direction exists."
        )

    shared_matrix = torch.stack(aggregate_residuals, dim=0).float()
    shared_basis = core.orthonormal_row_basis(
        shared_matrix, max_rank=SHARED_RANK
    )
    if shared_basis.ndim != 2 or int(shared_basis.shape[0]) != SHARED_RANK:
        raise RuntimeError("Could not construct the fixed rank-1 shared failure basis")

    shared_safe_overlap = (
        float(
            (shared_basis @ safe_basis.transpose(0, 1))
            .abs().max().detach().cpu()
        )
        if safe_basis.numel()
        else 0.0
    )

    local_rank_cap = max(0, int(repair_rank) - SHARED_RANK)
    kept_ids: List[int] = []
    final_bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []

    for tid in selected_ids:
        gradients = row_gradients[int(tid)]
        if not gradients:
            reports.append({
                "token_id": int(tid),
                "gradient_count": 0,
                "shared_rank_actual": SHARED_RANK,
                "local_rank_actual": 0,
                "final_rank_actual": 0,
                "skipped": True,
            })
            continue

        matrix = torch.stack(gradients, dim=0).float()
        safe_residual = _project_away(matrix, safe_basis)
        local_residual = _project_away(safe_residual, shared_basis)
        local_residual_norm = float(local_residual.norm().detach().cpu())

        local_basis = shared_basis.new_empty((0, shared_basis.shape[1]))
        if local_rank_cap > 0 and local_residual_norm > 1e-9:
            local_basis = core.orthonormal_row_basis(
                local_residual, max_rank=int(local_rank_cap)
            )

        blocks = [shared_basis]
        if local_basis.numel():
            blocks.append(local_basis)
        combined = torch.cat(blocks, dim=0)
        # Numerical re-orthogonalization only; rank remains <= repair_rank.
        final_basis = core.orthonormal_row_basis(
            combined, max_rank=int(repair_rank)
        )
        if final_basis.ndim != 2 or final_basis.shape[0] <= 0:
            reports.append({
                "token_id": int(tid),
                "gradient_count": int(matrix.shape[0]),
                "shared_rank_actual": SHARED_RANK,
                "local_rank_actual": int(local_basis.shape[0]),
                "final_rank_actual": 0,
                "skipped": True,
            })
            continue

        safe_overlap = (
            float(
                (final_basis @ safe_basis.transpose(0, 1))
                .abs().max().detach().cpu()
            )
            if safe_basis.numel()
            else 0.0
        )
        shared_overlap = float(
            (final_basis @ shared_basis.transpose(0, 1))
            .abs().max().detach().cpu()
        )

        kept_ids.append(int(tid))
        final_bases.append(final_basis.contiguous())
        reports.append({
            "token_id": int(tid),
            "gradient_count": int(matrix.shape[0]),
            "raw_gradient_norm": float(matrix.norm().detach().cpu()),
            "safe_projected_gradient_norm": float(safe_residual.norm().detach().cpu()),
            "local_residual_gradient_norm": local_residual_norm,
            "repair_rank_total_cap": int(repair_rank),
            "shared_rank_actual": SHARED_RANK,
            "local_rank_cap": int(local_rank_cap),
            "local_rank_actual": int(local_basis.shape[0]),
            "final_rank_actual": int(final_basis.shape[0]),
            "max_abs_overlap_with_safe_basis": safe_overlap,
            "max_abs_overlap_with_shared_basis": shared_overlap,
            "skipped": False,
        })

    if not kept_ids:
        raise RuntimeError("No editable row retained a hybrid gradient basis")

    HYBRID_BASIS_REPORT = {
        "shared_rank_by_construction": SHARED_RANK,
        "shared_rank_actual": int(shared_basis.shape[0]),
        "shared_source": (
            "top-1 rowspace direction of unit-normalized per-failure aggregate "
            "MCF sequence-margin gradients after exact safe-space projection"
        ),
        "failure_records_contributing": int(len(aggregate_residuals)),
        "aggregate_raw_gradient_norm_mean": float(
            sum(aggregate_raw_norms) / max(1, len(aggregate_raw_norms))
        ),
        "aggregate_residual_gradient_norm_mean": float(
            sum(aggregate_residual_norms) / max(1, len(aggregate_residual_norms))
        ),
        "shared_max_abs_overlap_with_safe_basis": shared_safe_overlap,
        "repair_rank_semantics": (
            "total per-row rank cap: one shared direction plus up to repair_rank-1 local directions"
        ),
    }
    return kept_ids, final_bases, reports


def _arg_value(name: str) -> str | None:
    for index, value in enumerate(sys.argv[:-1]):
        if value == name:
            return sys.argv[index + 1]
    prefix = name + "="
    for value in sys.argv[1:]:
        if value.startswith(prefix):
            return value[len(prefix):]
    return None


def _rewrite_summary() -> None:
    output_dir = _arg_value("--output-dir")
    if not output_dir:
        return
    path = Path(output_dir) / "repair_summary.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["schema_version"] = 6
    payload["method"] = METHOD
    payload["protocol"] = PROTOCOL
    payload["hybrid_shared_local_repair"] = True
    payload["shared_basis"] = HYBRID_BASIS_REPORT
    payload["row_basis_source"] = (
        "rank-1 shared protected-null aggregate MCF-margin gradient direction + "
        "row-specific protected-null residual MCF-margin gradients"
    )
    payload["parameterization"] = (
        "Delta w_s = c_shared,s B_shared + c_local,s B_local,s; "
        "all directions in exact P nullspace"
    )
    payload["repair_rank_semantics"] = (
        "repair-rank is total per-row cap; shared rank fixed at 1, local cap=repair-rank-1"
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "v6_summary_rewritten": str(path),
        "shared_basis": HYBRID_BASIS_REPORT,
        "hybrid_shared_local_repair": True,
    }, indent=2))


def main() -> None:
    # Reuse the battle-tested v5 optimizer, hard protection, exact bf16 check,
    # and minimum-feasible save.  Only basis construction changes.
    v5.METHOD = METHOD
    v5.PROTOCOL = PROTOCOL
    v5.build_rowwise_margin_gradient_bases = build_hybrid_margin_gradient_bases
    v5.main()
    _rewrite_summary()


if __name__ == "__main__":
    main()
