#!/usr/bin/env python3
"""MQuAKE SURE Stage 2 v4.1: numerically stable robust P-nullspace repair.

This is a numerical bug-fix wrapper around v4.  The v4 architecture and
objective are unchanged.  The only change is construction of the protected and
repair row-space bases.

Why this exists
---------------
The first v4 seed-1 run reported a large ``nullspace-leak`` even though the
repair was intended to be orthogonal to every protected hidden state.  The
cause is that the generic float32 row-space helper can retain tiny singular
vectors of the already-projected residual.  Those roundoff directions are not
reliably orthogonal after the basis is cast/materialized, so the hard P guard
starts rejecting otherwise useful optimizer steps.

v4.1 therefore:
  * computes SVDs in float64;
  * uses an explicit relative singular-value tolerance;
  * stores the first-call protected basis in float64;
  * reprojects the second-call residual against that protected basis in float64;
  * reprojects the candidate repair basis once more and QR-orthonormalizes it;
  * validates the resulting basis before returning it to the unchanged v4
    optimizer/guard/minimum-norm-shrink implementation.

No official AtomicGen prompt, retain example, target_new, or other held-out
field is introduced.  Stage-1 v4 checkpoints can be reused unchanged.
"""
from __future__ import annotations

from typing import Optional

import torch

import sure_canonical_core as core
import mquake_sure_stage2_robust_nullspace_v4 as v4


# The two calls made by v4.main are, in order:
#   1. rowspace(H_P)
#   2. rowspace(H_F - Proj_P(H_F))
# Keep the protected basis from call 1 so call 2 can be cleaned in float64.
_PROTECTED_BASIS64: Optional[torch.Tensor] = None
_CALL_INDEX = 0


def _svd_row_basis64(rows64: torch.Tensor, *, rtol: float) -> torch.Tensor:
    """Return a numerically stable orthonormal row basis in float64."""
    if rows64.numel() == 0:
        hidden = rows64.shape[-1] if rows64.ndim == 2 else 0
        return rows64.new_empty((0, hidden), dtype=torch.float64)
    if rows64.ndim != 2:
        raise ValueError("row-basis input must be rank 2")

    rows64 = rows64.to(dtype=torch.float64)
    _, singular_values, vh = torch.linalg.svd(rows64, full_matrices=False)
    if singular_values.numel() == 0:
        return rows64.new_empty((0, rows64.shape[1]), dtype=torch.float64)

    smax = singular_values.max().clamp_min(torch.finfo(torch.float64).tiny)
    # The explicit relative floor rejects residual directions created only by
    # float32 projection/SVD roundoff while retaining genuine failed-case
    # directions.  The machine-epsilon term is included for scale safety.
    rel_tol = float(rtol) * smax
    eps_tol = max(rows64.shape) * torch.finfo(torch.float64).eps * smax
    tolerance = torch.maximum(rel_tol, eps_tol)
    rank = int((singular_values > tolerance).sum().item())
    if rank <= 0:
        raise RuntimeError(
            "stable row-space basis is empty; residual contains no direction "
            "above the numerical tolerance"
        )
    return vh[:rank].contiguous()


def stable_orthonormal_row_basis(
    rows: torch.Tensor,
    max_rank: Optional[int],
) -> torch.Tensor:
    """Drop-in replacement for core.orthonormal_row_basis used by Stage2 v4."""
    global _PROTECTED_BASIS64, _CALL_INDEX
    _CALL_INDEX += 1

    source_device = rows.device
    rows64 = rows.detach().to(dtype=torch.float64)

    if _CALL_INDEX == 1:
        basis64 = _svd_row_basis64(rows64, rtol=1e-10)
        if max_rank is not None:
            basis64 = basis64[: int(max_rank)]
        _PROTECTED_BASIS64 = basis64.detach().clone()
        gram_err = float(
            (
                basis64 @ basis64.transpose(0, 1)
                - torch.eye(basis64.shape[0], device=basis64.device, dtype=torch.float64)
            )
            .abs()
            .max()
            .item()
        )
        print(
            "Stage2-v4.1 stable protected basis: rank={} gram_err={:.3e}".format(
                basis64.shape[0], gram_err
            )
        )
        return basis64.to(device=source_device, dtype=torch.float32)

    if _CALL_INDEX != 2 or _PROTECTED_BASIS64 is None:
        # v4 currently needs exactly two bases.  Fall back to a stable generic
        # basis for any future extra call rather than silently using stale P.
        basis64 = _svd_row_basis64(rows64, rtol=1e-8)
        if max_rank is not None:
            basis64 = basis64[: int(max_rank)]
        return basis64.to(device=source_device, dtype=torch.float32)

    p64 = _PROTECTED_BASIS64.to(device=rows64.device, dtype=torch.float64)

    # Remove any float32 projection residue again in float64.
    cleaned64 = rows64 - (rows64 @ p64.transpose(0, 1)) @ p64
    basis64 = _svd_row_basis64(cleaned64, rtol=1e-8)

    # Explicitly force candidate repair directions back into P^perp.  QR on
    # the transpose gives an orthonormal row basis without reintroducing P.
    basis64 = basis64 - (basis64 @ p64.transpose(0, 1)) @ p64
    q, _ = torch.linalg.qr(basis64.transpose(0, 1), mode="reduced")
    basis64 = q.transpose(0, 1).contiguous()

    # One final projection + QR suppresses accumulated roundoff at high rank.
    basis64 = basis64 - (basis64 @ p64.transpose(0, 1)) @ p64
    q, _ = torch.linalg.qr(basis64.transpose(0, 1), mode="reduced")
    basis64 = q.transpose(0, 1).contiguous()

    if max_rank is not None:
        basis64 = basis64[: int(max_rank)]

    leak64 = float((p64 @ basis64.transpose(0, 1)).abs().max().item())
    gram_err = float(
        (
            basis64 @ basis64.transpose(0, 1)
            - torch.eye(basis64.shape[0], device=basis64.device, dtype=torch.float64)
        )
        .abs()
        .max()
        .item()
    )
    if leak64 > 1e-8:
        raise RuntimeError(
            f"v4.1 repair basis failed float64 P-nullspace validation: leak={leak64}"
        )
    print(
        "Stage2-v4.1 stable repair basis: rank={} P_leak64={:.3e} gram_err={:.3e}".format(
            basis64.shape[0], leak64, gram_err
        )
    )
    return basis64.to(device=source_device, dtype=torch.float32)


def main() -> None:
    # v4 imports the same core module object, so this replaces only basis
    # construction while retaining all v4 gating, objective, hard protection,
    # best-checkpoint logic, shrink, materialization, and reporting.
    core.orthonormal_row_basis = stable_orthonormal_row_basis
    v4.main()


if __name__ == "__main__":
    main()
