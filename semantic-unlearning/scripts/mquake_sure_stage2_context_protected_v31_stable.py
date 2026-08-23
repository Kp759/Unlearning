#!/usr/bin/env python3
"""Numerically stable launcher for context-protected MQuAKE SURE v3 Stage 2.

This wrapper changes only the linear-algebra implementation used to construct
protected/repair row spaces.  It keeps the exact same locked training-visible
cases, p-median repair margin, optimizer, P guard, KL budget, and sparse LM-head
parameterization as mquake_sure_stage2_context_protected_v3.py.

The first row-basis call (B_G for H_P + H_NS) uses float64 SVD with rtol=1e-10.
The second call (B_F for residual H_F) first explicitly removes B_G again in
float64, uses float64 SVD with rtol=1e-8, then performs two project+QR cleanup
passes.  This prevents weak protected directions from being discarded by the
scale-dependent float32 matrix-rank tolerance.
"""
from __future__ import annotations

from typing import Optional

import torch

import sure_canonical_core as core


_ORIGINAL_BASIS = core.orthonormal_row_basis
_PROTECTED_BASIS64: Optional[torch.Tensor] = None
_CALL_INDEX = 0


def _svd_row_basis64(rows: torch.Tensor, *, rtol: float, max_rank: Optional[int]) -> torch.Tensor:
    if rows.numel() == 0:
        hidden = rows.shape[-1] if rows.ndim == 2 else 0
        return rows.new_empty((0, hidden), dtype=torch.float64)
    x = rows.detach().to(dtype=torch.float64)
    _, s, vh = torch.linalg.svd(x, full_matrices=False)
    threshold = float(rtol) * s.max().clamp_min(1.0)
    rank = int((s > threshold).sum().item())
    if max_rank is not None:
        rank = min(rank, int(max_rank))
    if rank <= 0:
        return x.new_empty((0, x.shape[1]))
    return vh[:rank].contiguous()


def _project_out(rows64: torch.Tensor, basis64: torch.Tensor) -> torch.Tensor:
    if rows64.numel() == 0 or basis64.numel() == 0:
        return rows64
    return rows64 - (rows64 @ basis64.transpose(0, 1)) @ basis64


def _cleanup_repair_basis(basis64: torch.Tensor, protected64: torch.Tensor) -> torch.Tensor:
    b = basis64
    if b.numel() == 0:
        return b
    for _ in range(2):
        b = _project_out(b, protected64)
        q, _ = torch.linalg.qr(b.transpose(0, 1), mode="reduced")
        b = q.transpose(0, 1).contiguous()
    return b


def stable_context_row_basis(rows: torch.Tensor, max_rank: Optional[int]) -> torch.Tensor:
    global _CALL_INDEX, _PROTECTED_BASIS64
    _CALL_INDEX += 1

    if _CALL_INDEX == 1:
        b64 = _svd_row_basis64(rows, rtol=1e-10, max_rank=max_rank)
        _PROTECTED_BASIS64 = b64.detach()
        gram_err = 0.0
        if b64.numel():
            eye = torch.eye(b64.shape[0], dtype=torch.float64, device=b64.device)
            gram_err = float((b64 @ b64.transpose(0, 1) - eye).abs().max().detach().cpu())
        print(
            "Stable nullspace B_G: rows={} rank={} rtol=1e-10 gram_err64={:.3e}".format(
                int(rows.shape[0]), int(b64.shape[0]), gram_err
            )
        )
        return b64.to(device=rows.device, dtype=torch.float32)

    if _CALL_INDEX == 2:
        if _PROTECTED_BASIS64 is None:
            raise RuntimeError("protected basis missing before repair-basis construction")
        x64 = rows.detach().to(dtype=torch.float64)
        pg = _PROTECTED_BASIS64.to(device=x64.device, dtype=torch.float64)
        x64 = _project_out(x64, pg)
        b64 = _svd_row_basis64(x64, rtol=1e-8, max_rank=max_rank)
        b64 = _cleanup_repair_basis(b64, pg)
        leak64 = 0.0
        gram_err = 0.0
        if b64.numel():
            leak64 = float((pg @ b64.transpose(0, 1)).abs().max().detach().cpu())
            eye = torch.eye(b64.shape[0], dtype=torch.float64, device=b64.device)
            gram_err = float((b64 @ b64.transpose(0, 1) - eye).abs().max().detach().cpu())
        print(
            "Stable nullspace B_F: rows={} rank={} rtol=1e-8 protected_leak64={:.3e} gram_err64={:.3e}".format(
                int(rows.shape[0]), int(b64.shape[0]), leak64, gram_err
            )
        )
        if leak64 > 1e-8:
            raise RuntimeError(f"float64 protected-nullspace leak too large: {leak64}")
        return b64.to(device=rows.device, dtype=torch.float32)

    # The context-protected implementation should make exactly two basis calls.
    # Fall back rather than silently applying the special tolerances elsewhere.
    return _ORIGINAL_BASIS(rows, max_rank)


def main() -> None:
    core.orthonormal_row_basis = stable_context_row_basis
    import mquake_sure_stage2_context_protected_v3 as impl
    impl.main()


if __name__ == "__main__":
    main()
