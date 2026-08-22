#!/usr/bin/env python3
"""Numerically stable entrypoint for Directional SURE v2.

This wrapper preserves the v2 training protocol and replaces only its
orthonormal-basis constructor. CUDA FP32 SVD right-singular vectors are
explicitly re-orthogonalized with reduced QR. If GPU FP32 QR still exceeds the
strict orthogonality tolerance, a CPU float64 QR fallback is used for that
basis before returning it on the original device in FP32.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch

import mcf_sensitive_rows_directional_gagd_v2 as V2


ORTHOGONALITY_ATOL = 1e-4
FALLBACK_ATOL = 2e-4


def _max_orthogonality_error(basis: torch.Tensor) -> float:
    if basis.numel() == 0:
        return 0.0
    rank = int(basis.shape[0])
    gram = basis.float() @ basis.float().T
    eye = torch.eye(rank, device=gram.device, dtype=gram.dtype)
    return float((gram - eye).abs().max().detach().cpu())


def _qr_reorthogonalize(rows: torch.Tensor) -> Tuple[torch.Tensor, str, float]:
    """Return an orthonormal row basis spanning rows, with a robust fallback."""
    if rows.numel() == 0:
        return rows.detach().float().contiguous(), "empty", 0.0

    raw = rows.detach().float().contiguous()
    q, _r = torch.linalg.qr(raw.T, mode="reduced")
    basis = q.T.contiguous().float()
    error = _max_orthogonality_error(basis)
    method = "gpu_fp32_qr" if raw.is_cuda else "cpu_fp32_qr"

    if error > ORTHOGONALITY_ATOL:
        # A second QR is cheap relative to the SVD and often removes accumulated
        # FP32 loss of orthogonality for wide protected bases.
        q2, _r2 = torch.linalg.qr(basis.T, mode="reduced")
        basis = q2.T.contiguous().float()
        error = _max_orthogonality_error(basis)
        method += "_twice"

    if error > ORTHOGONALITY_ATOL:
        # Rare numerical fallback: do only the re-orthogonalization in float64
        # on CPU, then return to the original device. This does not change the
        # represented span, only its numerical conditioning.
        device = raw.device
        cpu = raw.detach().to(device="cpu", dtype=torch.float64)
        q64, _r64 = torch.linalg.qr(cpu.T, mode="reduced")
        basis = q64.T.to(device=device, dtype=torch.float32).contiguous()
        error = _max_orthogonality_error(basis)
        method = "cpu_float64_qr_fallback"

    if error > FALLBACK_ATOL:
        raise RuntimeError(
            "stable basis re-orthogonalization failed: "
            f"max |BB^T-I|={error:.6g}"
        )
    return basis, method, error


def stable_orthonormal_basis(
    rows: torch.Tensor, rank_cap: int
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """SVD rank selection followed by explicit stable QR re-orthogonalization."""
    values = rows.detach().float()
    if values.ndim != 2:
        raise ValueError("basis rows must be a matrix")
    hidden = int(values.shape[1])
    if values.shape[0] == 0 or values.numel() == 0:
        return values.new_empty((0, hidden)), {
            "source_row_count": 0,
            "effective_rank": 0,
            "kept_rank": 0,
            "energy_fraction_retained": 0.0,
            "singular_values": [],
            "reorthogonalization_method": "empty",
            "max_abs_BBt_minus_I": 0.0,
        }

    _u, singular, vh = torch.linalg.svd(values, full_matrices=False)
    tol = (
        max(values.shape)
        * torch.finfo(values.dtype).eps
        * singular.max().clamp_min(1.0)
    )
    effective = int((singular > tol).sum().item())
    keep = effective if int(rank_cap) <= 0 else min(effective, int(rank_cap))
    raw_basis = vh[:keep].float().contiguous()
    basis, method, orth_error = _qr_reorthogonalize(raw_basis)

    total_energy = float(singular.square().sum().detach().cpu())
    kept_energy = float(singular[:keep].square().sum().detach().cpu())
    return basis, {
        "source_row_count": int(values.shape[0]),
        "effective_rank": int(effective),
        "kept_rank": int(keep),
        "energy_fraction_retained": (
            kept_energy / total_energy if total_energy > 0.0 else 0.0
        ),
        "singular_values": [
            float(x) for x in singular[:keep].detach().cpu().tolist()
        ],
        "reorthogonalization_method": method,
        "max_abs_BBt_minus_I": float(orth_error),
    }


# Patch only the numerical basis constructor. refresh_geometry() resolves the
# module global at runtime, so every initial/periodic/final refresh uses this.
V2._orthonormal_basis = stable_orthonormal_basis


if __name__ == "__main__":
    V2.main()
