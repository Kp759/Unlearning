#!/usr/bin/env python3
"""Soft retain-residualized SURE Stage 1.

Wrapper around sure_stage1_context_retain_residualized.py that replaces hard
retain-subspace subtraction with

    H_soft = H_F - alpha * Proj_QR(H_F),  0 <= alpha <= 1.

All optimization, leakage checks, direct constraints, scale selection, sparse
sensitive LM-head editing, and checkpoint writing remain inherited from the
hard-residualized implementation. alpha=1 reproduces hard residualization;
alpha=0 leaves the forget hidden states unchanged while still using the same
code path.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import sure_stage1_context_retain_residualized as hard


def _pop_float_arg(name: str, default: float) -> float:
    if name not in sys.argv:
        return float(default)
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a value")
    value = float(sys.argv[i + 1])
    del sys.argv[i:i + 2]
    return value


def _arg_value(name: str) -> str:
    if name not in sys.argv:
        raise SystemExit(f"Missing required argument {name}")
    i = sys.argv.index(name)
    if i + 1 >= len(sys.argv):
        raise SystemExit(f"{name} requires a value")
    return sys.argv[i + 1]


RESIDUAL_ALPHA = _pop_float_arg("--residual-alpha", 0.5)
if not 0.0 <= RESIDUAL_ALPHA <= 1.0:
    raise SystemExit("--residual-alpha must be in [0, 1]")
OUTPUT_DIR = _arg_value("--output-dir")


@torch.no_grad()
def build_soft_retain_residualized_bases(
    model,
    tok,
    forget_cases: Sequence[hard.core.SensitivePredictionCase],
    retain_cases: Sequence[hard.core.SensitivePredictionCase],
    *,
    selected_ids: Sequence[int],
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    retain_rank_cap: int,
    row_rank_cap: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[Dict[str, Any]], Dict[str, Any]]:
    forget_hidden = hard.core.forward_last_hidden(model, tok, forget_cases, device, batch_size)
    retain_hidden = hard.core.forward_last_hidden(model, tok, retain_cases, device, batch_size)
    if forget_hidden.ndim != 2 or retain_hidden.ndim != 2:
        raise RuntimeError("hidden-state caches must be rank-2 matrices")
    if forget_hidden.shape[1] != retain_hidden.shape[1]:
        raise RuntimeError("forget/retain hidden sizes differ")

    retain_max_rank = None if retain_rank_cap == 0 else int(retain_rank_cap)
    retain_basis = hard.core.orthonormal_row_basis(retain_hidden, max_rank=retain_max_rank)
    if retain_basis.ndim != 2 or retain_basis.shape[0] <= 0:
        raise RuntimeError("retain-context basis has zero numerical rank")

    tids = hard.core.official_target_ids(tok, forget_cases, llama_like=llama_like, device=device).detach()
    row_max_rank = None if row_rank_cap == 0 else int(row_rank_cap)

    q = retain_basis.to(device=forget_hidden.device, dtype=torch.float32)
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    all_residual_ratios: List[float] = []
    all_projection_ratios: List[float] = []
    all_removed_ratios: List[float] = []

    for token_id in [int(x) for x in selected_ids]:
        rows = forget_hidden[tids.eq(token_id)].float()
        if rows.numel() == 0:
            raise RuntimeError(f"Sensitive token {token_id} has no forget hidden rows")

        projection = (rows @ q.transpose(0, 1)) @ q
        residual = rows - float(RESIDUAL_ALPHA) * projection
        original_norm = rows.norm(dim=1).clamp_min(1e-12)
        residual_ratio = residual.norm(dim=1) / original_norm
        projection_ratio = projection.norm(dim=1) / original_norm
        removed_ratio = float(RESIDUAL_ALPHA) * projection_ratio

        basis = hard.core.orthonormal_row_basis(residual, max_rank=row_max_rank)
        if basis.ndim != 2 or basis.shape[0] <= 0:
            raise RuntimeError(f"Sensitive token {token_id} has zero soft-residual context rank")

        retain_dot = basis.float() @ q.transpose(0, 1)
        max_abs_retain_dot = float(retain_dot.abs().max().cpu()) if retain_dot.numel() else 0.0

        rr = [float(x) for x in residual_ratio.detach().cpu().tolist()]
        pp = [float(x) for x in projection_ratio.detach().cpu().tolist()]
        mm = [float(x) for x in removed_ratio.detach().cpu().tolist()]
        all_residual_ratios.extend(rr)
        all_projection_ratios.extend(pp)
        all_removed_ratios.extend(mm)

        bases.append(basis.detach().float().contiguous())
        reports.append({
            "token_id": token_id,
            "context_count": int(rows.shape[0]),
            "context_rank_after_residualization": int(basis.shape[0]),
            "hidden_size": int(basis.shape[1]),
            "residual_alpha": float(RESIDUAL_ALPHA),
            "mean_residual_norm_ratio": float(residual_ratio.mean().cpu()),
            "min_residual_norm_ratio": float(residual_ratio.min().cpu()),
            "max_residual_norm_ratio": float(residual_ratio.max().cpu()),
            "mean_retain_projection_norm_ratio": float(projection_ratio.mean().cpu()),
            "mean_removed_projection_norm_ratio": float(removed_ratio.mean().cpu()),
            "max_abs_basis_dot_retain_basis": max_abs_retain_dot,
        })

    residual_tensor = torch.tensor(all_residual_ratios, dtype=torch.float32)
    projection_tensor = torch.tensor(all_projection_ratios, dtype=torch.float32)
    removed_tensor = torch.tensor(all_removed_ratios, dtype=torch.float32)
    diagnostics: Dict[str, Any] = {
        "residualization_mode": "soft",
        "residual_alpha": float(RESIDUAL_ALPHA),
        "retain_hidden_count": int(retain_hidden.shape[0]),
        "retain_basis_rank": int(retain_basis.shape[0]),
        "hidden_size": int(retain_basis.shape[1]),
        "forget_prediction_case_count": int(forget_hidden.shape[0]),
        "mean_residual_norm_ratio": float(residual_tensor.mean()),
        "median_residual_norm_ratio": float(residual_tensor.median()),
        "min_residual_norm_ratio": float(residual_tensor.min()),
        "max_residual_norm_ratio": float(residual_tensor.max()),
        "mean_retain_projection_norm_ratio": float(projection_tensor.mean()),
        "median_retain_projection_norm_ratio": float(projection_tensor.median()),
        "min_retain_projection_norm_ratio": float(projection_tensor.min()),
        "max_retain_projection_norm_ratio": float(projection_tensor.max()),
        "mean_removed_projection_norm_ratio": float(removed_tensor.mean()),
        "median_removed_projection_norm_ratio": float(removed_tensor.median()),
        "max_abs_row_basis_dot_retain_basis": max(float(r["max_abs_basis_dot_retain_basis"]) for r in reports),
    }
    return retain_basis.detach().float().contiguous(), bases, reports, diagnostics


hard.build_retain_residualized_bases = build_soft_retain_residualized_bases
hard.main()

p = Path(OUTPUT_DIR).resolve() / "config_used.json"
if p.exists():
    d = json.loads(p.read_text(encoding="utf-8"))
    d["method"] = "SURE-LM-retain-soft-residualized-row-specific-stage1"
    d["protocol"] = "sure_retain_soft_residualized_stage1_v1"
    d["retain_residualization"] = True
    d["retain_residualization_mode"] = "soft"
    d["residual_alpha"] = float(RESIDUAL_ALPHA)
    p.write_text(json.dumps(d, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("soft residual alpha:", RESIDUAL_ALPHA)
