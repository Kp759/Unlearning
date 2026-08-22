#!/usr/bin/env python3
"""MCF Directional SURE v3: safe + shared forget geometry with protected preservation.

This is a new SURE-family ablation built from the complementary v1/v2 findings:

* v1: GA in the full forget span gives strong forgetting but destroys locality.
* v2: GA only in the protected-orthogonal forget span preserves utility but is too weak.

v3 decomposes each sensitive token's current forget hidden states H_F,y relative to
an observed protected hidden bank H_P (relation-donor controls + external Wikipedia):

    H_shared,y = Proj_BP(H_F,y)
    H_safe,y   = H_F,y - H_shared,y

with bases B_shared,y and B_safe,y.  The untied LM-head selected-row update is
parameterized as three independent sparse FP32 deltas:

    Delta W_y = Delta W_safe,y + Delta W_shared,y + Delta W_preserve,y

subject to

    Delta W_safe,y     in span(B_safe,y)
    Delta W_shared,y   in span(B_shared,y)
    Delta W_preserve,y in span(B_P) intersect span(B_shared,y)^perp.

Thus the three output components cannot directly cancel each other.  GA is routed
through the safe and shared components, but only for direct records whose current
training-visible target_true-vs-target_new margin is below the solver margin.
The shared component is additionally clipped row-wise to a hard maximum FP32 logit
drift over the observed protected hidden bank.  GD / donor-locality / Wikipedia
preservation gradients are routed only through Delta W_preserve.

Sensitive input-embedding rows remain sparse trainable FP32 deltas and receive the
original mixed SURE GA + preservation signal.  The transformer and every Base
parameter are frozen.  Because input-row learning changes hidden states, protected,
safe, and shared geometry is refreshed periodically; output optimizer moments are
reset after reprojection while input optimizer state is retained.

The direct target_new field is never used as a replacement training target.  It is
used only in the training-visible direct margin gate, stopping criterion, and final
shared-delta scale selection.  No official MCF neighborhood, paraphrase, retain-1000,
or PPL evaluation example is visible to training or checkpoint selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch
from torch import nn
from tqdm import tqdm

import gagd_compare as gagd
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_sensitive_rows_directional_gagd_v2 as V2
import mcf_sensitive_rows_directional_gagd_v2_stable as V2S
import mcf_sensitive_rows_projected_gagd as projected
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-LM-MCF-directional-GAGD-v3-safe-shared-protected"
PROTOCOL = "mcf_target_true_directional_gagd_safe_shared_protected_v3"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--input-row-lr", type=float, default=5e-5)
    p.add_argument("--lm-row-lr", type=float, default=1e-4)
    p.add_argument("--optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--row-delta-weight", type=float, default=0.01)

    p.add_argument("--basis-refresh-every", type=int, default=25)
    p.add_argument("--protected-basis-rank", type=int, default=0)
    p.add_argument("--safe-rank", type=int, default=0)
    p.add_argument("--shared-rank", type=int, default=0)
    p.add_argument(
        "--shared-protected-logit-drift-max",
        type=float,
        default=0.05,
        help="Hard per-sensitive-row FP32 logit drift cap of shared GA on H_P",
    )
    p.add_argument("--solver-margin", type=float, default=0.25)
    p.add_argument("--acceptance-margin", type=float, default=0.05)
    p.add_argument(
        "--candidate-shared-scales",
        default="0,0.03125,0.0625,0.09375,0.125,0.1875,0.25,0.375,0.5,0.625,0.75,0.875,1",
    )

    p.add_argument("--subject-control-count", type=int, default=4)
    p.add_argument("--locality-batch-size", type=int, default=4)
    p.add_argument("--locality-cache-batch-size", type=int, default=8)
    p.add_argument("--locality-kl-weight", type=float, default=2.0)
    p.add_argument("--locality-sensitive-logit-weight", type=float, default=5.0)

    p.add_argument("--utility-wikipedia-dir", required=True)
    p.add_argument("--utility-sample-size", type=int, default=200)
    p.add_argument("--utility-batch-size", type=int, default=4)
    p.add_argument("--utility-cache-batch-size", type=int, default=8)
    p.add_argument("--utility-max-length", type=int, default=128)
    p.add_argument("--utility-seed", type=int, default=1)
    p.add_argument("--utility-exclude-first", type=int, default=20)
    p.add_argument("--utility-kl-weight", type=float, default=2.0)

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)

    positive = (
        a.forget_num, a.steps, a.batch_size, a.cache_batch_size,
        a.input_row_lr, a.lm_row_lr, a.check_every, a.ga_weight,
        a.basis_refresh_every, a.subject_control_count,
        a.locality_batch_size, a.locality_cache_batch_size,
        a.locality_kl_weight, a.locality_sensitive_logit_weight,
        a.utility_sample_size, a.utility_batch_size, a.utility_cache_batch_size,
        a.utility_max_length, a.utility_kl_weight,
        a.shared_protected_logit_drift_max,
    )
    if any(float(v) <= 0 for v in positive):
        p.error("counts, LRs, cadence, positive weights, and drift cap must be positive")
    nonnegative = (
        a.gd_weight, a.grad_clip, a.row_delta_weight,
        a.protected_basis_rank, a.safe_rank, a.shared_rank,
        a.acceptance_margin, a.solver_margin, a.utility_exclude_first,
    )
    if any(float(v) < 0 for v in nonnegative):
        p.error("GD/clip/delta/ranks/margins/exclusion must be non-negative")
    if a.solver_margin < a.acceptance_margin:
        p.error("solver-margin must be >= acceptance-margin")
    if a.utility_exclude_first < 20:
        p.error("utility-exclude-first must be at least 20 to protect fixed PPL prefix")
    try:
        a.candidate_shared_scales = core.parse_scales(a.candidate_shared_scales)
    except ValueError as exc:
        p.error(str(exc))
    if 1.0 not in a.candidate_shared_scales:
        p.error("candidate-shared-scales must include 1")
    return a


def _optimizer(
    input_param: nn.Parameter,
    output_params: Iterable[nn.Parameter],
    kind: str,
    input_lr: float,
    output_lr: float,
):
    groups = [
        {"params": [input_param], "lr": float(input_lr)},
        {"params": list(output_params), "lr": float(output_lr)},
    ]
    if kind == "sgd":
        return torch.optim.SGD(groups)
    if kind == "adam":
        return torch.optim.Adam(groups)
    return torch.optim.AdamW(groups, weight_decay=0.0)


def _stable_basis(rows: torch.Tensor, rank_cap: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    return V2S.stable_orthonormal_basis(rows, int(rank_cap))


def _basis_projection(values: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    x = values.float()
    b = basis.to(device=x.device, dtype=torch.float32)
    if not b.numel():
        return torch.zeros_like(x)
    return (x @ b.T) @ b


def build_safe_shared_bases(
    forget_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    selected_ids: Sequence[int],
    protected_basis: torch.Tensor,
    *,
    safe_rank: int = 0,
    shared_rank: int = 0,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
    """Token-wise decomposition of forget geometry into safe and protected-shared spans."""
    h = forget_hidden.detach().float()
    tids = target_ids.detach().long().to(h.device)
    bp = protected_basis.to(device=h.device, dtype=torch.float32)
    if h.ndim != 2 or tids.ndim != 1 or h.shape[0] != tids.shape[0]:
        raise ValueError("forget hidden/target-id shape mismatch")

    safe_bases: Dict[int, torch.Tensor] = {}
    shared_bases: Dict[int, torch.Tensor] = {}
    receipt: Dict[str, Any] = {}
    for token_id in sorted({int(x) for x in selected_ids}):
        rows = torch.nonzero(tids == token_id, as_tuple=False).flatten()
        if not rows.numel():
            raise RuntimeError(f"selected token {token_id} has no forget hidden states")
        hf = h.index_select(0, rows).contiguous()
        shared_rows = _basis_projection(hf, bp)
        safe_rows = hf - shared_rows

        safe_basis, safe_diag = _stable_basis(safe_rows, int(safe_rank))
        shared_basis, shared_diag = _stable_basis(shared_rows, int(shared_rank))
        forget_basis, forget_diag = _stable_basis(hf, 0)

        forget_energy = float(hf.square().sum().detach().cpu())
        safe_energy = float(safe_rows.square().sum().detach().cpu())
        shared_energy = float(shared_rows.square().sum().detach().cpu())
        safe_bp_overlap = (
            float((safe_basis @ bp.T).abs().max().detach().cpu())
            if safe_basis.numel() and bp.numel() else 0.0
        )
        shared_bp_residual = 0.0
        if shared_basis.numel() and bp.numel():
            residual = shared_basis - _basis_projection(shared_basis, bp)
            shared_bp_residual = float(residual.abs().max().detach().cpu())
        safe_shared_overlap = (
            float((safe_basis @ shared_basis.T).abs().max().detach().cpu())
            if safe_basis.numel() and shared_basis.numel() else 0.0
        )

        principal_cosines: List[float] = []
        if forget_basis.numel() and bp.numel():
            principal_cosines = [
                float(x)
                for x in torch.linalg.svdvals(forget_basis @ bp.T)
                .clamp(0.0, 1.0).detach().cpu().tolist()
            ]

        safe_bases[token_id] = safe_basis.detach()
        shared_bases[token_id] = shared_basis.detach()
        receipt[str(token_id)] = {
            "case_count": int(rows.numel()),
            "forget_energy": forget_energy,
            "safe_energy": safe_energy,
            "shared_energy": shared_energy,
            "safe_energy_fraction": safe_energy / forget_energy if forget_energy > 0 else 0.0,
            "shared_energy_fraction": shared_energy / forget_energy if forget_energy > 0 else 0.0,
            "energy_fraction_sum": (
                (safe_energy + shared_energy) / forget_energy if forget_energy > 0 else 0.0
            ),
            "safe_rank": int(safe_basis.shape[0]),
            "shared_rank": int(shared_basis.shape[0]),
            "forget_rank": int(forget_basis.shape[0]),
            "max_abs_safe_BP_overlap": safe_bp_overlap,
            "max_abs_shared_outside_BP": shared_bp_residual,
            "max_abs_safe_shared_overlap": safe_shared_overlap,
            "principal_cosines_forget_vs_protected": principal_cosines,
            "principal_cosine_max": max(principal_cosines) if principal_cosines else 0.0,
            "principal_cosine_min": min(principal_cosines) if principal_cosines else 0.0,
            "safe_basis_diagnostics": safe_diag,
            "shared_basis_diagnostics": shared_diag,
            "forget_basis_diagnostics": forget_diag,
        }
    return safe_bases, shared_bases, receipt


def project_rows_protected_only(
    rows: torch.Tensor,
    row_ids: Sequence[int],
    protected_basis: torch.Tensor,
    shared_bases: Mapping[int, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Project each row into BP with its token's B_shared component removed."""
    protected, pdiag = V2.project_rows_global_span(rows, protected_basis)
    shared_part, _ = V2.project_rows_tokenwise_span(protected, row_ids, shared_bases)
    result = protected - shared_part
    return result, {
        "protected_kept_norm": float(pdiag["kept_norm"]),
        "shared_removed_norm": float(shared_part.norm().detach().cpu()),
        "protected_only_norm": float(result.norm().detach().cpu()),
    }


@torch.no_grad()
def enforce_shared_drift_budget_(
    shared_delta: torch.Tensor,
    protected_hidden: torch.Tensor,
    max_abs_drift: float,
) -> Dict[str, float]:
    """Row-wise scale shared GA so |h_p dot delta_y| never exceeds the budget."""
    if shared_delta.ndim != 2 or protected_hidden.ndim != 2:
        raise ValueError("shared delta and protected hidden must be matrices")
    if shared_delta.shape[1] != protected_hidden.shape[1]:
        raise ValueError("shared/protected hidden-size mismatch")
    h = protected_hidden.to(device=shared_delta.device, dtype=torch.float32)
    values = shared_delta.detach().float()
    if not h.numel() or not values.numel():
        return {
            "max_drift_before": 0.0,
            "max_drift_after": 0.0,
            "scaled_row_count": 0,
            "minimum_row_scale": 1.0,
        }
    drift = (h @ values.T).abs()
    row_max = drift.max(dim=0).values
    budget = float(max_abs_drift)
    scales = torch.ones_like(row_max)
    violating = row_max > budget
    scales[violating] = budget / row_max[violating].clamp_min(1e-12)
    shared_delta.mul_(scales[:, None].to(dtype=shared_delta.dtype))
    after = (h @ shared_delta.detach().float().T).abs().max()
    return {
        "max_drift_before": float(row_max.max().detach().cpu()),
        "max_drift_after": float(after.detach().cpu()),
        "scaled_row_count": int(violating.sum().item()),
        "minimum_row_scale": float(scales.min().detach().cpu()),
    }


@torch.no_grad()
def enforce_output_geometry_(
    safe_delta: torch.Tensor,
    shared_delta: torch.Tensor,
    preserve_delta: torch.Tensor,
    row_ids: Sequence[int],
    safe_bases: Mapping[int, torch.Tensor],
    shared_bases: Mapping[int, torch.Tensor],
    protected_basis: torch.Tensor,
    protected_hidden: torch.Tensor,
    shared_drift_max: float,
) -> Dict[str, Any]:
    safe_proj, safe_diag = V2.project_rows_tokenwise_span(
        safe_delta.detach(), row_ids, safe_bases
    )
    shared_proj, shared_diag = V2.project_rows_tokenwise_span(
        shared_delta.detach(), row_ids, shared_bases
    )
    preserve_proj, preserve_diag = project_rows_protected_only(
        preserve_delta.detach(), row_ids, protected_basis, shared_bases
    )
    safe_removed = safe_delta.detach().float() - safe_proj
    shared_removed = shared_delta.detach().float() - shared_proj
    preserve_removed = preserve_delta.detach().float() - preserve_proj
    safe_delta.copy_(safe_proj.to(dtype=safe_delta.dtype))
    shared_delta.copy_(shared_proj.to(dtype=shared_delta.dtype))
    preserve_delta.copy_(preserve_proj.to(dtype=preserve_delta.dtype))
    budget = enforce_shared_drift_budget_(
        shared_delta, protected_hidden, float(shared_drift_max)
    )
    return {
        "safe_removed_norm": float(safe_removed.norm().cpu()),
        "shared_removed_norm": float(shared_removed.norm().cpu()),
        "preserve_removed_norm": float(preserve_removed.norm().cpu()),
        "safe_kept_norm": float(safe_diag["kept_norm"]),
        "shared_kept_norm": float(shared_diag["kept_norm"]),
        "preserve_protected_only_norm": float(preserve_diag["protected_only_norm"]),
        "shared_drift_budget": budget,
    }


def geometry_summary(
    protected_diag: Mapping[str, Any],
    token_receipt: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    safe_f = [float(x["safe_energy_fraction"]) for x in token_receipt.values()]
    shared_f = [float(x["shared_energy_fraction"]) for x in token_receipt.values()]
    safe_r = [int(x["safe_rank"]) for x in token_receipt.values()]
    shared_r = [int(x["shared_rank"]) for x in token_receipt.values()]
    pc_max = [float(x["principal_cosine_max"]) for x in token_receipt.values()]
    return {
        "protected_source_row_count": int(protected_diag["source_row_count"]),
        "protected_effective_rank": int(protected_diag["effective_rank"]),
        "protected_kept_rank": int(protected_diag["kept_rank"]),
        "sensitive_token_count": int(len(token_receipt)),
        "safe_rank_min": min(safe_r) if safe_r else 0,
        "safe_rank_max": max(safe_r) if safe_r else 0,
        "shared_rank_min": min(shared_r) if shared_r else 0,
        "shared_rank_max": max(shared_r) if shared_r else 0,
        "safe_energy_fraction_min": min(safe_f) if safe_f else 0.0,
        "safe_energy_fraction_mean": sum(safe_f) / len(safe_f) if safe_f else 0.0,
        "safe_energy_fraction_max": max(safe_f) if safe_f else 0.0,
        "shared_energy_fraction_min": min(shared_f) if shared_f else 0.0,
        "shared_energy_fraction_mean": sum(shared_f) / len(shared_f) if shared_f else 0.0,
        "shared_energy_fraction_max": max(shared_f) if shared_f else 0.0,
        "principal_cosine_max_mean": sum(pc_max) / len(pc_max) if pc_max else 0.0,
        "principal_cosine_max_max": max(pc_max) if pc_max else 0.0,
    }


@torch.no_grad()
def refresh_geometry(
    model: nn.Module,
    tok: Any,
    cases: Sequence[Any],
    target_ids: torch.Tensor,
    selected_ids: Sequence[int],
    locality_prompts: Sequence[str],
    utility_prompts: Sequence[str],
    device: torch.device,
    *,
    forget_batch_size: int,
    locality_batch_size: int,
    utility_batch_size: int,
    protected_rank: int,
    safe_rank: int,
    shared_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, torch.Tensor], Dict[int, torch.Tensor], Dict[str, Any]]:
    forget_hidden = core.forward_last_hidden(
        model, tok, cases, device, int(forget_batch_size)
    ).detach().float()
    locality_hidden = V2._prompt_hidden(
        model, tok, locality_prompts, device, int(locality_batch_size)
    )
    utility_hidden = V2._prompt_hidden(
        model, tok, utility_prompts, device, int(utility_batch_size)
    )
    protected_hidden = torch.cat([locality_hidden, utility_hidden], dim=0).detach().float()
    protected_basis, protected_diag = _stable_basis(protected_hidden, int(protected_rank))
    safe_bases, shared_bases, token_receipt = build_safe_shared_bases(
        forget_hidden,
        target_ids.to(device=forget_hidden.device),
        selected_ids,
        protected_basis,
        safe_rank=int(safe_rank),
        shared_rank=int(shared_rank),
    )
    receipt = {
        "protected": protected_diag,
        "sensitive_tokens": token_receipt,
        "summary": geometry_summary(protected_diag, token_receipt),
    }
    return (
        protected_hidden.detach(),
        protected_basis.detach(),
        safe_bases,
        shared_bases,
        receipt,
    )


def _direct_margins(
    model: nn.Module,
    tok: Any,
    instances: Sequence[Any],
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> torch.Tensor:
    return stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(batch_size),
        "target_true", "target_new"
    ).detach().float().cpu()


def _margin_report(margins: torch.Tensor, threshold: float) -> Dict[str, Any]:
    m = margins.detach().float().cpu()
    return {
        "failures": int((m < float(threshold)).sum().item()),
        "minimum_margin": float(m.min()),
        "mean_margin": float(m.mean()),
        "maximum_margin": float(m.max()),
        "threshold": float(threshold),
    }


def _active_case_indices(
    cases: Sequence[Any], margins: torch.Tensor, solver_margin: float
) -> Tuple[List[int], List[int]]:
    active_records = [
        int(i) for i, value in enumerate(margins.tolist())
        if float(value) < float(solver_margin)
    ]
    active_set = set(active_records)
    active_cases = [
        i for i, case in enumerate(cases)
        if int(case.record_position) in active_set
    ]
    return active_records, active_cases


def _zero_if_none(value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(reference) if value is None else value


@torch.no_grad()
def directional_residual_diagnostics(
    safe_delta: torch.Tensor,
    shared_delta: torch.Tensor,
    preserve_delta: torch.Tensor,
    row_ids: Sequence[int],
    safe_bases: Mapping[int, torch.Tensor],
    shared_bases: Mapping[int, torch.Tensor],
    protected_basis: torch.Tensor,
    protected_hidden: torch.Tensor,
) -> Dict[str, float]:
    bp = protected_basis.to(device=safe_delta.device, dtype=torch.float32)
    hp = protected_hidden.to(device=safe_delta.device, dtype=torch.float32)
    safe_bp: List[float] = []
    shared_out_bp: List[float] = []
    preserve_shared: List[float] = []
    safe_shared_cos: List[float] = []
    for i, tid in enumerate(row_ids):
        safe = safe_delta[i].float()
        shared = shared_delta[i].float()
        preserve = preserve_delta[i].float()
        safe_bp.append(float((safe @ bp.T).abs().max().cpu()) if bp.numel() else 0.0)
        shared_resid = shared - _basis_projection(shared[None, :], bp)[0]
        shared_out_bp.append(float(shared_resid.abs().max().cpu()))
        bs = shared_bases.get(int(tid))
        if bs is not None and bs.numel():
            b = bs.to(device=safe_delta.device, dtype=torch.float32)
            preserve_shared.append(float((preserve @ b.T).abs().max().cpu()))
        else:
            preserve_shared.append(0.0)
        denom = safe.norm() * shared.norm()
        safe_shared_cos.append(float((safe @ shared).abs().div(denom.clamp_min(1e-12)).cpu()))
    shared_drift = (
        float((hp @ shared_delta.detach().float().T).abs().max().cpu())
        if hp.numel() and shared_delta.numel() else 0.0
    )
    return {
        "max_abs_safe_dot_protected_basis": max(safe_bp) if safe_bp else 0.0,
        "max_abs_shared_outside_protected_basis": max(shared_out_bp) if shared_out_bp else 0.0,
        "max_abs_preserve_dot_shared_basis": max(preserve_shared) if preserve_shared else 0.0,
        "max_abs_cosine_safe_vs_shared": max(safe_shared_cos) if safe_shared_cos else 0.0,
        "shared_protected_logit_drift_max": shared_drift,
        "checked_rows": int(len(row_ids)),
    }


def _choose_shared_scale(
    model: nn.Module,
    tok: Any,
    instances: Sequence[Any],
    shared_param: torch.Tensor,
    scales: Sequence[float],
    acceptance_margin: float,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    original = shared_param.detach().clone()
    reports: List[Dict[str, Any]] = []
    with torch.no_grad():
        for scale in sorted({float(x) for x in scales}):
            shared_param.copy_(original * float(scale))
            margins = _direct_margins(
                model, tok, instances, device, llama_like, int(batch_size)
            )
            report = _margin_report(margins, float(acceptance_margin))
            report["scale"] = float(scale)
            reports.append(report)
        zero = [r for r in reports if int(r["failures"]) == 0]
        if zero:
            chosen = min(zero, key=lambda r: float(r["scale"]))
        else:
            chosen = min(
                reports,
                key=lambda r: (int(r["failures"]), float(r["scale"])),
            )
        chosen_scale = float(chosen["scale"])
        shared_param.copy_(original * chosen_scale)
    return chosen_scale, reports


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = stage2.load_locked(
        "mcf", visible_path, manifest_path, int(a.seed), int(a.forget_num)
    )
    contract_helpers.assert_target_contract(manifest)
    contract_helpers.validate_direct_only_records(records)

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    model.eval()

    cases = core.expand_sensitive_cases(
        records, tok, sensitive_field="target_true", llama_like=llama_like
    )
    if not cases:
        raise RuntimeError("no target_true sensitive PredictionCases were created")
    same_prompt_base_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=int(a.cache_batch_size)
    )
    all_tids = core.official_target_ids(
        tok, cases, llama_like=llama_like, device=device
    )
    sensitive_ids = sorted(set(int(x) for x in all_tids.detach().cpu().tolist()))
    sensitive_ids = [x for x in sensitive_ids if x not in gagd.special_token_ids(tok)]
    if not sensitive_ids:
        raise RuntimeError("no non-special target_true sensitive rows")

    benchmark_instances = stage2.mcf_instances(records)
    margins_before = _direct_margins(
        model, tok, benchmark_instances, device, llama_like, int(a.batch_size)
    )
    benchmark_before = _margin_report(margins_before, float(a.acceptance_margin))

    locality_prompts, locality_protected, locality_receipt = (
        projected.build_relation_locality_controls(
            records, tok, benchmark_instances, int(a.subject_control_count)
        )
    )
    print(f"Caching Base locality references for {len(locality_prompts)} prompts...", flush=True)
    _base_local_hidden, base_local_logits = projected.cache_relation_locality_reference(
        model, tok, locality_prompts, device, int(a.locality_cache_batch_size)
    )

    utility_prompts, utility_receipt = wikipedia_utility.build_utility_prompts(
        tok,
        Path(a.utility_wikipedia_dir).resolve(),
        sample_size=int(a.utility_sample_size),
        seed=int(a.utility_seed),
        exclude_first=int(a.utility_exclude_first),
        max_length=int(a.utility_max_length),
    )
    print(f"Caching Base Wikipedia logits for {len(utility_prompts)} prompts...", flush=True)
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Directional SURE v3 requires untied LM head")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("all Base parameters must be frozen; sparse hooks carry learning")

    ids_device = torch.tensor(sensitive_ids, dtype=torch.long, device=input_layer.weight.device)
    selected_base_input = input_layer.weight.detach().index_select(0, ids_device).float().cpu()
    selected_base_output = output_layer.weight.detach().index_select(
        0, ids_device.to(output_layer.weight.device)
    ).float().cpu()

    input_module = V2.SparseInputRowDelta(input_layer, sensitive_ids)
    hidden_size = int(output_layer.weight.shape[1])
    safe_module = core.SelectedRowDelta(
        len(sensitive_ids), hidden_size, direction_basis=None, device=device
    )
    shared_module = core.SelectedRowDelta(
        len(sensitive_ids), hidden_size, direction_basis=None, device=device
    )
    preserve_module = core.SelectedRowDelta(
        len(sensitive_ids), hidden_size, direction_basis=None, device=device
    )
    if any(m.raw_delta is None for m in (safe_module, shared_module, preserve_module)):
        raise RuntimeError("v3 requires unrestricted sparse FP32 output deltas")

    output_hook = core.register_output_delta_hook(
        output_layer,
        sensitive_ids,
        lambda: (
            safe_module.effective_delta()
            + shared_module.effective_delta()
            + preserve_module.effective_delta()
        ),
    )
    output_params = [
        safe_module.raw_delta,
        shared_module.raw_delta,
        preserve_module.raw_delta,
    ]
    opt = _optimizer(
        input_module.delta,
        output_params,
        a.optimizer,
        float(a.input_row_lr),
        float(a.lm_row_lr),
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    core.write_json(out_dir / "relation_locality_receipt.json", locality_receipt)
    core.write_json(out_dir / "wikipedia_utility_receipt.json", utility_receipt)

    print("Building initial safe/shared/protected geometry...", flush=True)
    (
        protected_hidden,
        protected_basis,
        safe_bases,
        shared_bases,
        geometry_receipt,
    ) = refresh_geometry(
        model, tok, cases, all_tids, sensitive_ids,
        locality_prompts, utility_prompts, device,
        forget_batch_size=int(a.cache_batch_size),
        locality_batch_size=int(a.locality_cache_batch_size),
        utility_batch_size=int(a.utility_cache_batch_size),
        protected_rank=int(a.protected_basis_rank),
        safe_rank=int(a.safe_rank),
        shared_rank=int(a.shared_rank),
    )
    geometry_receipt["step"] = 0
    core.write_json(out_dir / "initial_geometry_receipt.json", geometry_receipt)
    print("Initial geometry:", geometry_receipt["summary"], flush=True)

    enforce_output_geometry_(
        safe_module.raw_delta, shared_module.raw_delta, preserve_module.raw_delta,
        sensitive_ids, safe_bases, shared_bases, protected_basis, protected_hidden,
        float(a.shared_protected_logit_drift_max),
    )

    margins = margins_before.clone()
    active_records, active_case_ids = _active_case_indices(
        cases, margins, float(a.solver_margin)
    )
    active_sampler = (
        core.IndexSampler(len(active_case_ids), int(a.batch_size), int(a.seed) + 91001)
        if active_case_ids else None
    )
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 91003
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 91007
    )

    geometry_log = (out_dir / "basis_refresh_log.jsonl").open("w", encoding="utf-8")
    direct_log = (out_dir / "direct_gate_log.jsonl").open("w", encoding="utf-8")
    train_log = (out_dir / "train_log.jsonl").open("w", encoding="utf-8")
    geometry_log.write(json.dumps(geometry_receipt) + "\n")
    direct_log.write(json.dumps({
        "step": 0,
        "active_record_count": len(active_records),
        "active_records": active_records,
        "solver": _margin_report(margins, float(a.solver_margin)),
        "acceptance": _margin_report(margins, float(a.acceptance_margin)),
    }) + "\n")
    geometry_log.flush(); direct_log.flush()

    stopped_step = int(a.steps)
    try:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF Directional SURE v3"):
            if active_sampler is None:
                stopped_step = int(step - 1)
                break

            if step > 1 and (step - 1) % int(a.basis_refresh_every) == 0:
                (
                    protected_hidden,
                    protected_basis,
                    safe_bases,
                    shared_bases,
                    refresh,
                ) = refresh_geometry(
                    model, tok, cases, all_tids, sensitive_ids,
                    locality_prompts, utility_prompts, device,
                    forget_batch_size=int(a.cache_batch_size),
                    locality_batch_size=int(a.locality_cache_batch_size),
                    utility_batch_size=int(a.utility_cache_batch_size),
                    protected_rank=int(a.protected_basis_rank),
                    safe_rank=int(a.safe_rank),
                    shared_rank=int(a.shared_rank),
                )
                refresh["step"] = int(step - 1)
                refresh["output_reprojection"] = enforce_output_geometry_(
                    safe_module.raw_delta,
                    shared_module.raw_delta,
                    preserve_module.raw_delta,
                    sensitive_ids, safe_bases, shared_bases,
                    protected_basis, protected_hidden,
                    float(a.shared_protected_logit_drift_max),
                )
                for param in output_params:
                    opt.state[param].clear()
                refresh["output_optimizer_moments_reset"] = True
                refresh["input_optimizer_moments_preserved"] = True
                geometry_log.write(json.dumps(refresh) + "\n")
                geometry_log.flush()

            f_local = active_sampler.next()
            fidx = [active_case_ids[i] for i in f_local]
            forget_batch = [cases[i] for i in fidx]
            lidx = locality_sampler.next()
            locality_batch = [locality_prompts[i] for i in lidx]
            locality_ids = [locality_protected[i] for i in lidx]
            uidx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in uidx]

            opt.zero_grad(set_to_none=True)
            logits = core.forward_last_logits(model, tok, forget_batch, device)
            tids = core.official_target_ids(
                tok, forget_batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(
                logits, same_prompt_base_logits[fidx], tids
            )

            _lh, local_logits = projected._prompt_hidden_and_logits(
                model, tok, locality_batch, device
            )
            local_base = base_local_logits[lidx]
            lkl = projected.locality_kl(local_logits, local_base)
            lrow = projected.protected_sensitive_logit_mse(
                local_logits, local_base, locality_ids
            )
            utility_logits = wikipedia_utility._forward_prompt_logits(
                model, tok, utility_batch, device
            )
            ukl = wikipedia_utility.utility_kl(
                utility_logits, utility_base_logits[uidx]
            )

            input_reg = input_module.delta.square().mean()
            safe_reg = safe_module.raw_delta.square().mean()
            shared_reg = shared_module.raw_delta.square().mean()
            preserve_reg = preserve_module.raw_delta.square().mean()
            ga_objective = (
                float(a.ga_weight) * ga
                + 0.5 * float(a.row_delta_weight) * (input_reg + safe_reg + shared_reg)
            )
            preserve_objective = (
                float(a.gd_weight) * gd
                + float(a.locality_kl_weight) * lkl
                + float(a.locality_sensitive_logit_weight) * lrow
                + float(a.utility_kl_weight) * ukl
                + 0.5 * float(a.row_delta_weight) * (input_reg + preserve_reg)
            )
            if not torch.isfinite(ga_objective) or not torch.isfinite(preserve_objective):
                raise FloatingPointError(f"non-finite v3 objective at step {step}")

            ga_input_grad, safe_grad, shared_grad = torch.autograd.grad(
                ga_objective,
                [input_module.delta, safe_module.raw_delta, shared_module.raw_delta],
                retain_graph=True,
                allow_unused=True,
            )
            preserve_input_grad, preserve_grad = torch.autograd.grad(
                preserve_objective,
                [input_module.delta, preserve_module.raw_delta],
                retain_graph=False,
                allow_unused=True,
            )
            ga_input_grad = _zero_if_none(ga_input_grad, input_module.delta)
            preserve_input_grad = _zero_if_none(preserve_input_grad, input_module.delta)
            safe_grad = _zero_if_none(safe_grad, safe_module.raw_delta)
            shared_grad = _zero_if_none(shared_grad, shared_module.raw_delta)
            preserve_grad = _zero_if_none(preserve_grad, preserve_module.raw_delta)

            safe_grad_proj, safe_grad_diag = V2.project_rows_tokenwise_span(
                safe_grad, sensitive_ids, safe_bases
            )
            shared_grad_proj, shared_grad_diag = V2.project_rows_tokenwise_span(
                shared_grad, sensitive_ids, shared_bases
            )
            preserve_grad_proj, preserve_grad_diag = project_rows_protected_only(
                preserve_grad, sensitive_ids, protected_basis, shared_bases
            )

            input_module.delta.grad = (
                ga_input_grad + preserve_input_grad
            ).to(input_module.delta.dtype)
            safe_module.raw_delta.grad = safe_grad_proj.to(safe_module.raw_delta.dtype)
            shared_module.raw_delta.grad = shared_grad_proj.to(shared_module.raw_delta.dtype)
            preserve_module.raw_delta.grad = preserve_grad_proj.to(preserve_module.raw_delta.dtype)

            params = [input_module.delta] + output_params
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at step {step}")
            opt.step()
            geometry_projection = enforce_output_geometry_(
                safe_module.raw_delta,
                shared_module.raw_delta,
                preserve_module.raw_delta,
                sensitive_ids, safe_bases, shared_bases,
                protected_basis, protected_hidden,
                float(a.shared_protected_logit_drift_max),
            )

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                margins = _direct_margins(
                    model, tok, benchmark_instances, device, llama_like, int(a.batch_size)
                )
                active_records, active_case_ids = _active_case_indices(
                    cases, margins, float(a.solver_margin)
                )
                active_sampler = (
                    core.IndexSampler(
                        len(active_case_ids), int(a.batch_size),
                        int(a.seed) + 91001 + int(step)
                    ) if active_case_ids else None
                )
                gate_row = {
                    "step": int(step),
                    "active_record_count": int(len(active_records)),
                    "active_records": active_records,
                    "solver": _margin_report(margins, float(a.solver_margin)),
                    "acceptance": _margin_report(margins, float(a.acceptance_margin)),
                }
                direct_log.write(json.dumps(gate_row) + "\n")
                direct_log.flush()

                total_output = (
                    safe_module.effective_delta()
                    + shared_module.effective_delta()
                    + preserve_module.effective_delta()
                )
                row = {
                    "step": int(step),
                    "active_record_count": int(len(active_records)),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_same_prompt_non_sensitive_kl": float(gd.detach().cpu()),
                    "relation_locality_kl": float(lkl.detach().cpu()),
                    "relation_sensitive_logit_mse": float(lrow.detach().cpu()),
                    "wikipedia_utility_kl": float(ukl.detach().cpu()),
                    "input_delta_mse": float(input_module.delta.square().mean().detach().cpu()),
                    "safe_delta_mse": float(safe_module.raw_delta.square().mean().detach().cpu()),
                    "shared_delta_mse": float(shared_module.raw_delta.square().mean().detach().cpu()),
                    "preserve_delta_mse": float(preserve_module.raw_delta.square().mean().detach().cpu()),
                    "total_output_delta_mse": float(total_output.square().mean().detach().cpu()),
                    "ga_input_gradient_norm": float(ga_input_grad.norm().detach().cpu()),
                    "preserve_input_gradient_norm": float(preserve_input_grad.norm().detach().cpu()),
                    "safe_gradient_kept_norm": float(safe_grad_diag["kept_norm"]),
                    "shared_gradient_kept_norm": float(shared_grad_diag["kept_norm"]),
                    "preserve_gradient_protected_only_norm": float(preserve_grad_diag["protected_only_norm"]),
                    "geometry_projection": geometry_projection,
                    "benchmark_retain_seen": 0,
                    "official_neighborhood_seen": 0,
                    "official_paraphrase_seen": 0,
                    "PPL_seen": False,
                }
                train_log.write(json.dumps(row) + "\n")
                train_log.flush()

                if active_sampler is None:
                    stopped_step = int(step)
                    break
    finally:
        geometry_log.close(); direct_log.close(); train_log.close()

    del opt

    (
        protected_hidden,
        protected_basis,
        safe_bases,
        shared_bases,
        final_geometry,
    ) = refresh_geometry(
        model, tok, cases, all_tids, sensitive_ids,
        locality_prompts, utility_prompts, device,
        forget_batch_size=int(a.cache_batch_size),
        locality_batch_size=int(a.locality_cache_batch_size),
        utility_batch_size=int(a.utility_cache_batch_size),
        protected_rank=int(a.protected_basis_rank),
        safe_rank=int(a.safe_rank),
        shared_rank=int(a.shared_rank),
    )
    final_geometry["step"] = int(stopped_step)
    final_geometry["output_reprojection"] = enforce_output_geometry_(
        safe_module.raw_delta, shared_module.raw_delta, preserve_module.raw_delta,
        sensitive_ids, safe_bases, shared_bases,
        protected_basis, protected_hidden,
        float(a.shared_protected_logit_drift_max),
    )

    chosen_shared_scale, shared_scale_reports = _choose_shared_scale(
        model, tok, benchmark_instances, shared_module.raw_delta,
        a.candidate_shared_scales, float(a.acceptance_margin),
        device, llama_like, int(a.batch_size),
    )
    final_geometry["chosen_shared_scale"] = float(chosen_shared_scale)
    final_geometry["shared_scale_reports"] = shared_scale_reports
    core.write_json(out_dir / "final_geometry_receipt.json", final_geometry)
    core.write_json(out_dir / "shared_scale_reports.json", shared_scale_reports)

    benchmark_pre_materialize = _margin_report(
        _direct_margins(model, tok, benchmark_instances, device, llama_like, int(a.batch_size)),
        float(a.acceptance_margin),
    )
    locality_pre_materialize = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, base_local_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_pre_materialize = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_base_logits,
        device, int(a.utility_cache_batch_size)
    )
    directional_after = directional_residual_diagnostics(
        safe_module.raw_delta.detach(), shared_module.raw_delta.detach(),
        preserve_module.raw_delta.detach(), sensitive_ids,
        safe_bases, shared_bases, protected_basis, protected_hidden
    )

    input_delta_final = input_module.delta.detach().float().cpu()
    safe_final = safe_module.effective_delta().detach().float().cpu()
    shared_final = shared_module.effective_delta().detach().float().cpu()
    preserve_final = preserve_module.effective_delta().detach().float().cpu()
    total_output_final = safe_final + shared_final + preserve_final

    input_module.remove()
    output_hook.remove()
    V2._materialize_selected_delta(input_layer.weight, sensitive_ids, input_delta_final)
    core.materialize_output_delta(output_layer, sensitive_ids, total_output_final)
    model.eval()

    materialized_input = (
        input_layer.weight.index_select(0, ids_device).float().cpu() - selected_base_input
    )
    output_ids = ids_device.to(output_layer.weight.device)
    materialized_output = (
        output_layer.weight.index_select(0, output_ids).float().cpu() - selected_base_output
    )
    input_materialization_error = float((materialized_input - input_delta_final).abs().max())
    output_materialization_error = float((materialized_output - total_output_final).abs().max())

    benchmark_after = _margin_report(
        _direct_margins(model, tok, benchmark_instances, device, llama_like, int(a.batch_size)),
        float(a.acceptance_margin),
    )
    locality_after = projected.evaluate_locality_guards(
        model, tok, locality_prompts, locality_protected, base_local_logits,
        device, int(a.locality_cache_batch_size)
    )
    utility_after = wikipedia_utility.evaluate_utility_kl(
        model, tok, utility_prompts, utility_base_logits,
        device, int(a.utility_cache_batch_size)
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_model_path": str(Path(a.model_path).resolve()),
        "training_visible_path": str(visible_path),
        "split_manifest": str(manifest_path),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_unwanted": "requested_rewrite.target_true",
            "benchmark_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "target_new_used_as_replacement_training_target": False,
        "target_new_used_for_training_visible_direct_margin_gate": True,
        "target_new_used_for_shared_scale_selection": True,
        "transformer_trainable": False,
        "input_embedding_sensitive_rows_trainable": True,
        "input_embedding_non_sensitive_rows_trainable": False,
        "lm_head_untied": True,
        "lm_head_sensitive_rows_trainable": True,
        "lm_head_non_sensitive_rows_trainable": False,
        "output_components": {
            "safe_ga": "token-specific protected-orthogonal forget span",
            "shared_ga": "token-specific forget projection inside protected span with hard H_P logit-drift cap",
            "preservation": "protected span with token shared-forget component removed",
        },
        "selected_sensitive_token_ids": [int(x) for x in sensitive_ids],
        "selected_sensitive_row_count": int(len(sensitive_ids)),
        "official_neighborhood_seen": 0,
        "official_paraphrase_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "steps_requested": int(a.steps),
        "stopped_step": int(stopped_step),
        "solver_margin": float(a.solver_margin),
        "acceptance_margin": float(a.acceptance_margin),
        "shared_protected_logit_drift_max": float(a.shared_protected_logit_drift_max),
        "basis_refresh_every": int(a.basis_refresh_every),
        "protected_basis_rank_cap": int(a.protected_basis_rank),
        "safe_rank_cap": int(a.safe_rank),
        "shared_rank_cap": int(a.shared_rank),
        "chosen_shared_scale": float(chosen_shared_scale),
        "shared_scale_reports": shared_scale_reports,
        "weights": {
            "ga": float(a.ga_weight),
            "gd": float(a.gd_weight),
            "relation_locality_kl": float(a.locality_kl_weight),
            "relation_sensitive_logit": float(a.locality_sensitive_logit_weight),
            "wikipedia_utility_kl": float(a.utility_kl_weight),
            "row_delta": float(a.row_delta_weight),
        },
        "input_row_lr": float(a.input_row_lr),
        "lm_row_lr": float(a.lm_row_lr),
        "optimizer": a.optimizer,
        "initial_geometry": geometry_receipt,
        "final_geometry": final_geometry,
        "directional_residual_after": directional_after,
        "benchmark_pair_before": benchmark_before,
        "benchmark_pair_pre_materialize": benchmark_pre_materialize,
        "benchmark_pair_after": benchmark_after,
        "relation_locality_pre_materialize": locality_pre_materialize,
        "relation_locality_after": locality_after,
        "wikipedia_utility_pre_materialize": utility_pre_materialize,
        "wikipedia_utility_after": utility_after,
        "input_delta_mse": float(input_delta_final.square().mean()),
        "safe_output_delta_mse": float(safe_final.square().mean()),
        "shared_output_delta_mse": float(shared_final.square().mean()),
        "preserve_output_delta_mse": float(preserve_final.square().mean()),
        "input_materialization_max_abs_error": input_materialization_error,
        "output_materialization_max_abs_error": output_materialization_error,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "sensitive_rows_directional_gagd_v3_summary.json", summary)

    print("Directional SURE v3 checkpoint:", ckpt)
    print("Transformer trainable: False")
    print("Input sensitive rows trainable: True")
    print("LM head untied: True")
    print("Sensitive row count:", len(sensitive_ids))
    print("Stopped step:", stopped_step)
    print("Chosen shared scale:", chosen_shared_scale)
    print("Initial geometry:", geometry_receipt["summary"])
    print("Final geometry:", final_geometry["summary"])
    print("Benchmark pair before:", benchmark_before)
    print("Benchmark pair after:", benchmark_after)
    print("Relation-locality after:", locality_after)
    print("Wikipedia utility after:", utility_after)
    print("Directional residual after:", directional_after)
    print("Input materialization max abs error:", input_materialization_error)
    print("Output materialization max abs error:", output_materialization_error)
    print("Official MCF neighborhood/paraphrase/retain/PPL eval data were NOT used.")
    print("Run official evaluation only after this checkpoint is finalized.")


if __name__ == "__main__":
    main()
