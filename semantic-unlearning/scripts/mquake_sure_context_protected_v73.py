#!/usr/bin/env python3
"""SURE-MQuAKE V7.3: context-protected dual-basis sparse LM-head repair.

V7.3 keeps the locked MQuAKE forget-only firewall and improves V7.2 by
restricting not only which LM-head rows may move, but also which hidden-space
directions each editable row may use.

For the N visible direct rewrite token positions, let H be the Base final-hidden
matrix (row-normalized) and G = H H^T.  The dual matrix D = pinv(G) H satisfies
D H^T ~= I.  For an editable target row y, A_y is the set of Base-active
positions whose sensitive target token is y.  The update delta_w_y is restricted
to span(D[A_y]).  Therefore it can change logits at the contexts that require
suppression while being approximately orthogonal to every other visible
context.  This is a forget-visible, context-selective parameterization; retain,
AtomicGen, multihop, target_new and PPL data are never used for optimization or
selection.

The optimizer constrains all visible token decisions with a squared hinge,
plus same-prompt non-target KL and L2.  A candidate must pass an exact
materialized BF16 all-visible audit.  The passing update is then bisected toward
Base with repeated exact BF16 audits to find the smallest passing scale along
the Base-to-candidate ray.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_forget_only_active_repair as locked
import mquake_sure_active_rows_v72 as v72
import mquake_sure_sparse_lm_gagd_v7 as v7
import mquake_v7_locked_case_compat as compat
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v7.3-context-protected-dual-basis-bf16-repair"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Protected pretrained Base model")
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--forget-hinge-weight", type=float, default=10.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=2.5)
    p.add_argument("--gd-weight", type=float, default=10.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-3)
    p.add_argument("--coeff-l2-lambda", type=float, default=1e-4)
    p.add_argument("--target-logit-margin", type=float, default=0.01)
    p.add_argument("--bf16-buffer-margin", type=float, default=0.05)
    p.add_argument("--dual-pinv-rtol", type=float, default=1e-6)
    p.add_argument("--basis-rank-tol", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bf16-bisection-steps", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def orthonormal_row_basis(rows: torch.Tensor, tol: float) -> torch.Tensor:
    """Return a [rank, hidden] orthonormal basis for the row span."""
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("orthonormal_row_basis requires non-empty rank-2 rows")
    # The active set for one vocabulary row is small, so SVD here is cheap.
    _, s, vh = torch.linalg.svd(rows.float(), full_matrices=False)
    if s.numel() == 0:
        raise RuntimeError("empty singular spectrum for V7.3 row basis")
    cutoff = max(float(tol) * float(s.max().detach().cpu()), 1e-12)
    rank = int((s > cutoff).sum().item())
    if rank <= 0:
        raise RuntimeError("V7.3 row basis collapsed to rank zero")
    return vh[:rank].detach()


def build_dual_bases(
    hidden: torch.Tensor,
    target_ids: Sequence[int],
    base_active_positions: Sequence[int],
    selected_ids: Sequence[int],
    *,
    pinv_rtol: float,
    rank_tol: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[int]], Dict[str, Any]]:
    """Build padded per-row dual bases and an allowed-context mask.

    Returns:
      basis: [R, Kmax, D]
      basis_mask: [R, Kmax]
      active_by_token: token_id -> visible case indices
      diagnostics
    """
    h = hidden.float()
    h_norm = h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    hn = h / h_norm
    gram = hn @ hn.transpose(0, 1)

    # One solve for all contexts. Work in float64 for the small N x N inverse,
    # then convert the dual vectors back to float32 for optimization.
    gram64 = gram.double()
    gram_pinv64 = torch.linalg.pinv(gram64, rtol=float(pinv_rtol))
    dual = (gram_pinv64 @ hn.double()).float()
    dual_identity = dual @ hn.transpose(0, 1)
    eye = torch.eye(hn.shape[0], dtype=torch.float32, device=hn.device)
    identity_error = (dual_identity - eye).abs()

    active_by_token: Dict[int, List[int]] = {int(t): [] for t in selected_ids}
    active_set = set(int(i) for i in base_active_positions)
    for idx, token_id in enumerate(target_ids):
        token_id = int(token_id)
        if idx in active_set and token_id in active_by_token:
            active_by_token[token_id].append(idx)

    bases: List[torch.Tensor] = []
    ranks: List[int] = []
    protected_leakage_max: List[float] = []
    protected_leakage_mean: List[float] = []
    selected_lookup = {int(t): j for j, t in enumerate(selected_ids)}
    allowed = torch.zeros(
        (hn.shape[0], len(selected_ids)), dtype=torch.bool, device=hn.device
    )

    for token_id in selected_ids:
        indices = active_by_token[int(token_id)]
        if not indices:
            raise RuntimeError(f"selected row {token_id} has no Base-active contexts")
        idx_t = torch.tensor(indices, dtype=torch.long, device=hn.device)
        basis = orthonormal_row_basis(dual.index_select(0, idx_t), rank_tol)
        bases.append(basis)
        ranks.append(int(basis.shape[0]))
        allowed[idx_t, selected_lookup[int(token_id)]] = True

        response = hn @ basis.transpose(0, 1)
        protected_mask = torch.ones(hn.shape[0], dtype=torch.bool, device=hn.device)
        protected_mask[idx_t] = False
        protected_values = response[protected_mask].abs()
        protected_leakage_max.append(
            float(protected_values.max().detach().cpu()) if protected_values.numel() else 0.0
        )
        protected_leakage_mean.append(
            float(protected_values.mean().detach().cpu()) if protected_values.numel() else 0.0
        )

    max_rank = max(ranks)
    hidden_dim = int(h.shape[1])
    padded = torch.zeros(
        (len(selected_ids), max_rank, hidden_dim),
        dtype=torch.float32,
        device=h.device,
    )
    basis_mask = torch.zeros(
        (len(selected_ids), max_rank), dtype=torch.float32, device=h.device
    )
    for row, basis in enumerate(bases):
        rank = int(basis.shape[0])
        padded[row, :rank, :] = basis
        basis_mask[row, :rank] = 1.0

    diagnostics = {
        "visible_hidden_count": int(h.shape[0]),
        "hidden_dim": hidden_dim,
        "gram_rank": int(torch.linalg.matrix_rank(gram64).item()),
        "dual_pinv_rtol": float(pinv_rtol),
        "dual_identity_max_abs_error": float(identity_error.max().detach().cpu()),
        "dual_identity_mean_abs_error": float(identity_error.mean().detach().cpu()),
        "per_row_basis_ranks": ranks,
        "basis_rank_min": min(ranks),
        "basis_rank_max": max(ranks),
        "basis_rank_mean": float(sum(ranks) / len(ranks)),
        "total_trainable_coefficients": int(sum(ranks)),
        "full_dense_row_parameter_count": int(len(selected_ids) * hidden_dim),
        "parameter_reduction_ratio": float(sum(ranks) / (len(selected_ids) * hidden_dim)),
        "protected_basis_response_max": max(protected_leakage_max),
        "protected_basis_response_mean_of_means": float(
            sum(protected_leakage_mean) / len(protected_leakage_mean)
        ),
    }
    return padded, basis_mask, active_by_token, diagnostics


def effective_delta(
    coeff: torch.Tensor,
    basis: torch.Tensor,
    basis_mask: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("rk,rkd->rd", coeff * basis_mask, basis)


def context_drift_metrics(
    stacked: Mapping[str, torch.Tensor],
    delta: torch.Tensor,
    allowed_context_mask: torch.Tensor,
) -> Dict[str, float]:
    corrections = v7.corrections_from_delta(stacked, delta)
    protected = corrections.masked_select(~allowed_context_mask)
    allowed = corrections.masked_select(allowed_context_mask)
    return {
        "protected_context_abs_logit_drift_max": float(
            protected.abs().max().detach().cpu() if protected.numel() else 0.0
        ),
        "protected_context_abs_logit_drift_mean": float(
            protected.abs().mean().detach().cpu() if protected.numel() else 0.0
        ),
        "allowed_context_abs_logit_drift_mean": float(
            allowed.abs().mean().detach().cpu() if allowed.numel() else 0.0
        ),
        "allowed_context_abs_logit_drift_max": float(
            allowed.abs().max().detach().cpu() if allowed.numel() else 0.0
        ),
    }


def cached_feasible(
    stacked: Mapping[str, torch.Tensor], delta: torch.Tensor, required_margin: float
) -> bool:
    margins = v7.competitor_minus_sensitive_margins(stacked, delta)
    return bool(torch.all(margins >= float(required_margin)).item())


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, steps and batch-size must be positive")
    if a.lr <= 0 or a.forget_hinge_weight <= 0 or a.gd_weight < 0:
        raise ValueError("invalid V7.3 learning rate/forget/GD weights")
    if a.hardest_forget_hinge_weight < 0 or a.delta_l2_lambda < 0:
        raise ValueError("invalid V7.3 regularization weights")
    if a.coeff_l2_lambda < 0 or a.grad_clip < 0:
        raise ValueError("invalid V7.3 coefficient/gradient controls")
    if a.target_logit_margin < 0 or a.bf16_buffer_margin < 0:
        raise ValueError("target/buffer margins must be non-negative")
    if not (0.0 < a.dual_pinv_rtol < 1.0) or a.basis_rank_tol <= 0:
        raise ValueError("invalid V7.3 dual/basis tolerances")
    if a.bf16_bisection_steps <= 0:
        raise ValueError("bf16-bisection-steps must be positive")

    compat.install()
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.repair_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, split_manifest = locked.load_locked_records(
        visible_path, manifest_path, a.forget_num, a.seed
    )

    model_args = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = active.freeze_model_for_output_repair(model)
    output_weight = output_layer.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = int(input_weight.data_ptr())
    input_version = int(input_weight._version)
    device = gagd.first_device(model)
    llama_like = mquake.is_llama_like(model, tok)

    cases = v7.direct_rewrite_cases(records, tok, llama_like=llama_like)
    target_ids = v7.resolve_case_target_ids(tok, cases, llama_like=llama_like, device=device)
    special_ids = gagd.special_token_ids(tok)
    all_sensitive_ids = sorted(set(target_ids) - special_ids)
    if not all_sensitive_ids:
        raise RuntimeError("V7.3 found no sensitive LM-head rows")
    if set(target_ids) - set(all_sensitive_ids):
        raise RuntimeError("V7.3 sensitive targets unexpectedly include special-token ids")

    # Untouched Base geometry with all sensitive rows for exact activity.
    all_caches = v7.build_token_delta_caches(
        model, tok, cases, all_sensitive_ids,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        desc="cache MQuAKE V7.3 Base geometry",
    )
    all_stacked = v7.stack_cache_fields(all_caches, device=device)
    all_zero = torch.zeros(
        (len(all_sensitive_ids), int(output_weight.shape[1])),
        dtype=torch.float32, device=device,
    )
    base_margins = v7.competitor_minus_sensitive_margins(all_stacked, all_zero)
    base_active_positions = [
        idx for idx, margin in enumerate(base_margins.detach().cpu().tolist())
        if float(margin) <= 0.0
    ]
    if not base_active_positions:
        raise RuntimeError("protected Base has no active sensitive token decisions")
    selected_ids = sorted({int(target_ids[i]) for i in base_active_positions})

    # Re-cache using only editable rows. All visible cases remain constraints.
    caches = v7.build_token_delta_caches(
        model, tok, cases, selected_ids,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        desc="cache MQuAKE V7.3 all-visible constraints",
    )
    stacked = v7.stack_cache_fields(caches, device=device)
    zero = torch.zeros(
        (len(selected_ids), int(output_weight.shape[1])),
        dtype=torch.float32, device=device,
    )
    restricted_base_margins = v7.competitor_minus_sensitive_margins(stacked, zero)
    if not torch.allclose(base_margins, restricted_base_margins, atol=1e-5, rtol=0.0):
        diff = float((base_margins - restricted_base_margins).abs().max().detach().cpu())
        raise RuntimeError(f"V7.3 restricted Base margin reconstruction mismatch: {diff}")

    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    basis, basis_mask, active_by_token, basis_diag = build_dual_bases(
        stacked["hidden"], target_ids, base_active_positions, selected_ids,
        pinv_rtol=a.dual_pinv_rtol, rank_tol=a.basis_rank_tol,
    )
    allowed_context_mask = torch.zeros(
        (len(cases), len(selected_ids)), dtype=torch.bool, device=device
    )
    lookup = {int(t): j for j, t in enumerate(selected_ids)}
    for token_id, positions in active_by_token.items():
        col = lookup[int(token_id)]
        allowed_context_mask[
            torch.tensor(positions, dtype=torch.long, device=device), col
        ] = True

    coeff = torch.nn.Parameter(
        torch.zeros((len(selected_ids), int(basis.shape[1])), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW([coeff], lr=a.lr, weight_decay=0.0)
    required_margin = float(a.target_logit_margin + a.bf16_buffer_margin)

    feasible_delta: torch.Tensor | None = None
    feasible_coeff: torch.Tensor | None = None
    feasible_step: int | None = None
    feasible_exact_summary: Dict[str, Any] | None = None
    best_metrics = v7.metrics_from_delta(stacked, zero, target_margin=required_margin)
    best_delta = zero.detach().clone()
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = effective_delta(coeff, basis, basis_mask)
        margins = v7.competitor_minus_sensitive_margins(stacked, delta)
        errors = torch.relu(required_margin - margins)
        gd_kl = v7.same_prompt_non_target_kl(stacked, delta)
        delta_l2 = delta.square().sum()
        coeff_l2 = (coeff * basis_mask).square().sum()
        loss = (
            a.forget_hinge_weight * errors.square().mean()
            + a.hardest_forget_hinge_weight * errors.square().max()
            + a.gd_weight * gd_kl
            + a.delta_l2_lambda * delta_l2
            + a.coeff_l2_lambda * coeff_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V7.3 loss at step {step}")
        loss.backward()
        grad_norm_value = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_([coeff], a.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite V7.3 gradient norm at step {step}")
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()

        with torch.no_grad():
            candidate = effective_delta(coeff, basis, basis_mask).detach().clone()
            candidate_kl = float(v7.same_prompt_non_target_kl(stacked, candidate).detach().cpu())
            metrics = v7.metrics_from_delta(
                stacked, candidate, target_margin=required_margin, kl_value=candidate_kl
            )
            drift = context_drift_metrics(stacked, candidate, allowed_context_mask)
            if v72.priority(metrics) < v72.priority(best_metrics):
                best_metrics = dict(metrics)
                best_delta = candidate.detach().clone()

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            logs.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mean_squared_hinge": float(errors.square().mean().detach().cpu()),
                "hardest_squared_hinge": float(errors.square().max().detach().cpu()),
                "gd_same_prompt_non_target_kl": float(gd_kl.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
                "coefficient_l2": float(coeff_l2.detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
                **metrics,
                **drift,
            })
            print(
                f"v73-step={step} active={metrics['official_active_sensitive_token_count']} "
                f"required_unmet={metrics['buffered_margin_unmet_token_count']} "
                f"min_margin={metrics['minimum_competitor_minus_sensitive_margin']:.6g} "
                f"KL={candidate_kl:.6g} norm={metrics['selected_lm_head_delta_norm']:.6g} "
                f"protected_drift_max={drift['protected_context_abs_logit_drift_max']:.3g}"
            )

        if cached_feasible(stacked, candidate, required_margin):
            exact_reports, exact_summary = v72.exact_materialized_audit(
                model, tok, cases,
                output_weight=output_weight,
                selected_ids=selected_ids,
                base_selected_rows=base_selected_rows,
                delta=candidate,
                device=device, llama_like=llama_like,
                batch_size=a.batch_size,
                target_margin=a.target_logit_margin,
            )
            if v72.exact_pass(exact_summary):
                feasible_delta = candidate.detach().clone()
                feasible_coeff = (coeff * basis_mask).detach().clone()
                feasible_step = step
                feasible_exact_summary = dict(exact_summary)
                break
            v7.set_selected_rows(output_weight, selected_ids, base_selected_rows, zero)

    del optimizer
    root = gagd.resolve_output_path(a.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    ckpt = root / "checkpoint"
    v7.write_jsonl(root / "train_log.jsonl", logs)

    if feasible_delta is None or feasible_step is None or feasible_exact_summary is None:
        v7.set_selected_rows(output_weight, selected_ids, base_selected_rows, zero)
        write_json(root / "failure.json", {
            "status": "FAILED_NO_BF16_FEASIBLE_CONTEXT_PROTECTED_REPAIR",
            "method": METHOD,
            "seed": int(a.seed),
            "base_active_token_count": len(base_active_positions),
            "editable_row_count": len(selected_ids),
            "basis_diagnostics": basis_diag,
            "best_cached_metrics": best_metrics,
            "best_delta_norm": float(best_delta.norm().detach().cpu()),
        })
        raise RuntimeError("V7.3 did not find a BF16-feasible context-protected repair")

    # Exact BF16 shrink-back toward Base. low is infeasible; high is feasible.
    low, high = 0.0, 1.0
    bisection_log: List[Dict[str, Any]] = []
    _, zero_summary = v72.exact_materialized_audit(
        model, tok, cases,
        output_weight=output_weight,
        selected_ids=selected_ids,
        base_selected_rows=base_selected_rows,
        delta=zero,
        device=device, llama_like=llama_like,
        batch_size=a.batch_size, target_margin=a.target_logit_margin,
    )
    if v72.exact_pass(zero_summary):
        low = high = 0.0
    else:
        _, high_summary = v72.exact_materialized_audit(
            model, tok, cases,
            output_weight=output_weight,
            selected_ids=selected_ids,
            base_selected_rows=base_selected_rows,
            delta=feasible_delta,
            device=device, llama_like=llama_like,
            batch_size=a.batch_size, target_margin=a.target_logit_margin,
        )
        if not v72.exact_pass(high_summary):
            raise RuntimeError("V7.3 feasible candidate stopped passing BF16 audit")
        for iteration in range(1, a.bf16_bisection_steps + 1):
            mid = 0.5 * (low + high)
            mid_delta = feasible_delta * float(mid)
            _, mid_summary = v72.exact_materialized_audit(
                model, tok, cases,
                output_weight=output_weight,
                selected_ids=selected_ids,
                base_selected_rows=base_selected_rows,
                delta=mid_delta,
                device=device, llama_like=llama_like,
                batch_size=a.batch_size, target_margin=a.target_logit_margin,
            )
            passed = v72.exact_pass(mid_summary)
            bisection_log.append({
                "iteration": iteration,
                "alpha": mid,
                "passed": passed,
                "official_active_sensitive_token_count": int(mid_summary["official_active_sensitive_token_count"]),
                "target_margin_unmet_token_count": int(mid_summary["buffered_margin_unmet_token_count"]),
                "minimum_competitor_minus_sensitive_margin": float(mid_summary["minimum_competitor_minus_sensitive_margin"]),
                "scaled_delta_norm": float(mid_delta.norm().detach().cpu()),
            })
            if passed:
                high = mid
            else:
                low = mid

    final_alpha = float(high)
    final_delta = feasible_delta * final_alpha
    final_reports, final_summary = v72.exact_materialized_audit(
        model, tok, cases,
        output_weight=output_weight,
        selected_ids=selected_ids,
        base_selected_rows=base_selected_rows,
        delta=final_delta,
        device=device, llama_like=llama_like,
        batch_size=a.batch_size, target_margin=a.target_logit_margin,
    )
    if not v72.exact_pass(final_summary):
        raise RuntimeError("V7.3 final BF16 shrink-back checkpoint failed exact audit")
    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V7.3 modified input embeddings")

    final_drift = context_drift_metrics(stacked, final_delta, allowed_context_mask)
    v7.write_jsonl(root / "all_visible_tokens_after_bf16.jsonl", final_reports)
    v7.write_jsonl(root / "bf16_bisection_log.jsonl", bisection_log)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    active_map_json = {
        str(token_id): {
            "token": v7.decoded_token(tok, token_id),
            "active_positions": positions,
            "active_count": len(positions),
        }
        for token_id, positions in active_by_token.items()
    }
    write_json(root / "context_protected_basis.json", {
        "definition": "per-row span of dual hidden vectors for that row's untouched-Base active target contexts; approximately orthogonal to all other visible contexts",
        "base_active_token_case_count": len(base_active_positions),
        "editable_lm_head_row_count": len(selected_ids),
        "editable_token_ids": selected_ids,
        "active_contexts_by_token": active_map_json,
        "basis_diagnostics": basis_diag,
        "non_selected_lm_rows_exact_base_by_construction": True,
        "input_embeddings_exact_base_by_construction": True,
        "transformer_frozen": True,
    })

    summary = {
        "status": "PASS_V73_CONTEXT_PROTECTED_MINIMAL_BF16",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "all_sensitive_target_row_count": len(all_sensitive_ids),
        "base_active_sensitive_token_case_count": len(base_active_positions),
        "selected_base_active_target_row_count": len(selected_ids),
        "basis_diagnostics": basis_diag,
        "optimizer_feasible_step": feasible_step,
        "optimizer_feasible_delta_norm": float(feasible_delta.norm().detach().cpu()),
        "optimizer_feasible_coefficient_norm": float(feasible_coeff.norm().detach().cpu()) if feasible_coeff is not None else None,
        "bf16_minimal_ray_alpha": final_alpha,
        "selected_lm_head_delta_norm": float(final_delta.norm().detach().cpu()),
        "cached_required_margin": required_margin,
        "exact_target_margin": float(a.target_logit_margin),
        "first_feasible_exact_metrics": feasible_exact_summary,
        "materialized_bf16_metrics": final_summary,
        "final_context_drift": final_drift,
        "training_data_access": {
            "forget_instances": int(a.forget_num),
            "forget_atomic_facts": len(records),
            "prompt_types": ["requested_rewrite"],
            "benchmark_retain_instances": 0,
            "atomic_questions": 0,
            "multihop_questions": 0,
            "benchmark_counterfactual_targets": 0,
            "PPL": False,
        },
        "checkpoint_selection_uses_retain_or_heldout": False,
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "repair_summary.json", summary)
    write_json(root / "config_used.json", {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        **vars(a),
        "repair_visible_path_resolved": str(visible_path),
        "split_manifest_resolved": str(manifest_path),
        "split_sampling": split_manifest.get("sampling"),
        "parameter_scope": "Base-active target_true LM-head rows only, each restricted to its context-protected dual basis; transformer/input embeddings/all other LM rows exact Base",
        "base_activity_definition": "competitor_minus_sensitive_margin <= 0 on untouched Base",
        "direction_definition": "D=pinv(H_normalized H_normalized^T) H_normalized; row y update lies in span(D[A_y])",
        "forget_definition": "squared hinge relu(target_margin + bf16_buffer - competitor_minus_sensitive_margin)^2 over all visible token cases",
        "gd_definition": "same-prompt KL(Base_non-target || Current_non-target), sensitive target removed and renormalized",
        "selection_definition": "first BF16-exact feasible context-protected candidate then exact BF16 all-visible bisection toward Base; no retain/PPL/AtomicGen/multihop selection",
        "checkpoint": str(ckpt.resolve()),
    })

    print("===== SURE-MQuAKE V7.3 CONTEXT-PROTECTED MINIMAL BF16 REPAIR =====")
    print(
        f"instances={a.forget_num} atomic_facts={len(records)} token_cases={len(cases)} "
        f"all_sensitive_rows={len(all_sensitive_ids)} base_active_tokens={len(base_active_positions)} "
        f"editable_rows={len(selected_ids)} trainable_coeffs={basis_diag['total_trainable_coefficients']}"
    )
    print(
        f"basis_rank_mean={basis_diag['basis_rank_mean']:.3f} "
        f"param_ratio={basis_diag['parameter_reduction_ratio']:.6g} "
        f"dual_identity_maxerr={basis_diag['dual_identity_max_abs_error']:.3g} "
        f"protected_basis_response_max={basis_diag['protected_basis_response_max']:.3g}"
    )
    print(
        f"optimizer_feasible_step={feasible_step} candidate_norm={float(feasible_delta.norm().detach().cpu()):.6g} "
        f"final_alpha={final_alpha:.8f} final_norm={float(final_delta.norm().detach().cpu()):.6g}"
    )
    print(
        f"protected_context_drift_max={final_drift['protected_context_abs_logit_drift_max']:.6g} "
        f"protected_context_drift_mean={final_drift['protected_context_abs_logit_drift_mean']:.6g}"
    )
    print(
        f"BF16 official_active={final_summary['official_active_sensitive_token_count']} "
        f"margin_unmet={final_summary['buffered_margin_unmet_token_count']} "
        f"min_margin={final_summary['minimum_competitor_minus_sensitive_margin']:.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
