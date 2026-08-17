#!/usr/bin/env python3
"""SURE-MQuAKE V7.4: prompt-token-subspace protected sparse LM-head repair.

V7.4 keeps the locked MQuAKE forget-only firewall but broadens preservation
beyond the 284 answer-decision states.  For each locked direct rewrite fact, it
uses the original rewrite prompt (the token_index==0 case) and caches every
non-final Base hidden state in that prompt.  These prompt-token states are
forget-visible language contexts; benchmark retain instances, AtomicGen,
multihop, target_new, and PPL data are never loaded for optimization/selection.

A dominant prompt-language subspace Q is obtained by SVD of those Base hidden
states.  Each editable LM-head row is still selected only from untouched-Base
active sensitive target rows, but its repair directions are built from that
row's active answer-decision hidden states after projecting out Q.  Thus the
update is hard-orthogonal to the dominant prompt-language subspace.  A soft
actual prompt-logit-drift penalty additionally suppresses movement in the
remaining prompt-state residual directions.

The +BF16 buffer applies only to untouched-Base active answer decisions.
Untouched-Base safe decisions are constrained to remain safe.  All 284 answer
decisions must pass an exact materialized BF16 audit, after which the update is
bisected toward Base with repeated exact BF16 audits.
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
import mquake_sure_context_protected_v73 as v73
import mquake_sure_sparse_lm_gagd_v7 as v7
import mquake_v7_locked_case_compat as compat
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v7.4-prompt-token-subspace-protected-bf16-repair"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--forget-hinge-weight", type=float, default=10.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=2.5)
    p.add_argument("--safe-hinge-weight", type=float, default=10.0)
    p.add_argument("--gd-weight", type=float, default=10.0)
    p.add_argument("--prompt-drift-weight", type=float, default=1.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-3)
    p.add_argument("--coeff-l2-lambda", type=float, default=1e-4)
    p.add_argument("--target-logit-margin", type=float, default=0.0)
    p.add_argument("--bf16-buffer-margin", type=float, default=0.05)
    p.add_argument("--safe-margin-floor", type=float, default=1e-4)
    p.add_argument("--prompt-protect-rank", type=int, default=1024)
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


def _chunks(values: Sequence[Any], size: int) -> List[Sequence[Any]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


@torch.no_grad()
def collect_prompt_protection_hidden(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Collect Base hidden states for non-final tokens of original rewrite prompts.

    token_index==0 corresponds to the original direct rewrite prompt before any
    teacher-forced answer continuation.  The final prompt token is excluded
    because its hidden state is exactly the first answer-decision context that
    may need forgetting.  Earlier prompt positions are preservation contexts.
    """
    prompts = sorted({str(case.prompt) for case in cases if int(case.token_index) == 0})
    if not prompts:
        raise RuntimeError("V7.4 found no token_index==0 direct rewrite prompts")

    rows: List[torch.Tensor] = []
    kept_per_prompt: List[int] = []
    for batch in _chunks(prompts, batch_size):
        encoded = tok(list(batch), padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        hidden = output.hidden_states[-1].float()
        attention = encoded["attention_mask"]
        for row in range(len(batch)):
            valid = attention[row].nonzero(as_tuple=False).flatten()
            if valid.numel() <= 1:
                kept_per_prompt.append(0)
                continue
            protect_positions = valid[:-1]
            rows.append(hidden[row].index_select(0, protect_positions).detach())
            kept_per_prompt.append(int(protect_positions.numel()))
        del output, hidden

    if not rows:
        raise RuntimeError("V7.4 prompt protection produced no hidden states")
    stacked = torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)
    diag = {
        "original_rewrite_prompt_count": len(prompts),
        "prompt_protection_hidden_count": int(stacked.shape[0]),
        "prompt_token_hidden_dim": int(stacked.shape[1]),
        "protected_positions_min_per_prompt": min(kept_per_prompt),
        "protected_positions_max_per_prompt": max(kept_per_prompt),
        "protected_positions_mean_per_prompt": float(sum(kept_per_prompt) / len(kept_per_prompt)),
    }
    return stacked, diag


def build_prompt_subspace(
    prompt_hidden: torch.Tensor,
    requested_rank: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if requested_rank <= 0:
        raise ValueError("prompt-protect-rank must be positive")
    h = prompt_hidden.float()
    hn = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Full_matrices=False; the prompt matrix is much smaller than the model.
    _, s, vh = torch.linalg.svd(hn, full_matrices=False)
    numerical_rank = int((s > (float(s.max().detach().cpu()) * 1e-6)).sum().item())
    rank = min(int(requested_rank), int(vh.shape[0]), numerical_rank)
    if rank <= 0:
        raise RuntimeError("V7.4 prompt protection subspace collapsed to rank zero")
    q = vh[:rank].detach()
    total_energy = float(s.square().sum().detach().cpu())
    captured = float(s[:rank].square().sum().detach().cpu())
    diag = {
        "requested_prompt_protect_rank": int(requested_rank),
        "actual_prompt_protect_rank": int(rank),
        "prompt_hidden_numerical_rank": numerical_rank,
        "prompt_subspace_energy_fraction": 0.0 if total_energy <= 0 else captured / total_energy,
        "top_singular_value": float(s[0].detach().cpu()),
        "cutoff_singular_value": float(s[rank - 1].detach().cpu()),
    }
    return q, diag


def build_projected_active_bases(
    hidden: torch.Tensor,
    target_ids: Sequence[int],
    base_active_positions: Sequence[int],
    selected_ids: Sequence[int],
    prompt_basis: torch.Tensor,
    *,
    rank_tol: float,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, List[int]], Dict[str, Any]]:
    active_set = set(int(i) for i in base_active_positions)
    active_by_token: Dict[int, List[int]] = {int(t): [] for t in selected_ids}
    for idx, token_id in enumerate(target_ids):
        token_id = int(token_id)
        if idx in active_set and token_id in active_by_token:
            active_by_token[token_id].append(idx)

    bases: List[torch.Tensor] = []
    ranks: List[int] = []
    residual_ratios: List[float] = []
    q = prompt_basis.float()
    for token_id in selected_ids:
        positions = active_by_token[int(token_id)]
        if not positions:
            raise RuntimeError(f"V7.4 editable row {token_id} has no active contexts")
        idx_t = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        active_hidden = hidden.index_select(0, idx_t).float()
        projected = active_hidden - (active_hidden @ q.transpose(0, 1)) @ q
        raw_norm = active_hidden.norm().clamp_min(1e-12)
        residual_ratios.append(float((projected.norm() / raw_norm).detach().cpu()))
        basis = v73.orthonormal_row_basis(projected, rank_tol)
        bases.append(basis)
        ranks.append(int(basis.shape[0]))

    max_rank = max(ranks)
    hidden_dim = int(hidden.shape[1])
    padded = torch.zeros(
        (len(selected_ids), max_rank, hidden_dim),
        dtype=torch.float32,
        device=hidden.device,
    )
    mask = torch.zeros(
        (len(selected_ids), max_rank), dtype=torch.float32, device=hidden.device
    )
    for row, basis in enumerate(bases):
        r = int(basis.shape[0])
        padded[row, :r, :] = basis
        mask[row, :r] = 1.0

    # Verify hard orthogonality to Q after basis construction.
    leakage = torch.matmul(padded, q.transpose(0, 1)).abs()
    valid_leakage = leakage * mask.unsqueeze(-1)
    trainable = int(sum(ranks))
    full_dense = int(len(selected_ids) * hidden_dim)
    diag = {
        "per_row_basis_ranks": ranks,
        "basis_rank_min": min(ranks),
        "basis_rank_max": max(ranks),
        "basis_rank_mean": float(sum(ranks) / len(ranks)),
        "total_trainable_coefficients": trainable,
        "full_dense_row_parameter_count": full_dense,
        "parameter_reduction_ratio": float(trainable / full_dense),
        "active_hidden_residual_ratio_min": min(residual_ratios),
        "active_hidden_residual_ratio_max": max(residual_ratios),
        "active_hidden_residual_ratio_mean": float(sum(residual_ratios) / len(residual_ratios)),
        "basis_prompt_subspace_overlap_max": float(valid_leakage.max().detach().cpu()),
    }
    return padded, mask, active_by_token, diag


def effective_delta(coeff: torch.Tensor, basis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.einsum("rk,rkd->rd", coeff * mask, basis)


def prompt_drift_metrics(prompt_hidden: torch.Tensor, delta: torch.Tensor) -> Dict[str, float]:
    corrections = prompt_hidden.float() @ delta.transpose(0, 1)
    abs_c = corrections.abs()
    return {
        "prompt_context_abs_logit_drift_max": float(abs_c.max().detach().cpu()),
        "prompt_context_abs_logit_drift_mean": float(abs_c.mean().detach().cpu()),
        "prompt_context_rms_logit_drift": float(corrections.square().mean().sqrt().detach().cpu()),
    }


def cached_feasible(
    margins: torch.Tensor,
    base_active_mask: torch.Tensor,
    *,
    active_required_margin: float,
    safe_required_margin: float,
) -> bool:
    return bool(
        (
            torch.all(margins[base_active_mask] >= float(active_required_margin))
            & torch.all(margins[~base_active_mask] > float(safe_required_margin))
        ).item()
    )


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, steps and batch-size must be positive")
    if a.lr <= 0 or a.forget_hinge_weight <= 0 or a.safe_hinge_weight < 0:
        raise ValueError("invalid V7.4 optimization weights")
    if a.hardest_forget_hinge_weight < 0 or a.gd_weight < 0 or a.prompt_drift_weight < 0:
        raise ValueError("invalid V7.4 hinge/GD/prompt weights")
    if a.delta_l2_lambda < 0 or a.coeff_l2_lambda < 0 or a.grad_clip < 0:
        raise ValueError("invalid V7.4 regularization controls")
    if a.target_logit_margin < 0 or a.bf16_buffer_margin < 0 or a.safe_margin_floor < 0:
        raise ValueError("invalid V7.4 margins")
    if a.prompt_protect_rank <= 0 or a.basis_rank_tol <= 0 or a.bf16_bisection_steps <= 0:
        raise ValueError("invalid V7.4 rank/tolerance/bisection controls")

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
    if not all_sensitive_ids or set(target_ids) - set(all_sensitive_ids):
        raise RuntimeError("V7.4 sensitive target row construction failed")

    # Exact untouched-Base answer-decision geometry.
    all_caches = v7.build_token_delta_caches(
        model, tok, cases, all_sensitive_ids,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        desc="cache MQuAKE V7.4 Base answer geometry",
    )
    all_stacked = v7.stack_cache_fields(all_caches, device=device)
    all_zero = torch.zeros(
        (len(all_sensitive_ids), int(output_weight.shape[1])),
        dtype=torch.float32, device=device,
    )
    base_margins = v7.competitor_minus_sensitive_margins(all_stacked, all_zero)
    base_active_mask = base_margins <= 0.0
    base_active_positions = base_active_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    if not base_active_positions:
        raise RuntimeError("protected Base has no active sensitive token decisions")
    selected_ids = sorted({int(target_ids[i]) for i in base_active_positions})

    # All visible answer cases remain exact constraints with only active rows editable.
    caches = v7.build_token_delta_caches(
        model, tok, cases, selected_ids,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        desc="cache MQuAKE V7.4 all-visible constraints",
    )
    stacked = v7.stack_cache_fields(caches, device=device)
    zero = torch.zeros(
        (len(selected_ids), int(output_weight.shape[1])),
        dtype=torch.float32, device=device,
    )
    restricted = v7.competitor_minus_sensitive_margins(stacked, zero)
    if not torch.allclose(base_margins, restricted, atol=1e-5, rtol=0.0):
        diff = float((base_margins - restricted).abs().max().detach().cpu())
        raise RuntimeError(f"V7.4 restricted Base margin reconstruction mismatch: {diff}")

    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    prompt_hidden, prompt_diag = collect_prompt_protection_hidden(
        model, tok, cases, device=device, batch_size=a.batch_size
    )
    prompt_basis, subspace_diag = build_prompt_subspace(prompt_hidden, a.prompt_protect_rank)
    basis, basis_mask, active_by_token, basis_diag = build_projected_active_bases(
        stacked["hidden"], target_ids, base_active_positions, selected_ids, prompt_basis,
        rank_tol=a.basis_rank_tol,
    )

    coeff = torch.nn.Parameter(
        torch.zeros((len(selected_ids), int(basis.shape[1])), dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW([coeff], lr=a.lr, weight_decay=0.0)
    active_required_margin = float(a.target_logit_margin + a.bf16_buffer_margin)
    safe_required_margin = float(a.safe_margin_floor)

    feasible_delta: torch.Tensor | None = None
    feasible_coeff: torch.Tensor | None = None
    feasible_step: int | None = None
    feasible_exact_summary: Dict[str, Any] | None = None
    best_metrics = v7.metrics_from_delta(stacked, zero, target_margin=active_required_margin)
    best_delta = zero.detach().clone()
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = effective_delta(coeff, basis, basis_mask)
        margins = v7.competitor_minus_sensitive_margins(stacked, delta)
        active_errors = torch.relu(active_required_margin - margins[base_active_mask])
        safe_errors = torch.relu(safe_required_margin - margins[~base_active_mask])
        gd_kl = v7.same_prompt_non_target_kl(stacked, delta)
        prompt_corrections = prompt_hidden.float() @ delta.transpose(0, 1)
        prompt_drift_mse = prompt_corrections.square().mean()
        delta_l2 = delta.square().sum()
        coeff_l2 = (coeff * basis_mask).square().sum()
        loss = (
            a.forget_hinge_weight * active_errors.square().mean()
            + a.hardest_forget_hinge_weight * active_errors.square().max()
            + a.safe_hinge_weight * safe_errors.square().mean()
            + a.gd_weight * gd_kl
            + a.prompt_drift_weight * prompt_drift_mse
            + a.delta_l2_lambda * delta_l2
            + a.coeff_l2_lambda * coeff_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V7.4 loss at step {step}")
        loss.backward()
        grad_norm_value = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_([coeff], a.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite V7.4 gradient norm at step {step}")
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()

        with torch.no_grad():
            candidate = effective_delta(coeff, basis, basis_mask).detach().clone()
            candidate_margins = v7.competitor_minus_sensitive_margins(stacked, candidate)
            candidate_kl = float(v7.same_prompt_non_target_kl(stacked, candidate).detach().cpu())
            metrics = v7.metrics_from_delta(
                stacked, candidate, target_margin=active_required_margin, kl_value=candidate_kl
            )
            pdrift = prompt_drift_metrics(prompt_hidden, candidate)
            if v72.priority(metrics) < v72.priority(best_metrics):
                best_metrics = dict(metrics)
                best_delta = candidate.detach().clone()

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            active_unmet = int((candidate_margins[base_active_mask] < active_required_margin).sum().item())
            safe_failed = int((candidate_margins[~base_active_mask] <= safe_required_margin).sum().item())
            logs.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "active_buffer_unmet": active_unmet,
                "safe_failed": safe_failed,
                "gd_same_prompt_non_target_kl": float(gd_kl.detach().cpu()),
                "prompt_drift_mse": float(prompt_drift_mse.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
                "coefficient_l2": float(coeff_l2.detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
                **metrics,
                **pdrift,
            })
            print(
                f"v74-step={step} active={metrics['official_active_sensitive_token_count']} "
                f"active_buffer_unmet={active_unmet} safe_failed={safe_failed} "
                f"min_margin={metrics['minimum_competitor_minus_sensitive_margin']:.6g} "
                f"KL={candidate_kl:.6g} norm={metrics['selected_lm_head_delta_norm']:.6g} "
                f"prompt_drift_max={pdrift['prompt_context_abs_logit_drift_max']:.3g}"
            )

        if cached_feasible(
            candidate_margins,
            base_active_mask,
            active_required_margin=active_required_margin,
            safe_required_margin=safe_required_margin,
        ):
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
            "status": "FAILED_NO_BF16_FEASIBLE_PROMPT_PROTECTED_REPAIR",
            "method": METHOD,
            "seed": int(a.seed),
            "base_active_token_count": len(base_active_positions),
            "editable_row_count": len(selected_ids),
            "prompt_protection": {**prompt_diag, **subspace_diag},
            "basis_diagnostics": basis_diag,
            "best_cached_metrics": best_metrics,
            "best_delta_norm": float(best_delta.norm().detach().cpu()),
        })
        raise RuntimeError("V7.4 did not find a BF16-feasible prompt-protected repair")

    # Exact BF16 shrink-back toward Base.
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
            raise RuntimeError("V7.4 feasible candidate stopped passing BF16 audit")
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
        raise RuntimeError("V7.4 final BF16 shrink-back checkpoint failed exact audit")
    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V7.4 modified input embeddings")

    final_prompt_drift = prompt_drift_metrics(prompt_hidden, final_delta)
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
    write_json(root / "prompt_protection_basis.json", {
        "definition": "top Base hidden-state subspace over every non-final token of each original locked direct rewrite prompt; active repair directions are projected into its orthogonal complement",
        "prompt_protection": {**prompt_diag, **subspace_diag},
        "active_projected_basis": basis_diag,
        "active_contexts_by_token": active_map_json,
        "editable_token_ids": selected_ids,
        "non_selected_lm_rows_exact_base_by_construction": True,
        "input_embeddings_exact_base_by_construction": True,
        "transformer_frozen": True,
    })

    summary = {
        "status": "PASS_V74_PROMPT_PROTECTED_MINIMAL_BF16",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "all_sensitive_target_row_count": len(all_sensitive_ids),
        "base_active_sensitive_token_case_count": len(base_active_positions),
        "selected_base_active_target_row_count": len(selected_ids),
        "prompt_protection": {**prompt_diag, **subspace_diag},
        "basis_diagnostics": basis_diag,
        "optimizer_feasible_step": feasible_step,
        "optimizer_feasible_delta_norm": float(feasible_delta.norm().detach().cpu()),
        "optimizer_feasible_coefficient_norm": float(feasible_coeff.norm().detach().cpu()) if feasible_coeff is not None else None,
        "bf16_minimal_ray_alpha": final_alpha,
        "selected_lm_head_delta_norm": float(final_delta.norm().detach().cpu()),
        "active_cached_required_margin": active_required_margin,
        "safe_cached_required_margin": safe_required_margin,
        "exact_target_margin": float(a.target_logit_margin),
        "first_feasible_exact_metrics": feasible_exact_summary,
        "materialized_bf16_metrics": final_summary,
        "final_prompt_context_drift": final_prompt_drift,
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
        "parameter_scope": "Base-active target_true LM-head rows only; per-row repair directions projected out of dominant original-prompt token hidden subspace; transformer/input embeddings/all other LM rows exact Base",
        "prompt_protection_definition": "non-final hidden states of token_index==0 direct rewrite prompts only; no benchmark retain or PPL utility data",
        "forget_definition": "squared hinge on Base-active answer decisions with BF16 buffer; Base-safe answer decisions protected above safe floor",
        "selection_definition": "first BF16-exact feasible prompt-protected candidate then exact BF16 all-visible bisection toward Base",
        "checkpoint": str(ckpt.resolve()),
    })

    print("===== SURE-MQuAKE V7.4 PROMPT-TOKEN-PROTECTED MINIMAL BF16 REPAIR =====")
    print(
        f"instances={a.forget_num} atomic_facts={len(records)} token_cases={len(cases)} "
        f"all_sensitive_rows={len(all_sensitive_ids)} base_active_tokens={len(base_active_positions)} "
        f"editable_rows={len(selected_ids)} trainable_coeffs={basis_diag['total_trainable_coefficients']}"
    )
    print(
        f"prompt_hidden={prompt_diag['prompt_protection_hidden_count']} "
        f"protect_rank={subspace_diag['actual_prompt_protect_rank']} "
        f"protect_energy={subspace_diag['prompt_subspace_energy_fraction']:.6f} "
        f"active_residual_ratio_mean={basis_diag['active_hidden_residual_ratio_mean']:.6g} "
        f"basis_Q_overlap_max={basis_diag['basis_prompt_subspace_overlap_max']:.3g}"
    )
    print(
        f"optimizer_feasible_step={feasible_step} candidate_norm={float(feasible_delta.norm().detach().cpu()):.6g} "
        f"final_alpha={final_alpha:.8f} final_norm={float(final_delta.norm().detach().cpu()):.6g}"
    )
    print(
        f"prompt_context_drift_max={final_prompt_drift['prompt_context_abs_logit_drift_max']:.6g} "
        f"prompt_context_drift_mean={final_prompt_drift['prompt_context_abs_logit_drift_mean']:.6g}"
    )
    print(
        f"BF16 official_active={final_summary['official_active_sensitive_token_count']} "
        f"margin_unmet={final_summary['buffered_margin_unmet_token_count']} "
        f"min_margin={final_summary['minimum_competitor_minus_sensitive_margin']:.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
