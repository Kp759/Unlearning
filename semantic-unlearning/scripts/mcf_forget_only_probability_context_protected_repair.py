#!/usr/bin/env python3
"""MCF Stage-2 repair for absolute sensitive suppression with context protection.

This experiment reuses a locked target-true-sensitive Setting-5e Stage-1
checkpoint. In that locked training view:

    training target_new  = ORIGINAL MCF target_true  (sensitive)
    training target_true = ORIGINAL MCF target_new   (reference)

Only sensitive output rows (training target_new tokens) may change.

Unlike the older pairwise-only repair, this objective directly requires the
sensitive answer NLL to exceed an absolute floor while also enforcing a
sensitive-vs-reference margin. The hidden-direction basis is built from ALL
50 direct training forget contexts, not only the remaining active cases, so a
requested rank > 1 can be realized when the 50 training contexts span it.

A disjoint calibration file supplies ordinary factual hidden states. Official
MCF paraphrases, neighborhoods, and generation prompts must be absent from both
repair-visible and calibration files. Calibration hidden states are used twice:
(1) to construct retain-aware protected forget directions, and
(2) to penalize selected-row logit drift on calibration contexts.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_active_case_repair as repair
import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like

METHOD = "mcf_probability_context_protected_repair"
PROTOCOL = "target_true_sensitive_absolute_suppression_disjoint_context_protected_v1"
LOCKED_FIELDS = ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Locked Setting-5e Stage-1 checkpoint")
    p.add_argument("--experiment-config-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mcf-cache-path", required=True, help="Locked swapped forget-visible MCF")
    p.add_argument("--calibration-json", required=True, help="Disjoint original-MCF direct-only calibration records")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--sample-mode", choices=("official", "first"), default="official")

    p.add_argument("--sensitive-nll-floor", type=float, default=8.0)
    p.add_argument("--pairwise-margin", type=float, default=1.0)
    p.add_argument("--repair-rank", type=int, default=8)
    p.add_argument("--repair-steps", type=int, default=400)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--absolute-weight", type=float, default=2.0)
    p.add_argument("--pairwise-weight", type=float, default=1.0)
    p.add_argument("--reference-anchor-weight", type=float, default=0.25)
    p.add_argument("--context-weight", type=float, default=1.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-4)
    p.add_argument("--max-delta-norm", type=float, default=None)

    p.add_argument(
        "--context-protection",
        choices=("ridge", "project", "none"),
        default="ridge",
        help="How retain calibration hidden states shape the shared forget basis.",
    )
    p.add_argument("--retain-ridge-lambda", type=float, default=0.10)
    p.add_argument("--retain-rank-cap", type=int, default=128)
    p.add_argument("--stop-when-all-satisfied", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    p.add_argument("--margin-batch-size", type=int, default=4)
    p.add_argument("--save-model", action="store_true")
    return p


def validate_args(a: argparse.Namespace) -> None:
    positive = {
        "forget_num": a.forget_num,
        "sensitive_nll_floor": a.sensitive_nll_floor,
        "repair_rank": a.repair_rank,
        "repair_steps": a.repair_steps,
        "repair_lr": a.repair_lr,
        "absolute_weight": a.absolute_weight,
        "pairwise_weight": a.pairwise_weight,
        "margin_batch_size": a.margin_batch_size,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"--{name.replace('_','-')} must be positive and finite")
    if not math.isfinite(a.pairwise_margin) or a.pairwise_margin < 0:
        raise ValueError("--pairwise-margin must be finite and non-negative")
    for name in ("reference_anchor_weight", "context_weight", "delta_l2_lambda"):
        value = float(getattr(a, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_','-')} must be finite and non-negative")
    if a.context_protection == "ridge" and (
        not math.isfinite(a.retain_ridge_lambda) or a.retain_ridge_lambda <= 0
    ):
        raise ValueError("--retain-ridge-lambda must be positive for ridge protection")
    if a.retain_rank_cap < 0:
        raise ValueError("--retain-rank-cap must be non-negative")
    if a.max_delta_norm is not None and (
        not math.isfinite(a.max_delta_norm) or a.max_delta_norm <= 0
    ):
        raise ValueError("--max-delta-norm must be positive and finite")


def _normalize_rr(record: Mapping[str, Any], position: int) -> Dict[str, Any]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        if len(rr) != 1:
            raise ValueError(f"record {position}: expected one requested_rewrite")
        rr = rr[0]
    if not isinstance(rr, Mapping):
        raise ValueError(f"record {position}: requested_rewrite must be a mapping")
    return dict(rr)


def _assert_direct_only(records: Sequence[Mapping[str, Any]], label: str) -> None:
    for i, record in enumerate(records):
        rr = _normalize_rr(record, i)
        for field in LOCKED_FIELDS:
            top = record.get(field, [])
            nested = rr.get(field, [])
            if top:
                raise RuntimeError(f"{label} record {i} exposes evaluation-only {field}")
            if nested:
                raise RuntimeError(f"{label} requested_rewrite {i} exposes evaluation-only {field}")


def _direct_instances_from_original(records: Sequence[Mapping[str, Any]]) -> List[repair.MCFPromptInstance]:
    out: List[repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        rr = _normalize_rr(record, position)
        subject = str(rr.get("subject", ""))
        prompt = str(rr.get("prompt", "")).format(subject)
        tn = rr.get("target_new")
        tt = rr.get("target_true")
        if not isinstance(tn, Mapping) or not tn.get("str"):
            raise ValueError(f"calibration record {position} lacks target_new.str")
        if not isinstance(tt, Mapping) or not tt.get("str"):
            raise ValueError(f"calibration record {position} lacks target_true.str")
        out.append(
            repair.MCFPromptInstance(
                record_index=int(record.get("case_id", position)),
                sampled_position=position,
                prompt_type="calibration_direct",
                prompt_index=0,
                prompt=prompt,
                target_new=str(tn["str"]),
                target_true=str(tt["str"]),
            )
        )
    return out


def _selected_sensitive_rows(tok: Any, instances: Sequence[repair.MCFPromptInstance]) -> List[int]:
    selected: set[int] = set()
    for instance in instances:
        selected.update(gagd.token_ids_for_text(tok, gagd.normalize_answer(instance.target_new)))
    selected -= gagd.special_token_ids(tok)
    return sorted(selected)


def _svd_basis(rows: torch.Tensor, rank_cap: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = rows.float()
    if rows.ndim != 2 or rows.shape[0] == 0:
        raise ValueError("rows must be non-empty [n, hidden]")
    _, s, vh = torch.linalg.svd(rows, full_matrices=False)
    tol = max(rows.shape) * torch.finfo(rows.dtype).eps * s.max().clamp_min(1.0)
    rank = int((s > tol).sum().item())
    if rank_cap is not None and rank_cap > 0:
        rank = min(rank, int(rank_cap))
    return vh[:rank].contiguous(), s[:rank].contiguous()


def ridge_precondition_forget_rows(
    forget_rows: torch.Tensor,
    retain_rows: torch.Tensor,
    *,
    ridge_lambda: float,
    retain_rank_cap: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Apply (H_R^T H_R / n + lambda I)^-1 to forget rows using low-rank SVD."""
    if retain_rows.ndim != 2 or retain_rows.shape[0] == 0:
        raise ValueError("ridge protection requires non-empty retain rows")
    retain = retain_rows.float()
    forget = forget_rows.float()
    _, s, vh = torch.linalg.svd(retain, full_matrices=False)
    numerical_tol = max(retain.shape) * torch.finfo(retain.dtype).eps * s.max().clamp_min(1.0)
    numerical_rank = int((s > numerical_tol).sum().item())
    kept = numerical_rank
    if retain_rank_cap > 0:
        kept = min(kept, retain_rank_cap)
    vh = vh[:kept]
    s = s[:kept]
    n = float(retain.shape[0])

    if kept:
        coeff = forget @ vh.transpose(0, 1)
        parallel = coeff @ vh
        perpendicular = forget - parallel
        inv_parallel = (coeff / (s.square().div(n) + ridge_lambda)) @ vh
    else:
        parallel = torch.zeros_like(forget)
        perpendicular = forget
        inv_parallel = torch.zeros_like(forget)

    protected = inv_parallel + perpendicular / ridge_lambda
    protected = F.normalize(protected, p=2, dim=1)
    report = {
        "mode": "ridge",
        "ridge_lambda": float(ridge_lambda),
        "retain_hidden_rows": int(retain.shape[0]),
        "retain_numerical_rank": numerical_rank,
        "retain_rank_used": int(kept),
        "mean_forget_parallel_fraction_before": float(
            (parallel.norm(dim=1) / forget.norm(dim=1).clamp_min(1e-12)).mean().cpu()
        ),
    }
    return protected, report


def protected_forget_rows(
    forget_rows: torch.Tensor,
    retain_rows: torch.Tensor,
    *,
    mode: str,
    ridge_lambda: float,
    retain_rank_cap: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if mode == "none":
        return F.normalize(forget_rows.float(), p=2, dim=1), {
            "mode": "none",
            "retain_hidden_rows": int(retain_rows.shape[0]),
        }
    if mode == "ridge":
        return ridge_precondition_forget_rows(
            forget_rows,
            retain_rows,
            ridge_lambda=ridge_lambda,
            retain_rank_cap=retain_rank_cap,
        )

    retain_basis, _ = _svd_basis(retain_rows, retain_rank_cap if retain_rank_cap > 0 else None)
    residual = repair.project_rows_away(forget_rows.float(), retain_basis)
    residual_norm = residual.norm(dim=1)
    raw_norm = forget_rows.float().norm(dim=1).clamp_min(1e-12)
    keep = residual_norm > 1e-8
    if not bool(keep.any().item()):
        raise RuntimeError("all forget directions vanished after retain projection")
    residual = F.normalize(residual[keep], p=2, dim=1)
    return residual, {
        "mode": "project",
        "retain_hidden_rows": int(retain_rows.shape[0]),
        "retain_rank_used": int(retain_basis.shape[0]),
        "forget_hidden_rows_before": int(forget_rows.shape[0]),
        "forget_hidden_rows_after": int(residual.shape[0]),
        "mean_residual_fraction": float((residual_norm / raw_norm).mean().cpu()),
    }


def nll_vectors_from_caches(
    caches: Sequence[repair.RewriteDeltaCache], delta_rows: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    sensitive = torch.stack(
        [repair.answer_nll_from_delta_cache(cache.target_new, delta_rows) for cache in caches]
    )
    reference = torch.stack(
        [repair.answer_nll_from_delta_cache(cache.target_true, delta_rows) for cache in caches]
    )
    return sensitive, reference


def constraint_counts(
    sensitive: torch.Tensor,
    reference: torch.Tensor,
    *,
    sensitive_nll_floor: float,
    pairwise_margin: float,
) -> Dict[str, Any]:
    margin = sensitive - reference
    abs_fail = sensitive < sensitive_nll_floor
    pair_fail = margin < pairwise_margin
    combined = abs_fail | pair_fail
    return {
        "absolute_failures": int(abs_fail.sum().item()),
        "pairwise_failures": int(pair_fail.sum().item()),
        "combined_failures": int(combined.sum().item()),
        "minimum_sensitive_nll": float(sensitive.min().detach().cpu()),
        "mean_sensitive_nll": float(sensitive.mean().detach().cpu()),
        "minimum_pairwise_margin": float(margin.min().detach().cpu()),
        "mean_pairwise_margin": float(margin.mean().detach().cpu()),
        "sensitive_probability_proxy_mean": float(torch.exp(-sensitive).mean().detach().cpu()),
    }


def optimize(
    module: repair.SelectedRowDelta,
    caches: Sequence[repair.RewriteDeltaCache],
    retain_hidden: torch.Tensor,
    *,
    sensitive_nll_floor: float,
    pairwise_margin: float,
    repair_steps: int,
    repair_lr: float,
    absolute_weight: float,
    pairwise_weight: float,
    reference_anchor_weight: float,
    context_weight: float,
    delta_l2_lambda: float,
    max_delta_norm: float | None,
    stop_when_all_satisfied: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    optimizer = torch.optim.AdamW(module.parameters(), lr=repair_lr, weight_decay=0.0)
    logs: List[Dict[str, Any]] = []
    zero = module.effective_delta().detach().zero_()
    with torch.no_grad():
        _, base_reference = nll_vectors_from_caches(caches, zero)

    for step in range(1, repair_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        sensitive, reference = nll_vectors_from_caches(caches, delta)
        margin = sensitive - reference

        absolute_loss = torch.relu(sensitive_nll_floor - sensitive).square().mean()
        pairwise_loss = torch.relu(pairwise_margin - margin).square().mean()
        reference_anchor = (reference - base_reference).square().mean()
        if retain_hidden.numel():
            context_corrections = retain_hidden @ delta.transpose(0, 1)
            context_loss = context_corrections.square().mean()
        else:
            context_loss = delta.new_zeros(())
        delta_l2 = delta.square().sum()

        total = (
            absolute_weight * absolute_loss
            + pairwise_weight * pairwise_loss
            + reference_anchor_weight * reference_anchor
            + context_weight * context_loss
            + delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        total.backward()
        optimizer.step()
        before_norm, after_norm, projected = repair.constrain_effective_delta_norm(
            module, max_delta_norm
        )

        with torch.no_grad():
            updated_delta = module.effective_delta()
            updated_sensitive, updated_reference = nll_vectors_from_caches(caches, updated_delta)
            counts = constraint_counts(
                updated_sensitive,
                updated_reference,
                sensitive_nll_floor=sensitive_nll_floor,
                pairwise_margin=pairwise_margin,
            )
            logs.append(
                {
                    "step": step,
                    "total_loss": float(total.detach().cpu()),
                    "absolute_sensitive_loss": float(absolute_loss.detach().cpu()),
                    "pairwise_margin_loss": float(pairwise_loss.detach().cpu()),
                    "reference_anchor_loss": float(reference_anchor.detach().cpu()),
                    "context_logit_drift_loss": float(context_loss.detach().cpu()),
                    "delta_l2": float(delta_l2.detach().cpu()),
                    "effective_delta_norm_before_projection": before_norm,
                    "effective_delta_norm": after_norm,
                    "delta_norm_projected": projected,
                    **counts,
                }
            )
            if stop_when_all_satisfied and counts["combined_failures"] == 0:
                break

    with torch.no_grad():
        final_delta = module.effective_delta()
        final_sensitive, final_reference = nll_vectors_from_caches(caches, final_delta)
        final_counts = constraint_counts(
            final_sensitive,
            final_reference,
            sensitive_nll_floor=sensitive_nll_floor,
            pairwise_margin=pairwise_margin,
        )
        context_rms = (
            float((retain_hidden @ final_delta.transpose(0, 1)).square().mean().sqrt().cpu())
            if retain_hidden.numel()
            else 0.0
        )
    return logs, {
        "steps_completed": len(logs),
        "all_constraints_satisfied": final_counts["combined_failures"] == 0,
        "final": final_counts,
        "context_selected_logit_shift_rms": context_rms,
        "max_delta_norm": max_delta_norm,
    }


def main() -> None:
    a = build_parser().parse_args()
    validate_args(a)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    output_dir = gagd.resolve_output_path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_args = argparse.Namespace(seed=a.seed, forget_num=a.forget_num, retain_num=0, sample_mode=a.sample_mode)
    config_path, source_config, preserved_alphas = repair.recover_experiment_config(
        a.model_path, a.experiment_config_path
    )
    repair.validate_source_experiment_config(source_config, source_args)

    sampling_args = argparse.Namespace(
        mcf_cache_path=a.mcf_cache_path,
        mcf_url=gagd.MCF_URL,
        forget_num=a.forget_num,
        retain_num=0,
        seed=a.seed,
        sample_mode=a.sample_mode,
    )
    forget_records, unexpected_retain = repair.load_sampled_mcf_records(sampling_args)
    if unexpected_retain:
        raise RuntimeError("locked forget-only repair unexpectedly sampled retain records")
    forget_examples = [record.example for record in forget_records]
    forget_instances = repair.expand_prompt_instances(forget_records)
    if len(forget_instances) != len(forget_records) or any(
        instance.prompt_type != "rewrite" for instance in forget_instances
    ):
        raise RuntimeError("repair-visible MCF exposes non-direct evaluation prompts")

    calibration_path = Path(a.calibration_json).resolve()
    calibration_raw = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration_raw, list) or not all(isinstance(x, dict) for x in calibration_raw):
        raise ValueError("calibration JSON must be a list of MCF records")
    _assert_direct_only(calibration_raw, "calibration")
    calibration_instances = _direct_instances_from_original(calibration_raw)

    model_args = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    print(f"Loading Stage-1 checkpoint: {a.model_path}")
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
    output_embeddings = repair.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    groups = gagd.collect_post_training_token_groups(tok, forget_examples, [])

    before_reports = repair.evaluate_prompt_instance_margin_reports(
        model, tok, forget_instances, groups, a.pairwise_margin, device, a.margin_batch_size, llama_like
    )
    absolute_active_positions = [
        i
        for i, report in enumerate(before_reports)
        if float(report["target_new_nll"]) < a.sensitive_nll_floor
        or float(report["margin"]) < a.pairwise_margin
    ]
    active_instances = [forget_instances[i] for i in absolute_active_positions]
    selected_ids = _selected_sensitive_rows(tok, active_instances)

    print(
        f"Absolute/pairwise active direct cases: {len(active_instances)}/{len(forget_instances)}; "
        f"sensitive-only rows={len(selected_ids)}"
    )
    if not selected_ids:
        raise RuntimeError("no sensitive rows selected; Stage-2 would be a no-op")

    input_storage_pointer = model.get_input_embeddings().weight.detach().data_ptr()
    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    selected_before = output_weight.index_select(0, selected_tensor).detach().clone()

    print("Caching all 50 direct forget contexts for exact NLL objectives")
    forget_caches = repair.build_prompt_instance_delta_caches(
        model, tok, forget_instances, selected_ids, device, a.margin_batch_size, llama_like
    )
    print(f"Caching {len(calibration_instances)} disjoint calibration contexts")
    calibration_caches = repair.build_prompt_instance_delta_caches(
        model, tok, calibration_instances, selected_ids, device, a.margin_batch_size, llama_like
    )

    forget_hidden = torch.cat([cache.target_new.hidden for cache in forget_caches], dim=0).float()
    retain_hidden = torch.cat([cache.target_true.hidden for cache in calibration_caches], dim=0).float()

    protected_hidden, protection_report = protected_forget_rows(
        forget_hidden,
        retain_hidden,
        mode=a.context_protection,
        ridge_lambda=a.retain_ridge_lambda,
        retain_rank_cap=a.retain_rank_cap,
    )
    direction_basis = repair.orthonormal_row_basis(protected_hidden, max_rank=a.repair_rank)
    actual_rank = int(direction_basis.shape[0])
    if actual_rank <= 0:
        raise RuntimeError("protected all-forget basis has zero rank")
    print(
        f"Shared protected forget basis: requested rank={a.repair_rank}, actual rank={actual_rank}, "
        f"forget hidden rows={forget_hidden.shape[0]}, retain hidden rows={retain_hidden.shape[0]}"
    )

    delta_module = repair.SelectedRowDelta(
        len(selected_ids),
        output_weight.shape[1],
        direction_basis=direction_basis,
        retained_basis=None,
        device=output_weight.device,
    )

    logs, optimization = optimize(
        delta_module,
        forget_caches,
        retain_hidden,
        sensitive_nll_floor=a.sensitive_nll_floor,
        pairwise_margin=a.pairwise_margin,
        repair_steps=a.repair_steps,
        repair_lr=a.repair_lr,
        absolute_weight=a.absolute_weight,
        pairwise_weight=a.pairwise_weight,
        reference_anchor_weight=a.reference_anchor_weight,
        context_weight=a.context_weight,
        delta_l2_lambda=a.delta_l2_lambda,
        max_delta_norm=a.max_delta_norm,
        stop_when_all_satisfied=a.stop_when_all_satisfied,
    )
    repair.write_jsonl(output_dir / "repair_log.jsonl", logs)

    with torch.no_grad():
        final_delta = delta_module.effective_delta().detach()
        repair.materialize_selected_delta(output_weight, selected_ids, final_delta)

    if model.get_input_embeddings().weight.detach().data_ptr() != input_storage_pointer:
        raise RuntimeError("input embedding storage changed during Stage-2 repair")
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("input embeddings unexpectedly became trainable")

    after_reports = repair.evaluate_prompt_instance_margin_reports(
        model, tok, forget_instances, groups, a.pairwise_margin, device, a.margin_batch_size, llama_like
    )
    selected_after = output_weight.index_select(0, selected_tensor).detach().clone()
    selected_delta = selected_after.float() - selected_before.float()

    after_abs_fail = sum(
        1 for report in after_reports if float(report["target_new_nll"]) < a.sensitive_nll_floor
    )
    after_pair_fail = sum(
        1 for report in after_reports if float(report["margin"]) < a.pairwise_margin
    )

    config_used = {
        **vars(a),
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_experiment_config_path": str(config_path),
        "source_experiment_config": source_config,
        "preserved_5e_overlap_alphas": preserved_alphas,
        "original_sensitive_field": "target_true",
        "original_reference_field": "target_new",
        "training_sensitive_field": "target_new",
        "editable_rows": "active sensitive answer token rows only",
        "basis_contexts": "all direct forget sensitive-answer hidden states",
        "calibration_role": "disjoint direct-only original-MCF factual contexts",
        "official_paraphrases_used": False,
        "official_neighborhoods_used": False,
        "official_generation_prompts_used": False,
    }
    gagd.write_json(output_dir / "config_used.json", config_used)
    gagd.write_json(output_dir / "rewrite_margins_before.json", before_reports)
    gagd.write_json(output_dir / "rewrite_margins_after.json", after_reports)
    gagd.write_json(
        output_dir / "geometry_report.json",
        {
            "repair_rank_requested": int(a.repair_rank),
            "repair_rank_actual": actual_rank,
            "selected_sensitive_lm_head_rows": len(selected_ids),
            "selected_sensitive_token_ids": [int(x) for x in selected_ids],
            "selected_sensitive_tokens": {str(x): tok.decode([int(x)]) for x in selected_ids},
            "all_forget_hidden_rows": int(forget_hidden.shape[0]),
            "calibration_hidden_rows": int(retain_hidden.shape[0]),
            "context_protection": protection_report,
        },
    )
    summary = {
        "method": METHOD,
        "protocol_status": PROTOCOL,
        "target_semantics": {
            "sensitive": "ORIGINAL target_true",
            "reference": "ORIGINAL target_new",
            "locked_training_sensitive": "target_new",
            "locked_training_reference": "target_true",
        },
        "forget_records": len(forget_records),
        "calibration_records": len(calibration_instances),
        "active_direct_cases_before": len(active_instances),
        "absolute_failures_after": int(after_abs_fail),
        "pairwise_failures_after": int(after_pair_fail),
        "sensitive_nll_floor": float(a.sensitive_nll_floor),
        "pairwise_margin": float(a.pairwise_margin),
        "selected_lm_head_rows": len(selected_ids),
        "changed_selected_lm_head_rows": int(selected_delta.norm(dim=1).gt(0).sum().item()),
        "selected_lm_head_delta_norm": float(selected_delta.norm().cpu()),
        "repair_rank_requested": int(a.repair_rank),
        "repair_rank_actual": actual_rank,
        "optimization": optimization,
        "context_protection": protection_report,
        "input_embeddings_modified": False,
        "transformer_parameters_trainable": 0,
        "evaluation_probe_leakage": False,
    }
    gagd.write_json(output_dir / "repair_summary.json", summary)

    if a.save_model:
        checkpoint = output_dir / "checkpoint"
        repair.save_repair_checkpoint(model, tok, checkpoint, repair_config=config_used)
        print("Saved repaired checkpoint:", checkpoint)

    print("=" * 88)
    print("MCF ABSOLUTE SENSITIVE SUPPRESSION + CONTEXT PROTECTION")
    print(
        f"rank requested/actual: {a.repair_rank}/{actual_rank}; "
        f"absolute failures after: {after_abs_fail}/{len(forget_instances)}; "
        f"pairwise failures after: {after_pair_fail}/{len(forget_instances)}"
    )
    print("context selected-logit shift RMS:", optimization["context_selected_logit_shift_rms"])
    print("selected delta norm:", float(selected_delta.norm().cpu()))
    print("=" * 88)

    del delta_module, forget_caches, calibration_caches
    gc.collect()


if __name__ == "__main__":
    main()
