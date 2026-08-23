#!/usr/bin/env python3
"""MCF SURE Stage 2: row-specific protected repair + exact minimum-scale save.

This is the concrete locality fix for the protected MCF-margin Stage 2.

It keeps the Stage-1 checkpoint unchanged and fixes two structural problems:

1) A single global failure basis must not be shared by every sensitive output
   row.  For each target_true token row s we now build its own residual basis

       B_safe = rowspace(H_P U H_context), rank <= protected_rank
       R_F,s  = H_F,s - Proj_Bsafe(H_F,s)
       B_F,s  = rowspace(R_F,s), rank <= repair_rank
       Delta w_s = c_s B_F,s

   H_P comes from Stage-1 direct MCF-success records. H_context is the same
   pre-answer training-visible context state pool used by protected Stage 1.
   No official paraphrases, neighborhoods, retain records, or PPL text are read.

2) Once a feasible repair direction is found, do not save the full optimizer
   delta.  Find the smallest bf16-materialized scale alpha in [0,1] that still
   satisfies the exact direct MCF gate.  This is an algorithmic line search,
   not a hyperparameter sweep.

Primary direct margin:
    m_i = NLL_i(target_true) - NLL_i(target_new)

Training requires m_i >= --train-mcf-margin (default .10). Final exact bf16
materialization requires m_i >= --final-mcf-margin (default .05), providing
quantization headroom while minimizing the saved update norm.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
import sure_canonical_core as core
import sure_context_projection as context
import sure_stage2_sparse_repair as shared_stage2
import mcf_sure_protected_subspace_stage1 as stage1v2
import mcf_sure_protected_subspace_stage2_mcf_margin as v3

METHOD = "SURE-MCF-row-specific-protected-minimal-stage2"
PROTOCOL = "mcf_target_true_rowspecific_protected_minimal_stage2_v4"
BACKTRACK_SCALES = v3.BACKTRACK_SCALES


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Protected Stage-1 checkpoint")
    p.add_argument("--stage1-config-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--train-mcf-margin", type=float, default=0.10)
    p.add_argument("--final-mcf-margin", type=float, default=0.05)
    p.add_argument("--protected-mcf-margin-floor", type=float, default=0.0)
    p.add_argument("--atomic-margin", type=float, default=0.05)
    p.add_argument("--protected-rank", type=int, default=32)
    p.add_argument("--repair-rank", type=int, default=4)
    p.add_argument("--protected-kl-weight", type=float, default=1.0)
    p.add_argument("--protected-kl-max", type=float, default=0.05)
    p.add_argument("--delta-l2", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--scale-bisect-steps", type=int, default=12)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.forget_num, a.repair_steps, a.protected_rank, a.repair_rank,
           a.check_every, a.cache_batch_size, a.scale_bisect_steps) <= 0:
        p.error("counts/ranks/check/bisect settings must be positive")
    if a.repair_lr <= 0:
        p.error("repair-lr must be positive")
    if min(a.train_mcf_margin, a.final_mcf_margin, a.protected_mcf_margin_floor,
           a.atomic_margin, a.protected_kl_weight, a.protected_kl_max,
           a.delta_l2, a.grad_clip) < 0:
        p.error("margins/KL/L2/clip values must be non-negative")
    if a.final_mcf_margin > a.train_mcf_margin:
        p.error("final-mcf-margin must not exceed train-mcf-margin")
    return a


def build_safe_basis(
    *,
    model,
    tok,
    sensitive_cases,
    hidden: torch.Tensor,
    protected_atomic_positions: Sequence[int],
    device: torch.device,
    cache_batch_size: int,
    protected_rank: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # Reuse the exact Stage-1 definition of pre-answer context states.
    _, h_context = stage1v2.collect_prediction_and_context_hidden(
        model, tok, sensitive_cases, device, int(cache_batch_size)
    )
    blocks: List[torch.Tensor] = []
    if protected_atomic_positions:
        pidx = torch.tensor(
            [int(x) for x in protected_atomic_positions],
            dtype=torch.long,
            device=hidden.device,
        )
        blocks.append(hidden.index_select(0, pidx).float())
    if h_context.numel():
        blocks.append(h_context.float())
    if not blocks:
        basis = hidden.new_empty((0, hidden.shape[1]), dtype=torch.float32)
    else:
        safe_rows = torch.cat(blocks, dim=0)
        basis = core.orthonormal_row_basis(
            safe_rows, max_rank=int(protected_rank)
        )
    return basis.contiguous(), {
        "protected_atomic_rows": int(len(protected_atomic_positions)),
        "pre_answer_context_rows": int(h_context.shape[0]),
        "protected_rank_requested": int(protected_rank),
        "protected_rank_actual": int(basis.shape[0]),
        "definition": "rowspace(Stage1-success target_true states U all pre-answer direct context states)",
    }


def build_row_specific_failure_bases(
    *,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    failure_atomic_positions: Sequence[int],
    safe_basis: torch.Tensor,
    repair_rank: int,
    special_ids: Sequence[int],
) -> Tuple[List[int], List[torch.Tensor], List[Dict[str, Any]]]:
    special = {int(x) for x in special_ids}
    tids = target_ids.detach().cpu().tolist()
    candidate_ids = sorted(
        {int(tids[i]) for i in failure_atomic_positions if int(tids[i]) not in special}
    )
    kept_ids: List[int] = []
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for tid in candidate_ids:
        positions = [
            int(i) for i in failure_atomic_positions if int(tids[i]) == int(tid)
        ]
        idx = torch.tensor(positions, dtype=torch.long, device=hidden.device)
        h_row = hidden.index_select(0, idx).float()
        residual = stage1v2.project_away(h_row, safe_basis)
        residual_norm = float(residual.norm().detach().cpu())
        if not torch.isfinite(residual).all() or residual_norm <= 1e-7:
            reports.append({
                "token_id": int(tid),
                "failure_atomic_count": len(positions),
                "residual_norm": residual_norm,
                "repair_rank_actual": 0,
                "skipped_zero_residual": True,
            })
            continue
        basis = core.orthonormal_row_basis(
            residual, max_rank=int(repair_rank)
        )
        if basis.ndim != 2 or basis.shape[0] <= 0:
            continue
        overlap = (
            float((basis @ safe_basis.transpose(0, 1)).abs().max().detach().cpu())
            if safe_basis.numel() else 0.0
        )
        kept_ids.append(int(tid))
        bases.append(basis.detach().float().contiguous())
        reports.append({
            "token_id": int(tid),
            "failure_atomic_count": len(positions),
            "residual_norm": residual_norm,
            "repair_rank_requested": int(repair_rank),
            "repair_rank_actual": int(basis.shape[0]),
            "max_abs_overlap_with_safe_basis": overlap,
            "skipped_zero_residual": False,
        })
    if not kept_ids:
        raise RuntimeError("all row-specific failure residuals vanished in protected nullspace")
    return kept_ids, bases, reports


@torch.no_grad()
def exact_materialized_margins(
    *,
    model,
    tok,
    output_layer,
    selected_ids: Sequence[int],
    base_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    scale: float,
    instances,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ids = torch.tensor(
        [int(x) for x in selected_ids], dtype=torch.long,
        device=output_layer.weight.device,
    )
    trial = base_rows + delta_rows.to(
        device=base_rows.device, dtype=torch.float32
    ) * float(scale)
    # Quantize exactly as the saved model will quantize these rows.
    quantized = trial.to(dtype=output_layer.weight.dtype)
    output_layer.weight.index_copy_(0, ids, quantized)
    try:
        margins = shared_stage2.mcf_direct_margins(
            model, tok, instances, device, llama_like, int(batch_size),
            sensitive_field="target_true", reference_field="target_new",
        ).detach().float().cpu()
        effective_delta = (
            quantized.float() - base_rows.to(dtype=quantized.dtype).float()
        ).detach()
    finally:
        output_layer.weight.index_copy_(0, ids, base_rows)
    return margins, effective_delta


def exact_feasible(
    *,
    margins: torch.Tensor,
    effective_delta: torch.Tensor,
    final_margin: float,
    protected_records: Sequence[int],
    protected_floor: float,
    protected_atomic_positions: Sequence[int],
    atomic_base_logits: torch.Tensor,
    hidden: torch.Tensor,
    selected_ids: Sequence[int],
    protected_kl_max: float,
) -> Tuple[bool, Dict[str, Any]]:
    all_pass = bool((margins >= float(final_margin)).all().item())
    if protected_records:
        pidx = torch.tensor([int(x) for x in protected_records], dtype=torch.long)
        pvals = margins.index_select(0, pidx)
        p_reg = int((pvals < float(protected_floor)).sum().item())
        p_min = float(pvals.min().item())
    else:
        p_reg = 0
        p_min = None
    kl = v3.protected_kl(
        base_logits=atomic_base_logits,
        hidden=hidden,
        protected_atomic_positions=protected_atomic_positions,
        selected_ids=selected_ids,
        delta_rows=effective_delta.to(device=hidden.device),
    )
    kl_value = max(0.0, float(kl.detach().cpu()))
    ok = all_pass and p_reg == 0 and kl_value <= float(protected_kl_max)
    return ok, {
        "all_direct_records_above_final_margin": all_pass,
        "minimum_direct_margin": float(margins.min().item()),
        "protected_regressions": p_reg,
        "protected_min_margin": p_min,
        "protected_kl": kl_value,
    }


@torch.no_grad()
def minimum_exact_scale(
    *,
    model,
    tok,
    output_layer,
    selected_ids: Sequence[int],
    base_rows: torch.Tensor,
    delta_rows: torch.Tensor,
    instances,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    final_margin: float,
    protected_records: Sequence[int],
    protected_floor: float,
    protected_atomic_positions: Sequence[int],
    atomic_base_logits: torch.Tensor,
    hidden: torch.Tensor,
    protected_kl_max: float,
    bisect_steps: int,
) -> Tuple[float, torch.Tensor, Dict[str, Any], List[Dict[str, Any]]]:
    evaluations: List[Dict[str, Any]] = []

    def evaluate(alpha: float):
        margins, qdelta = exact_materialized_margins(
            model=model, tok=tok, output_layer=output_layer,
            selected_ids=selected_ids, base_rows=base_rows,
            delta_rows=delta_rows, scale=float(alpha), instances=instances,
            device=device, llama_like=llama_like, batch_size=batch_size,
        )
        ok, metrics = exact_feasible(
            margins=margins, effective_delta=qdelta,
            final_margin=final_margin, protected_records=protected_records,
            protected_floor=protected_floor,
            protected_atomic_positions=protected_atomic_positions,
            atomic_base_logits=atomic_base_logits, hidden=hidden,
            selected_ids=selected_ids, protected_kl_max=protected_kl_max,
        )
        evaluations.append({"scale": float(alpha), "feasible": bool(ok), **metrics})
        return ok, margins, qdelta, metrics

    ok1, margins1, qdelta1, metrics1 = evaluate(1.0)
    if not ok1:
        raise RuntimeError(
            "full learned delta is not exact-bf16 feasible; optimizer must continue"
        )
    ok0, _, _, _ = evaluate(0.0)
    if ok0:
        return 0.0, torch.zeros_like(delta_rows), metrics1, evaluations

    lo, hi = 0.0, 1.0
    best_delta = qdelta1
    best_metrics = metrics1
    for _ in range(int(bisect_steps)):
        mid = (lo + hi) / 2.0
        ok, _, qdelta, metrics = evaluate(mid)
        if ok:
            hi = mid
            best_delta = qdelta
            best_metrics = metrics
        else:
            lo = mid
    # Re-evaluate final hi so reported metrics correspond exactly to chosen scale.
    _, _, best_delta, best_metrics = evaluate(hi)
    return float(hi), best_delta, best_metrics, evaluations


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    stage1_config_path = Path(a.stage1_config_path).resolve()
    records, manifest, stage1_config = v3.load_and_validate_protocol(
        visible_path, manifest_path, stage1_config_path,
        int(a.seed), int(a.forget_num),
    )

    ns = argparse.Namespace(
        model_path=a.model_path, dtype=a.dtype, device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model lacks input embeddings")
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    sensitive_cases = context.expand_answer_field_cases(
        records, tok, field="target_true", llama_like=llama_like
    )
    hidden, atomic_base_logits, target_ids = v3.cache_atomic_state(
        model, tok, sensitive_cases, llama_like=llama_like,
        device=device, batch_size=int(a.cache_batch_size),
    )
    instances = shared_stage2.mcf_instances(records)
    baseline_margins = shared_stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(a.cache_batch_size),
        sensitive_field="target_true", reference_field="target_new",
    ).detach().float()
    protected_records, failure_records = v3.record_partition(
        baseline_margins, float(a.train_mcf_margin)
    )
    protected_atomic_positions = v3.atomic_positions_for_records(
        sensitive_cases, protected_records
    )
    failure_atomic_positions = v3.atomic_positions_for_records(
        sensitive_cases, failure_records
    )

    safe_basis, safe_report = build_safe_basis(
        model=model, tok=tok, sensitive_cases=sensitive_cases, hidden=hidden,
        protected_atomic_positions=protected_atomic_positions, device=device,
        cache_batch_size=int(a.cache_batch_size),
        protected_rank=int(a.protected_rank),
    )
    selected_ids, row_bases, row_reports = build_row_specific_failure_bases(
        hidden=hidden, target_ids=target_ids,
        failure_atomic_positions=failure_atomic_positions,
        safe_basis=safe_basis, repair_rank=int(a.repair_rank),
        special_ids=gagd.special_token_ids(tok),
    )

    all_caches = mcf_repair.build_prompt_instance_delta_caches(
        model, tok, instances, selected_ids, device,
        int(a.cache_batch_size), llama_like,
    )
    failure_caches = [all_caches[int(i)] for i in failure_records]

    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, row_bases, device=output_layer.weight.device
    )
    opt = torch.optim.AdamW(
        delta_module.parameters(), lr=float(a.repair_lr), weight_decay=0.0
    )
    params = list(delta_module.parameters())

    ids_tensor = torch.tensor(
        selected_ids, dtype=torch.long, device=output_layer.weight.device
    )
    base_rows = output_layer.weight.index_select(0, ids_tensor).detach().clone()
    frozen_hash_before = v3.hash_frozen_parameters(model, output_layer)

    logs: List[Dict[str, Any]] = []
    rollback_count = 0
    accepted_hist: Dict[str, int] = {}
    feasible_delta = None

    for step in range(1, int(a.repair_steps) + 1):
        opt.zero_grad(set_to_none=True)
        delta = delta_module.effective_delta()
        margins_f = v3.record_margins_from_caches(failure_caches, delta)
        hinge = F.relu(float(a.train_mcf_margin) - margins_f).square().mean()
        kl_p = v3.protected_kl(
            base_logits=atomic_base_logits, hidden=hidden,
            protected_atomic_positions=protected_atomic_positions,
            selected_ids=selected_ids, delta_rows=delta,
        )
        l2 = delta.square().mean()
        loss = hinge + float(a.protected_kl_weight) * kl_p + float(a.delta_l2) * l2
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")

        old_params = v3.parameter_snapshot(delta_module)
        opt_state_before = copy.deepcopy(opt.state_dict())
        loss.backward()
        if float(a.grad_clip) > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient at step {step}")
        else:
            grad_norm = None
        opt.step()
        proposed_params = v3.parameter_snapshot(delta_module)

        accepted_alpha = None
        accepted_protection = None
        for alpha in BACKTRACK_SCALES:
            v3.set_interpolated_parameters(
                delta_module, old_params, proposed_params, float(alpha)
            )
            protection = v3.hard_protection_metrics(
                all_record_caches=all_caches,
                protected_record_positions=protected_records,
                protected_atomic_positions=protected_atomic_positions,
                base_logits=atomic_base_logits, hidden=hidden,
                selected_ids=selected_ids,
                delta_rows=delta_module.effective_delta(),
                mcf_margin_floor=float(a.protected_mcf_margin_floor),
            )
            if (int(protection["protected_mcf_regressions"]) == 0 and
                    max(0.0, float(protection["protected_kl"])) <= float(a.protected_kl_max)):
                accepted_alpha = float(alpha)
                accepted_protection = protection
                break
        if accepted_alpha is None:
            v3.set_interpolated_parameters(delta_module, old_params, old_params, 1.0)
            opt.load_state_dict(opt_state_before)
            rollback_count += 1
            accepted_alpha = 0.0
            accepted_protection = v3.hard_protection_metrics(
                all_record_caches=all_caches,
                protected_record_positions=protected_records,
                protected_atomic_positions=protected_atomic_positions,
                base_logits=atomic_base_logits, hidden=hidden,
                selected_ids=selected_ids,
                delta_rows=delta_module.effective_delta(),
                mcf_margin_floor=float(a.protected_mcf_margin_floor),
            )
        else:
            key = f"{accepted_alpha:g}"
            accepted_hist[key] = int(accepted_hist.get(key, 0)) + 1

        with torch.no_grad():
            cur = delta_module.effective_delta().detach()
            cached_all = v3.record_margins_from_caches(all_caches, cur)
            cached_fail = int((cached_all < float(a.train_mcf_margin)).sum().item())
            exact_pass = False
            exact_min = None
            if cached_fail == 0:
                exact_margins, exact_qdelta = exact_materialized_margins(
                    model=model, tok=tok, output_layer=output_layer,
                    selected_ids=selected_ids, base_rows=base_rows,
                    delta_rows=cur, scale=1.0, instances=instances,
                    device=device, llama_like=llama_like,
                    batch_size=int(a.cache_batch_size),
                )
                exact_pass = bool(
                    (exact_margins >= float(a.final_mcf_margin)).all().item()
                )
                exact_min = float(exact_margins.min().item())
                if exact_pass:
                    feasible_delta = cur.clone()
            row = {
                "step": int(step),
                "cached_mcf_failures": cached_fail,
                "cached_min_margin": float(cached_all.min().detach().cpu()),
                "exact_bf16_feasible": bool(exact_pass),
                "exact_bf16_min_margin": exact_min,
                "protected_mcf_regressions": int(accepted_protection["protected_mcf_regressions"]),
                "protected_kl": max(0.0, float(accepted_protection["protected_kl"])),
                "accepted_backtrack_alpha": float(accepted_alpha),
                "delta_norm": float(cur.norm().cpu()),
            }
            if grad_norm is not None:
                row["grad_norm"] = float(grad_norm.detach().cpu())
            if (step == 1 or step % int(a.check_every) == 0 or
                    cached_fail == 0 or step == int(a.repair_steps)):
                logs.append(row)
                print(json.dumps(row))
            if feasible_delta is not None:
                break

    del opt
    if feasible_delta is None:
        raise RuntimeError(
            "No exact-bf16 feasible row-specific repair found; refusing to save an unsafe checkpoint"
        )

    selected_scale, quantized_delta, scale_metrics, scale_evals = minimum_exact_scale(
        model=model, tok=tok, output_layer=output_layer,
        selected_ids=selected_ids, base_rows=base_rows,
        delta_rows=feasible_delta, instances=instances, device=device,
        llama_like=llama_like, batch_size=int(a.cache_batch_size),
        final_margin=float(a.final_mcf_margin),
        protected_records=protected_records,
        protected_floor=float(a.protected_mcf_margin_floor),
        protected_atomic_positions=protected_atomic_positions,
        atomic_base_logits=atomic_base_logits, hidden=hidden,
        protected_kl_max=float(a.protected_kl_max),
        bisect_steps=int(a.scale_bisect_steps),
    )

    # Materialize the exact quantized delta selected by line search.
    final_rows = base_rows.float() + quantized_delta.to(base_rows.device).float()
    output_layer.weight.index_copy_(0, ids_tensor, final_rows.to(output_layer.weight.dtype))

    frozen_hash_after = v3.hash_frozen_parameters(model, output_layer)
    frozen_bit_exact = frozen_hash_before == frozen_hash_after
    if not frozen_bit_exact:
        raise RuntimeError("Stage 2 changed embeddings and/or transformer parameters")

    final_margins = shared_stage2.mcf_direct_margins(
        model, tok, instances, device, llama_like, int(a.cache_batch_size),
        sensitive_field="target_true", reference_field="target_new",
    ).detach().float().cpu()
    final_failures = [
        i for i, x in enumerate(final_margins.tolist())
        if float(x) < float(a.final_mcf_margin)
    ]
    final_atomic, _ = stage1v2.evaluate_atomic_cases(
        model, tok, sensitive_cases, llama_like=llama_like,
        device=device, batch_size=int(a.cache_batch_size),
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    final_gate = {
        "all_mcf_direct_records_pass_final_margin": len(final_failures) == 0,
        "minimum_final_mcf_margin": float(final_margins.min().item()),
        "protected_regressions_zero": int(scale_metrics["protected_regressions"]) == 0,
        "protected_kl_within_limit": float(scale_metrics["protected_kl"]) <= float(a.protected_kl_max),
        "embeddings_and_transformer_bit_exact": bool(frozen_bit_exact),
    }
    final_gate["passed"] = bool(
        final_gate["all_mcf_direct_records_pass_final_margin"]
        and final_gate["protected_regressions_zero"]
        and final_gate["protected_kl_within_limit"]
        and final_gate["embeddings_and_transformer_bit_exact"]
    )

    summary: Dict[str, Any] = {
        "schema_version": 4,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_stage1_protocol": stage1_config.get("protocol"),
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "primary_gate": "NLL(target_true)-NLL(target_new)",
        "train_mcf_margin": float(a.train_mcf_margin),
        "final_mcf_margin": float(a.final_mcf_margin),
        "stage1_mcf_success_count_P": len(protected_records),
        "stage1_mcf_failure_count_F": len(failure_records),
        "safe_basis": safe_report,
        "row_specific_repair": True,
        "row_basis_reports": row_reports,
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "parameterization": "Delta w_s = c_s B_F,s",
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "repair_objective": "MCF sequence-margin hinge(F) + KL(P) + L2",
        "atomic_gate_role": "diagnostic_only",
        "optimizer_logs": logs,
        "accepted_backtrack_histogram": accepted_hist,
        "rollback_count": int(rollback_count),
        "unscaled_feasible_delta_norm": float(feasible_delta.norm().cpu()),
        "minimum_exact_bf16_scale": float(selected_scale),
        "final_effective_delta_norm": float(quantized_delta.norm().cpu()),
        "scale_line_search": scale_evals,
        "final_mcf_record_failure_count": len(final_failures),
        "final_mcf_record_failure_positions": final_failures,
        "final_mcf_record_min_margin": float(final_margins.min().item()),
        "final_atomic_failure_count_diagnostic": int(
            (final_atomic < float(a.atomic_margin)).sum().item()
        ),
        "final_atomic_min_margin_diagnostic": float(final_atomic.min().item()),
        "final_gate": final_gate,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
        "training_visible_sha256": stage1v2.sha256_file(visible_path),
        "split_manifest_sha256": stage1v2.sha256_file(manifest_path),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    print(json.dumps(summary, indent=2))
    print(
        f"Row-specific minimal Stage 2: MCF failures {len(failure_records)} -> {len(final_failures)}; "
        f"scale={selected_scale:.6f}; norm {float(feasible_delta.norm().cpu()):.4f} -> "
        f"{float(quantized_delta.norm().cpu()):.4f}; gate={final_gate['passed']}"
    )
    print(f"Final checkpoint: {ckpt}")
    if not final_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
