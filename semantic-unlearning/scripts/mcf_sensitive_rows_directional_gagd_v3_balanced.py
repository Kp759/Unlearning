#!/usr/bin/env python3
"""Recommended Directional SURE v3 entrypoint with separate GA and GD sampling.

This preserves the v3 architecture from mcf_sensitive_rows_directional_gagd_v3.py
but fixes one training-loop detail: GA samples only currently failing direct facts,
while same-prompt GD continues to sample all training-visible sensitive cases.
Relation-donor and Wikipedia preservation also remain global.  All geometry,
hard shared-drift constraints, active direct-margin gating, and final shared-scale
selection are inherited from the v3 helper module.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from tqdm import tqdm

import gagd_compare as gagd
import mcf_frozen_head_representation_repair as contract_helpers
import mcf_sensitive_rows_directional_gagd_v2 as V2
import mcf_sensitive_rows_directional_gagd_v3 as V3
import mcf_sensitive_rows_projected_gagd as projected
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_stage1_gagd_w1k as wikipedia_utility
import sure_stage2_sparse_repair as stage2


METHOD = "SURE-LM-MCF-directional-GAGD-v3-safe-shared-protected-balanced"
PROTOCOL = "mcf_target_true_directional_gagd_safe_shared_protected_v3_balanced"
parse_args = V3.parse_args


def _new_active_sampler(active_case_ids: List[int], a: argparse.Namespace, step: int):
    if not active_case_ids:
        return None
    return core.IndexSampler(
        len(active_case_ids), int(a.batch_size), int(a.seed) + 92001 + int(step)
    )


def main(argv=None) -> None:
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
    margins_before = V3._direct_margins(
        model, tok, benchmark_instances, device, llama_like, int(a.batch_size)
    )
    benchmark_before = V3._margin_report(
        margins_before, float(a.acceptance_margin)
    )

    locality_prompts, locality_protected, locality_receipt = (
        projected.build_relation_locality_controls(
            records, tok, benchmark_instances, int(a.subject_control_count)
        )
    )
    print(
        f"Caching Base locality references for {len(locality_prompts)} prompts...",
        flush=True,
    )
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
    print(
        f"Caching Base Wikipedia logits for {len(utility_prompts)} prompts...",
        flush=True,
    )
    utility_base_logits = wikipedia_utility.cache_utility_base_logits(
        model, tok, utility_prompts, device, int(a.utility_cache_batch_size)
    )

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Directional SURE v3 requires an untied LM head")
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("all Base parameters must remain frozen")

    ids_device = torch.tensor(
        sensitive_ids, dtype=torch.long, device=input_layer.weight.device
    )
    selected_base_input = (
        input_layer.weight.detach().index_select(0, ids_device).float().cpu()
    )
    selected_base_output = (
        output_layer.weight.detach()
        .index_select(0, ids_device.to(output_layer.weight.device))
        .float().cpu()
    )

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
    opt = V3._optimizer(
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
    ) = V3.refresh_geometry(
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

    V3.enforce_output_geometry_(
        safe_module.raw_delta,
        shared_module.raw_delta,
        preserve_module.raw_delta,
        sensitive_ids,
        safe_bases,
        shared_bases,
        protected_basis,
        protected_hidden,
        float(a.shared_protected_logit_drift_max),
    )

    margins = margins_before.clone()
    active_records, active_case_ids = V3._active_case_indices(
        cases, margins, float(a.solver_margin)
    )
    active_sampler = _new_active_sampler(active_case_ids, a, 0)
    gd_sampler = core.IndexSampler(
        len(cases), int(a.batch_size), int(a.seed) + 92005
    )
    locality_sampler = core.IndexSampler(
        len(locality_prompts), int(a.locality_batch_size), int(a.seed) + 92007
    )
    utility_sampler = core.IndexSampler(
        len(utility_prompts), int(a.utility_batch_size), int(a.utility_seed) + 92009
    )

    geometry_log = (out_dir / "basis_refresh_log.jsonl").open("w", encoding="utf-8")
    direct_log = (out_dir / "direct_gate_log.jsonl").open("w", encoding="utf-8")
    train_log = (out_dir / "train_log.jsonl").open("w", encoding="utf-8")
    geometry_log.write(json.dumps(geometry_receipt) + "\n")
    direct_log.write(json.dumps({
        "step": 0,
        "active_record_count": len(active_records),
        "active_records": active_records,
        "solver": V3._margin_report(margins, float(a.solver_margin)),
        "acceptance": V3._margin_report(margins, float(a.acceptance_margin)),
    }) + "\n")
    geometry_log.flush()
    direct_log.flush()

    stopped_step = int(a.steps)
    try:
        for step in tqdm(range(1, int(a.steps) + 1), desc="MCF Directional SURE v3 balanced"):
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
                ) = V3.refresh_geometry(
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
                refresh["output_reprojection"] = V3.enforce_output_geometry_(
                    safe_module.raw_delta,
                    shared_module.raw_delta,
                    preserve_module.raw_delta,
                    sensitive_ids,
                    safe_bases,
                    shared_bases,
                    protected_basis,
                    protected_hidden,
                    float(a.shared_protected_logit_drift_max),
                )
                for param in output_params:
                    opt.state[param].clear()
                refresh["output_optimizer_moments_reset"] = True
                refresh["input_optimizer_moments_preserved"] = True
                geometry_log.write(json.dumps(refresh) + "\n")
                geometry_log.flush()

            # GA: active direct failures only.
            f_local = active_sampler.next()
            fidx = [active_case_ids[i] for i in f_local]
            forget_batch = [cases[i] for i in fidx]

            # GD: all training-visible cases, independent of the active GA tail.
            gidx = gd_sampler.next()
            gd_batch = [cases[i] for i in gidx]

            lidx = locality_sampler.next()
            locality_batch = [locality_prompts[i] for i in lidx]
            locality_ids = [locality_protected[i] for i in lidx]
            uidx = utility_sampler.next()
            utility_batch = [utility_prompts[i] for i in uidx]

            opt.zero_grad(set_to_none=True)

            ga_logits = core.forward_last_logits(model, tok, forget_batch, device)
            ga_tids = core.official_target_ids(
                tok, forget_batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(ga_logits, ga_tids)

            gd_logits = core.forward_last_logits(model, tok, gd_batch, device)
            gd_tids = core.official_target_ids(
                tok, gd_batch, llama_like=llama_like, device=device
            )
            gd = core.gd_non_sensitive_kl(
                gd_logits, same_prompt_base_logits[gidx], gd_tids
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
                + 0.5 * float(a.row_delta_weight)
                * (input_reg + safe_reg + shared_reg)
            )
            preserve_objective = (
                float(a.gd_weight) * gd
                + float(a.locality_kl_weight) * lkl
                + float(a.locality_sensitive_logit_weight) * lrow
                + float(a.utility_kl_weight) * ukl
                + 0.5 * float(a.row_delta_weight)
                * (input_reg + preserve_reg)
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
            ga_input_grad = V3._zero_if_none(ga_input_grad, input_module.delta)
            preserve_input_grad = V3._zero_if_none(
                preserve_input_grad, input_module.delta
            )
            safe_grad = V3._zero_if_none(safe_grad, safe_module.raw_delta)
            shared_grad = V3._zero_if_none(shared_grad, shared_module.raw_delta)
            preserve_grad = V3._zero_if_none(
                preserve_grad, preserve_module.raw_delta
            )

            safe_grad_proj, safe_grad_diag = V2.project_rows_tokenwise_span(
                safe_grad, sensitive_ids, safe_bases
            )
            shared_grad_proj, shared_grad_diag = V2.project_rows_tokenwise_span(
                shared_grad, sensitive_ids, shared_bases
            )
            preserve_grad_proj, preserve_grad_diag = V3.project_rows_protected_only(
                preserve_grad, sensitive_ids, protected_basis, shared_bases
            )

            input_module.delta.grad = (
                ga_input_grad + preserve_input_grad
            ).to(input_module.delta.dtype)
            safe_module.raw_delta.grad = safe_grad_proj.to(
                safe_module.raw_delta.dtype
            )
            shared_module.raw_delta.grad = shared_grad_proj.to(
                shared_module.raw_delta.dtype
            )
            preserve_module.raw_delta.grad = preserve_grad_proj.to(
                preserve_module.raw_delta.dtype
            )

            params = [input_module.delta] + output_params
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(params, float(a.grad_clip))
                if a.grad_clip > 0 else None
            )
            if grad_norm is not None and not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite grad norm at step {step}")
            opt.step()
            geometry_projection = V3.enforce_output_geometry_(
                safe_module.raw_delta,
                shared_module.raw_delta,
                preserve_module.raw_delta,
                sensitive_ids,
                safe_bases,
                shared_bases,
                protected_basis,
                protected_hidden,
                float(a.shared_protected_logit_drift_max),
            )

            if step == 1 or step % int(a.check_every) == 0 or step == int(a.steps):
                margins = V3._direct_margins(
                    model, tok, benchmark_instances, device, llama_like,
                    int(a.batch_size)
                )
                active_records, active_case_ids = V3._active_case_indices(
                    cases, margins, float(a.solver_margin)
                )
                active_sampler = _new_active_sampler(active_case_ids, a, step)
                gate_row = {
                    "step": int(step),
                    "active_record_count": int(len(active_records)),
                    "active_records": active_records,
                    "solver": V3._margin_report(margins, float(a.solver_margin)),
                    "acceptance": V3._margin_report(
                        margins, float(a.acceptance_margin)
                    ),
                }
                direct_log.write(json.dumps(gate_row) + "\n")
                direct_log.flush()

                total_output = (
                    safe_module.effective_delta()
                    + shared_module.effective_delta()
                    + preserve_module.effective_delta()
                )
                train_log.write(json.dumps({
                    "step": int(step),
                    "active_record_count": int(len(active_records)),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_same_prompt_non_sensitive_kl": float(gd.detach().cpu()),
                    "gd_sampling_scope": "all_training_visible_sensitive_cases",
                    "ga_sampling_scope": "active_direct_failures_only",
                    "relation_locality_kl": float(lkl.detach().cpu()),
                    "relation_sensitive_logit_mse": float(lrow.detach().cpu()),
                    "wikipedia_utility_kl": float(ukl.detach().cpu()),
                    "input_delta_mse": float(
                        input_module.delta.square().mean().detach().cpu()
                    ),
                    "safe_delta_mse": float(
                        safe_module.raw_delta.square().mean().detach().cpu()
                    ),
                    "shared_delta_mse": float(
                        shared_module.raw_delta.square().mean().detach().cpu()
                    ),
                    "preserve_delta_mse": float(
                        preserve_module.raw_delta.square().mean().detach().cpu()
                    ),
                    "total_output_delta_mse": float(
                        total_output.square().mean().detach().cpu()
                    ),
                    "safe_gradient_kept_norm": float(safe_grad_diag["kept_norm"]),
                    "shared_gradient_kept_norm": float(shared_grad_diag["kept_norm"]),
                    "preserve_gradient_protected_only_norm": float(
                        preserve_grad_diag["protected_only_norm"]
                    ),
                    "geometry_projection": geometry_projection,
                    "benchmark_retain_seen": 0,
                    "official_neighborhood_seen": 0,
                    "official_paraphrase_seen": 0,
                    "PPL_seen": False,
                }) + "\n")
                train_log.flush()

                if active_sampler is None:
                    stopped_step = int(step)
                    break
    finally:
        geometry_log.close()
        direct_log.close()
        train_log.close()

    del opt

    (
        protected_hidden,
        protected_basis,
        safe_bases,
        shared_bases,
        final_geometry,
    ) = V3.refresh_geometry(
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
    final_geometry["output_reprojection"] = V3.enforce_output_geometry_(
        safe_module.raw_delta,
        shared_module.raw_delta,
        preserve_module.raw_delta,
        sensitive_ids,
        safe_bases,
        shared_bases,
        protected_basis,
        protected_hidden,
        float(a.shared_protected_logit_drift_max),
    )

    chosen_shared_scale, shared_scale_reports = V3._choose_shared_scale(
        model, tok, benchmark_instances, shared_module.raw_delta,
        a.candidate_shared_scales, float(a.acceptance_margin),
        device, llama_like, int(a.batch_size)
    )
    final_geometry["chosen_shared_scale"] = float(chosen_shared_scale)
    final_geometry["shared_scale_reports"] = shared_scale_reports
    core.write_json(out_dir / "final_geometry_receipt.json", final_geometry)
    core.write_json(out_dir / "shared_scale_reports.json", shared_scale_reports)

    benchmark_pre_materialize = V3._margin_report(
        V3._direct_margins(
            model, tok, benchmark_instances, device, llama_like, int(a.batch_size)
        ),
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
    directional_after = V3.directional_residual_diagnostics(
        safe_module.raw_delta.detach(),
        shared_module.raw_delta.detach(),
        preserve_module.raw_delta.detach(),
        sensitive_ids,
        safe_bases,
        shared_bases,
        protected_basis,
        protected_hidden,
    )

    input_delta_final = input_module.delta.detach().float().cpu()
    safe_final = safe_module.effective_delta().detach().float().cpu()
    shared_final = shared_module.effective_delta().detach().float().cpu()
    preserve_final = preserve_module.effective_delta().detach().float().cpu()
    total_output_final = safe_final + shared_final + preserve_final

    input_module.remove()
    output_hook.remove()
    V2._materialize_selected_delta(
        input_layer.weight, sensitive_ids, input_delta_final
    )
    core.materialize_output_delta(
        output_layer, sensitive_ids, total_output_final
    )
    model.eval()

    materialized_input = (
        input_layer.weight.index_select(0, ids_device).float().cpu()
        - selected_base_input
    )
    output_ids = ids_device.to(output_layer.weight.device)
    materialized_output = (
        output_layer.weight.index_select(0, output_ids).float().cpu()
        - selected_base_output
    )
    input_materialization_error = float(
        (materialized_input - input_delta_final).abs().max()
    )
    output_materialization_error = float(
        (materialized_output - total_output_final).abs().max()
    )

    benchmark_after = V3._margin_report(
        V3._direct_margins(
            model, tok, benchmark_instances, device, llama_like, int(a.batch_size)
        ),
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

    summary: Dict[str, Any] = {
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
        "lm_head_untied": True,
        "selected_sensitive_token_ids": [int(x) for x in sensitive_ids],
        "selected_sensitive_row_count": int(len(sensitive_ids)),
        "ga_sampling_scope": "active_direct_failures_only",
        "gd_sampling_scope": "all_training_visible_sensitive_cases",
        "output_components": {
            "safe_ga": "protected-orthogonal forget span",
            "shared_ga": "forget projection inside protected span with hard H_P drift cap",
            "preservation": "protected span minus token shared-forget span",
        },
        "official_neighborhood_seen": 0,
        "official_paraphrase_seen": 0,
        "benchmark_retain_seen": 0,
        "PPL_seen": False,
        "steps_requested": int(a.steps),
        "stopped_step": int(stopped_step),
        "solver_margin": float(a.solver_margin),
        "acceptance_margin": float(a.acceptance_margin),
        "shared_protected_logit_drift_max": float(
            a.shared_protected_logit_drift_max
        ),
        "basis_refresh_every": int(a.basis_refresh_every),
        "chosen_shared_scale": float(chosen_shared_scale),
        "shared_scale_reports": shared_scale_reports,
        "weights": {
            "ga": float(a.ga_weight),
            "gd": float(a.gd_weight),
            "relation_locality_kl": float(a.locality_kl_weight),
            "relation_sensitive_logit": float(
                a.locality_sensitive_logit_weight
            ),
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
    core.write_json(
        out_dir / "sensitive_rows_directional_gagd_v3_summary.json", summary
    )

    print("Directional SURE v3 balanced checkpoint:", ckpt)
    print("Transformer trainable: False")
    print("Input sensitive rows trainable: True")
    print("LM head untied: True")
    print("Sensitive row count:", len(sensitive_ids))
    print("GA sampling: active failures only")
    print("GD sampling: all training-visible cases")
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
