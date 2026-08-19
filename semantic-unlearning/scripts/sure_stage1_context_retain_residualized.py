#!/usr/bin/env python3
"""Retain-residualized row-specific SURE Stage 1 for MCF and ZsRE.

This is a locality-focused ablation of the fixed-shared Stage-1 architecture.
The transformer and input embeddings stay frozen at Base, and only sensitive
LM-head rows are editable.  For each sensitive token row s, the direct-forget
hidden states H_F,s are first residualized against a shared retain-context
subspace Q_R built from a disjoint prompt-only retain-training set:

    H_tilde_F,s = H_F,s - (H_F,s Q_R^T) Q_R

The row-specific basis B_s is then built from H_tilde_F,s and the sparse row
update is constrained to delta_w_s = a_s B_s.

Training objective remains the original shared Stage-1 GA/GD objective; retain
prompts affect only the allowed update geometry (no retain-KL term here).
No paraphrase, neighborhood, generation, retain-eval, or answer-label signal is
used during training or model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_context_projection as context
import sure_retain_kl as retain
import sure_shared_suppression as shared


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--training-visible-retain-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-train-num", type=int, default=1000)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=0.0)
    p.add_argument("--context-rank", type=int, default=2,
                   help="Per-row residualized forget-context rank cap; 0 = full numerical rank.")
    p.add_argument("--retain-rank", type=int, default=64,
                   help="Shared retain-context subspace rank cap; 0 = full numerical rank.")
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--min-sensitive-nll-increase", type=float, default=4.0)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(a: argparse.Namespace):
    forget_path = Path(a.training_visible_path).resolve()
    retain_path = Path(a.training_visible_retain_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    forget_records = json.loads(forget_path.read_text(encoding="utf-8"))
    retain_records = json.loads(retain_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(forget_records, list) or len(forget_records) != a.forget_num:
        raise RuntimeError(f"Expected {a.forget_num} direct forget records")
    if not isinstance(retain_records, list) or len(retain_records) != a.retain_train_num:
        raise RuntimeError(f"Expected {a.retain_train_num} retain-train records")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("split manifest seed mismatch")

    sampling = manifest.get("sampling", {})
    if int(sampling.get("retain_train_eval_overlap", -1)) != 0:
        raise RuntimeError("retain-train/eval overlap is not zero")
    if set(sampling.get("retain_train_case_ids", [])) & set(sampling.get("retain_eval_case_ids", [])):
        raise RuntimeError("retain-train and retain-eval IDs overlap")

    sensitive_field = core.sensitive_answer_field(a.dataset)
    for i, record in enumerate(forget_records):
        if (record.get("paraphrase_prompts") or record.get("neighborhood_prompts")
                or record.get("generation_prompts")):
            raise RuntimeError(f"forget record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get(sensitive_field, {}).get("str"):
            raise RuntimeError(f"forget record {i} lacks sensitive field")
        if a.dataset == "zsre" and "target_new" in rr:
            raise RuntimeError("ZsRE retain-residualized path forbids target_new/neutral targets")

    for i, record in enumerate(retain_records):
        if (record.get("paraphrase_prompts") or record.get("neighborhood_prompts")
                or record.get("generation_prompts")):
            raise RuntimeError(f"retain record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get("prompt"):
            raise RuntimeError(f"retain record {i} lacks direct prompt")
        if "target_true" in rr or "target_new" in rr:
            raise RuntimeError("retain-train must be prompt-only; answer labels leaked")

    return forget_records, retain_records, manifest


@torch.no_grad()
def build_retain_residualized_bases(
    model,
    tok,
    forget_cases: Sequence[core.SensitivePredictionCase],
    retain_cases: Sequence[core.SensitivePredictionCase],
    *,
    selected_ids: Sequence[int],
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    retain_rank_cap: int,
    row_rank_cap: int,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[Dict[str, Any]], Dict[str, Any]]:
    """Build shared retain basis, then row-specific bases from forget residuals."""
    forget_hidden = core.forward_last_hidden(model, tok, forget_cases, device, batch_size)
    retain_hidden = core.forward_last_hidden(model, tok, retain_cases, device, batch_size)
    if forget_hidden.ndim != 2 or retain_hidden.ndim != 2:
        raise RuntimeError("hidden-state caches must be rank-2 matrices")
    if forget_hidden.shape[1] != retain_hidden.shape[1]:
        raise RuntimeError("forget/retain hidden sizes differ")

    retain_max_rank = None if retain_rank_cap == 0 else int(retain_rank_cap)
    retain_basis = core.orthonormal_row_basis(retain_hidden, max_rank=retain_max_rank)
    if retain_basis.ndim != 2 or retain_basis.shape[0] <= 0:
        raise RuntimeError("retain-context basis has zero numerical rank")

    tids = core.official_target_ids(
        tok, forget_cases, llama_like=llama_like, device=device
    ).detach()
    row_max_rank = None if row_rank_cap == 0 else int(row_rank_cap)

    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    all_residual_ratios: List[float] = []
    all_overlap_ratios: List[float] = []

    q = retain_basis.to(device=forget_hidden.device, dtype=torch.float32)
    for token_id in [int(x) for x in selected_ids]:
        rows = forget_hidden[tids.eq(token_id)].float()
        if rows.numel() == 0:
            raise RuntimeError(f"Sensitive token {token_id} has no forget hidden rows")

        projection = (rows @ q.transpose(0, 1)) @ q
        residual = rows - projection
        original_norm = rows.norm(dim=1).clamp_min(1e-12)
        residual_ratio = residual.norm(dim=1) / original_norm
        overlap_ratio = projection.norm(dim=1) / original_norm

        basis = core.orthonormal_row_basis(residual, max_rank=row_max_rank)
        if basis.ndim != 2 or basis.shape[0] <= 0:
            raise RuntimeError(
                f"Sensitive token {token_id} has zero residual context rank after retain projection"
            )

        orthogonality = basis.float() @ q.transpose(0, 1)
        max_abs_retain_dot = float(orthogonality.abs().max().cpu()) if orthogonality.numel() else 0.0

        rr = [float(x) for x in residual_ratio.detach().cpu().tolist()]
        oo = [float(x) for x in overlap_ratio.detach().cpu().tolist()]
        all_residual_ratios.extend(rr)
        all_overlap_ratios.extend(oo)

        bases.append(basis.detach().float().contiguous())
        reports.append({
            "token_id": token_id,
            "context_count": int(rows.shape[0]),
            "context_rank_after_residualization": int(basis.shape[0]),
            "hidden_size": int(basis.shape[1]),
            "mean_residual_norm_ratio": float(residual_ratio.mean().cpu()),
            "min_residual_norm_ratio": float(residual_ratio.min().cpu()),
            "max_residual_norm_ratio": float(residual_ratio.max().cpu()),
            "mean_retain_projection_norm_ratio": float(overlap_ratio.mean().cpu()),
            "max_abs_basis_dot_retain_basis": max_abs_retain_dot,
        })

    residual_tensor = torch.tensor(all_residual_ratios, dtype=torch.float32)
    overlap_tensor = torch.tensor(all_overlap_ratios, dtype=torch.float32)
    diagnostics = {
        "retain_hidden_count": int(retain_hidden.shape[0]),
        "retain_basis_rank": int(retain_basis.shape[0]),
        "hidden_size": int(retain_basis.shape[1]),
        "forget_prediction_case_count": int(forget_hidden.shape[0]),
        "mean_residual_norm_ratio": float(residual_tensor.mean()),
        "median_residual_norm_ratio": float(residual_tensor.median()),
        "min_residual_norm_ratio": float(residual_tensor.min()),
        "max_residual_norm_ratio": float(residual_tensor.max()),
        "mean_retain_projection_norm_ratio": float(overlap_tensor.mean()),
        "median_retain_projection_norm_ratio": float(overlap_tensor.median()),
        "min_retain_projection_norm_ratio": float(overlap_tensor.min()),
        "max_retain_projection_norm_ratio": float(overlap_tensor.max()),
        "max_abs_row_basis_dot_retain_basis": max(
            float(r["max_abs_basis_dot_retain_basis"]) for r in reports
        ),
    }
    return retain_basis.detach().float().contiguous(), bases, reports, diagnostics


def main() -> None:
    a = parse_args()
    if min(a.forget_num, a.retain_train_num, a.steps, a.batch_size, a.cache_batch_size) <= 0:
        raise ValueError("counts/steps/batches must be positive")
    if a.lr <= 0 or a.ga_weight <= 0 or min(a.gd_weight, a.delta_l2) < 0:
        raise ValueError("invalid optimization settings")
    if min(a.context_rank, a.retain_rank, a.constraint_margin, a.min_sensitive_nll_increase) < 0:
        raise ValueError("rank/constraint values must be non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    forget_records, retain_records, manifest = load_locked(a)
    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    forget_cases = core.expand_sensitive_cases(
        forget_records, tok, dataset=a.dataset, llama_like=llama_like
    )
    retain_cases = retain.retain_prompt_cases(retain_records)
    if not forget_cases or not retain_cases:
        raise RuntimeError("forget and retain cases must both be non-empty")

    sensitive_tids = core.official_target_ids(
        tok, forget_cases, llama_like=llama_like, device=device
    )
    selected_ids = sorted(set(int(x) for x in sensitive_tids.detach().cpu().tolist()))

    retain_basis, row_bases, row_reports, residual_diagnostics = build_retain_residualized_bases(
        model,
        tok,
        forget_cases,
        retain_cases,
        selected_ids=selected_ids,
        llama_like=llama_like,
        device=device,
        batch_size=a.cache_batch_size,
        retain_rank_cap=a.retain_rank,
        row_rank_cap=a.context_rank,
    )
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, row_bases, device=output_layer.weight.device
    )

    base_logits = core.cache_base_logits(
        model, tok, forget_cases, device, batch_size=a.cache_batch_size
    )
    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_logits_path = out_dir / "base_sensitive_case_logits.pt"
    retain_basis_path = out_dir / "retain_context_basis.pt"
    torch.save(base_logits, base_logits_path)
    torch.save(retain_basis.cpu(), retain_basis_path)

    optimizer = torch.optim.AdamW(delta_module.parameters(), lr=a.lr, weight_decay=0.0)
    sampler = core.IndexSampler(len(forget_cases), a.batch_size, a.seed)
    hook = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, a.steps + 1):
                idx = sampler.next()
                batch = [forget_cases[i] for i in idx]
                optimizer.zero_grad(set_to_none=True)
                logits = core.forward_last_logits(model, tok, batch, device)
                tids = core.official_target_ids(
                    tok, batch, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(logits, tids)
                gd = core.gd_non_sensitive_kl(logits, base_logits[idx], tids)
                delta = delta_module.effective_delta()
                l2 = delta.square().mean()
                loss = a.ga_weight * ga + a.gd_weight * gd + a.delta_l2 * l2
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Stage-1 loss at step {step}")
                loss.backward()
                if a.grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        list(delta_module.parameters()), a.grad_clip
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError("non-finite gradient norm")
                optimizer.step()

                if step == 1 or step % 25 == 0 or step == a.steps:
                    log_f.write(json.dumps({
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "forget_non_sensitive_gd_kl": float(gd.detach().cpu()),
                        "delta_norm": float(delta.detach().norm().cpu()),
                        "retain_geometry_only": True,
                        "retain_kl_used": False,
                        "heldout_probes_seen": 0,
                        "retain_eval_seen": 0,
                    }) + "\n")
                    log_f.flush()
    finally:
        hook.remove()
    del optimizer

    trained_delta = delta_module.effective_delta().detach().clone()
    scales = core.parse_scales(a.candidate_scales)
    scale_reports = []
    for scale in scales:
        h = core.register_output_delta_hook(
            output_layer,
            selected_ids,
            lambda scale=scale: trained_delta * float(scale),
        )
        try:
            state = shared.evaluate_shared_constraints(
                model,
                tok,
                forget_cases,
                base_logits,
                llama_like=llama_like,
                device=device,
                batch_size=a.cache_batch_size,
            )
        finally:
            h.remove()
        failures = shared.count_failures(
            state["logit_margin"],
            state["sensitive_nll_increase"],
            required_logit_margin=a.constraint_margin,
            required_nll_increase=a.min_sensitive_nll_increase,
        )
        scale_reports.append({
            "scale": float(scale),
            "direct_failures": int(failures),
            "minimum_suppression_margin": float(state["logit_margin"].min().cpu()),
            "minimum_sensitive_nll_increase": float(state["sensitive_nll_increase"].min().cpu()),
            "mean_sensitive_nll_increase": float(state["sensitive_nll_increase"].mean().cpu()),
            "effective_delta_norm": float(trained_delta.norm().cpu() * float(scale)),
        })

    selected_scale = core.choose_scale(scale_reports)
    final_delta = trained_delta * float(selected_scale)
    core.materialize_output_delta(output_layer, selected_ids, final_delta)

    final_state = shared.evaluate_shared_constraints(
        model,
        tok,
        forget_cases,
        base_logits,
        llama_like=llama_like,
        device=device,
        batch_size=a.cache_batch_size,
    )
    final_failures = shared.count_failures(
        final_state["logit_margin"],
        final_state["sensitive_nll_increase"],
        required_logit_margin=a.constraint_margin,
        required_nll_increase=a.min_sensitive_nll_increase,
    )

    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "method": "SURE-LM-retain-residualized-row-specific-stage1",
        "protocol": "sure_retain_residualized_stage1_v1",
        "dataset": a.dataset,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "retain_train_num": int(a.retain_train_num),
        "retain_eval_num": int(manifest.get("sampling", {}).get("retain_eval_num", 0)),
        "retain_train_eval_overlap": 0,
        "sensitive_answer_field": core.sensitive_answer_field(a.dataset),
        "architecture_shared_across_mcf_zsre": True,
        "objective": "GA_sensitive + forget_non_sensitive_distribution_KL + delta_L2",
        "retain_role": "geometry_only_no_KL",
        "retain_answer_labels_seen": False,
        "heldout_probes_seen": 0,
        "retain_eval_seen": 0,
        "input_embeddings_modified": False,
        "input_embeddings_equal_base_by_construction": True,
        "transformer_trainable": 0,
        "lm_head_untied": True,
        "editable_rows": "sensitive_answer_rows_only",
        "row_specific_context_projection": True,
        "retain_residualization": True,
        "retain_basis_rank_cap": int(a.retain_rank),
        "retain_basis_rank_actual": int(retain_basis.shape[0]),
        "context_rank_cap": int(a.context_rank),
        "row_basis_reports": row_reports,
        "residualization_diagnostics": residual_diagnostics,
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "ga_weight": float(a.ga_weight),
        "gd_weight": float(a.gd_weight),
        "delta_l2": float(a.delta_l2),
        "constraint_margin": float(a.constraint_margin),
        "min_sensitive_nll_increase": float(a.min_sensitive_nll_increase),
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "final_direct_failures": int(final_failures),
        "minimum_final_suppression_margin": float(final_state["logit_margin"].min().cpu()),
        "minimum_final_sensitive_nll_increase": float(final_state["sensitive_nll_increase"].min().cpu()),
        "mean_final_sensitive_nll_increase": float(final_state["sensitive_nll_increase"].mean().cpu()),
        "base_logits_cache": str(base_logits_path.resolve()),
        "retain_basis_path": str(retain_basis_path.resolve()),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "config_used.json", config)

    print("Retain-residualized Stage 1 checkpoint:", ckpt)
    print("dataset:", a.dataset)
    print("sensitive field:", config["sensitive_answer_field"])
    print("retain basis rank:", config["retain_basis_rank_actual"])
    print("mean forget residual ratio:", residual_diagnostics["mean_residual_norm_ratio"])
    print("mean retain overlap ratio:", residual_diagnostics["mean_retain_projection_norm_ratio"])
    print("selected scale:", config["selected_scale"])
    print("final direct failures:", config["final_direct_failures"])
    print("minimum sensitive NLL increase:", config["minimum_final_sensitive_nll_increase"])


if __name__ == "__main__":
    main()
