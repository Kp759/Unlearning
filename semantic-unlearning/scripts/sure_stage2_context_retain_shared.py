#!/usr/bin/env python3
"""Retain-protected fixed shared SURE Stage 2 for MCF and ZsRE.

Stage 2 uses exactly the same objective as retain-protected Stage 1, restricted
on the forget side to residual direct sensitive PredictionCases.  The same
1,000 disjoint retain-train prompts remain active through a full-distribution
KL teacher loss.  No held-out probes or retain-eval records are used.
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
    p.add_argument("--base-logits-cache", required=True)
    p.add_argument("--base-retain-logits-cache", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-train-num", type=int, default=1000)
    p.add_argument("--candidate-ranks", default="2,8,0")
    p.add_argument("--repair-steps", type=int, default=800)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--ga-weight", type=float, default=4.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--retain-kl-weight", type=float, default=1.0)
    p.add_argument("--repair-l2", type=float, default=1e-6)
    p.add_argument("--constraint-margin", type=float, default=0.25)
    p.add_argument("--required-nll-increase", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--retain-batch-size", type=int, default=4)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(a: argparse.Namespace):
    forget = json.loads(Path(a.training_visible_path).resolve().read_text(encoding="utf-8"))
    retained = json.loads(Path(a.training_visible_retain_path).resolve().read_text(encoding="utf-8"))
    manifest = json.loads(Path(a.split_manifest).resolve().read_text(encoding="utf-8"))
    if len(forget) != a.forget_num or len(retained) != a.retain_train_num:
        raise RuntimeError("locked forget/retain counts do not match requested counts")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("split manifest seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("retain_train_eval_overlap", -1)) != 0:
        raise RuntimeError("retain-train/eval overlap is not zero")
    sensitive_field = core.sensitive_answer_field(a.dataset)
    for i, record in enumerate(forget):
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts") or record.get("generation_prompts"):
            raise RuntimeError(f"forget record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get(sensitive_field, {}).get("str"):
            raise RuntimeError(f"forget record {i} lacks sensitive field")
    for i, record in enumerate(retained):
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get("prompt"):
            raise RuntimeError(f"retain record {i} lacks prompt")
        if "target_true" in rr or "target_new" in rr:
            raise RuntimeError("retain-train answer labels leaked")
    return forget, retained, manifest


def count_constraints(report: Dict[str, torch.Tensor], a: argparse.Namespace) -> int:
    return shared.count_failures(
        report["logit_margin"], report["sensitive_nll_increase"],
        required_logit_margin=a.constraint_margin,
        required_nll_increase=a.required_nll_increase,
    )


def candidate_key(report: Dict[str, Any], order: int) -> Tuple[int, int, float]:
    return (int(report["direct_failures"]), int(order), float(report["delta_norm"]))


def optimize_candidate(
    *,
    a: argparse.Namespace,
    forget_cases,
    retain_cases,
    active_indices: Sequence[int],
    selected_ids: Sequence[int],
    bases,
    basis_reports,
    base_forget_logits: torch.Tensor,
    base_retain_logits: torch.Tensor,
    model,
    tok,
    output_layer,
    llama_like: bool,
    device: torch.device,
    requested_rank: int,
    order: int,
):
    active_cases = [forget_cases[i] for i in active_indices]
    active_base_logits = base_forget_logits[list(active_indices)]
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=output_layer.weight.device
    )
    optimizer = torch.optim.AdamW(
        delta_module.parameters(), lr=a.repair_lr, weight_decay=0.0
    )
    forget_sampler = core.IndexSampler(len(active_cases), a.batch_size, a.seed + 100003)
    retain_sampler = core.IndexSampler(len(retain_cases), a.retain_batch_size, a.seed + 300007)

    best_failures = 10**9
    best_loss = float("inf")
    best_step = 0
    best_delta = delta_module.effective_delta().detach().clone()
    logs: List[Dict[str, Any]] = []

    hook = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    try:
        model.eval()
        for step in range(1, a.repair_steps + 1):
            fidx = forget_sampler.next()
            ridx = retain_sampler.next()
            fbatch = [active_cases[i] for i in fidx]
            rbatch = [retain_cases[i] for i in ridx]
            optimizer.zero_grad(set_to_none=True)

            flogits = core.forward_last_logits(model, tok, fbatch, device)
            ftids = core.official_target_ids(
                tok, fbatch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(flogits, ftids)
            forget_gd = core.gd_non_sensitive_kl(
                flogits, active_base_logits[fidx], ftids
            )

            rlogits = core.forward_last_logits(model, tok, rbatch, device)
            retain_kl = retain.full_distribution_kl(
                rlogits, base_retain_logits[ridx]
            )

            delta = delta_module.effective_delta()
            l2 = delta.square().mean()
            loss = (
                a.ga_weight * ga
                + a.gd_weight * forget_gd
                + a.retain_kl_weight * retain_kl
                + a.repair_l2 * l2
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage-2 loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), 1.0)
            optimizer.step()

            if step == 1 or step % a.check_every == 0 or step == a.repair_steps:
                constraints = shared.evaluate_shared_constraints(
                    model, tok, forget_cases, base_forget_logits,
                    llama_like=llama_like, device=device, batch_size=a.batch_size,
                )
                failures = count_constraints(constraints, a)
                cur = delta_module.effective_delta().detach()
                cur_loss = float(loss.detach().cpu())
                row = {
                    "step": int(step),
                    "direct_failures": int(failures),
                    "minimum_logit_margin": float(constraints["logit_margin"].min().cpu()),
                    "minimum_sensitive_nll_increase": float(constraints["sensitive_nll_increase"].min().cpu()),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "forget_non_sensitive_gd_kl": float(forget_gd.detach().cpu()),
                    "retain_full_distribution_kl": float(retain_kl.detach().cpu()),
                    "loss": cur_loss,
                    "delta_norm": float(cur.norm().cpu()),
                }
                logs.append(row)
                if failures < best_failures or (failures == best_failures and cur_loss < best_loss):
                    best_failures = failures
                    best_loss = cur_loss
                    best_step = step
                    best_delta = cur.clone()
                if failures == 0:
                    break
    finally:
        hook.remove()
    del optimizer

    h = core.register_output_delta_hook(output_layer, selected_ids, lambda: best_delta)
    try:
        constraints = shared.evaluate_shared_constraints(
            model, tok, forget_cases, base_forget_logits,
            llama_like=llama_like, device=device, batch_size=a.batch_size,
        )
    finally:
        h.remove()

    report = {
        "requested_context_rank": int(requested_rank),
        "rank_semantics": "0=full_observed_row_specific_forget_context_rank",
        "row_basis_reports": basis_reports,
        "row_context_ranks": [int(x["context_rank"]) for x in basis_reports],
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "best_step": int(best_step),
        "direct_failures": int(count_constraints(constraints, a)),
        "minimum_logit_margin": float(constraints["logit_margin"].min().cpu()),
        "minimum_sensitive_nll_increase": float(constraints["sensitive_nll_increase"].min().cpu()),
        "delta_norm": float(best_delta.norm().cpu()),
        "candidate_order": int(order),
        "logs": logs,
    }
    return report, best_delta


def main() -> None:
    a = parse_args()
    if min(a.forget_num, a.retain_train_num, a.repair_steps, a.batch_size, a.retain_batch_size, a.check_every) <= 0:
        raise ValueError("counts/steps/batches must be positive")
    if a.repair_lr <= 0 or a.ga_weight <= 0 or min(a.gd_weight, a.retain_kl_weight, a.repair_l2) < 0:
        raise ValueError("invalid optimization settings")
    if min(a.constraint_margin, a.required_nll_increase) < 0:
        raise ValueError("constraints must be non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    ranks = core.parse_rank_list(a.candidate_ranks)
    scales = core.parse_scales(a.candidate_scales)
    forget_records, retain_records, manifest = load_locked(a)

    ns = argparse.Namespace(model_path=a.model_path, dtype=a.dtype,
                            device_map=a.device_map, gradient_checkpointing=False)
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
    base_forget_logits = torch.load(
        Path(a.base_logits_cache).resolve(), map_location="cpu"
    )
    base_retain_logits = torch.load(
        Path(a.base_retain_logits_cache).resolve(), map_location="cpu"
    )
    if base_forget_logits.shape[0] != len(forget_cases):
        raise RuntimeError("Base forget-logit cache misaligned")
    if base_retain_logits.shape[0] != len(retain_cases):
        raise RuntimeError("Base retain-logit cache misaligned")

    before = shared.evaluate_shared_constraints(
        model, tok, forget_cases, base_forget_logits,
        llama_like=llama_like, device=device, batch_size=a.batch_size,
    )
    mask = shared.failure_mask(
        before["logit_margin"], before["sensitive_nll_increase"],
        required_logit_margin=a.constraint_margin,
        required_nll_increase=a.required_nll_increase,
    )
    active_indices = [i for i, flag in enumerate(mask.detach().cpu().tolist()) if bool(flag)]
    active_before = len(active_indices)

    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "checkpoint"
    candidate_reports: List[Dict[str, Any]] = []
    candidate_deltas: List[torch.Tensor] = []
    selected_ids: List[int] = []
    scale_reports: List[Dict[str, Any]] = []
    selected_scale = 0.0
    chosen_index = None

    if active_indices:
        active_cases = [forget_cases[i] for i in active_indices]
        active_tids = core.official_target_ids(
            tok, active_cases, llama_like=llama_like, device=device
        )
        selected_ids = sorted(set(int(x) for x in active_tids.detach().cpu().tolist()))

        for order, rank in enumerate(ranks):
            max_rank = None if rank == 0 else int(rank)
            bases, basis_reports = context.build_row_specific_bases(
                model, tok, active_cases, selected_ids=selected_ids,
                llama_like=llama_like, device=device, batch_size=a.batch_size,
                max_rank=max_rank,
            )
            report, delta = optimize_candidate(
                a=a,
                forget_cases=forget_cases,
                retain_cases=retain_cases,
                active_indices=active_indices,
                selected_ids=selected_ids,
                bases=bases,
                basis_reports=basis_reports,
                base_forget_logits=base_forget_logits,
                base_retain_logits=base_retain_logits,
                model=model,
                tok=tok,
                output_layer=output_layer,
                llama_like=llama_like,
                device=device,
                requested_rank=rank,
                order=order,
            )
            candidate_reports.append(report)
            candidate_deltas.append(delta)
            print("Retain-protected repair candidate", {
                "rank": rank,
                "row_ranks": report["row_context_ranks"],
                "direct_failures": report["direct_failures"],
                "delta_norm": report["delta_norm"],
            })
            if report["direct_failures"] == 0:
                break

        chosen_index = min(
            range(len(candidate_reports)),
            key=lambda i: candidate_key(candidate_reports[i], i),
        )
        chosen_delta = candidate_deltas[chosen_index]
        for scale in scales:
            h = core.register_output_delta_hook(
                output_layer, selected_ids,
                lambda scale=scale: chosen_delta * float(scale),
            )
            try:
                constraints = shared.evaluate_shared_constraints(
                    model, tok, forget_cases, base_forget_logits,
                    llama_like=llama_like, device=device, batch_size=a.batch_size,
                )
            finally:
                h.remove()
            scale_reports.append({
                "scale": float(scale),
                "direct_failures": int(count_constraints(constraints, a)),
                "minimum_logit_margin": float(constraints["logit_margin"].min().cpu()),
                "minimum_sensitive_nll_increase": float(constraints["sensitive_nll_increase"].min().cpu()),
                "effective_delta_norm": float(chosen_delta.norm().cpu() * float(scale)),
            })
        selected_scale = core.choose_scale(scale_reports)
        core.materialize_output_delta(
            output_layer, selected_ids, chosen_delta * float(selected_scale)
        )

    final_constraints = shared.evaluate_shared_constraints(
        model, tok, forget_cases, base_forget_logits,
        llama_like=llama_like, device=device, batch_size=a.batch_size,
    )
    final_failures = count_constraints(final_constraints, a)

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    summary: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": "sure_fixed_shared_retain_v3",
        "method": "SURE-LM-fixed-shared-retain-stage2",
        "dataset": a.dataset,
        "seed": a.seed,
        "forget_num": a.forget_num,
        "retain_train_num": a.retain_train_num,
        "retain_eval_num": int(manifest.get("sampling", {}).get("retain_eval_num", 0)),
        "retain_train_eval_overlap": 0,
        "architecture_shared_across_mcf_zsre": True,
        "objective": "GA_sensitive + forget_non_sensitive_KL + retain_full_distribution_KL + L2",
        "ga_weight": a.ga_weight,
        "gd_weight": a.gd_weight,
        "retain_kl_weight": a.retain_kl_weight,
        "constraint_logit_margin": a.constraint_margin,
        "required_sensitive_nll_increase": a.required_nll_increase,
        "candidate_ranks": ranks,
        "candidate_reports": candidate_reports,
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "chosen_candidate": candidate_reports[chosen_index] if chosen_index is not None else None,
        "selected_scale": float(selected_scale),
        "active_before": int(active_before),
        "active_after": int(final_failures),
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "input_embeddings_modified": False,
        "transformer_trainable": 0,
        "retain_answer_labels_seen": False,
        "heldout_probes_seen": 0,
        "retain_eval_seen": 0,
        "repair_steps": a.repair_steps,
        "repair_lr": a.repair_lr,
        "forget_batch_size": a.batch_size,
        "retain_batch_size": a.retain_batch_size,
        "minimum_final_logit_margin": float(final_constraints["logit_margin"].min().cpu()),
        "minimum_final_sensitive_nll_increase": float(final_constraints["sensitive_nll_increase"].min().cpu()),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "repair_summary.json", summary)
    core.write_json(out_dir / "rank_candidates.json", candidate_reports)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)
    print(f"Retain-protected shared Stage 2 {a.dataset}: failures {active_before} -> {final_failures}; rows={len(selected_ids)}; scale={selected_scale:g}")


if __name__ == "__main__":
    main()
