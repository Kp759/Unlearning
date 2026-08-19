#!/usr/bin/env python3
"""Retain-protected fixed shared SURE Stage 1 for MCF and ZsRE.

Identical architecture across benchmarks:
  * direct forget sensitive-token GA;
  * forget-context non-sensitive-distribution GD to frozen Base;
  * full-distribution KL on a disjoint retain-train prompt set;
  * frozen transformer and Base input embeddings;
  * only sensitive LM-head rows editable;
  * row-specific direct-forget context projection;
  * identical direct constraints and scale selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

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
    p.add_argument("--retain-batch-size", type=int, default=4)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=4.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--retain-kl-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=0.0)
    p.add_argument("--context-rank", type=int, default=2)
    p.add_argument("--constraint-margin", type=float, default=0.25)
    p.add_argument("--required-nll-increase", type=float, default=4.0)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def load_locked(a: argparse.Namespace):
    forget_path = Path(a.training_visible_path).resolve()
    retain_path = Path(a.training_visible_retain_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    forget = json.loads(forget_path.read_text(encoding="utf-8"))
    retained = json.loads(retain_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(forget) != a.forget_num:
        raise RuntimeError(f"Expected {a.forget_num} forget records")
    if len(retained) != a.retain_train_num:
        raise RuntimeError(f"Expected {a.retain_train_num} retain-train records")
    if int(manifest.get("seed", -1)) != a.seed:
        raise RuntimeError("split manifest seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("retain_train_eval_overlap", -1)) != 0:
        raise RuntimeError("retain-train/eval overlap is not zero")
    if set(sampling.get("retain_train_case_ids", [])) & set(sampling.get("retain_eval_case_ids", [])):
        raise RuntimeError("retain-train/eval IDs overlap")
    sensitive_field = core.sensitive_answer_field(a.dataset)
    for i, record in enumerate(forget):
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts") or record.get("generation_prompts"):
            raise RuntimeError(f"forget record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get(sensitive_field, {}).get("str"):
            raise RuntimeError(f"forget record {i} lacks sensitive field")
    for i, record in enumerate(retained):
        if record.get("paraphrase_prompts") or record.get("neighborhood_prompts") or record.get("generation_prompts"):
            raise RuntimeError(f"retain record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping) or not rr.get("prompt"):
            raise RuntimeError(f"retain record {i} lacks direct prompt")
        if "target_true" in rr or "target_new" in rr:
            raise RuntimeError("retain-train must be prompt-only; answer labels leaked")
    return forget, retained, manifest


def failure_count(report: Dict[str, torch.Tensor], a: argparse.Namespace) -> int:
    return shared.count_failures(
        report["logit_margin"], report["sensitive_nll_increase"],
        required_logit_margin=a.constraint_margin,
        required_nll_increase=a.required_nll_increase,
    )


def main() -> None:
    a = parse_args()
    if min(a.forget_num, a.retain_train_num, a.steps, a.batch_size, a.retain_batch_size, a.cache_batch_size) <= 0:
        raise ValueError("counts/steps/batches must be positive")
    if a.lr <= 0 or a.ga_weight <= 0 or min(a.gd_weight, a.retain_kl_weight, a.delta_l2) < 0:
        raise ValueError("invalid optimization settings")
    if min(a.context_rank, a.constraint_margin, a.required_nll_increase) < 0:
        raise ValueError("context rank/constraints must be non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

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
    if not forget_cases or not retain_cases:
        raise RuntimeError("forget and retain PredictionCases must be non-empty")

    sensitive_tids = core.official_target_ids(
        tok, forget_cases, llama_like=llama_like, device=device
    )
    selected_ids = sorted(set(int(x) for x in sensitive_tids.detach().cpu().tolist()))
    max_rank = None if a.context_rank == 0 else int(a.context_rank)
    bases, basis_reports = context.build_row_specific_bases(
        model, tok, forget_cases, selected_ids=selected_ids, llama_like=llama_like,
        device=device, batch_size=a.cache_batch_size, max_rank=max_rank,
    )
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=output_layer.weight.device
    )

    # Frozen Base teachers.  Retain cache is stored in fp16 to reduce disk/RAM.
    base_forget_logits = core.cache_base_logits(
        model, tok, forget_cases, device, batch_size=a.cache_batch_size
    )
    base_retain_logits = core.cache_base_logits(
        model, tok, retain_cases, device, batch_size=a.cache_batch_size
    ).to(torch.float16)

    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_forget_path = out_dir / "base_sensitive_case_logits.pt"
    base_retain_path = out_dir / "base_retain_prompt_logits_fp16.pt"
    torch.save(base_forget_logits, base_forget_path)
    torch.save(base_retain_logits, base_retain_path)

    optimizer = torch.optim.AdamW(delta_module.parameters(), lr=a.lr, weight_decay=0.0)
    forget_sampler = core.IndexSampler(len(forget_cases), a.batch_size, a.seed)
    retain_sampler = core.IndexSampler(len(retain_cases), a.retain_batch_size, a.seed + 200003)
    hook = core.register_output_delta_hook(output_layer, selected_ids, delta_module.effective_delta)
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, a.steps + 1):
                fidx = forget_sampler.next()
                ridx = retain_sampler.next()
                fbatch = [forget_cases[i] for i in fidx]
                rbatch = [retain_cases[i] for i in ridx]
                optimizer.zero_grad(set_to_none=True)

                flogits = core.forward_last_logits(model, tok, fbatch, device)
                ftids = core.official_target_ids(
                    tok, fbatch, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(flogits, ftids)
                forget_gd = core.gd_non_sensitive_kl(flogits, base_forget_logits[fidx], ftids)

                rlogits = core.forward_last_logits(model, tok, rbatch, device)
                retain_kl = retain.full_distribution_kl(rlogits, base_retain_logits[ridx])

                delta = delta_module.effective_delta()
                l2 = delta.square().mean()
                loss = (
                    a.ga_weight * ga
                    + a.gd_weight * forget_gd
                    + a.retain_kl_weight * retain_kl
                    + a.delta_l2 * l2
                )
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
                        "forget_non_sensitive_gd_kl": float(forget_gd.detach().cpu()),
                        "retain_full_distribution_kl": float(retain_kl.detach().cpu()),
                        "delta_norm": float(delta.detach().norm().cpu()),
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
            output_layer, selected_ids,
            lambda scale=scale: trained_delta * float(scale),
        )
        try:
            constraints = shared.evaluate_shared_constraints(
                model, tok, forget_cases, base_forget_logits,
                llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
            )
        finally:
            h.remove()
        scale_reports.append({
            "scale": float(scale),
            "direct_failures": failure_count(constraints, a),
            "minimum_logit_margin": float(constraints["logit_margin"].min().cpu()),
            "minimum_sensitive_nll_increase": float(constraints["sensitive_nll_increase"].min().cpu()),
            "effective_delta_norm": float(trained_delta.norm().cpu() * float(scale)),
        })
    selected_scale = core.choose_scale(scale_reports)
    final_delta = trained_delta * float(selected_scale)
    core.materialize_output_delta(output_layer, selected_ids, final_delta)

    final_constraints = shared.evaluate_shared_constraints(
        model, tok, forget_cases, base_forget_logits,
        llama_like=llama_like, device=device, batch_size=a.cache_batch_size,
    )
    final_failures = failure_count(final_constraints, a)

    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": "sure_fixed_shared_retain_v3",
        "method": "SURE-LM-fixed-shared-retain-stage1",
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
        "steps": a.steps,
        "lr": a.lr,
        "forget_batch_size": a.batch_size,
        "retain_batch_size": a.retain_batch_size,
        "context_rank_cap": a.context_rank,
        "constraint_logit_margin": a.constraint_margin,
        "required_sensitive_nll_increase": a.required_nll_increase,
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "row_basis_reports": basis_reports,
        "input_embeddings_modified": False,
        "transformer_trainable": 0,
        "retain_answer_labels_seen": False,
        "heldout_probes_seen": 0,
        "retain_eval_seen": 0,
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "final_direct_failures": int(final_failures),
        "minimum_final_logit_margin": float(final_constraints["logit_margin"].min().cpu()),
        "minimum_final_sensitive_nll_increase": float(final_constraints["sensitive_nll_increase"].min().cpu()),
        "base_sensitive_logits_cache": str(base_forget_path.resolve()),
        "base_retain_logits_cache": str(base_retain_path.resolve()),
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "config_used.json", config)
    print("Retain-protected shared Stage 1 checkpoint:", ckpt)
    print("dataset:", a.dataset)
    print("forget PredictionCases:", len(forget_cases))
    print("retain prompt cases:", len(retain_cases))
    print("selected scale:", selected_scale)
    print("final direct failures:", final_failures)


if __name__ == "__main__":
    main()
