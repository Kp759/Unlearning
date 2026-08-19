#!/usr/bin/env python3
"""Stage-1-only diagnostic for contrastive SURE-LM v2.

Runs the exact shared v2 Stage-1 learner and writes the complete training-visible
scale frontier. It never runs Stage 2 or official evaluation. This is intended
to diagnose whether direct forgetting, retain-action, norm, or KL-proxy guards
are binding before changing any locked hyperparameter.
"""
from __future__ import annotations

import argparse
import json

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_retain_kl as retain
import sure_contrastive_two_stage_v2 as v2


def parse_args():
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
    p.add_argument("--stage1-steps", type=int, default=600)
    p.add_argument("--stage1-batch-size", type=int, default=1)
    p.add_argument("--retain-batch-size", type=int, default=64)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--stage1-lr", type=float, default=5e-3)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--retain-action-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=0.0)
    p.add_argument("--contrastive-rank", type=int, default=2)
    p.add_argument("--contrastive-eps", type=float, default=1e-3)
    p.add_argument("--retain-weight-clip", type=float, default=10.0)
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--min-sensitive-nll-increase", type=float, default=4.0)
    p.add_argument("--retain-action-budget", type=float, default=0.25)
    p.add_argument("--max-total-delta-norm", type=float, default=1.5)
    p.add_argument("--max-forget-nonsensitive-kl", type=float, default=0.25)
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def main():
    a = parse_args()
    if a.contrastive_rank <= 0:
        raise ValueError("contrastive rank must be positive")
    scales = core.parse_scales(a.candidate_scales)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    forget_records, retain_records, manifest = v2.load_locked(a)
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
    forget_tids = core.official_target_ids(
        tok, forget_cases, llama_like=llama_like, device=device
    ).detach()
    selected_ids = sorted(set(int(x) for x in forget_tids.detach().cpu().tolist()))

    base_forget_logits = core.cache_base_logits(
        model, tok, forget_cases, device, batch_size=a.cache_batch_size
    )
    forget_hidden = core.forward_last_hidden(
        model, tok, forget_cases, device, a.cache_batch_size
    ).float().detach()
    retain_hidden, raw_probs, retain_stats = v2.cache_retain_hidden_and_probs(
        model, tok, retain_cases, selected_ids,
        device=device, batch_size=a.cache_batch_size,
    )
    retain_weights = v2.normalized_clipped_weights(raw_probs, a.retain_weight_clip)
    retain_hidden = retain_hidden.to(device=device, dtype=torch.float32)
    retain_weights = retain_weights.to(device=device, dtype=torch.float32)

    row_bases, row_reports = v2.build_contrastive_bases_from_hidden(
        forget_hidden, forget_tids, retain_hidden, retain_weights,
        all_selected_ids=selected_ids, requested_ids=selected_ids,
        rank_cap=a.contrastive_rank, relative_eps=a.contrastive_eps,
    )

    out = gagd.resolve_output_path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trained_delta = v2.optimize_stage1(
        a=a, model=model, tok=tok, output_layer=output_layer,
        selected_ids=selected_ids, row_bases=row_bases,
        forget_cases=forget_cases, base_forget_logits=base_forget_logits,
        retain_hidden=retain_hidden, retain_weights=retain_weights,
        llama_like=llama_like, device=device, out_dir=out,
    )
    torch.save({"row_ids": selected_ids, "delta": trained_delta.detach().cpu()},
               out / "stage1_unscaled_trained_delta.pt")

    reports = [
        v2.evaluate_stage1_scale(
            a=a, model=model, tok=tok, output_layer=output_layer,
            selected_ids=selected_ids, trained_delta=trained_delta, scale=s,
            forget_cases=forget_cases, base_forget_logits=base_forget_logits,
            retain_hidden=retain_hidden, retain_weights=retain_weights,
            llama_like=llama_like, device=device,
        )
        for s in scales
    ]
    chosen, mode = v2.choose_stage1_report(reports)
    core.write_json(out / "stage1_scale_reports.json", reports)
    core.write_json(out / "stage1_diagnostic_summary.json", {
        "schema_version": 1,
        "dataset": a.dataset,
        "seed": a.seed,
        "source_protocol": manifest.get("protocol"),
        "contrastive_rank": a.contrastive_rank,
        "contrastive_eps": a.contrastive_eps,
        "retain_weight_clip": a.retain_weight_clip,
        "constraint_margin": a.constraint_margin,
        "min_sensitive_nll_increase": a.min_sensitive_nll_increase,
        "retain_action_budget": a.retain_action_budget,
        "max_total_delta_norm": a.max_total_delta_norm,
        "max_forget_nonsensitive_kl": a.max_forget_nonsensitive_kl,
        "selection_mode": mode,
        "selected": chosen,
        "retain_stats": retain_stats,
        "row_basis_reports": row_reports,
        "heldout_probes_seen": 0,
        "retain_eval_seen": 0,
        "official_eval_run": False,
    })

    print("Stage-1 diagnostic complete:", out)
    print("selection_mode:", mode)
    print("selected:", json.dumps(chosen, sort_keys=True))
    print("\nscale frontier")
    for r in reports:
        print(json.dumps(r, sort_keys=True))


if __name__ == "__main__":
    main()
