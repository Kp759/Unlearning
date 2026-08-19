#!/usr/bin/env python3
"""Fixed shared context-conditioned SURE Stage 1 for MCF and ZsRE.

Architecture is identical across benchmarks:
  * direct forget prompts only;
  * teacher-forced GA on the sensitive answer;
  * GD = KL preservation of the frozen-Base non-sensitive vocabulary
    distribution, with the sensitive token removed and the rest renormalized;
  * transformer and input embeddings remain exactly Base;
  * only sensitive LM-head rows are trainable;
  * each sensitive row is constrained to its own direct forget-context hidden
    subspace;
  * direct-only scale selection uses the SAME two shared constraints:
      - best-other minus sensitive logit >= logit margin;
      - sensitive-token NLL increase vs Base >= NLL-increase threshold.

The benchmark adapter changes only which answer field is sensitive:
  * MCF  -> target_new in the locked training view
            (for target-true-sensitive MCF this slot contains ORIGINAL target_true)
  * ZsRE -> original target_true
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
import sure_shared_suppression as shared


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("mcf", "zsre"), required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=0.0)
    p.add_argument("--context-rank", type=int, default=2,
                   help="Per-row direct-forget context rank cap; 0 means full observed numerical rank.")
    p.add_argument("--constraint-margin", type=float, default=0.05)
    p.add_argument("--min-sensitive-nll-increase", type=float, default=4.0,
                   help="Minimum per-sensitive-token NLL increase versus frozen Base.")
    p.add_argument("--candidate-scales", default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def validate_locked(dataset: str, visible_path: Path, manifest_path: Path, seed: int, forget_num: int):
    records = json.loads(visible_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != forget_num:
        raise RuntimeError(f"Expected {forget_num} direct training-visible forget records")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("Split manifest seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != forget_num:
        raise RuntimeError("Split manifest forget count mismatch")
    expected = [int(x) for x in sampling.get("forget_case_ids", [])]
    actual = [int(r.get("case_id", -1)) for r in records]
    if expected and expected != actual:
        raise RuntimeError("Training-visible IDs do not match split manifest")

    sensitive_field = core.sensitive_answer_field(dataset)
    for i, record in enumerate(records):
        if (record.get("paraphrase_prompts") or record.get("neighborhood_prompts")
                or record.get("generation_prompts")):
            raise RuntimeError(f"Record {i} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise RuntimeError(f"Record {i} lacks requested_rewrite")
        if not rr.get(sensitive_field, {}).get("str"):
            raise RuntimeError(f"Record {i} lacks sensitive {sensitive_field}.str")
        if dataset == "zsre" and "target_new" in rr:
            raise RuntimeError("Fixed shared ZsRE architecture forbids target_new/neutral targets")
    return records, manifest


def main() -> None:
    a = parse_args()
    if min(a.steps, a.batch_size, a.cache_batch_size) <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if a.lr <= 0 or a.ga_weight <= 0 or a.gd_weight < 0 or a.delta_l2 < 0:
        raise ValueError("invalid optimization weights")
    if a.context_rank < 0 or a.constraint_margin < 0 or a.min_sensitive_nll_increase < 0:
        raise ValueError("context rank and direct constraints must be non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = validate_locked(a.dataset, visible_path, manifest_path, a.seed, a.forget_num)

    ns = argparse.Namespace(model_path=a.model_path, dtype=a.dtype,
                            device_map=a.device_map, gradient_checkpointing=False)
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    cases = core.expand_sensitive_cases(records, tok, dataset=a.dataset, llama_like=llama_like)
    if not cases:
        raise RuntimeError("No sensitive teacher-forced PredictionCases")
    all_tids = core.official_target_ids(tok, cases, llama_like=llama_like, device=device)
    selected_ids = sorted(set(int(x) for x in all_tids.detach().cpu().tolist()))

    max_rank = None if a.context_rank == 0 else int(a.context_rank)
    bases, basis_reports = context.build_row_specific_bases(
        model, tok, cases, selected_ids=selected_ids, llama_like=llama_like,
        device=device, batch_size=a.cache_batch_size, max_rank=max_rank,
    )
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=output_layer.weight.device
    )

    # Exact frozen-Base teacher on the same direct forget PredictionCases.
    base_logits = core.cache_base_logits(
        model, tok, cases, device, batch_size=a.cache_batch_size
    )
    out_dir = gagd.resolve_output_path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_logits_path = out_dir / "base_sensitive_case_logits.pt"
    torch.save(base_logits, base_logits_path)

    opt = torch.optim.AdamW(delta_module.parameters(), lr=a.lr, weight_decay=0.0)
    sampler = core.IndexSampler(len(cases), a.batch_size, a.seed)
    hook = core.register_output_delta_hook(output_layer, selected_ids, delta_module.effective_delta)
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, a.steps + 1):
                idx = sampler.next()
                batch = [cases[i] for i in idx]
                opt.zero_grad(set_to_none=True)
                logits = core.forward_last_logits(model, tok, batch, device)
                tids = core.official_target_ids(tok, batch, llama_like=llama_like, device=device)
                ga = core.ga_sensitive_logprob(logits, tids)
                gd = core.gd_non_sensitive_kl(logits, base_logits[idx], tids)
                delta = delta_module.effective_delta()
                l2 = delta.square().mean()
                loss = a.ga_weight * ga + a.gd_weight * gd + a.delta_l2 * l2
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite Stage-1 loss at step {step}")
                loss.backward()
                grad_norm = (torch.nn.utils.clip_grad_norm_(list(delta_module.parameters()), a.grad_clip)
                             if a.grad_clip > 0 else None)
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at step {step}")
                opt.step()
                if step == 1 or step % 25 == 0 or step == a.steps:
                    log_f.write(json.dumps({
                        "step": step,
                        "total_loss": float(loss.detach().cpu()),
                        "ga_sensitive_logprob": float(ga.detach().cpu()),
                        "gd_non_sensitive_kl": float(gd.detach().cpu()),
                        "delta_l2": float(l2.detach().cpu()),
                        "delta_norm": float(delta.detach().norm().cpu()),
                        "benchmark_retain_seen": 0,
                        "heldout_probes_seen": 0,
                    }) + "\n")
                    log_f.flush()
    finally:
        hook.remove()
    del opt

    trained_delta = delta_module.effective_delta().detach().clone()
    scales = core.parse_scales(a.candidate_scales)
    scale_reports = []
    for scale in scales:
        h = core.register_output_delta_hook(
            output_layer, selected_ids,
            lambda scale=scale: trained_delta * float(scale),
        )
        try:
            state = shared.evaluate_shared_constraints(
                model, tok, cases, base_logits,
                llama_like=llama_like, device=device,
                batch_size=a.cache_batch_size,
            )
        finally:
            h.remove()
        failures = shared.count_failures(
            state["logit_margin"], state["sensitive_nll_increase"],
            required_logit_margin=a.constraint_margin,
            required_nll_increase=a.min_sensitive_nll_increase,
        )
        scale_reports.append({
            "scale": float(scale),
            "direct_failures": failures,
            "minimum_suppression_margin": float(state["logit_margin"].min().detach().cpu()),
            "minimum_sensitive_nll_increase": float(state["sensitive_nll_increase"].min().detach().cpu()),
            "mean_sensitive_nll_increase": float(state["sensitive_nll_increase"].mean().detach().cpu()),
            "effective_delta_norm": float(trained_delta.norm().cpu() * scale),
        })
    selected_scale = core.choose_scale(scale_reports)
    final_delta = trained_delta * float(selected_scale)
    core.materialize_output_delta(output_layer, selected_ids, final_delta)
    final_state = shared.evaluate_shared_constraints(
        model, tok, cases, base_logits,
        llama_like=llama_like, device=device,
        batch_size=a.cache_batch_size,
    )
    final_failures = shared.count_failures(
        final_state["logit_margin"], final_state["sensitive_nll_increase"],
        required_logit_margin=a.constraint_margin,
        required_nll_increase=a.min_sensitive_nll_increase,
    )

    ckpt = out_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 2,
        "method": "SURE-LM-fixed-shared-context-stage1",
        "dataset": a.dataset,
        "protocol": "sure_fixed_shared_context_v2",
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "sensitive_answer_field": core.sensitive_answer_field(a.dataset),
        "prediction_case_count": len(cases),
        "architecture_shared_across_mcf_zsre": True,
        "objective": "ga_sensitive_logprob + gd_non_sensitive_distribution_kl + delta_l2",
        "no_reference_answer_ce": True,
        "input_embeddings_modified": False,
        "input_embeddings_equal_base_by_construction": True,
        "transformer_trainable": 0,
        "lm_head_untied": True,
        "editable_rows": "sensitive_answer_rows_only",
        "row_specific_context_projection": True,
        "context_rank_cap": int(a.context_rank),
        "row_basis_reports": basis_reports,
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "ga_weight": float(a.ga_weight),
        "gd_weight": float(a.gd_weight),
        "delta_l2": float(a.delta_l2),
        "constraint_semantics": [
            "best_non_sensitive_logit_minus_sensitive_logit",
            "sensitive_token_nll_increase_vs_frozen_base",
        ],
        "constraint_margin": float(a.constraint_margin),
        "min_sensitive_nll_increase": float(a.min_sensitive_nll_increase),
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "final_direct_failures": int(final_failures),
        "minimum_final_suppression_margin": float(final_state["logit_margin"].min().detach().cpu()),
        "minimum_final_sensitive_nll_increase": float(final_state["sensitive_nll_increase"].min().detach().cpu()),
        "mean_final_sensitive_nll_increase": float(final_state["sensitive_nll_increase"].mean().detach().cpu()),
        "base_logits_cache": str(base_logits_path.resolve()),
        "checkpoint": str(ckpt.resolve()),
        "benchmark_retain_seen": 0,
        "heldout_probes_seen": 0,
    }
    core.write_json(out_dir / "config_used.json", config)
    print("Fixed shared Stage 1 checkpoint:", ckpt)
    print("dataset:", a.dataset)
    print("sensitive field:", config["sensitive_answer_field"])
    print("direct PredictionCases:", len(cases))
    print("final direct failures:", config["final_direct_failures"])
    print("minimum sensitive NLL increase:", config["minimum_final_sensitive_nll_increase"])


if __name__ == "__main__":
    main()
