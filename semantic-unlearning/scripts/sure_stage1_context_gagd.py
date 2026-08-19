#!/usr/bin/env python3
"""Context-conditioned canonical SURE Stage 1 for target-true-sensitive MCF.

Expected locked adapter semantics:
  canonical target_new  = ORIGINAL target_true = sensitive answer
  canonical target_true = ORIGINAL target_new  = non-sensitive/reference answer

Only sparse sensitive LM-head rows are trainable through row-specific direct
forget-context bases.  Input embeddings and transformer blocks remain exactly
Base.  The direct-only objective combines:
  * GA on the sensitive answer;
  * explicit GD/CE on the non-sensitive/reference answer;
  * frozen-Base KL preservation on the non-sensitive vocabulary distribution;
  * optional sparse-delta L2.

No paraphrase, neighborhood, retain, or PPL data is opened.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import gagd_active_case_repair as mcf_repair
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_canonical_core as core
import sure_context_projection as context


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--reference-batch-size", type=int, default=1)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--emb-lm-lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--reference-gd-weight", type=float, default=1.0)
    p.add_argument("--distribution-kl-weight", type=float, default=1.0)
    p.add_argument("--delta-l2", type=float, default=0.0)
    p.add_argument(
        "--context-rank",
        type=int,
        default=0,
        help="Per-sensitive-row forget-context rank cap; 0 uses full numerical rank.",
    )
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--stage1-constraint-margin",
        type=float,
        default=0.0,
        help="Direct-only sensitive-reference NLL gap required by Stage-1 scale selection.",
    )
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def validate_locked(
    visible_path: Path,
    manifest_path: Path,
    seed: int,
    forget_num: int,
):
    records = json.loads(visible_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != forget_num:
        raise RuntimeError(f"Expected {forget_num} training-visible forget records")
    if int(manifest.get("seed", -1)) != seed:
        raise RuntimeError("Split manifest seed mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != forget_num:
        raise RuntimeError("Split manifest forget count mismatch")
    expected = [int(x) for x in sampling.get("forget_case_ids", [])]
    actual = [int(r.get("case_id", -1)) for r in records]
    if expected and expected != actual:
        raise RuntimeError("Training-visible IDs do not match split manifest")
    semantics = manifest.get("target_semantics", {})
    if semantics.get("original_sensitive_field") != "target_true":
        raise RuntimeError("Stage 1 requires the target-true-sensitive MCF adapter")
    if semantics.get("training_sensitive_slot") != "target_new":
        raise RuntimeError("Canonical sensitive slot must be target_new")
    for index, record in enumerate(records):
        if (
            record.get("paraphrase_prompts")
            or record.get("neighborhood_prompts")
            or record.get("generation_prompts")
        ):
            raise RuntimeError(f"Record {index} exposes held-out probes")
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise RuntimeError(f"Record {index} lacks requested_rewrite")
        for field in ("target_new", "target_true"):
            if not rr.get(field, {}).get("str"):
                raise RuntimeError(f"Record {index} lacks {field}.str")
    return records, manifest


def mcf_instances(records: Sequence[Mapping[str, Any]]) -> List[mcf_repair.MCFPromptInstance]:
    instances: List[mcf_repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        instances.append(
            mcf_repair.MCFPromptInstance(
                record_index=int(record.get("case_id", position)),
                sampled_position=position,
                prompt_type="rewrite",
                prompt_index=0,
                prompt=str(rr["prompt"]).format(subject),
                target_new=str(rr["target_new"]["str"]),
                target_true=str(rr["target_true"]["str"]),
            )
        )
    return instances


@torch.no_grad()
def direct_margins(model, tok, instances, device, llama_like, batch_size):
    values: List[torch.Tensor] = []
    for start in range(0, len(instances), batch_size):
        new_nll, true_nll = mcf_repair.official_prompt_instance_nll_tensors(
            model,
            tok,
            instances[start : start + batch_size],
            device,
            llama_like,
        )
        values.append((new_nll - true_nll).float())
    return torch.cat(values, dim=0) if values else torch.empty(0, device=device)


def main() -> None:
    a = parse_args()
    if min(a.steps, a.batch_size, a.reference_batch_size, a.cache_batch_size) <= 0:
        raise ValueError("steps and batch sizes must be positive")
    if a.emb_lm_lr <= 0 or a.ga_weight <= 0:
        raise ValueError("learning rate and GA weight must be positive")
    if a.reference_gd_weight < 0 or a.distribution_kl_weight < 0 or a.delta_l2 < 0:
        raise ValueError("GD/KL/L2 weights must be non-negative")
    if a.context_rank < 0 or a.stage1_constraint_margin < 0:
        raise ValueError("context rank and Stage-1 margin must be non-negative")

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    visible_path = Path(a.training_visible_path).resolve()
    manifest_path = Path(a.split_manifest).resolve()
    records, manifest = validate_locked(
        visible_path, manifest_path, a.seed, a.forget_num
    )

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

    sensitive_cases = context.expand_answer_field_cases(
        records, tok, field="target_new", llama_like=llama_like
    )
    reference_cases = context.expand_answer_field_cases(
        records, tok, field="target_true", llama_like=llama_like
    )
    if not sensitive_cases or not reference_cases:
        raise RuntimeError("Sensitive/reference teacher-forced cases must be non-empty")

    sensitive_tids_all = core.official_target_ids(
        tok, sensitive_cases, llama_like=llama_like, device=device
    )
    selected_ids = sorted(
        set(int(x) for x in sensitive_tids_all.detach().cpu().tolist())
    )
    max_rank = None if a.context_rank == 0 else a.context_rank
    bases, basis_reports = context.build_row_specific_bases(
        model,
        tok,
        sensitive_cases,
        selected_ids=selected_ids,
        llama_like=llama_like,
        device=device,
        batch_size=a.cache_batch_size,
        max_rank=max_rank,
    )
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, bases, device=output_layer.weight.device
    )

    # Frozen-Base teacher distribution before any hook/delta is active.
    base_sensitive_logits = core.cache_base_logits(
        model,
        tok,
        sensitive_cases,
        device,
        batch_size=a.cache_batch_size,
    )

    opt = torch.optim.AdamW(
        delta_module.parameters(), lr=a.emb_lm_lr, weight_decay=0.0
    )
    sensitive_sampler = core.IndexSampler(len(sensitive_cases), a.batch_size, a.seed)
    reference_sampler = core.IndexSampler(
        len(reference_cases), a.reference_batch_size, a.seed + 7919
    )

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    hook = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, a.steps + 1):
                sens_idx = sensitive_sampler.next()
                ref_idx = reference_sampler.next()
                sens_batch = [sensitive_cases[i] for i in sens_idx]
                ref_batch = [reference_cases[i] for i in ref_idx]

                opt.zero_grad(set_to_none=True)

                sens_logits = core.forward_last_logits(model, tok, sens_batch, device)
                sens_tids = core.official_target_ids(
                    tok, sens_batch, llama_like=llama_like, device=device
                )
                ga = core.ga_sensitive_logprob(sens_logits, sens_tids)
                dist_kl = core.gd_non_sensitive_kl(
                    sens_logits,
                    base_sensitive_logits[sens_idx],
                    sens_tids,
                )

                ref_logits = core.forward_last_logits(model, tok, ref_batch, device)
                ref_tids = core.official_target_ids(
                    tok, ref_batch, llama_like=llama_like, device=device
                )
                reference_gd = F.cross_entropy(ref_logits.float(), ref_tids)

                delta = delta_module.effective_delta()
                l2 = delta.square().mean()
                total = (
                    a.ga_weight * ga
                    + a.reference_gd_weight * reference_gd
                    + a.distribution_kl_weight * dist_kl
                    + a.delta_l2 * l2
                )
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite context Stage-1 loss at step {step}")
                total.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(delta_module.parameters()), a.grad_clip
                ) if a.grad_clip > 0 else None
                if grad_norm is not None and not torch.isfinite(grad_norm):
                    raise FloatingPointError(f"Non-finite gradient norm at step {step}")
                opt.step()

                if step == 1 or step % 25 == 0 or step == a.steps:
                    log_f.write(
                        json.dumps(
                            {
                                "step": step,
                                "total_loss": float(total.detach().cpu()),
                                "ga_sensitive_logprob": float(ga.detach().cpu()),
                                "reference_gd_ce": float(reference_gd.detach().cpu()),
                                "gd_non_sensitive_distribution_kl": float(dist_kl.detach().cpu()),
                                "delta_l2": float(l2.detach().cpu()),
                                "delta_norm": float(delta.detach().norm().cpu()),
                                "ga_weight": float(a.ga_weight),
                                "reference_gd_weight": float(a.reference_gd_weight),
                                "distribution_kl_weight": float(a.distribution_kl_weight),
                                "benchmark_retain_seen": 0,
                                "heldout_probes_seen": 0,
                            }
                        )
                        + "\n"
                    )
                    log_f.flush()
    finally:
        hook.remove()
    del opt

    trained_delta = delta_module.effective_delta().detach().clone()
    instances = mcf_instances(records)
    scales = core.parse_scales(a.candidate_scales)
    scale_reports: List[Dict[str, Any]] = []
    for scale in scales:
        handle = core.register_output_delta_hook(
            output_layer,
            selected_ids,
            lambda scale=scale: trained_delta * float(scale),
        )
        try:
            margins = direct_margins(
                model, tok, instances, device, llama_like, a.cache_batch_size
            )
        finally:
            handle.remove()
        scale_reports.append(
            {
                "scale": float(scale),
                "direct_failures": int(
                    (margins < a.stage1_constraint_margin).sum().item()
                ),
                "minimum_margin": float(margins.min().detach().cpu()),
                "effective_delta_norm": float(trained_delta.norm().cpu() * scale),
            }
        )
    selected_scale = core.choose_scale(scale_reports)
    final_delta = trained_delta * float(selected_scale)
    core.materialize_output_delta(output_layer, selected_ids, final_delta)
    final_margins = direct_margins(
        model, tok, instances, device, llama_like, a.cache_batch_size
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "method": "SURE-LM-context-conditioned-stage1-gagd",
        "dataset": "mcf",
        "protocol": "sure_context_target_true_locked_direct_only",
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_semantics": {
            "canonical_sensitive_slot": "target_new",
            "canonical_reference_slot": "target_true",
            "original_sensitive_field": "target_true",
            "original_reference_field": "target_new",
        },
        "sensitive_prediction_cases": len(sensitive_cases),
        "reference_prediction_cases": len(reference_cases),
        "selected_lm_head_rows": len(selected_ids),
        "selected_token_ids": selected_ids,
        "row_specific_context_projection": True,
        "context_rank_cap": int(a.context_rank),
        "row_basis_reports": basis_reports,
        "trainable_parameters": int(delta_module.trainable_parameter_count),
        "input_embeddings_modified": False,
        "input_embeddings_equal_base_by_construction": True,
        "transformer_trainable": 0,
        "lm_head_untied": True,
        "editable_rows": "sensitive_answer_rows_only",
        "reference_answer_rows_editable": False,
        "ga_loss": "mean(log p_theta(sensitive_token)); minimized",
        "reference_gd_loss": "teacher-forced CE on non-sensitive/reference answer; minimized",
        "distribution_gd_loss": "KL(base_non_sensitive || current_non_sensitive) with sensitive token removed",
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "reference_batch_size": int(a.reference_batch_size),
        "emb_lm_lr": float(a.emb_lm_lr),
        "ga_weight": float(a.ga_weight),
        "reference_gd_weight": float(a.reference_gd_weight),
        "distribution_kl_weight": float(a.distribution_kl_weight),
        "delta_l2": float(a.delta_l2),
        "candidate_scales": scales,
        "scale_reports": scale_reports,
        "selected_scale": float(selected_scale),
        "stage1_constraint_margin": float(a.stage1_constraint_margin),
        "direct_failures_after": int(
            (final_margins < a.stage1_constraint_margin).sum().item()
        ),
        "minimum_direct_margin_after": float(final_margins.min().detach().cpu()),
        "effective_delta_norm": float(final_delta.norm().cpu()),
        "benchmark_retain_seen": 0,
        "heldout_paraphrases_seen": 0,
        "neighborhood_prompts_seen": 0,
        "PPL_seen": False,
        "checkpoint": str(ckpt.resolve()),
    }
    core.write_json(out_dir / "config_used.json", config)
    core.write_json(out_dir / "context_bases.json", basis_reports)
    core.write_json(out_dir / "scale_sweep_direct_only.json", scale_reports)

    print("Context-conditioned Stage-1 checkpoint:", ckpt)
    print("selected sensitive rows:", len(selected_ids))
    print("row context ranks:", [x["context_rank"] for x in basis_reports])
    print("selected Stage-1 scale:", selected_scale)
    print(
        "direct failures after Stage 1:",
        config["direct_failures_after"],
        "/",
        len(instances),
    )


if __name__ == "__main__":
    main()
