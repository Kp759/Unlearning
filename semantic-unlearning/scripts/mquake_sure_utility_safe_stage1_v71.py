#!/usr/bin/env python3
"""SURE-MQuAKE V7.1 Stage 1: utility-safe sparse sensitive-row forgetting.

This stage keeps the V7 locked data and parameter firewall but replaces the
unbounded sensitive-token GA term with a saturating native-MQuAKE margin loss.
Once a sensitive token safely loses to a competitor, its forget gradient rapidly
vanishes. Same-prompt non-target KL remains the Base-preservation term.

No benchmark retain, AtomicGen, multi-hop, counterfactual target, or PPL data is
loaded for optimization or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_forget_only_active_repair as locked
import mquake_sure_sparse_lm_gagd_v7 as v7
import mquake_v7_locked_case_compat as compat
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v7.1-utility-safe-bounded-margin-GD"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--hardest-forget-weight", type=float, default=0.05)
    p.add_argument("--gd-weight", type=float, default=10.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-4)
    p.add_argument("--forget-temperature", type=float, default=2.0)
    p.add_argument("--target-logit-margin", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--stop-when-all-satisfied", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def bounded_forget_terms(margins: torch.Tensor, target_margin: float, temperature: float) -> torch.Tensor:
    # d/dmargin is bounded in magnitude and tends to zero after the margin is
    # safely satisfied, unlike the old -NLL GA objective which keeps pushing.
    return F.softplus((float(target_margin) - margins) / float(temperature)) * float(temperature)


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, steps and batch-size must be positive")
    if a.lr <= 0 or a.forget_weight <= 0 or a.gd_weight < 0:
        raise ValueError("invalid learning rate/forget/GD weights")
    if a.hardest_forget_weight < 0 or a.delta_l2_lambda < 0 or a.grad_clip < 0:
        raise ValueError("invalid regularization controls")
    if a.forget_temperature <= 0 or a.target_logit_margin < 0:
        raise ValueError("temperature must be positive and target margin non-negative")

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
    sensitive_ids = sorted(set(target_ids) - special_ids)
    if not sensitive_ids:
        raise RuntimeError("no sensitive LM-head rows")
    if set(target_ids) - set(sensitive_ids):
        raise RuntimeError("official sensitive target unexpectedly includes special token ids")

    selected_tensor = torch.tensor(sensitive_ids, dtype=torch.long, device=output_weight.device)
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()
    caches = v7.build_token_delta_caches(
        model, tok, cases, sensitive_ids,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        desc="cache MQuAKE V7.1 utility-safe Stage1",
    )
    stacked = v7.stack_cache_fields(caches, device=device)

    zero = torch.zeros(
        (len(sensitive_ids), int(output_weight.shape[1])),
        dtype=torch.float32, device=device,
    )
    base_kl = float(v7.same_prompt_non_target_kl(stacked, zero).detach().cpu())
    base_metrics = v7.metrics_from_delta(
        stacked, zero, target_margin=a.target_logit_margin, kl_value=base_kl
    )

    module = active.SelectedRowDelta(
        len(sensitive_ids), int(output_weight.shape[1]),
        direction_basis=None, retained_basis=None, device=device,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=a.lr, weight_decay=0.0)

    best_delta = zero.detach().clone()
    best_metrics = dict(base_metrics)
    base_margins = v7.competitor_minus_sensitive_margins(stacked, zero)
    base_terms = bounded_forget_terms(base_margins, a.target_logit_margin, a.forget_temperature)
    best_objective = float(
        (a.forget_weight * base_terms.mean() + a.hardest_forget_weight * base_terms.max()).detach().cpu()
    )
    best_step = 0
    stopped_early = False
    logs: List[Dict[str, Any]] = []
    steps_completed = 0

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        margins = v7.competitor_minus_sensitive_margins(stacked, delta)
        forget_terms = bounded_forget_terms(margins, a.target_logit_margin, a.forget_temperature)
        gd_kl = v7.same_prompt_non_target_kl(stacked, delta)
        delta_l2 = delta.square().sum()
        loss = (
            a.forget_weight * forget_terms.mean()
            + a.hardest_forget_weight * forget_terms.max()
            + a.gd_weight * gd_kl
            + a.delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V7.1 Stage1 loss at step {step}")
        loss.backward()
        grad_norm_value = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), a.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite V7.1 gradient norm at step {step}")
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()
        steps_completed = step

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_margins = v7.competitor_minus_sensitive_margins(stacked, candidate)
            candidate_terms = bounded_forget_terms(candidate_margins, a.target_logit_margin, a.forget_temperature)
            candidate_kl_t = v7.same_prompt_non_target_kl(stacked, candidate)
            candidate_kl = float(candidate_kl_t.detach().cpu())
            candidate_l2 = candidate.square().sum()
            candidate_objective_t = (
                a.forget_weight * candidate_terms.mean()
                + a.hardest_forget_weight * candidate_terms.max()
                + a.gd_weight * candidate_kl_t
                + a.delta_l2_lambda * candidate_l2
            )
            candidate_objective = float(candidate_objective_t.detach().cpu())
            candidate_metrics = v7.metrics_from_delta(
                stacked, candidate, target_margin=a.target_logit_margin, kl_value=candidate_kl
            )
            if candidate_objective < best_objective:
                best_objective = candidate_objective
                best_delta = candidate.detach().clone()
                best_metrics = dict(candidate_metrics)
                best_step = step

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "bounded_forget_mean": float(forget_terms.mean().detach().cpu()),
                "bounded_forget_hardest": float(forget_terms.max().detach().cpu()),
                "gd_same_prompt_non_target_kl": float(gd_kl.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
                "selection_objective": candidate_objective,
                **candidate_metrics,
            }
            logs.append(row)
            print(
                f"v71-step={step} active={candidate_metrics['official_active_sensitive_token_count']} "
                f"margin_unmet={candidate_metrics['buffered_margin_unmet_token_count']} "
                f"min_margin={candidate_metrics['minimum_competitor_minus_sensitive_margin']:.6g} "
                f"KL={candidate_kl:.6g} norm={candidate_metrics['selected_lm_head_delta_norm']:.6g} "
                f"obj={candidate_objective:.6g}"
            )

        if a.stop_when_all_satisfied and candidate_metrics["buffered_margin_unmet_token_count"] == 0:
            if candidate_objective <= best_objective + 1e-12:
                best_delta = candidate.detach().clone()
                best_metrics = dict(candidate_metrics)
                best_objective = candidate_objective
                best_step = step
            stopped_early = True
            break

    del optimizer
    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    v7.write_jsonl(root / "train_log.jsonl", logs)

    v7.set_selected_rows(output_weight, sensitive_ids, base_selected_rows, best_delta)
    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V7.1 Stage1 modified input embeddings")

    exact_reports, exact_summary = v7.exact_materialized_reports(
        model, tok, cases,
        device=device, llama_like=llama_like, batch_size=a.batch_size,
        target_margin=a.target_logit_margin,
    )
    v7.write_jsonl(root / "all_visible_tokens_after_bf16.jsonl", exact_reports)

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    write_json(root / "sensitive_lm_rows.json", {
        "sensitive_row_definition": "union of official target_true token ids on locked direct requested_rewrite positions",
        "sensitive_row_count": len(sensitive_ids),
        "sensitive_token_ids": sensitive_ids,
        "sensitive_tokens": {str(t): v7.decoded_token(tok, t) for t in sensitive_ids},
        "non_sensitive_lm_rows_exact_base_by_construction": True,
        "input_embeddings_exact_base_by_construction": True,
        "transformer_frozen": True,
    })

    summary = {
        "status": "PASS_STAGE1_UTILITY_SAFE",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "best_step": best_step,
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "selection_objective": best_objective,
        "sensitive_lm_head_row_count": len(sensitive_ids),
        "selected_lm_head_delta_norm": float(best_delta.norm().detach().cpu()),
        "cached_metrics": best_metrics,
        "materialized_bf16_metrics": exact_summary,
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
        "parameter_scope": "sparse sensitive target_true LM-head rows only; transformer/input embeddings/non-sensitive rows exact Base",
        "forget_definition": "saturating softplus of target_margin - native competitor_minus_sensitive margin; gradients vanish after safe forgetting",
        "gd_definition": "same-prompt KL(Base_non-target || Current_non-target), sensitive target removed and renormalized",
        "selection_definition": "minimum forget+GD+L2 objective using forget-visible data only",
        "checkpoint": str(ckpt.resolve()),
    })

    print("===== SURE-MQuAKE V7.1 UTILITY-SAFE STAGE1 =====")
    print(
        f"instances={a.forget_num} atomic_facts={len(records)} token_cases={len(cases)} "
        f"sensitive_rows={len(sensitive_ids)} best_step={best_step} obj={best_objective:.6g}"
    )
    print(
        f"BF16 official_active={exact_summary['official_active_sensitive_token_count']} "
        f"margin_unmet={exact_summary['buffered_margin_unmet_token_count']} "
        f"min_margin={exact_summary['minimum_competitor_minus_sensitive_margin']:.6g} "
        f"delta_norm={float(best_delta.norm().detach().cpu()):.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
