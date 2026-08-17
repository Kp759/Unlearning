#!/usr/bin/env python3
"""SURE-MQuAKE V7.2: Base-active-row minimal BF16-safe sparse repair.

V7.2 is a utility-focused refinement of V7.1.

Data firewall:
- only direct requested_rewrite facts from the same 50 sampled forget instances;
- no benchmark retain instances, AtomicGen questions, multi-hop questions,
  counterfactual targets, or PPL text are used for optimization/selection.

Parameter firewall:
- transformer blocks remain frozen and exact Base;
- input embeddings remain frozen and exact Base;
- non-selected LM-head rows remain exact Base;
- editable LM-head rows are ONLY target_true token rows belonging to token
  decisions that are active in the untouched protected Base (margin <= 0).

Optimization:
- all visible token decisions are constrained, because an edited shared row can
  become a competitor for another case;
- squared hinge has exactly zero forget gradient once the cached BF16-buffered
  margin is satisfied;
- same-prompt non-target KL and L2 discourage collateral movement;
- once a BF16-exact feasible candidate is found, the update is bisected back
  toward Base using ACTUAL materialized BF16 all-visible audits, yielding the
  smallest located BF16-safe scale along that Base-to-candidate ray.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_active_case_repair as active
import gagd_compare as gagd
import mquake_forget_only_active_repair as locked
import mquake_sure_sparse_lm_gagd_v7 as v7
import mquake_v7_locked_case_compat as compat
import mquake_zero_unlearn_official_eval as mquake


METHOD = "SURE-MQuAKE-v7.2-base-active-rows-minimal-bf16-repair"
PROTOCOL = "mquake_zerounlearn_forget_only_locked_probes"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Protected pretrained Base model")
    p.add_argument("--repair-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--forget-hinge-weight", type=float, default=1.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=0.25)
    p.add_argument("--gd-weight", type=float, default=10.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-3)
    p.add_argument("--target-logit-margin", type=float, default=0.01)
    p.add_argument("--bf16-buffer-margin", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--bf16-bisection-steps", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def priority(metrics: Mapping[str, Any]) -> Tuple[int, int, float, float, float]:
    return (
        int(metrics["official_active_sensitive_token_count"]),
        int(metrics["buffered_margin_unmet_token_count"]),
        -float(metrics["minimum_competitor_minus_sensitive_margin"]),
        float(metrics["same_prompt_non_target_kl"]),
        float(metrics["selected_lm_head_delta_norm"]),
    )


def cached_feasible(
    stacked: Mapping[str, torch.Tensor],
    delta: torch.Tensor,
    required_margin: float,
) -> bool:
    margins = v7.competitor_minus_sensitive_margins(stacked, delta)
    return bool(torch.all(margins >= float(required_margin)).item())


@torch.no_grad()
def exact_materialized_audit(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    *,
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    base_selected_rows: torch.Tensor,
    delta: torch.Tensor,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    target_margin: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    v7.set_selected_rows(output_weight, selected_ids, base_selected_rows, delta)
    return v7.exact_materialized_reports(
        model,
        tok,
        cases,
        device=device,
        llama_like=llama_like,
        batch_size=batch_size,
        target_margin=target_margin,
    )


def exact_pass(summary: Mapping[str, Any]) -> bool:
    return (
        int(summary["official_active_sensitive_token_count"]) == 0
        and int(summary["buffered_margin_unmet_token_count"]) == 0
    )


def main() -> None:
    a = parse_args()
    if a.forget_num <= 0 or a.steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, steps and batch-size must be positive")
    if a.lr <= 0 or a.forget_hinge_weight <= 0 or a.gd_weight < 0:
        raise ValueError("invalid V7.2 learning rate/forget/GD weights")
    if a.hardest_forget_hinge_weight < 0 or a.delta_l2_lambda < 0 or a.grad_clip < 0:
        raise ValueError("invalid V7.2 regularization controls")
    if a.target_logit_margin < 0 or a.bf16_buffer_margin < 0:
        raise ValueError("target/buffer margins must be non-negative")
    if a.bf16_bisection_steps <= 0:
        raise ValueError("bf16-bisection-steps must be positive")

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
    target_ids = v7.resolve_case_target_ids(
        tok, cases, llama_like=llama_like, device=device
    )
    special_ids = gagd.special_token_ids(tok)
    all_sensitive_ids = sorted(set(target_ids) - special_ids)
    if not all_sensitive_ids:
        raise RuntimeError("MQuAKE visible sensitive tokens produced no LM-head rows")
    missing = sorted(set(target_ids) - set(all_sensitive_ids))
    if missing:
        raise RuntimeError(
            f"official sensitive target tokens unexpectedly include special ids: {missing}"
        )

    # Pass 1: untouched Base geometry using all sensitive rows only so we can
    # reconstruct every exact competitor-minus-sensitive margin analytically.
    all_caches = v7.build_token_delta_caches(
        model,
        tok,
        cases,
        all_sensitive_ids,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        desc="cache MQuAKE V7.2 Base activity",
    )
    all_stacked = v7.stack_cache_fields(all_caches, device=device)
    all_zero = torch.zeros(
        (len(all_sensitive_ids), int(output_weight.shape[1])),
        dtype=torch.float32,
        device=device,
    )
    base_margins = v7.competitor_minus_sensitive_margins(all_stacked, all_zero)
    base_active_positions = [
        idx for idx, margin in enumerate(base_margins.detach().cpu().tolist())
        if float(margin) <= 0.0
    ]
    if not base_active_positions:
        raise RuntimeError("protected Base has no active sensitive token decisions to repair")

    selected_ids = sorted({int(target_ids[idx]) for idx in base_active_positions})
    if not selected_ids:
        raise RuntimeError("Base-active cases produced no editable target rows")

    selected_tensor = torch.tensor(
        selected_ids, dtype=torch.long, device=output_weight.device
    )
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    # Pass 2: cache ALL visible cases against only the Base-active target rows.
    # This preserves the parameter firewall while still detecting collateral
    # effects where an edited row becomes a competitor elsewhere.
    caches = v7.build_token_delta_caches(
        model,
        tok,
        cases,
        selected_ids,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        desc="cache MQuAKE V7.2 all-visible constraints",
    )
    stacked = v7.stack_cache_fields(caches, device=device)
    zero = torch.zeros(
        (len(selected_ids), int(output_weight.shape[1])),
        dtype=torch.float32,
        device=device,
    )
    restricted_base_margins = v7.competitor_minus_sensitive_margins(stacked, zero)
    if not torch.allclose(base_margins, restricted_base_margins, atol=1e-5, rtol=0.0):
        max_diff = float((base_margins - restricted_base_margins).abs().max().detach().cpu())
        raise RuntimeError(f"restricted-row Base margin reconstruction mismatch: {max_diff}")

    base_kl = float(v7.same_prompt_non_target_kl(stacked, zero).detach().cpu())
    base_metrics = v7.metrics_from_delta(
        stacked,
        zero,
        target_margin=a.target_logit_margin,
        kl_value=base_kl,
    )
    required_margin = float(a.target_logit_margin + a.bf16_buffer_margin)

    module = active.SelectedRowDelta(
        len(selected_ids),
        int(output_weight.shape[1]),
        direction_basis=None,
        retained_basis=None,
        device=device,
    )
    optimizer = torch.optim.AdamW(module.parameters(), lr=a.lr, weight_decay=0.0)

    best_candidate = zero.detach().clone()
    best_metrics = dict(base_metrics)
    feasible_candidate: torch.Tensor | None = None
    feasible_step: int | None = None
    feasible_exact_summary: Dict[str, Any] | None = None
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        margins = v7.competitor_minus_sensitive_margins(stacked, delta)
        errors = torch.relu(required_margin - margins)
        gd_kl = v7.same_prompt_non_target_kl(stacked, delta)
        delta_l2 = delta.square().sum()
        loss = (
            a.forget_hinge_weight * errors.square().mean()
            + a.hardest_forget_hinge_weight * errors.square().max()
            + a.gd_weight * gd_kl
            + a.delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite V7.2 loss at step {step}")
        loss.backward()
        grad_norm_value = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), a.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite V7.2 gradient norm at step {step}")
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_kl = float(
                v7.same_prompt_non_target_kl(stacked, candidate).detach().cpu()
            )
            candidate_metrics = v7.metrics_from_delta(
                stacked,
                candidate,
                target_margin=required_margin,
                kl_value=candidate_kl,
            )
            if priority(candidate_metrics) < priority(best_metrics):
                best_candidate = candidate.detach().clone()
                best_metrics = dict(candidate_metrics)

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "mean_squared_hinge": float(errors.square().mean().detach().cpu()),
                "hardest_squared_hinge": float(errors.square().max().detach().cpu()),
                "gd_same_prompt_non_target_kl": float(gd_kl.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
                **candidate_metrics,
            }
            logs.append(row)
            print(
                f"v72-step={step} active={candidate_metrics['official_active_sensitive_token_count']} "
                f"required_unmet={candidate_metrics['buffered_margin_unmet_token_count']} "
                f"min_margin={candidate_metrics['minimum_competitor_minus_sensitive_margin']:.6g} "
                f"KL={candidate_kl:.6g} norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
            )

        if cached_feasible(stacked, candidate, required_margin):
            # Fail closed on the real BF16 model, not only the FP32 cache.
            exact_reports, exact_summary = exact_materialized_audit(
                model,
                tok,
                cases,
                output_weight=output_weight,
                selected_ids=selected_ids,
                base_selected_rows=base_selected_rows,
                delta=candidate,
                device=device,
                llama_like=llama_like,
                batch_size=a.batch_size,
                target_margin=a.target_logit_margin,
            )
            if exact_pass(exact_summary):
                feasible_candidate = candidate.detach().clone()
                feasible_step = step
                feasible_exact_summary = dict(exact_summary)
                break
            # Restore exact Base rows before continuing cached optimization.
            v7.set_selected_rows(output_weight, selected_ids, base_selected_rows, zero)

    del optimizer
    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)
    v7.write_jsonl(root / "train_log.jsonl", logs)

    if feasible_candidate is None or feasible_step is None or feasible_exact_summary is None:
        v7.set_selected_rows(output_weight, selected_ids, base_selected_rows, zero)
        write_json(
            root / "failure.json",
            {
                "status": "FAILED_NO_BF16_FEASIBLE_BASE_ACTIVE_ROW_REPAIR",
                "method": METHOD,
                "seed": int(a.seed),
                "base_active_token_count": len(base_active_positions),
                "selected_base_active_target_row_count": len(selected_ids),
                "required_cached_margin": required_margin,
                "exact_target_margin": float(a.target_logit_margin),
                "best_cached_metrics": best_metrics,
                "best_candidate_norm": float(best_candidate.norm().detach().cpu()),
            },
        )
        raise RuntimeError("V7.2 did not find a BF16-feasible Base-active-row repair")

    # BF16-exact shrink-back along the Base -> feasible candidate ray.
    # Invariant: low is known infeasible, high is known feasible.
    low = 0.0
    high = 1.0
    bisection_log: List[Dict[str, Any]] = []

    zero_reports, zero_summary = exact_materialized_audit(
        model,
        tok,
        cases,
        output_weight=output_weight,
        selected_ids=selected_ids,
        base_selected_rows=base_selected_rows,
        delta=zero,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        target_margin=a.target_logit_margin,
    )
    if exact_pass(zero_summary):
        low = high = 0.0
    else:
        # Re-establish the known feasible high endpoint after auditing zero.
        high_reports, high_summary = exact_materialized_audit(
            model,
            tok,
            cases,
            output_weight=output_weight,
            selected_ids=selected_ids,
            base_selected_rows=base_selected_rows,
            delta=feasible_candidate,
            device=device,
            llama_like=llama_like,
            batch_size=a.batch_size,
            target_margin=a.target_logit_margin,
        )
        if not exact_pass(high_summary):
            raise RuntimeError("V7.2 feasible candidate stopped passing BF16 audit")

        for iteration in range(1, a.bf16_bisection_steps + 1):
            mid = 0.5 * (low + high)
            mid_delta = feasible_candidate * float(mid)
            _, mid_summary = exact_materialized_audit(
                model,
                tok,
                cases,
                output_weight=output_weight,
                selected_ids=selected_ids,
                base_selected_rows=base_selected_rows,
                delta=mid_delta,
                device=device,
                llama_like=llama_like,
                batch_size=a.batch_size,
                target_margin=a.target_logit_margin,
            )
            passed = exact_pass(mid_summary)
            bisection_log.append(
                {
                    "iteration": iteration,
                    "alpha": mid,
                    "passed": passed,
                    "official_active_sensitive_token_count": int(
                        mid_summary["official_active_sensitive_token_count"]
                    ),
                    "target_margin_unmet_token_count": int(
                        mid_summary["buffered_margin_unmet_token_count"]
                    ),
                    "minimum_competitor_minus_sensitive_margin": float(
                        mid_summary["minimum_competitor_minus_sensitive_margin"]
                    ),
                    "scaled_delta_norm": float((mid_delta).norm().detach().cpu()),
                }
            )
            if passed:
                high = mid
            else:
                low = mid

    final_alpha = float(high)
    final_delta = feasible_candidate * final_alpha
    final_reports, final_summary = exact_materialized_audit(
        model,
        tok,
        cases,
        output_weight=output_weight,
        selected_ids=selected_ids,
        base_selected_rows=base_selected_rows,
        delta=final_delta,
        device=device,
        llama_like=llama_like,
        batch_size=a.batch_size,
        target_margin=a.target_logit_margin,
    )
    if not exact_pass(final_summary):
        raise RuntimeError("V7.2 final BF16 shrink-back checkpoint failed exact audit")

    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V7.2 modified input embeddings")

    v7.write_jsonl(root / "all_visible_tokens_after_bf16.jsonl", final_reports)
    v7.write_jsonl(root / "bf16_bisection_log.jsonl", bisection_log)
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    selected_tokens = {str(t): v7.decoded_token(tok, t) for t in selected_ids}
    write_json(
        root / "selected_base_active_lm_rows.json",
        {
            "definition": "target_true token rows belonging only to margin<=0 token decisions in untouched protected Base",
            "base_active_token_case_count": len(base_active_positions),
            "base_active_token_positions": base_active_positions,
            "selected_lm_head_row_count": len(selected_ids),
            "selected_lm_head_token_ids": selected_ids,
            "selected_lm_head_tokens": selected_tokens,
            "all_sensitive_row_count_for_audit_only": len(all_sensitive_ids),
            "non_selected_lm_rows_exact_base_by_construction": True,
            "input_embeddings_exact_base_by_construction": True,
            "transformer_frozen": True,
        },
    )

    summary = {
        "status": "PASS_V72_BASE_ACTIVE_ROWS_MINIMAL_BF16",
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(a.seed),
        "forget_instances": int(a.forget_num),
        "forget_atomic_facts": len(records),
        "visible_sensitive_token_cases": len(cases),
        "all_sensitive_target_row_count": len(all_sensitive_ids),
        "base_active_sensitive_token_case_count": len(base_active_positions),
        "selected_base_active_target_row_count": len(selected_ids),
        "optimizer_feasible_step": feasible_step,
        "optimizer_feasible_delta_norm": float(feasible_candidate.norm().detach().cpu()),
        "bf16_minimal_ray_alpha": final_alpha,
        "selected_lm_head_delta_norm": float(final_delta.norm().detach().cpu()),
        "cached_required_margin": required_margin,
        "exact_target_margin": float(a.target_logit_margin),
        "first_feasible_exact_metrics": feasible_exact_summary,
        "materialized_bf16_metrics": final_summary,
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
    write_json(
        root / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            **vars(a),
            "repair_visible_path_resolved": str(visible_path),
            "split_manifest_resolved": str(manifest_path),
            "split_sampling": split_manifest.get("sampling"),
            "parameter_scope": "Base-active target_true LM-head rows only; transformer/input embeddings/all other LM rows exact Base",
            "base_activity_definition": "competitor_minus_sensitive_margin <= 0 on untouched Base",
            "forget_definition": "squared hinge relu(target_margin + bf16_buffer - competitor_minus_sensitive_margin)^2 over all visible token cases",
            "gd_definition": "same-prompt KL(Base_non-target || Current_non-target), sensitive target removed and renormalized",
            "selection_definition": "first BF16-exact feasible candidate, then actual BF16 all-visible bisection toward Base; no retain/PPL/AtomicGen/multihop selection",
            "checkpoint": str(ckpt.resolve()),
        },
    )

    print("===== SURE-MQuAKE V7.2 BASE-ACTIVE-ROW MINIMAL BF16 REPAIR =====")
    print(
        f"instances={a.forget_num} atomic_facts={len(records)} token_cases={len(cases)} "
        f"all_sensitive_rows={len(all_sensitive_ids)} base_active_tokens={len(base_active_positions)} "
        f"editable_rows={len(selected_ids)}"
    )
    print(
        f"optimizer_feasible_step={feasible_step} candidate_norm={float(feasible_candidate.norm().detach().cpu()):.6g} "
        f"final_alpha={final_alpha:.8f} final_norm={float(final_delta.norm().detach().cpu()):.6g}"
    )
    print(
        f"BF16 official_active={final_summary['official_active_sensitive_token_count']} "
        f"margin_unmet={final_summary['buffered_margin_unmet_token_count']} "
        f"min_margin={final_summary['minimum_competitor_minus_sensitive_margin']:.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
