#!/usr/bin/env python3
"""Leakage-controlled MCF Stage-2 repair with row-specific forget-context bases.

This script supports two closely related row-conditioned Stage-2 experiments.
Both reuse the canonical target-true-sensitive r8/m1 Stage-1 checkpoint, direct
forget records, pairwise margin objective, optimizer, hinge weight, L2 weight,
margin target, and evaluation protocol.

Canonical/global repair:
    every selected output row shares one rank-r basis B_F built from all active
    direct forget answer-token hidden states.

Row-conditioned repair:
    each selected row s gets its own basis B_{F,s}, built only from active direct
    forget hidden states where token s is the teacher-forced target for the
    corresponding answer field.  Therefore

        delta_w_s = a_s B_{F,s}.

``--row-scope sensitive_only`` (default) is the main SURE-specific architecture:
it edits only training target_new rows.  In the locked target-true-sensitive MCF
view, training target_new is ORIGINAL MCF target_true, i.e. the sensitive answer.
Relative to the old global baseline, this changes both row scope and geometry.

``--row-scope all_answer_rows`` preserves the old target_new+target_true editable
row scope and changes only global-basis -> row-specific-basis geometry.  It is the
clean geometry-only control.

No MCF retain examples, official paraphrases, neighborhoods, generation prompts,
or external utility examples are used during repair.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import gagd_active_case_repair as repair
import gagd_compare as gagd
import sure_context_projection as context
from mcf_zero_unlearn_official_eval import is_llama_like

METHOD = "mcf_forget_only_rowspecific_repair"
PROTOCOL = "target_true_sensitive_rowspecific_forget_only_locked_probes"


def build_parser() -> argparse.ArgumentParser:
    p = repair.build_parser()
    p.description = __doc__
    p.add_argument(
        "--row-scope",
        choices=("sensitive_only", "all_answer_rows"),
        default="sensitive_only",
        help=(
            "sensitive_only edits training target_new rows only (ORIGINAL target_true); "
            "all_answer_rows matches the canonical target_new+target_true row scope."
        ),
    )
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.forget_num <= 0:
        raise ValueError("--forget-num must be positive")
    if args.retain_num != 0:
        raise ValueError("Row-specific forget-only repair requires --retain-num 0")
    if args.repair_mode != "minimal_optimize":
        raise ValueError("Row-specific repair supports only minimal_optimize")
    if args.retain_kl_mu != 0 or args.retain_calibration_num != 0:
        raise ValueError("Row-specific ablation uses no benchmark retain KL/calibration")
    if args.project_away_retain_hidden:
        raise ValueError("Row-specific ablation does not project against retain hidden states")
    if args.reference_model_path is not None:
        raise ValueError("Row-specific ablation does not use a reference model")
    if args.run_official_mcf_eval:
        raise ValueError("Evaluate only after the frozen repaired checkpoint is saved")
    if not math.isfinite(args.active_margin) or args.active_margin < 0:
        raise ValueError("--active-margin must be finite and non-negative")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("repair steps/lr must be positive")
    if args.hinge_weight <= 0 or args.delta_l2_lambda < 0:
        raise ValueError("invalid repair regularization weights")
    if args.repair_rank <= 0:
        raise ValueError("Row-specific experiment requires positive --repair-rank")
    if args.margin_batch_size <= 0:
        raise ValueError("--margin-batch-size must be positive")


def _selected_sensitive_rows(
    tok: Any,
    active_instances: Sequence[repair.MCFPromptInstance],
) -> List[int]:
    selected: set[int] = set()
    for instance in active_instances:
        selected.update(
            gagd.token_ids_for_text(tok, gagd.normalize_answer(instance.target_new))
        )
    selected -= gagd.special_token_ids(tok)
    return sorted(selected)


def _selected_rows(
    tok: Any,
    active_instances: Sequence[repair.MCFPromptInstance],
    groups: gagd.PostTrainingTokenGroups,
    row_scope: str,
) -> List[int]:
    if row_scope == "sensitive_only":
        return _selected_sensitive_rows(tok, active_instances)
    return repair.selected_rows_for_active_instances(
        tok, active_instances, groups, "minimal_optimize"
    )


def build_row_specific_bases_from_caches(
    caches: Sequence[repair.RewriteDeltaCache],
    active_positions: Sequence[int],
    selected_ids: Sequence[int],
    *,
    row_scope: str,
    max_rank: int,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Build B_{F,s} only from active contexts where row s is a target token."""
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for selected_column, token_id in enumerate(selected_ids):
        row_hidden: List[torch.Tensor] = []
        source_counts = {"sensitive_target_new": 0, "reference_target_true": 0}
        for position in active_positions:
            cache = caches[position]
            answer_sources = [("sensitive_target_new", cache.target_new)]
            if row_scope == "all_answer_rows":
                answer_sources.append(("reference_target_true", cache.target_true))
            for source_name, answer_cache in answer_sources:
                mask = answer_cache.target_selected_columns.eq(selected_column)
                if bool(mask.any().item()):
                    rows = answer_cache.hidden[mask]
                    row_hidden.append(rows)
                    source_counts[source_name] += int(rows.shape[0])
        if not row_hidden:
            raise RuntimeError(
                f"Selected row token_id={token_id} has no matching direct forget target contexts"
            )
        hidden = torch.cat(row_hidden, dim=0).float()
        basis = repair.orthonormal_row_basis(hidden, max_rank=max_rank)
        if basis.ndim != 2 or basis.shape[0] <= 0:
            raise RuntimeError(
                f"Selected row token_id={token_id} has zero numerical context rank"
            )
        bases.append(basis.detach().float().contiguous())
        reports.append(
            {
                "token_id": int(token_id),
                "context_count": int(hidden.shape[0]),
                "context_rank": int(basis.shape[0]),
                "rank_cap": int(max_rank),
                "source_counts": source_counts,
            }
        )
    return bases, reports


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path, source_config, preserved_alphas = repair.recover_experiment_config(
        args.model_path, args.experiment_config_path
    )
    repair.validate_source_experiment_config(source_config, args)
    config_used: Dict[str, Any] = {
        **vars(args),
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_experiment_config_path": str(config_path),
        "source_experiment_config": source_config,
        "preserved_5e_overlap_alphas": preserved_alphas,
        "repair_uses_official_paraphrases": False,
        "repair_prompt_scope": "requested_rewrite_only",
        "evaluation_probes_locked_during_repair": True,
        "benchmark_retain_examples_used_during_repair": 0,
        "external_utility_examples_used_during_repair": 0,
        "original_sensitive_field": "target_true",
        "counterfactual_reference_field": "target_new",
        "training_sensitive_field": "target_new",
        "repair_geometry": "row_specific_direct_forget_target_context_basis",
        "comparison_note": (
            "sensitive_only changes editable row scope and basis geometry relative to the old global baseline; "
            "all_answer_rows is the geometry-only control"
        ),
    }
    gagd.write_json(output_dir / "config_used.json", config_used)

    print("Loading locked target-true-sensitive MCF forget records")
    forget_records, retain_records = repair.load_sampled_mcf_records(args)
    if retain_records:
        raise RuntimeError("Forget-only row-specific repair unexpectedly sampled retain records")
    forget_examples = [record.example for record in forget_records]
    forget_prompt_instances = repair.expand_prompt_instances(forget_records)
    if any(instance.prompt_type != "rewrite" for instance in forget_prompt_instances):
        raise RuntimeError("Locked repair-visible MCF unexpectedly exposes held-out paraphrases")

    print(f"Loading Stage-1 checkpoint: {args.model_path}")
    model, tok = gagd.load_model_and_tokenizer(
        repair._model_loading_args(args), for_training=False
    )
    output_embeddings = repair.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    groups = gagd.collect_post_training_token_groups(tok, forget_examples, [])
    before_reports = repair.evaluate_prompt_instance_margin_reports(
        model, tok, forget_prompt_instances, groups, args.active_margin,
        device, args.margin_batch_size, llama_like
    )
    original_margin_values = [float(x["margin"]) for x in before_reports]
    original_margin_tensor = torch.tensor(
        original_margin_values, dtype=torch.float32, device=output_weight.device
    )
    required_margins = repair.build_required_margin_tensor(
        original_margin_tensor,
        active_margin=args.active_margin,
        protected_margin_floor=args.protected_margin_floor,
    )
    repair.attach_margin_requirement_metadata(
        before_reports, original_margin_values, required_margins, args.active_margin
    )
    active_positions = repair.select_active_positions(before_reports, args.active_margin)
    active_instances = repair._active_instances(forget_prompt_instances, active_positions)
    before_active_payload = repair.active_report_payload(before_reports, args.active_margin)
    selected_ids = _selected_rows(tok, active_instances, groups, args.row_scope)

    gagd.write_json(output_dir / "rewrite_margins_before.json", before_reports)
    gagd.write_json(output_dir / "active_cases_before.json", before_active_payload)

    print(
        f"Active direct cases before repair: {len(active_positions)}/{len(forget_prompt_instances)}; "
        f"row_scope={args.row_scope}; selected rows={len(selected_ids)}"
    )

    input_storage_pointer = model.get_input_embeddings().weight.detach().data_ptr()
    selected_before = (
        output_weight.index_select(
            0,
            torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device),
        ).detach().clone()
        if selected_ids else output_weight.new_empty((0, output_weight.shape[1]))
    )

    row_basis_reports: List[Dict[str, Any]] = []
    repair_logs: List[Dict[str, Any]] = []
    optimization_summary: Dict[str, Any] = {
        "steps_completed": 0,
        "stopped_early": False,
        "all_satisfied": repair.all_margins_satisfied(original_margin_tensor, required_margins),
        "training_prompt_instances": len(forget_prompt_instances),
    }

    if selected_ids:
        print("Caching exact direct sparse-delta pairwise objectives")
        prompt_caches = repair.build_prompt_instance_delta_caches(
            model, tok, forget_prompt_instances, selected_ids, device,
            args.margin_batch_size, llama_like
        )
        bases, row_basis_reports = build_row_specific_bases_from_caches(
            prompt_caches,
            active_positions,
            selected_ids,
            row_scope=args.row_scope,
            max_rank=args.repair_rank,
        )
        row_ranks = [int(x["context_rank"]) for x in row_basis_reports]
        print(
            "Using row-specific forget-context repair: "
            f"rows={len(selected_ids)}, rank_cap={args.repair_rank}, row_ranks={row_ranks}"
        )

        delta_module = context.RowSpecificProjectedDelta(
            selected_ids, bases, device=output_weight.device
        )
        margin_fn = lambda delta: repair.margins_from_delta_caches(prompt_caches, delta)
        zero_kl_fn = lambda delta: delta.new_zeros(())
        repair_logs, optimization_summary = repair.optimize_selected_delta(
            delta_module,
            margin_fn,
            zero_kl_fn,
            required_margins=required_margins,
            repair_steps=args.repair_steps,
            repair_lr=args.repair_lr,
            repair_optimizer=args.repair_optimizer,
            hinge_weight=args.hinge_weight,
            delta_l2_lambda=args.delta_l2_lambda,
            retain_kl_mu=0.0,
            stop_when_all_satisfied=args.stop_when_all_satisfied,
            max_delta_norm=args.max_delta_norm,
        )
        with torch.no_grad():
            repair.materialize_selected_delta(
                output_weight, selected_ids, delta_module.effective_delta()
            )
        del delta_module, prompt_caches
        gc.collect()

    repair.write_jsonl(output_dir / "repair_log.jsonl", repair_logs)

    if model.get_input_embeddings().weight.detach().data_ptr() != input_storage_pointer:
        raise RuntimeError("Input embedding storage changed during Stage-2 repair")
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("Input embeddings unexpectedly became trainable")

    after_reports = repair.evaluate_prompt_instance_margin_reports(
        model, tok, forget_prompt_instances, groups, args.active_margin,
        device, args.margin_batch_size, llama_like
    )
    repair.attach_margin_requirement_metadata(
        after_reports, original_margin_values, required_margins, args.active_margin
    )
    transitions = repair.prompt_margin_transitions(
        before_reports, after_reports, args.active_margin
    )
    after_active_payload = repair.active_report_payload(after_reports, args.active_margin)
    gagd.write_json(output_dir / "rewrite_margins_after.json", after_reports)
    gagd.write_json(output_dir / "active_cases_after.json", after_active_payload)

    selected_after = (
        output_weight.index_select(
            0,
            torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device),
        ).detach().clone()
        if selected_ids else selected_before
    )
    selected_delta = selected_after.float() - selected_before.float()
    row_ranks = [int(x["context_rank"]) for x in row_basis_reports]
    actual_max_rank = max(row_ranks, default=0)
    actual_mean_rank = sum(row_ranks) / len(row_ranks) if row_ranks else 0.0
    true_rome_failures = [r for r in after_reports if float(r["margin"]) <= 0.0]
    repair_target_slips = [r for r in after_reports if float(r["margin"]) < args.active_margin]

    geometry_report = {
        "row_scope": args.row_scope,
        "rank_cap": int(args.repair_rank),
        "selected_lm_head_token_ids": [int(x) for x in selected_ids],
        "selected_lm_head_tokens": {
            str(token_id): tok.decode([int(token_id)]) for token_id in selected_ids
        },
        "row_basis_reports": row_basis_reports,
        "row_context_ranks": row_ranks,
        "max_row_context_rank": int(actual_max_rank),
        "mean_row_context_rank": float(actual_mean_rank),
    }
    gagd.write_json(output_dir / "row_specific_geometry.json", geometry_report)

    repair_summary = {
        "method": METHOD,
        "protocol_status": PROTOCOL,
        "protocol_status_reason": (
            "Stage 2 used only direct requested_rewrite forget records in the locked "
            "target-true-sensitive training view. Each selected output row was "
            "restricted to hidden contexts where that token was a teacher-forced target."
        ),
        "repair_mode": "minimal_optimize_rowspecific",
        "repair_geometry": "row_specific_direct_forget_target_context_basis",
        "row_scope": args.row_scope,
        "comparison_note": config_used["comparison_note"],
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "source_experiment_config_path": str(config_path),
        "preserved_5e_overlap_alphas": preserved_alphas,
        "forget_records": len(forget_records),
        "forget_prompt_instances": len(forget_prompt_instances),
        "retain_records": 0,
        "benchmark_retain_examples_used_during_repair": 0,
        "external_utility_examples_used_during_repair": 0,
        "repair_uses_official_paraphrases": False,
        "repair_prompt_scope": "requested_rewrite_only",
        "evaluation_probes_locked_during_repair": True,
        "original_sensitive_field": "target_true",
        "counterfactual_reference_field": "target_new",
        "training_sensitive_field": "target_new",
        "active_margin": float(args.active_margin),
        "active_cases_before": len(active_positions),
        "active_cases_after": int(after_active_payload["count"]),
        "selected_lm_head_rows": len(selected_ids),
        "selected_lm_head_token_ids": [int(x) for x in selected_ids],
        "changed_selected_lm_head_rows": (
            int(selected_delta.norm(dim=1).gt(0).sum().item()) if selected_ids else 0
        ),
        "selected_lm_head_delta_norm": float(selected_delta.norm().cpu()),
        "input_embeddings_modified": False,
        "transformer_parameters_trainable": 0,
        "repair_rank_requested": int(args.repair_rank),
        "repair_rank_actual": int(actual_max_rank),
        "row_context_ranks": row_ranks,
        "mean_row_context_rank": float(actual_mean_rank),
        "row_basis_reports": row_basis_reports,
        "optimization": optimization_summary,
        "minimum_margin_before": min((float(r["margin"]) for r in before_reports), default=None),
        "minimum_margin_after": min((float(r["margin"]) for r in after_reports), default=None),
        "repair_target_slip_count_after": len(repair_target_slips),
        "true_rome_failure_count_after": len(true_rome_failures),
        "newly_below_repair_target_positions": transitions["newly_activated_positions"],
    }
    gagd.write_json(output_dir / "repair_summary.json", repair_summary)

    print(
        "Post-materialization direct diagnostics: "
        f"margin<{args.active_margin}: {len(repair_target_slips)}; "
        f"true ROME margin<=0: {len(true_rome_failures)}; "
        f"minimum margin={repair_summary['minimum_margin_after']}"
    )

    if args.save_model:
        checkpoint_dir = output_dir / "checkpoint"
        repair.save_repair_checkpoint(model, tok, checkpoint_dir, repair_config=config_used)
        print(f"Saved row-specific repaired checkpoint to {checkpoint_dir}")

    print(
        f"Done: active direct cases {len(active_positions)} -> "
        f"{after_active_payload['count']}; outputs in {output_dir}"
    )


if __name__ == "__main__":
    main()
