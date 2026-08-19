#!/usr/bin/env python3
"""MCF Stage-2 sparse LM-head repair with an external utility covariance guard.

This is a leakage-controlled target-true-sensitive Stage-2 variant. It keeps the
existing direct-only sparse repair mechanics, but regularizes the selected
LM-head row delta so that it produces little logit drift on a fixed external
utility corpus.

Protocol:
  * MCF retain records are not used during repair (retain_num must be 0).
  * Official MCF paraphrase/neighborhood/generation probes must be locked in the
    training-visible MCF copy.
  * Utility text is loaded from data/wikidata (or --wikidata-dir), independently
    of MCF. By default the first 20 train texts are excluded because the official
    PPL evaluator uses those texts; 500 texts are sampled from the remainder
    with a fixed utility seed.
  * Transformer and input embeddings are frozen. Only selected lm_head rows move.
  * Rank restriction is still built only from active direct forget hidden states.

The utility objective is

    L_U = (1 / |S|) sum_{r in S} delta_w_r^T C_U delta_w_r,
    C_U = (1 / T) sum_t h_t h_t^T,

which equals the mean squared selected-row logit drift over utility hidden states.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from datasets import load_from_disk

import gagd_active_case_repair as repair
import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like

METHOD = "mcf_forget_only_active_repair_external_utility"


def build_parser() -> argparse.ArgumentParser:
    p = repair.build_parser()
    p.description = __doc__
    p.add_argument("--utility-num", type=int, default=500)
    p.add_argument("--utility-seed", type=int, default=1729)
    p.add_argument(
        "--utility-exclude-first",
        type=int,
        default=20,
        help="Exclude initial Wikidata train texts reserved by the PPL evaluator.",
    )
    p.add_argument("--utility-weight", type=float, default=1.0)
    p.add_argument("--utility-max-length", type=int, default=100)
    p.add_argument("--utility-batch-size", type=int, default=4)
    return p


def validate_args(args: argparse.Namespace) -> None:
    if args.forget_num <= 0:
        raise ValueError("--forget-num must be positive")
    if args.retain_num != 0:
        raise ValueError("External-utility repair requires --retain-num 0")
    if args.repair_mode != "minimal_optimize":
        raise ValueError("External-utility repair supports only --repair-mode minimal_optimize")
    if args.retain_kl_mu != 0:
        raise ValueError("Use --retain-kl-mu 0; utility is handled by the external covariance guard")
    if args.project_away_retain_hidden:
        raise ValueError("Use --no-project-away-retain-hidden for this first utility ablation")
    if args.retain_calibration_num != 0:
        raise ValueError("Use --retain-calibration-num 0; MCF retain examples remain evaluation-only")
    if args.reference_model_path is not None:
        raise ValueError("External-utility repair does not use --reference-model-path")
    if args.run_official_mcf_eval:
        raise ValueError("Evaluate the frozen checkpoint separately; do not expose official probes in repair")
    if not math.isfinite(args.active_margin) or args.active_margin < 0:
        raise ValueError("--active-margin must be finite and non-negative")
    if not math.isfinite(args.protected_margin_floor):
        raise ValueError("--protected-margin-floor must be finite")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("--repair-steps and --repair-lr must be positive")
    if args.hinge_weight <= 0 or args.delta_l2_lambda < 0:
        raise ValueError("invalid repair regularization weights")
    if args.repair_rank <= 0:
        raise ValueError("This utility ablation requires a positive --repair-rank")
    if args.margin_batch_size <= 0:
        raise ValueError("--margin-batch-size must be positive")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("--max-delta-norm must be finite and non-negative")
    if args.utility_num <= 0:
        raise ValueError("--utility-num must be positive")
    if args.utility_exclude_first < 0:
        raise ValueError("--utility-exclude-first must be non-negative")
    if not math.isfinite(args.utility_weight) or args.utility_weight < 0:
        raise ValueError("--utility-weight must be finite and non-negative")
    if args.utility_max_length < 2 or args.utility_batch_size <= 0:
        raise ValueError("utility max length must be >=2 and utility batch size positive")


def load_utility_texts(
    wikidata_dir: str,
    *,
    count: int,
    seed: int,
    exclude_first: int,
) -> Tuple[List[str], List[int], str]:
    path = Path(wikidata_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Utility dataset not found: {path}")
    dataset = load_from_disk(str(path))
    if "train" not in dataset:
        raise ValueError("Utility dataset must contain a train split")
    train = dataset["train"]
    if "text" not in train.column_names:
        raise ValueError("Utility dataset train split must contain a text column")
    texts = train["text"]
    eligible = [
        i
        for i in range(exclude_first, len(texts))
        if isinstance(texts[i], str) and texts[i].strip()
    ]
    if count > len(eligible):
        raise ValueError(
            f"Requested {count} utility texts but only {len(eligible)} eligible texts remain "
            f"after excluding the first {exclude_first}"
        )
    sampled = random.Random(seed).sample(eligible, k=count)
    sampled.sort()
    selected = [texts[i].strip() for i in sampled]
    digest = hashlib.sha256(
        "\n\n<UTILITY_RECORD>\n\n".join(selected).encode("utf-8")
    ).hexdigest()
    return selected, sampled, digest


@torch.no_grad()
def build_utility_second_moment(
    model: torch.nn.Module,
    tok: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Tuple[torch.Tensor, int, float]:
    hidden_size = int(model.get_output_embeddings().weight.shape[1])
    second_moment = torch.zeros(
        (hidden_size, hidden_size), device=device, dtype=torch.float32
    )
    state_count = 0
    model.eval()

    old_padding_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            encoded = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            output = model(
                **encoded,
                output_hidden_states=True,
                use_cache=False,
            )
            hidden = output.hidden_states[-1].float()
            mask = encoded["attention_mask"].bool()
            predictor_mask = mask.clone()
            lengths = mask.sum(dim=1)
            for row, length in enumerate(lengths.detach().cpu().tolist()):
                if length > 0:
                    predictor_mask[row, int(length) - 1] = False
            states = hidden[predictor_mask]
            if states.numel():
                second_moment.add_(states.transpose(0, 1) @ states)
                state_count += int(states.shape[0])
            del output, hidden, states
    finally:
        tok.padding_side = old_padding_side

    if state_count <= 0:
        raise RuntimeError("Utility corpus produced zero predictor hidden states")
    second_moment.div_(float(state_count))
    trace = float(torch.trace(second_moment).detach().cpu())
    return second_moment, state_count, trace


def utility_row_drift_loss(
    delta_rows: torch.Tensor,
    utility_second_moment: torch.Tensor,
) -> torch.Tensor:
    if delta_rows.numel() == 0:
        return delta_rows.new_zeros(())
    cov = utility_second_moment.to(device=delta_rows.device, dtype=delta_rows.dtype)
    return ((delta_rows @ cov) * delta_rows).sum() / delta_rows.shape[0]


def optimize_selected_delta_with_utility(
    delta_module: repair.SelectedRowDelta,
    margin_fn,
    *,
    utility_second_moment: torch.Tensor,
    required_margins: torch.Tensor,
    repair_steps: int,
    repair_lr: float,
    repair_optimizer: str,
    hinge_weight: float,
    delta_l2_lambda: float,
    utility_weight: float,
    stop_when_all_satisfied: bool,
    max_delta_norm: Optional[float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    logs: List[Dict[str, Any]] = []
    steps_completed = 0
    stopped_early = False
    norm_projection_steps = 0

    with torch.no_grad():
        initial_margins = margin_fn(delta_module.effective_delta())
    required = required_margins.to(
        device=initial_margins.device, dtype=initial_margins.dtype
    )
    if required.shape != initial_margins.shape:
        raise ValueError("required margins must cover every training prompt instance")
    required_min = float(required.min().detach().cpu()) if required.numel() else None

    if stop_when_all_satisfied and repair.all_margins_satisfied(initial_margins, required):
        return logs, {
            "steps_completed": 0,
            "stopped_early": True,
            "all_satisfied": True,
            "training_prompt_instances": int(required.numel()),
            "required_margin_min": required_min,
            "max_delta_norm": max_delta_norm,
            "delta_norm_projection_steps": 0,
            "final_utility_row_drift_mse": 0.0,
            "final_utility_row_drift_rms": 0.0,
        }

    optimizer = repair.make_repair_optimizer(delta_module, repair_optimizer, repair_lr)
    for step in range(1, repair_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta_rows = delta_module.effective_delta()
        margins = margin_fn(delta_rows)
        hinge = repair.squared_hinge_loss(margins, required)
        delta_l2 = delta_rows.square().sum()
        utility = utility_row_drift_loss(delta_rows, utility_second_moment)
        total = (
            hinge_weight * hinge
            + delta_l2_lambda * delta_l2
            + utility_weight * utility
        )
        if not torch.isfinite(total):
            raise FloatingPointError(
                f"Non-finite utility-protected repair loss at step {step}: "
                f"{float(total.detach().cpu())}"
            )
        total.backward()
        optimizer.step()
        steps_completed = step
        before_norm, after_norm, projected = repair.constrain_effective_delta_norm(
            delta_module, max_delta_norm
        )
        if projected:
            norm_projection_steps += 1

        with torch.no_grad():
            updated_delta = delta_module.effective_delta()
            updated_margins = margin_fn(updated_delta)
            all_satisfied = repair.all_margins_satisfied(updated_margins, required)
            updated_utility = utility_row_drift_loss(
                updated_delta, utility_second_moment
            )
            logs.append(
                {
                    "step": int(step),
                    "total_loss": float(total.detach().cpu()),
                    "squared_hinge": float(hinge.detach().cpu()),
                    "weighted_hinge": float((hinge_weight * hinge).detach().cpu()),
                    "delta_l2": float(delta_l2.detach().cpu()),
                    "weighted_delta_l2": float((delta_l2_lambda * delta_l2).detach().cpu()),
                    "utility_row_drift_mse_before_step": float(utility.detach().cpu()),
                    "weighted_utility_row_drift": float((utility_weight * utility).detach().cpu()),
                    "utility_row_drift_mse_after_step": float(updated_utility.detach().cpu()),
                    "minimum_margin_before_step": float(margins.min().detach().cpu()),
                    "minimum_margin_after_step": float(updated_margins.min().detach().cpu()),
                    "minimum_required_margin": required_min,
                    "unsatisfied_after_step": int((updated_margins < required).sum().item()),
                    "all_training_prompt_instances_satisfied": bool(all_satisfied),
                    "delta_norm_projected": bool(projected),
                    "effective_delta_norm_before_projection": float(before_norm),
                    "effective_delta_norm": float(after_norm),
                }
            )
        if stop_when_all_satisfied and all_satisfied:
            stopped_early = True
            break

    with torch.no_grad():
        final_delta = delta_module.effective_delta()
        final_margins = margin_fn(final_delta)
        final_utility = utility_row_drift_loss(final_delta, utility_second_moment)
    return logs, {
        "steps_completed": int(steps_completed),
        "stopped_early": bool(stopped_early),
        "all_satisfied": repair.all_margins_satisfied(final_margins, required),
        "training_prompt_instances": int(required.numel()),
        "required_margin_min": required_min,
        "max_delta_norm": max_delta_norm,
        "delta_norm_projection_steps": int(norm_projection_steps),
        "minimum_final_margin_slack": (
            float((final_margins - required).min().detach().cpu())
            if required.numel()
            else None
        ),
        "final_utility_row_drift_mse": float(final_utility.detach().cpu()),
        "final_utility_row_drift_rms": float(final_utility.clamp_min(0).sqrt().detach().cpu()),
    }


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
        "source_experiment_config_path": str(config_path),
        "source_experiment_config": source_config,
        "preserved_5e_overlap_alphas": preserved_alphas,
        "repair_uses_official_paraphrases": False,
        "repair_prompt_scope": "requested_rewrite_only",
        "evaluation_probes_locked_during_repair": True,
        "benchmark_retain_examples_used_during_repair": 0,
        "utility_source": "wikidata_train_external_text",
        "utility_guard": "selected_lm_head_row_logit_drift_second_moment",
        "ppl_overlap_policy": "exclude_first_20_wikidata_train_texts",
    }
    gagd.write_json(output_dir / "config_used.json", config_used)

    print("Loading locked target-true-sensitive MCF forget records")
    forget_records, retain_records = repair.load_sampled_mcf_records(args)
    if retain_records:
        raise RuntimeError("MCF retain records unexpectedly entered utility repair")
    if any(record.paraphrase_prompts for record in forget_records):
        raise RuntimeError("Locked repair-visible MCF unexpectedly exposes paraphrases")
    forget_examples = [record.example for record in forget_records]
    forget_prompt_instances = repair.expand_prompt_instances(forget_records)
    if len(forget_prompt_instances) != len(forget_records):
        raise RuntimeError("Utility repair must see requested_rewrite prompts only")

    print(f"Loading Stage-1 checkpoint: {args.model_path}")
    model, tok = gagd.load_model_and_tokenizer(
        repair._model_loading_args(args), for_training=False
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_embeddings = repair.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)

    groups = gagd.collect_post_training_token_groups(tok, forget_examples, [])
    before_reports = repair.evaluate_prompt_instance_margin_reports(
        model,
        tok,
        forget_prompt_instances,
        groups,
        args.active_margin,
        device,
        args.margin_batch_size,
        llama_like,
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
        before_reports,
        original_margin_values,
        required_margins,
        args.active_margin,
    )
    active_positions = repair.select_active_positions(before_reports, args.active_margin)
    active_instances = repair._active_instances(forget_prompt_instances, active_positions)
    before_active_payload = repair.active_report_payload(before_reports, args.active_margin)
    selected_ids = repair.selected_rows_for_active_instances(
        tok, active_instances, groups, args.repair_mode
    )

    gagd.write_json(output_dir / "rewrite_margins_before.json", before_reports)
    gagd.write_json(output_dir / "active_cases_before.json", before_active_payload)

    utility_texts, utility_indices, utility_sha256 = load_utility_texts(
        args.wikidata_dir,
        count=args.utility_num,
        seed=args.utility_seed,
        exclude_first=args.utility_exclude_first,
    )
    print(
        f"Building utility second moment from {len(utility_texts)} external texts "
        f"(seed={args.utility_seed}, excluding first {args.utility_exclude_first})"
    )
    utility_second_moment, utility_state_count, utility_cov_trace = (
        build_utility_second_moment(
            model,
            tok,
            utility_texts,
            device=device,
            max_length=args.utility_max_length,
            batch_size=args.utility_batch_size,
        )
    )
    utility_manifest = {
        "schema_version": 1,
        "source": str(Path(args.wikidata_dir).resolve()),
        "split": "train",
        "text_column": "text",
        "utility_num": int(args.utility_num),
        "utility_seed": int(args.utility_seed),
        "excluded_prefix_count": int(args.utility_exclude_first),
        "excluded_reason": "official PPL evaluator uses first 20 Wikidata train texts",
        "sampled_indices": utility_indices,
        "sampled_text_sha256": utility_sha256,
        "max_length": int(args.utility_max_length),
        "predictor_hidden_state_count": int(utility_state_count),
        "second_moment_trace": float(utility_cov_trace),
    }
    gagd.write_json(output_dir / "utility_manifest.json", utility_manifest)

    input_storage_pointer = model.get_input_embeddings().weight.detach().data_ptr()
    selected_before = (
        output_weight.index_select(
            0,
            torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device),
        ).detach().clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )

    actual_rank = 0
    repair_logs: List[Dict[str, Any]] = []
    optimization_summary: Dict[str, Any] = {
        "steps_completed": 0,
        "stopped_early": True,
        "all_satisfied": repair.all_margins_satisfied(
            original_margin_tensor, required_margins
        ),
        "training_prompt_instances": len(forget_prompt_instances),
        "required_margin_min": (
            float(required_margins.min().detach().cpu()) if required_margins.numel() else None
        ),
        "final_utility_row_drift_mse": 0.0,
        "final_utility_row_drift_rms": 0.0,
    }

    if selected_ids:
        print("Caching direct forget sparse-delta objectives")
        prompt_caches = repair.build_prompt_instance_delta_caches(
            model,
            tok,
            forget_prompt_instances,
            selected_ids,
            device,
            args.margin_batch_size,
            llama_like,
        )
        active_hidden = torch.cat(
            [
                answer_cache.hidden
                for position in active_positions
                for answer_cache in (
                    prompt_caches[position].target_new,
                    prompt_caches[position].target_true,
                )
            ],
            dim=0,
        )
        direction_basis = repair.orthonormal_row_basis(
            active_hidden, max_rank=args.repair_rank
        )
        actual_rank = int(direction_basis.shape[0])
        if actual_rank == 0:
            raise RuntimeError("Active direct forget hidden basis has rank zero")
        print(
            f"Using rank-{actual_rank} forget-direction repair with utility weight "
            f"{args.utility_weight}"
        )

        delta_module = repair.SelectedRowDelta(
            len(selected_ids),
            output_weight.shape[1],
            direction_basis=direction_basis,
            retained_basis=None,
            device=output_weight.device,
        )
        margin_fn = lambda delta: repair.margins_from_delta_caches(prompt_caches, delta)
        repair_logs, optimization_summary = optimize_selected_delta_with_utility(
            delta_module,
            margin_fn,
            utility_second_moment=utility_second_moment,
            required_margins=required_margins,
            repair_steps=args.repair_steps,
            repair_lr=args.repair_lr,
            repair_optimizer=args.repair_optimizer,
            hinge_weight=args.hinge_weight,
            delta_l2_lambda=args.delta_l2_lambda,
            utility_weight=args.utility_weight,
            stop_when_all_satisfied=args.stop_when_all_satisfied,
            max_delta_norm=args.max_delta_norm,
        )
        with torch.no_grad():
            repair.materialize_selected_delta(
                output_weight, selected_ids, delta_module.effective_delta()
            )
        del delta_module, prompt_caches, active_hidden, direction_basis
        gc.collect()

    repair.write_jsonl(output_dir / "repair_log.jsonl", repair_logs)
    if model.get_input_embeddings().weight.detach().data_ptr() != input_storage_pointer:
        raise RuntimeError("Input embedding storage changed during Stage-2 utility repair")
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("Input embeddings unexpectedly became trainable")

    after_reports = repair.evaluate_prompt_instance_margin_reports(
        model,
        tok,
        forget_prompt_instances,
        groups,
        args.active_margin,
        device,
        args.margin_batch_size,
        llama_like,
    )
    repair.attach_margin_requirement_metadata(
        after_reports,
        original_margin_values,
        required_margins,
        args.active_margin,
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
        if selected_ids
        else selected_before
    )
    selected_delta = selected_after.float() - selected_before.float()

    repair_summary = {
        "method": METHOD,
        "protocol_status": "target_true_sensitive_forget_only_external_utility_locked_probes",
        "protocol_status_reason": (
            "Stage 2 used only the 50 sampled requested_rewrite forget prompts plus a "
            "fixed external Wikidata utility slice. No MCF retain, paraphrase, neighborhood, "
            "generation, or PPL-evaluation texts entered repair."
        ),
        "repair_mode": args.repair_mode,
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "source_experiment_config_path": str(config_path),
        "preserved_5e_overlap_alphas": preserved_alphas,
        "forget_records": len(forget_records),
        "forget_prompt_instances": len(forget_prompt_instances),
        "retain_records": 0,
        "benchmark_retain_examples_used_during_repair": 0,
        "repair_uses_official_paraphrases": False,
        "evaluation_probes_locked_during_repair": True,
        "active_margin": float(args.active_margin),
        "protected_margin_floor": float(args.protected_margin_floor),
        **repair.protection_summary_fields(
            transitions, required_margins, args.max_delta_norm
        ),
        "active_prompt_instances_before": len(active_positions),
        "active_prompt_instances_after": after_active_payload["active_prompt_count"],
        "active_parent_records_before": before_active_payload["active_parent_record_count"],
        "active_parent_records_after": after_active_payload["active_parent_record_count"],
        "active_cases_before": len(active_positions),
        "active_cases_after": after_active_payload["count"],
        "selected_lm_head_rows": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "changed_selected_lm_head_rows": (
            int(selected_delta.norm(dim=1).gt(0).sum().item()) if selected_ids else 0
        ),
        "selected_lm_head_delta_norm": float(selected_delta.norm().cpu()),
        "input_embeddings_modified": False,
        "transformer_parameters_trainable": 0,
        "repair_rank_requested": int(args.repair_rank),
        "repair_rank_actual": int(actual_rank),
        "utility": {
            **utility_manifest,
            "weight": float(args.utility_weight),
            "objective": "mean_selected_row_E[(delta_w^T h_utility)^2]",
            "final_row_drift_mse": optimization_summary.get(
                "final_utility_row_drift_mse"
            ),
            "final_row_drift_rms": optimization_summary.get(
                "final_utility_row_drift_rms"
            ),
        },
        "optimization": optimization_summary,
        "minimum_margin_before": min(
            (float(x["margin"]) for x in before_reports), default=None
        ),
        "minimum_margin_after": min(
            (float(x["margin"]) for x in after_reports), default=None
        ),
        "minimum_official_compatible_margin_before": min(
            (float(x["official_compatible_margin"]) for x in before_reports),
            default=None,
        ),
        "minimum_official_compatible_margin_after": min(
            (float(x["official_compatible_margin"]) for x in after_reports),
            default=None,
        ),
    }
    gagd.write_json(output_dir / "repair_summary.json", repair_summary)
    repair.raise_if_new_prompt_failures(transitions, after_reports)

    if args.save_model:
        checkpoint_dir = output_dir / "checkpoint"
        repair.save_repair_checkpoint(
            model,
            tok,
            checkpoint_dir,
            repair_config=config_used,
        )
        print(f"Saved utility-protected repaired checkpoint to {checkpoint_dir}")

    print(
        "Done: direct active prompt instances "
        f"{len(active_positions)} -> {after_active_payload['active_prompt_count']}; "
        f"utility texts={args.utility_num}; rank={actual_rank}; outputs={output_dir}"
    )


if __name__ == "__main__":
    main()
