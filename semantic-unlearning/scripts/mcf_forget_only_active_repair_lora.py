#!/usr/bin/env python3
"""Sparse-LoRA Stage-2 repair for the locked forget-only MCF protocol.

This is an ablation of ``mcf_forget_only_active_repair.py``.  Everything up to
active-case detection is kept identical, but the selected LM-head row delta is
parameterized as a learned LoRA factorization instead of the fixed-SVD-basis
parameterization used by ``SelectedRowDelta``::

    current SURE repair:  DeltaW_A = C @ B_fixed
    sparse LoRA repair:   DeltaW_A = B_lora @ A_lora * (alpha / rank)

Only the same active LM-head vocabulary rows may change.  Transformer weights,
input embeddings, all non-selected LM-head rows, MCF retain records, official
paraphrases, and neighborhood probes remain unavailable to Stage 2.

For the fairest comparison with the registered 3B run, use ``--repair-rank 1``:
the previous run requested rank 2 but every activated seed had actual SVD rank
1.  Rank 2 can be run separately as an additional LoRA-capacity ablation.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

import gagd_active_case_repair as repair
import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like


METHOD = "mcf_forget_only_sparse_lora_repair"


class SparseLoRADelta(nn.Module):
    """Trainable low-rank delta restricted to selected LM-head rows only."""

    def __init__(
        self,
        n_rows: int,
        hidden_size: int,
        rank: int,
        *,
        alpha: Optional[float],
        device: torch.device,
    ) -> None:
        super().__init__()
        if n_rows <= 0:
            raise ValueError("SparseLoRADelta requires at least one selected row")
        if hidden_size <= 0 or rank <= 0:
            raise ValueError("hidden_size and rank must be positive")
        self.n_rows = int(n_rows)
        self.hidden_size = int(hidden_size)
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA alpha must be finite and positive")
        self.scaling = self.alpha / self.rank

        # Standard LoRA-style initialization: A is random, B is zero, so the
        # initial effective delta is exactly zero while gradients can flow.
        self.lora_A = nn.Parameter(
            torch.empty((self.rank, self.hidden_size), device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros((self.n_rows, self.rank), device=device, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def effective_delta(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.lora_A.numel() + self.lora_B.numel())


def parse_args() -> argparse.Namespace:
    parser = repair.build_parser()
    parser.description = __doc__
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=None,
        help=(
            "LoRA alpha. Default: rank, giving scaling alpha/rank = 1 so the "
            "comparison changes only the factorization, not an extra scale."
        ),
    )
    return parser.parse_args()


def validate_forget_only_lora(args: argparse.Namespace) -> None:
    if args.repair_mode != "minimal_optimize":
        raise ValueError("Sparse LoRA ablation supports --repair-mode minimal_optimize only")
    if args.forget_num <= 0:
        raise ValueError("--forget-num must be positive")
    if args.retain_num != 0:
        raise ValueError("Locked sparse-LoRA repair requires --retain-num 0")
    if args.retain_kl_mu != 0:
        raise ValueError("Locked sparse-LoRA repair requires --retain-kl-mu 0")
    if args.project_away_retain_hidden:
        raise ValueError("Locked sparse-LoRA repair forbids retain-hidden projection")
    if args.retain_calibration_num != 0:
        raise ValueError("Locked sparse-LoRA repair requires retain calibration = 0")
    if args.reference_model_path is not None:
        raise ValueError("Locked sparse-LoRA repair does not use a reference model")
    if args.run_official_mcf_eval:
        raise ValueError("Final official evaluation must remain outside Stage 2")
    if args.repair_rank <= 0:
        raise ValueError("Sparse LoRA requires --repair-rank > 0")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("repair steps and learning rate must be positive")
    if args.hinge_weight <= 0 or args.delta_l2_lambda < 0:
        raise ValueError("invalid repair regularization weights")
    if args.margin_batch_size <= 0:
        raise ValueError("margin batch size must be positive")
    if not math.isfinite(args.active_margin) or args.active_margin < 0:
        raise ValueError("active margin must be finite and non-negative")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("max delta norm must be finite and non-negative")
    if args.lora_alpha is not None and (
        not math.isfinite(args.lora_alpha) or args.lora_alpha <= 0
    ):
        raise ValueError("LoRA alpha must be finite and positive")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    validate_forget_only_lora(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path, source_config, preserved_alphas = repair.recover_experiment_config(
        args.model_path, args.experiment_config_path
    )
    repair.validate_source_experiment_config(source_config, args)

    config_used: Dict[str, Any] = {
        **vars(args),
        "method": METHOD,
        "protocol": "zerounlearn_data_access_forget_only_locked_probes",
        "repair_parameterization": "sparse_lora_selected_rows",
        "repair_prompt_scope": "requested_rewrite_only",
        "evaluation_probes_locked_during_repair": True,
        "benchmark_retain_examples_used_during_repair": 0,
        "source_experiment_config_path": str(config_path),
        "source_experiment_config": source_config,
        "preserved_5e_overlap_alphas": preserved_alphas,
        "lora_scaling_rule": "alpha / rank",
        "lora_alpha_effective": float(args.repair_rank if args.lora_alpha is None else args.lora_alpha),
    }
    write_json(output_dir / "config_used.json", config_used)

    forget_records, retain_records = repair.load_sampled_mcf_records(args)
    if retain_records:
        raise RuntimeError("Sparse-LoRA Stage 2 unexpectedly loaded MCF retain records")
    if any(record.paraphrase_prompts for record in forget_records):
        raise RuntimeError("Repair-visible MCF exposed paraphrases to sparse LoRA")

    forget_examples = [record.example for record in forget_records]
    forget_prompt_instances = repair.expand_prompt_instances(forget_records)

    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(model_args, for_training=False)
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
    original_margin_values = [float(report["margin"]) for report in before_reports]
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
    active_instances = [forget_prompt_instances[position] for position in active_positions]
    selected_ids = repair.selected_rows_for_active_instances(
        tok, active_instances, groups, args.repair_mode
    )

    write_json(output_dir / "rewrite_margins_before.json", before_reports)
    write_json(
        output_dir / "active_cases_before.json",
        repair.active_report_payload(before_reports, args.active_margin),
    )

    input_weight = model.get_input_embeddings().weight
    input_storage_pointer = input_weight.data_ptr()
    selected_before = (
        output_weight.index_select(
            0,
            torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device),
        ).detach().clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )

    repair_logs = []
    optimization_summary: Dict[str, Any] = {
        "steps_completed": 0,
        "stopped_early": False,
        "all_satisfied": repair.all_margins_satisfied(
            original_margin_tensor, required_margins
        ),
        "training_prompt_instances": len(forget_prompt_instances),
        "required_margin_min": (
            float(required_margins.min().detach().cpu())
            if required_margins.numel()
            else None
        ),
        "max_delta_norm": args.max_delta_norm,
        "delta_norm_projection_steps": 0,
    }

    lora_rank_actual = 0
    lora_alpha_effective = float(
        args.repair_rank if args.lora_alpha is None else args.lora_alpha
    )
    trainable_stage2_params = 0
    lora_shapes: Dict[str, Any] = {
        "A": None,
        "B": None,
        "effective_delta": [0, int(output_weight.shape[1])],
    }

    if selected_ids:
        prompt_instance_caches = repair.build_prompt_instance_delta_caches(
            model,
            tok,
            forget_prompt_instances,
            selected_ids,
            device,
            args.margin_batch_size,
            llama_like,
        )
        delta_module = SparseLoRADelta(
            n_rows=len(selected_ids),
            hidden_size=int(output_weight.shape[1]),
            rank=args.repair_rank,
            alpha=args.lora_alpha,
            device=output_weight.device,
        )
        lora_rank_actual = min(
            args.repair_rank, len(selected_ids), int(output_weight.shape[1])
        )
        trainable_stage2_params = delta_module.trainable_parameter_count
        lora_shapes = {
            "A": list(delta_module.lora_A.shape),
            "B": list(delta_module.lora_B.shape),
            "effective_delta": [len(selected_ids), int(output_weight.shape[1])],
        }

        margin_fn = lambda delta: repair.margins_from_delta_caches(
            prompt_instance_caches, delta
        )
        kl_fn = lambda delta: delta.new_zeros(())
        repair_logs, optimization_summary = repair.optimize_selected_delta(
            delta_module,
            margin_fn,
            kl_fn,
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
                output_weight,
                selected_ids,
                delta_module.effective_delta(),
            )

    repair.write_jsonl(output_dir / "repair_log.jsonl", repair_logs)

    if model.get_input_embeddings().weight.data_ptr() != input_storage_pointer:
        raise RuntimeError("Input embedding storage changed during sparse-LoRA repair")
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
    repair.raise_if_new_prompt_failures(transitions, after_reports)
    after_active_payload = repair.active_report_payload(after_reports, args.active_margin)
    write_json(output_dir / "rewrite_margins_after.json", after_reports)
    write_json(output_dir / "active_cases_after.json", after_active_payload)

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
        "schema_version": 1,
        "method": METHOD,
        "protocol_status": "zerounlearn_data_access_forget_only_locked_probes",
        "repair_parameterization": "sparse_lora_selected_rows",
        "repair_prompt_scope": "requested_rewrite_only",
        "repair_uses_official_paraphrases": False,
        "evaluation_probes_locked_during_repair": True,
        "benchmark_retain_examples_used_during_repair": 0,
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "source_experiment_config_path": str(config_path),
        "preserved_5e_overlap_alphas": preserved_alphas,
        "forget_records": len(forget_records),
        "forget_prompt_instances": len(forget_prompt_instances),
        "retain_records": 0,
        "active_margin": args.active_margin,
        "protected_margin_floor": args.protected_margin_floor,
        **repair.protection_summary_fields(
            transitions, required_margins, args.max_delta_norm
        ),
        "active_prompt_instances_before": len(active_positions),
        "active_prompt_instances_after": after_active_payload["active_prompt_count"],
        "selected_lm_head_rows": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "changed_selected_lm_head_rows": (
            int(selected_delta.norm(dim=1).gt(0).sum().item()) if selected_ids else 0
        ),
        "selected_lm_head_delta_norm": float(selected_delta.norm().cpu()),
        "input_embeddings_modified": False,
        "transformer_parameters_trainable": 0,
        "lora_rank_requested": args.repair_rank,
        "lora_rank_effective_upper_bound": lora_rank_actual,
        "lora_alpha": lora_alpha_effective,
        "lora_scaling": lora_alpha_effective / args.repair_rank,
        "lora_shapes": lora_shapes,
        "stage2_trainable_parameters": trainable_stage2_params,
        "optimization": optimization_summary,
        "minimum_margin_before": min(original_margin_values, default=None),
        "minimum_margin_after": min(
            (float(report["margin"]) for report in after_reports), default=None
        ),
    }
    write_json(output_dir / "repair_summary.json", repair_summary)

    config_used.update(
        {
            "selected_lm_head_rows": len(selected_ids),
            "selected_lm_head_token_ids": selected_ids,
            "lora_rank_effective_upper_bound": lora_rank_actual,
            "lora_shapes": lora_shapes,
            "stage2_trainable_parameters": trainable_stage2_params,
        }
    )
    write_json(output_dir / "config_used.json", config_used)

    if args.save_model:
        repair.save_repair_checkpoint(
            model, tok, checkpoint_dir, repair_config=config_used
        )

    print(f"Selected LM-head rows: {len(selected_ids)}")
    print(f"LoRA rank requested: {args.repair_rank}")
    print(f"LoRA effective rank upper bound: {lora_rank_actual}")
    print(f"Stage-2 trainable parameters: {trainable_stage2_params}")
    print(f"Effective delta shape: {lora_shapes['effective_delta']}")
    if args.save_model:
        print(f"Sparse-LoRA repaired checkpoint: {checkpoint_dir}")


if __name__ == "__main__":
    main()
