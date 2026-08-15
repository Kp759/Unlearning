#!/usr/bin/env python3
"""SURE-TOFU Stage 1B: unrestricted sparse active forgetting.

Starts from the frozen Stage-1A GA/GD checkpoint.  Only LM-head rows required
by residual active direct forget answers are editable; transformer and input
embeddings stay frozen.  Rank 0 means an unrestricted selected-row delta in
the full hidden dimension.  Optimization uses only the same training-visible
50 direct forget QAs and stops only when every direct forget NLL constraint is
satisfied (including a buffer).

No retain95, paraphrase, same-author holdout, real-authors, world-facts, PPL,
or final-evaluation metric is loaded or used for selection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_active_forget_repair as native
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "SURE-TOFU-rank0-active-forgetting-Stage1B"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--target-nll-buffer", type=float, default=0.25)
    p.add_argument("--forget-hinge-weight", type=float, default=100.0)
    p.add_argument("--hardest-forget-hinge-weight", type=float, default=25.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--repair-steps", type=int, default=10000)
    p.add_argument("--repair-lr", type=float, default=5e-3)
    p.add_argument("--repair-optimizer", choices=("sgd", "adam", "adamw"), default="adamw")
    p.add_argument("--row-selection", choices=("all", "rare_topk"), default="all")
    p.add_argument("--rows-per-example", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--comparison-tolerance", type=float, default=1e-6)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    a = parse_args()
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, repair-steps and batch-size must be positive")
    if a.repair_lr <= 0 or a.delta_l2_lambda < 0:
        raise ValueError("invalid optimization controls")
    if a.target_nll_buffer < 0 or a.rows_per_example <= 0:
        raise ValueError("invalid target/row controls")

    forget_path = Path(a.forget_json).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    tok_data = AutoTokenizer.from_pretrained(a.model_path)
    if tok_data.pad_token is None:
        tok_data.pad_token = tok_data.eos_token
    instances, source_indices = locked.load_forget_instances(forget_path, tok_data, a.forget_num)

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = input_weight.data_ptr()
    input_version = input_weight._version
    device = gagd.first_device(model)

    baseline_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach()
    required_nll, initially_active_mask, target_nll = native.build_required_forget_nll(
        baseline_nll,
        target_probability=a.target_forget_answer_probability,
        target_nll_buffer=a.target_nll_buffer,
    )
    active_positions = initially_active_mask.nonzero(as_tuple=False).flatten().cpu().tolist()
    selected_ids, selection_report = locked.select_answer_rows(
        tok,
        instances,
        active_positions,
        max_length=a.max_length,
        row_selection=a.row_selection,
        rows_per_example=a.rows_per_example,
    )
    if active_positions and not selected_ids:
        raise RuntimeError("residual active forget answers produced no editable LM-head rows")

    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    baseline_rows = (
        output_weight.index_select(0, selected_tensor).detach().clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )

    zero_delta = torch.zeros_like(baseline_rows, dtype=torch.float32)
    baseline_metrics = locked.metrics(
        baseline_nll,
        required_nll,
        zero_delta,
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    locked.write_json(
        root / "forget_instances_before.json",
        locked.report_instances(
            instances, baseline_nll, required_nll, a.target_forget_answer_probability
        ),
    )
    write_json(
        root / "selected_lm_head_rows.json",
        {
            "repair_rank_requested": 0,
            "repair_rank_semantics": "unrestricted selected-row delta in full hidden dimension",
            "initially_active_forget_instance_count": len(active_positions),
            "active_positions": active_positions,
            "selected_lm_head_row_count": len(selected_ids),
            "selected_lm_head_token_ids": selected_ids,
            "selected_lm_head_tokens": {
                str(token_id): locked.decoded_token(tok, token_id) for token_id in selected_ids
            },
            "row_selection": a.row_selection,
            "row_selection_report": selection_report,
            "selection_source": "training-visible direct forget answers only",
            "retain_or_heldout_data_consulted": False,
        },
    )

    best_delta = zero_delta.detach().cpu().clone()
    best_metrics = dict(baseline_metrics)
    best_step = 0
    steps_completed = 0
    logs: List[Dict[str, Any]] = []

    if active_positions:
        caches = tofu.build_answer_delta_caches(
            model,
            tok,
            instances,
            selected_ids,
            device,
            batch_size=a.batch_size,
            max_length=a.max_length,
        )
        packed = tofu.pack_answer_delta_caches(caches)
        delta_module = active.SelectedRowDelta(
            len(selected_ids),
            output_weight.shape[1],
            direction_basis=None,
            retained_basis=None,
            device=device,
        )
        optimizer = active.make_repair_optimizer(delta_module, a.repair_optimizer, a.repair_lr)

        for step in range(1, a.repair_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            delta = delta_module.effective_delta()
            current_nll = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
            errors = torch.relu(required_nll.to(current_nll) - current_nll)
            loss = (
                a.forget_hinge_weight * errors.square().mean()
                + a.hardest_forget_hinge_weight * errors.square().max()
                + a.delta_l2_lambda * delta.square().sum()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite rank-0 loss at step {step}")
            loss.backward()
            optimizer.step()
            steps_completed = step

            with torch.no_grad():
                candidate_delta = delta_module.effective_delta()
                candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate_delta)
                candidate_metrics = locked.metrics(
                    candidate_nll,
                    required_nll,
                    candidate_delta,
                    target_nll=target_nll,
                    tolerance=a.comparison_tolerance,
                )
                if locked.priority(candidate_metrics) < locked.priority(best_metrics):
                    best_delta = candidate_delta.detach().cpu().clone()
                    best_metrics = dict(candidate_metrics)
                    best_step = step

            if step == 1 or step % a.log_every == 0:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "forget_hinge": float(errors.square().mean().detach().cpu()),
                    "hardest_forget_hinge": float(errors.square().max().detach().cpu()),
                    "delta_l2": float(delta.square().sum().detach().cpu()),
                    **candidate_metrics,
                }
                logs.append(row)
                print(
                    f"step={step} active={candidate_metrics['active_forget_instance_count']} "
                    f"buffered={candidate_metrics['buffered_forget_constraint_unmet_count']} "
                    f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g}"
                )

            if (
                candidate_metrics["active_forget_instance_count"] == 0
                and candidate_metrics["buffered_forget_constraint_unmet_count"] == 0
            ):
                if locked.priority(candidate_metrics) <= locked.priority(best_metrics):
                    best_delta = candidate_delta.detach().cpu().clone()
                    best_metrics = dict(candidate_metrics)
                    best_step = step
                break
        del optimizer

    write_jsonl(root / "repair_log.jsonl", logs)
    qualified = (
        best_metrics["active_forget_instance_count"] == 0
        and best_metrics["buffered_forget_constraint_unmet_count"] == 0
    )
    if not qualified:
        write_json(
            root / "repair_summary.json",
            {
                "status": "FAILED_NO_FEASIBLE_RANK0_FORGET_SOLUTION",
                "best_step": best_step,
                "steps_completed": steps_completed,
                "best_metrics_cached": best_metrics,
                "repair_rank_requested": 0,
                "selected_lm_head_row_count": len(selected_ids),
            },
        )
        raise RuntimeError("SURE-TOFU Stage1B did not satisfy every direct forget constraint")

    tofu.set_selected_lm_head_rows(output_weight, selected_ids, baseline_rows, best_delta)
    if input_weight.data_ptr() != input_pointer or input_weight._version != input_version:
        raise RuntimeError("Stage1B modified input embeddings")

    materialized_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach()
    materialized_metrics = locked.metrics(
        materialized_nll,
        required_nll,
        best_delta.to(device),
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    final_pass = (
        materialized_metrics["active_forget_instance_count"] == 0
        and materialized_metrics["buffered_forget_constraint_unmet_count"] == 0
    )
    if not final_pass:
        write_json(
            root / "repair_summary.json",
            {
                "status": "FAILED_BF16_MATERIALIZATION_FORGET_CONSTRAINT",
                "best_step": best_step,
                "best_metrics_cached": best_metrics,
                "materialized_metrics": materialized_metrics,
                "repair_rank_requested": 0,
            },
        )
        raise RuntimeError("rank-0 solution lost a direct forget constraint after materialization")

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    locked.write_json(
        root / "forget_instances_after.json",
        locked.report_instances(
            instances, materialized_nll, required_nll, a.target_forget_answer_probability
        ),
    )
    summary = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "best_step": best_step,
        "steps_completed": steps_completed,
        "repair_rank_requested": 0,
        "repair_rank_actual": 0,
        "repair_rank_semantics": "unrestricted selected-row delta",
        "initially_active_forget_instance_count": len(active_positions),
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "best_metrics_cached": best_metrics,
        "materialized_metrics": materialized_metrics,
        "training_data_access": {
            "direct_forget_qas": a.forget_num,
            "retain95": 0,
            "paraphrases": 0,
            "same_author_holdout": 0,
            "real_authors": 0,
            "world_facts": 0,
            "PPL": False,
        },
        "checkpoint": str(ckpt.resolve()),
    }
    write_json(root / "repair_summary.json", summary)
    write_json(
        root / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "model_path": a.model_path,
            "forget_json": str(forget_path),
            "seed": a.seed,
            "forget_num": a.forget_num,
            "target_forget_answer_probability": a.target_forget_answer_probability,
            "target_nll_buffer": a.target_nll_buffer,
            "repair_steps": a.repair_steps,
            "repair_lr": a.repair_lr,
            "repair_optimizer": a.repair_optimizer,
            "repair_rank": 0,
            "delta_l2_lambda": a.delta_l2_lambda,
            "row_selection": a.row_selection,
            "stage1b_visible_source_indices": source_indices,
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print(f"SURE-TOFU Stage1B PASS: {ckpt}")
    print(
        f"rank0 selected_rows={len(selected_ids)} best_step={best_step} "
        f"mean_prob={materialized_metrics['forget_answer_probability_mean']:.8g} "
        f"max_prob={materialized_metrics['forget_answer_probability_max']:.8g}"
    )


if __name__ == "__main__":
    main()
