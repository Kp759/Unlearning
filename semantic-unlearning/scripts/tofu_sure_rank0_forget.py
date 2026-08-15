#!/usr/bin/env python3
"""SURE-TOFU Stage 1B: sparse rank-0 forgetting with answer-row restoration.

Starts from the frozen Stage-1A GA/GD checkpoint.  Before rank-0 optimization,
this stage audits the same training-visible 50 direct forget QAs at token
resolution.  Answer-token rows that are not needed by a residual direct-forget
constraint are restored exactly to the original Full-TOFU Base in BOTH input
embeddings and the detached LM head.  Only sensitive answer-token rows retain
Stage-1A displacement, and rank-0 may edit only those sensitive LM-head rows.

Sensitivity is conservative and data-local:

* a sequence must currently violate its required direct-forget NLL;
* within that sequence, a target-token row is sensitive when that token's NLL
  is below the sequence's required NLL;
* sensitivity is global by vocabulary row: if a token ID is sensitive anywhere,
  that shared row is never restored as non-sensitive;
* after candidate non-sensitive rows are snapped to Base, all 50 constraints
  are re-audited.  Any newly/remaining violating sequence promotes its deficient
  answer-token rows back to the sensitive set.  A fallback promotes all answer
  rows for that violating sequence if token-level promotion stalls.

The promotion loop therefore fails closed: a row is called non-sensitive only
when restoring it to Base is compatible with the visible direct-forget
constraints.  No retain95, paraphrase, same-author holdout, real-authors,
world-facts, PPL, or final-evaluation metric is loaded or used for selection.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_active_forget_repair as native
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "SURE-TOFU-rank0-sensitive-row-restored-Stage1B"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Frozen Stage1A checkpoint")
    p.add_argument(
        "--reference-model-path",
        required=True,
        help="Original Full-TOFU checkpoint used only to restore visible answer rows.",
    )
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--comparison-tolerance", type=float, default=1e-6)
    p.add_argument("--max-promotion-rounds", type=int, default=64)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def all_answer_rows(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    *,
    max_length: int,
) -> List[int]:
    return locked.answer_rows_for_instances(
        tok,
        instances,
        list(range(len(instances))),
        max_length=max_length,
    )


def violating_sequence_positions(
    nll: torch.Tensor,
    required_nll: torch.Tensor,
    tolerance: float,
) -> List[int]:
    required = required_nll.to(device=nll.device, dtype=nll.dtype)
    return (
        (nll < (required - tolerance))
        .nonzero(as_tuple=False)
        .flatten()
        .detach()
        .cpu()
        .tolist()
    )


def sensitive_rows_from_token_deficits(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    all_selected_ids: Sequence[int],
    required_nll: torch.Tensor,
    positions: Sequence[int],
    *,
    tolerance: float,
) -> List[int]:
    """Return globally sensitive target rows from token-level NLL deficits."""
    selected: set[int] = set()
    for position in positions:
        cache = caches[position]
        requirement = required_nll[position].to(
            device=cache.base_token_nll.device,
            dtype=cache.base_token_nll.dtype,
        )
        deficient = cache.base_token_nll < (requirement - tolerance)
        columns = cache.target_selected_columns[deficient]
        for column in columns.detach().cpu().tolist():
            if int(column) < 0:
                raise RuntimeError(
                    "all-answer cache lost a target row while deriving sensitivity"
                )
            selected.add(int(all_selected_ids[int(column)]))
    return sorted(selected)


def rows_for_positions(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    positions: Sequence[int],
    *,
    max_length: int,
) -> List[int]:
    return locked.answer_rows_for_instances(
        tok,
        instances,
        positions,
        max_length=max_length,
    )


@torch.no_grad()
def load_reference_answer_rows(
    model_path: str,
    answer_ids: Sequence[int],
    dtype_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load only requested Full-TOFU input/output rows and return them on CPU."""
    dtype = gagd.torch_dtype(dtype_name)
    reference = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    input_weight = reference.get_input_embeddings().weight
    output_weight = reference.get_output_embeddings().weight
    input_ids = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    output_ids = input_ids.to(output_weight.device)
    input_rows = input_weight.index_select(0, input_ids).detach().cpu().clone()
    output_rows = output_weight.index_select(0, output_ids).detach().cpu().clone()
    del reference
    gc.collect()
    return input_rows, output_rows


@torch.no_grad()
def apply_answer_row_policy(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    all_ids: Sequence[int],
    sensitive_ids: Sequence[int],
    stage1a_input_rows: torch.Tensor,
    stage1a_output_rows: torch.Tensor,
    base_input_rows: torch.Tensor,
    base_output_rows: torch.Tensor,
) -> None:
    """Keep sensitive rows at Stage1A; restore all other visible answer rows to Base."""
    if not (
        len(all_ids)
        == stage1a_input_rows.shape[0]
        == stage1a_output_rows.shape[0]
        == base_input_rows.shape[0]
        == base_output_rows.shape[0]
    ):
        raise ValueError("answer-row snapshot lengths do not agree")
    all_set = set(int(token_id) for token_id in all_ids)
    sensitive_set = set(int(token_id) for token_id in sensitive_ids)
    if not sensitive_set.issubset(all_set):
        raise ValueError("sensitive answer rows must be a subset of all visible answer rows")

    all_input_ids = torch.tensor(all_ids, dtype=torch.long, device=input_weight.device)
    all_output_ids = all_input_ids.to(output_weight.device)
    desired_input = base_input_rows.to(device=input_weight.device, dtype=input_weight.dtype).clone()
    desired_output = base_output_rows.to(device=output_weight.device, dtype=output_weight.dtype).clone()
    id_to_position = {int(token_id): position for position, token_id in enumerate(all_ids)}
    sensitive_positions = [id_to_position[token_id] for token_id in sorted(sensitive_set)]
    if sensitive_positions:
        pos_cpu = torch.tensor(sensitive_positions, dtype=torch.long)
        desired_input.index_copy_(
            0,
            pos_cpu.to(desired_input.device),
            stage1a_input_rows.index_select(0, pos_cpu).to(
                device=desired_input.device, dtype=desired_input.dtype
            ),
        )
        desired_output.index_copy_(
            0,
            pos_cpu.to(desired_output.device),
            stage1a_output_rows.index_select(0, pos_cpu).to(
                device=desired_output.device, dtype=desired_output.dtype
            ),
        )
    input_weight.index_copy_(0, all_input_ids, desired_input)
    output_weight.index_copy_(0, all_output_ids, desired_output)


def answer_row_restoration_error(
    weight: torch.Tensor,
    all_ids: Sequence[int],
    non_sensitive_ids: Sequence[int],
    base_rows: torch.Tensor,
) -> float:
    if not non_sensitive_ids:
        return 0.0
    id_to_position = {int(token_id): position for position, token_id in enumerate(all_ids)}
    base_positions = torch.tensor(
        [id_to_position[int(token_id)] for token_id in non_sensitive_ids],
        dtype=torch.long,
    )
    weight_ids = torch.tensor(non_sensitive_ids, dtype=torch.long, device=weight.device)
    current = weight.index_select(0, weight_ids).detach().float().cpu()
    expected = base_rows.index_select(0, base_positions).detach().float().cpu()
    return float((current - expected).abs().max().item())


def sure_priority(metrics: Dict[str, Any]) -> tuple[int, int, float, float]:
    """For SURE, once feasible prefer the smallest edit rather than deeper forgetting."""
    return (
        int(metrics["active_forget_instance_count"]),
        int(metrics["buffered_forget_constraint_unmet_count"]),
        float(metrics["selected_lm_head_delta_norm"]),
        float(metrics["forget_answer_probability_max"]),
    )


def main() -> None:
    a = parse_args()
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, repair-steps and batch-size must be positive")
    if a.repair_lr <= 0 or a.delta_l2_lambda < 0:
        raise ValueError("invalid optimization controls")
    if a.max_promotion_rounds <= 0 or a.comparison_tolerance < 0:
        raise ValueError("invalid promotion/tolerance controls")

    forget_path = Path(a.forget_json).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
    reference_path = Path(a.reference_model_path).resolve()
    if not reference_path.is_dir():
        raise FileNotFoundError(reference_path)
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
    device = gagd.first_device(model)

    # Audit the untouched Stage1A checkpoint first.  Its NLLs define which
    # sequences still need forgetting and the buffered requirements are fixed
    # before any row restoration occurs.
    stage1a_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    required_nll, _, target_nll = native.build_required_forget_nll(
        stage1a_nll,
        target_probability=a.target_forget_answer_probability,
        target_nll_buffer=a.target_nll_buffer,
    )
    locked.write_json(
        root / "forget_instances_stage1a_before_row_restore.json",
        locked.report_instances(
            instances, stage1a_nll, required_nll, a.target_forget_answer_probability
        ),
    )

    answer_ids = all_answer_rows(tok, instances, max_length=a.max_length)
    if not answer_ids:
        raise RuntimeError("visible direct forget answers produced no answer-token rows")
    answer_tensor_input = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    answer_tensor_output = answer_tensor_input.to(output_weight.device)
    stage1a_input_rows = input_weight.index_select(0, answer_tensor_input).detach().cpu().clone()
    stage1a_output_rows = output_weight.index_select(0, answer_tensor_output).detach().cpu().clone()
    base_input_rows, base_output_rows = load_reference_answer_rows(
        str(reference_path), answer_ids, a.dtype
    )

    # Build a cache over every visible answer row so target token IDs can be
    # recovered exactly from target_selected_columns.  Only violating sequences
    # contribute sensitive rows.
    initial_all_cache = tofu.build_answer_delta_caches(
        model,
        tok,
        instances,
        answer_ids,
        device,
        batch_size=a.batch_size,
        max_length=a.max_length,
    )
    violating = violating_sequence_positions(
        stage1a_nll, required_nll, a.comparison_tolerance
    )
    sensitive_ids: set[int] = set(
        sensitive_rows_from_token_deficits(
            initial_all_cache,
            answer_ids,
            required_nll,
            violating,
            tolerance=a.comparison_tolerance,
        )
    )
    if violating and not sensitive_ids:
        sensitive_ids.update(
            rows_for_positions(tok, instances, violating, max_length=a.max_length)
        )

    promotion_rounds: List[Dict[str, Any]] = []
    current_nll = stage1a_nll
    for promotion_round in range(a.max_promotion_rounds):
        apply_answer_row_policy(
            input_weight,
            output_weight,
            answer_ids,
            sorted(sensitive_ids),
            stage1a_input_rows,
            stage1a_output_rows,
            base_input_rows,
            base_output_rows,
        )
        current_nll = tofu.score_answer_instances(
            model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
        ).detach().float()
        violating = violating_sequence_positions(
            current_nll, required_nll, a.comparison_tolerance
        )
        before = set(sensitive_ids)
        promoted: set[int] = set()
        fallback_used = False
        if violating:
            fresh_cache = tofu.build_answer_delta_caches(
                model,
                tok,
                instances,
                answer_ids,
                device,
                batch_size=a.batch_size,
                max_length=a.max_length,
            )
            promoted.update(
                sensitive_rows_from_token_deficits(
                    fresh_cache,
                    answer_ids,
                    required_nll,
                    violating,
                    tolerance=a.comparison_tolerance,
                )
            )
            promoted -= before
            if not promoted:
                promoted.update(
                    rows_for_positions(tok, instances, violating, max_length=a.max_length)
                )
                promoted -= before
                fallback_used = True
            sensitive_ids.update(promoted)

        promotion_rounds.append(
            {
                "round": promotion_round,
                "violating_sequence_count": len(violating),
                "violating_positions": violating,
                "sensitive_row_count_before": len(before),
                "newly_promoted_row_count": len(promoted),
                "newly_promoted_token_ids": sorted(promoted),
                "fallback_promote_all_answer_rows": fallback_used,
                "sensitive_row_count_after": len(sensitive_ids),
            }
        )
        if not violating or not promoted:
            break
    else:
        raise RuntimeError("answer-row sensitivity promotion exceeded max-promotion-rounds")

    # Re-apply the final policy after the last promotion so all newly promoted
    # rows are at Stage1A and every remaining answer row is exactly Base.
    apply_answer_row_policy(
        input_weight,
        output_weight,
        answer_ids,
        sorted(sensitive_ids),
        stage1a_input_rows,
        stage1a_output_rows,
        base_input_rows,
        base_output_rows,
    )
    pre_rank0_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
    pre_rank0_violating = violating_sequence_positions(
        pre_rank0_nll, required_nll, a.comparison_tolerance
    )

    selected_ids = sorted(sensitive_ids)
    non_sensitive_ids = sorted(set(answer_ids) - set(selected_ids))
    if pre_rank0_violating and not selected_ids:
        raise RuntimeError("violating direct forget constraints have no sensitive answer rows")

    selected_tensor = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    baseline_rows = (
        output_weight.index_select(0, selected_tensor).detach().clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )
    zero_delta = torch.zeros_like(baseline_rows, dtype=torch.float32)
    baseline_metrics = locked.metrics(
        pre_rank0_nll,
        required_nll,
        zero_delta,
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )

    input_pointer = input_weight.data_ptr()
    input_version = input_weight._version
    locked.write_json(
        root / "forget_instances_before.json",
        locked.report_instances(
            instances, pre_rank0_nll, required_nll, a.target_forget_answer_probability
        ),
    )
    write_json(
        root / "answer_row_restoration.json",
        {
            "policy": "restore_non_sensitive_visible_answer_rows_to_full_tofu_base",
            "sensitivity_unit": "global vocabulary row derived from teacher-forced token deficits",
            "all_visible_answer_row_count": len(answer_ids),
            "all_visible_answer_token_ids": answer_ids,
            "sensitive_answer_row_count": len(selected_ids),
            "sensitive_answer_token_ids": selected_ids,
            "sensitive_answer_tokens": {
                str(token_id): locked.decoded_token(tok, token_id) for token_id in selected_ids
            },
            "non_sensitive_answer_row_count": len(non_sensitive_ids),
            "non_sensitive_answer_token_ids": non_sensitive_ids,
            "non_sensitive_answer_tokens": {
                str(token_id): locked.decoded_token(tok, token_id)
                for token_id in non_sensitive_ids
            },
            "input_embedding_policy": "non-sensitive answer rows exact Full-TOFU Base; sensitive rows Stage1A",
            "lm_head_policy_before_rank0": "non-sensitive answer rows exact Full-TOFU Base; sensitive rows Stage1A",
            "promotion_rounds": promotion_rounds,
            "pre_rank0_violating_sequence_count": len(pre_rank0_violating),
            "pre_rank0_violating_positions": pre_rank0_violating,
            "retain_or_heldout_data_consulted": False,
        },
    )

    best_delta = zero_delta.detach().cpu().clone()
    best_metrics = dict(baseline_metrics)
    best_step = 0
    steps_completed = 0
    logs: List[Dict[str, Any]] = []

    if pre_rank0_violating:
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
            current = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
            errors = torch.relu(required_nll.to(current) - current)
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
                candidate_nll = tofu.answer_nlls_from_packed_delta_cache(
                    packed, candidate_delta
                )
                candidate_metrics = locked.metrics(
                    candidate_nll,
                    required_nll,
                    candidate_delta,
                    target_nll=target_nll,
                    tolerance=a.comparison_tolerance,
                )
                if sure_priority(candidate_metrics) < sure_priority(best_metrics):
                    best_delta = candidate_delta.detach().cpu().clone()
                    best_metrics = dict(candidate_metrics)
                    best_step = step

            if step == 1 or step % a.log_every == 0:
                logs.append(
                    {
                        "step": step,
                        "loss": float(loss.detach().cpu()),
                        "forget_hinge": float(errors.square().mean().detach().cpu()),
                        "hardest_forget_hinge": float(errors.square().max().detach().cpu()),
                        "delta_l2": float(delta.square().sum().detach().cpu()),
                        **candidate_metrics,
                    }
                )
                print(
                    f"step={step} active={candidate_metrics['active_forget_instance_count']} "
                    f"buffered={candidate_metrics['buffered_forget_constraint_unmet_count']} "
                    f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g} "
                    f"norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
                )

            if (
                candidate_metrics["active_forget_instance_count"] == 0
                and candidate_metrics["buffered_forget_constraint_unmet_count"] == 0
            ):
                if sure_priority(candidate_metrics) <= sure_priority(best_metrics):
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
                "sensitive_answer_row_count": len(selected_ids),
                "non_sensitive_answer_row_count": len(non_sensitive_ids),
            },
        )
        raise RuntimeError("SURE-TOFU Stage1B did not satisfy every direct forget constraint")

    tofu.set_selected_lm_head_rows(output_weight, selected_ids, baseline_rows, best_delta)
    if input_weight.data_ptr() != input_pointer or input_weight._version != input_version:
        raise RuntimeError("rank-0 optimizer modified input embeddings")

    # Reassert the exact Base restoration for non-sensitive answer rows after
    # rank-0 materialization.  Sensitive input rows remain Stage1A; sensitive
    # output rows retain Stage1A + rank-0.
    id_to_position = {int(token_id): position for position, token_id in enumerate(answer_ids)}
    if non_sensitive_ids:
        ns_positions = torch.tensor(
            [id_to_position[token_id] for token_id in non_sensitive_ids],
            dtype=torch.long,
        )
        ns_input_ids = torch.tensor(
            non_sensitive_ids, dtype=torch.long, device=input_weight.device
        )
        ns_output_ids = ns_input_ids.to(output_weight.device)
        with torch.no_grad():
            input_weight.index_copy_(
                0,
                ns_input_ids,
                base_input_rows.index_select(0, ns_positions).to(
                    device=input_weight.device, dtype=input_weight.dtype
                ),
            )
            output_weight.index_copy_(
                0,
                ns_output_ids,
                base_output_rows.index_select(0, ns_positions).to(
                    device=output_weight.device, dtype=output_weight.dtype
                ),
            )

    materialized_nll = tofu.score_answer_instances(
        model, tok, instances, device, batch_size=a.batch_size, max_length=a.max_length
    ).detach().float()
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
    input_restore_error = answer_row_restoration_error(
        input_weight,
        answer_ids,
        non_sensitive_ids,
        base_input_rows,
    )
    output_restore_error = answer_row_restoration_error(
        output_weight,
        answer_ids,
        non_sensitive_ids,
        base_output_rows,
    )
    if input_restore_error != 0.0 or output_restore_error != 0.0:
        raise RuntimeError(
            "non-sensitive answer rows are not exact Base after rank-0: "
            f"input={input_restore_error} output={output_restore_error}"
        )
    if not final_pass:
        write_json(
            root / "repair_summary.json",
            {
                "status": "FAILED_POST_RESTORE_FORGET_CONSTRAINT",
                "best_step": best_step,
                "best_metrics_cached": best_metrics,
                "materialized_metrics": materialized_metrics,
                "repair_rank_requested": 0,
                "sensitive_answer_row_count": len(selected_ids),
                "non_sensitive_answer_row_count": len(non_sensitive_ids),
            },
        )
        raise RuntimeError(
            "rank-0 solution lost a direct forget constraint after exact non-sensitive restoration"
        )

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
        "repair_rank_semantics": "unrestricted selected-row delta on sensitive answer rows only",
        "all_visible_answer_row_count": len(answer_ids),
        "sensitive_answer_row_count": len(selected_ids),
        "sensitive_answer_token_ids": selected_ids,
        "non_sensitive_answer_row_count": len(non_sensitive_ids),
        "non_sensitive_answer_token_ids": non_sensitive_ids,
        "non_sensitive_input_base_max_abs_error": input_restore_error,
        "non_sensitive_output_base_max_abs_error": output_restore_error,
        "best_metrics_cached": best_metrics,
        "materialized_metrics": materialized_metrics,
        "training_data_access": {
            "direct_forget_qas": a.forget_num,
            "full_tofu_reference_rows": "visible answer token rows only",
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
            "schema_version": 2,
            "method": METHOD,
            "protocol": PROTOCOL,
            "model_path": a.model_path,
            "reference_model_path": str(reference_path),
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
            "sensitivity_rule": "violating sequence + teacher-forced token NLL below sequence required NLL + fail-closed promotion",
            "answer_row_restoration": "non-sensitive visible answer rows exact Full-TOFU Base in input embedding and LM head",
            "stage1b_visible_source_indices": source_indices,
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print(
        "SURE-TOFU Stage1B PASS "
        f"sensitive_rows={len(selected_ids)} non_sensitive_rows={len(non_sensitive_ids)} "
        f"rank0_norm={materialized_metrics['selected_lm_head_delta_norm']:.6g} "
        f"max_prob={materialized_metrics['forget_answer_probability_max']:.8g}"
    )
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
