#!/usr/bin/env python3
"""Sparse TOFU Stage 2 using only the same 50 direct forget QAs as Stage 1.

Unlike the native full-utility TOFU repair, this locked ZeroUnlearn-style
variant has no access to retain95, real-authors, world-facts, paraphrases, or
perturbed answers. The transformer and input embeddings stay frozen.

Two LM-head row-selection policies are supported:

* ``all`` preserves the original behavior and edits the union of all answer
  token rows from initially active direct forget examples.
* ``rare_topk`` uses only the same training-visible direct forget answers to
  compute answer-token document frequency, removes punctuation-only tokens,
  and keeps the rarest ``--rows-per-example`` content-bearing rows for each
  initially active forget example. This concentrates a bounded repair norm on
  fact-specific output rows without consulting retain or evaluation data.

The low-rank repair basis is still built only from active direct-forget answer
hidden states.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_gagd_active_forget_repair as native
import tofu_gagd_neighborhood_confidence as tofu
from controlled_unlearning_protocol import load_json_or_jsonl


METHOD = "tofu_forget_only_active_repair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--forget-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    parser.add_argument("--target-nll-buffer", type=float, default=0.25)
    parser.add_argument("--forget-hinge-weight", type=float, default=100.0)
    parser.add_argument("--hardest-forget-hinge-weight", type=float, default=25.0)
    parser.add_argument("--delta-l2-lambda", type=float, default=1e-5)
    parser.add_argument("--repair-steps", type=int, default=5000)
    parser.add_argument("--repair-lr", type=float, default=2e-2)
    parser.add_argument(
        "--repair-optimizer", choices=["sgd", "adam", "adamw"], default="adamw"
    )
    parser.add_argument("--repair-rank", type=int, default=64)
    parser.add_argument("--basis-max-rows", type=int, default=2048)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument(
        "--row-selection",
        choices=["all", "rare_topk"],
        default="all",
        help=(
            "Editable LM-head row policy. 'all' reproduces the original union "
            "of active answer rows. 'rare_topk' selects rare content-bearing "
            "answer rows using only the training-visible direct forget QAs."
        ),
    )
    parser.add_argument(
        "--rows-per-example",
        type=int,
        default=3,
        help="Number of rare answer-token rows retained per active example in rare_topk mode.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--comparison-tolerance", type=float, default=1e-6)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-best-effort", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_forget_instances(
    path: Path,
    tok: Any,
    expected_count: int,
) -> Tuple[List[tofu.TOFUAnswerInstance], List[int]]:
    rows = load_json_or_jsonl(path)
    if len(rows) != expected_count:
        raise ValueError(
            f"locked forget file has {len(rows)} rows, expected {expected_count}"
        )
    instances: List[tofu.TOFUAnswerInstance] = []
    source_indices: List[int] = []
    for position, row in enumerate(rows):
        allowed = {"question", "answer", "_source_index"}
        extras = set(row) - allowed
        if extras:
            raise ValueError(
                f"Stage2 row {position} exposes evaluation-only fields: {sorted(extras)}"
            )
        if not row.get("question") or not row.get("answer"):
            raise ValueError(f"Stage2 row {position} lacks question/answer")
        source_index = int(row.get("_source_index", position))
        question = str(row["question"])
        instances.append(
            tofu.TOFUAnswerInstance(
                split="forget_locked",
                source_index=source_index,
                sampled_position=position,
                question=question,
                answer=str(row["answer"]),
                prompt=tofu.format_question_prompt(tok, question),
            )
        )
        source_indices.append(source_index)
    return instances, source_indices


def answer_token_ids(
    tok: Any,
    instance: tofu.TOFUAnswerInstance,
    *,
    max_length: int,
) -> List[int]:
    full_ids, prompt_length = tofu.answer_sequence_components(tok, instance, max_length)
    specials = gagd.special_token_ids(tok)
    return [int(token_id) for token_id in full_ids[prompt_length:] if token_id not in specials]


def answer_rows_for_instances(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    positions: Sequence[int],
    *,
    max_length: int,
) -> List[int]:
    selected: set[int] = set()
    for position in positions:
        selected.update(
            answer_token_ids(tok, instances[position], max_length=max_length)
        )
    return sorted(selected)


def decoded_token(tok: Any, token_id: int) -> str:
    return tok.decode(
        [int(token_id)],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def is_content_bearing_token(tok: Any, token_id: int) -> bool:
    text = decoded_token(tok, token_id).strip()
    if not text:
        return False
    # Exclude punctuation/symbol-only rows such as '.', ',', and quotes while
    # keeping alphabetic/numeric words and subword pieces used in names.
    return any(character.isalnum() for character in text)


def rare_topk_answer_rows(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    positions: Sequence[int],
    *,
    max_length: int,
    rows_per_example: int,
) -> Tuple[List[int], Dict[str, Any]]:
    """Select rare content-bearing answer rows using direct forget QAs only.

    Document frequency is computed over all training-visible direct forget
    answers, not retain/evaluation data. For each active example we rank unique
    content-bearing answer tokens by ascending document frequency and preserve
    original answer order as a deterministic tie-breaker.
    """
    if rows_per_example <= 0:
        raise ValueError("rows-per-example must be positive in rare_topk mode")

    all_answer_ids: List[List[int]] = [
        answer_token_ids(tok, instance, max_length=max_length)
        for instance in instances
    ]
    document_frequency: Counter[int] = Counter()
    for token_ids in all_answer_ids:
        document_frequency.update(set(token_ids))

    selected_union: set[int] = set()
    per_example: List[Dict[str, Any]] = []
    fallback_example_count = 0

    for position in positions:
        token_ids = all_answer_ids[position]
        # Deduplicate while preserving first answer occurrence.
        unique_ids = list(dict.fromkeys(token_ids))
        content_ids = [
            token_id
            for token_id in unique_ids
            if is_content_bearing_token(tok, token_id)
        ]
        fallback_used = False
        candidates = content_ids
        if not candidates:
            # This should be very rare for TOFU prose answers. Keep a safe
            # fallback so every active deletion request remains editable.
            candidates = unique_ids
            fallback_used = True
            fallback_example_count += 1
        if not candidates:
            raise RuntimeError(
                f"active forget example at position {position} has no editable answer rows"
            )

        first_position = {token_id: index for index, token_id in enumerate(unique_ids)}
        ranked = sorted(
            candidates,
            key=lambda token_id: (
                int(document_frequency[token_id]),
                first_position[token_id],
                int(token_id),
            ),
        )
        chosen = ranked[:rows_per_example]
        selected_union.update(chosen)

        per_example.append(
            {
                "position": int(position),
                "source_index": int(instances[position].source_index),
                "candidate_row_count": len(candidates),
                "fallback_to_noncontent_rows": fallback_used,
                "selected": [
                    {
                        "token_id": int(token_id),
                        "token": decoded_token(tok, token_id),
                        "document_frequency": int(document_frequency[token_id]),
                        "idf": float(
                            math.log(
                                (len(instances) + 1.0)
                                / (float(document_frequency[token_id]) + 1.0)
                            )
                        ),
                    }
                    for token_id in chosen
                ],
            }
        )

    histogram = Counter(document_frequency.values())
    report = {
        "policy": "rare_topk",
        "rows_per_example": rows_per_example,
        "document_frequency_corpus": "training_visible_direct_forget_answers_only",
        "document_frequency_example_count": len(instances),
        "unique_answer_token_row_count": len(document_frequency),
        "content_filter": "decoded token must contain at least one alphanumeric character",
        "fallback_example_count": fallback_example_count,
        "document_frequency_histogram": {
            str(df): int(count) for df, count in sorted(histogram.items())
        },
        "per_example": per_example,
    }
    return sorted(selected_union), report


def select_answer_rows(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    positions: Sequence[int],
    *,
    max_length: int,
    row_selection: str,
    rows_per_example: int,
) -> Tuple[List[int], Dict[str, Any]]:
    if row_selection == "all":
        selected = answer_rows_for_instances(
            tok, instances, positions, max_length=max_length
        )
        return selected, {
            "policy": "all",
            "rows_per_example": None,
            "document_frequency_corpus": None,
        }
    if row_selection == "rare_topk":
        return rare_topk_answer_rows(
            tok,
            instances,
            positions,
            max_length=max_length,
            rows_per_example=rows_per_example,
        )
    raise ValueError(f"unsupported row-selection policy: {row_selection}")


def metrics(
    nll: torch.Tensor,
    required: torch.Tensor,
    delta: torch.Tensor,
    *,
    target_nll: float,
    tolerance: float,
) -> Dict[str, Any]:
    probabilities = torch.exp(-nll.float())
    active_mask = nll < (target_nll - tolerance)
    buffered_mask = nll < (required.to(nll) - tolerance)
    return {
        "active_forget_instance_count": int(active_mask.sum().detach().cpu()),
        "buffered_forget_constraint_unmet_count": int(
            buffered_mask.sum().detach().cpu()
        ),
        "forget_answer_probability_mean": float(probabilities.mean().detach().cpu()),
        "forget_answer_probability_max": float(probabilities.max().detach().cpu()),
        "minimum_forget_answer_nll": float(nll.min().detach().cpu()),
        "minimum_forget_nll_slack": float(
            (nll - required.to(nll)).min().detach().cpu()
        ),
        "selected_lm_head_delta_norm": float(delta.norm().detach().cpu()),
    }


def priority(row: Dict[str, Any]) -> Tuple[int, int, float, float]:
    return (
        int(row["active_forget_instance_count"]),
        int(row["buffered_forget_constraint_unmet_count"]),
        float(row["forget_answer_probability_max"]),
        float(row["selected_lm_head_delta_norm"]),
    )


def report_instances(
    instances: Sequence[tofu.TOFUAnswerInstance],
    nll: torch.Tensor,
    required: torch.Tensor,
    target_probability: float,
) -> List[Dict[str, Any]]:
    rows = tofu.instance_reports(instances, nll, required)
    for row in rows:
        row["target_answer_probability"] = target_probability
        row["active"] = float(row["answer_probability"]) > target_probability
    return rows


def model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if args.forget_num <= 0 or args.repair_steps <= 0 or args.batch_size <= 0:
        raise ValueError("forget-num, repair-steps and batch-size must be positive")
    if args.repair_lr <= 0 or args.repair_rank <= 0 or args.basis_max_rows <= 0:
        raise ValueError("repair lr/rank/basis controls must be positive")
    if args.target_nll_buffer < 0 or args.delta_l2_lambda < 0:
        raise ValueError("buffer and L2 weight must be non-negative")
    if args.forget_hinge_weight <= 0 or args.hardest_forget_hinge_weight < 0:
        raise ValueError("invalid forget hinge weights")
    if args.rows_per_example <= 0:
        raise ValueError("rows-per-example must be positive")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("max-delta-norm must be finite and non-negative")

    forget_path = Path(args.forget_json).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)

    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoint"
    output_dir.mkdir(parents=True, exist_ok=True)

    tok_for_data = AutoTokenizer.from_pretrained(args.model_path)
    if tok_for_data.pad_token is None:
        tok_for_data.pad_token = tok_for_data.eos_token
    instances, source_indices = load_forget_instances(
        forget_path, tok_for_data, args.forget_num
    )

    model, tok = gagd.load_model_and_tokenizer(model_args(args), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = input_weight.data_ptr()
    input_version = input_weight._version
    device = gagd.first_device(model)

    baseline_nll = tofu.score_answer_instances(
        model,
        tok,
        instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    required_nll, initially_active_mask, target_nll = native.build_required_forget_nll(
        baseline_nll,
        target_probability=args.target_forget_answer_probability,
        target_nll_buffer=args.target_nll_buffer,
    )
    active_positions = (
        initially_active_mask.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
    )
    selected_ids, row_selection_report = select_answer_rows(
        tok,
        instances,
        active_positions,
        max_length=args.max_length,
        row_selection=args.row_selection,
        rows_per_example=args.rows_per_example,
    )
    if active_positions and not selected_ids:
        raise RuntimeError("active forget answers produced no editable LM-head rows")

    selected_tensor = torch.tensor(
        selected_ids, dtype=torch.long, device=output_weight.device
    )
    baseline_rows = (
        output_weight.index_select(0, selected_tensor).detach().clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )

    selected_report = {
        "initially_active_forget_instance_count": len(active_positions),
        "active_positions": active_positions,
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "selected_lm_head_tokens": {
            str(token_id): decoded_token(tok, token_id) for token_id in selected_ids
        },
        "selection_source": "initially_active_direct_forget_answers_only",
        "row_selection": args.row_selection,
        "rows_per_example": (
            args.rows_per_example if args.row_selection == "rare_topk" else None
        ),
        "row_selection_report": row_selection_report,
        "retain_or_utility_rows_consulted": False,
        "paraphrases_consulted": False,
    }
    write_json(output_dir / "selected_lm_head_rows.json", selected_report)
    write_json(
        output_dir / "forget_instances_before.json",
        report_instances(
            instances,
            baseline_nll,
            required_nll,
            args.target_forget_answer_probability,
        ),
    )

    zero_delta = torch.zeros_like(baseline_rows, dtype=torch.float32)
    baseline_metrics = metrics(
        baseline_nll,
        required_nll,
        zero_delta,
        target_nll=target_nll,
        tolerance=args.comparison_tolerance,
    )
    write_json(output_dir / "baseline_local_metrics.json", baseline_metrics)

    best_delta = zero_delta.detach().cpu().clone()
    best_metrics = dict(baseline_metrics)
    best_step = 0
    logs: List[Dict[str, Any]] = []
    steps_completed = 0
    stopped_early = not active_positions
    direction_basis: torch.Tensor | None = None

    if active_positions:
        caches = tofu.build_answer_delta_caches(
            model,
            tok,
            instances,
            selected_ids,
            device,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        packed = tofu.pack_answer_delta_caches(caches)
        active_caches = [caches[position] for position in active_positions]
        direction_basis = tofu.limited_hidden_row_basis(
            active_caches,
            max_rank=args.repair_rank,
            max_rows=args.basis_max_rows,
        )
        if direction_basis.shape[0] == 0:
            raise RuntimeError("active answer hidden states produced a zero-rank repair basis")
        delta_module = active.SelectedRowDelta(
            len(selected_ids),
            output_weight.shape[1],
            direction_basis=direction_basis,
            retained_basis=None,
            device=device,
        )
        optimizer = active.make_repair_optimizer(
            delta_module, args.repair_optimizer, args.repair_lr
        )

        for step in range(1, args.repair_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            delta = delta_module.effective_delta()
            current = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
            errors = torch.relu(required_nll.to(current) - current)
            loss = (
                args.forget_hinge_weight * errors.square().mean()
                + args.hardest_forget_hinge_weight * errors.square().max()
                + args.delta_l2_lambda * delta.square().sum()
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite Stage2 loss at step {step}")
            loss.backward()
            optimizer.step()
            norm_before, norm_after, norm_projected = active.constrain_effective_delta_norm(
                delta_module, args.max_delta_norm
            )
            steps_completed = step

            with torch.no_grad():
                candidate_delta = delta_module.effective_delta()
                candidate_nll = tofu.answer_nlls_from_packed_delta_cache(
                    packed, candidate_delta
                )
                candidate_metrics = metrics(
                    candidate_nll,
                    required_nll,
                    candidate_delta,
                    target_nll=target_nll,
                    tolerance=args.comparison_tolerance,
                )
                if priority(candidate_metrics) < priority(best_metrics):
                    best_delta = candidate_delta.detach().cpu().clone()
                    best_metrics = dict(candidate_metrics)
                    best_step = step

            if step == 1 or step % args.log_every == 0:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "forget_hinge": float(errors.square().mean().detach().cpu()),
                    "hardest_forget_hinge": float(errors.square().max().detach().cpu()),
                    "delta_l2": float(delta.square().sum().detach().cpu()),
                    "delta_norm_before_projection": norm_before,
                    "delta_norm": norm_after,
                    "delta_norm_projected": norm_projected,
                    **candidate_metrics,
                }
                logs.append(row)
                print(
                    f"step={step} active={candidate_metrics['active_forget_instance_count']} "
                    f"buffered={candidate_metrics['buffered_forget_constraint_unmet_count']} "
                    f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g}"
                )

            if (
                args.stop_when_all_satisfied
                and candidate_metrics["active_forget_instance_count"] == 0
                and candidate_metrics["buffered_forget_constraint_unmet_count"] == 0
            ):
                stopped_early = True
                if priority(candidate_metrics) <= priority(best_metrics):
                    best_delta = candidate_delta.detach().cpu().clone()
                    best_metrics = dict(candidate_metrics)
                    best_step = step
                break

    write_jsonl(output_dir / "repair_log.jsonl", logs)
    qualified_before_materialization = (
        best_metrics["active_forget_instance_count"] == 0
        and best_metrics["buffered_forget_constraint_unmet_count"] == 0
    )
    if not qualified_before_materialization and not args.save_best_effort:
        write_json(
            output_dir / "repair_summary.json",
            {
                "status": "FAILED_NO_QUALIFIED_FORGET_ONLY_CANDIDATE",
                "best_step": best_step,
                "best_metrics": best_metrics,
                "steps_completed": steps_completed,
                "row_selection": args.row_selection,
                "rows_per_example": args.rows_per_example,
                "selected_lm_head_row_count": len(selected_ids),
            },
        )
        raise RuntimeError(
            "No Stage2 candidate satisfied every direct forget target+buffer constraint"
        )

    tofu.set_selected_lm_head_rows(
        output_weight,
        selected_ids,
        baseline_rows,
        best_delta,
    )
    if input_weight.data_ptr() != input_pointer or input_weight._version != input_version:
        raise RuntimeError("Stage2 modified input embeddings")

    materialized_nll = tofu.score_answer_instances(
        model,
        tok,
        instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    materialized_metrics = metrics(
        materialized_nll,
        required_nll,
        best_delta.to(device),
        target_nll=target_nll,
        tolerance=args.comparison_tolerance,
    )
    final_target_met = materialized_metrics["active_forget_instance_count"] == 0
    if not final_target_met and not args.save_best_effort:
        write_json(
            output_dir / "repair_summary.json",
            {
                "status": "FAILED_BF16_MATERIALIZATION_FORGET_TARGET",
                "best_step": best_step,
                "best_metrics_cached": best_metrics,
                "materialized_metrics": materialized_metrics,
                "row_selection": args.row_selection,
                "rows_per_example": args.rows_per_example,
                "selected_lm_head_row_count": len(selected_ids),
            },
        )
        raise RuntimeError("materialized Stage2 checkpoint misses direct forget target")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tok.save_pretrained(checkpoint_dir)
    write_json(
        output_dir / "forget_instances_after.json",
        report_instances(
            instances,
            materialized_nll,
            required_nll,
            args.target_forget_answer_probability,
        ),
    )

    config = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
        "model_path": args.model_path,
        "forget_json": str(forget_path),
        "seed": args.seed,
        "forget_num": args.forget_num,
        "retain_num": 0,
        "paraphrases_used_during_repair": False,
        "perturbed_answers_used_during_repair": False,
        "real_authors_used_during_repair": False,
        "world_facts_used_during_repair": False,
        "target_forget_answer_probability": args.target_forget_answer_probability,
        "target_nll_buffer": args.target_nll_buffer,
        "repair_steps": args.repair_steps,
        "repair_lr": args.repair_lr,
        "repair_optimizer": args.repair_optimizer,
        "repair_rank_requested": args.repair_rank,
        "basis_max_rows": args.basis_max_rows,
        "max_delta_norm": args.max_delta_norm,
        "row_selection": args.row_selection,
        "rows_per_example": (
            args.rows_per_example if args.row_selection == "rare_topk" else None
        ),
        "forget_hinge_weight": args.forget_hinge_weight,
        "hardest_forget_hinge_weight": args.hardest_forget_hinge_weight,
        "delta_l2_lambda": args.delta_l2_lambda,
        "selected_lm_head_row_count": len(selected_ids),
        "stage2_visible_source_indices": source_indices,
        "checkpoint": str(checkpoint_dir),
    }
    write_json(output_dir / "config_used.json", config)
    summary = {
        "status": "PASS" if final_target_met else "BEST_EFFORT",
        "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
        "best_step": best_step,
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "initially_active_forget_instance_count": len(active_positions),
        "selected_lm_head_row_count": len(selected_ids),
        "row_selection": args.row_selection,
        "rows_per_example": (
            args.rows_per_example if args.row_selection == "rare_topk" else None
        ),
        "repair_rank_requested": args.repair_rank,
        "repair_rank_actual": (
            int(direction_basis.shape[0])
            if active_positions and direction_basis is not None
            else 0
        ),
        "best_metrics_cached": best_metrics,
        "materialized_metrics": materialized_metrics,
        "training_data_access": {
            "direct_forget_qas": args.forget_num,
            "retain95": 0,
            "paraphrases": 0,
            "perturbed_answers": 0,
            "real_authors": 0,
            "world_facts": 0,
        },
        "checkpoint": str(checkpoint_dir),
    }
    write_json(output_dir / "repair_summary.json", summary)
    print(f"TOFU locked Stage2 checkpoint: {checkpoint_dir}")
    print(
        f"row selection: {args.row_selection}; selected rows={len(selected_ids)}; "
        f"rows/example={args.rows_per_example if args.row_selection == 'rare_topk' else 'all'}"
    )
    print(
        f"direct forget max probability: {materialized_metrics['forget_answer_probability_max']:.8g}; "
        f"target={args.target_forget_answer_probability:.8g}"
    )


if __name__ == "__main__":
    main()
