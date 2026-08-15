#!/usr/bin/env python3
"""SURE-TOFU Stage 1B v3: progressive sparse sensitive-row rank-0 forgetting.

V3 changes only the sensitive answer-row selection policy.  The direct-forget
requirements, exact Full-TOFU Base restoration of non-sensitive rows, rank-0
optimizer, and boundary bisection are inherited from the validated v2 path.

Protocol:
1. Freeze requirements from the Stage1A checkpoint on the same 50 visible QAs.
2. Rank each answer's unique token rows using ONLY the visible forget answers:
   content-bearing rows first, then ascending answer-document frequency, then
   original token order.  Punctuation/symbol rows remain at the end as a safe
   fallback so the ranking can eventually cover every answer row.
3. Start with the top K rows per currently violating QA (default K=3).
4. Restore every other visible answer-token row exactly to Full-TOFU Base in
   BOTH input embeddings and LM head.
5. Try unrestricted selected-row rank-0 forgetting.  At its first feasible
   optimizer crossing, bisect to a near-boundary solution.
6. Only if that restricted problem is infeasible after the full repair budget,
   add the next P ranked rows (default P=1) for each still-failing QA and retry.
7. Stop at the first feasible sparse row set.  Reassert all non-sensitive rows
   to Base, audit all 50 direct constraints after materialization, then freeze.

No retain95, paraphrases, same-author heldout, real-authors, world-facts, PPL,
or final-evaluation metric is loaded or used for selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_active_forget_repair as native
import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_rank0_forget as old
import tofu_sure_rank0_forget_restored as v2


METHOD = "SURE-TOFU-progressive-sparse-rank0-v3"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Frozen Stage1A checkpoint")
    p.add_argument("--reference-model-path", required=True, help="Original Full-TOFU checkpoint")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--initial-rows-per-example", type=int, default=3)
    p.add_argument("--promotion-rows-per-example", type=int, default=1)
    p.add_argument("--max-promotion-rounds", type=int, default=64)
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
    p.add_argument("--boundary-bisection-steps", type=int, default=30)
    p.add_argument("--boundary-safety-fraction", type=float, default=0.002)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_progressive_rankings(
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    *,
    max_length: int,
) -> tuple[List[List[int]], Dict[str, Any]]:
    """Return a complete deterministic per-example row ranking.

    Training-visible answer document frequency is the only cross-example signal.
    Content rows are preferred, but non-content answer rows are retained at the
    end of each ranking so progressive promotion can fail closed by eventually
    exposing every answer-token row if necessary.
    """
    answer_ids = [
        locked.answer_token_ids(tok, instance, max_length=max_length)
        for instance in instances
    ]
    df: Counter[int] = Counter()
    for token_ids in answer_ids:
        df.update(set(token_ids))

    rankings: List[List[int]] = []
    per_example: List[Dict[str, Any]] = []
    for position, token_ids in enumerate(answer_ids):
        unique_ids = list(dict.fromkeys(int(x) for x in token_ids))
        if not unique_ids:
            raise RuntimeError(f"forget example {position} has no answer-token rows")
        first = {token_id: idx for idx, token_id in enumerate(unique_ids)}
        ranked = sorted(
            unique_ids,
            key=lambda token_id: (
                0 if locked.is_content_bearing_token(tok, token_id) else 1,
                int(df[token_id]),
                first[token_id],
                int(token_id),
            ),
        )
        rankings.append(ranked)
        per_example.append(
            {
                "position": position,
                "source_index": int(instances[position].source_index),
                "answer_unique_row_count": len(unique_ids),
                "content_row_count": sum(
                    1 for token_id in unique_ids
                    if locked.is_content_bearing_token(tok, token_id)
                ),
                "ranking": [
                    {
                        "token_id": int(token_id),
                        "token": locked.decoded_token(tok, token_id),
                        "content_bearing": bool(locked.is_content_bearing_token(tok, token_id)),
                        "document_frequency": int(df[token_id]),
                    }
                    for token_id in ranked
                ],
            }
        )

    histogram = Counter(df.values())
    return rankings, {
        "policy": "progressive_rare_content_first",
        "corpus": "training_visible_direct_forget_answers_only",
        "ranking_key": [
            "content-bearing before punctuation/symbol-only",
            "ascending answer document frequency",
            "original answer-token order",
            "token id deterministic tie-break",
        ],
        "document_frequency_histogram": {
            str(k): int(v) for k, v in sorted(histogram.items())
        },
        "per_example": per_example,
    }


def add_next_rows(
    sensitive: set[int],
    rankings: Sequence[Sequence[int]],
    positions: Sequence[int],
    count_per_example: int,
) -> Dict[int, List[int]]:
    """Promote up to count_per_example unseen ranked rows for each position."""
    promoted_by_position: Dict[int, List[int]] = {}
    for position in positions:
        chosen: List[int] = []
        for token_id in rankings[position]:
            token_id = int(token_id)
            if token_id in sensitive or token_id in chosen:
                continue
            chosen.append(token_id)
            if len(chosen) >= count_per_example:
                break
        if chosen:
            promoted_by_position[int(position)] = chosen
            sensitive.update(chosen)
    return promoted_by_position


def main() -> None:
    a = parse_args()
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")
    if a.forget_num <= 0 or a.repair_steps <= 0 or a.batch_size <= 0:
        raise ValueError("forget-num, repair-steps and batch-size must be positive")
    if a.initial_rows_per_example <= 0 or a.promotion_rows_per_example <= 0:
        raise ValueError("progressive row counts must be positive")
    if a.max_promotion_rounds <= 0 or a.boundary_bisection_steps <= 0:
        raise ValueError("promotion/bisection controls must be positive")
    if a.repair_lr <= 0 or a.delta_l2_lambda < 0:
        raise ValueError("invalid optimizer controls")
    if not 0.0 <= a.boundary_safety_fraction <= 0.1:
        raise ValueError("boundary-safety-fraction must lie in [0,0.1]")

    forget_path = Path(a.forget_json).resolve()
    reference_path = Path(a.reference_model_path).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)
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
    instances, source_indices = locked.load_forget_instances(
        forget_path, tok_data, a.forget_num
    )

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    input_weight = model.get_input_embeddings().weight
    output_weight = output_embeddings.weight
    device = gagd.first_device(model)

    # Freeze direct-forget requirements before any v3 restoration/selection.
    stage1a_nll = tofu.score_answer_instances(
        model, tok, instances, device,
        batch_size=a.batch_size, max_length=a.max_length,
    ).detach().float()
    required_nll, _, target_nll = native.build_required_forget_nll(
        stage1a_nll,
        target_probability=a.target_forget_answer_probability,
        target_nll_buffer=a.target_nll_buffer,
    )
    locked.write_json(
        root / "forget_instances_stage1a_before_row_restore.json",
        locked.report_instances(
            instances, stage1a_nll, required_nll,
            a.target_forget_answer_probability,
        ),
    )

    answer_ids = old.all_answer_rows(tok, instances, max_length=a.max_length)
    if not answer_ids:
        raise RuntimeError("visible direct answers produced no vocabulary rows")
    ids_i = torch.tensor(answer_ids, dtype=torch.long, device=input_weight.device)
    ids_o = ids_i.to(output_weight.device)
    stage1a_input_rows = input_weight.index_select(0, ids_i).detach().cpu().clone()
    stage1a_output_rows = output_weight.index_select(0, ids_o).detach().cpu().clone()
    base_input_rows, base_output_rows = v2.load_reference_rows(
        str(reference_path), answer_ids, a.dtype
    )

    rankings, ranking_report = build_progressive_rankings(
        tok, instances, max_length=a.max_length
    )
    initial_violating = old.violating_sequence_positions(
        stage1a_nll, required_nll, a.comparison_tolerance
    )
    sensitive_ids: set[int] = set()
    initial_by_position = add_next_rows(
        sensitive_ids,
        rankings,
        initial_violating,
        a.initial_rows_per_example,
    )
    if initial_violating and not sensitive_ids:
        raise RuntimeError("progressive selector produced no initial sensitive rows")

    write_json(
        root / "progressive_row_ranking.json",
        {
            **ranking_report,
            "initial_rows_per_example": a.initial_rows_per_example,
            "promotion_rows_per_example": a.promotion_rows_per_example,
            "initial_violating_positions": initial_violating,
            "initial_selected_by_position": {
                str(k): v for k, v in sorted(initial_by_position.items())
            },
            "initial_sensitive_unique_row_count": len(sensitive_ids),
            "all_visible_answer_unique_row_count": len(answer_ids),
            "retain_or_heldout_data_consulted": False,
        },
    )

    attempts: List[Dict[str, Any]] = []
    final_attempt: v2.RepairAttempt | None = None
    final_baseline_rows: torch.Tensor | None = None

    for promotion_round in range(a.max_promotion_rounds):
        old.apply_answer_row_policy(
            input_weight,
            output_weight,
            answer_ids,
            sorted(sensitive_ids),
            stage1a_input_rows,
            stage1a_output_rows,
            base_input_rows,
            base_output_rows,
        )
        selected_ids = sorted(sensitive_ids)
        non_sensitive_ids = sorted(set(answer_ids) - sensitive_ids)
        selected_tensor = torch.tensor(
            selected_ids, dtype=torch.long, device=output_weight.device
        )
        baseline_rows = (
            output_weight.index_select(0, selected_tensor).detach().clone()
            if selected_ids
            else output_weight.new_empty((0, output_weight.shape[1]))
        )

        pre_nll = tofu.score_answer_instances(
            model, tok, instances, device,
            batch_size=a.batch_size, max_length=a.max_length,
        ).detach().float()
        pre_violating = old.violating_sequence_positions(
            pre_nll, required_nll, a.comparison_tolerance
        )
        print(
            f"v3-round={promotion_round} sensitive={len(selected_ids)} "
            f"non_sensitive={len(non_sensitive_ids)} "
            f"pre_rank0_violating={len(pre_violating)}"
        )

        attempt = v2.run_restricted_rank0(
            model, tok, instances, selected_ids,
            required_nll, target_nll, a, device,
        )
        attempt_report: Dict[str, Any] = {
            "round": promotion_round,
            "sensitive_row_count": len(selected_ids),
            "non_sensitive_row_count": len(non_sensitive_ids),
            "pre_rank0_violating_sequence_count": len(pre_violating),
            "repair_feasible": attempt.feasible,
            "repair_step": attempt.step,
            "optimizer_crossing_step": attempt.optimizer_crossing_step,
            "boundary_alpha": attempt.boundary_alpha,
            "candidate_delta_norm": float(attempt.delta_cpu.norm().item()),
            "candidate_forget_answer_probability_max": float(
                torch.exp(-attempt.nll_cpu).max().item()
            ),
            "candidate_minimum_nll_slack": float(
                (attempt.nll_cpu - required_nll.detach().cpu()).min().item()
            ),
        }
        attempts.append(attempt_report)

        if attempt.feasible:
            final_attempt = attempt
            final_baseline_rows = baseline_rows
            break

        residual_positions = old.violating_sequence_positions(
            attempt.nll_cpu,
            required_nll.detach().cpu(),
            a.comparison_tolerance,
        )
        promoted_by_position = add_next_rows(
            sensitive_ids,
            rankings,
            residual_positions,
            a.promotion_rows_per_example,
        )
        promoted_ids = sorted(
            {token_id for ids in promoted_by_position.values() for token_id in ids}
        )
        attempt_report["residual_violating_positions"] = residual_positions
        attempt_report["promoted_by_position"] = {
            str(k): v for k, v in sorted(promoted_by_position.items())
        }
        attempt_report["promoted_token_ids"] = promoted_ids
        attempt_report["promoted_row_count"] = len(promoted_ids)
        if not promoted_ids:
            write_json(root / "promotion_attempts.json", attempts)
            raise RuntimeError(
                "v3 restricted rank-0 failed and residual QAs have no ranked rows left to promote"
            )
    else:
        write_json(root / "promotion_attempts.json", attempts)
        raise RuntimeError("v3 exceeded max progressive promotion rounds")

    assert final_attempt is not None
    assert final_baseline_rows is not None
    selected_ids = sorted(sensitive_ids)
    non_sensitive_ids = sorted(set(answer_ids) - sensitive_ids)

    # Deterministically reconstruct the final sparse row policy and materialize
    # only the final rank-0 delta.
    old.apply_answer_row_policy(
        input_weight,
        output_weight,
        answer_ids,
        selected_ids,
        stage1a_input_rows,
        stage1a_output_rows,
        base_input_rows,
        base_output_rows,
    )
    tofu.set_selected_lm_head_rows(
        output_weight,
        selected_ids,
        final_baseline_rows,
        final_attempt.delta_cpu,
    )

    id_to_position = {int(token_id): pos for pos, token_id in enumerate(answer_ids)}
    if non_sensitive_ids:
        ns_pos = torch.tensor(
            [id_to_position[token_id] for token_id in non_sensitive_ids],
            dtype=torch.long,
        )
        ns_i = torch.tensor(
            non_sensitive_ids, dtype=torch.long, device=input_weight.device
        )
        ns_o = ns_i.to(output_weight.device)
        input_weight.index_copy_(
            0,
            ns_i,
            base_input_rows.index_select(0, ns_pos).to(
                device=input_weight.device, dtype=input_weight.dtype
            ),
        )
        output_weight.index_copy_(
            0,
            ns_o,
            base_output_rows.index_select(0, ns_pos).to(
                device=output_weight.device, dtype=output_weight.dtype
            ),
        )

    materialized_nll = tofu.score_answer_instances(
        model, tok, instances, device,
        batch_size=a.batch_size, max_length=a.max_length,
    ).detach().float()
    materialized_metrics = locked.metrics(
        materialized_nll,
        required_nll,
        final_attempt.delta_cpu.to(device),
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    if not v2.feasible_nll(
        materialized_nll, required_nll, a.comparison_tolerance
    ):
        write_json(
            root / "repair_summary.json",
            {
                "status": "FAILED_DTYPE_MATERIALIZATION",
                "materialized_metrics": materialized_metrics,
                "promotion_attempts": attempts,
            },
        )
        raise RuntimeError("v3 near-boundary solution lost feasibility after materialization")

    input_restore_error = old.answer_row_restoration_error(
        input_weight, answer_ids, non_sensitive_ids, base_input_rows
    )
    output_restore_error = old.answer_row_restoration_error(
        output_weight, answer_ids, non_sensitive_ids, base_output_rows
    )
    if input_restore_error != 0.0 or output_restore_error != 0.0:
        raise RuntimeError(
            f"v3 non-sensitive rows are not exact Base: "
            f"input={input_restore_error} output={output_restore_error}"
        )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)
    locked.write_json(
        root / "forget_instances_after.json",
        locked.report_instances(
            instances, materialized_nll, required_nll,
            a.target_forget_answer_probability,
        ),
    )
    write_json(root / "promotion_attempts.json", attempts)
    write_json(
        root / "answer_row_restoration.json",
        {
            "policy": "v3 progressive sparse sensitive rows; every non-sensitive visible answer row exact Full-TOFU Base in embedding and LM head",
            "initial_sensitive_row_count": attempts[0]["sensitive_row_count"],
            "all_visible_answer_row_count": len(answer_ids),
            "sensitive_answer_row_count": len(selected_ids),
            "sensitive_answer_token_ids": selected_ids,
            "non_sensitive_answer_row_count": len(non_sensitive_ids),
            "non_sensitive_answer_token_ids": non_sensitive_ids,
            "promotion_rounds": attempts,
            "promotion_trigger": "only after a full restricted rank-0 attempt is infeasible",
            "retain_or_heldout_data_consulted": False,
        },
    )

    minimum_slack = float(
        (materialized_nll - required_nll).min().detach().cpu().item()
    )
    summary = {
        "status": "PASS",
        "method": METHOD,
        "protocol": PROTOCOL,
        "best_step": final_attempt.step,
        "optimizer_crossing_step": final_attempt.optimizer_crossing_step,
        "boundary_alpha": final_attempt.boundary_alpha,
        "repair_rank_requested": 0,
        "repair_rank_actual": 0,
        "repair_rank_semantics": "unrestricted selected-row delta on progressively selected sparse sensitive answer rows",
        "all_visible_answer_row_count": len(answer_ids),
        "initial_sensitive_answer_row_count": attempts[0]["sensitive_row_count"],
        "sensitive_answer_row_count": len(selected_ids),
        "non_sensitive_answer_row_count": len(non_sensitive_ids),
        "selected_lm_head_row_count": len(selected_ids),
        "selected_lm_head_token_ids": selected_ids,
        "non_sensitive_input_base_max_abs_error": input_restore_error,
        "non_sensitive_output_base_max_abs_error": output_restore_error,
        "materialized_metrics": materialized_metrics,
        "minimum_materialized_nll_slack": minimum_slack,
        "promotion_attempts": attempts,
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
            "schema_version": 4,
            "method": METHOD,
            "protocol": PROTOCOL,
            "model_path": a.model_path,
            "reference_model_path": str(reference_path),
            "forget_json": str(forget_path),
            "seed": a.seed,
            "forget_num": a.forget_num,
            "initial_rows_per_example": a.initial_rows_per_example,
            "promotion_rows_per_example": a.promotion_rows_per_example,
            "max_promotion_rounds": a.max_promotion_rounds,
            "target_forget_answer_probability": a.target_forget_answer_probability,
            "target_nll_buffer": a.target_nll_buffer,
            "repair_steps": a.repair_steps,
            "repair_lr": a.repair_lr,
            "repair_optimizer": a.repair_optimizer,
            "repair_rank": 0,
            "delta_l2_lambda": a.delta_l2_lambda,
            "boundary_bisection_steps": a.boundary_bisection_steps,
            "boundary_safety_fraction": a.boundary_safety_fraction,
            "sensitivity_rule": "rare/content-first top-K per violating visible QA; add next ranked rows only after restricted rank-0 infeasibility",
            "answer_row_restoration": "all non-sensitive visible answer rows exact Full-TOFU Base in input embedding and LM head",
            "stage1b_visible_source_indices": source_indices,
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print(
        "SURE-TOFU Stage1B-v3 PASS "
        f"initial_sensitive={attempts[0]['sensitive_row_count']} "
        f"final_sensitive={len(selected_ids)} non_sensitive={len(non_sensitive_ids)} "
        f"rounds={len(attempts)} cross_step={final_attempt.optimizer_crossing_step} "
        f"alpha={final_attempt.boundary_alpha} "
        f"norm={materialized_metrics['selected_lm_head_delta_norm']:.6g} "
        f"max_prob={materialized_metrics['forget_answer_probability_max']:.8g} "
        f"min_slack={minimum_slack:.6g}"
    )
    print(f"checkpoint: {ckpt}")


if __name__ == "__main__":
    main()
