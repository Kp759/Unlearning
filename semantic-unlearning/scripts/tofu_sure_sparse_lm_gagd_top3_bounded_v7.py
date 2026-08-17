#!/usr/bin/env python3
"""SURE-TOFU V7 bounded-top3 Stage 1.

Controlled follow-up to the destructive unbounded top-3 run.

For each of the 50 training-visible forget QAs we select the same top-K
(default 3) rare/content-bearing answer-token LM-head rows.  Only those rows
are editable; transformer blocks, all input embeddings, and every other
LM-head row remain exact Full-TOFU Base.

The key change is bounded GA.  At each answer position whose true target token
is selected sensitive, Stage 1 only asks for a fixed probability reduction
relative to Base (default 10x):

    NLL_current >= NLL_base + log(suppression_factor).

The GA term is a squared hinge and becomes exactly zero once the requested
suppression is reached.  Stage 1 selection/stopping is based on these bounded
sensitive-token constraints, NOT on whether complete answers already meet the
final 3e-4 deletion threshold.  Same-prompt non-target KL remains the GD
preservation term on all answer contexts from the same 50 visible QAs.

The final full-answer 3e-4 constraint is deliberately left to Stage 2.
No retain, paraphrase, same-author holdout, PPL, real-author, or world-fact data
is loaded or used by this stage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_forget_only_active_repair as locked
import tofu_gagd_neighborhood_confidence as tofu
from tofu_sure_rowspecific_null_v6_stable import stable_same_prompt_non_target_kl


METHOD = "SURE-TOFU-v7-top3-bounded-sensitive-GAGD"
PROTOCOL = "tofu_author_balanced_forget_only_locked_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True, help="Protected Full-TOFU Base")
    p.add_argument("--forget-json", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--ga-weight", type=float, default=2.0)
    p.add_argument("--gd-weight", type=float, default=1.0)
    p.add_argument("--delta-l2-lambda", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--target-forget-answer-probability", type=float, default=3e-4)
    p.add_argument("--sensitive-rows-per-example", type=int, default=3)
    p.add_argument(
        "--suppression-factor",
        type=float,
        default=10.0,
        help="Required Base/current probability ratio at selected sensitive target positions.",
    )
    p.add_argument(
        "--stop-when-bounded-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--comparison-tolerance", type=float, default=1e-6)
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


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def equal_example_non_target_kl(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    if not caches:
        return delta_rows.new_zeros(())
    return torch.stack(
        [stable_same_prompt_non_target_kl([cache], delta_rows) for cache in caches]
    ).mean()


def token_nll_and_sensitive_mask(
    cache: tofu.TOFUAnswerDeltaCache,
    delta_rows: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    corrections = cache.hidden.float() @ delta_rows.transpose(0, 1)
    log_shift = active._log_partition_shift(cache.selected_probs.float(), corrections)
    target_correction = corrections.new_zeros(corrections.shape[0])
    selected_mask = cache.target_selected_columns.ge(0)
    if selected_mask.any():
        target_correction[selected_mask] = corrections[
            selected_mask,
            cache.target_selected_columns[selected_mask],
        ]
    token_nll = cache.base_token_nll.float() + log_shift - target_correction
    return token_nll, selected_mask


def bounded_sensitive_terms(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
    *,
    log_suppression: float,
    tolerance: float,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Equal-example bounded GA hinge and diagnostic statistics."""
    per_example_loss: List[torch.Tensor] = []
    current_probs: List[torch.Tensor] = []
    base_probs: List[torch.Tensor] = []
    slacks: List[torch.Tensor] = []
    total_positions = 0
    unmet_positions = 0
    examples_with_positions = 0
    examples_with_unmet = 0

    for cache in caches:
        token_nll, selected_mask = token_nll_and_sensitive_mask(cache, delta_rows)
        if not selected_mask.any():
            raise RuntimeError(
                "bounded top3 selector produced a QA with no selected true-target positions"
            )
        current = token_nll[selected_mask]
        base = cache.base_token_nll.float()[selected_mask]
        required = base + float(log_suppression)
        error = torch.relu(required - current)
        slack = current - required

        per_example_loss.append(error.square().mean())
        current_probs.append(torch.exp(-current))
        base_probs.append(torch.exp(-base))
        slacks.append(slack)

        local_unmet = int((slack < -tolerance).sum().detach().cpu())
        total_positions += int(selected_mask.sum().detach().cpu())
        unmet_positions += local_unmet
        examples_with_positions += 1
        examples_with_unmet += int(local_unmet > 0)

    if not per_example_loss:
        raise RuntimeError("no sensitive target positions available for bounded GA")

    joined_current = torch.cat(current_probs)
    joined_base = torch.cat(base_probs)
    joined_slack = torch.cat(slacks)
    probability_ratio = joined_current / joined_base.clamp_min(torch.finfo(torch.float32).tiny)
    stats = {
        "bounded_sensitive_target_position_count": total_positions,
        "bounded_unmet_target_position_count": unmet_positions,
        "bounded_unmet_example_count": examples_with_unmet,
        "bounded_ga_example_coverage": examples_with_positions,
        "bounded_minimum_nll_slack": float(joined_slack.min().detach().cpu()),
        "bounded_mean_nll_slack": float(joined_slack.mean().detach().cpu()),
        "sensitive_probability_ratio_to_base_mean": float(probability_ratio.mean().detach().cpu()),
        "sensitive_probability_ratio_to_base_max": float(probability_ratio.max().detach().cpu()),
        "sensitive_target_probability_mean": float(joined_current.mean().detach().cpu()),
        "sensitive_target_probability_max": float(joined_current.max().detach().cpu()),
    }
    return torch.stack(per_example_loss).mean(), stats


def candidate_priority(
    bounded_hinge: float,
    bounded_stats: Dict[str, Any],
    kl_value: float,
    delta_norm: float,
) -> Tuple[int, int, float, float, float]:
    # Stage 1 is selected by its own bounded objective, never by final full-answer
    # deletion efficacy.  Once all bounded constraints are met, KL then norm break ties.
    return (
        int(bounded_stats["bounded_unmet_example_count"]),
        int(bounded_stats["bounded_unmet_target_position_count"]),
        float(bounded_hinge),
        float(kl_value),
        float(delta_norm),
    )


def main() -> None:
    a = parse_args()
    if a.forget_num != 50:
        raise ValueError("V7 bounded-top3 locked experiment fixes --forget-num=50")
    if a.steps <= 0 or a.lr <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("steps/lr/batch-size/max-length must be positive")
    if a.sensitive_rows_per_example <= 0:
        raise ValueError("sensitive-rows-per-example must be positive")
    if not math.isfinite(a.suppression_factor) or a.suppression_factor <= 1.0:
        raise ValueError("suppression-factor must be finite and > 1")
    if a.ga_weight <= 0 or a.gd_weight < 0 or a.delta_l2_lambda < 0 or a.grad_clip < 0:
        raise ValueError("invalid GA/GD/regularization controls")
    if not 0.0 < a.target_forget_answer_probability < 1.0:
        raise ValueError("target-forget-answer-probability must lie in (0,1)")

    forget_path = Path(a.forget_json).resolve()
    if not forget_path.is_file():
        raise FileNotFoundError(forget_path)

    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    root = gagd.resolve_output_path(a.output_dir)
    ckpt = root / "checkpoint"
    root.mkdir(parents=True, exist_ok=True)

    data_tok = AutoTokenizer.from_pretrained(a.model_path)
    if data_tok.pad_token is None:
        data_tok.pad_token = data_tok.eos_token
    instances, source_indices = locked.load_forget_instances(
        forget_path, data_tok, a.forget_num
    )

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = int(input_weight.data_ptr())
    input_version = int(input_weight._version)
    device = gagd.first_device(model)

    all_positions = list(range(len(instances)))
    sensitive_ids, selection_report = locked.rare_topk_answer_rows(
        tok,
        instances,
        all_positions,
        max_length=a.max_length,
        rows_per_example=a.sensitive_rows_per_example,
    )
    if not sensitive_ids:
        raise RuntimeError("bounded top3 selector produced no sensitive LM-head rows")

    selected_tensor = torch.tensor(
        sensitive_ids, dtype=torch.long, device=output_weight.device
    )
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    caches = tofu.build_answer_delta_caches(
        model,
        tok,
        instances,
        sensitive_ids,
        device,
        batch_size=a.batch_size,
        max_length=a.max_length,
    )
    packed = tofu.pack_answer_delta_caches(caches)
    zero = torch.zeros(
        (len(sensitive_ids), int(output_weight.shape[1])),
        dtype=torch.float32,
        device=device,
    )

    log_suppression = math.log(a.suppression_factor)
    base_bounded_hinge_t, base_bounded_stats = bounded_sensitive_terms(
        caches,
        zero,
        log_suppression=log_suppression,
        tolerance=a.comparison_tolerance,
    )
    if int(base_bounded_stats["bounded_ga_example_coverage"]) != len(instances):
        raise RuntimeError(
            "bounded GA coverage mismatch: "
            f"{base_bounded_stats['bounded_ga_example_coverage']}/{len(instances)} visible QAs"
        )

    base_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero).detach()
    final_target_nll = -math.log(a.target_forget_answer_probability)
    full_answer_required_nll = base_nll.new_full(base_nll.shape, final_target_nll)
    base_full_answer_metrics = locked.metrics(
        base_nll,
        full_answer_required_nll,
        zero,
        target_nll=final_target_nll,
        tolerance=a.comparison_tolerance,
    )

    module = active.SelectedRowDelta(
        len(sensitive_ids),
        int(output_weight.shape[1]),
        direction_basis=None,
        retained_basis=None,
        device=device,
    )
    optimizer = active.make_repair_optimizer(module, "adamw", a.lr)

    best_delta = zero.detach().clone()
    best_bounded_hinge = float(base_bounded_hinge_t.detach().cpu())
    best_bounded_stats = dict(base_bounded_stats)
    best_kl = float(equal_example_non_target_kl(caches, zero).detach().cpu())
    best_full_answer_metrics = dict(base_full_answer_metrics)
    best_step = 0
    steps_completed = 0
    stopped_early = False
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        bounded_hinge, bounded_stats = bounded_sensitive_terms(
            caches,
            delta,
            log_suppression=log_suppression,
            tolerance=a.comparison_tolerance,
        )
        gd_non_target_kl = equal_example_non_target_kl(caches, delta)
        delta_l2 = delta.square().sum()
        loss = (
            a.ga_weight * bounded_hinge
            + a.gd_weight * gd_non_target_kl
            + a.delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite bounded-top3 Stage1 loss at step {step}")
        loss.backward()

        grad_norm_value = None
        if a.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), a.grad_clip)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at step {step}")
            grad_norm_value = float(grad_norm.detach().cpu())
        optimizer.step()
        steps_completed = step

        with torch.no_grad():
            candidate = module.effective_delta().detach().clone()
            candidate_bounded_hinge_t, candidate_bounded_stats = bounded_sensitive_terms(
                caches,
                candidate,
                log_suppression=log_suppression,
                tolerance=a.comparison_tolerance,
            )
            candidate_bounded_hinge = float(candidate_bounded_hinge_t.detach().cpu())
            candidate_kl = float(
                equal_example_non_target_kl(caches, candidate).detach().cpu()
            )
            candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
            candidate_full_metrics = locked.metrics(
                candidate_nll,
                full_answer_required_nll,
                candidate,
                target_nll=final_target_nll,
                tolerance=a.comparison_tolerance,
            )
            candidate_norm = float(candidate.norm().detach().cpu())

            if candidate_priority(
                candidate_bounded_hinge,
                candidate_bounded_stats,
                candidate_kl,
                candidate_norm,
            ) < candidate_priority(
                best_bounded_hinge,
                best_bounded_stats,
                best_kl,
                float(best_delta.norm().detach().cpu()),
            ):
                best_delta = candidate.detach().clone()
                best_bounded_hinge = candidate_bounded_hinge
                best_bounded_stats = dict(candidate_bounded_stats)
                best_kl = candidate_kl
                best_full_answer_metrics = dict(candidate_full_metrics)
                best_step = step

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "bounded_ga_hinge": float(bounded_hinge.detach().cpu()),
                "gd_same_prompt_non_target_kl": float(gd_non_target_kl.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
                **candidate_bounded_stats,
                **candidate_full_metrics,
            }
            logs.append(row)
            print(
                f"stage1-bounded-top3-step={step} "
                f"bounded_unmet={candidate_bounded_stats['bounded_unmet_target_position_count']} "
                f"bounded_examples={candidate_bounded_stats['bounded_unmet_example_count']} "
                f"active_full_answers={candidate_full_metrics['active_forget_instance_count']} "
                f"max_full_prob={candidate_full_metrics['forget_answer_probability_max']:.8g} "
                f"KL={candidate_kl:.6g} norm={candidate_norm:.6g}"
            )

        if (
            a.stop_when_bounded_satisfied
            and int(candidate_bounded_stats["bounded_unmet_target_position_count"]) == 0
        ):
            best_delta = candidate.detach().clone()
            best_bounded_hinge = candidate_bounded_hinge
            best_bounded_stats = dict(candidate_bounded_stats)
            best_kl = candidate_kl
            best_full_answer_metrics = dict(candidate_full_metrics)
            best_step = step
            stopped_early = True
            break

    del optimizer
    write_jsonl(root / "train_log.jsonl", logs)

    tofu.set_selected_lm_head_rows(
        output_weight,
        sensitive_ids,
        base_selected_rows,
        best_delta,
    )
    if int(input_weight.data_ptr()) != input_pointer or int(input_weight._version) != input_version:
        raise RuntimeError("V7 bounded-top3 Stage1 modified input embeddings")

    materialized_nll = tofu.score_answer_instances(
        model,
        tok,
        instances,
        device,
        batch_size=a.batch_size,
        max_length=a.max_length,
    ).detach().float()
    materialized_full_metrics = locked.metrics(
        materialized_nll,
        full_answer_required_nll,
        best_delta.to(device),
        target_nll=final_target_nll,
        tolerance=a.comparison_tolerance,
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    write_json(
        root / "sensitive_lm_rows.json",
        {
            "sensitive_row_definition": (
                "union of top-K rare/content-bearing answer-token rows selected independently "
                "for each of the 50 training-visible forget QAs"
            ),
            "sensitive_rows_per_example": a.sensitive_rows_per_example,
            "sensitive_row_count": len(sensitive_ids),
            "sensitive_token_ids": sensitive_ids,
            "sensitive_tokens": {
                str(token_id): locked.decoded_token(tok, token_id)
                for token_id in sensitive_ids
            },
            "selection_report": selection_report,
            "bounded_ga_suppression_factor": a.suppression_factor,
            "bounded_ga_log_nll_margin": log_suppression,
            "bounded_ga_definition": (
                "relu(NLL_base + log(suppression_factor) - NLL_current)^2 "
                "on selected sensitive true-target positions only"
            ),
            "non_sensitive_lm_rows_exact_base_by_construction": True,
            "input_embeddings_exact_base_by_construction": True,
            "transformer_frozen": True,
            "selection_uses_only_training_visible_direct_forget_answers": True,
        },
    )
    write_json(
        root / "forget_instances_before.json",
        locked.report_instances(
            instances,
            base_nll,
            full_answer_required_nll,
            a.target_forget_answer_probability,
        ),
    )
    write_json(
        root / "forget_instances_after.json",
        locked.report_instances(
            instances,
            materialized_nll,
            full_answer_required_nll,
            a.target_forget_answer_probability,
        ),
    )

    summary = {
        "status": "PASS_STAGE1_BOUNDED",
        "method": METHOD,
        "protocol": PROTOCOL,
        "best_step": best_step,
        "steps_completed": steps_completed,
        "stopped_early_on_bounded_constraints": stopped_early,
        "sensitive_rows_per_example": a.sensitive_rows_per_example,
        "sensitive_lm_head_row_count": len(sensitive_ids),
        "suppression_factor": a.suppression_factor,
        "suppression_nll_margin": log_suppression,
        "bounded_ga_hinge": best_bounded_hinge,
        "bounded_sensitive_metrics": best_bounded_stats,
        "selected_lm_head_delta_norm": float(best_delta.norm().detach().cpu()),
        "same_prompt_non_target_kl": best_kl,
        "cached_metrics": best_full_answer_metrics,
        "materialized_metrics": materialized_full_metrics,
        "stage1_does_not_select_or_stop_on_final_full_answer_threshold": True,
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
            **vars(a),
            "forget_json_resolved": str(forget_path),
            "training_visible_source_indices": source_indices,
            "parameter_scope": (
                "top-K rare/content sensitive LM-head rows only; transformer/input embeddings/"
                "all other LM rows exact Base"
            ),
            "ga_definition": (
                "bounded squared hinge requiring a fixed probability suppression relative to Base "
                "at selected sensitive true-target positions"
            ),
            "gd_definition": (
                "same-prompt KL(Base_non-target || Current_non-target), per-position true answer "
                "token removed and renormalized"
            ),
            "stage1_selection_and_stopping": (
                "bounded sensitive-token constraints only; final full-answer deletion threshold "
                "is diagnostic and reserved for Stage2"
            ),
            "checkpoint": str(ckpt.resolve()),
        },
    )

    print("===== SURE-TOFU V7 BOUNDED TOP3 STAGE1 =====")
    print(
        f"rows_per_example={a.sensitive_rows_per_example} "
        f"sensitive_rows={len(sensitive_ids)} suppression_factor={a.suppression_factor:g} "
        f"best_step={best_step} bounded_unmet={best_bounded_stats['bounded_unmet_target_position_count']} "
        f"active_full_answers={materialized_full_metrics['active_forget_instance_count']} "
        f"max_full_prob={materialized_full_metrics['forget_answer_probability_max']:.8g} "
        f"KL={best_kl:.6g} norm={float(best_delta.norm().detach().cpu()):.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
