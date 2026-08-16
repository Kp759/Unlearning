#!/usr/bin/env python3
"""SURE-TOFU V7-top3 Stage 1: conservative sparse LM-head GA/GD.

This is the controlled variant of ``tofu_sure_sparse_lm_gagd_v7.py`` in which
Stage 1 does NOT treat every answer-token row as sensitive.  For each of the 50
training-visible forget QAs, it deterministically selects the top K (default 3)
rare/content-bearing answer-token rows using only those same 50 answers.

Only the union of those selected LM-head rows is editable.  Transformer blocks,
all input embeddings, and every non-selected LM-head row remain exact Full-TOFU
Base throughout optimization.

GA is applied ONLY at answer positions whose true target token is one of the
selected sensitive rows.  GD remains the same leakage-free same-prompt
non-target KL protection used by V7, evaluated on all answer positions from the
same 50 visible forget QAs.  Full-answer direct forgetting is used only for the
training-visible stopping/audit condition; no retain, paraphrase, same-author,
PPL, real-author, or world-fact data is loaded.
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


METHOD = "SURE-TOFU-v7-top3-sensitive-sparse-lmhead-GAGD"
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
    p.add_argument("--stop-when-all-satisfied", action=argparse.BooleanOptionalAction, default=True)
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
    """Weight each of the 50 visible forget QAs equally in the GD/KL term."""
    if not caches:
        return delta_rows.new_zeros(())
    return torch.stack(
        [stable_same_prompt_non_target_kl([cache], delta_rows) for cache in caches]
    ).mean()


def sensitive_target_ga_logprob(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> Tuple[torch.Tensor, int, int]:
    """Mean log p(true token) only where the true token is selected sensitive.

    Each QA contributes equally: within a QA, all occurrences of selected
    sensitive target rows are averaged first, then the 50 per-QA values are
    averaged.  The top-K selector guarantees at least one selected target row
    per QA, but we fail closed if tokenization ever violates that assumption.
    """
    per_example: List[torch.Tensor] = []
    total_positions = 0
    examples_with_positions = 0
    for cache in caches:
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
        if not selected_mask.any():
            raise RuntimeError(
                "top3 sensitive selector produced a QA with no selected true-target positions"
            )
        selected_nll = token_nll[selected_mask]
        per_example.append(-selected_nll.mean())
        total_positions += int(selected_mask.sum().detach().cpu())
        examples_with_positions += 1
    if not per_example:
        raise RuntimeError("no sensitive target positions available for GA")
    return torch.stack(per_example).mean(), total_positions, examples_with_positions


def sensitive_target_probability_stats(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> Dict[str, float]:
    probs: List[torch.Tensor] = []
    for cache in caches:
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
            probs.append(torch.exp(-token_nll[selected_mask]))
    if not probs:
        return {"sensitive_target_probability_mean": 0.0, "sensitive_target_probability_max": 0.0}
    joined = torch.cat(probs)
    return {
        "sensitive_target_probability_mean": float(joined.mean().detach().cpu()),
        "sensitive_target_probability_max": float(joined.max().detach().cpu()),
    }


def stage1_priority(
    metrics: Dict[str, Any],
    kl_value: float,
) -> Tuple[int, int, float, float, float]:
    return (
        int(metrics["active_forget_instance_count"]),
        int(metrics["buffered_forget_constraint_unmet_count"]),
        float(metrics["forget_answer_probability_max"]),
        float(kl_value),
        float(metrics["selected_lm_head_delta_norm"]),
    )


def main() -> None:
    a = parse_args()
    if a.forget_num != 50:
        raise ValueError("V7-top3 locked experiment fixes --forget-num=50")
    if a.steps <= 0 or a.lr <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("steps/lr/batch-size/max-length must be positive")
    if a.sensitive_rows_per_example <= 0:
        raise ValueError("sensitive-rows-per-example must be positive")
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
        raise RuntimeError("top3 selector produced no sensitive LM-head rows")

    selected_tensor = torch.tensor(
        sensitive_ids, dtype=torch.long, device=output_weight.device
    )
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    # Exact Full-TOFU caches: transformer/input embeddings never move, and only
    # selected output rows are represented by the learned sparse delta.
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

    # Fail closed: every one of the 50 examples must contain at least one true
    # target occurrence belonging to its/another QA's selected sensitive rows.
    _, sensitive_target_position_count, ga_example_count = sensitive_target_ga_logprob(
        caches, zero
    )
    if ga_example_count != len(instances):
        raise RuntimeError(
            f"GA coverage mismatch: {ga_example_count}/{len(instances)} visible QAs"
        )

    base_nll = tofu.answer_nlls_from_packed_delta_cache(packed, zero).detach()
    target_nll = -math.log(a.target_forget_answer_probability)
    required_nll = base_nll.new_full(base_nll.shape, target_nll)
    base_metrics = locked.metrics(
        base_nll,
        required_nll,
        zero,
        target_nll=target_nll,
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
    best_metrics = dict(base_metrics)
    best_kl = float(equal_example_non_target_kl(caches, zero).detach().cpu())
    best_step = 0
    steps_completed = 0
    stopped_early = False
    logs: List[Dict[str, Any]] = []

    for step in range(1, a.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        current_nll = tofu.answer_nlls_from_packed_delta_cache(packed, delta)
        ga_sensitive_logprob, _, _ = sensitive_target_ga_logprob(caches, delta)
        gd_non_target_kl = equal_example_non_target_kl(caches, delta)
        delta_l2 = delta.square().sum()
        loss = (
            a.ga_weight * ga_sensitive_logprob
            + a.gd_weight * gd_non_target_kl
            + a.delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite top3 sparse GA/GD loss at step {step}")
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
            candidate_nll = tofu.answer_nlls_from_packed_delta_cache(packed, candidate)
            candidate_kl_t = equal_example_non_target_kl(caches, candidate)
            candidate_kl = float(candidate_kl_t.detach().cpu())
            candidate_metrics = locked.metrics(
                candidate_nll,
                required_nll,
                candidate,
                target_nll=target_nll,
                tolerance=a.comparison_tolerance,
            )
            candidate_sensitive = sensitive_target_probability_stats(caches, candidate)
            if stage1_priority(candidate_metrics, candidate_kl) < stage1_priority(
                best_metrics, best_kl
            ):
                best_delta = candidate.detach().clone()
                best_metrics = dict(candidate_metrics)
                best_kl = candidate_kl
                best_step = step

        if step == 1 or step % a.log_every == 0 or step == a.steps:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "ga_sensitive_logprob": float(ga_sensitive_logprob.detach().cpu()),
                "gd_same_prompt_non_target_kl": float(gd_non_target_kl.detach().cpu()),
                "delta_l2": float(delta_l2.detach().cpu()),
                "gradient_norm_before_clip": grad_norm_value,
                **candidate_sensitive,
                **candidate_metrics,
            }
            logs.append(row)
            print(
                f"stage1-top3-step={step} active={candidate_metrics['active_forget_instance_count']} "
                f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g} "
                f"sens_p={candidate_sensitive['sensitive_target_probability_mean']:.6g} "
                f"KL={candidate_kl:.6g} norm={candidate_metrics['selected_lm_head_delta_norm']:.6g}"
            )

        if (
            a.stop_when_all_satisfied
            and candidate_metrics["active_forget_instance_count"] == 0
            and candidate_metrics["buffered_forget_constraint_unmet_count"] == 0
        ):
            best_delta = candidate.detach().clone()
            best_metrics = dict(candidate_metrics)
            best_kl = candidate_kl
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
        raise RuntimeError("V7-top3 Stage1 modified input embeddings")

    materialized_nll = tofu.score_answer_instances(
        model,
        tok,
        instances,
        device,
        batch_size=a.batch_size,
        max_length=a.max_length,
    ).detach().float()
    materialized_metrics = locked.metrics(
        materialized_nll,
        required_nll,
        best_delta.to(device),
        target_nll=target_nll,
        tolerance=a.comparison_tolerance,
    )
    final_sensitive_stats = sensitive_target_probability_stats(caches, best_delta.to(device))

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
            "sensitive_target_position_count": sensitive_target_position_count,
            "ga_example_coverage": ga_example_count,
            "sensitive_token_ids": sensitive_ids,
            "sensitive_tokens": {
                str(token_id): locked.decoded_token(tok, token_id)
                for token_id in sensitive_ids
            },
            "selection_report": selection_report,
            "non_sensitive_lm_rows_exact_base_by_construction": True,
            "input_embeddings_exact_base_by_construction": True,
            "transformer_frozen": True,
            "selection_uses_only_training_visible_direct_forget_answers": True,
        },
    )
    write_json(
        root / "forget_instances_before.json",
        locked.report_instances(
            instances, base_nll, required_nll, a.target_forget_answer_probability
        ),
    )
    write_json(
        root / "forget_instances_after.json",
        locked.report_instances(
            instances,
            materialized_nll,
            required_nll,
            a.target_forget_answer_probability,
        ),
    )

    summary = {
        "status": "PASS_STAGE1",
        "method": METHOD,
        "protocol": PROTOCOL,
        "best_step": best_step,
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "sensitive_rows_per_example": a.sensitive_rows_per_example,
        "sensitive_lm_head_row_count": len(sensitive_ids),
        "sensitive_target_position_count": sensitive_target_position_count,
        "selected_lm_head_delta_norm": float(best_delta.norm().detach().cpu()),
        "same_prompt_non_target_kl": best_kl,
        **final_sensitive_stats,
        "cached_metrics": best_metrics,
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
            **vars(a),
            "forget_json_resolved": str(forget_path),
            "training_visible_source_indices": source_indices,
            "parameter_scope": (
                "top-K rare/content sensitive LM-head rows only; transformer/input embeddings/"
                "all other LM rows exact Base"
            ),
            "ga_definition": (
                "mean log probability of true answer tokens only at selected sensitive target positions, minimized"
            ),
            "gd_definition": (
                "same-prompt KL(Base_non-target || Current_non-target), per-position true answer token removed and renormalized"
            ),
            "checkpoint": str(ckpt.resolve()),
        },
    )

    print("===== SURE-TOFU V7 TOP3 STAGE1 =====")
    print(
        f"rows_per_example={a.sensitive_rows_per_example} "
        f"sensitive_rows={len(sensitive_ids)} sensitive_positions={sensitive_target_position_count} "
        f"best_step={best_step} active_after={materialized_metrics['active_forget_instance_count']} "
        f"max_prob={materialized_metrics['forget_answer_probability_max']:.8g} "
        f"KL={best_kl:.6g} norm={float(best_delta.norm().detach().cpu()):.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
