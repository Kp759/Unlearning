#!/usr/bin/env python3
"""SURE-TOFU V7 Stage 1: sparse LM-head-only GA/GD on 50 visible forget QAs.

This stage implements the simplest leakage-free formulation:

* the protected Full-TOFU model is the immutable starting point;
* transformer blocks and ALL input embeddings stay frozen/exact Base;
* only LM-head vocabulary rows that occur in the 50 training-visible answers
  are editable (the sensitive answer rows);
* GA suppresses the true answer probability on those same 50 QAs;
* GD preserves the Base distribution after removing each position's true target
  token and renormalizing (same-prompt non-target KL);
* every non-sensitive LM-head row remains exactly Base throughout optimization.

The optimization is performed on sparse output-row deltas against cached Base
hidden states, so no dense LM-head optimizer state and no embedding update are
created.  No retain, paraphrase, same-author holdout, PPL, real-author, or
world-fact data is loaded or used.
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


METHOD = "SURE-TOFU-v7-sparse-lmhead-same-prompt-GAGD"
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
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def equal_example_non_target_kl(
    caches: Sequence[tofu.TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    """Match Stage1A semantics by weighting each of the 50 QAs equally."""
    if not caches:
        return delta_rows.new_zeros(())
    return torch.stack(
        [stable_same_prompt_non_target_kl([cache], delta_rows) for cache in caches]
    ).mean()


def stage1_priority(metrics: Dict[str, Any], kl_value: float) -> Tuple[int, int, float, float, float]:
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
        raise ValueError("V7 locked experiment fixes --forget-num=50")
    if a.steps <= 0 or a.lr <= 0 or a.batch_size <= 0 or a.max_length <= 0:
        raise ValueError("steps/lr/batch-size/max-length must be positive")
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
    instances, source_indices = locked.load_forget_instances(forget_path, data_tok, a.forget_num)

    model, tok = gagd.load_model_and_tokenizer(locked.model_args(a), for_training=False)
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_pointer = int(input_weight.data_ptr())
    input_version = int(input_weight._version)
    device = gagd.first_device(model)

    all_positions = list(range(len(instances)))
    sensitive_ids = locked.answer_rows_for_instances(
        tok, instances, all_positions, max_length=a.max_length
    )
    if not sensitive_ids:
        raise RuntimeError("50 visible forget answers produced no sensitive LM-head rows")

    selected_tensor = torch.tensor(sensitive_ids, dtype=torch.long, device=output_weight.device)
    base_selected_rows = output_weight.index_select(0, selected_tensor).detach().clone()

    # Base caches are exact because the transformer/input embeddings never move,
    # and only these selected output rows are parameterized by a delta.
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
        ga_sensitive_logprob = -current_nll.mean()
        gd_non_target_kl = equal_example_non_target_kl(caches, delta)
        delta_l2 = delta.square().sum()
        loss = (
            a.ga_weight * ga_sensitive_logprob
            + a.gd_weight * gd_non_target_kl
            + a.delta_l2_lambda * delta_l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite sparse GA/GD loss at step {step}")
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
            if stage1_priority(candidate_metrics, candidate_kl) < stage1_priority(best_metrics, best_kl):
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
                **candidate_metrics,
            }
            logs.append(row)
            print(
                f"stage1-step={step} active={candidate_metrics['active_forget_instance_count']} "
                f"max_prob={candidate_metrics['forget_answer_probability_max']:.8g} "
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
        raise RuntimeError("V7 Stage1 modified input embeddings")

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

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    write_json(
        root / "sensitive_lm_rows.json",
        {
            "sensitive_row_definition": "union of non-special answer-token vocabulary rows from the 50 training-visible forget QAs",
            "sensitive_row_count": len(sensitive_ids),
            "sensitive_token_ids": sensitive_ids,
            "sensitive_tokens": {
                str(token_id): locked.decoded_token(tok, token_id) for token_id in sensitive_ids
            },
            "non_sensitive_lm_rows_exact_base_by_construction": True,
            "input_embeddings_exact_base_by_construction": True,
            "transformer_frozen": True,
        },
    )
    write_json(
        root / "forget_instances_before.json",
        locked.report_instances(instances, base_nll, required_nll, a.target_forget_answer_probability),
    )
    write_json(
        root / "forget_instances_after.json",
        locked.report_instances(instances, materialized_nll, required_nll, a.target_forget_answer_probability),
    )
    summary = {
        "status": "PASS_STAGE1",
        "method": METHOD,
        "protocol": PROTOCOL,
        "best_step": best_step,
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "sensitive_lm_head_row_count": len(sensitive_ids),
        "selected_lm_head_delta_norm": float(best_delta.norm().detach().cpu()),
        "same_prompt_non_target_kl": best_kl,
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
            "parameter_scope": "sparse answer-token LM-head rows only; transformer/input embeddings/non-sensitive LM rows exact Base",
            "gd_definition": "same-prompt KL(Base_non-target || Current_non-target), true answer token removed and renormalized",
            "checkpoint": str(ckpt.resolve()),
        },
    )
    print("===== SURE-TOFU V7 STAGE1 =====")
    print(
        f"sensitive_rows={len(sensitive_ids)} best_step={best_step} "
        f"active_after={materialized_metrics['active_forget_instance_count']} "
        f"max_prob={materialized_metrics['forget_answer_probability_max']:.8g} "
        f"KL={best_kl:.6g} norm={float(best_delta.norm().detach().cpu()):.6g}"
    )
    print("checkpoint:", ckpt)


if __name__ == "__main__":
    main()
