#!/usr/bin/env python3
"""RWKU v3.2: answer-level frozen-base-head repair with 1K exact-KL utility preservation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_checkpoint_receipt as checkpoint_receipt
import rwku_representation as representation
import rwku_sure_head_only_w1k as head
import rwku_sure_hidden_direction_v31_w1k as v31
import rwku_sure_hidden_direction_w1k as v3
import rwku_sure_repr_rescue_w1k as v2
import sure_canonical_core as core

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_hidden_direction_v32_kl_w1k_seed0.json"
SCHEMA = "rwku_sure_head_hidden_direction_v32_kl_w1k_configuration_v1"
EXPERIMENT_ID = "rwku-h-w1k-stephen-king-hidden-direction-seed0-v32-kl"
PROTOCOL_STATUS = "rwku_target_only_auxwiki_sure_hidden_direction_v32_kl_w1k_posthoc_development"


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_configuration(path: Path) -> Dict[str, Any]:
    cfg = read_json(path)
    required = {
        "schema_version": SCHEMA,
        "configuration_id": EXPERIMENT_ID,
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_metrics_observed_before_method_design": True,
        "seed": 0,
        "target_entity": "Stephen King",
        "target_entity_id": "rwku:1_Stephen_King",
        "source_head_only_configuration_id": head.EXPECTED_CONFIGURATION_ID,
        "neutral_target": "Unknown",
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"v3.2 configuration changed {key}")

    h = cfg.get("hidden_direction", {})
    if h.get("objective") != "answer_level_frozen_base_head_sensitive_vs_neutral_plus_exact_utility_kl":
        raise ValueError("v3.2 objective changed")
    if h.get("sensitive_view_scope") != "all_generated_atomic_views":
        raise ValueError("v3.2 sensitive-view scope changed")
    if tuple(h.get("rank_ladder", ())) != (1, 2, 4):
        raise ValueError("v3.2 rank ladder must remain 1,2,4")
    if int(h.get("last_n_layers", 0)) != 1 or tuple(h.get("target_modules", ())) != ("down_proj",):
        raise ValueError("v3.2 must remain final-layer down_proj only")
    locked_h = {
        "answer_batch_size": 8,
        "answer_eval_batch_size": 8,
        "checkpoint_interval": 25,
        "checkpoint_kl_prompt_count": 128,
        "frozen_base_head_training_margin": 0.5,
        "frozen_base_head_answer_weight": 8.0,
        "edited_head_pairwise_target": 0.5,
        "edited_head_answer_weight": 2.0,
        "utility_hidden_weight": 2.0,
        "utility_kl_weight": 50.0,
        "utility_kl_batch_size": 4,
        "utility_train_prompt_count": 1000,
        "utility_gate_prompt_count": 1000,
        "max_relative_frobenius_delta": 0.01,
    }
    for key, expected in locked_h.items():
        if h.get(key) != expected:
            raise ValueError(f"v3.2 hidden-direction setting changed {key}")
    if int(h.get("steps", 0)) != 300 or float(h.get("learning_rate", 0.0)) != 0.0005:
        raise ValueError("v3.2 optimization schedule changed")

    acceptance = cfg.get("acceptance", {})
    locked_acceptance = {
        "required_pairwise_margin": 0.01,
        "required_direct_success": 100.0,
        "required_other_atomic_view_success": 100.0,
        "max_frozen_base_head_recovery": 0.0,
        "min_frozen_base_head_demotion_margin": 0.05,
        "max_head_delta_norm": 1.5,
        "utility_kl_mean_budget": 0.01,
        "utility_kl_p95_budget": 0.05,
        "utility_kl_max_budget": 0.5,
        "checkpoint_dtype": "bf16",
        "device_map": "single",
    }
    for key, expected in locked_acceptance.items():
        if acceptance.get(key) != expected:
            raise ValueError(f"v3.2 acceptance changed {key}")

    boundary = cfg.get("data_boundary", {})
    for key in (
        "official_rwku_records_available_to_learner",
        "official_rwku_records_used_for_checkpoint_selection",
        "neighbor_prompts_used_for_training_or_selection",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"v3.2 data boundary changed {key}")
    if boundary.get("external_wikipedia_only_utility") is not True:
        raise ValueError("v3.2 utility must remain external Wikipedia")
    if boundary.get("utility_train_gate_overlap_allowed") is not False:
        raise ValueError("v3.2 utility train/gate overlap must remain forbidden")
    return cfg


def _pad_utility_contexts(tokenizer, contexts, *, device):
    return v2._pad_context_batch(tokenizer, contexts, device=device)


def differentiable_utility_metrics(model, tokenizer, contexts, base_w, *, device):
    """Exact full-vocabulary Base||Edited KL plus final-hidden preservation."""
    if not contexts:
        raise ValueError("v3.2 KL utility batch is empty")
    input_ids, attention, lengths = _pad_utility_contexts(tokenizer, contexts, device=device)
    hidden, _ = wikipedia._final_hidden_only(model, {"input_ids": input_ids, "attention_mask": attention})
    rows = hidden[
        torch.arange(hidden.shape[0], device=hidden.device),
        lengths.to(hidden.device) - 1,
    ].float()
    base_hidden = torch.stack([x[2].float() for x in contexts]).to(device=rows.device, dtype=torch.float32)
    edited_w = model.get_output_embeddings().weight
    edited_logits = F.linear(rows.to(dtype=edited_w.dtype), edited_w).float()
    base_logits = F.linear(base_hidden.to(dtype=base_w.dtype), base_w).float()

    base_logp = F.log_softmax(base_logits, dim=-1)
    edited_logp = F.log_softmax(edited_logits, dim=-1)
    kl = (base_logp.exp() * (base_logp - edited_logp)).sum(dim=-1).clamp_min(0.0)
    if not torch.isfinite(kl).all():
        raise FloatingPointError("v3.2 utility KL became non-finite")

    hidden_rel_mse = (
        (rows - base_hidden).square().mean(dim=-1)
        / base_hidden.square().mean(dim=-1).clamp_min(1e-6)
    )
    if not torch.isfinite(hidden_rel_mse).all():
        raise FloatingPointError("v3.2 utility hidden loss became non-finite")
    return kl.mean(), kl, hidden_rel_mse.mean()


@torch.no_grad()
def checkpoint_utility_report(model, tokenizer, contexts, base_w, *, device):
    values = []
    hidden_values = []
    batch_size = 4
    for start in range(0, len(contexts), batch_size):
        mean_kl, vector, hidden_mean = differentiable_utility_metrics(
            model,
            tokenizer,
            contexts[start : start + batch_size],
            base_w,
            device=device,
        )
        del mean_kl
        values.append(vector.detach().cpu())
        hidden_values.append(float(hidden_mean.detach().cpu()))
    vector = torch.cat(values).float()
    return {
        "utility_kl_mean": float(vector.mean().item()),
        "utility_kl_p95": float(torch.quantile(vector, 0.95).item()),
        "utility_kl_max": float(vector.max().item()),
        "utility_prompt_count": int(vector.numel()),
        "utility_hidden_relative_mse_mean": float(sum(hidden_values) / max(len(hidden_values), 1)),
        "utility_kl_kind": "exact_full_vocabulary_base_to_edited_optimization_pool",
    }


def checkpoint_metrics(model, tokenizer, prompt_records, cases, base_w, utility, cfg, llama_like, device):
    proxy = v31.answer_proxy_report(model, cases, base_w, device)
    atomic = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
    )
    failures = int(atomic.get("direct_margin_failures", 0)) + int(atomic.get("generated_subject_margin_failures", 0))
    take = min(int(cfg["hidden_direction"]["checkpoint_kl_prompt_count"]), len(utility))
    utility_report = checkpoint_utility_report(model, tokenizer, utility[:take], base_w, device=device)
    return proxy, atomic, failures, utility_report


def train_rank(model, tokenizer, prompt_records, cases, utility, base_w, rank, cfg, llama_like, device, log_path):
    h = cfg["hidden_direction"]
    if len(utility) != int(h["utility_train_prompt_count"]):
        raise ValueError(
            f"v3.2 requires exactly {h['utility_train_prompt_count']} optimization utility contexts, got {len(utility)}"
        )
    adapter_cfg = representation.RepresentationConfig(
        steps=int(h["steps"]),
        learning_rate=float(h["learning_rate"]),
        weight_decay=float(h["weight_decay"]),
        rank=int(rank),
        alpha=float(rank),
        dropout=0.0,
        layer_indices=(),
        last_n_layers=1,
        target_modules=("down_proj",),
        seed=int(cfg["seed"]),
    )
    handles = representation.inject_lora_adapters(model, adapter_cfg)
    originals = v2.capture_adapter_base_weights(handles)
    parameters = representation.adapter_parameters(handles)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(h["learning_rate"]),
        weight_decay=float(h["weight_decay"]),
    )
    answer_bs = min(int(h["answer_batch_size"]), len(cases))
    utility_bs = min(int(h["utility_kl_batch_size"]), len(utility))
    interval = int(h["checkpoint_interval"])
    history = []
    checks = []
    best_key = None
    best_state = None
    best_step = None
    started = time.perf_counter()
    representation.set_adapter_scale(handles, 1.0)
    model.eval()

    for step in range(int(h["steps"])):
        optimizer.zero_grad(set_to_none=True)

        answer_indices = [(step * answer_bs + j) % len(cases) for j in range(answer_bs)]
        answer_batch = [cases[i] for i in answer_indices]
        bsens, bneu, esens, eneu = v31.answer_nlls(model, tokenizer, answer_batch, base_w, device)
        base_sep = bsens - bneu
        edited_sep = esens - eneu
        base_loss = F.relu(float(h["frozen_base_head_training_margin"]) - base_sep).square().mean()
        edited_loss = F.relu(float(h["edited_head_pairwise_target"]) - edited_sep).square().mean()

        utility_indices = [(step * utility_bs + j) % len(utility) for j in range(utility_bs)]
        utility_batch = [utility[i] for i in utility_indices]
        l2 = v2.adapter_l2(handles)

        answer_objective = (
            float(h["frozen_base_head_answer_weight"]) * base_loss
            + float(h["edited_head_answer_weight"]) * edited_loss
            + float(h["adapter_l2_weight"]) * l2
        )
        if not torch.isfinite(answer_objective):
            raise FloatingPointError("v3.2 answer objective became non-finite")
        answer_objective.backward()

        utility_kl_values = []
        utility_hidden_values = []
        for utility_context in utility_batch:
            utility_kl_i, _, utility_hidden_i = differentiable_utility_metrics(
                model,
                tokenizer,
                [utility_context],
                base_w,
                device=device,
            )
            utility_objective_i = (
                float(h["utility_hidden_weight"]) * utility_hidden_i
                + float(h["utility_kl_weight"]) * utility_kl_i
            ) / float(utility_bs)
            if not torch.isfinite(utility_objective_i):
                raise FloatingPointError("v3.2 utility objective became non-finite")
            utility_objective_i.backward()
            utility_kl_values.append(utility_kl_i.detach())
            utility_hidden_values.append(utility_hidden_i.detach())

        utility_kl = torch.stack(utility_kl_values).mean()
        utility_hidden = torch.stack(utility_hidden_values).mean()
        loss = (
            answer_objective.detach()
            + float(h["utility_hidden_weight"]) * utility_hidden
            + float(h["utility_kl_weight"]) * utility_kl
        )
        torch.nn.utils.clip_grad_norm_(parameters, float(h["grad_clip"]))
        optimizer.step()

        if step == 0 or (step + 1) % 25 == 0 or step + 1 == int(h["steps"]):
            row = {
                "step": step + 1,
                "rank": int(rank),
                "loss": float(loss.detach().cpu()),
                "batch_base_answer_hinge": float(base_loss.detach().cpu()),
                "batch_base_answer_separation_mean": float(base_sep.mean().detach().cpu()),
                "batch_base_answer_recovery_percentage": float((base_sep <= 0).float().mean().mul(100).detach().cpu()),
                "batch_edited_answer_hinge": float(edited_loss.detach().cpu()),
                "batch_edited_answer_separation_mean": float(edited_sep.mean().detach().cpu()),
                "utility_kl_mean": float(utility_kl.detach().cpu()),
                "utility_hidden_relative_mse": float(utility_hidden.detach().cpu()),
            }
            history.append(row)
            print(
                "v32-rank{} step {:3d}: loss={:.6f} base_sep={:.4f} base_rec={:.2f}% edited_sep={:.4f} wiki_kl={:.6f} wiki_hidden={:.6f}".format(
                    rank,
                    step + 1,
                    row["loss"],
                    row["batch_base_answer_separation_mean"],
                    row["batch_base_answer_recovery_percentage"],
                    row["batch_edited_answer_separation_mean"],
                    row["utility_kl_mean"],
                    row["utility_hidden_relative_mse"],
                )
            )

        if (step + 1) % interval == 0 or step + 1 == int(h["steps"]):
            proxy, atomic, atomic_failures, utility_report = checkpoint_metrics(
                model,
                tokenizer,
                prompt_records,
                cases,
                base_w,
                utility,
                cfg,
                llama_like,
                device,
            )
            feasible_behavior = bool(
                int(proxy["recovery_count"]) == 0
                and float(proxy["minimum_demotion_margin"]) >= float(cfg["acceptance"]["min_frozen_base_head_demotion_margin"])
                and atomic_failures == 0
            )
            if feasible_behavior:
                selection_key = (
                    0,
                    float(utility_report["utility_kl_mean"]),
                    float(utility_report["utility_kl_p95"]),
                    float(utility_report["utility_kl_max"]),
                    float(utility_report["utility_hidden_relative_mse_mean"]),
                )
            else:
                margin_shortfall = max(
                    0.0,
                    float(cfg["acceptance"]["min_frozen_base_head_demotion_margin"])
                    - float(proxy["minimum_demotion_margin"]),
                )
                selection_key = (
                    1,
                    int(proxy["recovery_count"]),
                    margin_shortfall,
                    int(atomic_failures),
                    float(utility_report["utility_kl_mean"]),
                )
            checks.append(
                {
                    "step": step + 1,
                    "behavior_constraint_satisfied": feasible_behavior,
                    "selection_key": list(selection_key),
                    "frozen_base_head_answer_proxy": proxy,
                    "atomic": atomic,
                    "atomic_margin_failure_count": atomic_failures,
                    "optimization_utility": utility_report,
                }
            )
            print(
                "  v32 checkpoint step {}: answer_rec={:.2f}% minsep={:.4f} atomic_fail={} optKL={:.6f}/{:.6f}/{:.6f} hidden={:.6f} eligible={}".format(
                    step + 1,
                    proxy["recovery_percentage"],
                    proxy["minimum_demotion_margin"],
                    atomic_failures,
                    utility_report["utility_kl_mean"],
                    utility_report["utility_kl_p95"],
                    utility_report["utility_kl_max"],
                    utility_report["utility_hidden_relative_mse_mean"],
                    feasible_behavior,
                )
            )
            if best_key is None or selection_key < best_key:
                best_key = selection_key
                best_state = v31.adapter_state(handles)
                best_step = step + 1

    if best_state is None:
        raise RuntimeError("v3.2 selected no adapter checkpoint")
    v31.restore_state(handles, best_state)
    report = {
        "rank": int(rank),
        "steps": int(h["steps"]),
        "best_checkpoint_step": int(best_step),
        "best_selection_key": list(best_key),
        "selection_policy": "first satisfy 0 frozen-base-head recovery, min answer margin >= acceptance, and zero atomic failures; then minimize exact optimization-pool KL mean/p95/max",
        "utility_train_prompt_count": len(utility),
        "utility_kl_batch_size": utility_bs,
        "utility_kl_weight": float(h["utility_kl_weight"]),
        "history": history,
        "checkpoint_evaluations": checks,
        "training_seconds": time.perf_counter() - started,
        "official_rwku_records_accessed": False,
    }
    core.write_json(log_path, report)
    return handles, originals, report


def configure(cfg):
    v3.SCRIPT_PATH = SCRIPT_PATH
    v3.SCHEMA = SCHEMA
    v3.EXPERIMENT_ID = EXPERIMENT_ID
    v3.PROTOCOL_STATUS = PROTOCOL_STATUS
    v3.DEFAULT_CONFIGURATION = DEFAULT_CONFIGURATION
    v3.load_configuration = load_configuration
    v3.build_direction_cases = v31.build_answer_cases
    v3.frozen_proxy_report = v31.answer_proxy_report
    v3.proxy_safe = v31.proxy_safe
    v3.train_rank = train_rank
    v31._RUNTIME["answer_eval_batch_size"] = int(cfg["hidden_direction"]["answer_eval_batch_size"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default="local_pinned_snapshot")
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--generator-receipt", type=Path, required=True)
    parser.add_argument("--utility-cache", type=Path, required=True)
    parser.add_argument("--wikipedia-dir", type=Path, required=True)
    parser.add_argument("--source-head-only-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    args = parser.parse_args()

    cfg = load_configuration(args.configuration)
    if args.experiment_id != cfg["configuration_id"]:
        raise ValueError("v3.2 experiment ID must equal locked configuration ID")
    configure(cfg)

    original_loader = gagd.load_model_and_tokenizer
    original_receipt_create = checkpoint_receipt.create_checkpoint_receipt
    original_exact_gate = v2.exact_full_vocab_utility_kl

    def wrapped_receipt_create(*a, **kw):
        implementation_files = list(kw.get("implementation_files", []))
        dependency = PROJECT_ROOT / "scripts" / "rwku_sure_hidden_direction_v31_w1k.py"
        if dependency not in implementation_files:
            implementation_files.append(dependency)
        kw["implementation_files"] = implementation_files
        return original_receipt_create(*a, **kw)

    def wrapped_exact_gate(model, tokenizer, contexts, **kw):
        expected = int(cfg["hidden_direction"]["utility_gate_prompt_count"])
        if len(contexts) != expected:
            raise ValueError(f"v3.2 requires exactly {expected} disjoint final utility-gate contexts, got {len(contexts)}")
        return original_exact_gate(model, tokenizer, contexts, **kw)

    def wrapped_loader(*a, **kw):
        model, tokenizer = original_loader(*a, **kw)
        v31._RUNTIME["tokenizer"] = tokenizer
        return model, tokenizer

    gagd.load_model_and_tokenizer = wrapped_loader
    checkpoint_receipt.create_checkpoint_receipt = wrapped_receipt_create
    v2.exact_full_vocab_utility_kl = wrapped_exact_gate
    sys.argv = [
        "rwku_sure_hidden_direction_v32_kl_w1k.py",
        "--model-path", args.model_path,
        "--model-revision", args.model_revision,
        "--training-bundle", str(args.training_bundle),
        "--generator-receipt", str(args.generator_receipt),
        "--utility-cache", str(args.utility_cache),
        "--wikipedia-dir", str(args.wikipedia_dir),
        "--source-head-only-run", str(args.source_head_only_run),
        "--output-root", str(args.output_root),
        "--experiment-id", args.experiment_id,
        "--configuration", str(args.configuration),
    ]
    try:
        v3.main()
    finally:
        gagd.load_model_and_tokenizer = original_loader
        checkpoint_receipt.create_checkpoint_receipt = original_receipt_create
        v2.exact_full_vocab_utility_kl = original_exact_gate


if __name__ == "__main__":
    main()
