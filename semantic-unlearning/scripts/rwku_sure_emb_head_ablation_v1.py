#!/usr/bin/env python3
"""RWKU development ablations: embeddings+LM head, with/without final down_proj LoRA.

This script starts from the same source Stage-1 sparse-head checkpoint used by
RWKU v3.2. It keeps an immutable copy of the original tied vocabulary matrix W0
for the frozen-decoder recovery probe, while optionally training:

  A) input embeddings + untied LM head
  B) input embeddings + untied LM head + rank-1 final-layer down_proj LoRA

The 1K Wikipedia optimization pool and v3.2 KL/hidden losses are reused. The
previously opened v3.2 1K guard is explicitly skipped; a fresh disjoint 1K guard
slice is opened only once after checkpoint selection. No official RWKU records
are loaded by this script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_representation as representation
import rwku_sure_head_only_w1k as head
import rwku_sure_hidden_direction_v31_w1k as v31
import rwku_sure_hidden_direction_v32_kl_w1k as v32
import rwku_sure_hidden_direction_w1k as v3
import rwku_sure_repr_rescue_w1k as v2
import sure_canonical_core as core
import sure_minimal_two_stage as learner

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_emb_head_ablation_v1_seed0.json"
V32_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_hidden_direction_v32_kl_w1k_seed0.json"
SCHEMA = "rwku_sure_emb_head_ablation_v1_configuration_v1"
EXPERIMENT_ID = "rwku-stephen-king-emb-head-ablation-seed0-v1"
VARIANTS = ("emb_head", "emb_head_downproj")


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    core.write_json(path, dict(value))


def sha_indices(values: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in values]).encode()).hexdigest()


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
        "neutral_target": "Unknown",
        "source_head_only_configuration_id": head.EXPECTED_CONFIGURATION_ID,
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"ablation configuration changed {key}")
    if tuple(cfg.get("variants", ())) != VARIANTS:
        raise ValueError(f"variants must remain {VARIANTS}")
    opt = cfg.get("optimization", {})
    locked = {
        "steps": 300,
        "answer_batch_size": 8,
        "answer_eval_batch_size": 8,
        "checkpoint_interval": 25,
        "checkpoint_kl_prompt_count": 128,
        "embedding_head_learning_rate": 0.0001,
        "downproj_learning_rate": 0.0005,
        "frozen_base_head_training_margin": 0.5,
        "frozen_base_head_answer_weight": 8.0,
        "edited_head_pairwise_target": 0.5,
        "edited_head_answer_weight": 2.0,
        "utility_hidden_weight": 2.0,
        "utility_kl_weight": 50.0,
        "utility_kl_batch_size": 4,
        "downproj_rank": 1,
        "utility_train_prompt_count": 1000,
        "fresh_confirmatory_gate_prompt_count": 1000,
        "opened_gate_skip_count": 1000,
        "max_relative_frobenius_downproj": 0.01,
    }
    for key, expected in locked.items():
        if opt.get(key) != expected:
            raise ValueError(f"ablation optimization setting changed {key}")
    acc = cfg.get("acceptance", {})
    locked_acc = {
        "required_pairwise_margin": 0.01,
        "required_direct_success": 100.0,
        "required_other_atomic_view_success": 100.0,
        "max_frozen_base_head_recovery": 0.0,
        "min_frozen_base_head_demotion_margin": 0.05,
        "utility_kl_mean_budget": 0.01,
        "utility_kl_p95_budget": 0.05,
        "utility_kl_max_budget": 0.5,
        "checkpoint_dtype": "bf16",
        "device_map": "single",
    }
    for key, expected in locked_acc.items():
        if acc.get(key) != expected:
            raise ValueError(f"ablation acceptance changed {key}")
    boundary = cfg.get("data_boundary", {})
    for key in (
        "official_rwku_records_available_to_learner",
        "official_rwku_records_used_for_checkpoint_selection",
        "neighbor_prompts_used_for_training_or_selection",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"ablation data boundary changed {key}")
    if boundary.get("external_wikipedia_only_utility") is not True:
        raise ValueError("utility must remain external Wikipedia")
    if boundary.get("previously_opened_v32_gate_excluded_from_selection") is not True:
        raise ValueError("opened v3.2 guard must remain excluded")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-bundle", type=Path, required=True)
    p.add_argument("--generator-receipt", type=Path, required=True)
    p.add_argument("--utility-cache", type=Path, required=True)
    p.add_argument("--wikipedia-dir", type=Path, required=True)
    p.add_argument("--source-head-only-run", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


@torch.no_grad()
def chunked_delta_report(current: torch.Tensor, reference: torch.Tensor, chunk_rows: int = 2048) -> Dict[str, float]:
    if current.shape != reference.shape:
        raise ValueError("delta report tensors differ in shape")
    delta_sq = 0.0
    ref_sq = 0.0
    rows = int(current.shape[0])
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        cur = current[start:stop].detach().float()
        ref = reference[start:stop].to(device=cur.device, dtype=torch.float32)
        delta_sq += float((cur - ref).square().sum().item())
        ref_sq += float(ref.square().sum().item())
    absolute = math.sqrt(max(delta_sq, 0.0))
    reference_norm = math.sqrt(max(ref_sq, 0.0))
    return {
        "absolute_frobenius": absolute,
        "reference_frobenius": reference_norm,
        "relative_frobenius": absolute / max(reference_norm, 1e-12),
    }


def snapshot_trainable_state(
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    handles: Sequence[Any],
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "input_weight": input_weight.detach().to(device="cpu", copy=True),
        "output_weight": output_weight.detach().to(device="cpu", copy=True),
    }
    if handles:
        state["adapter_state"] = v31.adapter_state(handles)
    return state


@torch.no_grad()
def restore_trainable_state(
    state: Mapping[str, Any],
    input_weight: torch.Tensor,
    output_weight: torch.Tensor,
    handles: Sequence[Any],
) -> None:
    input_weight.copy_(state["input_weight"].to(device=input_weight.device, dtype=input_weight.dtype))
    output_weight.copy_(state["output_weight"].to(device=output_weight.device, dtype=output_weight.dtype))
    if handles:
        v31.restore_state(handles, state["adapter_state"])
        representation.set_adapter_scale(handles, 1.0)


def behavior_report(model, tokenizer, prompt_records, cases, base_w0, cfg, llama_like, device) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    proxy = v31.answer_proxy_report(model, cases, base_w0, device)
    atomic = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
    )
    safe = bool(v2.behavior_safe(atomic) and v31.proxy_safe(proxy, cfg))
    return proxy, atomic, safe


def utility_safe(report: Mapping[str, Any], cfg: Mapping[str, Any]) -> Tuple[Dict[str, bool], bool]:
    acc = cfg["acceptance"]
    checks = {
        "mean": float(report["utility_kl_mean"]) <= float(acc["utility_kl_mean_budget"]),
        "p95": float(report["utility_kl_p95"]) <= float(acc["utility_kl_p95_budget"]),
        "max": float(report["utility_kl_max"]) <= float(acc["utility_kl_max_budget"]),
    }
    return checks, bool(all(checks.values()))


def setup_problem(args: argparse.Namespace, cfg: Mapping[str, Any], out: Path):
    v32_cfg = v32.load_configuration(V32_CONFIGURATION)
    source = v2.verify_source_run(Path(args.source_head_only_run).resolve(), v32_cfg)
    source_cfg = head.load_locked_configuration(v3.SOURCE_CONFIGURATION)
    views, bundle_audit, generator_audit = head.load_atomic_bundle(
        Path(args.training_bundle).resolve(),
        Path(args.generator_receipt).resolve(),
        source_cfg,
    )
    generator_model_audit = head.validate_generator_base_model(generator_audit, args.model_path)

    gagd.set_seed(int(cfg["seed"]))
    gagd.require_cuda_if_needed(cfg["acceptance"]["device_map"])
    margs = argparse.Namespace(
        model_path=args.model_path,
        dtype=cfg["acceptance"]["checkpoint_dtype"],
        device_map=cfg["acceptance"]["device_map"],
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(margs, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tokenizer)
    prompt_records = head.compile_prompt_records(views, tokenizer, neutral_target=str(cfg["neutral_target"]))

    inp = model.get_input_embeddings()
    outp = model.get_output_embeddings()
    if inp is None or outp is None or inp.weight.data_ptr() != outp.weight.data_ptr():
        raise ValueError("ablation requires the base model to begin with tied vocabulary weights")
    identity = wikipedia.model_identity(model, tokenizer, args.model_path)
    output_layer = core.untie_and_freeze_output_head(model)
    input_weight = model.get_input_embeddings().weight
    if input_weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("output head did not untie from input embeddings")

    # Immutable original decoder used by every frozen-W0 probe and Base KL reference.
    base_w0 = input_weight.detach().clone()
    base_head_reference = base_w0.detach().clone()

    payload = torch.load(source["stage1_delta"], map_location="cpu")
    selected_ids = [int(x) for x in payload["row_ids"]]
    selected_tensor = torch.tensor(selected_ids, device=output_layer.weight.device, dtype=torch.long)
    stage1_delta = payload["delta"].float().to(device=output_layer.weight.device)
    base_head_rows = output_layer.weight.index_select(0, selected_tensor).detach().clone()
    base_input_rows = base_w0.index_select(0, selected_tensor.to(base_w0.device)).detach()
    maxdiff = float((base_input_rows.float() - base_head_rows.to(base_input_rows.device).float()).abs().max().item())
    if maxdiff != 0.0:
        raise RuntimeError(f"base input readout differs from cloned head before sparse edit: {maxdiff}")

    optimization = head.optimization_namespace(source_cfg, prompt_count=len(prompt_records))
    second, utility_hidden, utility_lse, utility_meta = learner.load_utility_cache(
        Path(args.utility_cache).resolve(),
        expected_sample_size=optimization.utility_sample_size,
        expected_prompt_count=optimization.utility_prompt_count,
        expected_hidden_size=int(output_layer.weight.shape[1]),
        expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"],
    )
    del second
    utility_audit = head.validate_w1k_utility_metadata(utility_meta, source_cfg)
    selected_base_probs = v2.selected_base_probabilities_full_head(
        output_layer,
        selected_ids,
        utility_hidden,
        utility_lse,
        batch_size=optimization.utility_eval_batch_size,
    )
    train_idx, guard_idx, pool_report = learner.build_disjoint_token_conditioned_utility_pools(
        selected_base_probabilities=selected_base_probs,
        selected_ids=selected_ids,
        topk_per_row=optimization.utility_token_topk_per_row,
        uniform_prompt_count=optimization.utility_uniform_prompt_count,
        split_seed=optimization.utility_pool_seed,
    )

    core.materialize_output_delta(output_layer, selected_ids, stage1_delta)
    stage1_head_reference = output_layer.weight.detach().clone()
    actual_stage1_delta = learner.actual_selected_delta(output_layer, selected_ids, base_head_rows.float())
    source_stage1 = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
    )
    declared = read_json(source["stage1_report"])
    if source_stage1.get("pairwise_margin_failure_positions") != declared.get("pairwise_margin_failure_positions"):
        raise RuntimeError("reloaded source Stage-1 failure positions differ")

    neutral_ids = v2._completion_token_ids(tokenizer, str(cfg["neutral_target"]), llama_like)
    sensitive_ids = [x for x in selected_ids if x not in set(neutral_ids)]
    cases, direction_audit = v31.build_answer_cases(
        model,
        tokenizer,
        prompt_records,
        base_w0,
        sensitive_ids,
        neutral_ids,
        float(cfg["optimization"]["frozen_base_head_training_margin"]),
        llama_like,
        device,
    )
    v31._RUNTIME["tokenizer"] = tokenizer
    v31._RUNTIME["answer_eval_batch_size"] = int(cfg["optimization"]["answer_eval_batch_size"])

    texts, wiki_meta = wikipedia.load_wikipedia_train(Path(args.wikipedia_dir).resolve())
    opt = cfg["optimization"]
    train_count = int(opt["utility_train_prompt_count"])
    skip_count = int(opt["opened_gate_skip_count"])
    fresh_count = int(opt["fresh_confirmatory_gate_prompt_count"])
    if int(train_idx.numel()) < train_count:
        raise RuntimeError(f"utility train pool has only {int(train_idx.numel())} contexts; need {train_count}")
    if int(guard_idx.numel()) < skip_count + fresh_count:
        raise RuntimeError(
            f"utility guard pool has only {int(guard_idx.numel())} contexts; need at least {skip_count + fresh_count} "
            "to skip the opened v3.2 gate and create a fresh confirmatory gate"
        )
    train_indices = train_idx[:train_count].tolist()
    opened_indices = guard_idx[:skip_count].tolist()
    fresh_indices = guard_idx[skip_count : skip_count + fresh_count].tolist()
    if set(train_indices) & set(opened_indices) or set(train_indices) & set(fresh_indices) or set(opened_indices) & set(fresh_indices):
        raise RuntimeError("utility train/opened/fresh pools are not disjoint")
    train_contexts = v2.build_utility_contexts(tokenizer, texts, utility_meta, utility_hidden, train_indices)
    fresh_gate_contexts = v2.build_utility_contexts(tokenizer, texts, utility_meta, utility_hidden, fresh_indices)

    write_json(
        out / "protocol_report.json",
        {
            "configuration_id": cfg["configuration_id"],
            "variant": args.variant,
            "development_only": True,
            "posthoc_development_target": True,
            "official_rwku_records_accessed": False,
            "source_head_only_run": str(Path(args.source_head_only_run).resolve()),
            "bundle_audit": bundle_audit,
            "generator_model_audit": generator_model_audit,
            "utility_audit": utility_audit,
            "utility_pool_report": pool_report,
            "wikipedia_dataset": wiki_meta,
            "utility_train_prompt_count": len(train_contexts),
            "opened_v32_gate_skipped_prompt_count": len(opened_indices),
            "fresh_confirmatory_gate_prompt_count": len(fresh_gate_contexts),
            "utility_train_indices_sha256": sha_indices(train_indices),
            "opened_v32_gate_indices_sha256": sha_indices(opened_indices),
            "fresh_confirmatory_gate_indices_sha256": sha_indices(fresh_indices),
            "all_three_utility_slices_disjoint": True,
            "base_readout_validation_max_abs_diff": maxdiff,
            "direction_audit": direction_audit,
            "source_stage1_atomic": source_stage1,
            "source_stage1_sparse_head_delta_norm": float(actual_stage1_delta.norm().detach().cpu()),
        },
    )

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "llama_like": llama_like,
        "prompt_records": prompt_records,
        "cases": cases,
        "input_weight": input_weight,
        "output_layer": output_layer,
        "base_w0": base_w0,
        "base_head_reference": base_head_reference,
        "stage1_head_reference": stage1_head_reference,
        "train_contexts": train_contexts,
        "fresh_gate_contexts": fresh_gate_contexts,
    }


def train_variant(problem: Mapping[str, Any], cfg: Mapping[str, Any], variant: str, out: Path) -> Dict[str, Any]:
    model = problem["model"]
    tokenizer = problem["tokenizer"]
    device = problem["device"]
    llama_like = problem["llama_like"]
    prompt_records = problem["prompt_records"]
    cases = problem["cases"]
    input_weight = problem["input_weight"]
    output_weight = problem["output_layer"].weight
    base_w0 = problem["base_w0"]
    train_contexts = problem["train_contexts"]
    opt_cfg = cfg["optimization"]

    # Freeze everything, then open exactly the requested ablation parameters.
    for param in model.parameters():
        param.requires_grad_(False)
    input_weight.requires_grad_(True)
    output_weight.requires_grad_(True)

    handles: List[Any] = []
    originals: List[torch.Tensor] = []
    adapter_params: List[torch.nn.Parameter] = []
    if variant == "emb_head_downproj":
        rcfg = representation.RepresentationConfig(
            steps=int(opt_cfg["steps"]),
            learning_rate=float(opt_cfg["downproj_learning_rate"]),
            weight_decay=float(opt_cfg["weight_decay"]),
            rank=int(opt_cfg["downproj_rank"]),
            alpha=float(opt_cfg["downproj_alpha"]),
            dropout=0.0,
            layer_indices=(),
            last_n_layers=1,
            target_modules=("down_proj",),
            seed=int(cfg["seed"]),
        )
        handles = list(representation.inject_lora_adapters(model, rcfg))
        originals = list(v2.capture_adapter_base_weights(handles))
        adapter_params = list(representation.adapter_parameters(handles))
        representation.set_adapter_scale(handles, 1.0)

    parameter_groups: List[Dict[str, Any]] = [
        {
            "params": [input_weight, output_weight],
            "lr": float(opt_cfg["embedding_head_learning_rate"]),
            "weight_decay": float(opt_cfg["weight_decay"]),
        }
    ]
    if adapter_params:
        parameter_groups.append(
            {
                "params": adapter_params,
                "lr": float(opt_cfg["downproj_learning_rate"]),
                "weight_decay": float(opt_cfg["weight_decay"]),
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups)
    all_trainable = [input_weight, output_weight, *adapter_params]
    answer_bs = min(int(opt_cfg["answer_batch_size"]), len(cases))
    utility_bs = min(int(opt_cfg["utility_kl_batch_size"]), len(train_contexts))
    checkpoint_take = min(int(opt_cfg["checkpoint_kl_prompt_count"]), len(train_contexts))

    history: List[Dict[str, Any]] = []
    checkpoints: List[Dict[str, Any]] = []
    best_key: Optional[Tuple[Any, ...]] = None
    best_state: Optional[Dict[str, Any]] = None
    best_step: Optional[int] = None
    started = time.perf_counter()
    model.eval()

    for step in range(int(opt_cfg["steps"])):
        optimizer.zero_grad(set_to_none=True)
        answer_indices = [(step * answer_bs + j) % len(cases) for j in range(answer_bs)]
        answer_batch = [cases[i] for i in answer_indices]
        base_sensitive, base_neutral, edited_sensitive, edited_neutral = v31.answer_nlls(
            model, tokenizer, answer_batch, base_w0, device
        )
        base_sep = base_sensitive - base_neutral
        edited_sep = edited_sensitive - edited_neutral
        base_loss = torch.nn.functional.relu(float(opt_cfg["frozen_base_head_training_margin"]) - base_sep).square().mean()
        edited_loss = torch.nn.functional.relu(float(opt_cfg["edited_head_pairwise_target"]) - edited_sep).square().mean()
        adapter_l2 = v2.adapter_l2(handles) if handles else torch.zeros((), device=device, dtype=torch.float32)
        answer_objective = (
            float(opt_cfg["frozen_base_head_answer_weight"]) * base_loss
            + float(opt_cfg["edited_head_answer_weight"]) * edited_loss
            + float(opt_cfg["downproj_l2_weight"]) * adapter_l2
        )
        if not torch.isfinite(answer_objective):
            raise FloatingPointError("answer objective became non-finite")
        answer_objective.backward()

        utility_indices = [(step * utility_bs + j) % len(train_contexts) for j in range(utility_bs)]
        utility_kl_values: List[torch.Tensor] = []
        utility_hidden_values: List[torch.Tensor] = []
        for index in utility_indices:
            utility_kl_i, _, utility_hidden_i = v32.differentiable_utility_metrics(
                model,
                tokenizer,
                [train_contexts[index]],
                base_w0,
                device=device,
            )
            utility_objective_i = (
                float(opt_cfg["utility_hidden_weight"]) * utility_hidden_i
                + float(opt_cfg["utility_kl_weight"]) * utility_kl_i
            ) / float(utility_bs)
            if not torch.isfinite(utility_objective_i):
                raise FloatingPointError("utility objective became non-finite")
            utility_objective_i.backward()
            utility_kl_values.append(utility_kl_i.detach())
            utility_hidden_values.append(utility_hidden_i.detach())

        torch.nn.utils.clip_grad_norm_(all_trainable, float(opt_cfg["grad_clip"]))
        optimizer.step()

        utility_kl = torch.stack(utility_kl_values).mean()
        utility_hidden = torch.stack(utility_hidden_values).mean()
        total_for_log = (
            answer_objective.detach()
            + float(opt_cfg["utility_hidden_weight"]) * utility_hidden
            + float(opt_cfg["utility_kl_weight"]) * utility_kl
        )
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == int(opt_cfg["steps"]):
            row = {
                "step": step + 1,
                "variant": variant,
                "loss": float(total_for_log.cpu()),
                "batch_frozen_w0_separation_mean": float(base_sep.mean().detach().cpu()),
                "batch_frozen_w0_recovery_percentage": float((base_sep <= 0).float().mean().mul(100).detach().cpu()),
                "batch_edited_head_separation_mean": float(edited_sep.mean().detach().cpu()),
                "utility_kl_mean": float(utility_kl.cpu()),
                "utility_hidden_relative_mse": float(utility_hidden.cpu()),
            }
            history.append(row)
            print(
                "{} step {:3d}: loss={:.6f} base_sep={:.4f} base_rec={:.2f}% edited_sep={:.4f} wiki_kl={:.6f} wiki_hidden={:.6f}".format(
                    variant,
                    step + 1,
                    row["loss"],
                    row["batch_frozen_w0_separation_mean"],
                    row["batch_frozen_w0_recovery_percentage"],
                    row["batch_edited_head_separation_mean"],
                    row["utility_kl_mean"],
                    row["utility_hidden_relative_mse"],
                )
            )

        if (step + 1) % int(opt_cfg["checkpoint_interval"]) == 0 or step + 1 == int(opt_cfg["steps"]):
            proxy, atomic, behavior_ok = behavior_report(
                model, tokenizer, prompt_records, cases, base_w0, cfg, llama_like, device
            )
            utility_report = v32.checkpoint_utility_report(
                model,
                tokenizer,
                train_contexts[:checkpoint_take],
                base_w0,
                device=device,
            )
            emb_delta = chunked_delta_report(input_weight, base_w0)
            head_from_stage1 = chunked_delta_report(output_weight, problem["stage1_head_reference"])
            head_from_base = chunked_delta_report(output_weight, problem["base_head_reference"])
            downproj = v2.representation_delta_report(handles, originals) if handles else {
                "absolute_frobenius": 0.0,
                "relative_frobenius": 0.0,
            }
            if behavior_ok:
                key: Tuple[Any, ...] = (
                    0,
                    float(utility_report["utility_kl_mean"]),
                    float(utility_report["utility_kl_p95"]),
                    float(utility_report["utility_kl_max"]),
                    float(emb_delta["relative_frobenius"] + head_from_stage1["relative_frobenius"] + downproj["relative_frobenius"]),
                )
            else:
                margin_shortfall = max(
                    0.0,
                    float(cfg["acceptance"]["min_frozen_base_head_demotion_margin"])
                    - float(proxy["minimum_demotion_margin"]),
                )
                key = (
                    1,
                    int(proxy["recovery_count"]),
                    margin_shortfall,
                    int(atomic.get("direct_margin_failures", 0)) + int(atomic.get("generated_subject_margin_failures", 0)),
                    float(utility_report["utility_kl_mean"]),
                )
            checkpoint = {
                "step": step + 1,
                "selection_key": list(key),
                "behavior_safe": behavior_ok,
                "frozen_base_head_proxy": proxy,
                "atomic": atomic,
                "optimization_utility": utility_report,
                "embedding_delta_from_base": emb_delta,
                "lm_head_delta_from_stage1": head_from_stage1,
                "lm_head_delta_from_base": head_from_base,
                "downproj_delta": downproj,
            }
            checkpoints.append(checkpoint)
            print(
                "  checkpoint {}: rec={:.2f}% minmargin={:.4f} optKL={:.6f}/{:.6f}/{:.6f} emb_rel={:.6f} head_rel={:.6f} down_rel={:.6f} behavior_ok={}".format(
                    step + 1,
                    float(proxy["recovery_percentage"]),
                    float(proxy["minimum_demotion_margin"]),
                    float(utility_report["utility_kl_mean"]),
                    float(utility_report["utility_kl_p95"]),
                    float(utility_report["utility_kl_max"]),
                    float(emb_delta["relative_frobenius"]),
                    float(head_from_stage1["relative_frobenius"]),
                    float(downproj["relative_frobenius"]),
                    behavior_ok,
                )
            )
            if best_key is None or key < best_key:
                best_key = key
                best_step = step + 1
                best_state = snapshot_trainable_state(input_weight, output_weight, handles)

    if best_state is None or best_step is None or best_key is None:
        raise RuntimeError("ablation selected no checkpoint")
    restore_trainable_state(best_state, input_weight, output_weight, handles)

    # For the composite variant, physically materialize the rank-1 adapter at full scale.
    materialization = None
    if handles:
        materialization = v2.materialize_adapter_candidate(handles, originals, 1.0)

    final_proxy, final_atomic, behavior_ok = behavior_report(
        model, tokenizer, prompt_records, cases, base_w0, cfg, llama_like, device
    )
    final_emb_delta = chunked_delta_report(input_weight, base_w0)
    final_head_stage1_delta = chunked_delta_report(output_weight, problem["stage1_head_reference"])
    final_head_base_delta = chunked_delta_report(output_weight, problem["base_head_reference"])
    final_downproj = v2.representation_delta_report(handles, originals) if handles else {
        "absolute_frobenius": 0.0,
        "relative_frobenius": 0.0,
    }

    # This is the only place the fresh confirmatory utility guard is opened.
    fresh_utility = v32.checkpoint_utility_report(
        model,
        tokenizer,
        problem["fresh_gate_contexts"],
        base_w0,
        device=device,
    )
    fresh_checks, fresh_safe = utility_safe(fresh_utility, cfg)
    downproj_safe = bool(
        variant == "emb_head"
        or float(final_downproj["relative_frobenius"]) <= float(opt_cfg["max_relative_frobenius_downproj"])
    )
    feasible = bool(behavior_ok and fresh_safe and downproj_safe)

    report: Dict[str, Any] = {
        "schema_version": "rwku_sure_emb_head_ablation_v1_result_v1",
        "configuration_id": cfg["configuration_id"],
        "variant": variant,
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_records_accessed": False,
        "best_checkpoint_step": int(best_step),
        "best_selection_key": list(best_key),
        "selection_used_only_generated_sensitive_views_and_optimization_wikipedia": True,
        "previously_opened_v32_gate_used_for_selection": False,
        "fresh_confirmatory_gate_opened_after_selection": True,
        "frozen_base_head_proxy": final_proxy,
        "atomic": final_atomic,
        "embedding_delta_from_base": final_emb_delta,
        "lm_head_delta_from_stage1": final_head_stage1_delta,
        "lm_head_delta_from_base": final_head_base_delta,
        "downproj_delta": final_downproj,
        "downproj_materialization": materialization,
        "fresh_confirmatory_utility_kl": fresh_utility,
        "fresh_confirmatory_utility_checks": fresh_checks,
        "behavior_safe": behavior_ok,
        "fresh_utility_safe": fresh_safe,
        "downproj_norm_safe": downproj_safe,
        "feasible_under_declared_variant_gates": feasible,
        "training_seconds": time.perf_counter() - started,
        "history": history,
        "checkpoint_evaluations": checkpoints,
        "interpretation_note": (
            "Embedding and LM-head drift are reported, not thresholded, because this is an exploratory architecture ablation. "
            "The composite variant retains the predeclared <=1% down_proj relative-Frobenius gate."
        ),
    }
    write_json(out / "result.json", report)

    if args_save_checkpoint := False:
        del args_save_checkpoint
    return report


def main() -> None:
    args = parse_args()
    cfg = load_configuration(Path(args.configuration).resolve())
    out = Path(args.output_root).resolve() / cfg["configuration_id"] / args.variant
    if out.exists():
        raise FileExistsError(f"refusing to overwrite ablation output: {out}")
    out.mkdir(parents=True)
    problem = setup_problem(args, cfg, out)
    report = train_variant(problem, cfg, args.variant, out)

    if args.save_checkpoint and bool(report["feasible_under_declared_variant_gates"]):
        checkpoint = out / "checkpoint"
        problem["model"].save_pretrained(checkpoint)
        problem["tokenizer"].save_pretrained(checkpoint)
        print(f"Saved feasible development checkpoint: {checkpoint}")

    fresh = report["fresh_confirmatory_utility_kl"]
    proxy = report["frozen_base_head_proxy"]
    print("\nRWKU EMB/HEAD ABLATION RESULT")
    print(f"variant: {args.variant}")
    print(f"best checkpoint step: {report['best_checkpoint_step']}")
    print(f"frozen-W0 recovery: {proxy['recovery_percentage']:.2f}%")
    print(f"frozen-W0 minimum margin: {proxy['minimum_demotion_margin']:.6f}")
    print(
        "fresh utility KL mean/p95/max: {:.6f} / {:.6f} / {:.6f}".format(
            fresh["utility_kl_mean"], fresh["utility_kl_p95"], fresh["utility_kl_max"]
        )
    )
    print(f"embedding relative Frobenius: {report['embedding_delta_from_base']['relative_frobenius']:.6f}")
    print(f"LM-head relative Frobenius from Stage1: {report['lm_head_delta_from_stage1']['relative_frobenius']:.6f}")
    print(f"down_proj relative Frobenius: {report['downproj_delta']['relative_frobenius']:.6f}")
    print(f"feasible under declared variant gates: {report['feasible_under_declared_variant_gates']}")
    print(f"report: {out / 'result.json'}")


if __name__ == "__main__":
    main()
