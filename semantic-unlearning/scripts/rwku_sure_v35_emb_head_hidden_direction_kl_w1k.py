#!/usr/bin/env python3
"""RWKU v3.5: v3.2 decoder-aware repair + trainable embeddings and LM head.

Development-only Stephen King continuation. Starts from the same frozen Stage-1
sparse-head artifact used by v3.2. The v3.2 objective is preserved, but the
trainable parameter set is enlarged to:
  * input embedding matrix E,
  * untied deployed LM head W_edit,
  * final-layer MLP down_proj rank-r LoRA.

An immutable copy of the original tied vocabulary matrix W0 is retained as the
frozen decoder for answer-recovery probes and as the Base distribution readout
for Wikipedia KL. Official RWKU rows and neighbor prompts remain unavailable.

The already-opened v3.2 held-out Wiki gate is skipped. Selection uses only
generated sensitive views plus the 1K optimization Wiki pool. A fresh disjoint
1K Wiki slice is opened only once after candidate selection.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_experiment
import rwku_representation as representation
import rwku_sure_head_only_w1k as head
import rwku_sure_hidden_direction_v31_w1k as v31
import rwku_sure_hidden_direction_v32_kl_w1k as v32
import rwku_sure_repr_rescue_w1k as v2
import sure_canonical_core as core
import sure_minimal_two_stage as learner

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_v35_emb_head_hidden_direction_kl_w1k_seed0.json"
SOURCE_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_only_w1k_seed0.json"
SCHEMA = "rwku_sure_v35_emb_head_hidden_direction_kl_w1k_configuration_v1"
EXPERIMENT_ID = "rwku-h-w1k-stephen-king-emb-head-hidden-direction-seed0-v35-kl"
LEARNER_DIR = "sure_v35_emb_head_hidden_direction_w1k"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-bundle", type=Path, required=True)
    p.add_argument("--generator-receipt", type=Path, required=True)
    p.add_argument("--utility-cache", type=Path, required=True)
    p.add_argument("--wikipedia-dir", type=Path, required=True)
    p.add_argument("--source-head-only-run", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    core.write_json(path, dict(value))


def sha_indices(values: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in values], separators=(",", ":")).encode()).hexdigest()


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
            raise ValueError(f"v3.5 configuration changed {key}")
    components = cfg.get("trainable_components", {})
    expected_components = {
        "input_embeddings": True,
        "untied_lm_head": True,
        "final_mlp_down_proj_lora": True,
        "all_other_transformer_parameters": False,
    }
    for key, expected in expected_components.items():
        if components.get(key) != expected:
            raise ValueError(f"v3.5 trainable component changed {key}")
    opt = cfg.get("optimization", {})
    if opt.get("objective") != "v32_answer_level_frozen_W0_plus_edited_head_plus_exact_utility_KL_with_joint_embedding_head_downproj_training":
        raise ValueError("v3.5 objective changed")
    if opt.get("sensitive_view_scope") != "all_generated_atomic_views":
        raise ValueError("v3.5 sensitive view scope changed")
    if tuple(opt.get("rank_ladder", ())) != (1, 2, 4):
        raise ValueError("v3.5 rank ladder must remain 1,2,4")
    if int(opt.get("last_n_layers", 0)) != 1 or tuple(opt.get("target_modules", ())) != ("down_proj",):
        raise ValueError("v3.5 representation target changed")
    locked_opt = {
        "steps": 300,
        "embedding_learning_rate": 0.0001,
        "lm_head_learning_rate": 0.0001,
        "downproj_learning_rate": 0.0005,
        "weight_decay": 0.0,
        "grad_clip": 1.0,
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
        "adapter_l2_weight": 0.0001,
        "utility_train_prompt_count": 1000,
        "opened_v32_gate_skip_count": 1000,
        "fresh_confirmatory_gate_prompt_count": 1000,
        "utility_context_batch_size": 4,
        "max_relative_frobenius_downproj_delta": 0.01,
    }
    for key, expected in locked_opt.items():
        if opt.get(key) != expected:
            raise ValueError(f"v3.5 optimization setting changed {key}")
    if tuple(float(x) for x in opt.get("candidate_scales", ())) != (0.125, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("v3.5 candidate scales changed")
    acc = cfg.get("acceptance", {})
    locked_acc = {
        "required_pairwise_margin": 0.01,
        "required_direct_success": 100.0,
        "required_other_atomic_view_success": 100.0,
        "max_frozen_base_head_recovery": 0.0,
        "min_frozen_base_head_demotion_margin": 0.05,
        "max_total_lm_head_delta_norm_from_base": 1.5,
        "utility_kl_mean_budget": 0.01,
        "utility_kl_p95_budget": 0.05,
        "utility_kl_max_budget": 0.5,
        "checkpoint_dtype": "bf16",
        "device_map": "single",
    }
    for key, expected in locked_acc.items():
        if acc.get(key) != expected:
            raise ValueError(f"v3.5 acceptance changed {key}")
    boundary = cfg.get("data_boundary", {})
    for key in ("official_rwku_records_available_to_learner", "official_rwku_records_used_for_checkpoint_selection", "neighbor_prompts_used_for_training_or_selection"):
        if boundary.get(key) is not False:
            raise ValueError(f"v3.5 data boundary changed {key}")
    if boundary.get("external_wikipedia_only_utility") is not True:
        raise ValueError("v3.5 utility must remain external Wikipedia")
    if boundary.get("previously_opened_v32_gate_excluded_from_training_and_selection") is not True:
        raise ValueError("opened v3.2 utility gate must remain excluded")
    if boundary.get("utility_train_opened_gate_fresh_gate_overlap_allowed") is not False:
        raise ValueError("v3.5 utility slices must remain disjoint")
    return cfg


def state_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(output_root=Path(args.output_root), experiment_id=str(args.experiment_id), training_source=rwku_experiment.TRAINING_SOURCE_TARGET_ONLY)


def verify_prepared_state(args: argparse.Namespace, cfg: Mapping[str, Any]) -> Tuple[SimpleNamespace, Path]:
    state_args = state_namespace(args)
    state = rwku_experiment._read_state(state_args)
    if state.get("state") != "PREPARED":
        raise ValueError(f"v3.5 requires PREPARED state, got {state.get('state')}")
    if state.get("training_source") != rwku_experiment.TRAINING_SOURCE_TARGET_ONLY:
        raise ValueError("v3.5 prepared state belongs to another training source")
    if state.get("official_evaluation_opened") is not False:
        raise ValueError("official RWKU evaluation is already opened")
    target = state.get("target", {})
    if target.get("seed") != 0 or target.get("subject") != cfg["target_entity"]:
        raise ValueError("prepared state is not Stephen King seed0")
    run_dir = Path(args.output_root).resolve() / str(args.experiment_id)
    if (run_dir / "checkpoint_receipt.json").exists():
        raise FileExistsError("v3.5 run unexpectedly has a checkpoint receipt")
    return state_args, run_dir


@torch.no_grad()
def tensor_delta_report(current: torch.Tensor, reference: torch.Tensor, chunk_rows: int = 2048) -> Dict[str, float]:
    if tuple(current.shape) != tuple(reference.shape):
        raise ValueError("delta-report shapes differ")
    delta_sq = 0.0
    ref_sq = 0.0
    rows = int(current.shape[0])
    for start in range(0, rows, int(chunk_rows)):
        stop = min(rows, start + int(chunk_rows))
        cur = current[start:stop].detach().float()
        ref = reference[start:stop].to(device=cur.device, dtype=torch.float32)
        delta_sq += float((cur - ref).square().sum().item())
        ref_sq += float(ref.square().sum().item())
    delta = math.sqrt(max(delta_sq, 0.0))
    ref = math.sqrt(max(ref_sq, 0.0))
    return {"delta_frobenius": delta, "reference_frobenius": ref, "relative_frobenius": delta / max(ref, 1e-12)}


@torch.no_grad()
def restore_weight(current: torch.Tensor, reference: torch.Tensor) -> None:
    current.copy_(reference.to(device=current.device, dtype=current.dtype))


def utility_checks(report: Mapping[str, Any], cfg: Mapping[str, Any]) -> Tuple[Dict[str, bool], bool]:
    acc = cfg["acceptance"]
    checks = {
        "mean": float(report["utility_kl_mean"]) <= float(acc["utility_kl_mean_budget"]),
        "p95": float(report["utility_kl_p95"]) <= float(acc["utility_kl_p95_budget"]),
        "max": float(report["utility_kl_max"]) <= float(acc["utility_kl_max_budget"]),
    }
    return checks, bool(all(checks.values()))


def behavior_report(model, tokenizer, prompt_records, cases, base_w0, cfg, llama_like, device) -> Tuple[Dict[str, Any], Dict[str, Any], bool]:
    proxy = v31.answer_proxy_report(model, cases, base_w0, device)
    atomic = head.materialized_atomic_report(model, tokenizer, prompt_records, device, llama_like=llama_like, required_margin=float(cfg["acceptance"]["required_pairwise_margin"]))
    return proxy, atomic, bool(v2.behavior_safe(atomic) and v31.proxy_safe(proxy, cfg))


def setup_problem(args: argparse.Namespace, cfg: Mapping[str, Any], out: Path):
    source = v2.verify_source_run(Path(args.source_head_only_run).resolve(), cfg)
    source_cfg = head.load_locked_configuration(SOURCE_CONFIGURATION)
    views, bundle_audit, generator_audit = head.load_atomic_bundle(Path(args.training_bundle).resolve(), Path(args.generator_receipt).resolve(), source_cfg)
    generator_model_audit = head.validate_generator_base_model(generator_audit, args.model_path)
    gagd.set_seed(int(cfg["seed"]))
    gagd.require_cuda_if_needed(cfg["acceptance"]["device_map"])
    margs = argparse.Namespace(model_path=args.model_path, dtype=cfg["acceptance"]["checkpoint_dtype"], device_map=cfg["acceptance"]["device_map"], gradient_checkpointing=False)
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
        raise ValueError("v3.5 requires the base model to begin with tied vocabulary weights")
    identity = wikipedia.model_identity(model, tokenizer, args.model_path)
    output_layer = core.untie_and_freeze_output_head(model)
    input_weight = model.get_input_embeddings().weight
    if input_weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("v3.5 output head failed to untie from input embeddings")
    base_w0 = input_weight.detach().clone()
    base_w0.requires_grad_(False)
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
    second, utility_hidden, utility_lse, utility_meta = learner.load_utility_cache(Path(args.utility_cache).resolve(), expected_sample_size=optimization.utility_sample_size, expected_prompt_count=optimization.utility_prompt_count, expected_hidden_size=int(output_layer.weight.shape[1]), expected_model_probe=identity["model_probe_sha256"], expected_tokenizer_probe=identity["tokenizer_probe_sha256"])
    del second
    utility_audit = head.validate_w1k_utility_metadata(utility_meta, source_cfg)
    selected_base_probs = v2.selected_base_probabilities_full_head(output_layer, selected_ids, utility_hidden, utility_lse, batch_size=optimization.utility_eval_batch_size)
    train_idx, guard_idx, pool_report = learner.build_disjoint_token_conditioned_utility_pools(selected_base_probabilities=selected_base_probs, selected_ids=selected_ids, topk_per_row=optimization.utility_token_topk_per_row, uniform_prompt_count=optimization.utility_uniform_prompt_count, split_seed=optimization.utility_pool_seed)
    core.materialize_output_delta(output_layer, selected_ids, stage1_delta)
    stage1_head_reference = output_layer.weight.detach().to(device="cpu", copy=True)
    actual_stage1_delta = learner.actual_selected_delta(output_layer, selected_ids, base_head_rows.float())
    source_stage1 = head.materialized_atomic_report(model, tokenizer, prompt_records, device, llama_like=llama_like, required_margin=float(cfg["acceptance"]["required_pairwise_margin"]))
    declared = read_json(source["stage1_report"])
    if source_stage1.get("pairwise_margin_failure_positions") != declared.get("pairwise_margin_failure_positions"):
        raise RuntimeError("reloaded Stage-1 failure positions differ from source run")
    neutral_ids = v2._completion_token_ids(tokenizer, str(cfg["neutral_target"]), llama_like)
    sensitive_ids = [x for x in selected_ids if x not in set(neutral_ids)]
    cases, direction_audit = v31.build_answer_cases(model, tokenizer, prompt_records, base_w0, sensitive_ids, neutral_ids, float(cfg["optimization"]["frozen_base_head_training_margin"]), llama_like, device)
    v31._RUNTIME["tokenizer"] = tokenizer
    v31._RUNTIME["answer_eval_batch_size"] = int(cfg["optimization"]["answer_eval_batch_size"])
    texts, wiki_meta = wikipedia.load_wikipedia_train(Path(args.wikipedia_dir).resolve())
    opt = cfg["optimization"]
    train_count = int(opt["utility_train_prompt_count"])
    opened_skip = int(opt["opened_v32_gate_skip_count"])
    fresh_count = int(opt["fresh_confirmatory_gate_prompt_count"])
    if int(train_idx.numel()) < train_count:
        raise RuntimeError(f"utility train pool has {int(train_idx.numel())}, needs {train_count}")
    if int(guard_idx.numel()) < opened_skip + fresh_count:
        raise RuntimeError(f"utility guard pool has {int(guard_idx.numel())}, needs {opened_skip + fresh_count} to exclude opened v3.2 gate")
    train_indices = train_idx[:train_count].tolist()
    opened_indices = guard_idx[:opened_skip].tolist()
    fresh_indices = guard_idx[opened_skip: opened_skip + fresh_count].tolist()
    a, b, c = set(train_indices), set(opened_indices), set(fresh_indices)
    if (a & b) or (a & c) or (b & c):
        raise RuntimeError("v3.5 utility train/opened/fresh slices overlap")
    train_contexts = v2.build_utility_contexts(tokenizer, texts, utility_meta, utility_hidden, train_indices)
    fresh_contexts = v2.build_utility_contexts(tokenizer, texts, utility_meta, utility_hidden, fresh_indices)
    write_json(out / "protocol_report.json", {
        "configuration_id": cfg["configuration_id"], "development_only": True, "posthoc_development_target": True,
        "official_rwku_records_accessed": False, "source_head_only_run": str(Path(args.source_head_only_run).resolve()),
        "bundle_audit": bundle_audit, "generator_model_audit": generator_model_audit, "utility_audit": utility_audit,
        "utility_pool_report": pool_report, "wikipedia_dataset": wiki_meta, "utility_train_prompt_count": len(train_contexts),
        "opened_v32_gate_excluded_prompt_count": len(opened_indices), "fresh_confirmatory_gate_prompt_count": len(fresh_contexts),
        "utility_train_indices_sha256": sha_indices(train_indices), "opened_v32_gate_indices_sha256": sha_indices(opened_indices),
        "fresh_confirmatory_gate_indices_sha256": sha_indices(fresh_indices), "utility_slices_disjoint": True,
        "base_readout_validation_max_abs_diff": maxdiff, "source_stage1_sparse_head_delta_norm": float(actual_stage1_delta.norm().detach().cpu()),
        "source_stage1_atomic": source_stage1, "direction_audit": direction_audit, "trainable_components": cfg["trainable_components"],
    })
    return {
        "model": model, "tokenizer": tokenizer, "device": device, "llama_like": llama_like, "prompt_records": prompt_records,
        "cases": cases, "input_weight": input_weight, "output_layer": output_layer, "base_w0": base_w0,
        "stage1_head_reference": stage1_head_reference, "train_contexts": train_contexts, "fresh_contexts": fresh_contexts,
    }


def snapshot_candidate(input_weight: torch.Tensor, output_weight: torch.Tensor, handles: Sequence[Any], *, rank: int, step: int, scale: float, selection_key: Tuple[Any, ...], candidate_report: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "rank": int(rank), "step": int(step), "scale": float(scale), "selection_key": tuple(selection_key),
        "input_weight": input_weight.detach().to(device="cpu", copy=True), "output_weight": output_weight.detach().to(device="cpu", copy=True),
        "adapter_state": v31.adapter_state(handles), "candidate_report": dict(candidate_report),
    }


def train_rank(problem: Mapping[str, Any], cfg: Mapping[str, Any], rank: int, out: Path) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
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
    restore_weight(input_weight, base_w0)
    restore_weight(output_weight, problem["stage1_head_reference"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_weight.requires_grad_(True)
    output_weight.requires_grad_(True)
    rep_cfg = representation.RepresentationConfig(steps=int(opt_cfg["steps"]), learning_rate=float(opt_cfg["downproj_learning_rate"]), weight_decay=float(opt_cfg["weight_decay"]), rank=int(rank), alpha=float(rank), dropout=0.0, layer_indices=(), last_n_layers=1, target_modules=("down_proj",), seed=int(cfg["seed"]))
    handles = list(representation.inject_lora_adapters(model, rep_cfg))
    originals = list(v2.capture_adapter_base_weights(handles))
    adapter_parameters = list(representation.adapter_parameters(handles))
    representation.set_adapter_scale(handles, 1.0)
    optimizer = torch.optim.AdamW([
        {"params": [input_weight], "lr": float(opt_cfg["embedding_learning_rate"]), "weight_decay": float(opt_cfg["weight_decay"])},
        {"params": [output_weight], "lr": float(opt_cfg["lm_head_learning_rate"]), "weight_decay": float(opt_cfg["weight_decay"])},
        {"params": adapter_parameters, "lr": float(opt_cfg["downproj_learning_rate"]), "weight_decay": float(opt_cfg["weight_decay"])},
    ])
    trainable = [input_weight, output_weight, *adapter_parameters]
    answer_bs = min(int(opt_cfg["answer_batch_size"]), len(cases))
    utility_bs = min(int(opt_cfg["utility_kl_batch_size"]), len(train_contexts))
    checkpoint_take = min(int(opt_cfg["checkpoint_kl_prompt_count"]), len(train_contexts))
    interval = int(opt_cfg["checkpoint_interval"])
    history: List[Dict[str, Any]] = []
    attempts: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[Any, ...]] = None
    started = time.perf_counter()
    model.eval()
    for step_index in range(int(opt_cfg["steps"])):
        optimizer.zero_grad(set_to_none=True)
        answer_indices = [(step_index * answer_bs + j) % len(cases) for j in range(answer_bs)]
        answer_batch = [cases[i] for i in answer_indices]
        base_sensitive, base_neutral, edited_sensitive, edited_neutral = v31.answer_nlls(model, tokenizer, answer_batch, base_w0, device)
        base_sep = base_sensitive - base_neutral
        edited_sep = edited_sensitive - edited_neutral
        base_loss = torch.nn.functional.relu(float(opt_cfg["frozen_base_head_training_margin"]) - base_sep).square().mean()
        edited_loss = torch.nn.functional.relu(float(opt_cfg["edited_head_pairwise_target"]) - edited_sep).square().mean()
        l2 = v2.adapter_l2(handles)
        answer_objective = float(opt_cfg["frozen_base_head_answer_weight"]) * base_loss + float(opt_cfg["edited_head_answer_weight"]) * edited_loss + float(opt_cfg["adapter_l2_weight"]) * l2
        if not torch.isfinite(answer_objective):
            raise FloatingPointError("v3.5 answer objective became non-finite")
        answer_objective.backward()
        utility_indices = [(step_index * utility_bs + j) % len(train_contexts) for j in range(utility_bs)]
        utility_kl_values: List[torch.Tensor] = []
        utility_hidden_values: List[torch.Tensor] = []
        for index in utility_indices:
            utility_kl_i, _, utility_hidden_i = v32.differentiable_utility_metrics(model, tokenizer, [train_contexts[index]], base_w0, device=device)
            utility_objective_i = (float(opt_cfg["utility_hidden_weight"]) * utility_hidden_i + float(opt_cfg["utility_kl_weight"]) * utility_kl_i) / float(utility_bs)
            if not torch.isfinite(utility_objective_i):
                raise FloatingPointError("v3.5 utility objective became non-finite")
            utility_objective_i.backward()
            utility_kl_values.append(utility_kl_i.detach())
            utility_hidden_values.append(utility_hidden_i.detach())
        torch.nn.utils.clip_grad_norm_(trainable, float(opt_cfg["grad_clip"]))
        optimizer.step()
        utility_kl = torch.stack(utility_kl_values).mean()
        utility_hidden = torch.stack(utility_hidden_values).mean()
        total_for_log = answer_objective.detach() + float(opt_cfg["utility_hidden_weight"]) * utility_hidden + float(opt_cfg["utility_kl_weight"]) * utility_kl
        step = step_index + 1
        if step == 1 or step % 25 == 0 or step == int(opt_cfg["steps"]):
            row = {
                "step": step, "rank": int(rank), "loss": float(total_for_log.cpu()), "batch_frozen_W0_hinge": float(base_loss.detach().cpu()),
                "batch_frozen_W0_separation_mean": float(base_sep.mean().detach().cpu()), "batch_frozen_W0_recovery_percentage": float((base_sep <= 0).float().mean().mul(100).detach().cpu()),
                "batch_edited_head_hinge": float(edited_loss.detach().cpu()), "batch_edited_head_separation_mean": float(edited_sep.mean().detach().cpu()),
                "utility_kl_mean": float(utility_kl.cpu()), "utility_hidden_relative_mse": float(utility_hidden.cpu()),
            }
            history.append(row)
            print("v35-rank{} step {:3d}: loss={:.6f} W0_sep={:.4f} W0_rec={:.2f}% edit_sep={:.4f} wiki_kl={:.6f} wiki_hidden={:.6f}".format(rank, step, row["loss"], row["batch_frozen_W0_separation_mean"], row["batch_frozen_W0_recovery_percentage"], row["batch_edited_head_separation_mean"], row["utility_kl_mean"], row["utility_hidden_relative_mse"]))
        if step % interval != 0 and step != int(opt_cfg["steps"]):
            continue
        for scale in [float(x) for x in opt_cfg["candidate_scales"]]:
            materialization = v2.materialize_adapter_candidate(handles, originals, scale)
            proxy, atomic, behavior_safe = behavior_report(model, tokenizer, prompt_records, cases, base_w0, cfg, llama_like, device)
            downproj = v2.representation_delta_report(handles, originals)
            emb_delta = tensor_delta_report(input_weight, base_w0)
            head_total = tensor_delta_report(output_weight, base_w0)
            head_from_stage1 = tensor_delta_report(output_weight, problem["stage1_head_reference"])
            head_safe = bool(float(head_total["delta_frobenius"]) <= float(cfg["acceptance"]["max_total_lm_head_delta_norm_from_base"]))
            downproj_safe = bool(float(downproj["relative_frobenius"]) <= float(opt_cfg["max_relative_frobenius_downproj_delta"]))
            pre_safe = bool(behavior_safe and head_safe and downproj_safe)
            candidate: Dict[str, Any] = {
                "rank": int(rank), "step": int(step), "scale": float(scale), "materialization": materialization,
                "frozen_base_head_proxy": proxy, "atomic": atomic, "behavior_safe": behavior_safe, "embedding_delta_from_base": emb_delta,
                "lm_head_delta_from_base": head_total, "lm_head_delta_from_stage1": head_from_stage1, "downproj_delta": downproj,
                "head_norm_safe": head_safe, "downproj_norm_safe": downproj_safe, "pre_utility_safe": pre_safe, "official_rwku_records_accessed": False,
            }
            if pre_safe:
                utility = v32.checkpoint_utility_report(model, tokenizer, train_contexts[:checkpoint_take], base_w0, device=device)
                candidate["optimization_utility"] = utility
                candidate["selection_key"] = [0, float(utility["utility_kl_mean"]), float(utility["utility_kl_p95"]), float(utility["utility_kl_max"]), float(emb_delta["relative_frobenius"]) + float(head_from_stage1["relative_frobenius"]) + float(downproj["relative_frobenius"])]
                key = tuple(candidate["selection_key"])
            else:
                margin_shortfall = max(0.0, float(cfg["acceptance"]["min_frozen_base_head_demotion_margin"]) - float(proxy["minimum_demotion_margin"]))
                atomic_failures = int(atomic.get("direct_margin_failures", 0)) + int(atomic.get("generated_subject_margin_failures", 0))
                candidate["selection_key"] = [1, int(proxy["recovery_count"]), margin_shortfall, atomic_failures, 0 if head_safe else 1, 0 if downproj_safe else 1]
                key = tuple(candidate["selection_key"])
            attempts.append(candidate)
            print("  v35 candidate step={} rank={} scale={}: FS={} other={} W0_rec={:.2f}% W0_min={:.4f} emb_rel={:.6f} head_norm={:.6f} down_rel={:.6f} pre_safe={}".format(step, rank, scale, atomic.get("FS"), atomic.get("generated_subject_FS"), float(proxy["recovery_percentage"]), float(proxy["minimum_demotion_margin"]), float(emb_delta["relative_frobenius"]), float(head_total["delta_frobenius"]), float(downproj["relative_frobenius"]), pre_safe))
            if pre_safe:
                utility = candidate["optimization_utility"]
                print("    opt Wiki KL mean/p95/max={:.6f}/{:.6f}/{:.6f}".format(float(utility["utility_kl_mean"]), float(utility["utility_kl_p95"]), float(utility["utility_kl_max"])))
            if pre_safe and (best_key is None or key < best_key):
                best_key = key
                best = snapshot_candidate(input_weight, output_weight, handles, rank=rank, step=step, scale=scale, selection_key=key, candidate_report=candidate)
            v2.restore_adapter_base_weights(handles, originals)
            representation.set_adapter_scale(handles, 1.0)
    report = {
        "rank": int(rank), "steps": int(opt_cfg["steps"]), "history": history, "attempts": attempts,
        "best_pre_utility_candidate": None if best is None else {"rank": best["rank"], "step": best["step"], "scale": best["scale"], "selection_key": list(best["selection_key"]), "candidate_report": best["candidate_report"]},
        "training_seconds": time.perf_counter() - started, "official_rwku_records_accessed": False,
    }
    write_json(out / f"rank{rank}_training_and_selection.json", report)
    v2.restore_adapter_base_weights(handles, originals)
    representation.remove_lora_adapters(handles, merge_scale=0.0)
    restore_weight(input_weight, base_w0)
    restore_weight(output_weight, problem["stage1_head_reference"])
    input_weight.requires_grad_(False)
    output_weight.requires_grad_(False)
    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best, report


def restore_selected_candidate(problem: Mapping[str, Any], cfg: Mapping[str, Any], selected: Mapping[str, Any]):
    model = problem["model"]
    restore_weight(problem["input_weight"], selected["input_weight"])
    restore_weight(problem["output_layer"].weight, selected["output_weight"])
    rank = int(selected["rank"])
    opt_cfg = cfg["optimization"]
    rep_cfg = representation.RepresentationConfig(steps=int(opt_cfg["steps"]), learning_rate=float(opt_cfg["downproj_learning_rate"]), weight_decay=float(opt_cfg["weight_decay"]), rank=rank, alpha=float(rank), dropout=0.0, layer_indices=(), last_n_layers=1, target_modules=("down_proj",), seed=int(cfg["seed"]))
    handles = list(representation.inject_lora_adapters(model, rep_cfg))
    originals = list(v2.capture_adapter_base_weights(handles))
    v31.restore_state(handles, selected["adapter_state"])
    representation.set_adapter_scale(handles, 1.0)
    materialization = v2.materialize_adapter_candidate(handles, originals, float(selected["scale"]))
    return handles, originals, materialization


def main() -> None:
    args = parse_args()
    cfg = load_configuration(Path(args.configuration).resolve())
    if args.experiment_id != cfg["configuration_id"]:
        raise ValueError("experiment-id must equal locked v3.5 configuration ID")
    _, run_dir = verify_prepared_state(args, cfg)
    out = run_dir / LEARNER_DIR
    if out.exists():
        raise FileExistsError(f"refusing to overwrite v3.5 learner output: {out}")
    out.mkdir(parents=True)
    problem = setup_problem(args, cfg, out)
    global_best: Optional[Dict[str, Any]] = None
    global_key: Optional[Tuple[Any, ...]] = None
    rank_summaries: List[Dict[str, Any]] = []
    for rank in [int(x) for x in cfg["optimization"]["rank_ladder"]]:
        best, report = train_rank(problem, cfg, rank, out)
        rank_summaries.append({"rank": rank, "best_pre_utility_candidate": report["best_pre_utility_candidate"]})
        if best is not None:
            key = tuple(best["selection_key"])
            if global_key is None or key < global_key:
                global_key = key
                global_best = best
    write_json(out / "rank_selection_summary.json", {
        "configuration_id": cfg["configuration_id"], "rank_summaries": rank_summaries,
        "selected": None if global_best is None else {"rank": global_best["rank"], "step": global_best["step"], "scale": global_best["scale"], "selection_key": list(global_best["selection_key"])},
        "selection_used_opened_v32_gate": False, "official_rwku_records_accessed": False,
    })
    if global_best is None:
        write_json(out / "infeasible.json", {
            "configuration_id": cfg["configuration_id"],
            "reason": "no checkpoint x scale candidate passed generated-view behavior, frozen-W0 recovery, LM-head norm, and <=1% down_proj pre-gates",
            "opened_v32_gate_used_for_selection": False, "official_rwku_records_accessed": False,
        })
        raise RuntimeError("RWKU v3.5 found no pre-utility-safe candidate")
    handles, originals, materialization = restore_selected_candidate(problem, cfg, global_best)
    model = problem["model"]
    tokenizer = problem["tokenizer"]
    device = problem["device"]
    final_proxy, final_atomic, final_behavior_safe = behavior_report(model, tokenizer, problem["prompt_records"], problem["cases"], problem["base_w0"], cfg, problem["llama_like"], device)
    final_embedding = tensor_delta_report(problem["input_weight"], problem["base_w0"])
    final_head_total = tensor_delta_report(problem["output_layer"].weight, problem["base_w0"])
    final_head_stage1 = tensor_delta_report(problem["output_layer"].weight, problem["stage1_head_reference"])
    final_downproj = v2.representation_delta_report(handles, originals)
    final_head_safe = bool(float(final_head_total["delta_frobenius"]) <= float(cfg["acceptance"]["max_total_lm_head_delta_norm_from_base"]))
    final_downproj_safe = bool(float(final_downproj["relative_frobenius"]) <= float(cfg["optimization"]["max_relative_frobenius_downproj_delta"]))
    fresh_utility = v32.checkpoint_utility_report(model, tokenizer, problem["fresh_contexts"], problem["base_w0"], device=device)
    fresh_checks, fresh_safe = utility_checks(fresh_utility, cfg)
    feasible = bool(final_behavior_safe and final_head_safe and final_downproj_safe and fresh_safe)
    result = {
        "schema_version": "rwku_sure_v35_emb_head_hidden_direction_result_v1", "configuration_id": cfg["configuration_id"], "method": cfg["method"],
        "development_only": True, "posthoc_development_target": True, "official_rwku_records_accessed": False,
        "selected_rank": int(global_best["rank"]), "selected_checkpoint_step": int(global_best["step"]), "selected_downproj_scale": float(global_best["scale"]),
        "selection_key": list(global_best["selection_key"]), "downproj_materialization": materialization, "frozen_base_head_proxy": final_proxy, "atomic": final_atomic,
        "embedding_delta_from_base": final_embedding, "lm_head_delta_from_base": final_head_total, "lm_head_delta_from_stage1": final_head_stage1, "downproj_delta": final_downproj,
        "behavior_safe": final_behavior_safe, "lm_head_norm_safe": final_head_safe, "downproj_norm_safe": final_downproj_safe,
        "fresh_confirmatory_utility_kl": fresh_utility, "fresh_confirmatory_utility_checks": fresh_checks, "fresh_confirmatory_utility_safe": fresh_safe,
        "feasible_under_v35_gates": feasible, "previously_opened_v32_gate_used_for_training_or_selection": False,
        "fresh_confirmatory_gate_opened_after_selection": True,
        "interpretation_note": "v3.5 is a post-hoc development extension of v3.2. The frozen W0 probe is immutable despite trainable input embeddings and deployed LM head. Embedding drift is reported explicitly; inherited hard intervention gates are LM-head total delta <=1.5 and down_proj relative Frobenius <=1%.",
    }
    write_json(out / "result.json", result)
    representation.remove_lora_adapters(handles, merge_scale=0.0)
    if args.save_checkpoint and feasible:
        checkpoint = out / "checkpoint"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        print(f"Saved feasible v3.5 development checkpoint: {checkpoint}")
    print("\nRWKU v3.5 RESULT")
    print("selected rank/step/scale: {}/{}/{}".format(result["selected_rank"], result["selected_checkpoint_step"], result["selected_downproj_scale"]))
    print("frozen-W0 recovery/min margin: {:.2f}% / {:.6f}".format(float(final_proxy["recovery_percentage"]), float(final_proxy["minimum_demotion_margin"])))
    print("atomic direct/other: {} / {}".format(final_atomic.get("FS"), final_atomic.get("generated_subject_FS")))
    print("embedding rel Frobenius: {:.6f}".format(float(final_embedding["relative_frobenius"])))
    print("LM-head total delta norm from Base: {:.6f}".format(float(final_head_total["delta_frobenius"])))
    print("down_proj rel Frobenius: {:.6f}".format(float(final_downproj["relative_frobenius"])))
    print("fresh Wiki KL mean/p95/max: {:.6f} / {:.6f} / {:.6f}".format(float(fresh_utility["utility_kl_mean"]), float(fresh_utility["utility_kl_p95"]), float(fresh_utility["utility_kl_max"])))
    print(f"feasible under v3.5 gates: {feasible}")
    print(f"result: {out / 'result.json'}")


if __name__ == "__main__":
    main()
