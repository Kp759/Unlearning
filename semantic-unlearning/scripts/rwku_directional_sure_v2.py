#!/usr/bin/env python3
"""Directional SURE v2 for RWKU target-only development.

This is a sparse vocabulary-interface port of Directional SURE v2:

* every transformer parameter is frozen;
* only content-bearing sensitive input-embedding rows are trainable;
* the LM head is untied and only the same sensitive rows are trainable;
* non-sensitive embedding/head rows remain exactly Base by construction;
* the loss is canonical SURE GA + same-prompt non-sensitive GD KL;
* LM-head GA gradients are restricted to a sensitive-exclusive basis B_S;
* LM-head GD/locality gradients are restricted to a protected basis B_P;
* B_S and B_P are refreshed every 25 optimization steps;
* official RWKU paraphrase/neighborhood/retain/PPL artifacts are unavailable.

The protected basis, checkpoint-selection utility, and final utility gate use
three disjoint slices of target-excluded external Wikipedia. Stephen King seed0
is post-hoc development only; this script never opens official RWKU evaluation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_experiment
import rwku_setting5e_utility_controlled as sparse_rows
import rwku_sure_head_only_w1k as head
import sure_canonical_core as core

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "directional_sure_v2_seed0.json"
SOURCE_BUNDLE_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_only_w1k_seed0.json"
SCHEMA = "rwku_directional_sure_v2_configuration_v1"
EXPERIMENT_ID = "rwku-directional-sure-v2-stephen-king-seed0"
LEARNER_DIR = "directional_sure_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-bundle", type=Path, required=True)
    p.add_argument("--generator-receipt", type=Path, required=True)
    p.add_argument("--wikipedia-dir", type=Path, required=True)
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


def sha_ints(values: Sequence[int]) -> str:
    payload = json.dumps([int(x) for x in values], separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Directional SURE v2 configuration changed {key}")

    components = cfg.get("trainable_components", {})
    locked_components = {
        "sensitive_input_embedding_rows": True,
        "sensitive_untied_lm_head_rows": True,
        "non_sensitive_input_embedding_rows": False,
        "non_sensitive_lm_head_rows": False,
        "transformer_parameters": False,
    }
    for key, expected in locked_components.items():
        if components.get(key) != expected:
            raise ValueError(f"Directional SURE v2 trainable component changed {key}")

    opt = cfg.get("optimization", {})
    locked_opt = {
        "objective": "canonical_SURE_GA_plus_same_prompt_non_sensitive_GD_KL_with_directional_lm_head_gradient_decomposition",
        "training_view_scope": "all_target_only_generated_atomic_views",
        "steps": 600,
        "batch_size": 1,
        "cache_batch_size": 8,
        "embedding_learning_rate": 0.00005,
        "lm_head_learning_rate": 0.0001,
        "ga_weight": 2.0,
        "gd_weight": 1.0,
        "grad_clip": 1.0,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "basis_refresh_interval": 25,
        "sensitive_exclusive_basis_rank": 8,
        "protected_basis_rank": 32,
        "protected_basis_context_count": 256,
        "selection_utility_context_count": 256,
        "fresh_gate_utility_context_count": 1000,
        "utility_max_length": 128,
        "utility_batch_size": 4,
        "checkpoint_interval": 25,
    }
    for key, expected in locked_opt.items():
        if opt.get(key) != expected:
            raise ValueError(f"Directional SURE v2 optimization changed {key}")

    acc = cfg.get("acceptance", {})
    locked_acc = {
        "required_pairwise_margin": 0.01,
        "required_direct_success": 100.0,
        "required_other_atomic_view_success": 100.0,
        "utility_kl_mean_budget": 0.01,
        "utility_kl_p95_budget": 0.05,
        "utility_kl_max_budget": 0.5,
        "checkpoint_dtype": "bf16",
        "device_map": "single",
        "non_sensitive_embedding_rows_exact_base": True,
        "non_sensitive_lm_head_rows_exact_base": True,
        "transformer_exactly_frozen": True,
    }
    for key, expected in locked_acc.items():
        if acc.get(key) != expected:
            raise ValueError(f"Directional SURE v2 acceptance changed {key}")

    boundary = cfg.get("data_boundary", {})
    false_keys = (
        "official_rwku_records_available_to_learner",
        "official_rwku_records_used_for_checkpoint_selection",
        "official_rwku_paraphrase_seen",
        "official_rwku_neighborhood_seen",
        "official_rwku_retain_seen",
        "official_rwku_ppl_text_seen",
        "basis_selection_fresh_gate_overlap_allowed",
    )
    for key in false_keys:
        if boundary.get(key) is not False:
            raise ValueError(f"Directional SURE v2 boundary changed {key}")
    if boundary.get("external_wikipedia_only_for_directional_protection_and_utility") is not True:
        raise ValueError("Directional SURE v2 utility must be external Wikipedia only")
    if boundary.get("wikipedia_target_casefold_exclusion") != "stephen king":
        raise ValueError("Directional SURE v2 target exclusion changed")
    if int(boundary.get("wikipedia_exclude_first_documents", -1)) != 20:
        raise ValueError("Directional SURE v2 Wikipedia prefix exclusion changed")
    return cfg


def state_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=Path(args.output_root),
        experiment_id=str(args.experiment_id),
        training_source=rwku_experiment.TRAINING_SOURCE_TARGET_ONLY,
    )


def verify_prepared_state(args: argparse.Namespace, cfg: Mapping[str, Any]) -> Path:
    ns = state_namespace(args)
    state = rwku_experiment._read_state(ns)
    if state.get("state") != "PREPARED":
        raise ValueError(f"Directional SURE v2 requires PREPARED state, got {state.get('state')}")
    if state.get("training_source") != rwku_experiment.TRAINING_SOURCE_TARGET_ONLY:
        raise ValueError("Prepared state belongs to another RWKU training track")
    if state.get("official_evaluation_opened") is not False:
        raise ValueError("Official RWKU evaluation is already opened")
    target = state.get("target", {})
    if target.get("seed") != 0 or target.get("subject") != cfg["target_entity"]:
        raise ValueError("Prepared state is not Stephen King seed0")
    run_dir = Path(args.output_root).resolve() / str(args.experiment_id)
    if (run_dir / "checkpoint_receipt.json").exists():
        raise FileExistsError("Directional SURE v2 run unexpectedly has a checkpoint receipt")
    return run_dir


def _parameter_versions_except_vocab(model: torch.nn.Module) -> Dict[str, int]:
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    excluded = {id(input_weight), id(output_weight)}
    return {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
        if id(parameter) not in excluded
    }


def assert_transformer_versions(model: torch.nn.Module, before: Mapping[str, int]) -> None:
    after = _parameter_versions_except_vocab(model)
    if dict(before) != after:
        changed = sorted(
            name for name in set(before) | set(after) if before.get(name) != after.get(name)
        )
        raise RuntimeError(f"Frozen transformer parameter versions changed: {changed[:10]}")


def _content_sensitive_rows(
    tokenizer: Any,
    cases: Sequence[core.SensitivePredictionCase],
    tids: torch.Tensor,
    source_cfg: Mapping[str, Any],
    prompt_count: int,
) -> Tuple[List[int], Dict[str, Any]]:
    empty_neutral = torch.empty((0,), dtype=torch.long, device=tids.device)
    selected, audit = head.select_edit_row_ids(
        tokenizer,
        cases,
        tids,
        empty_neutral,
        prompt_count=prompt_count,
        configuration=source_cfg,
    )
    selected_sensitive = [int(x) for x in audit["selected_sensitive_row_ids"]]
    if selected != selected_sensitive:
        raise RuntimeError("Directional SURE v2 unexpectedly selected non-sensitive rows")
    if not selected_sensitive:
        raise RuntimeError("Directional SURE v2 selected no sensitive vocabulary rows")
    return selected_sensitive, audit


def _eligible_wikipedia_indices(
    texts: Sequence[str], *, exclude_first: int, excluded_casefold: str, seed: int
) -> List[int]:
    eligible: List[int] = []
    needle = re.sub(r"\s+", " ", str(excluded_casefold)).strip().casefold()
    for index in range(int(exclude_first), len(texts)):
        text = texts[index]
        if not isinstance(text, str) or not text.strip():
            continue
        normalized = re.sub(r"\s+", " ", text).casefold()
        if needle and needle in normalized:
            continue
        eligible.append(index)
    random.Random(int(seed) + 271828).shuffle(eligible)
    return eligible


def _tokenize_external_contexts(
    tokenizer: Any,
    texts: Sequence[str],
    indices: Sequence[int],
    *,
    max_length: int,
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for index in indices:
        encoded = tokenizer(
            str(texts[int(index)]),
            truncation=True,
            max_length=int(max_length),
            add_special_tokens=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"][0].detach().cpu().contiguous()
        if ids.numel() < 2:
            continue
        contexts.append({"document_index": int(index), "input_ids": ids})
    return contexts


def build_external_slices(
    tokenizer: Any,
    texts: Sequence[str],
    cfg: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    opt = cfg["optimization"]
    boundary = cfg["data_boundary"]
    counts = [
        int(opt["protected_basis_context_count"]),
        int(opt["selection_utility_context_count"]),
        int(opt["fresh_gate_utility_context_count"]),
    ]
    needed = sum(counts)
    eligible = _eligible_wikipedia_indices(
        texts,
        exclude_first=int(boundary["wikipedia_exclude_first_documents"]),
        excluded_casefold=str(boundary["wikipedia_target_casefold_exclusion"]),
        seed=int(cfg["seed"]),
    )
    # Tokenize a surplus so short/empty tokenizations cannot silently shrink a slice.
    candidate_indices = eligible[: max(needed * 2, needed + 256)]
    tokenized = _tokenize_external_contexts(
        tokenizer,
        texts,
        candidate_indices,
        max_length=int(opt["utility_max_length"]),
    )
    if len(tokenized) < needed:
        raise RuntimeError(f"Only {len(tokenized)} usable external contexts; need {needed}")
    a, b, c = counts
    basis = tokenized[:a]
    selection = tokenized[a : a + b]
    fresh = tokenized[a + b : a + b + c]
    basis_ids = [int(x["document_index"]) for x in basis]
    selection_ids = [int(x["document_index"]) for x in selection]
    fresh_ids = [int(x["document_index"]) for x in fresh]
    if (set(basis_ids) & set(selection_ids)) or (set(basis_ids) & set(fresh_ids)) or (set(selection_ids) & set(fresh_ids)):
        raise RuntimeError("Directional SURE v2 external utility slices overlap")
    audit = {
        "protected_basis_context_count": len(basis),
        "selection_utility_context_count": len(selection),
        "fresh_gate_utility_context_count": len(fresh),
        "protected_basis_indices_sha256": sha_ints(basis_ids),
        "selection_indices_sha256": sha_ints(selection_ids),
        "fresh_gate_indices_sha256": sha_ints(fresh_ids),
        "slices_disjoint": True,
        "target_casefold_exclusion": boundary["wikipedia_target_casefold_exclusion"],
        "excluded_prefix_document_count": boundary["wikipedia_exclude_first_documents"],
        "official_rwku_records_accessed": False,
    }
    return basis, selection, fresh, audit


def _pad_contexts(tokenizer: Any, contexts: Sequence[Mapping[str, Any]], device: torch.device):
    if not contexts:
        raise ValueError("External context batch is empty")
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer lacks pad/eos token")
    lengths = torch.tensor([int(x["input_ids"].numel()) for x in contexts], dtype=torch.long)
    width = int(lengths.max().item())
    ids = torch.full((len(contexts), width), int(pad_id), dtype=torch.long)
    attention = torch.zeros_like(ids)
    for row, context in enumerate(contexts):
        value = context["input_ids"]
        ids[row, : value.numel()] = value
        attention[row, : value.numel()] = 1
    return ids.to(device), attention.to(device), lengths.to(device)


@torch.no_grad()
def external_final_hidden(
    model: torch.nn.Module,
    tokenizer: Any,
    contexts: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for start in range(0, len(contexts), int(batch_size)):
        batch = contexts[start : start + int(batch_size)]
        ids, attention, lengths = _pad_contexts(tokenizer, batch, device)
        hidden, _ = wikipedia._final_hidden_only(
            model, {"input_ids": ids, "attention_mask": attention}
        )
        rows = torch.arange(len(batch), device=device)
        values.append(hidden[rows, lengths - 1].detach().float())
    return torch.cat(values, dim=0)


def project_into_basis(rows: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if rows.ndim != 2 or basis.ndim != 2 or rows.shape[1] != basis.shape[1]:
        raise ValueError("Directional projection shape mismatch")
    if not basis.numel():
        return torch.zeros_like(rows)
    b = basis.to(device=rows.device, dtype=rows.dtype)
    return (rows @ b.transpose(0, 1)) @ b


@torch.no_grad()
def refresh_directional_bases(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[core.SensitivePredictionCase],
    protected_contexts: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    opt = cfg["optimization"]
    sensitive_hidden = core.forward_last_hidden(
        model,
        tokenizer,
        cases,
        device,
        batch_size=int(opt["cache_batch_size"]),
    )
    protected_hidden = external_final_hidden(
        model,
        tokenizer,
        protected_contexts,
        device=device,
        batch_size=int(opt["utility_batch_size"]),
    )
    bp = core.orthonormal_row_basis(
        protected_hidden, int(opt["protected_basis_rank"])
    )
    if not bp.numel():
        raise RuntimeError("Protected basis B_P is empty")
    sensitive_exclusive_rows = sparse_rows.project_rows_away_from_protected_basis(
        sensitive_hidden, bp
    )
    bs = core.orthonormal_row_basis(
        sensitive_exclusive_rows, int(opt["sensitive_exclusive_basis_rank"])
    )
    if not bs.numel():
        raise RuntimeError("Sensitive-exclusive basis B_S is empty")
    # Reproject and orthonormalize once to suppress numerical leakage into B_P.
    bs = sparse_rows.project_rows_away_from_protected_basis(bs, bp)
    bs = core.orthonormal_row_basis(bs, int(opt["sensitive_exclusive_basis_rank"]))
    overlap = float((bs @ bp.transpose(0, 1)).abs().max().detach().cpu())
    if overlap > 1e-4:
        raise RuntimeError(f"B_S/B_P overlap too large after exclusive projection: {overlap}")
    sensitive_energy = float(sensitive_hidden.square().sum().detach().cpu())
    exclusive_energy = float(sensitive_exclusive_rows.square().sum().detach().cpu())
    report = {
        "sensitive_hidden_count": int(sensitive_hidden.shape[0]),
        "protected_hidden_count": int(protected_hidden.shape[0]),
        "hidden_size": int(sensitive_hidden.shape[1]),
        "sensitive_exclusive_rank": int(bs.shape[0]),
        "protected_rank": int(bp.shape[0]),
        "max_abs_sensitive_protected_basis_overlap": overlap,
        "sensitive_energy_after_protected_projection_fraction": exclusive_energy / max(sensitive_energy, 1e-12),
        "official_rwku_records_accessed": False,
    }
    return bs.detach(), bp.detach(), report


@torch.no_grad()
def exact_external_kl_report(
    model: torch.nn.Module,
    tokenizer: Any,
    contexts: Sequence[Mapping[str, Any]],
    base_hidden: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, Any]:
    if len(contexts) != int(base_hidden.shape[0]):
        raise ValueError("External contexts/base hidden mismatch")
    output_layer = model.get_output_embeddings()
    base_weight = output_layer.weight
    values: List[torch.Tensor] = []
    hidden_mse: List[torch.Tensor] = []
    for start in range(0, len(contexts), int(batch_size)):
        batch = contexts[start : start + int(batch_size)]
        current_hidden = external_final_hidden(
            model, tokenizer, batch, device=device, batch_size=len(batch)
        )
        base_h = base_hidden[start : start + len(batch)].to(device=device, dtype=torch.float32)
        # Calling the output module applies the sparse selected-row hook.
        edited_logits = output_layer(current_hidden.to(dtype=base_weight.dtype)).float()
        # F.linear bypasses the hook and therefore uses immutable Base rows.
        base_logits = F.linear(base_h.to(dtype=base_weight.dtype), base_weight).float()
        base_logp = F.log_softmax(base_logits, dim=-1)
        edited_logp = F.log_softmax(edited_logits, dim=-1)
        kl = (base_logp.exp() * (base_logp - edited_logp)).sum(dim=-1).clamp_min(0.0)
        values.append(kl.detach().cpu())
        hidden_mse.append(
            ((current_hidden - base_h).square().mean(dim=-1) / base_h.square().mean(dim=-1).clamp_min(1e-6)).detach().cpu()
        )
    vector = torch.cat(values).float()
    hidden_vector = torch.cat(hidden_mse).float()
    return {
        "utility_kl_mean": float(vector.mean().item()),
        "utility_kl_p95": float(torch.quantile(vector, 0.95).item()),
        "utility_kl_max": float(vector.max().item()),
        "utility_prompt_count": int(vector.numel()),
        "utility_hidden_relative_mse_mean": float(hidden_vector.mean().item()),
        "utility_kl_kind": "exact_full_vocabulary_base_to_directional_sure",
        "official_rwku_records_accessed": False,
    }


def utility_safe(report: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    acc = cfg["acceptance"]
    return bool(
        float(report["utility_kl_mean"]) <= float(acc["utility_kl_mean_budget"])
        and float(report["utility_kl_p95"]) <= float(acc["utility_kl_p95_budget"])
        and float(report["utility_kl_max"]) <= float(acc["utility_kl_max_budget"])
    )


def atomic_safe(report: Mapping[str, Any], cfg: Mapping[str, Any]) -> bool:
    acc = cfg["acceptance"]
    return bool(
        float(report.get("FS", -1.0)) == float(acc["required_direct_success"])
        and float(report.get("generated_subject_FS", -1.0)) == float(acc["required_other_atomic_view_success"])
        and int(report.get("direct_margin_failures", -1)) == 0
        and int(report.get("generated_subject_margin_failures", -1)) == 0
    )


def snapshot_candidate(
    sparse: sparse_rows.SparseFP32RowDeltas,
    *,
    step: int,
    atomic: Mapping[str, Any],
    utility: Mapping[str, Any],
) -> Dict[str, Any]:
    input_delta, output_delta = sparse.snapshot()
    norms = sparse.delta_norms(input_delta, output_delta)
    key = (
        float(utility["utility_kl_mean"]),
        float(utility["utility_kl_p95"]),
        float(utility["utility_kl_max"]),
        float(norms["total_selected_row_delta_norm"]),
        int(step),
    )
    return {
        "step": int(step),
        "selection_key": list(key),
        "input_delta": input_delta,
        "output_delta": output_delta,
        "atomic": dict(atomic),
        "selection_utility": dict(utility),
        "delta_norms": norms,
    }


def _nonselected_equal_base(
    current: torch.Tensor,
    base_cpu: torch.Tensor,
    selected: Sequence[int],
    *,
    chunk_size: int = 2048,
) -> bool:
    selected_set = set(int(x) for x in selected)
    for start in range(0, int(current.shape[0]), int(chunk_size)):
        stop = min(start + int(chunk_size), int(current.shape[0]))
        keep = [i for i in range(start, stop) if i not in selected_set]
        if not keep:
            continue
        ids = torch.tensor(keep, dtype=torch.long, device=current.device)
        ref = base_cpu.index_select(0, ids.cpu()).to(device=current.device, dtype=current.dtype)
        if not torch.equal(current.index_select(0, ids), ref):
            return False
    return True


def main() -> None:
    args = parse_args()
    cfg = load_configuration(Path(args.configuration).resolve())
    if args.experiment_id != cfg["configuration_id"]:
        raise ValueError("experiment-id must equal locked Directional SURE v2 configuration ID")
    run_dir = verify_prepared_state(args, cfg)
    out = run_dir / LEARNER_DIR
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite Directional SURE v2 output: {out}")
    out.mkdir(parents=True)

    source_cfg = head.load_locked_configuration(SOURCE_BUNDLE_CONFIGURATION)
    views, bundle_audit, generator_audit = head.load_atomic_bundle(
        Path(args.training_bundle).resolve(),
        Path(args.generator_receipt).resolve(),
        source_cfg,
    )
    generator_model_audit = head.validate_generator_base_model(generator_audit, args.model_path)

    gagd.set_seed(int(cfg["seed"]))
    gagd.require_cuda_if_needed(str(cfg["acceptance"]["device_map"]))
    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=str(cfg["acceptance"]["checkpoint_dtype"]),
        device_map=str(cfg["acceptance"]["device_map"]),
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tokenizer)

    prompt_records = head.compile_prompt_records(
        views, tokenizer, neutral_target=str(cfg["neutral_target"])
    )
    cases = core.expand_sensitive_cases(
        prompt_records,
        tokenizer,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    if not cases:
        raise RuntimeError("Directional SURE v2 created no sensitive prediction cases")

    # Cache canonical Base same-prompt teacher logits before any editable deltas exist.
    base_logits = core.cache_base_logits(
        model,
        tokenizer,
        cases,
        device,
        batch_size=int(cfg["optimization"]["cache_batch_size"]),
    )
    tids_all = core.official_target_ids(
        tokenizer, cases, llama_like=llama_like, device=device
    )
    sensitive_rows, sensitive_row_audit = _content_sensitive_rows(
        tokenizer, cases, tids_all, source_cfg, len(prompt_records)
    )

    sample_ids = tokenizer(prompt_records[0]["prompt_text"], return_tensors="pt")["input_ids"].to(device)
    untie_audit = sparse_rows.untie_lm_head_preserve_logits(
        model, sample_input_ids=sample_ids
    )
    freeze_audit = sparse_rows.freeze_transformer_parameters(model)
    transformer_versions = _parameter_versions_except_vocab(model)
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if not torch.equal(input_layer.weight.detach(), output_layer.weight.detach()):
        raise RuntimeError("Untied Base input/output vocabulary weights are not initially identical")
    # One Base vocabulary snapshot is sufficient because Llama begins tied and untying is exact.
    base_vocab_cpu = input_layer.weight.detach().cpu().clone()

    sparse = sparse_rows.SparseFP32RowDeltas(
        model,
        selected_input_rows=sensitive_rows,
        selected_output_rows=sensitive_rows,
    )
    if sparse.input_delta.dtype != torch.float32 or sparse.output_delta.dtype != torch.float32:
        raise RuntimeError("Directional SURE v2 sparse row masters must be FP32")

    texts, wikipedia_meta = wikipedia.load_wikipedia_train(Path(args.wikipedia_dir).resolve())
    protected_contexts, selection_contexts, fresh_contexts, external_audit = build_external_slices(
        tokenizer, texts, cfg
    )
    utility_bs = int(cfg["optimization"]["utility_batch_size"])
    # Base hidden states are cached before training; fresh contexts are not evaluated again until selection is frozen.
    selection_base_hidden = external_final_hidden(
        model, tokenizer, selection_contexts, device=device, batch_size=utility_bs
    ).cpu()
    fresh_base_hidden = external_final_hidden(
        model, tokenizer, fresh_contexts, device=device, batch_size=utility_bs
    ).cpu()

    core.write_json(
        out / "protocol_report.json",
        {
            "schema_version": "rwku_directional_sure_v2_protocol_v1",
            "configuration_id": cfg["configuration_id"],
            "development_only": True,
            "posthoc_development_target": True,
            "official_rwku_records_accessed": False,
            "bundle_audit": bundle_audit,
            "generator_model_audit": generator_model_audit,
            "training_prompt_count": len(prompt_records),
            "sensitive_prediction_case_count": len(cases),
            "sensitive_row_audit": sensitive_row_audit,
            "selected_sensitive_row_count": len(sensitive_rows),
            "selected_sensitive_row_ids": sensitive_rows,
            "untie_audit": untie_audit,
            "freeze_audit": freeze_audit,
            "external_wikipedia_dataset": wikipedia_meta,
            "external_slices": external_audit,
            "objective": cfg["optimization"]["objective"],
            "ga_definition": "mean log p_theta(sensitive token | generated training prompt/prefix); minimized",
            "gd_definition": "KL(Base_non_sensitive || current_non_sensitive), current sensitive token removed and renormalized, same generated training decision",
            "basis_definition": {
                "B_P": "orthonormal span of current final hidden states on fixed external-Wikipedia protected contexts",
                "B_S": "orthonormal span of current sensitive-prediction hidden states after projection into orthogonal complement of B_P",
                "refresh_interval": int(cfg["optimization"]["basis_refresh_interval"]),
            },
            "parameter_locality": "FP32 sparse deltas over immutable Base vocabulary matrices; non-sensitive rows have no trainable parameter",
        },
    )

    opt_cfg = cfg["optimization"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [sparse.input_delta],
                "lr": float(opt_cfg["embedding_learning_rate"]),
                "weight_decay": 0.0,
            },
            {
                "params": [sparse.output_delta],
                "lr": float(opt_cfg["lm_head_learning_rate"]),
                "weight_decay": 0.0,
            },
        ]
    )
    sampler = core.IndexSampler(len(cases), int(opt_cfg["batch_size"]), int(cfg["seed"]))
    best: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[Any, ...]] = None
    basis_history: List[Dict[str, Any]] = []
    checkpoint_history: List[Dict[str, Any]] = []
    train_log_path = out / "train_log.jsonl"
    bs: Optional[torch.Tensor] = None
    bp: Optional[torch.Tensor] = None
    started = time.perf_counter()
    model.eval()

    with train_log_path.open("w", encoding="utf-8") as log_handle:
        for step in range(1, int(opt_cfg["steps"]) + 1):
            if step == 1 or (step - 1) % int(opt_cfg["basis_refresh_interval"]) == 0:
                bs, bp, basis_report = refresh_directional_bases(
                    model,
                    tokenizer,
                    cases,
                    protected_contexts,
                    cfg,
                    device=device,
                )
                basis_report = {
                    "refresh_before_step": int(step),
                    "updates_since_previous_refresh": 0 if step == 1 else int(opt_cfg["basis_refresh_interval"]),
                    **basis_report,
                }
                basis_history.append(basis_report)
                print(
                    "basis refresh before step {}: B_S rank={} B_P rank={} overlap={:.3e} exclusive_energy={:.4f}".format(
                        step,
                        basis_report["sensitive_exclusive_rank"],
                        basis_report["protected_rank"],
                        basis_report["max_abs_sensitive_protected_basis_overlap"],
                        basis_report["sensitive_energy_after_protected_projection_fraction"],
                    )
                )
            if bs is None or bp is None:
                raise RuntimeError("Directional bases are unavailable")

            idx = sampler.next()
            batch = [cases[i] for i in idx]
            logits = core.forward_last_logits(model, tokenizer, batch, device)
            tids = core.official_target_ids(
                tokenizer, batch, llama_like=llama_like, device=device
            )
            ga = core.ga_sensitive_logprob(logits, tids)
            gd = core.gd_non_sensitive_kl(logits, base_logits[idx], tids)
            params = (sparse.input_delta, sparse.output_delta)
            ga_grads = torch.autograd.grad(
                float(opt_cfg["ga_weight"]) * ga,
                params,
                retain_graph=True,
                allow_unused=True,
            )
            gd_grads = torch.autograd.grad(
                float(opt_cfg["gd_weight"]) * gd,
                params,
                retain_graph=False,
                allow_unused=True,
            )

            def grad_or_zero(value: Optional[torch.Tensor], parameter: torch.Tensor) -> torch.Tensor:
                return torch.zeros_like(parameter) if value is None else value.float()

            ga_emb = grad_or_zero(ga_grads[0], sparse.input_delta)
            ga_head = grad_or_zero(ga_grads[1], sparse.output_delta)
            gd_emb = grad_or_zero(gd_grads[0], sparse.input_delta)
            gd_head = grad_or_zero(gd_grads[1], sparse.output_delta)
            head_ga_directional = project_into_basis(ga_head, bs)
            head_gd_directional = project_into_basis(gd_head, bp)

            optimizer.zero_grad(set_to_none=True)
            sparse.input_delta.grad = (ga_emb + gd_emb).to(sparse.input_delta.dtype)
            sparse.output_delta.grad = (head_ga_directional + head_gd_directional).to(sparse.output_delta.dtype)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [sparse.input_delta, sparse.output_delta], float(opt_cfg["grad_clip"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"Non-finite Directional SURE gradient norm at step {step}")
            optimizer.step()

            if step == 1 or step % 25 == 0 or step == int(opt_cfg["steps"]):
                row = {
                    "step": int(step),
                    "total_loss_for_reporting": float((float(opt_cfg["ga_weight"]) * ga + float(opt_cfg["gd_weight"]) * gd).detach().cpu()),
                    "ga_sensitive_logprob": float(ga.detach().cpu()),
                    "gd_non_sensitive_kl": float(gd.detach().cpu()),
                    "gradient_norm_before_clip": float(grad_norm.detach().cpu()),
                    "embedding_ga_gradient_norm": float(ga_emb.norm().detach().cpu()),
                    "embedding_gd_gradient_norm": float(gd_emb.norm().detach().cpu()),
                    "lm_head_ga_gradient_norm_before_projection": float(ga_head.norm().detach().cpu()),
                    "lm_head_ga_BS_gradient_norm_after_projection": float(head_ga_directional.norm().detach().cpu()),
                    "lm_head_gd_gradient_norm_before_projection": float(gd_head.norm().detach().cpu()),
                    "lm_head_gd_BP_gradient_norm_after_projection": float(head_gd_directional.norm().detach().cpu()),
                    **sparse.delta_norms(),
                    "official_rwku_records_accessed": False,
                }
                log_handle.write(json.dumps(row) + "\n")
                log_handle.flush()
                print(
                    "directional step {:3d}: GA={:.6f} GD={:.6f} embΔ={:.4f} headΔ={:.4f} GA->BS={:.4f} GD->BP={:.4f}".format(
                        step,
                        row["ga_sensitive_logprob"],
                        row["gd_non_sensitive_kl"],
                        row["selected_input_row_delta_norm"],
                        row["selected_output_row_delta_norm"],
                        row["lm_head_ga_BS_gradient_norm_after_projection"],
                        row["lm_head_gd_BP_gradient_norm_after_projection"],
                    )
                )

            if step % int(opt_cfg["checkpoint_interval"]) != 0 and step != int(opt_cfg["steps"]):
                continue
            atomic = head.materialized_atomic_report(
                model,
                tokenizer,
                prompt_records,
                device,
                llama_like=llama_like,
                required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
            )
            selection_utility = exact_external_kl_report(
                model,
                tokenizer,
                selection_contexts,
                selection_base_hidden,
                device=device,
                batch_size=utility_bs,
            )
            a_safe = atomic_safe(atomic, cfg)
            u_safe = utility_safe(selection_utility, cfg)
            checkpoint = {
                "step": int(step),
                "atomic": atomic,
                "selection_utility": selection_utility,
                "atomic_safe": a_safe,
                "selection_utility_safe": u_safe,
                "eligible": bool(a_safe and u_safe),
                "delta_norms": sparse.delta_norms(),
                "official_rwku_records_accessed": False,
            }
            checkpoint_history.append(checkpoint)
            print(
                "  checkpoint {}: direct={} other={} KL={:.6f}/{:.6f}/{:.6f} eligible={}".format(
                    step,
                    atomic.get("FS"),
                    atomic.get("generated_subject_FS"),
                    selection_utility["utility_kl_mean"],
                    selection_utility["utility_kl_p95"],
                    selection_utility["utility_kl_max"],
                    checkpoint["eligible"],
                )
            )
            if checkpoint["eligible"]:
                candidate = snapshot_candidate(
                    sparse,
                    step=step,
                    atomic=atomic,
                    utility=selection_utility,
                )
                key = tuple(candidate["selection_key"])
                if best_key is None or key < best_key:
                    best_key = key
                    best = candidate

    training_seconds = time.perf_counter() - started
    core.write_json(out / "basis_refresh_history.json", basis_history)
    core.write_json(out / "checkpoint_history.json", checkpoint_history)
    assert_transformer_versions(model, transformer_versions)

    # Hooks apply sparse deltas functionally; the Base vocabulary parameters themselves must still be exact.
    if not torch.equal(input_layer.weight.detach().cpu(), base_vocab_cpu):
        raise RuntimeError("Base input vocabulary matrix changed before materialization")
    if not torch.equal(output_layer.weight.detach().cpu(), base_vocab_cpu):
        raise RuntimeError("Base output vocabulary matrix changed before materialization")

    if best is None:
        core.write_json(
            out / "result.json",
            {
                "schema_version": "rwku_directional_sure_v2_result_v1",
                "configuration_id": cfg["configuration_id"],
                "development_only": True,
                "posthoc_development_target": True,
                "official_rwku_records_accessed": False,
                "feasible": False,
                "reason": "no checkpoint passed generated atomic behavior plus external-Wikipedia selection KL gates",
                "selected_checkpoint_step": None,
                "training_seconds": training_seconds,
                "transformer_exactly_frozen": True,
                "non_sensitive_embedding_rows_exact_base": True,
                "non_sensitive_lm_head_rows_exact_base": True,
            },
        )
        raise RuntimeError("Directional SURE v2 found no eligible development checkpoint")

    with torch.no_grad():
        sparse.input_delta.copy_(best["input_delta"].to(device=sparse.input_delta.device))
        sparse.output_delta.copy_(best["output_delta"].to(device=sparse.output_delta.device))

    # The fresh external gate is opened only after selection is fixed.
    fresh_utility = exact_external_kl_report(
        model,
        tokenizer,
        fresh_contexts,
        fresh_base_hidden,
        device=device,
        batch_size=utility_bs,
    )
    fresh_safe = utility_safe(fresh_utility, cfg)
    final_atomic = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(cfg["acceptance"]["required_pairwise_margin"]),
    )
    final_atomic_safe = atomic_safe(final_atomic, cfg)
    feasible = bool(final_atomic_safe and fresh_safe)

    # Materialize only the selected rows. Every non-selected row stays exactly Base.
    sparse.materialize(best["input_delta"], best["output_delta"], 1.0)
    nonselected_input_equal = _nonselected_equal_base(
        input_layer.weight.detach(), base_vocab_cpu, sensitive_rows
    )
    nonselected_output_equal = _nonselected_equal_base(
        output_layer.weight.detach(), base_vocab_cpu, sensitive_rows
    )
    if not nonselected_input_equal or not nonselected_output_equal:
        raise RuntimeError("Directional SURE v2 changed a non-sensitive vocabulary row")
    assert_transformer_versions(model, transformer_versions)

    result = {
        "schema_version": "rwku_directional_sure_v2_result_v1",
        "configuration_id": cfg["configuration_id"],
        "method": cfg["method"],
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_records_accessed": False,
        "selected_checkpoint_step": int(best["step"]),
        "selection_key": list(best["selection_key"]),
        "selected_sensitive_row_count": len(sensitive_rows),
        "selected_sensitive_row_ids": sensitive_rows,
        "delta_norms": sparse.delta_norms(best["input_delta"], best["output_delta"]),
        "final_atomic": final_atomic,
        "final_atomic_safe": final_atomic_safe,
        "fresh_external_wikipedia_utility": fresh_utility,
        "fresh_external_wikipedia_utility_safe": fresh_safe,
        "feasible": feasible,
        "transformer_exactly_frozen": True,
        "non_sensitive_embedding_rows_exact_base": nonselected_input_equal,
        "non_sensitive_lm_head_rows_exact_base": nonselected_output_equal,
        "fresh_gate_opened_only_after_checkpoint_selection": True,
        "official_rwku_paraphrase_seen": False,
        "official_rwku_neighborhood_seen": False,
        "official_rwku_retain_seen": False,
        "official_rwku_ppl_text_seen": False,
        "training_seconds": training_seconds,
        "interpretation_note": "Post-hoc Stephen King development only. Directional SURE v2 edits sparse vocabulary-interface rows while the transformer remains exactly frozen. Official RWKU evaluation is not opened by this run.",
    }
    core.write_json(out / "result.json", result)

    if args.save_checkpoint and feasible:
        checkpoint = out / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        print(f"Saved feasible Directional SURE v2 checkpoint: {checkpoint}")

    print("\nRWKU DIRECTIONAL SURE v2 RESULT")
    print(f"selected checkpoint step: {result['selected_checkpoint_step']}")
    print("atomic direct/other: {} / {}".format(final_atomic.get("FS"), final_atomic.get("generated_subject_FS")))
    print(
        "fresh Wiki KL mean/p95/max: {:.6f} / {:.6f} / {:.6f}".format(
            fresh_utility["utility_kl_mean"],
            fresh_utility["utility_kl_p95"],
            fresh_utility["utility_kl_max"],
        )
    )
    print("selected input/head delta norm: {:.6f} / {:.6f}".format(
        result["delta_norms"]["selected_input_row_delta_norm"],
        result["delta_norms"]["selected_output_row_delta_norm"],
    ))
    print(f"transformer frozen: {result['transformer_exactly_frozen']}")
    print(f"non-sensitive input rows exact Base: {result['non_sensitive_embedding_rows_exact_base']}")
    print(f"non-sensitive head rows exact Base: {result['non_sensitive_lm_head_rows_exact_base']}")
    print(f"feasible under Directional SURE v2 gates: {feasible}")
    print(f"result: {out / 'result.json'}")

    del optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
