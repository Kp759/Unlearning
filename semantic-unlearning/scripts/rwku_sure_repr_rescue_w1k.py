#!/usr/bin/env python3
"""Minimal representation rescue for the locked RWKU-H-W1K head-only pilot.

This is a development-only continuation of the Stephen King atomic seed-0
experiment. It does NOT alter or overwrite the failed head-only run. It loads
that run's frozen Stage-1 sparse LM-head delta, keeps the LM head fixed, and
trains a tiny LoRA residual in only the last Llama MLP ``down_proj``.

Official RWKU rows remain inaccessible until this script freezes a checkpoint
receipt. Candidate selection requires every generated atomic view to pass after
physical BF16 materialization, the original sparse-head norm to remain <= 1.5,
a bounded relative transformer-weight delta, and a full-vocabulary Wikipedia
KL gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_artifact_access as artifact_access
import rwku_checkpoint_receipt as checkpoint_receipt
import rwku_experiment
import rwku_representation as representation
import rwku_sure_head_only_w1k as head
import sure_canonical_core as core
import sure_minimal_two_stage as learner

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_repr_rescue_w1k_seed0.json"
SOURCE_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "sure_head_only_w1k_seed0.json"
PROTOCOL_STATUS = "rwku_target_only_auxwiki_sure_head_repr_rescue_w1k_development"
SCHEMA = "rwku_sure_head_repr_rescue_w1k_configuration_v1"
EXPERIMENT_ID = "rwku-h-w1k-stephen-king-repr-rescue-seed0-v2"


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    core.write_json(path, value)


def load_configuration(path: Path) -> Dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported representation-rescue configuration schema")
    required = {
        "configuration_id": EXPERIMENT_ID,
        "development_only": True,
        "seed": 0,
        "target_entity": "Stephen King",
        "target_entity_id": "rwku:1_Stephen_King",
        "source_head_only_configuration_id": head.EXPECTED_CONFIGURATION_ID,
        "neutral_target": "Unknown",
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise ValueError(f"Representation-rescue configuration changed {key}")
    rep = value.get("representation", {})
    acceptance = value.get("acceptance", {})
    if tuple(rep.get("rank_ladder", ())) != (1, 2, 4):
        raise ValueError("Representation rank ladder must remain 1,2,4")
    if tuple(rep.get("target_modules", ())) != ("down_proj",):
        raise ValueError("Representation rescue must remain last-layer down_proj only")
    if int(rep.get("last_n_layers", 0)) != 1:
        raise ValueError("Representation rescue must remain last-layer only")
    if int(rep.get("steps", 0)) <= 0 or float(rep.get("learning_rate", 0.0)) <= 0:
        raise ValueError("Representation optimization settings are invalid")
    if int(rep.get("utility_train_prompt_count", 0)) <= 0:
        raise ValueError("utility_train_prompt_count must be positive")
    if int(rep.get("utility_gate_prompt_count", 0)) <= 0:
        raise ValueError("utility_gate_prompt_count must be positive")
    if not rep.get("candidate_scales"):
        raise ValueError("Representation candidate scales are empty")
    if any(not 0.0 < float(x) <= 1.0 for x in rep["candidate_scales"]):
        raise ValueError("Representation candidate scales must be in (0,1]")
    required_acceptance = {
        "required_pairwise_margin": 0.01,
        "required_direct_success": 100.0,
        "required_other_atomic_view_success": 100.0,
        "max_head_delta_norm": 1.5,
        "utility_kl_mean_budget": 0.01,
        "utility_kl_p95_budget": 0.05,
        "utility_kl_max_budget": 0.5,
        "checkpoint_dtype": "bf16",
        "device_map": "single",
    }
    for key, expected in required_acceptance.items():
        if acceptance.get(key) != expected:
            raise ValueError(f"Representation-rescue acceptance changed {key}")
    boundary = value.get("data_boundary", {})
    for key in (
        "official_rwku_records_available_to_learner",
        "official_rwku_records_used_for_checkpoint_selection",
        "neighbor_prompts_used_for_training_or_selection",
    ):
        if boundary.get(key) is not False:
            raise ValueError(f"Data boundary changed {key}")
    return value


def _state_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=Path(args.output_root),
        experiment_id=str(args.experiment_id),
        training_source=rwku_experiment.TRAINING_SOURCE_TARGET_ONLY,
    )


def verify_prepared_state(args: argparse.Namespace, configuration: Mapping[str, Any]) -> Tuple[SimpleNamespace, Path]:
    state_args = _state_namespace(args)
    state = rwku_experiment._read_state(state_args)
    if state.get("state") != "PREPARED":
        raise ValueError(f"Representation rescue requires PREPARED state, got {state.get('state')}")
    if state.get("training_source") != rwku_experiment.TRAINING_SOURCE_TARGET_ONLY:
        raise ValueError("Prepared state belongs to a different RWKU training source")
    if state.get("official_evaluation_opened") is not False:
        raise ValueError("Official RWKU evaluation was already opened")
    target = state.get("target", {})
    if target.get("seed") != 0 or target.get("subject") != configuration["target_entity"]:
        raise ValueError("Prepared state is not the locked Stephen King seed-0 target")
    run_dir = Path(args.output_root).resolve() / str(args.experiment_id)
    if (run_dir / "checkpoint_receipt.json").exists():
        raise FileExistsError("Representation-rescue checkpoint receipt already exists")
    return state_args, run_dir


def verify_source_run(source_run: Path, configuration: Mapping[str, Any]) -> Dict[str, Path]:
    source_run = Path(source_run).resolve()
    learner_dir = source_run / "sure_head_only_w1k"
    required = {
        "stage1_delta": learner_dir / "stage1_delta.pt",
        "stage1_report": learner_dir / "stage1_materialized_report.json",
        "selected_rows": learner_dir / "selected_row_report.json",
        "infeasible": learner_dir / "infeasible.json",
        "source_state": source_run / "experiment_state.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing source head-only {label}: {path}")
    if (source_run / "checkpoint_receipt.json").exists():
        raise ValueError("Source head-only run unexpectedly has a frozen checkpoint receipt")
    state = read_json(required["source_state"])
    if state.get("official_evaluation_opened") is not False:
        raise ValueError("Source head-only run already opened official RWKU evaluation")
    report = read_json(required["stage1_report"])
    if report.get("FS") != 100.0:
        raise ValueError("Source Stage 1 must retain 100% direct atomic success")
    failures = report.get("pairwise_margin_failure_positions", [])
    if not isinstance(failures, list) or not failures:
        raise ValueError("Source Stage 1 has no residual atomic-view failure to rescue")
    infeasible = read_json(required["infeasible"])
    if infeasible.get("official_rwku_records_accessed") is not False:
        raise ValueError("Source infeasible report indicates official RWKU access")
    if infeasible.get("configuration_id") != configuration["source_head_only_configuration_id"]:
        raise ValueError("Source infeasible report has the wrong configuration identity")
    return required


@torch.no_grad()
def selected_base_probabilities_full_head(
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    utility_hidden: torch.Tensor,
    base_logsumexp: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    ids = torch.tensor([int(x) for x in selected_ids], device=output_layer.weight.device, dtype=torch.long)
    chunks: List[torch.Tensor] = []
    for start in range(0, int(utility_hidden.shape[0]), int(batch_size)):
        hidden = utility_hidden[start : start + int(batch_size)].to(
            device=output_layer.weight.device, dtype=output_layer.weight.dtype
        )
        full_logits = output_layer(hidden)
        selected_logits = full_logits.index_select(-1, ids)
        log_z = base_logsumexp[start : start + len(hidden)].to(device=selected_logits.device, dtype=torch.float32)
        chunks.append(torch.exp(selected_logits.float() - log_z.unsqueeze(1)).detach().cpu())
    probabilities = torch.cat(chunks, dim=0).float().contiguous()
    maximum_mass = float(probabilities.sum(dim=1).max().item())
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise FloatingPointError("Selected Base utility probabilities are invalid")
    if maximum_mass > 1.001:
        raise FloatingPointError(f"Full-head selected Base probability mass exceeds one: {maximum_mass}")
    return probabilities


def behavior_safe(report: Mapping[str, Any]) -> bool:
    return bool(
        report.get("FS") == 100.0
        and report.get("generated_subject_FS") == 100.0
        and int(report.get("direct_margin_failures", -1)) == 0
        and int(report.get("generated_subject_margin_failures", -1)) == 0
    )


def _completion_token_ids(tokenizer: Any, text: str, llama_like: bool) -> List[int]:
    ids = [int(x) for x in tokenizer(f" {text}")["input_ids"]]
    if llama_like:
        ids = ids[1:]
    if not ids:
        raise ValueError(f"Completion tokenized to no tokens: {text!r}")
    return ids


def differentiable_average_nll(
    model: torch.nn.Module,
    tokenizer: Any,
    prefix: str,
    suffix: str,
    *,
    llama_like: bool,
    device: torch.device,
) -> torch.Tensor:
    prefix_ids = [int(x) for x in tokenizer(prefix)["input_ids"]]
    suffix_ids = _completion_token_ids(tokenizer, suffix, llama_like)
    combined = tokenizer(f"{prefix} {suffix}", return_tensors="pt")["input_ids"].to(device)
    start = len(prefix_ids) - 1
    positions = torch.arange(start, start + len(suffix_ids), device=device, dtype=torch.long)
    if int(positions[-1]) >= int(combined.shape[1]):
        raise RuntimeError("Completion positions exceed tokenized prompt length")
    hidden, _ = wikipedia._final_hidden_only(
        model, {"input_ids": combined, "attention_mask": torch.ones_like(combined)}
    )
    rows = hidden[0].index_select(0, positions)
    output = model.get_output_embeddings()
    if output is None:
        raise RuntimeError("Model has no output embeddings")
    logits = output(rows.to(dtype=output.weight.dtype)).float()
    targets = torch.tensor(suffix_ids, device=logits.device, dtype=torch.long)
    return -F.log_softmax(logits, dim=-1).gather(1, targets[:, None]).mean()


def pairwise_separation_loss(
    model: torch.nn.Module,
    tokenizer: Any,
    record: Mapping[str, Any],
    *,
    margin: float,
    llama_like: bool,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rewrite = record["requested_rewrite"]
    sensitive = str(rewrite["target_sensitive"]["str"])
    neutral = str(rewrite["target_reference"]["str"])
    prompt = str(record["prompt_text"])
    sensitive_nll = differentiable_average_nll(model, tokenizer, prompt, sensitive, llama_like=llama_like, device=device)
    neutral_nll = differentiable_average_nll(model, tokenizer, prompt, neutral, llama_like=llama_like, device=device)
    separation = sensitive_nll - neutral_nll
    return F.relu(float(margin) - separation).square(), separation


def build_utility_prefix(tokenizer: Any, text: str, record: Mapping[str, Any], *, max_length: int) -> torch.Tensor:
    encoded = tokenizer(str(text), truncation=True, max_length=int(max_length), return_tensors="pt")["input_ids"][0]
    position = int(record["predictor_token_position"])
    attended = int(record["attended_token_count"])
    if attended != int(encoded.numel()):
        raise ValueError(
            "Wikipedia replay token count differs from the frozen utility cache: "
            f"cache={attended} replay={encoded.numel()}"
        )
    if not 0 <= position < attended - 1:
        raise ValueError("Wikipedia predictor position is invalid")
    return encoded[: position + 1].contiguous()


def build_utility_contexts(
    tokenizer: Any,
    texts: Sequence[str],
    utility_metadata: Mapping[str, Any],
    utility_hidden: torch.Tensor,
    indices: Sequence[int],
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    records = utility_metadata.get("utility_prompt_records")
    if not isinstance(records, list) or len(records) != int(utility_hidden.shape[0]):
        raise ValueError("Utility cache lacks aligned utility_prompt_records")
    max_length = int(utility_metadata.get("max_length", 4096))
    contexts: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
    for value in indices:
        index = int(value)
        record = records[index]
        document_index = int(record["document_index"])
        if not 0 <= document_index < len(texts):
            raise ValueError("Utility document index is outside the Wikipedia corpus")
        prefix = build_utility_prefix(tokenizer, str(texts[document_index]), record, max_length=max_length)
        target = utility_hidden[index].float().cpu().contiguous()
        contexts.append((index, prefix, target))
    return contexts


def utility_hidden_loss(
    model: torch.nn.Module,
    context: Tuple[int, torch.Tensor, torch.Tensor],
    *,
    device: torch.device,
) -> torch.Tensor:
    _, ids, target = context
    input_ids = ids.unsqueeze(0).to(device)
    hidden, _ = wikipedia._final_hidden_only(
        model, {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    )
    current = hidden[0, -1].float()
    base = target.to(device=current.device, dtype=torch.float32)
    return F.mse_loss(current, base) / base.square().mean().clamp_min(1e-6)


def adapter_l2(handles: Sequence[representation.AdapterHandle]) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for handle in handles:
        values.append(handle.wrapper.lora_A.square().mean())
        values.append(handle.wrapper.lora_B.square().mean())
    return torch.stack(values).mean()


@torch.no_grad()
def capture_adapter_base_weights(handles: Sequence[representation.AdapterHandle]) -> List[torch.Tensor]:
    return [handle.wrapper.base.weight.detach().clone() for handle in handles]


@torch.no_grad()
def restore_adapter_base_weights(
    handles: Sequence[representation.AdapterHandle], originals: Sequence[torch.Tensor]
) -> None:
    if len(handles) != len(originals):
        raise ValueError("Adapter original-weight count mismatch")
    for handle, original in zip(handles, originals):
        handle.wrapper.base.weight.copy_(
            original.to(device=handle.wrapper.base.weight.device, dtype=handle.wrapper.base.weight.dtype)
        )
    representation.set_adapter_scale(handles, 0.0)


@torch.no_grad()
def materialize_adapter_candidate(
    handles: Sequence[representation.AdapterHandle], originals: Sequence[torch.Tensor], scale: float
) -> Dict[str, Any]:
    restore_adapter_base_weights(handles, originals)
    return representation._materialize_adapter_scale(handles, float(scale))


@torch.no_grad()
def representation_delta_report(
    handles: Sequence[representation.AdapterHandle], originals: Sequence[torch.Tensor]
) -> Dict[str, Any]:
    delta_sq = 0.0
    base_sq = 0.0
    per_module: List[Dict[str, Any]] = []
    for handle, original in zip(handles, originals):
        current = handle.wrapper.base.weight.detach().float()
        base = original.to(device=current.device, dtype=torch.float32)
        delta = current - base
        dnorm = float(delta.norm().item())
        bnorm = float(base.norm().item())
        delta_sq += dnorm * dnorm
        base_sq += bnorm * bnorm
        per_module.append(
            {
                "path": handle.path,
                "layer_index": int(handle.layer_index),
                "module_name": handle.module_name,
                "delta_frobenius": dnorm,
                "base_frobenius": bnorm,
                "relative_frobenius": dnorm / max(bnorm, 1e-12),
            }
        )
    total_delta = math.sqrt(delta_sq)
    total_base = math.sqrt(base_sq)
    return {
        "delta_frobenius": total_delta,
        "base_frobenius": total_base,
        "relative_frobenius": total_delta / max(total_base, 1e-12),
        "modules": per_module,
    }


def _pad_context_batch(
    tokenizer: Any,
    contexts: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise ValueError("Tokenizer has no pad or eos token")
    lengths = torch.tensor([int(x[1].numel()) for x in contexts], dtype=torch.long)
    width = int(lengths.max().item())
    input_ids = torch.full((len(contexts), width), int(pad_id), dtype=torch.long)
    attention = torch.zeros((len(contexts), width), dtype=torch.long)
    for row, (_, ids, _) in enumerate(contexts):
        length = int(ids.numel())
        input_ids[row, :length] = ids
        attention[row, :length] = 1
    return input_ids.to(device), attention.to(device), lengths.to(device)


@torch.no_grad()
def context_logits(
    model: torch.nn.Module,
    tokenizer: Any,
    contexts: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> torch.Tensor:
    input_ids, attention, lengths = _pad_context_batch(tokenizer, contexts, device=device)
    hidden, _ = wikipedia._final_hidden_only(model, {"input_ids": input_ids, "attention_mask": attention})
    rows = hidden[
        torch.arange(hidden.shape[0], device=hidden.device),
        lengths.to(hidden.device) - 1,
    ]
    output = model.get_output_embeddings()
    if output is None:
        raise RuntimeError("Model has no output embeddings")
    return output(rows.to(dtype=output.weight.dtype)).float()


@torch.no_grad()
def exact_full_vocab_utility_kl(
    model: torch.nn.Module,
    tokenizer: Any,
    contexts: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    *,
    output_layer: torch.nn.Module,
    selected_tensor: torch.Tensor,
    base_head_rows: torch.Tensor,
    edited_head_rows: torch.Tensor,
    handles: Sequence[representation.AdapterHandle],
    original_adapter_weights: Sequence[torch.Tensor],
    scale: float,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    if not contexts:
        raise ValueError("Exact utility KL requires at least one context")
    values: List[torch.Tensor] = []
    for start in range(0, len(contexts), int(batch_size)):
        batch = contexts[start : start + int(batch_size)]
        restore_adapter_base_weights(handles, original_adapter_weights)
        output_layer.weight.index_copy_(
            0,
            selected_tensor,
            base_head_rows.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype),
        )
        base_logits = context_logits(model, tokenizer, batch, device=device)
        materialize_adapter_candidate(handles, original_adapter_weights, float(scale))
        output_layer.weight.index_copy_(
            0,
            selected_tensor,
            edited_head_rows.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype),
        )
        edited_logits = context_logits(model, tokenizer, batch, device=device)
        base_logp = F.log_softmax(base_logits.float(), dim=-1)
        edited_logp = F.log_softmax(edited_logits.float(), dim=-1)
        kl = (base_logp.exp() * (base_logp - edited_logp)).sum(dim=-1)
        if not torch.isfinite(kl).all() or bool((kl < -1e-5).any()):
            raise FloatingPointError("Non-finite/negative full-vocabulary utility KL")
        values.append(kl.clamp_min(0.0).cpu())
    materialize_adapter_candidate(handles, original_adapter_weights, float(scale))
    output_layer.weight.index_copy_(
        0,
        selected_tensor,
        edited_head_rows.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype),
    )
    vector = torch.cat(values).float()
    return {
        "utility_kl_mean": float(vector.mean().item()),
        "utility_kl_median": float(vector.median().item()),
        "utility_kl_p95": float(torch.quantile(vector, 0.95).item()),
        "utility_kl_p99": float(torch.quantile(vector, 0.99).item()),
        "utility_kl_max": float(vector.max().item()),
        "utility_prompt_count": int(vector.numel()),
        "utility_kl_kind": "exact_full_vocabulary_base_to_edited",
    }


def train_rank(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_records: Sequence[Mapping[str, Any]],
    active_positions: Sequence[int],
    utility_train_contexts: Sequence[Tuple[int, torch.Tensor, torch.Tensor]],
    *,
    rank: int,
    configuration: Mapping[str, Any],
    llama_like: bool,
    device: torch.device,
    log_path: Path,
) -> Tuple[List[representation.AdapterHandle], List[torch.Tensor], Dict[str, Any]]:
    rep_cfg = configuration["representation"]
    adapter_cfg = representation.RepresentationConfig(
        steps=int(rep_cfg["steps"]),
        learning_rate=float(rep_cfg["learning_rate"]),
        weight_decay=float(rep_cfg["weight_decay"]),
        rank=int(rank),
        alpha=float(rank),
        dropout=0.0,
        layer_indices=(),
        last_n_layers=int(rep_cfg["last_n_layers"]),
        target_modules=tuple(rep_cfg["target_modules"]),
        seed=int(configuration["seed"]),
    )
    handles = representation.inject_lora_adapters(model, adapter_cfg)
    originals = capture_adapter_base_weights(handles)
    parameters = representation.adapter_parameters(handles)
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(rep_cfg["learning_rate"]),
        weight_decay=float(rep_cfg["weight_decay"]),
    )
    active = [int(x) for x in active_positions]
    preserve = [i for i in range(len(prompt_records)) if i not in set(active)]
    if not active or not preserve:
        raise ValueError("Representation rescue needs active and preservation views")
    if not utility_train_contexts:
        raise ValueError("Representation rescue needs external-Wikipedia train contexts")
    started = time.perf_counter()
    history: List[Dict[str, Any]] = []
    representation.set_adapter_scale(handles, 1.0)
    model.eval()
    for step in range(int(rep_cfg["steps"])):
        optimizer.zero_grad(set_to_none=True)
        active_record = prompt_records[active[step % len(active)]]
        active_loss, active_sep = pairwise_separation_loss(
            model,
            tokenizer,
            active_record,
            margin=float(rep_cfg["active_pairwise_target"]),
            llama_like=llama_like,
            device=device,
        )
        preserve_record = prompt_records[preserve[step % len(preserve)]]
        preserve_loss, preserve_sep = pairwise_separation_loss(
            model,
            tokenizer,
            preserve_record,
            margin=float(rep_cfg["preserve_pairwise_margin"]),
            llama_like=llama_like,
            device=device,
        )
        hidden_loss = utility_hidden_loss(
            model,
            utility_train_contexts[step % len(utility_train_contexts)],
            device=device,
        )
        l2 = adapter_l2(handles)
        loss = (
            float(rep_cfg["active_weight"]) * active_loss
            + float(rep_cfg["preserve_weight"]) * preserve_loss
            + float(rep_cfg["utility_hidden_weight"]) * hidden_loss
            + float(rep_cfg["adapter_l2_weight"]) * l2
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Representation-rescue loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(rep_cfg["grad_clip"]))
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == int(rep_cfg["steps"]):
            row = {
                "step": step + 1,
                "rank": int(rank),
                "loss": float(loss.detach().cpu()),
                "active_pairwise_hinge": float(active_loss.detach().cpu()),
                "active_separation": float(active_sep.detach().cpu()),
                "preserve_pairwise_hinge": float(preserve_loss.detach().cpu()),
                "preserve_separation": float(preserve_sep.detach().cpu()),
                "utility_hidden_relative_mse": float(hidden_loss.detach().cpu()),
                "adapter_l2": float(l2.detach().cpu()),
            }
            history.append(row)
            print(
                "repr-rank{} step {:3d}: loss={:.6f} active_sep={:.4f} preserve_sep={:.4f} wiki_hidden={:.6f}".format(
                    rank,
                    step + 1,
                    row["loss"],
                    row["active_separation"],
                    row["preserve_separation"],
                    row["utility_hidden_relative_mse"],
                )
            )
    model.eval()
    write_json(
        log_path,
        {
            "rank": int(rank),
            "steps": int(rep_cfg["steps"]),
            "training_seconds": time.perf_counter() - started,
            "history": history,
        },
    )
    return handles, originals, {"history": history}


def main() -> None:
    args = parse_args()
    configuration_path = Path(args.configuration).resolve()
    configuration = load_configuration(configuration_path)
    if args.experiment_id != configuration["configuration_id"]:
        raise ValueError("Experiment ID must equal the locked v2 configuration ID")
    state_args, run_dir = verify_prepared_state(args, configuration)
    source = verify_source_run(Path(args.source_head_only_run), configuration)
    output_dir = run_dir / "sure_head_repr_rescue_w1k"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite representation rescue: {output_dir}")
    output_dir.mkdir(parents=True)

    source_configuration = head.load_locked_configuration(SOURCE_CONFIGURATION)
    views, bundle_audit, generator_audit = head.load_atomic_bundle(
        Path(args.training_bundle).resolve(),
        Path(args.generator_receipt).resolve(),
        source_configuration,
    )
    generator_model_audit = head.validate_generator_base_model(generator_audit, args.model_path)
    rwku_experiment._write_state(
        state_args,
        "TRAINING",
        configuration_path=str(configuration_path),
        configuration_sha256=artifact_access.sha256_file(configuration_path),
        source_head_only_run=str(Path(args.source_head_only_run).resolve()),
        official_evaluation_opened=False,
    )

    gagd.set_seed(int(configuration["seed"]))
    gagd.require_cuda_if_needed(configuration["acceptance"]["device_map"])
    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=configuration["acceptance"]["checkpoint_dtype"],
        device_map=configuration["acceptance"]["device_map"],
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tokenizer)
    prompt_records = head.compile_prompt_records(
        views, tokenizer, neutral_target=str(configuration["neutral_target"])
    )

    identity = wikipedia.model_identity(model, tokenizer, args.model_path)
    output_layer = core.untie_and_freeze_output_head(model)
    selected_payload = torch.load(source["stage1_delta"], map_location="cpu")
    selected_ids = [int(x) for x in selected_payload["row_ids"]]
    stage1_delta = selected_payload["delta"].float().to(device)
    selected_tensor = torch.tensor(selected_ids, device=output_layer.weight.device, dtype=torch.long)
    base_head_rows = output_layer.weight.index_select(0, selected_tensor).detach().clone()

    optimization = head.optimization_namespace(source_configuration, prompt_count=len(prompt_records))
    utility_second_moment, utility_hidden, utility_logsumexp, utility_metadata = learner.load_utility_cache(
        Path(args.utility_cache).resolve(),
        expected_sample_size=optimization.utility_sample_size,
        expected_prompt_count=optimization.utility_prompt_count,
        expected_hidden_size=int(output_layer.weight.shape[1]),
        expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"],
    )
    del utility_second_moment
    utility_audit = head.validate_w1k_utility_metadata(utility_metadata, source_configuration)
    selected_probabilities = selected_base_probabilities_full_head(
        output_layer,
        selected_ids,
        utility_hidden,
        utility_logsumexp,
        batch_size=optimization.utility_eval_batch_size,
    )
    train_indices, guard_indices, utility_pool_report = learner.build_disjoint_token_conditioned_utility_pools(
        selected_base_probabilities=selected_probabilities,
        selected_ids=selected_ids,
        topk_per_row=optimization.utility_token_topk_per_row,
        uniform_prompt_count=optimization.utility_uniform_prompt_count,
        split_seed=optimization.utility_pool_seed,
    )

    core.materialize_output_delta(output_layer, selected_ids, stage1_delta)
    actual_stage1_delta = learner.actual_selected_delta(output_layer, selected_ids, base_head_rows.float())
    edited_head_rows = output_layer.weight.index_select(0, selected_tensor).detach().clone()
    head_norm = float(actual_stage1_delta.norm().detach().cpu())
    if head_norm > float(configuration["acceptance"]["max_head_delta_norm"]):
        raise ValueError("Source Stage-1 head delta exceeds the v2 locked norm budget")
    source_stage1 = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(configuration["acceptance"]["required_pairwise_margin"]),
    )
    declared_source = read_json(source["stage1_report"])
    if source_stage1.get("pairwise_margin_failure_positions") != declared_source.get("pairwise_margin_failure_positions"):
        raise RuntimeError("Reloaded source Stage-1 failure positions differ from the source report")
    active_positions = [int(x) for x in source_stage1["pairwise_margin_failure_positions"]]

    texts, wikipedia_metadata = wikipedia.load_wikipedia_train(Path(args.wikipedia_dir).resolve())
    rep_cfg = configuration["representation"]
    train_take = min(int(rep_cfg["utility_train_prompt_count"]), int(train_indices.numel()))
    gate_take = min(int(rep_cfg["utility_gate_prompt_count"]), int(guard_indices.numel()))
    train_contexts = build_utility_contexts(
        tokenizer, texts, utility_metadata, utility_hidden, train_indices[:train_take].tolist()
    )
    gate_contexts = build_utility_contexts(
        tokenizer, texts, utility_metadata, utility_hidden, guard_indices[:gate_take].tolist()
    )
    write_json(
        output_dir / "utility_replay_report.json",
        {
            "source_pool": utility_pool_report,
            "wikipedia_dataset": wikipedia_metadata,
            "train_prompt_count": len(train_contexts),
            "gate_prompt_count": len(gate_contexts),
            "train_indices_sha256": hashlib.sha256(json.dumps(train_indices[:train_take].tolist()).encode("utf-8")).hexdigest(),
            "gate_indices_sha256": hashlib.sha256(json.dumps(guard_indices[:gate_take].tolist()).encode("utf-8")).hexdigest(),
            "train_guard_overlap_count": len(set(train_indices[:train_take].tolist()) & set(guard_indices[:gate_take].tolist())),
            "official_rwku_records_accessed": False,
        },
    )

    all_attempts: List[Dict[str, Any]] = []
    chosen: Dict[str, Any] | None = None
    chosen_handles: List[representation.AdapterHandle] | None = None
    chosen_originals: List[torch.Tensor] | None = None
    started = time.perf_counter()

    for rank in [int(x) for x in rep_cfg["rank_ladder"]]:
        output_layer.weight.index_copy_(
            0,
            selected_tensor,
            edited_head_rows.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype),
        )
        handles, originals, _ = train_rank(
            model,
            tokenizer,
            prompt_records,
            active_positions,
            train_contexts,
            rank=rank,
            configuration=configuration,
            llama_like=llama_like,
            device=device,
            log_path=output_dir / f"rank{rank}_training_history.json",
        )

        rank_succeeded = False
        for scale in [float(x) for x in rep_cfg["candidate_scales"]]:
            materialization = materialize_adapter_candidate(handles, originals, scale)
            output_layer.weight.index_copy_(
                0,
                selected_tensor,
                edited_head_rows.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype),
            )
            atomic = head.materialized_atomic_report(
                model,
                tokenizer,
                prompt_records,
                device,
                llama_like=llama_like,
                required_margin=float(configuration["acceptance"]["required_pairwise_margin"]),
            )
            repr_delta = representation_delta_report(handles, originals)
            repr_safe = repr_delta["relative_frobenius"] <= float(rep_cfg["max_relative_frobenius_delta"])
            candidate: Dict[str, Any] = {
                "rank": rank,
                "scale": scale,
                "materialization": materialization,
                "atomic": atomic,
                "representation_delta": repr_delta,
                "behavior_safe": behavior_safe(atomic),
                "representation_norm_safe": bool(repr_safe),
                "head_delta_norm": head_norm,
                "head_norm_safe": head_norm <= float(configuration["acceptance"]["max_head_delta_norm"]),
                "official_rwku_records_accessed": False,
            }
            print(
                "repr candidate rank={} scale={}: FS={} other={} minsep={:.4f} relnorm={:.6f}".format(
                    rank,
                    scale,
                    atomic.get("FS"),
                    atomic.get("generated_subject_FS"),
                    float(atomic.get("minimum_overall_separation", float("nan"))),
                    float(repr_delta["relative_frobenius"]),
                )
            )
            if candidate["behavior_safe"] and candidate["representation_norm_safe"] and candidate["head_norm_safe"]:
                utility_kl = exact_full_vocab_utility_kl(
                    model,
                    tokenizer,
                    gate_contexts,
                    output_layer=output_layer,
                    selected_tensor=selected_tensor,
                    base_head_rows=base_head_rows,
                    edited_head_rows=edited_head_rows,
                    handles=handles,
                    original_adapter_weights=originals,
                    scale=scale,
                    device=device,
                    batch_size=int(rep_cfg["utility_context_batch_size"]),
                )
                checks = {
                    "mean": utility_kl["utility_kl_mean"] <= float(configuration["acceptance"]["utility_kl_mean_budget"]),
                    "p95": utility_kl["utility_kl_p95"] <= float(configuration["acceptance"]["utility_kl_p95_budget"]),
                    "max": utility_kl["utility_kl_max"] <= float(configuration["acceptance"]["utility_kl_max_budget"]),
                }
                candidate["utility_kl"] = utility_kl
                candidate["utility_guard_checks"] = checks
                candidate["utility_safe"] = bool(all(checks.values()))
                candidate["feasible"] = bool(candidate["utility_safe"])
                print(
                    "  exact W1K repr gate: mean={:.6f} p95={:.6f} max={:.6f} safe={}".format(
                        utility_kl["utility_kl_mean"],
                        utility_kl["utility_kl_p95"],
                        utility_kl["utility_kl_max"],
                        candidate["utility_safe"],
                    )
                )
            else:
                candidate["utility_safe"] = False
                candidate["feasible"] = False
                candidate["utility_gate_skipped"] = True
            all_attempts.append(candidate)
            write_json(
                output_dir / f"rank{rank}_scale{str(scale).replace('.', 'p')}_report.json",
                candidate,
            )
            if candidate["feasible"]:
                chosen = candidate
                chosen_handles = handles
                chosen_originals = originals
                rank_succeeded = True
                break

        if rank_succeeded:
            break
        restore_adapter_base_weights(handles, originals)
        output_layer.weight.index_copy_(
            0,
            selected_tensor,
            edited_head_rows.to(device=output_layer.weight.device, dtype=output_layer.weight.dtype),
        )
        representation.remove_lora_adapters(handles, merge_scale=0.0)

    write_json(output_dir / "representation_attempts.json", {"attempts": all_attempts})
    if chosen is None or chosen_handles is None or chosen_originals is None:
        write_json(
            output_dir / "infeasible.json",
            {
                "configuration_id": configuration["configuration_id"],
                "source_stage1": source_stage1,
                "active_positions": active_positions,
                "attempts": all_attempts,
                "reason": "no BF16-safe minimal representation rescue passed atomic and W1K gates",
                "official_rwku_records_accessed": False,
            },
        )
        raise RuntimeError("RWKU representation rescue found no feasible checkpoint")

    final_repr_delta = representation_delta_report(chosen_handles, chosen_originals)
    representation_delta_payload: Dict[str, Any] = {
        "rank": int(chosen["rank"]),
        "scale": float(chosen["scale"]),
        "modules": {},
    }
    for handle, original in zip(chosen_handles, chosen_originals):
        delta = handle.wrapper.base.weight.detach().float() - original.to(
            device=handle.wrapper.base.weight.device, dtype=torch.float32
        )
        representation_delta_payload["modules"][handle.path] = delta.cpu()
    representation_delta_path = output_dir / "representation_delta.pt"
    torch.save(representation_delta_payload, representation_delta_path)

    representation.remove_lora_adapters(chosen_handles, merge_scale=0.0)
    final_atomic = head.materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=float(configuration["acceptance"]["required_pairwise_margin"]),
    )
    if not behavior_safe(final_atomic):
        raise RuntimeError("Final unwrapped representation checkpoint failed atomic gate")

    checkpoint_path = output_dir / "checkpoint"
    learner.save_checkpoint(model, tokenizer, checkpoint_path)
    head_delta_path = output_dir / "stage1_sparse_head_delta.pt"
    torch.save({"row_ids": selected_ids, "delta": actual_stage1_delta.detach().cpu()}, head_delta_path)
    write_json(output_dir / "final_atomic_view_report.json", final_atomic)

    training_report = {
        "schema_version": "rwku_sure_head_repr_rescue_w1k_training_report_v1",
        "protocol_label": artifact_access.TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": PROTOCOL_STATUS,
        "method": configuration["method"],
        "configuration_id": configuration["configuration_id"],
        "development_only": True,
        "target": {
            "seed": 0,
            "entity": configuration["target_entity"],
            "entity_id": configuration["target_entity_id"],
        },
        "source_head_only_run": {
            "path": str(Path(args.source_head_only_run).resolve()),
            "stage1_delta_sha256": artifact_access.sha256_file(source["stage1_delta"]),
            "stage1_report_sha256": artifact_access.sha256_file(source["stage1_report"]),
            "infeasible_report_sha256": artifact_access.sha256_file(source["infeasible"]),
            "official_evaluation_opened": False,
        },
        "atomic_bundle": bundle_audit,
        "atomic_generator_base_model": generator_model_audit,
        "utility_cache": utility_audit,
        "source_stage1_atomic_report": source_stage1,
        "active_positions": active_positions,
        "head_delta_norm": head_norm,
        "representation_rescue": {
            "rank_ladder": rep_cfg["rank_ladder"],
            "target_modules": rep_cfg["target_modules"],
            "last_n_layers": rep_cfg["last_n_layers"],
            "chosen": chosen,
            "final_representation_delta": final_repr_delta,
        },
        "final_training_view_report": final_atomic,
        "training_seconds": time.perf_counter() - started,
        "official_rwku_records_accessed": False,
        "official_rwku_records_used_for_training_or_selection": False,
        "final_evaluation_used_for_training_or_selection": False,
    }
    training_report_path = output_dir / "training_report.json"
    write_json(training_report_path, training_report)

    receipt_path = run_dir / "checkpoint_receipt.json"
    method_configuration = {
        "method": configuration["method"],
        "configuration_id": configuration["configuration_id"],
        "configuration_path": str(configuration_path),
        "configuration_sha256": artifact_access.sha256_file(configuration_path),
        "editable_parameters": "stage1_sparse_lm_head_rows_plus_last_layer_down_proj_low_rank_rescue",
        "source_head_only_stage1_delta_sha256": artifact_access.sha256_file(source["stage1_delta"]),
        "representation_rank": int(chosen["rank"]),
        "representation_scale": float(chosen["scale"]),
        "representation_relative_frobenius": float(final_repr_delta["relative_frobenius"]),
        "head_delta_norm": head_norm,
        "utility_gate_kind": "exact_full_vocabulary_base_to_edited",
        "utility_gate_prompt_count": int(chosen["utility_kl"]["utility_prompt_count"]),
        "official_rwku_records_used_for_selection": False,
    }
    implementation_files = [
        SCRIPT_PATH,
        PROJECT_ROOT / "scripts" / "rwku_sure_head_only_w1k.py",
        PROJECT_ROOT / "scripts" / "rwku_representation.py",
        PROJECT_ROOT / "scripts" / "build_sure_wikipedia_stats.py",
        PROJECT_ROOT / "scripts" / "gagd_compare.py",
        PROJECT_ROOT / "scripts" / "rwku_artifact_access.py",
        PROJECT_ROOT / "scripts" / "rwku_checkpoint_receipt.py",
        PROJECT_ROOT / "scripts" / "rwku_eval.py",
        PROJECT_ROOT / "scripts" / "rwku_experiment.py",
        PROJECT_ROOT / "scripts" / "sure_canonical_core.py",
        PROJECT_ROOT / "scripts" / "sure_minimal_two_stage.py",
    ]
    receipt = checkpoint_receipt.create_checkpoint_receipt(
        destination=receipt_path,
        experiment_id=str(args.experiment_id),
        protocol_label=artifact_access.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=PROTOCOL_STATUS,
        target_entity=str(configuration["target_entity"]),
        target_entity_id=str(configuration["target_entity_id"]),
        base_model_identity=rwku_experiment.local_model_identity(args.model_path),
        base_model_revision=str(args.model_revision),
        tokenizer_identity={
            "name_or_path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "tokenizer_probe_sha256": identity["tokenizer_probe_sha256"],
        },
        checkpoint_paths=[checkpoint_path],
        training_bundle_path=Path(args.training_bundle).resolve(),
        optimization_protection_path=None,
        mcf_retain_optimization_paths=[],
        mcf_repair_gate_paths=[],
        matched_protection_train_path=None,
        matched_protection_gate_path=None,
        method_configuration=method_configuration,
        implementation_files=implementation_files,
        sampler_provenance={
            "atomic_view_order_sha256": bundle_audit["view_ids_sha256"],
            "utility_pool_seed": optimization.utility_pool_seed,
            "training_seed": int(configuration["seed"]),
            "representation_utility_train_indices_sha256": hashlib.sha256(
                json.dumps(train_indices[:train_take].tolist()).encode("utf-8")
            ).hexdigest(),
            "representation_utility_gate_indices_sha256": hashlib.sha256(
                json.dumps(guard_indices[:gate_take].tolist()).encode("utf-8")
            ).hexdigest(),
            "official_rwku_records_accessed": False,
        },
        generator_receipt_path=Path(args.generator_receipt).resolve(),
        official_locked_eval_path=run_dir / "official_locked_eval.json",
        confirmatory=False,
        additional_artifact_paths={
            "locked_configuration": configuration_path,
            "utility_cache": Path(args.utility_cache).resolve(),
            "source_head_only_stage1_delta": source["stage1_delta"],
            "source_head_only_infeasible_report": source["infeasible"],
            "training_report": training_report_path,
            "final_sparse_head_delta": head_delta_path,
            "final_representation_delta": representation_delta_path,
        },
    )
    rwku_experiment._write_state(
        state_args,
        "CHECKPOINT_FROZEN",
        checkpoint_receipt=str(receipt_path.resolve()),
        checkpoint_receipt_sha256=receipt["receipt_sha256"],
        official_evaluation_opened=False,
        representation_rescue_feasible=True,
        training_report=str(training_report_path.resolve()),
    )
    print(f"RWKU-H+R-W1K checkpoint frozen: {checkpoint_path}")
    print(f"Representation rescue rank/scale: {chosen['rank']}/{chosen['scale']}")
    print(f"Atomic direct success: {final_atomic['FS']}")
    print(f"Other atomic-view success: {final_atomic['generated_subject_FS']}")
    print(
        "Wikipedia full-vocabulary KL mean/p95/max: "
        f"{chosen['utility_kl']['utility_kl_mean']:.6f}/"
        f"{chosen['utility_kl']['utility_kl_p95']:.6f}/"
        f"{chosen['utility_kl']['utility_kl_max']:.6f}"
    )
    print("Official RWKU evaluation remains unopened; staged evaluation may run next.")


if __name__ == "__main__":
    main()
