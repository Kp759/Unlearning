#!/usr/bin/env python3
"""Embedding-keyed sparse-neuron conditional suppression for locked MCF.

Architecture
------------

    ordinary sparse subject embedding-row deltas (frozen Stage 1)
        -> frozen early Transformer layers
        -> record-specific contextual activation code
        -> sparse existing SwiGLU detector/actuator neurons
        -> frozen remaining Transformer
        -> exactly unchanged LM head

For each edit, a disjoint group of existing low-activation MLP neurons is
selected using only training-safe writer-on versus writer-off activations.
``gate_proj``/``up_proj`` rows learn to detect the complete embedding-induced
contextual code. The matching ``down_proj`` columns learn a suppression residual.
All edits are materialized into ordinary model weights; there is no tokenizer
expansion, string matcher, retrieval cache, runtime router, sidecar, adapter,
LoRA, logit bias, or LM-head update.

Data firewall
-------------

This learner has deliberately no ``--mcf-path`` argument.  It accepts only a
locked direct-forget training view, an audited training-safe context manifest,
and a frozen Stage-1 writer whose stored manifest hash must match exactly.
Official paraphrases, neighborhoods, retain prompts, PPL text, aliases, and
adversarial attacks are unavailable to this process.  They may be opened only
by separate post-checkpoint evaluation processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_active_case_repair as mcf_repair
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional_method
import mcf_compositional_marker_core as compositional_core
import mcf_embedding_keyed_neuron_core as neuron_core
import mcf_sure_directional_emb_lm_stage1 as directional
import mcf_sure_subject_directional_emb_stage1 as subject_writer
import mcf_synthetic_paraphrase_templates as synthetic
import sure_canonical_core as canonical


METHOD = "Embedding-Keyed Sparse-Neuron Conditional Suppression"
PROTOCOL = neuron_core.PROTOCOL
FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "MCF_PATH",
    "OFFICIAL_MCF_PATH",
    "OFFICIAL_EVAL_PATH",
    "PARAPHRASE_PATH",
    "NEIGHBORHOOD_PATH",
    "RETAIN_EVAL_PATH",
    "ADVERSARIAL_EVAL_PATH",
)
COUNT_DERIVED_FRACTION_ABS_TOLERANCE = 1e-7


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--stage1-state", required=True)
    parser.add_argument("--stage1-report", required=True)
    parser.add_argument("--stage1-writer-log", required=True)
    parser.add_argument("--clean-stage1-portability-preflight", required=True)
    parser.add_argument("--clean-stage1-acceptance", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--experiment-label", default="primary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument(
        "--writer-mode",
        choices=("embedding_keyed", "none"),
        default="embedding_keyed",
        help=(
            "embedding_keyed trains the proposed conjunction with the frozen "
            "sparse embedding writer. none is the independently optimized, "
            "matched-capacity sparse-MLP control and materializes no embedding edit."
        ),
    )

    parser.add_argument("--neuron-layer", type=int, default=27)
    parser.add_argument("--neurons-per-record", type=int, default=4)
    parser.add_argument("--dormant-fraction", type=float, default=0.20)
    parser.add_argument(
        "--selection-mode",
        choices=(
            "writer_contrastive",
            "base_context_contrastive",
            "dormant_random",
        ),
        default="writer_contrastive",
    )
    parser.add_argument("--selection-stability-weight", type=float, default=1.0)
    parser.add_argument("--selection-positive-contexts", type=int, default=8)
    parser.add_argument("--selection-negative-contexts", type=int, default=8)
    parser.add_argument("--selection-protected-prompts", type=int, default=8192)
    parser.add_argument(
        "--selection-min-corpus-prompts",
        type=int,
        default=4096,
        help=(
            "Reserve this many disjoint corpus prompts inside the protected-neuron "
            "profile. Training prompts cannot consume this quota."
        ),
    )
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=5000)
    parser.add_argument("--corpus-protection-prompts", type=int, default=8192)
    parser.add_argument(
        "--writer-preflight-amplitude-threshold", type=float, default=4.5
    )
    parser.add_argument(
        "--writer-preflight-min-global-fraction", type=float, default=0.95
    )
    parser.add_argument(
        "--writer-preflight-min-record-fraction", type=float, default=0.80
    )

    parser.add_argument("--detector-steps", type=int, default=1000)
    parser.add_argument("--detector-lr", type=float, default=1e-3)
    parser.add_argument(
        "--detector-record-batch",
        type=int,
        default=4,
        help=(
            "Record-microbatch capacity for detector gradient accumulation. "
            "Every optimizer update still covers every record."
        ),
    )
    parser.add_argument(
        "--detector-positive-contexts",
        choices=("all",),
        default="all",
        help="V3.1 locks detector training to every training-safe positive context.",
    )
    parser.add_argument(
        "--detector-negative-contexts",
        choices=("all",),
        default="all",
        help="V3.1 locks detector training to every training-safe negative context.",
    )
    parser.add_argument("--detector-tail-k", type=int, default=2)
    parser.add_argument(
        "--detector-response-mode",
        choices=("absolute_signed_group_activation",),
        default="absolute_signed_group_activation",
    )
    parser.add_argument("--detector-positive-floor", type=float, default=0.25)
    parser.add_argument("--detector-negative-weight", type=float, default=5.0)
    parser.add_argument("--detector-cross-weight", type=float, default=2.0)
    parser.add_argument("--detector-writer-off-weight", type=float, default=10.0)
    parser.add_argument("--detector-consistency-weight", type=float, default=1.0)
    parser.add_argument("--detector-l2", type=float, default=1e-5)
    parser.add_argument("--detector-relative-cap", type=float, default=1.0)
    parser.add_argument("--detector-off-abs-max", type=float, default=0.20)

    parser.add_argument("--actuator-steps", type=int, default=2000)
    parser.add_argument("--actuator-lr", type=float, default=5e-4)
    parser.add_argument("--actuator-batch-size", type=int, default=4)
    parser.add_argument("--actuator-protected-batch", type=int, default=4)
    parser.add_argument("--actuator-writer-off-every", type=int, default=1)
    parser.add_argument("--forget-margin", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=20.0)
    parser.add_argument("--reference-nll-weight", type=float, default=50.0)
    parser.add_argument("--reference-nll-tolerance", type=float, default=0.05)
    parser.add_argument("--protected-kl-weight", type=float, default=20.0)
    parser.add_argument("--writer-off-nll-weight", type=float, default=50.0)
    parser.add_argument("--actuator-l2", type=float, default=1e-4)
    parser.add_argument("--actuator-relative-cap", type=float, default=0.50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--kl-topk", type=int, default=64)
    parser.add_argument("--hook-materialization-tolerance", type=float, default=0.05)

    parser.add_argument(
        "--min-writer-necessary-direct-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--min-decoder-necessary-direct-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--require-writer-necessity",
        "--require-within-checkpoint-writer-dependence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the within-checkpoint writer-off intervention to break the "
            "joint solution. Disabled only for the preregistered no-writer control."
        ),
    )
    parser.add_argument(
        "--gate-policy",
        choices=("strict", "report"),
        default="strict",
        help="Strict requires the training-only detector certificate before saving.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument(
        "--save-rejected-checkpoint",
        action="store_true",
        help=(
            "Freeze a fixed-budget rejected checkpoint for evaluation. Allowed "
            "only for the independently trained no-writer control."
        ),
    )
    value = parser.parse_args(list(argv) if argv is not None else None)

    if int(value.forget_num) <= 0:
        parser.error("--forget-num must be positive")
    if int(value.neuron_layer) < 0:
        parser.error("--neuron-layer must be non-negative")
    if int(value.frequency_doc_start) < 20:
        parser.error(
            "--frequency-doc-start must be >= 20; documents 0:20 are reserved "
            "for held-out official PPL"
        )
    if int(value.neurons_per_record) <= 0:
        parser.error("--neurons-per-record must be positive")
    if not 0.0 < float(value.dormant_fraction) <= 1.0:
        parser.error("--dormant-fraction must lie in (0, 1]")
    if int(value.selection_protected_prompts) <= 0:
        parser.error("--selection-protected-prompts must be positive")
    if (
        min(
            int(value.selection_positive_contexts),
            int(value.selection_negative_contexts),
        )
        <= 0
    ):
        parser.error("selection positive/negative context counts must be positive")
    if (
        not 0
        <= int(value.selection_min_corpus_prompts)
        <= int(value.selection_protected_prompts)
    ):
        parser.error(
            "--selection-min-corpus-prompts must lie between zero and the "
            "protected-prompt budget"
        )
    if int(value.corpus_protection_prompts) < int(value.selection_min_corpus_prompts):
        parser.error(
            "--corpus-protection-prompts must cover the reserved selection corpus quota"
        )
    if int(value.detector_steps) < 0:
        parser.error("--detector-steps must be non-negative")
    if int(value.detector_record_batch) <= 0:
        parser.error("--detector-record-batch must be positive")
    if int(value.detector_tail_k) <= 0:
        parser.error("--detector-tail-k must be positive")
    if float(value.detector_positive_floor) < 0:
        parser.error("--detector-positive-floor must be non-negative")
    if float(value.detector_off_abs_max) < 0:
        parser.error("--detector-off-abs-max must be non-negative")
    for name in (
        "detector_negative_weight",
        "detector_cross_weight",
        "detector_writer_off_weight",
        "detector_consistency_weight",
        "detector_l2",
    ):
        if float(getattr(value, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    if float(value.writer_preflight_amplitude_threshold) < 0:
        parser.error("--writer-preflight-amplitude-threshold must be non-negative")
    for name in (
        "writer_preflight_min_global_fraction",
        "writer_preflight_min_record_fraction",
    ):
        if not 0.0 <= float(getattr(value, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    if value.writer_mode == "embedding_keyed":
        if value.selection_mode == "base_context_contrastive":
            parser.error(
                "embedding_keyed mode must use writer_contrastive or dormant_random selection"
            )
        if not bool(value.require_writer_necessity):
            parser.error(
                "embedding_keyed mode must retain the within-checkpoint writer-dependence gate"
            )
    else:
        if value.selection_mode == "writer_contrastive":
            parser.error("no-writer control cannot use writer_contrastive selection")
        if bool(value.require_writer_necessity):
            parser.error("no-writer control must disable --require-writer-necessity")
    if bool(value.save_rejected_checkpoint) and value.writer_mode != "none":
        parser.error(
            "--save-rejected-checkpoint is restricted to the no-writer control"
        )
    for name in ("detector_relative_cap", "actuator_relative_cap"):
        if not 0.0 < float(getattr(value, name)) <= 2.0:
            parser.error(f"--{name.replace('_', '-')} must lie in (0, 2]")
    if float(value.hook_materialization_tolerance) < 0:
        parser.error("--hook-materialization-tolerance must be non-negative")
    if value.device_map == "auto":
        parser.error("training sparse neurons requires --device-map single")
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "evaluation path leaked into learner environment: "
            + ", ".join(sorted(exposed))
        )


def _validate_experiment_registry(
    registry: Mapping[str, Any], args: argparse.Namespace
) -> None:
    if registry.get("protocol") != PROTOCOL:
        raise RuntimeError("experiment registry protocol mismatch")
    writer_prerequisite = registry.get("stage1_writer_prerequisite")
    if not isinstance(writer_prerequisite, Mapping):
        raise RuntimeError("registry lacks the clean Stage-1 writer prerequisite")
    expected_template_hash = compositional_method.sha256_json(
        synthetic.RELATION_ALTERNATE_TEMPLATES
    )
    expected_writer_prerequisite = {
        "protocol": compositional_core.PROTOCOL,
        "positive_context_policy": compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY,
        "relation_template_bank_sha256": expected_template_hash,
        "training_origin": "Base model with no resumed Stage-1 state",
        "writer_steps": 1200,
        "writer_steps_semantics": (
            "optimizer updates after full-record gradient accumulation"
        ),
        "writer_record_batch": 3,
        "writer_record_batch_semantics": ("gradient_accumulation_microbatch_capacity"),
        "writer_update_coverage": "all_records_accumulated",
        "writer_records_per_optimizer_update": 50,
        "writer_microbatches_per_optimizer_update": 17,
        "writer_optimizer_updates": 1200,
        "writer_record_exposures": 60000,
        "writer_record_local_reference_exposures": 3600,
        "writer_record_exposure_multiplier_vs_v6_1": 50 / 3,
        "writer_gradient_normalization": (
            "equal_record_mean_plus_global_prompt_mean_kl"
        ),
        "writer_kl_evaluation": (
            "exact_registered_topk_rows_without_full_vocabulary_materialization"
        ),
        "writer_gradient_conflict_audit_phases": ["initial", "final"],
        "writer_gradient_conflict_audit_objectives": [
            "positive_write",
            "full_writer",
        ],
        "writer_gradient_conflict_audit_hash_bound": True,
        "writer_positive_context_mode": "all",
        "writer_positive_context_batch": 7,
        "writer_positive_tail_k": 2,
        "writer_negative_context_batch": 5,
        "writer_objective": "mean_plus_worst_k_squared_shortfall",
        "cross_record_parameter_sharing_audit_required": True,
    }
    for key, expected_value in expected_writer_prerequisite.items():
        if writer_prerequisite.get(key) != expected_value:
            raise RuntimeError(f"registry clean-writer prerequisite mismatch: {key}")
    detector_revision = registry.get("detector_training_revision")
    expected_detector_revision = {
        "version": "v3.1",
        "training_input": "cached_selected_layer_mlp_input_hidden_states",
        "cache_dtype": "float32",
        "cache_device": "cpu",
        "cache_scope": [
            "writer_on_positive",
            "writer_on_negative",
            "writer_off_positive",
        ],
        "update_coverage": "all_records_accumulated",
        "record_microbatch_argument": "detector_record_batch",
        "records_per_optimizer_update": 50,
        "optimizer_updates": 1000,
        "record_exposures": 50000,
        "positive_context_mode": "all",
        "negative_context_mode": "all",
        "tail_k": 2,
        "positive_objective": "mean_plus_worst_k_squared_shortfall",
        "negative_objective": "mean_plus_worst_k_squared_gate_excess",
        "cross_objective": "mean_plus_worst_k_squared_gate_excess",
        "writer_off_objective": "mean_plus_worst_k_squared_gate_excess",
        "gradient_normalization": "equal_record_mean",
        "gradient_clip_frequency": "once_per_optimizer_update",
        "norm_projection_frequency": "once_per_optimizer_update",
        "official_evaluation_prompts_seen": 0,
    }
    if not isinstance(detector_revision, Mapping):
        raise RuntimeError("registry lacks the V3.1 detector-training revision")
    for key, expected_value in expected_detector_revision.items():
        if detector_revision.get(key) != expected_value:
            raise RuntimeError(f"registry V3.1 detector revision mismatch: {key}")
    label = str(args.experiment_label)
    if label == "primary":
        expected = registry.get("primary_configuration")
        if not isinstance(expected, Mapping):
            raise RuntimeError("registry lacks primary configuration")
    else:
        changes = None
        rows = registry.get("controlled_training_ablations", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and str(row.get("name")) == label:
                    changes = row.get("change_only")
                    break
        if changes is None and label.startswith("neurons_per_record_"):
            changes = {"neurons_per_record": int(label.rsplit("_", 1)[-1])}
        if changes is None and label.startswith("layer_"):
            changes = {"neuron_layer": int(label.rsplit("_", 1)[-1])}
        if not isinstance(changes, Mapping):
            raise RuntimeError(f"experiment label is not preregistered: {label!r}")
        primary = registry.get("primary_configuration")
        common = registry.get("ablation_common_configuration", {})
        if not isinstance(primary, Mapping) or not isinstance(common, Mapping):
            raise RuntimeError("registry lacks primary/ablation-common configuration")
        expected = {**primary, **common, **changes}
    for key, registered in expected.items():
        if not hasattr(args, str(key)):
            raise RuntimeError(f"registry key has no learner argument: {key!r}")
        observed = getattr(args, str(key))
        if isinstance(registered, float):
            matches = math.isclose(float(observed), float(registered), abs_tol=1e-12)
        else:
            matches = observed == registered
        if not matches:
            raise RuntimeError(
                f"experiment {label!r} diverges from registry: {key}="
                f"{observed!r}, expected {registered!r}"
            )


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    return compositional_method.distribution([float(value) for value in values])


def _tensor_digest(tensor: torch.Tensor, *, row_chunk: int = 256) -> str:
    digest = hashlib.sha256()
    detached = tensor.detach()
    for start in range(0, int(detached.shape[0]), int(row_chunk)):
        block = detached[start : start + int(row_chunk)].contiguous().cpu()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _resolve_swiglu_mlp(model: torch.nn.Module, layer_index: int) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("model does not expose model.layers")
    if int(layer_index) >= len(layers):
        raise ValueError(
            f"neuron layer {layer_index} is outside model with {len(layers)} layers"
        )
    mlp = getattr(layers[int(layer_index)], "mlp", None)
    if mlp is None:
        raise RuntimeError(f"layer {layer_index} has no MLP")
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, name):
            raise RuntimeError(
                f"layer {layer_index} MLP is not a supported SwiGLU module: missing {name}"
            )
    if mlp.gate_proj.weight.shape != mlp.up_proj.weight.shape:
        raise RuntimeError("gate_proj and up_proj shapes differ")
    if int(mlp.down_proj.weight.shape[1]) != int(mlp.gate_proj.weight.shape[0]):
        raise RuntimeError("down_proj input does not match SwiGLU intermediate size")
    return mlp


def _validate_firewall(
    context_manifest: Mapping[str, Any], stage1_state: Mapping[str, Any]
) -> None:
    data_access = context_manifest.get("data_access")
    if not isinstance(data_access, Mapping):
        raise RuntimeError("context manifest lacks data-access receipt")
    expected_zero = {
        "official_paraphrases_seen": 0,
        "official_neighborhoods_seen": 0,
        "benchmark_retain_seen": 0,
        "official_ppl_seen": False,
    }
    for key, expected in expected_zero.items():
        if data_access.get(key) != expected:
            raise RuntimeError(
                f"data firewall violation: {key}={data_access.get(key)!r}, "
                f"expected {expected!r}"
            )
    forbidden = {
        "paraphrase_prompts",
        "neighborhood_prompts",
        "retain_prompts",
        "official_ppl_text",
        "adversarial_prompts",
        "alias_prompts",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden.intersection(str(key) for key in value)
            if overlap:
                raise RuntimeError(
                    "training context artifact contains evaluation-only fields: "
                    + ", ".join(sorted(overlap))
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(context_manifest)
    stored_hash = str(stage1_state.get("context_manifest_sha256") or "")
    if not stored_hash:
        raise RuntimeError("Stage-1 state lacks its exact context-manifest hash")


def _validate_clean_stage1_lineage(
    context_manifest: Mapping[str, Any],
    stage1_state: Mapping[str, Any],
    stage1_report: Mapping[str, Any],
    context_path: Path,
    stage1_log_path: Path,
) -> Dict[str, Any]:
    """Require a fresh, relation-preserving Stage-1 writer artifact."""

    policy = context_manifest.get("positive_context_policy")
    if not isinstance(policy, Mapping):
        raise RuntimeError("context manifest lacks its positive-context policy")
    expected_policy = compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY
    if str(policy.get("name") or "") != expected_policy:
        raise RuntimeError(
            "neuron suppression requires the clean relation-templates-only "
            f"writer policy {expected_policy!r}"
        )
    if bool(policy.get("free_form_generated_surrogates_allowed")):
        raise RuntimeError(
            "clean writer policy unexpectedly allows free-form surrogates"
        )
    expected_template_hash = compositional_method.sha256_json(
        synthetic.RELATION_ALTERNATE_TEMPLATES
    )
    if str(policy.get("relation_template_bank_sha256") or "") != expected_template_hash:
        raise RuntimeError("clean writer relation-template bank hash is stale")
    counts = policy.get("source_prompt_counts")
    if (
        not isinstance(counts, Mapping)
        or int(counts.get("external_free_form_surrogate", -1)) != 0
    ):
        raise RuntimeError("clean writer contains external free-form surrogate prompts")
    if context_manifest.get("surrogate_receipt") is not None:
        raise RuntimeError("clean writer context manifest contains a surrogate receipt")
    coverage = context_manifest.get("synthetic_coverage")
    if (
        not isinstance(coverage, Mapping)
        or int(coverage.get("generic_fallback_records", -1)) != 0
    ):
        raise RuntimeError(
            "clean writer lacks explicit hand-authored relation coverage for every record"
        )
    rows = context_manifest.get("records")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("clean writer context manifest has no records")
    sharing = context_manifest.get("cross_record_parameter_sharing")
    selected_embedding_rows = context_manifest.get("selected_embedding_rows")
    if (
        not isinstance(sharing, Mapping)
        or not isinstance(selected_embedding_rows, list)
        or int(sharing.get("case_count", -1)) != len(rows)
        or int(sharing.get("selected_row_count", -1)) != len(selected_embedding_rows)
        or int(sharing.get("positive_prompt_count", -1))
        != sum(
            len(row.get("positive_prompts", []))
            for row in rows
            if isinstance(row, Mapping)
        )
    ):
        raise RuntimeError(
            "clean writer lacks its cross-record parameter-sharing audit"
        )
    allowed_sources = {
        "canonical_direct",
        "hand_authored_relation_template_or_corpus_prefix",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("clean writer context record is invalid")
        positives = row.get("positive_prompts")
        provenance = row.get("positive_prompt_provenance")
        if (
            not isinstance(positives, list)
            or not isinstance(provenance, list)
            or len(provenance) != len(positives)
        ):
            raise RuntimeError("clean writer prompt provenance is incomplete")
        for index, (prompt, source_row) in enumerate(zip(positives, provenance)):
            if (
                not isinstance(source_row, Mapping)
                or str(source_row.get("prompt") or "") != str(prompt)
                or str(source_row.get("source") or "") not in allowed_sources
            ):
                raise RuntimeError("clean writer prompt provenance is invalid")
            if index == 0 and source_row.get("source") != "canonical_direct":
                raise RuntimeError("clean writer direct prompt is not first")

    if str(stage1_state.get("protocol") or "") != compositional_core.PROTOCOL:
        raise RuntimeError("Stage-1 writer protocol is stale or incompatible")
    lineage = stage1_state.get("training_lineage")
    if not isinstance(lineage, Mapping):
        raise RuntimeError("Stage-1 state lacks a training-lineage receipt")
    if (
        not bool(lineage.get("from_scratch"))
        or str(lineage.get("mode") or "") != "from_scratch"
        or lineage.get("resumed_from") is not None
        or int(lineage.get("writer_steps", 0)) != 1200
    ):
        raise RuntimeError(
            "Stage-1 writer must be trained from Base for exactly 1,200 steps"
        )
    expected_optimization = {
        "record_batch": 3,
        "record_batch_semantics": "gradient_accumulation_microbatch_capacity",
        "update_coverage": "all_records_accumulated",
        "records_per_optimizer_update": 50,
        "microbatches_per_optimizer_update": 17,
        "optimizer_updates": 1200,
        "record_exposures": 60000,
        "record_local_reference_exposures": 3600,
        "record_exposure_multiplier_vs_record_local": 50 / 3,
        "gradient_normalization": ("equal_record_mean_plus_global_prompt_mean_kl"),
        "kl_evaluation": (
            "exact_registered_topk_rows_without_full_vocabulary_materialization"
        ),
        "gradient_conflict_audit_phases": ["initial", "final"],
        "gradient_conflict_audit_objectives": ["positive_write", "full_writer"],
        "positive_context_mode": "all",
        "positive_context_batch": 7,
        "positive_tail_k": 2,
        "negative_context_batch": 5,
        "objective": "mean_plus_worst_k_squared_shortfall",
    }
    observed_optimization = lineage.get("writer_optimization")
    if (
        not isinstance(observed_optimization, Mapping)
        or dict(observed_optimization) != expected_optimization
    ):
        raise RuntimeError(
            "Stage-1 writer was not trained with the registered V6.2 "
            "all-record accumulated objective"
        )
    if dict(stage1_state.get("writer_optimization", {})) != expected_optimization:
        raise RuntimeError("Stage-1 state writer-optimization receipt mismatch")
    context_hash = compositional_method.sha256_file(context_path)
    if str(lineage.get("current_context_manifest_sha256") or "") != context_hash:
        raise RuntimeError(
            "Stage-1 lineage is not bound to the current context manifest"
        )
    if str(lineage.get("positive_context_policy") or "") != expected_policy:
        raise RuntimeError("Stage-1 lineage records a different context policy")
    base_model_path = str(lineage.get("base_model_path") or "")
    base_transformer_fingerprint = float(
        lineage.get("base_transformer_fingerprint", float("nan"))
    )
    base_selected_rows_sha256 = str(
        lineage.get("base_selected_embedding_rows_sha256") or ""
    )
    if (
        not base_model_path
        or not math.isfinite(base_transformer_fingerprint)
        or len(base_selected_rows_sha256) != 64
    ):
        raise RuntimeError("Stage-1 lineage lacks its Base-model binding")

    if str(stage1_report.get("protocol") or "") != compositional_core.PROTOCOL:
        raise RuntimeError("Stage-1 writer report protocol mismatch")
    if str(stage1_report.get("context_manifest_sha256") or "") != context_hash:
        raise RuntimeError("Stage-1 report is not bound to the context manifest")
    if str(stage1_report.get("positive_context_policy") or "") != expected_policy:
        raise RuntimeError("Stage-1 report records a different context policy")
    writer_configuration = stage1_report.get("writer_configuration")
    if not isinstance(writer_configuration, Mapping) or any(
        writer_configuration.get(key) != value
        for key, value in expected_optimization.items()
    ):
        raise RuntimeError("Stage-1 report writer-optimization receipt mismatch")
    report_lineage = stage1_report.get("training_lineage")
    if not isinstance(report_lineage, Mapping) or dict(report_lineage) != dict(lineage):
        raise RuntimeError("Stage-1 state/report lineage receipts differ")

    state_gradient_audit = stage1_state.get("gradient_conflict_audit")
    report_gradient_audit = stage1_report.get("gradient_conflict_audit")
    gradient_audit_sha256 = str(
        stage1_state.get("gradient_conflict_audit_sha256") or ""
    )
    gradient_audit_path = stage1_log_path.with_name(
        "stage1_gradient_conflict_audit.json"
    )
    gradient_audit_file_sha256 = (
        compositional_method.sha256_file(gradient_audit_path)
        if gradient_audit_path.is_file()
        else ""
    )
    if (
        not isinstance(state_gradient_audit, Mapping)
        or not isinstance(report_gradient_audit, Mapping)
        or dict(state_gradient_audit) != dict(report_gradient_audit)
        or gradient_audit_sha256
        != compositional_method.sha256_json(state_gradient_audit)
        or str(stage1_report.get("gradient_conflict_audit_sha256") or "")
        != gradient_audit_sha256
        or str(lineage.get("gradient_conflict_audit_sha256") or "")
        != gradient_audit_sha256
        or not gradient_audit_file_sha256
        or str(stage1_state.get("gradient_conflict_audit_file_sha256") or "")
        != gradient_audit_file_sha256
        or str(stage1_report.get("gradient_conflict_audit_file_sha256") or "")
        != gradient_audit_file_sha256
        or str(lineage.get("gradient_conflict_audit_file_sha256") or "")
        != gradient_audit_file_sha256
        or state_gradient_audit.get("official_evaluation_opened") is not False
    ):
        raise RuntimeError("Stage-1 gradient-conflict audit binding is invalid")
    expected_case_ids = [int(row["case_id"]) for row in rows]
    for phase in ("initial", "final"):
        phase_report = state_gradient_audit.get(phase)
        if not isinstance(phase_report, Mapping) or phase_report.get("phase") != phase:
            raise RuntimeError("Stage-1 gradient-conflict audit phase is incomplete")
        for objective in ("positive_write", "full_writer"):
            objective_report = phase_report.get(objective)
            if (
                not isinstance(objective_report, Mapping)
                or [int(value) for value in objective_report.get("case_ids", [])]
                != expected_case_ids
            ):
                raise RuntimeError(
                    "Stage-1 gradient-conflict audit objective is incomplete"
                )
    for objective in ("positive_write", "full_writer"):
        initial_report = state_gradient_audit["initial"][objective]
        if (
            int(initial_report.get("nonzero_gradient_records", -1)) != len(rows)
            or int(initial_report.get("valid_pair_count", -1))
            != len(rows) * (len(rows) - 1) // 2
        ):
            raise RuntimeError(
                "Stage-1 initial gradient-conflict audit has zero gradients"
            )

    if not stage1_log_path.is_file():
        raise RuntimeError("Stage-1 writer log is missing")
    log_sha256 = compositional_method.sha256_file(stage1_log_path)
    expected_log_sha256 = str(stage1_state.get("writer_log_sha256") or "")
    if not expected_log_sha256 or log_sha256 != expected_log_sha256:
        raise RuntimeError("Stage-1 writer log hash does not match its state")
    if str(stage1_report.get("writer_log_sha256") or "") != log_sha256:
        raise RuntimeError("Stage-1 writer log hash does not match its report")
    with stage1_log_path.open("r", encoding="utf-8") as log_handle:
        log_events = sum(1 for line in log_handle if line.strip())
    expected_events = int(stage1_state.get("writer_log_event_count", -1))
    if log_events <= 0 or log_events != expected_events:
        raise RuntimeError("Stage-1 writer log is empty or has an invalid event count")
    if int(stage1_report.get("writer_log_event_count", -1)) != log_events:
        raise RuntimeError("Stage-1 writer report log-event count mismatch")
    return {
        "positive_context_policy": expected_policy,
        "protocol": compositional_core.PROTOCOL,
        "from_scratch": True,
        "writer_steps": int(lineage["writer_steps"]),
        "context_manifest_sha256": context_hash,
        "writer_log_sha256": log_sha256,
        "writer_log_event_count": log_events,
        "gradient_conflict_audit_sha256": gradient_audit_sha256,
        "gradient_conflict_audit_file_sha256": gradient_audit_file_sha256,
        "writer_optimization": expected_optimization,
        "base_model_path": base_model_path,
        "base_transformer_fingerprint": base_transformer_fingerprint,
        "base_selected_embedding_rows_sha256": base_selected_rows_sha256,
    }


def _context_sets_by_case(
    context_manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    rows = context_manifest.get("records")
    if not isinstance(rows, list):
        raise RuntimeError("context manifest lacks record contexts")
    by_case: Dict[int, Dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise RuntimeError("invalid context-manifest record")
        case_id = int(value["case_id"])
        positives = value.get("positive_prompts")
        negatives = value.get("negative_contexts")
        if not isinstance(positives, list) or not positives:
            raise RuntimeError(f"case {case_id} has no training positives")
        if not isinstance(negatives, list) or not negatives:
            raise RuntimeError(f"case {case_id} has no training negatives")
        by_case[case_id] = dict(value)
    expected = [int(record["case_id"]) for record in records]
    if set(by_case) != set(expected):
        raise RuntimeError("context-manifest cases do not match locked training view")
    for record in records:
        case_id = int(record["case_id"])
        direct = str(record["direct_prompt"])
        if str(by_case[case_id]["positive_prompts"][0]) != direct:
            raise RuntimeError(
                f"case {case_id} direct prompt changed in context manifest"
            )
    return by_case


def _validate_clean_stage1_acceptance(
    receipt: Mapping[str, Any],
    *,
    seed: int,
    case_ids: Sequence[int],
    expected_artifacts: Mapping[str, str],
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> Dict[str, Any]:
    """Require the integrity *and* replayed portability conjunction."""

    if receipt.get("kind") != "mcf_clean_stage1_writer_acceptance":
        raise RuntimeError("clean Stage-1 acceptance receipt has the wrong kind")
    if receipt.get("passed") is not True:
        raise RuntimeError("clean Stage-1 acceptance receipt did not pass")
    if str(receipt.get("protocol") or "") != compositional_core.PROTOCOL:
        raise RuntimeError("clean Stage-1 acceptance protocol mismatch")
    if int(receipt.get("seed", -1)) != int(seed):
        raise RuntimeError("clean Stage-1 acceptance seed mismatch")
    if [int(value) for value in receipt.get("case_ids", [])] != [
        int(value) for value in case_ids
    ]:
        raise RuntimeError("clean Stage-1 acceptance case IDs mismatch")
    checks = receipt.get("checks")
    required_checks = ("artifact_integrity", "training_safe_portability")
    if not isinstance(checks, Mapping) or not all(
        checks.get(key) is True for key in required_checks
    ):
        raise RuntimeError("clean Stage-1 acceptance conjunction is incomplete")
    if bool(receipt.get("official_evaluation_opened")):
        raise RuntimeError("clean Stage-1 acceptance crossed the evaluation firewall")
    artifacts = receipt.get("artifacts")
    if (
        not isinstance(artifacts, Mapping)
        or not str(artifacts.get("stage1_gradient_conflict_audit_sha256") or "")
        or any(
            str(artifacts.get(key) or "") != value
            for key, value in expected_artifacts.items()
        )
    ):
        raise RuntimeError("clean Stage-1 acceptance artifact binding mismatch")

    preflight = receipt.get("training_safe_portability")
    if not isinstance(preflight, Mapping) or preflight.get("passed") is not True:
        raise RuntimeError("clean Stage-1 portability preflight did not pass")
    if preflight.get("kind") != "mcf_clean_stage1_training_safe_portability_preflight":
        raise RuntimeError("clean Stage-1 portability preflight has the wrong kind")
    validate_writer_preflight_summary(
        preflight,
        amplitude_threshold=float(amplitude_threshold),
        minimum_global_fraction=float(minimum_global_fraction),
        minimum_record_fraction=float(minimum_record_fraction),
    )
    return {
        "kind": str(receipt["kind"]),
        "passed": True,
        "training_safe_portability": {
            "prompt_count": int(preflight["prompt_count"]),
            "complete_count": int(preflight["complete_count"]),
            "global_complete_fraction": float(preflight["global_complete_fraction"]),
            "minimum_record_complete_fraction": float(
                preflight["minimum_record_complete_fraction"]
            ),
            "amplitude_threshold": float(preflight["amplitude_threshold"]),
        },
    }


def _record_views(locked_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return compositional_method._record_views(locked_records)


def _unique_prompts(values: Sequence[str]) -> List[str]:
    return compositional_core.ordered_unique([str(value) for value in values])


def _round_robin_unique_prompts(
    groups: Sequence[Sequence[str]],
    *,
    limit: int,
    excluded: Sequence[str] = (),
) -> List[str]:
    """Select prompts without letting early records consume a fixed budget."""

    if int(limit) <= 0:
        return []
    normalized_groups = [_unique_prompts(group) for group in groups]
    positions = [0 for _ in normalized_groups]
    seen = {str(value) for value in excluded}
    selected: List[str] = []
    while len(selected) < int(limit):
        progressed = False
        for group_index, group in enumerate(normalized_groups):
            while positions[group_index] < len(group):
                prompt = str(group[positions[group_index]])
                positions[group_index] += 1
                if prompt in seen:
                    continue
                seen.add(prompt)
                selected.append(prompt)
                progressed = True
                break
            if len(selected) >= int(limit):
                break
        if not progressed:
            break
    return selected


def build_selection_protected_prompts(
    training_groups: Sequence[Sequence[str]],
    corpus_prompts: Sequence[str],
    *,
    total_limit: int,
    minimum_corpus: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build an auditable, stratified neuron-profile prompt bank.

    Corpus prompts are reserved first, then the training quota is filled in
    round-robin record order.  Spare capacity is backfilled from either source.
    This prevents the historical ``all_training + corpus`` truncation from
    silently deleting the broad-corpus profile or all prompts from later cases.
    """

    total_limit = int(total_limit)
    minimum_corpus = int(minimum_corpus)
    if total_limit <= 0 or not 0 <= minimum_corpus <= total_limit:
        raise ValueError("invalid protected-prompt quotas")
    corpus = _unique_prompts(corpus_prompts)
    if len(corpus) < minimum_corpus:
        raise RuntimeError(
            f"only {len(corpus)} unique corpus prompts are available; "
            f"{minimum_corpus} are required for broad neuron profiling"
        )
    reserved_corpus = corpus[:minimum_corpus]
    training_quota = total_limit - len(reserved_corpus)
    training = _round_robin_unique_prompts(
        training_groups,
        limit=training_quota,
        excluded=reserved_corpus,
    )
    selected = [*reserved_corpus, *training]
    if len(selected) < total_limit:
        selected_set = set(selected)
        selected.extend(
            prompt for prompt in corpus[minimum_corpus:] if prompt not in selected_set
        )
        selected = selected[:total_limit]
    if len(selected) < total_limit:
        selected.extend(
            _round_robin_unique_prompts(
                training_groups,
                limit=total_limit - len(selected),
                excluded=selected,
            )
        )
    selected = _unique_prompts(selected)[:total_limit]
    corpus_set = set(corpus)
    training_set = {
        prompt for group in training_groups for prompt in _unique_prompts(group)
    }
    selected_set = set(selected)
    source_counts = {
        "corpus": sum(prompt in corpus_set for prompt in selected),
        "training_only": sum(
            prompt in training_set and prompt not in corpus_set for prompt in selected
        ),
        "unclassified": sum(
            prompt not in training_set and prompt not in corpus_set
            for prompt in selected
        ),
    }
    represented_groups = sum(
        any(prompt in selected_set for prompt in _unique_prompts(group))
        for group in training_groups
    )
    return selected, {
        "total_limit": total_limit,
        "minimum_corpus_required": minimum_corpus,
        "selected_total": len(selected),
        "full_budget_passed": len(selected) == total_limit,
        "source_counts": source_counts,
        "training_groups_total": len(training_groups),
        "training_groups_represented": represented_groups,
        "corpus_quota_passed": source_counts["corpus"] >= minimum_corpus,
        "all_training_groups_represented": represented_groups == len(training_groups),
    }


def activation_tail_profile(
    activations: torch.Tensor,
    neuron_ids: Sequence[int],
    *,
    activation_threshold: float,
    down_column_norms: torch.Tensor | None = None,
) -> Dict[str, Any]:
    """Summarize how much selected neurons already function on a prompt bank."""

    values = activations.detach().float().cpu()
    if values.ndim != 2 or values.shape[1] != len(neuron_ids):
        raise ValueError("activation profile shape does not match selected neurons")
    if values.shape[0] == 0:
        raise ValueError("activation profile needs at least one prompt")
    if not torch.isfinite(values).all():
        raise ValueError("activation profile contains non-finite values")
    if float(activation_threshold) < 0:
        raise ValueError("activation threshold must be non-negative")
    if down_column_norms is not None:
        down_norms = down_column_norms.detach().float().cpu().reshape(-1)
        if down_norms.numel() != len(neuron_ids):
            raise ValueError("down-column norms do not match selected neurons")
    else:
        down_norms = None

    absolute = values.abs()
    quantile_levels = torch.tensor([0.99, 0.999], dtype=torch.float32)
    quantiles = torch.quantile(absolute, quantile_levels, dim=0)
    rms = values.square().mean(dim=0).sqrt()
    per_neuron: List[Dict[str, Any]] = []
    for column, neuron_id in enumerate(neuron_ids):
        row: Dict[str, Any] = {
            "neuron_id": int(neuron_id),
            "activation_rms": float(rms[column]),
            "activation_abs_p99": float(quantiles[0, column]),
            "activation_abs_p999": float(quantiles[1, column]),
            "activation_abs_max": float(absolute[:, column].max()),
            "activation_threshold_exceedance_fraction": float(
                (absolute[:, column] > float(activation_threshold)).float().mean()
            ),
        }
        if down_norms is not None:
            norm = float(down_norms[column])
            row.update(
                {
                    "base_down_column_norm": norm,
                    "base_residual_contribution_rms_bound": norm
                    * row["activation_rms"],
                    "base_residual_contribution_abs_p999_bound": norm
                    * row["activation_abs_p999"],
                }
            )
        per_neuron.append(row)

    return {
        "prompt_count": int(values.shape[0]),
        "neuron_count": int(values.shape[1]),
        "activation_threshold": float(activation_threshold),
        "activation_rms": _distribution([row["activation_rms"] for row in per_neuron]),
        "activation_abs_p999": _distribution(
            [row["activation_abs_p999"] for row in per_neuron]
        ),
        "activation_abs_max": _distribution(
            [row["activation_abs_max"] for row in per_neuron]
        ),
        "per_neuron": per_neuron,
    }


def summarize_writer_preflight(
    amplitude_groups: Sequence[torch.Tensor],
    *,
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> Dict[str, Any]:
    if not amplitude_groups or any(group.numel() == 0 for group in amplitude_groups):
        raise ValueError("writer preflight needs non-empty amplitudes per record")
    groups = [group.detach().float().reshape(-1).cpu() for group in amplitude_groups]
    if not all(torch.isfinite(group).all() for group in groups):
        raise ValueError("writer preflight contains non-finite amplitudes")
    per_record = []
    for group in groups:
        prompt_count = int(group.numel())
        complete_count = int((group >= float(amplitude_threshold)).sum())
        per_record.append(
            {
                "prompt_count": prompt_count,
                "complete_count": complete_count,
                "complete_fraction": complete_count / prompt_count,
                "amplitude": _distribution([float(value) for value in group]),
            }
        )
    total = sum(row["prompt_count"] for row in per_record)
    complete = sum(row["complete_count"] for row in per_record)
    global_fraction = complete / total
    checks = {
        "global_complete_fraction": global_fraction >= float(minimum_global_fraction),
        "minimum_record_complete_fraction": min(
            row["complete_fraction"] for row in per_record
        )
        >= float(minimum_record_fraction),
    }
    return {
        "amplitude_threshold": float(amplitude_threshold),
        "prompt_count": total,
        "complete_count": complete,
        "global_complete_fraction": global_fraction,
        "minimum_record_complete_fraction": min(
            row["complete_fraction"] for row in per_record
        ),
        "criterion": {
            "minimum_global_fraction": float(minimum_global_fraction),
            "minimum_record_fraction": float(minimum_record_fraction),
        },
        "per_record": per_record,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_writer_preflight_summary(
    summary: Mapping[str, Any],
    *,
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> None:
    """Recompute the portability decision from its per-record counts."""

    rows = summary.get("per_record")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("writer preflight has no per-record rows")
    prompt_count = 0
    complete_count = 0
    fractions: List[float] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise RuntimeError("writer preflight per-record row is invalid")
        prompts = int(row.get("prompt_count", 0))
        complete = int(row.get("complete_count", -1))
        if prompts <= 0 or complete < 0 or complete > prompts:
            raise RuntimeError("writer preflight per-record counts are invalid")
        fraction = complete / prompts
        if not math.isclose(
            float(row.get("complete_fraction", float("nan"))),
            fraction,
            rel_tol=0.0,
            abs_tol=COUNT_DERIVED_FRACTION_ABS_TOLERANCE,
        ):
            raise RuntimeError("writer preflight per-record fraction is inconsistent")
        prompt_count += prompts
        complete_count += complete
        fractions.append(fraction)
    global_fraction = complete_count / prompt_count
    minimum_observed = min(fractions)
    for observed, expected, label in (
        (summary.get("prompt_count"), prompt_count, "prompt count"),
        (summary.get("complete_count"), complete_count, "complete count"),
    ):
        if int(observed if observed is not None else -1) != expected:
            raise RuntimeError(f"writer preflight {label} is inconsistent")
    for observed, expected, label in (
        (
            float(summary.get("amplitude_threshold", float("nan"))),
            float(amplitude_threshold),
            "amplitude threshold",
        ),
    ):
        if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"writer preflight {label} is inconsistent")
    for observed, expected, label in (
        (
            float(summary.get("global_complete_fraction", float("nan"))),
            global_fraction,
            "global fraction",
        ),
        (
            float(summary.get("minimum_record_complete_fraction", float("nan"))),
            minimum_observed,
            "minimum-record fraction",
        ),
    ):
        if not math.isclose(
            observed,
            expected,
            rel_tol=0.0,
            abs_tol=COUNT_DERIVED_FRACTION_ABS_TOLERANCE,
        ):
            raise RuntimeError(f"writer preflight {label} is inconsistent")
    criterion = summary.get("criterion")
    if (
        not isinstance(criterion, Mapping)
        or not math.isclose(
            float(criterion.get("minimum_global_fraction", float("nan"))),
            float(minimum_global_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(criterion.get("minimum_record_fraction", float("nan"))),
            float(minimum_record_fraction),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("writer preflight registered fractions are inconsistent")
    expected_checks = {
        "global_complete_fraction": global_fraction >= float(minimum_global_fraction),
        "minimum_record_complete_fraction": minimum_observed
        >= float(minimum_record_fraction),
    }
    observed_checks = summary.get("checks")
    if not isinstance(observed_checks, Mapping) or any(
        observed_checks.get(key) is not value for key, value in expected_checks.items()
    ):
        raise RuntimeError("writer preflight Boolean checks are inconsistent")
    if summary.get("passed") is not all(expected_checks.values()):
        raise RuntimeError("writer preflight pass flag is inconsistent")


@torch.no_grad()
def measure_training_safe_writer_preflight(
    model: torch.nn.Module,
    tok: Any,
    embedding_writer: Any,
    positive_prompts_by_record: Sequence[Sequence[str]],
    marker_map: Mapping[Any, Any],
    case_ids: Sequence[int],
    device: torch.device,
    *,
    batch_size: int,
    amplitude_threshold: float,
    minimum_global_fraction: float,
    minimum_record_fraction: float,
) -> Dict[str, Any]:
    """Replay the exact pre-decoder writer gate on training-safe positives."""

    if len(positive_prompts_by_record) != len(case_ids):
        raise ValueError("writer preflight prompt groups/case IDs differ")
    previous_enabled = bool(embedding_writer.enabled)
    amplitude_groups: List[torch.Tensor] = []
    try:
        for position, prompts in enumerate(positive_prompts_by_record):
            case_id = int(case_ids[position])
            marker = marker_map.get(case_id, marker_map.get(str(case_id)))
            if not isinstance(marker, torch.Tensor) or marker.ndim != 1:
                raise RuntimeError(f"Stage-1 marker missing for case {case_id}")
            embedding_writer.enabled = False
            writer_off_hidden = compositional_method.batched_last_hidden_only(
                model,
                tok,
                list(prompts),
                device,
                batch_size=int(batch_size),
            )
            embedding_writer.enabled = True
            writer_on_hidden = compositional_method.batched_last_hidden_only(
                model,
                tok,
                list(prompts),
                device,
                batch_size=int(batch_size),
            )
            amplitude_groups.append(
                (writer_on_hidden - writer_off_hidden) @ marker.detach().float().cpu()
            )
    finally:
        embedding_writer.enabled = previous_enabled

    result = summarize_writer_preflight(
        amplitude_groups,
        amplitude_threshold=float(amplitude_threshold),
        minimum_global_fraction=float(minimum_global_fraction),
        minimum_record_fraction=float(minimum_record_fraction),
    )
    for index, row in enumerate(result["per_record"]):
        row["case_id"] = int(case_ids[index])
    result.update(
        {
            "kind": "frozen_writer_training_safe_pre_decoder_preflight",
            "applicable": True,
            "official_evaluation_prompts_seen": 0,
            "used_for_decoder_hyperparameter_selection": False,
        }
    )
    return result


@torch.no_grad()
def capture_base_last_token_activations(
    model: torch.nn.Module,
    tok: Any,
    mlp: torch.nn.Module,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Capture the ordinary SwiGLU activation entering ``down_proj``."""
    rows: List[torch.Tensor] = []
    captured: List[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, inputs: Any) -> None:
        captured.append(inputs[0])

    handle = mlp.down_proj.register_forward_pre_hook(pre_hook)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("activation capture requires a separate model backbone")
        for start in range(0, len(prompts), int(batch_size)):
            batch = list(prompts[start : start + int(batch_size)])
            encoded = tok(batch, padding=True, return_tensors="pt").to(device)
            captured.clear()
            backbone(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError(
                    f"expected one down-projection activation, captured {len(captured)}"
                )
            activation = captured[0]
            positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_rows = torch.arange(len(batch), device=device)
            rows.append(activation[batch_rows, positions, :].detach().float().cpu())
    finally:
        handle.remove()
        tok.padding_side = old_side
    if not rows:
        return torch.empty((0, int(mlp.gate_proj.weight.shape[0])))
    return torch.cat(rows, dim=0)


@torch.no_grad()
def capture_mlp_input_last_token_hidden_states(
    model: torch.nn.Module,
    tok: Any,
    mlp: torch.nn.Module,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Cache the exact frozen input used by selected detector gate/up rows.

    During detector training the sparse down projection is disabled, so edits
    to this MLP cannot affect the tensor entering it.  Caching that tensor once
    therefore removes repeated 3B-backbone execution without approximating the
    learned detector computation.  Values are stored as CPU float32 because
    ``SparseSwiGLUNeuronEditor.selected_activations`` applies the same float32
    conversion during an ordinary model forward.
    """
    if int(batch_size) <= 0:
        raise ValueError("batch_size must be positive")
    rows: List[torch.Tensor] = []
    captured: List[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, inputs: Any) -> None:
        captured.append(inputs[0])

    handle = mlp.register_forward_pre_hook(pre_hook)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("detector cache requires a separate model backbone")
        for start in range(0, len(prompts), int(batch_size)):
            batch = list(prompts[start : start + int(batch_size)])
            encoded = tok(batch, padding=True, return_tensors="pt").to(device)
            captured.clear()
            backbone(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError(
                    f"expected one MLP input, captured {len(captured)}"
                )
            hidden = captured[0]
            positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_rows = torch.arange(len(batch), device=device)
            rows.append(hidden[batch_rows, positions, :].detach().float().cpu())
    finally:
        handle.remove()
        tok.padding_side = old_side
    if not rows:
        return torch.empty((0, int(mlp.gate_proj.weight.shape[1])))
    return torch.cat(rows, dim=0)


def capture_grouped_mlp_input_hidden_states(
    model: torch.nn.Module,
    tok: Any,
    mlp: torch.nn.Module,
    prompt_groups: Sequence[Sequence[str]],
    device: torch.device,
    *,
    batch_size: int,
) -> List[torch.Tensor]:
    """Capture a flattened prompt bank once and restore record grouping."""
    sizes = [len(group) for group in prompt_groups]
    prompts = [str(prompt) for group in prompt_groups for prompt in group]
    hidden = capture_mlp_input_last_token_hidden_states(
        model, tok, mlp, prompts, device, batch_size=int(batch_size)
    )
    if int(hidden.shape[0]) != sum(sizes):
        raise RuntimeError("detector cache row count does not match prompt bank")
    result: List[torch.Tensor] = []
    start = 0
    for size in sizes:
        result.append(hidden[start : start + size])
        start += size
    return result


def capture_editor_last_token_activations(
    model: torch.nn.Module,
    tok: Any,
    editor: neuron_core.SparseSwiGLUNeuronEditor,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    editor.capture_activations = True
    try:
        encoded = tok(list(prompts), padding=True, return_tensors="pt").to(device)
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("detector training requires a separate model backbone")
        backbone(**encoded, use_cache=False, return_dict=True)
        activation = editor.last_edited_activations
        if activation is None:
            raise RuntimeError("sparse neuron editor did not capture activations")
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(prompts), device=device)
        return activation[rows, positions, :]
    finally:
        editor.capture_activations = False
        tok.padding_side = old_side


def _negative_prompt_instances(
    records: Sequence[Mapping[str, Any]],
    context_sets: Mapping[int, Mapping[str, Any]],
) -> List[mcf_repair.MCFPromptInstance]:
    instances: List[mcf_repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        case_id = int(record["case_id"])
        negatives = context_sets[case_id]["negative_contexts"]
        for prompt_index, row in enumerate(negatives):
            instances.append(
                mcf_repair.MCFPromptInstance(
                    record_index=case_id,
                    sampled_position=position,
                    prompt_type=str(row.get("kind") or "training_safe_negative"),
                    prompt_index=prompt_index,
                    prompt=str(row["prompt"]),
                    target_new=str(record["reference"]),
                    target_true=str(record["answer"]),
                )
            )
    return instances


def _failure_counts(
    margins: torch.Tensor,
    direct_flags: Sequence[bool],
    margin: float,
) -> Tuple[int, int]:
    threshold = float(margin) - 1e-6
    direct = sum(
        int(bool(direct_flags[index]) and float(value) < threshold)
        for index, value in enumerate(margins)
    )
    return direct, int((margins < threshold).sum())


def _topk_kl(
    logits: torch.Tensor,
    prompts: Sequence[str],
    cache: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    terms: List[torch.Tensor] = []
    for row, prompt in enumerate(prompts):
        ids = cache[str(prompt)]["top_ids"].to(device)
        target = cache[str(prompt)]["top_log_probs"].to(device)
        observed = torch.log_softmax(logits[row].float()[ids], dim=-1)
        terms.append(F.kl_div(observed, target, log_target=True, reduction="sum"))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _replace_embedding_rows(
    layer: torch.nn.Module, token_ids: Sequence[int], values: torch.Tensor
) -> None:
    ids = torch.tensor(
        [int(value) for value in token_ids],
        dtype=torch.long,
        device=layer.weight.device,
    )
    with torch.no_grad():
        layer.weight.index_copy_(
            0, ids, values.to(device=layer.weight.device, dtype=layer.weight.dtype)
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_environment_firewall()
    gagd.set_seed(int(args.seed))
    gagd.require_cuda_if_needed(args.device_map)
    py_rng = random.Random(int(args.seed) + 78103)
    out_dir = gagd.resolve_output_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visible_path = Path(args.training_visible_path).resolve()
    split_path = Path(args.split_manifest).resolve()
    context_path = Path(args.context_manifest).resolve()
    stage1_path = Path(args.stage1_state).resolve()
    stage1_report_path = Path(args.stage1_report).resolve()
    stage1_log_path = Path(args.stage1_writer_log).resolve()
    stage1_gradient_audit_path = stage1_log_path.with_name(
        "stage1_gradient_conflict_audit.json"
    )
    stage1_portability_path = Path(args.clean_stage1_portability_preflight).resolve()
    stage1_acceptance_path = Path(args.clean_stage1_acceptance).resolve()
    registry_path = Path(args.experiment_registry).resolve()
    for path in (
        visible_path,
        split_path,
        context_path,
        stage1_path,
        stage1_report_path,
        stage1_log_path,
        stage1_gradient_audit_path,
        stage1_portability_path,
        stage1_acceptance_path,
        registry_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    experiment_registry = _load_json(registry_path)
    _validate_experiment_registry(experiment_registry, args)

    locked_records, split_manifest = directional.validate_locked(
        visible_path, split_path, int(args.seed), int(args.forget_num)
    )
    if split_manifest.get("protocol") != locked_split.PROTOCOL:
        raise RuntimeError("neuron suppression requires the locked direct-only split")
    locked_split.assert_direct_only_training_view(locked_records)
    records = _record_views(locked_records)
    context_manifest = _load_json(context_path)
    stage1_state = torch.load(stage1_path, map_location="cpu", weights_only=False)
    if not isinstance(stage1_state, Mapping):
        raise RuntimeError("Stage-1 writer state must be a mapping")
    stage1_report = _load_json(stage1_report_path)
    stage1_portability = _load_json(stage1_portability_path)
    stage1_acceptance = _load_json(stage1_acceptance_path)
    _validate_firewall(context_manifest, stage1_state)
    if compositional_method.sha256_file(context_path) != str(
        stage1_state["context_manifest_sha256"]
    ):
        raise RuntimeError(
            "Stage-1 writer and training-safe context manifest hashes differ"
        )
    clean_writer_receipt = _validate_clean_stage1_lineage(
        context_manifest,
        stage1_state,
        stage1_report,
        context_path,
        stage1_log_path,
    )
    if str(Path(args.model_path).resolve()) != str(
        clean_writer_receipt["base_model_path"]
    ):
        raise RuntimeError("decoder Base-model path differs from Stage-1 lineage")
    visible_sha256 = compositional_method.sha256_file(visible_path)
    split_sha256 = compositional_method.sha256_file(split_path)
    if (
        str(context_manifest.get("source_training_visible_sha256") or "")
        != visible_sha256
    ):
        raise RuntimeError(
            "training-visible file is not the exact file bound into the Stage-1 "
            "context manifest"
        )
    if str(context_manifest.get("source_split_manifest_sha256") or "") != split_sha256:
        raise RuntimeError(
            "split manifest is not the exact manifest bound into the Stage-1 "
            "context manifest"
        )
    case_ids = [int(record["case_id"]) for record in records]
    if [int(value) for value in stage1_state.get("case_ids", [])] != case_ids:
        raise RuntimeError("Stage-1 cases do not match the locked training view")
    stored_preflight_threshold = stage1_state.get("writer_positive_amplitude_threshold")
    if stored_preflight_threshold is not None and not math.isclose(
        float(stored_preflight_threshold),
        float(args.writer_preflight_amplitude_threshold),
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "registered writer preflight threshold differs from the frozen writer"
        )
    if dict(stage1_acceptance.get("training_safe_portability", {})) != dict(
        stage1_portability
    ):
        raise RuntimeError(
            "clean Stage-1 acceptance embeds a different portability preflight"
        )
    clean_acceptance_receipt = _validate_clean_stage1_acceptance(
        stage1_acceptance,
        seed=int(args.seed),
        case_ids=case_ids,
        expected_artifacts={
            "training_visible_sha256": visible_sha256,
            "split_manifest_sha256": split_sha256,
            "context_manifest_sha256": compositional_method.sha256_file(context_path),
            "stage1_state_sha256": compositional_method.sha256_file(stage1_path),
            "stage1_report_sha256": compositional_method.sha256_file(
                stage1_report_path
            ),
            "stage1_writer_log_sha256": compositional_method.sha256_file(
                stage1_log_path
            ),
            "stage1_gradient_conflict_audit_sha256": (
                compositional_method.sha256_file(stage1_gradient_audit_path)
            ),
            "training_safe_portability_sha256": compositional_method.sha256_file(
                stage1_portability_path
            ),
        },
        amplitude_threshold=float(args.writer_preflight_amplitude_threshold),
        minimum_global_fraction=float(args.writer_preflight_min_global_fraction),
        minimum_record_fraction=float(args.writer_preflight_min_record_fraction),
    )
    context_sets = _context_sets_by_case(context_manifest, records)

    firewall_receipt = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_training_firewall",
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "forget_num": len(records),
        "training_visible_path": str(visible_path),
        "training_visible_sha256": visible_sha256,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_sha256,
        "context_manifest_path": str(context_path),
        "context_manifest_sha256": compositional_method.sha256_file(context_path),
        "stage1_state_path": str(stage1_path),
        "stage1_state_sha256": compositional_method.sha256_file(stage1_path),
        "stage1_report_path": str(stage1_report_path),
        "stage1_report_sha256": compositional_method.sha256_file(stage1_report_path),
        "stage1_writer_log_path": str(stage1_log_path),
        "stage1_writer_log_sha256": compositional_method.sha256_file(stage1_log_path),
        "stage1_gradient_conflict_audit_path": str(stage1_gradient_audit_path),
        "stage1_gradient_conflict_audit_sha256": (
            compositional_method.sha256_file(stage1_gradient_audit_path)
        ),
        "clean_stage1_portability_path": str(stage1_portability_path),
        "clean_stage1_portability_sha256": compositional_method.sha256_file(
            stage1_portability_path
        ),
        "clean_stage1_acceptance_path": str(stage1_acceptance_path),
        "clean_stage1_acceptance_sha256": compositional_method.sha256_file(
            stage1_acceptance_path
        ),
        "clean_stage1_writer": clean_writer_receipt,
        "clean_stage1_acceptance": clean_acceptance_receipt,
        "experiment_label": str(args.experiment_label),
        "experiment_registry_path": str(registry_path),
        "experiment_registry_sha256": compositional_method.sha256_file(registry_path),
        "official_evaluation_file_argument_exists": False,
        "forbidden_evaluation_environment_variables_present": [],
        "data_access": dict(context_manifest["data_access"]),
        "evaluation_only_unavailable_during_training": [
            "official paraphrases",
            "official neighborhoods",
            "benchmark retain prompts",
            "official PPL text",
            "aliases",
            "descriptions",
            "adversarial prompts",
            "relearning attacks",
        ],
        "used_for_checkpoint_selection": [
            "locked direct forget prompts",
            "training-safe hand-authored relation-template and corpus-prefix positives",
            "training-visible compositional negatives",
            "disjoint Wikipedia protection prefixes",
        ],
        "ppl_corpus_partition": {
            "official_evaluation_document_interval": [0, 20],
            "training_protection_document_interval": [
                int(args.frequency_doc_start),
                int(args.frequency_doc_start) + int(args.frequency_docs),
            ],
            "overlap": 0,
        },
    }
    gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = canonical.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model has no input embedding layer")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embeddings and LM head remain tied")
    device = gagd.first_device(model)
    llama_like = canonical.is_llama_like(model, tok)
    mlp = _resolve_swiglu_mlp(model, int(args.neuron_layer))
    output_head_digest_before = _tensor_digest(output_layer.weight)

    selected_embedding_rows = [
        int(value) for value in stage1_state.get("selected_embedding_rows", [])
    ]
    embedding_delta = stage1_state.get("embedding_delta")
    if not selected_embedding_rows or not isinstance(embedding_delta, torch.Tensor):
        raise RuntimeError("Stage-1 state lacks sparse embedding rows/delta")
    if embedding_delta.shape != (
        len(selected_embedding_rows),
        int(input_layer.weight.shape[1]),
    ):
        raise RuntimeError("Stage-1 embedding delta has incompatible shape")
    observed_transformer_fingerprint = (
        compositional_method.frozen_transformer_fingerprint(model)
    )
    if not math.isclose(
        observed_transformer_fingerprint,
        float(clean_writer_receipt["base_transformer_fingerprint"]),
        rel_tol=1e-12,
        abs_tol=1e-3,
    ):
        raise RuntimeError("decoder Transformer differs from Stage-1 Base model")
    selected_row_index = torch.tensor(
        selected_embedding_rows,
        dtype=torch.long,
        device=input_layer.weight.device,
    )
    observed_selected_rows_sha256 = compositional_method.tensor_sha256(
        input_layer.weight.index_select(0, selected_row_index)
    )
    if observed_selected_rows_sha256 != str(
        clean_writer_receipt["base_selected_embedding_rows_sha256"]
    ):
        raise RuntimeError(
            "decoder Base embedding rows differ from Stage-1 writer lineage"
        )
    writer_present = str(args.writer_mode) == "embedding_keyed"
    applied_embedding_delta = (
        embedding_delta if writer_present else torch.zeros_like(embedding_delta)
    )
    embedding_writer = neuron_core.ToggleableEmbeddingDelta(
        input_layer, selected_embedding_rows, applied_embedding_delta
    )

    (
        positive_instances,
        _owners,
        direct_flags,
    ) = compositional_method.build_prompt_instances(records, context_sets)
    negative_instances = _negative_prompt_instances(records, context_sets)
    training_prompt_groups: List[List[str]] = []
    positive_prompts_by_record: List[List[str]] = []
    negative_prompts_by_record: List[List[str]] = []
    for record in records:
        context = context_sets[int(record["case_id"])]
        positives = [str(prompt) for prompt in context["positive_prompts"]]
        negatives = [str(row["prompt"]) for row in context["negative_contexts"]]
        positive_prompts_by_record.append(positives)
        negative_prompts_by_record.append(negatives)
        training_prompt_groups.append([*positives, *negatives])

    if writer_present:
        marker_map = stage1_state.get("markers")
        if not isinstance(marker_map, Mapping):
            raise RuntimeError("Stage-1 state lacks marker directions for preflight")
        writer_preflight = measure_training_safe_writer_preflight(
            model,
            tok,
            embedding_writer,
            positive_prompts_by_record,
            marker_map,
            case_ids,
            device,
            batch_size=int(args.cache_batch_size),
            amplitude_threshold=float(args.writer_preflight_amplitude_threshold),
            minimum_global_fraction=float(args.writer_preflight_min_global_fraction),
            minimum_record_fraction=float(args.writer_preflight_min_record_fraction),
        )
    else:
        writer_preflight = {
            "kind": "frozen_writer_training_safe_pre_decoder_preflight",
            "applicable": False,
            "writer_mode": "none",
            "passed": True,
            "official_evaluation_prompts_seen": 0,
        }
    gagd.write_json(out_dir / "writer_preflight_report.json", writer_preflight)
    if not bool(writer_preflight["passed"]):
        raise SystemExit(
            "frozen writer failed its training-safe portability preflight; "
            "decoder construction is refused"
        )

    documents = subject_writer.load_frequency_documents(
        args.wikidata_dir, int(args.frequency_doc_start), int(args.frequency_docs)
    )
    if int(args.frequency_docs) > 0 and not documents:
        raise RuntimeError(
            f"no disjoint protection documents loaded from {args.wikidata_dir!r}"
        )
    corpus_prompts = synthetic.corpus_context_prefixes(
        documents,
        count=int(args.corpus_protection_prompts),
        seed=int(args.seed) + 1907,
    )
    protected_prompts, protected_prompt_report = build_selection_protected_prompts(
        training_prompt_groups,
        corpus_prompts,
        total_limit=int(args.selection_protected_prompts),
        minimum_corpus=int(args.selection_min_corpus_prompts),
    )
    if not protected_prompt_report["corpus_quota_passed"]:
        raise RuntimeError("broad-corpus neuron-selection quota was not met")
    if not protected_prompt_report["full_budget_passed"]:
        raise RuntimeError("neuron-selection protected prompt budget was not filled")
    if not protected_prompt_report["all_training_groups_represented"]:
        raise RuntimeError("neuron-selection bank omitted one or more record groups")

    print(f"\nStage 0: {args.writer_mode} sparse-neuron selection")
    selection_start = time.time()
    embedding_writer.enabled = False
    protected_off = capture_base_last_token_activations(
        model,
        tok,
        mlp,
        protected_prompts,
        device,
        batch_size=int(args.cache_batch_size),
    )
    record_writer_on: List[torch.Tensor] = []
    record_writer_off: List[torch.Tensor] = []
    record_context_negative: List[torch.Tensor] = []
    for position, prompts in enumerate(positive_prompts_by_record):
        selected = prompts[: int(args.selection_positive_contexts)]
        selected_negative = negative_prompts_by_record[position][
            : int(args.selection_negative_contexts)
        ]
        embedding_writer.enabled = False
        off = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected,
            device,
            batch_size=int(args.cache_batch_size),
        )
        embedding_writer.enabled = writer_present
        on = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected,
            device,
            batch_size=int(args.cache_batch_size),
        )
        record_writer_off.append(off)
        record_writer_on.append(on)
        embedding_writer.enabled = False
        context_negative = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected_negative,
            device,
            batch_size=int(args.cache_batch_size),
        )
        record_context_negative.append(context_negative)
        displacement_label = (
            "writer activation displacement"
            if writer_present
            else "base positive/negative mean separation"
        )
        separation = (
            float((on - off).abs().max())
            if writer_present
            else float((on.mean(dim=0) - context_negative.mean(dim=0)).abs().max())
        )
        print(
            f"  case {case_ids[position]}: selection contexts={len(selected)}, "
            f"max {displacement_label}={separation:.4f}"
        )
    ownership, sign_groups, selection_reports = neuron_core.select_record_owned_neurons(
        record_writer_on,
        record_writer_off,
        protected_off,
        neurons_per_record=int(args.neurons_per_record),
        dormant_fraction=float(args.dormant_fraction),
        stability_weight=float(args.selection_stability_weight),
        selection_mode=str(args.selection_mode),
        context_negative_activations=record_context_negative,
        generator=torch.Generator(device="cpu").manual_seed(int(args.seed) + 4109),
    )
    # The actuator is multiplied by the absolute SwiGLU activation at runtime;
    # orient the audit/training statistic by the writer-on (or Base-positive for
    # the no-writer control) activation itself, not by an unavailable paired
    # Base displacement.
    sign_groups = []
    for record_index, neurons in enumerate(ownership):
        index = torch.tensor(neurons, dtype=torch.long)
        means = record_writer_on[record_index].index_select(1, index).mean(dim=0)
        sign_groups.append(
            torch.where(means >= 0, torch.ones_like(means), -torch.ones_like(means))
        )
    selected_neurons, flat_signs_cpu, local_groups = neuron_core.flatten_ownership(
        ownership, sign_groups
    )
    selected_index = torch.tensor(selected_neurons, dtype=torch.long)
    selected_protected = protected_off.index_select(1, selected_index)
    base_down_norms = (
        mlp.down_proj.weight.detach()
        .float()
        .cpu()
        .index_select(1, selected_index)
        .norm(dim=0)
    )
    corpus_prompt_set = set(corpus_prompts)
    corpus_mask = torch.tensor(
        [prompt in corpus_prompt_set for prompt in protected_prompts],
        dtype=torch.bool,
    )
    if int(corpus_mask.sum()) < int(args.selection_min_corpus_prompts):
        raise RuntimeError("selected activation profile lost its corpus quota")
    selected_activation_profile = {
        "all_protected_prompts": activation_tail_profile(
            selected_protected,
            selected_neurons,
            activation_threshold=float(args.detector_off_abs_max),
            down_column_norms=base_down_norms,
        ),
        "broad_corpus_prompts": activation_tail_profile(
            selected_protected[corpus_mask],
            selected_neurons,
            activation_threshold=float(args.detector_off_abs_max),
            down_column_norms=base_down_norms,
        ),
        "interpretation": (
            "Baseline activation and residual-contribution bounds for every "
            "candidate ultimately edited. Quietness on the training prompts alone "
            "is not treated as evidence that a neuron lacks an original function."
        ),
    }
    selection_report = {
        "layer": int(args.neuron_layer),
        "zero_based_layer_index": True,
        "neurons_per_record": int(args.neurons_per_record),
        "selected_neuron_count": len(selected_neurons),
        "intermediate_size": int(mlp.gate_proj.weight.shape[0]),
        "dormant_fraction": float(args.dormant_fraction),
        "writer_mode": str(args.writer_mode),
        "selection_mode": str(args.selection_mode),
        "detector_response_mode": str(args.detector_response_mode),
        "protected_prompt_count": len(protected_prompts),
        "protected_prompt_bank": protected_prompt_report,
        "protected_activation_rms": _distribution(
            [float(value) for value in protected_off.square().mean(dim=0).sqrt()]
        ),
        "selected_baseline_activation_profile": selected_activation_profile,
        "ownership": [
            {"case_id": case_ids[index], **row}
            for index, row in enumerate(selection_reports)
        ],
        "selection_wall_time_seconds": time.time() - selection_start,
    }
    gagd.write_json(out_dir / "neuron_selection_report.json", selection_report)
    print(
        f"  selected {len(selected_neurons)} disjoint existing neurons at layer "
        f"{args.neuron_layer} ({args.neurons_per_record} per record)"
    )

    editor = neuron_core.SparseSwiGLUNeuronEditor(mlp, selected_neurons)
    editor.install(mlp)
    flat_signs = flat_signs_cpu.to(device)
    embedding_writer.enabled = writer_present

    print("\nStage 1: cache frozen layer-input states for detector training")
    editor.write_enabled = False
    editor.gate_delta.requires_grad_(True)
    editor.up_delta.requires_grad_(True)
    editor.down_delta.requires_grad_(False)
    embedding_writer.enabled = writer_present
    positive_hidden_cache = capture_grouped_mlp_input_hidden_states(
        model,
        tok,
        mlp,
        positive_prompts_by_record,
        device,
        batch_size=int(args.cache_batch_size),
    )
    negative_hidden_cache = capture_grouped_mlp_input_hidden_states(
        model,
        tok,
        mlp,
        negative_prompts_by_record,
        device,
        batch_size=int(args.cache_batch_size),
    )
    if writer_present:
        embedding_writer.enabled = False
        writer_off_hidden_cache = capture_grouped_mlp_input_hidden_states(
            model,
            tok,
            mlp,
            positive_prompts_by_record,
            device,
            batch_size=int(args.cache_batch_size),
        )
    else:
        writer_off_hidden_cache = [
            torch.empty((0, int(mlp.gate_proj.weight.shape[1]))) for _ in records
        ]
    embedding_writer.enabled = writer_present
    if any(int(rows.shape[0]) == 0 for rows in positive_hidden_cache):
        raise RuntimeError("every record needs at least one cached positive context")
    if any(int(rows.shape[0]) == 0 for rows in negative_hidden_cache):
        raise RuntimeError("every record needs at least one cached negative context")
    hidden_width = int(mlp.gate_proj.weight.shape[1])
    all_cache_rows = [
        *positive_hidden_cache,
        *negative_hidden_cache,
        *writer_off_hidden_cache,
    ]
    if any(
        rows.ndim != 2
        or int(rows.shape[1]) != hidden_width
        or rows.device.type != "cpu"
        or rows.dtype != torch.float32
        or not bool(torch.isfinite(rows).all())
        for rows in all_cache_rows
    ):
        raise RuntimeError("detector hidden-state cache failed shape/dtype validation")
    detector_cache_report = {
        "schema_version": 1,
        "kind": "training_only_frozen_mlp_input_detector_cache",
        "layer": int(args.neuron_layer),
        "hidden_width": hidden_width,
        "storage": "in_memory_cpu_float32",
        "detector_computation": (
            "cached_h -> selected gate_proj rows -> SiLU -> selected up_proj "
            "rows -> signed record-group activation"
        ),
        "cache_exactness": (
            "the selected MLP down projection is disabled during detector training; "
            "the cached MLP input is independent of learned gate/up rows"
        ),
        "records": len(records),
        "writer_on_positive_contexts": sum(
            int(rows.shape[0]) for rows in positive_hidden_cache
        ),
        "writer_on_negative_contexts": sum(
            int(rows.shape[0]) for rows in negative_hidden_cache
        ),
        "writer_off_positive_contexts": sum(
            int(rows.shape[0]) for rows in writer_off_hidden_cache
        ),
        "positive_context_mode": str(args.detector_positive_contexts),
        "negative_context_mode": str(args.detector_negative_contexts),
        "per_record": [
            {
                "record_index": index,
                "case_id": int(case_ids[index]),
                "writer_on_positive_contexts": int(
                    positive_hidden_cache[index].shape[0]
                ),
                "writer_on_negative_contexts": int(
                    negative_hidden_cache[index].shape[0]
                ),
                "writer_off_positive_contexts": int(
                    writer_off_hidden_cache[index].shape[0]
                ),
            }
            for index in range(len(records))
        ],
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(out_dir / "detector_hidden_cache_report.json", detector_cache_report)
    print(
        "  cached "
        f"{detector_cache_report['writer_on_positive_contexts']} writer-on positives, "
        f"{detector_cache_report['writer_on_negative_contexts']} writer-on negatives, "
        f"and {detector_cache_report['writer_off_positive_contexts']} writer-off "
        "positives"
    )

    def cached_detector_responses(hidden: torch.Tensor) -> torch.Tensor:
        if hidden.ndim != 2 or int(hidden.shape[1]) != hidden_width:
            raise ValueError("cached detector hidden states have the wrong shape")
        edited_activation = editor.edited_selected_activations(
            hidden.to(device=device, non_blocking=True)
        )
        return neuron_core.signed_group_activations(
            edited_activation, local_groups, flat_signs
        )

    print("\nStage 1: train globally balanced sparse contextual-code detector")
    detector_optimizer = torch.optim.AdamW(
        [editor.gate_delta, editor.up_delta],
        lr=float(args.detector_lr),
        weight_decay=0.0,
    )
    detector_log: List[Dict[str, float]] = []
    detector_microbatches = math.ceil(
        len(records) / max(1, int(args.detector_record_batch))
    )
    for step in range(1, int(args.detector_steps) + 1):
        detector_optimizer.zero_grad(set_to_none=True)
        accumulated = {
            "write": 0.0,
            "write_mean": 0.0,
            "write_tail": 0.0,
            "negative": 0.0,
            "negative_mean": 0.0,
            "negative_tail": 0.0,
            "cross": 0.0,
            "cross_mean": 0.0,
            "cross_tail": 0.0,
            "consistency": 0.0,
            "writer_off": 0.0,
            "writer_off_mean": 0.0,
            "writer_off_tail": 0.0,
        }
        for record_start in range(
            0, len(records), int(args.detector_record_batch)
        ):
            record_indices = list(
                range(
                    record_start,
                    min(
                        len(records),
                        record_start + int(args.detector_record_batch),
                    ),
                )
            )
            hidden_rows: List[torch.Tensor] = []
            batch_owners: List[int] = []
            positive_flags: List[bool] = []
            off_hidden_rows: List[torch.Tensor] = []
            off_owners: List[int] = []
            for record_index in record_indices:
                positive_hidden = positive_hidden_cache[record_index]
                negative_hidden = negative_hidden_cache[record_index]
                hidden_rows.extend((positive_hidden, negative_hidden))
                batch_owners.extend(
                    [record_index]
                    * (int(positive_hidden.shape[0]) + int(negative_hidden.shape[0]))
                )
                positive_flags.extend(
                    [True] * int(positive_hidden.shape[0])
                    + [False] * int(negative_hidden.shape[0])
                )
                if writer_present:
                    off_hidden = writer_off_hidden_cache[record_index]
                    off_hidden_rows.append(off_hidden)
                    off_owners.extend([record_index] * int(off_hidden.shape[0]))

            responses = cached_detector_responses(torch.cat(hidden_rows, dim=0))
            owner_tensor = torch.tensor(
                batch_owners, dtype=torch.long, device=device
            )
            positive_tensor = torch.tensor(
                positive_flags, dtype=torch.bool, device=device
            )
            detector_loss, pieces = neuron_core.detector_objective(
                responses,
                owner_tensor,
                positive_tensor,
                positive_floor=float(args.detector_positive_floor),
                off_abs_max=float(args.detector_off_abs_max),
                tail_k=int(args.detector_tail_k),
                negative_weight=float(args.detector_negative_weight),
                cross_weight=float(args.detector_cross_weight),
            )
            writer_off_loss = responses.sum() * 0.0
            writer_off_pieces = {
                "writer_off_mean": writer_off_loss,
                "writer_off_tail": writer_off_loss,
            }
            if writer_present:
                off_response = cached_detector_responses(
                    torch.cat(off_hidden_rows, dim=0)
                )
                off_owner_tensor = torch.tensor(
                    off_owners, dtype=torch.long, device=device
                )
                writer_off_loss, writer_off_pieces = (
                    neuron_core.detector_writer_off_objective(
                        off_response,
                        off_owner_tensor,
                        off_abs_max=float(args.detector_off_abs_max),
                        tail_k=int(args.detector_tail_k),
                    )
                )
            microbatch_total = (
                detector_loss
                + float(args.detector_consistency_weight) * pieces["consistency"]
                + float(args.detector_writer_off_weight) * writer_off_loss
            )
            record_scale = len(record_indices) / len(records)
            (record_scale * microbatch_total).backward()
            for name, value in pieces.items():
                accumulated[name] += record_scale * float(value.detach())
            accumulated["writer_off"] += record_scale * float(
                writer_off_loss.detach()
            )
            for name, value in writer_off_pieces.items():
                accumulated[name] += record_scale * float(value.detach())

        l2 = editor.gate_delta.square().mean() + editor.up_delta.square().mean()
        (float(args.detector_l2) * l2).backward()
        total_value = (
            accumulated["write"]
            + float(args.detector_negative_weight) * accumulated["negative"]
            + float(args.detector_cross_weight) * accumulated["cross"]
            + float(args.detector_consistency_weight) * accumulated["consistency"]
            + float(args.detector_writer_off_weight) * accumulated["writer_off"]
            + float(args.detector_l2) * float(l2.detach())
        )
        if not math.isfinite(total_value):
            raise FloatingPointError(f"non-finite detector loss at step {step}")
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                [editor.gate_delta, editor.up_delta], float(args.grad_clip)
            )
        detector_optimizer.step()
        cap = editor.clamp_relative_(
            detector_cap=float(args.detector_relative_cap),
            actuator_cap=float(args.actuator_relative_cap),
        )
        if step == 1 or step % 25 == 0 or step == int(args.detector_steps):
            row = {
                "step": step,
                "loss": total_value,
                **accumulated,
                "l2": float(l2.detach()),
                "records_per_optimizer_update": len(records),
                "microbatches_per_optimizer_update": detector_microbatches,
                "gate_max_relative_norm": cap["gate_max_relative_norm"],
                "up_max_relative_norm": cap["up_max_relative_norm"],
            }
            detector_log.append(row)
            print(
                f"  step {step:>4}: loss {row['loss']:.4f}, "
                f"write {row['write']:.4f}, off {row['writer_off']:.4f}, "
                f"cross {row['cross']:.4f}"
            )
    del detector_optimizer

    @torch.no_grad()
    def record_cached_detector_responses(
        hidden_groups: Sequence[torch.Tensor],
    ) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for record_index, hidden in enumerate(hidden_groups):
            if int(hidden.shape[0]) == 0:
                result.append(torch.empty(0))
                continue
            response = cached_detector_responses(hidden)
            result.append(response[:, record_index].detach().cpu())
        return result

    positive_detector = record_cached_detector_responses(positive_hidden_cache)
    negative_detector = record_cached_detector_responses(negative_hidden_cache)
    writer_off_detector = (
        record_cached_detector_responses(writer_off_hidden_cache)
        if writer_present
        else [torch.empty(0) for _ in positive_prompts_by_record]
    )
    detector_gate = neuron_core.detector_gate_report(
        positive_detector,
        negative_detector,
        writer_off_detector,
        positive_floor=float(args.detector_positive_floor),
        off_abs_max=float(args.detector_off_abs_max),
        require_writer_off=writer_present,
    )
    detector_gate["kind"] = (
        "training_only_embedding_code_detector_gate"
        if writer_present
        else "training_only_base_context_sparse_mlp_detector_gate"
    )
    detector_gate["writer_mode"] = str(args.writer_mode)
    detector_gate["optimization"] = {
        "revision": "v3.1",
        "update_coverage": "all_records_accumulated",
        "records_per_optimizer_update": len(records),
        "record_microbatch_capacity": int(args.detector_record_batch),
        "microbatches_per_optimizer_update": detector_microbatches,
        "positive_context_mode": str(args.detector_positive_contexts),
        "negative_context_mode": str(args.detector_negative_contexts),
        "tail_k": int(args.detector_tail_k),
        "positive_objective": "mean_plus_worst_k_squared_shortfall",
        "negative_objective": "mean_plus_worst_k_squared_gate_excess",
        "cross_objective": "mean_plus_worst_k_squared_gate_excess",
        "writer_off_objective": "mean_plus_worst_k_squared_gate_excess",
        "gradient_normalization": "equal_record_mean",
        "optimizer_steps": int(args.detector_steps),
        "record_exposures": int(args.detector_steps) * len(records),
        "gradient_clip_frequency": "once_per_optimizer_update",
        "norm_projection_frequency": "once_per_optimizer_update",
        "cached_mlp_inputs": True,
    }
    if len(detector_gate["per_record"]) != len(case_ids):
        raise RuntimeError("detector gate lost the locked record-index binding")
    for row, case_id in zip(detector_gate["per_record"], case_ids):
        row["case_id"] = int(case_id)
    detector_gate["record_index_binding"] = "locked training-visible record order"
    detector_gate["official_evaluation_prompts_seen"] = 0
    gagd.write_json(out_dir / "detector_gate_report.json", detector_gate)
    diagnostic_lines = [
        "record_index\tcase_id\tpositive_min\tnegative_abs_max\t"
        "writer_off_abs_max\tpassed"
    ]
    diagnostic_lines.extend(
        "\t".join(
            (
                str(row["record_index"]),
                str(row["case_id"]),
                f"{float(row['positive_min']):+.8f}",
                f"{float(row['negative_abs_max']):.8f}",
                f"{float(row['writer_off_abs_max']):.8f}",
                str(bool(row["passed"])).lower(),
            )
        )
        for row in detector_gate["per_record"]
    )
    (out_dir / "detector_gate_case_report.tsv").write_text(
        "\n".join(diagnostic_lines) + "\n", encoding="utf-8"
    )
    print(
        f"  detector gate: {detector_gate['passed_records']}/"
        f"{detector_gate['total_records']} records"
    )
    if writer_present and args.gate_policy == "strict" and not detector_gate["passed"]:
        rejection = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_training_only_rejection",
            "stage": "sparse_context_detector",
            "reason": "detector_gate_failed",
            "writer_mode": str(args.writer_mode),
            "detector_gate": detector_gate,
            "actuator_training_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
            "next_action": (
                "Redesign or retrain the frozen writer/detector under a newly "
                "registered training-only configuration; do not tune on official probes."
            ),
        }
        gagd.write_json(out_dir / "training_rejection.json", rejection)
        editor.remove()
        embedding_writer.remove()
        raise SystemExit(
            "embedding-keyed detector failed its locked training-only gate; "
            "actuator training and official evaluation are refused"
        )

    print("\nStage 2: train sparse MLP suppression actuator; LM head frozen")
    # Every Stage-2 preservation target is the writer-only model.  Detector
    # gate/up edits are part of the learned MLP eraser and therefore may not be
    # smuggled into the reference baseline.
    editor.enabled = False
    editor.gate_delta.requires_grad_(False)
    editor.up_delta.requires_grad_(False)
    editor.down_delta.requires_grad_(True)
    embedding_writer.enabled = writer_present
    pre_target_new, pre_target_true = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    pre_negative_new, pre_negative_true = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        negative_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    protected_for_kl = _unique_prompts(
        [
            *[instance.prompt for instance in negative_instances],
            *corpus_prompts,
        ]
    )
    writer_only_cache = compositional_method.cache_prompt_baselines(
        model,
        tok,
        protected_for_kl,
        device,
        batch_size=int(args.cache_batch_size),
        topk=int(args.kl_topk),
    )
    editor.enabled = True
    editor.write_enabled = True

    actuator_optimizer = torch.optim.AdamW(
        [editor.down_delta], lr=float(args.actuator_lr), weight_decay=0.0
    )
    actuator_log: List[Dict[str, float]] = []
    positive_order = list(range(len(positive_instances)))
    negative_order = list(range(len(negative_instances)))
    protected_order = list(range(len(protected_for_kl)))
    for step in range(1, int(args.actuator_steps) + 1):
        positive_indices = py_rng.sample(
            positive_order,
            min(int(args.actuator_batch_size), len(positive_order)),
        )
        negative_indices = py_rng.sample(
            negative_order,
            min(int(args.actuator_batch_size), len(negative_order)),
        )
        protected_indices = py_rng.sample(
            protected_order,
            min(int(args.actuator_protected_batch), len(protected_order)),
        )
        positive_batch = [positive_instances[index] for index in positive_indices]
        negative_batch = [negative_instances[index] for index in negative_indices]
        protected_batch = [protected_for_kl[index] for index in protected_indices]

        actuator_optimizer.zero_grad(set_to_none=True)
        embedding_writer.enabled = writer_present
        current_new, current_true = compositional_method.differentiable_instance_nlls(
            model, tok, positive_batch, device, llama_like=llama_like
        )
        margins = current_true - current_new
        margin_loss = F.relu(float(args.forget_margin) - margins).square().mean()
        reference_loss = compositional_method.reference_nll_regression_penalty(
            current_new,
            pre_target_new[positive_indices].to(device),
            float(args.reference_nll_tolerance),
        )
        negative_new, negative_true = compositional_method.differentiable_instance_nlls(
            model, tok, negative_batch, device, llama_like=llama_like
        )
        negative_preservation = (
            negative_new - pre_negative_new[negative_indices].to(device)
        ).square().mean() + (
            negative_true - pre_negative_true[negative_indices].to(device)
        ).square().mean()
        _hidden, protected_logits = compositional_method.forward_last_hidden_logits(
            model, tok, protected_batch, device
        )
        protected_kl = _topk_kl(
            protected_logits, protected_batch, writer_only_cache, device
        )

        writer_off_loss = margins.sum() * 0.0
        if (
            writer_present
            and int(args.actuator_writer_off_every) > 0
            and step % int(args.actuator_writer_off_every) == 0
        ):
            embedding_writer.enabled = False
            off_new, off_true = compositional_method.differentiable_instance_nlls(
                model, tok, positive_batch, device, llama_like=llama_like
            )
            # With the contextual key removed, the neuron edit must preserve
            # the writer-off behavior rather than becoming a stand-alone MLP edit.
            editor.write_enabled = False
            (
                base_off_new,
                base_off_true,
            ) = compositional_method.differentiable_instance_nlls(
                model, tok, positive_batch, device, llama_like=llama_like
            )
            editor.write_enabled = True
            writer_off_loss = (off_new - base_off_new.detach()).square().mean() + (
                off_true - base_off_true.detach()
            ).square().mean()
            embedding_writer.enabled = writer_present

        l2 = editor.down_delta.square().mean()
        total = (
            float(args.margin_weight) * margin_loss
            + float(args.reference_nll_weight) * reference_loss
            + negative_preservation
            + float(args.protected_kl_weight) * protected_kl
            + float(args.writer_off_nll_weight) * writer_off_loss
            + float(args.actuator_l2) * l2
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite actuator loss at step {step}")
        total.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_([editor.down_delta], float(args.grad_clip))
        actuator_optimizer.step()
        cap = editor.clamp_relative_(
            detector_cap=float(args.detector_relative_cap),
            actuator_cap=float(args.actuator_relative_cap),
        )
        if step == 1 or step % 25 == 0 or step == int(args.actuator_steps):
            row = {
                "step": step,
                "loss": float(total.detach()),
                "margin": float(margin_loss.detach()),
                "reference": float(reference_loss.detach()),
                "negative_preservation": float(negative_preservation.detach()),
                "protected_kl": float(protected_kl.detach()),
                "writer_off": float(writer_off_loss.detach()),
                "down_max_relative_norm": cap["down_max_relative_norm"],
            }
            actuator_log.append(row)
            print(
                f"  step {step:>4}: loss {row['loss']:.4f}, "
                f"margin {row['margin']:.4f}, KL {row['protected_kl']:.4f}, "
                f"writer-off {row['writer_off']:.4f}"
            )
        if int(args.check_every) > 0 and step % int(args.check_every) == 0:
            current_margins = compositional_method.evaluate_instance_margins(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.cache_batch_size),
            )
            direct_failure, positive_failure = _failure_counts(
                current_margins, direct_flags, float(args.forget_margin)
            )
            print(
                f"    training audit: direct fail {direct_failure}, "
                f"all-positive fail {positive_failure}, "
                f"min margin {float(current_margins.min()):+.4f}"
            )
    del actuator_optimizer

    embedding_writer.enabled = writer_present
    editor.enabled = True
    editor.write_enabled = True

    # Training-only causal ablations.  Every configuration uses the same fixed
    # learned tensors; none is selected using official evaluation probes.
    def evaluate_ablation(writer: bool, decoder: bool) -> torch.Tensor:
        embedding_writer.enabled = bool(writer and writer_present)
        editor.enabled = bool(decoder)
        return compositional_method.evaluate_instance_margins(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )

    full_hook_margins = evaluate_ablation(writer_present, True)
    embedding_only_margins = evaluate_ablation(writer_present, False)
    neuron_only_margins = evaluate_ablation(False, True)
    reconstructed_base_margins = evaluate_ablation(False, False)
    embedding_writer.enabled = writer_present
    editor.enabled = True
    full_direct_failures, full_positive_failures = _failure_counts(
        full_hook_margins, direct_flags, float(args.forget_margin)
    )
    embedding_only_direct, embedding_only_positive = _failure_counts(
        embedding_only_margins, direct_flags, float(args.forget_margin)
    )
    neuron_only_direct, neuron_only_positive = _failure_counts(
        neuron_only_margins, direct_flags, float(args.forget_margin)
    )
    base_direct, base_positive = _failure_counts(
        reconstructed_base_margins, direct_flags, float(args.forget_margin)
    )
    causal_ablation = {
        "kind": "locked_training_only_within_checkpoint_component_intervention",
        "interpretation_boundary": (
            "Shows which components the fitted checkpoint relies on; it does not "
            "establish architectural necessity relative to an independently "
            "optimized no-writer sparse-MLP control."
        ),
        "writer_mode": str(args.writer_mode),
        "configurations": {
            "full_embedding_plus_neuron": {
                "direct_failures": full_direct_failures,
                "positive_failures": full_positive_failures,
                "minimum_margin": float(full_hook_margins.min()),
            },
            "embedding_only": {
                "direct_failures": embedding_only_direct,
                "positive_failures": embedding_only_positive,
                "minimum_margin": float(embedding_only_margins.min()),
            },
            "neuron_only": {
                "direct_failures": neuron_only_direct,
                "positive_failures": neuron_only_positive,
                "minimum_margin": float(neuron_only_margins.min()),
            },
            "reconstructed_base": {
                "direct_failures": base_direct,
                "positive_failures": base_positive,
                "minimum_margin": float(reconstructed_base_margins.min()),
            },
        },
        "writer_margin_gain": _distribution(
            [float(value) for value in full_hook_margins - neuron_only_margins]
        ),
        "decoder_margin_gain": _distribution(
            [float(value) for value in full_hook_margins - embedding_only_margins]
        ),
        "writer_necessity_applicable": writer_present,
        "writer_is_necessary": bool(
            writer_present
            and neuron_only_direct
            >= math.ceil(
                float(args.min_writer_necessary_direct_fraction) * len(records)
            )
        ),
        "decoder_is_necessary": bool(
            embedding_only_direct
            >= math.ceil(
                float(args.min_decoder_necessary_direct_fraction) * len(records)
            )
        ),
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(out_dir / "causal_component_ablation.json", causal_ablation)

    print("\nMaterialization: ordinary embedding and SwiGLU weights")
    editor.remove()
    embedding_writer.remove()
    input_index = torch.tensor(
        selected_embedding_rows, dtype=torch.long, device=input_layer.weight.device
    )
    base_input_rows = input_layer.weight.index_select(0, input_index).detach().clone()
    directional.materialize_input_delta(
        input_layer, selected_embedding_rows, applied_embedding_delta
    )
    edited_input_rows = input_layer.weight.index_select(0, input_index).detach().clone()
    base_neuron_weights = neuron_core.SparseNeuronWeights(
        editor.base_gate_rows.detach().to(mlp.gate_proj.weight.dtype),
        editor.base_up_rows.detach().to(mlp.up_proj.weight.dtype),
        editor.base_down_columns.detach().to(mlp.down_proj.weight.dtype),
    )
    edited_neuron_weights = editor.materialize(mlp)
    (
        materialized_target_new,
        materialized_target_true,
    ) = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    materialized_margins = materialized_target_true - materialized_target_new
    materialized_direct_failures, materialized_positive_failures = _failure_counts(
        materialized_margins, direct_flags, float(args.forget_margin)
    )
    hook_materialization_drift = float(
        (materialized_margins - full_hook_margins).abs().max()
    )
    reference_drift = materialized_target_new - pre_target_new
    reference_regression_max = max(0.0, float(reference_drift.max()))
    output_head_digest_after = _tensor_digest(output_layer.weight)
    output_head_unchanged = output_head_digest_before == output_head_digest_after
    if not output_head_unchanged:
        raise RuntimeError("LM head changed despite the architectural invariant")

    norm_report = editor.relative_norm_report()
    model_parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    embedding_edited_scalar_count = (
        int(edited_input_rows.numel()) if writer_present else 0
    )
    neuron_edited_scalar_count = int(
        edited_neuron_weights.gate_rows.numel()
        + edited_neuron_weights.up_rows.numel()
        + edited_neuron_weights.down_columns.numel()
    )
    total_edited_scalar_count = (
        embedding_edited_scalar_count + neuron_edited_scalar_count
    )
    norm_cap_passed = bool(
        norm_report["gate_max_relative_norm"]
        <= float(args.detector_relative_cap) + 1e-6
        and norm_report["up_max_relative_norm"]
        <= float(args.detector_relative_cap) + 1e-6
        and norm_report["down_max_relative_norm"]
        <= float(args.actuator_relative_cap) + 1e-6
    )
    detector_policy_passed = bool(
        detector_gate["passed"] or args.gate_policy == "report"
    )
    writer_requirement_passed = bool(
        not args.require_writer_necessity or causal_ablation["writer_is_necessary"]
    )
    training_passed = bool(
        materialized_direct_failures == 0
        and materialized_positive_failures == 0
        and reference_regression_max <= float(args.reference_nll_tolerance) + 1e-6
        and norm_cap_passed
        and writer_requirement_passed
        and causal_ablation["decoder_is_necessary"]
        and detector_policy_passed
        and output_head_unchanged
        and hook_materialization_drift
        <= float(args.hook_materialization_tolerance) + 1e-6
    )

    checkpoint_path = out_dir / "checkpoint"
    rejected_control_checkpoint_saved = bool(
        args.save_checkpoint
        and args.save_rejected_checkpoint
        and not training_passed
        and not writer_present
    )
    checkpoint_saved = bool(
        args.save_checkpoint and (training_passed or rejected_control_checkpoint_saved)
    )
    if checkpoint_saved:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tok.save_pretrained(checkpoint_path)

    actual_input_delta = (
        edited_input_rows.detach().float().cpu()
        - base_input_rows.detach().float().cpu()
    )
    state_path = out_dir / "embedding_keyed_neuron_state.pt"
    torch.save(
        {
            "schema_version": 3,
            "method": METHOD,
            "protocol": PROTOCOL,
            "seed": int(args.seed),
            "experiment_label": str(args.experiment_label),
            "writer_mode": str(args.writer_mode),
            "case_ids": case_ids,
            "layer": int(args.neuron_layer),
            "selected_embedding_rows": selected_embedding_rows,
            "base_selected_embedding_rows": base_input_rows.detach().cpu(),
            "edited_selected_embedding_rows": edited_input_rows.detach().cpu(),
            "actual_embedding_delta": actual_input_delta,
            "selected_neurons": selected_neurons,
            "ownership": ownership,
            "flat_signs": flat_signs_cpu,
            "base_neuron_weights": {
                "gate_rows": base_neuron_weights.gate_rows.detach().cpu(),
                "up_rows": base_neuron_weights.up_rows.detach().cpu(),
                "down_columns": base_neuron_weights.down_columns.detach().cpu(),
            },
            "edited_neuron_weights": {
                "gate_rows": edited_neuron_weights.gate_rows.detach().cpu(),
                "up_rows": edited_neuron_weights.up_rows.detach().cpu(),
                "down_columns": edited_neuron_weights.down_columns.detach().cpu(),
            },
            "gate_delta": editor.gate_delta.detach().cpu(),
            "up_delta": editor.up_delta.detach().cpu(),
            "down_delta": editor.down_delta.detach().cpu(),
            "detector_positive_floor": float(args.detector_positive_floor),
            "detector_off_abs_max": float(args.detector_off_abs_max),
            "detector_response_mode": str(args.detector_response_mode),
            "detector_training_revision": "v3.1",
            "detector_tail_k": int(args.detector_tail_k),
            "detector_update_coverage": "all_records_accumulated",
            "writer_preflight_amplitude_threshold": float(
                args.writer_preflight_amplitude_threshold
            ),
            "writer_configuration": {
                "row_norm_cap": (
                    float(stage1_state["row_norm_cap"])
                    if stage1_state.get("row_norm_cap") is not None
                    else None
                ),
                "row_norm_cap_frequency_alpha": (
                    float(stage1_state["row_norm_cap_frequency_alpha"])
                    if stage1_state.get("row_norm_cap_frequency_alpha") is not None
                    else None
                ),
                "max_subject_token_frequency": (
                    int(stage1_state["max_subject_token_frequency"])
                    if stage1_state.get("max_subject_token_frequency") is not None
                    else None
                ),
            },
            "source_stage1_state_sha256": compositional_method.sha256_file(stage1_path),
            "detector_relative_cap": float(args.detector_relative_cap),
            "actuator_relative_cap": float(args.actuator_relative_cap),
            "hook_materialization_tolerance": float(
                args.hook_materialization_tolerance
            ),
            "context_manifest_sha256": compositional_method.sha256_file(context_path),
            "training_visible_sha256": compositional_method.sha256_file(visible_path),
            "output_head_sha256": output_head_digest_after,
        },
        state_path,
    )

    summary = {
        "schema_version": 3,
        "method": (
            METHOD
            if writer_present
            else "Matched Retrained Sparse-MLP Conditional Suppression Control"
        ),
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "experiment_label": str(args.experiment_label),
        "writer_mode": str(args.writer_mode),
        "forget_num": len(records),
        "model_path": str(Path(args.model_path).resolve()),
        "architecture": {
            "tokenizer_expanded": False,
            "new_token_added": False,
            "runtime_string_matcher": False,
            "external_router": False,
            "retrieval_cache": False,
            "sidecar": False,
            "adapter": False,
            "lora": False,
            "logit_bias": False,
            "lm_head_edited": False,
            "lm_head_sha256_before": output_head_digest_before,
            "lm_head_sha256_after": output_head_digest_after,
            "input_embedding_rows_loaded_from_frozen_writer": len(
                selected_embedding_rows
            ),
            "input_embedding_rows_edited": (
                len(selected_embedding_rows) if writer_present else 0
            ),
            "mlp_layer": int(args.neuron_layer),
            "selected_existing_mlp_neurons": len(selected_neurons),
            "neurons_per_record": int(args.neurons_per_record),
            "base_model_parameter_count": model_parameter_count,
            "embedding_edited_scalar_count": embedding_edited_scalar_count,
            "mlp_edited_scalar_count": neuron_edited_scalar_count,
            "total_edited_scalar_count": total_edited_scalar_count,
            "edited_parameter_fraction": (
                total_edited_scalar_count / max(1, model_parameter_count)
            ),
            "edited_mlp_tensors": [
                f"model.layers.{args.neuron_layer}.mlp.gate_proj.weight[selected rows]",
                f"model.layers.{args.neuron_layer}.mlp.up_proj.weight[selected rows]",
                f"model.layers.{args.neuron_layer}.mlp.down_proj.weight[selected columns]",
            ],
            "writer_mode": str(args.writer_mode),
            "writer": (
                "ordinary global sparse subject embedding-row deltas"
                if writer_present
                else "none; selected embedding rows remain bit-identical to Base"
            ),
            "context_composer": "frozen Transformer layers before the selected MLP",
            "detector": "record-owned sparse existing SwiGLU gate/up rows",
            "actuator": "matching sparse existing SwiGLU down columns",
            "output_projection": "unchanged base LM head",
        },
        "optimization_budget": {
            "detector_response_mode": str(args.detector_response_mode),
            "dormant_fraction": float(args.dormant_fraction),
            "selection_stability_weight": float(args.selection_stability_weight),
            "selection_positive_contexts": int(args.selection_positive_contexts),
            "selection_negative_contexts": int(args.selection_negative_contexts),
            "detector_steps": int(args.detector_steps),
            "detector_lr": float(args.detector_lr),
            "detector_record_batch": int(args.detector_record_batch),
            "detector_record_batch_semantics": (
                "gradient_accumulation_microbatch_capacity"
            ),
            "detector_update_coverage": "all_records_accumulated",
            "detector_records_per_optimizer_update": len(records),
            "detector_microbatches_per_optimizer_update": detector_microbatches,
            "detector_record_exposures": int(args.detector_steps) * len(records),
            "detector_positive_contexts": str(args.detector_positive_contexts),
            "detector_negative_contexts": str(args.detector_negative_contexts),
            "detector_tail_k": int(args.detector_tail_k),
            "detector_positive_objective": (
                "mean_plus_worst_k_squared_shortfall"
            ),
            "detector_negative_objective": (
                "mean_plus_worst_k_squared_gate_excess"
            ),
            "detector_cross_objective": (
                "mean_plus_worst_k_squared_gate_excess"
            ),
            "detector_writer_off_objective": (
                "mean_plus_worst_k_squared_gate_excess"
            ),
            "detector_gradient_normalization": "equal_record_mean",
            "detector_cached_mlp_inputs": True,
            "detector_positive_floor": float(args.detector_positive_floor),
            "detector_off_abs_max": float(args.detector_off_abs_max),
            "detector_negative_weight": float(args.detector_negative_weight),
            "detector_cross_weight": float(args.detector_cross_weight),
            "detector_consistency_weight": float(args.detector_consistency_weight),
            "detector_l2": float(args.detector_l2),
            "detector_relative_cap": float(args.detector_relative_cap),
            "actuator_steps": int(args.actuator_steps),
            "actuator_lr": float(args.actuator_lr),
            "actuator_batch_size": int(args.actuator_batch_size),
            "actuator_protected_batch": int(args.actuator_protected_batch),
            "actuator_writer_off_every": int(args.actuator_writer_off_every),
            "actuator_relative_cap": float(args.actuator_relative_cap),
            "actuator_l2": float(args.actuator_l2),
            "neurons_per_record": int(args.neurons_per_record),
            "selected_existing_mlp_neurons": len(selected_neurons),
            "mlp_layer": int(args.neuron_layer),
            "protected_prompt_count": len(protected_prompts),
            "protected_kl_weight": float(args.protected_kl_weight),
            "margin_weight": float(args.margin_weight),
            "reference_nll_weight": float(args.reference_nll_weight),
            "reference_nll_tolerance": float(args.reference_nll_tolerance),
            "forget_margin": float(args.forget_margin),
            "grad_clip": float(args.grad_clip),
            "kl_topk": int(args.kl_topk),
        },
        "data_firewall": firewall_receipt,
        "writer_preflight": writer_preflight,
        "writer_configuration": {
            "row_norm_cap": (
                float(stage1_state["row_norm_cap"])
                if stage1_state.get("row_norm_cap") is not None
                else None
            ),
            "row_norm_cap_frequency_alpha": (
                float(stage1_state["row_norm_cap_frequency_alpha"])
                if stage1_state.get("row_norm_cap_frequency_alpha") is not None
                else None
            ),
            "max_subject_token_frequency": (
                int(stage1_state["max_subject_token_frequency"])
                if stage1_state.get("max_subject_token_frequency") is not None
                else None
            ),
            "source_stage1_state_sha256": compositional_method.sha256_file(stage1_path),
        },
        "selection": selection_report,
        "detector": {
            "response_mode": str(args.detector_response_mode),
            "training_revision": "v3.1",
            "hidden_cache": detector_cache_report,
            "training_log": detector_log,
            "gate": detector_gate,
            "relative_norm_cap": float(args.detector_relative_cap),
        },
        "actuator": {
            "training_log": actuator_log,
            "relative_norm_cap": float(args.actuator_relative_cap),
            "norms": norm_report,
            "forget_margin": float(args.forget_margin),
            "reference_nll_tolerance": float(args.reference_nll_tolerance),
            "reference_nll_drift": _distribution(
                [float(value) for value in reference_drift]
            ),
            "reference_nll_regression_max": reference_regression_max,
            "hook_to_materialized_margin_abs_max": hook_materialization_drift,
            "hook_materialization_tolerance": float(
                args.hook_materialization_tolerance
            ),
            "direct_failures": materialized_direct_failures,
            "training_safe_positive_failures": materialized_positive_failures,
            "minimum_margin": float(materialized_margins.min()),
        },
        "causal_component_ablation": causal_ablation,
        "acceptance": {
            "writer_preflight_passed": bool(writer_preflight["passed"]),
            "zero_direct_failures": materialized_direct_failures == 0,
            "zero_training_safe_positive_failures": materialized_positive_failures == 0,
            "reference_nll_regression_within_tolerance": bool(
                reference_regression_max <= float(args.reference_nll_tolerance) + 1e-6
            ),
            "detector_gate_passed": bool(detector_gate["passed"]),
            "detector_gate_policy_passed": detector_policy_passed,
            "writer_necessary": bool(causal_ablation["writer_is_necessary"]),
            "writer_necessity_required": bool(args.require_writer_necessity),
            "writer_requirement_passed": writer_requirement_passed,
            "decoder_necessary": bool(causal_ablation["decoder_is_necessary"]),
            "hard_relative_norm_caps_passed": norm_cap_passed,
            "lm_head_bit_identical": output_head_unchanged,
            "hook_materialization_within_tolerance": bool(
                hook_materialization_drift
                <= float(args.hook_materialization_tolerance) + 1e-6
            ),
            "checkpoint_saved": checkpoint_saved,
            "rejected_control_checkpoint_saved": rejected_control_checkpoint_saved,
            "passed": training_passed,
        },
        "checkpoint": str(checkpoint_path) if checkpoint_saved else None,
        "state": str(state_path),
        "claim_boundary": (
            "This checkpoint is a context-conditional suppression intervention, "
            "not evidence of weight-level knowledge removal. Training establishes "
            "only locked-context fit and, when applicable, within-checkpoint key "
            "dependence. Official paraphrase/locality tails and the independently "
            "retrained no-writer control remain mandatory post-freeze evidence."
        ),
    }
    gagd.write_json(out_dir / "embedding_keyed_neuron_summary.json", summary)

    print("\n" + "=" * 76)
    print(f"  materialized direct failures          : {materialized_direct_failures}")
    print(f"  materialized positive failures        : {materialized_positive_failures}")
    print(
        f"  minimum training margin               : {float(materialized_margins.min()):+.4f}"
    )
    print(f"  maximum reference NLL regression      : {reference_regression_max:.4f}")
    if writer_present:
        print(
            "  within-checkpoint writer-off direct fails: "
            f"{neuron_only_direct}/{len(records)}"
        )
    else:
        print("  embedding writer                       : absent by construction")
    print(
        f"  without neuron decoder direct fails   : {embedding_only_direct}/{len(records)}"
    )
    print(f"  LM head bit-identical                 : {output_head_unchanged}")
    print(f"  selected existing neurons             : {len(selected_neurons)}")
    print("  evaluation probes opened              : 0")
    print("=" * 76)
    if not training_passed:
        if rejected_control_checkpoint_saved:
            print(
                "fixed-budget no-writer control was rejected by training acceptance; "
                "its frozen checkpoint is retained for mandatory post-freeze evaluation"
            )
            print(f"checkpoint: {checkpoint_path}")
            print(f"summary: {out_dir / 'embedding_keyed_neuron_summary.json'}")
            return
        raise SystemExit(
            f"{args.writer_mode} sparse-neuron run failed its locked training-only "
            "acceptance; official evaluation is refused"
        )
    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {out_dir / 'embedding_keyed_neuron_summary.json'}")


if __name__ == "__main__":
    main()
