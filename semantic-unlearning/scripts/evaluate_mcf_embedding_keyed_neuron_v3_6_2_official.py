#!/usr/bin/env python3
"""One-shot official MCF evaluation of the frozen V3.6.2 candidate.

This process is intentionally separate from every learner.  It validates the
exact frozen checkpoint and its complete training-only lineage before opening
the official MCF file, reconstructs the additive embedding/detector/actuator
runtime without changing Base parameters, and evaluates a fixed set of arms.

There are deliberately no command-line knobs for the seed, split sizes,
dtype, arms, metric thresholds, or checkpoint identity.  Those values are
fixed by the post-training/pre-evaluation protocol artifact.  An output
directory is single-use: a failed or interrupted evaluation is evidence and
must not be replaced by a retry in the same directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import mcf_embedding_keyed_neuron_core as neuron_core
import mcf_embedding_keyed_neuron_erasure as learner
import sure_canonical_core as canonical
from mcf_zero_unlearn_official_eval import (
    evaluate_record_split,
    is_llama_like,
    load_official_eval_records,
    load_official_ppl_text,
    official_perplexity,
)


PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_2_official_v1"
TRAINING_PROTOCOL = "mcf_embedding_keyed_sparse_neuron_suppression_v3_6_2"
WRITER_PROTOCOL = "mcf_context_composed_sparse_embedding_writer_v6_2"
EXPECTED_CANDIDATE_SHA256 = (
    "a54723022361aa3a32f83c493fdc46b4c3b25c837b8f3c6b45e36a935e596ac8"
)
EXPECTED_MCF_SHA256 = "977a6acce4705507b5fd6bfcea8f61cd78f9ed0f9cd9c9a6bcd6c8a3ed61c833"
EXPECTED_CASE_IDS = [
    12639,
    19763,
    11471,
    14617,
    12369,
    18555,
    17802,
    18175,
    16657,
    13877,
    11975,
    18431,
    10902,
    16824,
    17528,
    20390,
    10472,
    17735,
    14801,
    14186,
    20123,
    12112,
    15638,
    10939,
    10803,
    10854,
    19308,
    10588,
    16683,
    13986,
    17353,
    10913,
    19082,
    14070,
    17612,
    18561,
    19496,
    14256,
    16101,
    14220,
    14022,
    17968,
    15185,
    10790,
    17256,
    19554,
    12076,
    13483,
    20749,
    15294,
]
FIXED_ARM_ORDER = [
    "full_candidate",
    "reconstructed_base",
    "writer_only",
    "actuator_without_writer",
]
ARM_SWITCHES = {
    "full_candidate": (True, True),
    "reconstructed_base": (False, False),
    "writer_only": (True, False),
    "actuator_without_writer": (False, True),
}
FIXED_EVALUATION = {
    "seed": 1,
    "forget_num": 50,
    "retain_num": 1000,
    "sample_mode": "official",
    "dtype": "bf16",
    "device_map": "single",
    "include_ppl": True,
    "layer": 27,
    "detector_neurons_per_record": 4,
    "actuator_neurons_per_record": 16,
    "actuator_relative_cap": 1.5,
}
FIXED_THRESHOLDS = {
    "max_forget_eff": 0.0,
    "max_forget_gen": 0.0,
    "min_forget_margin": 0.1,
    "max_abs_forget_spe_delta": 0.2,
    "max_abs_retain_eff_gen_delta": 1.0,
    "max_abs_retain_spe_delta": 0.2,
    "max_abs_ppl_percent_delta": 5.0,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    return hashlib.sha256(tensor.view(torch.uint8).numpy().tobytes()).hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-run-dir", required=True)
    parser.add_argument("--stage1-state", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def validate_protocol(protocol: Mapping[str, Any], *, source_path: Path) -> None:
    require(protocol.get("schema_version") == 1, "official protocol schema mismatch")
    require(protocol.get("protocol") == PROTOCOL, "official protocol name mismatch")
    require(
        protocol.get("status") == "frozen_post_training_pre_official_evaluation",
        "official protocol is not in its frozen pre-evaluation state",
    )
    require(
        protocol.get("training_protocol") == TRAINING_PROTOCOL,
        "official protocol points to a different learner",
    )
    require(
        protocol.get("candidate_checkpoint_sha256") == EXPECTED_CANDIDATE_SHA256,
        "official protocol candidate hash mismatch",
    )
    require(
        protocol.get("official_mcf_sha256") == EXPECTED_MCF_SHA256,
        "official protocol MCF hash mismatch",
    )
    require(
        protocol.get("forget_case_ids") == EXPECTED_CASE_IDS,
        "official protocol forget-case order mismatch",
    )
    require(
        protocol.get("fixed_arm_order") == FIXED_ARM_ORDER,
        "official protocol arm order mismatch",
    )
    require(
        protocol.get("evaluation") == FIXED_EVALUATION,
        "official protocol evaluation configuration mismatch",
    )
    require(
        protocol.get("candidate_acceptance_thresholds") == FIXED_THRESHOLDS,
        "official protocol metric thresholds mismatch",
    )
    policy = protocol.get("one_shot_policy")
    require(isinstance(policy, Mapping), "official protocol lacks one-shot policy")
    required_true = (
        "fresh_output_directory_required",
        "resume_prohibited",
        "retry_prohibited",
        "checkpoint_selection_prohibited",
        "gradient_updates_prohibited",
        "candidate_mutation_prohibited",
        "all_fixed_arms_reported_regardless_of_outcome",
    )
    require(
        all(policy.get(key) is True for key in required_true),
        "official protocol weakens the one-shot firewall",
    )
    source_digest = sha256_file(source_path)
    require(
        protocol.get("evaluator_source_sha256") == source_digest,
        "evaluator source differs from the frozen official protocol",
    )


def _is_matrix(value: Any, rows: int | None = None) -> bool:
    return (
        isinstance(value, torch.Tensor)
        and value.ndim == 2
        and (rows is None or int(value.shape[0]) == int(rows))
        and bool(torch.isfinite(value.float()).all())
    )


def validate_candidate_structure(state: Mapping[str, Any]) -> None:
    require(state.get("schema_version") == 1, "candidate schema mismatch")
    require(
        state.get("kind") == "mcf_embedding_keyed_neuron_v3_6_2_candidate_state",
        "candidate kind mismatch",
    )
    require(state.get("protocol") == TRAINING_PROTOCOL, "candidate protocol mismatch")
    require(state.get("case_ids") == EXPECTED_CASE_IDS, "candidate cases mismatch")
    require(
        state.get("official_evaluation_prompts_seen") == 0,
        "candidate was not frozen before official evaluation",
    )

    embedding_ids = [int(value) for value in state.get("selected_embedding_rows", [])]
    embedding_delta = state.get("embedding_delta")
    require(
        embedding_ids and len(embedding_ids) == len(set(embedding_ids)),
        "invalid embedding-row ownership",
    )
    require(_is_matrix(embedding_delta, len(embedding_ids)), "invalid embedding delta")

    detector_ids = [int(value) for value in state.get("detector_neuron_ids", [])]
    groups = state.get("detector_local_groups")
    signs = state.get("detector_flat_signs")
    detector_gate = state.get("detector_gate_rows")
    detector_up = state.get("detector_up_rows")
    require(len(detector_ids) == 200, "candidate must contain 200 detector neurons")
    require(len(detector_ids) == len(set(detector_ids)), "detector neurons overlap")
    require(isinstance(groups, list) and len(groups) == 50, "invalid detector groups")
    normalized_groups = [[int(value) for value in group] for group in groups]
    require(
        all(len(group) == 4 for group in normalized_groups),
        "detector width is not four",
    )
    require(
        sorted(value for group in normalized_groups for value in group)
        == list(range(200)),
        "detector groups do not partition the detector rows",
    )
    require(
        isinstance(signs, torch.Tensor)
        and int(signs.numel()) == 200
        and bool(torch.all(signs.detach().float().abs().eq(1.0))),
        "invalid detector signs",
    )
    require(_is_matrix(detector_gate, 200), "invalid detector gate rows")
    require(_is_matrix(detector_up, 200), "invalid detector up rows")
    require(detector_gate.shape == detector_up.shape, "detector matrices differ")

    actuator_ids = [int(value) for value in state.get("actuator_neuron_ids", [])]
    owners = [int(value) for value in state.get("actuator_owner_indices", [])]
    down_delta = state.get("actuator_down_delta")
    require(state.get("actuator_width") == 16, "candidate actuator width is not 16")
    require(len(actuator_ids) == 800, "candidate must contain 800 actuator neurons")
    require(len(actuator_ids) == len(set(actuator_ids)), "actuator neurons overlap")
    require(
        set(detector_ids).isdisjoint(actuator_ids),
        "detector and actuator banks overlap",
    )
    require(len(owners) == 800, "actuator owners are incomplete")
    require(
        [owners.count(index) for index in range(50)] == [16] * 50,
        "actuator ownership is not 16 per record",
    )
    require(_is_matrix(down_delta), "invalid actuator down delta")
    require(
        tuple(down_delta.shape) == (int(detector_gate.shape[1]), 800),
        "actuator down delta has the wrong shape",
    )
    require(
        math.isclose(float(state.get("actuator_relative_cap")), 1.5, abs_tol=1e-12),
        "candidate actuator cap mismatch",
    )
    require(
        math.isclose(
            float(state.get("threshold_off_boundary")), 0.200001, abs_tol=1e-12
        )
        and math.isclose(
            float(state.get("threshold_on_boundary")), 0.249999, abs_tol=1e-12
        ),
        "candidate threshold boundaries mismatch",
    )
    acceptance = state.get("training_acceptance")
    require(isinstance(acceptance, Mapping), "candidate lacks training acceptance")
    require(acceptance.get("passed") is True, "candidate training acceptance failed")
    require(
        acceptance.get("official_evaluation_prompts_seen") == 0,
        "training acceptance opened official prompts",
    )


def _lineage_paths(training_root: Path, stage1_path: Path) -> Dict[str, Path]:
    method_dir = training_root / "method"
    writer_root = stage1_path.parent.parent
    return {
        "experiment_registry_sha256": training_root
        / "protocol"
        / "ablation_registry_v1.json",
        "training_visible_sha256": writer_root
        / "protocol"
        / "training_visible_target_aware_direct.json",
        "split_manifest_sha256": writer_root / "protocol" / "split_manifest.json",
        "context_manifest_sha256": writer_root / "method" / "context_manifest.json",
        "frozen_v3_5_5_success_import_sha256": method_dir
        / "frozen_v3_5_5_success_import.json",
        "frozen_v3_6_rejection_import_sha256": method_dir
        / "frozen_v3_6_rejection_import.json",
        "frozen_v3_6_1_rejection_import_sha256": method_dir
        / "frozen_v3_6_1_rejection_import.json",
        "nll_numerics_receipt_sha256": method_dir / "v3_6_2_nll_numerics_receipt.json",
        "negative_preservation_precedence_sha256": method_dir
        / "negative_preservation_precedence_report.json",
        "exact_v3_5_4_detector_replay_sha256": method_dir
        / "exact_v3_5_4_detector_replay.json",
        "exact_v3_5_5_width16_selection_replay_sha256": method_dir
        / "exact_v3_5_5_width16_selection_replay.json",
        "actuator_selection_sha256": method_dir
        / "actuator_neuron_selection_report.json",
        "positive_warm_start_sha256": method_dir / "v3_6_2_positive_warm_start.json",
        "full_preservation_training_sha256": method_dir
        / "v3_6_2_full_preservation_training.json",
        "actuator_endpoint_audit_sha256": method_dir
        / "v3_6_2_actuator_endpoint_audit.json",
        "protected_kl_audit_sha256": method_dir / "v3_6_2_protected_kl_audit.json",
        "protected_hard_tail_manifest_sha256": method_dir
        / "v3_6_2_protected_hard_tail_manifest.json",
        "causal_component_audit_sha256": method_dir
        / "v3_6_2_causal_component_audit.json",
    }


def validate_training_lineage(
    state: Mapping[str, Any],
    *,
    training_root: Path,
    stage1_path: Path,
    candidate_path: Path,
) -> Dict[str, Any]:
    method_dir = training_root / "method"
    require(
        candidate_path == method_dir / "v3_6_2_candidate_state.pt",
        "candidate must be the canonical V3.6.2 artifact",
    )
    require(
        sha256_file(candidate_path) == EXPECTED_CANDIDATE_SHA256,
        "candidate checkpoint SHA-256 mismatch",
    )
    require(
        sha256_file(stage1_path) == state.get("source_stage1_state_sha256"),
        "Stage-1 state differs from candidate lineage",
    )
    require(
        Path(str(state.get("source_stage1_state_path"))).resolve() == stage1_path,
        "Stage-1 path differs from candidate lineage",
    )
    stage1 = torch.load(stage1_path, map_location="cpu", weights_only=False)
    require(isinstance(stage1, Mapping), "Stage-1 state is not a mapping")
    require(
        stage1.get("protocol") == WRITER_PROTOCOL, "Stage-1 writer protocol mismatch"
    )
    writer_training_lineage = stage1.get("training_lineage")
    require(
        isinstance(writer_training_lineage, Mapping),
        "Stage-1 state lacks its training lineage",
    )
    base_selected_embedding_rows_sha256 = str(
        writer_training_lineage.get("base_selected_embedding_rows_sha256") or ""
    )
    require(
        len(base_selected_embedding_rows_sha256) == 64,
        "Stage-1 state lacks the Base selected-row hash",
    )
    require(
        [int(value) for value in stage1.get("selected_embedding_rows", [])]
        == [int(value) for value in state["selected_embedding_rows"]],
        "candidate embedding rows differ from Stage 1",
    )
    require(
        isinstance(stage1.get("embedding_delta"), torch.Tensor)
        and torch.equal(
            stage1["embedding_delta"].detach().cpu(),
            state["embedding_delta"].detach().cpu(),
        ),
        "candidate embedding delta differs from Stage 1",
    )

    lineage = state.get("lineage")
    require(isinstance(lineage, Mapping), "candidate lacks lineage hashes")
    lineage_receipts: Dict[str, Any] = {}
    for key, path in _lineage_paths(training_root, stage1_path).items():
        require(path.is_file(), f"lineage artifact is missing: {path}")
        observed = sha256_file(path)
        require(lineage.get(key) == observed, f"lineage hash mismatch: {key}")
        lineage_receipts[key] = {"path": str(path), "sha256": observed}

    full = read_json(method_dir / "v3_6_2_full_preservation_training.json")
    completion = read_json(method_dir / "training_only_v3_6_2_completion.json")
    endpoint = read_json(method_dir / "v3_6_2_actuator_endpoint_audit.json")
    protected = read_json(method_dir / "v3_6_2_protected_kl_audit.json")
    causal = read_json(method_dir / "v3_6_2_causal_component_audit.json")
    firewall = read_json(method_dir / "training_firewall_receipt.json")

    require(
        dict(state["training_acceptance"]) == dict(full),
        "candidate embeds a different full training report",
    )
    require(full.get("passed") is True, "full preservation report did not pass")
    require(
        full.get("complete_training_log") is True, "full training log is incomplete"
    )
    require(
        full.get("official_evaluation_prompts_seen") == 0,
        "full training report opened official prompts",
    )
    final = full.get("final_audit", {})
    require(
        isinstance(final, Mapping) and final.get("passed") is True,
        "final training audit failed",
    )
    require(
        final.get("direct_failures") == 0 and final.get("positive_failures") == 0,
        "candidate has training forget failures",
    )
    require(
        float(final.get("reference_nll_regression_max")) == 0.0
        and float(final.get("negative_nll_abs_max")) == 0.0
        and float(final.get("writer_off_nll_abs_max")) == 0.0,
        "candidate training locality is not exact",
    )
    require(
        full.get("detector_tensors_unchanged") is True,
        "detector tensors changed during actuator training",
    )
    require(
        full.get("lm_head_bit_identical") is True, "LM head changed during training"
    )
    require(full.get("norm_cap_passed") is True, "actuator cap failed")
    require(endpoint.get("complete") is True, "endpoint audit is incomplete")
    require(
        endpoint.get("post_projection_matches_final_fresh") is True,
        "fresh endpoint replay differs",
    )
    require(protected.get("passed") is True, "protected KL audit failed")
    require(
        float(protected.get("mean")) <= 0.05 and float(protected.get("max")) <= 0.5,
        "protected KL limits failed",
    )
    require(causal.get("passed") is True, "training causal audit failed")
    require(
        causal.get("writer_necessary") is True
        and causal.get("actuator_necessary") is True,
        "training causal components are not necessary",
    )
    require(
        causal.get("official_evaluation_prompts_seen") == 0,
        "causal audit opened official prompts",
    )

    require(
        completion.get("full_preservation_passed") is True,
        "completion does not pass preservation",
    )
    require(
        completion.get("candidate_checkpoint_saved") is True,
        "completion did not save the candidate",
    )
    require(
        completion.get("candidate_checkpoint_sha256") == EXPECTED_CANDIDATE_SHA256,
        "completion candidate hash mismatch",
    )
    require(
        completion.get("eligible_for_separate_official_evaluation") is True,
        "candidate is not eligible for official evaluation",
    )
    require(
        completion.get("official_evaluation_allowed_in_this_process") is False,
        "learner improperly allowed official evaluation",
    )
    require(
        completion.get("official_evaluation_prompts_seen") == 0,
        "learner completion opened official prompts",
    )

    require(
        firewall.get("protocol") == TRAINING_PROTOCOL,
        "training firewall protocol mismatch",
    )
    require(
        firewall.get("seed") == 1 and firewall.get("forget_num") == 50,
        "training firewall split mismatch",
    )
    require(
        firewall.get("official_evaluation_file_argument_exists") is False,
        "learner accepted an official path",
    )
    require(
        firewall.get("forbidden_evaluation_environment_variables_present") == [],
        "learner saw official environment variables",
    )
    require(
        firewall.get("stage1_state_sha256") == sha256_file(stage1_path),
        "firewall Stage-1 hash mismatch",
    )

    return {
        "candidate": {"path": str(candidate_path), "sha256": EXPECTED_CANDIDATE_SHA256},
        "stage1": {"path": str(stage1_path), "sha256": sha256_file(stage1_path)},
        "lineage": lineage_receipts,
        "training_acceptance": {
            "passed": True,
            "direct_failures": 0,
            "positive_failures": 0,
            "reference_nll_regression_max": 0.0,
            "negative_nll_abs_max": 0.0,
            "writer_off_nll_abs_max": 0.0,
            "protected_kl_mean": float(protected["mean"]),
            "protected_kl_p99": float(protected["p99"]),
            "protected_kl_max": float(protected["max"]),
            "detector_tensors_unchanged": True,
            "lm_head_bit_identical": True,
            "causal_component_audit_passed": True,
            "norm_cap_passed": True,
            "official_evaluation_prompts_seen": 0,
        },
        "base_selected_embedding_rows_sha256": (base_selected_embedding_rows_sha256),
    }


def parameter_versions(model: torch.nn.Module) -> Dict[str, int]:
    return {
        name: int(parameter._version) for name, parameter in model.named_parameters()
    }


def parameter_versions_unchanged(
    before: Mapping[str, int], after: Mapping[str, int]
) -> Dict[str, Any]:
    changed = sorted(
        name for name in set(before) | set(after) if before.get(name) != after.get(name)
    )
    return {"passed": not changed, "changed_parameters": changed}


def _load_runtime(
    *,
    model_path: Path,
    state: Mapping[str, Any],
    expected_base_embedding_rows_sha256: str,
) -> tuple[
    torch.nn.Module,
    Any,
    neuron_core.ToggleableEmbeddingDelta,
    neuron_core.SparseThresholdGatedActuatorBank,
    Dict[str, Any],
]:
    require(
        torch.cuda.is_available(),
        "CUDA is required for the fixed single-device evaluator",
    )
    require(
        Path(str(state.get("model_path"))).resolve() == model_path,
        "model path differs from the frozen candidate",
    )

    config = AutoConfig.from_pretrained(str(model_path))
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        config=config,
        dtype=torch.bfloat16,
    ).to("cuda")
    tied_before_read_only_clone = (
        model.get_input_embeddings().weight.data_ptr()
        == model.get_output_embeddings().weight.data_ptr()
    )
    canonical.untie_and_freeze_output_head(model)
    model.config.use_cache = False
    require(
        not any(parameter.requires_grad for parameter in model.parameters()),
        "Base model parameters require gradients",
    )
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    require(
        input_layer is not None and output_layer is not None,
        "model lacks embedding layers",
    )
    require(
        input_layer.weight.data_ptr() != output_layer.weight.data_ptr(),
        "runtime LM head remains tied",
    )
    require(
        learner._tensor_digest(output_layer.weight) == state.get("lm_head_sha256"),
        "Base LM-head hash mismatch",
    )
    selected_embedding_index = torch.tensor(
        [int(value) for value in state["selected_embedding_rows"]],
        dtype=torch.long,
        device=input_layer.weight.device,
    )
    require(
        tensor_sha256(input_layer.weight.index_select(0, selected_embedding_index))
        == expected_base_embedding_rows_sha256,
        "Base selected embedding rows differ from Stage-1 training",
    )

    mlp = learner._resolve_swiglu_mlp(model, int(FIXED_EVALUATION["layer"]))
    writer = neuron_core.ToggleableEmbeddingDelta(
        input_layer,
        state["selected_embedding_rows"],
        state["embedding_delta"],
    )
    bank = neuron_core.SparseThresholdGatedActuatorBank(
        mlp,
        state["actuator_neuron_ids"],
        state["actuator_owner_indices"],
        detector_gate_rows=state["detector_gate_rows"],
        detector_up_rows=state["detector_up_rows"],
        detector_local_groups=state["detector_local_groups"],
        detector_flat_signs=state["detector_flat_signs"],
        off_boundary=float(state["threshold_off_boundary"]),
        on_boundary=float(state["threshold_on_boundary"]),
    )
    require(
        learner._tensor_digest(bank.base_gate_rows)
        == state.get("actuator_base_gate_rows_sha256"),
        "Base actuator gate rows differ from training",
    )
    require(
        learner._tensor_digest(bank.base_up_rows)
        == state.get("actuator_base_up_rows_sha256"),
        "Base actuator up rows differ from training",
    )
    require(
        learner._tensor_digest(bank.base_down_columns)
        == state.get("actuator_base_down_columns_sha256"),
        "Base actuator down columns differ from training",
    )
    with torch.no_grad():
        bank.down_delta.copy_(
            state["actuator_down_delta"].to(
                device=bank.down_delta.device,
                dtype=bank.down_delta.dtype,
            )
        )
    bank.down_delta.requires_grad_(False)
    require(
        float(bank.down_relative_norms().max())
        <= float(FIXED_EVALUATION["actuator_relative_cap"]) + 1e-6,
        "loaded candidate violates its actuator norm cap",
    )
    bank.install(mlp)

    integrity = {
        "model_path": str(model_path),
        "dtype": "bfloat16",
        "device": str(next(model.parameters()).device),
        "model_parameters_require_grad": False,
        "input_output_embeddings_untied": True,
        "lm_head_was_tied_before_numeric_identity_clone": tied_before_read_only_clone,
        "lm_head_numeric_identity_clone_is_not_a_weight_update": True,
        "lm_head_sha256": learner._tensor_digest(output_layer.weight),
        "base_selected_embedding_rows_sha256": expected_base_embedding_rows_sha256,
        "embedding_delta_sha256": tensor_sha256(state["embedding_delta"]),
        "detector_gate_rows_sha256": tensor_sha256(state["detector_gate_rows"]),
        "detector_up_rows_sha256": tensor_sha256(state["detector_up_rows"]),
        "actuator_down_delta_sha256": tensor_sha256(bank.down_delta),
        "actuator_down_max_relative_norm": float(bank.down_relative_norms().max()),
        "optimizer_constructed": False,
        "gradient_updates_performed": 0,
        "base_weights_replaced_or_materialized": False,
    }
    return model, tokenizer, writer, bank, integrity


def create_fresh_output_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(
            f"official output already exists and cannot be resumed or retried: {path}"
        ) from exc


def _case_ids(records: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(record["case_id"]) for record in records]


def _prefix_count(records: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        + len(record.get("paraphrase_prompts", []))
        + len(record.get("neighborhood_prompts", []))
        for record in records
    )


@torch.inference_mode()
def evaluate_fixed_arm(
    *,
    label: str,
    model: torch.nn.Module,
    tokenizer: Any,
    writer: neuron_core.ToggleableEmbeddingDelta,
    bank: neuron_core.SparseThresholdGatedActuatorBank,
    forget_records: Sequence[Mapping[str, Any]],
    retain_records: Sequence[Mapping[str, Any]],
    ppl_text: str,
    model_path: Path,
) -> Dict[str, Any]:
    writer_enabled, actuator_enabled = ARM_SWITCHES[label]
    writer.enabled = writer_enabled
    bank.enabled = actuator_enabled
    bank.write_enabled = actuator_enabled
    model.eval()
    device = next(model.parameters()).device
    llama_like = is_llama_like(model, tokenizer)
    forget, forget_raw = evaluate_record_split(
        model, tokenizer, forget_records, device, llama_like, "forget"
    )
    retain, retain_raw = evaluate_record_split(
        model, tokenizer, retain_records, device, llama_like, "retain"
    )
    ppl = official_perplexity(
        model,
        tokenizer,
        ppl_text,
        device,
        max_input_length=100,
    )
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_v3_6_2_fixed_official_arm",
        "protocol": PROTOCOL,
        "arm": label,
        "switches": {
            "embedding_writer_enabled": writer_enabled,
            "isolated_actuator_enabled": actuator_enabled,
        },
        "method": f"v3.6.2_{label}",
        "model_dir": str(model_path),
        "dataset": "MCF",
        "sample_mode": FIXED_EVALUATION["sample_mode"],
        "seed": FIXED_EVALUATION["seed"],
        "unlearn_num": FIXED_EVALUATION["forget_num"],
        "retain_num": FIXED_EVALUATION["retain_num"],
        "llama_like": llama_like,
        "forget": forget,
        "retain": retain,
        "forget_PPL": ppl,
        "retain_PPL": ppl,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
        "used_for_training_checkpoint_selection_or_retry": False,
    }


def _number(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"official metric is non-finite: {label}")
    return result


def build_candidate_acceptance(
    arms: Mapping[str, Mapping[str, Any]],
    *,
    integrity_passed: bool,
) -> Dict[str, Any]:
    full = arms["full_candidate"]
    base = arms["reconstructed_base"]
    full_forget = full["forget"]
    full_retain = full["retain"]
    base_forget = base["forget"]
    base_retain = base["retain"]

    observed = {
        "forget_eff": _number(full_forget["Eff"], "forget Eff"),
        "forget_gen": _number(full_forget["Gen"], "forget Gen"),
        "forget_minimum_rewrite_paraphrase_margin": _number(
            full_forget["minimum_rewrite_paraphrase_margin"],
            "forget minimum margin",
        ),
        "forget_spe_delta": _number(full_forget["Spe"], "forget Spe")
        - _number(base_forget["Spe"], "Base forget Spe"),
        "retain_eff_delta": _number(full_retain["Eff"], "retain Eff")
        - _number(base_retain["Eff"], "Base retain Eff"),
        "retain_gen_delta": _number(full_retain["Gen"], "retain Gen")
        - _number(base_retain["Gen"], "Base retain Gen"),
        "retain_spe_delta": _number(full_retain["Spe"], "retain Spe")
        - _number(base_retain["Spe"], "Base retain Spe"),
    }
    full_ppl = _number(full["forget_PPL"], "candidate PPL")
    base_ppl = _number(base["forget_PPL"], "Base PPL")
    require(base_ppl > 0.0, "Base PPL must be positive")
    observed["ppl_percent_delta"] = 100.0 * (full_ppl - base_ppl) / base_ppl

    checks = {
        "forget_eff_within_limit": observed["forget_eff"]
        <= FIXED_THRESHOLDS["max_forget_eff"],
        "forget_gen_within_limit": observed["forget_gen"]
        <= FIXED_THRESHOLDS["max_forget_gen"],
        "forget_margin_meets_floor": observed[
            "forget_minimum_rewrite_paraphrase_margin"
        ]
        >= FIXED_THRESHOLDS["min_forget_margin"],
        "forget_specificity_delta_within_limit": abs(observed["forget_spe_delta"])
        <= FIXED_THRESHOLDS["max_abs_forget_spe_delta"],
        "retain_eff_delta_within_limit": abs(observed["retain_eff_delta"])
        <= FIXED_THRESHOLDS["max_abs_retain_eff_gen_delta"],
        "retain_gen_delta_within_limit": abs(observed["retain_gen_delta"])
        <= FIXED_THRESHOLDS["max_abs_retain_eff_gen_delta"],
        "retain_specificity_delta_within_limit": abs(observed["retain_spe_delta"])
        <= FIXED_THRESHOLDS["max_abs_retain_spe_delta"],
        "ppl_percent_delta_within_limit": abs(observed["ppl_percent_delta"])
        <= FIXED_THRESHOLDS["max_abs_ppl_percent_delta"],
        "candidate_and_base_integrity_unchanged": bool(integrity_passed),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_candidate_acceptance",
        "thresholds": dict(FIXED_THRESHOLDS),
        "observed": observed,
        "checks": checks,
        "passed": not failures,
        "failure_reasons": failures,
        "used_for_training_checkpoint_selection_or_retry": False,
    }


def _git_head(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _arm_summary(value: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "switches": dict(value["switches"]),
        "forget": dict(value["forget"]),
        "retain": dict(value["retain"]),
        "PPL": value["forget_PPL"],
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_path = Path(args.model_path).resolve()
    training_root = Path(args.training_run_dir).resolve()
    stage1_path = Path(args.stage1_state).resolve()
    mcf_path = Path(args.mcf_path).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    protocol_path = Path(args.protocol).resolve()
    output_dir = Path(args.output_dir).resolve()
    source_path = Path(__file__).resolve()
    candidate_path = training_root / "method" / "v3_6_2_candidate_state.pt"

    for path in (
        model_path,
        training_root,
        stage1_path,
        mcf_path,
        wikidata_dir,
        protocol_path,
        candidate_path,
    ):
        require(path.exists(), f"required official-evaluation input is missing: {path}")
    require(not output_dir.exists(), f"official output already exists: {output_dir}")

    protocol = read_json(protocol_path)
    validate_protocol(protocol, source_path=source_path)
    require(
        sha256_file(candidate_path) == EXPECTED_CANDIDATE_SHA256,
        "candidate hash failed before deserialization",
    )
    state = torch.load(candidate_path, map_location="cpu", weights_only=False)
    require(isinstance(state, Mapping), "candidate state is not a mapping")
    validate_candidate_structure(state)
    lineage = validate_training_lineage(
        state,
        training_root=training_root,
        stage1_path=stage1_path,
        candidate_path=candidate_path,
    )

    model = None
    writer = None
    bank = None
    official_opened = False
    try:
        model, tokenizer, writer, bank, runtime_integrity = _load_runtime(
            model_path=model_path,
            state=state,
            expected_base_embedding_rows_sha256=str(
                lineage["base_selected_embedding_rows_sha256"]
            ),
        )
        torch.set_grad_enabled(False)
        versions_before = parameter_versions(model)
        lm_head_before = learner._tensor_digest(model.get_output_embeddings().weight)
        candidate_before = sha256_file(candidate_path)
        stage1_before = sha256_file(stage1_path)
        protocol_before = sha256_file(protocol_path)
        actuator_before = tensor_sha256(bank.down_delta)
        detector_gate_before = tensor_sha256(bank.detector_gate_rows)
        detector_up_before = tensor_sha256(bank.detector_up_rows)
        embedding_delta_before = tensor_sha256(writer.delta)

        create_fresh_output_dir(output_dir)
        pre_open = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_pre_open_firewall",
            "protocol": PROTOCOL,
            "created_at_utc": utc_now(),
            "official_evaluation_opened": False,
            "all_training_and_lineage_checks_passed": True,
            "candidate_checkpoint_sha256": candidate_before,
            "candidate_matches_frozen_protocol": candidate_before
            == EXPECTED_CANDIDATE_SHA256,
            "training_acceptance": lineage["training_acceptance"],
            "runtime_integrity": runtime_integrity,
            "fixed_evaluation": dict(FIXED_EVALUATION),
            "fixed_arm_order": list(FIXED_ARM_ORDER),
            "candidate_acceptance_thresholds": dict(FIXED_THRESHOLDS),
            "protocol_path": str(protocol_path),
            "protocol_sha256": protocol_before,
            "evaluator_source_path": str(source_path),
            "evaluator_source_sha256": sha256_file(source_path),
            "git_head": _git_head(source_path.parents[2]),
            "gradient_mode_enabled": torch.is_grad_enabled(),
            "optimizer_constructed": False,
            "gradient_updates_performed": 0,
            "checkpoint_selection_or_retry_allowed": False,
            "resume_allowed": False,
        }
        require(
            pre_open["gradient_mode_enabled"] is False, "gradient mode was not disabled"
        )
        pre_open["all_evaluation_forwards_use_inference_mode"] = True
        write_json(output_dir / "pre_open_firewall_receipt.json", pre_open)

        # Authorize and durably record the one-shot opening before the first
        # byte is read. If hashing or parsing fails, the terminal receipt still
        # truthfully records that official access began and forbids a retry.
        official_opened = True
        write_json(
            output_dir / "official_evaluation_opened.json",
            {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_open_event",
                "protocol": PROTOCOL,
                "opened_at_utc": utc_now(),
                "official_evaluation_opened": True,
                "official_mcf_path": str(mcf_path),
                "official_mcf_sha256": None,
                "expected_official_mcf_sha256": EXPECTED_MCF_SHA256,
                "candidate_checkpoint_sha256": candidate_before,
                "one_shot_evaluation": True,
                "resume_or_retry_allowed": False,
            },
        )
        mcf_sha = sha256_file(mcf_path)
        opened = read_json(output_dir / "official_evaluation_opened.json")
        write_json(
            output_dir / "official_evaluation_opened.json",
            {**opened, "official_mcf_sha256": mcf_sha},
        )
        require(mcf_sha == EXPECTED_MCF_SHA256, "official MCF dataset hash mismatch")
        forget_records, retain_records = load_official_eval_records(
            str(mcf_path),
            int(FIXED_EVALUATION["forget_num"]),
            int(FIXED_EVALUATION["retain_num"]),
            int(FIXED_EVALUATION["seed"]),
            str(FIXED_EVALUATION["sample_mode"]),
        )
        forget_case_ids = _case_ids(forget_records)
        retain_case_ids = _case_ids(retain_records)
        require(
            forget_case_ids == EXPECTED_CASE_IDS,
            "official forget split differs from the frozen candidate",
        )
        require(
            len(retain_case_ids) == 1000 and len(set(retain_case_ids)) == 1000,
            "official retain split is malformed",
        )
        require(
            set(forget_case_ids).isdisjoint(retain_case_ids),
            "official forget and retain splits overlap",
        )
        ppl_text = load_official_ppl_text(wikidata_dir)
        require(
            ppl_text is not None and ppl_text.strip(),
            "fixed official PPL text is unavailable",
        )

        opened = read_json(output_dir / "official_evaluation_opened.json")
        opened = {
            **opened,
            "forget_case_ids": forget_case_ids,
            "forget_case_ids_sha256": json_sha256(forget_case_ids),
            "retain_case_ids_sha256": json_sha256(retain_case_ids),
            "official_forget_records": len(forget_records),
            "official_retain_records": len(retain_records),
            "official_prefix_prompts_per_arm": _prefix_count(forget_records)
            + _prefix_count(retain_records),
            "ppl_text_sha256": sha256_text(ppl_text),
            "ppl_text_characters": len(ppl_text),
        }
        write_json(output_dir / "official_evaluation_opened.json", opened)

        arms: Dict[str, Mapping[str, Any]] = {}
        arm_artifacts: Dict[str, Any] = {}
        for label in FIXED_ARM_ORDER:
            result = evaluate_fixed_arm(
                label=label,
                model=model,
                tokenizer=tokenizer,
                writer=writer,
                bank=bank,
                forget_records=forget_records,
                retain_records=retain_records,
                ppl_text=ppl_text,
                model_path=model_path,
            )
            arm_path = output_dir / "arms" / f"{label}.json"
            write_json(arm_path, result)
            arms[label] = result
            arm_artifacts[label] = {
                "path": str(arm_path),
                "sha256": sha256_file(arm_path),
            }
            print(
                f"{label}: Eff={result['forget']['Eff']:.3f}, "
                f"Gen={result['forget']['Gen']:.3f}, "
                f"forget Spe={result['forget']['Spe']:.3f}, "
                f"retain Spe={result['retain']['Spe']:.3f}, "
                f"PPL={result['forget_PPL']:.4f}"
            )

        writer.enabled = True
        bank.enabled = True
        bank.write_enabled = True
        versions_after = parameter_versions(model)
        version_audit = parameter_versions_unchanged(versions_before, versions_after)
        integrity_checks = {
            "base_parameter_versions_unchanged": version_audit["passed"],
            "lm_head_bit_identical": learner._tensor_digest(
                model.get_output_embeddings().weight
            )
            == lm_head_before,
            "candidate_checkpoint_bit_identical": sha256_file(candidate_path)
            == candidate_before,
            "stage1_state_bit_identical": sha256_file(stage1_path) == stage1_before,
            "official_protocol_bit_identical": sha256_file(protocol_path)
            == protocol_before,
            "embedding_delta_bit_identical": tensor_sha256(writer.delta)
            == embedding_delta_before,
            "detector_gate_rows_bit_identical": tensor_sha256(bank.detector_gate_rows)
            == detector_gate_before,
            "detector_up_rows_bit_identical": tensor_sha256(bank.detector_up_rows)
            == detector_up_before,
            "actuator_down_delta_bit_identical": tensor_sha256(bank.down_delta)
            == actuator_before,
            "all_model_parameters_require_grad_false": not any(
                parameter.requires_grad for parameter in model.parameters()
            ),
            "actuator_parameter_requires_grad_false": not bank.down_delta.requires_grad,
            "optimizer_constructed": False,
            "gradient_updates_performed": 0,
        }
        integrity_passed = (
            all(
                value is True
                for key, value in integrity_checks.items()
                if key not in {"optimizer_constructed", "gradient_updates_performed"}
            )
            and integrity_checks["optimizer_constructed"] is False
            and integrity_checks["gradient_updates_performed"] == 0
        )
        integrity = {
            "passed": integrity_passed,
            "checks": integrity_checks,
            "changed_base_parameter_versions": version_audit["changed_parameters"],
        }
        acceptance = build_candidate_acceptance(
            arms,
            integrity_passed=integrity_passed,
        )
        official_result = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_6_2_one_shot_official_evaluation",
            "protocol": PROTOCOL,
            "completed_at_utc": utc_now(),
            "official_evaluation_opened": True,
            "official_evaluation_completed": True,
            "candidate_checkpoint_sha256": candidate_before,
            "official_mcf_sha256": mcf_sha,
            "ppl_text_sha256": sha256_text(ppl_text),
            "fixed_evaluation": dict(FIXED_EVALUATION),
            "fixed_arm_order": list(FIXED_ARM_ORDER),
            "arm_artifacts": arm_artifacts,
            "arms": {label: _arm_summary(arms[label]) for label in FIXED_ARM_ORDER},
            "candidate_behavioral_acceptance": acceptance,
            "runtime_integrity": integrity,
            "training_acceptance": lineage["training_acceptance"],
            "official_prefix_prompts_per_arm": opened[
                "official_prefix_prompts_per_arm"
            ],
            "official_prefix_prompt_arm_evaluations": opened[
                "official_prefix_prompts_per_arm"
            ]
            * len(FIXED_ARM_ORDER),
            "official_evaluation_processes": 1,
            "used_for_training_gradient_updates": False,
            "used_for_checkpoint_selection_or_early_stopping": False,
            "retry_or_resume_permitted": False,
            "candidate_or_model_artifact_mutation_detected": not integrity_passed,
            "paper_claim_readiness": {
                "candidate_official_acceptance_passed": acceptance["passed"],
                "matched_mlp_only_control_evaluated": False,
                "post_freeze_retain_tail_audit_completed": False,
                "latent_recovery_and_relearning_completed": False,
                "strong_unlearning_claim_ready": False,
                "interpretation": (
                    "This run evaluates the frozen primary candidate and fixed "
                    "within-checkpoint components. Matched independently trained "
                    "controls and recovery/relearning endpoints remain separate, "
                    "mandatory reported experiments."
                ),
            },
        }
        result_path = output_dir / "official_evaluation.json"
        write_json(result_path, official_result)
        manifest_paths = [
            output_dir / "pre_open_firewall_receipt.json",
            output_dir / "official_evaluation_opened.json",
            *[Path(value["path"]) for value in arm_artifacts.values()],
            result_path,
        ]
        manifest = {
            "schema_version": 1,
            "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_artifact_manifest",
            "protocol": PROTOCOL,
            "artifacts": [
                {"path": str(path), "sha256": sha256_file(path)}
                for path in manifest_paths
            ],
        }
        manifest_path = output_dir / "artifact_manifest.json"
        write_json(manifest_path, manifest)
        write_json(
            output_dir / "terminal_status.json",
            {
                "schema_version": 1,
                "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_terminal_status",
                "protocol": PROTOCOL,
                "status": "completed",
                "official_evaluation_opened": True,
                "official_evaluation_completed": True,
                "candidate_behavioral_acceptance_passed": acceptance["passed"],
                "official_evaluation_sha256": sha256_file(result_path),
                "artifact_manifest_sha256": sha256_file(manifest_path),
                "retry_or_resume_permitted": False,
            },
        )
        print(
            json.dumps(
                {
                    "candidate_behavioral_acceptance_passed": acceptance["passed"],
                    "failure_reasons": acceptance["failure_reasons"],
                    "candidate_checkpoint_sha256": candidate_before,
                    "official_evaluation": str(result_path),
                    "official_evaluation_sha256": sha256_file(result_path),
                    "used_for_checkpoint_selection_or_retry": False,
                },
                indent=2,
            )
        )
    except BaseException as exc:
        if output_dir.exists():
            write_json(
                output_dir / "terminal_status.json",
                {
                    "schema_version": 1,
                    "kind": "mcf_embedding_keyed_neuron_v3_6_2_official_terminal_status",
                    "protocol": PROTOCOL,
                    "status": "failed_after_official_open"
                    if official_opened
                    else "failed_before_official_open",
                    "official_evaluation_opened": official_opened,
                    "official_evaluation_completed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "partial_results_preserved": True,
                    "retry_or_resume_permitted": False,
                },
            )
        raise
    finally:
        if bank is not None:
            bank.remove()
        if writer is not None:
            writer.remove()


if __name__ == "__main__":
    main()
