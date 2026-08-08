#!/usr/bin/env python3
"""RWKU target-only utility-controlled Setting 5e method extension.

This is a new RWKU adapter, not unchanged Setting 5e.  It trains only declared
subject input rows and safe sensitive-answer output rows, selects a checkpoint
using generated facts plus disjoint target-independent protection data, and
then runs protected row-wise LM-head repair.  Official RWKU records remain
locked until an immutable checkpoint receipt crosses the one-way evaluation
boundary.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import rwku_experiment as legacy
from build_rwku_entity_facts import official_locked_descriptor
from build_rwku_matched_protection import build_matched_protection
from rwku_artifact_access import (
    ArtifactAccessError,
    TARGET_ONLY_PROTOCOL_LABEL,
    make_artifact,
    read_artifact,
    sha256_file,
    sha256_json,
    sha256_path,
    write_artifact,
)
from rwku_checkpoint_receipt import (
    assert_model_modification_allowed,
    create_checkpoint_receipt,
    load_receipt,
    mark_evaluation_complete,
    open_official_evaluation,
    verify_frozen_identities,
)
from rwku_data import ensure_target_data, target_for_seed
from rwku_eval import (
    build_frozen_head_probe,
    evaluate_perplexity,
    evaluate_qa_rows,
    evaluate_rwku,
    final_hidden_states,
    format_qa_prompt,
    generate_completions,
    load_wikidata_text,
    recovery_success,
    score_completions,
)
from rwku_rowwise_active_repair import (
    ACTIVE_SOURCE,
    ROW_SCALE_CANDIDATES,
    apply_rowwise_delta,
    classify_output_row,
    generated_forget_gates_pass,
    protection_gates_pass,
    select_rowwise_scales,
    selected_delta_norm,
    tokenizer_special_ids,
    validate_active_points,
)


SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
METHOD = "Setting 5e-UC + protected row-wise LM-head repair"
PRE_REPAIR_METHOD = "Setting 5e-UC before protected row-wise repair"
PROTOCOL_STATUS = "rwku_target_only_utility_controlled_setting5e_method_extension"
MATCHED_PROTECTION_COVERAGE_POLICY = (
    "allow_unmatched_generated_target_keys_but_audit"
)
STATE_SCHEMA_VERSION = "rwku_setting5e_utility_controlled_state_v1"
CONFIG_SCHEMA_VERSION = "rwku_setting5e_utility_controlled_configuration_v1"
DEFAULT_EXPOSURES = (2, 4, 6, 8, 10, 12, 15, 20)
DEFAULT_INTERPOLATION_SCALES = (
    0.015625,
    0.03125,
    0.0625,
    0.125,
    0.25,
    0.50,
    0.75,
    1.00,
)
MAX_PRE_REPAIR_CANDIDATES = 8
STATE_ORDER = {
    "PREPARED": 0,
    "TRAINING": 1,
    "CANDIDATES_EVALUATED": 2,
    "NO_FEASIBLE_CANDIDATE": 2,
    "CHECKPOINT_FROZEN": 3,
    "OFFICIAL_EVALUATION_OPENED": 4,
    "EVALUATION_COMPLETE": 5,
}
OFFICIAL_PATH_MARKERS = (
    "forget_level1.json",
    "forget_level2.json",
    "forget_level3.json",
    "forget_mia.json",
    "retain_mia.json",
    "neighbor_level1.json",
    "neighbor_level2.json",
    "retain_mmlu.json",
    "retain_bbh.json",
    "truthful.json",
    "triviaqa.json",
    "fluency.json",
    "official_locked_eval.json",
    "official_evaluation.json",
    "paper_rescore",
)


@dataclass(frozen=True)
class TrainingPoint:
    fact_id: str
    view_id: str
    prompt_style: str
    prompt: str
    sensitive_answer: str
    neutral_answer: str
    subject: str
    source_record_sha256: str


@dataclass(frozen=True)
class RowPolicy:
    selected_input_rows: Tuple[int, ...]
    selected_output_rows: Tuple[int, ...]
    input_audit: Tuple[Mapping[str, Any], ...]
    output_audit: Tuple[Mapping[str, Any], ...]
    protected_rows: Tuple[int, ...]
    document_frequency: Mapping[int, float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def strict_json_normalize(
    value: Any, *, path: str = ""
) -> Tuple[Any, List[Dict[str, str]]]:
    """Normalize non-finite scalars and return an RFC-6901 audit."""

    replacements: List[Dict[str, str]] = []

    def child(pointer: str, key: Any) -> str:
        escaped = str(key).replace("~", "~0").replace("/", "~1")
        return f"{pointer}/{escaped}" if pointer else f"/{escaped}"

    def visit(item: Any, pointer: str) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): visit(value, child(pointer, key))
                for key, value in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [
                visit(value, child(pointer, index)) for index, value in enumerate(item)
            ]
        if torch.is_tensor(item):
            if item.numel() != 1:
                raise TypeError(
                    f"Cannot JSON-serialize non-scalar tensor at {pointer or '/'}"
                )
            return visit(item.detach().cpu().item(), pointer)
        if isinstance(item, np.generic):
            return visit(item.item(), pointer)
        if isinstance(item, float) and not math.isfinite(item):
            classification = "nan"
            if math.isinf(item):
                classification = (
                    "positive_infinity" if item > 0 else "negative_infinity"
                )
            replacements.append({"path": pointer, "original": classification})
            return None
        return item

    return visit(value, path), replacements


def run_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / args.experiment_id


def state_path(args: argparse.Namespace) -> Path:
    return run_dir(args) / "experiment_state.json"


def read_state(args: argparse.Namespace) -> Dict[str, Any]:
    path = state_path(args)
    if not path.is_file():
        raise ValueError(f"Missing prepared experiment state: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported utility-controlled RWKU state schema")
    if value.get("experiment_id") != args.experiment_id:
        raise ValueError("Experiment state ID mismatch")
    return dict(value)


def write_state(args: argparse.Namespace, state_name: str, **extra: Any) -> None:
    if state_name not in STATE_ORDER:
        raise ValueError(f"Unknown utility-controlled RWKU state: {state_name}")
    existing: Dict[str, Any] = {}
    if state_path(args).is_file():
        existing = read_state(args)
        previous = str(existing["state"])
        if (
            previous in {"NO_FEASIBLE_CANDIDATE", "EVALUATION_COMPLETE"}
            and state_name != previous
        ):
            raise ValueError(
                f"Terminal RWKU state cannot transition: {previous} -> {state_name}"
            )
        if STATE_ORDER[state_name] < STATE_ORDER[previous]:
            raise ValueError(
                f"Backward RWKU state transition forbidden: {previous} -> {state_name}"
            )
    atomic_json_write(
        state_path(args),
        {
            **existing,
            "schema_version": STATE_SCHEMA_VERSION,
            "experiment_id": args.experiment_id,
            "state": state_name,
            **extra,
        },
    )


def reject_official_or_completed_path(path: Path, *, label: str) -> None:
    lowered = str(Path(path)).casefold()
    if any(marker in lowered for marker in OFFICIAL_PATH_MARKERS):
        raise ArtifactAccessError(
            f"{label} cannot use official/evaluation RWKU path: {path}"
        )
    if "/outputs/" in lowered and any(
        marker in lowered
        for marker in ("results.json", "checkpoint_receipt", "checkpoint/")
    ):
        raise ArtifactAccessError(f"{label} cannot consume a completed output: {path}")


def configuration_payload(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "method": METHOD,
        "protocol_status": PROTOCOL_STATUS,
        "objective": {
            "forget_weight": args.forget_weight,
            "retain_ce_weight": args.retain_ce_weight,
            "retain_kl_weight": args.retain_kl_weight,
            "protected_margin_weight": args.protected_margin_weight,
            "delta_l2_weight": args.delta_l2_weight,
            "forget_margin": args.forget_margin,
            "forget_token_margin": args.forget_token_margin,
            "probability_cap_weight": args.probability_cap_weight,
            "eos_margin_weight": args.eos_margin_weight,
        },
        "row_learning_rates": {
            "subject_input_delta": args.subject_input_delta_lr,
            "sensitive_output_delta": args.sensitive_output_delta_lr,
        },
        "maximum_retain_document_frequency": args.max_retain_document_frequency,
        "teacher_top_k": args.teacher_top_k,
        "requested_exposures_per_fact": list(args.exposures_per_fact),
        "candidate_interpolation_scales": list(args.candidate_scales),
        "rowwise_repair_scales": list(ROW_SCALE_CANDIDATES),
        "fixed_candidate_gates": fixed_gate_manifest(),
        "matched_protection_coverage_policy": MATCHED_PROTECTION_COVERAGE_POLICY,
        "reverse_prompts_enabled": False,
        "optimizer": "AdamW",
        "optimizer_state_scope": "fp32_sparse_selected_row_deltas_only",
        "transformer_frozen": True,
        "row_restore_after_every_step": True,
    }


def fixed_gate_manifest() -> Dict[str, Any]:
    return {
        "direct_generation_recovery": 0.0,
        "cloze_generation_recovery": 0.0,
        "paraphrase_generation_recovery": 0.0,
        "generated_geometric_answer_probability_max": 0.01,
        "active_violation_count": 0,
        "full_retain_probability_ratio_range": [0.995, 1.005],
        "geometric_retain_probability_ratio_range": [0.98, 1.02],
        "mean_retain_kl_max": 0.01,
        "p95_retain_kl_max": 0.05,
        "retain_top1_agreement_min": 0.99,
        "protected_answer_probability_ratio_min": 0.999,
        "protected_selected_row_logit_drift_max": 0.05,
        "protected_top1_changes": 0,
        "proxy_ppl_base_multiplier_max": 1.02,
        "nonselected_rows_equal_base": True,
    }


def validate_mode(args: argparse.Namespace) -> None:
    if bool(args.development) == bool(args.confirmatory):
        raise ValueError("Exactly one of --development or --confirmatory is required")
    if args.seed == 0 and args.confirmatory:
        raise ValueError(
            "Stephen King seed 0 is a development target for this revised method"
        )
    payload = configuration_payload(args)
    if args.confirmatory:
        if args.frozen_development_config is None:
            raise ValueError("Confirmatory mode requires --frozen-development-config")
        with Path(args.frozen_development_config).open("r", encoding="utf-8") as handle:
            frozen = json.load(handle)
        frozen_payload = frozen.get("configuration", frozen)
        if sha256_json(frozen_payload) != sha256_json(payload):
            raise ValueError(
                "Confirmatory configuration differs from frozen development manifest"
            )
    elif args.frozen_development_config is not None:
        raise ValueError(
            "--frozen-development-config is only valid with --confirmatory"
        )


def balanced_candidate_schedule(
    fact_count: int,
    exposures_per_fact: Sequence[int] = DEFAULT_EXPOSURES,
) -> List[Dict[str, Any]]:
    if fact_count <= 0:
        raise ValueError("Candidate schedule requires at least one fact")
    exposures = [int(value) for value in exposures_per_fact]
    if not exposures or exposures != sorted(set(exposures)) or exposures[0] <= 0:
        raise ValueError("Exposure schedule must be positive, unique, and increasing")
    return [
        {
            "requested_exposures_per_fact": exposure,
            "step": fact_count * exposure,
            "per_fact_exposures": [exposure] * fact_count,
            "exposure_imbalance": 0,
        }
        for exposure in exposures
    ]


def balanced_fact_order(fact_ids: Sequence[str], total_steps: int) -> List[str]:
    ordered = sorted({str(value) for value in fact_ids})
    if not ordered:
        raise ValueError("Balanced fact order requires fact IDs")
    return [ordered[index % len(ordered)] for index in range(int(total_steps))]


def interpolate_rows_from_base(
    weight: torch.Tensor,
    base: torch.Tensor,
    trained: torch.Tensor,
    row_ids: Sequence[int],
    scale: float,
) -> None:
    """Materialize ``Base + scale * checkpoint_delta`` non-cumulatively."""

    indices = torch.tensor(
        sorted({int(value) for value in row_ids}),
        dtype=torch.long,
        device=weight.device,
    )
    with torch.no_grad():
        weight.copy_(base.to(weight.device, weight.dtype))
        if indices.numel():
            base_rows = base.to(weight.device, weight.dtype).index_select(0, indices)
            trained_rows = trained.to(weight.device, weight.dtype).index_select(
                0, indices
            )
            weight.index_copy_(
                0, indices, base_rows + float(scale) * (trained_rows - base_rows)
            )


def rank_pre_repair_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    limit: int = MAX_PRE_REPAIR_CANDIDATES,
) -> List[Dict[str, Any]]:
    """Rank utility-safe Setting 5e candidates for active repair."""

    if limit <= 0:
        raise ValueError("Pre-repair candidate limit must be positive")
    safe = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("pre_repair_utility_safe") is True
    ]
    return sorted(
        safe,
        key=lambda candidate: (
            int(candidate["active_violation_count"]),
            float(candidate["generated_geometric_answer_probability"]),
            sum(
                float(candidate[name])
                for name in (
                    "direct_generation_recovery",
                    "cloze_generation_recovery",
                    "paraphrase_generation_recovery",
                )
            ),
            float(candidate["total_selected_row_delta_norm"]),
            int(candidate["checkpoint_step"]),
        ),
    )[: int(limit)]


def select_final_repaired_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    eligible = [
        dict(candidate)
        for candidate in candidates
        if candidate.get("final_eligible") is True
    ]
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda candidate: (
            float(candidate["total_selected_row_delta_norm_including_repair"]),
            int(candidate["checkpoint_step"]),
            float(candidate["interpolation_scale"]),
            float(candidate["rowwise_repair_norm"]),
        ),
    )


def select_eligible_candidate(
    candidates: Sequence[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Compatibility name for strict final, post-repair selection only."""

    return select_final_repaired_candidate(candidates)


def mark_no_feasible_candidate(args: argparse.Namespace) -> None:
    selected_paths = (
        run_dir(args) / "utility_controlled_setting5" / "selected_checkpoint",
        run_dir(args) / "rowwise_repair" / "selected_checkpoint",
        run_dir(args) / "checkpoint_receipt.json",
    )
    existing = [str(path) for path in selected_paths if path.exists()]
    if existing:
        raise RuntimeError(
            "Fail-closed no-feasible transition found a forbidden selected artifact: "
            + ", ".join(existing)
        )
    diagnostics_path = run_dir(args) / "training_diagnostics.json"
    if not diagnostics_path.is_file():
        raise RuntimeError(
            "NO_FEASIBLE_CANDIDATE requires training_diagnostics.json first"
        )
    write_state(
        args,
        "NO_FEASIBLE_CANDIDATE",
        failure_reason="no_repaired_candidate_passed_all_strict_final_gates",
        training_diagnostics_path=str(diagnostics_path.resolve()),
        training_diagnostics_sha256=sha256_file(diagnostics_path),
        selected_checkpoint_created=False,
        checkpoint_receipt_created=False,
    )


def _validate_candidate_metric_source(metrics: Mapping[str, Any]) -> None:
    if metrics.get("calibration_source") != ACTIVE_SOURCE:
        raise ValueError("Candidate gates require target-generated entity-fact views")
    if metrics.get("official_rwku_records_accessed") is not False:
        raise ValueError("Official RWKU artifacts cannot train or select a candidate")


def candidate_gate_report(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Pre-repair gate: protection/utility only; forgetting may remain nonzero."""

    _validate_candidate_metric_source(metrics)
    protection_ok, protection_failed = protection_gates_pass(metrics)
    return {
        "pre_repair_utility_safe": bool(protection_ok),
        "pre_repair_failed_protection_gates": protection_failed,
        "thresholds": fixed_gate_manifest(),
    }


def final_candidate_gate_report(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    """Strict final gate applied only after row-wise active repair."""

    _validate_candidate_metric_source(metrics)
    forget_ok, forget_failed = generated_forget_gates_pass(metrics)
    protection_ok, protection_failed = protection_gates_pass(metrics)
    return {
        "post_repair_generated_forget_gates_passed": bool(forget_ok),
        "post_repair_protection_gates_passed": bool(protection_ok),
        "post_repair_failed_generated_gates": forget_failed,
        "post_repair_failed_protection_gates": protection_failed,
        "final_eligible": bool(forget_ok and protection_ok),
        "thresholds": fixed_gate_manifest(),
    }


def _token_ids(tokenizer: Any, text: str) -> List[int]:
    encoder = getattr(tokenizer, "encode", None)
    if callable(encoder):
        return [int(value) for value in encoder(str(text), add_special_tokens=False)]
    encoded = tokenizer(str(text), add_special_tokens=False)
    return [int(value) for value in encoded["input_ids"]]


def _completion_token_ids(tokenizer: Any, answer: str) -> List[int]:
    """Tokenize an answer exactly as the teacher-forced evaluators do."""

    return _token_ids(tokenizer, " " + str(answer).lstrip())


def _subject_token_ids(tokenizer: Any, subject: str) -> List[int]:
    """Cover both sequence-initial and whitespace-prefixed subject pieces."""

    return sorted(
        {
            *_token_ids(tokenizer, str(subject)),
            *_token_ids(tokenizer, " " + str(subject).lstrip()),
        }
    )


def untie_lm_head_preserve_logits(
    model: nn.Module,
    *,
    sample_input_ids: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """Untie a shared output matrix by exact cloning and attest equal logits."""

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise ValueError("Model must expose input and output embeddings")
    tied = input_layer.weight.data_ptr() == output_layer.weight.data_ptr()
    before = None
    if sample_input_ids is not None:
        with torch.no_grad():
            before = model(input_ids=sample_input_ids).logits.detach().clone()
    if tied:
        replacement = nn.Linear(
            output_layer.weight.shape[1],
            output_layer.weight.shape[0],
            bias=getattr(output_layer, "bias", None) is not None,
            device=output_layer.weight.device,
            dtype=output_layer.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(output_layer.weight)
            if replacement.bias is not None:
                replacement.bias.copy_(output_layer.bias)
        model.set_output_embeddings(replacement)
        if hasattr(model, "config"):
            model.config.tie_word_embeddings = False
    output_after = model.get_output_embeddings()
    if output_after.weight.data_ptr() == input_layer.weight.data_ptr():
        raise RuntimeError("LM head remained tied after explicit clone")
    exact = True
    if before is not None:
        with torch.no_grad():
            after = model(input_ids=sample_input_ids).logits
        exact = torch.equal(before, after)
        if not exact:
            raise RuntimeError("Untying the LM head changed initial logits")
    return {
        "was_tied": tied,
        "explicitly_untied": tied,
        "input_output_storage_distinct": True,
        "initial_logits_bitwise_equal": exact,
    }


def freeze_transformer_parameters(model: nn.Module) -> Dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if len({input_weight.data_ptr(), output_weight.data_ptr()}) != 2:
        raise RuntimeError("Utility-controlled training requires an untied LM head")
    return {
        "transformer_frozen": all(
            not parameter.requires_grad
            for parameter in model.parameters()
        ),
        "base_input_embeddings_frozen": not input_weight.requires_grad,
        "base_output_head_frozen": not output_weight.requires_grad,
        "trainable_parameter_names": trainable,
    }


class SparseFP32RowDeltas(nn.Module):
    """FP32 sparse master deltas applied by hooks over immutable Base weights."""

    def __init__(
        self,
        model: nn.Module,
        selected_input_rows: Sequence[int],
        selected_output_rows: Sequence[int],
    ) -> None:
        super().__init__()
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "input_layer", model.get_input_embeddings())
        object.__setattr__(self, "output_layer", model.get_output_embeddings())
        if self.input_layer is None or self.output_layer is None:
            raise ValueError("Model must expose input and output embeddings")
        if self.input_layer.weight.requires_grad or self.output_layer.weight.requires_grad:
            raise ValueError("Base vocabulary matrices must be frozen")
        self.selected_input_rows = tuple(
            sorted({int(value) for value in selected_input_rows})
        )
        self.selected_output_rows = tuple(
            sorted({int(value) for value in selected_output_rows})
        )
        device = self.input_layer.weight.device
        input_indices = torch.tensor(
            self.selected_input_rows, dtype=torch.long, device=device
        )
        output_indices = torch.tensor(
            self.selected_output_rows, dtype=torch.long, device=device
        )
        self.register_buffer("input_indices", input_indices)
        self.register_buffer("output_indices", output_indices)
        self.register_buffer(
            "input_base_rows_native",
            self.input_layer.weight.detach().index_select(0, input_indices).clone(),
        )
        self.register_buffer(
            "output_base_rows_native",
            self.output_layer.weight.detach().index_select(0, output_indices).clone(),
        )
        self.register_buffer(
            "input_base_rows_fp32", self.input_base_rows_native.float().clone()
        )
        self.register_buffer(
            "output_base_rows_fp32", self.output_base_rows_native.float().clone()
        )
        self.input_delta = nn.Parameter(
            torch.zeros(
                (len(self.selected_input_rows), self.input_layer.weight.shape[1]),
                dtype=torch.float32,
                device=device,
            )
        )
        self.output_delta = nn.Parameter(
            torch.zeros(
                (len(self.selected_output_rows), self.output_layer.weight.shape[1]),
                dtype=torch.float32,
                device=device,
            )
        )
        lookup = torch.full(
            (self.input_layer.weight.shape[0],),
            -1,
            dtype=torch.long,
            device=device,
        )
        if input_indices.numel():
            lookup[input_indices] = torch.arange(
                input_indices.numel(), dtype=torch.long, device=device
            )
        self.register_buffer("input_row_lookup", lookup)
        self.enabled = True
        self._hooks = (
            self.input_layer.register_forward_hook(self._input_hook),
            self.output_layer.register_forward_hook(self._output_hook),
        )

    def _input_hook(
        self, _module: nn.Module, arguments: Tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled or not self.selected_input_rows:
            return output
        if not arguments:
            raise RuntimeError("Embedding hook did not receive input IDs")
        input_ids = arguments[0]
        positions = self.input_row_lookup[input_ids]
        valid = positions >= 0
        safe_positions = positions.clamp_min(0)
        correction = F.embedding(safe_positions, self.input_delta)
        correction = correction * valid.unsqueeze(-1)
        return output + correction.to(dtype=output.dtype)

    def _output_hook(
        self, _module: nn.Module, arguments: Tuple[Any, ...], output: torch.Tensor
    ) -> torch.Tensor:
        if not self.enabled or not self.selected_output_rows:
            return output
        if not arguments:
            raise RuntimeError("LM-head hook did not receive hidden states")
        hidden = arguments[0].float()
        correction = hidden @ self.output_delta.transpose(0, 1)
        result = output.float().clone()
        result[..., self.output_indices] = (
            result[..., self.output_indices] + correction
        )
        return result

    def snapshot(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.input_delta.detach().cpu().clone(),
            self.output_delta.detach().cpu().clone(),
        )

    def restore_base_rows(self) -> None:
        with torch.no_grad():
            if self.input_indices.numel():
                self.input_layer.weight.index_copy_(
                    0, self.input_indices, self.input_base_rows_native
                )
            if self.output_indices.numel():
                self.output_layer.weight.index_copy_(
                    0, self.output_indices, self.output_base_rows_native
                )

    def materialize(
        self,
        input_delta: torch.Tensor,
        output_delta: torch.Tensor,
        scale: float,
    ) -> None:
        """Copy Base + scale * FP32 master delta into selected rows only."""

        self.enabled = False
        with torch.no_grad():
            if self.input_indices.numel():
                values = self.input_base_rows_fp32 + float(scale) * input_delta.to(
                    self.input_base_rows_fp32.device, torch.float32
                )
                self.input_layer.weight.index_copy_(
                    0,
                    self.input_indices,
                    values.to(self.input_layer.weight.dtype),
                )
            if self.output_indices.numel():
                values = self.output_base_rows_fp32 + float(scale) * output_delta.to(
                    self.output_base_rows_fp32.device, torch.float32
                )
                self.output_layer.weight.index_copy_(
                    0,
                    self.output_indices,
                    values.to(self.output_layer.weight.dtype),
                )

    def resume_sparse_training(self) -> None:
        self.restore_base_rows()
        self.enabled = True

    def delta_norms(
        self,
        input_delta: Optional[torch.Tensor] = None,
        output_delta: Optional[torch.Tensor] = None,
        *,
        scale: float = 1.0,
    ) -> Dict[str, float]:
        input_value = self.input_delta if input_delta is None else input_delta
        output_value = self.output_delta if output_delta is None else output_delta
        input_norm = float(input_value.detach().float().norm().cpu()) * float(scale)
        output_norm = float(output_value.detach().float().norm().cpu()) * float(scale)
        return {
            "selected_input_row_delta_norm": input_norm,
            "selected_output_row_delta_norm": output_norm,
            "total_selected_row_delta_norm": math.sqrt(
                input_norm**2 + output_norm**2
            ),
        }

    @staticmethod
    def _nonselected_equal(
        current: torch.Tensor,
        reference: torch.Tensor,
        selected: Sequence[int],
        *,
        chunk_size: int = 2048,
    ) -> bool:
        excluded = set(int(value) for value in selected)
        for start in range(0, current.shape[0], chunk_size):
            stop = min(start + chunk_size, current.shape[0])
            keep = [index for index in range(start, stop) if index not in excluded]
            if keep:
                ids = torch.tensor(keep, dtype=torch.long, device=current.device)
                other = reference.index_select(0, ids.to(reference.device)).to(
                    current.device
                )
                if not torch.equal(current.index_select(0, ids), other):
                    return False
        return True

    def nonselected_equal_base(self, base_model: nn.Module) -> bool:
        return bool(
            self._nonselected_equal(
                self.input_layer.weight.detach(),
                base_model.get_input_embeddings().weight.detach(),
                self.selected_input_rows,
            )
            and self._nonselected_equal(
                self.output_layer.weight.detach(),
                base_model.get_output_embeddings().weight.detach(),
                self.selected_output_rows,
            )
        )

    def close(self) -> None:
        self.resume_sparse_training()
        for hook in self._hooks:
            hook.remove()


def retain_document_frequencies(
    tokenizer: Any, records: Sequence[Mapping[str, Any]]
) -> Dict[int, float]:
    counts: MutableMapping[int, int] = {}
    total = len(records)
    for wrapped in records:
        row = wrapped.get("record", wrapped)
        text = " ".join(
            str(row.get(field, ""))
            for field in (
                "prompt",
                "query",
                "text",
                "answer",
                "target",
                "target_true",
                "target_new",
            )
        )
        for token_id in set(_token_ids(tokenizer, text)):
            counts[token_id] = counts.get(token_id, 0) + 1
    denominator = max(total, 1)
    return {token_id: count / denominator for token_id, count in counts.items()}


def _record_text_answer(wrapped: Mapping[str, Any]) -> Tuple[str, str]:
    row = wrapped.get("record", wrapped)
    prompt = row.get("prompt") or row.get("query") or row.get("text")
    answer: Any = row.get("answer") or row.get("target_true") or row.get("target")
    if isinstance(answer, Mapping):
        answer = answer.get("str")
    if not prompt or not answer:
        raise ValueError("Protection/retain records require prompt and answer")
    return str(prompt), str(answer)


def build_row_policy(
    tokenizer: Any,
    training: Mapping[str, Any],
    protection_records: Sequence[Mapping[str, Any]],
    *,
    maximum_document_frequency: float,
) -> RowPolicy:
    views = list(training["payload"].get("views", []))
    subjects = {
        str(value)
        for view in views
        for value in [view.get("subject"), *view.get("subject_aliases", [])]
        if value
    }
    subjects.update(
        str(value)
        for value in training.get("metadata", {}).get("subject_aliases", [])
        if value
    )
    input_candidates = sorted(
        {token for subject in subjects for token in _subject_token_ids(tokenizer, subject)}
    )
    special = set(tokenizer_special_ids(tokenizer))
    frequencies = retain_document_frequencies(tokenizer, protection_records)
    protected_rows = {
        token_id
        for wrapped in protection_records
        for _, answer in [_record_text_answer(wrapped)]
        for token_id in _completion_token_ids(tokenizer, answer)
    }
    input_audit: List[Mapping[str, Any]] = []
    selected_input: List[int] = []
    for token_id in input_candidates:
        reasons = []
        if token_id in special:
            reasons.append("tokenizer_special_row")
        if token_id in protected_rows:
            reasons.append("matched_protection_overlap")
        if reasons:
            input_audit.append(
                {
                    "token_id": token_id,
                    "decoded_token_piece": tokenizer.decode([int(token_id)]),
                    "included": False,
                    "reasons": reasons,
                }
            )
        else:
            selected_input.append(token_id)
            input_audit.append(
                {
                    "token_id": token_id,
                    "decoded_token_piece": tokenizer.decode([int(token_id)]),
                    "included": True,
                    "reasons": ["declared_subject_or_alias"],
                }
            )

    answer_candidates: MutableMapping[int, set[str]] = {}
    for view in views:
        if view.get("boundary_expanding"):
            raise ValueError(
                "Relation-conditioned reverse views are forbidden in the primary method"
            )
        answer = str(
            view.get("sensitive_answer_alias")
            or view.get("canonical_sensitive_answer")
            or ""
        )
        for token_id in _completion_token_ids(tokenizer, answer):
            answer_candidates.setdefault(token_id, set()).add(answer)
    output_audit: List[Mapping[str, Any]] = []
    selected_output: List[int] = []
    for token_id in sorted(answer_candidates):
        classification = classify_output_row(
            token_id,
            tokenizer,
            protected_row_ids=protected_rows,
            retain_document_frequency=frequencies.get(token_id, 0.0),
            maximum_document_frequency=maximum_document_frequency,
        )
        row = {**asdict(classification), "answers": sorted(answer_candidates[token_id])}
        output_audit.append(row)
        if classification.eligible:
            selected_output.append(token_id)
    return RowPolicy(
        selected_input_rows=tuple(selected_input),
        selected_output_rows=tuple(selected_output),
        input_audit=tuple(input_audit),
        output_audit=tuple(output_audit),
        protected_rows=tuple(sorted(protected_rows)),
        document_frequency=frequencies,
    )


def compile_training_points(
    tokenizer: Any, training: Mapping[str, Any]
) -> List[TrainingPoint]:
    eos = getattr(tokenizer, "eos_token", None)
    if not eos:
        raise ValueError("Runtime tokenizer EOS is required")
    points: List[TrainingPoint] = []
    allowed_styles = {
        "direct question",
        "cloze",
        "deterministic paraphrase",
        "forced-prefix",
    }
    for view in training["payload"].get("views", []):
        if view.get("training_allowed") is not True:
            raise ValueError("Generated calibration contains a non-training view")
        style = str(view.get("prompt_style", ""))
        if style not in allowed_styles:
            raise ValueError(f"Unsupported generated prompt style: {style!r}")
        if view.get("boundary_expanding"):
            raise ValueError("Reverse prompts are disabled in the primary method")
        answer = str(
            view.get("sensitive_answer_alias") or view["canonical_sensitive_answer"]
        )
        row = {
            "query": str(view["query"]),
            "answer": answer,
            "subject": str(view["subject"]),
            "level": "generated",
            "type": style,
        }
        points.append(
            TrainingPoint(
                fact_id=str(view["fact_id"]),
                view_id=str(view["view_id"]),
                prompt_style=style,
                prompt=format_qa_prompt(tokenizer, row),
                sensitive_answer=answer,
                neutral_answer=str(eos),
                subject=str(view["subject"]),
                source_record_sha256=str(
                    view.get("source_record_sha256")
                    or view.get("raw_output_sha256")
                    or sha256_json(view)
                ),
            )
        )
    if not points:
        raise ValueError("Generated training bundle contains no method-visible views")
    return points


def topk_plus_tail_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    top_k: int,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Teacher-to-student KL over teacher top-k plus a normalized tail bucket."""

    teacher = teacher_logits.detach().float()
    student = student_logits.float()
    k = min(max(int(top_k), 1), teacher.shape[-1] - 1)
    teacher_prob = F.softmax(teacher, dim=-1)
    student_prob = F.softmax(student, dim=-1)
    top_prob, top_index = teacher_prob.topk(k, dim=-1)
    student_top = student_prob.gather(-1, top_index)
    teacher_tail = (1.0 - top_prob.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    student_tail = (1.0 - student_top.sum(dim=-1, keepdim=True)).clamp_min(1e-12)
    teacher_bucket = torch.cat([top_prob, teacher_tail], dim=-1).clamp_min(1e-12)
    student_bucket = torch.cat([student_top, student_tail], dim=-1).clamp_min(1e-12)
    per_position = (teacher_bucket * (teacher_bucket.log() - student_bucket.log())).sum(
        dim=-1
    )
    values = per_position[mask.bool()] if mask is not None else per_position.reshape(-1)
    if not values.numel():
        zero = student.sum() * 0.0
        return zero, zero.detach()
    return values.mean(), torch.quantile(values.detach(), 0.95)


def _completion_logprobs(
    model: nn.Module, tokenizer: Any, prompt: str, answer: str
) -> torch.Tensor:
    prompt_ids = _token_ids(tokenizer, prompt)
    answer_ids = _completion_token_ids(tokenizer, answer)
    if not prompt_ids or not answer_ids:
        raise ValueError("Prompt/answer tokenized to an empty sequence")
    device = next(model.parameters()).device
    sequence = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    logits = model(input_ids=sequence).logits.float()[0]
    positions = torch.arange(
        len(prompt_ids) - 1, len(prompt_ids) + len(answer_ids) - 1, device=device
    )
    targets = sequence[0, len(prompt_ids) :]
    return (
        F.log_softmax(logits.index_select(0, positions), dim=-1)
        .gather(1, targets[:, None])
        .squeeze(1)
    )


def forget_margin_loss(
    model: nn.Module,
    tokenizer: Any,
    point: TrainingPoint,
    *,
    margin: float,
) -> torch.Tensor:
    sensitive_nll = -_completion_logprobs(
        model, tokenizer, point.prompt, point.sensitive_answer
    ).mean()
    neutral_nll = -_completion_logprobs(
        model, tokenizer, point.prompt, point.neutral_answer
    ).mean()
    return F.relu(float(margin) + neutral_nll - sensitive_nll)


def token_demotion_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Demote targets below detached-selected strongest competitors."""

    if logits.ndim != 2 or target_ids.ndim != 1:
        raise ValueError("Token demotion expects [positions,vocab] logits and targets")
    if logits.shape[0] != target_ids.shape[0]:
        raise ValueError("Token demotion target count does not match logits")
    detached = logits.detach().clone()
    detached.scatter_(1, target_ids[:, None], -torch.inf)
    competitor_ids = detached.argmax(dim=-1)
    target_logits = logits.gather(1, target_ids[:, None]).squeeze(1)
    competitor_logits = logits.gather(1, competitor_ids[:, None]).squeeze(1)
    return F.relu(float(margin) + target_logits - competitor_logits).mean()


def probability_cap_loss(
    target_mean_logprob: torch.Tensor,
    *,
    probability_cap: float = 0.01,
) -> torch.Tensor:
    if not 0.0 < probability_cap < 1.0:
        raise ValueError("Probability cap must be strictly between zero and one")
    return F.relu(target_mean_logprob - math.log(probability_cap))


def forget_objective_components(
    model: nn.Module,
    tokenizer: Any,
    point: TrainingPoint,
    *,
    forget_token_margin: float,
    probability_cap_weight: float,
    eos_margin_weight: float,
    eos_margin: float,
) -> Dict[str, torch.Tensor]:
    prompt_ids = _token_ids(tokenizer, point.prompt)
    answer_ids = _completion_token_ids(tokenizer, point.sensitive_answer)
    if not prompt_ids or not answer_ids:
        raise ValueError("Forget point prompt/answer tokenized to an empty sequence")
    device = next(model.parameters()).device
    sequence = torch.tensor(
        [prompt_ids + answer_ids], dtype=torch.long, device=device
    )
    logits = model(input_ids=sequence).logits.float()[0]
    positions = torch.arange(
        len(prompt_ids) - 1,
        len(prompt_ids) + len(answer_ids) - 1,
        device=device,
    )
    position_logits = logits.index_select(0, positions)
    targets = sequence[0, len(prompt_ids) :]
    demotion = token_demotion_loss(
        position_logits, targets, margin=forget_token_margin
    )
    target_logprobs = F.log_softmax(position_logits, dim=-1).gather(
        1, targets[:, None]
    ).squeeze(1)
    cap = probability_cap_loss(target_logprobs.mean())
    eos_auxiliary = forget_margin_loss(
        model, tokenizer, point, margin=eos_margin
    )
    total = (
        demotion
        + float(probability_cap_weight) * cap
        + float(eos_margin_weight) * eos_auxiliary
    )
    return {
        "total_forget_loss": total,
        "token_demotion_loss": demotion,
        "probability_cap_loss": cap,
        "eos_auxiliary_margin_loss": eos_auxiliary,
        "target_mean_logprob": target_logprobs.mean(),
    }


def retain_answer_ce(
    model: nn.Module, tokenizer: Any, records: Sequence[Mapping[str, Any]]
) -> torch.Tensor:
    losses = [
        -_completion_logprobs(model, tokenizer, *_record_text_answer(record)).mean()
        for record in records
    ]
    return (
        torch.stack(losses).mean() if losses else next(model.parameters()).sum() * 0.0
    )


def protected_answer_hinge(
    model: nn.Module,
    teacher: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_ratio: float = 0.999,
) -> torch.Tensor:
    losses = []
    for record in records:
        prompt, answer = _record_text_answer(record)
        current = _completion_logprobs(model, tokenizer, prompt, answer).sum()
        with torch.no_grad():
            base = _completion_logprobs(teacher, tokenizer, prompt, answer).sum()
        losses.append(F.relu(base + math.log(minimum_ratio) - current))
    return (
        torch.stack(losses).mean() if losses else next(model.parameters()).sum() * 0.0
    )


def teacher_kl_for_records(
    model: nn.Module,
    teacher: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> Tuple[torch.Tensor, float]:
    means: List[torch.Tensor] = []
    p95s: List[torch.Tensor] = []
    device = next(model.parameters()).device
    for record in records:
        prompt, answer = _record_text_answer(record)
        ids = _token_ids(tokenizer, prompt + " " + answer.lstrip())
        inputs = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            teacher_logits = teacher(input_ids=inputs).logits
        student_logits = model(input_ids=inputs).logits
        mean_kl, p95 = topk_plus_tail_kl(student_logits, teacher_logits, top_k=top_k)
        means.append(mean_kl)
        p95s.append(p95)
    if not means:
        zero = next(model.parameters()).sum() * 0.0
        return zero, 0.0
    return torch.stack(means).mean(), float(torch.stack(p95s).max().cpu())


@torch.no_grad()
def _completion_top1_matches(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    answer: str,
) -> Tuple[int, int]:
    prompt_ids = _token_ids(tokenizer, prompt)
    answer_ids = _completion_token_ids(tokenizer, answer)
    device = next(model.parameters()).device
    sequence = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    logits = model(input_ids=sequence).logits[0]
    positions = torch.arange(
        len(prompt_ids) - 1, len(prompt_ids) + len(answer_ids) - 1, device=device
    )
    predictions = logits.index_select(0, positions).argmax(dim=-1)
    targets = sequence[0, len(prompt_ids) :]
    return int((predictions == targets).sum()), len(answer_ids)


@torch.no_grad()
def _prompt_distribution_metrics(
    model: nn.Module,
    base_model: nn.Module,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    selected_output_rows: Sequence[int],
    top_k: int,
) -> Dict[str, Any]:
    kls: List[float] = []
    agreements: List[float] = []
    drifts: List[float] = []
    device = next(model.parameters()).device
    selected = torch.tensor(
        sorted({int(value) for value in selected_output_rows}),
        dtype=torch.long,
        device=device,
    )
    for record in records:
        prompt, _ = _record_text_answer(record)
        ids = torch.tensor(
            [_token_ids(tokenizer, prompt)], dtype=torch.long, device=device
        )
        current = model(input_ids=ids).logits[:, -1, :]
        base = base_model(input_ids=ids).logits[:, -1, :]
        mean_kl, _ = topk_plus_tail_kl(current, base, top_k=top_k)
        kls.append(float(mean_kl.cpu()))
        agreements.append(float(current.argmax(-1).item() == base.argmax(-1).item()))
        if selected.numel():
            drift = (
                (
                    current.index_select(-1, selected).float()
                    - base.index_select(-1, selected).float()
                )
                .abs()
                .max()
            )
            drifts.append(float(drift.cpu()))
        else:
            drifts.append(0.0)
    return {
        "mean_retain_kl": float(np.mean(kls)) if kls else 0.0,
        "p95_retain_kl": float(np.quantile(kls, 0.95)) if kls else 0.0,
        "retain_top1_agreement": float(np.mean(agreements)) if agreements else 1.0,
        "protected_selected_row_logit_drift": max(drifts, default=0.0),
        "protected_top1_changes": int(sum(value == 0.0 for value in agreements)),
    }


def _safe_exp(value: float) -> float:
    return math.exp(max(-80.0, min(80.0, float(value))))


def evaluate_pre_freeze_candidate(
    model: nn.Module,
    base_model: nn.Module,
    tokenizer: Any,
    points: Sequence[TrainingPoint],
    protection_gate_records: Sequence[Mapping[str, Any]],
    *,
    selected_output_rows: Sequence[int],
    proxy_text: str,
    batch_size: int,
    teacher_top_k: int,
    nonselected_rows_equal_base: bool,
) -> Dict[str, Any]:
    """Evaluate using generated views and disjoint external gates only."""

    prompts = [point.prompt for point in points]
    outputs = generate_completions(
        model,
        tokenizer,
        prompts,
        batch_size=batch_size,
        max_new_tokens=30,
    )
    scores = score_completions(
        model,
        tokenizer,
        [(point.prompt, point.sensitive_answer) for point in points],
        batch_size=batch_size,
    )
    generated_details: List[Dict[str, Any]] = []
    top1_matches = 0
    top1_tokens = 0
    for point, output, score in zip(points, outputs, scores):
        matches, count = _completion_top1_matches(
            model, tokenizer, point.prompt, point.sensitive_answer
        )
        top1_matches += matches
        top1_tokens += count
        generated_details.append(
            {
                "fact_id": point.fact_id,
                "view_id": point.view_id,
                "prompt_style": point.prompt_style,
                "recovery_success": recovery_success(output, point.sensitive_answer),
                "full_sequence_answer_probability": _safe_exp(score.sum_logprob),
                "geometric_answer_token_probability": _safe_exp(score.mean_logprob),
                "first_token_probability": score.first_token_probability,
                "top1_answer_token_matches": matches,
                "answer_token_count": count,
            }
        )

    def recovery_for(style: str) -> float:
        rows = [row for row in generated_details if row["prompt_style"] == style]
        return (
            100.0 * sum(bool(row["recovery_success"]) for row in rows) / len(rows)
            if rows
            else 0.0
        )

    candidate_scores = score_completions(
        model,
        tokenizer,
        [_record_text_answer(record) for record in protection_gate_records],
        batch_size=batch_size,
    )
    base_scores = score_completions(
        base_model,
        tokenizer,
        [_record_text_answer(record) for record in protection_gate_records],
        batch_size=batch_size,
    )
    full_ratios = [
        _safe_exp(current.sum_logprob - base.sum_logprob)
        for current, base in zip(candidate_scores, base_scores)
    ]
    geometric_ratios = [
        _safe_exp(current.mean_logprob - base.mean_logprob)
        for current, base in zip(candidate_scores, base_scores)
    ]
    distribution = _prompt_distribution_metrics(
        model,
        base_model,
        tokenizer,
        protection_gate_records,
        selected_output_rows=selected_output_rows,
        top_k=teacher_top_k,
    )
    base_proxy = evaluate_perplexity(base_model, tokenizer, proxy_text)
    proxy = evaluate_perplexity(model, tokenizer, proxy_text)
    generated_geo = (
        float(
            np.mean(
                [row["geometric_answer_token_probability"] for row in generated_details]
            )
        )
        if generated_details
        else float("inf")
    )
    return {
        "calibration_source": ACTIVE_SOURCE,
        "official_rwku_records_accessed": False,
        "direct_generation_recovery": recovery_for("direct question"),
        "cloze_generation_recovery": recovery_for("cloze"),
        "paraphrase_generation_recovery": recovery_for("deterministic paraphrase"),
        "forced_prefix_recovery": recovery_for("forced-prefix"),
        "generated_full_sequence_answer_probability": float(
            np.mean(
                [row["full_sequence_answer_probability"] for row in generated_details]
            )
        ),
        "generated_geometric_answer_probability": generated_geo,
        "generated_first_token_probability": float(
            np.mean([row["first_token_probability"] for row in generated_details])
        ),
        "top1_answer_token_recovery": top1_matches / max(top1_tokens, 1),
        "active_violation_count": int(top1_matches),
        "full_retain_probability_ratio": float(np.mean(full_ratios))
        if full_ratios
        else 1.0,
        "geometric_retain_probability_ratio": float(np.mean(geometric_ratios))
        if geometric_ratios
        else 1.0,
        "p05_retain_probability_ratio": float(np.quantile(full_ratios, 0.05))
        if full_ratios
        else 1.0,
        "p95_retain_probability_ratio": float(np.quantile(full_ratios, 0.95))
        if full_ratios
        else 1.0,
        "protected_answer_margin_violations": int(
            sum(value < 0.999 for value in full_ratios)
        ),
        "protected_answer_probability_ratio": min(full_ratios, default=1.0),
        "proxy_ppl": proxy,
        "base_proxy_ppl": base_proxy,
        "nonselected_rows_equal_base": bool(nonselected_rows_equal_base),
        "generated_details": generated_details,
        **distribution,
    }


def _load_json_mapping(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


def _artifact_attests_official_access(artifact: Mapping[str, Any]) -> bool:
    return any(
        container.get("official_rwku_records_accessed") is True
        for container in (
            artifact,
            artifact.get("metadata", {}),
            artifact.get("payload", {}),
        )
        if isinstance(container, Mapping)
    )


def summarize_matched_protection_coverage(
    coverage: Sequence[Mapping[str, Any]],
    *,
    minimum_train_per_key: int,
    minimum_gate_per_key: int,
) -> Dict[str, Any]:
    """Validate and summarize the complete audited per-key coverage report."""

    normalized_keys: set[str] = set()
    insufficient_keys: List[str] = []
    covered_count = 0
    for row in coverage:
        normalized_key = str(row.get("normalized_key", ""))
        if not normalized_key:
            raise ValueError("Matched-protection coverage row lacks normalized_key")
        if normalized_key in normalized_keys:
            raise ValueError(
                f"Matched-protection coverage repeats key: {normalized_key}"
            )
        normalized_keys.add(normalized_key)
        try:
            optimization_count = int(row["optimization_count"])
            gate_count = int(row["gate_count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Matched-protection coverage counts are invalid: {normalized_key}"
            ) from error
        if optimization_count < 0 or gate_count < 0:
            raise ValueError(
                f"Matched-protection coverage counts are negative: {normalized_key}"
            )
        expected_status = (
            "covered"
            if optimization_count >= int(minimum_train_per_key)
            and gate_count >= int(minimum_gate_per_key)
            else "insufficient_coverage"
        )
        if row.get("coverage_status") != expected_status:
            raise ValueError(
                "Matched-protection coverage status disagrees with its counts: "
                f"{normalized_key}"
            )
        if expected_status == "covered":
            covered_count += 1
        else:
            insufficient_keys.append(normalized_key)
    return {
        "key_count": len(coverage),
        "covered_key_count": covered_count,
        "insufficient_key_count": len(insufficient_keys),
        "insufficient_keys": sorted(insufficient_keys),
    }


def validate_matched_protection_construction(
    protection_dir: Path,
    *,
    target_subject: str,
    minimum_train_per_key: int,
    minimum_gate_per_key: int,
) -> Dict[str, Any]:
    """Fail closed on unusable protection, not target-key match coverage."""

    directory = Path(protection_dir)
    train_path = directory / "matched_protection_train.json"
    gate_path = directory / "matched_protection_gate.json"
    coverage_path = directory / "matched_protection_coverage.json"
    for path in (train_path, gate_path, coverage_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing matched-protection artifact: {path}")

    train = read_artifact(
        train_path,
        stage="train",
        gradient=True,
        expected_role="optimization_protection",
    )
    gate = read_artifact(
        gate_path,
        stage="train",
        selection=True,
        expected_role="repair_selection_gate",
    )
    coverage_artifact = read_artifact(
        coverage_path,
        stage="prepare",
        expected_role="matched_protection_coverage",
    )
    for artifact in (train, gate, coverage_artifact):
        if _artifact_attests_official_access(artifact):
            raise ValueError(
                "Matched-protection construction accessed official/evaluation data"
            )

    # These checks validate every visible key's origin and every matched
    # record's target independence without imposing a minimum match count.
    legacy._validate_matched_protection_artifact(
        train, target_subject=target_subject
    )
    legacy._validate_matched_protection_artifact(
        gate, target_subject=target_subject
    )

    def record_hashes(artifact: Mapping[str, Any], label: str) -> set[str]:
        values: set[str] = set()
        for record in artifact["payload"].get("records", []):
            digest = str(record.get("content_sha256", ""))
            if not digest:
                raise ValueError(
                    f"{label} matched-protection record lacks content provenance hash"
                )
            values.add(digest)
        return values

    train_hashes = record_hashes(train, "train")
    gate_hashes = record_hashes(gate, "gate")
    overlap = train_hashes & gate_hashes
    if overlap:
        raise ValueError(
            "Matched-protection train/gate content hashes overlap: "
            + ", ".join(sorted(overlap)[:5])
        )

    coverage = coverage_artifact["payload"].get("coverage")
    warnings = coverage_artifact["payload"].get("warnings")
    if not isinstance(coverage, list) or not isinstance(warnings, list):
        raise ValueError("Matched-protection coverage report is malformed")
    summary = summarize_matched_protection_coverage(
        coverage,
        minimum_train_per_key=minimum_train_per_key,
        minimum_gate_per_key=minimum_gate_per_key,
    )
    expected_keys = {
        str(row["normalized_key"])
        for row in train["payload"].get("keys", [])
    }
    gate_keys = {
        str(row["normalized_key"])
        for row in gate["payload"].get("keys", [])
    }
    coverage_keys = {str(row["normalized_key"]) for row in coverage}
    if expected_keys != gate_keys or expected_keys != coverage_keys:
        raise ValueError(
            "Matched-protection coverage does not audit every generated/protected key"
        )
    if len(warnings) < summary["insufficient_key_count"]:
        raise ValueError(
            "Matched-protection coverage warnings omit insufficient keys"
        )
    return {
        "train": train,
        "gate": gate,
        "coverage": coverage_artifact,
        "coverage_summary": summary,
        "train_content_sha256": sorted(train_hashes),
        "gate_content_sha256": sorted(gate_hashes),
    }


def prepare_stage(args: argparse.Namespace) -> None:
    destination = run_dir(args)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(
            f"Refusing to overwrite existing experiment directory: {destination}"
        )
    target = target_for_seed(args.seed)
    for path, label in (
        (args.generated_entity_fact_bundle, "generated training bundle"),
        (args.generator_receipt, "generator receipt"),
        (args.model_path, "Base model"),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        if label != "Base model":
            reject_official_or_completed_path(Path(path), label=label)
    generator = read_artifact(
        args.generator_receipt,
        stage="prepare",
        expected_role="generator_receipt",
    )
    generator_payload = generator["payload"]
    if generator_payload.get("official_rwku_records_accessed") is not False:
        raise ValueError(
            "Generator receipt does not attest official_rwku_records_accessed=false"
        )
    generator_subject = (
        generator_payload.get("target_entity")
        or generator_payload.get("subject")
        or generator.get("metadata", {}).get("subject")
    )
    if generator_subject and generator_subject != target.subject:
        raise ValueError("Generator receipt target differs from seed target")
    configuration = configuration_payload(args)
    config_manifest = {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "configuration": configuration,
        "configuration_sha256": sha256_json(configuration),
        "development_source": bool(args.development),
        "frozen_at_utc": utc_now(),
    }
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json_write(destination / "configuration_manifest.json", config_manifest)
    metadata = {
        "seed": args.seed,
        "subject": target.subject,
        "entity_id": f"rwku:{target.directory}",
        "training_bundle_path": str(args.generated_entity_fact_bundle.resolve()),
        "training_bundle_sha256": sha256_file(args.generated_entity_fact_bundle),
        "generator_receipt_path": str(args.generator_receipt.resolve()),
        "generator_receipt_sha256": sha256_file(args.generator_receipt),
    }
    locked = make_artifact(
        "official_locked_eval",
        official_locked_descriptor(args.seed, include_level12=True),
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=PROTOCOL_STATUS,
        metadata=metadata,
    )
    write_artifact(destination / "official_locked_eval.json", locked)
    write_state(
        args,
        "PREPARED",
        method=METHOD,
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=PROTOCOL_STATUS,
        development=bool(args.development),
        confirmatory=bool(args.confirmatory),
        target={
            "seed": args.seed,
            "subject": target.subject,
            "entity_id": f"rwku:{target.directory}",
        },
        model_path=str(args.model_path.resolve()),
        model_sha256=sha256_path(args.model_path),
        model_revision=args.model_revision,
        dtype=args.dtype,
        prepared_training_bundle_path=metadata["training_bundle_path"],
        prepared_training_bundle_sha256=metadata["training_bundle_sha256"],
        prepared_generator_receipt_path=metadata["generator_receipt_path"],
        prepared_generator_receipt_sha256=metadata["generator_receipt_sha256"],
        configuration_manifest_path=str(
            (destination / "configuration_manifest.json").resolve()
        ),
        configuration_sha256=config_manifest["configuration_sha256"],
        official_evaluation_opened=False,
        official_rwku_records_accessed=False,
    )


def protection_stage(args: argparse.Namespace) -> None:
    state = read_state(args)
    if state.get("state") != "PREPARED":
        raise ValueError(
            f"Protection construction requires PREPARED, got {state.get('state')}"
        )
    if not args.protection_source:
        raise ValueError("Protection stage requires --protection-source")
    if args.mcf_path is None or not args.mcf_path.is_file():
        raise FileNotFoundError("Protection stage requires a local --mcf-path")
    for path in [args.mcf_path, *args.protection_source]:
        reject_official_or_completed_path(path, label="target-independent protection")
    tokenizer = None
    if args.tokenize_protection_rows:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            revision=args.model_revision,
            local_files_only=args.no_download,
        )
    protection_dir = run_dir(args) / "protection"
    result = build_matched_protection(
        training_bundle_path=args.generated_entity_fact_bundle,
        source_corpora=args.protection_source,
        output_dir=protection_dir,
        vocabulary_path=args.protection_vocabulary,
        split_seed=args.seed,
        minimum_train_per_key=args.minimum_protection_train_per_key,
        minimum_gate_per_key=args.minimum_protection_gate_per_key,
        strict=False,
        tokenizer=tokenizer,
    )
    protection_validation = validate_matched_protection_construction(
        protection_dir,
        target_subject=str(state["target"]["subject"]),
        minimum_train_per_key=args.minimum_protection_train_per_key,
        minimum_gate_per_key=args.minimum_protection_gate_per_key,
    )
    source_records, examples = legacy.load_mcf_retain(
        args.mcf_path,
        seed=args.seed,
        retain_num=args.mcf_optimization_count + args.mcf_gate_count,
    )
    records = [
        {
            "prompt": example.prompt,
            "answer": example.answer,
            "subject": example.subject,
            "target_new": example.target_new,
            "target_true": example.target_true,
            "source": "target_independent_mcf",
            "source_record_sha256": sha256_json(source_record),
        }
        for source_record, example in zip(source_records, examples)
    ]
    optimization_records = records[: args.mcf_optimization_count]
    gate_records = records[args.mcf_optimization_count :]
    optimization_hashes = {sha256_json(row) for row in optimization_records}
    gate_hashes = {sha256_json(row) for row in gate_records}
    if optimization_hashes & gate_hashes:
        raise ValueError("MCF optimization and gate partitions overlap")
    optimization_manifest = protection_dir / "mcf_optimization_manifest.json"
    gate_manifest = protection_dir / "mcf_gate_manifest.json"
    atomic_json_write(
        optimization_manifest,
        {
            "role": "optimization_protection",
            "gradient_allowed": True,
            "selection_allowed": False,
            "records": optimization_records,
            "record_sha256": sorted(optimization_hashes),
            "source_path": str(args.mcf_path.resolve()),
            "source_sha256": sha256_file(args.mcf_path),
        },
    )
    atomic_json_write(
        gate_manifest,
        {
            "role": "repair_selection_gate",
            "gradient_allowed": False,
            "selection_allowed": True,
            "must_run_under_no_grad": True,
            "records": gate_records,
            "record_sha256": sorted(gate_hashes),
            "source_path": str(args.mcf_path.resolve()),
            "source_sha256": sha256_file(args.mcf_path),
        },
    )
    train_path = protection_dir / "matched_protection_train.json"
    gate_path = protection_dir / "matched_protection_gate.json"
    coverage_summary = protection_validation["coverage_summary"]
    write_state(
        args,
        "PREPARED",
        protection_prepared=True,
        matched_protection_train_path=str(train_path.resolve()),
        matched_protection_train_sha256=sha256_file(train_path),
        matched_protection_gate_path=str(gate_path.resolve()),
        matched_protection_gate_sha256=sha256_file(gate_path),
        mcf_optimization_manifest_path=str(optimization_manifest.resolve()),
        mcf_optimization_manifest_sha256=sha256_file(optimization_manifest),
        mcf_gate_manifest_path=str(gate_manifest.resolve()),
        mcf_gate_manifest_sha256=sha256_file(gate_manifest),
        protection_coverage_count=len(result["coverage"]),
        matched_protection_key_count=coverage_summary["key_count"],
        matched_protection_covered_key_count=coverage_summary[
            "covered_key_count"
        ],
        matched_protection_insufficient_key_count=coverage_summary[
            "insufficient_key_count"
        ],
        matched_protection_insufficient_keys=coverage_summary[
            "insufficient_keys"
        ],
        official_rwku_records_accessed=False,
    )


def _verify_prepared_inputs(args: argparse.Namespace, state: Mapping[str, Any]) -> None:
    expected = {
        "prepared_training_bundle_path": args.generated_entity_fact_bundle,
        "prepared_generator_receipt_path": args.generator_receipt,
    }
    for field, path in expected.items():
        if state.get(field) != str(Path(path).resolve()):
            raise ValueError(f"{field} differs from PREPARED state")
    if sha256_file(args.generated_entity_fact_bundle) != state.get(
        "prepared_training_bundle_sha256"
    ):
        raise ValueError("Generated training bundle changed after PREPARED")
    if sha256_file(args.generator_receipt) != state.get(
        "prepared_generator_receipt_sha256"
    ):
        raise ValueError("Generator receipt changed after PREPARED")
    if sha256_path(args.model_path) != state.get("model_sha256"):
        raise ValueError("Pinned Base model changed after PREPARED")
    if sha256_json(configuration_payload(args)) != state.get("configuration_sha256"):
        raise ValueError("Utility-controlled configuration changed after PREPARED")


def _load_protection_inputs(
    state: Mapping[str, Any],
) -> Tuple[
    Dict[str, Any],
    Dict[str, Any],
    List[Mapping[str, Any]],
    List[Mapping[str, Any]],
    Path,
    Path,
    Path,
    Path,
]:
    if state.get("protection_prepared") is not True:
        raise ValueError("Run the protection stage before training")
    train_path = Path(str(state["matched_protection_train_path"]))
    gate_path = Path(str(state["matched_protection_gate_path"]))
    mcf_train_path = Path(str(state["mcf_optimization_manifest_path"]))
    mcf_gate_path = Path(str(state["mcf_gate_manifest_path"]))
    identities = (
        (train_path, "matched_protection_train_sha256"),
        (gate_path, "matched_protection_gate_sha256"),
        (mcf_train_path, "mcf_optimization_manifest_sha256"),
        (mcf_gate_path, "mcf_gate_manifest_sha256"),
    )
    for path, field in identities:
        if sha256_file(path) != state.get(field):
            raise ValueError(f"Frozen pre-training protection identity changed: {path}")
    matched_train = read_artifact(
        train_path,
        stage="train",
        gradient=True,
        expected_role="optimization_protection",
    )
    matched_gate = read_artifact(
        gate_path,
        stage="train",
        selection=True,
        expected_role="repair_selection_gate",
    )
    mcf_train = _load_json_mapping(mcf_train_path)
    mcf_gate = _load_json_mapping(mcf_gate_path)
    if (
        mcf_train.get("gradient_allowed") is not True
        or mcf_train.get("selection_allowed") is not False
    ):
        raise ValueError("Invalid MCF optimization partition permissions")
    if (
        mcf_gate.get("gradient_allowed") is not False
        or mcf_gate.get("selection_allowed") is not True
    ):
        raise ValueError("Invalid MCF gate partition permissions")
    if set(mcf_train.get("record_sha256", [])) & set(mcf_gate.get("record_sha256", [])):
        raise ValueError("MCF optimization and gate partitions are not disjoint")
    optimization_records = [
        *mcf_train.get("records", []),
        *matched_train["payload"].get("records", []),
    ]
    gate_records = [
        *mcf_gate.get("records", []),
        *matched_gate["payload"].get("records", []),
    ]
    return (
        matched_train,
        matched_gate,
        optimization_records,
        gate_records,
        train_path,
        gate_path,
        mcf_train_path,
        mcf_gate_path,
    )


def _sample_records(
    records: Sequence[Mapping[str, Any]],
    *,
    step: int,
    count: int,
) -> List[Mapping[str, Any]]:
    if not records:
        return []
    return [records[(step * count + offset) % len(records)] for offset in range(count)]


def _build_active_points(
    points: Sequence[TrainingPoint],
    tokenizer: Any,
    *,
    bundle_path: Path,
    bundle_sha256: str,
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for point in points:
        for answer_position, token_id in enumerate(
            _completion_token_ids(tokenizer, point.sensitive_answer)
        ):
            result.append(
                {
                    "fact_id": point.fact_id,
                    "view_id": point.view_id,
                    "prompt_style": point.prompt_style,
                    "answer_alias": point.sensitive_answer,
                    "answer_position": answer_position,
                    "token_id": int(token_id),
                    "prompt": point.prompt,
                    "source_record_sha256": point.source_record_sha256,
                    "training_bundle_sha256": bundle_sha256,
                    "source_path": str(bundle_path.resolve()),
                    "source_artifact_role": "training_bundle",
                    "level": "generated",
                    "active_source": ACTIVE_SOURCE,
                }
            )
    return validate_active_points(
        result,
        training_bundle_path=bundle_path,
        training_bundle_sha256=bundle_sha256,
    )


def active_position_support_report(
    active_points: Sequence[Mapping[str, Any]],
    eligible_rows: Sequence[int],
) -> Dict[str, Any]:
    eligible = {int(value) for value in eligible_rows}
    supported = [
        dict(point)
        for point in active_points
        if int(point["token_id"]) in eligible
    ]
    unsupported = [
        dict(point)
        for point in active_points
        if int(point["token_id"]) not in eligible
    ]
    by_fact: MutableMapping[str, List[Dict[str, Any]]] = {}
    for point in unsupported:
        by_fact.setdefault(str(point["fact_id"]), []).append(point)
    return {
        "eligible_sensitive_token_positions": supported,
        "unsupported_sensitive_token_positions": unsupported,
        "unsupported_positions_grouped_by_fact": {
            key: values for key, values in sorted(by_fact.items())
        },
        "supported_answer_position_count": len(supported),
        "unsupported_answer_position_count": len(unsupported),
        "total_answer_position_count": len(active_points),
        "percentage_answer_positions_supported": (
            100.0 * len(supported) / len(active_points) if active_points else 0.0
        ),
    }


def write_training_diagnostics(
    args: argparse.Namespace, payload: Mapping[str, Any]
) -> Path:
    path = run_dir(args) / "training_diagnostics.json"
    normalized, replacements = strict_json_normalize(dict(payload))
    normalized["serialization"] = {
        "policy": "non_finite_numeric_values_to_json_null",
        "replacement_count": len(replacements),
        "replacements": replacements,
    }
    atomic_json_write(path, normalized)
    return path


def learn_unscaled_repair_deltas(
    model: nn.Module,
    tokenizer: Any,
    active_points: Sequence[Mapping[str, Any]],
    *,
    eligible_rows: Sequence[int],
    steps: int,
    learning_rate: float,
    margin: float,
) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """Learn row deltas without changing the model or any checkpoint."""

    output_weight = model.get_output_embeddings().weight
    device = output_weight.device
    eligible = set(int(value) for value in eligible_rows)
    supported = [point for point in active_points if int(point["token_id"]) in eligible]
    unsupported = [
        point for point in active_points if int(point["token_id"]) not in eligible
    ]
    row_ids = sorted({int(point["token_id"]) for point in supported})
    if not row_ids:
        return {}, {
            "supported_active_position_count": 0,
            "unsupported_active_positions": unsupported,
            "optimization_steps": 0,
        }
    row_to_index = {row_id: index for index, row_id in enumerate(row_ids)}
    hidden_rows: List[torch.Tensor] = []
    competitor_logits: List[torch.Tensor] = []
    target_indices: List[int] = []
    with torch.no_grad():
        for point in supported:
            prompt_ids = _token_ids(tokenizer, str(point["prompt"]))
            answer_ids = _completion_token_ids(tokenizer, str(point["answer_alias"]))
            position = int(point["answer_position"])
            prefix = prompt_ids + answer_ids[:position]
            inputs = torch.tensor([prefix], dtype=torch.long, device=device)
            hidden = (
                final_hidden_states(model, input_ids=inputs)[:, -1, :][0]
                .detach()
                .float()
            )
            logits = output_weight.float().matmul(hidden)
            token_id = int(point["token_id"])
            logits[token_id] = -torch.inf
            hidden_rows.append(hidden)
            competitor_logits.append(logits.max().detach())
            target_indices.append(row_to_index[token_id])
    hidden_tensor = torch.stack(hidden_rows)
    competitor_tensor = torch.stack(competitor_logits)
    target_index = torch.tensor(target_indices, dtype=torch.long, device=device)
    base_rows = (
        output_weight.detach()
        .float()
        .index_select(0, torch.tensor(row_ids, dtype=torch.long, device=device))
    )
    delta = nn.Parameter(torch.zeros_like(base_rows))
    optimizer = torch.optim.AdamW([delta], lr=float(learning_rate), weight_decay=0.0)
    losses: List[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        target_rows = (base_rows + delta).index_select(0, target_index)
        target_logits = (target_rows * hidden_tensor).sum(dim=-1)
        loss = F.relu(float(margin) + target_logits - competitor_tensor).mean()
        loss = loss + 1e-6 * delta.pow(2).mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    deltas = {
        row_id: delta.detach()[index].clone() for index, row_id in enumerate(row_ids)
    }
    return deltas, {
        "supported_active_position_count": len(supported),
        "unsupported_active_positions": unsupported,
        "optimization_steps": int(steps),
        "learning_rate": float(learning_rate),
        "loss_history": losses,
    }


def _save_checkpoint(model: nn.Module, tokenizer: Any, path: Path) -> None:
    if Path(path).exists():
        raise ValueError(f"Refusing to overwrite checkpoint: {path}")
    legacy.save_checkpoint(model, tokenizer, path)


def _materialized_selected_delta_norms(
    adapter: SparseFP32RowDeltas,
) -> Dict[str, float]:
    input_delta = (
        adapter.input_layer.weight.detach()
        .index_select(0, adapter.input_indices)
        .float()
        - adapter.input_base_rows_native.float()
    )
    output_delta = (
        adapter.output_layer.weight.detach()
        .index_select(0, adapter.output_indices)
        .float()
        - adapter.output_base_rows_native.float()
    )
    input_norm = float(input_delta.norm().cpu())
    output_norm = float(output_delta.norm().cpu())
    return {
        "selected_input_row_delta_norm": input_norm,
        "selected_output_row_delta_norm": output_norm,
        "total_selected_row_delta_norm_including_repair": math.sqrt(
            input_norm**2 + output_norm**2
        ),
    }


def train_stage(args: argparse.Namespace) -> None:
    """Train FP32 sparse deltas, then repair up to eight utility-safe candidates."""

    state = read_state(args)
    if state.get("state") != "PREPARED":
        raise ValueError(f"Training requires PREPARED, got {state.get('state')}")
    receipt_path = run_dir(args) / "checkpoint_receipt.json"
    if receipt_path.exists():
        assert_model_modification_allowed(
            receipt_path, experiment_id=args.experiment_id
        )
    _verify_prepared_inputs(args, state)
    training = read_artifact(
        args.generated_entity_fact_bundle,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    if training.get("protocol_label") != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Training bundle is not the target-only generated-corpus track")
    generator = read_artifact(
        args.generator_receipt,
        stage="train",
        expected_role="generator_receipt",
    )
    if generator["payload"].get("final_entity_fact_bundle_sha256") != training.get(
        "sha256"
    ):
        raise ValueError("Generator receipt and training-bundle artifact hashes differ")
    legacy._validate_training_bundle_sources(
        training, training_source=legacy.TRAINING_SOURCE_TARGET_ONLY
    )
    (
        matched_train,
        matched_gate,
        optimization_records,
        gate_records,
        matched_train_path,
        matched_gate_path,
        mcf_train_path,
        mcf_gate_path,
    ) = _load_protection_inputs(state)
    legacy._validate_matched_protection_artifact(
        matched_train, target_subject=state["target"]["subject"]
    )
    legacy._validate_matched_protection_artifact(
        matched_gate, target_subject=state["target"]["subject"]
    )
    if not optimization_records or not gate_records:
        raise ValueError(
            "Utility-controlled training requires non-empty optimization and gate partitions"
        )
    proxy_text = load_wikidata_text(args.wikidata_dir)
    if not proxy_text:
        raise FileNotFoundError(
            "A fixed target-independent Wikidata proxy-PPL corpus is required"
        )

    write_state(
        args,
        "TRAINING",
        training_started_at_utc=utc_now(),
        official_rwku_records_accessed=False,
    )
    legacy.set_all_seeds(args.seed)
    dtype = legacy.dtype_from_name(args.dtype)
    model, tokenizer = legacy.load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        for_training=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    teacher, teacher_tokenizer = legacy.load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        for_training=False,
        gradient_checkpointing=False,
    )
    del teacher_tokenizer
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    sample = tokenizer("RWKU sparse-delta initialization", return_tensors="pt")
    sample_ids = sample["input_ids"].to(next(model.parameters()).device)
    untie_report = untie_lm_head_preserve_logits(model, sample_input_ids=sample_ids)
    freeze_report = freeze_transformer_parameters(model)
    row_policy = build_row_policy(
        tokenizer,
        training,
        optimization_records,
        maximum_document_frequency=args.max_retain_document_frequency,
    )
    if not row_policy.selected_input_rows:
        raise ValueError("No safe declared subject input rows are trainable")
    if not row_policy.selected_output_rows:
        raise ValueError("No safe sensitive-answer output rows are trainable")
    adapter = SparseFP32RowDeltas(
        model,
        row_policy.selected_input_rows,
        row_policy.selected_output_rows,
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [adapter.input_delta],
                "lr": args.subject_input_delta_lr,
                "weight_decay": 0.0,
            },
            {
                "params": [adapter.output_delta],
                "lr": args.sensitive_output_delta_lr,
                "weight_decay": 0.0,
            },
        ]
    )
    points = compile_training_points(tokenizer, training)
    by_fact: MutableMapping[str, List[TrainingPoint]] = {}
    for point in points:
        by_fact.setdefault(point.fact_id, []).append(point)
    fact_ids = sorted(by_fact)
    schedule = balanced_candidate_schedule(len(fact_ids), args.exposures_per_fact)
    total_steps = schedule[-1]["step"]
    fact_order = balanced_fact_order(fact_ids, total_steps)
    exposure_counts = {fact_id: 0 for fact_id in fact_ids}
    schedule_by_step = {row["step"]: row for row in schedule}
    candidate_rows: List[Dict[str, Any]] = []
    checkpoint_deltas: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    loss_log: List[Dict[str, Any]] = []
    checkpoint_root = run_dir(args) / "utility_controlled_setting5" / "checkpoints"
    candidate_report_path = (
        run_dir(args) / "utility_controlled_setting5" / "candidate_report.json"
    )
    bundle_sha = sha256_file(args.generated_entity_fact_bundle)
    generator_sha = sha256_file(args.generator_receipt)
    active_points = _build_active_points(
        points,
        tokenizer,
        bundle_path=args.generated_entity_fact_bundle,
        bundle_sha256=bundle_sha,
    )
    support_report = active_position_support_report(
        active_points, row_policy.selected_output_rows
    )
    started = time.perf_counter()
    model.train()
    for step_index, fact_id in enumerate(fact_order, start=1):
        fact_points = sorted(by_fact[fact_id], key=lambda point: point.view_id)
        point = fact_points[exposure_counts[fact_id] % len(fact_points)]
        exposure_counts[fact_id] += 1
        retain_batch = _sample_records(
            optimization_records,
            step=step_index - 1,
            count=args.retain_batch_size,
        )
        optimizer.zero_grad(set_to_none=True)
        forget_parts = forget_objective_components(
            model,
            tokenizer,
            point,
            forget_token_margin=args.forget_token_margin,
            probability_cap_weight=args.probability_cap_weight,
            eos_margin_weight=args.eos_margin_weight,
            eos_margin=args.forget_margin,
        )
        retain_ce = retain_answer_ce(model, tokenizer, retain_batch)
        retain_kl, retain_kl_p95 = teacher_kl_for_records(
            model,
            teacher,
            tokenizer,
            retain_batch,
            top_k=args.teacher_top_k,
        )
        protected = protected_answer_hinge(
            model, teacher, tokenizer, retain_batch
        )
        delta_l2 = adapter.input_delta.pow(2).sum() + adapter.output_delta.pow(2).sum()
        total = (
            args.forget_weight * forget_parts["total_forget_loss"]
            + args.retain_ce_weight * retain_ce
            + args.retain_kl_weight * retain_kl
            + args.protected_margin_weight * protected
            + args.delta_l2_weight * delta_l2
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite sparse-delta loss at step {step_index}")
        total.backward()
        input_gradient_norm = float(
            adapter.input_delta.grad.detach().float().norm().cpu()
            if adapter.input_delta.grad is not None
            else 0.0
        )
        output_gradient_norm = float(
            adapter.output_delta.grad.detach().float().norm().cpu()
            if adapter.output_delta.grad is not None
            else 0.0
        )
        torch.nn.utils.clip_grad_norm_(
            [adapter.input_delta, adapter.output_delta], args.grad_clip
        )
        optimizer.step()
        # Base matrices are excluded from the optimizer and remain immutable.
        if not torch.equal(
            adapter.input_layer.weight.index_select(0, adapter.input_indices),
            adapter.input_base_rows_native,
        ) or not torch.equal(
            adapter.output_layer.weight.index_select(0, adapter.output_indices),
            adapter.output_base_rows_native,
        ):
            raise RuntimeError("Sparse-delta optimization modified a Base row")
        norms = adapter.delta_norms()
        loss_log.append(
            {
                "step": step_index,
                "fact_id": fact_id,
                "view_id": point.view_id,
                "total_forget_loss": float(
                    forget_parts["total_forget_loss"].detach().cpu()
                ),
                "token_demotion_loss": float(
                    forget_parts["token_demotion_loss"].detach().cpu()
                ),
                "probability_cap_loss": float(
                    forget_parts["probability_cap_loss"].detach().cpu()
                ),
                "eos_auxiliary_margin_loss": float(
                    forget_parts["eos_auxiliary_margin_loss"].detach().cpu()
                ),
                "retain_answer_ce": float(retain_ce.detach().cpu()),
                "teacher_kl_mean": float(retain_kl.detach().cpu()),
                "teacher_kl_p95": retain_kl_p95,
                "protected_answer_hinge": float(protected.detach().cpu()),
                "selected_row_delta_l2": float(delta_l2.detach().cpu()),
                "input_delta_gradient_norm": input_gradient_norm,
                "output_delta_gradient_norm": output_gradient_norm,
                "input_delta_norm": norms["selected_input_row_delta_norm"],
                "output_delta_norm": norms["selected_output_row_delta_norm"],
                "total_loss": float(total.detach().cpu()),
            }
        )
        if step_index not in schedule_by_step:
            continue
        model.eval()
        input_snapshot, output_snapshot = adapter.snapshot()
        checkpoint_deltas[step_index] = (input_snapshot, output_snapshot)
        adapter.materialize(input_snapshot, output_snapshot, 1.0)
        if not adapter.nonselected_equal_base(teacher):
            raise RuntimeError("Checkpoint materialization changed a nonselected row")
        _save_checkpoint(
            model, tokenizer, checkpoint_root / f"step_{step_index}"
        )
        for scale in args.candidate_scales:
            adapter.materialize(input_snapshot, output_snapshot, scale)
            metrics = evaluate_pre_freeze_candidate(
                model,
                teacher,
                tokenizer,
                points,
                gate_records,
                selected_output_rows=row_policy.selected_output_rows,
                proxy_text=proxy_text,
                batch_size=args.candidate_eval_batch_size,
                teacher_top_k=args.teacher_top_k,
                nonselected_rows_equal_base=adapter.nonselected_equal_base(teacher),
            )
            gates = candidate_gate_report(metrics)
            candidate_rows.append(
                {
                    "candidate_id": f"step_{step_index}_scale_{float(scale):g}",
                    "checkpoint_step": step_index,
                    "requested_exposures_per_fact": schedule_by_step[step_index][
                        "requested_exposures_per_fact"
                    ],
                    "per_fact_exposure_counts": dict(exposure_counts),
                    "exposure_imbalance": max(exposure_counts.values())
                    - min(exposure_counts.values()),
                    "interpolation_scale": float(scale),
                    **adapter.delta_norms(
                        input_snapshot, output_snapshot, scale=float(scale)
                    ),
                    **metrics,
                    **gates,
                    "post_repair_generated_forget_gates_passed": None,
                    "post_repair_protection_gates_passed": None,
                    "final_eligible": None,
                }
            )
        adapter.resume_sparse_training()
        model.train()

    utility_safe = rank_pre_repair_candidates(candidate_rows)
    repair_outcomes: List[Dict[str, Any]] = []
    final_states: Dict[str, Dict[str, torch.Tensor]] = {}
    output_audit_by_id = {
        int(row["token_id"]): row for row in row_policy.output_audit
    }
    model.eval()
    for pre_candidate in utility_safe:
        candidate_id = str(pre_candidate["candidate_id"])
        checkpoint_step = int(pre_candidate["checkpoint_step"])
        scale = float(pre_candidate["interpolation_scale"])
        input_snapshot, output_snapshot = checkpoint_deltas[checkpoint_step]
        adapter.materialize(input_snapshot, output_snapshot, scale)
        pre_input_rows = (
            adapter.input_layer.weight.index_select(0, adapter.input_indices)
            .detach()
            .cpu()
            .clone()
        )
        pre_output_rows = (
            adapter.output_layer.weight.index_select(0, adapter.output_indices)
            .detach()
            .cpu()
            .clone()
        )
        unscaled_deltas, repair_optimization = learn_unscaled_repair_deltas(
            model,
            tokenizer,
            active_points,
            eligible_rows=row_policy.selected_output_rows,
            steps=args.repair_steps,
            learning_rate=args.repair_lr,
            margin=args.repair_margin,
        )
        immutable_repair_weight = adapter.output_layer.weight.detach().clone()

        def repair_evaluator(
            scales: Mapping[int, float],
        ) -> Mapping[str, Any]:
            apply_rowwise_delta(
                adapter.output_layer.weight,
                immutable_repair_weight,
                unscaled_deltas,
                scales,
            )
            return evaluate_pre_freeze_candidate(
                model,
                teacher,
                tokenizer,
                points,
                gate_records,
                selected_output_rows=row_policy.selected_output_rows,
                proxy_text=proxy_text,
                batch_size=args.candidate_eval_batch_size,
                teacher_top_k=args.teacher_top_k,
                nonselected_rows_equal_base=adapter.nonselected_equal_base(teacher),
            )

        baseline_repair_metrics = dict(repair_evaluator({}))
        contributions: Dict[int, Dict[str, float]] = {}
        for row_id in sorted(unscaled_deltas):
            metrics = dict(repair_evaluator({row_id: 1.0}))
            contributions[row_id] = {
                "generated_efficacy_contribution": max(
                    0.0,
                    baseline_repair_metrics[
                        "generated_geometric_answer_probability"
                    ]
                    - metrics["generated_geometric_answer_probability"],
                ),
                "protected_drift_contribution": max(
                    0.0,
                    metrics["protected_selected_row_logit_drift"]
                    - baseline_repair_metrics[
                        "protected_selected_row_logit_drift"
                    ],
                ),
            }
        repair_selection = select_rowwise_scales(
            sorted(unscaled_deltas),
            evaluate=repair_evaluator,
            row_contributions=contributions,
        )
        selected_scales = {
            int(key): float(value)
            for key, value in repair_selection["selected_scale_by_row"].items()
        }
        final_metrics = dict(repair_evaluator(selected_scales))
        final_gates = final_candidate_gate_report(final_metrics)
        rowwise_norm = selected_delta_norm(unscaled_deltas, selected_scales)
        total_norms = _materialized_selected_delta_norms(adapter)
        supported_by_row: MutableMapping[int, List[Mapping[str, Any]]] = {}
        for active_point in active_points:
            token_id = int(active_point["token_id"])
            if token_id in unscaled_deltas:
                supported_by_row.setdefault(token_id, []).append(active_point)
        outcome = {
            "candidate_id": candidate_id,
            "checkpoint_step": checkpoint_step,
            "interpolation_scale": scale,
            "pre_repair_utility_safe": True,
            "pre_repair_failed_protection_gates": [],
            "pre_repair_metrics": pre_candidate,
            "repair_attempted": True,
            "selected_scale_by_row": {
                str(key): value for key, value in selected_scales.items()
            },
            "rowwise_repair_norm": rowwise_norm,
            **total_norms,
            **final_metrics,
            **final_gates,
            "repair_selection": repair_selection,
            "repair_optimization": repair_optimization,
            "row_contributions": {
                str(key): value for key, value in contributions.items()
            },
            "row_details": [
                {
                    "token_id": row_id,
                    "decoded_token_piece": output_audit_by_id[row_id][
                        "decoded_token_piece"
                    ],
                    "eligibility_class": output_audit_by_id[row_id][
                        "eligibility_class"
                    ],
                    "selected_scale": selected_scales.get(row_id, 0.0),
                    "active_positions_supported": supported_by_row.get(row_id, []),
                    **contributions.get(row_id, {}),
                }
                for row_id in sorted(unscaled_deltas)
            ],
            "unsupported_positions_grouped_by_fact": support_report[
                "unsupported_positions_grouped_by_fact"
            ],
        }
        repair_outcomes.append(outcome)
        if outcome["final_eligible"]:
            final_states[candidate_id] = {
                "pre_input_rows": pre_input_rows,
                "pre_output_rows": pre_output_rows,
                "repaired_output_rows": (
                    adapter.output_layer.weight.index_select(
                        0, adapter.output_indices
                    )
                    .detach()
                    .cpu()
                    .clone()
                ),
            }
        adapter.resume_sparse_training()

    selected_meta = select_final_repaired_candidate(repair_outcomes)
    report = {
        "schema_version": "rwku_setting5e_uc_candidate_report_v2",
        "method": METHOD,
        "protocol_status": PROTOCOL_STATUS,
        "selection_sources": [
            "generated_training_bundle_views",
            "disjoint_mcf_gate_partition",
            "disjoint_matched_protection_gate",
            "target_independent_wikidata_proxy_ppl",
        ],
        "official_rwku_records_accessed": False,
        "fixed_thresholds": fixed_gate_manifest(),
        "schedule": schedule,
        "candidate_scales": list(args.candidate_scales),
        "candidates": candidate_rows,
        "utility_safe_candidate_count": sum(
            candidate["pre_repair_utility_safe"] for candidate in candidate_rows
        ),
        "candidates_sent_to_active_repair": [
            candidate["candidate_id"] for candidate in utility_safe
        ],
        "maximum_candidates_sent_to_active_repair": MAX_PRE_REPAIR_CANDIDATES,
        "repair_outcomes": repair_outcomes,
        "selected_candidate": selected_meta,
        "pre_repair_ranking_order": [
            "fewer_active_violation_count",
            "smaller_generated_geometric_answer_probability",
            "lower_direct_cloze_paraphrase_recovery_sum",
            "smaller_selected_row_delta_norm",
            "earlier_checkpoint",
        ],
        "final_selection_order": [
            "smallest_total_selected_row_delta_norm_including_repair",
            "earliest_setting5_checkpoint",
            "smallest_setting5_interpolation_scale",
            "smallest_rowwise_repair_norm",
        ],
        "active_position_support": support_report,
    }
    candidate_report_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_report, report_replacements = strict_json_normalize(report)
    normalized_report["serialization"] = {
        "policy": "non_finite_numeric_values_to_json_null",
        "replacement_count": len(report_replacements),
        "replacements": report_replacements,
    }
    atomic_json_write(candidate_report_path, normalized_report)
    write_state(
        args,
        "CANDIDATES_EVALUATED",
        candidate_report_path=str(candidate_report_path.resolve()),
        candidate_report_sha256=sha256_file(candidate_report_path),
        candidate_count=len(candidate_rows),
        utility_safe_candidate_count=len(
            [row for row in candidate_rows if row["pre_repair_utility_safe"]]
        ),
        candidates_sent_to_active_repair=len(utility_safe),
        official_rwku_records_accessed=False,
    )

    diagnostics = {
        "schema_version": "rwku_setting5e_uc_training_diagnostics_v1",
        "method": METHOD,
        "protocol_status": PROTOCOL_STATUS,
        "official_rwku_records_accessed": False,
        "generated_training_bundle_path": str(
            args.generated_entity_fact_bundle.resolve()
        ),
        "generated_training_bundle_sha256": bundle_sha,
        "generator_receipt_sha256": generator_sha,
        "selected_input_row_ids": list(row_policy.selected_input_rows),
        "selected_output_row_ids": list(row_policy.selected_output_rows),
        "selected_input_rows": list(row_policy.input_audit),
        "selected_and_excluded_output_rows": list(row_policy.output_audit),
        "fp32_sparse_delta_dtypes": {
            "input_delta": str(adapter.input_delta.dtype),
            "output_delta": str(adapter.output_delta.dtype),
        },
        "fp32_sparse_delta_norms": adapter.delta_norms(),
        "input_delta_gradient_norm": max(
            (row["input_delta_gradient_norm"] for row in loss_log), default=0.0
        ),
        "output_delta_gradient_norm": max(
            (row["output_delta_gradient_norm"] for row in loss_log), default=0.0
        ),
        "loss_log": loss_log,
        "candidate_metrics": candidate_rows,
        "utility_safe_candidate_count": len(
            [row for row in candidate_rows if row["pre_repair_utility_safe"]]
        ),
        "candidates_sent_to_active_repair": [
            candidate["candidate_id"] for candidate in utility_safe
        ],
        "repair_outcomes": repair_outcomes,
        "active_position_support": support_report,
        "schedule": schedule,
        "final_exposure_counts": exposure_counts,
        "training_seconds": time.perf_counter() - started,
    }
    training_diagnostics_path = write_training_diagnostics(args, diagnostics)
    if selected_meta is None:
        mark_no_feasible_candidate(args)
        adapter.close()
        legacy.release_model(model)
        legacy.release_model(teacher)
        return

    selected_state = final_states[str(selected_meta["candidate_id"])]
    adapter.enabled = False
    adapter.restore_base_rows()
    with torch.no_grad():
        adapter.input_layer.weight.index_copy_(
            0,
            adapter.input_indices,
            selected_state["pre_input_rows"].to(
                adapter.input_layer.weight.device,
                adapter.input_layer.weight.dtype,
            ),
        )
        adapter.output_layer.weight.index_copy_(
            0,
            adapter.output_indices,
            selected_state["pre_output_rows"].to(
                adapter.output_layer.weight.device,
                adapter.output_layer.weight.dtype,
            ),
        )
    selected_checkpoint = (
        run_dir(args) / "utility_controlled_setting5" / "selected_checkpoint"
    )
    _save_checkpoint(model, tokenizer, selected_checkpoint)
    with torch.no_grad():
        adapter.output_layer.weight.index_copy_(
            0,
            adapter.output_indices,
            selected_state["repaired_output_rows"].to(
                adapter.output_layer.weight.device,
                adapter.output_layer.weight.dtype,
            ),
        )
    repaired_checkpoint = run_dir(args) / "rowwise_repair" / "selected_checkpoint"
    _save_checkpoint(model, tokenizer, repaired_checkpoint)
    repair_report_path = run_dir(args) / "rowwise_repair" / "repair_report.json"
    repair_report_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_repair, repair_replacements = strict_json_normalize(selected_meta)
    normalized_repair["serialization"] = {
        "policy": "non_finite_numeric_values_to_json_null",
        "replacement_count": len(repair_replacements),
        "replacements": repair_replacements,
    }
    atomic_json_write(repair_report_path, normalized_repair)

    training_report = {
        "schema_version": "rwku_setting5e_uc_training_report_v2",
        "method": METHOD,
        "protocol_status": PROTOCOL_STATUS,
        "fresh_base_model_loaded": True,
        "teacher_base_model_frozen": True,
        "official_rwku_records_accessed": False,
        "untie": untie_report,
        "freezing": freeze_report,
        "row_policy": {
            "selected_input_rows": list(row_policy.selected_input_rows),
            "selected_output_rows": list(row_policy.selected_output_rows),
            "input_row_audit": list(row_policy.input_audit),
            "output_row_audit": list(row_policy.output_audit),
        },
        "objective": configuration_payload(args)["objective"],
        "schedule": schedule,
        "final_exposure_counts": exposure_counts,
        "exposure_imbalance": max(exposure_counts.values())
        - min(exposure_counts.values()),
        "loss_log": loss_log,
        "selected_candidate": selected_meta,
        "candidate_report_path": str(candidate_report_path.resolve()),
        "repair_report_path": str(repair_report_path.resolve()),
        "training_diagnostics_path": str(training_diagnostics_path.resolve()),
        "training_seconds": time.perf_counter() - started,
    }
    training_report_path = run_dir(args) / "training_report.json"
    normalized_training, training_replacements = strict_json_normalize(training_report)
    normalized_training["serialization"] = {
        "policy": "non_finite_numeric_values_to_json_null",
        "replacement_count": len(training_replacements),
        "replacements": training_replacements,
    }
    atomic_json_write(training_report_path, normalized_training)
    target = target_for_seed(args.seed)
    method_configuration = {
        **configuration_payload(args),
        "selected_checkpoint_step": selected_meta["checkpoint_step"],
        "selected_interpolation_scale": selected_meta["interpolation_scale"],
        "selected_input_row_ids": list(row_policy.selected_input_rows),
        "selected_output_row_ids": list(row_policy.selected_output_rows),
        "rowwise_repair_scales": selected_meta["selected_scale_by_row"],
        "candidate_report_path": str(candidate_report_path.resolve()),
        "candidate_report_sha256": sha256_file(candidate_report_path),
        "repair_report_path": str(repair_report_path.resolve()),
        "repair_report_sha256": sha256_file(repair_report_path),
        "training_report_path": str(training_report_path.resolve()),
        "training_report_sha256": sha256_file(training_report_path),
        "training_diagnostics_path": str(training_diagnostics_path.resolve()),
        "training_diagnostics_sha256": sha256_file(training_diagnostics_path),
        "exact_command": [sys.executable, str(SCRIPT_PATH), *sys.argv[1:]],
        "development": bool(args.development),
        "confirmatory": bool(args.confirmatory),
    }
    receipt = create_checkpoint_receipt(
        destination=receipt_path,
        experiment_id=args.experiment_id,
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=PROTOCOL_STATUS,
        target_entity=target.subject,
        target_entity_id=f"rwku:{target.directory}",
        base_model_identity=legacy.local_model_identity(args.model_path),
        base_model_revision=args.model_revision,
        tokenizer_identity={
            "name_or_path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
        },
        checkpoint_paths=[selected_checkpoint, repaired_checkpoint],
        training_bundle_path=args.generated_entity_fact_bundle,
        optimization_protection_path=matched_train_path,
        mcf_retain_optimization_paths=[mcf_train_path],
        mcf_repair_gate_paths=[mcf_gate_path],
        matched_protection_train_path=matched_train_path,
        matched_protection_gate_path=matched_gate_path,
        method_configuration=method_configuration,
        implementation_files=[
            SCRIPT_PATH,
            SEMANTIC_ROOT / "scripts" / "rwku_rowwise_active_repair.py",
            SEMANTIC_ROOT / "scripts" / "rwku_artifact_access.py",
            SEMANTIC_ROOT / "scripts" / "rwku_checkpoint_receipt.py",
            SEMANTIC_ROOT / "scripts" / "rwku_eval.py",
        ],
        sampler_provenance={
            "schedule": schedule,
            "final_exposure_counts": exposure_counts,
            "exposure_imbalance": max(exposure_counts.values())
            - min(exposure_counts.values()),
        },
        generator_receipt_path=args.generator_receipt,
        official_locked_eval_path=run_dir(args) / "official_locked_eval.json",
        confirmatory=args.confirmatory,
        additional_artifact_paths={
            "configuration_manifest": run_dir(args) / "configuration_manifest.json",
            "candidate_report": candidate_report_path,
            "repair_report": repair_report_path,
            "training_report": training_report_path,
            "training_diagnostics": training_diagnostics_path,
            "base_model_source": args.model_path,
        },
    )
    write_state(
        args,
        "CHECKPOINT_FROZEN",
        checkpoint_receipt=str(receipt_path.resolve()),
        checkpoint_receipt_sha256=receipt["receipt_sha256"],
        selected_checkpoint_path=str(selected_checkpoint.resolve()),
        repaired_checkpoint_path=str(repaired_checkpoint.resolve()),
        training_diagnostics_path=str(training_diagnostics_path.resolve()),
        training_diagnostics_sha256=sha256_file(training_diagnostics_path),
        official_evaluation_opened=False,
    )
    adapter.close()
    legacy.release_model(model)
    legacy.release_model(teacher)


def _official_level_metrics(
    model: nn.Module,
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> Dict[str, Any]:
    summary, details = evaluate_qa_rows(
        model,
        tokenizer,
        rows,
        batch_size=batch_size,
        score_answers=True,
    )
    geometric = [float(row["answer_geometric_probability"]) for row in details]
    full = [_safe_exp(float(row["answer_sum_logprob"])) for row in details]
    first = [float(row["answer_first_token_probability"]) for row in details]
    return {
        "count": len(details),
        "generation_recovery": float(summary["recovery_accuracy"]),
        "paper_code_compatible_eff": 100.0 * float(np.mean(geometric))
        if geometric
        else float("nan"),
        "literal_full_sequence_answer_probability": float(np.mean(full))
        if full
        else float("nan"),
        "first_token_probability": float(np.mean(first)) if first else float("nan"),
        "details": details,
    }


def _evaluate_official_model(
    *,
    method: str,
    model: nn.Module,
    tokenizer: Any,
    target: Any,
    datasets: Mapping[str, Sequence[Mapping[str, Any]]],
    args: argparse.Namespace,
    frozen_probe: Any,
    base_retain: Optional[Mapping[str, float]],
) -> Dict[str, Any]:
    result = evaluate_rwku(
        method=method,
        model=model,
        tokenizer=tokenizer,
        subject=target.subject,
        held_out_cloze=list(datasets["forget_level1.json"]),
        held_out_direct=list(datasets["forget_level2.json"]),
        datasets=datasets,
        wikidata_dir=args.wikidata_dir,
        batch_size=args.eval_batch_size,
        base_retain_mean_logprobs=base_retain,
        frozen_head_probe=frozen_probe,
        limits={},
        skip_ppl=False,
    )
    result["official_level_metrics"] = {
        "level1": _official_level_metrics(
            model,
            tokenizer,
            datasets["forget_level1.json"],
            batch_size=args.eval_batch_size,
        ),
        "level2": _official_level_metrics(
            model,
            tokenizer,
            datasets["forget_level2.json"],
            batch_size=args.eval_batch_size,
        ),
        "level3": _official_level_metrics(
            model,
            tokenizer,
            datasets["forget_level3.json"],
            batch_size=args.eval_batch_size,
        ),
    }
    return result


def _verify_evaluation_contract(
    args: argparse.Namespace,
    state: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    if state.get("state") != "CHECKPOINT_FROZEN":
        raise ValueError(
            f"Evaluation requires CHECKPOINT_FROZEN, got {state.get('state')}"
        )
    if receipt.get("state") != "CHECKPOINT_FROZEN":
        raise ValueError("Checkpoint receipt is not frozen and unopened")
    if receipt.get("protocol_status") != PROTOCOL_STATUS:
        raise ValueError("Checkpoint receipt belongs to a different RWKU method")
    if receipt.get("experiment_id") != args.experiment_id:
        raise ValueError("Checkpoint receipt experiment ID mismatch")
    if receipt.get("base_model_revision") != args.model_revision:
        raise ValueError("Base model revision differs from frozen receipt")
    if sha256_json(legacy.local_model_identity(args.model_path)) != sha256_json(
        receipt.get("base_model_identity", {})
    ):
        raise ValueError("Base model identity differs from frozen receipt")
    if (
        sha256_path(args.model_path)
        != receipt["artifacts"]["base_model_source"]["sha256"]
    ):
        raise ValueError("Base model changed after checkpoint freeze")
    if bool(receipt.get("confirmatory")) != bool(args.confirmatory):
        raise ValueError("Development/confirmatory mode differs from frozen receipt")
    if sha256_json(configuration_payload(args)) != state.get("configuration_sha256"):
        raise ValueError("Evaluation configuration differs from PREPARED state")
    verify_frozen_identities(receipt)


def evaluate_stage(args: argparse.Namespace) -> None:
    state = read_state(args)
    receipt_path = run_dir(args) / "checkpoint_receipt.json"
    receipt = load_receipt(receipt_path)
    _verify_evaluation_contract(args, state, receipt)
    result_path = run_dir(args) / "official_evaluation.json"
    if result_path.exists():
        raise ValueError(
            f"Refusing to overwrite completed official result: {result_path}"
        )
    opened = open_official_evaluation(receipt_path, experiment_id=args.experiment_id)
    write_state(
        args,
        "OFFICIAL_EVALUATION_OPENED",
        official_evaluation_opened=True,
        official_evaluation_opened_at_utc=opened["official_evaluation_opened_at_utc"],
    )
    try:
        locked = read_artifact(
            run_dir(args) / "official_locked_eval.json",
            stage="evaluate",
            evaluation=True,
            expected_role="official_locked_eval",
        )
        target, datasets, file_hashes = ensure_target_data(
            args.data_root,
            args.seed,
            allow_download=not args.no_download,
        )
        for filename, descriptor in locked["payload"]["files"].items():
            if file_hashes.get(filename) != descriptor["sha256"]:
                raise ValueError(f"Official locked RWKU file changed: {filename}")
        dtype = legacy.dtype_from_name(args.dtype)
        base_model, tokenizer = legacy.load_model_and_tokenizer(
            args.model_path,
            dtype=dtype,
            for_training=False,
            gradient_checkpointing=False,
        )
        all_answers = [
            str(row["answer"])
            for filename in (
                "forget_level1.json",
                "forget_level2.json",
                "forget_level3.json",
            )
            for row in datasets[filename]
        ]
        frozen_probe = build_frozen_head_probe(
            base_model,
            tokenizer,
            datasets["forget_level2.json"],
            additional_answers=all_answers,
        )
        base_result = _evaluate_official_model(
            method="Base model",
            model=base_model,
            tokenizer=tokenizer,
            target=target,
            datasets=datasets,
            args=args,
            frozen_probe=frozen_probe,
            base_retain=None,
        )
        base_retain = base_result["retain_reference_mean_logprobs"]
        legacy.release_model(base_model)
        del base_model

        pre_path = Path(receipt["checkpoint_paths"][0]["path"])
        pre_model, pre_tokenizer = legacy.load_model_and_tokenizer(
            pre_path,
            dtype=dtype,
            for_training=False,
            gradient_checkpointing=False,
        )
        pre_result = _evaluate_official_model(
            method=PRE_REPAIR_METHOD,
            model=pre_model,
            tokenizer=pre_tokenizer,
            target=target,
            datasets=datasets,
            args=args,
            frozen_probe=frozen_probe,
            base_retain=base_retain,
        )
        legacy.release_model(pre_model)
        del pre_model, pre_tokenizer

        repaired_path = Path(receipt["checkpoint_paths"][1]["path"])
        repaired_model, repaired_tokenizer = legacy.load_model_and_tokenizer(
            repaired_path,
            dtype=dtype,
            for_training=False,
            gradient_checkpointing=False,
        )
        repaired_result = _evaluate_official_model(
            method=METHOD,
            model=repaired_model,
            tokenizer=repaired_tokenizer,
            target=target,
            datasets=datasets,
            args=args,
            frozen_probe=frozen_probe,
            base_retain=base_retain,
        )
        legacy.release_model(repaired_model)
        del repaired_model, repaired_tokenizer
        gc.collect()
        torch.cuda.empty_cache()

        raw_result = {
            "schema_version": "rwku_setting5e_uc_official_evaluation_v1",
            "method": METHOD,
            "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
            "protocol_status": PROTOCOL_STATUS,
            "development": bool(args.development),
            "confirmatory": bool(args.confirmatory),
            "official_evaluation_opened_at_utc": opened[
                "official_evaluation_opened_at_utc"
            ],
            "metric_definitions_changed_after_open": False,
            "checkpoint_modified_after_open": False,
            "base": base_result,
            "selected_utility_controlled_setting5_before_repair": pre_result,
            "selected_setting5_uc_protected_rowwise_repair": repaired_result,
        }
        normalized, replacements = strict_json_normalize(raw_result)
        normalized["serialization"] = {
            "policy": "non_finite_numeric_values_to_json_null",
            "strict_json_allow_nan": False,
            "replacement_count": len(replacements),
            "replacements": replacements,
        }
        atomic_json_write(result_path, normalized)
        with result_path.open("r", encoding="utf-8") as handle:
            validated = json.load(handle)
        if validated.get("serialization", {}).get("replacement_count") != len(
            replacements
        ):
            raise ValueError("Strict evaluation serialization audit did not round-trip")
        verify_frozen_identities(load_receipt(receipt_path))
        completed = mark_evaluation_complete(
            receipt_path, experiment_id=args.experiment_id
        )
        write_state(
            args,
            "EVALUATION_COMPLETE",
            official_evaluation_opened=True,
            official_evaluation_opened_at_utc=opened[
                "official_evaluation_opened_at_utc"
            ],
            evaluation_completed_at_utc=completed["evaluation_completed_at_utc"],
            result_path=str(result_path.resolve()),
            result_sha256=sha256_file(result_path),
            serialization=normalized["serialization"],
        )
    except Exception as exc:
        diagnostic, replacements = strict_json_normalize(
            {
                "status": "official_evaluation_failed_after_open",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "state_preserved": "OFFICIAL_EVALUATION_OPENED",
                "checkpoint_modified": False,
                "failed_at_utc": utc_now(),
            }
        )
        diagnostic["serialization_replacements"] = replacements
        atomic_json_write(run_dir(args) / "evaluation_failure.json", diagnostic)
        raise


def parse_float_list(value: str) -> Tuple[float, ...]:
    values = tuple(
        float(part.strip()) for part in str(value).split(",") if part.strip()
    )
    if not values:
        raise argparse.ArgumentTypeError(
            "Expected a non-empty comma-separated float list"
        )
    return values


def parse_int_list(value: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError(
            "Expected a non-empty comma-separated integer list"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("prepare", "protection", "train", "evaluate"), required=True
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--generated-entity-fact-bundle", type=Path, required=True)
    parser.add_argument("--generator-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--wikidata-dir", type=Path, required=True)
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32", "bfloat16", "float16", "float32"),
        default="bf16",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--development", action="store_true")
    mode.add_argument("--confirmatory", action="store_true")
    parser.add_argument("--frozen-development-config", type=Path)
    parser.add_argument("--mcf-path", type=Path)
    parser.add_argument("--protection-source", type=Path, action="append", default=[])
    parser.add_argument("--protection-vocabulary", type=Path)
    parser.add_argument(
        "--tokenize-protection-rows",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum-protection-train-per-key", type=int, default=1)
    parser.add_argument("--minimum-protection-gate-per-key", type=int, default=1)
    parser.add_argument("--mcf-optimization-count", type=int, default=128)
    parser.add_argument("--mcf-gate-count", type=int, default=128)
    parser.add_argument(
        "--subject-input-delta-lr",
        "--subject-input-lr",
        dest="subject_input_delta_lr",
        type=float,
        default=2e-4,
    )
    parser.add_argument(
        "--sensitive-output-delta-lr",
        "--sensitive-output-lr",
        dest="sensitive_output_delta_lr",
        type=float,
        default=5e-4,
    )
    parser.add_argument("--forget-weight", type=float, default=8.0)
    parser.add_argument("--retain-ce-weight", type=float, default=2.0)
    parser.add_argument("--retain-kl-weight", type=float, default=4.0)
    parser.add_argument("--protected-margin-weight", type=float, default=10.0)
    parser.add_argument("--delta-l2-weight", type=float, default=1e-4)
    parser.add_argument("--forget-margin", type=float, default=1.0)
    parser.add_argument("--forget-token-margin", type=float, default=2.0)
    parser.add_argument("--probability-cap-weight", type=float, default=2.0)
    parser.add_argument("--eos-margin-weight", type=float, default=0.25)
    parser.add_argument("--retain-batch-size", type=int, default=4)
    parser.add_argument("--teacher-top-k", type=int, default=128)
    parser.add_argument("--max-retain-document-frequency", type=float, default=0.01)
    parser.add_argument(
        "--exposures-per-fact", type=parse_int_list, default=DEFAULT_EXPOSURES
    )
    parser.add_argument(
        "--candidate-scales",
        type=parse_float_list,
        default=DEFAULT_INTERPOLATION_SCALES,
    )
    parser.add_argument("--candidate-eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--repair-steps", type=int, default=200)
    parser.add_argument("--repair-lr", type=float, default=1e-2)
    parser.add_argument("--repair-margin", type=float, default=0.25)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--no-download", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    validate_mode(args)
    if args.exposures_per_fact != tuple(sorted(set(args.exposures_per_fact))):
        raise ValueError("--exposures-per-fact must be unique and increasing")
    if tuple(args.candidate_scales) != DEFAULT_INTERPOLATION_SCALES:
        raise ValueError(
            "Primary method requires candidate scales "
            "0.015625,0.03125,0.0625,0.125,0.25,0.50,0.75,1.00"
        )
    positive_names = (
        "subject_input_delta_lr",
        "sensitive_output_delta_lr",
        "forget_weight",
        "retain_ce_weight",
        "retain_kl_weight",
        "protected_margin_weight",
        "delta_l2_weight",
        "probability_cap_weight",
        "eos_margin_weight",
        "retain_batch_size",
        "teacher_top_k",
        "candidate_eval_batch_size",
        "repair_steps",
        "repair_lr",
        "eval_batch_size",
    )
    for name in positive_names:
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.forget_token_margin < 0 or args.forget_margin < 0:
        raise ValueError("Forget token/EOS margins must be non-negative")
    if not 0.0 <= args.max_retain_document_frequency <= 1.0:
        raise ValueError("--max-retain-document-frequency must be in [0,1]")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.no_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "protection":
        protection_stage(args)
    elif args.stage == "train":
        if not torch.cuda.is_available():
            raise RuntimeError("Utility-controlled RWKU training requires CUDA")
        train_stage(args)
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("Official RWKU evaluation requires CUDA")
        evaluate_stage(args)


if __name__ == "__main__":
    main()
