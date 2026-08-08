#!/usr/bin/env python3
"""Read-only BF16 selected-logit drift audit for saved RWKU UC candidates.

The utility derives the Base model, dtype, protection gate, selected rows, and
gate configuration from an existing run. It never opens official RWKU data
and never writes a checkpoint, experiment state, or result artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import rwku_experiment as legacy
from rwku_artifact_access import read_artifact, sha256_file, sha256_json, sha256_path
from rwku_setting5e_utility_controlled import (
    _artifact_attests_official_access,
    _load_json_mapping,
    _prompt_distribution_metrics,
    reject_official_or_completed_path,
)


def validate_read_only_audit_state(state: Mapping[str, Any]) -> None:
    if state.get("official_evaluation_opened") is not False:
        raise ValueError("BF16 drift audit requires official evaluation to remain locked")
    if state.get("official_rwku_records_accessed") is not False:
        raise ValueError("BF16 drift audit cannot use a run that accessed official RWKU data")
    if not state.get("protection_prepared"):
        raise ValueError("BF16 drift audit requires the frozen pre-freeze protection gate")
    if state.get("state") not in {
        "CANDIDATES_EVALUATED",
        "NO_FEASIBLE_CANDIDATE",
        "CHECKPOINT_FROZEN",
    }:
        raise ValueError("Run state does not contain auditable pre-freeze candidates")


def _verified_path(state: Mapping[str, Any], path_field: str, hash_field: str) -> Path:
    path = Path(str(state[path_field])).resolve()
    reject_official_or_completed_path(path, label="BF16 drift audit protection gate")
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != state.get(hash_field):
        raise ValueError(f"Frozen audit artifact identity changed: {path}")
    return path


def load_pre_freeze_gate_records(
    state: Mapping[str, Any],
) -> Tuple[List[Mapping[str, Any]], Dict[str, str]]:
    """Load only the disjoint target-independent pre-freeze selection gate."""

    validate_read_only_audit_state(state)
    mcf_path = _verified_path(
        state, "mcf_gate_manifest_path", "mcf_gate_manifest_sha256"
    )
    matched_path = _verified_path(
        state, "matched_protection_gate_path", "matched_protection_gate_sha256"
    )
    mcf_gate = _load_json_mapping(mcf_path)
    matched_gate = read_artifact(
        matched_path,
        stage="train",
        selection=True,
        expected_role="repair_selection_gate",
    )
    if (
        mcf_gate.get("gradient_allowed") is not False
        or mcf_gate.get("selection_allowed") is not True
    ):
        raise ValueError("MCF audit gate permissions differ from the pre-freeze contract")
    if _artifact_attests_official_access(matched_gate) or mcf_gate.get(
        "official_rwku_records_accessed"
    ) is True:
        raise ValueError("Official RWKU data cannot enter the BF16 drift audit")
    records = [
        *mcf_gate.get("records", []),
        *matched_gate["payload"].get("records", []),
    ]
    if not records:
        raise ValueError("Frozen pre-freeze protection gate is empty")
    return records, {
        "mcf_gate_path": str(mcf_path),
        "mcf_gate_sha256": sha256_file(mcf_path),
        "matched_protection_gate_path": str(matched_path),
        "matched_protection_gate_sha256": sha256_file(matched_path),
    }


def _read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


def _candidate_checkpoint(run_dir: Path, value: Path) -> Path:
    checkpoint = Path(value).resolve()
    checkpoint_root = (
        Path(run_dir).resolve() / "utility_controlled_setting5" / "checkpoints"
    )
    if not checkpoint.is_relative_to(checkpoint_root):
        raise ValueError("Candidate checkpoint must be a saved pre-freeze run checkpoint")
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    return checkpoint


def _scale_selected_rows(
    candidate_model: torch.nn.Module,
    base_model: torch.nn.Module,
    *,
    input_row_ids: Sequence[int],
    output_row_ids: Sequence[int],
    scale: float,
) -> None:
    with torch.no_grad():
        for candidate_weight, base_weight, row_ids in (
            (
                candidate_model.get_input_embeddings().weight,
                base_model.get_input_embeddings().weight,
                input_row_ids,
            ),
            (
                candidate_model.get_output_embeddings().weight,
                base_model.get_output_embeddings().weight,
                output_row_ids,
            ),
        ):
            indices = torch.tensor(
                sorted({int(value) for value in row_ids}),
                dtype=torch.long,
                device=candidate_weight.device,
            )
            if not indices.numel():
                continue
            base_rows = base_weight.detach().index_select(
                0, indices.to(base_weight.device)
            ).to(candidate_weight.device, torch.float32)
            trained_rows = candidate_weight.detach().index_select(0, indices).float()
            values = base_rows + float(scale) * (trained_rows - base_rows)
            candidate_weight.index_copy_(0, indices, values.to(candidate_weight.dtype))


def audit(args: argparse.Namespace) -> Dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    state_path = run_dir / "experiment_state.json"
    state_bytes = state_path.read_bytes()
    state = _read_json(state_path)
    validate_read_only_audit_state(state)
    checkpoint = _candidate_checkpoint(run_dir, args.candidate_checkpoint)
    checkpoint_sha_before = sha256_path(checkpoint)
    model_path = Path(str(state["model_path"])).resolve()
    if sha256_path(model_path) != state.get("model_sha256"):
        raise ValueError("Frozen Base model identity changed")
    base_sha_before = sha256_path(model_path)
    configuration_path = Path(str(state["configuration_manifest_path"])).resolve()
    configuration = _read_json(configuration_path)["configuration"]
    if sha256_json(configuration) != state.get("configuration_sha256"):
        raise ValueError("Frozen utility-controlled configuration identity changed")
    allowed_scales = [
        float(value) for value in configuration["candidate_interpolation_scales"]
    ]
    if float(args.candidate_scale) not in allowed_scales:
        raise ValueError("Audit scale is not in the frozen candidate schedule")
    diagnostics_path = Path(str(state["training_diagnostics_path"])).resolve()
    if sha256_file(diagnostics_path) != state.get("training_diagnostics_sha256"):
        raise ValueError("Frozen training diagnostics identity changed")
    diagnostics = _read_json(diagnostics_path)
    input_rows = [int(value) for value in diagnostics["selected_input_row_ids"]]
    output_rows = [int(value) for value in diagnostics["selected_output_row_ids"]]
    gate_records, gate_identity = load_pre_freeze_gate_records(state)

    dtype = legacy.dtype_from_name(str(state["dtype"]))
    base_model, tokenizer = legacy.load_model_and_tokenizer(
        str(model_path),
        dtype=dtype,
        for_training=False,
        gradient_checkpointing=False,
    )
    candidate_model, candidate_tokenizer = legacy.load_model_and_tokenizer(
        str(checkpoint),
        dtype=dtype,
        for_training=False,
        gradient_checkpointing=False,
    )
    del candidate_tokenizer
    try:
        base_model.eval()
        candidate_model.eval()
        _scale_selected_rows(
            candidate_model,
            base_model,
            input_row_ids=input_rows,
            output_row_ids=output_rows,
            scale=float(args.candidate_scale),
        )
        metrics = _prompt_distribution_metrics(
            candidate_model,
            base_model,
            tokenizer,
            gate_records,
            selected_output_rows=output_rows,
            top_k=int(configuration["teacher_top_k"]),
        )
    finally:
        legacy.release_model(candidate_model)
        legacy.release_model(base_model)

    if state_path.read_bytes() != state_bytes:
        raise RuntimeError("Read-only BF16 audit changed experiment state")
    if sha256_path(checkpoint) != checkpoint_sha_before:
        raise RuntimeError("Read-only BF16 audit changed the candidate checkpoint")
    if sha256_path(model_path) != base_sha_before:
        raise RuntimeError("Read-only BF16 audit changed the Base model")
    return {
        "schema_version": "rwku_bf16_drift_audit_v1",
        "status": "read_only_complete",
        "official_rwku_records_accessed": False,
        "experiment_id": state["experiment_id"],
        "candidate_checkpoint": str(checkpoint),
        "candidate_checkpoint_sha256": checkpoint_sha_before,
        "candidate_scale": float(args.candidate_scale),
        "base_model_path": str(model_path),
        "base_model_sha256": base_sha_before,
        "selected_output_row_ids": output_rows,
        "protection_gate_record_count": len(gate_records),
        "protection_gate_identity": gate_identity,
        "checkpoint_modified": False,
        "experiment_state_modified": False,
        "metrics": metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-scale", type=float, default=1.0)
    parser.add_argument("--no-download", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.no_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    result = audit(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
