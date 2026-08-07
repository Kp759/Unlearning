#!/usr/bin/env python3
"""Fail-closed checkpoint receipts and RWKU experiment state transitions."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence

from rwku_artifact_access import (
    TARGET_ONLY_PROTOCOL_LABEL,
    canonical_json_bytes,
    read_artifact,
    sha256_file,
    sha256_json,
    sha256_path,
)


RECEIPT_SCHEMA_VERSION = "rwku_checkpoint_receipt_v1"
STATES = (
    "PREPARED",
    "TRAINING",
    "CHECKPOINT_FROZEN",
    "OFFICIAL_EVALUATION_OPENED",
    "EVALUATION_COMPLETE",
)


class CheckpointReceiptError(ValueError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _receipt_digest(receipt: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in receipt.items() if key != "receipt_sha256"})


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    # Reuse the artifact module's durable public behavior without making the
    # receipt an artifact role.  A sibling temp file and os.replace are atomic.
    import os
    import tempfile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent)
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


def _identity(path: Path) -> Dict[str, str]:
    target = Path(path)
    return {"path": str(target.resolve()), "sha256": sha256_path(target)}


def create_checkpoint_receipt(
    *,
    destination: Path,
    experiment_id: str,
    protocol_label: str,
    protocol_status: str,
    target_entity: str,
    target_entity_id: str,
    base_model_identity: Mapping[str, Any],
    base_model_revision: str,
    tokenizer_identity: Mapping[str, Any],
    checkpoint_paths: Sequence[Path],
    training_bundle_path: Path,
    optimization_protection_path: Path | None,
    mcf_retain_optimization_paths: Sequence[Path],
    mcf_repair_gate_paths: Sequence[Path],
    matched_protection_train_path: Path | None,
    matched_protection_gate_path: Path | None,
    method_configuration: Mapping[str, Any],
    implementation_files: Sequence[Path],
    sampler_provenance: Mapping[str, Any],
    generator_receipt_path: Path | None,
    official_locked_eval_path: Path,
    confirmatory: bool,
    additional_artifact_paths: Mapping[str, Path] | None = None,
) -> Dict[str, Any]:
    training = read_artifact(
        training_bundle_path,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    if training["protocol_label"] != protocol_label:
        raise CheckpointReceiptError("Training bundle protocol label differs from run")
    if protocol_label == TARGET_ONLY_PROTOCOL_LABEL and generator_receipt_path is None:
        raise CheckpointReceiptError("Target-only checkpoint receipt requires generator receipt")
    checkpoints = [_identity(path) for path in checkpoint_paths]
    if not checkpoints:
        raise CheckpointReceiptError("At least one frozen checkpoint path is required")
    artifacts: Dict[str, Any] = {
        "training_bundle": _identity(training_bundle_path),
        "optimization_protection": _identity(optimization_protection_path) if optimization_protection_path else None,
        "mcf_retain_optimization": [_identity(path) for path in mcf_retain_optimization_paths],
        "mcf_repair_gate": [_identity(path) for path in mcf_repair_gate_paths],
        "matched_protection_train": _identity(matched_protection_train_path) if matched_protection_train_path else None,
        "matched_protection_gate": _identity(matched_protection_gate_path) if matched_protection_gate_path else None,
        "generator_receipt": _identity(generator_receipt_path) if generator_receipt_path else None,
        "official_locked_eval": _identity(official_locked_eval_path),
    }
    for name, path in (additional_artifact_paths or {}).items():
        key = str(name).strip()
        if not key or key in artifacts:
            raise CheckpointReceiptError(
                f"Invalid or duplicate additional checkpoint artifact name: {name!r}"
            )
        artifacts[key] = _identity(Path(path))
    completed = utc_now()
    receipt: Dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "protocol_label": protocol_label,
        "protocol_status": protocol_status,
        "target_entity": target_entity,
        "target_entity_id": target_entity_id,
        "base_model_identity": dict(base_model_identity),
        "base_model_revision": base_model_revision,
        "tokenizer_identity": dict(tokenizer_identity),
        "checkpoint_paths": checkpoints,
        "training_bundle_sha256": artifacts["training_bundle"]["sha256"],
        "optimization_protection_sha256": artifacts["optimization_protection"]["sha256"] if artifacts["optimization_protection"] else None,
        "mcf_retain_optimization_hashes": [row["sha256"] for row in artifacts["mcf_retain_optimization"]],
        "mcf_repair_gate_hashes": [row["sha256"] for row in artifacts["mcf_repair_gate"]],
        "matched_protection_train_sha256": artifacts["matched_protection_train"]["sha256"] if artifacts["matched_protection_train"] else None,
        "matched_protection_gate_sha256": artifacts["matched_protection_gate"]["sha256"] if artifacts["matched_protection_gate"] else None,
        "method_configuration": dict(method_configuration),
        "method_configuration_sha256": sha256_json(method_configuration),
        "implementation_file_sha256": {
            str(Path(path).resolve()): sha256_file(Path(path)) for path in implementation_files
        },
        "generator_receipt_sha256": artifacts["generator_receipt"]["sha256"] if artifacts["generator_receipt"] else None,
        "sampler_provenance": dict(sampler_provenance),
        "artifacts": artifacts,
        "training_completed_timestamp_utc": completed,
        "checkpoint_frozen_timestamp_utc": completed,
        "official_evaluation_opened": False,
        "official_evaluation_opened_at_utc": None,
        "evaluation_completed_at_utc": None,
        "state": "CHECKPOINT_FROZEN",
        "confirmatory": bool(confirmatory),
        "confirmatory_status": "confirmatory_not_yet_opened" if confirmatory else "exploratory",
    }
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    _atomic_write(destination, receipt)
    return receipt


def load_receipt(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise CheckpointReceiptError("Unsupported checkpoint receipt schema")
    if value.get("state") not in STATES:
        raise CheckpointReceiptError(f"Unknown experiment state: {value.get('state')}")
    if value.get("receipt_sha256") != _receipt_digest(value):
        raise CheckpointReceiptError("Checkpoint receipt integrity hash mismatch")
    return value


def _verify_identity(identity: Mapping[str, Any], label: str) -> None:
    path = Path(str(identity["path"]))
    actual = sha256_path(path)
    if actual != identity.get("sha256"):
        raise CheckpointReceiptError(
            f"{label} identity changed after freeze: declared={identity.get('sha256')}, actual={actual}"
        )


def verify_frozen_identities(receipt: Mapping[str, Any]) -> None:
    for index, checkpoint in enumerate(receipt["checkpoint_paths"]):
        _verify_identity(checkpoint, f"checkpoint[{index}]")
    for name, identity in receipt["artifacts"].items():
        if identity is None:
            continue
        if isinstance(identity, list):
            for index, item in enumerate(identity):
                _verify_identity(item, f"artifact {name}[{index}]")
        else:
            _verify_identity(identity, f"artifact {name}")
    for path_string, expected in receipt["implementation_file_sha256"].items():
        actual = sha256_file(Path(path_string))
        if actual != expected:
            raise CheckpointReceiptError(
                f"Method implementation changed after freeze: {path_string}"
            )
    if sha256_json(receipt["method_configuration"]) != receipt["method_configuration_sha256"]:
        raise CheckpointReceiptError("Method configuration identity mismatch")


def open_official_evaluation(
    receipt_path: Path,
    *,
    experiment_id: str,
) -> Dict[str, Any]:
    """Verify all identities and atomically cross the one-way data boundary."""

    receipt = load_receipt(receipt_path)
    if receipt["experiment_id"] != experiment_id:
        raise CheckpointReceiptError("Receipt experiment ID does not match invocation")
    if receipt["state"] != "CHECKPOINT_FROZEN":
        raise CheckpointReceiptError(
            f"Official evaluation requires CHECKPOINT_FROZEN, got {receipt['state']}"
        )
    if receipt["official_evaluation_opened"] is not False:
        raise CheckpointReceiptError("Official evaluation was already opened")
    verify_frozen_identities(receipt)
    receipt["official_evaluation_opened"] = True
    receipt["official_evaluation_opened_at_utc"] = utc_now()
    receipt["state"] = "OFFICIAL_EVALUATION_OPENED"
    if receipt["confirmatory"]:
        receipt["confirmatory_status"] = "confirmatory_official_evaluation_opened"
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    _atomic_write(receipt_path, receipt)
    return receipt


def mark_evaluation_complete(receipt_path: Path, *, experiment_id: str) -> Dict[str, Any]:
    receipt = load_receipt(receipt_path)
    if receipt["experiment_id"] != experiment_id:
        raise CheckpointReceiptError("Receipt experiment ID does not match invocation")
    if receipt["state"] != "OFFICIAL_EVALUATION_OPENED":
        raise CheckpointReceiptError("Evaluation can complete only after official data opening")
    verify_frozen_identities(receipt)
    receipt["state"] = "EVALUATION_COMPLETE"
    receipt["evaluation_completed_at_utc"] = utc_now()
    receipt["confirmatory_status"] = "confirmatory_complete" if receipt["confirmatory"] else "exploratory_complete"
    receipt["receipt_sha256"] = _receipt_digest(receipt)
    _atomic_write(receipt_path, receipt)
    return receipt


def assert_model_modification_allowed(receipt_path: Path, *, experiment_id: str) -> None:
    receipt = load_receipt(receipt_path)
    if receipt["experiment_id"] != experiment_id:
        return
    if receipt["state"] in {
        "CHECKPOINT_FROZEN",
        "OFFICIAL_EVALUATION_OPENED",
        "EVALUATION_COMPLETE",
    }:
        raise CheckpointReceiptError(
            "Model modification, repair-scale changes, and checkpoint replacement are "
            "forbidden after checkpoint freeze; start a new experiment ID. "
            "The new run is not confirmatory with respect to already observed results."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    opened = subparsers.add_parser("open-evaluation")
    opened.add_argument("--receipt", type=Path, required=True)
    opened.add_argument("--experiment-id", required=True)
    complete = subparsers.add_parser("complete-evaluation")
    complete.add_argument("--receipt", type=Path, required=True)
    complete.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    if args.command == "verify":
        receipt = load_receipt(args.receipt)
        verify_frozen_identities(receipt)
    elif args.command == "open-evaluation":
        receipt = open_official_evaluation(args.receipt, experiment_id=args.experiment_id)
    else:
        receipt = mark_evaluation_complete(args.receipt, experiment_id=args.experiment_id)
    print(json.dumps({"experiment_id": receipt["experiment_id"], "state": receipt["state"]}, indent=2))


if __name__ == "__main__":
    main()
