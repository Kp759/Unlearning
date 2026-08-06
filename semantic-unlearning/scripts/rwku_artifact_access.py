#!/usr/bin/env python3
"""RWKU artifact envelopes and enforced stage/permission checks.

This module intentionally uses only the Python standard library.  It is safe
to import in CPU-only preparation and generator dry-run processes.  Artifact
hashes cover the immutable payload and permission metadata; the ``sha256``
field itself is excluded from the digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence


ARTIFACT_SCHEMA_VERSION = "rwku_artifact_v1"

PROBE_PROTOCOL_LABEL = "rwku_probe_assisted_entity_fact_portability_nonofficial"
PROBE_PROTOCOL_STATUS = "nonofficial_probe_assisted_entity_fact_portability"
TARGET_ONLY_PROTOCOL_LABEL = (
    "rwku_target_only_generated_entity_corpus_method_extension"
)
TARGET_ONLY_PROTOCOL_STATUS = (
    "official_protocol_different_model_confirmatory_method_extension"
)
EVALUATION_CONDITIONED_REPAIR_STATUS = (
    "native_data_and_metrics_but_evaluation_conditioned_repair"
)
LEGACY_PROTOCOL_STATUS = "prompt_held_out_only_legacy_nonofficial"


ROLE_PERMISSIONS: Mapping[str, Mapping[str, Any]] = {
    "fact_catalog": {
        "gradient_allowed": False,
        "selection_allowed": False,
        "evaluation_only": False,
        "allowed_stages": ["prepare"],
    },
    "split_manifest": {
        "gradient_allowed": False,
        "selection_allowed": False,
        "evaluation_only": False,
        "allowed_stages": ["prepare", "train", "evaluate"],
    },
    "generator_receipt": {
        "gradient_allowed": False,
        "selection_allowed": False,
        "evaluation_only": False,
        "allowed_stages": ["prepare", "train", "evaluate"],
    },
    "training_bundle": {
        "gradient_allowed": True,
        "selection_allowed": False,
        "evaluation_only": False,
        "allowed_stages": ["train"],
    },
    "optimization_protection": {
        "gradient_allowed": True,
        "selection_allowed": False,
        "evaluation_only": False,
        "allowed_stages": ["train"],
    },
    "repair_selection_gate": {
        "gradient_allowed": False,
        "selection_allowed": True,
        "evaluation_only": False,
        "allowed_stages": ["train"],
    },
    "seen_fact_unseen_prompt_eval": {
        "gradient_allowed": False,
        "selection_allowed": False,
        "evaluation_only": True,
        "allowed_stages": ["evaluate"],
    },
    "unseen_fact_eval": {
        "gradient_allowed": False,
        "selection_allowed": False,
        "evaluation_only": True,
        "allowed_stages": ["evaluate"],
    },
    "official_locked_eval": {
        "gradient_allowed": False,
        "selection_allowed": False,
        "evaluation_only": True,
        "allowed_stages": ["evaluate"],
    },
    "matched_protection_coverage": {
        "gradient_allowed": False,
        "selection_allowed": False,
        "evaluation_only": False,
        "allowed_stages": ["prepare", "train"],
    },
}


class ArtifactAccessError(ValueError):
    """Raised when an artifact is corrupt or used outside its declared role."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one file or a directory tree including relative file names."""

    target = Path(path)
    if target.is_file():
        return sha256_file(target)
    if not target.is_dir():
        raise FileNotFoundError(f"Cannot hash missing path: {target}")
    files = sorted(candidate for candidate in target.rglob("*") if candidate.is_file())
    if not files:
        raise ValueError(f"Cannot hash empty directory: {target}")
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(target).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def _digestable_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in artifact.items() if key != "sha256"}


def make_artifact(
    role: str,
    payload: Any,
    *,
    protocol_label: str,
    protocol_status: str,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if role not in ROLE_PERMISSIONS:
        raise ArtifactAccessError(f"Unknown RWKU artifact role: {role}")
    permissions = dict(ROLE_PERMISSIONS[role])
    permissions["allowed_stages"] = list(permissions["allowed_stages"])
    artifact: Dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_role": role,
        "protocol_label": str(protocol_label),
        "protocol_status": str(protocol_status),
        **permissions,
        "metadata": dict(metadata or {}),
        "payload": payload,
    }
    artifact["sha256"] = sha256_json(_digestable_artifact(artifact))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactAccessError(
            f"Unsupported RWKU artifact schema: {artifact.get('schema_version')!r}"
        )
    role = str(artifact.get("artifact_role", ""))
    if role not in ROLE_PERMISSIONS:
        raise ArtifactAccessError(f"Unknown RWKU artifact role: {role!r}")
    expected_permissions = ROLE_PERMISSIONS[role]
    for field in ("gradient_allowed", "selection_allowed", "evaluation_only"):
        if artifact.get(field) is not expected_permissions[field]:
            raise ArtifactAccessError(
                f"Artifact {role} changes immutable permission {field}"
            )
    if list(artifact.get("allowed_stages", [])) != list(
        expected_permissions["allowed_stages"]
    ):
        raise ArtifactAccessError(
            f"Artifact {role} changes immutable allowed_stages"
        )
    declared = str(artifact.get("sha256", ""))
    actual = sha256_json(_digestable_artifact(artifact))
    if declared != actual:
        raise ArtifactAccessError(
            f"Artifact payload hash mismatch: declared={declared}, actual={actual}"
        )
    return dict(artifact)


def assert_artifact_access(
    artifact: Mapping[str, Any],
    *,
    stage: str,
    gradient: bool = False,
    selection: bool = False,
    evaluation: bool = False,
) -> None:
    validated = validate_artifact(artifact)
    role = validated["artifact_role"]
    if stage not in validated["allowed_stages"]:
        raise ArtifactAccessError(
            f"Artifact role {role} is forbidden in stage {stage}; allowed="
            f"{validated['allowed_stages']}"
        )
    if stage == "train" and validated["evaluation_only"]:
        raise ArtifactAccessError(
            f"Evaluation-only artifact {role} cannot be opened during training"
        )
    if gradient and not validated["gradient_allowed"]:
        raise ArtifactAccessError(
            f"Artifact role {role} cannot participate in a backward pass"
        )
    if selection and not validated["selection_allowed"]:
        raise ArtifactAccessError(
            f"Artifact role {role} cannot select a checkpoint or repair scale"
        )
    if evaluation and not validated["evaluation_only"]:
        raise ArtifactAccessError(
            f"Artifact role {role} is not an evaluation artifact"
        )


def _atomic_json_write(path: Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
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


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> str:
    validated = validate_artifact(artifact)
    _atomic_json_write(Path(path), validated)
    return sha256_file(Path(path))


def read_artifact(
    path: Path,
    *,
    stage: str,
    gradient: bool = False,
    selection: bool = False,
    evaluation: bool = False,
    expected_role: str | None = None,
) -> Dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ArtifactAccessError(f"RWKU artifact must be a JSON object: {source}")
    validated = validate_artifact(value)
    if expected_role is not None and validated["artifact_role"] != expected_role:
        raise ArtifactAccessError(
            f"Expected artifact role {expected_role}, got {validated['artifact_role']}"
        )
    assert_artifact_access(
        validated,
        stage=stage,
        gradient=gradient,
        selection=selection,
        evaluation=evaluation,
    )
    validated["_artifact_path"] = str(source.resolve())
    validated["_artifact_file_sha256"] = sha256_file(source)
    return validated


def assert_content_disjoint(
    left: Iterable[str],
    right: Iterable[str],
    *,
    left_name: str,
    right_name: str,
) -> None:
    overlap = set(left) & set(right)
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise ArtifactAccessError(
            f"{left_name} and {right_name} overlap by content hash: {preview}"
        )


def artifact_permissions(role: str) -> Dict[str, Any]:
    if role not in ROLE_PERMISSIONS:
        raise ArtifactAccessError(f"Unknown RWKU artifact role: {role}")
    result = dict(ROLE_PERMISSIONS[role])
    result["allowed_stages"] = list(result["allowed_stages"])
    return result


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArtifactAccessError",
    "EVALUATION_CONDITIONED_REPAIR_STATUS",
    "LEGACY_PROTOCOL_STATUS",
    "PROBE_PROTOCOL_LABEL",
    "PROBE_PROTOCOL_STATUS",
    "ROLE_PERMISSIONS",
    "TARGET_ONLY_PROTOCOL_LABEL",
    "TARGET_ONLY_PROTOCOL_STATUS",
    "artifact_permissions",
    "assert_artifact_access",
    "assert_content_disjoint",
    "canonical_json_bytes",
    "make_artifact",
    "read_artifact",
    "sha256_file",
    "sha256_json",
    "sha256_path",
    "validate_artifact",
    "write_artifact",
]
