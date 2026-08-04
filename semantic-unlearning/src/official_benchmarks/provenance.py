"""Run provenance and official-identity validation without heavyweight imports."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from .registry import PROJECT_ROOT


METHOD_IMPLEMENTATION_FILES = (
    "scripts/gagd_compare.py",
    "scripts/gagd_active_case_repair.py",
    "scripts/run_gagd_active_case_repair.sh",
    "scripts/tofu_gagd_four_settings_official.py",
    "scripts/tofu_gagd_setting5e_restore.py",
    "scripts/tofu_gagd_active_forget_repair.py",
    "scripts/run_tofu_gagd_neighborhood_confidence.sh",
    "scripts/zsre_gagd_setting5e_active_repair.py",
    "scripts/run_zsre_gagd_setting5e_active_repair.sh",
    "scripts/run_three_benchmark_experiments.sh",
)

UNRESOLVED_MARKERS = (
    "REPLACE_",
    "PIN_REQUIRED",
    "UNRESOLVED",
    "UNKNOWN",
    "<",
    ">",
    "${",
)


class ProvenanceError(ValueError):
    """Raised when an official manifest has unresolved or conflicting identity."""


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> Optional[str]:
    if not path.is_dir():
        return None
    files = sorted(
        candidate for candidate in path.rglob("*") if candidate.is_file()
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(candidate)))
    return digest.hexdigest()


def method_file_hashes(project_root: Path = PROJECT_ROOT) -> Dict[str, Optional[str]]:
    hashes: Dict[str, Optional[str]] = {}
    for relative in METHOD_IMPLEMENTATION_FILES:
        path = project_root / relative
        hashes[relative] = sha256_file(path) if path.is_file() else None
    return hashes


def _git(args: Sequence[str], cwd: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def repository_identity(project_root: Path = PROJECT_ROOT) -> Dict[str, Any]:
    commit = _git(("rev-parse", "HEAD"), project_root)
    status = _git(("status", "--porcelain"), project_root)
    return {
        "commit": commit,
        "dirty": None if status is None else bool(status),
        "status_porcelain": None if status is None else status.splitlines(),
    }


def dependency_versions() -> Dict[str, str]:
    packages = (
        "datasets",
        "huggingface-hub",
        "lm-eval",
        "numpy",
        "safetensors",
        "tokenizers",
        "torch",
        "transformers",
    )
    versions: Dict[str, str] = {"python": platform.python_version()}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def load_json_if_present(path: Path | None) -> Dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ProvenanceError(f"Expected a JSON object: {path}")
    return payload


def unresolved(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip()
        return not stripped or any(marker in stripped.upper() for marker in UNRESOLVED_MARKERS)
    return False


def model_identity(model_entry: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not model_entry:
        return {
            "id": None,
            "path": None,
            "revision": None,
            "architecture": None,
            "tokenizer": {"id": None, "revision": None},
            "role": None,
        }
    path_text = model_entry.get("path")
    path = Path(os.path.expandvars(str(path_text))).expanduser() if path_text else None
    config = load_json_if_present(path / "config.json" if path else None)
    architecture = model_entry.get("architecture")
    if not architecture:
        architectures = config.get("architectures")
        if isinstance(architectures, list) and architectures:
            architecture = architectures[0]
        else:
            architecture = config.get("model_type")
    return {
        "id": model_entry.get("id"),
        "path": None if path is None else str(path),
        "revision": model_entry.get("revision"),
        "architecture": architecture,
        "tokenizer": dict(model_entry.get("tokenizer") or {}),
        "role": model_entry.get("role"),
    }


def manifest_template(
    track: Mapping[str, Any],
    command: Sequence[str] | str,
    *,
    method: str,
    output_dir: Path,
    model_entry: Optional[Mapping[str, Any]] = None,
    source_lock: Optional[Mapping[str, Any]] = None,
    dataset_lock: Optional[Mapping[str, Any]] = None,
    evaluator_lock: Optional[Mapping[str, Any]] = None,
    status: str = "planned",
    failure_reason: Optional[str] = None,
) -> Dict[str, Any]:
    command_text = command if isinstance(command, str) else " ".join(command)
    source_lock = dict(source_lock or {})
    dataset_lock = dict(dataset_lock or {})
    evaluator_lock = dict(evaluator_lock or {})
    identity = model_identity(model_entry)
    comparison_identities = {
        str(role): model_identity({**dict(entry), "role": str(role)})
        for role, entry in ((model_entry or {}).get("comparison_models") or {}).items()
    }
    return {
        "schema_version": 1,
        "timestamp": utc_timestamp(),
        "repository": repository_identity(),
        "method_implementation_file_hashes": method_file_hashes(),
        "complete_command": command_text,
        "benchmark_id": track["id"],
        "benchmark_source": {
            "url": track["official_source"],
            "locked_revision": source_lock.get("revision"),
            "lock_id": track["source_lock_id"],
        },
        "evaluator": {
            "source_url": track["official_evaluator"]["source"],
            "locked_revision": evaluator_lock.get("revision"),
            "entrypoint": track["official_evaluator"]["entrypoint"],
        },
        "dataset": {
            "id": track["dataset"]["id"],
            "split": track["default_split"],
            "revision": dataset_lock.get("revision"),
            "fingerprint": dataset_lock.get("fingerprint"),
        },
        "model": identity,
        "comparison_models": comparison_identities,
        "model_role": identity.get("role"),
        "method": method,
        "method_configuration": dict(track.get("method_configuration") or {}),
        "method_hyperparameters": dict(track.get("method_hyperparameters") or {}),
        "seed": track["seed"],
        "dtype": track.get("dtype"),
        "device_information": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_device_map": track.get("device_map"),
            "gpu_initialized": False,
        },
        "dependency_versions": dependency_versions(),
        "output_directory": str(output_dir.resolve()),
        "output_checkpoint_hash": None,
        "native_metrics": [dict(metric) for metric in track["native_metrics"]],
        "metric_values": {},
        "official_protocol": False,
        "status": status,
        "failure_reason": failure_reason,
    }


def manifest_identity_errors(manifest: Mapping[str, Any]) -> List[str]:
    errors: List[str] = []
    required_paths = {
        "repository.commit": manifest.get("repository", {}).get("commit"),
        "benchmark_source.locked_revision": manifest.get("benchmark_source", {}).get("locked_revision"),
        "evaluator.locked_revision": manifest.get("evaluator", {}).get("locked_revision"),
        "dataset.id": manifest.get("dataset", {}).get("id"),
        "dataset.revision": manifest.get("dataset", {}).get("revision"),
        "dataset.fingerprint": manifest.get("dataset", {}).get("fingerprint"),
        "model.id": manifest.get("model", {}).get("id"),
        "model.path": manifest.get("model", {}).get("path"),
        "model.revision": manifest.get("model", {}).get("revision"),
        "model.architecture": manifest.get("model", {}).get("architecture"),
        "model.tokenizer.id": manifest.get("model", {}).get("tokenizer", {}).get("id"),
        "model.tokenizer.revision": manifest.get("model", {}).get("tokenizer", {}).get("revision"),
        "model_role": manifest.get("model_role"),
    }
    for label, value in required_paths.items():
        if unresolved(value):
            errors.append(f"unresolved {label}")
    for role, identity in (manifest.get("comparison_models") or {}).items():
        comparison_fields = {
            f"comparison_models.{role}.id": identity.get("id"),
            f"comparison_models.{role}.path": identity.get("path"),
            f"comparison_models.{role}.revision": identity.get("revision"),
            f"comparison_models.{role}.architecture": identity.get("architecture"),
            f"comparison_models.{role}.tokenizer.id": identity.get("tokenizer", {}).get("id"),
            f"comparison_models.{role}.tokenizer.revision": identity.get("tokenizer", {}).get("revision"),
        }
        for label, value in comparison_fields.items():
            if unresolved(value):
                errors.append(f"unresolved {label}")
    if not manifest.get("method_hyperparameters"):
        errors.append("method_hyperparameters are empty")
    if not manifest.get("native_metrics"):
        errors.append("native_metrics are empty")
    return errors


def validate_manifest(manifest: Mapping[str, Any], *, official: bool) -> None:
    errors = manifest_identity_errors(manifest)
    if official and errors:
        raise ProvenanceError(
            "Official manifest has unresolved identities: " + "; ".join(errors)
        )


def write_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
