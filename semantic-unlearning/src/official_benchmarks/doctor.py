"""CPU-only source, artifact, identity, and compatibility audit."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .provenance import load_json_if_present, unresolved
from .registry import PROJECT_ROOT


READY_STATUSES = frozenset({"READY_NATIVE", "READY_WITH_DATA_ADAPTER"})
GENERIC_FORBIDDEN_PREFIXES = (
    "tofu_",
    "muse_",
    "ugbench_",
    "pch_",
    "hubble_",
)


class DoctorError(ValueError):
    """Raised for an invalid official artifact mapping."""


def expand_path(value: object) -> Optional[Path]:
    if value is None:
        return None
    text = os.path.expandvars(str(value))
    if unresolved(text) or "${" in text:
        return None
    return Path(text).expanduser().resolve()


def _same_path(left: object, right: object) -> bool:
    left_path = expand_path(left)
    right_path = expand_path(right)
    return left_path is not None and right_path is not None and left_path == right_path


def validate_official_model(
    track: Mapping[str, Any],
    model_entry: Mapping[str, Any],
    *,
    generic_model_path: object = None,
) -> None:
    benchmark_id = str(track["id"])
    role = model_entry.get("role")
    required_roles = set(track.get("required_models") or [])
    if role not in required_roles:
        raise DoctorError(
            f"{benchmark_id} model role {role!r} does not satisfy required roles "
            f"{sorted(required_roles)}"
        )
    if benchmark_id.startswith(GENERIC_FORBIDDEN_PREFIXES):
        if _same_path(model_entry.get("path"), generic_model_path):
            raise DoctorError(
                f"{benchmark_id} requires its official Full/Target model; the generic "
                "model is forbidden"
            )
        model_id = str(model_entry.get("id") or "").lower()
        if model_id in {
            "meta-llama/llama-3.2-3b-instruct",
            "llama-3.2-3b-instruct",
            "generic",
        }:
            raise DoctorError(
                f"{benchmark_id} cannot use a generic model ID as its official target"
            )
    available_roles = {str(role)} | {
        str(name) for name in (model_entry.get("comparison_models") or {})
    }
    missing_roles = required_roles - available_roles
    if missing_roles:
        raise DoctorError(
            f"{benchmark_id} is missing required comparison model roles: "
            f"{sorted(missing_roles)}"
        )


def _is_pinned_revision(entry: Optional[Mapping[str, Any]]) -> bool:
    if not entry:
        return False
    revision = entry.get("revision")
    if unresolved(revision):
        return False
    revision_text = str(revision)
    lock_type = entry.get("lock_type", "commit")
    if lock_type == "commit":
        return bool(re.fullmatch(r"[0-9a-fA-F]{40}", revision_text))
    if lock_type in {"tag", "dataset_revision", "paper"}:
        return revision_text.lower() not in {"main", "master", "latest"}
    return False


def _model_has_weights(path: Path) -> bool:
    patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in patterns)


def _model_artifact_ok(model_entry: Mapping[str, Any], *, tokenizer_required: bool) -> bool:
    model_path = expand_path(model_entry.get("path"))
    identity_ok = bool(
        not unresolved(model_entry.get("id"))
        and not unresolved(model_entry.get("revision"))
    )
    if tokenizer_required:
        identity_ok = bool(
            identity_ok
            and not unresolved(model_entry.get("architecture"))
            and not unresolved((model_entry.get("tokenizer") or {}).get("id"))
            and not unresolved((model_entry.get("tokenizer") or {}).get("revision"))
        )
    return bool(
        identity_ok
        and model_path
        and model_path.is_dir()
        and (model_path / "config.json").is_file()
        and _model_has_weights(model_path)
    )


def _git_revision(path: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(path),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _entry(mapping: Mapping[str, Any], collection: str, key: str) -> Mapping[str, Any]:
    value = mapping.get(collection, {}).get(key, {})
    return value if isinstance(value, dict) else {}


def audit_track(
    track: Mapping[str, Any],
    *,
    models: Mapping[str, Any],
    source_locks: Mapping[str, Any],
    official_bench_root: Path,
    generic_model_path: object = None,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str, blocker: Optional[str] = None) -> None:
        checks.append(
            {"name": name, "ok": bool(ok), "detail": detail, "blocker": blocker}
        )

    compatibility = track["method_status"]
    compatible = compatibility in READY_STATUSES
    check(
        "method_compatibility",
        compatible,
        track["method_reason"],
        None if compatible else compatibility,
    )

    local_files = track.get("method_entrypoints") or []
    missing_local = [relative for relative in local_files if not (PROJECT_ROOT / relative).is_file()]
    check(
        "method_entrypoints",
        not missing_local,
        "all local entrypoints present" if not missing_local else f"missing: {missing_local}",
        "BLOCKED_MISSING_EVALUATOR" if missing_local else None,
    )

    source_entry = _entry(source_locks, "sources", track["source_lock_id"])
    source_pinned = _is_pinned_revision(source_entry)
    check(
        "source_lock",
        source_pinned,
        f"{track['source_lock_id']} revision={source_entry.get('revision')!r}",
        "BLOCKED_UNPINNED_SOURCE" if not source_pinned else None,
    )
    source_path = expand_path(source_entry.get("path"))
    if source_path is None and source_entry.get("checkout"):
        source_path = (official_bench_root / str(source_entry["checkout"])).resolve()
    source_exists = bool(source_path and source_path.is_dir())
    check(
        "source_checkout",
        source_exists,
        str(source_path) if source_path else "unresolved checkout",
        "BLOCKED_UNPINNED_SOURCE" if not source_exists else None,
    )
    if source_exists and source_pinned and source_entry.get("lock_type", "commit") == "commit":
        observed_revision = _git_revision(source_path)  # type: ignore[arg-type]
        expected_revision = source_entry.get("revision")
        matches = observed_revision == expected_revision
        check(
            "source_checkout_revision",
            matches,
            f"expected={expected_revision}, observed={observed_revision}",
            "BLOCKED_UNPINNED_SOURCE" if not matches else None,
        )

    dataset_entry = _entry(source_locks, "datasets", track["dataset"]["lock_id"])
    dataset_pinned = _is_pinned_revision(dataset_entry)
    dataset_fingerprint = dataset_entry.get("fingerprint")
    dataset_identity_ok = dataset_pinned and not unresolved(dataset_fingerprint)
    check(
        "dataset_identity",
        dataset_identity_ok,
        f"revision={dataset_entry.get('revision')!r}, fingerprint={dataset_fingerprint!r}",
        "BLOCKED_MISSING_DATA" if not dataset_identity_ok else None,
    )
    dataset_path = expand_path(dataset_entry.get("path"))
    dataset_exists = bool(dataset_path and dataset_path.exists())
    check(
        "dataset_artifact",
        dataset_exists,
        str(dataset_path) if dataset_path else "unresolved dataset path",
        "BLOCKED_MISSING_DATA" if not dataset_exists else None,
    )

    evaluator_lock_id = track["official_evaluator"]["lock_id"]
    evaluator_entry = _entry(source_locks, "evaluators", evaluator_lock_id)
    evaluator_pinned = _is_pinned_revision(evaluator_entry)
    check(
        "evaluator_lock",
        evaluator_pinned,
        f"{evaluator_lock_id} revision={evaluator_entry.get('revision')!r}",
        "BLOCKED_MISSING_EVALUATOR" if not evaluator_pinned else None,
    )
    evaluator_path = expand_path(evaluator_entry.get("path"))
    if evaluator_path is None and evaluator_entry.get("checkout"):
        evaluator_path = (official_bench_root / str(evaluator_entry["checkout"])).resolve()
    evaluator_exists = bool(evaluator_path and evaluator_path.exists())
    check(
        "evaluator_artifact",
        evaluator_exists,
        str(evaluator_path) if evaluator_path else "unresolved evaluator path",
        "BLOCKED_MISSING_EVALUATOR" if not evaluator_exists else None,
    )

    model_entry = _entry(models, "tracks", track["id"])
    model_ok = bool(model_entry)
    model_detail = "model mapping absent"
    if model_entry:
        try:
            validate_official_model(
                track,
                model_entry,
                generic_model_path=generic_model_path,
            )
            model_path = expand_path(model_entry.get("path"))
            model_ok = _model_artifact_ok(model_entry, tokenizer_required=True)
            comparison_models = model_entry.get("comparison_models") or {}
            model_ok = bool(
                model_ok
                and all(
                    _model_artifact_ok(comparison, tokenizer_required=True)
                    for comparison in comparison_models.values()
                )
            )
            model_detail = str(model_path) if model_path else "unresolved model path"
        except DoctorError as exc:
            model_ok = False
            model_detail = str(exc)
    check(
        "target_model_identity",
        model_ok,
        model_detail,
        "BLOCKED_MISSING_TARGET_MODEL" if not model_ok else None,
    )

    blockers = [item["blocker"] for item in checks if not item["ok"] and item["blocker"]]
    if compatibility not in READY_STATUSES:
        effective_status = compatibility
    else:
        priority = (
            "BLOCKED_UNPINNED_SOURCE",
            "BLOCKED_MISSING_TARGET_MODEL",
            "BLOCKED_MISSING_DATA",
            "BLOCKED_MISSING_EVALUATOR",
        )
        effective_status = compatibility
        for candidate in priority:
            if candidate in blockers:
                effective_status = candidate
                break
    return {
        "benchmark_id": track["id"],
        "declared_method_status": compatibility,
        "effective_status": effective_status,
        "official_ready": effective_status in READY_STATUSES and all(
            item["ok"] for item in checks
        ),
        "checks": checks,
    }


def run_doctor(
    tracks: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    models_path: Optional[Path] = None,
    source_lock_path: Optional[Path] = None,
) -> Dict[str, Any]:
    models_path = models_path or Path(
        os.environ.get(
            "OFFICIAL_MODELS_CONFIG",
            PROJECT_ROOT / "config" / "official_benchmarks" / "models.json",
        )
    )
    source_lock_path = source_lock_path or Path(
        os.environ.get(
            "OFFICIAL_SOURCE_LOCK",
            PROJECT_ROOT / "config" / "official_benchmarks" / "source_lock.json",
        )
    )
    models = load_json_if_present(models_path)
    source_locks = load_json_if_present(source_lock_path)
    official_bench_root = Path(
        os.environ.get("OFFICIAL_BENCH_ROOT", "/mnt/train/official-unlearning-benchmarks")
    ).expanduser()
    generic_model_path = os.environ.get(
        "GENERIC_MODEL_PATH", "/mnt/train/models/Llama-3.2-3B-Instruct"
    )
    reports = [
        audit_track(
            track,
            models=models,
            source_locks=source_locks,
            official_bench_root=official_bench_root,
            generic_model_path=generic_model_path,
        )
        for track in tracks
    ]
    payload = {
        "schema_version": 1,
        "cpu_only": True,
        "torch_imported": "torch" in __import__("sys").modules,
        "models_config": str(models_path),
        "source_lock": str(source_lock_path),
        "official_bench_root": str(official_bench_root),
        "tracks": reports,
        "summary": {
            "official_ready": sum(bool(item["official_ready"]) for item in reports),
            "total": len(reports),
            "effective_status_counts": {
                status: sum(item["effective_status"] == status for item in reports)
                for status in sorted({item["effective_status"] for item in reports})
            },
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "doctor.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "doctor.md").open("w", encoding="utf-8") as handle:
        handle.write("# Official benchmark doctor\n\n")
        handle.write("| Track | Declared | Effective | Official ready |\n")
        handle.write("| --- | --- | --- | :---: |\n")
        for item in reports:
            handle.write(
                f"| {item['benchmark_id']} | {item['declared_method_status']} | "
                f"{item['effective_status']} | {'yes' if item['official_ready'] else 'no'} |\n"
            )
    return payload
