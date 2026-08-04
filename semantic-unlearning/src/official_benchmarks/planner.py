"""CPU-only exact command planning and manifest reservation."""

from __future__ import annotations

import json
import os
import shlex
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .provenance import load_json_if_present, manifest_template, write_manifest
from .registry import PROJECT_ROOT


READY_STATUSES = frozenset({"READY_NATIVE", "READY_WITH_DATA_ADAPTER"})


def _model_entry(models: Mapping[str, Any], benchmark_id: str) -> Mapping[str, Any]:
    value = models.get("tracks", {}).get(benchmark_id, {})
    return value if isinstance(value, dict) else {}


def _lock_entry(locks: Mapping[str, Any], collection: str, lock_id: str) -> Mapping[str, Any]:
    value = locks.get(collection, {}).get(lock_id, {})
    return value if isinstance(value, dict) else {}


def default_model_expression(track: Mapping[str, Any]) -> str:
    benchmark_id = track["id"]
    if benchmark_id in {"mcf_zerounlearn_official", "zsre_zerounlearn_official"}:
        return "${GENERIC_MODEL_PATH:?Set GENERIC_MODEL_PATH to the declared MCF/ZsRE target}"
    variable = str(track.get("model_environment_variable") or "OFFICIAL_TARGET_MODEL")
    return "${" + variable + ":?Set " + variable + " to the pinned official target model}"


def command_for_track(
    track: Mapping[str, Any],
    output_dir: Path,
    *,
    model_entry: Optional[Mapping[str, Any]] = None,
) -> str:
    template = track["evaluation_command_template"]
    if not isinstance(template, str) or not template.strip():
        raise ValueError(f"{track['id']} has no evaluation command template")
    configured_model_path = (model_entry or {}).get("path")
    model_path = (
        shlex.quote(str(configured_model_path))
        if configured_model_path and "REPLACE_" not in str(configured_model_path)
        else default_model_expression(track)
    )
    replacements = {
        "__OUTPUT_DIR__": shlex.quote(str(output_dir)),
        "__MODEL_PATH__": model_path,
        "__PROJECT_ROOT__": shlex.quote(str(PROJECT_ROOT)),
        "__BENCHMARK_ID__": str(track["id"]),
    }
    command = template
    for marker, value in replacements.items():
        command = command.replace(marker, value)
    return command


def plan_tracks(
    tracks: Sequence[Mapping[str, Any]],
    *,
    output_dir: Path,
    method: str,
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
    locks = load_json_if_present(source_lock_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    planned: List[Dict[str, Any]] = []
    setup_lines = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    run_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "", f"cd {shlex.quote(str(PROJECT_ROOT))}", ""]
    for track in tracks:
        benchmark_id = track["id"]
        run_output = Path("outputs") / "official_benchmarks" / "runs" / benchmark_id
        command = command_for_track(
            track,
            run_output,
            model_entry=_model_entry(models, benchmark_id),
        )
        runnable = track["method_status"] in READY_STATUSES
        setup_lines.extend(
            [
                f"# {benchmark_id}",
                str(track["setup_command_template"]),
                "",
            ]
        )
        if runnable:
            run_lines.extend([f"# {benchmark_id}", command, ""])
        else:
            run_lines.extend(
                [
                    f"# {benchmark_id}: {track['method_status']}",
                    f"# {track['method_reason']}",
                    "",
                ]
            )
        source_lock = _lock_entry(locks, "sources", track["source_lock_id"])
        dataset_lock = _lock_entry(locks, "datasets", track["dataset"]["lock_id"])
        evaluator_lock = _lock_entry(
            locks,
            "evaluators",
            track["official_evaluator"]["lock_id"],
        )
        failure_reason = None if runnable else track["method_reason"]
        manifest = manifest_template(
            track,
            command,
            method=method,
            output_dir=run_output,
            model_entry=_model_entry(models, benchmark_id),
            source_lock=source_lock,
            dataset_lock=dataset_lock,
            evaluator_lock=evaluator_lock,
            status="planned" if runnable else track["method_status"],
            failure_reason=failure_reason,
        )
        manifest_path = output_dir / "manifests" / benchmark_id / "run_manifest.json"
        write_manifest(manifest_path, manifest)
        planned.append(
            {
                "benchmark_id": benchmark_id,
                "method_status": track["method_status"],
                "input_contract": track["input_contract"],
                "setup_command": track["setup_command_template"],
                "run_command": command if runnable else None,
                "evaluation_command": command,
                "manifest_reservation": str(manifest_path),
                "reason": track["method_reason"],
            }
        )
    (output_dir / "setup_commands.sh").write_text(
        "\n".join(setup_lines).rstrip() + "\n", encoding="utf-8"
    )
    (output_dir / "run_commands.sh").write_text(
        "\n".join(run_lines).rstrip() + "\n", encoding="utf-8"
    )
    payload = {
        "schema_version": 1,
        "cpu_only": True,
        "method": method,
        "models_config": str(models_path),
        "source_lock": str(source_lock_path),
        "tracks": planned,
    }
    with (output_dir / "plan.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return payload
