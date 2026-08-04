"""Fail-closed invocation of existing native method/evaluator pipelines."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .doctor import READY_STATUSES, audit_track
from .planner import command_for_track
from .provenance import (
    load_json_if_present,
    manifest_template,
    sha256_directory,
    validate_manifest,
    write_manifest,
)
from .registry import PROJECT_ROOT


class RunRefused(RuntimeError):
    """Raised before subprocess launch when official requirements are unresolved."""


def _read_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    return value if isinstance(value, dict) else {}


def extract_native_metrics(
    track: Mapping[str, Any], output_dir: Path
) -> tuple[Dict[str, Any], list[str]]:
    """Read only native outputs produced by the existing wrapped pipelines."""

    benchmark_id = track["id"]
    values: Dict[str, Any] = {}
    sources: list[str] = []
    if benchmark_id == "mcf_zerounlearn_official":
        for path in sorted(output_dir.rglob("aggregate.json")):
            payload = _read_json(path)
            if payload.get("dataset") != "MCF":
                continue
            for row in payload.get("rows", []):
                if row.get("method_key") == "protected_lm_head_repair":
                    for metric in ("Eff", "Gen", "Spe", "PPL"):
                        values[metric] = {
                            "mean": row.get(f"{metric}_mean"),
                            "std": row.get(f"{metric}_std"),
                        }
                    sources.append(str(path))
                    return values, sources
    elif benchmark_id == "zsre_zerounlearn_official":
        for path in sorted(output_dir.rglob("aggregate.json")):
            payload = _read_json(path)
            if payload.get("dataset") != "ZsRE":
                continue
            for row in payload.get("rows", []):
                if row.get("method") == "Selected":
                    mapping = {
                        "Eff": "forget_Eff_down",
                        "Gen": "forget_Gen_down",
                        "Spe": "forget_Spe_up",
                        "PPL": "PPL_down",
                    }
                    for metric, field in mapping.items():
                        values[metric] = {
                            "mean": row.get(f"{field}_mean"),
                            "std": row.get(f"{field}_std"),
                        }
                    sources.append(str(path))
                    return values, sources
    elif benchmark_id == "tofu_forget05":
        candidates = sorted(output_dir.rglob("tofu_active_forget_repair_summary.json"))
        if candidates:
            path = candidates[-1]
            payload = _read_json(path)
            aliases = {
                "real_authors_rougeL_recall": "tofu_real_authors_rougeL_recall",
                "world_facts_rougeL_recall": "tofu_world_facts_rougeL_recall",
            }
            for metric in track["native_metrics"]:
                name = metric["name"]
                key = name if name in payload else aliases.get(name)
                if key and key in payload:
                    values[name] = payload[key]
            sources.append(str(path))
    return values, sources


def _entry(mapping: Mapping[str, Any], collection: str, key: str) -> Mapping[str, Any]:
    value = mapping.get(collection, {}).get(key, {})
    return value if isinstance(value, dict) else {}


def run_track(
    track: Mapping[str, Any],
    *,
    method: str,
    output_dir: Path,
    execute: bool,
    models_path: Optional[Path] = None,
    source_lock_path: Optional[Path] = None,
) -> Dict[str, Any]:
    if method != "our_method":
        raise RunRefused(
            "Stage 1 run supports only our_method; official baselines are metadata-only "
            "until their pinned upstream repositories are explicitly selected"
        )
    if track["method_status"] not in READY_STATUSES:
        raise RunRefused(
            f"{track['id']} is {track['method_status']}: {track['method_reason']}"
        )
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
    official_bench_root = Path(
        os.environ.get("OFFICIAL_BENCH_ROOT", "/mnt/train/official-unlearning-benchmarks")
    )
    model_entry = _entry(models, "tracks", track["id"])
    audit = audit_track(
        track,
        models=models,
        source_locks=locks,
        official_bench_root=official_bench_root,
        generic_model_path=os.environ.get(
            "GENERIC_MODEL_PATH", "/mnt/train/models/Llama-3.2-3B-Instruct"
        ),
    )
    command = command_for_track(track, output_dir, model_entry=model_entry)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run_manifest.json"
    manifest = manifest_template(
        track,
        command,
        method=method,
        output_dir=output_dir,
        model_entry=model_entry,
        source_lock=_entry(locks, "sources", track["source_lock_id"]),
        dataset_lock=_entry(locks, "datasets", track["dataset"]["lock_id"]),
        evaluator_lock=_entry(
            locks, "evaluators", track["official_evaluator"]["lock_id"]
        ),
        status="dry_run" if not execute else "preflight",
    )
    if not audit["official_ready"]:
        failures = [
            item["detail"] for item in audit["checks"] if not item["ok"]
        ]
        manifest["status"] = audit["effective_status"]
        manifest["failure_reason"] = "; ".join(failures)
        write_manifest(manifest_path, manifest)
        raise RunRefused(
            f"{track['id']} unresolved ({audit['effective_status']}): "
            + "; ".join(failures)
        )
    validate_manifest(manifest, official=True)
    if not execute:
        write_manifest(manifest_path, manifest)
        return {"executed": False, "command": command, "manifest": str(manifest_path)}

    manifest["status"] = "running"
    write_manifest(manifest_path, manifest)
    completed = subprocess.run(
        ["bash", "-lc", command],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if completed.returncode != 0:
        manifest["status"] = "failed"
        manifest["failure_reason"] = f"native pipeline exited {completed.returncode}"
        write_manifest(manifest_path, manifest)
        raise subprocess.CalledProcessError(completed.returncode, command)
    checkpoint_candidates = sorted(output_dir.rglob("checkpoint"))
    if checkpoint_candidates:
        manifest["output_checkpoint_hash"] = sha256_directory(checkpoint_candidates[-1])
    metric_values, metric_sources = extract_native_metrics(track, output_dir)
    manifest["metric_values"] = metric_values
    manifest["native_metric_sources"] = metric_sources
    manifest["status"] = "completed"
    manifest["official_protocol"] = True
    manifest["failure_reason"] = None
    write_manifest(manifest_path, manifest)
    return {"executed": True, "command": command, "manifest": str(manifest_path)}
