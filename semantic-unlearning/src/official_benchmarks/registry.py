"""Load and validate the machine-readable official benchmark registry."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "official_benchmarks" / "registry.json"

ALLOWED_STATUSES = frozenset(
    {
        "READY_NATIVE",
        "READY_WITH_DATA_ADAPTER",
        "EVALUATION_ONLY",
        "NEEDS_METHOD_EXTENSION",
        "BLOCKED_MISSING_TARGET_MODEL",
        "BLOCKED_MISSING_DATA",
        "BLOCKED_MISSING_EVALUATOR",
        "BLOCKED_UNPINNED_SOURCE",
    }
)
ALLOWED_CONTRACTS = frozenset(
    {
        "qa_fact_request",
        "sequence_or_document_forget",
        "target_entity_only",
        "sequential_deletion_requests",
        "evaluation_overlay",
    }
)
ALLOWED_DIRECTIONS = frozenset({"higher", "lower", "reference", "report"})


class RegistryError(ValueError):
    """Raised when registry metadata is incomplete or scientifically invalid."""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise RegistryError(f"Registry file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RegistryError(f"Invalid JSON in registry {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RegistryError("Registry root must be a JSON object")
    return payload


def _require_keys(item: Mapping[str, Any], keys: Iterable[str], label: str) -> None:
    missing = [key for key in keys if key not in item]
    if missing:
        raise RegistryError(f"{label} is missing required fields: {', '.join(missing)}")


def validate_registry(payload: Mapping[str, Any]) -> None:
    _require_keys(
        payload,
        ("schema_version", "tracks", "evaluation_profiles", "baseline_methods"),
        "registry",
    )
    tracks = payload["tracks"]
    if not isinstance(tracks, list) or len(tracks) != 15:
        raise RegistryError("Registry must contain exactly the 15 requested benchmark tracks")
    required = (
        "id",
        "classification",
        "official_source",
        "source_lock_id",
        "expected_source_revision",
        "dataset",
        "data_roles",
        "required_models",
        "expected_tokenizer",
        "official_evaluator",
        "native_metrics",
        "default_split",
        "seed",
        "input_contract",
        "method_status",
        "required_environment",
        "setup_command_template",
        "evaluation_command_template",
        "scientific_restrictions",
        "method_reason",
    )
    ids: List[str] = []
    for index, track in enumerate(tracks):
        if not isinstance(track, dict):
            raise RegistryError(f"track[{index}] must be an object")
        label = f"track[{index}]"
        _require_keys(track, required, label)
        if track["classification"] != "benchmark":
            raise RegistryError(f"{track['id']} must be classified as benchmark")
        if track["method_status"] not in ALLOWED_STATUSES:
            raise RegistryError(
                f"{track['id']} has invalid status {track['method_status']!r}"
            )
        if track["input_contract"] not in ALLOWED_CONTRACTS:
            raise RegistryError(
                f"{track['id']} has invalid input contract {track['input_contract']!r}"
            )
        if not isinstance(track["native_metrics"], list) or not track["native_metrics"]:
            raise RegistryError(f"{track['id']} must declare native metrics")
        metric_names = set()
        for metric in track["native_metrics"]:
            _require_keys(metric, ("name", "direction"), f"{track['id']} metric")
            if metric["name"] in metric_names:
                raise RegistryError(
                    f"{track['id']} repeats native metric {metric['name']!r}"
                )
            metric_names.add(metric["name"])
            if metric["direction"] not in ALLOWED_DIRECTIONS:
                raise RegistryError(
                    f"{track['id']} metric {metric['name']} has invalid direction"
                )
        ids.append(track["id"])
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise RegistryError(f"Duplicate benchmark IDs: {', '.join(duplicates)}")

    for collection_name, expected_classification in (
        ("evaluation_profiles", "evaluation_profile"),
        ("baseline_methods", "baseline_method"),
    ):
        collection = payload[collection_name]
        if not isinstance(collection, list):
            raise RegistryError(f"{collection_name} must be a list")
        for item in collection:
            _require_keys(item, ("id", "classification"), collection_name)
            if item["classification"] != expected_classification:
                raise RegistryError(
                    f"{item['id']} must be classified as {expected_classification}"
                )

    baseline_ids = {item["id"] for item in payload["baseline_methods"]}
    expected_baselines = {"zerounlearn", "rmu", "permu", "rule", "fit", "shred"}
    if baseline_ids != expected_baselines:
        raise RegistryError(
            "baseline_methods must contain exactly zerounlearn, rmu, permu, "
            "rule, fit, and shred"
        )
    overlap = set(ids) & baseline_ids
    if overlap:
        raise RegistryError(f"Methods cannot be benchmark tracks: {sorted(overlap)}")


def load_registry(path: Path | str | None = None) -> Dict[str, Any]:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    payload = _read_json(registry_path)
    validate_registry(payload)
    payload["_path"] = str(registry_path.resolve())
    return payload


def get_track(registry: Mapping[str, Any], benchmark_id: str) -> Dict[str, Any]:
    for track in registry["tracks"]:
        if track["id"] == benchmark_id:
            return dict(track)
    known = ", ".join(track["id"] for track in registry["tracks"])
    raise RegistryError(f"Unknown benchmark {benchmark_id!r}; expected one of: {known}")


def select_tracks(registry: Mapping[str, Any], suite: str | Sequence[str]) -> List[Dict[str, Any]]:
    if isinstance(suite, str):
        requested = [part.strip() for part in suite.split(",") if part.strip()]
    else:
        requested = list(suite)
    if not requested or requested == ["all"]:
        return [dict(track) for track in registry["tracks"]]
    if "all" in requested:
        raise RegistryError("suite 'all' cannot be combined with individual benchmark IDs")
    return [get_track(registry, benchmark_id) for benchmark_id in requested]
