"""Aggregate manifests while preserving native metric names and directions."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping


class AggregationError(ValueError):
    """Raised for invalid, duplicate, or non-finite metric payloads."""


def _read_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise AggregationError(f"Manifest must be an object: {path}")
    return payload


def aggregate_runs(runs_root: Path, output_dir: Path) -> Dict[str, Any]:
    manifest_paths = sorted(runs_root.rglob("run_manifest.json"))
    rows: List[Dict[str, Any]] = []
    runs: List[Dict[str, Any]] = []
    for path in manifest_paths:
        manifest = _read_manifest(path)
        directions = {
            item["name"]: item["direction"] for item in manifest.get("native_metrics", [])
        }
        values = manifest.get("metric_values") or {}
        undeclared = sorted(set(values) - set(directions))
        if undeclared:
            raise AggregationError(
                f"{path} has undeclared native metrics: {', '.join(undeclared)}"
            )
        run_metrics: List[Dict[str, Any]] = []
        for name, direction in directions.items():
            value = values.get(name)
            if isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise AggregationError(f"{path} metric {name} is non-finite")
            metric = {"name": name, "direction": direction, "value": value}
            run_metrics.append(metric)
            rows.append(
                {
                    "benchmark_id": manifest.get("benchmark_id"),
                    "manifest": str(path),
                    "status": manifest.get("status"),
                    **metric,
                }
            )
        runs.append(
            {
                "benchmark_id": manifest.get("benchmark_id"),
                "manifest": str(path),
                "status": manifest.get("status"),
                "official_protocol": bool(manifest.get("official_protocol")),
                "native_metrics": run_metrics,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "runs_root": str(runs_root),
        "run_count": len(runs),
        "runs": runs,
        "normalization": None,
        "note": "No universal unlearning score is computed; native metrics and directions are retained.",
    }
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "native_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("benchmark_id", "manifest", "status", "name", "direction", "value"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "aggregate.md").open("w", encoding="utf-8") as handle:
        handle.write("# Official benchmark native metrics\n\n")
        handle.write("No universal unlearning score is computed.\n\n")
        handle.write("| Benchmark | Status | Metric | Direction | Value |\n")
        handle.write("| --- | --- | --- | --- | ---: |\n")
        for row in rows:
            handle.write(
                f"| {row['benchmark_id']} | {row['status']} | {row['name']} | "
                f"{row['direction']} | {row['value']} |\n"
            )
    return payload
