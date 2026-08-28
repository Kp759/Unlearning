#!/usr/bin/env python3
"""Aggregate preregistered training-only neuron-erasure ablations.

This utility reads method summaries only. It has no benchmark or official-eval
input, so its table cannot become an evaluation-driven model-selection loop.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation-root", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def _load(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _labels(registry: Mapping[str, Any]) -> List[str]:
    labels = [
        str(row["name"])
        for row in registry.get("controlled_training_ablations", [])
        if isinstance(row, Mapping)
    ]
    capacity = registry.get("capacity_ablations", {})
    if isinstance(capacity, Mapping):
        labels.extend(
            f"neurons_per_record_{int(value)}"
            for value in capacity.get("neurons_per_record", [])
            if int(value) != 4
        )
        labels.extend(
            f"layer_{int(value)}"
            for value in capacity.get("neuron_layer_zero_based", [])
            if int(value) != 8
        )
    return labels


def _row(label: str, summary: Mapping[str, Any] | None) -> Dict[str, Any]:
    if summary is None:
        return {"label": label, "status": "missing"}
    acceptance = summary.get("acceptance", {})
    actuator = summary.get("actuator", {})
    detector = summary.get("detector", {})
    gate = detector.get("gate", {}) if isinstance(detector, Mapping) else {}
    causal = summary.get("causal_component_ablation", {})
    architecture = summary.get("architecture", {})
    return {
        "label": label,
        "status": "pass" if acceptance.get("passed") else "fail",
        "direct_failures": actuator.get("direct_failures"),
        "positive_failures": actuator.get("training_safe_positive_failures"),
        "minimum_margin": actuator.get("minimum_margin"),
        "reference_nll_regression_max": actuator.get("reference_nll_regression_max"),
        "detector_records_passed": gate.get("passed_records"),
        "detector_records_total": gate.get("total_records"),
        "writer_necessary": causal.get("writer_is_necessary"),
        "decoder_necessary": causal.get("decoder_is_necessary"),
        "selected_neurons": architecture.get("selected_existing_mlp_neurons"),
        "edited_parameter_fraction": architecture.get("edited_parameter_fraction"),
        "official_evaluation_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    root = Path(args.ablation_root).resolve()
    registry = _load(Path(args.registry).resolve())
    rows: List[Dict[str, Any]] = []
    for label in _labels(registry):
        path = root / label / "embedding_keyed_neuron_summary.json"
        rows.append(_row(label, _load(path) if path.is_file() else None))
    output = Path(args.out_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_training_only_ablation_aggregate",
        "official_evaluation_opened": False,
        "rows": rows,
    }
    (output / "training_ablation_aggregate.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    columns = sorted({key for row in rows for key in row})
    with (output / "training_ablation_aggregate.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Training-only embedding-keyed neuron ablations",
        "",
        "Official evaluation opened: **no**",
        "",
        "| Ablation | Status | Direct failures | Positive failures | Detector pass | Writer needed | Decoder needed |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        detector = (
            f"{row.get('detector_records_passed')}/{row.get('detector_records_total')}"
            if row.get("detector_records_total") is not None
            else "-"
        )
        lines.append(
            f"| {row['label']} | {row['status']} | "
            f"{row.get('direct_failures', '-')} | {row.get('positive_failures', '-')} | "
            f"{detector} | {row.get('writer_necessary', '-')} | "
            f"{row.get('decoder_necessary', '-')} |"
        )
    (output / "training_ablation_aggregate.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"wrote {output / 'training_ablation_aggregate.md'}")


if __name__ == "__main__":
    main()
