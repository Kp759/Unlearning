#!/usr/bin/env python3
"""Verify a clean, from-Base MCF Stage-1 writer without opening eval probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import mcf_compositional_marker_write_read as writer
import mcf_embedding_keyed_neuron_erasure as neuron


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def verify(
    *,
    training_visible_path: Path,
    split_manifest_path: Path,
    context_manifest_path: Path,
    stage1_state_path: Path,
    stage1_report_path: Path,
    stage1_log_path: Path,
    portability_preflight_path: Path,
) -> Dict[str, Any]:
    context = _load_json(context_manifest_path)
    report = _load_json(stage1_report_path)
    state_value = torch.load(stage1_state_path, map_location="cpu", weights_only=False)
    if not isinstance(state_value, Mapping):
        raise RuntimeError("Stage-1 writer state must be a mapping")
    state = dict(state_value)

    neuron._validate_firewall(context, state)
    context_sha256 = writer.sha256_file(context_manifest_path)
    if context_sha256 != str(state.get("context_manifest_sha256") or ""):
        raise RuntimeError("Stage-1 state/context hash mismatch")
    training_sha256 = writer.sha256_file(training_visible_path)
    split_sha256 = writer.sha256_file(split_manifest_path)
    if str(context.get("source_training_visible_sha256") or "") != training_sha256:
        raise RuntimeError("context manifest/training-visible hash mismatch")
    if str(context.get("source_split_manifest_sha256") or "") != split_sha256:
        raise RuntimeError("context manifest/split hash mismatch")

    lineage = neuron._validate_clean_stage1_lineage(
        context,
        state,
        report,
        context_manifest_path,
        stage1_log_path,
    )
    case_ids = [int(value) for value in state.get("case_ids", [])]
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise RuntimeError("Stage-1 state case IDs are empty or non-unique")
    preflight = _load_json(portability_preflight_path)
    if preflight.get("kind") != "mcf_clean_stage1_training_safe_portability_preflight":
        raise RuntimeError("clean Stage-1 portability receipt has the wrong kind")
    if str(preflight.get("protocol") or "") != str(state["protocol"]):
        raise RuntimeError("clean Stage-1 portability protocol mismatch")
    if int(preflight.get("seed", -1)) != int(state["seed"]):
        raise RuntimeError("clean Stage-1 portability seed mismatch")
    if [int(value) for value in preflight.get("case_ids", [])] != case_ids:
        raise RuntimeError("clean Stage-1 portability case IDs mismatch")
    if bool(preflight.get("official_evaluation_opened")) or bool(
        preflight.get("decoder_constructed")
    ):
        raise RuntimeError("clean Stage-1 portability preflight crossed its firewall")
    neuron.validate_writer_preflight_summary(
        preflight,
        amplitude_threshold=4.5,
        minimum_global_fraction=0.95,
        minimum_record_fraction=0.80,
    )
    preflight_binding = preflight.get("binding")
    expected_binding = {
        "context_manifest_sha256": context_sha256,
        "stage1_state_sha256": writer.sha256_file(stage1_state_path),
        "stage1_report_sha256": writer.sha256_file(stage1_report_path),
        "stage1_writer_log_sha256": writer.sha256_file(stage1_log_path),
        "stage1_gradient_conflict_audit_sha256": writer.sha256_file(
            stage1_log_path.with_name("stage1_gradient_conflict_audit.json")
        ),
    }
    if not isinstance(preflight_binding, Mapping) or any(
        str(preflight_binding.get(key) or "") != value
        for key, value in expected_binding.items()
    ):
        raise RuntimeError("clean Stage-1 portability artifact binding mismatch")
    portability_passed = bool(preflight.get("passed"))
    gradient_audit_path = stage1_log_path.with_name(
        "stage1_gradient_conflict_audit.json"
    )
    if not gradient_audit_path.is_file():
        raise RuntimeError("Stage-1 gradient-conflict audit file is missing")
    return {
        "schema_version": 2,
        "kind": "mcf_clean_stage1_writer_acceptance",
        "passed": portability_passed,
        "checks": {
            "artifact_integrity": True,
            "training_safe_portability": portability_passed,
        },
        "protocol": str(state["protocol"]),
        "seed": int(state["seed"]),
        "forget_num": len(case_ids),
        "case_ids": case_ids,
        "lineage": lineage,
        "artifacts": {
            "training_visible_sha256": training_sha256,
            "split_manifest_sha256": split_sha256,
            "context_manifest_sha256": context_sha256,
            "stage1_state_sha256": writer.sha256_file(stage1_state_path),
            "stage1_report_sha256": writer.sha256_file(stage1_report_path),
            "stage1_writer_log_sha256": writer.sha256_file(stage1_log_path),
            "stage1_gradient_conflict_audit_sha256": writer.sha256_file(
                gradient_audit_path
            ),
            "training_safe_portability_sha256": writer.sha256_file(
                portability_preflight_path
            ),
        },
        "training_safe_portability": preflight,
        "official_evaluation_opened": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-visible-path", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--stage1-state", type=Path, required=True)
    parser.add_argument("--stage1-report", type=Path, required=True)
    parser.add_argument("--stage1-writer-log", type=Path, required=True)
    parser.add_argument("--training-safe-portability", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = verify(
        training_visible_path=args.training_visible_path.resolve(),
        split_manifest_path=args.split_manifest.resolve(),
        context_manifest_path=args.context_manifest.resolve(),
        stage1_state_path=args.stage1_state.resolve(),
        stage1_report_path=args.stage1_report.resolve(),
        stage1_log_path=args.stage1_writer_log.resolve(),
        portability_preflight_path=args.training_safe_portability.resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if not bool(receipt["passed"]):
        raise SystemExit(
            "clean Stage-1 writer failed the unchanged 4.5 / 95% / 80% "
            "training-safe portability gate"
        )


if __name__ == "__main__":
    main()
