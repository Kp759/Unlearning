#!/usr/bin/env python3
"""Validate a utility-controlled RWKU Setting 5e run without modifying it."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

from rwku_artifact_access import sha256_file
from rwku_checkpoint_receipt import load_receipt, verify_frozen_identities
from rwku_rowwise_active_repair import ACTIVE_SOURCE
from rwku_setting5e_utility_controlled import (
    METHOD,
    PROTOCOL_STATUS,
    STATE_SCHEMA_VERSION,
    fixed_gate_manifest,
)


def _read(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return dict(value)


def _assert_finite_or_null(value: Any, *, path: str = "") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_or_null(child, path=f"{path}/{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite_or_null(child, path=f"{path}/{index}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value remains in strict JSON at {path or '/'}")


def validate_run(run_dir: Path, *, expected_state: str | None = None) -> Dict[str, Any]:
    root = Path(run_dir).resolve()
    state = _read(root / "experiment_state.json")
    if state.get("schema_version") != STATE_SCHEMA_VERSION:
        raise ValueError("Unsupported utility-controlled experiment state")
    if state.get("method") != METHOD or state.get("protocol_status") != PROTOCOL_STATUS:
        raise ValueError(
            "Run is not the utility-controlled Setting 5e method extension"
        )
    if expected_state and state.get("state") != expected_state:
        raise ValueError(f"Expected state {expected_state}, got {state.get('state')}")
    candidate_path = root / "utility_controlled_setting5" / "candidate_report.json"
    if state.get("state") in {
        "CANDIDATES_EVALUATED",
        "NO_FEASIBLE_CANDIDATE",
        "CHECKPOINT_FROZEN",
        "OFFICIAL_EVALUATION_OPENED",
        "EVALUATION_COMPLETE",
    }:
        candidate = _read(candidate_path)
        if candidate.get("official_rwku_records_accessed") is not False:
            raise ValueError("Candidate report does not attest official-data isolation")
        if candidate.get("fixed_thresholds") != fixed_gate_manifest():
            raise ValueError("Candidate fixed thresholds changed")
    else:
        candidate = {}

    if state.get("state") == "NO_FEASIBLE_CANDIDATE":
        forbidden = [
            root / "checkpoint_receipt.json",
            root / "utility_controlled_setting5" / "selected_checkpoint",
            root / "rowwise_repair" / "selected_checkpoint",
        ]
        if any(path.exists() for path in forbidden):
            raise ValueError(
                "No-feasible run contains a selected checkpoint or receipt"
            )
        if candidate.get("selected_candidate") is not None:
            raise ValueError("No-feasible candidate report claims a selection")
        return {
            "valid": True,
            "state": state["state"],
            "selected_checkpoint": False,
            "official_evaluation": False,
        }

    receipt = None
    if state.get("state") in {
        "CHECKPOINT_FROZEN",
        "OFFICIAL_EVALUATION_OPENED",
        "EVALUATION_COMPLETE",
    }:
        receipt = load_receipt(root / "checkpoint_receipt.json")
        verify_frozen_identities(receipt)
        if receipt.get("protocol_status") != PROTOCOL_STATUS:
            raise ValueError("Checkpoint receipt has a different method status")
        selected = candidate.get("selected_candidate")
        if not isinstance(selected, Mapping) or selected.get("eligible") is not True:
            raise ValueError("Frozen run lacks a fully eligible selected candidate")
        repair_path = root / "rowwise_repair" / "repair_report.json"
        repair = _read(repair_path)
        if repair.get("active_source") != ACTIVE_SOURCE:
            raise ValueError("Repair active_source is not target-generated views")
        if repair.get("selected_success") is not True:
            raise ValueError("Repair report does not contain a valid selected repair")
        method = receipt.get("method_configuration", {})
        for field, path in (
            ("candidate_report_sha256", candidate_path),
            ("repair_report_sha256", repair_path),
            ("training_report_sha256", root / "training_report.json"),
        ):
            if method.get(field) != sha256_file(path):
                raise ValueError(f"Frozen method artifact changed: {path}")

    result_path = root / "official_evaluation.json"
    if state.get("state") == "EVALUATION_COMPLETE":
        result = _read(result_path)
        _assert_finite_or_null(result)
        serialization = result.get("serialization", {})
        if serialization.get("policy") != "non_finite_numeric_values_to_json_null":
            raise ValueError(
                "Official result lacks strict non-finite serialization policy"
            )
        if serialization.get("strict_json_allow_nan") is not False:
            raise ValueError("Official result does not attest allow_nan=false")
        if state.get("result_sha256") != sha256_file(result_path):
            raise ValueError("Official result differs from completion state")
    elif result_path.exists():
        raise ValueError("Official result exists before EVALUATION_COMPLETE")

    return {
        "valid": True,
        "state": state["state"],
        "selected_checkpoint": receipt is not None,
        "official_evaluation": state.get("state") == "EVALUATION_COMPLETE",
        "development": bool(state.get("development")),
        "confirmatory": bool(state.get("confirmatory")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--expect-state",
        choices=(
            "PREPARED",
            "TRAINING",
            "CANDIDATES_EVALUATED",
            "NO_FEASIBLE_CANDIDATE",
            "CHECKPOINT_FROZEN",
            "OFFICIAL_EVALUATION_OPENED",
            "EVALUATION_COMPLETE",
        ),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            validate_run(args.run_dir, expected_state=args.expect_state), indent=2
        )
    )


if __name__ == "__main__":
    main()
