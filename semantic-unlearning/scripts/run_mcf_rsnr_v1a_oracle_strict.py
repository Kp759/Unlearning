#!/usr/bin/env python3
"""Strict public entrypoint for RSNR-V1A training.

The core trainer intentionally saves diagnostic artifacts even when the final
registered training gate is not satisfied.  This wrapper preserves those
artifacts, annotates completion.json with an explicit success bit, and exits
nonzero when any forget case fails the joint 5-view training gate.

All launchers and reproducibility commands should use this entrypoint rather
than invoking run_mcf_rsnr_v1a_oracle.py directly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import run_mcf_rsnr_v1a_oracle as core


def validate_completion(payload: Mapping[str, Any], *, expected_count: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if payload.get("protocol") != core.PROTOCOL:
        reasons.append("protocol_mismatch")
    if int(payload.get("joint_passed", -1)) != int(expected_count):
        reasons.append("joint_passed_mismatch")
    if int(payload.get("joint_failures", -1)) != 0:
        reasons.append("joint_failures_nonzero")
    if payload.get("adapter_saved") is not True:
        reasons.append("adapter_not_saved")
    if payload.get("base_weights_modified") is not False:
        reasons.append("base_weights_modified")
    if payload.get("heldout_probe_text_used") is not False:
        reasons.append("heldout_probe_text_used")
    return not reasons, reasons


def mark_completion(path: Path, *, expected_count: int) -> tuple[bool, list[str]]:
    if not path.is_file():
        raise RuntimeError(f"RSNR core trainer returned without completion artifact: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("RSNR completion artifact is not a JSON object")
    passed, reasons = validate_completion(payload, expected_count=expected_count)
    payload["training_gate_passed"] = bool(passed)
    payload["run_successful"] = bool(passed)
    payload["failure_reasons"] = list(reasons)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return passed, reasons


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(argv) if argv is not None else sys.argv[1:]
    parsed = core.parse_args(args_list)
    core.main(args_list)

    completion_path = Path(parsed.output_dir).resolve() / "method" / "completion.json"
    passed, reasons = mark_completion(completion_path, expected_count=int(parsed.forget_num))
    if not passed:
        print(
            json.dumps(
                {
                    "protocol": core.PROTOCOL,
                    "training_gate_passed": False,
                    "failure_reasons": reasons,
                    "diagnostic_artifacts_preserved": True,
                },
                indent=2,
            ),
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(2)

    print(
        json.dumps(
            {
                "protocol": core.PROTOCOL,
                "training_gate_passed": True,
                "run_successful": True,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
