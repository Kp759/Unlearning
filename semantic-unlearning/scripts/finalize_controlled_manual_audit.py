#!/usr/bin/env python3
"""Attach a completed human audit without rerunning or reopening final tests."""

from __future__ import annotations

import argparse
from pathlib import Path

from controlled_unlearning_protocol import read_json, sha256_file, write_json
from evaluate_controlled_unlearning import (
    _manual_audit_status,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-summary", required=True)
    parser.add_argument("--completed-audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary_path = Path(args.evaluation_summary).resolve()
    summary = read_json(summary_path)
    if summary.get("kind") != "controlled_unlearning_evaluation":
        raise ValueError("Input is not a controlled evaluation summary")
    if summary.get("phase") != "test":
        raise ValueError("Manual finalization is intended for final test only")
    queue_path = Path(summary["manual_audit"]["queue_path"]).resolve()
    if sha256_file(queue_path) != summary["manual_audit"]["queue_sha256"]:
        raise ValueError("Manual audit queue changed after evaluation")
    queue = read_jsonl(queue_path)
    status = _manual_audit_status(
        queue,
        Path(args.completed_audit).resolve(),
        minimum_agreement_rate=float(
            summary["manual_audit"]["minimum_agreement_rate"]
        ),
    )
    if not status["complete"]:
        raise RuntimeError("Manual audit is incomplete")
    audited = {
        **summary,
        "manual_audit": {
            **status,
            "queue_path": str(queue_path),
            "queue_sha256": sha256_file(queue_path),
        },
        "release_ready": bool(
            not summary.get("controls", {}).get("final_test_rerun", False)
            and status["gate_passed"]
            and summary.get("utility_guardrail", {}).get("passed") is True
        ),
        "finalization": {
            "original_summary": str(summary_path),
            "original_summary_sha256": sha256_file(summary_path),
            "test_was_rerun": False,
            "test_results_used_for_repair": False,
        },
    }
    output = Path(args.output).resolve()
    write_json(output, audited)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
