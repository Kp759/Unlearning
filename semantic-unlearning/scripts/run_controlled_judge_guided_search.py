#!/usr/bin/env python3
"""Run a preregistered Judge-A outer loop over repair candidates.

Judge A is not treated as a differentiable truth oracle. Instead, each
candidate Setting 5e + LM-head repair configuration is applied to the same
validation deletion requests from a fresh base model, then Judge A and
model-native metrics select among candidates within fixed utility/locality
tolerances. The final-test bundle is never opened.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

from controlled_llm_judge import load_judge_config, public_judge_config
from controlled_unlearning_protocol import (
    load_development_bundle,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _candidate_spec(path: Path) -> Dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError(f"Candidate spec {path} is not an object")
    for field in (
        "schema_version",
        "candidate_id",
        "dataset",
        "base_model_path",
    ):
        if field not in value:
            raise ValueError(f"Candidate spec {path} lacks {field}")
    return dict(value)


def _run(
    command: List[str],
    *,
    commands: List[List[str]],
    dry_run: bool,
) -> None:
    commands.append(command)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-bundle", required=True)
    parser.add_argument(
        "--candidate-spec",
        action="append",
        required=True,
    )
    parser.add_argument("--judge-a-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--utility-tolerance", type=float, default=0.02)
    parser.add_argument("--locality-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--min-utility-probability-ratio",
        type=float,
        default=0.98,
    )
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--run-train-stage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run each candidate on train deletion requests for development "
            "diagnostics before the fresh-base validation application."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    args = parser.parse_args()

    if not 0.0 < args.utility_tolerance < 1.0:
        raise ValueError("--utility-tolerance must lie in (0,1)")
    if not 0.0 < args.locality_tolerance < 1.0:
        raise ValueError("--locality-tolerance must lie in (0,1)")
    if not 0.0 < args.min_utility_probability_ratio <= 1.0:
        raise ValueError(
            "--min-utility-probability-ratio must lie in (0,1]"
        )
    bundle_path = Path(args.development_bundle).resolve()
    bundle = load_development_bundle(bundle_path)
    judge_path = Path(args.judge_a_config).resolve()
    judge = load_judge_config(judge_path)
    if judge.role != "judge_a_development":
        raise ValueError("Search requires a Judge-A development config")
    spec_paths = [Path(value).resolve() for value in args.candidate_spec]
    specs = [_candidate_spec(path) for path in spec_paths]
    candidate_ids = [str(spec["candidate_id"]) for spec in specs]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate IDs must be unique")
    for spec in specs:
        if spec["dataset"] != bundle["dataset"]:
            raise ValueError("Candidate spec dataset does not match bundle")
        if (
            spec.get("seed") is not None
            and int(spec["seed"]) != int(bundle["seed"])
        ):
            raise ValueError("Candidate spec seed does not match bundle")
    base_paths = {str(spec["base_model_path"]) for spec in specs}
    if len(base_paths) != 1:
        raise ValueError(
            "All candidates must start from the exact same base model path"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preregistration = {
        "schema_version": 1,
        "kind": "controlled_judge_a_search_preregistration",
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "development_bundle_sha256": bundle["bundle_sha256"],
        "final_apply_commitment_sha256": bundle[
            "final_apply_bundle_sha256"
        ],
        "test_bundle_commitment_sha256": bundle[
            "test_bundle_sha256"
        ],
        "test_bundle_opened": False,
        "candidate_specs": [
            {
                "candidate_id": spec["candidate_id"],
                "path": str(path),
                "file_sha256": sha256_file(path),
                "content_sha256": sha256_json(spec),
            }
            for path, spec in zip(spec_paths, specs)
        ],
        "judge_a": public_judge_config(judge),
        "selection_tolerances": {
            "utility_absolute_drop": args.utility_tolerance,
            "locality_absolute_drop": args.locality_tolerance,
            "minimum_utility_answer_probability_ratio": (
                args.min_utility_probability_ratio
            ),
        },
        "selection_rule": (
            "utility/locality eligibility, then strict fact-level forget "
            "pass rate, sensitive probability, and retained utility"
        ),
        "train_stage_diagnostics_enabled": args.run_train_stage,
    }
    preregistration_path = output_dir / "preregistration.json"
    write_json(preregistration_path, preregistration)

    commands: List[List[str]] = []
    def evaluation_args(partition: str) -> List[str]:
        values = [
            "--bundle",
            str(bundle_path),
            "--phase",
            "development",
            "--partition",
            partition,
            "--judge-config",
            str(judge_path),
            "--batch-size",
            str(args.eval_batch_size),
            "--score-batch-size",
            str(args.score_batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dtype",
            args.dtype,
            "--device-map",
            args.device_map,
        ]
        if args.use_chat_template is True:
            values.append("--use-chat-template")
        elif args.use_chat_template is False:
            values.append("--no-use-chat-template")
        return values

    validation_eval_args = evaluation_args("validation")

    baseline_dir = output_dir / "baseline_validation"
    baseline_command = [
        args.python,
        "scripts/evaluate_controlled_unlearning.py",
        *validation_eval_args,
        "--model-path",
        next(iter(base_paths)),
        "--candidate-id",
        "base",
        "--output-dir",
        str(baseline_dir),
    ]
    _run(baseline_command, commands=commands, dry_run=args.dry_run)

    candidate_summaries: Dict[str, Path] = {}
    for spec_path, spec in zip(spec_paths, specs):
        candidate_id = str(spec["candidate_id"])
        candidate_root = output_dir / "candidates" / candidate_id
        if args.run_train_stage:
            train_application_command = [
                args.python,
                "scripts/run_controlled_setting5e.py",
                "--bundle",
                str(bundle_path),
                "--phase",
                "development",
                "--stage",
                "train",
                "--candidate-spec",
                str(spec_path),
                "--output-dir",
                str(candidate_root / "train_application"),
            ]
            _run(
                train_application_command,
                commands=commands,
                dry_run=args.dry_run,
            )
            train_application_dir = candidate_root / "train_application"
            if args.dry_run:
                train_checkpoint = (
                    train_application_dir / "DRY_RUN_SELECTED_CHECKPOINT"
                )
            else:
                train_receipt = read_json(
                    train_application_dir / "application_receipt.json"
                )
                train_checkpoint = Path(
                    train_receipt["selected_checkpoint"]
                )
            train_evaluation_command = [
                args.python,
                "scripts/evaluate_controlled_unlearning.py",
                *evaluation_args("train"),
                "--model-path",
                str(train_checkpoint),
                "--candidate-id",
                candidate_id,
                "--output-dir",
                str(candidate_root / "judge_a_train_diagnostics"),
            ]
            _run(
                train_evaluation_command,
                commands=commands,
                dry_run=args.dry_run,
            )
        application_dir = candidate_root / "application"
        application_command = [
            args.python,
            "scripts/run_controlled_setting5e.py",
            "--bundle",
            str(bundle_path),
            "--phase",
            "development",
            "--stage",
            "validation",
            "--candidate-spec",
            str(spec_path),
            "--output-dir",
            str(application_dir),
        ]
        _run(
            application_command,
            commands=commands,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            checkpoint = application_dir / "DRY_RUN_SELECTED_CHECKPOINT"
        else:
            receipt = read_json(application_dir / "application_receipt.json")
            checkpoint = Path(receipt["selected_checkpoint"])
        evaluation_dir = candidate_root / "judge_a_validation"
        evaluation_command = [
            args.python,
            "scripts/evaluate_controlled_unlearning.py",
            *validation_eval_args,
            "--model-path",
            str(checkpoint),
            "--candidate-id",
            candidate_id,
            "--output-dir",
            str(evaluation_dir),
        ]
        _run(
            evaluation_command,
            commands=commands,
            dry_run=args.dry_run,
        )
        candidate_summaries[candidate_id] = (
            evaluation_dir / "evaluation_summary.json"
        )

    receipt_path = output_dir / "selection_receipt.json"
    selection_command = [
        args.python,
        "scripts/select_controlled_candidate.py",
        "--development-bundle",
        str(bundle_path),
        "--baseline-summary",
        str(baseline_dir / "evaluation_summary.json"),
        "--utility-tolerance",
        str(args.utility_tolerance),
        "--locality-tolerance",
        str(args.locality_tolerance),
        "--min-utility-probability-ratio",
        str(args.min_utility_probability_ratio),
        "--output",
        str(receipt_path),
    ]
    for candidate_id, spec_path in zip(candidate_ids, spec_paths):
        selection_command.extend(
            [
                "--candidate",
                f"{candidate_id}={candidate_summaries[candidate_id]}",
                "--candidate-spec",
                f"{candidate_id}={spec_path}",
            ]
        )
    _run(
        selection_command,
        commands=commands,
        dry_run=args.dry_run,
    )
    write_json(
        output_dir / "search_commands.json",
        {
            "schema_version": 1,
            "preregistration": str(preregistration_path),
            "dry_run": args.dry_run,
            "commands": [
                {
                    "argv": command,
                    "shell_rendering_for_review": shlex.join(command),
                }
                for command in commands
            ],
            "selection_receipt": (
                None if args.dry_run else str(receipt_path)
            ),
            "test_bundle_opened": False,
            "test_results_used": False,
        },
    )
    if args.dry_run:
        print(f"Wrote dry-run plan to {output_dir / 'search_commands.json'}")
    else:
        print(f"Wrote selection receipt to {receipt_path}")


if __name__ == "__main__":
    main()
