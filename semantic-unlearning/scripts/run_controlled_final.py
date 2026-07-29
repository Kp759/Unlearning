#!/usr/bin/env python3
"""Apply a frozen candidate to held-out requests, then run Judge B once."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from controlled_llm_judge import (
    assert_independent_judges,
    load_judge_config,
)
from controlled_unlearning_protocol import (
    load_final_apply_bundle,
    load_test_bundle,
    read_json,
    sha256_file,
    sha256_json,
    validate_mcf_post_reload_acceptance,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]


def _validate_pretest_inputs(
    selection: Mapping[str, Any],
    *,
    final_apply_path: Path,
    judge_b_path: Path,
) -> None:
    if selection.get("kind") != "controlled_candidate_selection_receipt":
        raise ValueError("Invalid candidate-selection receipt")
    stored_hash = str(selection.get("receipt_sha256", ""))
    unhashed = dict(selection)
    unhashed.pop("receipt_sha256", None)
    if not stored_hash or sha256_json(unhashed) != stored_hash:
        raise ValueError("Selection receipt hash mismatch")
    if selection.get("hyperparameters_frozen") is not True:
        raise ValueError("Selection receipt did not freeze hyperparameters")
    if selection.get("test_results_used_for_selection") is not False:
        raise ValueError("Selection receipt does not prove test isolation")
    final_apply = load_final_apply_bundle(final_apply_path)
    if (
        final_apply["bundle_sha256"]
        != selection.get("final_apply_bundle_sha256")
    ):
        raise ValueError("Prompt-free final-apply commitment mismatch")
    for field in ("protocol_id", "dataset", "fold"):
        if final_apply[field] != selection.get(field):
            raise ValueError(
                f"Final-apply {field} does not match selection receipt"
            )
    judge_b = load_judge_config(judge_b_path)
    if judge_b.role != "judge_b_final":
        raise ValueError("Final run requires a Judge-B config")
    judge_a = selection.get("judge_a")
    if not isinstance(judge_a, Mapping):
        raise ValueError("Selection receipt lacks Judge A identity")
    assert_independent_judges(judge_a, judge_b)


def _validate_application_before_test(
    application: Mapping[str, Any],
    *,
    dataset: str,
) -> None:
    if application.get("kind") != "controlled_model_application_receipt":
        raise ValueError("Invalid final application receipt")
    if application.get("status") != "accepted":
        raise ValueError("Final application was not accepted")
    if application.get("dry_run"):
        raise ValueError("A dry-run application cannot unlock final testing")
    if dataset == "mcf":
        validate_mcf_post_reload_acceptance(
            application.get("post_reload_acceptance")
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-bundle", required=True)
    parser.add_argument("--selection-receipt", required=True)
    parser.add_argument("--judge-b-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--manual-audit-count", type=int, default=30)
    parser.add_argument(
        "--min-manual-judge-agreement",
        type=float,
        default=0.80,
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument(
        "--use-chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    if args.manual_audit_count < 21:
        raise ValueError(
            "--manual-audit-count must be at least 21 so every expected "
            "behavior/style stratum can be represented"
        )
    if not 0.0 <= args.min_manual_judge_agreement <= 1.0:
        raise ValueError(
            "--min-manual-judge-agreement must lie in [0,1]"
        )
    bundle_path = Path(args.test_bundle).resolve()
    selection_path = Path(args.selection_receipt).resolve()
    selection = read_json(selection_path)
    candidate_id = str(selection["selected_candidate_id"])
    selected_spec = selection.get("selected_candidate_spec")
    if not isinstance(selected_spec, dict):
        raise ValueError("Selection receipt lacks embedded candidate spec")
    base_model_path = str(selected_spec.get("base_model_path", "")).strip()
    if not base_model_path:
        raise ValueError("Frozen candidate spec lacks base_model_path")
    final_apply_path = Path(
        selection["final_apply_bundle_path"]
    ).resolve()
    judge_b_path = Path(args.judge_b_config).resolve()
    _validate_pretest_inputs(
        selection,
        final_apply_path=final_apply_path,
        judge_b_path=judge_b_path,
    )
    if (
        str(Path(selection.get("test_bundle_path", "")).resolve())
        != str(bundle_path)
    ):
        raise ValueError(
            "Selection receipt points to a different locked test bundle"
        )
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(
            "Final output directory must be new and empty; this prevents "
            "accidental repeated final-test access"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "final_preregistration.json",
        {
            "schema_version": 1,
            "kind": "controlled_final_preregistration",
            "protocol_id": selection["protocol_id"],
            "dataset": selection["dataset"],
            "fold": selection["fold"],
            "test_bundle_sha256": selection["test_bundle_sha256"],
            "final_apply_bundle": str(final_apply_path),
            "final_apply_bundle_sha256": selection[
                "final_apply_bundle_sha256"
            ],
            "selection_receipt": str(selection_path),
            "selection_receipt_sha256": sha256_file(selection_path),
            "candidate_id": candidate_id,
            "base_reference_model_path": base_model_path,
            "judge_b_config": str(judge_b_path),
            "judge_b_config_sha256": sha256_file(judge_b_path),
            "utility_tolerance": selection["utility_tolerance"],
            "locality_tolerance": selection["locality_tolerance"],
            "minimum_utility_answer_probability_ratio": selection[
                "min_utility_probability_ratio"
            ],
            "manual_audit_count": args.manual_audit_count,
            "minimum_manual_judge_agreement": (
                args.min_manual_judge_agreement
            ),
            "base_reference_and_candidate_evaluation_planned": True,
            "test_results_may_change_repair": False,
        },
    )

    application_dir = output_dir / "application"
    subprocess.run(
        [
            args.python,
            "scripts/run_controlled_setting5e.py",
            "--bundle",
            str(final_apply_path),
            "--phase",
            "final_apply",
            "--stage",
            "final_apply",
            "--selection-receipt",
            str(selection_path),
            "--output-dir",
            str(application_dir),
        ],
        cwd=PROJECT_DIR,
        check=True,
    )
    application_receipt_path = application_dir / "application_receipt.json"
    application = read_json(application_receipt_path)
    _validate_application_before_test(
        application,
        dataset=str(selection["dataset"]),
    )
    checkpoint = str(application["selected_checkpoint"])
    # Only after repair has completed and the application receipt is frozen
    # does this process parse the Judge-B prompt bundle.
    bundle = load_test_bundle(bundle_path)
    if bundle["bundle_sha256"] != selection["test_bundle_sha256"]:
        raise ValueError("Locked test bundle commitment mismatch")
    for field in ("protocol_id", "dataset", "fold"):
        if bundle[field] != selection[field]:
            raise ValueError(
                f"Locked test bundle {field} does not match selection receipt"
            )
    audit_strata = {
        (
            str(case["expected_behavior"]),
            str(case["style"]),
        )
        for case in bundle["prompt_cases"]
    }
    if args.manual_audit_count < len(audit_strata):
        raise ValueError(
            "--manual-audit-count must cover every locked behavior/style "
            f"stratum; need at least {len(audit_strata)}"
        )
    common_evaluation = [
        args.python,
        "scripts/evaluate_controlled_unlearning.py",
        "--bundle",
        str(bundle_path),
        "--phase",
        "test",
        "--judge-config",
        str(judge_b_path),
        "--selection-receipt",
        str(selection_path),
        "--batch-size",
        str(args.batch_size),
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
        common_evaluation.append("--use-chat-template")
    elif args.use_chat_template is False:
        common_evaluation.append("--no-use-chat-template")

    baseline_dir = output_dir / "judge_b_base_reference"
    baseline_command = [
        *common_evaluation,
        "--evaluation-role",
        "base_reference",
        "--model-path",
        base_model_path,
        "--candidate-id",
        "base-reference",
        "--output-dir",
        str(baseline_dir),
    ]
    subprocess.run(baseline_command, cwd=PROJECT_DIR, check=True)
    baseline_summary_path = baseline_dir / "evaluation_summary.json"

    evaluation_dir = output_dir / "judge_b_final"
    command = [
        *common_evaluation,
        "--evaluation-role",
        "candidate",
        "--model-path",
        checkpoint,
        "--candidate-id",
        candidate_id,
        "--application-receipt",
        str(application_receipt_path),
        "--baseline-summary",
        str(baseline_summary_path),
        "--output-dir",
        str(evaluation_dir),
        "--manual-audit-count",
        str(args.manual_audit_count),
        "--min-manual-judge-agreement",
        str(args.min_manual_judge_agreement),
    ]
    subprocess.run(command, cwd=PROJECT_DIR, check=True)
    print(
        f"Final evaluation written to "
        f"{evaluation_dir / 'evaluation_summary.json'}"
    )
    print(
        f"Complete the human queue at "
        f"{evaluation_dir / 'manual_audit_queue.jsonl'}; finalize it without "
        "rerunning the test."
    )


if __name__ == "__main__":
    main()
