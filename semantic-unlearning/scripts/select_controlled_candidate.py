#!/usr/bin/env python3
"""Select Setting 5e + repair hyperparameters using Judge A validation only.

The selector never opens the committed final-test bundle. It enforces a
predefined absolute utility/locality tolerance (default: two percentage
points), selects the best feasible forgetting result, and emits the receipt
required to unlock a final-apply run and Judge B evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from controlled_unlearning_protocol import (
    load_development_bundle,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)


def _parse_assignment(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected ID=PATH, got {value!r}")
    candidate_id, raw_path = value.split("=", 1)
    candidate_id = candidate_id.strip()
    if not candidate_id or not raw_path.strip():
        raise ValueError(f"Expected non-empty ID=PATH, got {value!r}")
    return candidate_id, Path(raw_path).resolve()


def _metric(
    summary: Mapping[str, Any],
    behavior: str,
    field: str,
) -> Optional[float]:
    value = (
        summary.get("metrics", {})
        .get("by_behavior", {})
        .get(behavior, {})
        .get(field)
    )
    return None if value is None else float(value)


def _weighted_utility_pass(summary: Mapping[str, Any]) -> Optional[float]:
    return _weighted_utility_metric(summary, "judge_pass_rate")


def _weighted_utility_metric(
    summary: Mapping[str, Any],
    field: str,
) -> Optional[float]:
    behaviors = summary.get("metrics", {}).get("by_behavior", {})
    numerator = 0.0
    denominator = 0
    for behavior in ("answer_correctly", "preserve_locality"):
        values = behaviors.get(behavior, {})
        count = int(values.get("count", 0) or 0)
        metric_value = values.get(field)
        if count and metric_value is not None:
            numerator += count * float(metric_value)
            denominator += count
    return numerator / denominator if denominator else None


def _validate_validation_summary(
    summary: Mapping[str, Any],
    *,
    development_bundle: Mapping[str, Any],
    expected_id: Optional[str] = None,
) -> None:
    if summary.get("kind") != "controlled_unlearning_evaluation":
        raise ValueError("Candidate is not a controlled evaluation summary")
    required = {
        "phase": "development",
        "partition": "validation",
        "protocol_id": development_bundle["protocol_id"],
        "dataset": development_bundle["dataset"],
        "fold": development_bundle["fold"],
        "bundle_sha256": development_bundle["bundle_sha256"],
    }
    for field, expected in required.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"Validation summary {field}={summary.get(field)!r}; "
                f"expected {expected!r}"
            )
    if expected_id is not None and summary.get("candidate_id") != expected_id:
        raise ValueError(
            f"Candidate assignment {expected_id!r} points to summary for "
            f"{summary.get('candidate_id')!r}"
        )
    judge = summary.get("judge", {})
    if judge.get("role") != "judge_a_development":
        raise ValueError("Candidate was not scored by Judge A")
    controls = summary.get("controls", {})
    if not controls.get("token_probability_metrics_present"):
        raise ValueError("Candidate lacks mandatory token-probability metrics")


def _candidate_report(
    candidate_id: str,
    summary: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    utility_tolerance: float,
    locality_tolerance: float,
    min_utility_probability_ratio: float,
) -> Dict[str, Any]:
    forget_pass = _metric(
        summary,
        "avoid_sensitive",
        "record_strict_all_prompts_pass_rate",
    )
    forget_prompt_pass = _metric(
        summary,
        "avoid_sensitive",
        "judge_pass_rate",
    )
    sensitive_probability = _metric(
        summary,
        "avoid_sensitive",
        "mean_max_sensitive_probability",
    )
    utility_pass = _weighted_utility_pass(summary)
    baseline_utility = _weighted_utility_pass(baseline)
    utility_probability = _weighted_utility_metric(
        summary,
        "mean_max_acceptable_probability",
    )
    baseline_utility_probability = _weighted_utility_metric(
        baseline,
        "mean_max_acceptable_probability",
    )
    locality_pass = _metric(
        summary,
        "preserve_locality",
        "judge_pass_rate",
    )
    baseline_locality = _metric(
        baseline,
        "preserve_locality",
        "judge_pass_rate",
    )
    if forget_pass is None:
        raise ValueError(
            f"Candidate {candidate_id} has no strict fact-level forget metric"
        )
    if utility_pass is None or baseline_utility is None:
        raise ValueError(
            f"Candidate {candidate_id} or baseline lacks utility judgments"
        )
    utility_drop = baseline_utility - utility_pass
    if (
        utility_probability is None
        or baseline_utility_probability is None
        or baseline_utility_probability <= 0.0
    ):
        raise ValueError(
            f"Candidate {candidate_id} or baseline lacks usable utility "
            "answer-probability metrics"
        )
    utility_probability_ratio = (
        utility_probability / baseline_utility_probability
    )
    locality_drop = (
        None
        if locality_pass is None or baseline_locality is None
        else baseline_locality - locality_pass
    )
    utility_ok = utility_drop <= utility_tolerance
    utility_probability_ok = (
        utility_probability_ratio >= min_utility_probability_ratio
    )
    locality_ok = (
        True
        if locality_drop is None
        else locality_drop <= locality_tolerance
    )
    return {
        "candidate_id": candidate_id,
        "forget_strict_fact_pass_rate": forget_pass,
        "forget_prompt_judge_pass_rate": forget_prompt_pass,
        "mean_sensitive_answer_probability": sensitive_probability,
        "utility_judge_pass_rate": utility_pass,
        "baseline_utility_judge_pass_rate": baseline_utility,
        "utility_absolute_drop": utility_drop,
        "utility_tolerance": utility_tolerance,
        "utility_passed": utility_ok,
        "utility_answer_probability": utility_probability,
        "baseline_utility_answer_probability": (
            baseline_utility_probability
        ),
        "utility_answer_probability_ratio": utility_probability_ratio,
        "minimum_utility_answer_probability_ratio": (
            min_utility_probability_ratio
        ),
        "utility_answer_probability_passed": utility_probability_ok,
        "locality_judge_pass_rate": locality_pass,
        "baseline_locality_judge_pass_rate": baseline_locality,
        "locality_absolute_drop": locality_drop,
        "locality_tolerance": locality_tolerance,
        "locality_passed": locality_ok,
        "eligible": bool(
            utility_ok and utility_probability_ok and locality_ok
        ),
    }


def _selection_key(report: Mapping[str, Any]) -> Tuple[float, float, float, str]:
    sensitive = report.get("mean_sensitive_answer_probability")
    sensitive_value = float("inf") if sensitive is None else float(sensitive)
    return (
        -float(report["forget_strict_fact_pass_rate"]),
        sensitive_value,
        -float(report["utility_judge_pass_rate"]),
        str(report["candidate_id"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development-bundle", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate validation summary as ID=PATH; repeat as needed.",
    )
    parser.add_argument(
        "--candidate-spec",
        action="append",
        required=True,
        help=(
            "Preregistered Setting 5e + repair configuration as ID=PATH; "
            "one is required for every candidate."
        ),
    )
    parser.add_argument(
        "--utility-tolerance",
        type=float,
        default=0.02,
        help="Maximum absolute utility judge-pass-rate drop from Base.",
    )
    parser.add_argument(
        "--locality-tolerance",
        type=float,
        default=0.02,
        help="Maximum absolute locality judge-pass-rate drop from Base.",
    )
    parser.add_argument(
        "--min-utility-probability-ratio",
        type=float,
        default=0.98,
        help=(
            "Candidate weighted retain/locality answer probability must be "
            "at least this fraction of Base."
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.utility_tolerance < 1.0:
        raise ValueError("--utility-tolerance must lie strictly in (0,1)")
    if not 0.0 < args.locality_tolerance < 1.0:
        raise ValueError("--locality-tolerance must lie strictly in (0,1)")
    if not 0.0 < args.min_utility_probability_ratio <= 1.0:
        raise ValueError(
            "--min-utility-probability-ratio must lie in (0,1]"
        )
    development_path = Path(args.development_bundle).resolve()
    development = load_development_bundle(development_path)
    baseline_path = Path(args.baseline_summary).resolve()
    baseline = read_json(baseline_path)
    _validate_validation_summary(
        baseline,
        development_bundle=development,
    )

    summary_paths = dict(_parse_assignment(value) for value in args.candidate)
    spec_paths = dict(
        _parse_assignment(value) for value in args.candidate_spec
    )
    if set(summary_paths) != set(spec_paths):
        raise ValueError(
            "Candidate summary IDs and candidate-spec IDs must match exactly"
        )
    if len(summary_paths) < 1:
        raise ValueError("At least one candidate is required")
    judge_a = baseline["judge"]
    reports: List[Dict[str, Any]] = []
    summaries: Dict[str, Mapping[str, Any]] = {}
    specs: Dict[str, Any] = {}
    for candidate_id in sorted(summary_paths):
        summary = read_json(summary_paths[candidate_id])
        _validate_validation_summary(
            summary,
            development_bundle=development,
            expected_id=candidate_id,
        )
        if sha256_json(summary["judge"]) != sha256_json(judge_a):
            raise ValueError(
                f"Candidate {candidate_id} used a different Judge A"
            )
        spec = read_json(spec_paths[candidate_id])
        if not isinstance(spec, Mapping):
            raise ValueError(
                f"Candidate spec {spec_paths[candidate_id]} is not an object"
            )
        if str(spec.get("candidate_id", "")) != candidate_id:
            raise ValueError(
                f"Candidate spec must declare candidate_id={candidate_id!r}"
            )
        if str(spec.get("dataset", "")) != development["dataset"]:
            raise ValueError(
                f"Candidate spec {candidate_id} targets the wrong dataset"
            )
        summaries[candidate_id] = summary
        specs[candidate_id] = dict(spec)
        report = _candidate_report(
            candidate_id,
            summary,
            baseline,
            utility_tolerance=args.utility_tolerance,
            locality_tolerance=args.locality_tolerance,
            min_utility_probability_ratio=(
                args.min_utility_probability_ratio
            ),
        )
        report["validation_summary"] = str(summary_paths[candidate_id])
        report["validation_summary_sha256"] = sha256_file(
            summary_paths[candidate_id]
        )
        report["candidate_spec"] = str(spec_paths[candidate_id])
        report["candidate_spec_sha256"] = sha256_file(
            spec_paths[candidate_id]
        )
        reports.append(report)
    eligible = [report for report in reports if report["eligible"]]
    if not eligible:
        report_path = Path(args.output).with_suffix(".rejected.json")
        write_json(
            report_path,
            {
                "kind": "controlled_candidate_selection_failure",
                "protocol_id": development["protocol_id"],
                "dataset": development["dataset"],
                "fold": development["fold"],
                "reason": "no_candidate_within_predefined_utility_tolerance",
                "candidate_reports": reports,
            },
        )
        raise RuntimeError(
            "No candidate met the predefined utility/locality tolerances; "
            f"details: {report_path}"
        )
    selected = min(eligible, key=_selection_key)
    selected_id = str(selected["candidate_id"])
    receipt = {
        "schema_version": 1,
        "kind": "controlled_candidate_selection_receipt",
        "protocol_id": development["protocol_id"],
        "dataset": development["dataset"],
        "fold": development["fold"],
        "development_bundle": str(development_path),
        "development_bundle_sha256": development["bundle_sha256"],
        # This is a commitment copied from the development bundle. The
        # selector deliberately does not open the final-test file.
        "final_apply_bundle_path": str(
            (
                development_path.parent
                / development["final_apply_bundle_path"]
            ).resolve()
        ),
        "final_apply_bundle_sha256": development[
            "final_apply_bundle_sha256"
        ],
        "test_bundle_path": str(
            (
                development_path.parent
                / development["test_bundle_path"]
            ).resolve()
        ),
        "test_bundle_sha256": development["test_bundle_sha256"],
        "selected_candidate_id": selected_id,
        "selected_candidate_spec": specs[selected_id],
        "selected_candidate_spec_path": str(spec_paths[selected_id]),
        "selected_candidate_spec_sha256": sha256_file(
            spec_paths[selected_id]
        ),
        "selected_validation_model_identity": summaries[selected_id][
            "model_identity"
        ],
        "judge_a": judge_a,
        "utility_tolerance": args.utility_tolerance,
        "locality_tolerance": args.locality_tolerance,
        "min_utility_probability_ratio": (
            args.min_utility_probability_ratio
        ),
        "selection_rule": (
            "eligible utility/locality first; then highest strict fact-level "
            "forget Judge-A pass rate, lowest sensitive-answer probability, "
            "highest utility"
        ),
        "candidate_reports": sorted(
            reports,
            key=lambda report: str(report["candidate_id"]),
        ),
        "hyperparameters_frozen": True,
        "final_apply_bundle_opened_by_selector": False,
        "test_bundle_opened_by_selector": False,
        "test_results_used_for_selection": False,
    }
    unhashed = dict(receipt)
    receipt["receipt_sha256"] = sha256_json(unhashed)
    output = Path(args.output).resolve()
    write_json(output, receipt)
    print(f"Selected {selected_id}; wrote {output}")


if __name__ == "__main__":
    main()
