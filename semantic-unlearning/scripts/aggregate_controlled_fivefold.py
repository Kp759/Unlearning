#!/usr/bin/env python3
"""Aggregate exactly five locked, manually audited Judge-B fold results."""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from controlled_unlearning_protocol import N_FOLDS, read_json, write_json


T_CRITICAL_95_DF4 = 2.7764451051977987


def _parse_result(value: str) -> Tuple[int, Path]:
    if "=" not in value:
        raise ValueError(f"Expected FOLD=PATH, got {value!r}")
    fold_text, path_text = value.split("=", 1)
    return int(fold_text), Path(path_text).resolve()


def _number(
    result: Mapping[str, Any],
    behavior: str,
    field: str,
) -> Optional[float]:
    value = (
        result.get("metrics", {})
        .get("by_behavior", {})
        .get(behavior, {})
        .get(field)
    )
    return None if value is None else float(value)


def _series(values: Sequence[Optional[float]]) -> Dict[str, Any]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {
            "fold_values": list(values),
            "mean": None,
            "sample_std": None,
            "ci95": None,
        }
    average = statistics.mean(present)
    std = statistics.stdev(present) if len(present) > 1 else 0.0
    half_width = (
        T_CRITICAL_95_DF4 * std / math.sqrt(len(present))
        if len(present) == N_FOLDS
        else None
    )
    return {
        "fold_values": list(values),
        "mean": average,
        "sample_std": std,
        "ci95": (
            [average - half_width, average + half_width]
            if half_width is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="Audited final evaluation as FOLD=PATH; provide all five.",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = read_json(manifest_path)
    if manifest.get("kind") != "controlled_unlearning_manifest":
        raise ValueError("Input is not a controlled protocol manifest")
    if int(manifest.get("n_folds", -1)) != N_FOLDS:
        raise ValueError(f"Manifest must contain exactly {N_FOLDS} folds")
    result_paths = dict(_parse_result(value) for value in args.result)
    if set(result_paths) != set(range(N_FOLDS)):
        raise ValueError(
            f"Results must contain folds 0..{N_FOLDS - 1} exactly"
        )
    manifest_folds = {
        int(item["fold"]): item
        for item in manifest["folds"]
    }
    results: List[Dict[str, Any]] = []
    candidate_ids: List[str] = []
    candidate_spec_hashes: List[str] = []
    release_flags: List[bool] = []
    manual_gate_flags: List[bool] = []
    utility_guardrail_flags: List[bool] = []
    for fold in range(N_FOLDS):
        result = read_json(result_paths[fold])
        required = {
            "kind": "controlled_unlearning_evaluation",
            "phase": "test",
            "protocol_id": manifest["protocol_id"],
            "dataset": manifest["dataset"],
            "fold": fold,
            "bundle_sha256": manifest_folds[fold]["test_bundle_sha256"],
        }
        for field, expected in required.items():
            if result.get(field) != expected:
                raise ValueError(
                    f"Fold {fold} {field}={result.get(field)!r}; "
                    f"expected {expected!r}"
                )
        if not result.get("manual_audit", {}).get("complete"):
            raise ValueError(f"Fold {fold} lacks completed manual audit")
        if not isinstance(result.get("release_ready"), bool):
            raise ValueError(f"Fold {fold} lacks release_ready status")
        if not isinstance(
            result.get("manual_audit", {}).get("gate_passed"),
            bool,
        ):
            raise ValueError(f"Fold {fold} lacks manual-audit gate status")
        if not isinstance(
            result.get("utility_guardrail", {}).get("passed"),
            bool,
        ):
            raise ValueError(
                f"Fold {fold} lacks a locked-test Base utility guardrail"
            )
        controls = result.get("controls", {})
        if controls.get("test_results_used_for_repair") is not False:
            raise ValueError(f"Fold {fold} does not prove test isolation")
        if controls.get("final_test_rerun"):
            raise ValueError(f"Fold {fold} is a disclosed final-test rerun")
        candidate_ids.append(str(result["candidate_id"]))
        candidate_spec_hash = str(
            result.get("candidate_spec_sha256", "")
        )
        if not candidate_spec_hash:
            raise ValueError(
                f"Fold {fold} lacks a frozen candidate-spec hash"
            )
        candidate_spec_hashes.append(candidate_spec_hash)
        release_flags.append(bool(result["release_ready"]))
        manual_gate_flags.append(
            bool(result["manual_audit"]["gate_passed"])
        )
        utility_guardrail_flags.append(
            bool(result["utility_guardrail"]["passed"])
        )
        results.append(result)

    metrics = {
        "forget_strict_fact_pass_rate": _series(
            [
                _number(
                    result,
                    "avoid_sensitive",
                    "record_strict_all_prompts_pass_rate",
                )
                for result in results
            ]
        ),
        "forget_judge_pass_rate": _series(
            [
                _number(result, "avoid_sensitive", "judge_pass_rate")
                for result in results
            ]
        ),
        "forget_sensitive_probability": _series(
            [
                _number(
                    result,
                    "avoid_sensitive",
                    "mean_max_sensitive_probability",
                )
                for result in results
            ]
        ),
        "forget_sensitive_preference_rate": _series(
            [
                _number(
                    result,
                    "avoid_sensitive",
                    "sensitive_preference_rate",
                )
                for result in results
            ]
        ),
        "retain_judge_pass_rate": _series(
            [
                _number(result, "answer_correctly", "judge_pass_rate")
                for result in results
            ]
        ),
        "retain_acceptable_answer_probability": _series(
            [
                _number(
                    result,
                    "answer_correctly",
                    "mean_max_acceptable_probability",
                )
                for result in results
            ]
        ),
        "locality_judge_pass_rate": _series(
            [
                _number(result, "preserve_locality", "judge_pass_rate")
                for result in results
            ]
        ),
        "locality_acceptable_answer_probability": _series(
            [
                _number(
                    result,
                    "preserve_locality",
                    "mean_max_acceptable_probability",
                )
                for result in results
            ]
        ),
        "utility_absolute_drop_vs_base": _series(
            [
                float(result["utility_guardrail"]["utility_absolute_drop"])
                for result in results
            ]
        ),
        "locality_absolute_drop_vs_base": _series(
            [
                float(result["utility_guardrail"]["locality_absolute_drop"])
                for result in results
            ]
        ),
        "utility_answer_probability_ratio_vs_base": _series(
            [
                float(
                    result["utility_guardrail"][
                        "utility_answer_probability_ratio"
                    ]
                )
                for result in results
            ]
        ),
        "manual_judge_agreement_rate": _series(
            [
                float(result["manual_audit"]["agreement_rate"])
                for result in results
            ]
        ),
    }
    style_names = sorted(
        {
            style
            for result in results
            for style in (
                result.get("metrics", {})
                .get("by_behavior_and_style", {})
                .get("avoid_sensitive", {})
            )
        }
    )
    styles = {
        style: _series(
            [
                (
                    float(
                        result["metrics"]["by_behavior_and_style"][
                            "avoid_sensitive"
                        ][style]["judge_pass_rate"]
                    )
                    if style
                    in result["metrics"]["by_behavior_and_style"].get(
                        "avoid_sensitive",
                        {},
                    )
                    else None
                )
                for result in results
            ]
        )
        for style in style_names
    }
    aggregate = {
        "schema_version": 1,
        "kind": "controlled_unlearning_fivefold_aggregate",
        "protocol_id": manifest["protocol_id"],
        "dataset": manifest["dataset"],
        "fold_selected_candidates": [
            {
                "fold": fold,
                "candidate_id": candidate_ids[fold],
                "candidate_spec_sha256": candidate_spec_hashes[fold],
            }
            for fold in range(N_FOLDS)
        ],
        "n_folds": N_FOLDS,
        "fold_results": [
            {
                "fold": fold,
                "path": str(result_paths[fold]),
                "case_count": results[fold]["case_count"],
            }
            for fold in range(N_FOLDS)
        ],
        "metrics": metrics,
        "forget_by_style": styles,
        "confidence_interval": (
            "two-sided 95% Student-t interval over five fold-level values"
        ),
        "controls": {
            "all_folds_release_ready": all(release_flags),
            "all_manual_audits_complete": True,
            "all_manual_audit_gates_passed": all(manual_gate_flags),
            "all_final_utility_guardrails_passed": all(
                utility_guardrail_flags
            ),
            "candidate_selection": (
                "nested per fold using that fold's development/validation "
                "data only"
            ),
            "test_results_used_for_repair": False,
        },
        "release_ready": all(release_flags),
    }
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "fivefold_summary.json", aggregate)
    lines = [
        f"# {manifest['dataset'].upper()} controlled five-fold evaluation",
        "",
        "Candidates were selected independently inside each fold from "
        "development/validation evidence only.",
        "",
        "| Metric | Mean | Sample SD | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        mean_value = values["mean"]
        std_value = values["sample_std"]
        interval = values["ci95"]
        rendered_interval = (
            "n/a"
            if interval is None
            else f"[{interval[0]:.6f}, {interval[1]:.6f}]"
        )
        lines.append(
            f"| {name} | "
            f"{'n/a' if mean_value is None else f'{mean_value:.6f}'} | "
            f"{'n/a' if std_value is None else f'{std_value:.6f}'} | "
            f"{rendered_interval} |"
        )
    lines.extend(
        [
            "",
            "All five folds used locked Judge-B prompts, completed manual "
            "audits, and reported token-probability/locality evidence. No "
            "final-test result was eligible for repair or candidate selection.",
            "",
            (
                "Release gate: PASS."
                if aggregate["release_ready"]
                else (
                    "Release gate: FAIL. Inspect fold-level manual-audit and "
                    "Base-relative utility guardrails."
                )
            ),
            "",
        ]
    )
    (output_dir / "fivefold_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(f"Wrote {output_dir / 'fivefold_summary.json'}")


if __name__ == "__main__":
    main()
