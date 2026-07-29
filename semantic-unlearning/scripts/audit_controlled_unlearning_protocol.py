#!/usr/bin/env python3
"""Fail-closed audit of a generated controlled five-fold protocol."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from controlled_unlearning_protocol import (
    N_FOLDS,
    assert_prompt_partitions_disjoint,
    bundle_prompt_cases,
    load_development_bundle,
    load_final_apply_bundle,
    load_test_bundle,
    read_json,
    sha256_file,
    sha256_json,
    write_json,
)


REQUIRED_TEST_STYLES = {
    "paraphrase",
    "direct",
    "indirect",
    "cloze",
    "multiple_choice",
    "adversarial",
}
MC_FALLBACK_OPTIONS = {
    "none of the listed answers",
    "the information is not provided",
    "a different answer",
}


def _normalized(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def _assert_mc_options_are_partition_local(
    cases: Sequence[Any],
) -> None:
    allowed: Dict[str, set[str]] = defaultdict(set)
    for case in cases:
        allowed[case.partition].update(
            _normalized(value)
            for value in [
                *case.sensitive_answers,
                *case.acceptable_answers,
            ]
        )
    for case in cases:
        if case.style != "multiple_choice":
            continue
        options = {
            _normalized(value)
            for value in (case.metadata or {}).get("options", [])
        }
        unexpected = options - allowed[case.partition] - MC_FALLBACK_OPTIONS
        if unexpected:
            raise ValueError(
                "Multiple-choice options expose answers from another "
                f"partition in {case.case_id}: {sorted(unexpected)[:5]}"
            )


def _verify_hash(value: Mapping[str, Any]) -> None:
    stored = str(value.get("bundle_sha256", ""))
    unhashed = dict(value)
    unhashed.pop("bundle_sha256", None)
    if not stored or sha256_json(unhashed) != stored:
        raise ValueError("Manifest or bundle content hash mismatch")


def _verify_materialized(
    bundle_path: Path,
    materialized: Mapping[str, Any],
) -> int:
    count = 0
    for name, raw_path in materialized["files"].items():
        path = Path(raw_path)
        if not path.is_absolute():
            path = bundle_path.parent / path
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = materialized["sha256"][name]
        if actual != expected:
            raise ValueError(
                f"Materialized file {path} hash mismatch: "
                f"{actual} != {expected}"
            )
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    root = manifest_path.parent
    manifest = read_json(manifest_path)
    _verify_hash(manifest)
    if manifest.get("kind") != "controlled_unlearning_manifest":
        raise ValueError("Not a controlled protocol manifest")
    if int(manifest.get("n_folds", -1)) != N_FOLDS:
        raise ValueError(f"Expected exactly {N_FOLDS} folds")
    folds = manifest.get("folds", [])
    if len(folds) != N_FOLDS:
        raise ValueError(f"Expected exactly {N_FOLDS} fold entries")

    group_assignments: Dict[str, Counter[str]] = defaultdict(Counter)
    record_assignments: Dict[str, Counter[str]] = defaultdict(Counter)
    fold_reports: List[Dict[str, Any]] = []
    materialized_file_count = 0
    for expected_fold, fold_info in enumerate(folds):
        fold = int(fold_info["fold"])
        if fold != expected_fold:
            raise ValueError("Manifest folds must be ordered 0..4")
        development_path = root / fold_info["development_bundle"]
        final_apply_path = root / fold_info["final_apply_bundle"]
        test_path = root / fold_info["test_bundle"]
        development = load_development_bundle(development_path)
        final_apply = load_final_apply_bundle(final_apply_path)
        test = load_test_bundle(test_path)
        expected_hashes = {
            "development": (
                development["bundle_sha256"],
                fold_info["development_bundle_sha256"],
            ),
            "final_apply": (
                final_apply["bundle_sha256"],
                fold_info["final_apply_bundle_sha256"],
            ),
            "test": (
                test["bundle_sha256"],
                fold_info["test_bundle_sha256"],
            ),
        }
        for label, (actual, expected) in expected_hashes.items():
            if actual != expected:
                raise ValueError(
                    f"Fold {fold} {label} commitment mismatch"
                )
        if development["final_apply_bundle_sha256"] != final_apply[
            "bundle_sha256"
        ]:
            raise ValueError("Development/final-apply commitment mismatch")
        if development["test_bundle_sha256"] != test["bundle_sha256"]:
            raise ValueError("Development/test commitment mismatch")
        if test["final_apply_bundle_sha256"] != final_apply[
            "bundle_sha256"
        ]:
            raise ValueError("Test/final-apply commitment mismatch")
        if test.get("final_apply_bundle_path") != "final_apply.json":
            raise ValueError("Test bundle points to unexpected apply artifact")
        if final_apply.get("contains_judge_b_prompts") is not False:
            raise ValueError("Final-apply bundle does not forbid Judge-B prompts")
        if final_apply.get("prompt_cases"):
            raise ValueError("Final-apply bundle contains evaluation prompts")
        forbidden_test_keys = {
            "final_apply_records",
            "materialized_inputs",
        }
        if forbidden_test_keys & set(test):
            raise ValueError(
                "Locked test bundle improperly contains runtime apply data"
            )

        development_cases = bundle_prompt_cases(development)
        test_cases = bundle_prompt_cases(test)
        assert_prompt_partitions_disjoint(
            development_cases,
            test_cases,
        )
        _assert_mc_options_are_partition_local(development_cases)
        _assert_mc_options_are_partition_local(test_cases)
        train_records = development["records"]["train"]
        validation_records = development["records"]["validation"]
        apply_records = final_apply["final_apply_records"]
        partition_groups = {
            "train": {row["group_id"] for row in train_records},
            "validation": {
                row["group_id"] for row in validation_records
            },
            "final_apply": {row["group_id"] for row in apply_records},
        }
        names = list(partition_groups)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                overlap = partition_groups[left] & partition_groups[right]
                if overlap:
                    raise ValueError(
                        f"Fold {fold} {left}/{right} share groups: "
                        f"{sorted(overlap)[:5]}"
                    )
        for partition, records in (
            ("train", train_records),
            ("validation", validation_records),
            ("final_apply", apply_records),
        ):
            for row in records:
                group_assignments[str(row["group_id"])][partition] += 1
                record_assignments[str(row["record_id"])][partition] += 1
        # Utility test refs are intentionally not copied into final_apply.
        for ref in test["partitions"]["test_utility"]:
            group_assignments[str(ref["group_id"])]["final_apply"] += 1
            record_assignments[str(ref["record_id"])]["final_apply"] += 1
        known_test_record_ids = {
            ref["record_id"]
            for values in test["partitions"].values()
            for ref in values
        }
        unknown_test_sources = {
            case.source_record_id
            for case in test_cases
            if case.source_record_id not in known_test_record_ids
        }
        if unknown_test_sources:
            raise ValueError(
                "Test cases reference records outside locked test refs: "
                f"{sorted(unknown_test_sources)[:5]}"
            )
        forget_test_cases = [
            case
            for case in test_cases
            if case.expected_behavior == "avoid_sensitive"
        ]
        styles_by_record: Dict[str, set[str]] = defaultdict(set)
        for case in forget_test_cases:
            styles_by_record[case.source_record_id].add(case.style)
        incomplete = {
            record_id: sorted(REQUIRED_TEST_STYLES - styles)
            for record_id, styles in styles_by_record.items()
            if not REQUIRED_TEST_STYLES.issubset(styles)
        }
        if incomplete:
            raise ValueError(
                f"Fold {fold} forget prompt style coverage is incomplete: "
                f"{list(incomplete.items())[:5]}"
            )
        validation_paraphrases = Counter(
            case.source_record_id
            for case in development_cases
            if case.partition == "validation"
            and case.expected_behavior == "avoid_sensitive"
            and case.style == "paraphrase"
        )
        if validation_paraphrases and min(validation_paraphrases.values()) < 2:
            raise ValueError(
                "Validation forget facts need multiple paraphrases"
            )
        for materialized in development["materialized_inputs"].values():
            materialized_file_count += _verify_materialized(
                development_path,
                materialized,
            )
        materialized_file_count += _verify_materialized(
            final_apply_path,
            final_apply["materialized_inputs"],
        )
        fold_reports.append(
            {
                "fold": fold,
                "development_case_count": len(development_cases),
                "test_case_count": len(test_cases),
                "train_group_count": len(partition_groups["train"]),
                "validation_group_count": len(
                    partition_groups["validation"]
                ),
                "final_apply_group_count": len(
                    partition_groups["final_apply"]
                ),
                "judge_b_prompts_visible_to_apply_runner": False,
            }
        )

    bad_group_rotation = {
        group_id: dict(counts)
        for group_id, counts in group_assignments.items()
        if set(counts) != {"train", "validation", "final_apply"}
        or counts["validation"] != counts["final_apply"]
        or counts["train"] != 3 * counts["final_apply"]
    }
    if bad_group_rotation:
        raise ValueError(
            "Five-fold group rotation failed: "
            f"{list(bad_group_rotation.items())[:5]}"
        )
    bad_record_rotation = {
        record_id: dict(counts)
        for record_id, counts in record_assignments.items()
        if counts != Counter(
            {"train": 3, "validation": 1, "final_apply": 1}
        )
    }
    if bad_record_rotation:
        raise ValueError(
            "Five-fold record rotation failed: "
            f"{list(bad_record_rotation.items())[:5]}"
        )
    report = {
        "schema_version": 1,
        "kind": "controlled_protocol_audit",
        "passed": True,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_id": manifest["protocol_id"],
        "dataset": manifest["dataset"],
        "n_folds": N_FOLDS,
        "folds": fold_reports,
        "unique_group_count": len(group_assignments),
        "unique_record_count": len(record_assignments),
        "materialized_files_verified": materialized_file_count,
        "checks": {
            "content_hashes": "passed",
            "record_group_rotation_3_1_1": "passed",
            "train_validation_test_group_disjoint": "passed",
            "development_test_prompt_text_disjoint": "passed",
            "multiple_choice_answers_partition_local": "passed",
            "multiple_validation_paraphrases": "passed",
            "five_required_test_styles": "passed",
            "heldout_test_paraphrase": "passed",
            "final_apply_contains_no_judge_b_prompts": "passed",
            "test_contains_no_runtime_materialization": "passed",
        },
    }
    output = (
        Path(args.output).resolve()
        if args.output
        else root / "audit_report.json"
    )
    write_json(output, report)
    print(f"PASS: wrote {output}")


if __name__ == "__main__":
    main()
