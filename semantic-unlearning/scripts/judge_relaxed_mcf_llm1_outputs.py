#!/usr/bin/env python3
"""Judge immutable relaxed MCF LLM1 outputs without regenerating LLM1.

This runner is intentionally post-hoc. It evaluates already-saved Base and/or
unlearned responses from ``run_relaxed_mcf_llm1_test.py`` with an independent
Judge B. It never modifies checkpoints and never feeds judgments back into
repair or candidate selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from controlled_llm_judge import (
    JudgeClient,
    assert_judge_phase,
    load_judge_config,
    public_judge_config,
)
from controlled_unlearning_protocol import (
    bundle_prompt_cases,
    load_test_bundle,
    read_json,
    sha256_file,
    stable_stratified_manual_audit_sample,
    write_json,
)
from evaluate_controlled_unlearning import read_jsonl, write_jsonl


FORGET_LABELS = (
    "FACT_LEAKED",
    "PARTIAL_LEAK",
    "CORRECT_ABSTENTION",
    "HALLUCINATION",
    "AMBIGUOUS",
)


def _new_judge_b_v2_output_dir(root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = root / f"judge_b_v2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _resolve_relative(anchor: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = anchor.parent / path
    return path.resolve()


def _load_case_rows(
    path: Path,
    cases: Sequence[Any],
) -> list[Dict[str, Any]]:
    rows = read_jsonl(path)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", ""))
        if not case_id:
            raise ValueError(f"Row in {path} lacks case_id")
        if case_id in by_id:
            raise ValueError(f"Duplicate case_id {case_id} in {path}")
        by_id[case_id] = dict(row)

    required = {str(case.case_id) for case in cases}
    missing = sorted(required - set(by_id))
    if missing:
        raise ValueError(
            f"Case mismatch for {path}: missing={missing[:5]}"
        )

    ordered: list[Dict[str, Any]] = []
    for case in cases:
        row = by_id[str(case.case_id)]
        checks = {
            "prompt": case.prompt,
            "expected_behavior": case.expected_behavior,
            "style": case.style,
            "source_record_id": case.source_record_id,
        }
        for field, expected in checks.items():
            if str(row.get(field)) != str(expected):
                raise ValueError(
                    f"Immutable case mismatch for {case.case_id}: {field}"
                )
        ordered.append(row)
    return ordered


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = Counter(
        str(row["judgment"]["classification"]).upper()
        for row in rows
    )
    n = len(rows)
    leak_count = counts["FACT_LEAKED"] + counts["PARTIAL_LEAK"]
    return {
        "case_count": n,
        "classification_counts": {
            label: int(counts[label]) for label in FORGET_LABELS
        },
        "semantic_leak_rate": leak_count / n if n else None,
        "fact_leak_rate": counts["FACT_LEAKED"] / n if n else None,
        "partial_leak_rate": counts["PARTIAL_LEAK"] / n if n else None,
        "correct_abstention_rate": (
            counts["CORRECT_ABSTENTION"] / n if n else None
        ),
        "hallucination_rate": counts["HALLUCINATION"] / n if n else None,
        "ambiguous_rate": counts["AMBIGUOUS"] / n if n else None,
        "judge_pass_rate": (
            sum(1 for row in rows if row["judgment"]["passed"]) / n
            if n
            else None
        ),
        "mean_confidence": (
            sum(float(row["judgment"]["confidence"]) for row in rows) / n
            if n
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm1-summary", required=True)
    parser.add_argument("--judge-b-config", required=True)
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Parent directory under which a new timestamped "
            "judge_b_v2_<UTC timestamp> directory is created"
        ),
    )
    parser.add_argument(
        "--models",
        choices=["base", "unlearned", "both"],
        default="both",
    )
    parser.add_argument(
        "--behavior",
        choices=["avoid_sensitive", "answer_correctly", "preserve_locality", "all"],
        default="avoid_sensitive",
    )
    parser.add_argument("--manual-audit-count", type=int, default=30)
    parser.add_argument("--confirm-posthoc-judge-b", action="store_true")
    args = parser.parse_args()

    if not args.confirm_posthoc_judge_b:
        raise SystemExit(
            "Refusing to judge locked outputs without --confirm-posthoc-judge-b"
        )
    if args.manual_audit_count < 0:
        raise ValueError("--manual-audit-count must be non-negative")
    if args.behavior != "avoid_sensitive":
        raise ValueError(
            "This post-hoc runner currently supports --behavior avoid_sensitive "
            "only; use the original preregistered runner for utility judging."
        )

    output_root = Path(args.output_dir).resolve()

    summary_path = Path(args.llm1_summary).resolve()
    llm1 = read_json(summary_path)
    expected = {
        "kind": "relaxed_mcf_llm1_test_evaluation",
        "protocol_mode": "relaxed_report_only",
        "phase": "test",
        "dataset": "mcf",
        "judge_used": False,
        "test_results_used_for_repair": False,
    }
    for field, value in expected.items():
        if llm1.get(field) != value:
            raise ValueError(
                f"LLM1 summary {field}={llm1.get(field)!r}; expected {value!r}"
            )

    prereg_path = _resolve_relative(
        summary_path,
        str(llm1["preregistration"]),
    )
    prereg = read_json(prereg_path)
    test_path = _resolve_relative(
        prereg_path,
        str(prereg["test_bundle_path"]),
    )
    bundle = load_test_bundle(test_path)
    if bundle["bundle_sha256"] != llm1["test_bundle_sha256"]:
        raise ValueError("Locked test bundle commitment mismatch")
    for field in ("dataset", "fold"):
        if bundle[field] != llm1[field]:
            raise ValueError(f"Locked test {field} mismatch")

    judge_config = load_judge_config(Path(args.judge_b_config).resolve())
    assert_judge_phase(judge_config, bundle_phase="test")
    judge = JudgeClient(judge_config)

    all_cases = bundle_prompt_cases(bundle)
    cases = [
        case
        for case in all_cases
        if case.expected_behavior == "avoid_sensitive"
    ]
    if not cases:
        raise ValueError("No locked avoid_sensitive cases were found")

    source_dir = summary_path.parent / "llm1_test"
    labels = ["base", "unlearned"] if args.models == "both" else [args.models]

    audit_seed = int(bundle["seed"]) + int(bundle["fold"]) * 1009
    planned_audit = stable_stratified_manual_audit_sample(
        [
            {
                "case_id": case.case_id,
                "expected_behavior": case.expected_behavior,
                "style": case.style,
            }
            for case in cases
        ],
        count=min(args.manual_audit_count, len(cases)),
        seed=audit_seed,
    )
    audit_ids = {str(row["case_id"]) for row in planned_audit}

    output_dir = _new_judge_b_v2_output_dir(output_root)
    model_results: Dict[str, Any] = {}
    judged_by_label: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for label in labels:
        source_path = source_dir / f"{label}_case_results.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows = _load_case_rows(source_path, cases)
        judged_rows: list[Dict[str, Any]] = []
        for index, (case, row) in enumerate(zip(cases, rows), 1):
            response = str(row.get("response", ""))
            judgment = judge.judge(case, response)
            judged_rows.append(
                {
                    **row,
                    "model_label": label,
                    "judgment": judgment,
                    "case_position": index,
                }
            )
            if index % 10 == 0 or index == len(cases):
                print(f"{label}: judged {index}/{len(cases)}")

        out_path = output_dir / f"{label}_judge_b_results.jsonl"
        write_jsonl(out_path, judged_rows)
        model_results[label] = {
            "source_path": str(source_path),
            "source_sha256": sha256_file(source_path),
            "judged_results": str(out_path),
            "judged_results_sha256": sha256_file(out_path),
            "metrics": _summary(judged_rows),
        }
        judged_by_label[label] = {
            str(row["case_id"]): row for row in judged_rows
        }

    comparison = None
    if "base" in model_results and "unlearned" in model_results:
        base_metrics = model_results["base"]["metrics"]
        unlearned_metrics = model_results["unlearned"]["metrics"]
        comparison = {
            "semantic_leak_rate_change": (
                unlearned_metrics["semantic_leak_rate"]
                - base_metrics["semantic_leak_rate"]
            ),
            "semantic_leak_relative_reduction": (
                None
                if not base_metrics["semantic_leak_rate"]
                else 1.0
                - unlearned_metrics["semantic_leak_rate"]
                / base_metrics["semantic_leak_rate"]
            ),
            "correct_abstention_rate_change": (
                unlearned_metrics["correct_abstention_rate"]
                - base_metrics["correct_abstention_rate"]
            ),
            "hallucination_rate_change": (
                unlearned_metrics["hallucination_rate"]
                - base_metrics["hallucination_rate"]
            ),
        }

    audit_rows = []
    for case in cases:
        if str(case.case_id) not in audit_ids:
            continue
        audit_rows.append(
            {
                "case_id": case.case_id,
                "source_record_id": case.source_record_id,
                "style": case.style,
                "expected_behavior": case.expected_behavior,
                "prompt": case.prompt,
                "sensitive_answers": list(case.sensitive_answers),
                "acceptable_answers": list(case.acceptable_answers),
                "base": judged_by_label.get("base", {}).get(str(case.case_id)),
                "unlearned": judged_by_label.get("unlearned", {}).get(
                    str(case.case_id)
                ),
                "human_base_classification": None,
                "human_unlearned_classification": None,
                "reviewer": None,
                "notes": None,
            }
        )
    audit_path = output_dir / "manual_audit_queue.jsonl"
    write_jsonl(audit_path, audit_rows)

    final_summary = {
        "schema_version": 2,
        "kind": "relaxed_mcf_posthoc_judge_b_v2_evaluation",
        "protocol_mode": "relaxed_report_only_posthoc_judge_b",
        "posthoc": True,
        "not_originally_preregistered": True,
        "test_results_used_for_repair": False,
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "candidate_id": llm1["candidate_id"],
        "behavior": "avoid_sensitive",
        "case_count_per_model": len(cases),
        "llm1_summary": str(summary_path),
        "llm1_summary_sha256": sha256_file(summary_path),
        "test_bundle": str(test_path),
        "test_bundle_sha256": bundle["bundle_sha256"],
        "judge": public_judge_config(judge_config),
        "prompt_version": judge_config.prompt_version,
        "judge_request_count": judge.request_count,
        "models": model_results,
        "comparison": comparison,
        "manual_audit": {
            "selection_uses_outputs": False,
            "count": len(audit_rows),
            "queue_path": str(audit_path),
            "queue_sha256": sha256_file(audit_path),
            "complete": False,
        },
    }
    summary_out = output_dir / "judge_b_summary.json"
    write_json(summary_out, final_summary)
    print(f"Wrote {summary_out}")
    print(f"Manual audit queue: {audit_path}")


if __name__ == "__main__":
    main()
