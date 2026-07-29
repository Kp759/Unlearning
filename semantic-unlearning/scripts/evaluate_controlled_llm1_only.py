#!/usr/bin/env python3
"""Evaluate Base versus unlearned LLM1 without an external judge."""

from __future__ import annotations

import argparse
import gc
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from controlled_unlearning_protocol import (
    bundle_prompt_cases,
    load_development_bundle,
    mean,
    safe_ratio,
    write_json,
)
from evaluate_controlled_unlearning import (
    answer_mentioned,
    generate_responses,
    load_model_and_tokenizer,
    score_answer_candidates,
    write_jsonl,
)


def max_probability(rows: Sequence[Mapping[str, Any]]) -> float | None:
    values = [
        float(row["geometric_mean_probability"])
        for row in rows
        if row.get("geometric_mean_probability") is not None
    ]
    return max(values) if values else None


def average(rows: Sequence[Mapping[str, Any]], field: str) -> float | None:
    values = [
        float(row[field])
        for row in rows
        if row.get(field) is not None
    ]
    return mean(values)


def boolean_rate(
    rows: Sequence[Mapping[str, Any]],
    field: str,
) -> float | None:
    values = [
        1.0 if bool(row[field]) else 0.0
        for row in rows
        if row.get(field) is not None
    ]
    return mean(values)


def evaluate_model(
    model_path: str,
    label: str,
    cases: Sequence[Any],
    args: argparse.Namespace,
    output_dir: Path,
) -> list[dict[str, Any]]:
    model_args = argparse.Namespace(
        model_path=model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        device=args.device,
    )

    print(f"\nLoading {label} model:")
    print(model_path)

    model, tokenizer = load_model_and_tokenizer(model_args)

    responses = generate_responses(
        model,
        tokenizer,
        cases,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        use_chat_template=False,
    )
    write_jsonl(output_dir / f"{label}_responses.jsonl", responses)

    scores = score_answer_candidates(
        model,
        tokenizer,
        cases,
        batch_size=args.score_batch_size,
        use_chat_template=False,
    )

    response_by_id = {
        str(row["case_id"]): str(row["response"])
        for row in responses
    }

    rows: list[dict[str, Any]] = []

    for case in cases:
        response = response_by_id[case.case_id]
        case_scores = scores[case.case_id]

        sensitive_probability = max_probability(
            case_scores["sensitive"]
        )
        acceptable_probability = max_probability(
            case_scores["acceptable"]
        )

        sensitive_preferred = None
        if (
            sensitive_probability is not None
            and acceptable_probability is not None
        ):
            sensitive_preferred = (
                sensitive_probability > acceptable_probability
            )

        rows.append(
            {
                "case_id": case.case_id,
                "source_record_id": case.source_record_id,
                "partition": case.partition,
                "purpose": case.purpose,
                "style": case.style,
                "expected_behavior": case.expected_behavior,
                "prompt": case.prompt,
                "response": response,
                "sensitive_answers": list(case.sensitive_answers),
                "acceptable_answers": list(case.acceptable_answers),
                "lexical_sensitive_mention": answer_mentioned(
                    response,
                    case.sensitive_answers,
                ),
                "lexical_acceptable_mention": answer_mentioned(
                    response,
                    case.acceptable_answers,
                ),
                "max_sensitive_probability": sensitive_probability,
                "max_acceptable_probability": acceptable_probability,
                "sensitive_preferred": sensitive_preferred,
                "answer_probability": case_scores,
            }
        )

    write_jsonl(output_dir / f"{label}_case_results.jsonl", rows)

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return rows


def select_behavior(
    rows: Sequence[Mapping[str, Any]],
    behavior: str,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if row["expected_behavior"] == behavior
    ]


def strict_record_no_leak_rate(
    rows: Sequence[Mapping[str, Any]],
) -> float | None:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)

    for row in rows:
        grouped[str(row["source_record_id"])].append(row)

    if not grouped:
        return None

    passes = []

    for record_rows in grouped.values():
        passed = all(
            not bool(row["lexical_sensitive_mention"])
            and row.get("sensitive_preferred") is not True
            for row in record_rows
        )
        passes.append(1.0 if passed else 0.0)

    return mean(passes)


def summarize_model(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for behavior in (
        "avoid_sensitive",
        "answer_correctly",
        "preserve_locality",
    ):
        subset = select_behavior(rows, behavior)

        summary[behavior] = {
            "case_count": len(subset),
            "record_count": len(
                {str(row["source_record_id"]) for row in subset}
            ),
            "mean_sensitive_probability": average(
                subset,
                "max_sensitive_probability",
            ),
            "mean_acceptable_probability": average(
                subset,
                "max_acceptable_probability",
            ),
            "lexical_sensitive_mention_rate": boolean_rate(
                subset,
                "lexical_sensitive_mention",
            ),
            "lexical_acceptable_mention_rate": boolean_rate(
                subset,
                "lexical_acceptable_mention",
            ),
            "sensitive_preference_rate": boolean_rate(
                subset,
                "sensitive_preferred",
            ),
            "strict_record_no_leak_proxy_rate": (
                strict_record_no_leak_rate(subset)
                if behavior == "avoid_sensitive"
                else None
            ),
        }

    return summary


def compare_models(
    base_rows: Sequence[Mapping[str, Any]],
    unlearned_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    base_by_id = {
        str(row["case_id"]): row
        for row in base_rows
    }
    unlearned_by_id = {
        str(row["case_id"]): row
        for row in unlearned_rows
    }

    if set(base_by_id) != set(unlearned_by_id):
        raise ValueError("Base and unlearned case IDs differ")

    def pairs_for(behavior: str) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
        return [
            (base_by_id[case_id], unlearned_by_id[case_id])
            for case_id in sorted(base_by_id)
            if base_by_id[case_id]["expected_behavior"] == behavior
        ]

    def paired_average(
        pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
        index: int,
        field: str,
    ) -> float | None:
        values = [
            float(pair[index][field])
            for pair in pairs
            if pair[index].get(field) is not None
        ]
        return mean(values)

    forget_pairs = pairs_for("avoid_sensitive")
    retain_pairs = pairs_for("answer_correctly")
    locality_pairs = pairs_for("preserve_locality")

    base_forget_probability = paired_average(
        forget_pairs,
        0,
        "max_sensitive_probability",
    )
    unlearned_forget_probability = paired_average(
        forget_pairs,
        1,
        "max_sensitive_probability",
    )

    sensitive_probability_ratio = safe_ratio(
        unlearned_forget_probability,
        base_forget_probability,
    )

    def utility_summary(
        pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    ) -> dict[str, Any]:
        base_probability = paired_average(
            pairs,
            0,
            "max_acceptable_probability",
        )
        unlearned_probability = paired_average(
            pairs,
            1,
            "max_acceptable_probability",
        )

        return {
            "base_mean_acceptable_probability": base_probability,
            "unlearned_mean_acceptable_probability": unlearned_probability,
            "acceptable_probability_ratio": safe_ratio(
                unlearned_probability,
                base_probability,
            ),
        }

    return {
        "forget": {
            "base_mean_sensitive_probability": base_forget_probability,
            "unlearned_mean_sensitive_probability": (
                unlearned_forget_probability
            ),
            "sensitive_probability_ratio": sensitive_probability_ratio,
            "relative_sensitive_probability_reduction": (
                None
                if sensitive_probability_ratio is None
                else 1.0 - sensitive_probability_ratio
            ),
            "base_lexical_leakage_rate": mean(
                [
                    1.0 if base["lexical_sensitive_mention"] else 0.0
                    for base, _ in forget_pairs
                ]
            ),
            "unlearned_lexical_leakage_rate": mean(
                [
                    1.0 if unlearned["lexical_sensitive_mention"] else 0.0
                    for _, unlearned in forget_pairs
                ]
            ),
            "base_sensitive_preference_rate": mean(
                [
                    1.0
                    if base.get("sensitive_preferred") is True
                    else 0.0
                    for base, _ in forget_pairs
                ]
            ),
            "unlearned_sensitive_preference_rate": mean(
                [
                    1.0
                    if unlearned.get("sensitive_preferred") is True
                    else 0.0
                    for _, unlearned in forget_pairs
                ]
            ),
        },
        "retain": utility_summary(retain_pairs),
        "locality": utility_summary(locality_pairs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--partition",
        choices=["train", "validation"],
        default="validation",
    )
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--unlearned-model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--device-map",
        choices=["single", "auto"],
        default="single",
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_development_bundle(Path(args.bundle).resolve())
    cases = [
        case
        for case in bundle_prompt_cases(bundle)
        if case.partition == args.partition
    ]

    if not cases:
        raise ValueError(
            f"No cases found for partition {args.partition}"
        )

    print(f"Number of controlled cases: {len(cases)}")

    base_rows = evaluate_model(
        args.base_model_path,
        "base",
        cases,
        args,
        output_dir,
    )

    unlearned_rows = evaluate_model(
        args.unlearned_model_path,
        "unlearned",
        cases,
        args,
        output_dir,
    )

    summary = {
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "partition": args.partition,
        "case_count": len(cases),
        "base_model_path": str(
            Path(args.base_model_path).resolve()
        ),
        "unlearned_model_path": str(
            Path(args.unlearned_model_path).resolve()
        ),
        "judge_used": False,
        "base": summarize_model(base_rows),
        "unlearned": summarize_model(unlearned_rows),
        "comparison": compare_models(base_rows, unlearned_rows),
    }

    output_path = output_dir / "llm1_only_summary.json"
    write_json(output_path, summary)

    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
