#!/usr/bin/env python3
"""Evaluate Base versus unlearned TOFU LLM1 without an external judge.

The generic LLM1-only evaluator historically used plain prompts for MCF/ZsRE.
TOFU requires the tokenizer chat template, matching the controlled protocol and
the TOFU training/evaluation path. This wrapper forces that prompt protocol and
never loads Judge A or Judge B.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import evaluate_controlled_llm1_only as llm1
from controlled_unlearning_protocol import (
    bundle_prompt_cases,
    load_development_bundle,
    write_json,
)
from evaluate_controlled_unlearning import (
    generate_responses as generate_responses_shared,
    score_answer_candidates as score_answer_candidates_shared,
)


def generate_responses_tofu(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    batch_size: int,
    max_new_tokens: int,
    use_chat_template: bool,
):
    del use_chat_template
    return generate_responses_shared(
        model,
        tokenizer,
        cases,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        use_chat_template=True,
    )


def score_answer_candidates_tofu(
    model: Any,
    tokenizer: Any,
    cases: Sequence[Any],
    *,
    batch_size: int,
    use_chat_template: bool,
):
    del use_chat_template
    return score_answer_candidates_shared(
        model,
        tokenizer,
        cases,
        batch_size=batch_size,
        use_chat_template=True,
    )


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = Path(args.bundle).resolve()
    bundle = load_development_bundle(bundle_path)
    if bundle.get("dataset") != "tofu":
        raise ValueError(
            "This evaluator is TOFU-only; received "
            f"dataset={bundle.get('dataset')!r}"
        )

    cases = [
        case
        for case in bundle_prompt_cases(bundle)
        if case.partition == args.partition
    ]
    if not cases:
        raise ValueError(
            f"No cases found for partition {args.partition}"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # evaluate_model resolves these module globals at runtime.
    llm1.generate_responses = generate_responses_tofu
    llm1.score_answer_candidates = score_answer_candidates_tofu

    print(f"Number of controlled TOFU cases: {len(cases)}")
    print("Prompt protocol: tokenizer chat template")
    print("External judge used: False")

    base_rows = llm1.evaluate_model(
        args.base_model_path,
        "base",
        cases,
        args,
        output_dir,
    )
    unlearned_rows = llm1.evaluate_model(
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
        "base_model_path": str(Path(args.base_model_path).resolve()),
        "unlearned_model_path": str(
            Path(args.unlearned_model_path).resolve()
        ),
        "judge_used": False,
        "use_chat_template": True,
        "base": llm1.summarize_model(base_rows),
        "unlearned": llm1.summarize_model(unlearned_rows),
        "comparison": llm1.compare_models(base_rows, unlearned_rows),
    }

    output_path = output_dir / "llm1_only_summary.json"
    write_json(output_path, summary)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
