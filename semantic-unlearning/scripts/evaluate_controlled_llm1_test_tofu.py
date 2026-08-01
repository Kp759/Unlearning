#!/usr/bin/env python3
"""Post-hoc LLM1-only evaluation of a TOFU checkpoint on a locked test bundle."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import evaluate_controlled_llm1_only as llm1
from controlled_unlearning_protocol import (
    bundle_prompt_cases,
    load_test_bundle,
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
    parser.add_argument("--test-bundle", required=True)
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
    parser.add_argument(
        "--confirm-open-locked-test",
        action="store_true",
        help="Required confirmation because this opens the locked test bundle.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not args.confirm_open_locked_test:
        raise SystemExit(
            "Refusing to open the locked test bundle without "
            "--confirm-open-locked-test"
        )

    bundle_path = Path(args.test_bundle).resolve()
    bundle = load_test_bundle(bundle_path)
    if bundle.get("dataset") != "tofu":
        raise ValueError(
            "This evaluator is TOFU-only; received "
            f"dataset={bundle.get('dataset')!r}"
        )

    cases = bundle_prompt_cases(bundle)
    if not cases:
        raise ValueError("The TOFU test bundle has no prompt cases")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty test output directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    llm1.generate_responses = generate_responses_tofu
    llm1.score_answer_candidates = score_answer_candidates_tofu

    print(f"Number of locked TOFU test cases: {len(cases)}")
    print("Prompt protocol: tokenizer chat template")
    print("External judge used: False")
    print("Evaluation status: post-hoc test analysis")

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
        "partition": "test",
        "case_count": len(cases),
        "base_model_path": str(Path(args.base_model_path).resolve()),
        "unlearned_model_path": str(
            Path(args.unlearned_model_path).resolve()
        ),
        "judge_used": False,
        "use_chat_template": True,
        "evaluation_status": "post_hoc_test_analysis",
        "warning": (
            "This checkpoint was selected/evaluated without a frozen Judge-A "
            "selection receipt and is not an official final-apply result."
        ),
        "base": llm1.summarize_model(base_rows),
        "unlearned": llm1.summarize_model(unlearned_rows),
        "comparison": llm1.compare_models(base_rows, unlearned_rows),
    }

    output_path = output_dir / "llm1_test_summary.json"
    write_json(output_path, summary)
    print(f"\nWrote: {output_path}")


if __name__ == "__main__":
    main()
