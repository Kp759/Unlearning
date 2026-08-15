#!/usr/bin/env python3
"""Evaluate a frozen TOFU locked checkpoint on local final-eval-only views.

This evaluator is intentionally path-based: it reads the exact files emitted by
``build_tofu_zerounlearn_locked_split.py`` and never resamples them.  It reports
(1) the 50 direct deletion requests, (2) their benchmark-provided unseen
paraphrased questions, (3) the remaining 150 forget05 direct QAs, (4) their
paraphrases, and (5) 1,000 retain95 utility QAs.  Perturbed-answer truth-ratio
is computed only for the paraphrase views, after the checkpoint is frozen.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from tqdm import tqdm

from controlled_unlearning_protocol import load_json_or_jsonl
from tofu_eval import Evaluator


GROUPS = (
    "forget_direct",
    "forget_paraphrase",
    "heldout_direct",
    "heldout_paraphrase",
    "retain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-model-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--skip-generation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip greedy generation/ROUGE-L for a faster probability-only pass.",
    )
    return parser.parse_args()


def dtype_name(value: str) -> str:
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[value]


def safe_mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def load_group(eval_dir: Path, name: str) -> List[Dict[str, Any]]:
    path = eval_dir / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = load_json_or_jsonl(path)
    for position, row in enumerate(rows):
        if not row.get("question") or not row.get("answer"):
            raise ValueError(f"{name} row {position} lacks question/answer")
    return [dict(row) for row in rows]


def perturbed_answers(row: Dict[str, Any]) -> List[str]:
    value = row.get("perturbed_answer", row.get("perturbed_answers"))
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def truth_ratio(evaluator: Evaluator, row: Dict[str, Any]) -> Optional[float]:
    wrong = perturbed_answers(row)
    if not wrong:
        return None
    correct_logp = evaluator.answer_log_prob(str(row["question"]), str(row["answer"]))
    wrong_logps = [
        evaluator.answer_log_prob(str(row["question"]), answer) for answer in wrong
    ]
    return float(math.exp(safe_mean(wrong_logps) - correct_logp))


def evaluate_rows(
    evaluator: Evaluator,
    rows: Sequence[Dict[str, Any]],
    *,
    name: str,
    max_new_tokens: int,
    skip_generation: bool,
    include_truth_ratio: bool,
) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    probabilities: List[float] = []
    rouge_values: List[float] = []
    truth_values: List[float] = []

    for row in tqdm(rows, desc=f"locked-eval[{name}]"):
        question = str(row["question"])
        answer = str(row["answer"])
        logp = evaluator.answer_log_prob(question, answer)
        probability = 0.0 if logp < -50 else float(math.exp(logp))
        probabilities.append(probability)
        item: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "source_index": row.get("_source_index"),
            "answer_log_probability": logp,
            "answer_probability": probability,
        }
        if include_truth_ratio:
            tr = truth_ratio(evaluator, row)
            item["truth_ratio_raw"] = tr
            if tr is not None:
                truth_values.append(tr)
        if not skip_generation:
            generated = evaluator.generate_answer(question, max_new_tokens=max_new_tokens)
            rouge = evaluator.rouge_l_recall(answer, generated)
            item["generated"] = generated
            item["rougeL_recall"] = rouge
            rouge_values.append(rouge)
        details.append(item)

    summary: Dict[str, Any] = {
        "count": len(rows),
        "answer_probability_mean": safe_mean(probabilities),
        "answer_probability_max": max(probabilities) if probabilities else float("nan"),
        "answer_probability_min": min(probabilities) if probabilities else float("nan"),
        "rougeL_recall_mean": (
            safe_mean(rouge_values) if not skip_generation else None
        ),
        "truth_ratio_raw_mean": safe_mean(truth_values) if truth_values else None,
    }
    return {"summary": summary, "records": details}


def evaluate_reference_probabilities(
    evaluator: Evaluator,
    groups: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, rows in groups.items():
        values = [
            evaluator.answer_prob(str(row["question"]), str(row["answer"]))
            for row in tqdm(rows, desc=f"reference[{name}]")
        ]
        result[name] = {
            "count": len(values),
            "answer_probability_mean": safe_mean(values),
            "answer_probability_max": max(values) if values else float("nan"),
        }
    return result


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir).resolve()
    output_path = Path(args.output).resolve()
    groups = {name: load_group(eval_dir, name) for name in GROUPS}

    if len(groups["forget_direct"]) != len(groups["forget_paraphrase"]):
        raise ValueError("direct and paraphrased selected forget views are misaligned")
    if len(groups["heldout_direct"]) != len(groups["heldout_paraphrase"]):
        raise ValueError("direct and paraphrased held-out forget views are misaligned")
    for direct_name, para_name in (
        ("forget_direct", "forget_paraphrase"),
        ("heldout_direct", "heldout_paraphrase"),
    ):
        direct_ids = [row.get("_source_index") for row in groups[direct_name]]
        para_ids = [row.get("_source_index") for row in groups[para_name]]
        if direct_ids != para_ids:
            raise ValueError(f"{direct_name}/{para_name} source indices differ")

    evaluator = Evaluator(
        args.model_dir,
        args.device,
        dtype_name(args.dtype),
        args.max_length,
    )
    results: Dict[str, Any] = {}
    for name, rows in groups.items():
        results[name] = evaluate_rows(
            evaluator,
            rows,
            name=name,
            max_new_tokens=args.max_new_tokens,
            skip_generation=args.skip_generation,
            include_truth_ratio=name.endswith("paraphrase"),
        )

    reference = None
    if args.reference_model_dir:
        reference_evaluator = Evaluator(
            args.reference_model_dir,
            args.device,
            dtype_name(args.dtype),
            args.max_length,
        )
        reference = evaluate_reference_probabilities(reference_evaluator, groups)

    summary = {
        "schema_version": 1,
        "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
        "seed": args.seed,
        "model_dir": str(Path(args.model_dir).resolve()),
        "reference_model_dir": (
            str(Path(args.reference_model_dir).resolve())
            if args.reference_model_dir
            else None
        ),
        "eval_dir": str(eval_dir),
        "generation_enabled": not args.skip_generation,
        "groups": {name: value["summary"] for name, value in results.items()},
        "primary_generalization": {
            "definition": "benchmark-provided paraphrased questions for the same 50 deletion QAs",
            "answer_probability_mean": results["forget_paraphrase"]["summary"]["answer_probability_mean"],
            "rougeL_recall_mean": results["forget_paraphrase"]["summary"]["rougeL_recall_mean"],
        },
        "secondary_generalization": {
            "definition": "remaining forget05 QA facts never exposed to Stage1/Stage2",
            "direct_answer_probability_mean": results["heldout_direct"]["summary"]["answer_probability_mean"],
            "paraphrase_answer_probability_mean": results["heldout_paraphrase"]["summary"]["answer_probability_mean"],
        },
        "reference_probabilities": reference,
    }
    if reference:
        retain_ref = reference["retain"]["answer_probability_mean"]
        retain_cur = results["retain"]["summary"]["answer_probability_mean"]
        summary["retain_answer_probability_ratio_to_reference"] = (
            retain_cur / retain_ref if retain_ref > 0 else None
        )

    payload = {"summary": summary, "details": results}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"locked TOFU evaluation: {output_path}")


if __name__ == "__main__":
    main()
