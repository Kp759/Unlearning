#!/usr/bin/env python3
"""Strictly aggregate the ten target-specific RWKU runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from rwku_data import RWKU_DATASET_REVISION, TARGETS_BY_SEED
from rwku_experiment import METHOD_ORDER


SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_INPUT_ROOT = SEMANTIC_ROOT / "outputs" / "rwku"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "aggregate"

FORGET_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("cloze_target_recovery", "Cloze recovery ↓", "pp"),
    ("cloze_target_probability", "Cloze probability ↓", "prob"),
    ("direct_target_qa_recovery", "Direct QA recovery ↓", "pp"),
    ("direct_target_qa_probability", "Direct QA probability ↓", "prob"),
    ("paraphrased_target_recovery", "Paraphrase recovery ↓", "pp"),
    ("alias_question_recovery", "Alias-question recovery ↓", "pp"),
    ("alias_question_coverage", "Alias-question coverage", "ratio"),
    ("adversarial_recovery_success", "Adversarial recovery ↓", "pp"),
    (
        "membership_inference_attack_advantage",
        "MIA advantage ↓",
        "prob",
    ),
    (
        "target_answer_token_probability",
        "Target-token probability ↓",
        "prob",
    ),
)

RETAIN_METRICS: Tuple[Tuple[str, str, str], ...] = (
    ("neighboring_entity_accuracy", "Neighbor accuracy ↑", "pp"),
    ("general_utility", "General utility ↑", "pp"),
    ("mmlu_accuracy", "MMLU accuracy ↑", "pp"),
    ("reasoning_accuracy", "Reasoning accuracy ↑", "pp"),
    ("truthfulness_factuality", "Truth/factuality macro ↑", "pp"),
    ("truthfulness_accuracy", "TruthfulQA MC1 ↑", "pp"),
    ("factuality_f1", "TriviaQA factuality F1 ↑", "pp"),
    ("perplexity", "PPL ↓", "ppl"),
    ("full_retain_probability_ratio", "Full-retain ratio ↑", "ratio"),
)

CONTROL_METRICS: Tuple[Tuple[str, str, str], ...] = (
    (
        "full_answer_mean_log_likelihood",
        "Full-answer log likelihood ↓",
        "logp",
    ),
    (
        "full_answer_geometric_probability",
        "Full-answer probability ↓",
        "prob",
    ),
    (
        "forced_answer_prefix_recovery",
        "Forced-prefix recovery ↓",
        "pp",
    ),
    (
        "forced_answer_prefix_probability",
        "Forced-prefix probability ↓",
        "prob",
    ),
    (
        "forced_answer_prefix_coverage",
        "Forced-prefix coverage",
        "ratio",
    ),
    ("answer_alias_recovery", "Answer-alias recovery ↓", "pp"),
    ("answer_alias_probability", "Answer-alias probability ↓", "prob"),
    ("answer_alias_coverage", "Answer-alias coverage", "ratio"),
    ("multiple_choice_recovery", "Multiple-choice recovery ↓", "pp"),
    ("open_ended_recovery", "Open-ended recovery ↓", "pp"),
    (
        "frozen_base_head_probe_recovery",
        "Frozen-head probe recovery ↓",
        "pp",
    ),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a subset of seeds for smoke/debug aggregation.",
    )
    return parser


def load_runs(
    input_root: Path,
    *,
    allow_incomplete: bool,
) -> List[Dict[str, Any]]:
    paths = sorted(Path(input_root).glob("seed*/results.json"))
    if not paths:
        raise FileNotFoundError(f"No RWKU results.json files under {input_root}")
    by_seed: Dict[int, Dict[str, Any]] = {}
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            run = json.load(handle)
        seed = int(run["seed"])
        if seed in by_seed:
            raise ValueError(f"Duplicate RWKU seed {seed}: {path}")
        if run.get("status") != "complete":
            raise ValueError(f"RWKU seed {seed} is not marked complete")
        if run.get("rwku_dataset_revision") != RWKU_DATASET_REVISION:
            raise ValueError(
                f"RWKU seed {seed} dataset revision does not match pinned revision"
            )
        expected = TARGETS_BY_SEED[seed]
        target = run["target"]
        if (
            target.get("seed") != seed
            or target.get("subject") != expected.subject
            or target.get("directory") != expected.directory
        ):
            raise ValueError(f"RWKU seed {seed} target mapping is incorrect")
        if set(run["results"]) != set(METHOD_ORDER):
            raise ValueError(
                f"RWKU seed {seed} must contain all five methods; found "
                f"{sorted(run['results'])}"
            )
        by_seed[seed] = run
    expected_seeds = set(range(10))
    actual_seeds = set(by_seed)
    if not allow_incomplete and actual_seeds != expected_seeds:
        raise ValueError(
            "Strict RWKU aggregation requires seeds 0-9; "
            f"missing={sorted(expected_seeds - actual_seeds)}, "
            f"extra={sorted(actual_seeds - expected_seeds)}"
        )
    if not actual_seeds <= expected_seeds:
        raise ValueError(f"Unexpected RWKU seeds: {sorted(actual_seeds - expected_seeds)}")
    return [by_seed[seed] for seed in sorted(by_seed)]


def metric_values(
    runs: Sequence[Mapping[str, Any]],
    *,
    method: str,
    section: str,
    metric: str,
) -> List[float]:
    values: List[float] = []
    for run in runs:
        value = run["results"][method]["summary"][section].get(metric)
        if value is None:
            if metric in {
                "alias_question_recovery",
                "forced_answer_prefix_recovery",
                "forced_answer_prefix_probability",
                "answer_alias_recovery",
                "answer_alias_probability",
            }:
                continue
            raise ValueError(
                f"Missing {section}.{metric} for seed {run['seed']} / {method}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(
                f"Non-finite {section}.{metric} for seed {run['seed']} / {method}"
            )
        values.append(number)
    if not values:
        raise ValueError(f"No eligible observations for {section}.{metric} / {method}")
    return values


def aggregate_section(
    runs: Sequence[Mapping[str, Any]],
    *,
    section: str,
    metrics: Sequence[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        row: Dict[str, Any] = {
            "method": method,
            "n_targets": len(runs),
        }
        for key, _, _ in metrics:
            values = metric_values(
                runs,
                method=method,
                section=section,
                metric=key,
            )
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_std"] = float(np.std(values, ddof=0))
            row[f"{key}_n"] = len(values)
            row[f"{key}_values"] = values
        rows.append(row)
    return rows


def format_value(mean_value: float, std_value: float, kind: str) -> str:
    if kind == "prob":
        return f"{mean_value:.6f} ± {std_value:.6f}"
    if kind == "ratio":
        return f"{mean_value:.4f} ± {std_value:.4f}"
    if kind == "logp":
        return f"{mean_value:.4f} ± {std_value:.4f}"
    if kind == "ppl":
        return f"{mean_value:.4f} ± {std_value:.4f}"
    return f"{mean_value:.3f} ± {std_value:.3f}"


def markdown_table(
    title: str,
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[Tuple[str, str, str]],
) -> List[str]:
    headers = ["Method", *[label for _, label, _ in metrics]]
    lines = [
        f"## {title}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [str(row["method"])]
        for key, _, kind in metrics:
            values.append(
                format_value(
                    float(row[f"{key}_mean"]),
                    float(row[f"{key}_std"]),
                    kind,
                )
                + (
                    f" (n={int(row[f'{key}_n'])})"
                    if int(row[f"{key}_n"]) != int(row["n_targets"])
                    else ""
                )
            )
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return lines


def write_csv(
    path: Path,
    rows_by_section: Mapping[str, Sequence[Mapping[str, Any]]],
    metrics_by_section: Mapping[str, Sequence[Tuple[str, str, str]]],
) -> None:
    output: List[Dict[str, Any]] = []
    for section, rows in rows_by_section.items():
        metrics = metrics_by_section[section]
        for row in rows:
            for key, label, _ in metrics:
                output.append(
                    {
                        "section": section,
                        "method": row["method"],
                        "metric": key,
                        "label": label,
                        "n_targets": row["n_targets"],
                        "n_metric": row[f"{key}_n"],
                        "mean": row[f"{key}_mean"],
                        "std_population": row[f"{key}_std"],
                        "seed_values": json.dumps(row[f"{key}_values"]),
                    }
                )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def main() -> None:
    args = build_parser().parse_args()
    runs = load_runs(args.input_root, allow_incomplete=args.allow_incomplete)
    metrics_by_section = {
        "forget": FORGET_METRICS,
        "retain": RETAIN_METRICS,
        "controls": CONTROL_METRICS,
    }
    rows_by_section = {
        section: aggregate_section(
            runs,
            section=section,
            metrics=metrics,
        )
        for section, metrics in metrics_by_section.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / "rwku_aggregate.csv",
        rows_by_section,
        metrics_by_section,
    )
    output_json = {
        "dataset": "RWKU",
        "dataset_revision": RWKU_DATASET_REVISION,
        "seeds": [int(run["seed"]) for run in runs],
        "targets": [run["target"]["subject"] for run in runs],
        "std_definition": "population standard deviation (ddof=0)",
        "sections": rows_by_section,
    }
    with (args.output_dir / "rwku_aggregate.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(output_json, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    lines = [
        "# RWKU results",
        "",
        (
            f"Ten independent single-target runs, seeds "
            f"{min(output_json['seeds'])}–{max(output_json['seeds'])}; "
            "mean ± population standard deviation."
        ),
        "",
    ]
    lines.extend(
        markdown_table(
            "Forget metrics",
            rows_by_section["forget"],
            FORGET_METRICS,
        )
    )
    lines.extend(
        markdown_table(
            "Retain metrics",
            rows_by_section["retain"],
            RETAIN_METRICS,
        )
    )
    lines.extend(
        markdown_table(
            "Alternative-output and repair controls",
            rows_by_section["controls"],
            CONTROL_METRICS,
        )
    )
    lines.extend(
        [
            "Probabilities and MIA advantage are in [0,1]; accuracies are "
            "percentage points. The frozen-head probe applies the untouched "
            "base LM-head answer rows to each method's live hidden states.",
            "",
        ]
    )
    (args.output_dir / "rwku_aggregate.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        f"Aggregated {len(runs)} RWKU targets to "
        f"{args.output_dir / 'rwku_aggregate.md'}"
    )


if __name__ == "__main__":
    main()
