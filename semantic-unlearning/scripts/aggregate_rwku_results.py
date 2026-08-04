#!/usr/bin/env python3
"""Strictly aggregate the ten target-specific RWKU runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from rwku_data import (
    RWKU_CODE_REVISION,
    RWKU_DATASET_REVISION,
    TARGETS_BY_SEED,
)
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
    (
        "multiple_choice_single_order_recovery",
        "MC single-order recovery ↓",
        "pp",
    ),
    ("open_ended_recovery", "Open-ended recovery ↓", "pp"),
    (
        "frozen_base_head_probe_recovery",
        "Frozen-head probe recovery ↓",
        "pp",
    ),
    (
        "frozen_base_head_probe_chance_accuracy",
        "Frozen-head chance accuracy",
        "pp",
    ),
    (
        "frozen_base_head_probe_target_probability",
        "Frozen-head target probability ↓",
        "prob",
    ),
    (
        "frozen_base_head_probe_normalized_rank",
        "Frozen-head normalized rank ↑",
        "ratio",
    ),
)


def protocol_fingerprint(run: Mapping[str, Any]) -> str:
    """Fingerprint fields that must be identical across target runs."""

    mcf = run.get("mcf_retain_provenance", {})
    representation = run.get("representation")
    if isinstance(representation, Mapping):
        # The optimization seed is intentionally target/seed-specific.  It is
        # part of each run's provenance, but must not make otherwise identical
        # ten-target protocols look incomparable.
        representation = {
            key: value
            for key, value in representation.items()
            if key != "seed"
        }
    payload = {
        "rwku_code_revision": run.get("rwku_code_revision"),
        "rwku_dataset_revision": run.get("rwku_dataset_revision"),
        "methods": run.get("methods"),
        "method_order": run.get("method_order"),
        "model_path": run.get("model_path"),
        "model_identity": run.get("model_identity"),
        "dtype": run.get("dtype"),
        "implementation_file_sha256": run.get(
            "implementation_file_sha256"
        ),
        "zero_unlearn": run.get("zero_unlearn"),
        "wikidata_corpus": run.get("wikidata_corpus"),
        "calibration_fraction": run.get("calibration_fraction"),
        "setting5": run.get("setting5"),
        "repair": run.get("repair"),
        "representation": representation,
        "external_retain_partitions": run.get("external_retain_partitions"),
        "mcf_file_sha256": (
            mcf.get("file_sha256") if isinstance(mcf, Mapping) else None
        ),
        "evaluation_limits": run.get("evaluation_limits"),
        "skip_ppl": run.get("skip_ppl"),
        "eval_batch_size": run.get("eval_batch_size"),
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
    common_fingerprint: Optional[str] = None
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
        if run.get("rwku_code_revision") != RWKU_CODE_REVISION:
            raise ValueError(
                f"RWKU seed {seed} code revision does not match pinned revision"
            )
        if not allow_incomplete:
            required_protocol_fields = {
                "rwku_code_revision",
                "methods",
                "method_order",
                "methods_run",
                "model_path",
                "model_identity",
                "dtype",
                "implementation_file_sha256",
                "calibration_fraction",
                "setting5",
                "repair",
                "representation",
                "external_retain_partitions",
                "mcf_retain_provenance",
                "zero_unlearn",
                "wikidata_corpus",
                "evaluation_limits",
                "skip_ppl",
                "eval_batch_size",
            }
            missing_fields = sorted(required_protocol_fields - set(run))
            if missing_fields:
                raise ValueError(
                    f"RWKU seed {seed} lacks strict protocol fields: "
                    f"{missing_fields}"
                )
            if run["methods"] != list(METHOD_ORDER):
                raise ValueError(
                    f"RWKU seed {seed} methods are not the fixed final protocol"
                )
            if run["method_order"] != list(METHOD_ORDER):
                raise ValueError(
                    f"RWKU seed {seed} method_order is not canonical"
                )
            if run["methods_run"] != list(METHOD_ORDER):
                raise ValueError(
                    f"RWKU seed {seed} methods_run is incomplete or non-canonical"
                )
            if run["evaluation_limits"] != {} or run["skip_ppl"] is not False:
                raise ValueError(
                    f"RWKU seed {seed} is a bounded/skip-PPL run and cannot "
                    "enter strict final aggregation"
                )
            mcf_protocol = run["mcf_retain_provenance"]
            if (
                not isinstance(mcf_protocol, Mapping)
                or not mcf_protocol.get("file_sha256")
                or mcf_protocol.get("partitions_disjoint") is not True
                or mcf_protocol.get(
                    "example_content_partitions_disjoint"
                )
                is not True
            ):
                raise ValueError(
                    f"RWKU seed {seed} lacks valid disjoint MCF provenance"
                )
            for field in (
                "model_identity",
                "implementation_file_sha256",
                "setting5",
                "repair",
                "representation",
                "zero_unlearn",
                "wikidata_corpus",
            ):
                if not isinstance(run[field], Mapping) or not run[field]:
                    raise ValueError(
                        f"RWKU seed {seed} has empty strict protocol field "
                        f"{field}"
                    )
        expected = TARGETS_BY_SEED[seed]
        target = run["target"]
        if (
            target.get("seed") != seed
            or target.get("subject") != expected.subject
            or target.get("directory") != expected.directory
        ):
            raise ValueError(f"RWKU seed {seed} target mapping is incorrect")
        expected_methods = set(METHOD_ORDER)
        actual_methods = set(run["results"])
        if actual_methods != expected_methods:
            raise ValueError(
                f"RWKU seed {seed} must contain all {len(METHOD_ORDER)} "
                "configured methods; "
                f"missing={sorted(expected_methods - actual_methods)}, "
                f"extra={sorted(actual_methods - expected_methods)}"
            )
        if not allow_incomplete:
            for method in METHOD_ORDER:
                if method == METHOD_ORDER[0]:
                    continue
                contract = run["results"][method].get("success_contract")
                if not isinstance(contract, Mapping):
                    raise ValueError(
                        f"RWKU seed {seed} / {method} lacks a success contract"
                    )
                if contract.get("all_required_metrics_evaluated") is not True:
                    raise ValueError(
                        f"RWKU seed {seed} / {method} has an incomplete "
                        "success contract"
                    )
        fingerprint = protocol_fingerprint(run)
        if common_fingerprint is None:
            common_fingerprint = fingerprint
        elif fingerprint != common_fingerprint:
            raise ValueError(
                f"RWKU seed {seed} protocol/config fingerprint differs from "
                "the other runs"
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


def aggregate_success_contracts(
    runs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in METHOD_ORDER:
        if method == METHOD_ORDER[0]:
            continue
        contracts = [run["results"][method]["success_contract"] for run in runs]
        failures: Counter[str] = Counter()
        for contract in contracts:
            failures.update(str(value) for value in contract["failed_criteria"])
        passed = sum(bool(contract["passed"]) for contract in contracts)
        rows.append(
            {
                "method": method,
                "passed_targets": passed,
                "target_count": len(contracts),
                "all_targets_passed": passed == len(contracts),
                "failed_criterion_counts": dict(sorted(failures.items())),
            }
        )
    return rows


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
    success_contracts = aggregate_success_contracts(runs)
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
        "protocol_fingerprint": protocol_fingerprint(runs[0]),
        "std_definition": "population standard deviation (ddof=0)",
        "sections": rows_by_section,
        "success_contracts": success_contracts,
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
            f"{len(output_json['seeds'])} independent single-target runs, seeds "
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
        [
            "## Strict success contract",
            "",
            "| Method | Targets passing all criteria | All targets pass |",
            "| --- | ---: | --- |",
            *[
                "| "
                + str(row["method"])
                + " | "
                + f"{row['passed_targets']}/{row['target_count']}"
                + " | "
                + ("yes" if row["all_targets_passed"] else "no")
                + " |"
                for row in success_contracts
            ],
            "",
        ]
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
