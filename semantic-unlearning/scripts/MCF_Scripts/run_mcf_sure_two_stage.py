#!/usr/bin/env python3
"""Run the single canonical two-stage SURE experiment for MCF.

This entry point deliberately fixes the CounterFact answer roles without any
field swapping:

    target_true = sensitive/original answer to forget
    target_new  = non-sensitive CounterFact replacement to learn

The training objective is therefore bounded GA on ``target_true``, bounded GD
on ``target_new``, and an external-Wikipedia KL guard.  Stage 1 is a Rank-4
sparse LM-head edit.  Stage 2 is a conditional residual repair with the fixed
Rank 2 -> 4 -> 8 ladder; it runs only when the materialized Stage-1 checkpoint
does not meet every direct constraint.

Official MCF paraphrases, neighborhood prompts, retain records, and PPL text
are opened only after the checkpoint has been selected and saved.

Paper-facing metrics use one explicit, lower-is-better contract:

    Eff = 100 * E[ NLL(target_true) < NLL(target_new) ] on direct prompts
    Gen = 100 * E[ NLL(target_true) < NLL(target_new) ] on paraphrases
    Spe = 100 * E[ exp(-NLL(true)) - exp(-NLL(new)) ] on neighborhoods
    PPL = the fixed ZeroUnlearn-compatible Wikipedia perplexity probe

Eff/Gen measure residual preference for the sensitive answer.  Exact ties are
reported separately and count as neither residual preference nor forgetting
success.  FS/GFS and sensitive-answer token probabilities are also emitted as
audits, but are not substituted for Eff/Gen.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

METHOD = "SURE-MCF-2Stage-target_true-sensitive"
PROTOCOL = "sure_mcf_two_stage_target_true_sensitive_v1"
METRIC_SCHEMA = "mcf_target_true_sensitive_eff_gen_lower_v1"

TARGET_CONTRACT: Dict[str, Any] = {
    "sensitive_answer": "requested_rewrite.target_true",
    "non_sensitive_replacement": "requested_rewrite.target_new",
    "field_swapping": False,
    "stage1_target_true_operation": "bounded_gradient_ascent_increase_NLL",
    "stage1_target_new_operation": "gradient_descent_decrease_NLL",
    "stage2_operation": "residual_direct_pairwise_constraint_repair",
}

METRIC_CONTRACT: Dict[str, Any] = {
    "Eff": {
        "direction": "lower_is_better",
        "definition": (
            "100 * macro_mean_records(mean_direct_prompts("
            "I[NLL(target_true) < NLL(target_new)]))"
        ),
        "meaning": "residual direct preference for the sensitive target_true",
    },
    "Gen": {
        "direction": "lower_is_better",
        "definition": (
            "100 * macro_mean_records(mean_paraphrases("
            "I[NLL(target_true) < NLL(target_new)]))"
        ),
        "meaning": "residual paraphrase preference for the sensitive target_true",
    },
    "Spe": {
        "direction": "higher_is_better",
        "definition": (
            "100 * macro_mean_records(mean_neighborhoods("
            "exp(-NLL(target_true)) - exp(-NLL(target_new))))"
        ),
        "meaning": "preservation of the factual answer on locality prompts",
    },
    "PPL": {
        "direction": "lower_or_base_stable_is_better",
        "definition": "ZeroUnlearn-compatible perplexity on the fixed Wikipedia probe",
    },
    "ties": "exact NLL ties count as neither Eff/Gen nor FS/GFS success",
    "averaging": "prompt means within each record, then macro mean across records",
}

ARCHITECTURE: Dict[str, Any] = {
    "editable_parameters": "union target_true/target_new LM-head rows only",
    "official_probe_access_before_checkpoint": False,
    "benchmark_retain_examples_used_for_training": 0,
    "stage1": {
        "rank": 4,
        "steps": 600,
        "learning_rate": 0.005,
        "pairwise_target": 1.0,
        "target_true_nll_increase": 2.0,
        "target_new_nll_decrease": 1.0,
        "pairwise_weight": 100.0,
        "target_true_ga_weight": 10.0,
        "target_new_gd_weight": 10.0,
        "wikipedia_kl_weight": 1.0,
        "delta_l2_weight": 1e-4,
        "candidate_scales": (
            "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,"
            ".046875,.03125,.015625,.0078125,0"
        ),
    },
    "stage2": {
        "conditional_on_stage1_residual_failures": True,
        "rank_ladder": "2,4,8",
        "solver_margins": "0.5,1.0,2.0",
        "max_iterations": 500,
        "ftol": 1e-9,
        "constraint_tolerance": 1e-5,
        "residual_l2_weight": 1e-4,
        "constraint_context_weight": 0.05,
    },
    "acceptance": {
        "direct_pairwise_margin": 0.01,
        "materialized_direct_FS": 100.0,
        "utility_KL_mean_max": 0.01,
        "utility_KL_p95_max": 0.05,
        "utility_KL_max_max": 0.5,
        "total_delta_norm_max": 1.5,
        "GFS_used_for_checkpoint_selection": False,
    },
}


@dataclass(frozen=True)
class Step:
    label: str
    command: List[str]


@dataclass(frozen=True)
class SeedPaths:
    root: Path
    protocol_dir: Path
    training_visible: Path
    split_manifest: Path
    learner_dir: Path
    checkpoint: Path
    base_eval: Path
    final_eval: Path
    metrics: Path


def seed_paths(output_root: Path, seed: int) -> SeedPaths:
    root = output_root / f"seed{seed}"
    protocol_dir = root / "protocol"
    learner_dir = root / "sure_two_stage_learner"
    return SeedPaths(
        root=root,
        protocol_dir=protocol_dir,
        training_visible=(protocol_dir / "training_visible_target_aware_direct.json"),
        split_manifest=protocol_dir / "split_manifest.json",
        learner_dir=learner_dir,
        checkpoint=learner_dir / "checkpoint",
        base_eval=root / "base_official_eval.json",
        final_eval=root / "final_official_eval.json",
        metrics=root / "metrics_eff_gen_spe_ppl.json",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _pstdev(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _finite_nll(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite: {value!r}")
    return number


def pairwise_prompt_metrics(
    rows: Sequence[Mapping[str, Any]], prompt_key: str
) -> Dict[str, Any]:
    """Compute target_true-sensitive metrics with record-level macro averaging."""

    true_preference_by_record: List[float] = []
    new_preference_by_record: List[float] = []
    tie_by_record: List[float] = []
    true_probability_by_record: List[float] = []
    new_probability_by_record: List[float] = []
    separation_by_record: List[float] = []
    prompt_instances = 0
    true_wins = 0
    new_wins = 0
    ties = 0

    for row_index, row in enumerate(rows):
        prompts = row.get("post", {}).get(prompt_key, [])
        if not prompts:
            continue
        record_true_preference: List[float] = []
        record_new_preference: List[float] = []
        record_ties: List[float] = []
        record_true_probability: List[float] = []
        record_new_probability: List[float] = []
        record_separation: List[float] = []
        for prompt_index, values in enumerate(prompts):
            true_nll = _finite_nll(
                values["target_true"],
                f"row {row_index} prompt {prompt_index} target_true NLL",
            )
            new_nll = _finite_nll(
                values["target_new"],
                f"row {row_index} prompt {prompt_index} target_new NLL",
            )
            prompt_instances += 1
            true_preferred = true_nll < new_nll
            new_preferred = true_nll > new_nll
            tied = true_nll == new_nll
            true_wins += int(true_preferred)
            new_wins += int(new_preferred)
            ties += int(tied)
            record_true_preference.append(float(true_preferred))
            record_new_preference.append(float(new_preferred))
            record_ties.append(float(tied))
            record_true_probability.append(math.exp(-true_nll))
            record_new_probability.append(math.exp(-new_nll))
            record_separation.append(true_nll - new_nll)
        true_preference_by_record.append(statistics.mean(record_true_preference))
        new_preference_by_record.append(statistics.mean(record_new_preference))
        tie_by_record.append(statistics.mean(record_ties))
        true_probability_by_record.append(statistics.mean(record_true_probability))
        new_probability_by_record.append(statistics.mean(record_new_probability))
        separation_by_record.append(statistics.mean(record_separation))

    if not true_preference_by_record:
        raise RuntimeError(f"No prompts were available for {prompt_key}")

    return {
        "record_count": len(true_preference_by_record),
        "prompt_instance_count": prompt_instances,
        "target_true_preferred_prompt_instances": true_wins,
        "target_new_preferred_prompt_instances": new_wins,
        "exact_tie_prompt_instances": ties,
        "target_true_preference_percent": 100.0
        * statistics.mean(true_preference_by_record),
        "target_new_preference_percent": 100.0
        * statistics.mean(new_preference_by_record),
        "exact_tie_percent": 100.0 * statistics.mean(tie_by_record),
        "target_true_geometric_token_probability_percent": (
            100.0 * statistics.mean(true_probability_by_record)
        ),
        "target_new_geometric_token_probability_percent": (
            100.0 * statistics.mean(new_probability_by_record)
        ),
        "nll_separation_true_minus_new": statistics.mean(separation_by_record),
    }


def specificity_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    pair = pairwise_prompt_metrics(rows, "neighborhood_prompts_probs")
    probability_differences_by_record: List[float] = []
    for row_index, row in enumerate(rows):
        prompts = row.get("post", {}).get("neighborhood_prompts_probs", [])
        if not prompts:
            continue
        values: List[float] = []
        for prompt_index, prompt in enumerate(prompts):
            true_nll = _finite_nll(
                prompt["target_true"],
                f"row {row_index} neighborhood {prompt_index} target_true NLL",
            )
            new_nll = _finite_nll(
                prompt["target_new"],
                f"row {row_index} neighborhood {prompt_index} target_new NLL",
            )
            values.append(math.exp(-true_nll) - math.exp(-new_nll))
        probability_differences_by_record.append(statistics.mean(values))
    pair["Spe"] = 100.0 * statistics.mean(probability_differences_by_record)
    pair["Spe_success"] = pair["target_true_preference_percent"]
    return pair


def _assert_matching_eval_records(
    base_rows: Sequence[Mapping[str, Any]],
    post_rows: Sequence[Mapping[str, Any]],
    split_name: str,
) -> None:
    if len(base_rows) != len(post_rows):
        raise RuntimeError(f"{split_name}: base/post record counts differ")
    for index, (base, post) in enumerate(zip(base_rows, post_rows)):
        if base.get("requested_rewrite") != post.get("requested_rewrite"):
            raise RuntimeError(
                f"{split_name}: base/post requested_rewrite mismatch at {index}"
            )


def _target_text(rewrite: Mapping[str, Any], field: str) -> str:
    value = rewrite.get(field)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"requested_rewrite.{field} is missing")
    text = str(value.get("str", "")).strip()
    if not text:
        raise RuntimeError(f"requested_rewrite.{field}.str is empty")
    return text


def validate_data_boundary(
    manifest: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    evaluated_forget_rows: Sequence[Mapping[str, Any]],
    *,
    expected_seed: int,
    expected_forget_num: int,
) -> None:
    if manifest.get("dataset") != "mcf":
        raise RuntimeError("split manifest must declare dataset=mcf")
    if int(manifest.get("seed", -1)) != expected_seed:
        raise RuntimeError("split manifest seed does not match the run")
    adapter = manifest.get("learner_adapter_contract", {})
    required_adapter = {
        "sensitive_answer_field": "target_true",
        "reference_answer_field": "target_new",
        "direct_only": True,
        "official_paraphrases_visible_to_learner": False,
    }
    for key, expected in required_adapter.items():
        if adapter.get(key) != expected:
            raise RuntimeError(f"invalid learner role contract for {key}")
    roles = manifest.get("data_roles", {})
    if roles.get("GFS_checkpoint_selection") is not False:
        raise RuntimeError("GFS must be evaluation-only")
    if roles.get("heldout_probes_visible_during_training") is not False:
        raise RuntimeError("held-out MCF probes leaked into training")
    if int(roles.get("benchmark_retain_examples_visible_to_training", -1)) != 0:
        raise RuntimeError("benchmark retain examples leaked into training")
    if len(training_rows) != expected_forget_num:
        raise RuntimeError("training-visible forget count is incorrect")
    if len(evaluated_forget_rows) != expected_forget_num:
        raise RuntimeError("evaluated forget count is incorrect")

    forbidden = {
        "paraphrase_prompts",
        "neighborhood_prompts",
        "attribute_prompts",
        "generation_prompts",
    }
    for index, (training, evaluated) in enumerate(
        zip(training_rows, evaluated_forget_rows)
    ):
        leaked = sorted(forbidden.intersection(training))
        if leaked:
            raise RuntimeError(
                f"training-visible row {index} contains evaluation probes: {leaked}"
            )
        training_rewrite = training.get("requested_rewrite", {})
        evaluated_rewrite = evaluated.get("requested_rewrite", {})
        for field in ("prompt", "subject"):
            if str(training_rewrite.get(field, "")) != str(
                evaluated_rewrite.get(field, "")
            ):
                raise RuntimeError(
                    f"training/evaluation {field} mismatch at forget row {index}"
                )
        for field in ("target_true", "target_new"):
            if _target_text(training_rewrite, field) != _target_text(
                evaluated_rewrite, field
            ):
                raise RuntimeError(
                    f"training/evaluation {field} mismatch at forget row {index}"
                )


def _primary_metrics(eval_payload: Mapping[str, Any]) -> Dict[str, float]:
    forget_rows = eval_payload.get("forget_raw", [])
    direct = pairwise_prompt_metrics(forget_rows, "rewrite_prompts_probs")
    paraphrase = pairwise_prompt_metrics(forget_rows, "paraphrase_prompts_probs")
    specificity = specificity_metrics(forget_rows)
    ppl = eval_payload.get("forget_PPL")
    if ppl is None or not math.isfinite(float(ppl)):
        raise RuntimeError("official PPL was not computed or is non-finite")
    return {
        "Eff": float(direct["target_true_preference_percent"]),
        "Gen": float(paraphrase["target_true_preference_percent"]),
        "Spe": float(specificity["Spe"]),
        "PPL": float(ppl),
    }


def build_seed_report(
    *,
    base: Mapping[str, Any],
    post: Mapping[str, Any],
    manifest: Mapping[str, Any],
    training_rows: Sequence[Mapping[str, Any]],
    seed: int,
    forget_num: int,
    retain_num: int,
    paths: SeedPaths | None = None,
) -> Dict[str, Any]:
    for key in ("seed", "unlearn_num", "retain_num", "sample_mode"):
        if base.get(key) != post.get(key):
            raise RuntimeError(f"base/post evaluation mismatch for {key}")
    if int(post.get("seed", -1)) != seed:
        raise RuntimeError("evaluation seed does not match the run")
    if int(post.get("unlearn_num", -1)) != forget_num:
        raise RuntimeError("evaluation forget count does not match the run")
    if int(post.get("retain_num", -1)) != retain_num:
        raise RuntimeError("evaluation retain count does not match the run")
    if post.get("sample_mode") != "official":
        raise RuntimeError("MCF comparison requires sample_mode=official")

    base_forget = base.get("forget_raw", [])
    post_forget = post.get("forget_raw", [])
    base_retain = base.get("retain_raw", [])
    post_retain = post.get("retain_raw", [])
    _assert_matching_eval_records(base_forget, post_forget, "forget")
    _assert_matching_eval_records(base_retain, post_retain, "retain")
    if len(post_retain) != retain_num:
        raise RuntimeError("evaluated retain count is incorrect")
    validate_data_boundary(
        manifest,
        training_rows,
        post_forget,
        expected_seed=seed,
        expected_forget_num=forget_num,
    )

    base_direct = pairwise_prompt_metrics(base_forget, "rewrite_prompts_probs")
    post_direct = pairwise_prompt_metrics(post_forget, "rewrite_prompts_probs")
    base_paraphrase = pairwise_prompt_metrics(base_forget, "paraphrase_prompts_probs")
    post_paraphrase = pairwise_prompt_metrics(post_forget, "paraphrase_prompts_probs")
    base_specificity = specificity_metrics(base_forget)
    post_specificity = specificity_metrics(post_forget)
    base_primary = _primary_metrics(base)
    post_primary = _primary_metrics(post)

    retain_direct = pairwise_prompt_metrics(post_retain, "rewrite_prompts_probs")
    retain_paraphrase = pairwise_prompt_metrics(post_retain, "paraphrase_prompts_probs")

    provenance: Dict[str, Any] = {}
    if paths is not None:
        provenance = {
            "base_eval_json": str(paths.base_eval),
            "final_eval_json": str(paths.final_eval),
            "split_manifest": str(paths.split_manifest),
            "training_visible": str(paths.training_visible),
            "checkpoint": str(paths.checkpoint),
        }

    return {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "metric_schema": METRIC_SCHEMA,
        "method": METHOD,
        "dataset": "MCF",
        "seed": seed,
        "target_contract": TARGET_CONTRACT,
        "metric_contract": METRIC_CONTRACT,
        "primary_metrics": {
            name: {
                "value": value,
                "direction": METRIC_CONTRACT[name]["direction"],
            }
            for name, value in post_primary.items()
        },
        "base_primary_metrics": {
            name: {
                "value": value,
                "direction": METRIC_CONTRACT[name]["direction"],
            }
            for name, value in base_primary.items()
        },
        "delta_post_minus_base": {
            name: post_primary[name] - base_primary[name] for name in post_primary
        },
        "forget_audits": {
            "FS_direct_higher_is_better": post_direct["target_new_preference_percent"],
            "GFS_paraphrase_higher_is_better": post_paraphrase[
                "target_new_preference_percent"
            ],
            "Eff_exact_ties_percent": post_direct["exact_tie_percent"],
            "Gen_exact_ties_percent": post_paraphrase["exact_tie_percent"],
            "SensitiveProbability_direct_percent": post_direct[
                "target_true_geometric_token_probability_percent"
            ],
            "SensitiveProbability_paraphrase_percent": post_paraphrase[
                "target_true_geometric_token_probability_percent"
            ],
            "NLL_separation_true_minus_new_direct": post_direct[
                "nll_separation_true_minus_new"
            ],
            "NLL_separation_true_minus_new_paraphrase": post_paraphrase[
                "nll_separation_true_minus_new"
            ],
            "Spe_success_higher_is_better": post_specificity["Spe_success"],
        },
        "base_forget_audits": {
            "FS_direct_higher_is_better": base_direct["target_new_preference_percent"],
            "GFS_paraphrase_higher_is_better": base_paraphrase[
                "target_new_preference_percent"
            ],
            "SensitiveProbability_direct_percent": base_direct[
                "target_true_geometric_token_probability_percent"
            ],
            "SensitiveProbability_paraphrase_percent": base_paraphrase[
                "target_true_geometric_token_probability_percent"
            ],
            "Spe_success_higher_is_better": base_specificity["Spe_success"],
        },
        "retain_utility_audits": {
            "factual_target_true_preference_direct_percent": retain_direct[
                "target_true_preference_percent"
            ],
            "factual_target_true_preference_paraphrase_percent": retain_paraphrase[
                "target_true_preference_percent"
            ],
            "record_count": len(post_retain),
        },
        "counts": {
            "forget_records": len(post_forget),
            "retain_records": len(post_retain),
            "direct_prompt_instances": post_direct["prompt_instance_count"],
            "paraphrase_prompt_instances": post_paraphrase["prompt_instance_count"],
            "neighborhood_prompt_instances": post_specificity["prompt_instance_count"],
        },
        "comparison_warning": (
            "Only compare these Eff/Gen values with baseline raw predictions "
            "recomputed under the same target_true-sensitive metric schema. "
            "Do not mix them with a target_new-sensitive interpretation or an "
            "inverted rewrite-success alias."
        ),
        "provenance": provenance,
    }


def _value(report: Mapping[str, Any], section: str, metric: str) -> float:
    return float(report[section][metric]["value"])


def _format_mean_sd(mean: float, sd: float, metric: str) -> str:
    decimals = 4 if metric == "PPL" else 2
    return f"{mean:.{decimals}f} +/- {sd:.{decimals}f}"


def aggregate_reports(
    reports: Sequence[Mapping[str, Any]], output_root: Path
) -> Dict[str, Any]:
    if not reports:
        raise ValueError("cannot aggregate an empty report list")
    metrics = ("Eff", "Gen", "Spe", "PPL")
    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "metric_schema": METRIC_SCHEMA,
        "target_contract": TARGET_CONTRACT,
        "metric_contract": METRIC_CONTRACT,
        "seeds": [int(report["seed"]) for report in reports],
        "paper_ready_seed_count": len(reports) >= 10,
        "methods": {},
    }
    table_rows: List[Dict[str, str]] = []
    for label, section in (
        ("Base", "base_primary_metrics"),
        (METHOD, "primary_metrics"),
    ):
        summaries: Dict[str, Any] = {}
        row: Dict[str, str] = {"Method": label, "Seeds": str(len(reports))}
        for metric in metrics:
            values = [_value(report, section, metric) for report in reports]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            summaries[metric] = {
                "mean": float(mean),
                "sample_sd": float(sd),
                "values": [float(value) for value in values],
                "direction": METRIC_CONTRACT[metric]["direction"],
            }
            arrow = "uparrow" if metric == "Spe" else "downarrow"
            row[f"{metric}_{arrow}"] = _format_mean_sd(mean, sd, metric)
        aggregate["methods"][label] = summaries
        table_rows.append(row)

    write_json(output_root / "aggregate_metrics.json", aggregate)
    fieldnames = [
        "Method",
        "Seeds",
        "Eff_downarrow",
        "Gen_downarrow",
        "Spe_uparrow",
        "PPL_downarrow",
    ]
    csv_path = output_root / "table1_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)

    markdown = [
        "# MCF Table 1 rows",
        "",
        "`target_true` is sensitive; `target_new` is the non-sensitive replacement.",
        "Eff/Gen are residual-sensitive preference rates (lower is better).",
        "",
        "| Method | Seeds | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in table_rows:
        markdown.append(
            "| {Method} | {Seeds} | {Eff_downarrow} | {Gen_downarrow} | "
            "{Spe_uparrow} | {PPL_downarrow} |".format(**row)
        )
    markdown.extend(
        [
            "",
            "Do not paste a baseline into this table unless its raw predictions were ",
            "scored with the same target-role and metric contract.",
        ]
    )
    (output_root / "table1_rows.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    return aggregate


def _python(script_name: str, *arguments: Any) -> List[str]:
    return [
        sys.executable,
        str(SCRIPTS_DIR / script_name),
        *(str(argument) for argument in arguments),
    ]


def learner_command(
    args: argparse.Namespace,
    paths: SeedPaths,
    utility_cache: Path,
    utility_prompts: int,
    min_utility_documents: int,
    min_utility_prompts: int,
    seed: int,
) -> List[str]:
    stage1 = ARCHITECTURE["stage1"]
    stage2 = ARCHITECTURE["stage2"]
    acceptance = ARCHITECTURE["acceptance"]
    command = _python(
        "sure_mcf_target_aware_direct_only.py",
        "--model-path",
        args.model_path,
        "--training-visible-path",
        paths.training_visible,
        "--split-manifest",
        paths.split_manifest,
        "--utility-cache",
        utility_cache,
        "--output-dir",
        paths.learner_dir,
        "--seed",
        seed,
        "--forget-num",
        args.forget_num,
        "--utility-sample-size",
        args.utility_docs,
        "--utility-prompt-count",
        utility_prompts,
        "--require-min-utility-documents",
        min_utility_documents,
        "--require-min-utility-prompts",
        min_utility_prompts,
        "--utility-token-topk-per-row",
        128,
        "--utility-uniform-prompt-count",
        1024,
        "--utility-pool-seed",
        1,
        "--utility-train-batch-size",
        128,
        "--utility-eval-batch-size",
        512,
        "--cache-batch-size",
        8,
        "--stage1-rank",
        stage1["rank"],
        "--stage1-steps",
        stage1["steps"],
        "--stage1-lr",
        stage1["learning_rate"],
        "--stage1-pairwise-target",
        stage1["pairwise_target"],
        "--stage1-true-nll-increase",
        stage1["target_true_nll_increase"],
        "--stage1-new-nll-decrease",
        stage1["target_new_nll_decrease"],
        "--stage1-pairwise-weight",
        stage1["pairwise_weight"],
        "--stage1-true-ga-weight",
        stage1["target_true_ga_weight"],
        "--stage1-new-gd-weight",
        stage1["target_new_gd_weight"],
        "--stage1-utility-kl-weight",
        stage1["wikipedia_kl_weight"],
        "--stage1-l2-weight",
        stage1["delta_l2_weight"],
        "--stage1-candidate-scales",
        stage1["candidate_scales"],
        "--required-pairwise-margin",
        acceptance["direct_pairwise_margin"],
        "--stage2-solver-margins",
        stage2["solver_margins"],
        "--stage2-rank-ladder",
        stage2["rank_ladder"],
        "--stage2-maxiter",
        stage2["max_iterations"],
        "--stage2-ftol",
        stage2["ftol"],
        "--stage2-constraint-tolerance",
        stage2["constraint_tolerance"],
        "--stage2-residual-l2-weight",
        stage2["residual_l2_weight"],
        "--constraint-context-weight",
        stage2["constraint_context_weight"],
        "--contrastive-eps",
        0.001,
        "--utility-kl-mean-budget",
        acceptance["utility_KL_mean_max"],
        "--utility-kl-p95-budget",
        acceptance["utility_KL_p95_max"],
        "--utility-kl-max-budget",
        acceptance["utility_KL_max_max"],
        "--max-total-delta-norm",
        acceptance["total_delta_norm_max"],
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
    )
    if args.require_corpus_protocol:
        command.extend(
            ["--require-utility-corpus-protocol", args.require_corpus_protocol]
        )
    return command


def seed_command_plan(
    args: argparse.Namespace,
    paths: SeedPaths,
    utility_cache: Path,
    utility_prompts: int,
    min_utility_documents: int,
    min_utility_prompts: int,
    seed: int,
) -> List[Step]:
    common_eval = [
        "--mcf-path",
        args.mcf_path,
        "--wikidata-dir",
        args.wikipedia_dir,
        "--unlearn-num",
        args.forget_num,
        "--retain-num",
        args.retain_num,
        "--seed",
        seed,
        "--sample-mode",
        "official",
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
        "--quiet",
    ]
    return [
        Step(
            "LOCKED DIRECT-ONLY SPLIT",
            _python(
                "build_mcf_sure_target_aware_direct_split.py",
                "--mcf-path",
                args.mcf_path,
                "--output-dir",
                paths.protocol_dir,
                "--seed",
                seed,
                "--forget-num",
                args.forget_num,
                "--retain-eval-num",
                args.retain_num,
            ),
        ),
        Step(
            "STAGE 1 + CONDITIONAL STAGE 2",
            learner_command(
                args,
                paths,
                utility_cache,
                utility_prompts,
                min_utility_documents,
                min_utility_prompts,
                seed,
            ),
        ),
        Step(
            "BASE OFFICIAL EVALUATION",
            _python(
                "mcf_zero_unlearn_official_eval.py",
                "--model-dir",
                args.model_path,
                "--out",
                paths.base_eval,
                *common_eval,
            ),
        ),
        Step(
            "FROZEN SURE OFFICIAL EVALUATION",
            _python(
                "mcf_zero_unlearn_official_eval.py",
                "--model-dir",
                paths.checkpoint,
                "--out",
                paths.final_eval,
                *common_eval,
            ),
        ),
    ]


def utility_cache_command(
    args: argparse.Namespace,
    utility_cache: Path,
    utility_wikipedia_dir: Path,
    utility_prompts: int,
    min_utility_documents: int,
    min_utility_prompts: int,
) -> List[str]:
    command = _python(
        "build_sure_wikipedia_stats.py",
        "--model-path",
        args.model_path,
        "--wikidata-dir",
        utility_wikipedia_dir,
        "--output-path",
        utility_cache,
        "--sample-size",
        args.utility_docs,
        "--require-min-documents",
        min_utility_documents,
        "--require-min-prompts",
        min_utility_prompts,
        "--utility-seed",
        1,
        "--exclude-first",
        20,
        "--utility-max-length",
        4096,
        "--utility-batch-size",
        1,
        "--utility-prompt-count",
        utility_prompts,
        "--utility-logit-batch-size",
        64,
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
    )
    if args.require_corpus_protocol:
        command.extend(["--require-corpus-protocol", args.require_corpus_protocol])
    return command


def run_step(step: Step, *, dry_run: bool) -> None:
    print(f"\n===== {step.label} =====")
    print(shlex.join(step.command))
    if not dry_run:
        subprocess.run(step.command, cwd=PROJECT_ROOT, check=True)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", required=True, help="Local HF model directory")
    parser.add_argument(
        "--mcf-path",
        default=str(PROJECT_ROOT / "data" / "multi_counterfact.json"),
    )
    parser.add_argument(
        "--wikipedia-dir",
        default=str(PROJECT_ROOT / "data" / "wikidata"),
        help="DatasetDict used for the fixed PPL probe",
    )
    parser.add_argument(
        "--utility-wikipedia-dir",
        default=None,
        help="External Wikipedia DatasetDict; defaults to --wikipedia-dir",
    )
    parser.add_argument(
        "--utility-cache",
        default=None,
        help="Reuse/build this model-specific SURE utility cache",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "outputs" / "mcf_sure_two_stage"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument(
        "--utility-docs",
        type=int,
        default=100_000,
        help="Wikipedia document coverage (use 1000 for the W1K pilot)",
    )
    parser.add_argument(
        "--utility-prompts",
        type=int,
        default=100_000,
        help="Cached predictor-state candidates (independent of document count)",
    )
    parser.add_argument(
        "--allow-capped-utility-cache",
        action="store_true",
        help="Allow fewer documents/prompts than requested; marks the run as a pilot",
    )
    parser.add_argument(
        "--require-corpus-protocol",
        default="",
        help="Optional prepared-Wikipedia receipt protocol to require",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the locked contract and every command without writing or running",
    )
    args = parser.parse_args(argv)
    if args.forget_num <= 0 or args.retain_num <= 0:
        parser.error("--forget-num and --retain-num must be positive")
    if args.utility_docs <= 0 or args.utility_prompts <= 0:
        parser.error("--utility-docs and --utility-prompts must be positive")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds must contain one or more unique integers")
    return args


def _validate_paths(args: argparse.Namespace, utility_cache: Path) -> None:
    model_path = _path(args.model_path)
    mcf_path = _path(args.mcf_path)
    wikipedia_dir = _path(args.wikipedia_dir)
    utility_wikipedia_dir = _path(args.utility_wikipedia_dir or args.wikipedia_dir)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_path}")
    if not mcf_path.is_file():
        raise FileNotFoundError(f"MCF JSON not found: {mcf_path}")
    if not wikipedia_dir.is_dir():
        raise FileNotFoundError(f"PPL Wikipedia dataset not found: {wikipedia_dir}")
    if not utility_cache.is_file() and not utility_wikipedia_dir.is_dir():
        raise FileNotFoundError(
            f"utility Wikipedia dataset not found: {utility_wikipedia_dir}"
        )
    for helper in (
        "build_sure_wikipedia_stats.py",
        "build_mcf_sure_target_aware_direct_split.py",
        "sure_mcf_target_aware_direct_only.py",
        "mcf_zero_unlearn_official_eval.py",
    ):
        if not (SCRIPTS_DIR / helper).is_file():
            raise FileNotFoundError(f"required SURE component is missing: {helper}")


def print_contract(args: argparse.Namespace, utility_cache: Path) -> None:
    print("=" * 72)
    print("MCF SURE TWO-STAGE — LOCKED TARGET CONTRACT")
    print("target_true : SENSITIVE -> bounded GA -> increase NLL")
    print("target_new  : NON-SENSITIVE REPLACEMENT -> GD -> decrease NLL")
    print("Eff / Gen   : residual sensitive preference, LOWER is better")
    print("Spe         : neighborhood preservation, HIGHER is better")
    print("PPL         : fluency, LOWER/base-stable is better")
    print("Official paraphrases/neighborhoods/retain/PPL: evaluation-only")
    print(f"Wikipedia utility cache: {utility_cache}")
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.model_path = str(_path(args.model_path))
    args.mcf_path = str(_path(args.mcf_path))
    args.wikipedia_dir = str(_path(args.wikipedia_dir))
    if args.utility_wikipedia_dir is not None:
        args.utility_wikipedia_dir = str(_path(args.utility_wikipedia_dir))
    output_root = _path(args.output_root)
    utility_prompts = int(args.utility_prompts)
    if args.utility_cache:
        utility_cache = _path(args.utility_cache)
    else:
        model_tag = Path(args.model_path).name
        utility_cache = (
            PROJECT_ROOT
            / "outputs"
            / "sure_wikipedia_stats"
            / (
                f"{model_tag}_token_conditioned_docs{args.utility_docs}_"
                f"candidates{utility_prompts}_v3.pt"
            )
        )
    utility_wikipedia_dir = _path(args.utility_wikipedia_dir or args.wikipedia_dir)
    min_utility_documents = 0 if args.allow_capped_utility_cache else args.utility_docs
    min_utility_prompts = (
        0 if args.allow_capped_utility_cache else int(math.ceil(0.9 * utility_prompts))
    )

    print_contract(args, utility_cache)
    if not args.dry_run:
        _validate_paths(args, utility_cache)
        if output_root.exists():
            raise FileExistsError(
                f"refusing to overwrite output root: {output_root}. "
                "Choose a new --output-root."
            )

    if not utility_cache.is_file():
        run_step(
            Step(
                "BUILD EXTERNAL WIKIPEDIA UTILITY CACHE",
                utility_cache_command(
                    args,
                    utility_cache,
                    utility_wikipedia_dir,
                    utility_prompts,
                    min_utility_documents,
                    min_utility_prompts,
                ),
            ),
            dry_run=args.dry_run,
        )
    else:
        print(f"Reusing utility cache: {utility_cache}")

    reports: List[Dict[str, Any]] = []
    for seed in args.seeds:
        paths = seed_paths(output_root, seed)
        for step in seed_command_plan(
            args,
            paths,
            utility_cache,
            utility_prompts,
            min_utility_documents,
            min_utility_prompts,
            seed,
        ):
            run_step(step, dry_run=args.dry_run)
        if args.dry_run:
            continue

        base = json.loads(paths.base_eval.read_text(encoding="utf-8"))
        post = json.loads(paths.final_eval.read_text(encoding="utf-8"))
        manifest = json.loads(paths.split_manifest.read_text(encoding="utf-8"))
        training_rows = json.loads(paths.training_visible.read_text(encoding="utf-8"))
        report = build_seed_report(
            base=base,
            post=post,
            manifest=manifest,
            training_rows=training_rows,
            seed=seed,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            paths=paths,
        )
        write_json(paths.metrics, report)
        reports.append(report)
        primary = report["primary_metrics"]
        print("\n===== PAPER METRICS (TARGET_TRUE SENSITIVE) =====")
        for metric in ("Eff", "Gen", "Spe", "PPL"):
            print(
                f"{metric:<4} {primary[metric]['value']:>10.4f}  "
                f"{primary[metric]['direction']}"
            )
        print(f"Wrote: {paths.metrics}")

    if args.dry_run:
        print("\nDry run complete; no files were written and no models were loaded.")
        return

    run_config = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "method": METHOD,
        "target_contract": TARGET_CONTRACT,
        "metric_contract": METRIC_CONTRACT,
        "architecture": ARCHITECTURE,
        "inputs": {
            "model_path": args.model_path,
            "mcf_path": args.mcf_path,
            "mcf_sha256": sha256_file(Path(args.mcf_path)),
            "wikipedia_dir": args.wikipedia_dir,
            "utility_wikipedia_dir": str(utility_wikipedia_dir),
            "utility_cache": str(utility_cache),
            "utility_docs_requested": args.utility_docs,
            "utility_prompts_requested": utility_prompts,
            "minimum_utility_documents": min_utility_documents,
            "minimum_utility_prompts": min_utility_prompts,
            "capped_utility_cache_allowed": args.allow_capped_utility_cache,
            "required_corpus_protocol": args.require_corpus_protocol or None,
        },
        "seeds": list(args.seeds),
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
        "dtype": args.dtype,
        "device_map": args.device_map,
    }
    write_json(output_root / "run_config.json", run_config)
    aggregate = aggregate_reports(reports, output_root)
    print(f"\nComplete: {output_root}")
    print(f"Table rows: {output_root / 'table1_rows.md'}")
    if not aggregate["paper_ready_seed_count"]:
        print(
            "NOTE: fewer than 10 seeds; treat this as a pilot, not a paper aggregate."
        )


if __name__ == "__main__":
    main()
