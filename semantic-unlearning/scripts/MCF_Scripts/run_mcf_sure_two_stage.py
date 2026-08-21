#!/usr/bin/env python3
"""Run a locked canonical two-stage SURE experiment for MCF.

This entry point deliberately fixes the CounterFact answer roles without any
field swapping:

    target_true = sensitive/original answer to forget
    target_new  = non-sensitive CounterFact replacement to learn

The direct-only treatment uses bounded GA on ``target_true``, bounded GD on
``target_new``, and an external-Wikipedia KL guard. The paired-context recovery
treatment adds four locked, answer-cued same-subject views plus syntactically
matched external-Wikipedia locality views; it never reads an official MCF
probe before checkpoint freeze. Stage 1 is a Rank-4 sparse LM-head edit. Stage
2 is a conditional residual repair with the fixed Rank 2 -> 4 -> 8 ladder.

Official MCF paraphrases, neighborhood prompts, retain records, and PPL text
are opened only after the checkpoint has been selected and saved. A separate
prompt-only 1,000-record retain file is then used for exact sparse-row KL.

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
import copy
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
DIRECT_TREATMENT = "direct_only"
PAIRED_RECOVERY_TREATMENT = "paired_context_recovery"
PAIRED_RECOVERY_METHOD = "SURE-MCF-2Stage-paired-answer-cue-recovery"
PAIRED_RECOVERY_PROTOCOL = "sure_mcf_two_stage_paired_answer_cue_recovery_v1"
PAIRED_CONTEXT_PROFILE = "paired_answer_cue_v1"
PAPER_CONFIRMATORY_SEED_COUNT = 10

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

PAIRED_RECOVERY: Dict[str, Any] = {
    "utility_documents": 10_000,
    "context_profile": PAIRED_CONTEXT_PROFILE,
    "generated_subject_contexts_per_record": 4,
    "external_locality_contexts_per_record": 128,
    "external_context_lead_characters": 256,
    "locality_token_topk_per_row": 64,
    "locality_uniform_prompt_count": 512,
    "locality_pool_seed": 1,
    "stage1_locality_kl_weight": 10.0,
    "stage2_locality_kl_weight": 1.0,
    "locality_kl_mean_budget": 0.01,
    "locality_kl_p95_budget": 0.05,
    "locality_kl_max_budget": 0.5,
    "acceptance": {
        "FS_direct_min": 100.0,
        "GFS_paraphrase_min": 79.0,
        "Spe_success_min": 61.8,
        "Spe_margin_min": 3.69,
        "PPL_max": 11.2,
        "partial_GFS_strictly_above": 48.0,
    },
}

RUNTIME_SOURCE_FILES = (
    "scripts/MCF_Scripts/run_mcf_sure_two_stage.py",
    "scripts/run_mcf_sure_v9_gfs_recovery.sh",
    "scripts/audit_sure_exact_retain_kl.py",
    "scripts/build_mcf_sure_target_aware_direct_split.py",
    "scripts/build_sure_mcf_external_contexts.py",
    "scripts/build_sure_wikipedia_stats.py",
    "scripts/gagd_compare.py",
    "scripts/mcf_zero_unlearn_official_eval.py",
    "scripts/sure_canonical_core.py",
    "scripts/sure_context_projection.py",
    "scripts/sure_contrastive_two_stage_v2.py",
    "scripts/sure_mcf_direct_fs_repair.py",
    "scripts/sure_mcf_target_aware_direct_only.py",
    "scripts/sure_mcf_target_aware_two_stage.py",
    "scripts/sure_minimal_two_stage.py",
    "scripts/sure_retain_kl.py",
    "scripts/sure_shared_suppression.py",
)


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
    retain_audit: Path
    external_contexts: Path
    learner_dir: Path
    checkpoint: Path
    final_delta: Path
    learner_config: Path
    base_eval: Path
    final_eval: Path
    exact_retain_kl: Path
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
        retain_audit=(protocol_dir / "evaluation_only_retain_prompts.json"),
        external_contexts=(
            protocol_dir / "external_subject_locality_contexts.json"
        ),
        learner_dir=learner_dir,
        checkpoint=learner_dir / "checkpoint",
        final_delta=learner_dir / "final_total_delta.pt",
        learner_config=learner_dir / "config_used.json",
        base_eval=root / "base_official_eval.json",
        final_eval=root / "final_official_eval.json",
        exact_retain_kl=root / "posthoc_exact_retain_kl.json",
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


def treatment_method_protocol(treatment: str) -> tuple[str, str]:
    if treatment == DIRECT_TREATMENT:
        return METHOD, PROTOCOL
    if treatment == PAIRED_RECOVERY_TREATMENT:
        return PAIRED_RECOVERY_METHOD, PAIRED_RECOVERY_PROTOCOL
    raise ValueError(f"unsupported treatment: {treatment}")


def target_contract_for_treatment(treatment: str) -> Dict[str, Any]:
    contract = copy.deepcopy(TARGET_CONTRACT)
    if treatment == PAIRED_RECOVERY_TREATMENT:
        contract["stage2_operation"] = (
            "residual_direct_and_generated_pairwise_constraint_repair"
        )
    elif treatment != DIRECT_TREATMENT:
        raise ValueError(f"unsupported treatment: {treatment}")
    return contract


def architecture_for_treatment(treatment: str) -> Dict[str, Any]:
    architecture = copy.deepcopy(ARCHITECTURE)
    architecture["treatment"] = treatment
    if treatment == PAIRED_RECOVERY_TREATMENT:
        architecture.update(
            {
                "training_prompt_scope": "direct_plus_generated_subject",
                "external_context_treatment": copy.deepcopy(PAIRED_RECOVERY),
                "official_probe_access_before_checkpoint": False,
            }
        )
    elif treatment == DIRECT_TREATMENT:
        architecture["training_prompt_scope"] = "direct_only"
    else:
        raise ValueError(f"unsupported treatment: {treatment}")
    return architecture


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.rstrip()


def source_provenance(*, require_clean: bool) -> Dict[str, Any]:
    """Fingerprint runtime source and reject uncommitted script changes."""
    repository_root = Path(_git_output("rev-parse", "--show-toplevel")).resolve()
    commit = _git_output("rev-parse", "HEAD")
    dirty_output = _git_output(
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        "scripts",
    )
    dirty_entries = [line for line in dirty_output.splitlines() if line.strip()]
    if require_clean and dirty_entries:
        formatted = "\n".join(dirty_entries[:20])
        raise RuntimeError(
            "canonical SURE execution requires a clean semantic-unlearning/scripts "
            "source tree; commit or restore these entries first:\n" + formatted
        )

    source_hashes: Dict[str, str] = {}
    for relative in RUNTIME_SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime source file is missing: {path}")
        source_hashes[relative] = sha256_file(path)
    return {
        "git_repository_root": str(repository_root),
        "git_commit": commit,
        "runtime_source_scope": "semantic-unlearning/scripts",
        "runtime_source_clean": not dirty_entries,
        "runtime_source_status": dirty_entries,
        "runtime_source_sha256": source_hashes,
    }


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


def exact_retain_kl_summary(
    payload: Mapping[str, Any] | None,
    *,
    expected_retain_num: int,
) -> Dict[str, Any] | None:
    if payload is None:
        return None
    if payload.get("audit") != "exact_sparse_output_row_retain_kl":
        raise RuntimeError("retain audit has the wrong protocol")
    if payload.get("retain_role") != "post_training_official_retain_prompt_only":
        raise RuntimeError("retain KL was not computed from the evaluation-only file")
    if int(payload.get("retain_eval_seen_during_training_or_selection", -1)) != 0:
        raise RuntimeError("retain evaluation influenced training or selection")
    if int(payload.get("retain_prompt_count", -1)) != expected_retain_num:
        raise RuntimeError("exact retain-KL prompt count differs from --retain-num")
    values = payload.get("exact_kl_base_to_edited")
    if not isinstance(values, Mapping):
        raise RuntimeError("exact retain-KL summary is missing")
    required = ("mean", "median", "p95", "p99", "max")
    summary = {name: float(values[name]) for name in required}
    if not all(math.isfinite(value) and value >= 0 for value in summary.values()):
        raise RuntimeError("exact retain-KL summary contains invalid values")
    summary["counts_above_threshold"] = dict(
        values.get("counts_above_threshold", {})
    )
    summary["selected_row_count"] = int(payload.get("selected_row_count", 0))
    return summary


def recovery_acceptance_report(
    report: Mapping[str, Any], learner_config: Mapping[str, Any]
) -> Dict[str, Any]:
    thresholds = PAIRED_RECOVERY["acceptance"]
    audits = report["forget_audits"]
    primary = report["primary_metrics"]
    final = learner_config.get("final", {})
    if not isinstance(final, Mapping):
        raise RuntimeError("paired recovery learner config lacks a final report")
    observed = {
        "FS_direct": float(audits["FS_direct_higher_is_better"]),
        "GFS_paraphrase": float(audits["GFS_paraphrase_higher_is_better"]),
        "Spe_success": float(audits["Spe_success_higher_is_better"]),
        "Spe_margin": float(primary["Spe"]["value"]),
        "PPL": float(primary["PPL"]["value"]),
        "utility_safe": final.get("utility_safe") is True,
        "locality_safe": final.get("locality_safe") is True,
    }
    guards_pass = observed["utility_safe"] and observed["locality_safe"]
    locality_pass = (
        observed["Spe_success"] >= thresholds["Spe_success_min"]
        and observed["Spe_margin"] >= thresholds["Spe_margin_min"]
        and observed["PPL"] <= thresholds["PPL_max"]
        and guards_pass
    )
    full = (
        observed["FS_direct"] >= thresholds["FS_direct_min"]
        and observed["GFS_paraphrase"] >= thresholds["GFS_paraphrase_min"]
        and locality_pass
    )
    partial = (
        observed["FS_direct"] >= thresholds["FS_direct_min"]
        and thresholds["partial_GFS_strictly_above"]
        < observed["GFS_paraphrase"]
        < thresholds["GFS_paraphrase_min"]
        and locality_pass
    )
    return {
        "predeclared_thresholds": copy.deepcopy(thresholds),
        "observed": observed,
        "every_declared_utility_locality_guard_passed": guards_pass,
        "classification": (
            "full_recovery" if full else "partial_recovery" if partial else "reject"
        ),
        "official_metrics_used_for_checkpoint_selection": False,
    }


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
    exact_retain_kl: Mapping[str, Any] | None = None,
    learner_config: Mapping[str, Any] | None = None,
    method: str = METHOD,
    protocol: str = PROTOCOL,
    treatment: str = DIRECT_TREATMENT,
    replicate_role: str = "confirmatory",
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

    base_retain_direct = pairwise_prompt_metrics(
        base_retain, "rewrite_prompts_probs"
    )
    post_retain_direct = pairwise_prompt_metrics(
        post_retain, "rewrite_prompts_probs"
    )
    base_retain_paraphrase = pairwise_prompt_metrics(
        base_retain, "paraphrase_prompts_probs"
    )
    post_retain_paraphrase = pairwise_prompt_metrics(
        post_retain, "paraphrase_prompts_probs"
    )
    retain_kl = exact_retain_kl_summary(
        exact_retain_kl, expected_retain_num=retain_num
    )

    provenance: Dict[str, Any] = {}
    if paths is not None:
        provenance = {
            "base_eval_json": str(paths.base_eval),
            "final_eval_json": str(paths.final_eval),
            "split_manifest": str(paths.split_manifest),
            "training_visible": str(paths.training_visible),
            "evaluation_only_retain_prompts": str(paths.retain_audit),
            "external_contexts": (
                str(paths.external_contexts)
                if paths.external_contexts.is_file()
                else None
            ),
            "checkpoint": str(paths.checkpoint),
            "final_sparse_delta": str(paths.final_delta),
            "learner_config": str(paths.learner_config),
            "exact_retain_kl": str(paths.exact_retain_kl),
        }

    report = {
        "schema_version": 1,
        "protocol": protocol,
        "metric_schema": METRIC_SCHEMA,
        "method": method,
        "treatment": treatment,
        "replicate_role": replicate_role,
        "dataset": "MCF",
        "seed": seed,
        "target_contract": target_contract_for_treatment(treatment),
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
            "factual_target_true_preference_direct_percent": post_retain_direct[
                "target_true_preference_percent"
            ],
            "factual_target_true_preference_paraphrase_percent": post_retain_paraphrase[
                "target_true_preference_percent"
            ],
            "base_factual_target_true_preference_direct_percent": base_retain_direct[
                "target_true_preference_percent"
            ],
            "base_factual_target_true_preference_paraphrase_percent": base_retain_paraphrase[
                "target_true_preference_percent"
            ],
            "delta_post_minus_base_direct_percent": (
                post_retain_direct["target_true_preference_percent"]
                - base_retain_direct["target_true_preference_percent"]
            ),
            "delta_post_minus_base_paraphrase_percent": (
                post_retain_paraphrase["target_true_preference_percent"]
                - base_retain_paraphrase["target_true_preference_percent"]
            ),
            "exact_sparse_output_row_KL_base_to_edited": retain_kl,
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
    if treatment == PAIRED_RECOVERY_TREATMENT:
        if learner_config is None:
            raise RuntimeError("paired recovery report requires learner config")
        report["recovery_acceptance"] = recovery_acceptance_report(
            report, learner_config
        )
    return report


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
    methods = {str(report.get("method", METHOD)) for report in reports}
    protocols = {str(report.get("protocol", PROTOCOL)) for report in reports}
    if len(methods) != 1 or len(protocols) != 1:
        raise ValueError("one aggregate cannot mix methods or protocols")
    method = next(iter(methods))
    protocol = next(iter(protocols))
    development_reports = [
        report for report in reports if report.get("replicate_role") == "development"
    ]
    confirmatory_reports = [
        report
        for report in reports
        if report.get("replicate_role", "confirmatory") == "confirmatory"
    ]
    analysis_reports = confirmatory_reports or list(reports)
    metrics = ("Eff", "Gen", "Spe", "PPL")
    aggregate: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": protocol,
        "metric_schema": METRIC_SCHEMA,
        "target_contract": reports[0].get("target_contract", TARGET_CONTRACT),
        "metric_contract": METRIC_CONTRACT,
        "seeds": [int(report["seed"]) for report in reports],
        "development_seeds": [int(report["seed"]) for report in development_reports],
        "confirmatory_seeds": [int(report["seed"]) for report in confirmatory_reports],
        "analysis_scope": (
            "confirmatory_only" if confirmatory_reports else "pilot_all_seeds"
        ),
        "required_confirmatory_seed_count": PAPER_CONFIRMATORY_SEED_COUNT,
        "paper_ready_seed_count": (
            len(confirmatory_reports) >= PAPER_CONFIRMATORY_SEED_COUNT
        ),
        "methods": {},
    }
    table_rows: List[Dict[str, str]] = []
    for label, section in (
        ("Base", "base_primary_metrics"),
        (method, "primary_metrics"),
    ):
        summaries: Dict[str, Any] = {}
        row: Dict[str, str] = {
            "Method": label,
            "Seeds": str(len(analysis_reports)),
        }
        for metric in metrics:
            values = [_value(report, section, metric) for report in analysis_reports]
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

    retain_metric_paths = {
        "base_direct_percent": "base_factual_target_true_preference_direct_percent",
        "post_direct_percent": "factual_target_true_preference_direct_percent",
        "delta_direct_percent": "delta_post_minus_base_direct_percent",
        "base_paraphrase_percent": "base_factual_target_true_preference_paraphrase_percent",
        "post_paraphrase_percent": "factual_target_true_preference_paraphrase_percent",
        "delta_paraphrase_percent": "delta_post_minus_base_paraphrase_percent",
    }
    aggregate["retain_utility_audits"] = {}
    for output_name, report_name in retain_metric_paths.items():
        values = [
            float(report["retain_utility_audits"][report_name])
            for report in analysis_reports
        ]
        aggregate["retain_utility_audits"][output_name] = {
            "mean": float(statistics.mean(values)),
            "sample_sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
            "values": values,
        }
    exact_kl_reports = [
        report["retain_utility_audits"].get(
            "exact_sparse_output_row_KL_base_to_edited"
        )
        for report in analysis_reports
    ]
    if all(isinstance(value, Mapping) for value in exact_kl_reports):
        aggregate["retain_utility_audits"]["exact_sparse_output_row_KL"] = {
            metric: {
                "mean": float(
                    statistics.mean(float(value[metric]) for value in exact_kl_reports)
                ),
                "values": [float(value[metric]) for value in exact_kl_reports],
            }
            for metric in ("mean", "p95", "max")
        }

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
            (
                "Development seeds are excluded when confirmatory seeds are present. "
                f"Paper-ready status requires {PAPER_CONFIRMATORY_SEED_COUNT} "
                "confirmatory seeds."
            ),
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
    architecture = architecture_for_treatment(args.treatment)
    stage1 = architecture["stage1"]
    stage2 = architecture["stage2"]
    acceptance = architecture["acceptance"]
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
    if args.treatment == PAIRED_RECOVERY_TREATMENT:
        command.extend(
            [
                "--external-contexts",
                str(paths.external_contexts),
                "--locality-token-topk-per-row",
                str(PAIRED_RECOVERY["locality_token_topk_per_row"]),
                "--locality-uniform-prompt-count",
                str(PAIRED_RECOVERY["locality_uniform_prompt_count"]),
                "--locality-pool-seed",
                str(PAIRED_RECOVERY["locality_pool_seed"]),
                "--locality-train-batch-size",
                "128",
                "--locality-eval-batch-size",
                "512",
                "--stage1-locality-kl-weight",
                str(PAIRED_RECOVERY["stage1_locality_kl_weight"]),
                "--stage2-locality-kl-weight",
                str(PAIRED_RECOVERY["stage2_locality_kl_weight"]),
                "--locality-kl-mean-budget",
                str(PAIRED_RECOVERY["locality_kl_mean_budget"]),
                "--locality-kl-p95-budget",
                str(PAIRED_RECOVERY["locality_kl_p95_budget"]),
                "--locality-kl-max-budget",
                str(PAIRED_RECOVERY["locality_kl_max_budget"]),
            ]
        )
    return command


def external_context_command(
    args: argparse.Namespace,
    paths: SeedPaths,
    *,
    seed: int,
) -> List[str]:
    utility_wikipedia_dir = args.utility_wikipedia_dir or args.wikipedia_dir
    command = _python(
        "build_sure_mcf_external_contexts.py",
        "--training-visible-path",
        paths.training_visible,
        "--wikipedia-dir",
        utility_wikipedia_dir,
        "--output-path",
        paths.external_contexts,
        "--corpus-document-limit",
        args.utility_docs,
        "--exclude-first",
        20,
        "--contexts-per-record",
        PAIRED_RECOVERY["external_locality_contexts_per_record"],
        "--lead-chars",
        PAIRED_RECOVERY["external_context_lead_characters"],
        "--context-profile",
        PAIRED_CONTEXT_PROFILE,
        "--seed",
        seed,
    )
    if args.require_corpus_protocol:
        command.extend(["--require-corpus-protocol", args.require_corpus_protocol])
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
    steps = [
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
    ]
    if args.treatment == PAIRED_RECOVERY_TREATMENT:
        steps.append(
            Step(
                "BUILD LOCKED PAIRED EXTERNAL CONTEXTS",
                external_context_command(args, paths, seed=seed),
            )
        )
    steps.extend(
        [
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
            Step(
                "POST-CHECKPOINT EXACT RETAIN KL AUDIT",
                _python(
                    "audit_sure_exact_retain_kl.py",
                    "--model-path",
                    args.model_path,
                    "--retain-prompt-path",
                    paths.retain_audit,
                    "--delta-path",
                    paths.final_delta,
                    "--scale",
                    1,
                    "--output-json",
                    paths.exact_retain_kl,
                    "--batch-size",
                    8,
                    "--dtype",
                    args.dtype,
                    "--device-map",
                    args.device_map,
                ),
            ),
        ]
    )
    return steps


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
    print(f"\n===== {step.label} =====", flush=True)
    print(shlex.join(step.command), flush=True)
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
    parser.add_argument(
        "--treatment",
        choices=(DIRECT_TREATMENT, PAIRED_RECOVERY_TREATMENT),
        default=DIRECT_TREATMENT,
        help=(
            "Locked training treatment. paired_context_recovery uses the "
            "predeclared W10K paired-answer-cue context treatment."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--development-seeds",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Additional seeds excluded from paper aggregates. Seed 1 is always "
            "development evidence when present."
        ),
    )
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
    if args.development_seeds is None:
        args.development_seeds = []
    if len(args.development_seeds) != len(set(args.development_seeds)):
        parser.error("--development-seeds must be unique")
    if not set(args.development_seeds).issubset(set(args.seeds)):
        parser.error("--development-seeds must be a subset of --seeds")
    development_seed_set = set(args.development_seeds)
    if 1 in args.seeds:
        development_seed_set.add(1)
    args.development_seeds = [
        seed for seed in args.seeds if seed in development_seed_set
    ]
    if args.treatment == PAIRED_RECOVERY_TREATMENT:
        if args.utility_docs != int(PAIRED_RECOVERY["utility_documents"]):
            parser.error(
                "paired_context_recovery is locked to --utility-docs 10000"
            )
        if args.allow_capped_utility_cache:
            parser.error("paired_context_recovery cannot use a capped utility cache")
        if args.require_corpus_protocol != "sure_external_wikipedia_corpus_v1":
            parser.error(
                "paired_context_recovery requires "
                "--require-corpus-protocol sure_external_wikipedia_corpus_v1"
            )
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
        "build_sure_mcf_external_contexts.py",
        "sure_mcf_target_aware_direct_only.py",
        "mcf_zero_unlearn_official_eval.py",
        "audit_sure_exact_retain_kl.py",
    ):
        if not (SCRIPTS_DIR / helper).is_file():
            raise FileNotFoundError(f"required SURE component is missing: {helper}")


def print_contract(args: argparse.Namespace, utility_cache: Path) -> None:
    print("=" * 72, flush=True)
    print("MCF SURE TWO-STAGE — LOCKED TARGET CONTRACT", flush=True)
    print(f"treatment   : {args.treatment}", flush=True)
    if args.treatment == PAIRED_RECOVERY_TREATMENT:
        print(
            f"context view: {PAIRED_CONTEXT_PROFILE} (W10K locked recovery)",
            flush=True,
        )
    print("target_true : SENSITIVE -> bounded GA -> increase NLL", flush=True)
    print("target_new  : NON-SENSITIVE REPLACEMENT -> GD -> decrease NLL", flush=True)
    print("Eff / Gen   : residual sensitive preference, LOWER is better", flush=True)
    print("Spe         : neighborhood preservation, HIGHER is better", flush=True)
    print("PPL         : fluency, LOWER/base-stable is better", flush=True)
    print(
        "Official paraphrases/neighborhoods/retain/PPL: evaluation-only",
        flush=True,
    )
    print(f"Wikipedia utility cache: {utility_cache}", flush=True)
    print("=" * 72, flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    method, protocol = treatment_method_protocol(args.treatment)
    architecture = architecture_for_treatment(args.treatment)
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
    execution_provenance = source_provenance(require_clean=not args.dry_run)
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
        print(f"Reusing utility cache: {utility_cache}", flush=True)

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
        exact_retain_kl = json.loads(
            paths.exact_retain_kl.read_text(encoding="utf-8")
        )
        learner_config = json.loads(
            paths.learner_config.read_text(encoding="utf-8")
        )
        report = build_seed_report(
            base=base,
            post=post,
            manifest=manifest,
            training_rows=training_rows,
            seed=seed,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            paths=paths,
            exact_retain_kl=exact_retain_kl,
            learner_config=learner_config,
            method=method,
            protocol=protocol,
            treatment=args.treatment,
            replicate_role=(
                "development" if seed in args.development_seeds else "confirmatory"
            ),
        )
        report["execution_provenance"] = execution_provenance
        report["learner_selection"] = {
            "selection_mode": learner_config.get("selected", {}).get(
                "selection_mode"
            ),
            "utility_cache_metadata": learner_config.get("utility_cache_metadata"),
            "utility_probability_reconstruction": learner_config.get(
                "utility_probability_reconstruction"
            ),
        }
        write_json(paths.metrics, report)
        reports.append(report)
        primary = report["primary_metrics"]
        print("\n===== PAPER METRICS (TARGET_TRUE SENSITIVE) =====", flush=True)
        for metric in ("Eff", "Gen", "Spe", "PPL"):
            print(
                f"{metric:<4} {primary[metric]['value']:>10.4f}  "
                f"{primary[metric]['direction']}",
                flush=True,
            )
        print(
            "Base: "
            + ", ".join(
                f"{metric}={report['base_primary_metrics'][metric]['value']:.4f}"
                for metric in ("Eff", "Gen", "Spe", "PPL")
            ),
            flush=True,
        )
        print(
            "Selection mode:",
            report["learner_selection"]["selection_mode"],
            flush=True,
        )
        if "recovery_acceptance" in report:
            print(
                "Recovery classification:",
                report["recovery_acceptance"]["classification"],
                flush=True,
            )
        print(f"Wrote: {paths.metrics}", flush=True)

    if args.dry_run:
        print(
            "\nDry run complete; no files were written and no models were loaded.",
            flush=True,
        )
        return

    run_config = {
        "schema_version": 1,
        "protocol": protocol,
        "method": method,
        "treatment": args.treatment,
        "target_contract": target_contract_for_treatment(args.treatment),
        "metric_contract": METRIC_CONTRACT,
        "architecture": architecture,
        "execution_provenance": execution_provenance,
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
        "development_seeds": list(args.development_seeds),
        "confirmatory_seeds": [
            seed for seed in args.seeds if seed not in args.development_seeds
        ],
        "required_confirmatory_seed_count": PAPER_CONFIRMATORY_SEED_COUNT,
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
        "dtype": args.dtype,
        "device_map": args.device_map,
    }
    first_utility_metadata = reports[0].get("learner_selection", {}).get(
        "utility_cache_metadata"
    )
    if isinstance(first_utility_metadata, Mapping):
        run_config["observed_utility_cache"] = {
            "actual_document_sample_size": int(
                first_utility_metadata.get("actual_document_sample_size", 0)
            ),
            "actual_utility_prompt_count": int(
                first_utility_metadata.get("actual_utility_prompt_count", 0)
            ),
            "predictor_hidden_state_count": int(
                first_utility_metadata.get("predictor_hidden_state_count", 0)
            ),
            "utility_hidden_sha256": first_utility_metadata.get(
                "utility_hidden_sha256"
            ),
            "base_logsumexp_sha256": first_utility_metadata.get(
                "base_logsumexp_sha256"
            ),
        }
    write_json(output_root / "run_config.json", run_config)
    aggregate = aggregate_reports(reports, output_root)
    print(f"\nComplete: {output_root}", flush=True)
    print(f"Table rows: {output_root / 'table1_rows.md'}", flush=True)
    if not aggregate["paper_ready_seed_count"]:
        print(
            "NOTE: fewer than 10 confirmatory seeds; treat this as a pilot, "
            "not a paper aggregate.",
            flush=True,
        )


if __name__ == "__main__":
    main()
