#!/usr/bin/env python3
"""Train the locked RWKU-H-W1K Stephen King head-only feasibility run.

This learner accepts only the independently generated RWKU v3 atomic training
bundle, its generator receipt, a target-excluded W1K Wikipedia cache, and the
base model.  It has no argument for official RWKU probes.  The generic staged
RWKU evaluator may open those probes only after this program freezes a
checkpoint receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import build_rwku_entity_facts as entity_facts
import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import mcf_zero_unlearn_official_eval as nll_scorer
import rwku_artifact_access as artifact_access
import rwku_checkpoint_receipt as checkpoint_receipt
import rwku_eval
import rwku_experiment
import sure_canonical_core as core
import sure_mcf_direct_fs_repair as exact
import sure_mcf_target_aware_two_stage as joint
import sure_minimal_two_stage as learner


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = (
    PROJECT_ROOT / "config" / "rwku" / "sure_head_only_w1k_seed0.json"
)
EXPECTED_CONFIGURATION_ID = "rwku-h-w1k-stephen-king-atomic-seed0-v1"
EXPECTED_SCHEMA = "rwku_sure_head_only_w1k_configuration_v1"
EXPECTED_GENERATION_CONFIGURATION = "llama32_3b_target_corpus_v3_atomic_facts"
EXPECTED_CONFIGURATION_SHA256 = (
    "924062c9a4addb01d21d0bfe41c54ce9db311e75a87fbe9acd9145efa1dc5bef"
)
EXPERIMENT_ID = "rwku-h-w1k-stephen-king-atomic-seed0-v1"
PROTOCOL_STATUS = "rwku_target_only_auxwiki_sure_head_only_w1k_development"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-revision", default="local_pinned_snapshot")
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--generator-receipt", type=Path, required=True)
    parser.add_argument("--utility-cache", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    return parser.parse_args(argv)


def load_locked_configuration(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("Unsupported RWKU-H-W1K configuration schema")
    required_identity = {
        "configuration_id": EXPECTED_CONFIGURATION_ID,
        "development_only": True,
        "seed": 0,
        "target_entity": "Stephen King",
        "target_entity_id": "rwku:1_Stephen_King",
        "required_generation_configuration_id": EXPECTED_GENERATION_CONFIGURATION,
        "neutral_target": "Unknown",
        "editable_parameters": "content_sensitive_answer_plus_neutral_lm_head_rows_only",
    }
    for key, expected in required_identity.items():
        if value.get(key) != expected:
            raise ValueError(f"Locked RWKU-H-W1K configuration changed {key}")
    utility = value.get("utility", {})
    acceptance = value.get("acceptance", {})
    stage1 = value.get("stage1", {})
    boundary = value.get("data_boundary", {})
    fixed_checks = {
        "Wikipedia document count": (utility.get("document_count"), 1000),
        "Wikipedia prompt count": (utility.get("requested_prompt_count"), 100000),
        "Wikipedia minimum prompts": (utility.get("minimum_prompt_count"), 90000),
        "Wikipedia target exclusion": (
            utility.get("required_casefold_exclusion"),
            "stephen king",
        ),
        "Stage-1 rank": (stage1.get("rank"), 4),
        "checkpoint dtype": (acceptance.get("checkpoint_dtype"), "bf16"),
        "device map": (acceptance.get("device_map"), "single"),
        "official learner access": (
            boundary.get("official_rwku_records_available_to_learner"),
            False,
        ),
        "official checkpoint selection": (
            boundary.get("official_rwku_records_used_for_checkpoint_selection"),
            False,
        ),
    }
    for label, (actual, expected) in fixed_checks.items():
        if actual != expected:
            raise ValueError(f"Locked RWKU-H-W1K configuration changed {label}")
    allowed_styles = tuple(value.get("allowed_training_view_styles", ()))
    expected_styles = (
        "direct question",
        "cloze",
        "deterministic paraphrase",
        "forced-prefix",
    )
    if allowed_styles != expected_styles:
        raise ValueError("Locked atomic training-view styles changed")
    row_filter = value.get("sensitive_row_filter", {})
    expected_excluded_tokens = (
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "the",
        "to",
    )
    if (
        row_filter.get("exclude_punctuation_only") is not True
        or tuple(row_filter.get("excluded_casefold_tokens", ()))
        != expected_excluded_tokens
        or row_filter.get("require_content_row_per_atomic_answer") is not True
    ):
        raise ValueError("Locked sensitive-answer LM-head row filter changed")
    if int(value.get("minimum_atomic_fact_count", 0)) < 1:
        raise ValueError("minimum_atomic_fact_count must be positive")
    if artifact_access.sha256_file(path) != EXPECTED_CONFIGURATION_SHA256:
        raise ValueError("Locked RWKU-H-W1K configuration digest changed")
    return value


def optimization_namespace(
    configuration: Mapping[str, Any], *, prompt_count: int
) -> argparse.Namespace:
    utility = configuration["utility"]
    stage1 = configuration["stage1"]
    stage2 = configuration["stage2"]
    acceptance = configuration["acceptance"]
    return argparse.Namespace(
        seed=int(configuration["seed"]),
        forget_num=int(prompt_count),
        utility_sample_size=int(utility["document_count"]),
        utility_prompt_count=int(utility["requested_prompt_count"]),
        utility_token_topk_per_row=int(utility["token_topk_per_row"]),
        utility_uniform_prompt_count=int(utility["uniform_prompt_count"]),
        utility_pool_seed=int(utility["pool_seed"]),
        utility_train_batch_size=int(utility["train_batch_size"]),
        utility_eval_batch_size=int(utility["eval_batch_size"]),
        cache_batch_size=int(utility["cache_batch_size"]),
        stage1_rank=int(stage1["rank"]),
        stage1_steps=int(stage1["steps"]),
        stage1_lr=float(stage1["learning_rate"]),
        stage1_pairwise_target=float(stage1["pairwise_target"]),
        stage1_true_nll_increase=float(stage1["sensitive_nll_increase"]),
        stage1_new_nll_decrease=float(stage1["neutral_nll_decrease"]),
        stage1_pairwise_weight=float(stage1["pairwise_weight"]),
        stage1_true_ga_weight=float(stage1["sensitive_ga_weight"]),
        stage1_new_gd_weight=float(stage1["neutral_gd_weight"]),
        stage1_utility_kl_weight=float(utility["kl_weight"]),
        stage1_l2_weight=float(stage1["l2_weight"]),
        stage1_candidate_scales=str(stage1["candidate_scales"]),
        required_pairwise_margin=float(acceptance["required_pairwise_margin"]),
        stage2_solver_margins=str(stage2["solver_margins"]),
        stage2_rank_ladder=str(stage2["rank_ladder"]),
        stage2_maxiter=int(stage2["max_iterations"]),
        stage2_ftol=float(stage2["ftol"]),
        stage2_constraint_tolerance=float(stage2["constraint_tolerance"]),
        stage2_residual_l2_weight=float(stage2["residual_l2_weight"]),
        constraint_context_weight=float(stage2["constraint_context_weight"]),
        contrastive_eps=float(stage2["contrastive_epsilon"]),
        utility_kl_mean_budget=float(utility["kl_mean_budget"]),
        utility_kl_p95_budget=float(utility["kl_p95_budget"]),
        utility_kl_max_budget=float(utility["kl_max_budget"]),
        max_total_delta_norm=float(acceptance["max_total_delta_norm"]),
        dtype=str(acceptance["checkpoint_dtype"]),
        device_map=str(acceptance["device_map"]),
    )


def _artifact_identity_fields(
    artifact: Mapping[str, Any], configuration: Mapping[str, Any], label: str
) -> None:
    if artifact.get("protocol_label") != artifact_access.TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError(f"{label} is not a target-only RWKU artifact")
    if artifact.get("protocol_status") != artifact_access.TARGET_ONLY_PROTOCOL_STATUS:
        raise ValueError(f"{label} has an unexpected protocol status")
    metadata = artifact.get("metadata", {})
    expected = {
        "entity_id": configuration["target_entity_id"],
        "subject": configuration["target_entity"],
        "seed": configuration["seed"],
        "generation_configuration_id": configuration[
            "required_generation_configuration_id"
        ],
    }
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise ValueError(f"{label} metadata changed {key}")


def validate_atomic_views(
    views: Sequence[Mapping[str, Any]], configuration: Mapping[str, Any]
) -> Dict[str, Any]:
    if not views:
        raise ValueError("Atomic training bundle contains no views")
    allowed = set(configuration["allowed_training_view_styles"])
    required_per_fact = {"direct question", "cloze", "deterministic paraphrase"}
    seen_view_ids: set[str] = set()
    fact_styles: Dict[str, set[str]] = {}
    fact_identity: Dict[str, Tuple[str, str, str]] = {}
    style_counts: Dict[str, int] = {style: 0 for style in sorted(allowed)}
    for position, view in enumerate(views):
        if not isinstance(view, Mapping):
            raise ValueError(f"Atomic view {position} is not an object")
        if view.get("schema_version") != entity_facts.ENTITY_FACT_SCHEMA_VERSION:
            raise ValueError(f"Atomic view {position} has an unsupported schema")
        if view.get("training_allowed") is not True:
            raise ValueError("Atomic bundle exposes a non-training view")
        if bool(view.get("boundary_expanding", False)):
            raise ValueError("Boundary-expanding atomic views are forbidden")
        if view.get("source_file") != "generated_raw_corpus.json":
            raise ValueError("Atomic view source is not independently generated")
        if view.get("level") != "generated":
            raise ValueError("Atomic training views must have level=generated")
        if view.get("subject") != configuration["target_entity"]:
            raise ValueError("Atomic view subject is not Stephen King")
        if view.get("entity_id") != configuration["target_entity_id"]:
            raise ValueError("Atomic view entity ID is not the locked target")
        style = str(view.get("prompt_style", ""))
        if style not in allowed or view.get("query_type") != style:
            raise ValueError(f"Forbidden or inconsistent atomic view style: {style!r}")
        query = str(view.get("query", "")).strip()
        canonical = str(view.get("canonical_sensitive_answer", "")).strip()
        alias = str(view.get("sensitive_answer_alias", "")).strip()
        fact_id = str(view.get("fact_id", "")).strip()
        relation_id = str(view.get("relation_id", "")).strip()
        view_id = str(view.get("view_id", "")).strip()
        if not all((query, canonical, alias, fact_id, relation_id, view_id)):
            raise ValueError(f"Atomic view {position} has an empty required field")
        if configuration["target_entity"].casefold() not in query.casefold():
            raise ValueError("Every atomic training query must name the target")
        if alias.casefold() == str(configuration["neutral_target"]).casefold():
            raise ValueError("Sensitive answer collides with the neutral target")
        if view_id in seen_view_ids:
            raise ValueError(f"Duplicate atomic view ID: {view_id}")
        expected_content = entity_facts.view_content_sha256(view)
        expected_view_id = hashlib.sha256(
            f"{fact_id}:{expected_content}".encode("utf-8")
        ).hexdigest()
        if (
            view.get("view_content_sha256") != expected_content
            or view_id != expected_view_id
        ):
            raise ValueError("Atomic view content identity is invalid")
        if style == "forced-prefix":
            if "Answer prefix:" not in query or alias == canonical:
                raise ValueError("Forced-prefix view must train only its answer suffix")
        seen_view_ids.add(view_id)
        style_counts[style] += 1
        fact_styles.setdefault(fact_id, set()).add(style)
        identity = (relation_id, canonical, str(view["entity_id"]))
        if fact_id in fact_identity and fact_identity[fact_id] != identity:
            raise ValueError("Atomic views disagree on fact identity")
        fact_identity[fact_id] = identity
    minimum = int(configuration["minimum_atomic_fact_count"])
    if len(fact_styles) < minimum:
        raise ValueError(
            f"Atomic bundle has {len(fact_styles)} facts; at least {minimum} required"
        )
    for fact_id, styles in fact_styles.items():
        if not required_per_fact.issubset(styles):
            missing = sorted(required_per_fact - styles)
            raise ValueError(f"Atomic fact {fact_id} lacks required views: {missing}")
        if any(
            sum(
                str(view.get("fact_id")) == fact_id
                and str(view.get("prompt_style")) == style
                for view in views
            )
            != 1
            for style in styles
        ):
            raise ValueError(f"Atomic fact {fact_id} repeats a view style")
    return {
        "fact_count": len(fact_styles),
        "view_count": len(views),
        "style_counts": style_counts,
        "fact_ids": sorted(fact_styles),
        "view_ids_sha256": artifact_access.sha256_json(
            [str(view["view_id"]) for view in views]
        ),
        "boundary_expanding_view_count": 0,
    }


def load_atomic_bundle(
    bundle_path: Path,
    generator_receipt_path: Path,
    configuration: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    bundle = artifact_access.read_artifact(
        bundle_path,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    generator = artifact_access.read_artifact(
        generator_receipt_path,
        stage="train",
        expected_role="generator_receipt",
    )
    _artifact_identity_fields(bundle, configuration, "training bundle")
    _artifact_identity_fields(generator, configuration, "generator receipt")
    receipt = generator.get("payload", {})
    required_receipt = {
        "status": "complete",
        "target_entity": configuration["target_entity"],
        "entity_id": configuration["target_entity_id"],
        "protocol_label": artifact_access.TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": artifact_access.TARGET_ONLY_PROTOCOL_STATUS,
        "official_rwku_records_accessed": False,
        "fact_extractor_implementation": "atomic_relation_fact_extractor_v1",
        "parser_implementation_revision": "complete_json_object_atomic_v1",
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise ValueError(f"Generator receipt changed {key}")
    if receipt.get("random_seeds") != [0]:
        raise ValueError("Generator receipt is not the seed-0 atomic bundle")
    extraction = receipt.get("extraction_configuration", {})
    if extraction.get("reverse_prompts_enabled") is not False:
        raise ValueError("Atomic generator enabled reverse prompts")
    if receipt.get("final_entity_fact_bundle_sha256") != bundle.get("sha256"):
        raise ValueError("Generator receipt does not identify the training bundle")
    views = bundle.get("payload", {}).get("views")
    if not isinstance(views, list):
        raise ValueError("Atomic training bundle payload lacks views")
    audit = validate_atomic_views(views, configuration)
    if int(receipt.get("accepted_fact_count", -1)) != audit["fact_count"]:
        raise ValueError("Generator accepted-fact count differs from the bundle")
    return [dict(view) for view in views], audit, receipt


def validate_generator_base_model(
    generator_receipt: Mapping[str, Any], model_path: str
) -> Dict[str, Any]:
    """Require the atomic facts to have been generated by this base snapshot."""

    declared = str(generator_receipt.get("generator_model_path", "")).strip()
    revision = str(generator_receipt.get("generator_model_revision", "")).strip()
    snapshot = generator_receipt.get("local_snapshot_identity", {})
    if not declared or not revision or not isinstance(snapshot, Mapping):
        raise ValueError("Generator receipt lacks a pinned base-model identity")
    if snapshot.get("exists") is not True:
        raise ValueError("Generator receipt did not use a local model snapshot")
    declared_path = Path(declared).expanduser().resolve()
    requested_path = Path(model_path).expanduser().resolve()
    if declared_path != requested_path:
        raise ValueError(
            "Atomic corpus generator snapshot differs from the unlearning base model"
        )
    return {
        "generator_model_path": str(declared_path),
        "unlearning_model_path": str(requested_path),
        "generator_model_revision": revision,
        "same_local_snapshot": True,
    }


def compile_prompt_records(
    views: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    neutral_target: str,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    prompt_index_by_fact: Dict[str, int] = {}
    for position, view in enumerate(views):
        style = str(view["prompt_style"])
        level = "1" if style == "cloze" else "2"
        row = {
            "query": str(view["query"]),
            "answer": str(view["sensitive_answer_alias"]),
            "subject": str(view["subject"]),
            "level": level,
            "type": style,
        }
        prompt = rwku_eval.format_qa_prompt(tokenizer, row)
        fact_id = str(view["fact_id"])
        local_index = prompt_index_by_fact.get(fact_id, 0)
        prompt_index_by_fact[fact_id] = local_index + 1
        records.append(
            {
                "case_id": position,
                "source_record_position": position,
                "prompt_kind": (
                    "direct" if style == "direct question" else "generated_subject"
                ),
                "prompt_index": local_index,
                "prompt_text": prompt,
                "fact_id": fact_id,
                "view_id": str(view["view_id"]),
                "prompt_style": style,
                "requested_rewrite": {
                    "prompt": "{}",
                    "subject": prompt,
                    "target_sensitive": {
                        "str": str(view["sensitive_answer_alias"]).strip()
                    },
                    "target_reference": {"str": str(neutral_target).strip()},
                },
            }
        )
    if len({record["prompt_text"] for record in records}) != len(records):
        raise ValueError("Atomic prompt compiler produced duplicate rendered prompts")
    return records


def validate_w1k_utility_metadata(
    metadata: Mapping[str, Any], configuration: Mapping[str, Any]
) -> Dict[str, Any]:
    utility = configuration["utility"]
    checks = {
        "requested_document_sample_size": int(utility["document_count"]),
        "actual_document_sample_size": int(utility["document_count"]),
        "requested_utility_prompt_count": int(utility["requested_prompt_count"]),
        "required_minimum_utility_prompt_count": int(utility["minimum_prompt_count"]),
        "excluded_prefix_document_count": int(utility["exclude_first_documents"]),
    }
    for key, expected in checks.items():
        if int(metadata.get(key, -1)) != expected:
            raise ValueError(f"W1K utility cache changed {key}")
    actual_prompts = int(metadata.get("actual_utility_prompt_count", 0))
    if (
        not int(utility["minimum_prompt_count"])
        <= actual_prompts
        <= int(utility["requested_prompt_count"])
    ):
        raise ValueError("W1K utility cache predictor count is outside the lock")
    corpus_receipt = metadata.get("corpus_receipt")
    if (
        not isinstance(corpus_receipt, Mapping)
        or corpus_receipt.get("protocol") != utility["corpus_protocol"]
    ):
        raise ValueError("W1K utility cache lacks the prepared Wikipedia receipt")
    exclusions = wikipedia.normalize_casefold_substrings(
        metadata.get("excluded_casefold_substrings", [])
    )
    required_exclusion = str(utility["required_casefold_exclusion"]).casefold()
    if exclusions != (required_exclusion,):
        raise ValueError("W1K utility cache does not exclusively exclude Stephen King")
    rejected = int(metadata.get("target_substring_excluded_document_count", -1))
    digest = metadata.get("target_substring_excluded_document_indices_sha256")
    if rejected < 0 or not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("W1K utility target-exclusion audit is incomplete")
    return {
        "document_count": int(metadata["actual_document_sample_size"]),
        "predictor_count": actual_prompts,
        "target_exclusion": required_exclusion,
        "target_matching_documents_rejected": rejected,
        "corpus_protocol": corpus_receipt["protocol"],
    }


def select_edit_row_ids(
    tokenizer: Any,
    sensitive_cases: Sequence[core.SensitivePredictionCase],
    sensitive_ids: torch.Tensor,
    neutral_ids: torch.Tensor,
    *,
    prompt_count: int,
    configuration: Mapping[str, Any],
) -> Tuple[List[int], Dict[str, Any]]:
    """Select content-bearing sensitive rows plus every neutral-target row."""

    if len(sensitive_cases) != int(sensitive_ids.numel()):
        raise ValueError("Sensitive cases and token IDs do not align")
    row_filter = configuration["sensitive_row_filter"]
    excluded = set(str(value) for value in row_filter["excluded_casefold_tokens"])
    selected_sensitive: set[int] = set()
    rejected_sensitive: set[int] = set()
    coverage = [0] * int(prompt_count)
    decoded: Dict[int, str] = {}
    for case, raw_token_id in zip(
        sensitive_cases, sensitive_ids.detach().cpu().tolist()
    ):
        token_id = int(raw_token_id)
        token_text = str(tokenizer.decode([token_id]))
        decoded[token_id] = token_text
        normalized = re.sub(r"[\W_]+", "", token_text.casefold(), flags=re.UNICODE)
        content_bearing = bool(normalized) and normalized not in excluded
        if content_bearing:
            selected_sensitive.add(token_id)
            coverage[int(case.record_position)] += 1
        else:
            rejected_sensitive.add(token_id)
    if row_filter["require_content_row_per_atomic_answer"] is True:
        uncovered = [index for index, count in enumerate(coverage) if count == 0]
        if uncovered:
            raise ValueError(
                "Sensitive row filter left atomic prompts without a content row: "
                f"{uncovered[:10]}"
            )
    neutral = set(int(value) for value in neutral_ids.detach().cpu().tolist())
    selected = sorted(selected_sensitive | neutral)
    if not selected:
        raise ValueError("Sensitive/neutral row selection is empty")
    return selected, {
        "filter": dict(row_filter),
        "selected_sensitive_row_ids": sorted(selected_sensitive),
        "rejected_sensitive_row_ids": sorted(rejected_sensitive - neutral),
        "neutral_row_ids": sorted(neutral),
        "selected_row_ids": selected,
        "selected_row_decodings": {
            str(token_id): str(tokenizer.decode([token_id])) for token_id in selected
        },
        "rejected_sensitive_row_decodings": {
            str(token_id): decoded[token_id]
            for token_id in sorted(rejected_sensitive - neutral)
        },
        "minimum_content_rows_per_prompt": min(coverage),
    }


@torch.no_grad()
def materialized_atomic_report(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
    required_margin: float,
) -> Dict[str, Any]:
    groups: Dict[Tuple[str, str], List[Tuple[int, Mapping[str, Any]]]] = {}
    for position, record in enumerate(prompt_records):
        rewrite = record["requested_rewrite"]
        key = (
            str(rewrite["target_sensitive"]["str"]),
            str(rewrite["target_reference"]["str"]),
        )
        groups.setdefault(key, []).append((position, record))
    separations: List[float | None] = [None] * len(prompt_records)
    for (sensitive, neutral), entries in groups.items():
        scores = nll_scorer.official_test_batch_prediction(
            model,
            tokenizer,
            [str(record["prompt_text"]) for _, record in entries],
            neutral,
            sensitive,
            device,
            llama_like=llama_like,
        )
        if len(scores) != len(entries):
            raise RuntimeError("NLL scorer omitted an atomic training prompt")
        for (position, _), score in zip(entries, scores):
            separations[position] = float(score["target_true"] - score["target_new"])
    if any(value is None for value in separations):
        raise RuntimeError("Atomic materialization report is incomplete")
    report = joint.grouped_pairwise_report(
        torch.tensor(separations, dtype=torch.float32),
        prompt_records,
        required_margin=required_margin,
    )
    report.update(
        {
            "scorer": "teacher_forced_sequence_nll_sensitive_minus_neutral",
            "checkpoint_dtype_forward": True,
            "training_prompt_scope": "atomic_direct_cloze_paraphrase_forced_prefix",
            "official_rwku_records_evaluated": False,
            "official_rwku_records_used_for_selection": False,
        }
    )
    return report


def atomic_candidate_feasible(report: Mapping[str, Any]) -> bool:
    return bool(
        int(report.get("direct_prompt_count", 0)) > 0
        and int(report.get("generated_subject_prompt_count", 0)) > 0
        and report.get("FS") == 100.0
        and report.get("generated_subject_FS") == 100.0
        and int(report.get("direct_margin_failures", -1)) == 0
        and int(report.get("generated_subject_margin_failures", -1)) == 0
        and report.get("utility_safe") is True
    )


def _parameter_versions_except_output(
    model: torch.nn.Module, output_layer: torch.nn.Module
) -> Dict[str, int]:
    return {
        name: int(parameter._version)
        for name, parameter in model.named_parameters()
        if parameter is not output_layer.weight
    }


def assert_head_only_versions(
    model: torch.nn.Module,
    output_layer: torch.nn.Module,
    before: Mapping[str, int],
) -> None:
    after = _parameter_versions_except_output(model, output_layer)
    if dict(before) != after:
        changed = sorted(
            name
            for name in set(before) | set(after)
            if before.get(name) != after.get(name)
        )
        raise RuntimeError(f"Non-LM-head parameters changed: {changed[:10]}")
    if model.get_input_embeddings().weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("Input embeddings became tied to the edited LM head")


def _state_namespace(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        output_root=Path(args.output_root),
        experiment_id=str(args.experiment_id),
        training_source=rwku_experiment.TRAINING_SOURCE_TARGET_ONLY,
    )


def _verify_prepared_state(
    args: argparse.Namespace,
    configuration: Mapping[str, Any],
) -> Tuple[SimpleNamespace, Dict[str, Any], Path]:
    state_args = _state_namespace(args)
    state = rwku_experiment._read_state(state_args)
    if state.get("state") != "PREPARED":
        raise ValueError(
            f"RWKU-H-W1K training requires PREPARED, got {state.get('state')}"
        )
    if state.get("training_source") != rwku_experiment.TRAINING_SOURCE_TARGET_ONLY:
        raise ValueError("Prepared state belongs to a different RWKU training track")
    if state.get("confirmatory") is not False:
        raise ValueError("RWKU-H-W1K is a development-only feasibility run")
    target = state.get("target", {})
    if (
        target.get("seed") != 0
        or target.get("subject") != configuration["target_entity"]
    ):
        raise ValueError("Prepared state is not the seed-0 Stephen King target")
    bundle_path = Path(args.training_bundle).resolve()
    generator_path = Path(args.generator_receipt).resolve()
    if Path(state.get("prepared_training_bundle_path", "")).resolve() != bundle_path:
        raise ValueError("Learner bundle path differs from the prepared state")
    if state.get("prepared_training_bundle_file_sha256") != artifact_access.sha256_file(
        bundle_path
    ):
        raise ValueError("Learner bundle changed after preparation")
    if state.get(
        "prepared_generator_receipt_file_sha256"
    ) != artifact_access.sha256_file(generator_path):
        raise ValueError("Generator receipt changed after preparation")
    if state.get("official_evaluation_opened") is not False:
        raise ValueError("Official RWKU evaluation was already opened")
    run_dir = Path(args.output_root).resolve() / str(args.experiment_id)
    receipt_path = run_dir / "checkpoint_receipt.json"
    if receipt_path.exists():
        checkpoint_receipt.assert_model_modification_allowed(
            receipt_path, experiment_id=str(args.experiment_id)
        )
        raise FileExistsError(f"Checkpoint receipt already exists: {receipt_path}")
    return state_args, state, run_dir


def main() -> None:
    args = parse_args()
    configuration_path = Path(args.configuration).resolve()
    configuration = load_locked_configuration(configuration_path)
    if args.experiment_id != configuration["configuration_id"]:
        raise ValueError("Experiment ID must equal the locked configuration ID")
    state_args, _state, run_dir = _verify_prepared_state(args, configuration)
    learner_dir = run_dir / "sure_head_only_w1k"
    if learner_dir.exists():
        raise FileExistsError(f"Refusing to overwrite learner output: {learner_dir}")
    views, bundle_audit, generator_audit = load_atomic_bundle(
        Path(args.training_bundle).resolve(),
        Path(args.generator_receipt).resolve(),
        configuration,
    )
    generator_model_audit = validate_generator_base_model(
        generator_audit, args.model_path
    )
    rwku_experiment._write_state(
        state_args,
        "TRAINING",
        configuration_path=str(configuration_path),
        configuration_sha256=artifact_access.sha256_file(configuration_path),
        official_evaluation_opened=False,
    )
    learner_dir.mkdir(parents=True)

    started = time.perf_counter()
    provisional = optimization_namespace(configuration, prompt_count=len(views))
    stage1_scales, solver_margins, rank_ladder = joint.validate_args(provisional)
    gagd.set_seed(provisional.seed)
    gagd.require_cuda_if_needed(provisional.device_map)
    model_args = argparse.Namespace(
        model_path=args.model_path,
        dtype=provisional.dtype,
        device_map=provisional.device_map,
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(model_args, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    prompt_records = compile_prompt_records(
        views,
        tokenizer,
        neutral_target=str(configuration["neutral_target"]),
    )
    optimization = optimization_namespace(
        configuration, prompt_count=len(prompt_records)
    )

    identity = wikipedia.model_identity(model, tokenizer, args.model_path)
    original_output = model.get_output_embeddings()
    if original_output is None:
        raise RuntimeError("Base model has no output head")
    hidden_size = int(original_output.weight.shape[1])
    (
        utility_second_moment,
        utility_hidden,
        utility_logsumexp,
        utility_metadata,
    ) = learner.load_utility_cache(
        Path(args.utility_cache).resolve(),
        expected_sample_size=optimization.utility_sample_size,
        expected_prompt_count=optimization.utility_prompt_count,
        expected_hidden_size=hidden_size,
        expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"],
    )
    utility_audit = validate_w1k_utility_metadata(utility_metadata, configuration)

    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tokenizer)
    frozen_versions = _parameter_versions_except_output(model, output_layer)
    sensitive_cases = core.expand_sensitive_cases(
        prompt_records,
        tokenizer,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    neutral_cases = core.expand_sensitive_cases(
        prompt_records,
        tokenizer,
        sensitive_field="target_reference",
        llama_like=llama_like,
    )
    sensitive_ids = core.official_target_ids(
        tokenizer, sensitive_cases, llama_like=llama_like, device=device
    ).detach()
    neutral_ids = core.official_target_ids(
        tokenizer, neutral_cases, llama_like=llama_like, device=device
    ).detach()
    selected_ids, row_filter_report = select_edit_row_ids(
        tokenizer,
        sensitive_cases,
        sensitive_ids,
        neutral_ids,
        prompt_count=len(prompt_records),
        configuration=configuration,
    )
    selected_tensor = torch.tensor(
        selected_ids, device=output_layer.weight.device, dtype=torch.long
    )
    base_rows = output_layer.weight.index_select(0, selected_tensor).detach().float()
    selected_row_report = {
        "editable_parameters": configuration["editable_parameters"],
        "selected_row_count": len(selected_ids),
        "selected_row_ids": selected_ids,
        "selected_row_decodings": row_filter_report["selected_row_decodings"],
        "sensitive_answer_content_rows": row_filter_report[
            "selected_sensitive_row_ids"
        ],
        "sensitive_answer_filtered_rows": row_filter_report[
            "rejected_sensitive_row_ids"
        ],
        "neutral_target_rows": row_filter_report["neutral_row_ids"],
        "sensitive_row_filter": row_filter_report,
        "subject_name_rows_added": False,
        "input_embedding_rows_editable": False,
        "transformer_parameters_editable": False,
    }
    core.write_json(learner_dir / "selected_row_report.json", selected_row_report)

    utility_probabilities = learner.selected_base_probabilities(
        output_layer,
        selected_ids,
        utility_hidden,
        utility_logsumexp,
        device=device,
        batch_size=optimization.utility_eval_batch_size,
    )
    (
        train_indices,
        guard_indices,
        utility_pool_report,
    ) = learner.build_disjoint_token_conditioned_utility_pools(
        selected_base_probabilities=utility_probabilities,
        selected_ids=selected_ids,
        topk_per_row=optimization.utility_token_topk_per_row,
        uniform_prompt_count=optimization.utility_uniform_prompt_count,
        split_seed=optimization.utility_pool_seed,
    )
    utility_train_hidden = (
        utility_hidden.index_select(0, train_indices).contiguous().to(device)
    )
    utility_train_probabilities = (
        utility_probabilities.index_select(0, train_indices).contiguous().to(device)
    )
    utility_guard_hidden = utility_hidden.index_select(0, guard_indices).contiguous()
    utility_guard_probabilities = utility_probabilities.index_select(
        0, guard_indices
    ).contiguous()
    utility_pool_report["rwku_h_w1k_audit"] = utility_audit
    core.write_json(learner_dir / "utility_pool_report.json", utility_pool_report)

    base_sensitive_logits = learner.cache_logits_preserving_dtype(
        model,
        tokenizer,
        sensitive_cases,
        device,
        optimization.cache_batch_size,
    )
    base_neutral_logits = learner.cache_logits_preserving_dtype(
        model, tokenizer, neutral_cases, device, optimization.cache_batch_size
    )
    sensitive_hidden = core.forward_last_hidden(
        model, tokenizer, sensitive_cases, device, optimization.cache_batch_size
    ).float()
    neutral_hidden = core.forward_last_hidden(
        model, tokenizer, neutral_cases, device, optimization.cache_batch_size
    ).float()
    sensitive_positions = exact.record_positions(sensitive_cases, device=device)
    neutral_positions = exact.record_positions(neutral_cases, device=device)
    sensitive_cache = exact.build_sequence_cache(
        base_sensitive_logits,
        sensitive_hidden,
        sensitive_ids,
        sensitive_positions,
        selected_ids,
        record_count=len(prompt_records),
        device=device,
    )
    neutral_cache = exact.build_sequence_cache(
        base_neutral_logits,
        neutral_hidden,
        neutral_ids,
        neutral_positions,
        selected_ids,
        record_count=len(prompt_records),
        device=device,
    )
    zero = torch.zeros(
        (len(selected_ids), hidden_size), device=device, dtype=torch.float32
    )
    base_sensitive_nll = exact.exact_sequence_record_nll(sensitive_cache, zero).detach()
    base_neutral_nll = exact.exact_sequence_record_nll(neutral_cache, zero).detach()
    masks = joint.prompt_kind_masks(prompt_records, device=device)
    base_report = materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=optimization.required_pairwise_margin,
    )
    core.write_json(learner_dir / "base_atomic_view_report.json", base_report)

    stage1_bases, stage1_basis_report = joint.build_joint_bases(
        sensitive_hidden,
        sensitive_ids,
        neutral_hidden,
        neutral_ids,
        utility_second_moment,
        requested_ids=selected_ids,
        rank_cap=optimization.stage1_rank,
        relative_eps=optimization.contrastive_eps,
        constraint_context_weight=optimization.constraint_context_weight,
    )
    core.write_json(learner_dir / "stage1_basis_report.json", stage1_basis_report)
    trained_stage1 = joint.optimize_stage1(
        optimization,
        selected_ids,
        stage1_bases,
        sensitive_cache,
        neutral_cache,
        base_sensitive_nll,
        base_neutral_nll,
        masks,
        utility_train_hidden,
        utility_train_probabilities,
        learner_dir,
        device=device,
    )
    stage1_delta, stage1_selected, stage1_reports = joint.choose_stage1_delta(
        optimization,
        trained_stage1,
        stage1_scales,
        sensitive_cache,
        neutral_cache,
        prompt_records,
        utility_guard_hidden,
        utility_guard_probabilities,
        device=device,
    )
    torch.save(
        {"row_ids": selected_ids, "delta": stage1_delta.detach().cpu()},
        learner_dir / "stage1_delta.pt",
    )
    core.write_json(learner_dir / "stage1_scale_reports.json", stage1_reports)
    core.write_json(learner_dir / "stage1_selected_report.json", stage1_selected)

    with learner.temporary_materialized_output_delta(
        output_layer, selected_ids, stage1_delta
    ):
        actual_stage1_delta = learner.actual_selected_delta(
            output_layer, selected_ids, base_rows
        )
        stage1_materialized = materialized_atomic_report(
            model,
            tokenizer,
            prompt_records,
            device,
            llama_like=llama_like,
            required_margin=optimization.required_pairwise_margin,
        )
    joint.add_utility_report(
        stage1_materialized,
        actual_stage1_delta,
        utility_guard_hidden,
        utility_guard_probabilities,
        optimization,
        device=device,
    )
    core.write_json(
        learner_dir / "stage1_materialized_report.json", stage1_materialized
    )

    final_delta: torch.Tensor | None = None
    selected_metadata: Dict[str, Any]
    if atomic_candidate_feasible(stage1_materialized):
        final_delta = actual_stage1_delta
        selected_metadata = {
            "selection_mode": "rank4_stage1_materialized_all_atomic_views_safe",
            "stage1": stage1_materialized,
        }
    else:
        failure_positions = [
            int(value)
            for value in stage1_materialized["pairwise_margin_failure_positions"]
        ]
        if not failure_positions:
            core.write_json(
                learner_dir / "infeasible.json",
                {
                    "configuration_id": configuration["configuration_id"],
                    "stage1": stage1_materialized,
                    "reason": "behavioral constraints passed but a utility guard failed",
                    "official_rwku_records_accessed": False,
                },
            )
            raise RuntimeError("RWKU-H-W1K Stage 1 is behaviorally complete but unsafe")
        failure_set = set(failure_positions)
        sensitive_failure_mask = torch.tensor(
            [case.record_position in failure_set for case in sensitive_cases],
            device=device,
            dtype=torch.bool,
        )
        neutral_failure_mask = torch.tensor(
            [case.record_position in failure_set for case in neutral_cases],
            device=device,
            dtype=torch.bool,
        )
        failure_token_ids = set(
            int(value)
            for value in sensitive_ids[sensitive_failure_mask].detach().cpu().tolist()
        ) | set(
            int(value)
            for value in neutral_ids[neutral_failure_mask].detach().cpu().tolist()
        )
        active_ids = sorted(failure_token_ids & set(selected_ids))
        if not active_ids:
            raise RuntimeError(
                "Residual failures have no editable content/neutral rows"
            )
        attempts: List[Dict[str, Any]] = []
        chosen: Dict[str, Any] | None = None
        for rank in rank_ladder:
            bases, basis_report = joint.build_joint_bases(
                sensitive_hidden,
                sensitive_ids,
                neutral_hidden,
                neutral_ids,
                utility_second_moment,
                requested_ids=active_ids,
                rank_cap=rank,
                relative_eps=optimization.contrastive_eps,
                constraint_context_weight=optimization.constraint_context_weight,
            )
            core.write_json(
                learner_dir / f"stage2_rank{rank}_basis_report.json", basis_report
            )
            for solver_target in solver_margins:
                residual, history, solver_report = joint.solve_residual(
                    optimization,
                    rank=rank,
                    solver_target=solver_target,
                    row_bases=bases,
                    active_ids=active_ids,
                    selected_ids=selected_ids,
                    stage1_delta=actual_stage1_delta,
                    true_cache=sensitive_cache,
                    reference_cache=neutral_cache,
                    prompt_records=prompt_records,
                    utility_hidden=utility_train_hidden,
                    utility_probabilities=utility_train_probabilities,
                )
                tag = str(solver_target).replace(".", "p")
                core.write_json(
                    learner_dir / f"stage2_rank{rank}_margin{tag}_solver_history.json",
                    history,
                )
                combined = learner.total_delta_with_residual(
                    actual_stage1_delta, selected_ids, residual, active_ids
                )
                if not bool(solver_report["continuous_feasible"]):
                    materialized = {
                        "rank": rank,
                        "solver_target": float(solver_target),
                        "continuous_feasible": False,
                        "feasible": False,
                        "materialization_skipped": True,
                        "continuous_minimum_separation": solver_report[
                            "minimum_overall_separation"
                        ],
                        "continuous_total_delta_norm": solver_report[
                            "total_delta_norm"
                        ],
                    }
                else:
                    with learner.temporary_materialized_output_delta(
                        output_layer, selected_ids, combined
                    ):
                        actual_combined = learner.actual_selected_delta(
                            output_layer, selected_ids, base_rows
                        )
                        materialized = materialized_atomic_report(
                            model,
                            tokenizer,
                            prompt_records,
                            device,
                            llama_like=llama_like,
                            required_margin=optimization.required_pairwise_margin,
                        )
                    joint.add_utility_report(
                        materialized,
                        actual_combined,
                        utility_guard_hidden,
                        utility_guard_probabilities,
                        optimization,
                        device=device,
                    )
                    materialized.update(
                        {
                            "rank": rank,
                            "solver_target": float(solver_target),
                            "residual_delta_norm": float(residual.norm().cpu()),
                            "continuous_feasible": True,
                        }
                    )
                    materialized["feasible"] = atomic_candidate_feasible(materialized)
                core.write_json(
                    learner_dir
                    / f"stage2_rank{rank}_margin{tag}_materialized_report.json",
                    materialized,
                )
                attempt = {
                    "rank": rank,
                    "solver_target": float(solver_target),
                    "solver": solver_report,
                    "materialized": materialized,
                }
                attempts.append(attempt)
                if materialized["feasible"]:
                    chosen = {**attempt, "delta": actual_combined.detach()}
                    break
            if chosen is not None:
                break
        core.write_json(learner_dir / "stage2_attempts.json", attempts)
        if chosen is None:
            core.write_json(
                learner_dir / "infeasible.json",
                {
                    "configuration_id": configuration["configuration_id"],
                    "stage1": stage1_materialized,
                    "active_row_ids": active_ids,
                    "stage2_attempts": attempts,
                    "reason": "no BF16-safe atomic-view checkpoint passed utility guards",
                    "official_rwku_records_accessed": False,
                },
            )
            raise RuntimeError("RWKU-H-W1K found no feasible head-only checkpoint")
        final_delta = chosen["delta"]
        selected_metadata = {
            "selection_mode": "minimum_utility_residual_all_atomic_views_safe",
            "rank": chosen["rank"],
            "solver_target": chosen["solver_target"],
            "solver": chosen["solver"],
            "materialized": chosen["materialized"],
        }

    if final_delta is None:
        raise AssertionError("RWKU-H-W1K selection produced no final delta")
    core.materialize_output_delta(output_layer, selected_ids, final_delta)
    actual_final_delta = learner.actual_selected_delta(
        output_layer, selected_ids, base_rows
    )
    final_report = materialized_atomic_report(
        model,
        tokenizer,
        prompt_records,
        device,
        llama_like=llama_like,
        required_margin=optimization.required_pairwise_margin,
    )
    joint.add_utility_report(
        final_report,
        actual_final_delta,
        utility_guard_hidden,
        utility_guard_probabilities,
        optimization,
        device=device,
    )
    if not atomic_candidate_feasible(final_report):
        raise RuntimeError("Final RWKU-H-W1K materialization failed its locked gate")
    assert_head_only_versions(model, output_layer, frozen_versions)

    checkpoint_path = learner_dir / "checkpoint"
    learner.save_checkpoint(model, tokenizer, checkpoint_path)
    delta_path = learner_dir / "final_total_delta.pt"
    torch.save(
        {"row_ids": selected_ids, "delta": actual_final_delta.detach().cpu()},
        delta_path,
    )
    core.write_json(learner_dir / "final_atomic_view_report.json", final_report)
    core.write_json(
        learner_dir / "config_used.json",
        {
            "configuration": configuration,
            "configuration_sha256": artifact_access.sha256_file(configuration_path),
            "bundle_audit": bundle_audit,
            "generator_audit": {
                "status": generator_audit["status"],
                "accepted_fact_count": generator_audit["accepted_fact_count"],
                "official_rwku_records_accessed": generator_audit[
                    "official_rwku_records_accessed"
                ],
                "base_model": generator_model_audit,
            },
            "utility_audit": utility_audit,
            "selected_rows": selected_row_report,
            "selected": selected_metadata,
            "final": final_report,
        },
    )
    before_failures = len(base_report["pairwise_margin_failure_positions"])
    after_failures = len(final_report["pairwise_margin_failure_positions"])
    training_report = {
        "schema_version": "rwku_sure_head_only_w1k_training_report_v1",
        "protocol_label": artifact_access.TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": PROTOCOL_STATUS,
        "method": configuration["method"],
        "configuration_id": configuration["configuration_id"],
        "development_only": True,
        "target": {
            "seed": 0,
            "entity": configuration["target_entity"],
            "entity_id": configuration["target_entity_id"],
        },
        "objective": "bounded_GA(atomic_sensitive_answer)+bounded_GD(Unknown)+external_Wikipedia_KL",
        "atomic_bundle": bundle_audit,
        "atomic_generator_base_model": generator_model_audit,
        "utility_cache": utility_audit,
        "head_only_invariants": {
            "transformer_frozen": True,
            "input_embeddings_frozen_and_untied": True,
            "selected_lm_head_row_count": len(selected_ids),
            "selected_lm_head_row_ids": selected_ids,
            "all_non_output_parameter_versions_unchanged": True,
            "rank4_stage1": True,
            "residual_rank_ladder": list(rank_ladder),
        },
        "repair": {
            "active_prompt_instances_before": before_failures,
            "active_prompt_instances_after": after_failures,
            "selection": selected_metadata,
        },
        "base_training_view_report": base_report,
        "final_training_view_report": final_report,
        "training_seconds": time.perf_counter() - started,
        "official_rwku_records_accessed": False,
        "official_rwku_records_used_for_training_or_selection": False,
        "final_evaluation_used_for_training_or_selection": False,
    }
    training_report_path = learner_dir / "training_report.json"
    core.write_json(training_report_path, training_report)

    receipt_path = run_dir / "checkpoint_receipt.json"
    method_configuration = {
        "method": configuration["method"],
        "configuration_id": configuration["configuration_id"],
        "configuration_path": str(configuration_path),
        "configuration_sha256": artifact_access.sha256_file(configuration_path),
        "training_report_path": str(training_report_path.resolve()),
        "training_report_sha256": artifact_access.sha256_file(training_report_path),
        "editable_parameters": configuration["editable_parameters"],
        "selected_lm_head_row_count": len(selected_ids),
        "stage1_rank": optimization.stage1_rank,
        "utility_document_count": utility_audit["document_count"],
        "utility_predictor_count": utility_audit["predictor_count"],
        "neutral_target": configuration["neutral_target"],
        "official_rwku_records_used_for_selection": False,
    }
    implementation_files = [
        SCRIPT_PATH,
        PROJECT_ROOT / "scripts" / "build_rwku_entity_facts.py",
        PROJECT_ROOT / "scripts" / "gagd_compare.py",
        PROJECT_ROOT / "scripts" / "mcf_zero_unlearn_official_eval.py",
        PROJECT_ROOT / "scripts" / "rwku_artifact_access.py",
        PROJECT_ROOT / "scripts" / "rwku_checkpoint_receipt.py",
        PROJECT_ROOT / "scripts" / "rwku_eval.py",
        PROJECT_ROOT / "scripts" / "rwku_experiment.py",
        PROJECT_ROOT / "scripts" / "sure_canonical_core.py",
        PROJECT_ROOT / "scripts" / "sure_mcf_direct_fs_repair.py",
        PROJECT_ROOT / "scripts" / "sure_mcf_target_aware_two_stage.py",
        PROJECT_ROOT / "scripts" / "sure_minimal_two_stage.py",
        PROJECT_ROOT / "scripts" / "build_sure_wikipedia_stats.py",
    ]
    receipt = checkpoint_receipt.create_checkpoint_receipt(
        destination=receipt_path,
        experiment_id=str(args.experiment_id),
        protocol_label=artifact_access.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=PROTOCOL_STATUS,
        target_entity=str(configuration["target_entity"]),
        target_entity_id=str(configuration["target_entity_id"]),
        base_model_identity=rwku_experiment.local_model_identity(args.model_path),
        base_model_revision=str(args.model_revision),
        tokenizer_identity={
            "name_or_path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
            "tokenizer_probe_sha256": identity["tokenizer_probe_sha256"],
        },
        checkpoint_paths=[checkpoint_path],
        training_bundle_path=Path(args.training_bundle).resolve(),
        optimization_protection_path=None,
        mcf_retain_optimization_paths=[],
        mcf_repair_gate_paths=[],
        matched_protection_train_path=None,
        matched_protection_gate_path=None,
        method_configuration=method_configuration,
        implementation_files=implementation_files,
        sampler_provenance={
            "atomic_view_order_sha256": bundle_audit["view_ids_sha256"],
            "utility_pool_seed": optimization.utility_pool_seed,
            "training_seed": optimization.seed,
            "official_rwku_records_accessed": False,
        },
        generator_receipt_path=Path(args.generator_receipt).resolve(),
        official_locked_eval_path=run_dir / "official_locked_eval.json",
        confirmatory=False,
        additional_artifact_paths={
            "locked_configuration": configuration_path,
            "utility_cache": Path(args.utility_cache).resolve(),
            "training_report": training_report_path,
            "final_sparse_delta": delta_path,
        },
    )
    rwku_experiment._write_state(
        state_args,
        "CHECKPOINT_FROZEN",
        checkpoint_receipt=str(receipt_path.resolve()),
        checkpoint_receipt_sha256=receipt["receipt_sha256"],
        official_evaluation_opened=False,
        head_only_feasible=True,
        training_report=str(training_report_path.resolve()),
    )
    print(f"RWKU-H-W1K head-only checkpoint frozen: {checkpoint_path}")
    print(f"Atomic direct success: {final_report['FS']}")
    print(f"Other atomic-view success: {final_report['generated_subject_FS']}")
    print(f"Wikipedia utility mean KL: {final_report['utility_kl_mean']}")
    print("Official RWKU evaluation remains unopened; use the staged evaluator next.")


if __name__ == "__main__":
    main()
