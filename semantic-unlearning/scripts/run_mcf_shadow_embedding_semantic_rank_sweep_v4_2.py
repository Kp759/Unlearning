#!/usr/bin/env python3
"""Run V4.2's training-only nested shared semantic-rank sweep.

The failed V4.0 and V4.1 runs are mandatory hash-bound inputs.  V4.1 showed
that a four-dimensional shared relation subspace underfit both fit and
development cells.  V4.2 keeps the passive shadow marker, relation tying,
large negative bank, and Base main path fixed while sweeping the nested ranks
4, 8, 16, and the available full relation-contrast rank (capped at 32).

V4.1 development prompts are now consumed fit data.  A newly authored family
selects the smallest passing rank.  The still-unopened V4.1 certification
family is evaluated once for that selected rank after every optimizer is
gone.  A passing router is frozen for a separate actuator experiment; this
process constructs no actuator and has no official-evaluation interface.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_embedding_keyed_neuron_core as legacy_core
import mcf_embedding_keyed_neuron_erasure as legacy
import mcf_shadow_embedding_semantic_rank_sweep_v4_2_core as core
import mcf_shadow_embedding_semantic_router_core as v41_core
import mcf_shadow_relation_prompts as relation_prompts
import mcf_sure_directional_emb_lm_stage1 as directional
import mcf_sure_subject_directional_emb_stage1 as subject_writer
import mcf_synthetic_paraphrase_templates as synthetic
import scoped_span_edit as scoped
import train_mcf_shadow_embedding_router_v4 as v4_train
import train_mcf_shadow_embedding_semantic_router_v4_1 as v41_train


PROTOCOL = core.PROTOCOL


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--stage1-state", required=True)
    parser.add_argument("--stage1-report", required=True)
    parser.add_argument("--stage1-writer-log", required=True)
    parser.add_argument("--clean-stage1-portability-preflight", required=True)
    parser.add_argument("--clean-stage1-acceptance", required=True)
    parser.add_argument("--failed-v4-run-dir", required=True)
    parser.add_argument("--failed-v4-1-run-dir", required=True)
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=4096)
    parser.add_argument("--corpus-prefixes", type=int, default=256)
    parser.add_argument("--marker-rms-threshold", type=float, default=1e-6)
    parser.add_argument("--semantic-router-steps", type=int, default=300)
    parser.add_argument("--semantic-router-lr", type=float, default=0.01)
    parser.add_argument("--semantic-router-weight-decay", type=float, default=0.01)
    parser.add_argument("--semantic-router-coefficient-norm-cap", type=float, default=4.0)
    parser.add_argument("--semantic-selection-check-every", type=int, default=10)
    parser.add_argument("--semantic-selection-patience", type=int, default=10)
    parser.add_argument("--semantic-training-positive-floor", type=float, default=1.0)
    parser.add_argument("--semantic-training-negative-ceiling", type=float, default=-1.0)
    parser.add_argument("--semantic-certificate-positive-floor", type=float, default=0.25)
    parser.add_argument("--semantic-certificate-negative-ceiling", type=float, default=-0.25)
    parser.add_argument("--semantic-tail-k", type=int, default=4)
    parser.add_argument("--fit-positive-variants-per-old-family", type=int, default=8)
    parser.add_argument("--fit-wrong-relations-per-old-family", type=int, default=64)
    parser.add_argument("--consumed-v4-1-positive-variants", type=int, default=4)
    parser.add_argument("--consumed-v4-1-wrong-relations", type=int, default=32)
    parser.add_argument("--development-positive-variants", type=int, default=4)
    parser.add_argument("--development-wrong-relations", type=int, default=64)
    parser.add_argument("--certification-positive-variants", type=int, default=4)
    parser.add_argument("--certification-wrong-relations", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    counts = (
        args.forget_num,
        args.frequency_docs,
        args.corpus_prefixes,
        args.semantic_router_steps,
        args.semantic_selection_check_every,
        args.semantic_selection_patience,
        args.semantic_tail_k,
        args.fit_positive_variants_per_old_family,
        args.fit_wrong_relations_per_old_family,
        args.consumed_v4_1_positive_variants,
        args.consumed_v4_1_wrong_relations,
        args.development_positive_variants,
        args.development_wrong_relations,
        args.certification_positive_variants,
        args.certification_wrong_relations,
        args.batch_size,
    )
    if any(int(value) <= 0 for value in counts):
        parser.error("all count and step arguments must be positive")
    if int(args.frequency_doc_start) < 20:
        parser.error("documents 0:20 remain reserved for official PPL")
    if not float(args.marker_rms_threshold) > 0:
        parser.error("marker RMS threshold must be strictly positive")
    return args


def validate_failed_v4_1(root: Path) -> Dict[str, Any]:
    method = root / "method"
    paths = {
        "completion": method / "completion.json",
        "capacity": method / "semantic_capacity_report.json",
        "training_log": method / "semantic_router_training_log.json",
        "preflight": method / "factorized_router_preflight.json",
        "marker": method / "marker_preflight.json",
        "firewall": method / "training_firewall_receipt.json",
        "prompt_manifest": method / "development_prompt_manifest.json",
        "registry": root / "protocol" / "experiment_registry.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = v4_train.load_json(paths["completion"])
    capacity = v4_train.load_json(paths["capacity"])
    training_log = v4_train.load_json(paths["training_log"])
    preflight = v4_train.load_json(paths["preflight"])
    marker = v4_train.load_json(paths["marker"])
    firewall = v4_train.load_json(paths["firewall"])
    registry = v4_train.load_json(paths["registry"])
    selection = training_log.get("selection", {})
    fit_semantic = preflight.get("fit_semantic", {})
    development_semantic = preflight.get("development_semantic", {})
    forbidden_outputs = [
        method / "factorized_router_certificate.json",
        method / "actuator_training.json",
        method / "v4_1_factorized_candidate.pt",
    ]
    checks = {
        "source_protocol": completion.get("protocol") == v41_core.PROTOCOL,
        "source_failed_development": completion.get("passed") is False
        and completion.get("stage") == "fit_or_development_router_preflight"
        and completion.get("candidate_saved") is False,
        "rank_four": int(capacity.get("shared_rank", -1)) == 4
        and int(capacity.get("rejected_v4_independent_parameters", -1)) == 153600
        and int(capacity.get("fit_negative_cells_per_record_min", -1)) >= 100,
        "selection_failed": selection.get("selection_split") == "v4_1_development"
        and selection.get("certification_split_visible_to_selection") is False
        and int(selection.get("best_optimizer_step", -1)) == 1
        and int(selection.get("stopped_optimizer_step", -1)) == 100,
        "hard_certificates_failed": preflight.get("passed") is False
        and (
            int(fit_semantic.get("positive_failures", 0))
            + int(fit_semantic.get("negative_failures", 0))
            > 0
        )
        and (
            int(development_semantic.get("positive_failures", 0))
            + int(development_semantic.get("negative_failures", 0))
            > 0
        ),
        "marker_still_structural": marker.get("passed") is True
        and float(marker.get("writer_off_marker_rms", float("nan"))) == 0.0,
        "certification_unopened": preflight.get("certification_prompts_opened") is False
        and all(not path.exists() for path in forbidden_outputs),
        "official_closed": int(completion.get("official_evaluation_prompts_seen", -1))
        == 0
        and int(firewall.get("official_evaluation_prompts_seen", -1)) == 0
        and registry.get("official_evaluation_prohibited") is True,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError("failed V4.1 lineage mismatch: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "kind": "mcf_shadow_semantic_v4_1_rejection_import",
        "source_run_dir": str(root.resolve()),
        "source_artifacts": {
            name: {"path": str(path.resolve()), "sha256": v4_train.sha256_file(path)}
            for name, path in paths.items()
        },
        "diagnosis": {
            "shared_rank": 4,
            "best_optimizer_step": 1,
            "stopped_optimizer_step": 100,
            "fit_failures_at_selected_checkpoint": int(
                fit_semantic.get("positive_failures", 0)
                + fit_semantic.get("negative_failures", 0)
            ),
            "development_failures_at_selected_checkpoint": int(
                development_semantic.get("positive_failures", 0)
                + development_semantic.get("negative_failures", 0)
            ),
            "conclusion": "shared_rank_four_underfit_relation_semantics",
        },
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    if registry.get("protocol") != PROTOCOL or int(registry.get("schema_version", -1)) != 1:
        raise RuntimeError("V4.2 experiment registry is stale")
    architecture = registry.get("architecture")
    expected_architecture = {
        "base_embedding_main_path": True,
        "v6_2_delta_shadow_path_only": True,
        "shadow_gradients": False,
        "exact_complete_subject_key": True,
        "marker_gate": "shadow_minus_base_rms",
        "semantic_gate": "frozen_shared_rank_relation_router",
        "rank_arms": [4, 8, 16, "full_relation_contrast_rank_capped_at_32"],
        "nested_shared_basis_prefixes": True,
        "semantic_basis_trainable": False,
        "relation_tied_coefficients": True,
        "record_specific_hidden_vectors": False,
        "actuator_constructed": False,
        "layer": int(args.layer),
        "marker_rms_threshold": float(args.marker_rms_threshold),
        "base_embedding_mutation": False,
        "lm_head_mutation": False,
    }
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != value for key, value in expected_architecture.items()
    ):
        raise RuntimeError("V4.2 registry architecture mismatch")
    locked = registry.get("locked_training")
    expected_locked = {
        "semantic_router_optimizer_updates_max_per_arm": int(args.semantic_router_steps),
        "semantic_router_lr": float(args.semantic_router_lr),
        "semantic_router_weight_decay": float(args.semantic_router_weight_decay),
        "semantic_router_coefficient_norm_cap": float(
            args.semantic_router_coefficient_norm_cap
        ),
        "semantic_selection_check_every": int(args.semantic_selection_check_every),
        "semantic_selection_patience_checks": int(args.semantic_selection_patience),
        "semantic_training_positive_floor": float(
            args.semantic_training_positive_floor
        ),
        "semantic_training_negative_ceiling": float(
            args.semantic_training_negative_ceiling
        ),
        "semantic_certificate_positive_floor": float(
            args.semantic_certificate_positive_floor
        ),
        "semantic_certificate_negative_ceiling": float(
            args.semantic_certificate_negative_ceiling
        ),
        "semantic_tail_k": int(args.semantic_tail_k),
        "marker_rms_threshold": float(args.marker_rms_threshold),
        "fit_positive_variants_per_old_family": int(
            args.fit_positive_variants_per_old_family
        ),
        "fit_wrong_relations_per_old_family": int(
            args.fit_wrong_relations_per_old_family
        ),
        "consumed_v4_1_positive_variants": int(
            args.consumed_v4_1_positive_variants
        ),
        "consumed_v4_1_wrong_relations": int(
            args.consumed_v4_1_wrong_relations
        ),
        "development_positive_variants": int(
            args.development_positive_variants
        ),
        "development_wrong_relations": int(args.development_wrong_relations),
        "certification_positive_variants": int(
            args.certification_positive_variants
        ),
        "certification_wrong_relations": int(args.certification_wrong_relations),
    }
    if not isinstance(locked, Mapping) or any(
        locked.get(key) != value for key, value in expected_locked.items()
    ):
        raise RuntimeError("V4.2 registry optimizer mismatch")
    prompt_policy = registry.get("development_prompt_policy")
    expected_policy = {
        "v4_1_consumed_development_family": "fit",
        "fit_wrong_relation_negatives_per_record_total": (
            2 * int(args.fit_wrong_relations_per_old_family)
            + int(args.consumed_v4_1_wrong_relations)
        ),
        "v4_2_development_family": "rank_and_checkpoint_selection_no_gradients",
        "v4_1_certification_family": (
            "opened_once_for_selected_rank_after_all_optimizers"
        ),
    }
    if not isinstance(prompt_policy, Mapping) or any(
        prompt_policy.get(key) != value for key, value in expected_policy.items()
    ):
        raise RuntimeError("V4.2 registry split policy mismatch")
    if (
        registry.get("source_v4_0_rejection_required") is not True
        or registry.get("source_v4_1_rejection_required") is not True
        or int(registry.get("seed", -1)) != int(args.seed)
        or int(registry.get("forget_num", -1)) != int(args.forget_num)
        or registry.get("official_evaluation_prohibited") is not True
        or registry.get("official_policy", {}).get(
            "seed_1_v3_6_2_official_result_consumed"
        )
        is not True
    ):
        raise RuntimeError("V4.2 must prohibit official evaluation")


def write_failure(output: Path, *, stage: str, certification_opened: bool) -> None:
    v4_train.write_json(
        output / "completion.json",
        {
            "schema_version": 1,
            "kind": "mcf_shadow_semantic_rank_sweep_v4_2_completion",
            "protocol": PROTOCOL,
            "passed": False,
            "stage": str(stage),
            "certified_router_saved": False,
            "actuator_optimizer_constructed": False,
            "certification_open_count": 1 if certification_opened else 0,
            "optimizer_steps_after_certification_open": 0,
            "official_evaluation_prompts_seen": 0,
        },
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    v4_train.validate_environment_firewall()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

    failed_v4 = v41_train.validate_failed_v4(Path(args.failed_v4_run_dir).resolve())
    failed_v41 = validate_failed_v4_1(Path(args.failed_v4_1_run_dir).resolve())
    v4_train.write_json(output / "frozen_v4_rejection_import.json", failed_v4)
    v4_train.write_json(output / "frozen_v4_1_rejection_import.json", failed_v41)

    paths = {
        "training_visible": Path(args.training_visible_path).resolve(),
        "split_manifest": Path(args.split_manifest).resolve(),
        "context_manifest": Path(args.context_manifest).resolve(),
        "stage1_state": Path(args.stage1_state).resolve(),
        "stage1_report": Path(args.stage1_report).resolve(),
        "stage1_writer_log": Path(args.stage1_writer_log).resolve(),
        "clean_stage1_portability": Path(args.clean_stage1_portability_preflight).resolve(),
        "clean_stage1_acceptance": Path(args.clean_stage1_acceptance).resolve(),
        "experiment_registry": Path(args.experiment_registry).resolve(),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = v4_train.load_json(paths["experiment_registry"])
    validate_registry(registry, args)
    locked_records, split_manifest = directional.validate_locked(
        paths["training_visible"],
        paths["split_manifest"],
        int(args.seed),
        int(args.forget_num),
    )
    if split_manifest.get("protocol") != locked_split.PROTOCOL:
        raise RuntimeError("V4.2 requires the locked direct-only split")
    locked_split.assert_direct_only_training_view(locked_records)
    records = compositional._record_views(locked_records)
    context_manifest = v4_train.load_json(paths["context_manifest"])
    stage1_state = torch.load(paths["stage1_state"], map_location="cpu", weights_only=False)
    if not isinstance(stage1_state, Mapping):
        raise RuntimeError("V6.2 writer state must be a mapping")
    stage1_report = v4_train.load_json(paths["stage1_report"])
    legacy._validate_firewall(context_manifest, stage1_state)
    clean_writer = legacy._validate_clean_stage1_lineage(
        context_manifest,
        stage1_state,
        stage1_report,
        paths["context_manifest"],
        paths["stage1_writer_log"],
    )
    if str(Path(args.model_path).resolve()) != str(clean_writer["base_model_path"]):
        raise RuntimeError("V4.2 Base model differs from V6.2 lineage")
    if v4_train.sha256_file(paths["context_manifest"]) != str(
        stage1_state["context_manifest_sha256"]
    ):
        raise RuntimeError("V6.2 writer/context hashes differ")
    acceptance = v4_train.load_json(paths["clean_stage1_acceptance"])
    portability = v4_train.load_json(paths["clean_stage1_portability"])
    if (
        acceptance.get("passed") is not True
        or bool(acceptance.get("official_evaluation_opened"))
        or dict(acceptance.get("training_safe_portability", {})) != dict(portability)
    ):
        raise RuntimeError("clean V6.2 acceptance is invalid")
    context_sets = legacy._context_sets_by_case(context_manifest, records)
    case_ids = [int(record["case_id"]) for record in records]
    if [int(value) for value in stage1_state.get("case_ids", [])] != case_ids:
        raise RuntimeError("V6.2/V4.2 case order mismatch")

    source_hashes = {name: v4_train.sha256_file(path) for name, path in paths.items()}
    source_hashes["failed_v4_completion"] = failed_v4["source_artifacts"][
        "completion"
    ]["sha256"]
    source_hashes["failed_v4_1_completion"] = failed_v41["source_artifacts"][
        "completion"
    ]["sha256"]
    firewall = {
        "schema_version": 1,
        "kind": "mcf_shadow_semantic_rank_sweep_training_firewall",
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "forget_num": len(records),
        "source_hashes": source_hashes,
        "v4_1_development_role": "consumed_fit_data",
        "v4_2_development_role": "rank_and_checkpoint_selection_only",
        "v4_1_certification_role": "unopened_until_one_selected_rank",
        "seed_1_blind_official_reuse_prohibited": True,
        "official_evaluation_arguments_available": False,
        "official_evaluation_prompts_seen": 0,
    }
    v4_train.write_json(output / "training_firewall_receipt.json", firewall)

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    device = gagd.first_device(model)
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None or output_embedding is None:
        raise RuntimeError("model lacks input/output embeddings")
    layers = scoped.find_decoder_layers(model)
    if not 0 <= int(args.layer) < len(layers):
        raise RuntimeError("V4.2 layer is outside Base model")
    layer = layers[int(args.layer)]
    hidden_size = int(input_embedding.weight.shape[1])
    embedding_rows = [int(value) for value in stage1_state.get("selected_embedding_rows", [])]
    embedding_delta = stage1_state.get("embedding_delta")
    if not embedding_rows or not isinstance(embedding_delta, torch.Tensor) or embedding_delta.shape != (
        len(embedding_rows),
        hidden_size,
    ):
        raise RuntimeError("V6.2 embedding marker is missing or incompatible")
    if not math.isclose(
        float(compositional.frozen_transformer_fingerprint(model)),
        float(clean_writer["base_transformer_fingerprint"]),
        rel_tol=1e-12,
        abs_tol=1e-3,
    ):
        raise RuntimeError("V4.2 Transformer differs from V6.2 Base")
    selected_index = torch.tensor(
        embedding_rows, dtype=torch.long, device=input_embedding.weight.device
    )
    if v4_train.tensor_sha256(input_embedding.weight.index_select(0, selected_index)) != str(
        clean_writer["base_selected_embedding_rows_sha256"]
    ):
        raise RuntimeError("V4.2 Base embedding rows differ from V6.2")
    embedding_hash = v4_train.tensor_sha256(input_embedding.weight)
    lm_head_hash = v4_train.tensor_sha256(output_embedding.weight)
    writer = legacy_core.ToggleableEmbeddingDelta(
        input_embedding, embedding_rows, embedding_delta
    )
    writer.enabled = False
    subjects = [str(record["subject"]) for record in records]
    patterns = scoped.build_subject_patterns(tok, subjects)
    span_router = scoped.SpanGateRouter(
        input_embedding, patterns, subjects=subjects, model=model
    )

    documents = subject_writer.load_frequency_documents(
        args.wikidata_dir, int(args.frequency_doc_start), int(args.frequency_docs)
    )
    if not documents:
        raise RuntimeError("V4.2 could not load development Wikipedia documents")
    prefixes = synthetic.corpus_context_prefixes(
        documents, count=int(args.corpus_prefixes), seed=int(args.seed) + 4049
    )
    if len(prefixes) < int(args.corpus_prefixes):
        raise RuntimeError("V4.2 prefix bank is incomplete")

    writer_specs = v4_train.context_positive_specs(records, context_sets)
    old_calibration_positive = relation_prompts.build_positive_specs(
        records,
        split="calibration",
        corpus_prefixes=prefixes,
        variants_per_record=int(args.fit_positive_variants_per_old_family),
    )
    consumed_v4_heldout_positive = relation_prompts.build_positive_specs(
        records,
        split="heldout",
        corpus_prefixes=prefixes,
        variants_per_record=int(args.fit_positive_variants_per_old_family),
    )
    old_calibration_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="calibration",
        variants_per_record=int(args.fit_wrong_relations_per_old_family),
        corpus_prefixes=prefixes,
    )
    consumed_v4_heldout_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="heldout",
        variants_per_record=int(args.fit_wrong_relations_per_old_family),
        corpus_prefixes=prefixes,
    )
    consumed_v41_development_positive = relation_prompts.build_positive_specs(
        records,
        split="v4_1_development",
        variants_per_record=int(args.consumed_v4_1_positive_variants),
        corpus_prefixes=prefixes,
    )
    consumed_v41_development_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="v4_1_development",
        variants_per_record=int(args.consumed_v4_1_wrong_relations),
        corpus_prefixes=prefixes,
    )
    development_positive = relation_prompts.build_positive_specs(
        records,
        split="v4_2_development",
        variants_per_record=int(args.development_positive_variants),
        corpus_prefixes=prefixes,
    )
    development_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="v4_2_development",
        variants_per_record=int(args.development_wrong_relations),
        corpus_prefixes=prefixes,
    )
    certification_positive = relation_prompts.build_positive_specs(
        records,
        split="v4_1_certification",
        variants_per_record=int(args.certification_positive_variants),
        corpus_prefixes=prefixes,
    )
    certification_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="v4_1_certification",
        variants_per_record=int(args.certification_wrong_relations),
        corpus_prefixes=prefixes,
    )
    fit_specs = v4_train.unique_specs(
        [
            *writer_specs,
            *old_calibration_positive,
            *consumed_v4_heldout_positive,
            *old_calibration_negative,
            *consumed_v4_heldout_negative,
            *consumed_v41_development_positive,
            *consumed_v41_development_negative,
        ]
    )
    development_specs = v4_train.unique_specs(
        [*development_positive, *development_negative]
    )
    certification_specs = v4_train.unique_specs(
        [*certification_positive, *certification_negative]
    )
    prompt_sets = {
        "fit": {spec.prompt for spec in fit_specs},
        "development": {spec.prompt for spec in development_specs},
        "certification": {spec.prompt for spec in certification_specs},
    }
    if (
        prompt_sets["fit"] & prompt_sets["development"]
        or prompt_sets["fit"] & prompt_sets["certification"]
        or prompt_sets["development"] & prompt_sets["certification"]
    ):
        raise RuntimeError("V4.2 fit/development/certification prompts overlap")
    manifest = {
        "schema_version": 1,
        "kind": "mcf_v4_2_semantic_rank_sweep_prompt_manifest",
        "protocol": PROTOCOL,
        "counts": {
            "fit": len(fit_specs),
            "development": len(development_specs),
            "certification": len(certification_specs),
            "fit_positive": sum(spec.positive for spec in fit_specs),
            "fit_negative": sum(not spec.positive for spec in fit_specs),
            "development_positive": len(development_positive),
            "development_negative": len(development_negative),
            "certification_positive": len(certification_positive),
            "certification_negative": len(certification_negative),
        },
        "split_policy": {
            "fit": "optimizer_gradients",
            "development": "rank_checkpoint_selection_no_gradients",
            "certification": "one_selected_rank_once_after_all_optimizers",
        },
        "fit_specs": [spec.json() for spec in fit_specs],
        "development_specs": [spec.json() for spec in development_specs],
        "certification_specs": [spec.json() for spec in certification_specs],
        "official_evaluation_prompts_seen": 0,
    }
    v4_train.write_json(output / "development_prompt_manifest.json", manifest)

    print("\nStage 1: cache V4.2 fit and development Base/marker states")
    fit_delta, fit_base, fit_active = v4_train.capture_shadow_difference_last_states(
        model,
        tok,
        layer,
        writer,
        span_router,
        [spec.prompt for spec in fit_specs],
        device,
        batch_size=int(args.batch_size),
    )
    fit_labels = v4_train.labels_for_specs(fit_specs, fit_active, len(records))
    development_delta, development_base, development_active = (
        v4_train.capture_shadow_difference_last_states(
            model,
            tok,
            layer,
            writer,
            span_router,
            [spec.prompt for spec in development_specs],
            device,
            batch_size=int(args.batch_size),
        )
    )
    development_labels = v4_train.labels_for_specs(
        development_specs, development_active, len(records)
    )
    fit_marker = v41_core.marker_certificate(
        fit_delta.square().mean(dim=1).sqrt(),
        fit_active,
        fit_labels,
        threshold=float(args.marker_rms_threshold),
    )
    development_marker = v41_core.marker_certificate(
        development_delta.square().mean(dim=1).sqrt(),
        development_active,
        development_labels,
        threshold=float(args.marker_rms_threshold),
    )
    marker_report = {
        "fit": fit_marker,
        "development": development_marker,
        "writer_off_marker_rms": 0.0,
        "passed": bool(fit_marker["passed"] and development_marker["passed"]),
        "official_evaluation_prompts_seen": 0,
    }
    v4_train.write_json(output / "marker_preflight.json", marker_report)
    if not marker_report["passed"]:
        write_failure(output, stage="fit_or_development_marker", certification_opened=False)
        raise SystemExit("V4.2 marker preflight failed")

    unique_relation_ids = sorted({str(record["relation_id"]) for record in records})
    relation_lookup = {
        relation_id: index for index, relation_id in enumerate(unique_relation_ids)
    }
    record_relation_index = torch.tensor(
        [relation_lookup[str(record["relation_id"])] for record in records],
        dtype=torch.long,
    )
    rank_arms = core.registered_ranks(len(unique_relation_ids))
    maximum_basis, maximum_basis_report = v41_core.fit_shared_contrast_basis(
        fit_base,
        fit_active,
        fit_labels,
        record_relation_index,
        rank=max(rank_arms),
    )
    negative_counts = (fit_active & ~fit_labels).sum(dim=0)
    capacity = {
        "rejected_v4_independent_parameters": len(records) * hidden_size,
        "rejected_v4_1_shared_rank": 4,
        "distinct_relations": len(unique_relation_ids),
        "registered_rank_arms": list(rank_arms),
        "maximum_basis": maximum_basis_report,
        "fit_positive_cells": int(fit_labels.sum()),
        "fit_negative_cells": int((fit_active & ~fit_labels).sum()),
        "fit_negative_cells_per_record_min": int(negative_counts.min()),
        "fit_negative_cells_per_record_median": float(
            negative_counts.float().median()
        ),
        "basis_nested_prefixes": True,
        "record_specific_hidden_vectors": 0,
    }
    v4_train.write_json(output / "semantic_capacity_report.json", capacity)
    if capacity["fit_negative_cells_per_record_min"] < 150:
        raise RuntimeError("V4.2 requires at least 150 fit negatives per record")

    print("\nStage 1a: nested shared semantic-rank sweep")
    arms: List[Dict[str, Any]] = []
    selected_router: v41_core.BaseSemanticRouter | None = None
    selected_rank: int | None = None
    for rank in rank_arms:
        print(f"\n  rank {rank}: fresh relation coefficients")
        router = v41_core.BaseSemanticRouter(
            len(records),
            hidden_size,
            maximum_basis[:rank].clone(),
            record_relation_index,
        ).to(device)
        events, selection = v41_train.fit_semantic_router(
            router,
            fit_base,
            fit_active,
            fit_labels,
            development_base,
            development_active,
            development_labels,
            args,
            selection_split="v4_2_development",
        )
        fit_report = v41_train.semantic_certificate_with_violations(
            router,
            fit_base,
            fit_active,
            fit_labels,
            fit_specs,
            positive_floor=float(args.semantic_certificate_positive_floor),
            negative_ceiling=float(args.semantic_certificate_negative_ceiling),
        )
        development_report = v41_train.semantic_certificate_with_violations(
            router,
            development_base,
            development_active,
            development_labels,
            development_specs,
            positive_floor=float(args.semantic_certificate_positive_floor),
            negative_ceiling=float(args.semantic_certificate_negative_ceiling),
        )
        arm = {
            "rank": int(rank),
            "trainable_relation_parameters": router.trainable_parameter_count,
            "fit": fit_report,
            "development": development_report,
            "selection": selection,
            "events": events,
            "passed": bool(fit_report["passed"] and development_report["passed"]),
            "weights_retained": False,
        }
        if arm["passed"]:
            selected_router = router
            selected_rank = int(rank)
            arm["weights_retained"] = True
        arms.append(arm)
        v4_train.write_json(output / f"rank_{rank}_training.json", arm)
        if arm["passed"]:
            break
    sweep_report = {
        "schema_version": 1,
        "kind": "mcf_v4_2_nested_shared_semantic_rank_sweep",
        "protocol": PROTOCOL,
        "registered_rank_arms": list(rank_arms),
        "evaluated_rank_arms": [int(arm["rank"]) for arm in arms],
        "selection_rule": "smallest_fit_and_fresh_development_passing_rank",
        "selected_smallest_passing_rank": selected_rank,
        "passed": selected_router is not None,
        "certification_opened": False,
        "actuator_optimizer_constructed": False,
        "arms": arms,
        "official_evaluation_prompts_seen": 0,
    }
    v4_train.write_json(output / "rank_sweep_report.json", sweep_report)
    if selected_router is None or selected_rank is None:
        write_failure(output, stage="no_rank_passed_development", certification_opened=False)
        raise SystemExit("V4.2 no registered shared rank passed development")
    for parameter in selected_router.parameters():
        parameter.requires_grad_(False)

    print(
        f"\nStage 1b: open untouched certification for selected rank {selected_rank}"
    )
    certification_delta, certification_base, certification_active = (
        v4_train.capture_shadow_difference_last_states(
            model,
            tok,
            layer,
            writer,
            span_router,
            [spec.prompt for spec in certification_specs],
            device,
            batch_size=int(args.batch_size),
        )
    )
    certification_labels = v4_train.labels_for_specs(
        certification_specs, certification_active, len(records)
    )
    certification_marker = v41_core.marker_certificate(
        certification_delta.square().mean(dim=1).sqrt(),
        certification_active,
        certification_labels,
        threshold=float(args.marker_rms_threshold),
    )
    certification_semantic = v41_train.semantic_certificate_with_violations(
        selected_router,
        certification_base,
        certification_active,
        certification_labels,
        certification_specs,
        positive_floor=float(args.semantic_certificate_positive_floor),
        negative_ceiling=float(args.semantic_certificate_negative_ceiling),
    )
    certification = {
        "schema_version": 1,
        "kind": "mcf_v4_2_one_shot_semantic_certification",
        "protocol": PROTOCOL,
        "selected_rank": selected_rank,
        "marker": certification_marker,
        "semantic": certification_semantic,
        "certification_open_count": 1,
        "optimizer_steps_after_certification_open": 0,
        "actuator_optimizer_constructed": False,
        "passed": bool(
            certification_marker["passed"] and certification_semantic["passed"]
        ),
        "official_evaluation_prompts_seen": 0,
    }
    v4_train.write_json(output / "one_shot_certification.json", certification)
    if not certification["passed"]:
        write_failure(output, stage="one_shot_certification", certification_opened=True)
        raise SystemExit("V4.2 selected rank failed one-shot certification")

    integrity = {
        "base_embedding_bit_identical": v4_train.tensor_sha256(input_embedding.weight)
        == embedding_hash,
        "lm_head_bit_identical": v4_train.tensor_sha256(output_embedding.weight)
        == lm_head_hash,
        "base_parameters_trainable": int(
            sum(parameter.requires_grad for parameter in model.parameters())
        ),
        "semantic_basis_trainable": selected_router.shared_basis.requires_grad,
        "semantic_parameters_trainable": int(
            sum(parameter.requires_grad for parameter in selected_router.parameters())
        ),
        "actuator_optimizer_constructed": False,
    }
    integrity["passed"] = bool(
        integrity["base_embedding_bit_identical"]
        and integrity["lm_head_bit_identical"]
        and integrity["base_parameters_trainable"] == 0
        and integrity["semantic_basis_trainable"] is False
        and integrity["semantic_parameters_trainable"] == 0
    )
    v4_train.write_json(output / "integrity_audit.json", integrity)
    if not integrity["passed"]:
        write_failure(output, stage="integrity", certification_opened=True)
        raise SystemExit("V4.2 integrity audit failed")

    state = core.certified_router_state(
        selected_router,
        case_ids=case_ids,
        relation_ids=unique_relation_ids,
        source_hashes=source_hashes,
        selected_rank=selected_rank,
        layer_index=int(args.layer),
        marker_rms_threshold=float(args.marker_rms_threshold),
    )
    state_path = output / "v4_2_certified_router_state.pt"
    torch.save(state, state_path)
    state_sha256 = v4_train.sha256_file(state_path)
    completion = {
        "schema_version": 1,
        "kind": "mcf_shadow_semantic_rank_sweep_v4_2_completion",
        "protocol": PROTOCOL,
        "passed": True,
        "registered_rank_arms": list(rank_arms),
        "selected_smallest_passing_rank": selected_rank,
        "certified_router_saved": True,
        "certified_router_sha256": state_sha256,
        "certification_open_count": 1,
        "optimizer_steps_after_certification_open": 0,
        "actuator_optimizer_constructed": False,
        "eligible_for_separate_actuator_training": True,
        "eligible_for_official_evaluation": False,
        "seed_1_blind_official_reuse_prohibited": True,
        "official_evaluation_prompts_seen": 0,
        "conclusion": "smallest_shared_semantic_rank_certified_router_frozen",
    }
    v4_train.write_json(output / "completion.json", completion)
    span_router.close()
    writer.remove()
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
