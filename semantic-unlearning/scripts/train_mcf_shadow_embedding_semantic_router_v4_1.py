#!/usr/bin/env python3
"""Train V4.1's capacity-limited shadow-marker/Base-semantic MCF router.

The rejected V4.0 development result is a required, hash-bound input.  Its
former held-out scaffolds are explicitly consumed fit data here.  A second
family is development-only model selection, and a third disjoint family is
opened once for certification after selection.  No official dataset path or
prompt is available to this learner.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_embedding_keyed_neuron_core as legacy_core
import mcf_embedding_keyed_neuron_erasure as legacy
import mcf_shadow_embedding_router_core as v4_core
import mcf_shadow_embedding_semantic_router_core as core
import mcf_shadow_relation_prompts as relation_prompts
import mcf_sure_directional_emb_lm_stage1 as directional
import mcf_sure_subject_directional_emb_stage1 as subject_writer
import mcf_synthetic_paraphrase_templates as synthetic
import scoped_span_edit as scoped
import sure_canonical_core as canonical
import train_mcf_shadow_embedding_router_v4 as v4_train


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
    parser.add_argument("--semantic-shared-rank", type=int, default=4)
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
    parser.add_argument("--development-positive-variants-per-record", type=int, default=4)
    parser.add_argument("--development-wrong-relations-per-record", type=int, default=32)
    parser.add_argument("--certification-positive-variants-per-record", type=int, default=4)
    parser.add_argument("--certification-wrong-relations-per-record", type=int, default=64)
    parser.add_argument("--actuator-steps", type=int, default=100)
    parser.add_argument("--actuator-lr", type=float, default=0.01)
    parser.add_argument("--actuator-relative-cap", type=float, default=1.5)
    parser.add_argument("--forget-margin", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--check-every", type=int, default=5)
    parser.add_argument("--identity-protected-prompts", type=int, default=512)
    parser.add_argument("--minimum-writer-necessity-fraction", type=float, default=0.5)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single",), default="single")
    parser.add_argument("--save-candidate", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    counts = (
        args.forget_num,
        args.frequency_docs,
        args.corpus_prefixes,
        args.semantic_shared_rank,
        args.semantic_router_steps,
        args.semantic_selection_check_every,
        args.semantic_selection_patience,
        args.semantic_tail_k,
        args.fit_positive_variants_per_old_family,
        args.fit_wrong_relations_per_old_family,
        args.development_positive_variants_per_record,
        args.development_wrong_relations_per_record,
        args.certification_positive_variants_per_record,
        args.certification_wrong_relations_per_record,
        args.actuator_steps,
        args.batch_size,
        args.check_every,
        args.identity_protected_prompts,
    )
    if any(int(value) <= 0 for value in counts):
        parser.error("all count and optimizer-step arguments must be positive")
    if int(args.frequency_doc_start) < 20:
        parser.error("documents 0:20 remain reserved for official PPL")
    if not float(args.marker_rms_threshold) > 0:
        parser.error("marker RMS threshold must be strictly positive")
    if not 1 <= int(args.semantic_shared_rank) <= 4:
        parser.error("semantic shared rank must lie in [1, 4]")
    return args


def validate_failed_v4(root: Path) -> Dict[str, Any]:
    method = root / "method"
    paths = {
        "certificate": method / "router_certificate.json",
        "completion": method / "completion.json",
        "firewall": method / "training_firewall_receipt.json",
        "prompt_manifest": method / "development_prompt_manifest.json",
        "registry": root / "protocol" / "experiment_registry.json",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    certificate = v4_train.load_json(paths["certificate"])
    completion = v4_train.load_json(paths["completion"])
    firewall = v4_train.load_json(paths["firewall"])
    registry = v4_train.load_json(paths["registry"])
    optimization = certificate.get("optimization", {})
    heldout = certificate.get("heldout", {})
    writer_off = certificate.get("writer_off_zero_feature", {})
    checks = {
        "source_protocol": completion.get("protocol") == v4_core.PROTOCOL,
        "source_failed_closed": completion.get("passed") is False
        and completion.get("stage") == "heldout_router_certificate"
        and completion.get("candidate_saved") is False,
        "optimization_passed": optimization.get("passed") is True
        and int(optimization.get("positive_failures", -1)) == 0
        and int(optimization.get("negative_failures", -1)) == 0,
        "heldout_failed": heldout.get("passed") is False
        and int(heldout.get("positive_failures", -1)) == 33
        and int(heldout.get("negative_failures", -1)) == 44
        and math.isclose(
            float(heldout.get("positive_min", float("nan"))),
            -4.526548385620117,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(heldout.get("negative_max", float("nan"))),
            7.249982833862305,
            abs_tol=1e-9,
        ),
        "writer_off_exact": writer_off.get("passed") is True
        and int(writer_off.get("open_gates", -1)) == 0
        and float(writer_off.get("maximum", float("nan"))) == -1.0,
        "official_closed": int(firewall.get("official_evaluation_prompts_seen", -1))
        == 0
        and registry.get("official_evaluation_prohibited") is True,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError("failed V4 lineage mismatch: " + ", ".join(failed))
    return {
        "schema_version": 1,
        "kind": "mcf_shadow_router_v4_rejection_import",
        "source_run_dir": str(root.resolve()),
        "source_artifacts": {
            name: {"path": str(path.resolve()), "sha256": v4_train.sha256_file(path)}
            for name, path in paths.items()
        },
        "diagnosis": {
            "optimization_positive_failures": 0,
            "optimization_negative_failures": 0,
            "heldout_positive_failures": 33,
            "heldout_positive_cells": 200,
            "heldout_negative_failures": 44,
            "heldout_negative_cells": 100,
            "writer_off_open_gates": 0,
            "rejected_router_trainable_parameters": 50 * 3072,
            "rejected_fit_cells": 546 + 100,
            "rejected_parameters_per_fit_cell": (50 * 3072) / (546 + 100),
            "conclusion": (
                "independent_3072d_per_record_router_memorized_fit_and_failed_heldout"
            ),
        },
        "checks": checks,
        "passed": True,
        "official_evaluation_prompts_seen": 0,
    }


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    if registry.get("protocol") != PROTOCOL or int(registry.get("schema_version", -1)) != 1:
        raise RuntimeError("V4.1 experiment registry is stale")
    architecture = registry.get("architecture")
    expected = {
        "base_embedding_main_path": True,
        "v6_2_delta_shadow_path_only": True,
        "exact_complete_subject_key": True,
        "marker_gate": "shadow_minus_base_rms",
        "semantic_gate": "frozen_shared_rank_relation_router",
        "semantic_shared_rank": int(args.semantic_shared_rank),
        "semantic_basis_trainable": False,
        "semantic_relation_tied_coefficients": True,
        "semantic_record_specific_hidden_vectors": False,
        "joint_gate": "subject_and_marker_and_semantic",
        "constant_record_residual_actuator": True,
        "layer": int(args.layer),
        "marker_rms_threshold": float(args.marker_rms_threshold),
    }
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("V4.1 registry architecture mismatch")
    locked = registry.get("locked_training")
    locked_expected = {
        "semantic_shared_rank": int(args.semantic_shared_rank),
        "semantic_router_optimizer_updates_max": int(args.semantic_router_steps),
        "semantic_router_lr": float(args.semantic_router_lr),
        "semantic_router_weight_decay": float(args.semantic_router_weight_decay),
        "semantic_router_coefficient_norm_cap": float(
            args.semantic_router_coefficient_norm_cap
        ),
        "semantic_selection_check_every": int(
            args.semantic_selection_check_every
        ),
        "semantic_selection_patience_checks": int(
            args.semantic_selection_patience
        ),
    }
    if not isinstance(locked, Mapping) or any(
        locked.get(key) != value for key, value in locked_expected.items()
    ):
        raise RuntimeError("V4.1 registry optimizer/model-selection mismatch")
    prompt_policy = registry.get("development_prompt_policy")
    prompt_expected = {
        "fit_positive_variants_per_old_family_per_record": int(
            args.fit_positive_variants_per_old_family
        ),
        "fit_wrong_relation_negatives_per_old_family_per_record": int(
            args.fit_wrong_relations_per_old_family
        ),
        "development_positive_variants_per_record": int(
            args.development_positive_variants_per_record
        ),
        "development_wrong_relation_negatives_per_record": int(
            args.development_wrong_relations_per_record
        ),
        "certification_positive_variants_per_record": int(
            args.certification_positive_variants_per_record
        ),
        "certification_wrong_relation_negatives_per_record": int(
            args.certification_wrong_relations_per_record
        ),
    }
    if not isinstance(prompt_policy, Mapping) or any(
        prompt_policy.get(key) != value for key, value in prompt_expected.items()
    ):
        raise RuntimeError("V4.1 registry development-split mismatch")
    if registry.get("official_evaluation_prohibited") is not True:
        raise RuntimeError("V4.1 registry must prohibit official evaluation")


def semantic_certificate_with_violations(
    router: core.BaseSemanticRouter,
    base_states: torch.Tensor,
    active: torch.Tensor,
    labels: torch.Tensor,
    specs: Sequence[relation_prompts.ShadowPromptSpec],
    *,
    positive_floor: float,
    negative_ceiling: float,
) -> Dict[str, Any]:
    device = router.relation_coefficients.device
    with torch.no_grad():
        scores = router.scores(base_states.to(device)).detach().cpu()
    report = v4_core.router_certificate(
        scores,
        active,
        labels,
        positive_floor=float(positive_floor),
        negative_ceiling=float(negative_ceiling),
    )
    violations: List[Dict[str, Any]] = []
    for row, spec in enumerate(specs):
        for group in range(int(scores.shape[1])):
            if not bool(active[row, group]):
                continue
            positive = bool(labels[row, group])
            score = float(scores[row, group])
            failed = (
                score < float(positive_floor)
                if positive
                else score > float(negative_ceiling)
            )
            if failed:
                violations.append(
                    {
                        "prompt_index": row,
                        "prompt_sha256": hashlib.sha256(
                            spec.prompt.encode("utf-8")
                        ).hexdigest(),
                        "source_case_id": int(spec.case_id),
                        "source_relation_id": str(spec.relation_id),
                        "source_positive": bool(spec.positive),
                        "detector_group_index": group,
                        "label_positive": positive,
                        "score": score,
                    }
                )
    report["violating_cells"] = violations
    return report


def fit_semantic_router(
    router: core.BaseSemanticRouter,
    fit_base_states: torch.Tensor,
    fit_active: torch.Tensor,
    fit_labels: torch.Tensor,
    development_base_states: torch.Tensor,
    development_active: torch.Tensor,
    development_labels: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Optimize only relation coefficients and select on disjoint development.

    The shared hidden-dimensional basis is a frozen buffer.  Development
    metrics may choose/early-stop a checkpoint but never contribute a gradient.
    The certification bank is deliberately absent from this function.
    """

    device = router.relation_coefficients.device
    fit_base_states = fit_base_states.to(device)
    fit_active = fit_active.to(device)
    fit_labels = fit_labels.to(device)
    development_base_states = development_base_states.to(device)
    development_active = development_active.to(device)
    development_labels = development_labels.to(device)
    parameters = [router.relation_coefficients, router.relation_bias]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(args.semantic_router_lr),
        weight_decay=float(args.semantic_router_weight_decay),
    )
    events: List[Dict[str, Any]] = []
    best_key: Tuple[float, ...] | None = None
    best_state: Dict[str, torch.Tensor] | None = None
    best_step: int | None = None
    checks_without_improvement = 0
    stopped_step = int(args.semantic_router_steps)
    for step in range(1, int(args.semantic_router_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        fit_scores = router.scores(fit_base_states)
        loss, metrics = v4_core.record_balanced_router_hinge(
            fit_scores,
            fit_active,
            fit_labels,
            positive_floor=float(args.semantic_training_positive_floor),
            negative_ceiling=float(args.semantic_training_negative_ceiling),
            tail_k=int(args.semantic_tail_k),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        router.clamp_coefficient_norm_(
            float(args.semantic_router_coefficient_norm_cap)
        )
        check = (
            step == 1
            or step % int(args.semantic_selection_check_every) == 0
            or step == int(args.semantic_router_steps)
        )
        if check:
            with torch.no_grad():
                fit_scores = router.scores(fit_base_states)
                development_scores = router.scores(development_base_states)
                fit_certificate = v4_core.router_certificate(
                    fit_scores,
                    fit_active,
                    fit_labels,
                    positive_floor=float(args.semantic_certificate_positive_floor),
                    negative_ceiling=float(args.semantic_certificate_negative_ceiling),
                )
                development_certificate = v4_core.router_certificate(
                    development_scores,
                    development_active,
                    development_labels,
                    positive_floor=float(args.semantic_certificate_positive_floor),
                    negative_ceiling=float(args.semantic_certificate_negative_ceiling),
                )
                development_loss, _ = v4_core.record_balanced_router_hinge(
                    development_scores,
                    development_active,
                    development_labels,
                    positive_floor=float(args.semantic_certificate_positive_floor),
                    negative_ceiling=float(args.semantic_certificate_negative_ceiling),
                    tail_k=int(args.semantic_tail_k),
                )
                fit_eval_loss, _ = v4_core.record_balanced_router_hinge(
                    fit_scores,
                    fit_active,
                    fit_labels,
                    positive_floor=float(args.semantic_certificate_positive_floor),
                    negative_ceiling=float(args.semantic_certificate_negative_ceiling),
                    tail_k=int(args.semantic_tail_k),
                )
            development_failures = int(
                development_certificate["positive_failures"]
                + development_certificate["negative_failures"]
            )
            fit_failures = int(
                fit_certificate["positive_failures"]
                + fit_certificate["negative_failures"]
            )
            selection_key = (
                float(development_failures),
                float(development_loss),
                float(fit_failures),
                float(fit_eval_loss),
                float(step),
            )
            improved = best_key is None or selection_key < best_key
            if improved:
                best_key = selection_key
                best_step = step
                best_state = {
                    "relation_coefficients": (
                        router.relation_coefficients.detach().cpu().clone()
                    ),
                    "relation_bias": router.relation_bias.detach().cpu().clone(),
                }
                checks_without_improvement = 0
            else:
                checks_without_improvement += 1
            event = {
                "optimizer_step": step,
                "loss": float(loss.detach()),
                "positive_violations": int(metrics["positive_violations"]),
                "negative_violations": int(metrics["negative_violations"]),
                "fit_certificate": fit_certificate,
                "development_certificate": development_certificate,
                "development_certificate_loss": float(development_loss),
                "selection_improved": bool(improved),
                "checks_without_improvement": checks_without_improvement,
            }
            events.append(event)
            print(
                f"  semantic step {step:4d}: loss {event['loss']:.6f}, "
                f"fit fail {fit_failures}, dev fail {development_failures}, "
                f"dev +min {development_certificate['positive_min']:.4f}, "
                f"dev -max {development_certificate['negative_max']:.4f}"
            )
            if checks_without_improvement >= int(args.semantic_selection_patience):
                stopped_step = step
                break
    if best_state is None or best_step is None or best_key is None:
        raise RuntimeError("semantic model selection produced no checkpoint")
    with torch.no_grad():
        router.relation_coefficients.copy_(
            best_state["relation_coefficients"].to(device)
        )
        router.relation_bias.copy_(best_state["relation_bias"].to(device))
    selection = {
        "selection_split": "v4_1_development",
        "certification_split_visible_to_selection": False,
        "best_optimizer_step": int(best_step),
        "stopped_optimizer_step": int(stopped_step),
        "maximum_optimizer_steps": int(args.semantic_router_steps),
        "selection_key": [float(value) for value in best_key],
        "early_stopping_patience_checks": int(args.semantic_selection_patience),
        "shared_basis_trainable": False,
        "trainable_parameter_count": router.trainable_parameter_count,
    }
    return events, selection


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    v4_train.validate_environment_firewall()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    failed_v4 = validate_failed_v4(Path(args.failed_v4_run_dir).resolve())
    v4_train.write_json(output / "frozen_v4_rejection_import.json", failed_v4)

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
        raise RuntimeError("V4.1 requires the locked direct-only split")
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
        raise RuntimeError("V4.1 Base model differs from V6.2 lineage")
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
        raise RuntimeError("V6.2/V4.1 case order mismatch")

    source_hashes = {name: v4_train.sha256_file(path) for name, path in paths.items()}
    source_hashes["failed_v4_router_certificate"] = failed_v4["source_artifacts"][
        "certificate"
    ]["sha256"]
    firewall = {
        "schema_version": 1,
        "kind": "mcf_shadow_marker_base_semantic_router_training_firewall",
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "forget_num": len(records),
        "source_hashes": source_hashes,
        "v4_0_heldout_role": "consumed_fit_data",
        "v4_1_development_role": "checkpoint_selection_only_no_gradients",
        "v4_1_certification_role": "one_shot_after_model_selection",
        "seed_1_official_result_role": "human_architecture_diagnosis_only_not_learner_input",
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
    llama_like = canonical.is_llama_like(model, tok)
    input_embedding = model.get_input_embeddings()
    output_embedding = model.get_output_embeddings()
    if input_embedding is None or output_embedding is None:
        raise RuntimeError("model lacks input/output embeddings")
    layers = scoped.find_decoder_layers(model)
    if int(args.layer) >= len(layers):
        raise RuntimeError("V4.1 layer is outside Base model")
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
        raise RuntimeError("V4.1 Transformer differs from V6.2 Base")
    selected_index = torch.tensor(
        embedding_rows, dtype=torch.long, device=input_embedding.weight.device
    )
    if v4_train.tensor_sha256(input_embedding.weight.index_select(0, selected_index)) != str(
        clean_writer["base_selected_embedding_rows_sha256"]
    ):
        raise RuntimeError("V4.1 Base embedding rows differ from V6.2")
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
        raise RuntimeError("V4.1 could not load development Wikipedia documents")
    prefixes = synthetic.corpus_context_prefixes(
        documents, count=int(args.corpus_prefixes), seed=int(args.seed) + 4049
    )
    if len(prefixes) < int(args.corpus_prefixes):
        raise RuntimeError("V4.1 prefix bank is incomplete")

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
    development_positive = relation_prompts.build_positive_specs(
        records,
        split="v4_1_development",
        corpus_prefixes=prefixes,
        variants_per_record=int(args.development_positive_variants_per_record),
    )
    development_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="v4_1_development",
        variants_per_record=int(args.development_wrong_relations_per_record),
        corpus_prefixes=prefixes,
    )
    certification_positive = relation_prompts.build_positive_specs(
        records,
        split="v4_1_certification",
        corpus_prefixes=prefixes,
        variants_per_record=int(args.certification_positive_variants_per_record),
    )
    certification_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="v4_1_certification",
        variants_per_record=int(args.certification_wrong_relations_per_record),
        corpus_prefixes=prefixes,
    )
    fit_specs = v4_train.unique_specs(
        [
            *writer_specs,
            *old_calibration_positive,
            *consumed_v4_heldout_positive,
            *old_calibration_negative,
            *consumed_v4_heldout_negative,
        ]
    )
    development_specs = v4_train.unique_specs(
        [*development_positive, *development_negative]
    )
    certification_specs = v4_train.unique_specs(
        [*certification_positive, *certification_negative]
    )
    split_prompt_sets = {
        "fit": {spec.prompt for spec in fit_specs},
        "development": {spec.prompt for spec in development_specs},
        "certification": {spec.prompt for spec in certification_specs},
    }
    if (
        split_prompt_sets["fit"] & split_prompt_sets["development"]
        or split_prompt_sets["fit"] & split_prompt_sets["certification"]
        or split_prompt_sets["development"] & split_prompt_sets["certification"]
    ):
        raise RuntimeError("V4.1 fit/development/certification prompts overlap")
    manifest = {
        "schema_version": 1,
        "kind": "mcf_v4_1_factorized_development_prompt_manifest",
        "protocol": PROTOCOL,
        "counts": {
            "writer_training_positive": len(writer_specs),
            "old_calibration_positive": len(old_calibration_positive),
            "consumed_v4_heldout_positive": len(consumed_v4_heldout_positive),
            "old_calibration_wrong_relation": len(old_calibration_negative),
            "consumed_v4_heldout_wrong_relation": len(
                consumed_v4_heldout_negative
            ),
            "development_positive": len(development_positive),
            "development_wrong_relation": len(development_negative),
            "certification_positive": len(certification_positive),
            "certification_wrong_relation": len(certification_negative),
        },
        "v4_0_heldout_is_no_longer_heldout": True,
        "split_policy": {
            "fit": "optimizer_gradients",
            "development": "checkpoint_selection_and_early_stopping_no_gradients",
            "certification": "opened_once_after_selection_never_used_for_updates",
        },
        "fit_specs": [spec.json() for spec in fit_specs],
        "development_specs": [spec.json() for spec in development_specs],
        "certification_specs": [spec.json() for spec in certification_specs],
        "official_evaluation_prompts_seen": 0,
    }
    v4_train.write_json(output / "development_prompt_manifest.json", manifest)

    print("\nStage 1: cache fit and development Base/marker states")
    fit_delta, fit_base, fit_active = (
        v4_train.capture_shadow_difference_last_states(
            model,
            tok,
            layer,
            writer,
            span_router,
            [spec.prompt for spec in fit_specs],
            device,
            batch_size=int(args.batch_size),
        )
    )
    fit_labels = v4_train.labels_for_specs(
        fit_specs, fit_active, len(records)
    )
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
    fit_marker_report = core.marker_certificate(
        fit_delta.square().mean(dim=1).sqrt(),
        fit_active,
        fit_labels,
        threshold=float(args.marker_rms_threshold),
    )
    development_marker_report = core.marker_certificate(
        development_delta.square().mean(dim=1).sqrt(),
        development_active,
        development_labels,
        threshold=float(args.marker_rms_threshold),
    )
    v4_train.write_json(
        output / "marker_preflight.json",
        {
            "protocol": PROTOCOL,
            "fit": fit_marker_report,
            "development": development_marker_report,
            "writer_off_marker_rms": 0.0,
            "passed": bool(
                fit_marker_report["passed"]
                and development_marker_report["passed"]
            ),
            "official_evaluation_prompts_seen": 0,
        },
    )
    if not fit_marker_report["passed"] or not development_marker_report["passed"]:
        v4_train.write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "passed": False,
                "stage": "fit_or_development_marker_preflight",
                "candidate_saved": False,
                "official_evaluation_prompts_seen": 0,
            },
        )
        raise SystemExit("V4.1 embedding-marker presence failed before semantic fitting")

    unique_relation_ids = sorted({str(record["relation_id"]) for record in records})
    relation_lookup = {
        relation_id: index for index, relation_id in enumerate(unique_relation_ids)
    }
    record_relation_index = torch.tensor(
        [relation_lookup[str(record["relation_id"])] for record in records],
        dtype=torch.long,
    )
    shared_basis, shared_basis_report = core.fit_shared_contrast_basis(
        fit_base,
        fit_active,
        fit_labels,
        record_relation_index,
        rank=int(args.semantic_shared_rank),
    )
    semantic_router = core.BaseSemanticRouter(
        len(records),
        hidden_size,
        shared_basis,
        record_relation_index,
    ).to(device)
    capacity_report = {
        "rejected_v4_independent_parameters": len(records) * hidden_size,
        "shared_rank": int(semantic_router.rank),
        "distinct_relations": len(unique_relation_ids),
        "shared_fitted_basis_values": int(semantic_router.shared_basis.numel()),
        "trainable_relation_parameters": semantic_router.trainable_parameter_count,
        "total_stored_semantic_values": int(
            semantic_router.shared_basis.numel()
            + semantic_router.trainable_parameter_count
        ),
        "coefficients_referenced_per_record": int(semantic_router.rank),
        "independently_fitted_coefficients_per_record": 0,
        "record_specific_hidden_dimensional_vectors": 0,
        "fit_positive_cells": int(fit_labels.sum()),
        "fit_negative_cells": int((fit_active & ~fit_labels).sum()),
        "fit_negative_cells_per_record_min": int(
            (fit_active & ~fit_labels).sum(dim=0).min()
        ),
        "fit_negative_cells_per_record_median": float(
            (fit_active & ~fit_labels).sum(dim=0).float().median()
        ),
        "shared_basis": shared_basis_report,
        "relation_ids": unique_relation_ids,
    }
    v4_train.write_json(output / "semantic_capacity_report.json", capacity_report)
    if capacity_report["fit_negative_cells_per_record_min"] < 100:
        raise RuntimeError("V4.1 requires at least 100 fit negatives per record")

    print("\nStage 1a: fit shared-rank relation semantics; select on development")
    semantic_log, semantic_selection = fit_semantic_router(
        semantic_router,
        fit_base,
        fit_active,
        fit_labels,
        development_base,
        development_active,
        development_labels,
        args,
    )
    fit_semantic_report = semantic_certificate_with_violations(
        semantic_router,
        fit_base,
        fit_active,
        fit_labels,
        fit_specs,
        positive_floor=float(args.semantic_certificate_positive_floor),
        negative_ceiling=float(args.semantic_certificate_negative_ceiling),
    )
    development_semantic_report = semantic_certificate_with_violations(
        semantic_router,
        development_base,
        development_active,
        development_labels,
        development_specs,
        positive_floor=float(args.semantic_certificate_positive_floor),
        negative_ceiling=float(args.semantic_certificate_negative_ceiling),
    )
    v4_train.write_json(
        output / "semantic_router_training_log.json",
        {
            "events": semantic_log,
            "selection": semantic_selection,
            "capacity": capacity_report,
        },
    )
    router_preflight = {
        "schema_version": 1,
        "kind": "mcf_shadow_marker_base_semantic_router_preflight",
        "protocol": PROTOCOL,
        "capacity": capacity_report,
        "model_selection": semantic_selection,
        "fit_marker": fit_marker_report,
        "fit_semantic": fit_semantic_report,
        "development_marker": development_marker_report,
        "development_semantic": development_semantic_report,
        "writer_off": {
            "marker_rms": 0.0,
            "marker_gate_open": False,
            "semantic_gate_cannot_act_without_marker": True,
            "passed": True,
        },
        "passed": bool(
            fit_marker_report["passed"]
            and fit_semantic_report["passed"]
            and development_marker_report["passed"]
            and development_semantic_report["passed"]
        ),
        "certification_prompts_opened": False,
        "official_evaluation_prompts_seen": 0,
    }
    v4_train.write_json(output / "factorized_router_preflight.json", router_preflight)
    if not router_preflight["passed"]:
        v4_train.write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "passed": False,
                "stage": "fit_or_development_router_preflight",
                "candidate_saved": False,
                "official_evaluation_prompts_seen": 0,
            },
        )
        raise SystemExit("V4.1 low-rank router failed development preflight")

    for parameter in semantic_router.parameters():
        parameter.requires_grad_(False)
    reference_norms: List[torch.Tensor] = []
    for owner in range(len(records)):
        rows = [
            index
            for index, spec in enumerate(fit_specs)
            if spec.positive and int(spec.owner_index) == owner
        ]
        reference_norms.append(fit_base[rows].norm(dim=1).median())
    branch = core.ShadowMarkerSemanticResidualBranch(
        layer,
        span_router,
        semantic_router,
        hidden_size,
        marker_rms_threshold=float(args.marker_rms_threshold),
        residual_reference_norms=torch.stack(reference_norms).to(device),
    ).to(device)
    wrapper = v4_core.ShadowDualPathCausalLM(model, writer, branch)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    branch.residual.requires_grad_(True)

    actuator_specs = v4_train.unique_specs(
        [*writer_specs, *consumed_v4_heldout_positive]
    )
    actuator_instances, actuator_owners = v4_train.make_instances(
        actuator_specs, records
    )
    certification_instances, _certification_owners = v4_train.make_instances(
        certification_positive, records
    )
    print("\nStage 2: train the same constant residual behind the factorized gate")
    actuator_log, actuator_endpoint = v4_train.train_actuator(
        wrapper,
        branch,
        tok,
        actuator_instances,
        actuator_owners,
        device,
        llama_like=llama_like,
        args=args,
    )
    v4_train.write_json(
        output / "actuator_training.json",
        {
            "events": actuator_log,
            "endpoint": actuator_endpoint,
            "residual_initialization": "bit_exact_zero",
            "full_visible_context_coverage_per_update": True,
        },
    )
    if not actuator_endpoint["margin"]["passed"]:
        v4_train.write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "passed": False,
                "stage": "fit_actuator_reachability",
                "candidate_saved": False,
                "certification_prompts_opened": False,
                "official_evaluation_prompts_seen": 0,
            },
        )
        raise SystemExit("V4.1 actuator failed visible positive reachability")

    print("\nStage 3: open one-shot certification after all optimizer updates")
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
    certification_marker_report = core.marker_certificate(
        certification_delta.square().mean(dim=1).sqrt(),
        certification_active,
        certification_labels,
        threshold=float(args.marker_rms_threshold),
    )
    certification_semantic_report = semantic_certificate_with_violations(
        semantic_router,
        certification_base,
        certification_active,
        certification_labels,
        certification_specs,
        positive_floor=float(args.semantic_certificate_positive_floor),
        negative_ceiling=float(args.semantic_certificate_negative_ceiling),
    )
    router_report = {
        **router_preflight,
        "kind": "mcf_shadow_marker_base_semantic_router_certificate",
        "certification_marker": certification_marker_report,
        "certification_semantic": certification_semantic_report,
        "certification_open_count": 1,
        "optimizer_steps_after_certification_open": 0,
        "passed": bool(
            router_preflight["passed"]
            and certification_marker_report["passed"]
            and certification_semantic_report["passed"]
        ),
    }
    router_report.pop("certification_prompts_opened", None)
    v4_train.write_json(output / "factorized_router_certificate.json", router_report)
    if not router_report["passed"]:
        v4_train.write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "passed": False,
                "stage": "one_shot_v4_1_certification_router_certificate",
                "candidate_saved": False,
                "optimizer_steps_after_certification_open": 0,
                "official_evaluation_prompts_seen": 0,
            },
        )
        raise SystemExit("V4.1 low-rank router failed one-shot certification")

    print("\nStage 3a: certification portability and exact locality")
    certification_margins = v4_train.evaluate_margins(
        wrapper,
        tok,
        certification_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.batch_size),
    )
    certification_margin_report = v4_train.margin_report(
        certification_margins, floor=float(args.forget_margin)
    )
    positive_prompts = {spec.prompt for spec in actuator_specs}
    identity_prompts: List[str] = []
    for record in records:
        for row in context_sets[int(record["case_id"])]["negative_contexts"]:
            prompt = str(row["prompt"])
            if prompt not in positive_prompts:
                identity_prompts.append(prompt)
    identity_prompts.extend(spec.prompt for spec in old_calibration_negative)
    identity_prompts.extend(spec.prompt for spec in consumed_v4_heldout_negative)
    identity_prompts.extend(spec.prompt for spec in development_negative)
    identity_prompts.extend(spec.prompt for spec in certification_negative)
    identity_prompts.extend(prefixes[: int(args.identity_protected_prompts)])
    identity_prompts = list(dict.fromkeys(identity_prompts))
    identity_report = v4_train.closed_route_identity_audit(
        model,
        wrapper,
        writer,
        branch,
        tok,
        identity_prompts,
        device,
        batch_size=int(args.batch_size),
    )
    branch.shadow_writer_enabled = False
    without_writer = v4_train.evaluate_margins(
        wrapper,
        tok,
        certification_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.batch_size),
    )
    branch.shadow_writer_enabled = True
    writer_failures = int(without_writer.lt(float(args.forget_margin) - 1e-6).sum())
    writer_necessity = {
        "certification_instances": len(certification_instances),
        "failures_without_shadow_embedding": writer_failures,
        "failure_fraction": writer_failures / len(certification_instances),
        "required_fraction": float(args.minimum_writer_necessity_fraction),
        "marker_gate_without_writer": False,
        "passed": writer_failures
        >= math.ceil(
            len(certification_instances)
            * float(args.minimum_writer_necessity_fraction)
        ),
    }
    integrity = {
        "base_embedding_bit_identical": v4_train.tensor_sha256(input_embedding.weight)
        == embedding_hash,
        "lm_head_bit_identical": v4_train.tensor_sha256(output_embedding.weight)
        == lm_head_hash,
        "base_parameters_trainable": int(
            sum(parameter.requires_grad for parameter in model.parameters())
        ),
    }
    integrity["passed"] = bool(
        integrity["base_embedding_bit_identical"]
        and integrity["lm_head_bit_identical"]
        and integrity["base_parameters_trainable"] == 0
    )
    final = {
        "schema_version": 1,
        "kind": "mcf_v4_1_factorized_training_only_final_audit",
        "protocol": PROTOCOL,
        "router": router_report,
        "certification_forget_margin": certification_margin_report,
        "closed_route_identity": identity_report,
        "shadow_embedding_necessity": writer_necessity,
        "integrity": integrity,
        "residual_relative_norm_max": float(branch.relative_residual_norms().max()),
        "residual_relative_cap": float(args.actuator_relative_cap),
        "official_evaluation_prompts_seen": 0,
    }
    final["passed"] = bool(
        router_report["passed"]
        and certification_margin_report["passed"]
        and identity_report["passed"]
        and writer_necessity["passed"]
        and integrity["passed"]
        and final["residual_relative_norm_max"]
        <= float(args.actuator_relative_cap) + 1e-6
    )
    v4_train.write_json(output / "final_training_only_audit.json", final)

    candidate_path = output / "v4_1_factorized_candidate.pt"
    candidate_saved = False
    candidate_sha256 = None
    if final["passed"] and bool(args.save_candidate):
        state = core.factorized_candidate_state(
            layer_index=int(args.layer),
            case_ids=case_ids,
            subjects=subjects,
            subject_patterns=patterns,
            embedding_row_ids=embedding_rows,
            embedding_delta=embedding_delta,
            relation_ids=unique_relation_ids,
            semantic_router=semantic_router,
            branch=branch,
            source_hashes=source_hashes,
        )
        torch.save(state, candidate_path)
        candidate_sha256 = v4_train.sha256_file(candidate_path)
        candidate_saved = True
    completion = {
        "schema_version": 1,
        "kind": "mcf_v4_1_factorized_training_only_completion",
        "protocol": PROTOCOL,
        "passed": bool(final["passed"]),
        "architecture": {
            "main_forward_embedding": "unaltered_Base",
            "shadow_embedding_marker": "frozen_V6.2_retained",
            "embedding_permanently_materialized": False,
            "joint_gate": (
                "exact_subject_AND_shadow_marker_AND_shared_rank_Base_semantics"
            ),
            "semantic_shared_rank": int(semantic_router.rank),
            "semantic_basis_trainable": False,
            "semantic_trainable_parameters": semantic_router.trainable_parameter_count,
            "fit_negative_cells_per_record_min": capacity_report[
                "fit_negative_cells_per_record_min"
            ],
            "layer": int(args.layer),
        },
        "candidate_saved": candidate_saved,
        "candidate_sha256": candidate_sha256,
        "seed_1_blind_official_reuse_prohibited": True,
        "eligible_for_fresh_seed_official_preregistration": bool(
            final["passed"] and candidate_saved
        ),
        "official_evaluation_allowed_in_this_process": False,
        "official_evaluation_prompts_seen": 0,
        "certification_open_count": 1,
        "optimizer_steps_after_certification_open": 0,
        "conclusion": (
            "factorized_shadow_marker_semantic_router_training_only_passed"
            if final["passed"]
            else "factorized_shadow_marker_semantic_router_failed_closed"
        ),
    }
    v4_train.write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2))
    if not final["passed"]:
        raise SystemExit("V4.1 failed one or more development-only gates")


if __name__ == "__main__":
    main()
