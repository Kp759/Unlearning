#!/usr/bin/env python3
"""Train and certify the passive shadow-embedding MCF router (V4.0).

This is a training/development-only process.  It has no MCF dataset argument
and refuses every environment variable historically used to expose official
paraphrases, neighborhoods, retain records, or PPL text.  The frozen V6.2
embedding delta is retained, but it is applied only in a no-gradient shadow
forward.  The returned model forward always begins from unmodified Base
embeddings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import torch

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_active_case_repair as mcf_repair
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_embedding_keyed_neuron_core as legacy_core
import mcf_embedding_keyed_neuron_erasure as legacy
import mcf_shadow_embedding_router_core as shadow
import mcf_shadow_relation_prompts as relation_prompts
import mcf_sure_directional_emb_lm_stage1 as directional
import mcf_sure_subject_directional_emb_stage1 as subject_writer
import mcf_synthetic_paraphrase_templates as synthetic
import scoped_span_edit as scoped
import sure_canonical_core as canonical


PROTOCOL = shadow.PROTOCOL
FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES = (
    "MCF_PATH",
    "OFFICIAL_MCF_PATH",
    "OFFICIAL_EVAL_PATH",
    "PARAPHRASE_PATH",
    "NEIGHBORHOOD_PATH",
    "RETAIN_EVAL_PATH",
    "ADVERSARIAL_EVAL_PATH",
)


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
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument("--layer", type=int, default=27)
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=4096)
    parser.add_argument("--corpus-prefixes", type=int, default=256)
    parser.add_argument("--router-steps", type=int, default=400)
    parser.add_argument("--router-lr", type=float, default=0.01)
    parser.add_argument("--router-weight-decay", type=float, default=1e-4)
    parser.add_argument("--router-row-norm-cap", type=float, default=32.0)
    parser.add_argument("--router-training-positive-floor", type=float, default=1.0)
    parser.add_argument("--router-training-negative-ceiling", type=float, default=-0.25)
    parser.add_argument("--router-certificate-positive-floor", type=float, default=0.25)
    parser.add_argument("--router-certificate-negative-ceiling", type=float, default=-0.25)
    parser.add_argument("--router-tail-k", type=int, default=2)
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
    positive_counts = (
        args.forget_num,
        args.frequency_docs,
        args.corpus_prefixes,
        args.router_steps,
        args.router_tail_k,
        args.actuator_steps,
        args.batch_size,
        args.check_every,
        args.identity_protected_prompts,
    )
    if any(int(value) <= 0 for value in positive_counts):
        parser.error("all counts and optimizer-step budgets must be positive")
    if int(args.frequency_doc_start) < 20:
        parser.error("documents 0:20 are reserved for official PPL")
    if int(args.layer) < 0:
        parser.error("--layer must be non-negative")
    if not 0 < float(args.minimum_writer_necessity_fraction) <= 1:
        parser.error("writer necessity fraction must lie in (0, 1]")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return compositional.tensor_sha256(value.detach().cpu().contiguous())


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "official-evaluation path leaked into V4 learner: "
            + ", ".join(sorted(exposed))
        )


def validate_registry(registry: Mapping[str, Any], args: argparse.Namespace) -> None:
    if registry.get("protocol") != PROTOCOL or int(registry.get("schema_version", -1)) != 1:
        raise RuntimeError("shadow-router experiment registry is stale")
    architecture = registry.get("architecture")
    expected = {
        "base_embedding_main_path": True,
        "v6_2_delta_shadow_path_only": True,
        "shadow_gradients": False,
        "exact_complete_subject_key": True,
        "router_feature": "shadow_minus_base_layer_output",
        "constant_record_residual_actuator": True,
        "layer": int(args.layer),
        "fixed_writer_off_bias": -1.0,
        "gate_threshold": 0.0,
    }
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("registry architecture differs from the V4 implementation")
    if registry.get("official_evaluation_prohibited") is not True:
        raise RuntimeError("V4 training registry does not prohibit official evaluation")
    if registry.get("seed_1_v3_6_2_official_result_consumed") is not True:
        raise RuntimeError("registry omits the consumed V3.6.2 seed-1 result")


def unique_specs(values: Sequence[relation_prompts.ShadowPromptSpec]) -> List[relation_prompts.ShadowPromptSpec]:
    seen: set[Tuple[str, int, bool]] = set()
    result: List[relation_prompts.ShadowPromptSpec] = []
    for value in values:
        key = (str(value.prompt), int(value.owner_index), bool(value.positive))
        if key in seen:
            continue
        seen.add(key)
        result.append(value)
    return result


def context_positive_specs(
    records: Sequence[Mapping[str, Any]],
    context_sets: Mapping[int, Mapping[str, Any]],
) -> List[relation_prompts.ShadowPromptSpec]:
    values: List[relation_prompts.ShadowPromptSpec] = []
    for owner, record in enumerate(records):
        case_id = int(record["case_id"])
        for index, prompt in enumerate(context_sets[case_id]["positive_prompts"]):
            values.append(
                relation_prompts.ShadowPromptSpec(
                    prompt=str(prompt),
                    owner_index=owner,
                    case_id=case_id,
                    relation_id=str(record["relation_id"]),
                    split="writer_training",
                    positive=True,
                    family_index=index,
                    source="frozen_v6_2_training_safe_context_manifest",
                )
            )
    return values


def labels_for_specs(
    specs: Sequence[relation_prompts.ShadowPromptSpec],
    active_subjects: torch.Tensor,
    records: int,
) -> torch.Tensor:
    labels = torch.zeros((len(specs), int(records)), dtype=torch.bool)
    for row, spec in enumerate(specs):
        if spec.positive:
            labels[row, int(spec.owner_index)] = True
    if bool((labels & ~active_subjects.cpu()).any()):
        raise RuntimeError("a development positive lacks its exact full-subject key")
    return labels


@torch.no_grad()
def capture_shadow_difference_last_states(
    model: torch.nn.Module,
    tok: Any,
    layer: torch.nn.Module,
    embedding_writer: legacy_core.ToggleableEmbeddingDelta,
    span_router: scoped.SpanGateRouter,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    differences: List[torch.Tensor] = []
    base_rows: List[torch.Tensor] = []
    routes: List[torch.Tensor] = []
    captured: List[torch.Tensor] = []

    def hook(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
        captured.append(shadow._hidden_from_layer_output(output))

    handle = layer.register_forward_hook(hook)
    old_padding = getattr(tok, "padding_side", "right")
    old_writer = bool(embedding_writer.enabled)
    tok.padding_side = "right"
    try:
        for start in range(0, len(prompts), int(batch_size)):
            batch = list(prompts[start : start + int(batch_size)])
            encoded = tok(batch, padding=True, return_tensors="pt").to(device)
            positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_rows = torch.arange(len(batch), device=device)
            route = span_router.route(encoded["input_ids"])
            routes.append(route.active)

            captured.clear()
            embedding_writer.enabled = False
            model(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError("Base shadow cache expected one target-layer output")
            base = captured[0][batch_rows, positions, :].detach().float()

            captured.clear()
            embedding_writer.enabled = True
            model(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError("marked shadow cache expected one target-layer output")
            marked = captured[0][batch_rows, positions, :].detach().float()
            base_rows.append(base.cpu())
            differences.append((marked - base).cpu())
    finally:
        handle.remove()
        embedding_writer.enabled = old_writer
        tok.padding_side = old_padding
    return torch.cat(differences), torch.cat(base_rows), torch.cat(routes)


def fit_router(
    router: shadow.PassiveShadowRouter,
    features: torch.Tensor,
    active: torch.Tensor,
    labels: torch.Tensor,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    device = router.weight.device
    features = features.to(device)
    active = active.to(device)
    labels = labels.to(device)
    optimizer = torch.optim.AdamW(
        [router.weight],
        lr=float(args.router_lr),
        weight_decay=float(args.router_weight_decay),
    )
    log: List[Dict[str, Any]] = []
    for step in range(1, int(args.router_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = router.scores(features)
        loss, metrics = shadow.record_balanced_router_hinge(
            scores,
            active,
            labels,
            positive_floor=float(args.router_training_positive_floor),
            negative_ceiling=float(args.router_training_negative_ceiling),
            tail_k=int(args.router_tail_k),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([router.weight], 1.0)
        optimizer.step()
        router.clamp_row_norm_(float(args.router_row_norm_cap))
        if step == 1 or step % 25 == 0 or step == int(args.router_steps):
            with torch.no_grad():
                certificate = shadow.router_certificate(
                    router.scores(features),
                    active,
                    labels,
                    positive_floor=float(args.router_certificate_positive_floor),
                    negative_ceiling=float(args.router_certificate_negative_ceiling),
                )
            row = {
                "optimizer_step": step,
                "loss": float(loss.detach()),
                "positive_violations": int(metrics["positive_violations"]),
                "negative_violations": int(metrics["negative_violations"]),
                "certificate": certificate,
            }
            log.append(row)
            print(
                f"  router step {step:4d}: loss {row['loss']:.6f}, "
                f"positive_min {certificate['positive_min']:.4f}, "
                f"negative_max {certificate['negative_max']:.4f}"
            )
    return log


def make_instances(
    specs: Sequence[relation_prompts.ShadowPromptSpec],
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[mcf_repair.MCFPromptInstance], List[int]]:
    instances: List[mcf_repair.MCFPromptInstance] = []
    owners: List[int] = []
    for index, spec in enumerate(specs):
        if not spec.positive:
            continue
        record = records[int(spec.owner_index)]
        instances.append(
            mcf_repair.MCFPromptInstance(
                record_index=int(spec.case_id),
                sampled_position=int(spec.owner_index),
                prompt_type=str(spec.split),
                prompt_index=int(spec.family_index),
                prompt=str(spec.prompt),
                target_new=str(record["reference"]),
                target_true=str(record["answer"]),
            )
        )
        owners.append(int(spec.owner_index))
    return instances, owners


@torch.no_grad()
def evaluate_margins(
    model: torch.nn.Module,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    device: torch.device,
    *,
    llama_like: bool,
    batch_size: int,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    for start in range(0, len(instances), int(batch_size)):
        margin = compositional.differentiable_instance_margins(
            model,
            tok,
            instances[start : start + int(batch_size)],
            device,
            llama_like=llama_like,
        )
        values.append(margin.detach().cpu())
    return torch.cat(values) if values else torch.empty(0)


def margin_report(margins: torch.Tensor, *, floor: float) -> Dict[str, Any]:
    if not int(margins.numel()):
        raise ValueError("margin report requires instances")
    failures = int(margins.lt(float(floor) - 1e-6).sum())
    return {
        "instances": int(margins.numel()),
        "failures": failures,
        "minimum": float(margins.min()),
        "median": float(margins.median()),
        "maximum": float(margins.max()),
        "floor": float(floor),
        "passed": failures == 0,
    }


def train_actuator(
    wrapper: shadow.ShadowDualPathCausalLM,
    branch: shadow.ShadowEmbeddingResidualBranch,
    tok: Any,
    instances: Sequence[mcf_repair.MCFPromptInstance],
    owners: Sequence[int],
    device: torch.device,
    *,
    llama_like: bool,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if len(instances) != len(owners):
        raise ValueError("actuator instances and owners differ")
    owner_counts = torch.bincount(
        torch.tensor(owners, dtype=torch.long), minlength=branch.semantic_router.records
    ).float()
    if bool(owner_counts.eq(0).any()):
        raise RuntimeError("every record needs actuator optimization positives")
    per_instance_weights = torch.tensor(
        [1.0 / (len(owner_counts) * float(owner_counts[int(owner)])) for owner in owners],
        dtype=torch.float32,
        device=device,
    )
    optimizer = torch.optim.Adam([branch.residual], lr=float(args.actuator_lr))
    branch.zero_residual_()
    log: List[Dict[str, Any]] = []
    final: Dict[str, Any] = {}
    for step in range(1, int(args.actuator_steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(instances), int(args.batch_size)):
            stop = min(len(instances), start + int(args.batch_size))
            margins = compositional.differentiable_instance_margins(
                wrapper,
                tok,
                instances[start:stop],
                device,
                llama_like=llama_like,
            )
            hinge = torch.relu(float(args.forget_margin) - margins).square()
            loss = (hinge * per_instance_weights[start:stop]).sum()
            loss.backward()
        torch.nn.utils.clip_grad_norm_([branch.residual], 1.0)
        optimizer.step()
        branch.clamp_residual_relative_(float(args.actuator_relative_cap))
        if step == 1 or step % int(args.check_every) == 0:
            margins = evaluate_margins(
                wrapper,
                tok,
                instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.batch_size),
            )
            report = margin_report(margins, floor=float(args.forget_margin))
            relative = branch.relative_residual_norms().cpu()
            row = {
                "optimizer_step": step,
                "margin": report,
                "relative_residual_norm_max": float(relative.max()),
                "relative_residual_norm_median": float(relative.median()),
                "records_at_cap": int(
                    relative.ge(float(args.actuator_relative_cap) - 1e-6).sum()
                ),
            }
            log.append(row)
            print(
                f"  actuator step {step:3d}: failures {report['failures']}, "
                f"minimum {report['minimum']:.4f}, norm-max "
                f"{row['relative_residual_norm_max']:.4f}"
            )
            if report["passed"]:
                final = row
                break
    if not final:
        final = log[-1]
    return log, final


@torch.no_grad()
def closed_route_identity_audit(
    base_model: torch.nn.Module,
    wrapper: shadow.ShadowDualPathCausalLM,
    embedding_writer: legacy_core.ToggleableEmbeddingDelta,
    branch: shadow.ShadowEmbeddingResidualBranch,
    tok: Any,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> Dict[str, Any]:
    old_padding = getattr(tok, "padding_side", "right")
    old_writer = bool(embedding_writer.enabled)
    old_branch = bool(branch.enabled)
    tok.padding_side = "right"
    mismatched = 0
    maximum = 0.0
    gate_cells = 0
    try:
        for start in range(0, len(prompts), int(batch_size)):
            batch = list(prompts[start : start + int(batch_size)])
            encoded = tok(batch, padding=True, return_tensors="pt").to(device)
            positions = encoded["attention_mask"].sum(dim=1) - 1
            rows = torch.arange(len(batch), device=device)
            embedding_writer.enabled = False
            branch.enabled = False
            base = base_model(**encoded, use_cache=False, return_dict=True).logits[
                rows, positions, :
            ].detach()
            branch.enabled = True
            current = wrapper(**encoded, use_cache=False, return_dict=True).logits[
                rows, positions, :
            ].detach()
            if branch.last_gates is not None:
                gate_cells += int(branch.last_gates.sum())
            diff = (current.float() - base.float()).abs()
            maximum = max(maximum, float(diff.max()))
            mismatched += int(~current.eq(base).all(dim=1).sum())
    finally:
        embedding_writer.enabled = old_writer
        branch.enabled = old_branch
        tok.padding_side = old_padding
    return {
        "prompts": len(prompts),
        "mismatched_last_logit_rows": mismatched,
        "last_logit_abs_max": maximum,
        "open_gate_cells": gate_cells,
        "bit_identical": mismatched == 0 and maximum == 0.0,
        "passed": mismatched == 0 and maximum == 0.0 and gate_cells == 0,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_environment_firewall()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)

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
    registry = load_json(paths["experiment_registry"])
    validate_registry(registry, args)

    locked_records, split_manifest = directional.validate_locked(
        paths["training_visible"], paths["split_manifest"], int(args.seed), int(args.forget_num)
    )
    if split_manifest.get("protocol") != locked_split.PROTOCOL:
        raise RuntimeError("V4 requires the locked direct-only MCF split")
    locked_split.assert_direct_only_training_view(locked_records)
    records = compositional._record_views(locked_records)
    context_manifest = load_json(paths["context_manifest"])
    stage1_state = torch.load(paths["stage1_state"], map_location="cpu", weights_only=False)
    if not isinstance(stage1_state, Mapping):
        raise RuntimeError("V6.2 writer state must be a mapping")
    stage1_report = load_json(paths["stage1_report"])
    legacy._validate_firewall(context_manifest, stage1_state)
    clean_writer_receipt = legacy._validate_clean_stage1_lineage(
        context_manifest,
        stage1_state,
        stage1_report,
        paths["context_manifest"],
        paths["stage1_writer_log"],
    )
    if str(Path(args.model_path).resolve()) != str(
        clean_writer_receipt["base_model_path"]
    ):
        raise RuntimeError("V4 Base-model path differs from frozen V6.2 lineage")
    if sha256_file(paths["context_manifest"]) != str(stage1_state["context_manifest_sha256"]):
        raise RuntimeError("V6.2 writer and context manifest hashes differ")
    stage1_portability = load_json(paths["clean_stage1_portability"])
    stage1_acceptance = load_json(paths["clean_stage1_acceptance"])
    if (
        stage1_acceptance.get("kind") != "mcf_clean_stage1_writer_acceptance"
        or stage1_acceptance.get("passed") is not True
        or bool(stage1_acceptance.get("official_evaluation_opened"))
        or dict(stage1_acceptance.get("training_safe_portability", {}))
        != dict(stage1_portability)
    ):
        raise RuntimeError("clean V6.2 writer acceptance conjunction is invalid")
    context_sets = legacy._context_sets_by_case(context_manifest, records)
    case_ids = [int(record["case_id"]) for record in records]
    if [int(value) for value in stage1_state.get("case_ids", [])] != case_ids:
        raise RuntimeError("V6.2 writer case order differs from V4")
    coverage = relation_prompts.coverage_report(records)
    if not coverage["complete"]:
        raise RuntimeError(f"relation prompt bank lacks coverage: {coverage['missing_relation_ids']}")

    firewall_receipt = {
        "schema_version": 1,
        "kind": "mcf_passive_shadow_embedding_router_training_firewall",
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "forget_num": len(records),
        "source_hashes": {name: sha256_file(path) for name, path in paths.items()},
        "main_embedding_path": "Base_only",
        "v6_2_embedding_delta_role": "no_gradient_shadow_context_feature_only",
        "official_evaluation_arguments_available": False,
        "forbidden_evaluation_environment_variables_present": [],
        "official_evaluation_prompts_seen": 0,
    }
    write_json(output / "training_firewall_receipt.json", firewall_receipt)

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
        raise RuntimeError("model must expose input and output embeddings")
    layers = scoped.find_decoder_layers(model)
    if int(args.layer) >= len(layers):
        raise RuntimeError("registered shadow layer is outside this model")
    target_layer = layers[int(args.layer)]
    hidden_size = int(input_embedding.weight.shape[1])
    embedding_rows = [int(value) for value in stage1_state.get("selected_embedding_rows", [])]
    embedding_delta = stage1_state.get("embedding_delta")
    if not embedding_rows or not isinstance(embedding_delta, torch.Tensor):
        raise RuntimeError("V6.2 state lacks its sparse embedding delta")
    if embedding_delta.shape != (len(embedding_rows), hidden_size):
        raise RuntimeError("V6.2 sparse embedding delta shape is incompatible")
    observed_fingerprint = compositional.frozen_transformer_fingerprint(model)
    if not math.isclose(
        float(observed_fingerprint),
        float(clean_writer_receipt["base_transformer_fingerprint"]),
        rel_tol=1e-12,
        abs_tol=1e-3,
    ):
        raise RuntimeError("V4 Transformer differs from the V6.2 Base model")
    selected_index = torch.tensor(
        embedding_rows, dtype=torch.long, device=input_embedding.weight.device
    )
    if tensor_sha256(input_embedding.weight.index_select(0, selected_index)) != str(
        clean_writer_receipt["base_selected_embedding_rows_sha256"]
    ):
        raise RuntimeError("V4 Base embedding rows differ from V6.2 lineage")
    base_embedding_hash = tensor_sha256(input_embedding.weight)
    base_lm_head_hash = tensor_sha256(output_embedding.weight)
    embedding_writer = legacy_core.ToggleableEmbeddingDelta(
        input_embedding, embedding_rows, embedding_delta
    )
    embedding_writer.enabled = False
    subjects = [str(record["subject"]) for record in records]
    subject_patterns = scoped.build_subject_patterns(tok, subjects)
    span_router = scoped.SpanGateRouter(
        input_embedding, subject_patterns, subjects=subjects, model=model
    )

    documents = subject_writer.load_frequency_documents(
        args.wikidata_dir, int(args.frequency_doc_start), int(args.frequency_docs)
    )
    if not documents:
        raise RuntimeError("no development-only Wikipedia documents were loaded")
    corpus_prefixes = synthetic.corpus_context_prefixes(
        documents, count=int(args.corpus_prefixes), seed=int(args.seed) + 4049
    )
    if len(corpus_prefixes) < int(args.corpus_prefixes):
        raise RuntimeError("development corpus did not fill the registered prefix bank")

    writer_specs = context_positive_specs(records, context_sets)
    calibration_positive = relation_prompts.build_positive_specs(
        records, split="calibration", corpus_prefixes=corpus_prefixes
    )
    calibration_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="calibration",
        variants_per_record=2,
        corpus_prefixes=corpus_prefixes,
    )
    heldout_positive = relation_prompts.build_positive_specs(
        records, split="heldout", corpus_prefixes=corpus_prefixes
    )
    heldout_negative = relation_prompts.build_wrong_relation_specs(
        records,
        split="heldout",
        variants_per_record=2,
        corpus_prefixes=corpus_prefixes,
    )
    optimization_specs = unique_specs(
        [*writer_specs, *calibration_positive, *calibration_negative]
    )
    heldout_specs = unique_specs([*heldout_positive, *heldout_negative])
    prompt_manifest = {
        "schema_version": 1,
        "kind": "mcf_shadow_router_development_prompt_manifest",
        "protocol": PROTOCOL,
        "counts": {
            "frozen_writer_training_positive": len(writer_specs),
            "calibration_positive": len(calibration_positive),
            "calibration_wrong_relation": len(calibration_negative),
            "heldout_positive": len(heldout_positive),
            "heldout_wrong_relation": len(heldout_negative),
        },
        "calibration_used_for_optimization": True,
        "heldout_used_for_optimization": False,
        "heldout_features_evaluated_after_router_fit": True,
        "relation_coverage": coverage,
        "optimization_specs": [value.json() for value in optimization_specs],
        "heldout_specs": [value.json() for value in heldout_specs],
        "official_evaluation_prompts_seen": 0,
    }
    write_json(output / "development_prompt_manifest.json", prompt_manifest)

    print("\nStage 1: cache Base and frozen-V6.2-shadow contextual states")
    optimization_features, optimization_base, optimization_active = (
        capture_shadow_difference_last_states(
            model,
            tok,
            target_layer,
            embedding_writer,
            span_router,
            [value.prompt for value in optimization_specs],
            device,
            batch_size=int(args.batch_size),
        )
    )
    optimization_labels = labels_for_specs(
        optimization_specs, optimization_active, len(records)
    )

    semantic_router = shadow.PassiveShadowRouter(
        len(records), hidden_size, fixed_bias=-1.0, threshold=0.0
    ).to(device)
    print("\nStage 1a: fit the passive subject-and-relation router")
    router_log = fit_router(
        semantic_router,
        optimization_features,
        optimization_active,
        optimization_labels,
        args,
    )
    optimization_certificate = shadow.router_certificate(
        semantic_router.scores(optimization_features.to(device)).detach().cpu(),
        optimization_active,
        optimization_labels,
        positive_floor=float(args.router_certificate_positive_floor),
        negative_ceiling=float(args.router_certificate_negative_ceiling),
    )
    write_json(output / "router_training_log.json", {"events": router_log})
    if not optimization_certificate["passed"]:
        write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "passed": False,
                "stage": "router_optimization_certificate",
                "certificate": optimization_certificate,
                "candidate_saved": False,
                "official_evaluation_prompts_seen": 0,
            },
        )
        raise SystemExit("V4 router failed its optimization-set certificate; actuator refused")

    print("\nStage 1b: open the disjoint development held-out paraphrase bank")
    heldout_features, _heldout_base, heldout_active = capture_shadow_difference_last_states(
        model,
        tok,
        target_layer,
        embedding_writer,
        span_router,
        [value.prompt for value in heldout_specs],
        device,
        batch_size=int(args.batch_size),
    )
    heldout_labels = labels_for_specs(heldout_specs, heldout_active, len(records))
    heldout_certificate = shadow.router_certificate(
        semantic_router.scores(heldout_features.to(device)).detach().cpu(),
        heldout_active,
        heldout_labels,
        positive_floor=float(args.router_certificate_positive_floor),
        negative_ceiling=float(args.router_certificate_negative_ceiling),
    )
    zero_features = torch.zeros((1, hidden_size), device=device)
    zero_scores = semantic_router.scores(zero_features).detach().cpu()
    writer_off_certificate = {
        "scores_exact_fixed_bias": bool(zero_scores.eq(-1.0).all()),
        "maximum": float(zero_scores.max()),
        "open_gates": int(zero_scores.ge(0).sum()),
        "passed": bool(zero_scores.eq(-1.0).all()),
    }
    router_report = {
        "schema_version": 1,
        "kind": "mcf_shadow_embedding_router_certificate",
        "protocol": PROTOCOL,
        "optimization": optimization_certificate,
        "heldout": heldout_certificate,
        "writer_off_zero_feature": writer_off_certificate,
        "passed": bool(heldout_certificate["passed"] and writer_off_certificate["passed"]),
        "official_evaluation_prompts_seen": 0,
    }
    write_json(output / "router_certificate.json", router_report)
    if not router_report["passed"]:
        write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "passed": False,
                "stage": "heldout_router_certificate",
                "router_certificate": router_report,
                "candidate_saved": False,
                "official_evaluation_prompts_seen": 0,
            },
        )
        raise SystemExit("V4 router failed held-out development; actuator refused")

    for parameter in semantic_router.parameters():
        parameter.requires_grad_(False)
    # Scale each residual cap by the median frozen Base layer-state norm for
    # that record's optimization positives.  This is a model-size-stable norm
    # definition and does not borrow an existing neuron column.
    reference_norms: List[torch.Tensor] = []
    for owner in range(len(records)):
        indices = [
            index
            for index, spec in enumerate(optimization_specs)
            if spec.positive and int(spec.owner_index) == owner
        ]
        reference_norms.append(optimization_base[indices].norm(dim=1).median())
    branch = shadow.ShadowEmbeddingResidualBranch(
        target_layer,
        span_router,
        semantic_router,
        hidden_size,
        residual_reference_norms=torch.stack(reference_norms).to(device),
    ).to(device)
    wrapper = shadow.ShadowDualPathCausalLM(model, embedding_writer, branch)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    branch.residual.requires_grad_(True)

    actuator_specs = unique_specs([*writer_specs, *calibration_positive])
    actuator_instances, actuator_owners = make_instances(actuator_specs, records)
    heldout_instances, _ = make_instances(heldout_positive, records)
    print("\nStage 2: train the constant gated residual from exact zero")
    actuator_log, actuator_endpoint = train_actuator(
        wrapper,
        branch,
        tok,
        actuator_instances,
        actuator_owners,
        device,
        llama_like=llama_like,
        args=args,
    )
    write_json(
        output / "actuator_training.json",
        {
            "events": actuator_log,
            "endpoint": actuator_endpoint,
            "all_records_and_visible_contexts_per_update": True,
            "optimizer_steps_max": int(args.actuator_steps),
            "residual_initialization": "bit_exact_zero",
        },
    )
    if not actuator_endpoint["margin"]["passed"]:
        write_json(
            output / "completion.json",
            {
                "protocol": PROTOCOL,
                "passed": False,
                "stage": "actuator_visible_reachability",
                "candidate_saved": False,
                "official_evaluation_prompts_seen": 0,
            },
        )
        raise SystemExit("V4 constant residual did not reach all visible positives")

    print("\nStage 3: frozen held-out portability, locality, and causality audits")
    heldout_margins = evaluate_margins(
        wrapper,
        tok,
        heldout_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.batch_size),
    )
    heldout_margin_report = margin_report(
        heldout_margins, floor=float(args.forget_margin)
    )
    all_positive_prompts = {value.prompt for value in actuator_specs}
    negative_prompts: List[str] = []
    for record in records:
        for row in context_sets[int(record["case_id"])]["negative_contexts"]:
            prompt = str(row["prompt"])
            if prompt not in all_positive_prompts:
                negative_prompts.append(prompt)
    negative_prompts.extend(value.prompt for value in calibration_negative)
    negative_prompts.extend(value.prompt for value in heldout_negative)
    negative_prompts.extend(corpus_prefixes[: int(args.identity_protected_prompts)])
    negative_prompts = list(dict.fromkeys(negative_prompts))
    identity = closed_route_identity_audit(
        model,
        wrapper,
        embedding_writer,
        branch,
        tok,
        negative_prompts,
        device,
        batch_size=int(args.batch_size),
    )
    branch.shadow_writer_enabled = False
    without_writer_margins = evaluate_margins(
        wrapper,
        tok,
        heldout_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.batch_size),
    )
    branch.shadow_writer_enabled = True
    without_writer_failures = int(
        without_writer_margins.lt(float(args.forget_margin) - 1e-6).sum()
    )
    writer_necessity = {
        "heldout_instances": len(heldout_instances),
        "failures_without_shadow_embedding": without_writer_failures,
        "failure_fraction": without_writer_failures / len(heldout_instances),
        "required_fraction": float(args.minimum_writer_necessity_fraction),
        "zero_feature_score": -1.0,
        "passed": without_writer_failures
        >= math.ceil(
            len(heldout_instances) * float(args.minimum_writer_necessity_fraction)
        ),
    }
    integrity = {
        "base_embedding_sha256_before": base_embedding_hash,
        "base_embedding_sha256_after": tensor_sha256(input_embedding.weight),
        "lm_head_sha256_before": base_lm_head_hash,
        "lm_head_sha256_after": tensor_sha256(output_embedding.weight),
        "base_embedding_bit_identical": tensor_sha256(input_embedding.weight)
        == base_embedding_hash,
        "lm_head_bit_identical": tensor_sha256(output_embedding.weight)
        == base_lm_head_hash,
        "base_parameters_trainable": int(
            sum(parameter.requires_grad for parameter in model.parameters())
        ),
    }
    integrity["passed"] = bool(
        integrity["base_embedding_bit_identical"]
        and integrity["lm_head_bit_identical"]
        and integrity["base_parameters_trainable"] == 0
    )
    final_audit = {
        "schema_version": 1,
        "kind": "mcf_shadow_embedding_router_training_only_final_audit",
        "protocol": PROTOCOL,
        "router": router_report,
        "heldout_forget_margin": heldout_margin_report,
        "closed_route_identity": identity,
        "shadow_embedding_necessity": writer_necessity,
        "integrity": integrity,
        "residual_relative_norm_max": float(branch.relative_residual_norms().max()),
        "residual_relative_cap": float(args.actuator_relative_cap),
        "official_evaluation_prompts_seen": 0,
    }
    final_audit["passed"] = bool(
        router_report["passed"]
        and heldout_margin_report["passed"]
        and identity["passed"]
        and writer_necessity["passed"]
        and integrity["passed"]
        and final_audit["residual_relative_norm_max"]
        <= float(args.actuator_relative_cap) + 1e-6
    )
    write_json(output / "final_training_only_audit.json", final_audit)

    candidate_path = output / "v4_shadow_embedding_candidate.pt"
    candidate_saved = False
    candidate_sha256 = None
    if final_audit["passed"] and bool(args.save_candidate):
        state = shadow.shadow_candidate_state(
            layer_index=int(args.layer),
            case_ids=case_ids,
            subjects=subjects,
            subject_patterns=subject_patterns,
            embedding_row_ids=embedding_rows,
            embedding_delta=embedding_delta,
            semantic_router=semantic_router,
            branch=branch,
            source_hashes=firewall_receipt["source_hashes"],
        )
        torch.save(state, candidate_path)
        candidate_sha256 = sha256_file(candidate_path)
        candidate_saved = True
    completion = {
        "schema_version": 1,
        "kind": "mcf_shadow_embedding_router_training_only_completion",
        "protocol": PROTOCOL,
        "passed": bool(final_audit["passed"]),
        "architecture": {
            "main_forward_embedding": "unaltered_Base",
            "shadow_forward_embedding": "Base_plus_frozen_V6.2_delta",
            "embedding_retained": True,
            "embedding_permanently_materialized": False,
            "exact_subject_key": True,
            "semantic_feature": "shadow_minus_base_contextual_state",
            "actuator": "constant_record_specific_residual",
            "layer": int(args.layer),
        },
        "candidate_saved": candidate_saved,
        "candidate_sha256": candidate_sha256,
        "eligible_for_separately_preregistered_official_evaluation": bool(
            final_audit["passed"] and candidate_saved
        ),
        "official_evaluation_allowed_in_this_process": False,
        "official_evaluation_prompts_seen": 0,
        "conclusion": (
            "shadow_embedding_context_router_training_only_passed"
            if final_audit["passed"]
            else "shadow_embedding_context_router_failed_closed"
        ),
    }
    write_json(output / "completion.json", completion)
    print(json.dumps(completion, indent=2))
    if not final_audit["passed"]:
        raise SystemExit("V4 failed one or more development-only gates; candidate refused")


if __name__ == "__main__":
    main()
