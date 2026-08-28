#!/usr/bin/env python3
"""Embedding-keyed sparse neuron erasure for locked direct-only MCF.

Architecture
------------

    ordinary sparse subject embedding-row deltas (frozen Stage 1)
        -> frozen early Transformer layers
        -> record-specific contextual activation code
        -> sparse existing SwiGLU detector/actuator neurons
        -> frozen remaining Transformer
        -> exactly unchanged LM head

For each edit, a disjoint group of existing low-activation MLP neurons is
selected using only training-safe writer-on versus writer-off activations.
``gate_proj``/``up_proj`` rows learn to detect the complete embedding-induced
contextual code.  The matching ``down_proj`` columns learn an erasure residual.
All edits are materialized into ordinary model weights; there is no tokenizer
expansion, string matcher, retrieval cache, runtime router, sidecar, adapter,
LoRA, logit bias, or LM-head update.

Data firewall
-------------

This learner has deliberately no ``--mcf-path`` argument.  It accepts only a
locked direct-forget training view, an audited training-safe context manifest,
and a frozen Stage-1 writer whose stored manifest hash must match exactly.
Official paraphrases, neighborhoods, retain prompts, PPL text, aliases, and
adversarial attacks are unavailable to this process.  They may be opened only
by separate post-checkpoint evaluation processes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_active_case_repair as mcf_repair
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional_method
import mcf_compositional_marker_core as compositional_core
import mcf_embedding_keyed_neuron_core as neuron_core
import mcf_sure_directional_emb_lm_stage1 as directional
import mcf_sure_subject_directional_emb_stage1 as subject_writer
import mcf_synthetic_paraphrase_templates as synthetic
import sure_canonical_core as canonical


METHOD = "Embedding-Keyed Sparse Neuron Erasure"
PROTOCOL = neuron_core.PROTOCOL
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
    parser.add_argument("--experiment-registry", required=True)
    parser.add_argument("--experiment-label", default="primary")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forget-num", type=int, default=50)

    parser.add_argument("--neuron-layer", type=int, default=8)
    parser.add_argument("--neurons-per-record", type=int, default=4)
    parser.add_argument("--dormant-fraction", type=float, default=0.35)
    parser.add_argument(
        "--selection-mode",
        choices=("writer_contrastive", "dormant_random"),
        default="writer_contrastive",
    )
    parser.add_argument("--selection-stability-weight", type=float, default=1.0)
    parser.add_argument("--selection-positive-contexts", type=int, default=8)
    parser.add_argument("--selection-protected-prompts", type=int, default=1024)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--frequency-doc-start", type=int, default=20)
    parser.add_argument("--frequency-docs", type=int, default=5000)
    parser.add_argument("--corpus-protection-prompts", type=int, default=256)

    parser.add_argument("--detector-steps", type=int, default=600)
    parser.add_argument("--detector-lr", type=float, default=1e-3)
    parser.add_argument("--detector-record-batch", type=int, default=4)
    parser.add_argument("--detector-positive-contexts", type=int, default=2)
    parser.add_argument("--detector-negative-contexts", type=int, default=2)
    parser.add_argument("--detector-positive-floor", type=float, default=1.0)
    parser.add_argument("--detector-negative-weight", type=float, default=5.0)
    parser.add_argument("--detector-cross-weight", type=float, default=2.0)
    parser.add_argument("--detector-writer-off-weight", type=float, default=10.0)
    parser.add_argument("--detector-consistency-weight", type=float, default=1.0)
    parser.add_argument("--detector-l2", type=float, default=1e-5)
    parser.add_argument("--detector-relative-cap", type=float, default=0.40)
    parser.add_argument("--detector-off-abs-max", type=float, default=0.20)

    parser.add_argument("--actuator-steps", type=int, default=1000)
    parser.add_argument("--actuator-lr", type=float, default=5e-3)
    parser.add_argument("--actuator-batch-size", type=int, default=4)
    parser.add_argument("--actuator-protected-batch", type=int, default=4)
    parser.add_argument("--actuator-writer-off-every", type=int, default=4)
    parser.add_argument("--forget-margin", type=float, default=1.0)
    parser.add_argument("--margin-weight", type=float, default=100.0)
    parser.add_argument("--reference-nll-weight", type=float, default=10.0)
    parser.add_argument("--reference-nll-tolerance", type=float, default=0.05)
    parser.add_argument("--protected-kl-weight", type=float, default=10.0)
    parser.add_argument("--writer-off-nll-weight", type=float, default=10.0)
    parser.add_argument("--actuator-l2", type=float, default=1e-4)
    parser.add_argument("--actuator-relative-cap", type=float, default=0.40)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--check-every", type=int, default=50)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--kl-topk", type=int, default=64)
    parser.add_argument("--hook-materialization-tolerance", type=float, default=0.05)

    parser.add_argument(
        "--min-writer-necessary-direct-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--min-decoder-necessary-direct-fraction", type=float, default=0.50
    )
    parser.add_argument(
        "--gate-policy",
        choices=("strict", "report"),
        default="strict",
        help="Strict requires the training-only detector certificate before saving.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument("--save-checkpoint", action="store_true")
    value = parser.parse_args(list(argv) if argv is not None else None)

    if int(value.forget_num) <= 0:
        parser.error("--forget-num must be positive")
    if int(value.neuron_layer) < 0:
        parser.error("--neuron-layer must be non-negative")
    if int(value.frequency_doc_start) < 20:
        parser.error(
            "--frequency-doc-start must be >= 20; documents 0:20 are reserved "
            "for held-out official PPL"
        )
    if int(value.neurons_per_record) <= 0:
        parser.error("--neurons-per-record must be positive")
    if not 0.0 < float(value.dormant_fraction) <= 1.0:
        parser.error("--dormant-fraction must lie in (0, 1]")
    for name in ("detector_relative_cap", "actuator_relative_cap"):
        if not 0.0 < float(getattr(value, name)) <= 2.0:
            parser.error(f"--{name.replace('_', '-')} must lie in (0, 2]")
    if float(value.hook_materialization_tolerance) < 0:
        parser.error("--hook-materialization-tolerance must be non-negative")
    if value.device_map == "auto":
        parser.error("training sparse neurons requires --device-map single")
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _validate_environment_firewall() -> None:
    exposed = [
        name
        for name in FORBIDDEN_EVALUATION_ENVIRONMENT_VARIABLES
        if str(os.environ.get(name, "")).strip()
    ]
    if exposed:
        raise RuntimeError(
            "evaluation path leaked into learner environment: "
            + ", ".join(sorted(exposed))
        )


def _validate_experiment_registry(
    registry: Mapping[str, Any], args: argparse.Namespace
) -> None:
    if registry.get("protocol") != PROTOCOL:
        raise RuntimeError("experiment registry protocol mismatch")
    label = str(args.experiment_label)
    if label == "primary":
        expected = registry.get("primary_configuration")
        if not isinstance(expected, Mapping):
            raise RuntimeError("registry lacks primary configuration")
    else:
        changes = None
        rows = registry.get("controlled_training_ablations", [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, Mapping) and str(row.get("name")) == label:
                    changes = row.get("change_only")
                    break
        if changes is None and label.startswith("neurons_per_record_"):
            changes = {"neurons_per_record": int(label.rsplit("_", 1)[-1])}
        if changes is None and label.startswith("layer_"):
            changes = {"neuron_layer": int(label.rsplit("_", 1)[-1])}
        if not isinstance(changes, Mapping):
            raise RuntimeError(f"experiment label is not preregistered: {label!r}")
        primary = registry.get("primary_configuration")
        common = registry.get("ablation_common_configuration", {})
        if not isinstance(primary, Mapping) or not isinstance(common, Mapping):
            raise RuntimeError("registry lacks primary/ablation-common configuration")
        expected = {**primary, **common, **changes}
    for key, registered in expected.items():
        if not hasattr(args, str(key)):
            raise RuntimeError(f"registry key has no learner argument: {key!r}")
        observed = getattr(args, str(key))
        if isinstance(registered, float):
            matches = math.isclose(float(observed), float(registered), abs_tol=1e-12)
        else:
            matches = observed == registered
        if not matches:
            raise RuntimeError(
                f"experiment {label!r} diverges from registry: {key}="
                f"{observed!r}, expected {registered!r}"
            )


def _distribution(values: Sequence[float]) -> Dict[str, float]:
    return compositional_method.distribution([float(value) for value in values])


def _tensor_digest(tensor: torch.Tensor, *, row_chunk: int = 256) -> str:
    digest = hashlib.sha256()
    detached = tensor.detach()
    for start in range(0, int(detached.shape[0]), int(row_chunk)):
        block = detached[start : start + int(row_chunk)].contiguous().cpu()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _resolve_swiglu_mlp(model: torch.nn.Module, layer_index: int) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", None)
    if layers is None:
        raise RuntimeError("model does not expose model.layers")
    if int(layer_index) >= len(layers):
        raise ValueError(
            f"neuron layer {layer_index} is outside model with {len(layers)} layers"
        )
    mlp = getattr(layers[int(layer_index)], "mlp", None)
    if mlp is None:
        raise RuntimeError(f"layer {layer_index} has no MLP")
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, name):
            raise RuntimeError(
                f"layer {layer_index} MLP is not a supported SwiGLU module: missing {name}"
            )
    if mlp.gate_proj.weight.shape != mlp.up_proj.weight.shape:
        raise RuntimeError("gate_proj and up_proj shapes differ")
    if int(mlp.down_proj.weight.shape[1]) != int(mlp.gate_proj.weight.shape[0]):
        raise RuntimeError("down_proj input does not match SwiGLU intermediate size")
    return mlp


def _validate_firewall(
    context_manifest: Mapping[str, Any], stage1_state: Mapping[str, Any]
) -> None:
    data_access = context_manifest.get("data_access")
    if not isinstance(data_access, Mapping):
        raise RuntimeError("context manifest lacks data-access receipt")
    expected_zero = {
        "official_paraphrases_seen": 0,
        "official_neighborhoods_seen": 0,
        "benchmark_retain_seen": 0,
        "official_ppl_seen": False,
    }
    for key, expected in expected_zero.items():
        if data_access.get(key) != expected:
            raise RuntimeError(
                f"data firewall violation: {key}={data_access.get(key)!r}, "
                f"expected {expected!r}"
            )
    forbidden = {
        "paraphrase_prompts",
        "neighborhood_prompts",
        "retain_prompts",
        "official_ppl_text",
        "adversarial_prompts",
        "alias_prompts",
    }

    def walk(value: Any) -> None:
        if isinstance(value, Mapping):
            overlap = forbidden.intersection(str(key) for key in value)
            if overlap:
                raise RuntimeError(
                    "training context artifact contains evaluation-only fields: "
                    + ", ".join(sorted(overlap))
                )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(context_manifest)
    stored_hash = str(stage1_state.get("context_manifest_sha256") or "")
    if not stored_hash:
        raise RuntimeError("Stage-1 state lacks its exact context-manifest hash")


def _context_sets_by_case(
    context_manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> Dict[int, Dict[str, Any]]:
    rows = context_manifest.get("records")
    if not isinstance(rows, list):
        raise RuntimeError("context manifest lacks record contexts")
    by_case: Dict[int, Dict[str, Any]] = {}
    for value in rows:
        if not isinstance(value, Mapping):
            raise RuntimeError("invalid context-manifest record")
        case_id = int(value["case_id"])
        positives = value.get("positive_prompts")
        negatives = value.get("negative_contexts")
        if not isinstance(positives, list) or not positives:
            raise RuntimeError(f"case {case_id} has no training positives")
        if not isinstance(negatives, list) or not negatives:
            raise RuntimeError(f"case {case_id} has no training negatives")
        by_case[case_id] = dict(value)
    expected = [int(record["case_id"]) for record in records]
    if set(by_case) != set(expected):
        raise RuntimeError("context-manifest cases do not match locked training view")
    for record in records:
        case_id = int(record["case_id"])
        direct = str(record["direct_prompt"])
        if str(by_case[case_id]["positive_prompts"][0]) != direct:
            raise RuntimeError(
                f"case {case_id} direct prompt changed in context manifest"
            )
    return by_case


def _record_views(locked_records: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return compositional_method._record_views(locked_records)


def _unique_prompts(values: Sequence[str]) -> List[str]:
    return compositional_core.ordered_unique([str(value) for value in values])


@torch.no_grad()
def capture_base_last_token_activations(
    model: torch.nn.Module,
    tok: Any,
    mlp: torch.nn.Module,
    prompts: Sequence[str],
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Capture the ordinary SwiGLU activation entering ``down_proj``."""
    rows: List[torch.Tensor] = []
    captured: List[torch.Tensor] = []

    def pre_hook(_module: torch.nn.Module, inputs: Any) -> None:
        captured.append(inputs[0])

    handle = mlp.down_proj.register_forward_pre_hook(pre_hook)
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("activation capture requires a separate model backbone")
        for start in range(0, len(prompts), int(batch_size)):
            batch = list(prompts[start : start + int(batch_size)])
            encoded = tok(batch, padding=True, return_tensors="pt").to(device)
            captured.clear()
            backbone(**encoded, use_cache=False, return_dict=True)
            if len(captured) != 1:
                raise RuntimeError(
                    f"expected one down-projection activation, captured {len(captured)}"
                )
            activation = captured[0]
            positions = encoded["attention_mask"].sum(dim=1) - 1
            batch_rows = torch.arange(len(batch), device=device)
            rows.append(activation[batch_rows, positions, :].detach().float().cpu())
    finally:
        handle.remove()
        tok.padding_side = old_side
    if not rows:
        return torch.empty((0, int(mlp.gate_proj.weight.shape[0])))
    return torch.cat(rows, dim=0)


def capture_editor_last_token_activations(
    model: torch.nn.Module,
    tok: Any,
    editor: neuron_core.SparseSwiGLUNeuronEditor,
    prompts: Sequence[str],
    device: torch.device,
) -> torch.Tensor:
    old_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    editor.capture_activations = True
    try:
        encoded = tok(list(prompts), padding=True, return_tensors="pt").to(device)
        backbone = getattr(model, "model", None)
        if backbone is None or backbone is model:
            raise RuntimeError("detector training requires a separate model backbone")
        backbone(**encoded, use_cache=False, return_dict=True)
        activation = editor.last_edited_activations
        if activation is None:
            raise RuntimeError("sparse neuron editor did not capture activations")
        positions = encoded["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(len(prompts), device=device)
        return activation[rows, positions, :]
    finally:
        editor.capture_activations = False
        tok.padding_side = old_side


def _negative_prompt_instances(
    records: Sequence[Mapping[str, Any]],
    context_sets: Mapping[int, Mapping[str, Any]],
) -> List[mcf_repair.MCFPromptInstance]:
    instances: List[mcf_repair.MCFPromptInstance] = []
    for position, record in enumerate(records):
        case_id = int(record["case_id"])
        negatives = context_sets[case_id]["negative_contexts"]
        for prompt_index, row in enumerate(negatives):
            instances.append(
                mcf_repair.MCFPromptInstance(
                    record_index=case_id,
                    sampled_position=position,
                    prompt_type=str(row.get("kind") or "training_safe_negative"),
                    prompt_index=prompt_index,
                    prompt=str(row["prompt"]),
                    target_new=str(record["reference"]),
                    target_true=str(record["answer"]),
                )
            )
    return instances


def _failure_counts(
    margins: torch.Tensor,
    direct_flags: Sequence[bool],
    margin: float,
) -> Tuple[int, int]:
    threshold = float(margin) - 1e-6
    direct = sum(
        int(bool(direct_flags[index]) and float(value) < threshold)
        for index, value in enumerate(margins)
    )
    return direct, int((margins < threshold).sum())


def _topk_kl(
    logits: torch.Tensor,
    prompts: Sequence[str],
    cache: Mapping[str, Mapping[str, torch.Tensor]],
    device: torch.device,
) -> torch.Tensor:
    terms: List[torch.Tensor] = []
    for row, prompt in enumerate(prompts):
        ids = cache[str(prompt)]["top_ids"].to(device)
        target = cache[str(prompt)]["top_log_probs"].to(device)
        observed = torch.log_softmax(logits[row].float()[ids], dim=-1)
        terms.append(F.kl_div(observed, target, log_target=True, reduction="sum"))
    return torch.stack(terms).mean() if terms else logits.sum() * 0.0


def _replace_embedding_rows(
    layer: torch.nn.Module, token_ids: Sequence[int], values: torch.Tensor
) -> None:
    ids = torch.tensor(
        [int(value) for value in token_ids],
        dtype=torch.long,
        device=layer.weight.device,
    )
    with torch.no_grad():
        layer.weight.index_copy_(
            0, ids, values.to(device=layer.weight.device, dtype=layer.weight.dtype)
        )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_environment_firewall()
    gagd.set_seed(int(args.seed))
    gagd.require_cuda_if_needed(args.device_map)
    py_rng = random.Random(int(args.seed) + 78103)
    out_dir = gagd.resolve_output_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    visible_path = Path(args.training_visible_path).resolve()
    split_path = Path(args.split_manifest).resolve()
    context_path = Path(args.context_manifest).resolve()
    stage1_path = Path(args.stage1_state).resolve()
    registry_path = Path(args.experiment_registry).resolve()
    for path in (visible_path, split_path, context_path, stage1_path, registry_path):
        if not path.exists():
            raise FileNotFoundError(path)
    experiment_registry = _load_json(registry_path)
    _validate_experiment_registry(experiment_registry, args)

    locked_records, split_manifest = directional.validate_locked(
        visible_path, split_path, int(args.seed), int(args.forget_num)
    )
    if split_manifest.get("protocol") != locked_split.PROTOCOL:
        raise RuntimeError("neuron erasure requires the locked direct-only split")
    locked_split.assert_direct_only_training_view(locked_records)
    records = _record_views(locked_records)
    context_manifest = _load_json(context_path)
    stage1_state = torch.load(stage1_path, map_location="cpu", weights_only=False)
    if not isinstance(stage1_state, Mapping):
        raise RuntimeError("Stage-1 writer state must be a mapping")
    _validate_firewall(context_manifest, stage1_state)
    if compositional_method.sha256_file(context_path) != str(
        stage1_state["context_manifest_sha256"]
    ):
        raise RuntimeError(
            "Stage-1 writer and training-safe context manifest hashes differ"
        )
    visible_sha256 = compositional_method.sha256_file(visible_path)
    split_sha256 = compositional_method.sha256_file(split_path)
    if (
        str(context_manifest.get("source_training_visible_sha256") or "")
        != visible_sha256
    ):
        raise RuntimeError(
            "training-visible file is not the exact file bound into the Stage-1 "
            "context manifest"
        )
    if str(context_manifest.get("source_split_manifest_sha256") or "") != split_sha256:
        raise RuntimeError(
            "split manifest is not the exact manifest bound into the Stage-1 "
            "context manifest"
        )
    case_ids = [int(record["case_id"]) for record in records]
    if [int(value) for value in stage1_state.get("case_ids", [])] != case_ids:
        raise RuntimeError("Stage-1 cases do not match the locked training view")
    context_sets = _context_sets_by_case(context_manifest, records)

    firewall_receipt = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_training_firewall",
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "forget_num": len(records),
        "training_visible_path": str(visible_path),
        "training_visible_sha256": visible_sha256,
        "split_manifest_path": str(split_path),
        "split_manifest_sha256": split_sha256,
        "context_manifest_path": str(context_path),
        "context_manifest_sha256": compositional_method.sha256_file(context_path),
        "stage1_state_path": str(stage1_path),
        "stage1_state_sha256": compositional_method.sha256_file(stage1_path),
        "experiment_label": str(args.experiment_label),
        "experiment_registry_path": str(registry_path),
        "experiment_registry_sha256": compositional_method.sha256_file(registry_path),
        "official_evaluation_file_argument_exists": False,
        "forbidden_evaluation_environment_variables_present": [],
        "data_access": dict(context_manifest["data_access"]),
        "evaluation_only_unavailable_during_training": [
            "official paraphrases",
            "official neighborhoods",
            "benchmark retain prompts",
            "official PPL text",
            "aliases",
            "descriptions",
            "adversarial prompts",
            "relearning attacks",
        ],
        "used_for_checkpoint_selection": [
            "locked direct forget prompts",
            "training-safe synthetic/surrogate positives",
            "training-visible compositional negatives",
            "disjoint Wikipedia protection prefixes",
        ],
        "ppl_corpus_partition": {
            "official_evaluation_document_interval": [0, 20],
            "training_protection_document_interval": [
                int(args.frequency_doc_start),
                int(args.frequency_doc_start) + int(args.frequency_docs),
            ],
            "overlap": 0,
        },
    }
    gagd.write_json(out_dir / "training_firewall_receipt.json", firewall_receipt)

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    output_layer = canonical.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model has no input embedding layer")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("input embeddings and LM head remain tied")
    device = gagd.first_device(model)
    llama_like = canonical.is_llama_like(model, tok)
    mlp = _resolve_swiglu_mlp(model, int(args.neuron_layer))
    output_head_digest_before = _tensor_digest(output_layer.weight)

    selected_embedding_rows = [
        int(value) for value in stage1_state.get("selected_embedding_rows", [])
    ]
    embedding_delta = stage1_state.get("embedding_delta")
    if not selected_embedding_rows or not isinstance(embedding_delta, torch.Tensor):
        raise RuntimeError("Stage-1 state lacks sparse embedding rows/delta")
    if embedding_delta.shape != (
        len(selected_embedding_rows),
        int(input_layer.weight.shape[1]),
    ):
        raise RuntimeError("Stage-1 embedding delta has incompatible shape")
    embedding_writer = neuron_core.ToggleableEmbeddingDelta(
        input_layer, selected_embedding_rows, embedding_delta
    )

    (
        positive_instances,
        _owners,
        direct_flags,
    ) = compositional_method.build_prompt_instances(records, context_sets)
    negative_instances = _negative_prompt_instances(records, context_sets)
    all_training_prompts: List[str] = []
    positive_prompts_by_record: List[List[str]] = []
    negative_prompts_by_record: List[List[str]] = []
    for record in records:
        context = context_sets[int(record["case_id"])]
        positives = [str(prompt) for prompt in context["positive_prompts"]]
        negatives = [str(row["prompt"]) for row in context["negative_contexts"]]
        positive_prompts_by_record.append(positives)
        negative_prompts_by_record.append(negatives)
        all_training_prompts.extend(positives)
        all_training_prompts.extend(negatives)

    documents = subject_writer.load_frequency_documents(
        args.wikidata_dir, int(args.frequency_doc_start), int(args.frequency_docs)
    )
    if int(args.frequency_docs) > 0 and not documents:
        raise RuntimeError(
            f"no disjoint protection documents loaded from {args.wikidata_dir!r}"
        )
    corpus_prompts = synthetic.corpus_context_prefixes(
        documents,
        count=int(args.corpus_protection_prompts),
        seed=int(args.seed) + 1907,
    )
    protected_prompts = _unique_prompts(
        [
            *all_training_prompts,
            *corpus_prompts,
        ]
    )[: int(args.selection_protected_prompts)]

    print("\nStage 0: embedding-code neuron selection")
    selection_start = time.time()
    embedding_writer.enabled = False
    protected_off = capture_base_last_token_activations(
        model,
        tok,
        mlp,
        protected_prompts,
        device,
        batch_size=int(args.cache_batch_size),
    )
    record_writer_on: List[torch.Tensor] = []
    record_writer_off: List[torch.Tensor] = []
    for position, prompts in enumerate(positive_prompts_by_record):
        selected = prompts[: int(args.selection_positive_contexts)]
        embedding_writer.enabled = False
        off = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected,
            device,
            batch_size=int(args.cache_batch_size),
        )
        embedding_writer.enabled = True
        on = capture_base_last_token_activations(
            model,
            tok,
            mlp,
            selected,
            device,
            batch_size=int(args.cache_batch_size),
        )
        record_writer_off.append(off)
        record_writer_on.append(on)
        print(
            f"  case {case_ids[position]}: selection contexts={len(selected)}, "
            f"max writer activation displacement={float((on-off).abs().max()):.4f}"
        )
    ownership, sign_groups, selection_reports = neuron_core.select_record_owned_neurons(
        record_writer_on,
        record_writer_off,
        protected_off,
        neurons_per_record=int(args.neurons_per_record),
        dormant_fraction=float(args.dormant_fraction),
        stability_weight=float(args.selection_stability_weight),
        selection_mode=str(args.selection_mode),
        generator=torch.Generator(device="cpu").manual_seed(int(args.seed) + 4109),
    )
    selected_neurons, flat_signs_cpu, local_groups = neuron_core.flatten_ownership(
        ownership, sign_groups
    )
    selection_report = {
        "layer": int(args.neuron_layer),
        "zero_based_layer_index": True,
        "neurons_per_record": int(args.neurons_per_record),
        "selected_neuron_count": len(selected_neurons),
        "intermediate_size": int(mlp.gate_proj.weight.shape[0]),
        "dormant_fraction": float(args.dormant_fraction),
        "selection_mode": str(args.selection_mode),
        "protected_prompt_count": len(protected_prompts),
        "ownership": [
            {"case_id": case_ids[index], **row}
            for index, row in enumerate(selection_reports)
        ],
        "selection_wall_time_seconds": time.time() - selection_start,
    }
    gagd.write_json(out_dir / "neuron_selection_report.json", selection_report)
    print(
        f"  selected {len(selected_neurons)} disjoint existing neurons at layer "
        f"{args.neuron_layer} ({args.neurons_per_record} per record)"
    )

    editor = neuron_core.SparseSwiGLUNeuronEditor(mlp, selected_neurons)
    editor.install(mlp)
    flat_signs = flat_signs_cpu.to(device)
    base_selected_by_prompt: Dict[str, torch.Tensor] = {}
    embedding_writer.enabled = False
    all_prompt_order = _unique_prompts([*all_training_prompts, *corpus_prompts])
    base_selected = capture_base_last_token_activations(
        model,
        tok,
        mlp,
        all_prompt_order,
        device,
        batch_size=int(args.cache_batch_size),
    )[:, selected_neurons]
    for prompt, row in zip(all_prompt_order, base_selected):
        base_selected_by_prompt[prompt] = row
    embedding_writer.enabled = True

    print("\nStage 1: train sparse nonlinear contextual-code detector")
    editor.write_enabled = False
    editor.gate_delta.requires_grad_(True)
    editor.up_delta.requires_grad_(True)
    editor.down_delta.requires_grad_(False)
    detector_optimizer = torch.optim.AdamW(
        [editor.gate_delta, editor.up_delta],
        lr=float(args.detector_lr),
        weight_decay=0.0,
    )
    detector_log: List[Dict[str, float]] = []
    for step in range(1, int(args.detector_steps) + 1):
        record_indices = py_rng.sample(
            range(len(records)), min(int(args.detector_record_batch), len(records))
        )
        prompts: List[str] = []
        batch_owners: List[int] = []
        positive_flags: List[bool] = []
        positive_batch: List[str] = []
        for record_index in record_indices:
            positives = list(positive_prompts_by_record[record_index])
            negatives = list(negative_prompts_by_record[record_index])
            direct = positives[0]
            remaining = positives[1:]
            py_rng.shuffle(remaining)
            chosen_positive = [direct, *remaining][
                : int(args.detector_positive_contexts)
            ]
            py_rng.shuffle(negatives)
            chosen_negative = negatives[: int(args.detector_negative_contexts)]
            for prompt in chosen_positive:
                prompts.append(prompt)
                positive_batch.append(prompt)
                batch_owners.append(record_index)
                positive_flags.append(True)
            for prompt in chosen_negative:
                prompts.append(prompt)
                batch_owners.append(record_index)
                positive_flags.append(False)

        detector_optimizer.zero_grad(set_to_none=True)
        embedding_writer.enabled = True
        activation = capture_editor_last_token_activations(
            model, tok, editor, prompts, device
        )
        baseline = torch.stack(
            [base_selected_by_prompt[prompt] for prompt in prompts]
        ).to(device)
        responses = neuron_core.contextual_code_responses(
            activation, baseline, local_groups, flat_signs
        )
        owner_tensor = torch.tensor(batch_owners, dtype=torch.long, device=device)
        positive_tensor = torch.tensor(positive_flags, dtype=torch.bool, device=device)
        detector_loss, pieces = neuron_core.detector_objective(
            responses,
            owner_tensor,
            positive_tensor,
            positive_floor=float(args.detector_positive_floor),
            negative_weight=float(args.detector_negative_weight),
            cross_weight=float(args.detector_cross_weight),
        )
        consistency = responses.sum() * 0.0
        for record_index in record_indices:
            mask = positive_tensor & owner_tensor.eq(record_index)
            if int(mask.sum()) > 1:
                consistency = consistency + responses[mask, record_index].var(
                    unbiased=False
                )
        consistency = consistency / max(1, len(record_indices))

        embedding_writer.enabled = False
        off_activation = capture_editor_last_token_activations(
            model, tok, editor, positive_batch, device
        )
        off_baseline = torch.stack(
            [base_selected_by_prompt[prompt] for prompt in positive_batch]
        ).to(device)
        off_response = neuron_core.contextual_code_responses(
            off_activation, off_baseline, local_groups, flat_signs
        )
        writer_off_loss = off_response.square().mean()
        embedding_writer.enabled = True
        l2 = editor.gate_delta.square().mean() + editor.up_delta.square().mean()
        total = (
            detector_loss
            + float(args.detector_consistency_weight) * consistency
            + float(args.detector_writer_off_weight) * writer_off_loss
            + float(args.detector_l2) * l2
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite detector loss at step {step}")
        total.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                [editor.gate_delta, editor.up_delta], float(args.grad_clip)
            )
        detector_optimizer.step()
        cap = editor.clamp_relative_(
            detector_cap=float(args.detector_relative_cap),
            actuator_cap=float(args.actuator_relative_cap),
        )
        if step == 1 or step % 25 == 0 or step == int(args.detector_steps):
            row = {
                "step": step,
                "loss": float(total.detach()),
                "write": float(pieces["write"].detach()),
                "negative": float(pieces["negative"].detach()),
                "cross": float(pieces["cross"].detach()),
                "consistency": float(consistency.detach()),
                "writer_off": float(writer_off_loss.detach()),
                "gate_max_relative_norm": cap["gate_max_relative_norm"],
                "up_max_relative_norm": cap["up_max_relative_norm"],
            }
            detector_log.append(row)
            print(
                f"  step {step:>4}: loss {row['loss']:.4f}, "
                f"write {row['write']:.4f}, off {row['writer_off']:.4f}, "
                f"cross {row['cross']:.4f}"
            )
    del detector_optimizer

    @torch.no_grad()
    def record_detector_responses(
        prompt_groups: Sequence[Sequence[str]], *, writer_enabled: bool
    ) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        embedding_writer.enabled = bool(writer_enabled)
        editor.write_enabled = False
        for record_index, prompts in enumerate(prompt_groups):
            rows: List[torch.Tensor] = []
            for start in range(0, len(prompts), int(args.cache_batch_size)):
                batch = list(prompts[start : start + int(args.cache_batch_size)])
                activation = capture_editor_last_token_activations(
                    model, tok, editor, batch, device
                )
                baseline = torch.stack(
                    [base_selected_by_prompt[prompt] for prompt in batch]
                ).to(device)
                response = neuron_core.contextual_code_responses(
                    activation, baseline, local_groups, flat_signs
                )
                rows.append(response[:, record_index].detach().cpu())
            result.append(torch.cat(rows) if rows else torch.empty(0))
        embedding_writer.enabled = True
        return result

    positive_detector = record_detector_responses(
        positive_prompts_by_record, writer_enabled=True
    )
    negative_detector = record_detector_responses(
        negative_prompts_by_record, writer_enabled=True
    )
    writer_off_detector = record_detector_responses(
        positive_prompts_by_record, writer_enabled=False
    )
    detector_gate = neuron_core.detector_gate_report(
        positive_detector,
        negative_detector,
        writer_off_detector,
        positive_floor=float(args.detector_positive_floor),
        off_abs_max=float(args.detector_off_abs_max),
    )
    detector_gate["kind"] = "training_only_embedding_code_detector_gate"
    detector_gate["official_evaluation_prompts_seen"] = 0
    gagd.write_json(out_dir / "detector_gate_report.json", detector_gate)
    print(
        f"  detector gate: {detector_gate['passed_records']}/"
        f"{detector_gate['total_records']} records"
    )

    print("\nStage 2: train sparse MLP erasure actuator; LM head frozen")
    # Every Stage-2 preservation target is the writer-only model.  Detector
    # gate/up edits are part of the learned MLP eraser and therefore may not be
    # smuggled into the reference baseline.
    editor.enabled = False
    editor.gate_delta.requires_grad_(False)
    editor.up_delta.requires_grad_(False)
    editor.down_delta.requires_grad_(True)
    embedding_writer.enabled = True
    pre_target_new, pre_target_true = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    pre_negative_new, pre_negative_true = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        negative_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    protected_for_kl = _unique_prompts(
        [
            *[instance.prompt for instance in negative_instances],
            *corpus_prompts,
        ]
    )
    writer_only_cache = compositional_method.cache_prompt_baselines(
        model,
        tok,
        protected_for_kl,
        device,
        batch_size=int(args.cache_batch_size),
        topk=int(args.kl_topk),
    )
    editor.enabled = True
    editor.write_enabled = True

    actuator_optimizer = torch.optim.AdamW(
        [editor.down_delta], lr=float(args.actuator_lr), weight_decay=0.0
    )
    actuator_log: List[Dict[str, float]] = []
    positive_order = list(range(len(positive_instances)))
    negative_order = list(range(len(negative_instances)))
    protected_order = list(range(len(protected_for_kl)))
    for step in range(1, int(args.actuator_steps) + 1):
        positive_indices = py_rng.sample(
            positive_order,
            min(int(args.actuator_batch_size), len(positive_order)),
        )
        negative_indices = py_rng.sample(
            negative_order,
            min(int(args.actuator_batch_size), len(negative_order)),
        )
        protected_indices = py_rng.sample(
            protected_order,
            min(int(args.actuator_protected_batch), len(protected_order)),
        )
        positive_batch = [positive_instances[index] for index in positive_indices]
        negative_batch = [negative_instances[index] for index in negative_indices]
        protected_batch = [protected_for_kl[index] for index in protected_indices]

        actuator_optimizer.zero_grad(set_to_none=True)
        embedding_writer.enabled = True
        current_new, current_true = compositional_method.differentiable_instance_nlls(
            model, tok, positive_batch, device, llama_like=llama_like
        )
        margins = current_true - current_new
        margin_loss = F.relu(float(args.forget_margin) - margins).square().mean()
        reference_loss = compositional_method.reference_nll_regression_penalty(
            current_new,
            pre_target_new[positive_indices].to(device),
            float(args.reference_nll_tolerance),
        )
        negative_new, negative_true = compositional_method.differentiable_instance_nlls(
            model, tok, negative_batch, device, llama_like=llama_like
        )
        negative_preservation = (
            negative_new - pre_negative_new[negative_indices].to(device)
        ).square().mean() + (
            negative_true - pre_negative_true[negative_indices].to(device)
        ).square().mean()
        _hidden, protected_logits = compositional_method.forward_last_hidden_logits(
            model, tok, protected_batch, device
        )
        protected_kl = _topk_kl(
            protected_logits, protected_batch, writer_only_cache, device
        )

        writer_off_loss = margins.sum() * 0.0
        if (
            int(args.actuator_writer_off_every) > 0
            and step % int(args.actuator_writer_off_every) == 0
        ):
            embedding_writer.enabled = False
            off_new, off_true = compositional_method.differentiable_instance_nlls(
                model, tok, positive_batch, device, llama_like=llama_like
            )
            # With the contextual key removed, the neuron edit must preserve
            # the writer-off behavior rather than becoming a stand-alone MLP edit.
            editor.write_enabled = False
            (
                base_off_new,
                base_off_true,
            ) = compositional_method.differentiable_instance_nlls(
                model, tok, positive_batch, device, llama_like=llama_like
            )
            editor.write_enabled = True
            writer_off_loss = (off_new - base_off_new.detach()).square().mean() + (
                off_true - base_off_true.detach()
            ).square().mean()
            embedding_writer.enabled = True

        l2 = editor.down_delta.square().mean()
        total = (
            float(args.margin_weight) * margin_loss
            + float(args.reference_nll_weight) * reference_loss
            + negative_preservation
            + float(args.protected_kl_weight) * protected_kl
            + float(args.writer_off_nll_weight) * writer_off_loss
            + float(args.actuator_l2) * l2
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite actuator loss at step {step}")
        total.backward()
        if float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_([editor.down_delta], float(args.grad_clip))
        actuator_optimizer.step()
        cap = editor.clamp_relative_(
            detector_cap=float(args.detector_relative_cap),
            actuator_cap=float(args.actuator_relative_cap),
        )
        if step == 1 or step % 25 == 0 or step == int(args.actuator_steps):
            row = {
                "step": step,
                "loss": float(total.detach()),
                "margin": float(margin_loss.detach()),
                "reference": float(reference_loss.detach()),
                "negative_preservation": float(negative_preservation.detach()),
                "protected_kl": float(protected_kl.detach()),
                "writer_off": float(writer_off_loss.detach()),
                "down_max_relative_norm": cap["down_max_relative_norm"],
            }
            actuator_log.append(row)
            print(
                f"  step {step:>4}: loss {row['loss']:.4f}, "
                f"margin {row['margin']:.4f}, KL {row['protected_kl']:.4f}, "
                f"writer-off {row['writer_off']:.4f}"
            )
        if int(args.check_every) > 0 and step % int(args.check_every) == 0:
            current_margins = compositional_method.evaluate_instance_margins(
                model,
                tok,
                positive_instances,
                device,
                llama_like=llama_like,
                batch_size=int(args.cache_batch_size),
            )
            direct_failure, positive_failure = _failure_counts(
                current_margins, direct_flags, float(args.forget_margin)
            )
            print(
                f"    training audit: direct fail {direct_failure}, "
                f"all-positive fail {positive_failure}, "
                f"min margin {float(current_margins.min()):+.4f}"
            )
    del actuator_optimizer

    embedding_writer.enabled = True
    editor.enabled = True
    editor.write_enabled = True

    # Training-only causal ablations.  Every configuration uses the same fixed
    # learned tensors; none is selected using official evaluation probes.
    def evaluate_ablation(writer: bool, decoder: bool) -> torch.Tensor:
        embedding_writer.enabled = bool(writer)
        editor.enabled = bool(decoder)
        return compositional_method.evaluate_instance_margins(
            model,
            tok,
            positive_instances,
            device,
            llama_like=llama_like,
            batch_size=int(args.cache_batch_size),
        )

    full_hook_margins = evaluate_ablation(True, True)
    embedding_only_margins = evaluate_ablation(True, False)
    neuron_only_margins = evaluate_ablation(False, True)
    reconstructed_base_margins = evaluate_ablation(False, False)
    embedding_writer.enabled = True
    editor.enabled = True
    full_direct_failures, full_positive_failures = _failure_counts(
        full_hook_margins, direct_flags, float(args.forget_margin)
    )
    embedding_only_direct, embedding_only_positive = _failure_counts(
        embedding_only_margins, direct_flags, float(args.forget_margin)
    )
    neuron_only_direct, neuron_only_positive = _failure_counts(
        neuron_only_margins, direct_flags, float(args.forget_margin)
    )
    base_direct, base_positive = _failure_counts(
        reconstructed_base_margins, direct_flags, float(args.forget_margin)
    )
    causal_ablation = {
        "kind": "locked_training_only_four_way_causal_component_ablation",
        "configurations": {
            "full_embedding_plus_neuron": {
                "direct_failures": full_direct_failures,
                "positive_failures": full_positive_failures,
                "minimum_margin": float(full_hook_margins.min()),
            },
            "embedding_only": {
                "direct_failures": embedding_only_direct,
                "positive_failures": embedding_only_positive,
                "minimum_margin": float(embedding_only_margins.min()),
            },
            "neuron_only": {
                "direct_failures": neuron_only_direct,
                "positive_failures": neuron_only_positive,
                "minimum_margin": float(neuron_only_margins.min()),
            },
            "reconstructed_base": {
                "direct_failures": base_direct,
                "positive_failures": base_positive,
                "minimum_margin": float(reconstructed_base_margins.min()),
            },
        },
        "writer_margin_gain": _distribution(
            [float(value) for value in full_hook_margins - neuron_only_margins]
        ),
        "decoder_margin_gain": _distribution(
            [float(value) for value in full_hook_margins - embedding_only_margins]
        ),
        "writer_is_necessary": bool(
            neuron_only_direct
            >= math.ceil(
                float(args.min_writer_necessary_direct_fraction) * len(records)
            )
        ),
        "decoder_is_necessary": bool(
            embedding_only_direct
            >= math.ceil(
                float(args.min_decoder_necessary_direct_fraction) * len(records)
            )
        ),
        "official_evaluation_prompts_seen": 0,
    }
    gagd.write_json(out_dir / "causal_component_ablation.json", causal_ablation)

    print("\nMaterialization: ordinary embedding and SwiGLU weights")
    editor.remove()
    embedding_writer.remove()
    input_index = torch.tensor(
        selected_embedding_rows, dtype=torch.long, device=input_layer.weight.device
    )
    base_input_rows = input_layer.weight.index_select(0, input_index).detach().clone()
    directional.materialize_input_delta(
        input_layer, selected_embedding_rows, embedding_delta
    )
    edited_input_rows = input_layer.weight.index_select(0, input_index).detach().clone()
    base_neuron_weights = neuron_core.SparseNeuronWeights(
        editor.base_gate_rows.detach().to(mlp.gate_proj.weight.dtype),
        editor.base_up_rows.detach().to(mlp.up_proj.weight.dtype),
        editor.base_down_columns.detach().to(mlp.down_proj.weight.dtype),
    )
    edited_neuron_weights = editor.materialize(mlp)
    (
        materialized_target_new,
        materialized_target_true,
    ) = compositional_method.evaluate_instance_nlls(
        model,
        tok,
        positive_instances,
        device,
        llama_like=llama_like,
        batch_size=int(args.cache_batch_size),
    )
    materialized_margins = materialized_target_true - materialized_target_new
    materialized_direct_failures, materialized_positive_failures = _failure_counts(
        materialized_margins, direct_flags, float(args.forget_margin)
    )
    hook_materialization_drift = float(
        (materialized_margins - full_hook_margins).abs().max()
    )
    reference_drift = materialized_target_new - pre_target_new
    reference_regression_max = max(0.0, float(reference_drift.max()))
    output_head_digest_after = _tensor_digest(output_layer.weight)
    output_head_unchanged = output_head_digest_before == output_head_digest_after
    if not output_head_unchanged:
        raise RuntimeError("LM head changed despite the architectural invariant")

    norm_report = editor.relative_norm_report()
    model_parameter_count = sum(
        int(parameter.numel()) for parameter in model.parameters()
    )
    embedding_edited_scalar_count = int(edited_input_rows.numel())
    neuron_edited_scalar_count = int(
        edited_neuron_weights.gate_rows.numel()
        + edited_neuron_weights.up_rows.numel()
        + edited_neuron_weights.down_columns.numel()
    )
    total_edited_scalar_count = (
        embedding_edited_scalar_count + neuron_edited_scalar_count
    )
    norm_cap_passed = bool(
        norm_report["gate_max_relative_norm"]
        <= float(args.detector_relative_cap) + 1e-6
        and norm_report["up_max_relative_norm"]
        <= float(args.detector_relative_cap) + 1e-6
        and norm_report["down_max_relative_norm"]
        <= float(args.actuator_relative_cap) + 1e-6
    )
    detector_policy_passed = bool(
        detector_gate["passed"] or args.gate_policy == "report"
    )
    training_passed = bool(
        materialized_direct_failures == 0
        and materialized_positive_failures == 0
        and reference_regression_max <= float(args.reference_nll_tolerance) + 1e-6
        and norm_cap_passed
        and causal_ablation["writer_is_necessary"]
        and causal_ablation["decoder_is_necessary"]
        and detector_policy_passed
        and output_head_unchanged
        and hook_materialization_drift
        <= float(args.hook_materialization_tolerance) + 1e-6
    )

    checkpoint_path = out_dir / "checkpoint"
    checkpoint_saved = bool(args.save_checkpoint and training_passed)
    if checkpoint_saved:
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tok.save_pretrained(checkpoint_path)

    actual_input_delta = (
        edited_input_rows.detach().float().cpu()
        - base_input_rows.detach().float().cpu()
    )
    state_path = out_dir / "embedding_keyed_neuron_state.pt"
    torch.save(
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "seed": int(args.seed),
            "case_ids": case_ids,
            "layer": int(args.neuron_layer),
            "selected_embedding_rows": selected_embedding_rows,
            "base_selected_embedding_rows": base_input_rows.detach().cpu(),
            "edited_selected_embedding_rows": edited_input_rows.detach().cpu(),
            "actual_embedding_delta": actual_input_delta,
            "selected_neurons": selected_neurons,
            "ownership": ownership,
            "flat_signs": flat_signs_cpu,
            "base_neuron_weights": {
                "gate_rows": base_neuron_weights.gate_rows.detach().cpu(),
                "up_rows": base_neuron_weights.up_rows.detach().cpu(),
                "down_columns": base_neuron_weights.down_columns.detach().cpu(),
            },
            "edited_neuron_weights": {
                "gate_rows": edited_neuron_weights.gate_rows.detach().cpu(),
                "up_rows": edited_neuron_weights.up_rows.detach().cpu(),
                "down_columns": edited_neuron_weights.down_columns.detach().cpu(),
            },
            "gate_delta": editor.gate_delta.detach().cpu(),
            "up_delta": editor.up_delta.detach().cpu(),
            "down_delta": editor.down_delta.detach().cpu(),
            "detector_relative_cap": float(args.detector_relative_cap),
            "actuator_relative_cap": float(args.actuator_relative_cap),
            "hook_materialization_tolerance": float(
                args.hook_materialization_tolerance
            ),
            "context_manifest_sha256": compositional_method.sha256_file(context_path),
            "training_visible_sha256": compositional_method.sha256_file(visible_path),
            "output_head_sha256": output_head_digest_after,
        },
        state_path,
    )

    summary = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "seed": int(args.seed),
        "forget_num": len(records),
        "model_path": str(Path(args.model_path).resolve()),
        "architecture": {
            "tokenizer_expanded": False,
            "new_token_added": False,
            "runtime_string_matcher": False,
            "external_router": False,
            "retrieval_cache": False,
            "sidecar": False,
            "adapter": False,
            "lora": False,
            "logit_bias": False,
            "lm_head_edited": False,
            "lm_head_sha256_before": output_head_digest_before,
            "lm_head_sha256_after": output_head_digest_after,
            "input_embedding_rows_edited": len(selected_embedding_rows),
            "mlp_layer": int(args.neuron_layer),
            "selected_existing_mlp_neurons": len(selected_neurons),
            "neurons_per_record": int(args.neurons_per_record),
            "base_model_parameter_count": model_parameter_count,
            "embedding_edited_scalar_count": embedding_edited_scalar_count,
            "mlp_edited_scalar_count": neuron_edited_scalar_count,
            "total_edited_scalar_count": total_edited_scalar_count,
            "edited_parameter_fraction": (
                total_edited_scalar_count / max(1, model_parameter_count)
            ),
            "edited_mlp_tensors": [
                f"model.layers.{args.neuron_layer}.mlp.gate_proj.weight[selected rows]",
                f"model.layers.{args.neuron_layer}.mlp.up_proj.weight[selected rows]",
                f"model.layers.{args.neuron_layer}.mlp.down_proj.weight[selected columns]",
            ],
            "writer": "ordinary global sparse subject embedding-row deltas",
            "context_composer": "frozen Transformer layers before the selected MLP",
            "detector": "record-owned sparse existing SwiGLU gate/up rows",
            "actuator": "matching sparse existing SwiGLU down columns",
            "decoder": "unchanged base LM head",
        },
        "data_firewall": firewall_receipt,
        "selection": selection_report,
        "detector": {
            "training_log": detector_log,
            "gate": detector_gate,
            "relative_norm_cap": float(args.detector_relative_cap),
        },
        "actuator": {
            "training_log": actuator_log,
            "relative_norm_cap": float(args.actuator_relative_cap),
            "norms": norm_report,
            "forget_margin": float(args.forget_margin),
            "reference_nll_tolerance": float(args.reference_nll_tolerance),
            "reference_nll_drift": _distribution(
                [float(value) for value in reference_drift]
            ),
            "reference_nll_regression_max": reference_regression_max,
            "hook_to_materialized_margin_abs_max": hook_materialization_drift,
            "hook_materialization_tolerance": float(
                args.hook_materialization_tolerance
            ),
            "direct_failures": materialized_direct_failures,
            "training_safe_positive_failures": materialized_positive_failures,
            "minimum_margin": float(materialized_margins.min()),
        },
        "causal_component_ablation": causal_ablation,
        "acceptance": {
            "zero_direct_failures": materialized_direct_failures == 0,
            "zero_training_safe_positive_failures": materialized_positive_failures == 0,
            "reference_nll_regression_within_tolerance": bool(
                reference_regression_max <= float(args.reference_nll_tolerance) + 1e-6
            ),
            "detector_gate_passed": bool(detector_gate["passed"]),
            "detector_gate_policy_passed": detector_policy_passed,
            "writer_necessary": bool(causal_ablation["writer_is_necessary"]),
            "decoder_necessary": bool(causal_ablation["decoder_is_necessary"]),
            "hard_relative_norm_caps_passed": norm_cap_passed,
            "lm_head_bit_identical": output_head_unchanged,
            "hook_materialization_within_tolerance": bool(
                hook_materialization_drift
                <= float(args.hook_materialization_tolerance) + 1e-6
            ),
            "checkpoint_saved": checkpoint_saved,
            "passed": training_passed,
        },
        "checkpoint": str(checkpoint_path) if checkpoint_saved else None,
        "state": str(state_path),
        "claim_boundary": (
            "Training establishes a causal embedding-key to sparse-neuron pathway "
            "only on locked training-safe contexts. Unseen paraphrase, locality, "
            "PPL, alias, adversarial, latent-probe, and relearning behavior remains "
            "unknown until separate post-checkpoint evaluation."
        ),
    }
    gagd.write_json(out_dir / "embedding_keyed_neuron_summary.json", summary)

    print("\n" + "=" * 76)
    print(f"  materialized direct failures          : {materialized_direct_failures}")
    print(f"  materialized positive failures        : {materialized_positive_failures}")
    print(
        f"  minimum training margin               : {float(materialized_margins.min()):+.4f}"
    )
    print(f"  maximum reference NLL regression      : {reference_regression_max:.4f}")
    print(
        f"  without embedding writer direct fails : {neuron_only_direct}/{len(records)}"
    )
    print(
        f"  without neuron decoder direct fails   : {embedding_only_direct}/{len(records)}"
    )
    print(f"  LM head bit-identical                 : {output_head_unchanged}")
    print(f"  selected existing neurons             : {len(selected_neurons)}")
    print("  evaluation probes opened              : 0")
    print("=" * 76)
    if not training_passed:
        raise SystemExit(
            "embedding-keyed sparse neuron erasure failed its locked training-only "
            "acceptance; official evaluation is refused"
        )
    print(f"checkpoint: {checkpoint_path}")
    print(f"summary: {out_dir / 'embedding_keyed_neuron_summary.json'}")


if __name__ == "__main__":
    main()
