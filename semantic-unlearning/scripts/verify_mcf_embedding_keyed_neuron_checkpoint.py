#!/usr/bin/env python3
"""Fresh-process verification for an embedding-keyed neuron checkpoint.

This verifier opens only the locked direct view, its audited training-safe
context manifest, the learned-state receipt, and the serialized checkpoint.
It has no original-MCF/evaluation argument.  Official paraphrases,
neighborhoods, retain prompts, PPL text, aliases, and attacks remain unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import build_mcf_sure_target_aware_direct_split as locked_split
import gagd_compare as gagd
import mcf_compositional_marker_write_read as compositional
import mcf_embedding_keyed_neuron_core as core
import mcf_embedding_keyed_neuron_erasure as method
import sure_canonical_core as canonical


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--context-manifest", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--forget-margin", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    value = parser.parse_args(list(argv) if argv is not None else None)
    if int(value.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _tensor_digest(tensor: torch.Tensor, row_chunk: int = 256) -> str:
    digest = hashlib.sha256()
    detached = tensor.detach()
    for start in range(0, int(detached.shape[0]), int(row_chunk)):
        block = detached[start : start + int(row_chunk)].contiguous().cpu()
        digest.update(block.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _state_weights(state: Mapping[str, Any], name: str) -> core.SparseNeuronWeights:
    value = state.get(name)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"state lacks {name}")
    gate = value.get("gate_rows")
    up = value.get("up_rows")
    down = value.get("down_columns")
    if not all(isinstance(tensor, torch.Tensor) for tensor in (gate, up, down)):
        raise RuntimeError(f"state {name} is incomplete")
    return core.SparseNeuronWeights(gate, up, down)


def _exact_tensor_match(current: torch.Tensor, expected: torch.Tensor) -> bool:
    return bool(
        torch.equal(
            current.detach().cpu(), expected.detach().to(dtype=current.dtype).cpu()
        )
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    visible_path = Path(args.training_visible_path).resolve()
    context_path = Path(args.context_manifest).resolve()
    state_path = Path(args.state).resolve()
    checkpoint_path = Path(args.model_dir).resolve()
    for path in (visible_path, context_path, state_path, checkpoint_path):
        if not path.exists():
            raise FileNotFoundError(path)

    raw_records = json.loads(visible_path.read_text(encoding="utf-8"))
    if not isinstance(raw_records, list):
        raise RuntimeError("training-visible artifact must be a JSON list")
    locked_split.assert_direct_only_training_view(raw_records)
    records = compositional._record_views(raw_records)
    context_manifest = _load_json(context_path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("protocol") != method.PROTOCOL:
        raise RuntimeError("embedding-keyed neuron state protocol mismatch")
    method._validate_firewall(context_manifest, state)
    if _sha256(context_path) != str(state.get("context_manifest_sha256")):
        raise RuntimeError("context manifest is not the exact training-bound artifact")
    if _sha256(visible_path) != str(state.get("training_visible_sha256")):
        raise RuntimeError("training-visible artifact hash differs from learned state")
    if [int(record["case_id"]) for record in records] != [
        int(value) for value in state.get("case_ids", [])
    ]:
        raise RuntimeError("learned-state case IDs do not match locked records")
    contexts = method._context_sets_by_case(context_manifest, records)
    instances, _owners, direct_flags = compositional.build_prompt_instances(
        records, contexts
    )

    namespace = argparse.Namespace(
        model_path=str(checkpoint_path),
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    mlp = method._resolve_swiglu_mlp(model, int(state["layer"]))
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise RuntimeError("reloaded model lacks input/output embeddings")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("checkpoint unexpectedly retied input and output embeddings")

    embedding_ids = [int(value) for value in state["selected_embedding_rows"]]
    embedding_index = torch.tensor(
        embedding_ids, dtype=torch.long, device=input_layer.weight.device
    )
    current_embedding = input_layer.weight.index_select(0, embedding_index)
    embedding_exact = _exact_tensor_match(
        current_embedding, state["edited_selected_embedding_rows"]
    )

    neurons = [int(value) for value in state["selected_neurons"]]
    current_neurons = core.sparse_neuron_weights(mlp, neurons)
    expected_neurons = _state_weights(state, "edited_neuron_weights")
    gate_exact = _exact_tensor_match(
        current_neurons.gate_rows, expected_neurons.gate_rows
    )
    up_exact = _exact_tensor_match(current_neurons.up_rows, expected_neurons.up_rows)
    down_exact = _exact_tensor_match(
        current_neurons.down_columns, expected_neurons.down_columns
    )

    base_neurons = _state_weights(state, "base_neuron_weights")
    gate_relative = (
        current_neurons.gate_rows.float().cpu() - base_neurons.gate_rows.float()
    ).norm(dim=1) / base_neurons.gate_rows.float().norm(dim=1).clamp_min(1e-30)
    up_relative = (
        current_neurons.up_rows.float().cpu() - base_neurons.up_rows.float()
    ).norm(dim=1) / base_neurons.up_rows.float().norm(dim=1).clamp_min(1e-30)
    down_relative = (
        current_neurons.down_columns.float().cpu() - base_neurons.down_columns.float()
    ).norm(dim=0) / base_neurons.down_columns.float().norm(dim=0).clamp_min(1e-30)
    detector_cap = float(state["detector_relative_cap"])
    actuator_cap = float(state["actuator_relative_cap"])
    cap_passed = bool(
        float(gate_relative.max()) <= detector_cap + 1e-6
        and float(up_relative.max()) <= detector_cap + 1e-6
        and float(down_relative.max()) <= actuator_cap + 1e-6
    )
    output_digest = _tensor_digest(output_layer.weight)
    output_head_exact = output_digest == str(state.get("output_head_sha256"))

    margins = compositional.evaluate_instance_margins(
        model,
        tok,
        instances,
        device,
        llama_like=canonical.is_llama_like(model, tok),
        batch_size=int(args.batch_size),
    )
    threshold = float(args.forget_margin) - 1e-6
    failures = margins < threshold
    direct_failures = sum(
        int(bool(direct_flags[index]) and bool(failures[index]))
        for index in range(len(direct_flags))
    )
    positive_failures = int(failures.sum())
    passed = bool(
        direct_failures == 0
        and positive_failures == 0
        and embedding_exact
        and gate_exact
        and up_exact
        and down_exact
        and cap_passed
        and output_head_exact
    )
    payload = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_post_reload_acceptance",
        "checkpoint_was_reloaded": True,
        "official_compatible_nll_arithmetic": True,
        "observed": {
            "direct_prompt_instances": int(sum(bool(value) for value in direct_flags)),
            "training_safe_positive_instances": len(direct_flags),
            "direct_failures": direct_failures,
            "training_safe_positive_failures": positive_failures,
            "minimum_margin": float(margins.min()),
        },
        "serialization": {
            "embedding_rows_exact": embedding_exact,
            "gate_rows_exact": gate_exact,
            "up_rows_exact": up_exact,
            "down_columns_exact": down_exact,
            "lm_head_digest_exact": output_head_exact,
            "hard_relative_caps_passed": cap_passed,
            "gate_max_relative_norm": float(gate_relative.max()),
            "up_max_relative_norm": float(up_relative.max()),
            "down_max_relative_norm": float(down_relative.max()),
        },
        "data_access": {
            "official_paraphrases_seen": 0,
            "official_neighborhoods_seen": 0,
            "benchmark_retain_seen": 0,
            "official_ppl_seen": False,
            "adversarial_or_alias_probes_seen": 0,
        },
        "passed": passed,
    }
    output_path = Path(args.out).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"post-reload acceptance: {output_path}")
    if not passed:
        raise SystemExit("reloaded checkpoint failed locked training-only acceptance")


if __name__ == "__main__":
    main()
