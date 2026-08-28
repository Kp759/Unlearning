#!/usr/bin/env python3
"""Post-checkpoint PPL attribution for embedding-keyed neuron erasure.

This diagnostic is intentionally downstream of checkpoint freezing.  It opens
the official PPL text only to report the four fixed causal configurations:
full method, embedding writer only, sparse neuron decoder only, and reconstructed
base.  Its output is never consumed by training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_embedding_keyed_neuron_core as core
import mcf_embedding_keyed_neuron_erasure as method
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    load_official_ppl_text,
    official_perplexity,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    parser.add_argument("--max-input-length", type=int, default=100)
    value = parser.parse_args(list(argv) if argv is not None else None)
    if int(value.max_input_length) <= 1:
        parser.error("--max-input-length must exceed one")
    return value


def _matrix(state: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError(f"state lacks matrix {key!r}")
    return value.detach().cpu()


def _neuron_weights(state: Mapping[str, Any], key: str) -> core.SparseNeuronWeights:
    value = state.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"state lacks neuron weights {key!r}")
    tensors = [value.get(name) for name in ("gate_rows", "up_rows", "down_columns")]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise RuntimeError(f"state neuron weights {key!r} are incomplete")
    return core.SparseNeuronWeights(*[tensor.detach().cpu() for tensor in tensors])


def _percent_delta(value: float, base: float) -> float:
    return 100.0 * (float(value) - float(base)) / max(abs(float(base)), 1e-12)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    state_path = Path(args.state).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    for path in (model_dir, state_path, wikidata_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("protocol") != method.PROTOCOL:
        raise RuntimeError("embedding-keyed neuron state protocol mismatch")

    input_ids = [int(value) for value in state["selected_embedding_rows"]]
    neurons = [int(value) for value in state["selected_neurons"]]
    base_input = _matrix(state, "base_selected_embedding_rows")
    edited_input = _matrix(state, "edited_selected_embedding_rows")
    base_neurons = _neuron_weights(state, "base_neuron_weights")
    edited_neurons = _neuron_weights(state, "edited_neuron_weights")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs: Dict[str, Any] = {"torch_dtype": dtype_from_str(args.dtype)}
    if args.device_map == "auto":
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **kwargs)
    if args.device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")
    model.eval()
    model.config.use_cache = False
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise RuntimeError("model lacks input/output embeddings")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("component attribution requires an untied unchanged LM head")
    mlp = method._resolve_swiglu_mlp(model, int(state["layer"]))
    ppl_text = load_official_ppl_text(wikidata_dir)
    if ppl_text is None:
        raise RuntimeError(f"could not load official PPL text from {wikidata_dir}")
    device = next(model.parameters()).device

    modes = {
        "full_embedding_plus_neuron": (edited_input, edited_neurons),
        "embedding_only": (edited_input, base_neurons),
        "neuron_only": (base_input, edited_neurons),
        "reconstructed_base": (base_input, base_neurons),
    }
    values: Dict[str, float] = {}
    try:
        for label, (input_rows, neuron_rows) in modes.items():
            method._replace_embedding_rows(input_layer, input_ids, input_rows)
            core.replace_sparse_neuron_weights(mlp, neurons, neuron_rows)
            values[label] = float(
                official_perplexity(
                    model,
                    tokenizer,
                    ppl_text,
                    device,
                    max_input_length=int(args.max_input_length),
                )
            )
            print(f"{label:>29}: {values[label]:.6f}")
    finally:
        method._replace_embedding_rows(input_layer, input_ids, edited_input)
        core.replace_sparse_neuron_weights(mlp, neurons, edited_neurons)

    base = values["reconstructed_base"]
    payload = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_component_ppl_attribution",
        "ppl": values,
        "percent_delta_from_reconstructed_base": {
            key: _percent_delta(value, base) for key, value in values.items()
        },
        "sparse_parameters": {
            "input_embedding_rows": len(input_ids),
            "existing_swiglu_neurons": len(neurons),
            "lm_head_rows": 0,
        },
        "diagnostic_only": True,
        "used_for_training_or_checkpoint_selection": False,
        "opened_after_checkpoint_was_frozen": True,
        "benchmark_records_loaded": 0,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
