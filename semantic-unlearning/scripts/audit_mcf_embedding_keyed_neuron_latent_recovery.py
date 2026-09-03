#!/usr/bin/env python3
"""Post-freeze downstream residual logit-lens recovery audit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_embedding_keyed_neuron_core as core
import mcf_embedding_keyed_neuron_erasure as method
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    is_llama_like,
    load_official_eval_records,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument(
        "--sample-mode", choices=("official", "first"), default="official"
    )
    parser.add_argument("--layer-stride", type=int, default=4)
    parser.add_argument("--recovery-fraction", type=float, default=0.5)
    parser.add_argument("--recovery-gap", type=float, default=0.1)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    value = parser.parse_args(list(argv) if argv is not None else None)
    if int(value.layer_stride) <= 0:
        parser.error("--layer-stride must be positive")
    for name in ("recovery_fraction", "recovery_gap"):
        if not 0.0 <= float(getattr(value, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must lie in [0, 1]")
    return value


def summarize_candidate_nlls(
    rows: Sequence[Mapping[str, float]],
) -> Dict[str, Any]:
    if not rows:
        raise ValueError("latent summary needs prompt observations")
    margins = [
        float(row["target_new_nll"]) - float(row["target_true_nll"]) for row in rows
    ]
    if not all(math.isfinite(value) for value in margins):
        raise ValueError("latent summary contains non-finite NLLs")
    ordered = sorted(margins)
    return {
        "prompt_count": len(rows),
        "sensitive_preference_count": sum(value > 0.0 for value in margins),
        "sensitive_preference_fraction": sum(value > 0.0 for value in margins)
        / len(margins),
        "target_new_minus_true_nll": {
            "min": ordered[0],
            "median": ordered[len(ordered) // 2],
            "max": ordered[-1],
            "mean": sum(ordered) / len(ordered),
        },
    }


def _matrix(state: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError(f"state lacks matrix {key!r}")
    return value.detach().cpu()


def _weights(state: Mapping[str, Any], key: str) -> core.SparseNeuronWeights:
    value = state.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"state lacks neuron weights {key!r}")
    tensors = [value.get(name) for name in ("gate_rows", "up_rows", "down_columns")]
    if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise RuntimeError(f"state neuron weights {key!r} are incomplete")
    return core.SparseNeuronWeights(*[tensor.detach().cpu() for tensor in tensors])


def _answer_tokens(tokenizer: Any, answer: str, *, llama_like: bool) -> List[int]:
    values = tokenizer(f" {answer}")["input_ids"]
    if values and isinstance(values[0], list):
        values = values[0]
    result = [int(value) for value in values]
    return result[1:] if llama_like else result


@torch.no_grad()
def record_layer_nlls(
    model: torch.nn.Module,
    tokenizer: Any,
    prefixes: Sequence[str],
    target_true: str,
    target_new: str,
    layer_indices: Sequence[int],
    device: torch.device,
    *,
    llama_like: bool,
) -> Dict[str, List[Dict[str, float]]]:
    if not prefixes:
        return {str(layer): [] for layer in layer_indices}
    layers = model.model.layers
    final_norm = model.model.norm
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("model lacks output projection")
    captured: Dict[int, torch.Tensor] = {}
    handles = []
    for layer_index in layer_indices:

        def capture(
            _module: torch.nn.Module,
            _inputs: Any,
            output: Any,
            *,
            index: int = int(layer_index),
        ) -> None:
            captured[index] = output[0] if isinstance(output, tuple) else output

        handles.append(layers[int(layer_index)].register_forward_hook(capture))

    prefix_lengths = [len(tokenizer(prefix)["input_ids"]) for prefix in prefixes]
    true_tokens = _answer_tokens(tokenizer, target_true, llama_like=llama_like)
    new_tokens = _answer_tokens(tokenizer, target_new, llama_like=llama_like)
    if not true_tokens or not new_tokens:
        raise RuntimeError("target tokenization is empty")
    sequences: List[str] = []
    descriptors: List[Tuple[int, str, List[int]]] = []
    for prompt_index, prefix in enumerate(prefixes):
        sequences.extend([f"{prefix} {target_true}", f"{prefix} {target_new}"])
        descriptors.extend(
            [
                (prompt_index, "target_true", true_tokens),
                (prompt_index, "target_new", new_tokens),
            ]
        )
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(sequences, padding=True, return_tensors="pt").to(device)
        output = model(**encoded, use_cache=False, return_dict=True)
    finally:
        tokenizer.padding_side = old_side
        for handle in handles:
            handle.remove()
    if set(captured) != {int(value) for value in layer_indices}:
        raise RuntimeError("failed to capture every requested residual layer")

    result: Dict[str, List[Dict[str, float]]] = {
        str(layer): [{"target_true_nll": 0.0, "target_new_nll": 0.0} for _ in prefixes]
        for layer in layer_indices
    }
    result["final_model_output"] = [
        {"target_true_nll": 0.0, "target_new_nll": 0.0} for _ in prefixes
    ]
    counts = {
        "target_true": len(true_tokens),
        "target_new": len(new_tokens),
    }
    for layer_key in [str(value) for value in layer_indices] + ["final_model_output"]:
        hidden = captured[int(layer_key)] if layer_key != "final_model_output" else None
        for sequence_index, (prompt_index, target_name, tokens) in enumerate(
            descriptors
        ):
            positions = [
                prefix_lengths[prompt_index] + offset - 1
                for offset in range(len(tokens))
            ]
            if layer_key == "final_model_output":
                vectors = output.logits[sequence_index, positions, :].float()
            else:
                assert hidden is not None
                norm_device = next(final_norm.parameters()).device
                head_device = lm_head.weight.device
                states = hidden[sequence_index, positions, :].to(norm_device)
                normalized = final_norm(states).to(head_device)
                vectors = lm_head(normalized).float()
            log_probs = F.log_softmax(vectors, dim=1)
            token_index = torch.tensor(
                tokens, dtype=torch.long, device=log_probs.device
            )
            nll = -log_probs[
                torch.arange(len(tokens), device=log_probs.device), token_index
            ].mean()
            result[layer_key][prompt_index][f"{target_name}_nll"] = float(nll)
        for row in result[layer_key]:
            for target_name, count in counts.items():
                if count <= 0 or not math.isfinite(float(row[f"{target_name}_nll"])):
                    raise RuntimeError("invalid latent target NLL")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    state_path = Path(args.state).resolve()
    for path in (model_dir, state_path, Path(args.mcf_path).resolve()):
        if not path.exists():
            raise FileNotFoundError(path)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping) or state.get("protocol") != method.PROTOCOL:
        raise RuntimeError("embedding-keyed neuron state protocol mismatch")

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
    if input_layer is None:
        raise RuntimeError("model lacks input embeddings")
    mlp = method._resolve_swiglu_mlp(model, int(state["layer"]))
    device = input_layer.weight.device
    total_layers = len(model.model.layers)
    edit_layer = int(state["layer"])
    layer_indices = list(range(edit_layer, total_layers, int(args.layer_stride)))
    if total_layers - 1 not in layer_indices:
        layer_indices.append(total_layers - 1)

    input_ids = [int(value) for value in state["selected_embedding_rows"]]
    neurons = [int(value) for value in state["selected_neurons"]]
    base_input = _matrix(state, "base_selected_embedding_rows")
    edited_input = _matrix(state, "edited_selected_embedding_rows")
    base_neurons = _weights(state, "base_neuron_weights")
    edited_neurons = _weights(state, "edited_neuron_weights")
    forget_records, _retain = load_official_eval_records(
        args.mcf_path,
        int(args.unlearn_num),
        1,
        int(args.seed),
        str(args.sample_mode),
    )
    modes = {
        "edited": (edited_input, edited_neurons),
        "reconstructed_base": (base_input, base_neurons),
    }
    mode_rows: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    try:
        for mode_name, (input_rows, neuron_rows) in modes.items():
            method._replace_embedding_rows(input_layer, input_ids, input_rows)
            core.replace_sparse_neuron_weights(mlp, neurons, neuron_rows)
            accumulated: Dict[str, List[Dict[str, float]]] = {
                str(layer): [] for layer in layer_indices
            }
            accumulated["final_model_output"] = []
            for position, record in enumerate(forget_records):
                rewrite = record["requested_rewrite"]
                prefixes = [
                    str(rewrite["prompt"]).format(str(rewrite["subject"])),
                    *[str(value) for value in record.get("paraphrase_prompts", [])],
                ]
                observed = record_layer_nlls(
                    model,
                    tokenizer,
                    prefixes,
                    str(rewrite["target_true"]["str"]),
                    str(rewrite["target_new"]["str"]),
                    layer_indices,
                    device,
                    llama_like=is_llama_like(model, tokenizer),
                )
                for layer_key, values in observed.items():
                    accumulated[layer_key].extend(values)
                if position == 0 or (position + 1) % 10 == 0:
                    print(
                        f"{mode_name}: audited {position + 1}/{len(forget_records)} records"
                    )
            mode_rows[mode_name] = accumulated
    finally:
        method._replace_embedding_rows(input_layer, input_ids, edited_input)
        core.replace_sparse_neuron_weights(mlp, neurons, edited_neurons)

    summaries = {
        mode_name: {
            layer_key: summarize_candidate_nlls(values)
            for layer_key, values in layers.items()
        }
        for mode_name, layers in mode_rows.items()
    }
    edited_final = summaries["edited"]["final_model_output"][
        "sensitive_preference_fraction"
    ]
    edited_intermediate = {
        key: value
        for key, value in summaries["edited"].items()
        if key != "final_model_output"
    }
    recovered_layers = [
        key
        for key, value in edited_intermediate.items()
        if value["sensitive_preference_fraction"] >= float(args.recovery_fraction)
        and value["sensitive_preference_fraction"]
        >= edited_final + float(args.recovery_gap)
    ]
    final_output_sensitive = edited_final >= float(args.recovery_fraction)
    base_probe_maximum_sensitive_fraction = max(
        float(row["sensitive_preference_fraction"])
        for row in summaries["reconstructed_base"].values()
    )
    positive_control_passed = base_probe_maximum_sensitive_fraction >= float(
        args.recovery_fraction
    )
    payload = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_post_freeze_latent_recovery_audit",
        "dataset": "MCF",
        "seed": int(args.seed),
        "unlearn_num": int(args.unlearn_num),
        "sample_mode": str(args.sample_mode),
        "writer_mode": str(state.get("writer_mode") or "embedding_keyed"),
        "record_count": len(forget_records),
        "prompt_count": int(summaries["edited"]["final_model_output"]["prompt_count"]),
        "probe": "fixed final-norm plus unchanged-LM-head logit lens",
        "downstream_zero_based_block_layers": layer_indices,
        "prompt_scope": "official forget rewrites and paraphrases",
        "used_for_training_checkpoint_selection_or_retry": False,
        "fact_recoverable": bool(final_output_sensitive or recovered_layers),
        "final_output_sensitive": bool(final_output_sensitive),
        "reconstructed_base_positive_control": {
            "maximum_sensitive_preference_fraction": (
                base_probe_maximum_sensitive_fraction
            ),
            "passed": bool(positive_control_passed),
        },
        "positive_control_passed": bool(positive_control_passed),
        "recovered_layers": recovered_layers,
        "recovery_rule": {
            "minimum_sensitive_preference_fraction": float(args.recovery_fraction),
            "minimum_gap_above_final_model_output": float(args.recovery_gap),
        },
        "modes": summaries,
        "interpretation_boundary": (
            "A positive result falsifies complete representational removal. A "
            "negative fixed-lens result does not prove that no stronger probe can recover the fact."
        ),
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "fact_recoverable": payload["fact_recoverable"],
                "final_output_sensitive": payload["final_output_sensitive"],
                "recovered_layers": recovered_layers,
            },
            indent=2,
        )
    )
    print(f"latent recovery audit: {out}")


if __name__ == "__main__":
    main()
