#!/usr/bin/env python3
"""Post-freeze false-response and effect-tail audit on official retain prompts."""

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
    _official_prompt_groups,
    dtype_from_str,
    load_official_eval_records,
)


LEGACY_EVENTS = 24
LEGACY_PROMPTS = 13_000
LEGACY_RATE = LEGACY_EVENTS / LEGACY_PROMPTS
ONE_SIDED_95_Z = 1.6448536269514722


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=9000)
    parser.add_argument(
        "--sample-mode", choices=("official", "first"), default="official"
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--minimum-prompts", type=int, default=100000)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--kl-event-threshold", type=float, default=0.01)
    parser.add_argument("--logprob-event-threshold", type=float, default=0.1)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    value = parser.parse_args(list(argv) if argv is not None else None)
    if min(int(value.batch_size), int(value.topk), int(value.minimum_prompts)) <= 0:
        parser.error("batch size, top-k, and minimum prompts must be positive")
    if int(value.max_prompts) < 0:
        parser.error("--max-prompts must be non-negative")
    return value


def wilson_upper_bound(
    events: int, observations: int, *, z: float = ONE_SIDED_95_Z
) -> float:
    """One-sided Wilson upper confidence bound for a binomial rate."""

    events = int(events)
    observations = int(observations)
    if observations <= 0 or events < 0 or events > observations:
        raise ValueError("invalid binomial counts")
    p = events / observations
    z2 = float(z) ** 2
    center = p + z2 / (2.0 * observations)
    radius = float(z) * math.sqrt(
        p * (1.0 - p) / observations + z2 / (4.0 * observations**2)
    )
    return (center + radius) / (1.0 + z2 / observations)


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    finite = torch.tensor(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=torch.float64,
    )
    if finite.numel() == 0:
        return {"n": 0}
    quantiles = torch.quantile(
        finite, torch.tensor([0.5, 0.9, 0.99, 0.999], dtype=torch.float64)
    )
    return {
        "n": int(finite.numel()),
        "min": float(finite.min()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "p999": float(quantiles[3]),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def summarize_tail(
    response_maxima: Sequence[float],
    top1_changed: Sequence[bool],
    restricted_kls: Sequence[float],
    max_logprob_changes: Sequence[float],
    *,
    response_threshold: float,
    kl_threshold: float,
    logprob_threshold: float,
) -> Dict[str, Any]:
    count = len(response_maxima)
    if not (
        count == len(top1_changed) == len(restricted_kls) == len(max_logprob_changes)
    ):
        raise ValueError("tail observations must have equal lengths")
    if count == 0:
        raise ValueError("tail summary needs at least one observation")

    def event_row(events: int) -> Dict[str, Any]:
        return {
            "events": int(events),
            "prompts": count,
            "rate": float(events) / count,
            "one_sided_95pct_wilson_upper": wilson_upper_bound(events, count),
        }

    response_events = sum(
        float(value) > float(response_threshold) for value in response_maxima
    )
    top1_events = sum(bool(value) for value in top1_changed)
    kl_events = sum(float(value) > float(kl_threshold) for value in restricted_kls)
    logprob_events = sum(
        float(value) > float(logprob_threshold) for value in max_logprob_changes
    )
    return {
        "prompt_count": count,
        "thresholds": {
            "response_abs_max": float(response_threshold),
            "restricted_topk_kl": float(kl_threshold),
            "max_abs_topk_logprob_change": float(logprob_threshold),
        },
        "response_event": event_row(response_events),
        "top1_change_event": event_row(top1_events),
        "restricted_topk_kl_event": event_row(kl_events),
        "max_abs_topk_logprob_change_event": event_row(logprob_events),
        "response_abs_max": _distribution(response_maxima),
        "restricted_topk_kl": _distribution(restricted_kls),
        "max_abs_topk_logprob_change": _distribution(max_logprob_changes),
    }


def unique_prompt_groups(records: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    raw = _official_prompt_groups(records)
    raw["attribute"] = [
        str(prompt)
        for record in records
        for prompt in record.get("attribute_prompts", [])
    ]
    raw["generation"] = [
        str(prompt)
        for record in records
        for prompt in record.get("generation_prompts", [])
    ]
    result: Dict[str, List[str]] = {}
    globally_seen: set[str] = set()
    for group_name in (
        "rewrite",
        "paraphrase",
        "neighborhood",
        "attribute",
        "generation",
    ):
        selected: List[str] = []
        for value in raw[group_name]:
            prompt = str(value)
            if prompt in globally_seen:
                continue
            globally_seen.add(prompt)
            selected.append(prompt)
        result[group_name] = selected
    return result


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


@torch.no_grad()
def _forward_last_token(
    model: torch.nn.Module,
    tokenizer: Any,
    mlp: torch.nn.Module,
    selected_neurons: Sequence[int],
    prompts: Sequence[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    captured: List[torch.Tensor] = []

    def hook(_module: torch.nn.Module, inputs: Any) -> None:
        captured.append(inputs[0])

    handle = mlp.down_proj.register_forward_pre_hook(hook)
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        encoded = tokenizer(list(prompts), padding=True, return_tensors="pt").to(device)
        output = model(**encoded, use_cache=False, return_dict=True)
        if len(captured) != 1:
            raise RuntimeError(f"expected one MLP activation, captured {len(captured)}")
        activation_device = captured[0].device
        positions = (encoded["attention_mask"].sum(dim=1) - 1).to(activation_device)
        rows = torch.arange(len(prompts), device=activation_device)
        neuron_index = torch.tensor(
            [int(value) for value in selected_neurons],
            dtype=torch.long,
            device=captured[0].device,
        )
        activations = captured[0][rows, positions, :].index_select(1, neuron_index)
        logit_device = output.logits.device
        logit_rows = torch.arange(len(prompts), device=logit_device)
        logit_positions = positions.to(logit_device)
        logits = output.logits[logit_rows, logit_positions, :]
        return activations.detach().float().cpu(), logits.detach().float().cpu()
    finally:
        handle.remove()
        tokenizer.padding_side = old_side


def _effect_rows(
    base_logits: torch.Tensor, edited_logits: torch.Tensor, *, topk: int
) -> Tuple[List[bool], List[float], List[float]]:
    if base_logits.shape != edited_logits.shape or base_logits.ndim != 2:
        raise ValueError("base and edited logits must be equal rank-two tensors")
    k = min(int(topk), int(base_logits.shape[1]))
    ids = base_logits.topk(k, dim=1).indices
    base_full_logprob = F.log_softmax(base_logits, dim=1)
    edited_full_logprob = F.log_softmax(edited_logits, dim=1)
    base_selected = base_full_logprob.gather(1, ids)
    edited_selected = edited_full_logprob.gather(1, ids)
    base_logprob = F.log_softmax(base_selected, dim=1)
    edited_logprob = F.log_softmax(edited_selected, dim=1)
    restricted_kl = F.kl_div(
        edited_logprob, base_logprob, log_target=True, reduction="none"
    ).sum(dim=1)
    maximum_change = (edited_selected - base_selected).abs().max(dim=1).values
    top1_changed = base_logits.argmax(dim=1) != edited_logits.argmax(dim=1)
    return (
        [bool(value) for value in top1_changed],
        [float(value) for value in restricted_kl],
        [float(value) for value in maximum_change],
    )


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
    if state.get("detector_response_mode") != "absolute_signed_group_activation":
        raise RuntimeError("retain-tail audit requires the v2 absolute gate statistic")

    _forget, retain_records = load_official_eval_records(
        args.mcf_path,
        int(args.unlearn_num),
        int(args.retain_num),
        int(args.seed),
        str(args.sample_mode),
    )
    grouped = unique_prompt_groups(retain_records)
    ordered: List[Tuple[str, str]] = [
        (group, prompt) for group, prompts in grouped.items() for prompt in prompts
    ]
    if int(args.max_prompts) > 0:
        ordered = ordered[: int(args.max_prompts)]
    if len(ordered) < int(args.minimum_prompts):
        raise RuntimeError(
            f"retain tail has only {len(ordered)} unique prefixes; "
            f"{args.minimum_prompts} were preregistered"
        )

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

    input_ids = [int(value) for value in state["selected_embedding_rows"]]
    neurons = [int(value) for value in state["selected_neurons"]]
    base_input = _matrix(state, "base_selected_embedding_rows")
    edited_input = _matrix(state, "edited_selected_embedding_rows")
    base_neurons = _neuron_weights(state, "base_neuron_weights")
    edited_neurons = _neuron_weights(state, "edited_neuron_weights")
    ownership = [[int(value) for value in group] for group in state["ownership"]]
    signs = state.get("flat_signs")
    if not isinstance(signs, torch.Tensor):
        raise RuntimeError("state lacks flat detector signs")
    offsets = [0]
    for group in ownership:
        offsets.append(offsets[-1] + len(group))
    flattened, _signs, local_groups = core.flatten_ownership(
        ownership,
        [signs[offsets[index] : offsets[index + 1]] for index in range(len(ownership))],
    )
    if flattened != neurons:
        raise RuntimeError("state neuron ownership order is inconsistent")
    flat_signs = signs.detach().float().cpu()
    response_threshold = float(state["detector_off_abs_max"])

    observations: Dict[str, Dict[str, List[Any]]] = {
        group: {"response": [], "top1": [], "kl": [], "logprob": []}
        for group in (*grouped.keys(), "all")
    }
    try:
        for start in range(0, len(ordered), int(args.batch_size)):
            rows = ordered[start : start + int(args.batch_size)]
            prompts = [prompt for _group, prompt in rows]
            method._replace_embedding_rows(input_layer, input_ids, base_input)
            core.replace_sparse_neuron_weights(mlp, neurons, base_neurons)
            _base_activation, base_logits = _forward_last_token(
                model, tokenizer, mlp, neurons, prompts, device
            )
            method._replace_embedding_rows(input_layer, input_ids, edited_input)
            core.replace_sparse_neuron_weights(mlp, neurons, edited_neurons)
            edited_activation, edited_logits = _forward_last_token(
                model, tokenizer, mlp, neurons, prompts, device
            )
            responses = (
                core.signed_group_activations(
                    edited_activation, local_groups, flat_signs
                )
                .abs()
                .max(dim=1)
                .values
            )
            top1, kls, logprob = _effect_rows(
                base_logits, edited_logits, topk=int(args.topk)
            )
            for index, (group, _prompt) in enumerate(rows):
                for destination in (group, "all"):
                    observations[destination]["response"].append(
                        float(responses[index])
                    )
                    observations[destination]["top1"].append(bool(top1[index]))
                    observations[destination]["kl"].append(float(kls[index]))
                    observations[destination]["logprob"].append(float(logprob[index]))
            if start == 0 or (start + len(rows)) % 1000 < len(rows):
                print(f"audited {start + len(rows)}/{len(ordered)} retain prefixes")
    finally:
        method._replace_embedding_rows(input_layer, input_ids, edited_input)
        core.replace_sparse_neuron_weights(mlp, neurons, edited_neurons)

    groups = {
        group: summarize_tail(
            values["response"],
            values["top1"],
            values["kl"],
            values["logprob"],
            response_threshold=response_threshold,
            kl_threshold=float(args.kl_event_threshold),
            logprob_threshold=float(args.logprob_event_threshold),
        )
        for group, values in observations.items()
        if values["response"]
    }
    overall = groups["all"]
    response = overall["response_event"]
    top1 = overall["top1_change_event"]
    checks = {
        "minimum_unique_prompt_count": overall["prompt_count"]
        >= int(args.minimum_prompts),
        "response_rate_at_most_24_over_13000": response["rate"] <= LEGACY_RATE,
        "response_wilson_upper_at_most_24_over_13000": response[
            "one_sided_95pct_wilson_upper"
        ]
        <= LEGACY_RATE,
        "top1_change_rate_at_most_24_over_13000": top1["rate"] <= LEGACY_RATE,
    }
    payload = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_post_freeze_retain_tail_audit",
        "dataset": "MCF",
        "seed": int(args.seed),
        "unlearn_num": int(args.unlearn_num),
        "retain_num": int(args.retain_num),
        "sample_mode": str(args.sample_mode),
        "writer_mode": str(state.get("writer_mode") or "embedding_keyed"),
        "prefixes_only": True,
        "unique_prompts": True,
        "response_statistic": (
            "maximum absolute signed mean activation of any record-owned "
            "SwiGLU group; this is the actual continuous quantity multiplying "
            "the edited down columns, with no unavailable Base subtraction"
        ),
        "used_for_training_checkpoint_selection_or_retry": False,
        "legacy_comparator": {
            "events": LEGACY_EVENTS,
            "prompts": LEGACY_PROMPTS,
            "rate": LEGACY_RATE,
            "comparison_scope_warning": (
                "The legacy event used an exact sequence router and is an empirical "
                "bar, not an identical stochastic mechanism."
            ),
        },
        "groups": groups,
        "acceptance": {"checks": checks, "passed": all(checks.values())},
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["acceptance"], indent=2))
    print(f"retain tail audit: {out}")


if __name__ == "__main__":
    main()
