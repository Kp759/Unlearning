#!/usr/bin/env python3
"""Replay the clean frozen writer's training-safe portability gate.

This audit loads no official MCF evaluation prompt.  It reconstructs Base,
applies only the frozen sparse embedding delta, and calls the same measurement
function used immediately before sparse-neuron decoder construction.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import gagd_compare as gagd
import mcf_compositional_marker_write_read as writer_method
import mcf_embedding_keyed_neuron_core as neuron_core
import mcf_embedding_keyed_neuron_erasure as neuron_method


def _load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--stage1-state", type=Path, required=True)
    parser.add_argument("--stage1-report", type=Path, required=True)
    parser.add_argument("--stage1-writer-log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--amplitude-threshold", type=float, default=4.5)
    parser.add_argument("--minimum-global-fraction", type=float, default=0.95)
    parser.add_argument("--minimum-record-fraction", type=float, default=0.80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    if float(args.amplitude_threshold) < 0.0:
        parser.error("--amplitude-threshold must be non-negative")
    for name in ("minimum_global_fraction", "minimum_record_fraction"):
        if not 0.0 <= float(getattr(args, name)) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0,1]")
    return args


@torch.no_grad()
def run(args: argparse.Namespace) -> Dict[str, Any]:
    context_path = Path(args.context_manifest).resolve()
    state_path = Path(args.stage1_state).resolve()
    report_path = Path(args.stage1_report).resolve()
    log_path = Path(args.stage1_writer_log).resolve()
    for path in (context_path, state_path, report_path, log_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    context = _load_json(context_path)
    report = _load_json(report_path)
    state_value = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state_value, Mapping):
        raise RuntimeError("Stage-1 writer state must be a mapping")
    state = dict(state_value)
    neuron_method._validate_firewall(context, state)
    lineage = neuron_method._validate_clean_stage1_lineage(
        context, state, report, context_path, log_path
    )
    if str(Path(args.model_path).resolve()) != str(lineage["base_model_path"]):
        raise RuntimeError("preflight Base-model path differs from Stage-1 lineage")
    stored_threshold = float(state.get("writer_positive_amplitude_threshold", -1.0))
    if not math.isclose(
        stored_threshold,
        float(args.amplitude_threshold),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("preflight threshold differs from the frozen writer")

    namespace = argparse.Namespace(
        model_path=str(Path(args.model_path).resolve()),
        dtype=str(args.dtype),
        device_map=str(args.device_map),
        gradient_checkpointing=False,
    )
    model, tokenizer = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    model.config.use_cache = False
    device = gagd.first_device(model)
    observed_transformer_fingerprint = writer_method.frozen_transformer_fingerprint(
        model
    )
    if not math.isclose(
        observed_transformer_fingerprint,
        float(lineage["base_transformer_fingerprint"]),
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("preflight Transformer differs from Stage-1 Base")

    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model lacks input embeddings")
    selected_rows = [int(value) for value in state.get("selected_embedding_rows", [])]
    embedding_delta = state.get("embedding_delta")
    if not selected_rows or not isinstance(embedding_delta, torch.Tensor):
        raise RuntimeError("Stage-1 state lacks its sparse embedding delta")
    selected_index = torch.tensor(
        selected_rows, dtype=torch.long, device=input_layer.weight.device
    )
    observed_rows_sha256 = writer_method.tensor_sha256(
        input_layer.weight.index_select(0, selected_index)
    )
    if observed_rows_sha256 != str(lineage["base_selected_embedding_rows_sha256"]):
        raise RuntimeError("preflight embedding rows differ from Stage-1 Base")

    rows = context.get("records")
    if not isinstance(rows, list):
        raise RuntimeError("context manifest lacks record contexts")
    by_case = {int(row["case_id"]): row for row in rows if isinstance(row, Mapping)}
    case_ids = [int(value) for value in state.get("case_ids", [])]
    if set(by_case) != set(case_ids):
        raise RuntimeError("context/state case IDs differ")
    positive_groups = []
    for case_id in case_ids:
        prompts = by_case[case_id].get("positive_prompts")
        if not isinstance(prompts, list) or not prompts:
            raise RuntimeError(f"case {case_id} has no training-safe positives")
        positive_groups.append([str(prompt) for prompt in prompts])
    markers = state.get("markers")
    if not isinstance(markers, Mapping):
        raise RuntimeError("Stage-1 state lacks marker directions")

    embedding_writer = neuron_core.ToggleableEmbeddingDelta(
        input_layer, selected_rows, embedding_delta
    )
    try:
        result = neuron_method.measure_training_safe_writer_preflight(
            model,
            tokenizer,
            embedding_writer,
            positive_groups,
            markers,
            case_ids,
            device,
            batch_size=int(args.batch_size),
            amplitude_threshold=float(args.amplitude_threshold),
            minimum_global_fraction=float(args.minimum_global_fraction),
            minimum_record_fraction=float(args.minimum_record_fraction),
        )
    finally:
        embedding_writer.remove()

    result.update(
        {
            "schema_version": 1,
            "kind": "mcf_clean_stage1_training_safe_portability_preflight",
            "protocol": str(state["protocol"]),
            "seed": int(state["seed"]),
            "forget_num": len(case_ids),
            "case_ids": case_ids,
            "binding": {
                "context_manifest_sha256": writer_method.sha256_file(context_path),
                "stage1_state_sha256": writer_method.sha256_file(state_path),
                "stage1_report_sha256": writer_method.sha256_file(report_path),
                "stage1_writer_log_sha256": writer_method.sha256_file(log_path),
                "base_model_path": str(Path(args.model_path).resolve()),
            },
            "decoder_constructed": False,
            "official_evaluation_opened": False,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
