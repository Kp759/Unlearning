#!/usr/bin/env python3
"""Post-hoc PPL attribution for the two sparse compositional-marker edits.

The frozen checkpoint differs from the base model only on selected input
embedding rows and selected LM-head rows.  The method state stores the exact
base and edited values for both sets, so one checkpoint load is enough to
evaluate the combined edit, input-only edit, output-only edit, and reconstructed
base.  This script is diagnostic only and must run after checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
    parser.add_argument(
        "--dtype", choices=("bf16", "fp16", "fp32"), default="bf16"
    )
    parser.add_argument(
        "--device-map", choices=("single", "auto"), default="single"
    )
    parser.add_argument("--max-input-length", type=int, default=100)
    value = parser.parse_args(list(argv) if argv is not None else None)
    if int(value.max_input_length) <= 1:
        parser.error("--max-input-length must be greater than one")
    return value


def _required_tensor(state: Mapping[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise RuntimeError(f"method state lacks matrix {key!r}")
    return value.detach().cpu()


@torch.no_grad()
def replace_selected_rows(
    layer: torch.nn.Module,
    token_ids: Sequence[int],
    values: torch.Tensor,
) -> None:
    if values.ndim != 2 or values.shape[0] != len(token_ids):
        raise ValueError("selected row ids and values do not match")
    index = torch.tensor(token_ids, dtype=torch.long, device=layer.weight.device)
    layer.weight.index_copy_(
        0,
        index,
        values.to(device=layer.weight.device, dtype=layer.weight.dtype),
    )


def _percent_delta(value: float, base: float) -> float:
    return 100.0 * (float(value) - float(base)) / float(base)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    model_dir = Path(args.model_dir).resolve()
    state_path = Path(args.state).resolve()
    wikidata_dir = Path(args.wikidata_dir).resolve()
    for path in (model_dir, state_path, wikidata_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("method state must be a mapping")
    input_ids = [int(x) for x in state.get("selected_embedding_rows", [])]
    output_ids = [int(x) for x in state.get("selected_output_rows", [])]
    base_input = _required_tensor(state, "base_selected_embedding_rows")
    edited_input = _required_tensor(state, "edited_selected_embedding_rows")
    base_output = _required_tensor(state, "base_selected_output_rows")
    edited_output = _required_tensor(state, "edited_selected_output_rows")
    if not input_ids or not output_ids:
        raise RuntimeError("method state has no selected sparse rows")

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype_from_str(args.dtype)}
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    if args.device_map == "single":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for --device-map single")
        model = model.to("cuda")
    model.eval()
    model.config.use_cache = False
    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("component attribution requires an untied output head")

    ppl_text = load_official_ppl_text(wikidata_dir)
    if ppl_text is None:
        raise RuntimeError(f"could not load PPL text from {wikidata_dir}")
    device = next(model.parameters()).device
    modes = {
        "combined": (edited_input, edited_output),
        "input_only": (edited_input, base_output),
        "output_only": (base_input, edited_output),
        "reconstructed_base": (base_input, base_output),
    }
    values: Dict[str, float] = {}
    try:
        for label, (input_rows, output_rows) in modes.items():
            replace_selected_rows(input_layer, input_ids, input_rows)
            replace_selected_rows(output_layer, output_ids, output_rows)
            values[label] = float(
                official_perplexity(
                    model,
                    tokenizer,
                    ppl_text,
                    device,
                    max_input_length=int(args.max_input_length),
                )
            )
            print(f"{label:>18}: {values[label]:.6f}")
    finally:
        # Leave the in-memory object in its serialized combined state even if a
        # diagnostic forward raises; no file is mutated by this script.
        replace_selected_rows(input_layer, input_ids, edited_input)
        replace_selected_rows(output_layer, output_ids, edited_output)

    base = values["reconstructed_base"]
    payload = {
        "schema_version": 1,
        "kind": "mcf_compositional_marker_component_ppl_attribution",
        "model_dir": str(model_dir),
        "state": str(state_path),
        "ppl": values,
        "percent_delta_from_reconstructed_base": {
            key: _percent_delta(value, base) for key, value in values.items()
        },
        "sparse_rows": {
            "input_embedding": len(input_ids),
            "lm_head": len(output_ids),
        },
        "diagnostic_only": True,
        "used_for_training_or_checkpoint_selection": False,
        "benchmark_records_loaded": 0,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
