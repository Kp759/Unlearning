#!/usr/bin/env python3
"""Post-freeze official evaluation of fixed within-checkpoint components."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_embedding_keyed_neuron_core as core
import mcf_embedding_keyed_neuron_erasure as method
from mcf_zero_unlearn_official_eval import (
    dtype_from_str,
    evaluate_loaded_model_official,
    load_official_eval_records,
    official_summarize,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--state", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--wikidata-dir", required=True)
    parser.add_argument("--context-manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--unlearn-num", type=int, default=50)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument(
        "--sample-mode", choices=("official", "first"), default="official"
    )
    parser.add_argument("--include-ppl", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args(list(argv) if argv is not None else None)


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


def _load_frequency_by_case(path: str | None) -> Dict[int, int]:
    if not path:
        return {}
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("context manifest must be a JSON object")
    rows = payload.get("subject_row_selection")
    if not isinstance(rows, list):
        raise RuntimeError("context manifest lacks subject-row frequencies")
    result: Dict[int, int] = {}
    for row in rows:
        frequencies = [int(value) for value in row.get("kept_token_frequencies", [])]
        result[int(row["case_id"])] = max(frequencies) if frequencies else 0
    return result


def _frequency_strata(
    forget_records: Sequence[Mapping[str, Any]],
    forget_raw: Sequence[Mapping[str, Any]],
    frequency_by_case: Mapping[int, int],
) -> Dict[str, Any]:
    if not frequency_by_case:
        return {}
    definitions = {
        "rare": lambda value: value < 100,
        "medium": lambda value: 100 <= value < 1000,
        "common": lambda value: value >= 1000,
    }
    result: Dict[str, Any] = {}
    for name, predicate in definitions.items():
        selected: List[Mapping[str, Any]] = []
        case_ids: List[int] = []
        for record, raw in zip(forget_records, forget_raw):
            case_id = int(record["case_id"])
            if predicate(int(frequency_by_case[case_id])):
                selected.append(raw)
                case_ids.append(case_id)
        result[name] = {
            "record_count": len(selected),
            "case_ids": case_ids,
            "metrics": official_summarize("forget", selected) if selected else None,
        }
    return result


def _compact(
    result: Mapping[str, Any],
    forget_records: Sequence[Mapping[str, Any]],
    frequency_by_case: Mapping[int, int],
) -> Dict[str, Any]:
    return {
        "forget": result["forget"],
        "retain": result["retain"],
        "PPL": result["forget_PPL"],
        "forget_frequency_strata": _frequency_strata(
            forget_records, result["forget_raw"], frequency_by_case
        ),
    }


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
    input_ids = [int(value) for value in state["selected_embedding_rows"]]
    neurons = [int(value) for value in state["selected_neurons"]]
    base_input = _matrix(state, "base_selected_embedding_rows")
    edited_input = _matrix(state, "edited_selected_embedding_rows")
    base_neurons = _weights(state, "base_neuron_weights")
    edited_neurons = _weights(state, "edited_neuron_weights")
    forget_records, _retain_records = load_official_eval_records(
        args.mcf_path,
        int(args.unlearn_num),
        int(args.retain_num),
        int(args.seed),
        str(args.sample_mode),
    )
    frequency_by_case = _load_frequency_by_case(args.context_manifest)
    if frequency_by_case and any(
        int(record["case_id"]) not in frequency_by_case for record in forget_records
    ):
        raise RuntimeError("frequency manifest does not cover official forget cases")

    modes = {
        "full_embedding_plus_neuron": (edited_input, edited_neurons),
        "embedding_only": (edited_input, base_neurons),
        "neuron_only": (base_input, edited_neurons),
        "reconstructed_base": (base_input, base_neurons),
    }
    components: Dict[str, Any] = {}
    try:
        for label, (input_rows, neuron_rows) in modes.items():
            method._replace_embedding_rows(input_layer, input_ids, input_rows)
            core.replace_sparse_neuron_weights(mlp, neurons, neuron_rows)
            result = evaluate_loaded_model_official(
                method=label,
                model=model,
                tok=tokenizer,
                model_dir=model_dir,
                mcf_path=args.mcf_path,
                wikidata_dir=args.wikidata_dir,
                unlearn_num=int(args.unlearn_num),
                retain_num=int(args.retain_num),
                seed=int(args.seed),
                sample_mode=str(args.sample_mode),
                skip_ppl=not bool(args.include_ppl),
            )
            components[label] = _compact(result, forget_records, frequency_by_case)
            print(
                f"{label}: Eff={result['forget']['Eff']:.3f}, "
                f"Gen={result['forget']['Gen']:.3f}, "
                f"retain Spe={result['retain']['Spe']:.3f}"
            )
    finally:
        method._replace_embedding_rows(input_layer, input_ids, edited_input)
        core.replace_sparse_neuron_weights(mlp, neurons, edited_neurons)

    payload = {
        "schema_version": 1,
        "kind": "mcf_embedding_keyed_neuron_post_freeze_component_evaluation",
        "writer_mode": str(state.get("writer_mode") or "embedding_keyed"),
        "writer_configuration": dict(state.get("writer_configuration") or {}),
        "source_stage1_state_sha256": state.get("source_stage1_state_sha256"),
        "dataset": "MCF",
        "seed": int(args.seed),
        "unlearn_num": int(args.unlearn_num),
        "retain_num": int(args.retain_num),
        "sample_mode": str(args.sample_mode),
        "interpretation_boundary": (
            "These fixed interventions diagnose reliance of one fitted checkpoint. "
            "The independently retrained no-writer model is a separate artifact."
        ),
        "used_for_training_checkpoint_selection_or_retry": False,
        "frequency_stratum_definition": {
            "statistic": "maximum selected subject-row corpus frequency",
            "rare": "<100",
            "medium": "100-999",
            "common": ">=1000",
        },
        "components": components,
    }
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"component evaluation: {out}")


if __name__ == "__main__":
    main()
