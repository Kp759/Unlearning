#!/usr/bin/env python3
"""Cache fixed Wikipedia final-hidden second-moment statistics for SURE-LM.

This is a one-time, model-specific preprocessing step shared by MCF and ZsRE.
It requests ``--sample-size`` Wikipedia documents (100,000 by default), caps
that request to the eligible local dataset exactly as ZeroUnlearn's statistics
loader does, collects their causal predictor states, and stores

    C_U = (1 / N) sum_t h_t h_t^T.

The statistic is aligned with ZeroUnlearn's use of fixed Wikipedia second
moments, but lives at the final hidden state that feeds the LM head.  It never
contains MCF or ZsRE examples.  By default, the first 20 Wikipedia documents
are excluded so the texts used by this repository's PPL probe remain unseen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd


UTILITY_PROTOCOL = "sure_wikipedia_final_hidden_second_moment_v1"
DEFAULT_SAMPLE_SIZE = 100_000
MODEL_PROBE_VALUES_PER_TENSOR = 16


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


@torch.no_grad()
def model_probe_sha256(model: torch.nn.Module) -> str:
    """Return a cheap deterministic probe over every parameter tensor.

    This is not presented as a full cryptographic hash of multi-gigabyte model
    weights.  It samples fixed positions from every named parameter and is used
    to catch accidental cache/model mismatches before training.
    """
    digest = hashlib.sha256()
    parameter_count = 0
    for name, parameter in model.named_parameters():
        parameter_count += 1
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(int(x) for x in parameter.shape)).encode("ascii"))
        flat = parameter.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        take = min(MODEL_PROBE_VALUES_PER_TENSOR, int(flat.numel()))
        if take == 1:
            indices = torch.zeros(1, dtype=torch.long, device=flat.device)
        else:
            indices = (
                torch.linspace(
                    0,
                    int(flat.numel()) - 1,
                    steps=take,
                    dtype=torch.float64,
                    device=flat.device,
                )
                .round()
                .long()
            )
        values = flat.index_select(0, indices).float().cpu().contiguous()
        digest.update(values.numpy().tobytes())
    if parameter_count == 0:
        raise ValueError("Cannot fingerprint a model with no named parameters")
    return digest.hexdigest()


def tokenizer_probe_sha256(tok: Any) -> str:
    payload = {
        "class": tok.__class__.__name__,
        "vocab_size": int(len(tok)),
        "bos_token_id": getattr(tok, "bos_token_id", None),
        "eos_token_id": getattr(tok, "eos_token_id", None),
        "pad_token_id": getattr(tok, "pad_token_id", None),
        "probe_ids": [
            [int(x) for x in tok.encode(text)]
            for text in (
                "Wikipedia utility statistics.",
                "The quick brown fox jumps over the lazy dog.",
            )
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_identity(model: torch.nn.Module, tok: Any, model_path: str) -> Dict[str, Any]:
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise ValueError("Model must expose an output embedding weight")
    return {
        "model_source": str(Path(model_path).resolve()),
        "config_name_or_path": str(getattr(model.config, "_name_or_path", "")),
        "model_type": str(getattr(model.config, "model_type", "")),
        "hidden_size": int(output.weight.shape[1]),
        "vocab_size": int(output.weight.shape[0]),
        "model_probe_sha256": model_probe_sha256(model),
        "tokenizer_probe_sha256": tokenizer_probe_sha256(tok),
    }


def load_wikipedia_train(path: Path) -> Tuple[Sequence[str], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Wikipedia dataset not found: {path}")
    from datasets import load_from_disk

    loader = "datasets.load_from_disk"
    fallback_error = None
    try:
        loaded = load_from_disk(str(path.resolve()))
        if hasattr(loaded, "keys") and "train" in loaded:
            train = loaded["train"]
        else:
            train = loaded
        columns = list(getattr(train, "column_names", []))
        if "text" not in columns:
            raise ValueError("Wikipedia dataset must contain a train text column")
        texts = train["text"]
        row_count = int(len(train))
        fingerprint = str(getattr(train, "_fingerprint", ""))
    except Exception as error:
        # Some datasets/fsspec version pairs cannot reopen a perfectly valid
        # local save_to_disk artifact.  Read its Arrow shards directly rather
        # than downloading or silently changing the utility corpus.
        import pyarrow as pa
        import pyarrow.ipc as ipc

        split_root = (
            path / "train" if (path / "train" / "state.json").is_file() else path
        )
        state_path = split_root / "state.json"
        if not state_path.is_file():
            raise error
        state = json.loads(state_path.read_text(encoding="utf-8"))
        data_files = state.get("_data_files", [])
        filenames = [entry.get("filename") for entry in data_files]
        if not filenames or not all(isinstance(name, str) for name in filenames):
            raise error
        texts_list: List[str] = []
        for filename in filenames:
            shard_path = split_root / filename
            with pa.memory_map(str(shard_path), "r") as source:
                try:
                    table = ipc.RecordBatchStreamReader(source).read_all()
                except pa.ArrowInvalid:
                    source.seek(0)
                    table = ipc.RecordBatchFileReader(source).read_all()
            if "text" not in table.column_names:
                raise ValueError("Wikipedia Arrow shard does not contain a text column")
            texts_list.extend(table.column("text").to_pylist())
        texts = texts_list
        row_count = len(texts_list)
        fingerprint = str(state.get("_fingerprint", ""))
        loader = "direct_pyarrow_save_to_disk_fallback"
        fallback_error = f"{type(error).__name__}: {error}"
    metadata = {
        "dataset_source": str(path.resolve()),
        "split": "train",
        "text_column": "text",
        "dataset_row_count": row_count,
        "dataset_fingerprint": fingerprint,
        "dataset_loader": loader,
        "datasets_loader_fallback_reason": fallback_error,
    }
    return texts, metadata


def predictor_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Select attended states that have a following token in the same text."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [batch, sequence]")
    mask = attention_mask.bool()
    lengths = mask.sum(dim=1)
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return mask & (positions < (lengths - 1).clamp_min(0).unsqueeze(1))


def finalize_second_moment(
    unnormalized: torch.Tensor, count: int
) -> Tuple[torch.Tensor, float]:
    if count <= 0:
        raise ValueError("second moment requires at least one state")
    if unnormalized.ndim != 2 or unnormalized.shape[0] != unnormalized.shape[1]:
        raise ValueError("unnormalized second moment must be square")
    moment = unnormalized.float() / float(count)
    moment = 0.5 * (moment + moment.transpose(0, 1))
    if not torch.isfinite(moment).all():
        raise FloatingPointError("Wikipedia second moment contains non-finite values")
    diagonal_min = float(torch.diagonal(moment).min().detach().cpu())
    if diagonal_min < -1e-6:
        raise FloatingPointError(
            f"Wikipedia second moment has a negative diagonal: {diagonal_min}"
        )
    return moment, float(torch.trace(moment).detach().cpu())


def _move_batch(encoded: Mapping[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in encoded.items()
    }


def _final_hidden_only(
    model: torch.nn.Module, encoded: Mapping[str, Any]
) -> Tuple[torch.Tensor, str]:
    prefix = str(getattr(model, "base_model_prefix", ""))
    backbone = getattr(model, prefix, None) if prefix else None
    if backbone is not None and backbone is not model:
        output = backbone(**encoded, use_cache=False, return_dict=True)
        hidden = getattr(output, "last_hidden_state", None)
        if isinstance(hidden, torch.Tensor):
            return hidden, f"base_model_prefix:{prefix}"
    output = model(
        **encoded,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    hidden_states = getattr(output, "hidden_states", None)
    if not hidden_states:
        raise RuntimeError("Model did not return final hidden states")
    return hidden_states[-1], "causal_lm_hidden_states_fallback"


@torch.no_grad()
def build_second_moment(
    model: torch.nn.Module,
    tok: Any,
    texts: Sequence[str],
    *,
    document_order: Sequence[int],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if not document_order or max_length < 2 or batch_size <= 0:
        raise ValueError("invalid Wikipedia statistic dimensions")
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise ValueError("Model must expose output embeddings")
    hidden_size = int(output.weight.shape[1])
    unnormalized = None
    collected = 0
    forwarded_indices: List[int] = []
    backend = None
    model.eval()

    old_padding_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
        for start in range(0, len(document_order), batch_size):
            indices = [int(x) for x in document_order[start : start + batch_size]]
            chunk = [str(texts[index]) for index in indices]
            encoded = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = _move_batch(encoded, device)
            hidden, current_backend = _final_hidden_only(model, encoded)
            if backend is None:
                backend = current_backend
            elif backend != current_backend:
                raise RuntimeError("final-hidden extraction backend changed mid-run")
            mask = predictor_mask(encoded["attention_mask"])
            states = hidden[mask].float()
            if states.ndim != 2 or (
                states.numel() and int(states.shape[1]) != hidden_size
            ):
                raise RuntimeError("Wikipedia hidden-state shape mismatch")
            if states.numel():
                if unnormalized is None:
                    unnormalized = torch.zeros(
                        (hidden_size, hidden_size),
                        dtype=torch.float32,
                        device=states.device,
                    )
                unnormalized.add_(states.transpose(0, 1) @ states)
                collected += int(states.shape[0])
            forwarded_indices.extend(indices)
            if collected and collected % 10_000 < int(states.shape[0]):
                print(f"Wikipedia predictor states collected: {collected}")
    finally:
        tok.padding_side = old_padding_side

    if collected <= 0 or unnormalized is None:
        raise RuntimeError(
            "Selected Wikipedia documents produced zero predictor states"
        )
    moment, trace = finalize_second_moment(unnormalized, collected)
    moment = moment.detach().cpu().contiguous()
    forwarded_text_digest = hashlib.sha256()
    for index in forwarded_indices:
        forwarded_text_digest.update(str(index).encode("ascii"))
        forwarded_text_digest.update(b"\0")
        forwarded_text_digest.update(str(texts[index]).encode("utf-8"))
        forwarded_text_digest.update(b"\0")
    report = {
        "predictor_hidden_state_count": int(collected),
        "hidden_size": hidden_size,
        "second_moment_trace": trace,
        "second_moment_sha256": sha256_tensor(moment),
        "documents_forwarded": len(forwarded_indices),
        "forwarded_document_indices": forwarded_indices,
        "forwarded_text_sha256": forwarded_text_digest.hexdigest(),
        "hidden_extraction_backend": backend,
        "state_selection": (
            "attended final-layer LM-head input states with a following token "
            "from the fixed sampled document set"
        ),
    }
    return moment, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--utility-seed", type=int, default=1)
    parser.add_argument("--exclude-first", type=int, default=20)
    parser.add_argument("--utility-max-length", type=int, default=4096)
    parser.add_argument("--utility-batch-size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample-size must be positive")
    if args.exclude_first < 0:
        raise ValueError("exclude-first must be non-negative")
    if args.utility_max_length < 2 or args.utility_batch_size <= 0:
        raise ValueError("utility max length/batch size are invalid")

    output_path = Path(args.output_path).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Utility cache already exists: {output_path}; pass --overwrite explicitly"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gagd.set_seed(args.utility_seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    identity = model_identity(model, tok, args.model_path)
    device = gagd.first_device(model)

    texts, dataset_metadata = load_wikipedia_train(Path(args.wikidata_dir).resolve())
    eligible = [
        index
        for index in range(args.exclude_first, len(texts))
        if isinstance(texts[index], str) and texts[index].strip()
    ]
    if not eligible:
        raise ValueError("No eligible non-empty Wikipedia documents remain")
    order = list(eligible)
    random.Random(args.utility_seed).shuffle(order)
    selected_order = order[: min(args.sample_size, len(order))]
    if args.sample_size > len(eligible):
        print(
            "Wikipedia sample request exceeds eligible local corpus; "
            f"using all {len(eligible)} eligible documents, matching "
            "ZeroUnlearn's capped-sample behavior"
        )

    moment, statistic_report = build_second_moment(
        model,
        tok,
        texts,
        document_order=selected_order,
        device=device,
        max_length=args.utility_max_length,
        batch_size=args.utility_batch_size,
    )
    metadata = {
        "schema_version": 1,
        "protocol": UTILITY_PROTOCOL,
        **dataset_metadata,
        **identity,
        **statistic_report,
        "utility_seed": int(args.utility_seed),
        "requested_document_sample_size": int(args.sample_size),
        "actual_document_sample_size": len(selected_order),
        "sample_size_cap_policy": "min(requested, eligible_local_documents)",
        "excluded_prefix_document_count": int(args.exclude_first),
        "excluded_prefix_reason": (
            "repository PPL evaluator consumes the first 20 Wikipedia texts"
        ),
        "utility_max_length": int(args.utility_max_length),
        "utility_batch_size": int(args.utility_batch_size),
        "benchmark_examples_seen": 0,
        "benchmark_retain_examples_seen": 0,
        "heldout_benchmark_probes_seen": 0,
        "zero_unlearn_alignment": (
            "fixed external Wikipedia uncentered second moment; this cache uses "
            "final LM-head input states rather than a layer-specific MLP key"
        ),
    }
    payload = {"second_moment": moment, "metadata": metadata}
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)

    print("Wikipedia SURE statistic:", output_path)
    print("predictor states:", metadata["predictor_hidden_state_count"])
    print("hidden size:", metadata["hidden_size"])
    print("second-moment trace:", metadata["second_moment_trace"])
    print("model probe:", metadata["model_probe_sha256"])


if __name__ == "__main__":
    main()
