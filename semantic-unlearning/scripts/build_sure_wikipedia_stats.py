#!/usr/bin/env python3
"""Cache dataset-independent Wikipedia utility statistics for SURE-LM.

This is a one-time, model-specific preprocessing step shared by MCF and ZsRE.
It requests ``--sample-size`` Wikipedia documents (100,000 by default), caps
that request to the eligible local dataset exactly as ZeroUnlearn's statistics
loader does, collects their causal predictor states, and stores both

    C_U = (1 / N) sum_t h_t h_t^T

and a fixed predictor-state/Base-partition candidate reservoir
``(h_u, log Z_u)``. The reservoir is spread across token positions rather than
limited to one state per document, so a capped pilot corpus can still provide
many KL candidates. The former defines the contrastive generalized-eigen
basis. The latter lets any downstream dataset adapter select contexts where
its edited token rows have high Base probability and compute exact joint
sparse-head utility KL without another Wikipedia model pass.

The statistic is aligned with ZeroUnlearn's use of fixed Wikipedia second
moments, but lives at the final hidden state that feeds the LM head.  It never
contains MCF or ZsRE examples.  By default, the first 20 Wikipedia documents
are excluded so the texts used by this repository's PPL probe remain unseen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import gagd_compare as gagd


UTILITY_PROTOCOL = "sure_wikipedia_hidden_moment_and_predictor_reservoir_v3"
CACHE_SCHEMA_VERSION = 3
DEFAULT_SAMPLE_SIZE = 100_000
DEFAULT_UTILITY_PROMPT_COUNT = 100_000
DEFAULT_UTILITY_LOGIT_BATCH_SIZE = 64
MODEL_PROBE_VALUES_PER_TENSOR = 16
HASH_CHUNK_ELEMENTS = 1 << 20


def sha256_tensor(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous().reshape(-1)
    digest = hashlib.sha256()
    for start in range(0, int(value.numel()), HASH_CHUNK_ELEMENTS):
        chunk = value[start : start + HASH_CHUNK_ELEMENTS].float().contiguous()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


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
    receipt_path = path / "sure_wikipedia_corpus_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("Wikipedia corpus receipt must be a JSON object")
        if int(receipt.get("actual_article_count", -1)) != row_count:
            raise ValueError("Wikipedia corpus receipt row count mismatch")
        metadata["corpus_receipt"] = receipt
        metadata["corpus_receipt_sha256"] = hashlib.sha256(
            receipt_path.read_bytes()
        ).hexdigest()
    else:
        metadata["corpus_receipt"] = None
        metadata["corpus_receipt_sha256"] = None
    return texts, metadata


def predictor_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Select attended states that have a following token in the same text."""
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must be [batch, sequence]")
    mask = attention_mask.bool()
    lengths = mask.sum(dim=1)
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    return mask & (positions < (lengths - 1).clamp_min(0).unsqueeze(1))


def deterministic_predictor_position(
    document_index: int, attended_length: int, utility_seed: int
) -> int:
    """Choose one reproducible causal predictor position from a document."""
    if attended_length < 2:
        raise ValueError("a utility prompt document needs at least two tokens")
    digest = hashlib.sha256(
        f"{int(utility_seed)}:{int(document_index)}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (int(attended_length) - 1)


def deterministic_predictor_positions(
    document_index: int,
    attended_length: int,
    utility_seed: int,
    count: int,
) -> List[int]:
    """Choose distinct reproducible predictor positions within one document."""
    available = int(attended_length) - 1
    if available <= 0 or count <= 0:
        return []
    take = min(int(count), available)
    if take == 1:
        return [
            deterministic_predictor_position(
                document_index, attended_length, utility_seed
            )
        ]
    if take == available:
        return list(range(available))
    digest = hashlib.sha256(
        f"positions:{int(utility_seed)}:{int(document_index)}".encode("ascii")
    ).digest()
    seed = int.from_bytes(digest[:8], "big")
    return sorted(random.Random(seed).sample(range(available), take))


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
def base_logsumexp_for_hidden(
    model: torch.nn.Module,
    hidden_states: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Return Base ``logsumexp`` values for cached LM-head input states."""
    if hidden_states.ndim != 2 or hidden_states.shape[0] == 0:
        raise ValueError("utility hidden states must be non-empty [N, hidden]")
    if batch_size <= 0:
        raise ValueError("utility logit batch size must be positive")
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise ValueError("Model must expose output embeddings")
    head_device = output.weight.device
    values: List[torch.Tensor] = []
    for start in range(0, int(hidden_states.shape[0]), batch_size):
        batch = hidden_states[start : start + batch_size].to(
            device=head_device,
            dtype=output.weight.dtype,
        )
        logits = output(batch)
        values.append(torch.logsumexp(logits.float(), dim=-1).detach().cpu())
    result = torch.cat(values, dim=0).float().contiguous()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Base utility log-partitions are non-finite")
    return result


@torch.no_grad()
def build_second_moment(
    model: torch.nn.Module,
    tok: Any,
    texts: Sequence[str],
    *,
    document_order: Sequence[int],
    utility_prompt_document_indices: Sequence[int],
    utility_seed: int,
    utility_prompt_count: int | None = None,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    if not document_order or max_length < 2 or batch_size <= 0:
        raise ValueError("invalid Wikipedia statistic dimensions")
    output = model.get_output_embeddings()
    if output is None or not hasattr(output, "weight"):
        raise ValueError("Model must expose output embeddings")
    hidden_size = int(output.weight.shape[1])
    unnormalized = None
    collected = 0
    forwarded_indices: List[int] = []
    prompt_document_order = [
        int(index) for index in utility_prompt_document_indices
    ]
    prompt_documents = set(prompt_document_order)
    if not prompt_documents or not prompt_documents.issubset(
        {int(index) for index in document_order}
    ):
        raise ValueError("utility prompt documents must be a non-empty sampled subset")
    utility_hidden_rows: List[torch.Tensor] = []
    utility_prompt_records: List[Dict[str, int]] = []
    requested_prompt_count = (
        len(prompt_document_order)
        if utility_prompt_count is None
        else int(utility_prompt_count)
    )
    if requested_prompt_count <= 0:
        raise ValueError("utility prompt count must be positive")
    prompt_documents_remaining = len(prompt_document_order)
    prompt_slots_remaining = requested_prompt_count
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
            attended_lengths = encoded["attention_mask"].sum(dim=1).tolist()
            for row_index, document_index in enumerate(indices):
                length = int(attended_lengths[row_index])
                if document_index not in prompt_documents:
                    continue
                quota = math.ceil(
                    prompt_slots_remaining / max(1, prompt_documents_remaining)
                )
                positions = deterministic_predictor_positions(
                    document_index,
                    length,
                    utility_seed,
                    min(quota, prompt_slots_remaining),
                )
                for position in positions:
                    utility_hidden_rows.append(
                        hidden[row_index, position].detach().cpu().contiguous()
                    )
                    utility_prompt_records.append(
                        {
                            "document_index": int(document_index),
                            "predictor_token_position": int(position),
                            "attended_token_count": int(length),
                        }
                    )
                prompt_slots_remaining -= len(positions)
                prompt_documents_remaining -= 1
            forwarded_indices.extend(indices)
            if collected and collected % 10_000 < int(states.shape[0]):
                print(f"Wikipedia predictor states collected: {collected}")
    finally:
        tok.padding_side = old_padding_side

    if collected <= 0 or unnormalized is None:
        raise RuntimeError(
            "Selected Wikipedia documents produced zero predictor states"
        )
    if not utility_hidden_rows:
        raise RuntimeError("Selected Wikipedia documents produced no utility prompts")
    moment, trace = finalize_second_moment(unnormalized, collected)
    moment = moment.detach().cpu().contiguous()
    utility_hidden = torch.stack(utility_hidden_rows, dim=0).contiguous()
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
        "utility_prompt_count": int(utility_hidden.shape[0]),
        "requested_utility_prompt_count_within_builder": requested_prompt_count,
        "utility_prompt_records": utility_prompt_records,
        "utility_hidden_sha256": sha256_tensor(utility_hidden),
        "utility_prompt_sampling": (
            "deterministic distinct predictor positions spread across shuffled "
            "utility documents until the requested candidate reservoir is full"
        ),
        "state_selection": (
            "attended final-layer LM-head input states with a following token "
            "from the fixed sampled document set"
        ),
    }
    return moment, utility_hidden, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--wikidata-dir", default="data/wikidata")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument(
        "--require-min-documents",
        type=int,
        default=0,
        help=(
            "Fail instead of silently building a pilot cache when fewer than "
            "this many eligible documents are available"
        ),
    )
    parser.add_argument(
        "--require-min-prompts",
        type=int,
        default=0,
        help="Fail when the resulting predictor-state reservoir is smaller",
    )
    parser.add_argument(
        "--require-corpus-protocol",
        default="",
        help="Require a prepared-corpus receipt with this exact protocol",
    )
    parser.add_argument("--utility-seed", type=int, default=1)
    parser.add_argument("--exclude-first", type=int, default=20)
    parser.add_argument("--utility-max-length", type=int, default=4096)
    parser.add_argument("--utility-batch-size", type=int, default=1)
    parser.add_argument(
        "--utility-prompt-count",
        type=int,
        default=DEFAULT_UTILITY_PROMPT_COUNT,
        help="Number of disjoint Wikipedia predictor states cached for exact KL",
    )
    parser.add_argument(
        "--utility-logit-batch-size",
        type=int,
        default=DEFAULT_UTILITY_LOGIT_BATCH_SIZE,
    )
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
    if args.require_min_documents < 0 or args.require_min_prompts < 0:
        raise ValueError("minimum document/prompt requirements must be non-negative")
    if args.require_min_documents > args.sample_size:
        raise ValueError("require-min-documents cannot exceed sample-size")
    if args.require_min_prompts > args.utility_prompt_count:
        raise ValueError("require-min-prompts cannot exceed utility-prompt-count")
    if (
        args.utility_max_length < 2
        or args.utility_batch_size <= 0
        or args.utility_prompt_count <= 0
        or args.utility_logit_batch_size <= 0
    ):
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
    if args.require_corpus_protocol:
        receipt = dataset_metadata.get("corpus_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("protocol") != str(
            args.require_corpus_protocol
        ):
            raise RuntimeError(
                "Wikipedia corpus lacks the required prepared-corpus protocol: "
                f"{args.require_corpus_protocol}"
            )
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
    if len(selected_order) < int(args.require_min_documents):
        raise RuntimeError(
            "Wikipedia corpus is too small for the requested non-pilot cache: "
            f"required at least {args.require_min_documents} eligible documents, "
            f"found {len(selected_order)} after excluding {args.exclude_first}"
        )
    if args.sample_size > len(eligible):
        print(
            "Wikipedia sample request exceeds eligible local corpus; "
            f"using all {len(eligible)} eligible documents, matching "
            "ZeroUnlearn's capped-sample behavior"
        )

    prompt_document_indices = selected_order[
        : min(args.utility_prompt_count, len(selected_order))
    ]
    moment, utility_hidden, statistic_report = build_second_moment(
        model,
        tok,
        texts,
        document_order=selected_order,
        utility_prompt_document_indices=prompt_document_indices,
        utility_seed=args.utility_seed,
        utility_prompt_count=args.utility_prompt_count,
        device=device,
        max_length=args.utility_max_length,
        batch_size=args.utility_batch_size,
    )
    if int(utility_hidden.shape[0]) < int(args.require_min_prompts):
        raise RuntimeError(
            "Wikipedia predictor reservoir is too small: required at least "
            f"{args.require_min_prompts}, built {int(utility_hidden.shape[0])}"
        )
    base_logsumexp = base_logsumexp_for_hidden(
        model,
        utility_hidden,
        device=device,
        batch_size=args.utility_logit_batch_size,
    )
    metadata = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "protocol": UTILITY_PROTOCOL,
        **dataset_metadata,
        **identity,
        **statistic_report,
        "utility_seed": int(args.utility_seed),
        "requested_document_sample_size": int(args.sample_size),
        "actual_document_sample_size": len(selected_order),
        "required_minimum_document_sample_size": int(args.require_min_documents),
        "sample_size_cap_policy": "min(requested, eligible_local_documents)",
        "excluded_prefix_document_count": int(args.exclude_first),
        "excluded_prefix_reason": (
            "repository PPL evaluator consumes the first 20 Wikipedia texts"
        ),
        "utility_max_length": int(args.utility_max_length),
        "utility_batch_size": int(args.utility_batch_size),
        "requested_utility_prompt_count": int(args.utility_prompt_count),
        "actual_utility_prompt_count": int(utility_hidden.shape[0]),
        "required_minimum_utility_prompt_count": int(args.require_min_prompts),
        "utility_logit_batch_size": int(args.utility_logit_batch_size),
        "base_logsumexp_sha256": sha256_tensor(base_logsumexp),
        "benchmark_examples_seen": 0,
        "benchmark_retain_examples_seen": 0,
        "heldout_benchmark_probes_seen": 0,
        "zero_unlearn_alignment": (
            "fixed external Wikipedia uncentered second moment plus a fixed "
            "prompt-state/Base-partition sample; no benchmark retain examples"
        ),
    }
    payload = {
        "second_moment": moment,
        "utility_hidden_states": utility_hidden,
        "base_logsumexp": base_logsumexp,
        "metadata": metadata,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output_path)

    print("Wikipedia SURE statistic:", output_path)
    print("predictor states:", metadata["predictor_hidden_state_count"])
    print("hidden size:", metadata["hidden_size"])
    print("second-moment trace:", metadata["second_moment_trace"])
    print("exact-KL utility prompts:", metadata["actual_utility_prompt_count"])
    print("model probe:", metadata["model_probe_sha256"])


if __name__ == "__main__":
    main()
