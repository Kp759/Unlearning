#!/usr/bin/env python3
"""Build a PPL-disjoint Wikipedia second moment for SURE-v3 LM-head repair.

The statistic is aligned with the object edited by Stage 2: the final transformer
hidden state fed to ``lm_head``.  ``--documents`` is a requested source-document
count and ``--states-per-document`` defines the nominal target sample count:

    N_target = requested_documents * states_per_document.

If the local PPL-disjoint Wikipedia corpus contains fewer usable source documents
than requested, the builder uses every available document and redistributes the
same ``N_target`` predictor-state samples as evenly as possible across those
actual documents.  Thus a request for 1,000 documents x 100 states still produces
exactly 100,000 LM-head samples on the repository's current small local corpus,
but metadata truthfully records the smaller number of unique source documents.

Within each document, samples are balanced over all valid causal predictor
positions.  When the per-document quota exceeds the number of unique positions,
positions are reused deterministically and as evenly as possible.  Metadata
reports sampled-state count, unique-position count, replacement count, requested
and actual source-document counts, and the exact quota per source document.

The corpus loader is shared with ``build_sure_wikipedia_stats.py`` and the first
20 Wikipedia documents are excluded by default because this repository's PPL
probe consumes those texts.  Thus the external utility geometry is disjoint from
the current PPL evaluation probe.

No MQuAKE retain, AtomicGen, target_new, paraphrase, neighborhood, or multihop
field is read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import gagd_compare as gagd
from build_sure_wikipedia_stats import load_wikipedia_train


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--output", required=True)
    p.add_argument("--documents", type=int, default=1000)
    p.add_argument("--states-per-document", type=int, default=100)
    p.add_argument("--exclude-first", type=int, default=20)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--corpus-seed", type=int, default=1729)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def title_from_text(text: str, index: int) -> str:
    first = next((line.strip() for line in str(text).splitlines() if line.strip()), "")
    if not first:
        first = " ".join(str(text).split()[:12])
    return first[:200] if first else f"Wikipedia document {index}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_documents(
    texts: Sequence[str],
    tok,
    *,
    count: int,
    exclude_first: int,
    max_length: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Choose up to ``count`` deterministic PPL-disjoint usable documents."""
    if count <= 0 or max_length < 2:
        raise ValueError("documents/max-length settings are inconsistent")
    if exclude_first < 0 or exclude_first >= len(texts):
        raise ValueError("exclude-first is outside the Wikipedia corpus")

    order = list(range(exclude_first, len(texts)))
    random.Random(seed).shuffle(order)
    chosen: List[Dict[str, Any]] = []
    for index in order:
        text = str(texts[int(index)]).strip()
        if not text:
            continue
        ids = tok(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        available = int(len(ids) - 1)
        if available <= 0:
            continue
        chosen.append(
            {
                "index": int(index),
                "text": text,
                "title": title_from_text(text, int(index)),
                "token_count_truncated": int(len(ids)),
                "available_predictor_positions": available,
                "text_sha256": sha256_text(text),
            }
        )
        if len(chosen) == count:
            break

    if not chosen:
        raise RuntimeError(
            "No PPL-disjoint non-empty Wikipedia documents with a causal predictor state"
        )
    if len(chosen) < count:
        print(
            "Wikipedia document request exceeds the usable PPL-disjoint local corpus; "
            f"requested {count}, using all {len(chosen)} usable documents. "
            "The total predictor-state target is preserved exactly."
        )
    return chosen


def balanced_predictor_positions(
    length: int,
    count: int,
    device: torch.device,
) -> torch.Tensor:
    """Return exactly ``count`` balanced positions in [0, length-2]."""
    available = int(length) - 1
    if available <= 0 or count <= 0:
        raise ValueError("balanced predictor sampling requires positive sizes")

    if count <= available:
        pos = torch.linspace(
            0,
            available - 1,
            steps=count,
            device=device,
            dtype=torch.float64,
        ).round().long()
        if int(torch.unique(pos).numel()) != count:
            raise RuntimeError("evenly spaced predictor positions contain duplicates")
        return pos

    base = torch.arange(available, device=device, dtype=torch.long)
    full_repeats, remainder = divmod(int(count), available)
    pieces: List[torch.Tensor] = []
    if full_repeats:
        pieces.append(base.repeat(full_repeats))
    if remainder:
        extra = torch.linspace(
            0,
            available - 1,
            steps=remainder,
            device=device,
            dtype=torch.float64,
        ).round().long()
        pieces.append(extra)
    pos = torch.cat(pieces, dim=0)
    if int(pos.numel()) != int(count):
        raise RuntimeError("balanced predictor sampling produced wrong count")
    return pos


def balanced_document_quotas(total_samples: int, document_count: int) -> List[int]:
    """Split ``total_samples`` across documents with quotas differing by <=1."""
    if total_samples <= 0 or document_count <= 0:
        raise ValueError("sample/document counts must be positive")
    base, remainder = divmod(int(total_samples), int(document_count))
    if base <= 0:
        raise ValueError("target state count must be at least the actual document count")
    quotas = [base + (1 if i < remainder else 0) for i in range(document_count)]
    if sum(quotas) != int(total_samples):
        raise RuntimeError("document quota construction lost samples")
    return quotas


@torch.no_grad()
def accumulate_second_moment(model, tok, docs: Sequence[Dict[str, Any]], a: argparse.Namespace):
    prefix = str(getattr(model, "base_model_prefix", ""))
    backbone = getattr(model, prefix, None) if prefix else None
    if backbone is None or backbone is model:
        backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("Expected a causal LM exposing a final-hidden backbone")

    device = gagd.first_device(model)
    hidden_size = int(model.get_output_embeddings().weight.shape[1])
    moment = torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=device)
    sampled_state_count = 0
    unique_position_count = 0
    replacement_sample_count = 0
    document_sampling: List[Dict[str, Any]] = []

    target_state_count = int(a.documents) * int(a.states_per_document)
    quotas = balanced_document_quotas(target_state_count, len(docs))

    model.eval()
    for start in range(0, len(docs), a.batch_size):
        batch_docs = docs[start : start + a.batch_size]
        batch_quotas = quotas[start : start + len(batch_docs)]
        encoded = tok(
            [d["text"] for d in batch_docs],
            padding=True,
            truncation=True,
            max_length=a.max_length,
            return_tensors="pt",
        ).to(device)
        out = backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            use_cache=False,
            return_dict=True,
        )
        hidden = out.last_hidden_state
        lengths = encoded["attention_mask"].sum(dim=1)
        selected: List[torch.Tensor] = []

        for i, length_tensor in enumerate(lengths):
            length = int(length_tensor.item())
            available = length - 1
            if available <= 0:
                raise RuntimeError("chosen Wikipedia document lost all predictor states")
            quota = int(batch_quotas[i])
            pos = balanced_predictor_positions(length, quota, hidden.device)
            unique_used = int(torch.unique(pos).numel())
            selected.append(hidden[i].index_select(0, pos).float())
            unique_position_count += unique_used
            replacement = quota - unique_used
            replacement_sample_count += replacement
            document_sampling.append(
                {
                    "index": int(batch_docs[i]["index"]),
                    "available_predictor_positions": int(available),
                    "sampled_states": quota,
                    "unique_predictor_positions_used": int(unique_used),
                    "replacement_samples": int(replacement),
                }
            )

        x = torch.cat(selected, dim=0)
        moment.addmm_(x.transpose(0, 1), x)
        sampled_state_count += int(x.shape[0])
        done = min(start + len(batch_docs), len(docs))
        if done == len(docs) or done % 50 == 0:
            print(
                f"Wikipedia LM-head covariance: {done}/{len(docs)} actual docs, "
                f"{sampled_state_count}/{target_state_count} sampled states, "
                f"{unique_position_count} unique positions"
            )
        del out, hidden, x, selected

    if sampled_state_count != target_state_count:
        raise RuntimeError(
            f"sampled-state-count mismatch: got {sampled_state_count}, "
            f"expected {target_state_count}"
        )
    covariance = (moment / float(sampled_state_count)).detach().cpu()
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    audit = {
        "requested_document_count": int(a.documents),
        "actual_document_count": int(len(docs)),
        "document_cap_applied": bool(len(docs) < int(a.documents)),
        "nominal_requested_states_per_document": int(a.states_per_document),
        "target_state_count": int(target_state_count),
        "sampled_state_count": int(sampled_state_count),
        "minimum_actual_document_quota": int(min(quotas)),
        "maximum_actual_document_quota": int(max(quotas)),
        "unique_predictor_position_count": int(unique_position_count),
        "replacement_sample_count": int(replacement_sample_count),
        "replacement_fraction": float(replacement_sample_count / sampled_state_count),
        "sampling_rule": (
            "requested total state count preserved exactly; samples distributed "
            "as evenly as possible across all usable PPL-disjoint source documents; "
            "within each source document positions are balanced with deterministic reuse"
        ),
        "documents": document_sampling,
    }
    return covariance, audit


def main() -> None:
    a = parse_args()
    if min(a.documents, a.states_per_document, a.max_length, a.batch_size) <= 0:
        raise ValueError("all size settings must be positive")
    if a.max_length < 2:
        raise ValueError("max-length must be at least 2")

    corpus = Path(a.wikidata_dir).resolve()
    if not corpus.exists():
        raise FileNotFoundError(f"Wikipedia corpus not found: {corpus}")
    output = Path(a.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    gagd.set_seed(a.corpus_seed)
    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    texts, dataset_metadata = load_wikipedia_train(corpus)
    docs = choose_documents(
        texts,
        tok,
        count=a.documents,
        exclude_first=a.exclude_first,
        max_length=a.max_length,
        seed=a.corpus_seed,
    )
    target_state_count = int(a.documents) * int(a.states_per_document)
    print(
        f"Selected {len(docs)} actual deterministic PPL-disjoint Wikipedia documents "
        f"from {len(texts)} rows (requested {a.documents}); drawing exactly "
        f"{target_state_count} balanced LM-head samples in total"
    )
    covariance, sampling_audit = accumulate_second_moment(model, tok, docs, a)
    sampled_state_count = int(sampling_audit["sampled_state_count"])
    hidden_size = int(covariance.shape[0])
    symmetry_error = float((covariance - covariance.T).abs().max().item())
    diag = covariance.diag()

    metadata = {
        "schema_version": 4,
        "kind": "uncentered second moment of balanced sampled final hidden states feeding lm_head",
        "formula": "C=(1/N) sum h h^T over balanced source-document samples",
        "model_path": str(a.model_path),
        "wikidata_dir": str(corpus),
        **dataset_metadata,
        "corpus_seed": int(a.corpus_seed),
        "requested_document_count": int(a.documents),
        "document_count": int(len(docs)),
        "actual_document_count": int(len(docs)),
        "document_cap_applied": bool(len(docs) < int(a.documents)),
        "states_per_document": int(a.states_per_document),
        "nominal_requested_states_per_document": int(a.states_per_document),
        "target_state_count": int(target_state_count),
        "state_count": sampled_state_count,
        "sampled_state_count": sampled_state_count,
        "minimum_actual_document_quota": int(sampling_audit["minimum_actual_document_quota"]),
        "maximum_actual_document_quota": int(sampling_audit["maximum_actual_document_quota"]),
        "unique_predictor_position_count": int(sampling_audit["unique_predictor_position_count"]),
        "replacement_sample_count": int(sampling_audit["replacement_sample_count"]),
        "replacement_fraction": float(sampling_audit["replacement_fraction"]),
        "sampling_rule": sampling_audit["sampling_rule"],
        "all_samples_unique": bool(sampling_audit["replacement_sample_count"] == 0),
        "excluded_prefix_document_count": int(a.exclude_first),
        "excluded_prefix_reason": "repository PPL evaluator consumes the first 20 Wikipedia texts",
        "ppl_probe_disjoint": bool(a.exclude_first >= 20),
        "max_length": int(a.max_length),
        "hidden_size": hidden_size,
        "storage_dtype": "float32",
        "symmetry_error_max_abs": symmetry_error,
        "diag_min": float(diag.min().item()),
        "diag_mean": float(diag.mean().item()),
        "diag_max": float(diag.max().item()),
        "benchmark_retain_seen": 0,
        "official_atomicgen_seen": 0,
        "target_new_seen": False,
        "external_wikipedia_used": True,
        "documents": [
            {
                "index": d["index"],
                "title": d["title"],
                "token_count_truncated": d["token_count_truncated"],
                "available_predictor_positions": d["available_predictor_positions"],
                "text_sha256": d["text_sha256"],
            }
            for d in docs
        ],
        "document_sampling": sampling_audit["documents"],
    }
    torch.save({"covariance": covariance, "metadata": metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("===== WIKIPEDIA LM-HEAD COVARIANCE COMPLETE =====")
    print("output:", output)
    print("requested_documents:", a.documents)
    print("actual_documents:", len(docs))
    print("document_cap_applied:", len(docs) < int(a.documents))
    print("target_states:", target_state_count)
    print("sampled_states:", sampled_state_count)
    print("actual_document_quota_min:", sampling_audit["minimum_actual_document_quota"])
    print("actual_document_quota_max:", sampling_audit["maximum_actual_document_quota"])
    print("unique_predictor_positions:", sampling_audit["unique_predictor_position_count"])
    print("replacement_samples:", sampling_audit["replacement_sample_count"])
    print("replacement_fraction:", sampling_audit["replacement_fraction"])
    print("excluded_first_for_ppl:", a.exclude_first)
    print("hidden_size:", hidden_size)
    print("symmetry_error_max_abs:", symmetry_error)


if __name__ == "__main__":
    main()
