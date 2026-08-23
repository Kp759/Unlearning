#!/usr/bin/env python3
"""Build a PPL-disjoint Wikipedia second moment for SURE-v3 LM-head repair.

The statistic is aligned with the object edited by Stage 2: the final transformer
hidden state fed to ``lm_head``.  By default we deterministically select 1,000
Wikipedia documents and exactly 100 valid next-token states from each, yielding
100,000 LM-head input states:

    C_wiki = (1/N) sum_n h_n h_n^T.

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
    states_per_document: int,
    exclude_first: int,
    max_length: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if count <= 0 or states_per_document <= 0 or max_length <= states_per_document:
        raise ValueError("documents/states/max-length settings are inconsistent")
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
        if len(ids) - 1 < states_per_document:
            continue
        chosen.append(
            {
                "index": int(index),
                "text": text,
                "title": title_from_text(text, int(index)),
                "token_count_truncated": int(len(ids)),
                "text_sha256": sha256_text(text),
            }
        )
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(
            f"Could select only {len(chosen)}/{count} PPL-disjoint documents with at least "
            f"{states_per_document + 1} tokens after truncation"
        )
    return chosen


def evenly_spaced_positions(length: int, count: int, device: torch.device) -> torch.Tensor:
    pos = torch.linspace(0, length - 2, steps=count, device=device).round().long()
    if int(torch.unique(pos).numel()) != count:
        raise RuntimeError("evenly spaced state positions unexpectedly contain duplicates")
    return pos


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
    state_count = 0

    model.eval()
    for start in range(0, len(docs), a.batch_size):
        batch_docs = docs[start : start + a.batch_size]
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
            pos = evenly_spaced_positions(length, a.states_per_document, hidden.device)
            selected.append(hidden[i].index_select(0, pos).float())
        x = torch.cat(selected, dim=0)
        moment.addmm_(x.transpose(0, 1), x)
        state_count += int(x.shape[0])
        done = min(start + len(batch_docs), len(docs))
        if done == len(docs) or done % 100 == 0:
            print(f"Wikipedia LM-head covariance: {done}/{len(docs)} docs, {state_count} states")
        del out, hidden, x, selected

    expected = int(a.documents) * int(a.states_per_document)
    if state_count != expected:
        raise RuntimeError(f"state-count mismatch: got {state_count}, expected {expected}")
    covariance = (moment / float(state_count)).detach().cpu()
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    return covariance, state_count


def main() -> None:
    a = parse_args()
    if min(a.documents, a.states_per_document, a.max_length, a.batch_size) <= 0:
        raise ValueError("all size settings must be positive")
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
        states_per_document=a.states_per_document,
        exclude_first=a.exclude_first,
        max_length=a.max_length,
        seed=a.corpus_seed,
    )
    print(
        f"Selected {len(docs)} deterministic PPL-disjoint Wikipedia documents from {len(texts)} rows; "
        f"collecting {a.states_per_document} LM-head states/document"
    )
    covariance, state_count = accumulate_second_moment(model, tok, docs, a)
    hidden_size = int(covariance.shape[0])
    symmetry_error = float((covariance - covariance.T).abs().max().item())
    diag = covariance.diag()

    metadata = {
        "schema_version": 2,
        "kind": "uncentered second moment of final hidden states feeding lm_head",
        "formula": "C=(1/N) sum h h^T",
        "model_path": str(a.model_path),
        "wikidata_dir": str(corpus),
        **dataset_metadata,
        "corpus_seed": int(a.corpus_seed),
        "document_count": int(len(docs)),
        "states_per_document": int(a.states_per_document),
        "state_count": int(state_count),
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
                "text_sha256": d["text_sha256"],
            }
            for d in docs
        ],
    }
    torch.save({"covariance": covariance, "metadata": metadata}, output)
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("===== WIKIPEDIA LM-HEAD COVARIANCE COMPLETE =====")
    print("output:", output)
    print("documents:", len(docs))
    print("states:", state_count)
    print("excluded_first_for_ppl:", a.exclude_first)
    print("hidden_size:", hidden_size)
    print("symmetry_error_max_abs:", symmetry_error)


if __name__ == "__main__":
    main()
