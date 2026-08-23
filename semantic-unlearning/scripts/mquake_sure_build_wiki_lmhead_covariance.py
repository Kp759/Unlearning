#!/usr/bin/env python3
"""Build a reusable Wikipedia second moment for SURE-v3 LM-head repair.

The statistic is intentionally aligned with the object edited by Stage 2: the
final transformer hidden state that is fed to ``lm_head``.  We deterministically
select ``--documents`` Wikipedia documents from the local ``data/wikidata``
corpus and take exactly ``--states-per-document`` valid next-token hidden states
from each selected document.  The default is therefore 1,000 documents x 100
states = 100,000 LM-head input states.

The saved matrix is the uncentered second moment

    C_wiki = (1/N) sum_n h_n h_n^T,

matching the covariance/second-moment spirit used by ROME/MEMIT while operating
in the exact representation space touched by SURE's output-head edits.

This is an external unlabeled utility statistic.  It never reads MQuAKE retain,
AtomicGen, target_new, paraphrase, neighborhood, or multihop fields.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from datasets import Dataset, DatasetDict, load_from_disk

import gagd_compare as gagd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--output", required=True)
    p.add_argument("--documents", type=int, default=1000)
    p.add_argument("--states-per-document", type=int, default=100)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--corpus-seed", type=int, default=1729)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args()


def dataset_train(path: Path) -> Dataset:
    obj = load_from_disk(str(path))
    if isinstance(obj, DatasetDict):
        if "train" not in obj:
            raise RuntimeError(f"DatasetDict at {path} has no train split")
        ds = obj["train"]
    else:
        ds = obj
    if not isinstance(ds, Dataset):
        raise RuntimeError(f"Unsupported dataset object at {path}: {type(ds)!r}")
    return ds


def text_from_row(row: Dict[str, Any]) -> str:
    for key in ("text", "content", "article", "document", "body"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and len(value.split()) >= 20:
            return value.strip()
    return ""


def title_from_row(row: Dict[str, Any], text: str, index: int) -> str:
    for key in ("title", "name", "page_title"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:200]
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    if not first:
        first = " ".join(text.split()[:12])
    return first[:200] if first else f"Wikipedia document {index}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def choose_documents(
    ds: Dataset,
    tok,
    *,
    count: int,
    states_per_document: int,
    max_length: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if count <= 0 or states_per_document <= 0 or max_length <= states_per_document:
        raise ValueError("documents/states/max-length settings are inconsistent")
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    chosen: List[Dict[str, Any]] = []
    for index in order:
        row = dict(ds[int(index)])
        text = text_from_row(row)
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
                "title": title_from_row(row, text, int(index)),
                "token_count_truncated": int(len(ids)),
                "text_sha256": sha256_text(text),
            }
        )
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise RuntimeError(
            f"Could select only {len(chosen)}/{count} documents with at least "
            f"{states_per_document + 1} tokens after truncation"
        )
    return chosen


def evenly_spaced_positions(length: int, count: int, device: torch.device) -> torch.Tensor:
    # Positions 0..length-2 predict an observed next token.  Qualifying documents
    # guarantee at least count such positions, so rounded linspace is unique in
    # practice; assert this rather than silently duplicating states.
    pos = torch.linspace(0, length - 2, steps=count, device=device).round().long()
    if int(torch.unique(pos).numel()) != count:
        raise RuntimeError("evenly spaced state positions unexpectedly contain duplicates")
    return pos


@torch.no_grad()
def accumulate_second_moment(model, tok, docs: Sequence[Dict[str, Any]], a: argparse.Namespace):
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("Expected a causal LM exposing .model final hidden states")
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

    ds = dataset_train(corpus)
    docs = choose_documents(
        ds,
        tok,
        count=a.documents,
        states_per_document=a.states_per_document,
        max_length=a.max_length,
        seed=a.corpus_seed,
    )
    print(
        f"Selected {len(docs)} deterministic Wikipedia documents from {len(ds)} rows; "
        f"collecting {a.states_per_document} LM-head states/document"
    )
    covariance, state_count = accumulate_second_moment(model, tok, docs, a)
    hidden_size = int(covariance.shape[0])
    symmetry_error = float((covariance - covariance.T).abs().max().item())
    diag = covariance.diag()

    metadata = {
        "schema_version": 1,
        "kind": "uncentered second moment of final hidden states feeding lm_head",
        "formula": "C=(1/N) sum h h^T",
        "model_path": str(a.model_path),
        "wikidata_dir": str(corpus),
        "corpus_seed": int(a.corpus_seed),
        "document_count": int(len(docs)),
        "states_per_document": int(a.states_per_document),
        "state_count": int(state_count),
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
    print("hidden_size:", hidden_size)
    print("symmetry_error_max_abs:", symmetry_error)


if __name__ == "__main__":
    main()
