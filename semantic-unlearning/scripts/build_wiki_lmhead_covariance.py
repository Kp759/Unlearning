#!/usr/bin/env python3
"""Build final-hidden-state Wikipedia covariance for LM-head utility preservation.

The cache is designed for SURE MCF Stage 2, where only sparse LM-head rows are
edited.  For each of 1,000 Wikipedia documents we sample 100 causal prediction
positions from a single forward pass and accumulate the second moment

    C = E[h h^T]

of the final transformer hidden state h presented to the LM head.

Important leakage rule: by default the first 20 Wikipedia train documents are
reserved because mcf_zero_unlearn_official_eval.py uses exactly those texts for
its PPL evaluation.  They are never included in this covariance cache.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import Dataset, DatasetDict, load_from_disk

import gagd_compare as gagd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--wikidata-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--num-docs", type=int, default=1000)
    p.add_argument("--prompts-per-doc", type=int, default=100)
    p.add_argument("--skip-first-docs", type=int, default=20,
                   help="Reserve official PPL texts; evaluator uses first 20 train docs.")
    p.add_argument("--max-doc-tokens", type=int, default=1024)
    p.add_argument("--min-context-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    a = p.parse_args(list(argv) if argv is not None else None)
    if min(a.num_docs, a.prompts_per_doc, a.max_doc_tokens,
           a.min_context_tokens, a.batch_size) <= 0:
        p.error("document/prompt/token/batch counts must be positive")
    if a.skip_first_docs < 0:
        p.error("skip-first-docs must be non-negative")
    if a.min_context_tokens > a.max_doc_tokens:
        p.error("min-context-tokens cannot exceed max-doc-tokens")
    return a


def train_split(raw: Any):
    if isinstance(raw, DatasetDict):
        if "train" not in raw:
            raise ValueError("Wikipedia DatasetDict has no train split")
        return raw["train"]
    if isinstance(raw, Dataset):
        return raw
    if hasattr(raw, "keys") and "train" in raw:
        return raw["train"]
    return raw


def final_hidden(model, encoded: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return last transformer hidden state without materializing LM logits."""
    backbone = getattr(model, "model", None)
    if backbone is not None:
        out = backbone(
            input_ids=encoded["input_ids"],
            attention_mask=encoded.get("attention_mask"),
            use_cache=False,
            return_dict=True,
        )
        value = getattr(out, "last_hidden_state", None)
        if value is not None:
            return value
    out = model(**encoded, output_hidden_states=True, use_cache=False)
    return out.hidden_states[-1]


def sample_positions(length: int, count: int, minimum: int, device: torch.device) -> torch.Tensor:
    if length < minimum:
        return torch.empty((0,), dtype=torch.long, device=device)
    lo = minimum - 1
    hi = length - 1
    if count == 1:
        return torch.tensor([hi], dtype=torch.long, device=device)
    # Exactly count positions.  Very short documents may repeat a position;
    # duplicates are reported in metadata instead of silently reducing count.
    return torch.linspace(lo, hi, steps=count, device=device).round().long()


@torch.no_grad()
def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    wikidata_dir = Path(a.wikidata_dir).resolve()
    if not wikidata_dir.exists():
        raise FileNotFoundError(wikidata_dir)
    raw = load_from_disk(str(wikidata_dir))
    ds = train_split(raw)
    if "text" not in getattr(ds, "column_names", []):
        raise ValueError("Wikipedia dataset must contain a text column")

    eligible = list(range(int(a.skip_first_docs), len(ds)))
    if len(eligible) < int(a.num_docs):
        raise ValueError(
            f"Need {a.num_docs} docs after reserving first {a.skip_first_docs}; "
            f"only {len(eligible)} available"
        )
    rng = random.Random(int(a.seed))
    rng.shuffle(eligible)
    chosen = eligible[: int(a.num_docs)]

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    hidden_size = int(model.get_output_embeddings().weight.shape[1])

    cov_sum = torch.zeros((hidden_size, hidden_size), dtype=torch.float32, device=device)
    mean_sum = torch.zeros((hidden_size,), dtype=torch.float32, device=device)
    state_count = 0
    docs_used = 0
    duplicate_positions = 0
    skipped_short = 0

    for start in range(0, len(chosen), int(a.batch_size)):
        indices = chosen[start : start + int(a.batch_size)]
        texts = [str(ds[int(i)]["text"]) for i in indices]
        encoded = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=int(a.max_doc_tokens),
            return_tensors="pt",
        ).to(device)
        h = final_hidden(model, encoded).float()
        lengths = encoded["attention_mask"].sum(dim=1).tolist()
        blocks: List[torch.Tensor] = []
        for row, length_value in enumerate(lengths):
            length = int(length_value)
            positions = sample_positions(
                length,
                int(a.prompts_per_doc),
                int(a.min_context_tokens),
                device,
            )
            if positions.numel() == 0:
                skipped_short += 1
                continue
            duplicate_positions += int(positions.numel() - torch.unique(positions).numel())
            blocks.append(h[row].index_select(0, positions))
            docs_used += 1
        if not blocks:
            continue
        states = torch.cat(blocks, dim=0).float()
        cov_sum.add_(states.transpose(0, 1) @ states)
        mean_sum.add_(states.sum(dim=0))
        state_count += int(states.shape[0])
        if start == 0 or (start // int(a.batch_size) + 1) % 25 == 0:
            print(json.dumps({
                "docs_processed": min(start + int(a.batch_size), len(chosen)),
                "docs_used": docs_used,
                "hidden_states": state_count,
            }))

    if state_count == 0:
        raise RuntimeError("No Wikipedia hidden states collected")
    covariance = (cov_sum / float(state_count)).detach().cpu()
    mean = (mean_sum / float(state_count)).detach().cpu()
    average_variance = float(torch.trace(covariance).item() / hidden_size)
    if not torch.isfinite(covariance).all() or average_variance <= 0:
        raise RuntimeError("Invalid Wikipedia covariance")

    out = Path(a.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "kind": "lm_head_final_hidden_second_moment",
        "covariance": covariance,
        "mean": mean,
        "hidden_size": hidden_size,
        "hidden_state_count": int(state_count),
        "documents_requested": int(a.num_docs),
        "documents_used": int(docs_used),
        "prompts_per_document": int(a.prompts_per_doc),
        "document_indices": [int(x) for x in chosen],
        "skip_first_docs": int(a.skip_first_docs),
        "official_ppl_first_20_reserved": bool(int(a.skip_first_docs) >= 20),
        "max_doc_tokens": int(a.max_doc_tokens),
        "min_context_tokens": int(a.min_context_tokens),
        "duplicate_sample_positions": int(duplicate_positions),
        "short_documents_skipped": int(skipped_short),
        "average_second_moment_eigenvalue": average_variance,
        "model_path": str(Path(a.model_path).resolve()),
        "wikidata_dir": str(wikidata_dir),
        "seed": int(a.seed),
    }
    torch.save(payload, out)
    print(json.dumps({
        "covariance_cache": str(out),
        "documents_used": docs_used,
        "hidden_state_count": state_count,
        "hidden_size": hidden_size,
        "average_second_moment_eigenvalue": average_variance,
        "reserved_official_ppl_docs": int(a.skip_first_docs),
    }, indent=2))


if __name__ == "__main__":
    main()
