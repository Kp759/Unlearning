#!/usr/bin/env python3
"""Build final-hidden-state Wikipedia covariance for LM-head utility preservation.

The cache is designed for SURE MCF Stage 2, where only sparse LM-head rows are
edited.  The requested protocol is expressed as ``--num-docs`` times
``--prompts-per-doc`` predictor states (1000 x 100 = 100000 by default).

Local Wikipedia artifacts can contain fewer physical documents than requested.
Match the recent MQuAKE/SURE utility-cache policy instead of failing: cap the
document sample to the eligible local corpus, then spread the requested
predictor-state reservoir across multiple DISTINCT causal token positions in
those documents.  This preserves the requested state count when the capped
corpus has enough causal positions, while metadata explicitly marks reduced
document diversity as a pilot.

    C = E[h h^T]

uses final transformer hidden states h presented to the LM head.  States are
streamed into the second moment; the 100k states are never retained in memory.

Important leakage rule: by default the first 20 Wikipedia train documents are
reserved because mcf_zero_unlearn_official_eval.py uses exactly those texts for
its PPL evaluation. They are never included in this covariance cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import Dataset, DatasetDict, load_from_disk

import gagd_compare as gagd


CACHE_SCHEMA_VERSION = 2
SAMPLING_POLICY = "mquake_style_capped_documents_adaptive_distinct_predictor_positions_v1"


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
    if a.min_context_tokens >= a.max_doc_tokens:
        p.error("min-context-tokens must be smaller than max-doc-tokens")
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


def deterministic_predictor_positions(
    *,
    document_index: int,
    attended_length: int,
    seed: int,
    count: int,
    minimum_context_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    """Choose distinct reproducible causal predictor positions in one document.

    A predictor state must have a following token in the same document.  With
    ``minimum_context_tokens=8``, eligible hidden-state positions are therefore
    [7, attended_length-2].  This mirrors the MQuAKE utility-cache fix, while
    preserving the minimum-context convention used by this MCF builder.
    """
    lo = int(minimum_context_tokens) - 1
    hi = int(attended_length) - 2
    available = hi - lo + 1
    if available <= 0 or int(count) <= 0:
        return torch.empty((0,), dtype=torch.long, device=device)
    take = min(int(count), int(available))
    if take == available:
        values = list(range(lo, hi + 1))
    elif take == 1:
        digest = hashlib.sha256(
            f"wiki-cov:{int(seed)}:{int(document_index)}".encode("ascii")
        ).digest()
        offset = int.from_bytes(digest[:8], "big") % available
        values = [lo + offset]
    else:
        digest = hashlib.sha256(
            f"wiki-cov-positions:{int(seed)}:{int(document_index)}".encode("ascii")
        ).digest()
        local_seed = int.from_bytes(digest[:8], "big")
        values = sorted(random.Random(local_seed).sample(range(lo, hi + 1), take))
    return torch.tensor(values, dtype=torch.long, device=device)


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

    eligible = [
        int(i)
        for i in range(int(a.skip_first_docs), len(ds))
        if str(ds[int(i)]["text"]).strip()
    ]
    if not eligible:
        raise ValueError(
            f"No eligible Wikipedia documents remain after reserving first {a.skip_first_docs}"
        )
    rng = random.Random(int(a.seed))
    rng.shuffle(eligible)
    chosen = eligible[: min(int(a.num_docs), len(eligible))]
    document_cap_applied = len(chosen) < int(a.num_docs)
    if document_cap_applied:
        print(json.dumps({
            "warning": "requested Wikipedia document count exceeds eligible local corpus",
            "requested_documents": int(a.num_docs),
            "eligible_documents": len(eligible),
            "documents_selected": len(chosen),
            "policy": "cap documents to local corpus and spread requested predictor states across distinct token positions",
            "document_diversity_status": "pilot_capped_local_corpus",
        }))

    # Preserve the requested 1000 x 100 = 100000 state budget even when the
    # physical-document sample is capped.  This is the same core resolution as
    # the recent MQuAKE utility-cache fix: document diversity is capped, state
    # diversity is recovered from distinct causal predictor positions.
    requested_state_count = int(a.num_docs) * int(a.prompts_per_doc)
    state_slots_remaining = requested_state_count
    documents_remaining = len(chosen)

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
    skipped_short = 0
    available_predictor_positions = 0
    per_document_state_counts: List[int] = []

    old_padding_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "right"
    try:
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

            for row, (document_index, length_value) in enumerate(zip(indices, lengths)):
                length = int(length_value)
                available = max(0, length - int(a.min_context_tokens))
                available_predictor_positions += int(available)

                if state_slots_remaining <= 0:
                    quota = 0
                else:
                    quota = int(math.ceil(
                        state_slots_remaining / max(1, documents_remaining)
                    ))

                positions = deterministic_predictor_positions(
                    document_index=int(document_index),
                    attended_length=length,
                    seed=int(a.seed),
                    count=min(quota, state_slots_remaining),
                    minimum_context_tokens=int(a.min_context_tokens),
                    device=device,
                )
                documents_remaining -= 1

                count = int(positions.numel())
                per_document_state_counts.append(count)
                if count == 0:
                    if length <= int(a.min_context_tokens):
                        skipped_short += 1
                    continue

                blocks.append(h[row].index_select(0, positions))
                docs_used += 1
                state_slots_remaining -= count

            if blocks:
                states = torch.cat(blocks, dim=0).float()
                cov_sum.add_(states.transpose(0, 1) @ states)
                mean_sum.add_(states.sum(dim=0))
                state_count += int(states.shape[0])

            if start == 0 or (start // int(a.batch_size) + 1) % 25 == 0 or state_slots_remaining == 0:
                print(json.dumps({
                    "docs_processed": min(start + int(a.batch_size), len(chosen)),
                    "docs_selected": len(chosen),
                    "docs_used": docs_used,
                    "hidden_states": state_count,
                    "requested_hidden_states": requested_state_count,
                    "hidden_states_remaining": max(0, state_slots_remaining),
                }))
    finally:
        tok.padding_side = old_padding_side

    if state_count == 0:
        raise RuntimeError("No Wikipedia hidden states collected")

    state_target_filled = state_count >= requested_state_count
    if not state_target_filled:
        print(json.dumps({
            "warning": "local capped corpus lacks enough eligible causal positions to fill requested state reservoir",
            "requested_hidden_states": requested_state_count,
            "actual_hidden_states": state_count,
            "available_predictor_positions_after_truncation": available_predictor_positions,
            "status": "pilot_state_count_shortfall",
        }))

    covariance = (cov_sum / float(state_count)).detach().cpu()
    mean = (mean_sum / float(state_count)).detach().cpu()
    covariance = 0.5 * (covariance + covariance.transpose(0, 1))
    average_variance = float(torch.trace(covariance).item() / hidden_size)
    if not torch.isfinite(covariance).all() or average_variance <= 0:
        raise RuntimeError("Invalid Wikipedia covariance")

    out = Path(a.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    diversity_status = (
        "pilot_capped_local_corpus" if document_cap_applied else "requested_document_count_met"
    )
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": "lm_head_final_hidden_second_moment",
        "sampling_policy": SAMPLING_POLICY,
        "covariance": covariance,
        "mean": mean,
        "hidden_size": hidden_size,
        "hidden_state_count": int(state_count),
        "requested_hidden_state_count": int(requested_state_count),
        "hidden_state_target_filled": bool(state_target_filled),
        "documents_requested": int(a.num_docs),
        "eligible_documents": int(len(eligible)),
        "documents_selected": int(len(chosen)),
        "documents_used": int(docs_used),
        "document_sample_cap_applied": bool(document_cap_applied),
        "document_diversity_status": diversity_status,
        "sample_size_cap_policy": "min(requested_documents, eligible_local_documents)",
        "prompts_per_document_nominal": int(a.prompts_per_doc),
        "actual_mean_states_per_used_document": (
            float(state_count / docs_used) if docs_used else 0.0
        ),
        "per_document_state_count_min": int(min(per_document_state_counts)) if per_document_state_counts else 0,
        "per_document_state_count_max": int(max(per_document_state_counts)) if per_document_state_counts else 0,
        "available_predictor_positions_after_truncation": int(available_predictor_positions),
        "document_indices": [int(x) for x in chosen],
        "skip_first_docs": int(a.skip_first_docs),
        "official_ppl_first_20_reserved": bool(int(a.skip_first_docs) >= 20),
        "max_doc_tokens": int(a.max_doc_tokens),
        "min_context_tokens": int(a.min_context_tokens),
        "duplicate_sample_positions": 0,
        "short_documents_skipped": int(skipped_short),
        "average_second_moment_eigenvalue": average_variance,
        "model_path": str(Path(a.model_path).resolve()),
        "wikidata_dir": str(wikidata_dir),
        "seed": int(a.seed),
    }
    torch.save(payload, out)
    print(json.dumps({
        "covariance_cache": str(out),
        "documents_requested": int(a.num_docs),
        "documents_selected": len(chosen),
        "documents_used": docs_used,
        "document_diversity_status": diversity_status,
        "hidden_state_count": state_count,
        "requested_hidden_state_count": requested_state_count,
        "hidden_state_target_filled": bool(state_target_filled),
        "hidden_size": hidden_size,
        "average_second_moment_eigenvalue": average_variance,
        "reserved_official_ppl_docs": int(a.skip_first_docs),
        "sampling_policy": SAMPLING_POLICY,
    }, indent=2))


if __name__ == "__main__":
    main()
