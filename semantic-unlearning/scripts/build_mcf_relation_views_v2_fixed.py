#!/usr/bin/env python3
"""Contrastive-verifier entrypoint for build_mcf_relation_views_v2.

This keeps the V2 builder's leakage contract and generation logic, but replaces
the unreliable instruction-model SAME-vs-DIFFERENT log-prob judge with a
frozen-representation contrastive relation check.

For each generated candidate, the literal subject is replaced by the neutral
placeholder ENTITY.  The candidate embedding is compared against relation
prototypes built only from the sanitized direct forget prompts, grouped by
relation_id.  A candidate is accepted only if its target relation prototype is
more similar than every competing relation prototype by a positive margin and
is not globally dissimilar to its own relation.

No target_true/target_new value, official paraphrase, neighborhood, retain, or
generation probe is read by this verifier.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json
import os
import sys

import torch
import torch.nn.functional as F

import build_mcf_relation_views_v2 as builder


def _extract_input_ids(value: Any, *, label: str) -> torch.Tensor:
    if torch.is_tensor(value):
        ids = value
    elif isinstance(value, Mapping):
        if "input_ids" not in value:
            raise RuntimeError(f"{label} encoding mapping has no input_ids")
        ids = value["input_ids"]
    elif hasattr(value, "input_ids"):
        ids = value.input_ids
    else:
        raise RuntimeError(f"unsupported {label} encoding type: {type(value).__name__}")
    if not torch.is_tensor(ids) or ids.ndim != 2:
        raise RuntimeError(
            f"{label} input_ids must be rank-2 tensor, got "
            f"{type(ids).__name__} shape={getattr(ids, 'shape', None)}"
        )
    return ids


def _fixed_encode_generation_prompt(tokenizer: Any, text: str, device: torch.device) -> torch.Tensor:
    value: Any = None
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            value = tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        except Exception:
            value = None
    if value is None:
        value = tokenizer(text, return_tensors="pt")
    return _extract_input_ids(value, label="generation").to(device)


def _arg_value(name: str) -> str:
    for i, value in enumerate(sys.argv[:-1]):
        if value == name:
            return sys.argv[i + 1]
    raise RuntimeError(f"required argument {name} not found in argv")


def _load_relation_bank() -> tuple[dict[str, dict[str, str]], dict[str, list[str]]]:
    source = Path(_arg_value("--forget-direct")).resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))
    canonical_meta: dict[str, dict[str, str]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        rr = row["requested_rewrite"]
        subject = str(rr["subject"])
        relation_id = str(rr["relation_id"])
        template = str(rr["prompt"])
        canonical = builder.normalize_space(template.format(subject))
        relation_text = builder.normalize_space(template.format("ENTITY"))
        canonical_meta[canonical] = {
            "subject": subject,
            "relation_id": relation_id,
            "relation_text": relation_text,
        }
        grouped[relation_id].append(relation_text)
    return canonical_meta, dict(grouped)


_CANONICAL_META, _RELATION_TEXTS = _load_relation_bank()
_PROTO_CACHE: dict[str, torch.Tensor] | None = None
_LOW_MARGIN_PRINTS = 0
_LOW_MARGIN_LIMIT = 60


@torch.no_grad()
def _encode_texts(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    device: torch.device,
) -> torch.Tensor:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("contrastive verifier requires model.model backbone")
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=192,
        return_tensors="pt",
    ).to(device)
    out = backbone(**enc, use_cache=False, return_dict=True)
    hidden = out.last_hidden_state.float()
    mask = enc["attention_mask"].to(hidden.dtype).unsqueeze(-1)
    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return F.normalize(pooled, p=2, dim=-1)


@torch.no_grad()
def _relation_prototypes(model: Any, tokenizer: Any, device: torch.device) -> dict[str, torch.Tensor]:
    global _PROTO_CACHE
    if _PROTO_CACHE is not None:
        return _PROTO_CACHE
    rel_ids = sorted(_RELATION_TEXTS)
    flat: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for rid in rel_ids:
        start = len(flat)
        flat.extend(_RELATION_TEXTS[rid])
        spans[rid] = (start, len(flat))
    emb = _encode_texts(model, tokenizer, flat, device)
    protos: dict[str, torch.Tensor] = {}
    for rid in rel_ids:
        a, b = spans[rid]
        protos[rid] = F.normalize(emb[a:b].mean(dim=0), p=2, dim=0).cpu()
    _PROTO_CACHE = protos
    return protos


@torch.no_grad()
def _contrastive_equivalence_margin(
    model: Any,
    tokenizer: Any,
    canonical: str,
    candidate: str,
    device: torch.device,
) -> float:
    global _LOW_MARGIN_PRINTS
    key = builder.normalize_space(canonical)
    meta = _CANONICAL_META.get(key)
    if meta is None:
        raise RuntimeError(f"canonical prompt missing from sanitized relation bank: {canonical!r}")

    subject = meta["subject"]
    relation_id = meta["relation_id"]
    if candidate.count(subject) != 1:
        return -1.0
    candidate_relation = builder.normalize_space(candidate.replace(subject, "ENTITY", 1))

    protos = _relation_prototypes(model, tokenizer, device)
    c = _encode_texts(model, tokenizer, [candidate_relation], device)[0].cpu()
    target = float(torch.dot(c, protos[relation_id]).item())
    competitors = [(rid, float(torch.dot(c, p).item())) for rid, p in protos.items() if rid != relation_id]
    if competitors:
        competitor_id, competitor = max(competitors, key=lambda x: x[1])
    else:
        competitor_id, competitor = "<none>", -1.0

    gap = target - competitor
    # The returned scalar is intentionally the minimum of two requirements:
    # (1) target beats every competing relation; (2) target itself is not weak.
    # With launcher default threshold 0.02, this means gap >= .02 and
    # target cosine >= .52 (because target - .50 >= .02).
    score = min(gap, target - 0.50)

    threshold = float(os.environ.get("RELATION_EQ_MARGIN", "0.02"))
    if score < threshold and _LOW_MARGIN_PRINTS < _LOW_MARGIN_LIMIT:
        print(
            "[relation-v2 contrastive reject] "
            f"score={score:.4f} threshold={threshold:.4f} "
            f"target_sim={target:.4f} competitor_sim={competitor:.4f} "
            f"target_relation={relation_id!r} competitor_relation={competitor_id!r} "
            f"candidate={candidate!r}",
            flush=True,
        )
        _LOW_MARGIN_PRINTS += 1
    return float(score)


builder.encode_generation_prompt = _fixed_encode_generation_prompt
builder.equivalence_margin = _contrastive_equivalence_margin


if __name__ == "__main__":
    builder.main()
