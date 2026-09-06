#!/usr/bin/env python3
"""Reference-bank verifier entrypoint for build_mcf_relation_views_v2.

This wrapper keeps the V2 builder's leakage contract and generation logic, but
uses a richer training-only relation reference bank instead of a single vague
canonical prompt.

Reference data:
* sanitized ``training_visible_forget_direct.json``;
* the existing leakage-safe five-view training corpus supplied through
  ``VIEW_CORPUS`` (or ``MCF_V13_VIEW_CORPUS``).

No target_true/target_new value, official paraphrase, neighborhood, retain, or
official generation probe is read by this verifier.

A generated candidate is accepted only when:
1. it is an open-ended relation query (not a yes/no query);
2. it does not inject an obvious new named object/entity;
3. its frozen Base representation is closest to the target relation reference
   bank rather than a competing relation; and
4. if the semantic top-1 gap is small, it contains a lexical relation cue that
   is unique to the target relation in the training-only reference bank.

This is a corpus-quality filter, not the final router operating threshold.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import json
import os
import re
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


_STOP = {
    "entity", "the", "a", "an", "is", "was", "were", "are", "be", "been",
    "of", "to", "in", "on", "at", "for", "with", "as", "by", "from", "that",
    "which", "what", "who", "where", "when", "how", "does", "do", "did",
    "its", "their", "this", "these", "those", "it", "and", "or", "please",
    "identify", "name", "state", "specify", "give", "tell", "find", "request",
}


def _stem(word: str) -> str:
    w = word.casefold()
    for suffix in ("ization", "isation", "ation", "ition", "ment", "ingly", "edly", "ing", "ed", "es", "s"):
        if len(w) > len(suffix) + 3 and w.endswith(suffix):
            w = w[: -len(suffix)]
            break
    return w


def _cue_stems(text: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z'-]+", text):
        stem = _stem(token)
        if stem not in _STOP and len(stem) >= 3:
            out.add(stem)
    return out


def _load_relation_bank() -> tuple[dict[str, dict[str, str]], dict[str, list[str]], dict[str, set[str]]]:
    source = Path(_arg_value("--forget-direct")).resolve()
    rows = json.loads(source.read_text(encoding="utf-8"))

    canonical_meta: dict[str, dict[str, str]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    case_to_relation: dict[int, str] = {}

    for row in rows:
        rr = row["requested_rewrite"]
        cid = int(row["case_id"])
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
        case_to_relation[cid] = relation_id
        grouped[relation_id].append(relation_text)

    ref_env = os.environ.get("VIEW_CORPUS") or os.environ.get("MCF_V13_VIEW_CORPUS")
    if not ref_env:
        raise RuntimeError(
            "VIEW_CORPUS is required for the V2 reference-bank verifier; "
            "set it to the leakage-safe five-view training corpus"
        )
    ref_path = Path(ref_env).resolve()
    payload = json.loads(ref_path.read_text(encoding="utf-8"))
    leakage = payload.get("leakage_contract", {})
    forbidden_true = (
        "official_paraphrase_prompts_read",
        "official_neighborhood_prompts_read",
        "official_generation_prompts_read",
        "official_retain_records_read",
        "generator_received_target_true",
        "generator_received_target_new",
    )
    if any(leakage.get(k) is not False for k in forbidden_true):
        raise RuntimeError("reference view corpus fails leakage contract")

    for case in payload.get("cases", []):
        cid = int(case["case_id"])
        rid = case_to_relation.get(cid)
        if rid is None:
            continue
        for view in case.get("views", []):
            template = str(view.get("template", ""))
            if template.count("{}") != 1:
                continue
            relation_text = builder.normalize_space(template.format("ENTITY"))
            cues = _cue_stems(relation_text)
            if not cues:
                continue
            grouped[rid].append(relation_text)

    deduped: dict[str, list[str]] = {}
    for rid, texts in grouped.items():
        seen: set[str] = set()
        keep: list[str] = []
        for text in texts:
            key = builder.normalize_space(text).casefold()
            if key not in seen:
                seen.add(key)
                keep.append(builder.normalize_space(text))
        deduped[rid] = keep

    all_relation_stems: dict[str, set[str]] = {
        rid: set().union(*(_cue_stems(t) for t in texts)) if texts else set()
        for rid, texts in deduped.items()
    }
    unique_stems: dict[str, set[str]] = {}
    for rid, stems in all_relation_stems.items():
        others: set[str] = set()
        for rid2, stems2 in all_relation_stems.items():
            if rid2 != rid:
                others |= stems2
        unique_stems[rid] = stems - others

    print(
        json.dumps(
            {
                "relation_v2_reference_bank": str(ref_path),
                "relations": len(deduped),
                "reference_texts": sum(len(v) for v in deduped.values()),
                "source": "sanitized direct + leakage-safe legacy training views",
            },
            indent=2,
        ),
        flush=True,
    )
    return canonical_meta, deduped, unique_stems


_CANONICAL_META, _RELATION_TEXTS, _UNIQUE_STEMS = _load_relation_bank()
_REF_CACHE: dict[str, torch.Tensor] | None = None
_LOW_MARGIN_PRINTS = 0
_LOW_MARGIN_LIMIT = 80


@torch.no_grad()
def _encode_texts(model: Any, tokenizer: Any, texts: list[str], device: torch.device) -> torch.Tensor:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise RuntimeError("reference-bank verifier requires model.model backbone")
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
def _reference_embeddings(model: Any, tokenizer: Any, device: torch.device) -> dict[str, torch.Tensor]:
    global _REF_CACHE
    if _REF_CACHE is not None:
        return _REF_CACHE
    rel_ids = sorted(_RELATION_TEXTS)
    flat: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for rid in rel_ids:
        start = len(flat)
        flat.extend(_RELATION_TEXTS[rid])
        spans[rid] = (start, len(flat))
    emb = _encode_texts(model, tokenizer, flat, device).cpu()
    refs: dict[str, torch.Tensor] = {}
    for rid in rel_ids:
        a, b = spans[rid]
        refs[rid] = emb[a:b]
    _REF_CACHE = refs
    return refs


_ORIGINAL_TOO_VAGUE = builder.too_vague
_YES_NO = re.compile(r"^(?:does|do|did|is|are|was|were|has|have|had|can|could|will|would|should|may|might)\b", re.I)


def _has_new_named_object(candidate: str, subject: str) -> bool:
    text = candidate.replace(subject, "ENTITY", 1)
    if re.search(r"\b[A-Z]{2,}\b", text.replace("ENTITY", "")):
        return True
    proper = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    return any(x not in {"General Information"} for x in proper)


def _fixed_too_vague(candidate: str, subject: str) -> bool:
    if _ORIGINAL_TOO_VAGUE(candidate, subject):
        return True
    stripped = candidate.strip().lstrip("\"'` ")
    if _YES_NO.match(stripped):
        return True
    if _has_new_named_object(candidate, subject):
        return True
    return False


@torch.no_grad()
def _reference_bank_equivalence_margin(
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
    refs = _reference_embeddings(model, tokenizer, device)
    c = _encode_texts(model, tokenizer, [candidate_relation], device)[0].cpu()

    target = float((refs[relation_id] @ c).max().item())
    competitor_id = "<none>"
    competitor = -1.0
    for rid, matrix in refs.items():
        if rid == relation_id:
            continue
        value = float((matrix @ c).max().item())
        if value > competitor:
            competitor = value
            competitor_id = rid
    gap = target - competitor

    candidate_stems = _cue_stems(candidate_relation)
    unique_overlap = sorted(candidate_stems & _UNIQUE_STEMS.get(relation_id, set()))

    if target < 0.70 or gap < 0.0:
        score = -1.0
    elif gap >= 0.02:
        score = min(gap, target - 0.70)
    elif unique_overlap:
        score = min(0.001, target - 0.70)
    else:
        score = -1.0

    threshold = float(os.environ.get("RELATION_EQ_MARGIN", "0.0"))
    if score < threshold and _LOW_MARGIN_PRINTS < _LOW_MARGIN_LIMIT:
        print(
            "[relation-v2 reference reject] "
            f"score={score:.4f} threshold={threshold:.4f} "
            f"target_sim={target:.4f} competitor_sim={competitor:.4f} gap={gap:.4f} "
            f"target_relation={relation_id!r} competitor_relation={competitor_id!r} "
            f"unique_cues={unique_overlap!r} candidate={candidate!r}",
            flush=True,
        )
        _LOW_MARGIN_PRINTS += 1
    return float(score)


builder.encode_generation_prompt = _fixed_encode_generation_prompt
builder.too_vague = _fixed_too_vague
builder.equivalence_margin = _reference_bank_equivalence_margin


if __name__ == "__main__":
    builder.main()
