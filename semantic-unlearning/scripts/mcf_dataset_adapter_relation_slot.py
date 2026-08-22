#!/usr/bin/env python3
"""Answer-blind relation-slot gates for the MCF robust-prompt dataset adapter.

This module belongs to the *dataset adapter*, not the SURE optimizer.  It sees
only the locked subject/direct prompt and generated candidate text.  It never
receives target_true, target_new, official paraphrases, neighborhoods, retain
examples, or PPL text.

Pipeline implemented here:
  1. profile the answer slot in the locked direct prompt;
  2. classify a candidate against that relation slot;
  3. apply a deterministic answer-type compatibility gate.

A later semantic-equivalence judge is intentionally separate.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Sequence

import torch


ADAPTER_PROTOCOL = "mcf_answer_blind_relation_slot_adapter_v1"
ANSWER_TYPES = (
    "person",
    "organization",
    "location",
    "language",
    "occupation_or_role",
    "nationality_or_demonym",
    "date_or_year",
    "number_or_quantity",
    "product_or_artifact",
    "creative_work",
    "event",
    "other_entity",
    "other_value",
    "ambiguous",
)


def _chat_text(tok, instruction: str) -> str:
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


def _extract_json(text: str) -> Dict[str, Any] | None:
    text = str(text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start : end + 1])
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _clean_types(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    allowed = set(ANSWER_TYPES)
    out: List[str] = []
    for value in values:
        key = str(value).strip().lower()
        if key in allowed and key not in out:
            out.append(key)
    return out


def direct_profile_instruction(subject: str, direct_prompt: str) -> str:
    allowed = ", ".join(ANSWER_TYPES)
    return f"""You are profiling the UNKNOWN answer slot of a factual knowledge-base prompt.
Do NOT answer the fact and do not guess the missing value.

SUBJECT: {subject}
DIRECT PROMPT: {direct_prompt} [UNKNOWN]

Infer only the relation expressed by the words already present in DIRECT PROMPT.
Some prompts are incomplete sentence stems; that is normal.

Return whether this direct prompt is specific enough to safely create semantic
paraphrases.  If the relation is underspecified (for example, \"Person X, the\")
or the missing slot could represent unrelated relations, set augmentable=false.

answer_types must contain one or more values from this closed list only:
{allowed}

Examples:
- \"X works as [UNKNOWN]\" -> occupation_or_role
- \"X is produced by [UNKNOWN]\" -> organization or person, depending on wording;
  list every genuinely plausible type without guessing the fact
- \"X was founded in [UNKNOWN]\" can be date_or_year and/or location when the
  prompt alone does not disambiguate; keep both if both are linguistically possible
- \"X, the [UNKNOWN]\" -> augmentable=false because the relation is underspecified

Return ONLY one JSON object with exactly these keys:
{{"relation_label":"short snake_case relation label","relation_description":"short answer-blind description of what [UNKNOWN] denotes","answer_types":["one_or_more_allowed_types"],"augmentable":true,"ambiguity":"low|medium|high","reason":"short reason"}}"""


def candidate_profile_instruction(
    subject: str,
    direct_prompt: str,
    direct_profile: Mapping[str, Any],
    candidate: str,
) -> str:
    allowed = ", ".join(ANSWER_TYPES)
    relation = str(direct_profile.get("relation_label", ""))
    description = str(direct_profile.get("relation_description", ""))
    direct_types = _clean_types(direct_profile.get("answer_types"))
    return f"""You are checking a generated paraphrase against a locked factual answer slot.
Do NOT answer the fact and do not guess the missing value.

SUBJECT: {subject}
ORIGINAL: {direct_prompt} [UNKNOWN]
ORIGINAL RELATION LABEL: {relation}
ORIGINAL SLOT DESCRIPTION: {description}
ORIGINAL POSSIBLE ANSWER TYPES: {direct_types}
CANDIDATE: {candidate} [UNKNOWN]

Classify the CANDIDATE's own missing slot and compare its relation to ORIGINAL.
A question/completion-stem conversion, active/passive voice, tense change, or
synonym can preserve the relation.  Reject semantic narrowing or broadening.
Examples of narrowing include adding \"Japanese\" to \"automaker\" when ORIGINAL
only asks for a producer.  Reject relation changes such as founded-in ->
headquartered-in, formed-in -> created-by, or producer -> vehicle.

candidate_answer_types must use only this closed list:
{allowed}

Return ONLY one JSON object with exactly these keys:
{{"candidate_relation_label":"short snake_case label","candidate_answer_types":["allowed_type"],"same_relation":true,"adds_constraint_or_claim":false,"semantically_coherent":true,"reason":"short reason"}}"""


@torch.no_grad()
def _generate_json_batch(
    model,
    tok,
    prompts: Sequence[str],
    *,
    max_new_tokens: int,
) -> List[str]:
    if not prompts:
        return []
    device = next(model.parameters()).device
    old_padding_side = getattr(tok, "padding_side", "right")
    tok.padding_side = "left"
    try:
        texts = [_chat_text(tok, p) for p in prompts]
        enc = tok(list(texts), padding=True, return_tensors="pt").to(device)
        seq = model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        prefix = int(enc["input_ids"].shape[1])
        return [tok.decode(row[prefix:], skip_special_tokens=True).strip() for row in seq]
    finally:
        tok.padding_side = old_padding_side


def parse_direct_profile(text: str) -> Dict[str, Any]:
    value = _extract_json(text)
    types = _clean_types((value or {}).get("answer_types"))
    augmentable = bool(value) and value.get("augmentable") is True
    ambiguity = str((value or {}).get("ambiguity", "")).strip().lower()
    relation_label = str((value or {}).get("relation_label", "")).strip()
    relation_description = str((value or {}).get("relation_description", "")).strip()
    valid = (
        value is not None
        and bool(relation_label)
        and bool(relation_description)
        and bool(types)
        and ambiguity in {"low", "medium", "high"}
        and isinstance(value.get("augmentable"), bool)
    )
    # High ambiguity is allowed only when the model still explicitly declares
    # the relation augmentable.  A fully ambiguous slot type is never augmented.
    safe_to_augment = bool(valid and augmentable and "ambiguous" not in types)
    return {
        "parsed": value,
        "parse_ok": value is not None,
        "valid": bool(valid),
        "safe_to_augment": safe_to_augment,
        "relation_label": relation_label,
        "relation_description": relation_description,
        "answer_types": types,
        "ambiguity": ambiguity,
        "raw": str(text),
    }


def parse_candidate_profile(text: str) -> Dict[str, Any]:
    value = _extract_json(text)
    types = _clean_types((value or {}).get("candidate_answer_types"))
    relation_label = str((value or {}).get("candidate_relation_label", "")).strip()
    valid = (
        value is not None
        and bool(relation_label)
        and bool(types)
        and isinstance(value.get("same_relation"), bool)
        and isinstance(value.get("adds_constraint_or_claim"), bool)
        and isinstance(value.get("semantically_coherent"), bool)
    )
    relation_pass = bool(
        valid
        and value.get("same_relation") is True
        and value.get("adds_constraint_or_claim") is False
        and value.get("semantically_coherent") is True
    )
    return {
        "parsed": value,
        "parse_ok": value is not None,
        "valid": bool(valid),
        "relation_pass": relation_pass,
        "candidate_relation_label": relation_label,
        "candidate_answer_types": types,
        "raw": str(text),
    }


def answer_type_compatible(
    direct_types: Sequence[str],
    candidate_types: Sequence[str],
) -> Dict[str, Any]:
    direct = {str(x) for x in direct_types if str(x) in ANSWER_TYPES and str(x) != "ambiguous"}
    candidate = {str(x) for x in candidate_types if str(x) in ANSWER_TYPES and str(x) != "ambiguous"}
    overlap = sorted(direct & candidate)
    return {
        "compatible": bool(direct and candidate and overlap),
        "direct_types": sorted(direct),
        "candidate_types": sorted(candidate),
        "overlap": overlap,
    }


@torch.no_grad()
def profile_direct(
    model,
    tok,
    *,
    subject: str,
    direct_prompt: str,
    max_new_tokens: int = 128,
) -> Dict[str, Any]:
    raw = _generate_json_batch(
        model,
        tok,
        [direct_profile_instruction(subject, direct_prompt)],
        max_new_tokens=max_new_tokens,
    )[0]
    return parse_direct_profile(raw)


@torch.no_grad()
def profile_candidates(
    model,
    tok,
    *,
    subject: str,
    direct_prompt: str,
    direct_profile: Mapping[str, Any],
    candidates: Sequence[str],
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for start in range(0, len(candidates), int(batch_size)):
        batch = list(candidates[start : start + int(batch_size)])
        prompts = [
            candidate_profile_instruction(
                subject, direct_prompt, direct_profile, candidate
            )
            for candidate in batch
        ]
        raw = _generate_json_batch(
            model, tok, prompts, max_new_tokens=max_new_tokens
        )
        for candidate, text in zip(batch, raw):
            parsed = parse_candidate_profile(text)
            type_gate = answer_type_compatible(
                direct_profile.get("answer_types", []),
                parsed.get("candidate_answer_types", []),
            )
            results.append({
                "candidate": str(candidate),
                "candidate_profile": parsed,
                "relation_pass": bool(parsed["relation_pass"]),
                "answer_type_gate": type_gate,
                "answer_type_pass": bool(type_gate["compatible"]),
                "presemantic_pass": bool(
                    parsed["relation_pass"] and type_gate["compatible"]
                ),
            })
    return results
