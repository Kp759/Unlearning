#!/usr/bin/env python3
"""Calibrated answer-blind semantic validator for MCF surrogate prompts.

The validator compares factual *answer slots*, not sentence surface form.  MCF
prompts are commonly incomplete completion stems, so fragments are explicitly
allowed.  A statement stem, passive rewrite, possessive rewrite, or direct
question may all be equivalent when the same unknown value fills the same
relation.

The validator receives only:
  * subject,
  * locked formatted direct prompt,
  * candidate surrogate.

It never receives target_true, target_new, official paraphrases, neighborhoods,
retain examples, or PPL text.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

import torch

import mcf_surrogate_semantic_validator as v1


VALIDATOR_PROTOCOL = "mcf_direct_only_relation_slot_validator_v2"


def structural_rejection_reason(direct_prompt: str, candidate: str) -> str | None:
    return v1.structural_rejection_reason(direct_prompt, candidate)


def _chat_text(tok, instruction: str) -> str:
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


def relation_slot_instruction(subject: str, direct_prompt: str, candidate: str) -> str:
    return f"""You are validating two knowledge-base prompts with an UNKNOWN answer slot.
Do NOT answer the fact. Judge only whether the same unknown value would fill/answer both prompts for the same relation.

SUBJECT: {subject}
ORIGINAL: {direct_prompt} [UNKNOWN]
CANDIDATE: {candidate} [UNKNOWN]

Important calibration rules:
- ORIGINAL and CANDIDATE may be incomplete sentence stems. That is NORMAL and must not be rejected merely for being a fragment.
- A question and a completion stem can be equivalent.
- Active/passive voice, possessive wording, synonyms, tense changes, and word-order changes can be equivalent.
- Do not require identical wording.
- Reject if the candidate asks a different property/relation, changes what kind of entity/value belongs in [UNKNOWN], adds factual/background restrictions not present in ORIGINAL, or is semantically incoherent.

Examples of EQUIVALENT form changes:
ORIGINAL: "Ada Lovelace works as [UNKNOWN]"
CANDIDATE: "Ada Lovelace's occupation is [UNKNOWN]"
=> same slot/relation.

ORIGINAL: "A car is produced by [UNKNOWN]"
CANDIDATE: "Who manufactures the car? [UNKNOWN]"
=> same manufacturer slot.

Examples of NOT equivalent:
ORIGINAL: "A car is produced by [UNKNOWN]"
CANDIDATE: "What vehicle is the car manufactured by? [UNKNOWN]"
=> changes the answer type/relation.

ORIGINAL: "Person X is originally from [UNKNOWN]"
CANDIDATE: "Person X is a professional athlete [UNKNOWN]"
=> changes relation and adds a factual claim.

Return ONLY one JSON object on one line with exactly these keys:
{{"same_slot_relation":true,"same_answer_type":true,"no_added_factual_constraints":true,"semantically_coherent":true,"verdict":"EQUIVALENT","reason":"short reason"}}
Set any failed criterion false and verdict "NOT_EQUIVALENT"."""


def adversarial_instruction(subject: str, direct_prompt: str, candidate: str) -> str:
    return f"""Act as a conservative adversarial checker for two factual prompts with an UNKNOWN answer slot.
Do NOT answer the fact.

SUBJECT: {subject}
ORIGINAL: {direct_prompt} [UNKNOWN]
CANDIDATE: {candidate} [UNKNOWN]

A completion fragment is allowed. A question/stem conversion is allowed.
Flag ONLY substantive semantic problems:
- relation_shift: [UNKNOWN] represents a different relation/property;
- answer_type_shift: [UNKNOWN] would need a different kind of entity/value;
- added_constraint_or_claim: candidate adds factual/background restrictions or assertions that change/narrow the fact;
- semantic_malformation: wording no longer clearly requests one corresponding factual value;
- weak_wrapper: candidate mostly copies ORIGINAL inside a generic instruction wrapper.

Return ONLY one JSON object on one line with exactly these keys:
{{"relation_shift":false,"answer_type_shift":false,"added_constraint_or_claim":false,"semantic_malformation":false,"weak_wrapper":false,"verdict":"SAFE","reason":"short reason"}}
If any substantive problem exists, set it true and use verdict "UNSAFE"."""


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


def parse_relation_slot(text: str) -> Dict[str, Any]:
    value = _extract_json(text)
    keys = (
        "same_slot_relation",
        "same_answer_type",
        "no_added_factual_constraints",
        "semantically_coherent",
    )
    accepted = (
        bool(value)
        and all(value.get(k) is True for k in keys)
        and str(value.get("verdict", "")).upper() == "EQUIVALENT"
    )
    return {
        "parsed": value,
        "accepted": bool(accepted),
        "parse_ok": value is not None,
        "raw": str(text),
    }


def parse_adversarial(text: str) -> Dict[str, Any]:
    value = _extract_json(text)
    keys = (
        "relation_shift",
        "answer_type_shift",
        "added_constraint_or_claim",
        "semantic_malformation",
        "weak_wrapper",
    )
    accepted = (
        bool(value)
        and all(value.get(k) is False for k in keys)
        and str(value.get("verdict", "")).upper() == "SAFE"
    )
    return {
        "parsed": value,
        "accepted": bool(accepted),
        "parse_ok": value is not None,
        "raw": str(text),
    }


@torch.no_grad()
def _generate_json_batch(model, tok, prompts: Sequence[str], *, max_new_tokens: int) -> List[str]:
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


@torch.no_grad()
def validate_candidates(
    model,
    tok,
    *,
    subject: str,
    direct_prompt: str,
    candidates: Sequence[str],
    batch_size: int = 8,
    max_new_tokens: int = 96,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = [None] * len(candidates)  # type: ignore[list-item]
    judge_indices: List[int] = []
    for i, candidate in enumerate(candidates):
        reason = structural_rejection_reason(direct_prompt, candidate)
        if reason is not None:
            results[i] = {
                "candidate": str(candidate),
                "accepted": False,
                "structural_rejection": reason,
                "relation_slot": None,
                "adversarial": None,
            }
        else:
            judge_indices.append(i)

    for start in range(0, len(judge_indices), int(batch_size)):
        idxs = judge_indices[start : start + int(batch_size)]
        primary_prompts = [
            relation_slot_instruction(subject, direct_prompt, str(candidates[i]))
            for i in idxs
        ]
        adversarial_prompts = [
            adversarial_instruction(subject, direct_prompt, str(candidates[i]))
            for i in idxs
        ]
        primary_raw = _generate_json_batch(
            model, tok, primary_prompts, max_new_tokens=max_new_tokens
        )
        adversarial_raw = _generate_json_batch(
            model, tok, adversarial_prompts, max_new_tokens=max_new_tokens
        )
        for i, p_text, a_text in zip(idxs, primary_raw, adversarial_raw):
            primary = parse_relation_slot(p_text)
            adversarial = parse_adversarial(a_text)
            accepted = bool(primary["accepted"] and adversarial["accepted"])
            results[i] = {
                "candidate": str(candidates[i]),
                "accepted": accepted,
                "structural_rejection": None,
                "relation_slot": primary,
                "adversarial": adversarial,
            }
    return results


def rejection_reason(record: Dict[str, Any]) -> str:
    if record.get("accepted"):
        return "accepted"
    if record.get("structural_rejection"):
        return str(record["structural_rejection"])
    primary = record.get("relation_slot") or {}
    adv = record.get("adversarial") or {}
    if not primary.get("parse_ok", False):
        return "relation_slot_parse_failure"
    if not primary.get("accepted", False):
        parsed = primary.get("parsed") or {}
        failed = [
            k for k in (
                "same_slot_relation",
                "same_answer_type",
                "no_added_factual_constraints",
                "semantically_coherent",
            ) if parsed.get(k) is not True
        ]
        return "relation_slot:" + (",".join(failed) if failed else "reject")
    if not adv.get("parse_ok", False):
        return "adversarial_parse_failure"
    if not adv.get("accepted", False):
        parsed = adv.get("parsed") or {}
        flagged = [
            k for k in (
                "relation_shift",
                "answer_type_shift",
                "added_constraint_or_claim",
                "semantic_malformation",
                "weak_wrapper",
            ) if parsed.get(k) is True
        ]
        return "adversarial:" + (",".join(flagged) if flagged else "unsafe")
    return "rejected"
