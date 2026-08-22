#!/usr/bin/env python3
"""Boolean-consensus semantic validator for MCF surrogate prompts.

This is a calibration-only successor to mcf_surrogate_semantic_validator_v2.
The v2 judge sometimes emits internally inconsistent JSON such as all required
criterion booleans being correct while the free-form ``verdict`` string says
NOT_EQUIVALENT/UNSAFE.  For a small local 3B judge, the structured criterion
fields are the more reliable signal.

v3 therefore makes the explicit booleans authoritative:
  * primary pass iff all relation-slot criteria are True;
  * adversarial pass iff all problem flags are False.

The verdict field is retained verbatim as an audit diagnostic but does not veto
an otherwise internally consistent structured judgment.  Data access remains
answer-blind: subject + direct prompt + candidate only.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

import mcf_surrogate_semantic_validator_v2 as v2


VALIDATOR_PROTOCOL = "mcf_direct_only_relation_slot_boolean_consensus_v3"

structural_rejection_reason = v2.structural_rejection_reason
relation_slot_instruction = v2.relation_slot_instruction
adversarial_instruction = v2.adversarial_instruction
_generate_json_batch = v2._generate_json_batch
_extract_json = v2._extract_json


def parse_relation_slot(text: str) -> Dict[str, Any]:
    value = _extract_json(text)
    keys = (
        "same_slot_relation",
        "same_answer_type",
        "no_added_factual_constraints",
        "semantically_coherent",
    )
    criteria_ok = bool(value) and all(value.get(k) is True for k in keys)
    verdict = "" if not value else str(value.get("verdict", "")).upper()
    return {
        "parsed": value,
        "accepted": bool(criteria_ok),
        "parse_ok": value is not None,
        "criteria_ok": bool(criteria_ok),
        "verdict": verdict,
        "verdict_consistent": bool(verdict == "EQUIVALENT") if value else False,
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
    criteria_ok = bool(value) and all(value.get(k) is False for k in keys)
    verdict = "" if not value else str(value.get("verdict", "")).upper()
    return {
        "parsed": value,
        "accepted": bool(criteria_ok),
        "parse_ok": value is not None,
        "criteria_ok": bool(criteria_ok),
        "verdict": verdict,
        "verdict_consistent": bool(verdict == "SAFE") if value else False,
        "raw": str(text),
    }


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
            results[i] = {
                "candidate": str(candidates[i]),
                "accepted": bool(primary["accepted"] and adversarial["accepted"]),
                "structural_rejection": None,
                "relation_slot": primary,
                "adversarial": adversarial,
            }
    return results


def rejection_reason(record: Dict[str, Any]) -> str:
    if record.get("accepted"):
        primary = record.get("relation_slot") or {}
        adv = record.get("adversarial") or {}
        if not primary.get("verdict_consistent", True) or not adv.get("verdict_consistent", True):
            return "accepted_with_verdict_inconsistency"
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
        return "relation_slot:" + (",".join(failed) if failed else "criteria_reject")
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
        return "adversarial:" + (",".join(flagged) if flagged else "criteria_unsafe")
    return "rejected"
