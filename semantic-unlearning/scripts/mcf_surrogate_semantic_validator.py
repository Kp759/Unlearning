#!/usr/bin/env python3
"""Strict held-out-safe semantic validation for MCF surrogate prompts.

The validator receives ONLY:
  * the subject,
  * the formatted locked direct prompt,
  * one candidate surrogate.

It never receives target_true, target_new, official paraphrases, neighborhoods,
retain examples, or PPL text.  Acceptance requires consensus between two local
LLM checks:

1. equivalence judge: same missing relation/object and answer type, no added
   factual claims, grammatical, and completion-compatible;
2. adversarial critic: no relation shift, answer-type shift, added factual
   claim, malformed wording, or generic wrapper.

Obvious generic wrappers are rejected deterministically before any LLM call.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Sequence

import torch


VALIDATOR_PROTOCOL = "mcf_direct_only_semantic_equivalence_validator_v1"

_GENERIC_WRAPPER_MARKERS = (
    "complete this factual statement:",
    "complete the following fact:",
    "provide the missing factual completion:",
    "supply the missing continuation:",
    "finish this factual relation:",
    "give the missing object for this fact:",
    "answer with the missing completion:",
    "fill in the missing factual continuation:",
    "what completes this factual statement?",
    "which completion makes this fact complete?",
    "provide only the missing factual continuation:",
    "complete the relation expressed here:",
    "finish the following factual prompt:",
    "supply the factual continuation for:",
    "give the completion requested by this fact:",
    "complete this knowledge statement:",
    "provide the object that completes this relation:",
    "fill the missing end of this factual statement:",
    "return only the factual completion for:",
    "complete the factual relation below:",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def structural_rejection_reason(direct_prompt: str, candidate: str) -> str | None:
    """Reject obvious wrappers/duplicates before semantic judging."""
    d = _norm(direct_prompt)
    c = _norm(candidate)
    if not c:
        return "empty"
    if c == d:
        return "duplicate_direct"
    if any(c.startswith(marker) for marker in _GENERIC_WRAPPER_MARKERS):
        return "generic_wrapper"
    # If the entire direct prompt appears unchanged inside a longer candidate,
    # the candidate is almost always an instruction wrapper rather than a true
    # semantic paraphrase.  We intentionally reject these weak augmentations.
    if d and d in c and len(c) > len(d) + 8:
        return "contains_direct_prompt_verbatim"
    return None


def _chat_text(tok, instruction: str) -> str:
    if hasattr(tok, "apply_chat_template") and getattr(tok, "chat_template", None):
        return tok.apply_chat_template(
            [{"role": "user", "content": instruction}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return instruction


def equivalence_instruction(subject: str, direct_prompt: str, candidate: str) -> str:
    """Prompt for the affirmative semantic-equivalence judge.

    Deliberately has no answer arguments; callers cannot accidentally expose
    target_true/target_new through this API.
    """
    return f"""You are a strict semantic validator for factual-completion prompts.

Decide whether CANDIDATE asks for EXACTLY the same missing factual relation/object as ORIGINAL.
Do not answer either prompt and do not infer or state the missing fact.

SUBJECT: {subject}
ORIGINAL: {direct_prompt}
CANDIDATE: {candidate}

A valid paraphrase must satisfy ALL of these:
1. same_relation: it requests the same relation/object, not a related attribute;
2. same_answer_type: the same kind of answer would fill both prompts;
3. no_added_factual_claims: it does not inject biography, location, fictionality, dates, occupations, or other new factual context;
4. grammatical: it is coherent English rather than a malformed question/fragment;
5. completion_compatible: one missing answer can naturally complete/answer it in the same role as ORIGINAL.

Be conservative. If wording could change what answer is required, reject it.
Return ONLY one JSON object on one line, exactly with these keys:
{{"same_relation":true,"same_answer_type":true,"no_added_factual_claims":true,"grammatical":true,"completion_compatible":true,"verdict":"ACCEPT","reason":"short reason"}}
Use false and verdict "REJECT" for any failed criterion."""


def critic_instruction(subject: str, direct_prompt: str, candidate: str) -> str:
    """Prompt for an adversarial semantic critic, also answer-blind."""
    return f"""Act as an adversarial critic of a proposed paraphrase for a factual-completion prompt.

Do NOT answer the fact. Search for any reason the candidate might require a different missing answer than the original.

SUBJECT: {subject}
ORIGINAL: {direct_prompt}
CANDIDATE: {candidate}

Flag each problem independently:
- different_relation: asks for a different property/relation;
- different_answer_type: changes the expected type of missing answer;
- added_factual_claim: adds unsupported factual/background restrictions or assertions;
- malformed_or_incomplete: grammar/structure makes the requested relation unclear;
- generic_wrapper: mostly wraps/copies ORIGINAL instead of actually paraphrasing it.

Return ONLY one JSON object on one line, exactly with these keys:
{{"different_relation":false,"different_answer_type":false,"added_factual_claim":false,"malformed_or_incomplete":false,"generic_wrapper":false,"verdict":"SAFE","reason":"short reason"}}
If any problem exists, set it true and use verdict "UNSAFE"."""


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


def parse_equivalence(text: str) -> Dict[str, Any]:
    value = _extract_json(text)
    keys = (
        "same_relation",
        "same_answer_type",
        "no_added_factual_claims",
        "grammatical",
        "completion_compatible",
    )
    valid = bool(value) and all(value.get(k) is True for k in keys) and str(value.get("verdict", "")).upper() == "ACCEPT"
    return {
        "parsed": value,
        "accepted": bool(valid),
        "parse_ok": value is not None,
        "raw": str(text),
    }


def parse_critic(text: str) -> Dict[str, Any]:
    value = _extract_json(text)
    keys = (
        "different_relation",
        "different_answer_type",
        "added_factual_claim",
        "malformed_or_incomplete",
        "generic_wrapper",
    )
    valid = bool(value) and all(value.get(k) is False for k in keys) and str(value.get("verdict", "")).upper() == "SAFE"
    return {
        "parsed": value,
        "accepted": bool(valid),
        "parse_ok": value is not None,
        "raw": str(text),
    }


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
        texts = [_chat_text(tok, x) for x in prompts]
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
    """Return one auditable validation record per candidate."""
    results: List[Dict[str, Any]] = [None] * len(candidates)  # type: ignore[list-item]
    judge_indices: List[int] = []
    for i, candidate in enumerate(candidates):
        structural = structural_rejection_reason(direct_prompt, candidate)
        if structural is not None:
            results[i] = {
                "candidate": str(candidate),
                "accepted": False,
                "structural_rejection": structural,
                "equivalence": None,
                "critic": None,
            }
        else:
            judge_indices.append(i)

    for start in range(0, len(judge_indices), int(batch_size)):
        idxs = judge_indices[start : start + int(batch_size)]
        eq_prompts = [
            equivalence_instruction(subject, direct_prompt, str(candidates[i]))
            for i in idxs
        ]
        critic_prompts = [
            critic_instruction(subject, direct_prompt, str(candidates[i]))
            for i in idxs
        ]
        eq_raw = _generate_json_batch(
            model, tok, eq_prompts, max_new_tokens=max_new_tokens
        )
        critic_raw = _generate_json_batch(
            model, tok, critic_prompts, max_new_tokens=max_new_tokens
        )
        for i, eq_text, critic_text in zip(idxs, eq_raw, critic_raw):
            eq = parse_equivalence(eq_text)
            critic = parse_critic(critic_text)
            accepted = bool(eq["accepted"] and critic["accepted"])
            results[i] = {
                "candidate": str(candidates[i]),
                "accepted": accepted,
                "structural_rejection": None,
                "equivalence": eq,
                "critic": critic,
            }
    return results


def rejection_reason(record: Dict[str, Any]) -> str:
    if record.get("accepted"):
        return "accepted"
    if record.get("structural_rejection"):
        return str(record["structural_rejection"])
    eq = record.get("equivalence") or {}
    cr = record.get("critic") or {}
    if not eq.get("parse_ok", False):
        return "equivalence_parse_failure"
    if not eq.get("accepted", False):
        parsed = eq.get("parsed") or {}
        failed = [
            k for k in (
                "same_relation",
                "same_answer_type",
                "no_added_factual_claims",
                "grammatical",
                "completion_compatible",
            ) if parsed.get(k) is not True
        ]
        return "equivalence:" + (",".join(failed) if failed else "reject")
    if not cr.get("parse_ok", False):
        return "critic_parse_failure"
    if not cr.get("accepted", False):
        parsed = cr.get("parsed") or {}
        flagged = [
            k for k in (
                "different_relation",
                "different_answer_type",
                "added_factual_claim",
                "malformed_or_incomplete",
                "generic_wrapper",
            ) if parsed.get(k) is True
        ]
        return "critic:" + (",".join(flagged) if flagged else "unsafe")
    return "rejected"
