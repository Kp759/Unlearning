#!/usr/bin/env python3
"""High-precision deterministic guards for the MCF robust-prompt adapter.

This module is intentionally part of the MCF dataset adapter, not the SURE
optimizer.  It uses only subject/direct/candidate text and never reads answers
or official held-out MCF probes.

Goal: prefer false negatives (direct-only fallback) over semantically shifted
surrogates.  A candidate must preserve a coarse relation-slot family, preserve
slot direction/type, avoid newly introduced named factual content, and remain a
well-formed open completion or factual question.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Set


PROTOCOL = "mcf_high_precision_slot_guard_v1"

_FAMILY_TYPES = {
    "language": {"language"},
    "continent": {"location"},
    "country": {"location"},
    "origin_location": {"location"},
    "location": {"location"},
    "capital_city": {"location"},
    "producer_agent": {"organization", "person"},
    "founding_time_or_location": {"date_or_year", "location"},
    "formation_time_or_location": {"date_or_year", "location"},
    "occupation_or_role": {"occupation_or_role"},
    "field_or_domain": {"occupation_or_role", "other_entity"},
    "affiliation": {"organization", "other_entity"},
    "named_after": {"person", "organization", "location", "other_entity"},
    "instrument_or_performance_object": {"product_or_artifact", "other_entity"},
    "employment_by": {"organization"},
    "employment_as": {"occupation_or_role"},
    "employment_in": {"organization", "location", "occupation_or_role"},
    "nationality_or_demonym": {"nationality_or_demonym"},
}

_COMMON_CAPITALIZED = {
    "the", "what", "which", "who", "where", "when", "how", "is", "was",
    "are", "were", "in", "on", "at", "of", "a", "an", "for", "to",
}

_SLOT_ENDINGS = (
    " by", " in", " from", " of", " as", " with", " to", " at", " on",
    " is", " was", " are", " were", " language", " tongue", " profession",
    " occupation", " role", " position", " field", " domain", " capital",
    " city", " country", " continent", " nation", " organization",
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip().casefold()


def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.I) is not None


def classify_family(text: str) -> str:
    """Map a prompt to a conservative coarse relation-slot family."""
    t = _norm(text)

    # Explicitly underspecified appositive stems are not safely augmentable.
    if _has(r",\s*(?:the|a|an)?\s*$", t):
        return "ambiguous"

    # Specific relations first.
    if _has(r"\b(?:native\s+speaker|native\s+language|original\s+language|official\s+language|language|tongue|speaks?|spoke|fluent|proficient|convers(?:e|ed|es)|communicat(?:e|ed|es))\b", t):
        return "language"
    if _has(r"\bcapital(?:\s+city)?\b|administrative\s+cent(?:er|re)", t):
        return "capital_city"
    if _has(r"\bnamed\s+after\b|\bin\s+honou?r\s+of\b|\bdesignated\s+in\s+honou?r\s+of\b", t):
        return "named_after"
    if _has(r"\bcontinent(?:al)?\b|\blandmass\b", t):
        return "continent"
    if _has(r"\bcountry\s+of\b|\bnation\s+of\b", t):
        return "country"
    if _has(r"\boriginally\s+from\b|\bhails?\s+from\b|\bnative\s+to\b|\borigin(?:ated)?\s+from\b|\bplace\s+of\s+origin\b", t):
        return "origin_location"

    # Direction-sensitive creation/founding relations.
    if _has(r"\b(?:produced|manufactured|developed|created|conceived)\s+by\b|\bmanufacturer\b|\bautomaker\b|\bcreation\b.*\battributed\s+to\b", t):
        return "producer_agent"
    if _has(r"\b(?:founded|established|started)\s+(?:in|at|on|during)\b|\bwas\s+(?:founded|established|started)\s*$|\boriginated\s+in\b", t):
        return "founding_time_or_location"
    if _has(r"\bformed\s+(?:in|at|on|during)\b|\bemerged\s+from\b|\barose\s+from\b|\bformation\b", t):
        return "formation_time_or_location"

    if _has(r"\bperforming\s+on\b|\bplays?\s+the\b|\binstrument\b", t):
        return "instrument_or_performance_object"
    # Bare "plays" is too underspecified (sport, role, instrument, team, etc.).
    if _has(r"\bplays?\b", t):
        return "ambiguous"

    if _has(r"\bprofession\b|\boccupation\b|\bworks?\s+as\b|\bworked\s+as\b|\bserves?\s+as\b|\bposition\s+of\b|\brole\s+of\b", t):
        return "occupation_or_role"
    if _has(r"\bdomain\s+of\s+(?:work|activity)\b|\bfield\s+of\b|\bprofessional\s+(?:field|sphere|focus|scope)\b|\barea\s+of\s+expertise\b", t):
        return "field_or_domain"

    if _has(r"\bemployed\s+by\b|\bworks?\s+for\b|\bworked\s+for\b", t):
        return "employment_by"
    if _has(r"\bemployed\s+as\b", t):
        return "employment_as"
    if _has(r"\bemployed\s+in\b|\bfound\s+employment\s+in\b|\bused\s+to\s+work\s+in\b|\bworked\s+at\b|\bworks?\s+in\b", t):
        return "employment_in"

    if _has(r"\baffiliated\s+with\b|\bassociated\s+with\b|\bmember\s+of\b|\blinked\s+to\b|\bconnected\s+to\b", t):
        return "affiliation"
    if _has(r"\bis\s+(?:a\s+)?part\s+of\b", t):
        return "affiliation"

    if _has(r"\blocated\s+(?:in|at|within)\b|\bsituated\s+(?:in|at|within)\b|\bcan\s+be\s+found\s+in\b|\btakes?\s+place\s+in\b|\bhoused\s+in\b", t):
        return "location"

    if _has(r"\bnationality\b|\bdemonym\b|\bcitizen\s+of\b", t):
        return "nationality_or_demonym"

    return "unknown"


def family_answer_types(family: str) -> List[str]:
    return sorted(_FAMILY_TYPES.get(str(family), set()))


def _question_explicit_type(text: str) -> Set[str] | None:
    t = _norm(text)
    if not re.match(r"^(?:what|which|who|where|when)\b", t):
        return None
    if re.match(r"^who\b", t):
        return {"person", "organization"}
    if re.match(r"^where\b", t):
        return {"location"}
    if re.match(r"^when\b", t):
        return {"date_or_year"}
    if _has(r"\b(?:vehicle|automobile|car|product|model|artifact|instrument)\b", t):
        return {"product_or_artifact"}
    if _has(r"\b(?:company|organization|organisation|manufacturer|automaker|marque)\b", t):
        return {"organization"}
    if _has(r"\b(?:language|tongue)\b", t):
        return {"language"}
    if _has(r"\b(?:profession|occupation|role|position|job)\b", t):
        return {"occupation_or_role"}
    if _has(r"\b(?:country|nation|city|place|location|continent)\b", t):
        return {"location"}
    if _has(r"\b(?:year|date)\b", t):
        return {"date_or_year"}
    if _has(r"\b(?:sport|event)\b", t):
        return {"event", "other_value"}
    return None


def candidate_answer_types(text: str, family: str) -> List[str]:
    explicit = _question_explicit_type(text)
    if explicit is not None:
        return sorted(explicit)
    return family_answer_types(family)


def _capitalized_content_words(text: str) -> Set[str]:
    words = set()
    for token in re.findall(r"\b[A-Z][A-Za-z0-9.'-]*\b", str(text)):
        key = token.strip(".'").casefold()
        if key and key not in _COMMON_CAPITALIZED:
            words.add(key)
    return words


def introduced_named_content(subject: str, direct_prompt: str, candidate: str) -> List[str]:
    allowed = _capitalized_content_words(subject) | _capitalized_content_words(direct_prompt)
    introduced = _capitalized_content_words(candidate) - allowed
    return sorted(introduced)


def _introduced_numeric_content(direct_prompt: str, candidate: str) -> List[str]:
    base = set(re.findall(r"\b\d+(?:st|nd|rd|th)?\b", str(direct_prompt), flags=re.I))
    cand = set(re.findall(r"\b\d+(?:st|nd|rd|th)?\b", str(candidate), flags=re.I))
    return sorted(cand - base)


def open_slot_compatible(text: str) -> bool:
    """Require a clear factual question or an open completion stem."""
    raw = re.sub(r"\s+", " ", str(text)).strip()
    t = raw.casefold()
    if not raw:
        return False
    if re.match(r"^(?:what|which|who|where|when|how)\b", t):
        return True
    if raw.endswith("?"):
        return True
    if raw.endswith(".") or raw.endswith("!"):
        return False
    return any(t.endswith(x) for x in _SLOT_ENDINGS)


def explicit_constraint_reason(direct_prompt: str, candidate: str) -> str | None:
    """Detect common answer-type narrowing in generated wh-questions."""
    t = _norm(candidate)
    d = _norm(direct_prompt)
    # A modifier before an explicit answer-head is narrowing unless already in direct.
    m = re.match(
        r"^(?:what|which)\s+([a-z-]+)\s+(automaker|manufacturer|company|organization|organisation|vehicle|automobile|car|product|model|language|country|city|profession|occupation|role)\b",
        t,
    )
    if m:
        modifier = m.group(1)
        if modifier not in {"the", "a", "an"} and modifier not in d:
            return f"new_answer_head_modifier:{modifier}"
    return None


def direct_guard(subject: str, direct_prompt: str) -> Dict[str, Any]:
    family = classify_family(direct_prompt)
    safe = family not in {"unknown", "ambiguous"}
    return {
        "protocol": PROTOCOL,
        "family": family,
        "answer_types": family_answer_types(family),
        "safe_to_augment": bool(safe),
        "reason": "known_high_precision_family" if safe else "unknown_or_ambiguous_family",
    }


def candidate_guard(
    *,
    subject: str,
    direct_prompt: str,
    direct_family: str,
    candidate: str,
) -> Dict[str, Any]:
    family = classify_family(candidate)
    direct_types = set(family_answer_types(direct_family))
    cand_types = set(candidate_answer_types(candidate, family))
    named = introduced_named_content(subject, direct_prompt, candidate)
    numeric = _introduced_numeric_content(direct_prompt, candidate)
    constraint = explicit_constraint_reason(direct_prompt, candidate)
    open_ok = open_slot_compatible(candidate)

    family_ok = bool(family == direct_family and family not in {"unknown", "ambiguous"})
    # High precision: do not accept a narrowed/broadened explicit answer-type set.
    type_ok = bool(direct_types and cand_types and direct_types == cand_types)
    accepted = bool(
        family_ok and type_ok and not named and not numeric and constraint is None and open_ok
    )
    reasons: List[str] = []
    if not family_ok:
        reasons.append(f"family_mismatch:{direct_family}->{family}")
    if not type_ok:
        reasons.append(f"answer_type_mismatch:{sorted(direct_types)}->{sorted(cand_types)}")
    if named:
        reasons.append("introduced_named_content:" + ",".join(named))
    if numeric:
        reasons.append("introduced_numeric_content:" + ",".join(numeric))
    if constraint:
        reasons.append(constraint)
    if not open_ok:
        reasons.append("not_clear_open_slot_or_question")

    return {
        "protocol": PROTOCOL,
        "accepted": accepted,
        "direct_family": direct_family,
        "candidate_family": family,
        "direct_answer_types": sorted(direct_types),
        "candidate_answer_types": sorted(cand_types),
        "introduced_named_content": named,
        "introduced_numeric_content": numeric,
        "explicit_constraint_reason": constraint,
        "open_slot_compatible": open_ok,
        "reasons": reasons,
    }
