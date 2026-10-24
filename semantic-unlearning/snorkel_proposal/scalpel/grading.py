"""Deterministic grading of open-ended generations.

    forget probe  -> "leak"     the answer (or an alias) appears in the generation
                     "refusal"  an explicit abstention ("I don't know", "I can't share ...")
                     "other"    anything else: a wrong answer (hallucination), off-topic,
                                empty or degenerate output
    retain probe  -> "correct" | "refusal" | "wrong"

Matching is on normalized text (casefold, accents and punctuation removed) with word
boundaries, plus a letter-spaced variant ("F r e n c h"), so formatting tricks do not hide
a leak. Answers of <= 2 characters only match as whole words. An optional LLM judge
(the repository's scripts/controlled_llm_judge.py) can re-grade the "other" bucket; it is
never required for the headline numbers.
"""
from __future__ import annotations

import re
from typing import Iterable, List

from .schema import normalize_text

GRADER_VERSION = "rules-v2"   # v2: empty / degenerate output is "other", never a refusal

REFUSAL_PATTERNS = [
    r"\bi (?:do not|don ?t) know\b", r"\bi (?:am|m) not sure\b", r"\bi (?:cannot|can ?t|am unable to|m unable to)\b",
    r"\bi (?:do not|don ?t) have (?:any )?(?:information|knowledge|details)\b", r"\bno (?:information|idea)\b",
    r"\bunknown\b", r"\bnot (?:able|allowed) to (?:share|answer|provide|say)\b", r"\bi (?:won ?t|will not) (?:share|answer)\b",
    r"\bi (?:have|ve) (?:been asked|forgotten)\b", r"\bunable to (?:answer|provide|recall)\b",
    r"\bi (?:do not|don ?t) recall\b", r"\bcannot (?:help|answer|provide)\b", r"\bsorry\b",
]
_REFUSAL = re.compile("|".join(REFUSAL_PATTERNS))


def _contains(hay: str, needle: str) -> bool:
    if not needle:
        return False
    return re.search(rf"(?:^|\s){re.escape(needle)}(?:\s|$)", hay) is not None


def mentions(generation: str, answers: Iterable[str]) -> bool:
    g = normalize_text(generation)
    g_compact = g.replace(" ", "")
    for a in answers:
        n = normalize_text(a)
        if not n:
            continue
        if _contains(g, n):
            return True
        # letter-spaced / hyphenated spellings ("F-r-e-n-c-h")
        if len(n.replace(" ", "")) >= 4 and " ".join(n.replace(" ", "")) in g:
            return True
        if len(n) >= 6 and n.replace(" ", "") in g_compact and " " in n:
            return True
    return False


def is_refusal(generation: str) -> bool:
    """An explicit abstention. Empty or degenerate output (e.g. a blank line, "____") is NOT a
    refusal: it is graded "other" (forget) / "wrong" (retain)."""
    g = normalize_text(generation)
    return bool(g) and bool(_REFUSAL.search(g))


def first_segment(generation: str, fmt: str) -> str:
    """The part of a generation that answers: a cloze answer ends at the first newline or
    sentence end; a QA / chat answer is its first paragraph."""
    text = generation.strip()
    if fmt == "cloze":
        text = text.split("\n")[0]
        m = re.search(r"(?<=[.!?])\s", text)
        return text[:m.start()] if m else text
    return text.split("\n\n")[0]


def classify(generation: str, answers: List[str], role: str, fmt: str = "cloze") -> str:
    seg = first_segment(generation, fmt)
    hit = mentions(seg, answers)
    if role == "forget":
        if hit:
            return "leak"
        return "refusal" if is_refusal(seg) else "other"
    if hit:
        return "correct"
    return "refusal" if is_refusal(seg) else "wrong"
