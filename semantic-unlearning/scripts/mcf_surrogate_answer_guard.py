#!/usr/bin/env python3
"""Shared leakage guard for MCF surrogate prompts.

A locked direct prompt can already contain a string that is also present in one
of the training-visible answers (for example through the subject or relation
wording).  Such baseline occurrences are not leakage: the generator already
received the direct prompt.  A surrogate is rejected only when it introduces
*additional* whole-token occurrences of target_true or target_new beyond those
already present in the direct prompt.
"""
from __future__ import annotations

import re
from typing import Sequence


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_cmp(text: str) -> str:
    return normalize_space(text).strip(" \t\n\r\"'`.,:;!?").casefold()


def answer_occurrence_count(text: str, answer: str) -> int:
    haystack = normalize_cmp(text)
    needle = normalize_cmp(answer)
    if not needle:
        return 0
    return len(re.findall(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def contains_answer(text: str, answers: Sequence[str]) -> bool:
    return any(answer_occurrence_count(text, answer) > 0 for answer in answers)


def introduced_answer_occurrences(
    candidate: str,
    direct_prompt: str,
    answers: Sequence[str],
) -> bool:
    """Return True only if candidate adds answer mentions beyond direct prompt."""
    for answer in answers:
        if answer_occurrence_count(candidate, answer) > answer_occurrence_count(
            direct_prompt, answer
        ):
            return True
    return False
