#!/usr/bin/env python3
"""Robust v2 entrypoint for locked MCF surrogate paraphrase generation.

This wrapper keeps the v1 artifact schema/protocol and data-access contract, but
hardens generation in two ways:

1. Local-LLM generation is repeated over several deterministic RNG seeds so the
   semantic candidate pool is larger before validation.
2. Deterministic fallbacks wrap the already-formatted direct prompt WITHOUT
   repeating the subject outside that prompt. This matters when the subject
   string is also target_true or target_new: repeating the subject would create
   a new answer occurrence and the baseline-aware leakage guard would correctly
   reject the fallback.

Official MCF paraphrase/neighborhood/retain/PPL data remain unavailable.
"""
from __future__ import annotations

from typing import Any, Dict, List

import build_mcf_surrogate_paraphrases as base


_ORIGINAL_GENERATE = base.generate_local_surrogates


def deterministic_surrogates(subject: str, direct_prompt: str) -> List[str]:
    """Fallback variants that never duplicate ``subject`` outside direct_prompt."""
    del subject  # subject is already present in the formatted direct prompt
    direct = base._normalize_space(direct_prompt)
    prefixes = [
        "Complete this factual statement:",
        "Complete the following fact:",
        "Provide the missing factual completion:",
        "Supply the missing continuation:",
        "Finish this factual relation:",
        "Give the missing object for this fact:",
        "Answer with the missing completion:",
        "Fill in the missing factual continuation:",
        "What completes this factual statement?",
        "Which completion makes this fact complete?",
        "Provide only the missing factual continuation:",
        "Complete the relation expressed here:",
        "Finish the following factual prompt:",
        "Supply the factual continuation for:",
        "Give the completion requested by this fact:",
        "Complete this knowledge statement:",
        "Provide the object that completes this relation:",
        "Fill the missing end of this factual statement:",
        "Return only the factual completion for:",
        "Complete the factual relation below:",
    ]
    seen = {base._normalize_cmp(direct)}
    out: List[str] = []
    for prefix in prefixes:
        value = base._clean_candidate(f"{prefix} {direct}")
        key = base._normalize_cmp(value)
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def generate_local_surrogates(
    model,
    tok,
    *,
    subject: str,
    direct_prompt: str,
    count: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Dict[str, Any]:
    """Accumulate semantic candidates from three reproducible sampling rounds."""
    accepted: List[str] = []
    raw: List[str] = []
    # Large prime offsets make the rounds reproducible but independent enough to
    # diversify local generations. Validation/deduplication still happens in v1.
    for round_idx, offset in enumerate((0, 104729, 209759)):
        result = _ORIGINAL_GENERATE(
            model,
            tok,
            subject=subject,
            direct_prompt=direct_prompt,
            count=count,
            seed=int(seed) + int(offset),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        accepted.extend(result["accepted"])
        raw.extend(result["raw"])
    return {"accepted": accepted, "raw": raw}


def main(argv=None) -> None:
    # v1 main resolves these names from its module globals at runtime, so this
    # safely upgrades only candidate generation while reusing its locked split,
    # baseline-aware leakage validation, artifact schema, and receipts.
    base.deterministic_surrogates = deterministic_surrogates
    base.generate_local_surrogates = generate_local_surrogates
    base.main(argv)


if __name__ == "__main__":
    main()
