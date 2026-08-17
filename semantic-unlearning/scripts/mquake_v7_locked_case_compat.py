#!/usr/bin/env python3
"""Compatibility shim for rewrite-only locked MQuAKE V7 case expansion.

The source-locked evaluator's ``expand_prediction_cases`` historically built
both rewrite and AtomicGen prompt groups eagerly.  Locked repair-visible records
intentionally omit ``atomic_gen_prompt``.  Therefore a rewrite-only caller could
raise KeyError even though AtomicGen was not requested.

This shim preserves the evaluator's exact tokenization/prefix semantics while
reading ``atomic_gen_prompt`` only when ``prompt_types`` actually requests it.
It is intentionally tiny so Stage 1/2 can remain data-firewalled without
changing official final evaluation behavior.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence

import mquake_zero_unlearn_official_eval as mquake


def safe_expand_prediction_cases(
    record: Mapping[str, Any],
    tok: Any,
    *,
    llama_like: bool,
    prompt_types: Sequence[str] = ("rewrite", "atomic_gen"),
) -> List[mquake.PredictionCase]:
    """Exact evaluator semantics with lazy prompt-field access."""

    rewrite = record["requested_rewrite"]
    subject = str(rewrite["subject"])
    sensitive = str(rewrite["target_true"]["str"])
    target_ids = mquake.original_answer_token_ids(
        tok,
        sensitive,
        llama_like=llama_like,
    )

    cases: List[mquake.PredictionCase] = []
    for prompt_type in prompt_types:
        if prompt_type == "rewrite":
            prompts = [str(rewrite["prompt"]).format(subject)]
        elif prompt_type == "atomic_gen":
            if "atomic_gen_prompt" not in record:
                raise KeyError(
                    "atomic_gen_prompt is unavailable in this locked record; "
                    "AtomicGen is evaluation-only and must not be requested by "
                    "Stage 1/2"
                )
            prompts = [str(record["atomic_gen_prompt"])]
        else:
            raise ValueError(f"Unsupported MQuAKE prompt type: {prompt_type}")

        for prompt_index, prompt in enumerate(prompts):
            for token_index, token_id in enumerate(target_ids):
                decoded_prefix = tok.decode(target_ids[:token_index])
                if llama_like and token_index > 0:
                    evaluated_prompt = prompt + " " + decoded_prefix
                else:
                    evaluated_prompt = prompt + decoded_prefix
                cases.append(
                    mquake.PredictionCase(
                        case_id=int(record["case_id"]),
                        prompt_type=prompt_type,
                        prompt_index=prompt_index,
                        token_index=token_index,
                        prompt=evaluated_prompt,
                        target_text=tok.decode([token_id]),
                    )
                )
    return cases


def install() -> None:
    """Patch the already-imported evaluator module for this Python process."""

    mquake.expand_prediction_cases = safe_expand_prediction_cases
