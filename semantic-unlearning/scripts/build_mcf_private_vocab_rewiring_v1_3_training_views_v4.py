#!/usr/bin/env python3
"""V1.3 training-view generator revision 4: subject-slot generation.

This is a narrow generation-format repair for the 10-view V1.3b ablation.
The frozen Base model is instructed to emit the literal marker ``[SUBJECT]``
exactly once instead of reproducing a potentially long subject string.  Before
any candidate is scored or admitted, the marker is deterministically replaced
with the actual training-visible subject.  The existing subject-aware answer-
leak check, Base semantic/logprob gates, relation-template priority, and held-
out leakage firewall are otherwise unchanged.

No official paraphrase, neighborhood, generation, or retain text is read.
"""
from __future__ import annotations

import build_mcf_private_vocab_rewiring_v1_3_training_views_v3 as v3


SUBJECT_SLOT = "[SUBJECT]"
_CURRENT_SUBJECT: str | None = None
_ORIGINAL_CLEAN_GENERATED_LINES = v3.base.clean_generated_lines


def slot_instruction(subject: str, canonical: str, count: int, mode: int) -> str:
    """Request relation-preserving paraphrases with a deterministic subject slot."""
    global _CURRENT_SUBJECT
    _CURRENT_SUBJECT = str(subject)
    if canonical.count(subject) != 1:
        raise RuntimeError(
            f"canonical prompt must contain the literal subject exactly once: {subject!r}"
        )
    canonical_slot = canonical.replace(subject, SUBJECT_SLOT, 1)
    styles = [
        "Make only small syntactic changes. Keep all relation-bearing content words whenever possible.",
        "Convert the prompt into natural question-style variants without changing the requested relation.",
        "Create declarative completion variants with the same relation and no factual answer.",
        "Reorder or minimally rephrase the syntax while preserving the exact semantic slot being requested.",
    ]
    return f"""Create {count} TRAINING-ONLY paraphrases of this factual prompt.

Hard rules:
- Use the exact marker {SUBJECT_SLOT} exactly once in EVERY output line.
- Do NOT write or alter the real subject name; the pipeline will substitute it later.
- Preserve exactly the SAME semantic relation as the original prompt.
- Do NOT broaden, narrow, or switch the relation.
- Do NOT answer the fact and do NOT include any candidate factual value.
- Each output must still be completable by the same short answer type.
- {styles[mode % len(styles)]}
- Output exactly {count} lines and nothing else.

Original prompt:
{canonical_slot}

Paraphrases:"""


def clean_generated_lines_with_subject_slot(text: str) -> list[str]:
    """Replace one generated subject marker with the actual subject before scoring."""
    lines = _ORIGINAL_CLEAN_GENERATED_LINES(text)
    subject = _CURRENT_SUBJECT
    if not subject:
        raise RuntimeError("subject-slot cleaner called before generation instruction")
    out: list[str] = []
    for line in lines:
        if line.count(SUBJECT_SLOT) == 1:
            out.append(line.replace(SUBJECT_SLOT, subject, 1))
        else:
            # Leave malformed generations unchanged so the existing
            # subject-exactly-once validator rejects them normally.
            out.append(line)
    return out


def main() -> None:
    # Narrow monkeypatches used only by this entrypoint.  v3.main still performs
    # all data validation, same-relation protection-fit template admission,
    # answer-leak checks, Base semantic/logprob gates, manifest writing, and
    # no-held-out-fallback behavior.
    v3.minimal_instruction = slot_instruction
    v3.base.clean_generated_lines = clean_generated_lines_with_subject_slot
    v3.main()


if __name__ == "__main__":
    main()
