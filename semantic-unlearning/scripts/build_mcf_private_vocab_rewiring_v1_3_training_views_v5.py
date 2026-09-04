#!/usr/bin/env python3
"""V1.3 training-view generator revision 5: relation-anchored subject-slot generation.

This is a narrow training-view generation repair for the 10-view V1.3b ablation.
The scientific acceptance thresholds and downstream worst-1 objective are unchanged.

Compared with revision 4, generation no longer paraphrases only the forget case's
single canonical wording.  The wrapper captures the sanitized direct protection_fit
records already validated by revision 3 and cycles through templates with the exact
same relation_id as generation anchors.  The frozen Base model emits [SUBJECT]
exactly once; that marker is replaced with the real training-visible subject before
any Base semantic/logprob scoring or admission.

No official paraphrase, neighborhood, generation, or retain text is read.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import build_mcf_private_vocab_rewiring_v1_3_training_views_v3 as v3


SUBJECT_SLOT = "[SUBJECT]"
_ORIGINAL_VALIDATE_DIRECT = v3.validate_direct
_ORIGINAL_CLEAN_GENERATED_LINES = v3.base.clean_generated_lines

_FORGET_RELATION_BY_KEY: dict[tuple[str, str], str] = {}
_RELATION_TEMPLATES: dict[str, list[str]] = defaultdict(list)
_CURRENT_SUBJECT: str | None = None


def capture_direct(records: Sequence[Mapping[str, Any]], role: str) -> None:
    """Preserve v3 validation and capture only already-sanitized direct templates."""
    _ORIGINAL_VALIDATE_DIRECT(records, role)
    if role == "forget":
        for record in records:
            rr = record["requested_rewrite"]
            subject = str(rr["subject"])
            template = str(rr["prompt"])
            canonical = v3.base.normalize_space(template.format(subject))
            _FORGET_RELATION_BY_KEY[(subject, canonical)] = str(rr["relation_id"])
    elif role == "protection_fit":
        for record in records:
            rr = record["requested_rewrite"]
            rid = str(rr["relation_id"])
            template = str(rr["prompt"])
            if template not in _RELATION_TEMPLATES[rid]:
                _RELATION_TEMPLATES[rid].append(template)


def relation_anchored_instruction(subject: str, canonical: str, count: int, mode: int) -> str:
    """Generate minimal paraphrases around a same-relation training-visible anchor."""
    global _CURRENT_SUBJECT
    _CURRENT_SUBJECT = str(subject)

    key = (str(subject), v3.base.normalize_space(canonical))
    relation_id = _FORGET_RELATION_BY_KEY.get(key)
    if relation_id is None:
        raise RuntimeError(f"cannot resolve relation_id for forget prompt: {subject!r}")

    anchors = list(_RELATION_TEMPLATES.get(relation_id, []))
    # The canonical wording is also training-visible and is a valid anchor.  Keep
    # it first, then cycle through distinct same-relation protection-fit templates.
    canonical_template = canonical.replace(subject, "{}", 1) if canonical.count(subject) == 1 else None
    ordered: list[str] = []
    if canonical_template is not None:
        ordered.append(canonical_template)
    for template in anchors:
        if template not in ordered:
            ordered.append(template)
    if not ordered:
        raise RuntimeError(f"no training-visible anchor for relation {relation_id}")

    anchor_template = ordered[mode % len(ordered)]
    if anchor_template.count("{}") != 1:
        raise RuntimeError(f"invalid relation anchor for {relation_id}: {anchor_template!r}")
    anchor_slot = anchor_template.replace("{}", SUBJECT_SLOT, 1)

    styles = [
        "Make only a very small syntactic change; preserve the completion style and answer type.",
        "Change wording minimally while keeping all relation-bearing words or close synonyms.",
        "Reorder the syntax slightly but preserve the exact factual slot and completion behavior.",
        "Produce conservative lexical variants; do not turn a completion into a different task format.",
    ]

    return f"""Create {count} TRAINING-ONLY paraphrases of the anchor prompt below.

Relation identifier: {relation_id}

Hard rules:
- Use the exact marker {SUBJECT_SLOT} exactly once in EVERY output line.
- Do NOT write or alter the real subject name; the pipeline substitutes it later.
- Preserve exactly the SAME semantic relation and SAME short-answer type as the anchor.
- Preserve the anchor's completion/question style as closely as possible.
- Do NOT broaden, narrow, or switch the relation.
- Do NOT answer the fact and do NOT include any candidate factual value.
- {styles[mode % len(styles)]}
- Output exactly {count} lines and nothing else.

Same-relation training-visible anchor:
{anchor_slot}

Paraphrases:"""


def clean_generated_lines_with_subject_slot(text: str) -> list[str]:
    """Substitute exactly one generated subject slot before existing v3 validation."""
    lines = _ORIGINAL_CLEAN_GENERATED_LINES(text)
    subject = _CURRENT_SUBJECT
    if not subject:
        raise RuntimeError("subject-slot cleaner called before generation instruction")
    out: list[str] = []
    for line in lines:
        if line.count(SUBJECT_SLOT) == 1:
            out.append(line.replace(SUBJECT_SLOT, subject, 1))
        else:
            # Keep malformed output unchanged so v3's existing exactly-once subject
            # check rejects it; do not silently repair malformed generations.
            out.append(line)
    return out


def main() -> None:
    # Narrow monkeypatches only. v3.main retains sanitized-input validation,
    # relation-template direct admission, target leakage checks, unchanged Base
    # semantic/logprob gates, output provenance, and no-held-out-fallback behavior.
    v3.validate_direct = capture_direct
    v3.minimal_instruction = relation_anchored_instruction
    v3.base.clean_generated_lines = clean_generated_lines_with_subject_slot
    v3.main()


if __name__ == "__main__":
    main()
