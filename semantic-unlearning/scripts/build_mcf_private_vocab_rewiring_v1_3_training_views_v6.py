#!/usr/bin/env python3
"""V1.3b revision 6: conservative relation-anchored micro-paraphrases.

This wrapper keeps the revision-3 admission/scoring code and revision-5
training-visible same-relation anchors, but makes generation deliberately
lower-entropy and surface-local.  It does not change any semantic acceptance
threshold, held-out access rule, or downstream worst-1 objective.
"""
from __future__ import annotations

import build_mcf_private_vocab_rewiring_v1_3_training_views_v3 as v3
import build_mcf_private_vocab_rewiring_v1_3_training_views_v5 as v5


SUBJECT_SLOT = v5.SUBJECT_SLOT


def conservative_relation_instruction(subject: str, canonical: str, count: int, mode: int) -> str:
    """Generate distinct but deliberately tiny surface variants of a safe relation anchor."""
    v5._CURRENT_SUBJECT = str(subject)

    key = (str(subject), v3.base.normalize_space(canonical))
    relation_id = v5._FORGET_RELATION_BY_KEY.get(key)
    if relation_id is None:
        raise RuntimeError(f"cannot resolve relation_id for forget prompt: {subject!r}")

    anchors = list(v5._RELATION_TEMPLATES.get(relation_id, []))
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

    operations = [
        "Change only function words, punctuation, or very small word-order details. Keep relation-bearing content words unchanged whenever grammatical.",
        "Make a single conservative syntactic edit. Do not change question/completion style or answer type.",
        "Use a near-copy paraphrase: preserve the relation phrase and alter only surrounding syntax.",
        "Reorder only local syntax around the subject slot while preserving the factual slot exactly.",
    ]

    return f"""Create {count} DISTINCT TRAINING-ONLY MICRO-PARAPHRASES of the anchor below.

Relation identifier: {relation_id}

Hard rules:
- Use the exact marker {SUBJECT_SLOT} exactly once in EVERY output line.
- Do NOT write the real subject name.
- Preserve exactly the SAME semantic relation, factual slot, short-answer type, and question/completion style.
- Stay extremely close to the anchor. Do NOT broaden, narrow, reinterpret, or switch the relation.
- Do NOT answer the fact and do NOT include any candidate factual value.
- {operations[mode % len(operations)]}
- Every line must be distinct from the anchor and from the other lines.
- Output exactly {count} lines and nothing else.

Same-relation training-visible anchor:
{anchor_slot}

Micro-paraphrases:"""


def main() -> None:
    # Reuse v5's capture of sanitized direct relation anchors and subject-slot
    # substitution.  v3.main retains all unchanged admission/logprob gates and
    # the no-held-out-fallback contract.
    v3.validate_direct = v5.capture_direct
    v3.minimal_instruction = conservative_relation_instruction
    v3.base.clean_generated_lines = v5.clean_generated_lines_with_subject_slot
    v3.main()


if __name__ == "__main__":
    main()
