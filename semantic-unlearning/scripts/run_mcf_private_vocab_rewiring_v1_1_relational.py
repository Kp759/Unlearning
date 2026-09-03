#!/usr/bin/env python3
"""Registered V1.1 entrypoint with explicit same-subject relation preservation."""
from __future__ import annotations

import random
from typing import Any, Dict, Mapping, Sequence

import run_mcf_private_vocab_rewiring_v1_1 as runner


RELATION_RETAIN_PER_SUBJECT = 8
_ORIGINAL_RETAIN_BUILDER = runner.v1.make_retain_contexts


def make_relation_preserving_retain_contexts(
    forget_records: Sequence[Mapping[str, Any]],
    protection_fit: Sequence[Mapping[str, Any]],
) -> tuple[list[str], Dict[str, int]]:
    """Add training-safe other-relation frames for each forget subject.

    Relation templates come only from the visible protection-fit partition.  For
    each forget record, templates with the same relation_id as the forgotten
    relation are excluded.  The Base model supplies the teacher distribution, so
    no target labels or official evaluation prompts are exposed.
    """
    contexts, stats = _ORIGINAL_RETAIN_BUILDER(forget_records, protection_fit)
    templates: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in protection_fit:
        rr = record["requested_rewrite"]
        relation_id = str(rr["relation_id"])
        prompt = str(rr["prompt"])
        if "{}" not in prompt:
            continue
        key = (relation_id, prompt)
        if key not in seen:
            seen.add(key)
            templates.append(key)
    templates.sort()

    added: list[str] = []
    for record in forget_records:
        rr = record["requested_rewrite"]
        subject = str(rr["subject"])
        forget_relation = str(rr["relation_id"])
        candidates = [item for item in templates if item[0] != forget_relation]
        rng = random.Random(11017 + int(record["case_id"]))
        take = min(RELATION_RETAIN_PER_SUBJECT, len(candidates))
        chosen = rng.sample(candidates, take) if take else []
        for _relation_id, prompt in chosen:
            added.append(prompt.format(subject))

    combined = list(dict.fromkeys([*contexts, *added]))
    out_stats = dict(stats)
    out_stats.update(
        {
            "relation_templates_available": len(templates),
            "same_subject_different_relation_requested_per_forget": RELATION_RETAIN_PER_SUBJECT,
            "same_subject_different_relation_contexts_added": len(
                list(dict.fromkeys(added))
            ),
            "total_unique_contexts": len(combined),
        }
    )
    return combined, out_stats


def main() -> None:
    runner.v1.make_retain_contexts = make_relation_preserving_retain_contexts
    runner.main()


if __name__ == "__main__":
    main()
