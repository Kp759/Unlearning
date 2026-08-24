#!/usr/bin/env python3
"""Hand-authored synthetic paraphrase templates for MCF Stage-1 direction fitting.

MCF's real ``paraphrase_prompts`` combine two independent sources of surface
variation: (a) an unrelated narrative lead-in sentence with no semantic
relation to the fact, and (b) a genuinely different grammatical phrasing of
the relation itself, immediately before the answer continuation. Both are
authored here from scratch -- never copied from, or derived from, any
record's real ``paraphrase_prompts`` -- so the official held-out paraphrase
set stays uncontaminated and GFS/Gen remains an honest measure of whether
this transfers.

``RELATION_ALTERNATE_TEMPLATES`` covers the 34 relation ids used by the
standard MultiCounterFact release. Each entry gives 2 alternate cloze
templates, syntactically distinct from the dataset's own canonical
``requested_rewrite.prompt`` template for that relation. ``GENERIC_CONTEXT_PREFIXES``
are content-free lead-in sentences applied on top of a chosen template to add
the "irrelevant preceding context" axis. ``GENERIC_FALLBACK_TEMPLATES`` are a
relation-agnostic safety net used only if a record's ``relation_id`` is
missing or falls outside this bank (e.g. a future dataset revision), so the
pipeline degrades gracefully instead of failing closed.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence

RELATION_ALTERNATE_TEMPLATES: Dict[str, List[str]] = {
    "P27": ["The country that {} is a citizen of is", "{} holds citizenship in"],
    "P30": ["The continent where {} is located is", "{} is situated on the continent of"],
    "P413": ["The position that {} plays is", "In their sport, {} plays as a"],
    "P1412": ["{} writes and publishes in", "The language {} uses for writing is"],
    "P103": ["The native language of {} is", "{} grew up speaking"],
    "P495": ["{} originates from", "The country of origin of {} is"],
    "P176": ["The manufacturer of {} is", "{} is manufactured by"],
    "P17": ["The country where {} is located is", "{} is situated in the country of"],
    "P136": ["The genre that {} performs is", "{} is known for performing"],
    "P937": ["The city where {} worked is", "{} carried out their work in the city of"],
    "P106": ["The profession of {} is", "{} works as a"],
    "P20": ["The place where {} died is", "{} passed away in"],
    "P449": ["The network that originally aired {} is", "{} was originally broadcast on"],
    "P19": ["The birthplace of {} is", "{} was born in"],
    "P740": ["The location where {} was founded is", "{} was established in"],
    "P159": ["{} has its headquarters located in", "The headquarters of {} can be found in"],
    "P364": ["The original language of {} is", "{} was originally produced in the language of"],
    "P131": ["{} is located within", "The administrative region containing {} is"],
    "P37": ["The official language spoken in {} is", "In {}, the language people speak is"],
    "P178": ["The developer of {} is", "{} was developed by"],
    "P276": ["{} can be found in", "The location of {} is"],
    "P1303": ["The instrument that {} plays is", "{} is known for playing the"],
    "P101": ["{} specializes in the field of", "The area of expertise of {} is"],
    "P39": ["The position held by {} is", "{} serves in the position of"],
    "P127": ["{} is owned by", "The owner of {} is"],
    "P140": ["The religion followed by {} is", "{} practices the religion of"],
    "P108": ["{} is employed by", "The employer of {} is"],
    "P641": ["The sport that {} plays is", "{} is an athlete who plays"],
    "P138": ["{} was named in honor of", "The namesake of {} is"],
    "P407": ["The language that {} was written in is", "{} was composed in the language of"],
    "P463": ["{} is a member of", "The organization that {} belongs to is"],
    "P36": ["The capital of {} is", "{}'s capital city is"],
    "P190": ["The sister city of {} is", "{} is twinned with"],
    "P264": ["The record label of {} is", "{} is signed to the label"],
}

GENERIC_CONTEXT_PREFIXES: List[str] = [
    "According to publicly available records,",
    "As has been noted elsewhere,",
    "In an earlier account,",
    "Based on commonly cited sources,",
]

GENERIC_FALLBACK_TEMPLATES: List[str] = [
    "Completing the statement, {}",
    "To finish the sentence: {}",
]


def _relation_templates(relation_id: str, canonical_prompt: str) -> List[str]:
    templates = RELATION_ALTERNATE_TEMPLATES.get(relation_id)
    if templates:
        return templates
    return [tmpl.format(canonical_prompt) for tmpl in GENERIC_FALLBACK_TEMPLATES]


def synthetic_prompt_templates(
    *,
    relation_id: str,
    canonical_prompt: str,
    case_id: int,
    count: int,
) -> List[str]:
    """Deterministically build ``count`` alternate cloze templates (each still
    containing one ``{}`` subject placeholder) for one record.

    Candidates interleave a bare grammatical variant with a context-prefixed
    variant of the same template, covering both real paraphrase-variation
    axes before ever repeating a template.
    """
    if count <= 0:
        return []
    templates = _relation_templates(relation_id, canonical_prompt)
    candidates: List[str] = []
    prefix_cycle = 0
    for template in templates:
        candidates.append(template)
        prefix = GENERIC_CONTEXT_PREFIXES[(case_id + prefix_cycle) % len(GENERIC_CONTEXT_PREFIXES)]
        candidates.append(f"{prefix} {template}")
        prefix_cycle += 1
    # If more variants are requested than authored, keep cycling context
    # prefixes over the same templates rather than repeating an identical string.
    while len(candidates) < count:
        template = templates[prefix_cycle % len(templates)]
        prefix = GENERIC_CONTEXT_PREFIXES[(case_id + prefix_cycle) % len(GENERIC_CONTEXT_PREFIXES)]
        candidates.append(f"{prefix} {template}")
        prefix_cycle += 1
    return candidates[:count]


def build_synthetic_records(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
) -> List[Dict[str, Any]]:
    """Return a flat list of synthetic MCF records for direction fitting/GA.

    Each synthetic record has the same ``subject``/``target_true``/``target_new``
    as its source record, with ``requested_rewrite.prompt`` replaced by a
    hand-authored alternate template. Never reads or derives from the source
    record's real ``paraphrase_prompts``.
    """
    if count <= 0:
        return []
    synthetic: List[Dict[str, Any]] = []
    for record in records:
        rr = record.get("requested_rewrite")
        if not isinstance(rr, Mapping):
            raise ValueError("record lacks requested_rewrite")
        relation_id = str(rr.get("relation_id") or "")
        canonical_prompt = str(rr.get("prompt", ""))
        case_id = int(record.get("case_id", 0))
        templates = synthetic_prompt_templates(
            relation_id=relation_id,
            canonical_prompt=canonical_prompt,
            case_id=case_id,
            count=count,
        )
        for template in templates:
            synthetic_rr = dict(rr)
            synthetic_rr["prompt"] = template
            synthetic_record = dict(record)
            synthetic_record["requested_rewrite"] = synthetic_rr
            synthetic.append(synthetic_record)
    return synthetic
