#!/usr/bin/env python3
"""Development-only relation paraphrases for passive shadow routing.

The strings in this file are authored scaffolds built from Wikidata relation
names.  They do not read MultiCounterFact's official paraphrase or
neighborhood fields.  Calibration families may be used to train the router
and actuator; the held-out families are certificate-only and must never enter
an optimizer update.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Sequence


SCHEMA_VERSION = 1


RELATION_NOUN_PHRASES: Dict[str, str] = {
    "P27": "country of citizenship",
    "P30": "continent",
    "P413": "playing position",
    "P1412": "language used for writing",
    "P103": "native language",
    "P495": "country of origin",
    "P176": "manufacturer",
    "P17": "country",
    "P136": "genre",
    "P937": "work location",
    "P106": "profession",
    "P20": "place of death",
    "P449": "original broadcast network",
    "P19": "place of birth",
    "P740": "founding location",
    "P159": "headquarters location",
    "P364": "original language",
    "P131": "administrative region",
    "P37": "official language",
    "P178": "developer",
    "P276": "location",
    "P1303": "musical instrument",
    "P101": "field of work",
    "P39": "position held",
    "P127": "owner",
    "P140": "religion",
    "P108": "employer",
    "P641": "sport",
    "P138": "namesake",
    "P407": "language of the work",
    "P463": "member organization",
    "P36": "capital city",
    "P190": "sister city",
    "P264": "record label",
}


# Calibration prompts are visible to optimization.  Held-out prompts are only
# opened after router and actuator weights have been frozen for the run.
CALIBRATION_SCAFFOLDS: Sequence[str] = (
    "For {subject}, the recorded {relation} is",
    "In reference to {subject}, sources give the {relation} as",
    "The {relation} associated with {subject} is",
    "Asked for the {relation} of {subject}, the answer is",
)

HELDOUT_SCAFFOLDS: Sequence[str] = (
    "Identify the {relation} for {subject}:",
    "What is {subject}'s {relation}? It is",
    "Concerning {subject}, its {relation} is",
    "When asked about the {relation} connected with {subject}, respond with",
)


@dataclass(frozen=True)
class ShadowPromptSpec:
    prompt: str
    owner_index: int
    case_id: int
    relation_id: str
    split: str
    positive: bool
    family_index: int
    source: str

    def json(self) -> Dict[str, Any]:
        return asdict(self)


def _noun(relation_id: str) -> str:
    value = RELATION_NOUN_PHRASES.get(str(relation_id))
    if value is None:
        raise ValueError(f"no development relation noun registered for {relation_id!r}")
    return value


def _render(scaffold: str, *, subject: str, relation_id: str) -> str:
    return scaffold.format(subject=str(subject), relation=_noun(relation_id))


def _with_prefix(prompt: str, prefix: str | None) -> str:
    normalized = " ".join(str(prefix or "").split())
    return f"{normalized} {prompt}" if normalized else prompt


def build_positive_specs(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    corpus_prefixes: Sequence[str] = (),
) -> List[ShadowPromptSpec]:
    """Build relation-aware calibration or certificate-only positives."""

    if split == "calibration":
        scaffolds = CALIBRATION_SCAFFOLDS
    elif split == "heldout":
        scaffolds = HELDOUT_SCAFFOLDS
    else:
        raise ValueError("split must be calibration or heldout")
    prefixes = [str(value) for value in corpus_prefixes if str(value).strip()]
    values: List[ShadowPromptSpec] = []
    for owner, record in enumerate(records):
        case_id = int(record["case_id"])
        subject = str(record["subject"])
        relation_id = str(record["relation_id"])
        _noun(relation_id)
        for family, scaffold in enumerate(scaffolds):
            prompt = _render(scaffold, subject=subject, relation_id=relation_id)
            prefix = (
                prefixes[(case_id + family * 17) % len(prefixes)]
                if prefixes and family % 2 == 1
                else None
            )
            values.append(
                ShadowPromptSpec(
                    prompt=_with_prefix(prompt, prefix),
                    owner_index=owner,
                    case_id=case_id,
                    relation_id=relation_id,
                    split=split,
                    positive=True,
                    family_index=family,
                    source="authored_relation_noun_scaffold",
                )
            )
    return values


def build_wrong_relation_specs(
    records: Sequence[Mapping[str, Any]],
    *,
    split: str,
    variants_per_record: int = 2,
    corpus_prefixes: Sequence[str] = (),
) -> List[ShadowPromptSpec]:
    """Build exact-subject, different-relation hard negatives.

    A subject may legitimately occur in multiple forget records.  Distractor
    relations are therefore chosen outside the set registered for that exact
    subject, preventing a synthetic negative from contradicting a positive
    label for another record.
    """

    if int(variants_per_record) <= 0:
        return []
    scaffolds = (
        CALIBRATION_SCAFFOLDS if split == "calibration" else HELDOUT_SCAFFOLDS
    )
    if split not in {"calibration", "heldout"}:
        raise ValueError("split must be calibration or heldout")
    prefixes = [str(value) for value in corpus_prefixes if str(value).strip()]
    relation_ids = sorted(RELATION_NOUN_PHRASES)
    subject_relations: Dict[str, set[str]] = {}
    for record in records:
        subject_relations.setdefault(str(record["subject"]), set()).add(
            str(record["relation_id"])
        )

    values: List[ShadowPromptSpec] = []
    for owner, record in enumerate(records):
        subject = str(record["subject"])
        forbidden = subject_relations[subject]
        available = [value for value in relation_ids if value not in forbidden]
        if len(available) < int(variants_per_record):
            raise ValueError(f"insufficient distractor relations for {subject!r}")
        case_id = int(record["case_id"])
        start = (case_id * 13 + owner * 7) % len(available)
        for family in range(int(variants_per_record)):
            distractor = available[(start + family * 11) % len(available)]
            scaffold = scaffolds[(family + owner) % len(scaffolds)]
            prompt = _render(scaffold, subject=subject, relation_id=distractor)
            prefix = (
                prefixes[(case_id + family * 19) % len(prefixes)]
                if prefixes and family % 2 == 1
                else None
            )
            values.append(
                ShadowPromptSpec(
                    prompt=_with_prefix(prompt, prefix),
                    owner_index=owner,
                    case_id=case_id,
                    relation_id=distractor,
                    split=split,
                    positive=False,
                    family_index=family,
                    source="same_subject_different_relation_hard_negative",
                )
            )
    return values


def coverage_report(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    missing = sorted(
        {
            str(record.get("relation_id") or "<missing>")
            for record in records
            if str(record.get("relation_id") or "") not in RELATION_NOUN_PHRASES
        }
    )
    return {
        "records": len(records),
        "registered_relation_nouns": len(RELATION_NOUN_PHRASES),
        "missing_relation_ids": missing,
        "complete": not missing,
        "official_evaluation_prompts_seen": 0,
    }
