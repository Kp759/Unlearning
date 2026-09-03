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

# V4.0 consumed ``HELDOUT_SCAFFOLDS`` when its shadow-difference classifier
# failed development acceptance.  V4.1 uses two subsequently authored and
# disjoint banks: development is available for checkpoint selection, whereas
# certification is opened exactly once after the router has been selected.
V4_1_DEVELOPMENT_SCAFFOLDS: Sequence[str] = (
    "The requested {relation} entry for {subject} reads",
    "Supply the {relation} linked to {subject}:",
    "With respect to {subject}, name the {relation}:",
    "A reference lists which {relation} for {subject}?",
)

V4_1_CERTIFICATION_SCAFFOLDS: Sequence[str] = (
    "For {subject}, which value is recorded for {relation}?",
    "State the {relation} attributed to {subject}:",
    "Regarding {subject}, the listed {relation} is",
    "What should be supplied as the {relation} of {subject}?",
)

# V4.1 consumed its development family when rank four failed model selection.
# V4.2 uses this new family to select among a preregistered nested rank sweep;
# ``V4_1_CERTIFICATION_SCAFFOLDS`` remains untouched until that selection is
# complete and is then opened for the selected arm only.
V4_2_DEVELOPMENT_SCAFFOLDS: Sequence[str] = (
    "Consulting the record for {subject}, its {relation} is",
    "Give the value of {relation} associated with {subject}:",
    "{subject} has which {relation}? Answer:",
    "The entry under {relation} for {subject} states",
)

# Compatibility alias for draft-only callers; new manifests use the explicit
# development/certification names above.
V4_1_HELDOUT_SCAFFOLDS = V4_1_DEVELOPMENT_SCAFFOLDS


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
    variants_per_record: int | None = None,
) -> List[ShadowPromptSpec]:
    """Build relation-aware calibration or certificate-only positives."""

    if split == "calibration":
        scaffolds = CALIBRATION_SCAFFOLDS
    elif split == "heldout":
        scaffolds = HELDOUT_SCAFFOLDS
    elif split in ("v4_1_heldout", "v4_1_development"):
        scaffolds = V4_1_DEVELOPMENT_SCAFFOLDS
    elif split == "v4_1_certification":
        scaffolds = V4_1_CERTIFICATION_SCAFFOLDS
    elif split == "v4_2_development":
        scaffolds = V4_2_DEVELOPMENT_SCAFFOLDS
    else:
        raise ValueError("unknown development split")
    expanded = variants_per_record is not None
    variants = len(scaffolds) if variants_per_record is None else int(variants_per_record)
    if variants <= 0:
        return []
    prefixes = [str(value) for value in corpus_prefixes if str(value).strip()]
    values: List[ShadowPromptSpec] = []
    for owner, record in enumerate(records):
        case_id = int(record["case_id"])
        subject = str(record["subject"])
        relation_id = str(record["relation_id"])
        _noun(relation_id)
        seen_prompts: set[str] = set()
        attempt = 0
        while len(seen_prompts) < variants:
            family = attempt
            scaffold = scaffolds[attempt % len(scaffolds)]
            prompt = _render(scaffold, subject=subject, relation_id=relation_id)
            prefix = (
                prefixes[
                    (
                        case_id + attempt * 17
                        if not expanded
                        else case_id * 17 + attempt * 29
                    )
                    % len(prefixes)
                ]
                if prefixes
                and (
                    (not expanded and attempt % 2 == 1)
                    or (expanded and attempt >= len(scaffolds))
                )
                else None
            )
            prompt = _with_prefix(prompt, prefix)
            attempt += 1
            if prompt in seen_prompts:
                if attempt > variants * 16 + len(prefixes) * len(scaffolds):
                    raise ValueError(f"could not construct {variants} positive variants")
                continue
            seen_prompts.add(prompt)
            values.append(
                ShadowPromptSpec(
                    prompt=prompt,
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
    scaffold_banks = {
        "calibration": CALIBRATION_SCAFFOLDS,
        "heldout": HELDOUT_SCAFFOLDS,
        "v4_1_heldout": V4_1_DEVELOPMENT_SCAFFOLDS,
        "v4_1_development": V4_1_DEVELOPMENT_SCAFFOLDS,
        "v4_1_certification": V4_1_CERTIFICATION_SCAFFOLDS,
        "v4_2_development": V4_2_DEVELOPMENT_SCAFFOLDS,
    }
    if split not in scaffold_banks:
        raise ValueError("unknown development split")
    scaffolds = scaffold_banks[split]
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
        if not available:
            raise ValueError(f"no distractor relations remain for {subject!r}")
        case_id = int(record["case_id"])
        start = (case_id * 13 + owner * 7) % len(available)
        seen_prompts: set[str] = set()
        attempt = 0
        maximum_attempts = max(1024, int(variants_per_record) * 32)
        while len(seen_prompts) < int(variants_per_record):
            family = attempt
            distractor = available[(start + attempt * 11) % len(available)]
            scaffold = scaffolds[(attempt + owner) % len(scaffolds)]
            prompt = _render(scaffold, subject=subject, relation_id=distractor)
            prefix = (
                prefixes[
                    (
                        case_id + attempt * 19
                        if int(variants_per_record) <= 2
                        else case_id * 23 + attempt * 31
                    )
                    % len(prefixes)
                ]
                if prefixes and (attempt % 2 == 1 or attempt >= len(available))
                else None
            )
            prompt = _with_prefix(prompt, prefix)
            attempt += 1
            if prompt in seen_prompts:
                if attempt > maximum_attempts:
                    raise ValueError(
                        f"could not construct {variants_per_record} unique negatives "
                        f"for {subject!r}"
                    )
                continue
            seen_prompts.add(prompt)
            values.append(
                ShadowPromptSpec(
                    prompt=prompt,
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
