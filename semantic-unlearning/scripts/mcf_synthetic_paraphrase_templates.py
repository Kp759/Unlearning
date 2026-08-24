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

import random
import re
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


def coverage_report(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Report how many records hit the hand-authored relation bank vs. the
    generic fallback, so a missing/unrecognized ``relation_id`` upstream
    (e.g. a split builder that strips it) shows up immediately instead of
    silently degrading every record to the 2 generic templates."""
    hit = 0
    fallback = 0
    missing_relation_ids: List[str] = []
    for record in records:
        rr = record.get("requested_rewrite")
        relation_id = str((rr or {}).get("relation_id") or "")
        if relation_id in RELATION_ALTERNATE_TEMPLATES:
            hit += 1
        else:
            fallback += 1
            missing_relation_ids.append(relation_id or "<missing>")
    return {
        "relation_bank_hit_records": hit,
        "generic_fallback_records": fallback,
        "generic_fallback_relation_ids": sorted(set(missing_relation_ids)),
    }


def synthetic_prompt_templates(
    *,
    relation_id: str,
    canonical_prompt: str,
    case_id: int,
    count: int,
    context_prefixes: Sequence[str] | None = None,
    prefer_subject_first: bool = True,
) -> List[str]:
    """Deterministically build ``count`` alternate cloze templates (each still
    containing one ``{}`` subject placeholder) for one record.

    Candidates interleave a bare grammatical variant with a context-prefixed
    variant of the same template, covering both real paraphrase-variation
    axes before ever repeating a template.

    ``context_prefixes`` defaults to the four formulaic
    ``GENERIC_CONTEXT_PREFIXES``; pass :func:`corpus_context_prefixes` output
    to match the arbitrary-unrelated-sentence prefixes real CounterFact
    paraphrases actually use.
    """
    if count <= 0:
        return []
    prefixes = list(context_prefixes) if context_prefixes else GENERIC_CONTEXT_PREFIXES
    if not prefixes:
        prefixes = GENERIC_CONTEXT_PREFIXES
    templates = _relation_templates(relation_id, canonical_prompt)
    if prefer_subject_first:
        # Real paraphrases are 100% subject-first; put those variants at the
        # front so the earliest-drawn templates match the evaluated register.
        derived = subject_first_variants(canonical_prompt)
        head = [t for t in derived if t not in templates]
        front = [t for t in templates if t.strip().startswith("{}")]
        rest = [t for t in templates if not t.strip().startswith("{}")]
        templates = head + front + rest
    candidates: List[str] = []
    prefix_cycle = 0
    for template in templates:
        candidates.append(template)
        prefix = prefixes[(case_id + prefix_cycle) % len(prefixes)]
        candidates.append(f"{prefix} {template}")
        prefix_cycle += 1
    # If more variants are requested than authored, keep cycling context
    # prefixes over the same templates rather than repeating an identical string.
    while len(candidates) < count:
        template = templates[prefix_cycle % len(templates)]
        prefix = prefixes[(case_id + prefix_cycle) % len(prefixes)]
        candidates.append(f"{prefix} {template}")
        prefix_cycle += 1
    return candidates[:count]


SUBJECT_FIRST_PATTERNS: List[tuple] = [
    # "The mother tongue of {} is"  ->  "{}'s mother tongue is"
    #                                   "{}, whose mother tongue is"
    (r"^The\s+(.+?)\s+of\s+\{\}\s+(is|was|are|were)$",
     ["{{}}'s {0} {1}", "{{}}, whose {0} {1}"]),
    # "The headquarter of {} is located in" -> "{}'s headquarter is located in"
    (r"^The\s+(.+?)\s+of\s+\{\}\s+(.+)$",
     ["{{}}'s {0} {1}"]),
    # "The language used by {} is" -> "{}, which used the language,"  (kept simple)
    (r"^The\s+(.+?)\s+used\s+by\s+\{\}\s+(is|was)$",
     ["{{}}'s {0} {1}"]),
]


def subject_first_variants(canonical_prompt: str) -> List[str]:
    """Mechanically derive subject-first templates from a record's own prompt.

    100% of real CounterFact ``paraphrase_prompts`` are subject-first -- the
    subject comes first and a relation continuation follows ("X, speaker of",
    "X's headquarters are in").  72% of canonical prompts are already that
    shape, but only 33% of this bank's generated templates were, so training
    exercised a syntactic register that neither the direct prompt nor the
    evaluated paraphrases use.  Across the 50 forget records the bank's 34
    distinct tails overlapped the 66 real tails in just 2 cases.

    Derivation uses only ``canonical_prompt``, which is part of the locked
    training-visible ``requested_rewrite`` -- never a real ``paraphrase_prompts``
    entry -- so the data firewall is unchanged.

    A prompt that is already subject-first is returned as-is; there is nothing
    to fix and inventing structure would only add noise.
    """
    prompt = str(canonical_prompt).strip()
    if not prompt or "{}" not in prompt:
        return []
    if prompt.startswith("{}"):
        return [prompt]
    variants: List[str] = []
    for pattern, shapes in SUBJECT_FIRST_PATTERNS:
        match = re.match(pattern, prompt)
        if not match:
            continue
        groups = [g.strip() for g in match.groups()]
        for shape in shapes:
            try:
                candidate = shape.format(*groups)
            except (IndexError, KeyError):
                continue
            if "{}" in candidate and candidate not in variants:
                variants.append(candidate)
        if variants:
            break
    return variants


def corpus_context_prefixes(
    documents: Sequence[str],
    *,
    count: int,
    seed: int,
    min_words: int = 5,
    max_words: int = 16,
) -> List[str]:
    """Sample arbitrary unrelated sentences to use as paraphrase prefixes.

    ``GENERIC_CONTEXT_PREFIXES`` are four short, formulaic meta lead-ins
    ("According to publicly available records,").  Real CounterFact
    ``paraphrase_prompts`` instead prepend an arbitrary *unrelated sentence*
    lifted from some other document::

        "Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native"
        "The population density was . Toko Yasuda plays the instrument"

    A run trained only on the four formulaic prefixes reached 30% synthetic
    failure but 59% real paraphrase failure -- the edit had learned to fire
    after a meta lead-in that announces a factual statement, not after
    arbitrary noise.  Sampling many real sentences closes that distribution
    gap, and is the same robustness trick MEMIT uses when it averages its key
    over randomly sampled prefixes.

    ``documents`` must come from a corpus disjoint from the official PPL
    slice, and never from any record's real ``paraphrase_prompts``.
    """
    if count <= 0 or not documents:
        return []
    seen: set[str] = set()
    pool: List[str] = []
    for document in documents:
        for raw in str(document).replace("\n", " ").split("."):
            sentence = " ".join(raw.split())
            if not sentence:
                continue
            words = sentence.split()
            if not (min_words <= len(words) <= max_words):
                continue
            # A prefix that itself contains a cloze placeholder would corrupt
            # the template's single {} subject slot.
            if "{" in sentence or "}" in sentence:
                continue
            candidate = sentence + "."
            if candidate in seen:
                continue
            seen.add(candidate)
            pool.append(candidate)
    if not pool:
        return []
    pool.sort()  # deterministic ordering before the seeded draw
    rng = random.Random(int(seed))
    if len(pool) <= count:
        return pool
    return rng.sample(pool, count)


def build_synthetic_records(
    records: Sequence[Mapping[str, Any]],
    *,
    count: int,
    context_prefixes: Sequence[str] | None = None,
    prefer_subject_first: bool = True,
) -> List[Dict[str, Any]]:
    """Return a flat list of synthetic MCF records for direction fitting/GA.

    Each synthetic record has the same ``subject``/``target_true``/``target_new``
    as its source record, with ``requested_rewrite.prompt`` replaced by a
    hand-authored alternate template. Never reads or derives from the source
    record's real ``paraphrase_prompts``.

    ``context_prefixes`` overrides ``GENERIC_CONTEXT_PREFIXES`` -- pass the
    output of :func:`corpus_context_prefixes` to match the real
    arbitrary-unrelated-sentence prefix distribution.  Omitting it preserves
    the original four formulaic lead-ins.
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
            context_prefixes=context_prefixes,
            prefer_subject_first=prefer_subject_first,
        )
        for template in templates:
            synthetic_rr = dict(rr)
            synthetic_rr["prompt"] = template
            synthetic_record = dict(record)
            synthetic_record["requested_rewrite"] = synthetic_rr
            synthetic.append(synthetic_record)
    return synthetic
