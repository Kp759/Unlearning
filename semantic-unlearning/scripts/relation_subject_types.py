"""Subject-type inventory for the 34 MCF relations, for negative-control hygiene.

`_negative_prompts_for_fact` builds negatives by transplanting a fact's subject
into another fact's prompt template. When the two relations expect different
kinds of subject, the result is grammatical but semantically incoherent --
"The manufacturer of Danielle Darrieux is", "The native language of the Nokia
3310 is". Such a prompt sits far from everything in representation space, so
max cosine to the negative bank is systematically too low, the margin d too
high, and the fitted tau too permissive.

The failure is invisible under the current evaluation because the audit
negatives come from the same builder, so the bias is shared by the measurement
and the thing measured.

Types are assigned from what the relation's DOMAIN must be, not from what any
particular record happens to contain. Several relations accept more than one
domain and are typed as a set; compatibility is non-empty intersection, which
is deliberately permissive -- the goal is to catch person/product collisions,
not to adjudicate whether a band counts as an organization.

This is a hand-built table over a closed 34-relation inventory, not an
ontology. Anything outside it is reported as UNKNOWN and treated as compatible
so the pipeline degrades open rather than silently dropping controls.
"""
from __future__ import annotations


PERSON = "person"
ORGANIZATION = "organization"
CREATIVE_WORK = "creative_work"
PLACE = "place"
PRODUCT = "product"
UNKNOWN = "unknown"

ALL_TYPES = (PERSON, ORGANIZATION, CREATIVE_WORK, PLACE, PRODUCT)


# Domain of each relation: which kinds of subject the relation can take.
RELATION_SUBJECT_TYPES = {
    # People
    "P27":   {PERSON},                      # country of citizenship
    "P413":  {PERSON},                      # playing position
    "P1412": {PERSON},                      # language used for writing
    "P103":  {PERSON},                      # native language
    "P106":  {PERSON},                      # profession
    "P20":   {PERSON},                      # place of death
    "P19":   {PERSON},                      # place of birth
    "P1303": {PERSON},                      # musical instrument
    "P101":  {PERSON},                      # field of work
    "P39":   {PERSON},                      # position held
    "P140":  {PERSON},                      # religion
    "P108":  {PERSON},                      # employer
    "P641":  {PERSON},                      # sport
    "P937":  {PERSON},                      # work location

    # Organizations
    "P159":  {ORGANIZATION},                # headquarters location
    "P740":  {ORGANIZATION},                # founding location
    "P463":  {PERSON, ORGANIZATION},        # member of organization

    # Creative works and media
    "P136":  {CREATIVE_WORK, PERSON},       # genre (work or artist)
    "P449":  {CREATIVE_WORK},               # original broadcast network
    "P364":  {CREATIVE_WORK},               # original language
    "P407":  {CREATIVE_WORK},               # language of the work
    "P264":  {CREATIVE_WORK, PERSON},       # record label (album or artist)

    # Products and software
    "P176":  {PRODUCT},                     # manufacturer
    "P178":  {PRODUCT, CREATIVE_WORK},      # developer (software)

    # Places
    "P30":   {PLACE},                       # continent
    "P131":  {PLACE},                       # administrative region
    "P37":   {PLACE},                       # official language
    "P36":   {PLACE},                       # capital city
    "P190":  {PLACE},                       # sister city

    # Multi-domain: these genuinely accept several subject kinds, so they are
    # weak discriminators and are typed permissively on purpose.
    "P17":   {PLACE, ORGANIZATION, PRODUCT, CREATIVE_WORK},   # country
    "P276":  {PLACE, ORGANIZATION, CREATIVE_WORK, PRODUCT},   # location
    "P495":  {CREATIVE_WORK, PRODUCT, ORGANIZATION},          # country of origin
    "P127":  {ORGANIZATION, PRODUCT, CREATIVE_WORK, PLACE},   # owner
    "P138":  {PERSON, ORGANIZATION, PLACE, CREATIVE_WORK, PRODUCT},  # namesake
}

# Relations whose domain is broad enough that they discriminate little. A
# negative built from one of these is weak evidence either way, so the audit
# counts them separately rather than calling them valid.
PERMISSIVE_RELATIONS = frozenset(
    relation for relation, types in RELATION_SUBJECT_TYPES.items()
    if len(types) >= 3
)


def subject_types(relation_id):
    """Types the relation's subject may take; empty set if unknown."""
    return set(RELATION_SUBJECT_TYPES.get(str(relation_id), ()))


def infer_subject_type(relation_id):
    """Infer a fact's subject type from its own gold relation.

    A fact carrying P19 (place of birth) has a person subject; that is what
    makes the fact well-formed in the first place. For permissive relations
    this returns the full candidate set and the caller should treat the
    inference as weak.
    """
    return subject_types(relation_id) or {UNKNOWN}


def compatible(subject_relation, target_relation):
    """Can this subject plausibly appear in the target relation's template?

    Compatibility is non-empty intersection of candidate domains. Unknown
    relations are treated as compatible, so the check degrades open.
    """
    left = subject_types(subject_relation)
    right = subject_types(target_relation)
    if not left or not right:
        return True
    return bool(left & right)


def incompatibility_reason(subject_relation, target_relation):
    """Human-readable reason, or None when the pair is compatible."""
    if compatible(subject_relation, target_relation):
        return None
    left = sorted(subject_types(subject_relation))
    right = sorted(subject_types(target_relation))
    return (
        f"subject of {subject_relation} is {'/'.join(left)}; "
        f"{target_relation} expects {'/'.join(right)}"
    )


def coverage_report(relations):
    """Which of the supplied relation ids this table knows about."""
    known = [r for r in relations if str(r) in RELATION_SUBJECT_TYPES]
    unknown = sorted({str(r) for r in relations if str(r) not in RELATION_SUBJECT_TYPES})
    return {
        "relation_count": len(set(map(str, relations))),
        "known": len(set(known)),
        "unknown": unknown,
        "permissive": sorted(
            {str(r) for r in relations if str(r) in PERMISSIVE_RELATIONS}
        ),
        "table_size": len(RELATION_SUBJECT_TYPES),
    }
