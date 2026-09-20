"""Step 1 fix: a corrected train/development split and type-checked negatives.

Two defects in the shipped fitting data, both free to fix and both upstream of
every number the router produces.

Defect 1: the split is inverted.
`build_forget_examples` puts the generic `{relation}`/`{subject}` scaffolds in
BOTH splits and keeps the syntactically varied per-relation templates
(`RELATION_ALTERNATE_TEMPLATES`, plus their context-prefixed forms) entirely
in train. So the model fits on the diverse phrasings and generalization is
measured on the uniform ones. Worse, TRAIN_SCAFFOLDS and DEVELOPMENT_SCAFFOLDS
were written by the same hand in the same register: measured on token overlap,
development sits no further from train (0.125) than train sits from itself
(0.134). That is same-generator, different-sample, which is not a held-out set.

`rebalanced_split` moves the relation-alternates to development and keeps the
scaffolds in train. Development then contains genuinely different syntax and
dev route recall becomes a number that can fail. No new data is generated --
this is a relabelling of prompts the pipeline already builds.

Defect 2: negative controls are type-unchecked.
`_negative_prompts_for_fact` transplants a subject into another fact's
template regardless of whether the two relations accept the same kind of
subject, producing off-manifold controls like "The manufacturer of Danielle
Darrieux is". `type_checked_negatives` keeps the builder's ordering and count
but skips type collisions, preferring real same-subject competitors first and
falling back to compatible transplants.

Both changes make the router harder to satisfy, so expect dev recall to drop
and tau to tighten. That is the point: the previous numbers were measuring the
fitting rule rather than the representation.
"""
from __future__ import annotations

from collections import defaultdict

from fact_association_router_v2 import _norm, _replace_subject
from relation_subject_types import (
    PERMISSIVE_RELATIONS,
    compatible,
    incompatibility_reason,
)


# Prompt roles as `build_forget_examples` labels them. The relation-alternate
# families are the syntactically distinct ones and belong in development.
SCAFFOLD_ROLE_PREFIX = "authored_"
CANONICAL_ROLE = "canonical_rewrite"
ALTERNATE_ROLE_PREFIXES = ("relation_alternate_", "context_relation_alternate_")


def _role_family(role):
    role = str(role)
    if role == CANONICAL_ROLE:
        return "canonical"
    if role.startswith(SCAFFOLD_ROLE_PREFIX):
        return "scaffold"
    for prefix in ALTERNATE_ROLE_PREFIXES:
        if role.startswith(prefix):
            return "relation_alternate"
    return "other"


def rebalanced_split(examples, *, keep_canonical_in_train=True):
    """Reassign splits so development holds the syntactically distinct prompts.

    train        canonical rewrite + all generic scaffolds
    development  relation-alternate templates and their context-prefixed forms

    Returns (reassigned_examples, diagnostics). Input objects are not mutated;
    each example is returned as a dict with an updated `split`.
    """
    reassigned = []
    moved = defaultdict(int)
    for example in examples:
        row = dict(example) if not isinstance(example, dict) else dict(example)
        family = _role_family(row.get("role", ""))
        if family == "relation_alternate":
            target = "development"
        elif family == "canonical":
            target = "train" if keep_canonical_in_train else "development"
        elif family == "scaffold":
            target = "train"
        else:
            target = str(row.get("split", "train"))
        if target != str(row.get("split", "")):
            moved[f"{row.get('split')}->{target}:{family}"] += 1
        row["split"] = target
        row["split_rule"] = "rebalanced_v2"
        reassigned.append(row)

    counts = defaultdict(lambda: defaultdict(int))
    for row in reassigned:
        counts[row["split"]][_role_family(row.get("role", ""))] += 1
    per_fact = defaultdict(lambda: defaultdict(int))
    for row in reassigned:
        per_fact[row.get("fact_id")][row["split"]] += 1
    empty = [
        fact_id for fact_id, splits in per_fact.items()
        if splits.get("train", 0) == 0 or splits.get("development", 0) == 0
    ]
    diagnostics = {
        "rule": "rebalanced_v2",
        "description": (
            "development holds RELATION_ALTERNATE_TEMPLATES and their "
            "context-prefixed forms; train holds the canonical rewrite and the "
            "generic scaffolds"
        ),
        "moved": dict(moved),
        "counts_by_split": {k: dict(v) for k, v in counts.items()},
        "facts_missing_a_split": empty,
        "rationale": (
            "The shipped split measured generalization on the least diverse "
            "prompt family. Development prompts here differ from train in "
            "syntax, not only in wording."
        ),
    }
    if empty:
        raise ValueError(
            f"Rebalanced split left {len(empty)} facts without both splits: "
            f"{empty[:5]}"
        )
    return reassigned, diagnostics


def type_checked_negatives(
    fact_index,
    facts,
    positive_prompts_by_fact,
    negative_count,
    *,
    allow_permissive=True,
    strict=False,
):
    """Negatives with the shipped ordering, minus semantic type collisions.

    Priority is unchanged: real same-subject competing relations first (no text
    surgery, so always valid), then transplants. The only change is that a
    transplant whose donor relation cannot take this subject's type is skipped.

    allow_permissive  keep transplants into relations with three or more
                      candidate domains (P17, P276, P495, P127, P138). They
                      discriminate weakly but are not wrong.
    strict            raise if the budget cannot be filled, instead of
                      returning fewer negatives. Default False so a fact with
                      few compatible donors degrades rather than failing the run.
    """
    fact = facts[fact_index]
    subject = str(fact["subject"])
    subject_key = _norm(subject)
    relation = str(fact.get("relation", ""))
    own_positive = set(positive_prompts_by_fact[fact["id"]])
    negatives = []
    provenance = []

    for other_index, other in enumerate(facts):
        if other_index == fact_index or _norm(other["subject"]) != subject_key:
            continue
        for prompt in positive_prompts_by_fact[other["id"]]:
            if prompt not in own_positive and prompt not in negatives:
                negatives.append(prompt)
                provenance.append({
                    "prompt": prompt,
                    "kind": "same_subject_real",
                    "donor_relation": other.get("relation"),
                })

    skipped = []
    for offset in range(1, len(facts) + 1):
        if len(negatives) >= int(negative_count):
            break
        other_index = (fact_index + offset) % len(facts)
        if other_index == fact_index:
            continue
        other = facts[other_index]
        donor_relation = str(other.get("relation", ""))
        if not compatible(relation, donor_relation):
            skipped.append({
                "donor_relation": donor_relation,
                "reason": incompatibility_reason(relation, donor_relation),
            })
            continue
        if not allow_permissive and donor_relation in PERMISSIVE_RELATIONS:
            skipped.append({
                "donor_relation": donor_relation,
                "reason": "permissive donor relation excluded",
            })
            continue
        for prompt in positive_prompts_by_fact[other["id"]]:
            transplanted = _replace_subject(prompt, other["subject"], subject)
            if (
                transplanted
                and transplanted not in own_positive
                and transplanted not in negatives
            ):
                negatives.append(transplanted)
                provenance.append({
                    "prompt": transplanted,
                    "kind": (
                        "transplant_permissive"
                        if donor_relation in PERMISSIVE_RELATIONS
                        else "transplant_compatible"
                    ),
                    "donor_relation": donor_relation,
                })
            if len(negatives) >= int(negative_count):
                break

    if not negatives:
        raise ValueError(
            f"No type-compatible negative controls for {fact['id']} "
            f"(relation {relation}). Every donor collided on subject type; "
            f"supply Wikidata same-subject relations for this fact."
        )
    if strict and len(negatives) < int(negative_count):
        raise ValueError(
            f"Only {len(negatives)}/{negative_count} type-compatible negatives "
            f"for {fact['id']}"
        )
    return negatives[: int(negative_count)], {
        "fact_id": fact["id"],
        "relation": relation,
        "requested": int(negative_count),
        "produced": min(len(negatives), int(negative_count)),
        "provenance": provenance[: int(negative_count)],
        "skipped_type_collisions": skipped,
    }


def build_type_checked_negative_map(
    facts, positive_prompts_by_fact, negative_count, **kwargs
):
    """Type-checked negatives for every fact, plus a run-level report."""
    negatives, reports = {}, []
    for fact_index, fact in enumerate(facts):
        prompts, report = type_checked_negatives(
            fact_index, facts, positive_prompts_by_fact, negative_count, **kwargs
        )
        negatives[fact["id"]] = prompts
        reports.append(report)
    shortfall = [r for r in reports if r["produced"] < r["requested"]]
    return negatives, {
        "negative_count_requested": int(negative_count),
        "facts": len(facts),
        "facts_below_budget": len(shortfall),
        "mean_produced": (
            sum(r["produced"] for r in reports) / len(reports) if reports else 0.0
        ),
        "total_type_collisions_skipped": sum(
            len(r["skipped_type_collisions"]) for r in reports
        ),
        "per_fact": reports,
        "note": (
            "Facts below budget have few type-compatible donors in the bank. "
            "Fill them from Wikidata same-subject relations rather than "
            "relaxing the type check."
        ),
    }
