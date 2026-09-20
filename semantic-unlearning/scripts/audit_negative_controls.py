#!/usr/bin/env python3
"""Step 1 diagnostic: how many negative controls are semantically invalid?

Reconstructs exactly the negatives the shipped builder produces for a trained
run, then classifies each one:

  same_subject_real     another PROTECTED relation for the same subject, taken
                        verbatim with no text surgery. These are the good ones
                        and the class the router most needs to reject.
  transplant_compatible a subject transplanted into another fact's template
                        where the two relations accept the same kind of
                        subject. Plausible, keep.
  transplant_permissive as above, but the target relation accepts three or
                        more domains, so it barely discriminates. Counted
                        apart because "compatible" overstates it.
  transplant_invalid    a type collision: "The manufacturer of <a person> is".
                        Grammatical, semantically incoherent, off-manifold.
                        These depress max cos(negative), inflate d, and leave
                        tau too permissive.

The headline number is the invalid fraction. It bounds how much of the
reported positive/negative separation is an artifact of bad controls rather
than a property of the representation, and it needs no GPU -- the audit is
pure text and metadata over the artifact you already trained.

Usage
-----
python -u scripts/audit_negative_controls.py \
  --artifact outputs/<run>/fact_association_embeddings.pt \
  --examples outputs/<run>/association_examples.json \
  --output-dir outputs/<run>/negative_audit
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import torch

from fact_association_router_v2 import (
    _negative_prompts_for_fact,
    _norm,
    prompt_map_from_examples,
)
from relation_subject_types import (
    PERMISSIVE_RELATIONS,
    compatible,
    coverage_report,
    incompatibility_reason,
    infer_subject_type,
)


def classify(fact, facts, prompt, positives_by_fact, prompt_owner):
    """Label one negative control by how it was constructed."""
    owner_index = prompt_owner.get(prompt)
    if owner_index is not None:
        other = facts[owner_index]
        if _norm(other["subject"]) == _norm(fact["subject"]):
            return "same_subject_real", other["relation"], None
    # Not a verbatim protected prompt for this subject, so it came from the
    # transplant branch. Recover the donor template by matching the residual
    # text after the subject was swapped in.
    donor = _donor_relation(fact, facts, prompt, positives_by_fact)
    if donor is None:
        return "transplant_unmatched", None, None
    reason = incompatibility_reason(fact["relation"], donor)
    if reason is not None:
        return "transplant_invalid", donor, reason
    if str(donor) in PERMISSIVE_RELATIONS:
        return "transplant_permissive", donor, None
    return "transplant_compatible", donor, None


def _donor_relation(fact, facts, prompt, positives_by_fact):
    """Which fact's template this transplanted negative was built from."""
    subject = str(fact["subject"])
    skeleton = _norm(prompt).replace(_norm(subject), "\x00")
    for other in facts:
        if other["id"] == fact["id"]:
            continue
        for candidate in positives_by_fact.get(other["id"], []):
            other_skeleton = _norm(candidate).replace(
                _norm(str(other["subject"])), "\x00"
            )
            if other_skeleton == skeleton:
                return other["relation"]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--examples", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--negative-count", type=int, default=12)
    args = parser.parse_args(argv)

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    facts = list(artifact["facts"])
    raw_examples = json.loads(Path(args.examples).read_text())

    class _Row:
        __slots__ = ("split", "prompt", "fact_id")

        def __init__(self, row):
            self.split = row["split"]
            self.prompt = row["prompt"]
            self.fact_id = row["fact_id"]

    examples = [_Row(row) for row in raw_examples]
    positives_by_fact = prompt_map_from_examples(examples, split="train")

    prompt_owner = {}
    index_of = {fact["id"]: position for position, fact in enumerate(facts)}
    for fact_id, prompts in positives_by_fact.items():
        for prompt in prompts:
            prompt_owner.setdefault(prompt, index_of[fact_id])

    rows = []
    per_fact = []
    for fact_index, fact in enumerate(facts):
        negatives = _negative_prompts_for_fact(
            fact_index, facts, positives_by_fact, int(args.negative_count)
        )
        tally = Counter()
        for prompt in negatives:
            kind, donor, reason = classify(
                fact, facts, prompt, positives_by_fact, prompt_owner
            )
            tally[kind] += 1
            rows.append({
                "fact_id": fact["id"],
                "subject": fact["subject"],
                "relation": fact["relation"],
                "subject_type": sorted(infer_subject_type(fact["relation"])),
                "negative_prompt": prompt,
                "donor_relation": donor,
                "classification": kind,
                "incompatibility_reason": reason,
            })
        per_fact.append({
            "fact_id": fact["id"],
            "relation": fact["relation"],
            "negative_count": len(negatives),
            **{kind: tally.get(kind, 0) for kind in (
                "same_subject_real",
                "transplant_compatible",
                "transplant_permissive",
                "transplant_invalid",
                "transplant_unmatched",
            )},
            "invalid_fraction": (
                tally.get("transplant_invalid", 0) / len(negatives)
                if negatives else 0.0
            ),
        })

    totals = Counter(row["classification"] for row in rows)
    count = len(rows) or 1
    worst = sorted(
        per_fact, key=lambda row: row["invalid_fraction"], reverse=True
    )[:10]
    collisions = Counter(
        f"{row['relation']} -> {row['donor_relation']}"
        for row in rows if row["classification"] == "transplant_invalid"
    )

    report = {
        "schema_version": "negative_control_audit_v1",
        "artifact": str(Path(args.artifact).resolve()),
        "fact_count": len(facts),
        "negative_count_per_fact": int(args.negative_count),
        "total_negatives": len(rows),
        "counts": dict(totals),
        "fractions": {k: v / count for k, v in totals.items()},
        "invalid_fraction": totals.get("transplant_invalid", 0) / count,
        "weak_fraction": (
            totals.get("transplant_invalid", 0)
            + totals.get("transplant_permissive", 0)
        ) / count,
        "relation_coverage": coverage_report([f["relation"] for f in facts]),
        "most_common_type_collisions": collisions.most_common(15),
        "worst_facts_by_invalid_fraction": worst,
        "per_fact": per_fact,
        "interpretation": (
            "invalid_fraction bounds how much of the reported positive/negative "
            "separation comes from off-manifold controls rather than from the "
            "representation. Because the shipped audit negatives come from this "
            "same builder, that bias is shared by the measurement and the thing "
            "measured, so it cannot be detected from the existing diagnostics."
        ),
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "negative_control_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    (output / "negative_control_rows.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "negative_audit_complete",
        "total_negatives": len(rows),
        "counts": dict(totals),
        "invalid_fraction": report["invalid_fraction"],
        "weak_fraction": report["weak_fraction"],
        "top_collisions": collisions.most_common(5),
        "output": str(output / "negative_control_audit.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
