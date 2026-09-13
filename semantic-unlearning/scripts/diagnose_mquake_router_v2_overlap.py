#!/usr/bin/env python3
"""Diagnose MQuAKE Router V2 retain activations for atomic overlap/conflict.

CPU-only. No model forward pass is performed. The script reloads the exact
seed-1 forget/retain records, reads the saved official evaluation, and
classifies retain activations relative to the protected forget associations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from mquake_fact_association_embeddings import (
    association_key_from_record,
    normalized,
)
import mquake_zero_unlearn_official_eval as mquake


def _sr(record):
    rr = record["requested_rewrite"]
    return normalized(rr["subject"]), str(rr.get("relation_id"))


def _subject(record):
    return normalized(record["requested_rewrite"]["subject"])


def _fact_key(fact):
    if fact.get("association_key") is not None:
        return str(fact["association_key"])
    return (
        f"{normalized(fact['subject'])}\t"
        f"{str(fact.get('relation'))}\t"
        f"{normalized(fact['object'])}"
    )


def _fact_sr(fact):
    return normalized(fact["subject"]), str(fact.get("relation"))


def _record_category(record, forget_keys, forget_sr, forget_subjects):
    key = association_key_from_record(record)
    sr = _sr(record)
    subject = _subject(record)
    if key in forget_keys:
        return "exact_forget_association"
    if sr in forget_sr:
        return "same_subject_relation_different_object"
    if subject in forget_subjects:
        return "same_subject_different_relation"
    return "subject_disjoint"


def _route_pair_category(record, fact):
    record_key = association_key_from_record(record)
    fact_key = _fact_key(fact)
    record_subject, record_relation = _sr(record)
    fact_subject, fact_relation = _fact_sr(fact)
    if record_key == fact_key:
        return "exact_protected_association"
    if record_subject == fact_subject and record_relation == fact_relation:
        return "same_subject_relation_different_object"
    if record_subject == fact_subject:
        return "same_subject_different_relation"

    # Subject eligibility is token based, so a shorter protected subject can be
    # embedded in a longer retain subject (or vice versa).
    rs = record_subject.casefold()
    fs = fact_subject.casefold()
    if rs and fs and (rs in fs or fs in rs):
        return "lexical_subject_overlap"
    return "unexpected_nonmatching_subject"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        default="outputs/mquake_fact_assoc_router_v2_seed1",
    )
    p.add_argument(
        "--mquake-path",
        default="data/MQuAKE-CF-3k-v2.json",
    )
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    manifest = json.loads(
        (run_dir / "association_manifest.json").read_text()
    )
    evaluation = json.loads(
        (run_dir / "official_mquake_eval.json").read_text()
    )
    facts = list(manifest["facts"])

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        Path(manifest["model_path"]).resolve(),
        use_fast=True,
        local_files_only=args.local_files_only,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget_records, retain_records = mquake.load_official_eval_records(
        Path(args.mquake_path).resolve(),
        tok,
        forget_num=50,
        retain_num=1000,
        seed=1,
    )
    retain_by_case = {
        int(record["case_id"]): record
        for record in retain_records
    }

    forget_keys = {
        association_key_from_record(record)
        for record in forget_records
    }
    forget_sr = {_sr(record) for record in forget_records}
    forget_subjects = {_subject(record) for record in forget_records}

    retain_record_categories = Counter(
        _record_category(
            record, forget_keys, forget_sr, forget_subjects
        )
        for record in retain_records
    )

    raw = list(evaluation["retain_raw"])
    active_raw = [
        row for row in raw
        if bool(row.get("association_route_active"))
    ]

    # Count unique cases per prompt type so multi-token answers do not inflate
    # the overlap diagnosis.
    active_cases_by_prompt_type = {}
    active_case_categories_by_prompt_type = {}
    for prompt_type in ("rewrite", "atomic_gen"):
        case_ids = sorted({
            int(row["case_id"])
            for row in active_raw
            if row["prompt_type"] == prompt_type
        })
        active_cases_by_prompt_type[prompt_type] = len(case_ids)
        active_case_categories_by_prompt_type[prompt_type] = dict(
            Counter(
                _record_category(
                    retain_by_case[case_id],
                    forget_keys,
                    forget_sr,
                    forget_subjects,
                )
                for case_id in case_ids
            )
        )

    # Diagnose which protected fact each active route points at. Deduplicate by
    # (retain case, prompt type, protected row) to avoid answer-token inflation.
    unique_route_pairs = set()
    token_route_pair_counts = Counter()
    row_token_fires = Counter()
    for row in active_raw:
        case_id = int(row["case_id"])
        prompt_type = str(row["prompt_type"])
        for fact_row in row.get("active_fact_rows", []):
            fact_row = int(fact_row)
            unique_route_pairs.add((case_id, prompt_type, fact_row))
            category = _route_pair_category(
                retain_by_case[case_id],
                facts[fact_row],
            )
            token_route_pair_counts[category] += 1
            row_token_fires[fact_row] += 1

    unique_pair_categories = Counter()
    for case_id, prompt_type, fact_row in sorted(unique_route_pairs):
        unique_pair_categories[
            _route_pair_category(
                retain_by_case[case_id],
                facts[fact_row],
            )
        ] += 1

    top_route_magnets = []
    for fact_row, fires in row_token_fires.most_common(30):
        fact = facts[fact_row]
        top_route_magnets.append({
            "row": int(fact_row),
            "token_decision_fires": int(fires),
            "fact_id": fact.get("id"),
            "subject": fact.get("subject"),
            "relation": fact.get("relation"),
            "object": fact.get("object"),
            "association_key": fact.get("association_key"),
        })

    exact_retain_records = [
        {
            "case_id": int(record["case_id"]),
            "association_key": association_key_from_record(record),
            "subject": record["requested_rewrite"]["subject"],
            "relation": str(
                record["requested_rewrite"].get("relation_id")
            ),
            "object": record["requested_rewrite"]["target_true"]["str"],
        }
        for record in retain_records
        if association_key_from_record(record) in forget_keys
    ]

    result = {
        "dataset": "MQuAKE-CF-3k-v2",
        "seed": 1,
        "forget_atomic_records": len(forget_records),
        "protected_unique_associations": len(facts),
        "retain_atomic_records": len(retain_records),
        "retain_record_overlap_categories": dict(retain_record_categories),
        "exact_forget_associations_reappearing_in_retain_records": (
            len(exact_retain_records)
        ),
        "active_token_decisions": len(active_raw),
        "active_unique_cases_by_prompt_type": active_cases_by_prompt_type,
        "active_case_overlap_categories_by_prompt_type": (
            active_case_categories_by_prompt_type
        ),
        "unique_active_case_prompt_row_pairs": len(unique_route_pairs),
        "unique_route_pair_categories": dict(unique_pair_categories),
        "token_decision_route_pair_categories": dict(
            token_route_pair_counts
        ),
        "top_route_magnets": top_route_magnets,
        "exact_overlap_examples": exact_retain_records[:50],
        "interpretation_guide": {
            "exact_protected_association": (
                "retain record is the same normalized "
                "(subject, relation_id, target_true) fact as a protected fact"
            ),
            "same_subject_relation_different_object": (
                "natural subject+relation address is shared but object differs; "
                "a router that cannot observe the object cannot cleanly separate these"
            ),
            "same_subject_different_relation": (
                "genuine relation-selectivity error for the contextual router"
            ),
            "lexical_subject_overlap": (
                "subject token eligibility can match nested/overlapping entity surfaces"
            ),
            "unexpected_nonmatching_subject": (
                "should be investigated as a subject-mask or provenance issue"
            ),
        },
    }

    out = (
        Path(args.out).resolve()
        if args.out
        else run_dir / "router_v2_overlap_diagnostic.json"
    )
    out.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
