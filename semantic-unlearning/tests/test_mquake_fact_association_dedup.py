import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mquake_fact_association_embeddings as MFA


def _record(case_id, subject, relation, obj, prompt):
    return {
        "case_id": case_id,
        "mquake_case_id": case_id // 100,
        "source_index": case_id // 100,
        "rewrite_index": case_id % 100,
        "requested_rewrite": {
            "prompt": prompt,
            "subject": subject,
            "relation_id": relation,
            "target_true": {"str": obj},
        },
        "paraphrase_prompts": [],
        "neighborhood_prompts": [],
    }


def test_identical_subject_relation_object_records_share_one_vector():
    records = [
        _record(10000, "Italy", "P36", "Rome", "The capital of {} is"),
        _record(20000, "Italy", "P36", "Rome", "The capital of {} is"),
        _record(30000, "Italy", "P30", "Europe", "{} is located in the continent of"),
    ]

    facts, case_to_fact_id, diagnostics = MFA.build_association_facts(records)

    assert len(records) == 3
    assert len(facts) == 2
    assert diagnostics["duplicate_records_collapsed"] == 1
    assert diagnostics["duplicate_association_group_count"] == 1

    assert case_to_fact_id[10000] == case_to_fact_id[20000]
    assert case_to_fact_id[10000] != case_to_fact_id[30000]

    capital = next(
        fact for fact in facts
        if fact["relation"] == "P36"
    )
    assert capital["subject"] == "Italy"
    assert capital["object"] == "Rome"
    assert capital["atomic_occurrence_count"] == 2
    assert capital["occurrence_case_ids"] == [10000, 20000]


def test_duplicate_association_can_aggregate_multiple_allowed_direct_prompts():
    records = [
        _record(10000, "Italy", "P36", "Rome", "The capital of {} is"),
        _record(20000, "Italy", "P36", "Rome", "{} has the capital city"),
    ]

    facts, case_to_fact_id, diagnostics = MFA.build_association_facts(records)

    assert len(facts) == 1
    assert case_to_fact_id[10000] == case_to_fact_id[20000]
    assert facts[0]["canonical_prompts"] == [
        "The capital of Italy is",
        "Italy has the capital city",
    ]
    assert diagnostics["duplicate_records_collapsed"] == 1


def test_same_natural_address_with_different_objects_fails_closed():
    records = [
        _record(10000, "Italy", "P36", "Rome", "The capital of {} is"),
        _record(20000, "Italy", "P36", "Milan", "The capital of {} is"),
    ]

    with pytest.raises(ValueError, match="natural-address conflicts"):
        MFA.build_association_facts(records)
