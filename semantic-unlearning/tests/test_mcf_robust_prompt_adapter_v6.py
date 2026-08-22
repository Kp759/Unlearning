import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_dataset_adapter_relation_slot as A  # noqa: E402
import mcf_sure_h_then_genaware_lmhead_lora_v6 as T  # noqa: E402


def test_direct_relation_profile_can_mark_underspecified_prompt_direct_only():
    raw = json.dumps({
        "relation_label": "underspecified_appositive",
        "relation_description": "unclear property following the subject",
        "answer_types": ["ambiguous"],
        "augmentable": False,
        "ambiguity": "high",
        "reason": "prompt does not state a relation",
    })
    p = A.parse_direct_profile(raw)
    assert p["valid"] is True
    assert p["safe_to_augment"] is False


def test_relation_profile_allows_multiple_plausible_answer_types():
    raw = json.dumps({
        "relation_label": "founding_context",
        "relation_description": "the place or year in which an organization was founded",
        "answer_types": ["location", "date_or_year"],
        "augmentable": True,
        "ambiguity": "medium",
        "reason": "the preposition in is compatible with place or year",
    })
    p = A.parse_direct_profile(raw)
    assert p["safe_to_augment"] is True
    assert set(p["answer_types"]) == {"location", "date_or_year"}


def test_answer_type_gate_rejects_vehicle_for_manufacturer_slot():
    gate = A.answer_type_compatible(["organization", "person"], ["product_or_artifact"])
    assert gate["compatible"] is False
    assert gate["overlap"] == []


def test_answer_type_gate_accepts_location_when_direct_slot_is_ambiguous_place_or_year():
    gate = A.answer_type_compatible(["location", "date_or_year"], ["location"])
    assert gate["compatible"] is True
    assert gate["overlap"] == ["location"]


def test_candidate_relation_classifier_rejects_added_constraint_even_when_relation_matches():
    raw = json.dumps({
        "candidate_relation_label": "manufacturer",
        "candidate_answer_types": ["organization"],
        "same_relation": True,
        "adds_constraint_or_claim": True,
        "semantically_coherent": True,
        "reason": "adds Japanese automaker restriction",
    })
    p = A.parse_candidate_profile(raw)
    assert p["valid"] is True
    assert p["relation_pass"] is False


def test_candidate_relation_classifier_rejects_headquarters_for_founding_relation():
    raw = json.dumps({
        "candidate_relation_label": "headquarters_location",
        "candidate_answer_types": ["location"],
        "same_relation": False,
        "adds_constraint_or_claim": False,
        "semantically_coherent": True,
        "reason": "headquarters is not founding context",
    })
    p = A.parse_candidate_profile(raw)
    assert p["relation_pass"] is False


def _records():
    return [
        {
            "case_id": 10,
            "requested_rewrite": {
                "subject": "Toyota Tercel",
                "prompt": "{} is produced by",
                "target_true": {"str": "sensitive-a"},
                "target_new": {"str": "reference-a"},
            },
        },
        {
            "case_id": 11,
            "requested_rewrite": {
                "subject": "Rob Barrett",
                "prompt": "{}, the",
                "target_true": {"str": "sensitive-b"},
                "target_new": {"str": "reference-b"},
            },
        },
    ]


def _artifact():
    return {
        "schema_version": 1,
        "protocol": T.ARTIFACT_PROTOCOL,
        "adapter_protocol": T.ADAPTER_PROTOCOL,
        "seed": 1,
        "forget_num": 2,
        "min_surrogates": 3,
        "max_surrogates": 8,
        "generator": {
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "deterministic_wrapper_fallback_used": False,
        },
        "relation_slot_adapter": {
            "protocol": T.RELATION_PROTOCOL,
            "classifier_received_target_true": False,
            "classifier_received_target_new": False,
        },
        "semantic_validation": {
            "protocol": T.SEMANTIC_PROTOCOL,
            "validator_received_target_true": False,
            "validator_received_target_new": False,
            "validator_received_official_paraphrases": False,
        },
        "data_access": {
            "official_paraphrase_seen": 0,
            "official_neighborhood_seen": 0,
            "benchmark_retain_seen": 0,
            "official_PPL_seen": False,
        },
        "adapter_summary": {
            "records_with_robust_prompt_sets": 1,
            "records_direct_only": 1,
            "surrogate_prompts_total": 3,
        },
        "records": [
            {
                "case_id": 10,
                "sampled_position": 0,
                "subject": "Toyota Tercel",
                "direct_prompt": "Toyota Tercel is produced by",
                "augmentation_status": "robust_prompt_set",
                "surrogate_count": 3,
                "surrogate_prompts": [
                    "Toyota Tercel is manufactured by",
                    "The manufacturer of Toyota Tercel is",
                    "Who manufactures Toyota Tercel?",
                ],
            },
            {
                "case_id": 11,
                "sampled_position": 1,
                "subject": "Rob Barrett",
                "direct_prompt": "Rob Barrett, the",
                "augmentation_status": "direct_only",
                "surrogate_count": 0,
                "surrogate_prompts": [],
            },
        ],
    }


def test_v6_loader_accepts_variable_robust_prompt_sets_and_direct_only(tmp_path):
    p = tmp_path / "adapter.json"
    p.write_text(json.dumps(_artifact()), encoding="utf-8")
    data, prompts = T.load_surrogate_artifact(p, _records(), seed=1, forget_num=2)
    assert data["protocol"] == T.ARTIFACT_PROTOCOL
    assert len(prompts[0]) == 3
    assert prompts[1] == []


def test_v6_loader_rejects_one_or_two_surrogates(tmp_path):
    artifact = _artifact()
    artifact["records"][0]["surrogate_prompts"] = [
        "Toyota Tercel is manufactured by",
        "The manufacturer of Toyota Tercel is",
    ]
    artifact["records"][0]["surrogate_count"] = 2
    artifact["adapter_summary"]["surrogate_prompts_total"] = 2
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 3..8"):
        T.load_surrogate_artifact(p, _records(), seed=1, forget_num=2)


def test_v6_loader_rejects_answer_occurrence_introduced_by_surrogate(tmp_path):
    records = _records()
    records[0]["requested_rewrite"]["target_true"]["str"] = "Toyota"
    artifact = _artifact()
    # Direct already has Toyota once. This candidate adds a second occurrence.
    artifact["records"][0]["surrogate_prompts"][0] = "Toyota Tercel is manufactured by Toyota"
    p = tmp_path / "leak.json"
    p.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(RuntimeError, match="introduced an answer occurrence"):
        T.load_surrogate_artifact(p, records, seed=1, forget_num=2)
