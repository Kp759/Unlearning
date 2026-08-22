import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_surrogate_semantic_validator_v3 as V  # noqa: E402
import mcf_sure_h_then_genaware_lmhead_lora_v5 as T  # noqa: E402


def test_relation_slot_booleans_override_inconsistent_negative_verdict():
    raw = json.dumps({
        "same_slot_relation": True,
        "same_answer_type": True,
        "no_added_factual_constraints": True,
        "semantically_coherent": True,
        "verdict": "NOT_EQUIVALENT",
        "reason": "inconsistent verdict string",
    })
    r = V.parse_relation_slot(raw)
    assert r["parse_ok"] is True
    assert r["accepted"] is True
    assert r["criteria_ok"] is True
    assert r["verdict_consistent"] is False


def test_adversarial_booleans_override_inconsistent_unsafe_verdict():
    raw = json.dumps({
        "relation_shift": False,
        "answer_type_shift": False,
        "added_constraint_or_claim": False,
        "semantic_malformation": False,
        "weak_wrapper": False,
        "verdict": "UNSAFE",
        "reason": "inconsistent verdict string",
    })
    r = V.parse_adversarial(raw)
    assert r["parse_ok"] is True
    assert r["accepted"] is True
    assert r["criteria_ok"] is True
    assert r["verdict_consistent"] is False


def test_relation_slot_still_rejects_failed_structured_criterion():
    raw = json.dumps({
        "same_slot_relation": False,
        "same_answer_type": True,
        "no_added_factual_constraints": True,
        "semantically_coherent": True,
        "verdict": "EQUIVALENT",
        "reason": "bad relation despite positive verdict",
    })
    r = V.parse_relation_slot(raw)
    assert r["accepted"] is False
    assert r["criteria_ok"] is False


def test_adversarial_still_rejects_structured_problem_flag():
    raw = json.dumps({
        "relation_shift": True,
        "answer_type_shift": False,
        "added_constraint_or_claim": False,
        "semantic_malformation": False,
        "weak_wrapper": False,
        "verdict": "SAFE",
        "reason": "bad relation despite positive verdict",
    })
    r = V.parse_adversarial(raw)
    assert r["accepted"] is False
    assert r["criteria_ok"] is False


def _records():
    return [{
        "case_id": 7,
        "requested_rewrite": {
            "subject": "Ada",
            "prompt": "{} worked as",
            "target_true": {"str": "sensitive"},
            "target_new": {"str": "reference"},
        },
    }]


def _artifact(builder_protocol=T.SEMANTIC_BUILDER_PROTOCOL):
    return {
        "schema_version": 1,
        "protocol": "mcf_locked_direct_only_surrogate_paraphrases_v1",
        "builder_protocol": builder_protocol,
        "seed": 1,
        "forget_num": 1,
        "surrogates_per_record": 2,
        "generator": {
            "deterministic_wrapper_fallback_used": False,
            "generator_received_target_true": False,
            "generator_received_target_new": False,
        },
        "semantic_validation": {
            "enabled": True,
            "protocol": V.VALIDATOR_PROTOCOL,
            "dual_pass_consensus": True,
            "required_for_every_surrogate": True,
            "completion_fragments_explicitly_allowed": True,
            "structured_booleans_authoritative": True,
            "free_form_verdict_is_audit_only": True,
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
        "records": [{
            "case_id": 7,
            "sampled_position": 0,
            "subject": "Ada",
            "direct_prompt": "Ada worked as",
            "surrogate_prompts": ["Ada's occupation was", "What occupation did Ada have?"],
        }],
    }


def test_v5_trainer_accepts_v5_boolean_consensus_artifact(tmp_path):
    p = tmp_path / "v5.json"
    p.write_text(json.dumps(_artifact()), encoding="utf-8")
    data, prompts = T.load_surrogate_artifact(p, _records(), seed=1, forget_num=1)
    assert data["builder_protocol"] == T.SEMANTIC_BUILDER_PROTOCOL
    assert prompts == [["Ada's occupation was", "What occupation did Ada have?"]]


def test_v5_trainer_rejects_v4_artifact(tmp_path):
    p = tmp_path / "v4.json"
    p.write_text(
        json.dumps(_artifact("mcf_locked_direct_only_relation_slot_surrogates_v4")),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="v5 surrogate builder"):
        T.load_surrogate_artifact(p, _records(), seed=1, forget_num=1)
