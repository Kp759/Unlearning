import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_surrogate_semantic_validator_v2 as V  # noqa: E402
import mcf_sure_h_then_genaware_lmhead_lora_v4 as T  # noqa: E402


def test_relation_slot_parser_accepts_calibrated_fragment_equivalence():
    raw = json.dumps({
        "same_slot_relation": True,
        "same_answer_type": True,
        "no_added_factual_constraints": True,
        "semantically_coherent": True,
        "verdict": "EQUIVALENT",
        "reason": "same occupation slot",
    })
    r = V.parse_relation_slot(raw)
    assert r["parse_ok"] is True
    assert r["accepted"] is True


def test_adversarial_parser_rejects_relation_shift():
    raw = json.dumps({
        "relation_shift": True,
        "answer_type_shift": False,
        "added_constraint_or_claim": False,
        "semantic_malformation": False,
        "weak_wrapper": False,
        "verdict": "UNSAFE",
        "reason": "manufacturer became vehicle",
    })
    r = V.parse_adversarial(raw)
    assert r["parse_ok"] is True
    assert r["accepted"] is False


def test_calibration_prompt_explicitly_allows_completion_stems():
    prompt = V.relation_slot_instruction(
        "Rory O'Hanlon",
        "Rory O'Hanlon works as",
        "Rory O'Hanlon serves as",
    )
    assert "incomplete sentence stems" in prompt
    assert "NORMAL" in prompt
    assert "same unknown value" in prompt


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


def test_v4_trainer_accepts_only_calibrated_v4_artifact(tmp_path):
    p = tmp_path / "v4.json"
    p.write_text(json.dumps(_artifact()), encoding="utf-8")
    data, prompts = T.load_surrogate_artifact(p, _records(), seed=1, forget_num=1)
    assert data["builder_protocol"] == T.SEMANTIC_BUILDER_PROTOCOL
    assert prompts == [["Ada's occupation was", "What occupation did Ada have?"]]


def test_v4_trainer_rejects_old_v3_builder(tmp_path):
    p = tmp_path / "old.json"
    p.write_text(
        json.dumps(_artifact("mcf_locked_direct_only_semantic_surrogates_v3")),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="v4 surrogate builder"):
        T.load_surrogate_artifact(p, _records(), seed=1, forget_num=1)
