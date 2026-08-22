import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_mcf_surrogate_paraphrases as builder_base  # noqa: E402
import build_mcf_surrogate_paraphrases_v3 as builder_v3  # noqa: E402
import mcf_surrogate_semantic_validator as semantic  # noqa: E402
import mcf_sure_h_then_genaware_lmhead_lora_v3 as trainer_v3  # noqa: E402


def test_structural_validator_rejects_generic_wrappers_but_not_real_rewrites():
    direct = "Toyota Tercel is produced by"
    assert semantic.structural_rejection_reason(
        direct,
        "Complete this factual statement: Toyota Tercel is produced by",
    ) == "generic_wrapper"
    assert semantic.structural_rejection_reason(
        direct,
        "The Toyota Tercel is manufactured by",
    ) is None


def test_structural_validator_rejects_verbatim_direct_inside_longer_wrapper():
    direct = "Arkansas has an official language of"
    candidate = "Please answer carefully using one phrase: Arkansas has an official language of"
    assert semantic.structural_rejection_reason(direct, candidate) == "contains_direct_prompt_verbatim"


def test_equivalence_parser_requires_every_positive_criterion():
    good = json.dumps({
        "same_relation": True,
        "same_answer_type": True,
        "no_added_factual_claims": True,
        "grammatical": True,
        "completion_compatible": True,
        "verdict": "ACCEPT",
        "reason": "same relation",
    })
    assert semantic.parse_equivalence(good)["accepted"] is True

    bad = json.dumps({
        "same_relation": True,
        "same_answer_type": False,
        "no_added_factual_claims": True,
        "grammatical": True,
        "completion_compatible": True,
        "verdict": "ACCEPT",
        "reason": "answer type changed",
    })
    assert semantic.parse_equivalence(bad)["accepted"] is False


def test_critic_parser_requires_no_flags_and_safe_verdict():
    good = json.dumps({
        "different_relation": False,
        "different_answer_type": False,
        "added_factual_claim": False,
        "malformed_or_incomplete": False,
        "generic_wrapper": False,
        "verdict": "SAFE",
        "reason": "none",
    })
    assert semantic.parse_critic(good)["accepted"] is True

    bad = json.dumps({
        "different_relation": False,
        "different_answer_type": False,
        "added_factual_claim": True,
        "malformed_or_incomplete": False,
        "generic_wrapper": False,
        "verdict": "UNSAFE",
        "reason": "adds biographical context",
    })
    assert semantic.parse_critic(bad)["accepted"] is False


def test_semantic_judge_prompts_are_answer_blind_by_api():
    subject = "Matija Gubec"
    direct = "Matija Gubec is originally from"
    candidate = "Matija Gubec hails from"
    eq = semantic.equivalence_instruction(subject, direct, candidate)
    critic = semantic.critic_instruction(subject, direct, candidate)
    # These secrets are deliberately not accepted as function arguments and
    # therefore cannot leak into the semantic judge prompts.
    assert "Croatia_SECRET" not in eq
    assert "Slovenia_SECRET" not in eq
    assert "Croatia_SECRET" not in critic
    assert "Slovenia_SECRET" not in critic
    assert subject in eq and direct in eq and candidate in eq
    assert subject in critic and direct in critic and candidate in critic


def _record():
    return {
        "case_id": 7,
        "requested_rewrite": {
            "subject": "Ada Lovelace",
            "prompt": "{} worked in",
            "target_true": {"str": "sensitive"},
            "target_new": {"str": "reference"},
        },
    }


def _semantic_payload():
    return {
        "schema_version": 1,
        "protocol": builder_base.PROTOCOL,
        "builder_protocol": trainer_v3.SEMANTIC_BUILDER_PROTOCOL,
        "seed": 1,
        "forget_num": 1,
        "surrogates_per_record": 2,
        "generator": {
            "generator_received_target_true": False,
            "generator_received_target_new": False,
            "deterministic_wrapper_fallback_used": False,
        },
        "semantic_validation": {
            "enabled": True,
            "protocol": semantic.VALIDATOR_PROTOCOL,
            "dual_pass_consensus": True,
            "required_for_every_surrogate": True,
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
            "subject": "Ada Lovelace",
            "direct_prompt": "Ada Lovelace worked in",
            "surrogate_prompts": [
                "The field Ada Lovelace worked in was",
                "Ada Lovelace's field of work was",
            ],
        }],
    }


def test_v3_trainer_accepts_semantic_artifact_contract(tmp_path):
    p = tmp_path / "surrogates.json"
    p.write_text(json.dumps(_semantic_payload()), encoding="utf-8")
    data, prompts = trainer_v3.load_surrogate_artifact(
        p, [_record()], seed=1, forget_num=1
    )
    assert data["semantic_validation"]["enabled"] is True
    assert len(prompts) == 1
    assert len(prompts[0]) == 2


def test_v3_trainer_rejects_old_unvalidated_artifact(tmp_path):
    payload = _semantic_payload()
    payload.pop("builder_protocol")
    payload.pop("semantic_validation")
    p = tmp_path / "surrogates.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="semantically validated v3"):
        trainer_v3.load_surrogate_artifact(
            p, [_record()], seed=1, forget_num=1
        )


def test_v3_builder_defaults_to_strict_semantic_pipeline():
    args = builder_v3.parse_args([
        "--training-visible-path", "visible.json",
        "--split-manifest", "manifest.json",
        "--output", "out.json",
        "--generator-model-path", "model",
    ])
    assert args.surrogates_per_record == 8
    assert args.generation_rounds == 6
    assert args.judge_batch_size == 8
    assert args.validator_model_path is None
