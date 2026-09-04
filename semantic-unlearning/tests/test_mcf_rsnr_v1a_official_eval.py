from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_rsnr_v1a_official_eval_fresh_retain as ev  # noqa: E402


def _record(
    subject: str,
    relation: str,
    *,
    case_id: int = 1,
    paraphrases=None,
    neighborhoods=None,
    target_true: str = "Brussels",
):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "subject": subject,
            "relation_id": relation,
            "prompt": "The fact about {} is",
            "target_true": {"str": target_true, "id": f"Q{case_id}"},
            "target_new": {"str": "Fake"},
        },
        "paraphrase_prompts": list(paraphrases or []),
        "neighborhood_prompts": list(neighborhoods or []),
    }


def _membership(*rows):
    return {
        (int(row["case_id"]), str(row["requested_rewrite"]["subject"]), str(row["requested_rewrite"]["relation_id"]))
        for row in rows
    }


def test_sensitive_neighborhood_prompt_routes_on_per_prompt():
    forgotten = _record("BMW M5", "P176", case_id=17, target_true="BMW")
    parent = _record(
        "Unrelated car",
        "P176",
        case_id=88,
        neighborhoods=["BMW M5 is developed by"],
        target_true="Other maker",
    )
    router = ev.OraclePromptRouter(_membership(forgotten))
    routed = ev.route_record_prompts(parent, router)
    item = routed["neighborhood"][0]
    assert item["gated"] is True
    assert item["match"] == (17, "BMW M5", "P176")


def test_non_sensitive_neighborhood_same_relation_stays_off():
    forgotten = _record("BMW M5", "P176", case_id=17)
    parent = _record(
        "Unrelated car",
        "P176",
        case_id=88,
        neighborhoods=["Audi A6 is developed by"],
    )
    router = ev.OraclePromptRouter(_membership(forgotten))
    assert ev.route_record_prompts(parent, router)["neighborhood"][0]["gated"] is False


def test_forgotten_parent_paraphrase_can_use_oracle_metadata_fallback():
    forgotten = _record(
        "Belgium",
        "P36",
        case_id=7,
        paraphrases=["What is its capital?"],
    )
    router = ev.OraclePromptRouter(_membership(forgotten))
    routed = ev.route_record_prompts(forgotten, router)
    assert routed["rewrite"][0]["gated"] is True
    assert routed["paraphrase"][0]["gated"] is True
    assert routed["paraphrase"][0]["match"] == (7, "Belgium", "P36")


def test_same_relation_different_subject_does_not_fire():
    forgotten = _record("Belgium", "P36", case_id=7)
    router = ev.OraclePromptRouter(_membership(forgotten))
    assert router.resolve("The capital of France is", "P36") is None


def test_compound_prompt_with_two_distinct_forgotten_subjects_is_rejected():
    belgium = _record("Belgium", "P36", case_id=7)
    france = _record("France", "P36", case_id=8)
    router = ev.OraclePromptRouter(_membership(belgium, france))
    with pytest.raises(RuntimeError, match="multiple forgotten subjects"):
        router.resolve("The capitals of Belgium and France are", "P36")


def test_abstention_detection_has_exact_and_rule_semantic_modes():
    exact = ev.detect_abstention("I don't know.")
    assert exact == {"exact": True, "semantic": True}

    semantic = ev.detect_abstention("I'm not sure what the answer is.")
    assert semantic["exact"] is False
    assert semantic["semantic"] is True

    answer = ev.detect_abstention("The answer is Brussels.")
    assert answer == {"exact": False, "semantic": False}


def test_true_answer_and_alias_leakage_is_detected_case_insensitively():
    assert ev.leaked_aliases("My guess is BRUSSELS.", ["Brussels", "Bruxelles"]) == ["Brussels"]
    assert ev.leaked_aliases("I don't know.", ["Brussels", "Bruxelles"]) == []


def test_alias_map_includes_canonical_and_same_target_id_surface_forms():
    forget = _record("Belgium", "P36", case_id=7, target_true="Brussels")
    other_surface = _record("Other", "P999", case_id=7, target_true="City of Brussels")
    aliases = ev.build_true_alias_map([forget, other_surface], [forget])[("Belgium", "P36")]
    assert "Brussels" in aliases
    assert "City of Brussels" in aliases


def _artifact_fixture():
    locked = [_record("Belgium", "P36", case_id=7)]
    membership = [{"case_id": 7, "subject": "Belgium", "relation_id": "P36"}]
    adapter = {
        "protocol": ev.PROTOCOL,
        "abstention": ev.ABSTENTION,
        "forget_membership": membership,
    }
    sidecar = {
        "protocol": ev.PROTOCOL,
        "abstention_text": ev.ABSTENTION,
        "forget_membership": membership,
    }
    completion = {
        "protocol": ev.PROTOCOL,
        "joint_passed": 1,
        "joint_failures": 0,
        "adapter_saved": True,
        "base_weights_modified": False,
        "heldout_probe_text_used": False,
    }
    manifest = {"case_ids": {"forget": [7]}}
    return adapter, sidecar, completion, locked, manifest


def test_artifact_validation_rejects_stale_checkpoint_even_if_sidecar_agrees():
    adapter, sidecar, completion, locked, manifest = _artifact_fixture()
    stale = [{"case_id": 7, "subject": "France", "relation_id": "P36"}]
    adapter["forget_membership"] = stale
    sidecar["forget_membership"] = stale
    with pytest.raises(RuntimeError, match="does not match locked forget records"):
        ev.validate_artifact_correspondence(
            adapter_payload=adapter,
            sidecar=sidecar,
            completion=completion,
            locked_forget=locked,
            manifest=manifest,
            expected_count=1,
        )


def test_artifact_validation_rejects_failed_completion():
    adapter, sidecar, completion, locked, manifest = _artifact_fixture()
    completion["joint_passed"] = 0
    completion["joint_failures"] = 1
    with pytest.raises(RuntimeError, match="training gate failed"):
        ev.validate_artifact_correspondence(
            adapter_payload=adapter,
            sidecar=sidecar,
            completion=completion,
            locked_forget=locked,
            manifest=manifest,
            expected_count=1,
        )


def test_evaluator_documents_target_new_as_metric_reference_not_training_target():
    source = Path(ev.__file__).read_text(encoding="utf-8")
    assert '"target_new_used_for_training": False' in source
    assert '"official_eff_gen_still_compare_target_true_vs_target_new": True' in source
    assert '"idk_generation_directly_audited": True' in source
