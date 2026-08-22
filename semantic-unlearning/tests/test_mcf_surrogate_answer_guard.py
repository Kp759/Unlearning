import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_mcf_surrogate_paraphrases as builder  # noqa: E402
import mcf_surrogate_answer_guard as guard  # noqa: E402
import mcf_sure_h_then_genaware_lmhead_lora_v2 as trainer_v2  # noqa: E402


def test_guard_allows_answer_already_present_in_direct_prompt():
    direct = "United States musician X was born in"
    candidate = "For X, complete this fact: United States musician X was born in"
    assert guard.answer_occurrence_count(direct, "United States") == 1
    assert guard.answer_occurrence_count(candidate, "United States") == 1
    assert not guard.introduced_answer_occurrences(
        candidate, direct, ["United States", "France"]
    )


def test_guard_rejects_new_answer_occurrence():
    direct = "X was born in"
    candidate = "X was born in France"
    assert guard.introduced_answer_occurrences(candidate, direct, ["France"])


def test_builder_fallback_survives_baseline_answer_overlap():
    direct = "United States musician X was born in"
    candidates = builder.deterministic_surrogates("X", direct)
    accepted = builder._validated_unique(
        candidates,
        subject="X",
        direct_prompt=direct,
        answers=["United States", "France"],
        limit=8,
    )
    assert len(accepted) == 8


def test_v2_loader_allows_baseline_overlap_but_rejects_new_leak(tmp_path):
    record = {
        "case_id": 7,
        "requested_rewrite": {
            "subject": "X",
            "prompt": "United States musician {} was born in",
            "target_true": {"str": "United States"},
            "target_new": {"str": "France"},
        },
    }
    base_payload = {
        "schema_version": 1,
        "protocol": trainer_v2.base.SURROGATE_PROTOCOL,
        "seed": 1,
        "forget_num": 1,
        "data_access": {
            "official_paraphrase_seen": 0,
            "official_neighborhood_seen": 0,
            "benchmark_retain_seen": 0,
            "official_PPL_seen": False,
        },
        "records": [{
            "case_id": 7,
            "sampled_position": 0,
            "subject": "X",
            "direct_prompt": "United States musician X was born in",
            "surrogate_prompts": [
                "For X, complete this fact: United States musician X was born in"
            ],
        }],
    }
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(base_payload), encoding="utf-8")
    _, prompts = trainer_v2.load_surrogate_artifact(
        p, [record], seed=1, forget_num=1
    )
    assert len(prompts[0]) == 1

    bad = json.loads(json.dumps(base_payload))
    bad["records"][0]["surrogate_prompts"] = [
        "For X, answer France: United States musician X was born in"
    ]
    p2 = tmp_path / "bad.json"
    p2.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(RuntimeError, match="answer occurrence introduced"):
        trainer_v2.load_surrogate_artifact(p2, [record], seed=1, forget_num=1)
