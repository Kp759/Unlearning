import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torch

from run_mcf_rsnr_v1a_relation_router_v2 import (
    DiagonalRelationRouter,
    build_pair_corpus,
    candidate_case_ids,
    pair_feature,
    split_fact_views,
    subject_is_mentioned,
    synthetic_hard_negative_prompts,
)


def _forget_rows():
    return [
        {
            "case_id": 1,
            "requested_rewrite": {
                "subject": "Alpha Car",
                "relation_id": "manufacturer",
            },
        },
        {
            "case_id": 2,
            "requested_rewrite": {
                "subject": "Beta Person",
                "relation_id": "birth_place",
            },
        },
        {
            "case_id": 3,
            "requested_rewrite": {
                "subject": "Gamma Work",
                "relation_id": "genre",
            },
        },
    ]


def _views():
    return {
        1: ["Who makes {}?", "{} is made by", "Manufacturer of {} is", "Who manufactures {}?", "Maker of {}:"],
        2: ["Where was {} born?", "{} was born in", "Birthplace of {} is", "Where is {} from?", "Born where: {}"],
        3: ["What genre is {}?", "{} has genre", "Genre of {} is", "Which genre fits {}?", "Genre: {}"],
    }


def test_subject_candidate_matching_is_case_insensitive_and_specific():
    assert subject_is_mentioned("WHO MAKES ALPHA CAR?", "Alpha Car")
    assert not subject_is_mentioned("Who makes Beta Person?", "Alpha Car")


def test_split_fact_views_keeps_one_calibration_view_per_fact():
    facts = split_fact_views(_forget_rows(), _views())
    assert set(facts) == {1, 2, 3}
    for info in facts.values():
        assert len(info["train_prompts"]) == 4
        assert info["calib_prompt"] not in set(info["train_prompts"])
        assert info["subject"] in info["calib_prompt"]


def test_hard_negatives_keep_subject_but_change_relation_template():
    facts = split_fact_views(_forget_rows(), _views())
    negatives = synthetic_hard_negative_prompts(facts, 1, count=2, calibration=False)
    assert negatives
    assert all("Alpha Car" in text for text in negatives)
    assert all("makes Alpha Car" not in text for text in negatives)


def test_pair_corpus_contains_relation_hard_negatives_and_subject_only_negatives():
    facts = split_fact_views(_forget_rows(), _views())
    train, calib, meta = build_pair_corpus(facts)
    assert sum(int(x["label"] == 1) for x in train) == 12
    assert any(x["kind"] == "same_subject_different_relation_synthetic_train" for x in train)
    assert any(x["kind"] == "subject_only_train" for x in train)
    assert sum(int(x["label"] == 1) for x in calib) == 3
    assert meta["official_probe_text_used_for_router_fit"] is False
    assert meta["official_probe_text_used_for_threshold_calibration"] is False


def test_candidate_case_ids_uses_subject_before_relation_scoring():
    facts = split_fact_views(_forget_rows(), _views())
    assert candidate_case_ids("Where was Alpha Car created?", facts) == [1]
    assert candidate_case_ids("Tell me about an unrelated entity", facts) == []


def test_relation_pair_feature_is_elementwise_residual_agreement():
    q = torch.tensor([3.0, 2.0])
    anchor = torch.tensor([1.0, 2.0])
    proto = torch.tensor([1.0, 0.0])
    feature = pair_feature(q, anchor, proto)
    assert torch.allclose(feature, torch.tensor([1.0, 0.0]), atol=1e-6)


def test_router_parameter_count_is_hidden_size_plus_bias():
    router = DiagonalRelationRouter(3072)
    assert sum(p.numel() for p in router.parameters()) == 3073
