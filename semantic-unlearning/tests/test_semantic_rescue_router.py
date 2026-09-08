from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_semantic_rescue_router_seed1 as mod


def test_semantic_query_removes_exact_subject_identity():
    view = mod.RoutingView(
        selected_text="What is [TARGET]Albert Einstein[/TARGET]'s occupation?",
        selection_status="ATOMIC_PASSTHROUGH",
        selected_character_offsets=(0, 10),
        scope_supported=True,
        enumerated_subjects=("Albert Einstein",),
    )
    out = mod.semantic_query_text(view, "Albert Einstein")
    assert "Albert Einstein" not in out
    assert "TARGET_ENTITY" in out
    assert "occupation" in out


def test_collect_examples_uses_relation_specific_natural_families():
    contract = {
        "families": {
            "wh_question": ["Where was {} born?"],
            "possessive_question": ["What is {}'s place of birth?"],
            "alternative_question": ["In which place was {} born?"],
            "imperative_identify": ["Name the birthplace of {}."],
            "nominalized_question": ["What is the birth location of {}?"],
            "conversational_question": ["Please tell me where {} was born."],
        }
    }
    xs = mod.collect_examples(contract, limit=6)
    assert len(xs) == 6
    assert all("TARGET_ENTITY" in x for x in xs)
    assert any("born" in x for x in xs)


def test_semantic_decision_confidence_requires_absolute_yes_and_gap():
    profiles = [
        mod.RelationProfile("P1", "a", "a", ("a",), ()),
        mod.RelationProfile("P2", "b", "b", ("b",), ()),
        mod.RelationProfile("P3", "c", "c", ("c",), ()),
    ]
    scores = torch.tensor([
        [4.0, 1.0, 0.0],   # top=4 gap=3 => confidence 3
        [-0.5, -1.0, -2.0], # top negative => confidence negative
    ])
    out = mod.semantic_decisions(scores, profiles)
    assert out[0].relation == "P1"
    assert abs(out[0].confidence - 3.0) < 1e-9
    assert out[1].confidence < 0.0


def test_rescue_never_removes_existing_fix5o_activation():
    row = mod.Row(
        text="What is A's occupation?", subject="A", relation="P106",
        forbidden=True, kind="x", family="x", case_id=1, candidate=True,
    )
    view = mod.RoutingView(
        selected_text="What is [TARGET]A[/TARGET]'s occupation?",
        selection_status="ATOMIC_PASSTHROUGH", selected_character_offsets=None,
        scope_supported=True, enumerated_subjects=("A",),
    )
    fix = {"activates": True, "relation": "P106"}
    sem = mod.SemanticDecision("P101", 10.0, 0.0, 10.0, 10.0)
    active = mod.active_relations_for_row(
        row, view, fix, sem, eta_semantic=0.0, bank={("A", "P106"), ("A", "P101")}
    )
    assert active == {"P106"}


def test_rescue_adds_only_bank_binding_when_primary_is_inactive():
    row = mod.Row(
        text="What field does A work in?", subject="A", relation="P101",
        forbidden=True, kind="x", family="x", case_id=1, candidate=True,
    )
    view = mod.RoutingView(
        selected_text="What field does [TARGET]A[/TARGET] work in?",
        selection_status="ATOMIC_PASSTHROUGH", selected_character_offsets=None,
        scope_supported=True, enumerated_subjects=("A",),
    )
    fix = {"activates": False, "relation": "NONE"}
    sem = mod.SemanticDecision("P101", 5.0, 1.0, 4.0, 4.0)
    active = mod.active_relations_for_row(
        row, view, fix, sem, eta_semantic=2.0, bank={("A", "P101")}
    )
    assert active == {"P101"}
    active2 = mod.active_relations_for_row(
        row, view, fix, sem, eta_semantic=6.0, bank={("A", "P101")}
    )
    assert active2 == set()


def test_preservation_gate_enforces_every_declared_budget():
    good = {
        "wrong_forbidden_binding_accept_pct": 0.0,
        "permitted_false_activation_pct": 1.0,
        "candidate_present_permitted_false_activation_pct": 2.0,
        "permitted_query_false_activation_pct": 1.5,
        "mixed_query_permitted_companion_false_activation_pct": 2.0,
        "permitted_negative_families": {
            "same_subject_different_relation": {"false_activation_pct": 2.0}
        },
    }
    assert mod.preservation_ok(good, 0.02)
    bad = dict(good)
    bad["permitted_query_false_activation_pct"] = 2.01
    assert not mod.preservation_ok(bad, 0.02)


def test_calibration_can_choose_no_rescue_as_safe_fallback():
    rows = [
        mod.Row(
            text="q", subject="A", relation="P1", forbidden=False,
            kind="same_subject_different_relation", family="x", case_id=1,
            candidate=True,
        )
    ]
    views = [mod.RoutingView("q", "ATOMIC_PASSTHROUGH", None, True, ("A",))]
    fix = [{"activates": False, "relation": "NONE"}]
    sem = [mod.SemanticDecision("P1", 5.0, 0.0, 5.0, 5.0)]
    # Semantic calibration row is intentionally wrong, so the safest operating point
    # should reject the rescue rather than violate the permitted budget.
    sem_cal_rows = [mod.Row("z", "B", "P2", True, "x", "x", 2, candidate=True)]
    sem_cal = [mod.SemanticDecision("P1", 1.0, 0.0, 1.0, 1.0)]
    eta, payload = mod.calibrate_rescue_eta(
        sem_cal_rows, sem_cal, rows, views, fix, sem,
        bank={("A", "P1")}, epsilon=0.0,
    )
    assert eta > 5.0
    assert payload["policy"]["permitted_false_activation_pct"] == 0.0
