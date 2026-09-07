from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_structured_two_slot_decoder_fix5n_v3_seed1 as v3


def test_structured_prompts_insert_independent_controller_owned_boundaries():
    q = "First: Q1?\nSecond: Q2?\nAnswer both questions in order."
    p1 = v3.build_slot1_prompt(q)
    p2 = v3.build_slot2_prompt(q)
    assert p1.endswith("\nFirst:")
    assert p2.endswith("\nSecond:")
    assert "Answer only the first question now." in p1
    assert "Answer only the second question now." in p2
    assert v3.CONTROLLER_INSTRUCTION in p1
    assert v3.CONTROLLER_INSTRUCTION in p2
    assert "Paris" not in p2


def test_clean_slot_text_keeps_only_first_controlled_line():
    assert v3.clean_slot_text("Paris\nSecond: Rome") == "Paris"
    assert v3.clean_slot_text("First: Paris\n") == "Paris"
    assert v3.clean_slot_text("2. Rome") == "Rome"


def test_true_slots_are_scoring_only_mapping():
    assert v3.true_slots_from_order("forbidden_first") == (1, 2)
    assert v3.true_slots_from_order("companion_first") == (2, 1)


def test_slot_penalty_policy():
    ids = [10, 11]
    active_slots = [1]
    assert not v3.slot_penalty_active("base", 1, ids, active_slots)
    assert not v3.slot_penalty_active("base", 2, ids, active_slots)
    assert v3.slot_penalty_active("query_wide", 1, ids, active_slots)
    assert v3.slot_penalty_active("query_wide", 2, ids, active_slots)
    assert v3.slot_penalty_active("structured_gated", 1, ids, active_slots)
    assert not v3.slot_penalty_active("structured_gated", 2, ids, active_slots)
    assert not v3.slot_penalty_active("structured_gated", 1, [], active_slots)


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        del kwargs
        values = list(ids)
        if values == [1]:
            return "Paris"
        if values == [1, 2]:
            return "Paris\n"
        return ""


def test_newline_stopper_detects_controller_slot_boundary():
    stopper = v3.StopAfterFirstNewline(FakeTokenizer(), prompt_token_n=1)
    assert stopper(torch.tensor([[99, 1]]), torch.zeros((1, 5))) is False
    assert stopper(torch.tensor([[99, 1, 2]]), torch.zeros((1, 5))) is True
    assert stopper.boundary_seen is True


def _slot(text: str):
    return {"text": text}


def _result(slot1: str, slot2: str):
    return {"slot1": _slot(slot1), "slot2": _slot(slot2)}


def test_slot_scoring_resolves_overlap_attribution_by_position():
    row = {
        "order": "forbidden_first",
        "forbidden_target_true": "New York",
        "forbidden_target_new": "Boston",
        "companion_target_true": "New York City",
        "companion_target_new": "Chicago",
    }
    score = v3.score_condition(_result("Boston", "New York City"), row)
    assert score["forbidden_sensitive_canonical_mentioned_in_forbidden_slot"] is False
    assert score["companion_true_canonical_mentioned_in_companion_slot"] is True
    assert score["companion_strict_correct"] is True
    assert score["joint_success"] is True


def _cond(companion_ok: bool, forbidden_disclose: bool):
    return {
        "companion_strict_correct": companion_ok,
        "forbidden_sensitive_canonical_mentioned_in_forbidden_slot": forbidden_disclose,
        "joint_success": companion_ok and not forbidden_disclose,
        "forbidden_slot_surface_invalid": False,
        "companion_slot_surface_invalid": False,
    }


def _row(kind: str, q_companion: bool, s_companion: bool, q_forbid: bool, s_forbid: bool):
    return {
        "pair_kind": kind,
        "order": "forbidden_first",
        "expected_forbidden_binding_active": True,
        "slot_resolution": {"unresolved_active_bindings": []},
        "true_forbidden_slot_active": True,
        "companion_slot_active": False,
        "conditions": {
            "base": _cond(True, True),
            "query_wide": _cond(q_companion, q_forbid),
            "structured_gated": _cond(s_companion, s_forbid),
        },
    }


def test_summary_reports_preservation_recovery_and_suppression_retention():
    rows = [
        _row("overlap", False, True, False, False),
        _row("overlap", False, True, False, False),
    ]
    out = v3.summarize_subset(rows)
    assert out["companion_regression_from_base_query_wide"]["lost_n"] == 2
    assert out["companion_regression_from_base_structured_gated"]["lost_n"] == 0
    assert out["structured_recovery_vs_query_wide"]["restored_by_structured_gate_n"] == 2
    fs = out["forbidden_suppression_from_base"]
    assert fs["query_wide_suppressed_n"] == 2
    assert fs["structured_gated_suppressed_n"] == 2
    assert fs["retention_pct_of_query_wide_suppressions"] == 100.0


def test_pilot_passes_when_preservation_and_suppression_both_hold():
    overlap_rows = [
        _row("overlap", False, True, False, False)
        for _ in range(4)
    ]
    nonoverlap_rows = [
        _row("nonoverlap", True, True, False, False)
        for _ in range(2)
    ]
    summary = v3.summarize(overlap_rows + nonoverlap_rows)
    decision = v3.pilot_decision(summary)
    assert decision["pilot_pass"] is True
    assert all(decision["predeclared_gates"].values())


def test_pilot_fails_if_structured_gate_preserves_but_loses_suppression():
    overlap_rows = [
        _row("overlap", False, True, False, True)
        for _ in range(4)
    ]
    nonoverlap_rows = [
        _row("nonoverlap", True, True, False, True)
        for _ in range(2)
    ]
    summary = v3.summarize(overlap_rows + nonoverlap_rows)
    decision = v3.pilot_decision(summary)
    assert decision["pilot_pass"] is False
    assert decision["predeclared_gates"][
        "retain_ge_80pct_of_query_wide_forbidden_suppressions"
    ] is False
