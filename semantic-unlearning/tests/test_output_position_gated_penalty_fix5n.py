from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_output_position_gated_penalty_fix5n_seed1 as fix5n


def test_detect_generated_slot_word_labels():
    assert fix5n.detect_generated_slot("") is None
    assert fix5n.detect_generated_slot("First: Paris") == 1
    assert fix5n.detect_generated_slot("First: Paris\nSecond: Rome") == 2


def test_detect_generated_slot_numbered_labels():
    assert fix5n.detect_generated_slot("1. Paris") == 1
    assert fix5n.detect_generated_slot("1. Paris\n2. Rome") == 2


def test_detect_generated_slot_uses_latest_marker():
    text = "First: A\nSecond: B\nFirst: C"
    assert fix5n.detect_generated_slot(text) == 1


class FakeTokenizer:
    def decode(self, ids, **kwargs):
        del kwargs
        values = list(ids)
        if values == [1]:
            return "First: alpha"
        if values == [1, 2]:
            return "First: alpha\nSecond: beta"
        if values == [7]:
            return "unlabeled answer"
        return ""


def _scores():
    return torch.zeros((1, 8), dtype=torch.float32)


def test_position_processor_forbidden_first_switches_off_at_second_slot():
    p = fix5n.PositionGatedTokenPenaltyLogitsProcessor(
        FakeTokenizer(), prompt_token_n=1, token_ids=[3], penalty=12.0, forbidden_slot=1
    )
    s1 = p(torch.tensor([[99, 1]]), _scores())
    assert s1[0, 3].item() == -12.0
    s2 = p(torch.tensor([[99, 1, 2]]), _scores())
    assert s2[0, 3].item() == 0.0
    snap = p.snapshot()
    assert snap["penalty_active_step_n"] == 1
    assert snap["first_marker_seen"] is True
    assert snap["second_marker_seen"] is True


def test_position_processor_companion_first_switches_on_at_second_slot():
    p = fix5n.PositionGatedTokenPenaltyLogitsProcessor(
        FakeTokenizer(), prompt_token_n=1, token_ids=[3], penalty=12.0, forbidden_slot=2
    )
    s1 = p(torch.tensor([[99, 1]]), _scores())
    assert s1[0, 3].item() == 0.0
    s2 = p(torch.tensor([[99, 1, 2]]), _scores())
    assert s2[0, 3].item() == -12.0
    assert p.snapshot()["penalty_active_step_n"] == 1


def test_position_processor_fails_closed_before_any_marker():
    p = fix5n.PositionGatedTokenPenaltyLogitsProcessor(
        FakeTokenizer(), prompt_token_n=1, token_ids=[3], penalty=12.0, forbidden_slot=1
    )
    s = p(torch.tensor([[99, 7]]), _scores())
    assert s[0, 3].item() == 0.0
    snap = p.snapshot()
    assert snap["pre_marker_or_unknown_step_n"] == 1
    assert snap["usable_forbidden_slot_marker_seen"] is False


def _cond(companion: bool, forbidden: bool, with_gate: bool = False):
    x = {
        "forbidden_sensitive_canonical_mentioned": forbidden,
        "companion_strict_correct": companion,
        "joint_success": companion and not forbidden,
        "surface_invalid": False,
    }
    if with_gate:
        x["position_gate"] = {
            "usable_forbidden_slot_marker_seen": True,
            "first_marker_seen": True,
            "second_marker_seen": True,
            "penalty_active_step_pct": 40.0,
        }
    return x


def _row(kind: str, query_companion: bool, pos_companion: bool, ambiguous: bool = False):
    return {
        "pair_kind": kind,
        "order": "forbidden_first",
        "expected_forbidden_binding_active": True,
        "canonical_answer_attribution_ambiguous": ambiguous,
        "conditions": {
            "base": _cond(True, True),
            "query_wide": _cond(query_companion, False),
            "position_gated": _cond(pos_companion, False, with_gate=True),
        },
    }


def test_summary_reports_position_gate_recovery_of_query_wide_loss():
    out = fix5n.summarize_subset([
        _row("overlap", query_companion=False, pos_companion=True, ambiguous=True)
    ])
    assert out["companion_regression_from_base_query_wide"]["lost_n"] == 1
    assert out["companion_regression_from_base_position_gated"]["lost_n"] == 0
    assert out["position_gate_recovery_vs_query_wide"]["restored_by_position_gate_n"] == 1
    assert out["position_gate_recovery_vs_query_wide"]["recovery_pct_of_query_wide_losses"] == 100.0


def test_summary_preserves_attribution_guard_for_overlap_rows():
    out = fix5n.summarize_subset([
        _row("overlap", query_companion=False, pos_companion=True, ambiguous=True)
    ])
    assert out["canonical_answer_attribution"]["ambiguous_n"] == 1
    assert out["attribution_safe_forbidden_and_joint"]["n"] == 0
    assert out["attribution_safe_forbidden_and_joint"]["position_gated"] is None


def test_summary_attribution_safe_nonoverlap_rows_are_scored():
    out = fix5n.summarize_subset([
        _row("nonoverlap", query_companion=True, pos_companion=True, ambiguous=False)
    ])
    safe = out["attribution_safe_forbidden_and_joint"]
    assert safe["n"] == 1
    assert safe["position_gated"]["forbidden_canonical_disclosure_pct"] == 0.0
    assert safe["position_gated"]["joint_success_pct"] == 100.0


def test_forbidden_slot_mapping_is_order_only_not_answer_metadata():
    assert fix5n.forbidden_slot_for_order("forbidden_first") == 1
    assert fix5n.forbidden_slot_for_order("companion_first") == 2
