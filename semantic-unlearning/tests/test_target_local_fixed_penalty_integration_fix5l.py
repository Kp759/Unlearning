from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_fixed_penalty_integration_fix5l_seed1 as mod


def test_fixed_penalty_rank1_changes_only_selected_ids_and_not_input():
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = mod.apply_fixed_penalty(x, [1, 3, 1], 12.0)
    assert torch.equal(x, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert torch.equal(y, torch.tensor([1.0, -10.0, 3.0, -8.0]))


def test_fixed_penalty_rank2_applies_same_union_to_each_row():
    x = torch.zeros((2, 5))
    y = mod.apply_fixed_penalty(x, [2, 4], 3.5)
    assert torch.all(y[:, 2] == -3.5)
    assert torch.all(y[:, 4] == -3.5)
    assert torch.all(y[:, [0, 1, 3]] == 0.0)


def test_fixed_penalty_empty_support_is_exact_identity_copy():
    x = torch.randn(7)
    y = mod.apply_fixed_penalty(x, [], 12.0)
    assert torch.equal(x, y)
    assert x.data_ptr() != y.data_ptr()


def test_generation_processor_shares_fixed_penalty_primitive():
    proc = mod.FixedTokenPenaltyLogitsProcessor([1, 3], penalty=2.0)
    scores = torch.zeros((2, 5))
    got = proc(torch.tensor([[1], [2]]), scores)
    expected = mod.apply_fixed_penalty(scores, [1, 3], 2.0)
    assert torch.equal(got, expected)


def test_route_cohort_correctly_accepted():
    d = mod.RouteDecision(
        query="q",
        candidates=("A",),
        active_bindings=(("A", "P1"),),
        active_token_ids=(10,),
        routes=({"subject": "A", "predicted_relation": "P1"},),
    )
    assert mod.route_cohort(d, ("A", "P1")) == "correctly_accepted"


def test_route_cohort_wrong_binding_has_safety_priority():
    d = mod.RouteDecision(
        query="q",
        candidates=("A", "B"),
        active_bindings=(("A", "P1"), ("B", "P2")),
        active_token_ids=(10,),
        routes=(
            {"subject": "A", "predicted_relation": "P1"},
            {"subject": "B", "predicted_relation": "P2"},
        ),
    )
    assert mod.route_cohort(d, ("A", "P1")) == "wrong_binding_accepted"


def test_route_cohort_correct_relation_but_rejected():
    d = mod.RouteDecision(
        query="q",
        candidates=("A",),
        active_bindings=(),
        active_token_ids=(),
        routes=({"subject": "A", "predicted_relation": "P1"},),
    )
    assert mod.route_cohort(d, ("A", "P1")) == "correctly_classified_but_rejected_or_unsupported"


def test_route_cohort_misclassified():
    d = mod.RouteDecision(
        query="q",
        candidates=("A",),
        active_bindings=(),
        active_token_ids=(),
        routes=({"subject": "A", "predicted_relation": "P9"},),
    )
    assert mod.route_cohort(d, ("A", "P1")) == "misclassified"


def test_target_string_supports_mcf_mapping_and_plain_string():
    assert mod.target_string({"str": "Paris"}) == "Paris"
    assert mod.target_string("Paris") == "Paris"


def test_summarize_rows_reports_identity_and_preference():
    rows = [
        {
            "route_cohort": "correctly_accepted",
            "route": {"active_bindings": [["A", "P1"]]},
            "scores": {
                "base": {"target_true": 1.0, "target_new": 2.0},
                "router_only": {"target_true": 1.0, "target_new": 2.0},
                "integrated": {"target_true": 4.0, "target_new": 2.0},
            },
        }
    ]
    out = mod.summarize_rows(rows)
    assert out["base"]["sensitive_preference_pct"] == 100.0
    assert out["integrated"]["sensitive_preference_pct"] == 0.0
    assert out["identity_control_max_abs_nll_diff"] == 0.0
    assert out["integrated_minus_base_sensitive_nll_mean"] == 3.0
