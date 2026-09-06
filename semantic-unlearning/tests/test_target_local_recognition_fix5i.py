from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_local_recognition_baseline_fix5i_seed1.py"
spec = importlib.util.spec_from_file_location("fix5i_target_local", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

Row = m.Row


def row(
    text: str,
    subject: str,
    relation: str,
    *,
    forbidden: bool = False,
    kind: str = "retain_other",
    family: str = "x",
    case_id: int = 1,
    candidate: bool = True,
) -> Row:
    return Row(
        text=text,
        subject=subject,
        relation=relation,
        forbidden=forbidden,
        kind=kind,
        family=family,
        case_id=case_id,
        masked="",
        candidate=candidate,
    )


def test_registered_subject_enumeration_comes_from_query_text():
    got = m.registered_subject_candidates(
        "Belgium is compared with France.",
        ["Belgium", "France", "Germany"],
    )
    assert got == ["Belgium", "France"]


def test_explicit_two_clause_selector_is_target_local_without_relation_label():
    q = "First: Where was Belgium founded? Second: What language does France use?"
    view = m.routing_view(q, "Belgium", ["Belgium", "France"])
    assert view.scope_supported
    assert view.selection_status == "SELECTED_EXPLICIT_CLAUSE"
    assert view.selected_text == "Where was [TARGET]Belgium[/TARGET] founded?"
    assert "France" not in view.selected_text


def test_atomic_single_subject_query_is_passthrough_and_marked():
    q = "Which organization is Belgium a member of?"
    view = m.routing_view(q, "Belgium", ["Belgium", "France"])
    assert view.scope_supported
    assert view.selection_status == "ATOMIC_PASSTHROUGH"
    assert view.selected_text == "Which organization is [TARGET]Belgium[/TARGET] a member of?"


def test_unstructured_multi_subject_query_is_conservative_unsupported():
    q = "Compare Belgium with France and tell me which one is in Europe."
    view = m.routing_view(q, "Belgium", ["Belgium", "France"])
    assert not view.scope_supported
    assert view.selection_status == "UNSUPPORTED_UNSTRUCTURED_MULTI_SUBJECT"
    assert "[TARGET]Belgium[/TARGET]" in view.selected_text
    assert "OTHER_ENTITY" in view.selected_text


def test_mixed_query_builder_creates_forbidden_and_permitted_companion_routes():
    pos = row(
        "Which organization is Belgium a member of?",
        "Belgium",
        "P463",
        forbidden=True,
        kind="fix5_calib",
        case_id=10,
    )
    crossed_a = row(
        "First: Where was France born? Second: What language does Germany use?",
        "France",
        "P19",
        forbidden=False,
        kind="crossed_binding",
        case_id=11,
    )
    mixed = m.build_mixed_queries([pos, crossed_a], ["Belgium", "France", "Germany"], 1, "calib")
    assert len(mixed) == 2
    assert mixed[0].text == mixed[1].text
    assert mixed[0].forbidden is True
    assert mixed[1].forbidden is False
    assert mixed[0].subject == "Belgium"
    assert mixed[1].subject == "France"

    v0 = m.routing_view(mixed[0].text, mixed[0].subject, ["Belgium", "France", "Germany"])
    v1 = m.routing_view(mixed[1].text, mixed[1].subject, ["Belgium", "France", "Germany"])
    assert v0.scope_supported and v1.scope_supported
    assert "[TARGET]Belgium[/TARGET]" in v0.selected_text
    assert "[TARGET]France[/TARGET]" in v1.selected_text


def test_unsupported_scope_cannot_activate_even_with_high_forbidden_logit():
    r = row(
        "Belgium and France are discussed together.",
        "Belgium",
        "P463",
        forbidden=False,
        kind="retain_other",
    )
    view = m.RoutingView(
        selected_text="[TARGET]Belgium[/TARGET] and OTHER_ENTITY are discussed together.",
        selection_status="UNSUPPORTED_UNSTRUCTURED_MULTI_SUBJECT",
        selected_character_offsets=(0, 40),
        scope_supported=False,
        enumerated_subjects=("Belgium", "France"),
    )
    logits = torch.tensor([[10.0, 0.0]])
    d = m.decision_tensors(
        [r], logits, [view], eta=1.0,
        classes=["P463", m.NONE], none_idx=1,
        bank={("Belgium", "P463")},
    )
    assert bool(d["accepted_relation"][0]) is False
    assert bool(d["activates"][0]) is False


def test_whole_query_mixed_metrics_require_forbidden_hit_and_clean_companion():
    q = "First: ask A. Second: ask B."
    rows = [
        row(q, "A", "P1", forbidden=True, kind="mixed_forbidden_distractor_validation"),
        row(q, "B", "P2", forbidden=False, kind="mixed_forbidden_distractor_validation"),
    ]
    d = {
        "activates": torch.tensor([True, False]),
        "correct": torch.tensor([True, True]),
    }
    w = m.whole_query_report(rows, d)
    assert w["mixed_query_n"] == 1
    assert w["mixed_query_correct_forbidden_activation_pct"] == 100.0
    assert w["mixed_query_permitted_companion_false_activation_pct"] == 0.0
    assert w["mixed_query_joint_success_pct"] == 100.0
