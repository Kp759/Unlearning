from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_relation_head_compare_fix5d_seed1.py"
spec = importlib.util.spec_from_file_location("fix5d", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

Row = m.Row


def row(subject: str, relation: str, text: str, forbidden: bool = False) -> Row:
    return Row(
        text=text,
        subject=subject,
        relation=relation,
        forbidden=forbidden,
        kind="crossed_binding",
        family="nominalized_question",
        case_id=1,
        masked=text,
        candidate=True,
    )


def test_whole_query_counts_any_route_activation_once():
    text = "First: ask A relation. Second: ask B relation."
    rows = [row("A", "R3", text), row("B", "R3", text)]
    classes = ["R1", "R2", "R3", "NONE"]
    bank = {("A", "R1"), ("B", "R2")}
    logits = torch.tensor([
        [9.0, 0.0, 1.0, 0.0],  # route A predicts forbidden R1 -> activation
        [9.0, 0.0, 1.0, 0.0],  # route B predicts R1 -> not forbidden for B
    ])
    rep = m.whole_query_policy_report(rows, logits, eta=1.0, classes=classes, none_idx=3, bank=bank)
    assert rep["permitted_query_n"] == 1
    assert rep["permitted_query_false_activation_n"] == 1
    assert rep["permitted_query_false_activation_pct"] == 100.0
    assert rep["multi_route_permitted_query_n"] == 1


def test_same_masked_prediction_can_have_different_subject_policy_outcomes():
    text = "Ask TARGET_ENTITY about R3."
    rows = [row("A", "R3", text), row("B", "R3", text)]
    classes = ["R1", "R3", "NONE"]
    bank = {("A", "R1"), ("B", "R2")}
    logits = torch.tensor([[8.0, 1.0, 0.0], [8.0, 1.0, 0.0]])
    rep = m.policy_with_whole_query(rows, logits, eta=1.0, classes=classes, none_idx=2, bank=bank)
    assert rep["whole_query"]["permitted_query_false_activation_n"] == 1
    assert rep["whole_query"]["permitted_query_n"] == 1


def test_no_permitted_rows_returns_empty_whole_query_report():
    rows = [row("A", "R1", "Ask A R1", forbidden=True)]
    classes = ["R1", "NONE"]
    logits = torch.tensor([[5.0, 0.0]])
    rep = m.whole_query_policy_report(rows, logits, eta=1.0, classes=classes, none_idx=1, bank={("A", "R1")})
    assert rep["permitted_query_n"] == 0
    assert rep["permitted_query_false_activation_pct"] is None
