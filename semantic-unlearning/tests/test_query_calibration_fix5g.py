from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "replay_target_representation_query_calibration_fix5g_seed1.py"
spec = importlib.util.spec_from_file_location("fix5g", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

Row = m.Row


def row(text, subject, relation, forbidden=False, kind="crossed_binding", candidate=True):
    return Row(text=text, subject=subject, relation=relation, forbidden=forbidden, kind=kind, family="x", case_id=1, masked=text, candidate=candidate)


def test_query_report_counts_any_route_activation_once_per_query():
    rows = [
        row("First A. Second B.", "A", "R1"),
        row("First A. Second B.", "B", "R2"),
    ]
    classes = ["R1", "R2", "NONE"]
    bank = {("A", "R1")}
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    rep = m.query_report(rows, logits, 1.0, classes, 2, bank)
    assert rep["overall"]["n"] == 1
    assert rep["overall"]["false_activation_n"] == 1
    assert rep["overall"]["false_activation_pct"] == 100.0


def test_query_report_keeps_different_families_separate():
    rows = [
        row("q1", "A", "R2", kind="crossed_binding"),
        row("q2", "A", "R2", kind="same_subject_different_relation"),
    ]
    classes = ["R1", "R2", "NONE"]
    bank = {("A", "R1")}
    logits = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    rep = m.query_report(rows, logits, 1.0, classes, 2, bank)
    assert set(rep["by_family"]) == {"crossed_binding", "same_subject_different_relation"}


def test_replay_calibration_can_reject_threshold_that_violates_query_family_budget():
    rows = [
        row("positive", "A", "R1", forbidden=True, kind="fix5_calib"),
        row("pair", "A", "R2", kind="crossed_binding"),
        row("pair", "B", "R2", kind="crossed_binding"),
    ]
    classes = ["R1", "R2", "NONE"]
    bank = {("A", "R1")}
    logits = torch.tensor([
        [5.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
        [0.0, 3.0, 0.0],
    ])
    eta, report = m.replay_calibrate(rows, logits, classes, 2, bank, eps=0.02, eps_wrong=0.02, min_correct_accept=0.0)
    assert eta > 3.0
    assert report["query_family_false_activation_rates"]["crossed_binding"] == 0.0
