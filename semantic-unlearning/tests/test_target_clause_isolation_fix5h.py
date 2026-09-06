from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "diagnose_target_clause_isolation_fix5h_seed1.py"
spec = importlib.util.spec_from_file_location("fix5h", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def test_split_first_second():
    out = m.split_first_second("First: Where was A born? Second: What language does B use?")
    assert out == ("Where was A born?", "What language does B use?")


def test_isolate_first_clause_and_mark_target():
    text, status = m.isolate_target_clause(
        "First: Where was A born? Second: What language does B use?",
        "A",
        ["A", "B"],
    )
    assert status == "ok"
    assert text == "Where was [TARGET]A[/TARGET] born?"


def test_isolate_second_clause_and_mark_target():
    text, status = m.isolate_target_clause(
        "First: Where was A born? Second: What language does B use?",
        "B",
        ["A", "B"],
    )
    assert status == "ok"
    assert text == "What language does [TARGET]B[/TARGET] use?"


def test_nonstructured_query_is_reported_unsupported():
    text, status = m.isolate_target_clause("Where was A born?", "A", ["A"])
    assert text is None
    assert status == "not_first_second_format"


def test_subject_in_both_clauses_is_not_silently_selected():
    text, status = m.isolate_target_clause(
        "First: Compare A with B. Second: Tell me about A.",
        "A",
        ["A", "B"],
    )
    assert text is None
    assert status == "target_not_unique_to_one_clause"
