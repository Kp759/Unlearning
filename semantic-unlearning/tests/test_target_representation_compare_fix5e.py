from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_representation_compare_fix5e_seed1.py"
spec = importlib.util.spec_from_file_location("fix5e_target_rep", SCRIPT)
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
    masked: str = "",
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
        masked=masked,
        candidate=candidate,
    )


def test_target_preserving_mark_keeps_name_and_marks_span():
    text, found = m.target_preserving_text(
        "Which organization is Belgium a member of?",
        "Belgium",
        ["Belgium", "France"],
    )
    assert found
    assert text == "Which organization is [TARGET]Belgium[/TARGET] a member of?"
    assert "TARGET_ENTITY" not in text


def test_other_registered_subjects_keep_other_entity_treatment():
    text, found = m.target_preserving_text(
        "Compare Belgium with France.",
        "Belgium",
        ["Belgium", "France"],
    )
    assert found
    assert text == "Compare [TARGET]Belgium[/TARGET] with OTHER_ENTITY."


def test_marked_row_changes_only_representation_field():
    original = row(
        "What is Belgium's continent?",
        "Belgium",
        "P30",
        forbidden=True,
        kind="fix5_validation",
        family="conversational_question",
        case_id=99,
        masked="What is TARGET_ENTITY's continent?",
    )
    marked = m.marked_row(original, ["Belgium"])
    assert marked.masked == "What is [TARGET]Belgium[/TARGET]'s continent?"
    assert m.row_identity(marked) == m.row_identity(original)
    assert m.semantic_identity(marked) == m.semantic_identity(original)


def test_conflict_keys_detects_only_different_semantic_labels():
    parts = {
        "fit": [
            row("x", "A", "P30", masked="TARGET_ENTITY is in"),
            row("x2", "B", "P30", masked="TARGET_ENTITY is in"),
        ],
        "calib": [row("x3", "C", "P276", masked="TARGET_ENTITY is in")],
        "validation": [row("y", "D", "P463", masked="TARGET_ENTITY is a member of")],
    }
    conflicts = m.conflict_keys(parts)
    assert conflicts == {"target_entity is in": ["P276", "P30"]}


def test_coverage_rows_are_separate_and_subject_aware():
    parts = {
        "fit": [
            row("Ask A where it is.", "A", "P30", case_id=1, masked="Ask TARGET_ENTITY where it is."),
            row("Ask B where it is.", "B", "P276", case_id=2, masked="Ask TARGET_ENTITY where it is."),
        ],
        "calib": [],
        "validation": [],
    }
    conflicts = m.conflict_keys(parts)
    challenge = m.coverage_rows(parts, conflicts)
    assert len(challenge["fit"]) == 2
    assert {r.subject for r in challenge["fit"]} == {"A", "B"}


def test_marking_does_not_insert_relation_label():
    original = row(
        "Which organization is Belgium a member of?",
        "Belgium",
        "P463",
        masked="Which organization is TARGET_ENTITY a member of?",
    )
    marked = m.marked_row(original, ["Belgium"])
    assert "P463" not in marked.masked
    assert marked.relation == "P463"


def test_feature_index_shares_erased_text_but_not_distinct_marked_subjects():
    erased = [
        row("A is in", "A", "P30", masked="TARGET_ENTITY is in"),
        row("B is in", "B", "P30", masked="TARGET_ENTITY is in"),
    ]
    texts, idx = m.feature_index({"rows": erased})
    assert len(texts) == 1
    assert idx["rows"] == [0, 0]

    marked = m.mark_rows(erased, ["A", "B"])
    texts2, idx2 = m.feature_index({"rows": marked})
    assert len(texts2) == 2
    assert idx2["rows"] == [0, 1]


def test_contains_subsequence_for_marker_visibility():
    assert m.contains_subsequence([1, 2, 3, 4], [2, 3])
    assert not m.contains_subsequence([1, 2, 3, 4], [3, 2])
