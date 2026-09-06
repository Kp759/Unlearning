from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_relation_classifier_fix5b_seed1.py"
spec = importlib.util.spec_from_file_location("fix5b", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

Row = m.Row


def row(masked: str, relation: str, phase_kind: str = "x", *, forbidden: bool = False) -> Row:
    return Row(
        text=masked,
        subject="Belgium",
        relation=relation,
        forbidden=forbidden,
        kind=phase_kind,
        family="canonical_cloze",
        case_id=1,
        masked=masked,
        candidate=True,
    )


def test_conflicting_masked_labels_are_quarantined_not_relabelled():
    parts = {
        "fit": [row("TARGET_ENTITY, the", "P463"), row("TARGET_ENTITY, the", "P30")],
        "calib": [row("Which continent contains TARGET_ENTITY?", "P30")],
        "validation": [row("Where was TARGET_ENTITY born?", "P19")],
    }
    out, report = m.separate(parts)
    assert out["fit"] == []
    assert report["masked_label_conflict_unique_n"] == 1
    assert report["masked_label_conflict_rows_dropped"]["fit"] == 2
    item = report["masked_label_conflicts"][0]
    assert item["masked"] == "TARGET_ENTITY, the"
    assert item["relations"] == ["P30", "P463"]


def test_exact_cross_partition_overlap_is_kept_only_in_earliest_partition():
    q = "Which organization is TARGET_ENTITY a member of?"
    parts = {
        "fit": [row(q, "P463")],
        "calib": [row(q, "P463")],
        "validation": [row("What is TARGET_ENTITY's native language?", "P103")],
    }
    out, report = m.separate(parts)
    assert len(out["fit"]) == 1
    assert out["calib"] == []
    assert report["masked_overlap_dropped"]["calib"] == 1


def test_same_phase_same_label_policy_distinct_rows_survive_separation():
    q = "Which organization is TARGET_ENTITY a member of?"
    parts = {
        "fit": [
            row(q, "P463", "fix5_fit", forbidden=True),
            row(q, "P463", "same_subject_different_relation", forbidden=False),
        ],
        "calib": [],
        "validation": [],
    }
    out, report = m.separate(parts)
    assert len(out["fit"]) == 2
    assert report["masked_overlap_dropped"].get("fit", 0) == 0
    # Semantic training uses one copy, policy evaluation retains the two roles.
    assert len(m.base.dedup_sem(out["fit"])) == 1
    assert len(m.base.dedup_policy(out["fit"])) == 2


def test_nonconflicting_rows_survive():
    parts = {
        "fit": [row("Which organization is TARGET_ENTITY a member of?", "P463")],
        "calib": [row("On which continent is TARGET_ENTITY located?", "P30")],
        "validation": [row("What is TARGET_ENTITY's native language?", "P103")],
    }
    out, report = m.separate(parts)
    assert {k: len(v) for k, v in out.items()} == {"fit": 1, "calib": 1, "validation": 1}
    assert report["masked_label_conflict_unique_n"] == 0
