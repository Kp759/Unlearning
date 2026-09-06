from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_local_typed_masking_ablation_fix5k_seed1.py"
spec = importlib.util.spec_from_file_location("fix5k_typed", SCRIPT)
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


class FakeTokenizer:
    chat_template = None

    def __call__(self, text, add_special_tokens=False):
        table = {str(i): i for i in range(1, 10)}
        table.update({" " + str(i): 100 + i for i in range(1, 10)})
        if text in table:
            return {"input_ids": [table[text]]}
        return {"input_ids": [999, 998]}


def test_type_prompt_contains_subject_and_fixed_menu_only():
    tok = FakeTokenizer()
    prompt = m.subject_type_prompt("Belgium", tok)
    assert "Belgium" in prompt
    assert "PERSON" in prompt
    assert "COUNTRY_OR_TERRITORY" in prompt
    assert "relation_id" not in prompt
    assert "P463" not in prompt


def test_option_token_ids_accepts_single_token_digits():
    ids, prefix = m.option_token_ids(FakeTokenizer())
    assert ids == list(range(1, 10))
    assert prefix == ""


def test_typed_marker_is_coarse_and_contains_no_subject_name():
    assert m.typed_marker("PERSON") == "[TARGET_PERSON]"
    assert "Belgium" not in m.typed_marker("COUNTRY_OR_TERRITORY")


def test_apply_typed_target_removes_exact_identity_but_preserves_selector_contract():
    v = m.RoutingView(
        selected_text="Which organization is [TARGET]Belgium[/TARGET] a member of?",
        selection_status="ATOMIC_PASSTHROUGH",
        selected_character_offsets=(0, 52),
        scope_supported=True,
        enumerated_subjects=("Belgium",),
    )
    out = m.apply_typed_target(v, "Belgium", "COUNTRY_OR_TERRITORY")
    assert out.selected_text == "Which organization is [TARGET_COUNTRY_OR_TERRITORY] a member of?"
    assert "Belgium" not in out.selected_text
    assert out.selection_status == v.selection_status
    assert out.scope_supported is True
    assert out.selected_character_offsets == v.selected_character_offsets
    assert out.enumerated_subjects == v.enumerated_subjects


def test_apply_typed_target_fails_closed_if_exact_marker_missing():
    v = m.RoutingView(
        selected_text="Which organization is Belgium a member of?",
        selection_status="ATOMIC_PASSTHROUGH",
        selected_character_offsets=(0, 44),
        scope_supported=True,
        enumerated_subjects=("Belgium",),
    )
    out = m.apply_typed_target(v, "Belgium", "COUNTRY_OR_TERRITORY")
    assert out.scope_supported is False
    assert "TYPED_MARKER_MISSING" in out.selection_status


def test_typed_view_uses_subject_only_type_map_and_same_selector_scope():
    r = row("Which organization is Belgium a member of?", "Belgium", "P463", forbidden=True)
    type_map = {
        "Belgium": {
            "type": "COUNTRY_OR_TERRITORY",
            "option_probability": 0.9,
            "option_logit_margin": 3.0,
        }
    }
    exact = m.exact_view(r, ["Belgium"])
    typed = m.typed_view(r, ["Belgium"], type_map)
    assert exact.scope_supported == typed.scope_supported
    assert exact.selection_status == typed.selection_status
    assert "Belgium" in exact.selected_text
    assert "Belgium" not in typed.selected_text
    assert "[TARGET_COUNTRY_OR_TERRITORY]" in typed.selected_text


def test_representation_conflicts_reports_typed_collision_without_dropping_rows():
    rows = [
        row("Belgium is in", "Belgium", "P30", case_id=1),
        row("France is in", "France", "P276", case_id=2),
    ]
    views = [
        m.RoutingView("[TARGET_COUNTRY_OR_TERRITORY] is in", "ATOMIC_PASSTHROUGH", (0, 13), True, ("Belgium",)),
        m.RoutingView("[TARGET_COUNTRY_OR_TERRITORY] is in", "ATOMIC_PASSTHROUGH", (0, 12), True, ("France",)),
    ]
    report = m.representation_conflicts(rows, views)
    assert report["row_n"] == 2
    assert report["unique_text_n"] == 1
    assert report["conflicting_text_unique_n"] == 1
    assert report["conflicting_row_n"] == 2
    assert set(report["conflicts"][0]["relations"]) == {"P30", "P276"}


def test_feature_index_reuses_collapsed_typed_text_but_keeps_row_indices():
    views = [
        m.RoutingView("[TARGET_PERSON] plays", "ATOMIC_PASSTHROUGH", (0, 1), True, ("A",)),
        m.RoutingView("[TARGET_PERSON] plays", "ATOMIC_PASSTHROUGH", (0, 1), True, ("B",)),
    ]
    texts, idx = m.feature_index({"fit": views})
    assert texts == ["[TARGET_PERSON] plays"]
    assert idx["fit"] == [0, 0]


def test_identity_hash_ignores_representation_and_depends_on_original_rows():
    a = [row("Belgium is in", "Belgium", "P30", case_id=1)]
    b = [row("Belgium is in", "Belgium", "P30", case_id=1)]
    assert m.identity_hash(a) == m.identity_hash(b)


def test_type_label_menu_has_unique_fixed_categories():
    assert len(m.TYPE_LABELS) == 9
    assert len(set(m.TYPE_LABELS)) == 9
    assert m.TYPE_LABELS[-1] == "OTHER"
