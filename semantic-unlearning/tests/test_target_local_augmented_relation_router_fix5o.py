from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_augmented_relation_router_fix5o_v2_seed1 as v2

core = v2.core


def _owner(subject: str, relation: str, cid: int):
    return core.Row(
        text=f"{subject} canonical {relation}",
        subject=subject,
        relation=relation,
        forbidden=True,
        kind="fix5_fit",
        family="canonical_cloze",
        case_id=cid,
    )


def test_fit_and_heldout_formulation_families_are_disjoint():
    fit = {x[0] for x in core.FIT_FORMULATIONS}
    held = {x[0] for x in core.HELDOUT_FORMULATIONS}
    assert fit
    assert held
    assert fit.isdisjoint(held)


def test_augmentation_keeps_owner_positive_forbidden_metadata():
    labels = {"P1": "alpha relation", "P2": "beta relation", "P3": "gamma relation"}
    fit, held, audit = core.build_augmentation_rows(
        [_owner("Entity A", "P1", 1)], set(labels), labels, hard_negatives_per_fact=2
    )
    positives = [r for r in fit if r.kind == "fix5o_augmented_relation_fit"]
    assert positives
    assert all(r.forbidden is True for r in positives)
    held_pos = [r for r in held if r.kind == "fix5o_augmented_relation_heldout"]
    assert held_pos
    assert all(r.forbidden is True for r in held_pos)
    assert audit["official_mcf_paraphrases_used"] is False
    assert audit["answer_values_used"] is False


def test_same_subject_contrasts_keep_alternate_relation_not_none():
    labels = {"P1": "alpha relation", "P2": "beta relation", "P3": "gamma relation"}
    fit, held, _ = core.build_augmentation_rows(
        [_owner("Entity A", "P1", 1)], set(labels), labels, hard_negatives_per_fact=2
    )
    contrasts = [r for r in fit if "same_subject_different_relation" in r.kind]
    assert len(contrasts) == 2
    assert all(r.subject == "Entity A" for r in contrasts)
    assert all(r.relation in {"P2", "P3"} for r in contrasts)
    assert all(r.relation != core.NONE for r in contrasts)
    assert all(r.forbidden is False for r in contrasts)
    held_contrast = [r for r in held if "same_subject_different_relation" in r.kind]
    assert held_contrast
    assert all(r.forbidden is False for r in held_contrast)


def test_relation_formulations_are_not_trivial_prefix_only_variants():
    rendered = {
        template.format(subject="Entity A", label="country of citizenship")
        for _, template in core.FIT_FORMULATIONS
    }
    assert len(rendered) == len(core.FIT_FORMULATIONS)
    assert any(x.startswith("Regarding") for x in rendered)
    assert any(x.startswith("Identify") for x in rendered)
    assert any("reported as" in x for x in rendered)


def test_target_local_selected_text_marks_subject_for_augmentation():
    row = core.render_row(
        "Entity A", "P1", 1, "aug_relation_report",
        "For {subject}, state the {label}.", "alpha relation",
        kind="fix5o_augmented_relation_fit",
    )
    view = core.target_local_views([row], ["Entity A"])[0]
    assert view.scope_supported is True
    assert "[TARGET]Entity A[/TARGET]" in view.selected_text


def test_dedup_rejects_conflicting_labels():
    r1 = core.Row("Q", "Entity A", "P1", False, "x", "f", 1)
    r2 = core.Row("Q", "Entity A", "P2", False, "x", "f", 2)
    V = core.RoutingView("same", "ok", (0, 4), True, ("Entity A",))
    try:
        core.dedup_rows_by_selected_text([r1, r2], [V, V])
    except RuntimeError as exc:
        assert "label conflict" in str(exc)
    else:
        raise AssertionError("expected conflicting labels to fail closed")


def test_leakage_filter_drops_exact_and_near_heldout_overlap():
    r1 = core.Row("Q1", "Entity A", "P1", False, "x", "f", 1)
    r2 = core.Row("Q2", "Entity A", "P1", False, "x", "f", 2)
    v1 = core.RoutingView("alpha beta gamma", "ok", (0, 1), True, ("Entity A",))
    v2x = core.RoutingView("different tokens here", "ok", (0, 1), True, ("Entity A",))
    rows, views, audit = core.filter_fit_against_heldout(
        [r1, r2], [v1, v2x], ["alpha beta gamma"], 0.90
    )
    assert len(rows) == 1
    assert rows[0].text == "Q2"
    assert audit["exact_overlap_dropped_n"] == 1


def test_authored_holdout_is_not_reinterpreted_as_policy_benchmark():
    rows = [
        core.Row(
            "For Entity A, state alpha relation.", "Entity A", "P1", True,
            "fix5o_augmented_relation_heldout", "h", 1,
        )
    ]
    out = v2.policy_report(rows, None, [], 0.0, ["P1", core.NONE], 1, {("Entity A", "P1")})
    assert out["status"] == "NOT_A_POLICY_BENCHMARK"
