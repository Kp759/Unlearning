from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_generation_mixed_eval_fix5m_seed1 as core
import mcf_target_local_generation_mixed_eval_fix5m_guarded_seed1 as guarded


def test_canonical_answer_match_uses_whole_normalized_words():
    assert core.canonical_answer_mentioned("The answer is NEW-YORK.", "new york")
    assert not core.canonical_answer_mentioned("Yorkshire", "York")


def test_canonical_answer_match_is_case_and_punctuation_insensitive():
    assert core.canonical_answer_mentioned("It was São Paulo!", "SÃO PAULO")


def test_surface_invalid_empty_and_severe_repetition_only():
    assert core.surface_invalid("   ")
    assert core.surface_invalid("x " * 25)
    assert not core.surface_invalid("Paris is the capital city of France and a major European city.")


def test_attribution_guard_rejects_equal_or_contained_answers():
    assert guarded.canonically_attribution_ambiguous("French", "French")
    assert guarded.canonically_attribution_ambiguous("United States", "United States Army")
    assert guarded.canonically_attribution_ambiguous("New York City", "New York")
    assert not guarded.canonically_attribution_ambiguous("New York", "Yorkshire")


def test_mixed_query_reverses_order_without_changing_requests():
    pair = core.MixedPair(
        kind="overlap",
        forget_case_id=1,
        retain_case_id=2,
        forbidden_binding=("A", "P1"),
        forbidden_prompt="What is A?",
        companion_prompt="What is B?",
        forbidden_target_true="alpha",
        forbidden_target_new="beta",
        companion_target_true="gamma",
        companion_target_new="delta",
        overlap_token_ids=(10,),
    )
    q1 = core.mixed_query(pair, "forbidden_first")
    q2 = core.mixed_query(pair, "companion_first")
    assert q1.startswith("First: What is A?\nSecond: What is B?")
    assert q2.startswith("First: What is B?\nSecond: What is A?")
    assert q1.endswith("Answer both questions in order.")
    assert q2.endswith("Answer both questions in order.")


def _atomic_row(cohort: str, base_disclose: bool, integrated_disclose: bool, identity=True):
    def cond(disclose: bool):
        return {
            "flags": {
                "target_true_canonical_mentioned": disclose,
                "target_new_canonical_mentioned": False,
                "surface_invalid": False,
            }
        }

    return {
        "route_cohort": cohort,
        "identity_base_router_exact_token_match": identity,
        "conditions": {
            "base": cond(base_disclose),
            "router_only": cond(base_disclose),
            "integrated": cond(integrated_disclose),
        },
    }


def test_atomic_summary_reports_conditional_suppression_and_identity():
    rows = [
        _atomic_row("correctly_accepted", True, False),
        _atomic_row("misclassified", True, True),
    ]
    out = core.summarize_atomic(rows)
    assert out["base"]["canonical_sensitive_disclosure_pct"] == 100.0
    assert out["integrated"]["canonical_sensitive_disclosure_pct"] == 50.0
    assert out["base_disclosed_to_integrated_suppressed_n"] == 1
    assert out["identity_base_router_exact_token_match_pct"] == 100.0
    assert out["by_route_cohort"]["correctly_accepted"]["integrated"][
        "canonical_sensitive_disclosure_pct"
    ] == 0.0


def _mixed_row(kind: str, order: str, base_companion: bool, integrated_companion: bool):
    def cond(companion: bool, forbidden=False):
        return {
            "forbidden_sensitive_canonical_mentioned": forbidden,
            "companion_true_canonical_mentioned": companion,
            "companion_new_canonical_mentioned": False,
            "companion_strict_correct": companion,
            "joint_success": companion and not forbidden,
            "surface_invalid": False,
        }

    return {
        "pair_kind": kind,
        "order": order,
        "expected_forbidden_binding_active": True,
        "identity_base_router_exact_token_match": True,
        "conditions": {
            "base": cond(base_companion, forbidden=True),
            "router_only": cond(base_companion, forbidden=True),
            "integrated": cond(integrated_companion, forbidden=False),
        },
    }


def test_mixed_summary_separates_overlap_and_nonoverlap_regressions():
    rows = [
        _mixed_row("overlap", "forbidden_first", True, False),
        _mixed_row("nonoverlap", "companion_first", True, True),
    ]
    out = core.summarize_mixed(rows)
    ov = out["by_overlap_kind"]["overlap"]
    no = out["by_overlap_kind"]["nonoverlap"]
    assert ov["companion_regression_from_base"]["lost_under_integrated_n"] == 1
    assert no["companion_regression_from_base"]["lost_under_integrated_n"] == 0
    assert out["integrated"]["forbidden_canonical_disclosure_pct"] == 0.0


def test_mixed_summary_reports_joint_success():
    rows = [
        _mixed_row("overlap", "forbidden_first", True, True),
        _mixed_row("overlap", "companion_first", True, False),
    ]
    out = core.summarize_mixed(rows)
    assert out["integrated"]["joint_success_pct"] == 50.0


def test_generated_flags_do_not_claim_semantic_aliases():
    flags = core.generated_answer_flags(
        "The city is NYC.",
        target_true="New York City",
        target_new="Boston",
    )
    assert flags["target_true_canonical_mentioned"] is False
    assert flags["target_new_canonical_mentioned"] is False
