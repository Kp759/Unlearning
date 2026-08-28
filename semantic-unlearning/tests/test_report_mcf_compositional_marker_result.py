from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import report_mcf_compositional_marker_result as report


def payload(forget, retain, ppl):
    return {"forget": forget, "retain": retain, "PPL": ppl}


def metrics(eff, gen, spe, spe_success):
    return {
        "Eff": eff,
        "Gen": gen,
        "Spe": spe,
        "Spe_success": spe_success,
    }


def test_report_passes_exact_target_with_matched_locality_and_ppl():
    base = payload(
        metrics(84.0, 85.0, 12.57, 88.6),
        metrics(87.3, 85.0, 11.84, 84.82),
        16.625,
    )
    edited = payload(
        metrics(0.0, 0.0, 12.57, 88.6),
        metrics(87.3, 85.0, 11.82, 84.8),
        16.625,
    )
    result = report.build_report(
        base,
        edited,
        {"acceptance": {"passed": True}, "reader_gate": {"passed": True}},
        {"passed": True, "checkpoint_was_reloaded": True},
        max_abs_spe_delta=0.2,
        max_ppl_percent_delta=5.0,
    )

    assert result["status"] == "PASS"
    assert result["checks"]["Eff_zero"] is True
    assert result["checks"]["Gen_zero"] is True
    assert result["delta"]["PPL"] == 0.0


def test_report_fails_when_unseen_generalization_is_not_zero():
    base = payload(
        metrics(84.0, 85.0, 12.57, 88.6),
        metrics(87.3, 85.0, 11.84, 84.82),
        16.625,
    )
    edited = payload(
        metrics(0.0, 4.0, 12.57, 88.6),
        metrics(87.3, 85.0, 11.84, 84.82),
        16.625,
    )
    result = report.build_report(
        base,
        edited,
        {"acceptance": {"passed": True}, "reader_gate": {"passed": True}},
        {"passed": True, "checkpoint_was_reloaded": True},
        max_abs_spe_delta=0.2,
        max_ppl_percent_delta=5.0,
    )

    assert result["status"] == "FAIL"
    assert result["checks"]["Gen_zero"] is False


def test_report_rejects_a_checkpoint_that_failed_fresh_process_reload():
    base = payload(
        metrics(84.0, 85.0, 12.57, 88.6),
        metrics(87.3, 85.0, 11.84, 84.82),
        16.625,
    )
    edited = payload(
        metrics(0.0, 0.0, 12.57, 88.6),
        metrics(87.3, 85.0, 11.84, 84.82),
        16.625,
    )
    result = report.build_report(
        base,
        edited,
        {"acceptance": {"passed": True}, "reader_gate": {"passed": True}},
        {"passed": False, "checkpoint_was_reloaded": True},
        max_abs_spe_delta=0.2,
        max_ppl_percent_delta=5.0,
    )

    assert result["status"] == "FAIL"
    assert result["checks"]["post_reload_acceptance_passed"] is False


def test_report_rejects_retain_regression_even_when_forget_is_zero():
    base = payload(
        metrics(84.0, 85.0, 12.57, 88.6),
        metrics(87.3, 85.0, 11.84, 84.82),
        16.625,
    )
    edited = payload(
        metrics(0.0, 0.0, 12.57, 88.6),
        metrics(90.0, 85.0, 11.84, 84.82),
        16.625,
    )
    result = report.build_report(
        base,
        edited,
        {"acceptance": {"passed": True}, "reader_gate": {"passed": True}},
        {"passed": True, "checkpoint_was_reloaded": True},
        max_abs_spe_delta=0.2,
        max_ppl_percent_delta=5.0,
    )

    assert result["status"] == "FAIL"
    assert result["checks"]["abs_retain_Eff_delta_within_limit"] is False
