from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_rsnr_v1a_official_eval_fresh_retain as ev  # noqa: E402


def _record(subject: str, relation: str):
    return {
        "case_id": 1,
        "requested_rewrite": {
            "subject": subject,
            "relation_id": relation,
            "prompt": "The capital of {} is",
            "target_true": {"str": "Brussels"},
            "target_new": {"str": "Paris"},
        },
        "paraphrase_prompts": ["What is Belgium's capital?"],
        "neighborhood_prompts": ["The capital of France is"],
    }


def test_oracle_eval_routes_matching_fact_rewrite_and_paraphrase_only():
    record = _record("Belgium", "P36")
    flags = ev.routing_flags_for_record(record, {("Belgium", "P36")})
    assert flags == {
        "rewrite": True,
        "paraphrase": True,
        "neighborhood": False,
    }


def test_oracle_eval_keeps_nonmatching_fact_fully_on_base_path():
    record = _record("Belgium", "P38")
    flags = ev.routing_flags_for_record(record, {("Belgium", "P36")})
    assert flags == {
        "rewrite": False,
        "paraphrase": False,
        "neighborhood": False,
    }


def test_same_relation_different_subject_does_not_fire():
    record = _record("France", "P36")
    flags = ev.routing_flags_for_record(record, {("Belgium", "P36")})
    assert not any(flags.values())


def test_evaluator_documents_target_new_as_metric_reference_not_training_target():
    source = Path(ev.__file__).read_text(encoding="utf-8")
    assert '"target_new_used_for_training": False' in source
    assert '"official_eff_gen_still_compare_target_true_vs_target_new": True' in source
