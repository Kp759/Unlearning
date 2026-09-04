from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_private_vocab_rewiring_v1_4_relation_null_core as core  # noqa: E402


def _record(case_id: int, subject: str, relation: str, prompt: str):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "subject": subject,
            "relation_id": relation,
            "prompt": prompt,
            "target_true": {"str": "TRUE"},
            "target_new": {"str": "FAKE"},
        },
    }


def test_gate_examples_use_five_locked_positives_and_different_relation_negatives():
    forget = [_record(7, "Belgium", "P36", "The capital of {} is")]
    views = {
        7: [
            "The capital of {} is",
            "{}'s capital is",
            "The capital city of {} is",
            "For {}, the capital is",
            "Which city is the capital of {}?",
        ]
    }
    protection = [
        _record(100, "X", "P36", "The capital of {} is"),
        _record(101, "X", "P38", "The currency used by {} is"),
        _record(102, "X", "P30", "{} is located on the continent of"),
    ]
    rows = core.build_gate_examples(
        forget, views, protection, negatives_per_case=2, seed=1
    )
    positives = [row for row in rows if row.label == 1]
    negatives = [row for row in rows if row.label == 0]
    assert len(positives) == 5
    assert len(negatives) == 2
    assert all(row.case_id == 7 and row.subject == "Belgium" for row in rows)
    assert {row.relation_id for row in negatives} == {"P30", "P38"}
    assert all(row.relation_id != "P36" for row in negatives)
    assert all("Belgium" in row.text for row in rows)


def test_gate_threshold_calibration_requires_per_case_separation():
    logits = torch.tensor([3.0, 2.0, -1.0, -2.0, 5.0, 4.0, 0.0, -3.0])
    labels = torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    case_ids = [10, 10, 10, 10, 20, 20, 20, 20]
    thresholds, metrics = core.calibrate_case_thresholds(logits, labels, case_ids)
    assert metrics["accuracy"] == 1.0
    assert metrics["all_cases_perfectly_separable"] is True
    assert -1.0 < thresholds[10] < 2.0
    assert 0.0 < thresholds[20] < 4.0
    predictions = core.gate_predictions(logits, case_ids, thresholds)
    assert predictions.tolist() == [True, True, False, False, True, True, False, False]


def test_registry_encodes_relation_aware_null_not_fake_target():
    registry = json.loads(
        (ROOT / "protocols" / "mcf_private_vocab_rewiring_v1_4_relation_null_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["protocol"] == core.PROTOCOL
    assert registry["architecture"]["relation_aware_router"] is True
    assert registry["architecture"]["non_sensitive_path"] == "exact_base_token_ids"
    assert registry["architecture"]["transformer_frozen"] is True
    assert registry["architecture"]["lm_head_frozen_bit_identical"] is True
    assert registry["relation_gate"]["official_eval_text_used"] is False
    assert registry["relation_gate"]["positive_views_per_case"] == 5
    assert registry["relation_gate"]["forgotten_relation_excluded_from_negatives"] is True
    assert registry["null_objective"]["abstention_text"] == "I don't know"
    assert registry["null_objective"]["target_new_gradient"] is False
    assert registry["null_objective"]["internal_reserved_ids_are_output_targets"] is False
    assert registry["acceptance"]["minimum_abstention_minus_true_margin"] == 0.1
    assert registry["acceptance"]["minimum_true_logprob_drop"] == 2.0
    assert registry["acceptance"]["relative_private_row_cap"] == 0.5
    assert registry["leakage_firewall"]["final_certification_requires_new_untouched_seed"] is True
    assert registry["claim_boundary"]["latent_knowledge_erasure"] is False


def test_launcher_reuses_locked_five_view_protocol_and_unsets_full_mcf_env():
    text = (
        ROOT / "scripts" / "run_mcf_private_vocab_rewiring_v1_4_relation_null_manual.sh"
    ).read_text(encoding="utf-8")
    assert "training_visible_multiview_forget.json" in text
    assert "training_visible_forget_direct.json" in text
    assert "training_visible_protection_fit_direct.json" in text
    assert "unset MCF_PATH" in text
    assert "--mcf-path" not in text
    assert "--minimum-abstention-margin 0.1" in text
    assert "--minimum-true-suppression 2.0" in text
    assert "--relative-row-cap 0.5" in text
