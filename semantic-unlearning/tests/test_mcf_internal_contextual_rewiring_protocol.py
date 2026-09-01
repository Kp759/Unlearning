from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = ROOT / "protocols"


def _load(name: str) -> dict:
    return json.loads((PROTOCOLS / name).read_text())


def test_internal_rewiring_registry_locks_primary_architecture() -> None:
    registry = _load("mcf_internal_contextual_rewiring_v1_registry.json")
    architecture = registry["architecture"]
    editable = registry["editable_parameters"]

    assert registry["status"] == "terminal_rejected_training_only_classifier_preflight"
    assert registry["terminal_training_result"]["failure_stage"] == "calibration_positive_coverage"
    assert registry["terminal_training_result"]["actuator_constructed"] is False
    assert registry["terminal_training_result"]["official_evaluation_prompts_seen"] == 0
    assert architecture["external_string_router"] is False
    assert architecture["inference_sidecar"] is False
    assert architecture["subject_code_rank"] == 8
    assert architecture["detector_neurons_per_fact"] == 8
    assert architecture["actuator_neurons_per_fact"] == 16
    assert architecture["candidate_layers"] == [8, 12, 16, 20]
    assert architecture["detector_actuator_disjoint"] is True
    assert architecture["lm_head_untied_before_embedding_training"] is True
    assert architecture["lm_head_mutated"] is False
    assert editable["joint_subject_embedding_code"] is True
    assert editable["target_true_lm_head_rows"] is False
    assert editable["target_new_lm_head_rows"] is False


def test_overlap_and_classifier_contracts_are_contextual() -> None:
    registry = _load("mcf_internal_contextual_rewiring_v1_registry.json")
    overlap = registry["overlapping_subword_policy"]
    classifier = registry["classifier"]

    assert overlap["one_coherent_delta_per_token_row"] is True
    assert overlap["full_subject_coverage_required"] is True
    assert overlap["rarest_row_fallback_as_primary_policy"] is False
    assert classifier["output_semantics"] == "canonical_multi_label_fact_vector"
    assert classifier["fact_score_requires_subject_and_relation_evidence"] is True
    assert classifier["independent_full_hidden_vector_per_fact_forbidden"] is True
    assert "same_subject_different_relation" in classifier["hard_negative_families"]
    assert (
        "shared_subject_subword_without_complete_subject"
        in classifier["hard_negative_families"]
    )
    assert classifier["mandatory_certificate_hard_negative_families"] == [
        "same_subject_different_relation",
        "same_relation_different_subject",
        "shared_subject_subword_without_complete_subject",
        "broad_corpus_prompt",
        "writer_off_positive_context",
    ]
    assert (
        "alias_of_different_entity" in classifier["conditional_hard_negative_families"]
    )


def test_threshold_contract_has_an_unseen_statistical_certificate() -> None:
    registry = _load("mcf_internal_contextual_rewiring_v1_registry.json")
    threshold = registry["threshold_calibration"]
    cells = threshold["minimum_disjoint_negative_certification_cells"]

    assert threshold["legacy_raw_0_20_0_25_thresholds_reused"] is False
    assert threshold["threshold_frozen_before_certification"] is True
    assert threshold["maximum_certification_false_positive_cells"] == 0
    assert cells >= 300_000
    assert threshold["minimum_distinct_negative_certification_prompts"] >= 6_000
    assert (
        3.0 / cells
        <= threshold["conditional_cell_level_rule_of_three_95_percent_fpr_upper_bound"]
    )
    assert threshold["cell_independence_assumed"] is False
    assert threshold["ambiguous_or_unknown_state"] == "gate_closed"


def test_external_sidecar_lineage_is_terminal_control_only() -> None:
    retirement = _load("mcf_external_sidecar_lineage_retirement_v1.json")

    assert retirement["status"] == "terminal_research_architecture_retired_control_only"
    assert retirement["historical_artifacts_preserved"] is True
    assert retirement["new_candidate_builds_permitted"] is False
    assert retirement["new_seed_evaluations_permitted"] is False
    assert retirement["strong_unlearning_claim_permitted"] is False
    assert retirement["frozen_historical_registries_are_not_rewritten"] is True
    assert {
        "mcf_exact_subject_target_logit_sidecar_v5_0",
        "mcf_normalization_preserving_entity_sidecar_v6_0",
        "zsre_exact_subject_normalization_preserving_sidecar_v6_0",
    }.issubset(set(retirement["retired_protocols"]))
