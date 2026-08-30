from __future__ import annotations

import sys
from pathlib import Path
import json

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_embedding_keyed_neuron_core as core
import mcf_embedding_keyed_neuron_erasure as method
import audit_mcf_embedding_keyed_neuron_retain_tail as tail_audit
import audit_mcf_embedding_keyed_neuron_latent_recovery as latent_audit
import audit_mcf_embedding_keyed_neuron_relearning as relearning_audit
import audit_mcf_embedding_writer_shared_row_exposure as exposure_audit
import audit_mcf_frozen_writer_portability as portability_audit
import aggregate_mcf_context_gating_frequency_factorial as frequency_aggregate
import report_mcf_embedding_keyed_neuron_result as report
import report_mcf_detector_gate_cases as gate_cases
import verify_mcf_clean_stage1_writer as clean_writer_verify


class TinySwiGLU(torch.nn.Module):
    def __init__(self, hidden: int = 5, intermediate: int = 7):
        super().__init__()
        self.gate_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = torch.nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = torch.nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = torch.nn.functional.silu

    def forward(self, hidden):
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        )


def _tail_receipt(*, passed: bool = True, writer_mode: str = "embedding_keyed"):
    checks = {
        "minimum_unique_prompt_count": True,
        "response_rate_at_most_24_over_13000": passed,
        "response_wilson_upper_at_most_24_over_13000": passed,
        "top1_change_rate_at_most_24_over_13000": passed,
    }
    return {
        "kind": "mcf_embedding_keyed_neuron_post_freeze_retain_tail_audit",
        "dataset": "MCF",
        "seed": 1,
        "unlearn_num": 50,
        "retain_num": 9000,
        "sample_mode": "official",
        "writer_mode": writer_mode,
        "unique_prompts": True,
        "used_for_training_checkpoint_selection_or_retry": False,
        "groups": {"all": {"prompt_count": 100_000}},
        "acceptance": {"checks": checks, "passed": passed},
    }


def _portability_receipt(*, passed: bool = True):
    return {
        "kind": "mcf_frozen_stage1_writer_official_portability_audit",
        "dataset": "MCF",
        "seed": 1,
        "unlearn_num": 50,
        "sample_mode": "official",
        "writer_mode": "embedding_keyed",
        "threshold_binding_passed": True,
        "decoder_loaded": False,
        "writer_parameters_updated": False,
        "used_for_training_checkpoint_selection_or_retry": False,
        "prompt_count": 100,
        "record_count": 50,
        "by_prompt_type": {
            "rewrite": {"prompt_count": 50},
            "paraphrase": {"prompt_count": 50},
        },
        "acceptance": {"passed": passed},
    }


def _clean_stage1_acceptance_summary():
    return {
        "kind": "mcf_clean_stage1_writer_acceptance",
        "passed": True,
        "training_safe_portability": {
            "amplitude_threshold": 4.5,
            "prompt_count": 346,
            "complete_count": 346,
            "global_complete_fraction": 1.0,
            "minimum_record_complete_fraction": 1.0,
        },
    }


def _development_exposure_audit():
    return {
        "data_role": "consumed_development_evidence_not_blind_evaluation",
        "source_writer_protocol": method.compositional_core.PROTOCOL,
        "forget_records": 50,
        "reserved_official_retain_records_excluded": 1000,
        "development_retain_records_consumed": 9438,
        "forget_records_with_subject_word_overlap": 43,
        "unique_forget_subject_words": 110,
        "unique_forget_subject_words_reused": 65,
        "development_prompts_with_forget_subject_word": 4298,
        "forget_records_with_subject_subtoken_overlap": 50,
        "unique_forget_subject_token_ids": 236,
        "unique_forget_subject_token_ids_reused": 199,
        "development_prompts_with_forget_subject_subtoken": 5763,
        "actual_edited_embedding_rows": 234,
        "actual_edited_embedding_rows_reused": 198,
        "development_prompts_with_actual_edited_row": 5762,
        "forget_subjects_with_reused_actual_edited_row": 49,
        "official_evaluation_prompts_seen": 0,
    }


def _v3_5_1_forensic_scope():
    return {
        "training_only": True,
        "forensic_replay_only": True,
        "source_v3_5_rejection_hash_bound": True,
        "expected_writer_off_gate_cells": 17300,
        "expected_nonzero_writer_off_gate_cells": 1,
        "expected_source_case_id": 10803,
        "raw_signed_response_matrix_required": True,
        "source_context_index_and_provenance_required": True,
        "detector_group_and_owner_status_required": True,
        "threshold_calibration_prohibited": True,
        "detector_optimizer_construction_prohibited": True,
        "actuator_optimizer_construction_prohibited": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
        "ordinary_existing_weight_materialization_claimed": False,
        "downstream_architecture_change_checkpoint_controls_and_official_audits": (
            "deferred until the collision is identified and a separate fix is "
            "preregistered"
        ),
    }


def _v3_5_2_repair_scope():
    return {
        "training_only": True,
        "source_v3_5_1_forensics_hash_bound": True,
        "diagnosed_source_case_id": 10803,
        "diagnosed_source_context_index": 4,
        "diagnosed_detector_case_id": 17353,
        "diagnosed_detector_group_index": 30,
        "diagnosed_owner_group": False,
        "repair_targets_collision_identity_directly": False,
        "all_writer_off_groups_per_context": 50,
        "detector_repair_optimizer_updates": 100,
        "global_gate_required_before_actuator_feasibility": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }


def _v3_5_3_multilabel_scope():
    return {
        "training_only": True,
        "source_v3_5_2_rejection_hash_bound": True,
        "duplicate_prompt_sha256": (
            "9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38"
        ),
        "positive_source_case_id": 10472,
        "positive_source_context_index": 1,
        "negative_source_case_id": 19763,
        "negative_source_context_index": 4,
        "shared_detector_case_id": 10472,
        "repair_targets_prompt_identity_directly": False,
        "same_record_positive_negative_conflicts_allowed": False,
        "canonical_hidden_reuse_required": True,
        "global_worst_two_weight": 1.0,
        "detector_repair_optimizer_updates": 100,
        "multilabel_gate_required_before_actuator_feasibility": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }


def _v3_5_4_balanced_scope():
    return {
        "training_only": True,
        "source_v3_5_3_rejection_hash_bound": True,
        "source_v3_5_3_owner_gate_passed_records": 29,
        "source_v3_5_3_owner_gate_total_records": 50,
        "source_v3_5_3_positive_failures": 21,
        "source_v3_5_3_negative_failures": 0,
        "source_v3_5_3_writer_off_failures": 0,
        "canonical_hidden_reuse_required": True,
        "canonical_multilabel_semantics_unchanged": True,
        "complete_update_global_tail_optimization_weight": 0.0,
        "per_record_worst_two_retained": True,
        "all_writer_off_groups_per_context": 50,
        "first_update_component_gradient_audit_required": True,
        "detector_repair_optimizer_updates": 100,
        "multilabel_gate_required_before_actuator_feasibility": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }


def _v3_5_5_width_scope():
    return {
        "training_only": True,
        "source_v3_5_4_rejection_hash_bound": True,
        "source_v3_5_4_detector_passed_records": 50,
        "source_v3_5_4_detector_total_records": 50,
        "source_v3_5_4_actuator_cap": 1.5,
        "source_v3_5_4_actuator_columns": 200,
        "source_v3_5_4_saturated_columns": 147,
        "source_v3_5_4_positive_reachability_passed": False,
        "exact_detector_tensor_replay_required": True,
        "detector_neurons_per_record": 4,
        "actuator_widths_per_record": [4, 8, 16],
        "detector_actuator_neurons_disjoint": True,
        "nested_actuator_prefixes": True,
        "native_per_column_relative_cap": 1.5,
        "matched_width4_group_budget_controls": [8, 16],
        "matched_controls_used_for_width_selection": False,
        "optimizer_updates_per_arm": 100,
        "smallest_native_passing_width_selected": True,
        "every_fitted_actuator_discarded": True,
        "full_preservation_training_prohibited": True,
        "checkpoint_creation_prohibited": True,
        "official_evaluation_prohibited": True,
    }


def _detector_training_revision():
    return {
        "version": "v3.5.4_canonical_multilabel_balanced_tail_repair",
        "mode": "training_only_canonical_prompt_multilabel_balanced_repair",
        "primary_initialization": "frozen_v3_2",
        "source_v3_5_protocol": method.FROZEN_V3_5_PROTOCOL,
        "source_v3_5_1_protocol": method.FROZEN_V3_5_1_PROTOCOL,
        "source_v3_5_2_protocol": method.FROZEN_V3_5_2_PROTOCOL,
        "source_v3_5_3_protocol": method.FROZEN_V3_5_3_PROTOCOL,
        "source_v3_5_3_owner_gate_passed_records": 29,
        "source_v3_5_3_owner_gate_total_records": 50,
        "source_v3_5_3_positive_failures": 21,
        "source_protocol": method.FROZEN_DETECTOR_PROTOCOL,
        "source_training_revision": "v3.2",
        "source_gate_passed_records": 50,
        "source_gate_total_records": 50,
        "source_gate_passed": True,
        "source_optimizer_updates": 1000,
        "optimizer_constructed_this_run": True,
        "optimizer_updates_this_run": 100,
        "imported_tensors": ["gate_delta", "up_delta"],
        "source_down_delta_imported": False,
        "current_down_delta_reset_to_zero": True,
        "exact_source_artifact_hash_receipt_required": True,
        "fresh_full_context_source_replay_required_before_repair": True,
        "source_replay_abs_tolerance": (method.FROZEN_DETECTOR_REPLAY_ABS_TOLERANCE),
        "controls_initialization": "train",
        "control_optimizer_updates": 1000,
        "training_input": (
            "canonical_exact_prompt_cached_selected_layer_mlp_input_hidden_states"
        ),
        "cache_dtype": "float32",
        "cache_device": "cpu",
        "cache_scope": [
            "writer_on_positive",
            "writer_on_negative",
            "writer_off_positive",
        ],
        "duplicate_prompt_policy": (
            "one bit-identical hidden-state representative reused across every "
            "source-role occurrence"
        ),
        "prompt_label_semantics": (
            "active groups are every record for which the exact prompt is "
            "registered positive; source-relative negative roles leave those "
            "labels active"
        ),
        "update_coverage": "all_records_accumulated",
        "record_microbatch_argument": "detector_record_batch",
        "records_per_optimizer_update": 50,
        "optimizer_updates_total_per_detector": 100,
        "record_exposures": 5000,
        "positive_context_mode": "all",
        "negative_context_mode": "all",
        "tail_k": 2,
        "positive_objective": (
            "active_label_equal_record_mean_plus_worst_k_squared_shortfall"
        ),
        "negative_objective": (
            "source_owner_equal_record_mean_plus_worst_k_squared_excess"
        ),
        "cross_objective": (
            "inactive_label_equal_record_mean_plus_worst_k_squared_excess"
        ),
        "writer_off_objective": (
            "all_detector_groups_equal_source_record_mean_plus_worst_k_squared_excess"
        ),
        "writer_off_groups_per_context": 50,
        "global_tail_weight": 0.0,
        "global_tail_terms": "diagnostic_only_not_optimized",
        "training_positive_floor": 0.30,
        "training_off_abs_max": 0.15,
        "certificate_positive_floor": 0.25,
        "certificate_off_abs_max": 0.20,
        "certificate_abs_tolerance": 1e-7,
        "certificate_thresholds_unchanged_from_v3_1": True,
        "gradient_normalization": "equal_record_mean_plus_per_record_worst_k",
        "gradient_balance_audit": {
            "optimizer_step": 1,
            "parameter_grad_buffers_mutated": False,
            "components": [
                "positive_write",
                "source_negative",
                "inactive_cross",
                "writer_off_all_groups",
                "positive_consistency",
                "parameter_l2",
                "total",
            ],
        },
        "gradient_clip_frequency": "once_per_optimizer_update",
        "norm_projection_frequency": "once_per_optimizer_update",
        "endpoint_audit_phases": [
            "pre_update",
            "post_adam",
            "post_projection",
            "final_fresh_full_context_certificate",
        ],
        "complete_training_log_required": True,
        "official_evaluation_prompts_seen": 0,
    }


def _actuator_training_revision():
    return {
        "version": "v3.5.5",
        "mode": "discarded_separate_actuator_width_sweep",
        "source_v3_5_4_protocol": method.FROZEN_V3_5_4_PROTOCOL,
        "source_v3_5_4_artifact_hash_receipt_required": True,
        "source_v3_5_4_detector_tensor_hash_replay_required": True,
        "optimizer_constructed_this_run": True,
        "optimizer_updates_per_arm": 100,
        "frozen_training_contexts": {
            "writer_on_positive": 346,
            "writer_on_negative": 465,
            "writer_off_positive": 346,
        },
        "source_v3_5_4_rejection": {
            "detector_passed_records": 50,
            "detector_total_records": 50,
            "threshold_gate_passed": True,
            "positive_owner_gate_min": 1.0,
            "writer_off_gate_max": 0.0,
            "actuator_width": 4,
            "relative_norm_cap": 1.5,
            "saturated_columns": 147,
            "selected_columns": 200,
            "positive_reachability_passed": False,
        },
        "architecture": {
            "base_mlp_path": "bit_exact_untouched",
            "frozen_detector_features_per_record": 4,
            "actuator_features_per_record": [4, 8, 16],
            "detector_actuator_feature_sets_disjoint": True,
            "actuator_feature_sets_nested": True,
            "actuator_gate_up_rows": "frozen_unmodified_Base_rows",
            "residual_formula": (
                "BaseMLP(h) + threshold_gate_r(h) * BaseActuatorFeatures_r(h) @ "
                "down_delta_r.T"
            ),
            "off_boundary": 0.200001,
            "on_boundary": 0.249999,
            "threshold_gate": (
                "clip((response - off_boundary) / (on_boundary - off_boundary), 0, 1)"
            ),
            "down_delta_exact_zero_is_algebraic_identity": True,
            "original_base_down_columns_modified": False,
            "ordinary_existing_weight_materialization": False,
        },
        "feasibility": {
            "native_actuator_widths": [4, 8, 16],
            "native_per_column_relative_cap": 1.5,
            "matched_width4_group_budget_control_widths": [8, 16],
            "matched_controls_used_for_selection": False,
            "selection_rule": (
                "smallest native width with zero direct and positive failures"
            ),
            "initial_down_delta": "bit_exact_zero",
            "optimizer_state": "fresh_independent_adamw_per_arm",
            "optimizer_updates_per_arm": 100,
            "learning_rate": 5e-4,
            "records_per_optimizer_update": 50,
            "positive_contexts_per_optimizer_update": 346,
            "context_microbatch_capacity": 4,
            "tail_k": 2,
            "objective": (
                "equal_record_mean_plus_worst_two_squared_margin_shortfall_only"
            ),
            "forget_margin": 1.0,
            "gradient_clip_frequency": "once_per_optimizer_update",
            "norm_projection_frequency": "once_per_optimizer_update",
            "complete_incremental_training_log_required": True,
            "fresh_full_context_audit_required": True,
            "zero_actuator_identity_abs_tolerance": 1e-6,
            "positive_reachability_rule": (
                "direct_failures == 0 and positive_failures == 0"
            ),
            "writer_off_structural_selectivity_tolerance": 0.05,
            "every_fitted_down_delta_discarded": True,
            "full_preservation_objective_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
        },
        "official_evaluation_prompts_seen": 0,
    }


def _latent_receipt(*, fact_recoverable: bool):
    return {
        "kind": "mcf_embedding_keyed_neuron_post_freeze_latent_recovery_audit",
        "dataset": "MCF",
        "seed": 1,
        "unlearn_num": 50,
        "sample_mode": "official",
        "writer_mode": "embedding_keyed",
        "record_count": 50,
        "prompt_count": 100,
        "fact_recoverable": fact_recoverable,
        "used_for_training_checkpoint_selection_or_retry": False,
        "modes": {
            "edited": {"final_model_output": {}},
            "reconstructed_base": {"final_model_output": {}},
        },
        "positive_control_passed": True,
    }


def _relearning_receipt(*, fact_recoverable: bool):
    return {
        "kind": "mcf_embedding_keyed_neuron_post_freeze_relearning_attack",
        "dataset": "MCF",
        "seed": 1,
        "unlearn_num": 50,
        "sample_mode": "official",
        "writer_mode": "embedding_keyed",
        "record_count": 50,
        "fact_recoverable": fact_recoverable,
        "used_for_training_checkpoint_selection_or_retry": False,
        "attack": {"maximum_steps": 64},
        "curve": [{"step": 0}, {"step": 64}],
        "reconstructed_base_positive_control": {},
        "positive_control_passed": True,
    }


def test_record_owned_neuron_selection_is_disjoint_and_writer_sensitive():
    protected = torch.zeros(10, 8)
    protected[:, 6:] = 4.0
    off = [torch.zeros(3, 8), torch.zeros(3, 8)]
    writer = [off[0].clone(), off[1].clone()]
    writer[0][:, 0] = 3.0
    writer[0][:, 1] = -2.0
    writer[1][:, 2] = 4.0
    writer[1][:, 3] = 2.0

    ownership, signs, reports = core.select_record_owned_neurons(
        writer,
        off,
        protected,
        neurons_per_record=2,
        dormant_fraction=0.75,
    )

    assert set(ownership[0]).isdisjoint(ownership[1])
    assert set(ownership[0]) == {0, 1}
    assert set(ownership[1]) == {2, 3}
    assert signs[0].tolist() == [1.0, -1.0]
    assert len(reports) == 2


def test_shared_row_exposure_audit_statistics_handle_ties_and_delta_sources():
    assert exposure_audit.average_ranks([1.0, 1.0, 3.0]) == [1.5, 1.5, 3.0]
    correlation = exposure_audit.correlation_report([0, 2, 4], [1.0, 2.0, 3.0])
    assert correlation["pearson_frequency_vs_norm"] == pytest.approx(1.0)
    assert correlation["spearman_frequency_vs_norm"] == pytest.approx(1.0)

    direct, source = exposure_audit.tensor_from_state(
        {"embedding_delta": torch.tensor([[1.0, 2.0]])},
        direct_keys=("embedding_delta",),
        edited_key="edited",
        base_key="base",
    )
    assert source == "embedding_delta"
    assert torch.equal(direct, torch.tensor([[1.0, 2.0]]))

    paired, source = exposure_audit.tensor_from_state(
        {
            "edited": torch.tensor([[3.0, 5.0]]),
            "base": torch.tensor([[1.0, 2.0]]),
        },
        direct_keys=("missing",),
        edited_key="edited",
        base_key="base",
    )
    assert source == "edited-base"
    assert torch.equal(paired, torch.tensor([[2.0, 3.0]]))


def test_shared_row_exposure_materializes_only_direct_prompt_and_subject_words():
    record = {
        "requested_rewrite": {
            "prompt": "{} was born in",
            "subject": "John Smith",
        }
    }
    assert exposure_audit.materialize_prompt(record) == "John Smith was born in"
    assert exposure_audit.words("John Smith's Airport") == {
        "john",
        "smith",
        "s",
        "airport",
    }


def test_shared_row_exposure_main_excludes_reserved_retain_and_marks_dev_consumed(
    tmp_path, monkeypatch
):
    class FakeTokenizer:
        vocabulary = {
            "Reserved": 9,
            "John": 1,
            "works": 2,
            "at": 3,
            "Alpha": 4,
            "Other": 5,
            "place": 6,
            "was": 7,
            "born": 8,
        }

        def __call__(self, value, add_special_tokens=False):
            del add_special_tokens
            return {"input_ids": [self.vocabulary[token] for token in value.split()]}

        def convert_ids_to_tokens(self, token_id):
            return next(
                (
                    token
                    for token, value in self.vocabulary.items()
                    if value == token_id
                ),
                f"tok-{token_id}",
            )

        def decode(self, ids, **_kwargs):
            return self.convert_ids_to_tokens(ids[0])

    monkeypatch.setattr(
        exposure_audit.AutoTokenizer,
        "from_pretrained",
        lambda *_args, **_kwargs: FakeTokenizer(),
    )

    def record(prompt, person):
        return {"requested_rewrite": {"prompt": prompt, "subject": person}}

    raw = [
        record("Reserved {}", "John"),
        record("{} works at Alpha", "John"),
        record("Other place", "Nobody"),
        record("Other place", "Nobody"),
        record("{} was born", "John"),
        record("Other place", "Nobody"),
    ]
    mcf = tmp_path / "mcf.json"
    split = tmp_path / "split.json"
    state = tmp_path / "writer.pt"
    output = tmp_path / "audit.json"
    mcf.write_text(json.dumps(raw), encoding="utf-8")
    split.write_text(
        json.dumps(
            {
                "protocol": exposure_audit.LOCKED_SPLIT_PROTOCOL,
                "source_sha256": exposure_audit.sha256_file(mcf),
                "seed": 1,
                "sampling": {
                    "forget_num": 1,
                    "retain_eval_num": 1,
                    "forget_case_ids": [4],
                    "retain_eval_case_ids": [0],
                },
            }
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "protocol": exposure_audit.V6_2_WRITER_PROTOCOL,
            "seed": 1,
            "case_ids": [4],
            "selected_embedding_rows": [1, 42],
            "embedding_delta": torch.tensor([[3.0, 4.0], [0.0, 2.0]]),
        },
        state,
    )

    exposure_audit.main(
        [
            "--model-path",
            "fake-model",
            "--mcf-path",
            str(mcf),
            "--split-manifest",
            str(split),
            "--writer-state",
            str(state),
            "--output",
            str(output),
            "--top-k",
            "2",
        ]
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["data_firewall"]["development_retain_case_ids"] == [1, 2]
    assert report["data_firewall"]["reserved_official_retain_prompts_materialized"] == 0
    assert report["data_firewall"][
        "development_ids_must_never_be_described_as_blind_evaluation"
    ]
    assert report["actual_embedding_row_overlap"]["selected_rows_reused"] == 1
    assert (
        report["actual_embedding_row_overlap"]["development_prompts_with_selected_row"]
        == 1
    )
    assert report["embedding_row_norm_frequency_analysis"]["per_row"][0][
        "delta_l2_norm"
    ] == pytest.approx(5.0)


def test_dormant_random_selection_is_deterministic_and_disjoint():
    protected = torch.zeros(12, 10)
    off = [torch.zeros(2, 10), torch.zeros(2, 10)]
    writer = [torch.ones(2, 10), torch.ones(2, 10)]
    first, _, reports = core.select_record_owned_neurons(
        writer,
        off,
        protected,
        neurons_per_record=2,
        dormant_fraction=1.0,
        selection_mode="dormant_random",
        generator=torch.Generator().manual_seed(91),
    )
    second, _, _ = core.select_record_owned_neurons(
        writer,
        off,
        protected,
        neurons_per_record=2,
        dormant_fraction=1.0,
        selection_mode="dormant_random",
        generator=torch.Generator().manual_seed(91),
    )
    assert first == second
    assert set(first[0]).isdisjoint(first[1])
    assert reports[0]["selection_mode"] == "dormant_random"


def test_no_writer_selection_uses_base_context_contrast_without_embedding_delta():
    protected = torch.zeros(8, 6)
    positive = [torch.zeros(3, 6), torch.zeros(3, 6)]
    writer_off = [row.clone() for row in positive]
    negatives = [torch.zeros(4, 6), torch.zeros(4, 6)]
    positive[0][:, 1] = 3.0
    positive[1][:, 4] = -2.0

    ownership, signs, reports = core.select_record_owned_neurons(
        positive,
        writer_off,
        protected,
        context_negative_activations=negatives,
        neurons_per_record=1,
        dormant_fraction=1.0,
        selection_mode="base_context_contrastive",
    )

    assert ownership == [[1], [4]]
    assert signs[0].tolist() == [1.0]
    assert signs[1].tolist() == [-1.0]
    assert all(
        row["selection_score_kind"] == "base_positive_minus_context_negative"
        for row in reports
    )


def test_contextual_code_response_uses_owned_multibit_group():
    ownership = [[2, 4], [1, 7]]
    signs = [torch.tensor([1.0, -1.0]), torch.tensor([-1.0, 1.0])]
    flat, flat_signs, local = core.flatten_ownership(ownership, signs)
    assert flat == [2, 4, 1, 7]

    baseline = torch.zeros(2, 4)
    edited = torch.tensor([[3.0, -3.0, 0.0, 0.0], [0.0, 0.0, -2.0, 2.0]])
    response = core.contextual_code_responses(edited, baseline, local, flat_signs)
    assert torch.allclose(response, torch.tensor([[3.0, 0.0], [0.0, 2.0]]))


def test_absolute_group_response_is_the_runtime_actuator_gate():
    ownership = [[2, 4], [1, 7]]
    signs = [torch.tensor([1.0, -1.0]), torch.tensor([-1.0, 1.0])]
    _flat, flat_signs, local = core.flatten_ownership(ownership, signs)
    activations = torch.tensor([[3.0, -3.0, 2.0, -2.0]])
    response = core.signed_group_activations(activations, local, flat_signs)
    assert torch.allclose(response, torch.tensor([[3.0, -2.0]]))


def test_detector_objective_rewards_owned_positive_and_nulls_cross_codes():
    good = torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    owners = torch.tensor([0, 1, 0])
    positive = torch.tensor([True, True, False])
    good_loss, _ = core.detector_objective(
        good,
        owners,
        positive,
        positive_target=1.0,
        off_target_abs_max=0.2,
        tail_k=2,
        negative_weight=2.0,
        cross_weight=2.0,
    )
    bad = good.clone()
    bad[0, 1] = 3.0
    bad[2, 0] = 3.0
    bad_loss, _ = core.detector_objective(
        bad,
        owners,
        positive,
        positive_target=1.0,
        off_target_abs_max=0.2,
        tail_k=2,
        negative_weight=2.0,
        cross_weight=2.0,
    )
    assert good_loss == pytest.approx(0.0)
    assert bad_loss > good_loss


def test_detector_positive_loss_is_equal_record_mean_plus_worst_two():
    responses = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    owners = torch.tensor([0, 0, 0, 1, 1, 1])
    positive = torch.ones(6, dtype=torch.bool)
    loss, pieces = core.detector_objective(
        responses,
        owners,
        positive,
        positive_target=1.0,
        off_target_abs_max=0.2,
        tail_k=2,
        negative_weight=0.0,
        cross_weight=0.0,
    )

    # Record 0: mean shortfall 1/3 + worst-two mean 1/2. Record 1: zero.
    assert pieces["write_mean"] == pytest.approx(1.0 / 6.0)
    assert pieces["write_tail"] == pytest.approx(1.0 / 4.0)
    assert loss == pytest.approx(5.0 / 12.0)


def test_detector_off_context_losses_ignore_gate_compliant_nonzero_responses():
    owners = torch.tensor([0, 0, 1, 1])
    compliant = torch.tensor([[0.15, 9.0], [-0.20, 9.0], [9.0, -0.19], [9.0, 0.05]])
    loss, pieces = core.detector_writer_off_objective(
        compliant, owners, off_target_abs_max=0.2, tail_k=2
    )
    assert loss == pytest.approx(0.0)
    assert pieces["writer_off_mean"] == pytest.approx(0.0)
    assert pieces["writer_off_tail"] == pytest.approx(0.0)

    violating = compliant.clone()
    violating[0, 0] = 0.30
    violating_loss, _ = core.detector_writer_off_objective(
        violating, owners, off_target_abs_max=0.2, tail_k=2
    )
    assert violating_loss > loss


def test_global_writer_off_loss_catches_nonowner_collision_owner_loss_misses():
    owners = torch.tensor([0, 0, 1, 1])
    responses = torch.tensor(
        [
            [0.05, 0.22867718],
            [0.10, 0.00],
            [0.00, 0.05],
            [0.00, 0.10],
        ]
    )

    owner_only, _ = core.detector_writer_off_objective(
        responses,
        owners,
        off_target_abs_max=0.15,
        tail_k=2,
    )
    global_loss, pieces = core.detector_global_writer_off_objective(
        responses,
        owners,
        off_target_abs_max=0.15,
        tail_k=2,
    )

    assert owner_only == pytest.approx(0.0)
    assert global_loss > 0.0
    assert pieces["writer_off_mean"] > 0.0
    assert pieces["writer_off_tail"] > 0.0


def test_global_writer_off_loss_equal_weights_source_records():
    owners = torch.tensor([0, 1, 1, 1])
    responses = torch.tensor(
        [
            [0.30, 0.00],
            [0.00, 0.30],
            [0.00, 0.00],
            [0.00, 0.00],
        ]
    )
    loss, pieces = core.detector_global_writer_off_objective(
        responses,
        owners,
        off_target_abs_max=0.20,
        tail_k=1,
    )

    # Record 0: mean .005 + tail .01. Record 1: mean .0016667 + tail .01.
    assert pieces["writer_off_mean"] == pytest.approx((0.005 + 0.01 / 6.0) / 2)
    assert pieces["writer_off_tail"] == pytest.approx(0.01)
    assert loss == pytest.approx((0.015 + (0.01 / 6.0 + 0.01)) / 2)


def test_multilabel_detector_keeps_valid_label_active_in_negative_source_role():
    responses = torch.tensor([[0.30, 0.00], [0.30, 0.00], [0.00, 0.30]])
    owners = torch.tensor([0, 1, 1])
    positive_occurrences = torch.tensor([True, False, True])
    active = torch.tensor([[True, False], [True, False], [False, True]])

    multilabel_loss, pieces = core.detector_multilabel_objective(
        responses,
        owners,
        positive_occurrences,
        active,
        positive_target=0.30,
        off_target_abs_max=0.15,
        tail_k=2,
        negative_weight=5.0,
        cross_weight=2.0,
        global_tail_weight=1.0,
    )
    contradictory_old_loss, _ = core.detector_objective(
        responses,
        owners,
        positive_occurrences,
        positive_target=0.30,
        off_target_abs_max=0.15,
        tail_k=2,
        negative_weight=5.0,
        cross_weight=2.0,
    )

    assert multilabel_loss == pytest.approx(0.0)
    assert pieces["write_global_tail"] == pytest.approx(0.0)
    assert pieces["negative_global_tail"] == pytest.approx(0.0)
    assert contradictory_old_loss > 0.0


def test_multilabel_manifest_preserves_roles_and_canonicalizes_duplicate_state():
    positives = [["shared prompt"], ["record one positive"]]
    negatives = [["record zero negative"], ["shared prompt"]]
    manifest = method.build_multilabel_prompt_manifest(positives, negatives, [10, 20])
    shared = manifest["report"]["role_overlap_entries"][0]
    assert shared["active_group_indices"] == [0]
    assert shared["active_case_ids"] == [10]
    assert shared["negative_occurrences"] == [
        {
            "source_record_index": 1,
            "source_case_id": 20,
            "source_context_index": 0,
        }
    ]
    assert manifest["active_mask"][
        manifest["prompt_to_index"]["shared prompt"]
    ].tolist() == [True, False]

    prompt_groups = [*positives, *negatives]
    hidden_groups = [
        torch.tensor([[1.0, 2.0]]),
        torch.tensor([[3.0, 4.0]]),
        torch.tensor([[5.0, 6.0]]),
        torch.tensor([[9.0, 9.0]]),
    ]
    canonical, audit = method.canonicalize_grouped_hidden_states(
        prompt_groups,
        hidden_groups,
        canonical_prompts=manifest["canonical_prompts"],
        prompt_to_index=manifest["prompt_to_index"],
    )
    assert torch.equal(canonical[0][0], canonical[3][0])
    assert torch.equal(canonical[0][0], torch.tensor([1.0, 2.0]))
    assert audit["precanonicalization_duplicate_hidden_abs_max"] == pytest.approx(8.0)
    assert audit["postcanonicalization_duplicate_hidden_abs_max"] == 0.0
    assert audit["duplicate_occurrences_reused"] == 1


def test_multilabel_manifest_rejects_same_record_role_contradiction():
    with pytest.raises(RuntimeError, match="same record"):
        method.build_multilabel_prompt_manifest([["same"]], [["same"]], [10])


def test_global_writer_off_tail_adds_complete_update_extreme():
    responses = torch.zeros(100, 2)
    responses[0, 1] = 0.40
    owners = torch.tensor([0] * 50 + [1] * 50)
    base, _ = core.detector_global_writer_off_objective(
        responses,
        owners,
        off_target_abs_max=0.15,
        tail_k=2,
        global_tail_weight=0.0,
    )
    tailed, pieces = core.detector_global_writer_off_objective(
        responses,
        owners,
        off_target_abs_max=0.15,
        tail_k=2,
        global_tail_weight=1.0,
    )
    assert pieces["writer_off_global_tail"] > 0.0
    assert tailed - base == pytest.approx(pieces["writer_off_global_tail"])


def test_multilabel_global_tails_are_diagnostic_at_zero_weight():
    responses = torch.tensor([[0.10, 0.40], [0.30, 0.00]])
    owners = torch.tensor([0, 1])
    positive_occurrences = torch.tensor([True, True])
    active = torch.tensor([[True, False], [False, True]])
    balanced, balanced_pieces = core.detector_multilabel_objective(
        responses,
        owners,
        positive_occurrences,
        active,
        positive_target=0.30,
        off_target_abs_max=0.15,
        tail_k=2,
        negative_weight=5.0,
        cross_weight=2.0,
        global_tail_weight=0.0,
    )
    globally_tailed, tailed_pieces = core.detector_multilabel_objective(
        responses,
        owners,
        positive_occurrences,
        active,
        positive_target=0.30,
        off_target_abs_max=0.15,
        tail_k=2,
        negative_weight=5.0,
        cross_weight=2.0,
        global_tail_weight=1.0,
    )

    assert balanced_pieces["write_global_tail"] > 0.0
    assert balanced_pieces["cross_global_tail"] > 0.0
    assert balanced < globally_tailed
    assert balanced_pieces["write"] < tailed_pieces["write"]
    assert balanced_pieces["cross"] < tailed_pieces["cross"]


def test_detector_training_targets_are_separate_from_certificate_thresholds():
    responses = torch.tensor([[0.25], [0.18]])
    owners = torch.tensor([0, 0])
    positive = torch.tensor([True, False])
    loss, pieces = core.detector_objective(
        responses,
        owners,
        positive,
        positive_target=0.30,
        off_target_abs_max=0.15,
        tail_k=2,
        negative_weight=1.0,
        cross_weight=0.0,
    )

    assert pieces["write"] > 0
    assert pieces["negative"] > 0
    assert loss > 0


def test_detector_record_microbatch_accumulation_matches_one_global_objective():
    torch.manual_seed(23)
    owners = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    positive = torch.tensor([True, False] * 4)
    full_responses = torch.randn(8, 4, requires_grad=True)
    full_loss, _ = core.detector_objective(
        full_responses,
        owners,
        positive,
        positive_target=0.25,
        off_target_abs_max=0.2,
        tail_k=2,
        negative_weight=5.0,
        cross_weight=2.0,
    )
    full_loss.backward()
    full_gradient = full_responses.grad.detach().clone()

    micro_responses = full_responses.detach().clone().requires_grad_(True)
    accumulated_loss = micro_responses.sum() * 0.0
    for record_ids in ((0, 1), (2, 3)):
        mask = torch.zeros_like(owners, dtype=torch.bool)
        for record_id in record_ids:
            mask |= owners.eq(record_id)
        micro_loss, _ = core.detector_objective(
            micro_responses[mask],
            owners[mask],
            positive[mask],
            positive_target=0.25,
            off_target_abs_max=0.2,
            tail_k=2,
            negative_weight=5.0,
            cross_weight=2.0,
        )
        accumulated_loss = accumulated_loss + 0.5 * micro_loss
    accumulated_loss.backward()

    assert float(accumulated_loss.detach()) == pytest.approx(float(full_loss.detach()))
    assert torch.allclose(micro_responses.grad, full_gradient, atol=1e-7, rtol=1e-6)


def test_sparse_swiglu_hook_matches_materialized_weights_in_float32():
    torch.manual_seed(9)
    mlp = TinySwiGLU()
    hidden = torch.randn(3, 4, 5)
    base_output = mlp(hidden).detach()
    editor = core.SparseSwiGLUNeuronEditor(mlp, [1, 5])
    with torch.no_grad():
        editor.gate_delta.normal_(std=0.03)
        editor.up_delta.normal_(std=0.03)
        editor.down_delta.normal_(std=0.03)
    editor.install(mlp)
    hooked = mlp(hidden).detach()
    editor.remove()
    editor.materialize(mlp)
    materialized = mlp(hidden).detach()

    assert not torch.equal(base_output, hooked)
    assert torch.allclose(hooked, materialized, atol=2e-6, rtol=2e-6)


def test_sparse_swiglu_write_can_be_disabled_while_capturing_detector():
    torch.manual_seed(3)
    mlp = TinySwiGLU()
    hidden = torch.randn(2, 3, 5)
    expected = mlp(hidden).detach()
    editor = core.SparseSwiGLUNeuronEditor(mlp, [0, 2])
    with torch.no_grad():
        editor.gate_delta.fill_(0.05)
    editor.write_enabled = False
    editor.capture_activations = True
    editor.install(mlp)
    actual = mlp(hidden)
    editor.remove()

    assert torch.equal(expected, actual.detach())
    assert editor.last_edited_activations is not None
    assert editor.last_edited_activations.shape == (2, 3, 2)


def test_isolated_threshold_residual_is_exact_identity_at_zero_actuator():
    torch.manual_seed(31)
    mlp = TinySwiGLU()
    hidden = torch.randn(2, 3, 5)
    expected = mlp(hidden).detach()
    editor = core.SparseSwiGLUNeuronEditor(mlp, [0, 2])
    with torch.no_grad():
        editor.gate_delta.normal_(std=0.4)
        editor.up_delta.normal_(std=0.4)
        editor.down_delta.zero_()
    editor.configure_isolated_threshold_residual(
        [[0], [1]],
        torch.ones(2),
        off_boundary=0.20,
        on_boundary=0.25,
    )
    editor.install(mlp)
    actual = mlp(hidden).detach()
    editor.remove()

    assert torch.equal(actual, expected)
    with pytest.raises(RuntimeError, match="explicit internal branch"):
        editor.materialize(mlp)


def test_isolated_threshold_residual_writes_without_mutating_base_mlp_weights():
    mlp = TinySwiGLU()
    selected = torch.tensor([0, 2])
    with torch.no_grad():
        mlp.gate_proj.weight.index_fill_(0, selected, 0.0)
        mlp.up_proj.weight.index_fill_(0, selected, 0.0)
    hidden = torch.ones(1, 2, 5)
    base_output = mlp(hidden).detach()
    base_state = {
        name: parameter.detach().clone() for name, parameter in mlp.named_parameters()
    }
    editor = core.SparseSwiGLUNeuronEditor(mlp, selected.tolist())
    with torch.no_grad():
        editor.gate_delta.fill_(1.0)
        editor.up_delta.fill_(1.0)
        editor.down_delta.fill_(0.1)
    editor.gate_delta.requires_grad_(False)
    editor.up_delta.requires_grad_(False)
    editor.configure_isolated_threshold_residual(
        [[0], [1]], torch.ones(2), off_boundary=0.20, on_boundary=0.25
    )
    editor.install(mlp)
    edited_output = mlp(hidden)
    edited_output.sum().backward()
    editor.remove()

    assert not torch.equal(edited_output.detach(), base_output)
    assert editor.down_delta.grad is not None
    assert float(editor.down_delta.grad.abs().sum()) > 0.0
    for name, parameter in mlp.named_parameters():
        assert torch.equal(parameter.detach(), base_state[name])


def test_isolated_threshold_gate_maps_certificate_gap_to_zero_and_one():
    mlp = TinySwiGLU()
    editor = core.SparseSwiGLUNeuronEditor(mlp, [0, 2, 4, 6])
    editor.configure_isolated_threshold_residual(
        [[0, 1], [2, 3]],
        torch.ones(4),
        off_boundary=0.20,
        on_boundary=0.25,
    )
    activations = torch.tensor(
        [
            [0.10, 0.30, 0.25, 0.25],
            [0.15, 0.15, 0.30, 0.30],
        ]
    )
    features, responses, gates = editor.isolated_actuator_features_from_activations(
        activations
    )

    assert torch.allclose(responses, torch.tensor([[0.20, 0.25], [0.15, 0.30]]))
    assert torch.allclose(
        gates, torch.tensor([[0.0, 1.0], [0.0, 1.0]]), atol=1e-6, rtol=0.0
    )
    assert torch.equal(features[:, :2], torch.zeros(2, 2))
    assert torch.allclose(features[:, 2:], activations[:, 2:], atol=1e-6, rtol=0.0)


def test_nested_actuator_selection_is_disjoint_and_prefix_preserving():
    positive = [
        torch.ones(3, 12),
        torch.ones(4, 12) * 2.0,
    ]
    down_norms = torch.arange(1, 13, dtype=torch.float32)
    by_width, reports = core.select_nested_record_actuator_neurons(
        positive,
        down_norms,
        widths=[2, 4],
        excluded_neurons=[0, 1],
    )

    assert sorted(by_width) == [2, 4]
    assert len(reports) == 2
    assert all(len(group) == 4 for group in by_width[4])
    assert all(
        small == large[:2]
        for small, large in zip(by_width[2], by_width[4])
    )
    maximum = [neuron for group in by_width[4] for neuron in group]
    assert len(maximum) == len(set(maximum)) == 8
    assert not set(maximum).intersection({0, 1})


def test_separate_actuator_bank_is_identity_at_zero_and_gate_suppresses_write():
    mlp = TinySwiGLU(intermediate=7)
    with torch.no_grad():
        mlp.gate_proj.weight[2:4].fill_(1.0)
        mlp.up_proj.weight[2:4].fill_(1.0)
    hidden = torch.ones(1, 2, 5)
    expected = mlp(hidden).detach()

    bank = core.SparseThresholdGatedActuatorBank(
        mlp,
        [2, 3],
        [0, 0],
        detector_gate_rows=torch.ones(1, 5),
        detector_up_rows=torch.ones(1, 5),
        detector_local_groups=[[0]],
        detector_flat_signs=torch.ones(1),
        off_boundary=0.20,
        on_boundary=0.25,
    )
    bank.install(mlp)
    assert torch.equal(mlp(hidden).detach(), expected)
    with torch.no_grad():
        bank.down_delta.fill_(0.1)
    gated_on = mlp(hidden)
    gated_on.sum().backward()
    bank.remove()

    assert not torch.equal(gated_on.detach(), expected)
    assert bank.down_delta.grad is not None
    assert float(bank.down_delta.grad.abs().sum()) > 0.0

    suppressed = core.SparseThresholdGatedActuatorBank(
        mlp,
        [2, 3],
        [0, 0],
        detector_gate_rows=torch.zeros(1, 5),
        detector_up_rows=torch.zeros(1, 5),
        detector_local_groups=[[0]],
        detector_flat_signs=torch.ones(1),
        off_boundary=0.20,
        on_boundary=0.25,
    )
    with torch.no_grad():
        suppressed.down_delta.fill_(10.0)
    suppressed.install(mlp)
    gated_off = mlp(hidden).detach()
    suppressed.remove()
    assert torch.equal(gated_off, expected)


def test_separate_actuator_bank_projects_per_column_and_matched_group_budget():
    mlp = TinySwiGLU(intermediate=7)
    bank = core.SparseThresholdGatedActuatorBank(
        mlp,
        [1, 2, 3, 4],
        [0, 0, 1, 1],
        detector_gate_rows=torch.ones(2, 5),
        detector_up_rows=torch.ones(2, 5),
        detector_local_groups=[[0], [1]],
        detector_flat_signs=torch.ones(2),
        off_boundary=0.20,
        on_boundary=0.25,
    )
    with torch.no_grad():
        bank.down_delta.fill_(10.0)
    bank.clamp_down_relative_(1.5)
    assert bool((bank.down_relative_norms() <= 1.5 + 1e-6).all())
    budgets = torch.tensor([0.25, 0.50])
    bank.clamp_group_frobenius_(budgets)
    assert torch.allclose(
        bank.group_frobenius_norms(), budgets, atol=1e-6, rtol=1e-6
    )


def test_detector_cache_captures_exact_mlp_input_last_token_in_batches():
    class Encoded(dict):
        def to(self, device):
            return Encoded({key: value.to(device) for key, value in self.items()})

    class Tokenizer:
        padding_side = "left"

        def __call__(self, prompts, *, padding, return_tensors):
            assert padding and return_tensors == "pt"
            width = max(len(prompt) for prompt in prompts)
            input_ids = torch.zeros((len(prompts), width), dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for row, prompt in enumerate(prompts):
                size = len(prompt)
                input_ids[row, :size] = torch.arange(1, size + 1)
                attention[row, :size] = 1
            return Encoded(input_ids=input_ids, attention_mask=attention)

    class Backbone(torch.nn.Module):
        def __init__(self, mlp):
            super().__init__()
            self.mlp = mlp

        def forward(self, input_ids, attention_mask, **_kwargs):
            offsets = torch.arange(5, dtype=torch.float32).reshape(1, 1, 5)
            hidden = input_ids.float().unsqueeze(-1) + offsets
            self.mlp(hidden)
            return {"last_hidden_state": hidden}

    mlp = TinySwiGLU(hidden=5, intermediate=7)
    model = type("Model", (), {"model": Backbone(mlp)})()
    tok = Tokenizer()
    hidden = method.capture_mlp_input_last_token_hidden_states(
        model,
        tok,
        mlp,
        ["a", "bbb", "cc"],
        torch.device("cpu"),
        batch_size=2,
    )

    assert hidden.dtype == torch.float32
    assert hidden.device.type == "cpu"
    assert torch.equal(
        hidden,
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0, 7.0],
                [2.0, 3.0, 4.0, 5.0, 6.0],
            ]
        ),
    )
    assert tok.padding_side == "left"


def test_relative_caps_bound_each_materialized_neuron_component():
    torch.manual_seed(14)
    mlp = TinySwiGLU()
    editor = core.SparseSwiGLUNeuronEditor(mlp, [1, 4, 6])
    with torch.no_grad():
        editor.gate_delta.fill_(10.0)
        editor.up_delta.fill_(-10.0)
        editor.down_delta.fill_(10.0)
    report = editor.clamp_relative_(detector_cap=0.2, actuator_cap=0.3)

    assert report["gate_max_relative_norm"] <= 0.200001
    assert report["up_max_relative_norm"] <= 0.200001
    assert report["down_max_relative_norm"] <= 0.300001


def test_toggleable_embedding_delta_uses_existing_rows_only():
    embedding = torch.nn.Embedding(12, 4)
    ids = torch.tensor([[2, 3, 2]])
    baseline = embedding(ids).detach()
    delta = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    writer = core.ToggleableEmbeddingDelta(embedding, [2], delta)
    edited = embedding(ids).detach()
    writer.enabled = False
    restored = embedding(ids).detach()
    writer.remove()

    assert torch.equal(edited[:, 0], baseline[:, 0] + delta)
    assert torch.equal(edited[:, 1], baseline[:, 1])
    assert torch.equal(edited[:, 2], baseline[:, 2] + delta)
    assert torch.equal(restored, baseline)


def test_detector_gate_requires_writer_off_and_negative_silence():
    report = core.detector_gate_report(
        [torch.tensor([1.2, 1.5])],
        [torch.tensor([0.01, -0.03])],
        [torch.tensor([0.02, 0.01])],
        positive_floor=1.0,
        off_abs_max=0.1,
    )
    assert report["passed"]

    failed = core.detector_gate_report(
        [torch.tensor([1.2])],
        [torch.tensor([0.5])],
        [torch.tensor([0.02])],
        positive_floor=1.0,
        off_abs_max=0.1,
    )
    assert not failed["passed"]

    no_writer = core.detector_gate_report(
        [torch.tensor([1.2])],
        [torch.tensor([0.02])],
        [torch.tensor([99.0])],
        positive_floor=1.0,
        off_abs_max=0.1,
        require_writer_off=False,
    )
    assert no_writer["passed"]
    assert not no_writer["criterion"]["writer_off_required"]


def test_detector_gate_uses_preregistered_numerical_tolerance():
    exact = core.detector_gate_report(
        [torch.tensor([0.25])],
        [torch.tensor([0.20000005])],
        [torch.tensor([0.20000005])],
        positive_floor=0.25,
        off_abs_max=0.20,
    )
    tolerant = core.detector_gate_report(
        [torch.tensor([0.24999995])],
        [torch.tensor([0.20000005])],
        [torch.tensor([0.20000005])],
        positive_floor=0.25,
        off_abs_max=0.20,
        comparison_abs_tolerance=1e-7,
    )

    assert not exact["passed"]
    assert tolerant["passed"]
    assert tolerant["criterion"]["comparison_abs_tolerance"] == pytest.approx(1e-7)
    assert tolerant["per_record"][0]["positive_passed"]
    assert tolerant["per_record"][0]["negative_passed"]
    assert tolerant["per_record"][0]["writer_off_passed"]


def test_selected_neuron_ownership_hash_matches_jq_compact_newline_contract():
    assert method.selected_neuron_ownership_jq_compact_sha256([[4, 2], [9]]) == (
        "dbc00724c7fd383c8dfc1da4ae76d75db73ee785e4a77c8224005507bc7eaf55"
    )


def test_detector_endpoint_replay_comparison_binds_metrics_and_record_order():
    first = core.detector_gate_report(
        [torch.tensor([0.30])],
        [torch.tensor([0.10])],
        [torch.tensor([0.10])],
        positive_floor=0.25,
        off_abs_max=0.20,
        comparison_abs_tolerance=1e-7,
    )
    first["per_record"][0]["case_id"] = 10472
    second = json.loads(json.dumps(first))
    second["per_record"][0]["positive_min"] += 5e-8

    replay = method.compare_detector_gate_replays(first, second, abs_tolerance=1e-7)
    assert replay["passed"]
    assert replay["record_binding_match"]
    assert replay["decisions_match"]

    second["per_record"][0]["case_id"] = 14801
    assert not method.compare_detector_gate_replays(first, second, abs_tolerance=1e-7)[
        "passed"
    ]


def test_actuator_positive_objective_is_mean_plus_worst_two_shortfall():
    loss, pieces = core.actuator_positive_margin_objective(
        torch.tensor([0.0, 0.5, 1.5]), margin_floor=1.0, tail_k=2
    )
    assert pieces["mean"] == pytest.approx((1.0 + 0.25) / 3.0)
    assert pieces["tail"] == pytest.approx((1.0 + 0.25) / 2.0)
    assert loss == pytest.approx((1.0 + 0.25) / 3.0 + (1.0 + 0.25) / 2.0)


def test_paired_on_off_ratios_expose_activation_leakage():
    ratios = core.paired_on_off_ratios(
        torch.tensor([2.0, 1.0]),
        torch.tensor([0.5, 0.0]),
        epsilon=1e-2,
    )
    assert torch.allclose(ratios["writer_on_to_off_ratio"], torch.tensor([4.0, 100.0]))
    assert torch.allclose(
        ratios["writer_off_to_on_fraction"], torch.tensor([0.25, 0.0])
    )
    with pytest.raises(ValueError, match="equal shape"):
        core.paired_on_off_ratios(torch.ones(2), torch.ones(3), epsilon=1e-8)


def test_down_only_projection_preserves_frozen_detector_bit_exact():
    mlp = TinySwiGLU(hidden=5, intermediate=7)
    editor = core.SparseSwiGLUNeuronEditor(mlp, [0, 1])
    with torch.no_grad():
        editor.gate_delta.fill_(0.25)
        editor.up_delta.fill_(-0.125)
        editor.down_delta.fill_(10.0)
    gate_before = editor.gate_delta.detach().clone()
    up_before = editor.up_delta.detach().clone()
    report = editor.clamp_down_relative_(0.5)
    assert torch.equal(editor.gate_delta, gate_before)
    assert torch.equal(editor.up_delta, up_before)
    assert report["down_max_relative_norm"] <= 0.5 + 1e-6
    assert editor.down_relative_norms().numel() == 2


def test_cap_sweep_selects_smallest_positive_reachable_cap_separately_from_leakage():
    rows = [
        {
            "cap": 0.5,
            "positive_reachable": False,
            "writer_off_structural_selectivity_passed": True,
        },
        {
            "cap": 0.75,
            "positive_reachable": True,
            "writer_off_structural_selectivity_passed": False,
        },
        {
            "cap": 1.0,
            "positive_reachable": True,
            "writer_off_structural_selectivity_passed": True,
        },
    ]
    decision = core.actuator_cap_sweep_decision(rows)
    assert decision["selected_smallest_positive_reachable_cap"] == pytest.approx(0.75)
    assert decision["smallest_jointly_structurally_selective_cap"] == pytest.approx(1.0)
    assert decision["positive_reachability_passed"]
    assert decision["structural_selectivity_passed"]

    failed = core.actuator_cap_sweep_decision(
        [
            {
                "cap": cap,
                "positive_reachable": False,
                "writer_off_structural_selectivity_passed": False,
            }
            for cap in (0.5, 0.75, 1.0, 1.5, 2.0)
        ]
    )
    assert failed["selected_smallest_positive_reachable_cap"] is None
    assert failed["conclusion"] == (
        "isolated_threshold_branch_not_positive_reachable_at_registered_cap"
    )


def test_actuator_reference_and_writer_off_losses_have_tolerance_bands():
    reference_loss, _ = core.actuator_reference_regression_objective(
        torch.tensor([1.04, 1.20]),
        torch.tensor([1.00, 1.00]),
        tolerance=0.05,
        tail_k=2,
    )
    assert reference_loss == pytest.approx(2.0 * 0.15**2 / 2.0)

    writer_off_loss, _ = core.actuator_writer_off_objective(
        torch.tensor([1.04, 1.20]),
        torch.tensor([2.00, 1.90]),
        torch.tensor([1.00, 1.00]),
        torch.tensor([2.00, 2.00]),
        tolerance=0.05,
        tail_k=2,
    )
    # Squared violations are [0, .15^2, 0, .05^2].
    assert writer_off_loss == pytest.approx(0.00625 + 0.0125)


def test_actuator_endpoint_replay_uses_tolerance_and_record_binding():
    first = {
        "direct_failures": 0,
        "positive_failures": 0,
        "positive_contexts": 2,
        "minimum_margin": 1.01,
        "reference_nll_regression_max": 0.01,
        "writer_off_nll_abs_max": 0.02,
        "per_record": [
            {
                "record_index": 0,
                "case_id": 10472,
                "positive_contexts": 1,
                "positive_failures": 0,
                "direct_margin": 1.01,
                "positive_min": 1.01,
                "reference_nll_regression_max": 0.01,
                "writer_off_nll_abs_max": 0.02,
            },
            {
                "record_index": 1,
                "case_id": 14801,
                "positive_contexts": 1,
                "positive_failures": 0,
                "direct_margin": 1.02,
                "positive_min": 1.02,
                "reference_nll_regression_max": 0.0,
                "writer_off_nll_abs_max": 0.01,
            },
        ],
    }
    second = json.loads(json.dumps(first))
    second["minimum_margin"] += 5e-7
    replay = method.compare_actuator_audits(first, second, abs_tolerance=1e-6)
    assert replay["passed"]

    second["per_record"][1]["case_id"] = 999
    assert not method.compare_actuator_audits(first, second, abs_tolerance=1e-6)[
        "passed"
    ]


def test_frozen_v3_2_import_loads_only_passed_detector_tensors(tmp_path):
    mlp = TinySwiGLU(hidden=5, intermediate=7)
    editor = core.SparseSwiGLUNeuronEditor(mlp, [0, 1])
    ownership = [[0], [1]]
    ownership_hash = method.selected_neuron_ownership_jq_compact_sha256(ownership)
    signs = torch.tensor([1.0, -1.0])
    source_gate = torch.full_like(editor.gate_delta, 0.25)
    source_up = torch.full_like(editor.up_delta, -0.125)

    stage1_path = tmp_path / "stage1.pt"
    stage1_path.write_bytes(b"frozen-writer")
    method_dir = tmp_path / "source" / "method"
    method_dir.mkdir(parents=True)
    torch.save(
        {
            "protocol": method.FROZEN_DETECTOR_PROTOCOL,
            "detector_training_revision": "v3.2",
            "case_ids": [10472, 14801],
            "layer": 27,
            "selected_neurons": [0, 1],
            "ownership": ownership,
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
            "source_stage1_state_sha256": method.compositional_method.sha256_file(
                stage1_path
            ),
            "output_head_sha256": "head-sha256",
            "flat_signs": signs,
            "base_neuron_weights": {
                "gate_rows": editor.base_gate_rows.detach().clone(),
                "up_rows": editor.base_up_rows.detach().clone(),
                "down_columns": editor.base_down_columns.detach().clone(),
            },
            "gate_delta": source_gate,
            "up_delta": source_up,
            "down_delta": torch.full_like(editor.down_delta, 9.0),
        },
        method_dir / "embedding_keyed_neuron_state.pt",
    )
    artifacts = {
        "detector_gate_report.json": {
            "protocol": method.FROZEN_DETECTOR_PROTOCOL,
            "passed": True,
            "passed_records": 2,
            "total_records": 2,
            "criterion": {
                "positive_floor": 0.25,
                "negative_abs_max": 0.20,
                "writer_off_abs_max": 0.20,
                "comparison_abs_tolerance": 1e-7,
                "writer_off_required": True,
            },
            "official_evaluation_prompts_seen": 0,
        },
        "detector_endpoint_audit.json": {
            "complete": True,
            "official_evaluation_prompts_seen": 0,
        },
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "embedding_keyed_neuron_summary.json": {
            "protocol": method.FROZEN_DETECTOR_PROTOCOL,
            "acceptance": {
                "detector_gate_passed": True,
                "checkpoint_saved": False,
                "passed": False,
            },
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    with torch.no_grad():
        editor.down_delta.fill_(3.0)
    receipt, gate = method.import_frozen_v3_2_detector(
        tmp_path / "source",
        stage1_path=stage1_path,
        case_ids=[10472, 14801],
        layer=27,
        ownership=ownership,
        selected_neurons=[0, 1],
        flat_signs=signs,
        output_head_sha256="head-sha256",
        editor=editor,
    )

    assert receipt["passed"]
    assert gate["passed"]
    assert torch.equal(editor.gate_delta, source_gate)
    assert torch.equal(editor.up_delta, source_up)
    assert torch.count_nonzero(editor.down_delta) == 0


def test_frozen_v3_3_rejection_is_hash_bound_and_metric_exact(tmp_path):
    method_dir = tmp_path / "v3_3" / "method"
    method_dir.mkdir(parents=True)
    ownership_hash = "a" * 64
    case_ids = [10472, 14801]
    artifacts = {
        "detector_gate_report.json": {
            "protocol": method.FROZEN_V3_3_PROTOCOL,
            "passed": True,
            "passed_records": 2,
            "total_records": 2,
            "per_record": [
                {"case_id": case_id, "passed": True} for case_id in case_ids
            ],
            "official_evaluation_prompts_seen": 0,
        },
        "detector_endpoint_audit.json": {
            "protocol": method.FROZEN_V3_3_PROTOCOL,
            "official_evaluation_prompts_seen": 0,
        },
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "actuator_positive_only_feasibility.json": {
            "protocol": method.FROZEN_V3_3_PROTOCOL,
            "passed": False,
            "relative_norm_cap": 0.5,
            "optimizer_steps_recorded": 100,
            "complete_training_log": [
                {
                    "step": step,
                    "down_max_relative_norm": 0.49 if step < 15 else 0.5,
                }
                for step in range(1, 101)
            ],
            "final_audit": {
                "direct_failures": 40,
                "positive_failures": 300,
                "positive_contexts": 346,
                "minimum_margin": -8.75,
                "reference_nll_regression_max": 2.375,
                "writer_off_nll_abs_max": 4.8125,
                "norms": {"down_max_relative_norm": 0.5000000596046448},
            },
            "official_evaluation_prompts_seen": 0,
        },
        "training_rejection.json": {
            "full_actuator_training_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            "official_evaluation_prompts_seen": 0,
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = method.validate_frozen_v3_3_rejection(
        tmp_path / "v3_3", case_ids=case_ids, ownership_sha256=ownership_hash
    )
    assert receipt["passed"]
    assert receipt["observed"]["cap_saturation_step"] == 15
    assert receipt["observed"]["writer_off_nll_abs_max"] == pytest.approx(4.8125)

    artifacts["actuator_positive_only_feasibility.json"]["final_audit"][
        "writer_off_nll_abs_max"
    ] = 4.0
    (method_dir / "actuator_positive_only_feasibility.json").write_text(
        json.dumps(artifacts["actuator_positive_only_feasibility.json"]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="writer_off_drift"):
        method.validate_frozen_v3_3_rejection(
            tmp_path / "v3_3", case_ids=case_ids, ownership_sha256=ownership_hash
        )


def test_frozen_v3_4_rejection_binds_reachability_and_selectivity(tmp_path):
    method_dir = tmp_path / "v3_4" / "method"
    method_dir.mkdir(parents=True)
    ownership_hash = "b" * 64
    case_ids = [10472, 14801]
    zero_official = {"official_evaluation_prompts_seen": 0}
    artifacts = {
        "detector_gate_report.json": {
            "protocol": method.FROZEN_V3_4_PROTOCOL,
            "passed": True,
            "passed_records": 2,
            "total_records": 2,
            "per_record": [{"case_id": value} for value in case_ids],
            **zero_official,
        },
        "detector_endpoint_audit.json": {
            "protocol": method.FROZEN_V3_4_PROTOCOL,
            **zero_official,
        },
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "actuator_norm_cap_reachability_sweep.json": {
            "protocol": method.FROZEN_V3_4_PROTOCOL,
            "caps_preregistered": [0.5, 0.75, 1.0, 1.5, 2.0],
            "caps_completed": [0.5, 0.75, 1.0, 1.5, 2.0],
            "cap_artifacts": [
                {
                    "cap": cap,
                    "positive_reachable": cap >= 1.5,
                    "down_saturated_columns": 34 if cap == 1.5 else 0,
                }
                for cap in (0.5, 0.75, 1.0, 1.5, 2.0)
            ],
            "positive_reachability_passed": True,
            "selected_smallest_positive_reachable_cap": 1.5,
            "mechanism_readiness_passed": False,
            "structural_selectivity_diagnostic": {"passed": False},
            "conclusion": (
                "positive_reachability_found_but_structural_writer_off_selectivity_failed"
            ),
            "all_fitted_weights_discarded": True,
            "frozen_detector_tensors_unchanged": True,
            "full_preservation_objective_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "frozen_detector_selectivity_audit.json": {
            "protocol": method.FROZEN_V3_4_PROTOCOL,
            "aggregate": {
                "owned_signed_group_response": {
                    "writer_on_to_off_ratio": {"p10": 2.1545, "median": 3.2468}
                }
            },
            **zero_official,
        },
        "frozen_detector_zero_actuator_behavior_audit.json": {
            "protocol": method.FROZEN_V3_4_PROTOCOL,
            "writer_off_nll_abs_max": 0.25,
            "writer_off_preservation_passed": False,
            **zero_official,
        },
        "training_only_sweep_completion.json": {
            "protocol": method.FROZEN_V3_4_PROTOCOL,
            "positive_reachability_passed": True,
            "selected_smallest_positive_reachable_cap": 1.5,
            "mechanism_readiness_passed": False,
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "training_rejection.json": {
            "stage": "actuator_norm_cap_reachability_sweep",
            "reason": (
                "positive_reachability_without_structural_writer_off_selectivity"
            ),
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = method.validate_frozen_v3_4_rejection(
        tmp_path / "v3_4", case_ids=case_ids, ownership_sha256=ownership_hash
    )
    assert receipt["passed"]
    assert receipt["observed"]["selected_smallest_positive_reachable_cap"] == 1.5
    assert receipt["observed"]["zero_actuator_writer_off_nll_abs_max"] == 0.25

    artifacts["frozen_detector_zero_actuator_behavior_audit.json"][
        "writer_off_nll_abs_max"
    ] = 0.20
    (method_dir / "frozen_detector_zero_actuator_behavior_audit.json").write_text(
        json.dumps(artifacts["frozen_detector_zero_actuator_behavior_audit.json"]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="zero_actuator_drift"):
        method.validate_frozen_v3_4_rejection(
            tmp_path / "v3_4", case_ids=case_ids, ownership_sha256=ownership_hash
        )


def test_frozen_v3_5_rejection_binds_exactly_one_unresolved_gate_cell(tmp_path):
    method_dir = tmp_path / "v3_5" / "method"
    method_dir.mkdir(parents=True)
    ownership_hash = "c" * 64
    case_ids = [10803, *range(20000, 20049)]
    gate_max = 0.5735465884208679
    zero_official = {"official_evaluation_prompts_seen": 0}
    threshold_rows = [
        {
            "case_id": case_id,
            "writer_off_gate_abs_max": gate_max if index == 0 else 0.0,
        }
        for index, case_id in enumerate(case_ids)
    ]
    artifacts = {
        "detector_gate_report.json": {
            "protocol": method.FROZEN_V3_5_PROTOCOL,
            "passed": True,
            "passed_records": 50,
            "total_records": 50,
            "per_record": [{"case_id": value} for value in case_ids],
            **zero_official,
        },
        "detector_endpoint_audit.json": {
            "protocol": method.FROZEN_V3_5_PROTOCOL,
            **zero_official,
        },
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "isolated_threshold_gate_report.json": {
            "protocol": method.FROZEN_V3_5_PROTOCOL,
            "boundaries": {
                "runtime_off_boundary": 0.200001,
                "runtime_on_boundary": 0.249999,
            },
            "checks": {
                "all_positive_owner_gates_one": True,
                "all_positive_cross_gates_zero": True,
                "all_negative_gates_zero": True,
                "all_writer_off_gates_zero": False,
            },
            "aggregate": {
                "positive_owner_gate": {"n": 346, "min": 1.0},
                "positive_cross_gate": {"n": 16954, "max": 0.0},
                "negative_gate": {"n": 23250, "max": 0.0},
                "writer_off_gate": {
                    "n": 17300,
                    "mean": gate_max / 17300,
                    "max": gate_max,
                },
            },
            "per_record": threshold_rows,
            "passed": False,
            **zero_official,
        },
        "frozen_detector_selectivity_audit.json": {
            "protocol": method.FROZEN_V3_5_PROTOCOL,
            **zero_official,
        },
        "frozen_v3_2_detector_import.json": {"passed": True, **zero_official},
        "frozen_v3_4_rejection_import.json": {"passed": True, **zero_official},
        "training_rejection.json": {
            "stage": "isolated_threshold_gate",
            "reason": "frozen_detector_gap_did_not_map_to_exact_branch_gate",
            "actuator_training_started": False,
            "checkpoint_saved": False,
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = method.validate_frozen_v3_5_rejection(
        tmp_path / "v3_5", case_ids=case_ids, ownership_sha256=ownership_hash
    )
    assert receipt["passed"]
    assert receipt["observed"]["nonzero_writer_off_gate_cells"] == 1
    assert receipt["observed"]["writer_off_source_case_id"] == 10803
    assert "detector_group" not in receipt["observed"]

    artifacts["isolated_threshold_gate_report.json"]["aggregate"]["writer_off_gate"][
        "mean"
    ] = (2 * gate_max / 17300)
    (method_dir / "isolated_threshold_gate_report.json").write_text(
        json.dumps(artifacts["isolated_threshold_gate_report.json"]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="writer_off_single_cell"):
        method.validate_frozen_v3_5_rejection(
            tmp_path / "v3_5", case_ids=case_ids, ownership_sha256=ownership_hash
        )


def test_frozen_v3_5_1_forensics_binds_exact_nonowner_collision(tmp_path):
    method_dir = tmp_path / "v3_5_1" / "method"
    method_dir.mkdir(parents=True)
    ownership_hash = "d" * 64
    identity = {
        "source_case_id": 10803,
        "source_context_index": 4,
        "detector_case_id": 17353,
        "detector_group_index": 30,
        "owner_group": False,
    }
    zero_official = {"official_evaluation_prompts_seen": 0}
    artifacts = {
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "isolated_threshold_gate_report.json": {
            "protocol": method.FROZEN_V3_5_1_PROTOCOL,
            "gate_endpoint_violation_counts": {
                "positive_owner": 0,
                "positive_cross": 0,
                "negative": 0,
                "writer_off": 1,
            },
            **zero_official,
        },
        "v3_5_collision_forensics.json": {
            "protocol": method.FROZEN_V3_5_1_PROTOCOL,
            "single_writer_off_collision": identity,
            "ownerwise_certificate_consistent": True,
            **zero_official,
        },
        "training_only_v3_5_1_completion.json": {
            "protocol": method.FROZEN_V3_5_1_PROTOCOL,
            "complete": True,
            "result": "single_v3_5_writer_off_collision_identified",
            "diagnosis": "cross_record_detector_fired_without_embedding_writer",
            **identity,
            "raw_signed_response": 0.2286771833896637,
            "runtime_gate": 0.5735465884208679,
            "threshold_changed": False,
            "detector_tensors_changed": False,
            "actuator_tensors_changed": False,
            "detector_optimizer_constructed": False,
            "actuator_optimizer_constructed": False,
            "checkpoint_saved": False,
            **zero_official,
        },
        "training_rejection.json": {
            "stage": "v3.5_collision_forensics",
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = method.validate_frozen_v3_5_1_forensics(
        tmp_path / "v3_5_1", ownership_sha256=ownership_hash
    )
    assert receipt["passed"]
    assert receipt["diagnosis"] == (
        "cross_record_detector_fired_without_embedding_writer"
    )
    assert receipt["collision"]["detector_case_id"] == 17353
    assert receipt["collision"]["owner_group"] is False

    artifacts["training_only_v3_5_1_completion.json"]["owner_group"] = True
    (method_dir / "training_only_v3_5_1_completion.json").write_text(
        json.dumps(artifacts["training_only_v3_5_1_completion.json"]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="collision_identity"):
        method.validate_frozen_v3_5_1_forensics(
            tmp_path / "v3_5_1", ownership_sha256=ownership_hash
        )


def test_frozen_v3_5_2_rejection_binds_duplicate_prompt_contradiction(tmp_path):
    method_dir = tmp_path / "v3_5_2" / "method"
    method_dir.mkdir(parents=True)
    ownership_hash = "e" * 64
    prompt_hash = "9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38"
    positive = {
        "source_record_index": 16,
        "source_case_id": 10472,
        "source_context_index": 1,
        "source_prompt_sha256": prompt_hash,
        "detector_group_index": 16,
        "detector_case_id": 10472,
        "owner_group": True,
        "response": 0.22039906680583954,
    }
    negative = {
        "source_record_index": 1,
        "source_case_id": 19763,
        "source_context_index": 4,
        "source_prompt_sha256": prompt_hash,
        "detector_group_index": 16,
        "detector_case_id": 10472,
        "owner_group": False,
        "response": 0.21156451106071472,
    }
    zero_official = {"official_evaluation_prompts_seen": 0}
    artifacts = {
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "detector_gate_report.json": {
            "protocol": method.FROZEN_V3_5_2_PROTOCOL,
            "passed_records": 49,
            "total_records": 50,
            "failure_counts": {"positive": 1, "negative": 0, "writer_off": 0},
            **zero_official,
        },
        "detector_step_100_post_projection_global_isolation_gate.json": {
            "protocol": method.FROZEN_V3_5_2_PROTOCOL,
            "gate_endpoint_violation_counts": {
                "positive_owner": 1,
                "positive_cross": 0,
                "negative": 1,
                "writer_off": 0,
            },
            "response_certificate_violation_counts": {
                "positive_owner": 1,
                "positive_cross": 0,
                "negative": 1,
                "writer_off": 0,
            },
            "violating_cells": {
                "positive_owner": [positive],
                "positive_cross": [],
                "negative": [negative],
                "writer_off": [],
            },
            "aggregate": {"writer_off_response": {"min": -0.148, "max": 0.149}},
            **zero_official,
        },
        "detector_training_log.json": {
            "protocol": method.FROZEN_V3_5_2_PROTOCOL,
            "optimizer_steps_expected": 100,
            "optimizer_steps_recorded": 100,
            "complete": True,
            **zero_official,
        },
        "detector_endpoint_audit.json": {
            "complete": True,
            **zero_official,
        },
        "training_rejection.json": {
            "stage": "sparse_context_detector",
            "actuator_training_started": False,
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = method.validate_frozen_v3_5_2_rejection(
        tmp_path / "v3_5_2", ownership_sha256=ownership_hash
    )
    assert receipt["passed"]
    assert receipt["duplicate_prompt_sha256"] == prompt_hash
    assert receipt["writer_off_repair_succeeded"]
    assert receipt["positive_cell"]["source_case_id"] == 10472
    assert receipt["negative_cell"]["source_case_id"] == 19763

    artifacts["detector_step_100_post_projection_global_isolation_gate.json"][
        "violating_cells"
    ]["negative"][0]["source_prompt_sha256"] = "bad"
    (
        method_dir / "detector_step_100_post_projection_global_isolation_gate.json"
    ).write_text(
        json.dumps(
            artifacts["detector_step_100_post_projection_global_isolation_gate.json"]
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="single_negative_cell|same_exact_prompt"):
        method.validate_frozen_v3_5_2_rejection(
            tmp_path / "v3_5_2", ownership_sha256=ownership_hash
        )


def test_frozen_v3_5_3_rejection_binds_global_tail_positive_collapse(tmp_path):
    method_dir = tmp_path / "v3_5_3" / "method"
    method_dir.mkdir(parents=True)
    ownership_hash = "f" * 64
    zero_official = {"official_evaluation_prompts_seen": 0}
    artifacts = {
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "multilabel_prompt_manifest.json": {
            "protocol": method.FROZEN_V3_5_3_PROTOCOL,
            "frozen_v3_5_2_duplicate_prompt_reproduced": True,
            "same_record_positive_negative_conflicts": 0,
            **zero_official,
        },
        "detector_gate_report.json": {
            "protocol": method.FROZEN_V3_5_3_PROTOCOL,
            "passed_records": 29,
            "total_records": 50,
            "passed": False,
            "failure_counts": {"positive": 21, "negative": 0, "writer_off": 0},
            **zero_official,
        },
        "detector_step_100_post_projection_global_isolation_gate.json": {
            "protocol": method.FROZEN_V3_5_3_PROTOCOL,
            "response_certificate_violation_counts": {
                "positive_owner": 21,
                "writer_on_active": 21,
                "writer_on_inactive": 3,
                "source_negative_owner": 0,
                "writer_off": 0,
            },
            **zero_official,
        },
        "detector_training_log.json": {
            "protocol": method.FROZEN_V3_5_3_PROTOCOL,
            "optimizer_steps_expected": 100,
            "optimizer_steps_recorded": 100,
            "complete": True,
            "optimization": {"global_tail_weight": 1.0},
            **zero_official,
        },
        "detector_endpoint_audit.json": {
            "protocol": method.FROZEN_V3_5_3_PROTOCOL,
            "complete": True,
            **zero_official,
        },
        "training_rejection.json": {
            "stage": "sparse_context_detector",
            "actuator_training_started": False,
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = method.validate_frozen_v3_5_3_rejection(
        tmp_path / "v3_5_3", ownership_sha256=ownership_hash
    )
    assert receipt["passed"]
    assert receipt["owner_gate_passed_records"] == 29
    assert receipt["negative_and_writer_off_certificate_clean"]

    artifacts["detector_training_log.json"]["optimization"][
        "global_tail_weight"
    ] = 0.0
    (method_dir / "detector_training_log.json").write_text(
        json.dumps(artifacts["detector_training_log.json"]), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="global_tail_optimization_enabled"):
        method.validate_frozen_v3_5_3_rejection(
            tmp_path / "v3_5_3", ownership_sha256=ownership_hash
        )


def test_frozen_v3_5_4_rejection_binds_exact_detector_and_width_four_limit(
    tmp_path,
):
    method_dir = tmp_path / "v3_5_4" / "method"
    method_dir.mkdir(parents=True)
    ownership_hash = "1" * 64
    zero_official = {"official_evaluation_prompts_seen": 0}
    protocol = method.FROZEN_V3_5_4_PROTOCOL
    artifacts = {
        "neuron_selection_report.json": {
            "selected_neuron_ownership_jq_compact_sha256": ownership_hash,
        },
        "detector_gate_report.json": {
            "protocol": protocol,
            "passed_records": 50,
            "total_records": 50,
            "passed": True,
            **zero_official,
        },
        "isolated_threshold_gate_report.json": {
            "schema_version": 3,
            "kind": "mcf_embedding_keyed_neuron_multilabel_global_isolation_gate",
            "protocol": protocol,
            "passed": True,
            "checks": {
                "all_positive_source_owners_active": True,
                "all_writer_on_active_labels_one": True,
                "all_writer_on_inactive_labels_zero": True,
                "all_source_negative_owners_zero": True,
                "all_writer_off_labels_zero": True,
            },
            **zero_official,
        },
        "actuator_cap_1p50_feasibility.json": {
            "protocol": protocol,
            "cap": 1.5,
            "positive_reachable": False,
            "final_audit": {"positive_failures": 300},
            "down_norm_geometry": {
                "selected_columns": 200,
                "saturated_columns": 147,
            },
            "frozen_detector_tensors": {
                "gate_delta_sha256": "a" * 64,
                "up_delta_sha256": "b" * 64,
                "unchanged": True,
            },
            "fitted_weights_discarded": True,
            "down_delta_after_discard": "bit_exact_zero",
            **zero_official,
        },
        "v3_5_4_multilabel_actuator_feasibility.json": {
            "protocol": protocol,
            "all_fitted_weights_discarded": True,
            **zero_official,
        },
        "training_only_v3_5_4_completion.json": {
            "protocol": protocol,
            "positive_reachability_passed": False,
            "mechanism_readiness_passed": False,
            "conclusion": (
                "isolated_threshold_branch_not_positive_reachable_at_registered_cap"
            ),
            "full_preservation_objective_started": False,
            "checkpoint_saved": False,
            **zero_official,
        },
        "training_rejection.json": {
            "official_evaluation_allowed": False,
            **zero_official,
        },
        "training_firewall_receipt.json": {
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
        },
    }
    for name, payload in artifacts.items():
        (method_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    receipt = method.validate_frozen_v3_5_4_rejection(
        tmp_path / "v3_5_4", ownership_sha256=ownership_hash
    )
    assert receipt["passed"]
    assert receipt["detector_gate_delta_sha256"] == "a" * 64
    assert receipt["actuator_saturated_columns"] == 147

    artifacts["isolated_threshold_gate_report.json"]["checks"][
        "all_writer_on_inactive_labels_zero"
    ] = False
    (method_dir / "isolated_threshold_gate_report.json").write_text(
        json.dumps(artifacts["isolated_threshold_gate_report.json"]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="threshold_gate_perfect"):
        method.validate_frozen_v3_5_4_rejection(
            tmp_path / "v3_5_4", ownership_sha256=ownership_hash
        )
    artifacts["isolated_threshold_gate_report.json"]["checks"][
        "all_writer_on_inactive_labels_zero"
    ] = True
    (method_dir / "isolated_threshold_gate_report.json").write_text(
        json.dumps(artifacts["isolated_threshold_gate_report.json"]),
        encoding="utf-8",
    )

    artifacts["actuator_cap_1p50_feasibility.json"]["down_norm_geometry"][
        "saturated_columns"
    ] = 146
    (method_dir / "actuator_cap_1p50_feasibility.json").write_text(
        json.dumps(artifacts["actuator_cap_1p50_feasibility.json"]),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cap_geometry_matches_observed_run"):
        method.validate_frozen_v3_5_4_rejection(
            tmp_path / "v3_5_4", ownership_sha256=ownership_hash
        )


def test_detector_gate_case_tsv_binds_locked_record_order():
    gate = core.detector_gate_report(
        [torch.tensor([1.2])],
        [torch.tensor([0.01])],
        [torch.tensor([0.02])],
        positive_floor=1.0,
        off_abs_max=0.1,
    )
    output = gate_cases.detector_gate_case_tsv(gate, [{"case_id": 14801}])
    assert output.splitlines() == [
        "record_index\tcase_id\tpositive_min\tnegative_abs_max\t"
        "writer_off_abs_max\tpassed",
        "0\t14801\t+1.20000005\t0.01000000\t0.02000000\ttrue",
    ]

    with pytest.raises(RuntimeError, match="length mismatch"):
        gate_cases.detector_gate_case_tsv(gate, [])


def test_protected_prompt_bank_reserves_corpus_and_round_robins_records():
    training = [
        [f"record-{record}-prompt-{prompt}" for prompt in range(20)]
        for record in range(4)
    ]
    corpus = [f"corpus-{index}" for index in range(20)]
    selected, audit = method.build_selection_protected_prompts(
        training,
        corpus,
        total_limit=12,
        minimum_corpus=8,
    )

    assert selected[:8] == corpus[:8]
    assert len(selected) == 12
    assert audit["source_counts"]["corpus"] >= 8
    assert audit["training_groups_represented"] == 4
    assert audit["all_training_groups_represented"]


def test_activation_profile_reports_existing_neuron_function_tail():
    activations = torch.tensor([[0.0, 1.0], [0.1, 2.0], [0.3, 3.0], [0.5, 4.0]])
    profile = method.activation_tail_profile(
        activations,
        [7, 9],
        activation_threshold=0.2,
        down_column_norms=torch.tensor([2.0, 3.0]),
    )

    assert profile["prompt_count"] == 4
    assert profile["per_neuron"][0]["neuron_id"] == 7
    assert profile["per_neuron"][0][
        "activation_threshold_exceedance_fraction"
    ] == pytest.approx(0.5)
    assert profile["per_neuron"][1]["base_down_column_norm"] == 3.0


def test_writer_preflight_checks_global_and_worst_record_completeness():
    summary = method.summarize_writer_preflight(
        [torch.tensor([5.0, 5.0]), torch.tensor([5.0, 1.0])],
        amplitude_threshold=4.5,
        minimum_global_fraction=0.7,
        minimum_record_fraction=0.8,
    )
    assert summary["global_complete_fraction"] == pytest.approx(0.75)
    assert not summary["passed"]
    assert not summary["checks"]["minimum_record_complete_fraction"]


def test_writer_preflight_accepts_count_fractions_serialized_as_float32():
    summary = method.summarize_writer_preflight(
        [torch.full((7,), 5.0), torch.tensor([5.0] * 6 + [1.0])],
        amplitude_threshold=4.5,
        minimum_global_fraction=0.95,
        minimum_record_fraction=0.8,
    )
    for row in summary["per_record"]:
        row["complete_fraction"] = float(
            torch.tensor(row["complete_fraction"], dtype=torch.float32)
        )
    summary["global_complete_fraction"] = float(
        torch.tensor(summary["global_complete_fraction"], dtype=torch.float32)
    )
    summary["minimum_record_complete_fraction"] = float(
        torch.tensor(summary["minimum_record_complete_fraction"], dtype=torch.float32)
    )

    method.validate_writer_preflight_summary(
        summary,
        amplitude_threshold=4.5,
        minimum_global_fraction=0.95,
        minimum_record_fraction=0.8,
    )


def test_decoder_requires_conjunctive_clean_stage1_acceptance():
    artifact_hashes = {
        "training_visible_sha256": "visible",
        "split_manifest_sha256": "split",
        "context_manifest_sha256": "context",
        "stage1_state_sha256": "state",
        "stage1_report_sha256": "report",
        "stage1_writer_log_sha256": "log",
        "stage1_gradient_conflict_audit_sha256": "gradient-audit",
    }
    receipt = {
        "kind": "mcf_clean_stage1_writer_acceptance",
        "passed": True,
        "protocol": method.compositional_core.PROTOCOL,
        "seed": 1,
        "case_ids": [11, 12],
        "checks": {
            "artifact_integrity": True,
            "training_safe_portability": True,
        },
        "artifacts": artifact_hashes,
        "official_evaluation_opened": False,
        "training_safe_portability": {
            "kind": "mcf_clean_stage1_training_safe_portability_preflight",
            "passed": True,
            "amplitude_threshold": 4.5,
            "prompt_count": 14,
            "complete_count": 14,
            "global_complete_fraction": 1.0,
            "minimum_record_complete_fraction": 1.0,
            "per_record": [
                {
                    "case_id": 11,
                    "prompt_count": 7,
                    "complete_count": 7,
                    "complete_fraction": 1.0,
                },
                {
                    "case_id": 12,
                    "prompt_count": 7,
                    "complete_count": 7,
                    "complete_fraction": 1.0,
                },
            ],
            "criterion": {
                "minimum_global_fraction": 0.95,
                "minimum_record_fraction": 0.8,
            },
            "checks": {
                "global_complete_fraction": True,
                "minimum_record_complete_fraction": True,
            },
        },
    }
    accepted = method._validate_clean_stage1_acceptance(
        receipt,
        seed=1,
        case_ids=[11, 12],
        expected_artifacts=artifact_hashes,
        amplitude_threshold=4.5,
        minimum_global_fraction=0.95,
        minimum_record_fraction=0.8,
    )
    assert accepted["passed"]

    integrity_only = {
        **receipt,
        "checks": {
            "artifact_integrity": True,
            "training_safe_portability": False,
        },
    }
    with pytest.raises(RuntimeError, match="conjunction"):
        method._validate_clean_stage1_acceptance(
            integrity_only,
            seed=1,
            case_ids=[11, 12],
            expected_artifacts=artifact_hashes,
            amplitude_threshold=4.5,
            minimum_global_fraction=0.95,
            minimum_record_fraction=0.8,
        )


def test_clean_writer_receipt_reports_failed_portability_after_valid_integrity(
    tmp_path, monkeypatch
):
    training = tmp_path / "training.json"
    split = tmp_path / "split.json"
    context_path = tmp_path / "context.json"
    state_path = tmp_path / "writer.pt"
    report_path = tmp_path / "report.json"
    log_path = tmp_path / "writer.jsonl"
    gradient_audit_path = tmp_path / "stage1_gradient_conflict_audit.json"
    preflight_path = tmp_path / "preflight.json"
    training.write_text("{}\n", encoding="utf-8")
    split.write_text("{}\n", encoding="utf-8")
    context = {
        "source_training_visible_sha256": method.compositional_method.sha256_file(
            training
        ),
        "source_split_manifest_sha256": method.compositional_method.sha256_file(split),
        "data_access": {
            "official_paraphrases_seen": 0,
            "official_neighborhoods_seen": 0,
            "benchmark_retain_seen": 0,
            "official_ppl_seen": False,
        },
    }
    context_path.write_text(json.dumps(context), encoding="utf-8")
    context_hash = method.compositional_method.sha256_file(context_path)
    torch.save(
        {
            "protocol": method.compositional_core.PROTOCOL,
            "seed": 1,
            "case_ids": [17],
            "context_manifest_sha256": context_hash,
        },
        state_path,
    )
    report_path.write_text("{}\n", encoding="utf-8")
    log_path.write_text('{"step": 1}\n', encoding="utf-8")
    gradient_audit_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        clean_writer_verify.neuron,
        "_validate_clean_stage1_lineage",
        lambda *_args, **_kwargs: {"from_scratch": True, "writer_steps": 1200},
    )
    preflight = {
        "kind": "mcf_clean_stage1_training_safe_portability_preflight",
        "protocol": method.compositional_core.PROTOCOL,
        "seed": 1,
        "case_ids": [17],
        "amplitude_threshold": 4.5,
        "prompt_count": 1,
        "complete_count": 0,
        "global_complete_fraction": 0.0,
        "minimum_record_complete_fraction": 0.0,
        "criterion": {
            "minimum_global_fraction": 0.95,
            "minimum_record_fraction": 0.8,
        },
        "per_record": [
            {
                "case_id": 17,
                "prompt_count": 1,
                "complete_count": 0,
                "complete_fraction": 0.0,
            }
        ],
        "checks": {
            "global_complete_fraction": False,
            "minimum_record_complete_fraction": False,
        },
        "passed": False,
        "decoder_constructed": False,
        "official_evaluation_opened": False,
        "binding": {
            "context_manifest_sha256": context_hash,
            "stage1_state_sha256": method.compositional_method.sha256_file(state_path),
            "stage1_report_sha256": method.compositional_method.sha256_file(
                report_path
            ),
            "stage1_writer_log_sha256": method.compositional_method.sha256_file(
                log_path
            ),
            "stage1_gradient_conflict_audit_sha256": (
                method.compositional_method.sha256_file(gradient_audit_path)
            ),
        },
    }
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")

    receipt = clean_writer_verify.verify(
        training_visible_path=training,
        split_manifest_path=split,
        context_manifest_path=context_path,
        stage1_state_path=state_path,
        stage1_report_path=report_path,
        stage1_log_path=log_path,
        portability_preflight_path=preflight_path,
    )
    assert receipt["checks"]["artifact_integrity"]
    assert not receipt["checks"]["training_safe_portability"]
    assert not receipt["passed"]


def test_retain_tail_uses_binomial_bound_and_explicit_effect_events():
    assert tail_audit.wilson_upper_bound(0, 1000) > 0.0
    summary = tail_audit.summarize_tail(
        [0.01, 0.3, 0.02, 0.0],
        [False, True, False, False],
        [0.0, 0.02, 0.001, 0.0],
        [0.0, 0.2, 0.01, 0.0],
        response_threshold=0.2,
        kl_threshold=0.01,
        logprob_threshold=0.1,
    )
    assert summary["response_event"]["events"] == 1
    assert summary["top1_change_event"]["events"] == 1
    assert summary["restricted_topk_kl_event"]["events"] == 1
    assert summary["max_abs_topk_logprob_change_event"]["events"] == 1


def test_writer_portability_requires_global_and_worst_record_coverage():
    rows = [
        {
            "prompts": [
                {
                    "prompt_type": "rewrite",
                    "complete": True,
                    "own_marker_amplitude": 2.0,
                },
                {
                    "prompt_type": "paraphrase",
                    "complete": True,
                    "own_marker_amplitude": 1.5,
                },
            ]
        },
        {
            "prompts": [
                {
                    "prompt_type": "rewrite",
                    "complete": True,
                    "own_marker_amplitude": 2.0,
                },
                {
                    "prompt_type": "paraphrase",
                    "complete": False,
                    "own_marker_amplitude": 0.2,
                },
            ]
        },
    ]
    summary = portability_audit.summarize_portability(
        rows,
        minimum_global_complete_fraction=0.7,
        minimum_per_record_complete_fraction=0.8,
    )
    assert summary["global_complete_fraction"] == pytest.approx(0.75)
    assert not summary["acceptance"]["passed"]
    assert not summary["acceptance"]["checks"]["minimum_per_record_complete_fraction"]


def test_latent_recovery_summary_uses_sensitive_target_preference():
    summary = latent_audit.summarize_candidate_nlls(
        [
            {"target_true_nll": 1.0, "target_new_nll": 2.0},
            {"target_true_nll": 3.0, "target_new_nll": 2.0},
        ]
    )
    assert summary["sensitive_preference_count"] == 1
    assert summary["sensitive_preference_fraction"] == pytest.approx(0.5)


def test_relearning_recovery_requires_both_direct_and_paraphrase_return():
    curve = [
        {"step": 0, "Eff": 0.0, "Gen": 0.0},
        {"step": 4, "Eff": 80.0, "Gen": 20.0},
        {"step": 8, "Eff": 80.0, "Gen": 60.0},
    ]
    assert (
        relearning_audit.first_recovery_step(curve, minimum_eff=50.0, minimum_gen=50.0)
        == 8
    )


def test_frequency_factorial_tests_decoder_by_writer_cap_interaction():
    def component(spe: float, ppl: float, common_spe: float):
        return {
            "forget": {"Eff": 0.0, "Gen": 0.0, "Spe": spe},
            "retain": {"Eff": 80.0, "Gen": 80.0, "Spe": spe},
            "PPL": ppl,
            "forget_frequency_strata": {
                "common": {
                    "record_count": 3,
                    "metrics": {"Eff": 0.0, "Gen": 0.0, "Spe": common_spe},
                }
            },
        }

    payloads = {}
    expected = {
        "frequency_capped": {
            "row_norm_cap": 8.0,
            "row_norm_cap_frequency_alpha": 0.15,
            "max_subject_token_frequency": 1_000_000_000,
        },
        "uniform_same_cap": {
            "row_norm_cap": 8.0,
            "row_norm_cap_frequency_alpha": 0.0,
            "max_subject_token_frequency": 1_000_000_000,
        },
        "uniform_raised_cap": {
            "row_norm_cap": 16.0,
            "row_norm_cap_frequency_alpha": 0.0,
            "max_subject_token_frequency": 1_000_000_000,
        },
    }
    for condition in frequency_aggregate.CONDITIONS:
        base = component(10.0, 10.0, 10.0)
        if condition == "uniform_raised_cap":
            embedding = component(10.0, 10.0, 9.0)
            full = component(10.1, 10.1, 9.8)
        else:
            embedding = component(10.0, 10.0, 9.9)
            full = component(10.0, 10.0, 9.9)
        payloads[condition] = {
            "kind": "mcf_embedding_keyed_neuron_post_freeze_component_evaluation",
            "writer_mode": "embedding_keyed",
            "writer_configuration": expected[condition],
            "source_stage1_state_sha256": f"hash-{condition}",
            "dataset": "MCF",
            "sample_mode": "official",
            "seed": 1,
            "unlearn_num": 50,
            "retain_num": 1000,
            "used_for_training_checkpoint_selection_or_retry": False,
            "components": {
                "reconstructed_base": base,
                "embedding_only": embedding,
                "full_embedding_plus_neuron": full,
            },
        }
    result = frequency_aggregate.build_aggregate(
        payloads,
        expected_writer_conditions=expected,
        max_abs_spe_delta=0.2,
        max_abs_ppl_percent_delta=5.0,
    )
    assert result["acceptance"]["passed"]
    assert result["acceptance"]["checks"][
        "decoder_attenuates_common_token_leakage_increase"
    ]


def test_training_cli_exposes_no_original_mcf_or_official_eval_argument():
    args = method.parse_args(
        [
            "--model-path",
            "model",
            "--training-visible-path",
            "direct.json",
            "--split-manifest",
            "split.json",
            "--context-manifest",
            "contexts.json",
            "--stage1-state",
            "writer.pt",
            "--stage1-report",
            "writer-report.json",
            "--stage1-writer-log",
            "writer-log.jsonl",
            "--clean-stage1-portability-preflight",
            "clean-stage1-portability.json",
            "--clean-stage1-acceptance",
            "clean-stage1-acceptance.json",
            "--experiment-registry",
            "registry.json",
            "--output-dir",
            "out",
        ]
    )
    assert not hasattr(args, "mcf_path")
    assert not hasattr(args, "official_eval")
    assert not hasattr(args, "paraphrase_path")
    assert args.detector_positive_floor == pytest.approx(0.25)
    assert args.detector_off_abs_max == pytest.approx(0.20)
    assert args.detector_training_positive_floor == pytest.approx(0.30)
    assert args.detector_training_off_abs_max == pytest.approx(0.15)
    assert args.detector_certificate_abs_tolerance == pytest.approx(1e-7)
    assert args.detector_initialization == "train"
    assert not args.training_only_actuator_width_sweep
    assert args.actuator_feasibility_steps == 100
    assert args.actuator_feasibility_caps == [1.5]
    assert args.actuator_feasibility_widths == [4, 8, 16]
    assert args.actuator_architecture == "separate_threshold_gated_actuator_bank"
    assert args.threshold_gate_numerical_guard == pytest.approx(1e-6)
    assert args.detector_selectivity_ratio_epsilon == pytest.approx(1e-8)
    assert args.detector_selectivity_warning_ratio == pytest.approx(100.0)
    assert args.actuator_steps == 100
    assert args.actuator_protected_batch == 80
    assert args.actuator_positive_contexts == "all"
    assert args.actuator_negative_contexts == "all"
    assert args.actuator_tail_k == 2
    assert args.actuator_writer_off_nll_tolerance == pytest.approx(0.05)

    with pytest.raises(SystemExit):
        method.parse_args(
            [
                "--model-path",
                "model",
                "--training-visible-path",
                "direct.json",
                "--split-manifest",
                "split.json",
                "--context-manifest",
                "contexts.json",
                "--stage1-state",
                "writer.pt",
                "--stage1-report",
                "writer-report.json",
                "--stage1-writer-log",
                "writer-log.jsonl",
                "--clean-stage1-portability-preflight",
                "clean-stage1-portability.json",
                "--clean-stage1-acceptance",
                "clean-stage1-acceptance.json",
                "--experiment-registry",
                "registry.json",
                "--output-dir",
                "out",
                "--frequency-doc-start",
                "0",
            ]
        )


def test_no_writer_cli_requires_matched_control_settings():
    common = [
        "--model-path",
        "model",
        "--training-visible-path",
        "direct.json",
        "--split-manifest",
        "split.json",
        "--context-manifest",
        "contexts.json",
        "--stage1-state",
        "writer.pt",
        "--stage1-report",
        "writer-report.json",
        "--stage1-writer-log",
        "writer-log.jsonl",
        "--clean-stage1-portability-preflight",
        "clean-stage1-portability.json",
        "--clean-stage1-acceptance",
        "clean-stage1-acceptance.json",
        "--experiment-registry",
        "registry.json",
        "--output-dir",
        "out",
        "--writer-mode",
        "none",
    ]
    with pytest.raises(SystemExit):
        method.parse_args(common)

    args = method.parse_args(
        [
            *common,
            "--selection-mode",
            "base_context_contrastive",
            "--no-require-writer-necessity",
            "--save-rejected-checkpoint",
        ]
    )
    assert args.writer_mode == "none"
    assert not args.require_writer_necessity


def test_firewall_rejects_heldout_probe_content_recursively():
    clean = {
        "data_access": {
            "official_paraphrases_seen": 0,
            "official_neighborhoods_seen": 0,
            "benchmark_retain_seen": 0,
            "official_ppl_seen": False,
        },
        "records": [{"positive_prompts": ["direct"]}],
    }
    method._validate_firewall(clean, {"context_manifest_sha256": "abc"})
    leaked = dict(clean)
    leaked["records"] = [{"metadata": {"adversarial_prompts": ["leak"]}}]
    with pytest.raises(RuntimeError, match="evaluation-only fields"):
        method._validate_firewall(leaked, {"context_manifest_sha256": "abc"})


def test_clean_stage1_lineage_requires_from_scratch_training_and_nonempty_log(
    tmp_path,
):
    context = {
        "positive_context_policy": {
            "name": method.compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY,
            "free_form_generated_surrogates_allowed": False,
            "relation_template_bank_sha256": method.compositional_method.sha256_json(
                method.synthetic.RELATION_ALTERNATE_TEMPLATES
            ),
            "source_prompt_counts": {"external_free_form_surrogate": 0},
        },
        "surrogate_receipt": None,
        "synthetic_coverage": {"generic_fallback_records": 0},
        "selected_embedding_rows": [7],
        "cross_record_parameter_sharing": {
            "case_count": 1,
            "selected_row_count": 1,
            "positive_prompt_count": 2,
        },
        "records": [
            {
                "case_id": 11,
                "positive_prompts": ["Ada works as", "The profession of Ada is"],
                "positive_prompt_provenance": [
                    {"prompt": "Ada works as", "source": "canonical_direct"},
                    {
                        "prompt": "The profession of Ada is",
                        "source": ("hand_authored_relation_template_or_corpus_prefix"),
                    },
                ],
            }
        ],
    }
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context), encoding="utf-8")
    context_hash = method.compositional_method.sha256_file(context_path)
    log_path = tmp_path / "writer.jsonl"
    log_path.write_text('{"step": 1}\n', encoding="utf-8")
    log_hash = method.compositional_method.sha256_file(log_path)
    gradient_summary = method.compositional_core.gradient_conflict_summary(
        [torch.tensor([1.0])], [11]
    )
    gradient_audit = {
        "schema_version": 1,
        "protocol": method.compositional_core.PROTOCOL,
        "kind": "mcf_stage1_per_record_gradient_conflict_audit",
        "negative_cosine_tolerance": 1e-8,
        "interpretation": "test fixture",
        "initial": {
            "phase": "initial",
            "positive_write": gradient_summary,
            "full_writer": gradient_summary,
        },
        "final": {
            "phase": "final",
            "positive_write": gradient_summary,
            "full_writer": gradient_summary,
        },
        "official_evaluation_opened": False,
    }
    gradient_audit_hash = method.compositional_method.sha256_json(gradient_audit)
    gradient_audit_path = tmp_path / "stage1_gradient_conflict_audit.json"
    gradient_audit_path.write_text(json.dumps(gradient_audit), encoding="utf-8")
    gradient_audit_file_hash = method.compositional_method.sha256_file(
        gradient_audit_path
    )
    lineage = {
        "mode": "from_scratch",
        "same_context_manifest": True,
        "resumed_context_manifest_sha256": None,
        "current_context_manifest_sha256": context_hash,
        "writer_steps": 1200,
        "from_scratch": True,
        "resumed_from": None,
        "positive_context_policy": (
            method.compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY
        ),
        "writer_log_sha256": log_hash,
        "writer_log_event_count": 1,
        "gradient_conflict_audit_sha256": gradient_audit_hash,
        "gradient_conflict_audit_file_sha256": gradient_audit_file_hash,
        "base_model_path": "/models/base",
        "base_transformer_fingerprint": 123.5,
        "base_selected_embedding_rows_sha256": "a" * 64,
        "writer_optimization": {
            "record_batch": 3,
            "record_batch_semantics": ("gradient_accumulation_microbatch_capacity"),
            "update_coverage": "all_records_accumulated",
            "records_per_optimizer_update": 50,
            "microbatches_per_optimizer_update": 17,
            "optimizer_updates": 1200,
            "record_exposures": 60000,
            "record_local_reference_exposures": 3600,
            "record_exposure_multiplier_vs_record_local": 50 / 3,
            "gradient_normalization": ("equal_record_mean_plus_global_prompt_mean_kl"),
            "kl_evaluation": (
                "exact_registered_topk_rows_without_full_vocabulary_materialization"
            ),
            "gradient_conflict_audit_phases": ["initial", "final"],
            "gradient_conflict_audit_objectives": [
                "positive_write",
                "full_writer",
            ],
            "positive_context_mode": "all",
            "positive_context_batch": 7,
            "positive_tail_k": 2,
            "negative_context_batch": 5,
            "objective": "mean_plus_worst_k_squared_shortfall",
        },
    }
    state = {
        "protocol": method.compositional_core.PROTOCOL,
        "context_manifest_sha256": context_hash,
        "training_lineage": lineage,
        "writer_log_sha256": log_hash,
        "writer_log_event_count": 1,
        "gradient_conflict_audit": gradient_audit,
        "gradient_conflict_audit_sha256": gradient_audit_hash,
        "gradient_conflict_audit_file_sha256": gradient_audit_file_hash,
        "writer_optimization": lineage["writer_optimization"],
    }
    report = {
        "protocol": method.compositional_core.PROTOCOL,
        "context_manifest_sha256": context_hash,
        "positive_context_policy": (
            method.compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY
        ),
        "training_lineage": lineage,
        "writer_log_sha256": log_hash,
        "writer_log_event_count": 1,
        "gradient_conflict_audit": gradient_audit,
        "gradient_conflict_audit_sha256": gradient_audit_hash,
        "gradient_conflict_audit_file_sha256": gradient_audit_file_hash,
        "writer_configuration": lineage["writer_optimization"],
    }

    receipt = method._validate_clean_stage1_lineage(
        context, state, report, context_path, log_path
    )
    assert receipt["from_scratch"]
    assert receipt["writer_log_event_count"] == 1

    bad_provenance = json.loads(json.dumps(context))
    bad_provenance["records"][0]["positive_prompt_provenance"][1][
        "source"
    ] = "external_free_form_surrogate"
    with pytest.raises(RuntimeError, match="prompt provenance"):
        method._validate_clean_stage1_lineage(
            bad_provenance, state, report, context_path, log_path
        )

    resumed_state = {**state, "training_lineage": {**lineage, "from_scratch": False}}
    with pytest.raises(RuntimeError, match="trained from Base"):
        method._validate_clean_stage1_lineage(
            context, resumed_state, report, context_path, log_path
        )

    log_path.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="log hash"):
        method._validate_clean_stage1_lineage(
            context, state, report, context_path, log_path
        )


def test_environment_firewall_rejects_evaluation_paths(monkeypatch):
    method._validate_environment_firewall()
    monkeypatch.setenv("MCF_PATH", "/heldout/multi_counterfact.json")
    with pytest.raises(RuntimeError, match="leaked into learner environment"):
        method._validate_environment_firewall()


def test_primary_configuration_is_bound_to_preregistered_values():
    args = method.parse_args(
        [
            "--model-path",
            "model",
            "--training-visible-path",
            "direct.json",
            "--split-manifest",
            "split.json",
            "--context-manifest",
            "contexts.json",
            "--stage1-state",
            "writer.pt",
            "--stage1-report",
            "writer-report.json",
            "--stage1-writer-log",
            "writer-log.jsonl",
            "--clean-stage1-portability-preflight",
            "clean-stage1-portability.json",
            "--clean-stage1-acceptance",
            "clean-stage1-acceptance.json",
            "--experiment-registry",
            "registry.json",
            "--output-dir",
            "out",
        ]
    )
    registry = {
        "protocol": method.PROTOCOL,
        "stage1_writer_prerequisite": {
            "protocol": method.compositional_core.PROTOCOL,
            "positive_context_policy": (
                method.compositional_method.CLEAN_POSITIVE_CONTEXT_POLICY
            ),
            "relation_template_bank_sha256": (
                method.compositional_method.sha256_json(
                    method.synthetic.RELATION_ALTERNATE_TEMPLATES
                )
            ),
            "training_origin": "Base model with no resumed Stage-1 state",
            "writer_steps": 1200,
            "writer_steps_semantics": (
                "optimizer updates after full-record gradient accumulation"
            ),
            "writer_record_batch": 3,
            "writer_record_batch_semantics": (
                "gradient_accumulation_microbatch_capacity"
            ),
            "writer_update_coverage": "all_records_accumulated",
            "writer_records_per_optimizer_update": 50,
            "writer_microbatches_per_optimizer_update": 17,
            "writer_optimizer_updates": 1200,
            "writer_record_exposures": 60000,
            "writer_record_local_reference_exposures": 3600,
            "writer_record_exposure_multiplier_vs_v6_1": 50 / 3,
            "writer_gradient_normalization": (
                "equal_record_mean_plus_global_prompt_mean_kl"
            ),
            "writer_kl_evaluation": (
                "exact_registered_topk_rows_without_full_vocabulary_materialization"
            ),
            "writer_gradient_conflict_audit_phases": ["initial", "final"],
            "writer_gradient_conflict_audit_objectives": [
                "positive_write",
                "full_writer",
            ],
            "writer_gradient_conflict_audit_hash_bound": True,
            "writer_positive_context_mode": "all",
            "writer_positive_context_batch": 7,
            "writer_positive_tail_k": 2,
            "writer_negative_context_batch": 5,
            "writer_objective": "mean_plus_worst_k_squared_shortfall",
            "cross_record_parameter_sharing_audit_required": True,
        },
        "development_retain_shared_row_exposure_audit": (_development_exposure_audit()),
        "v3_5_1_scope": _v3_5_1_forensic_scope(),
        "v3_5_2_scope": _v3_5_2_repair_scope(),
        "v3_5_3_scope": _v3_5_3_multilabel_scope(),
        "v3_5_4_scope": _v3_5_4_balanced_scope(),
        "v3_5_5_scope": _v3_5_5_width_scope(),
        "detector_training_revision": _detector_training_revision(),
        "actuator_training_revision": _actuator_training_revision(),
        "selected_neuron_ownership_binding": {
            "scope": "primary_embedding_keyed_configuration",
            "source_runs": [
                "v3",
                "v3.1",
                "v3.2",
                "v3.3",
                "v3.4",
                "v3.5",
                "v3.5.1",
                "v3.5.2",
                "v3.5.3",
                "v3.5.4",
            ],
            "jq_projection": "[.ownership[].selected_neurons]",
            "jq_compact_sha256": (
                "acc3cc05868483f6c40a8909fca064b59c4ec4d000a76cf1ece6c3e818c750d1"
            ),
            "writer_configuration": {
                "row_norm_cap": 8.0,
                "row_norm_cap_frequency_alpha": 0.15,
                "max_subject_token_frequency": 1000000000,
            },
            "required": True,
        },
        "primary_configuration": {
            "forget_num": 50,
            "neuron_layer": 27,
            "neurons_per_record": 4,
            "selection_mode": "writer_contrastive",
            "dormant_fraction": 0.2,
            "detector_training_positive_floor": 0.3,
            "detector_training_off_abs_max": 0.15,
            "detector_certificate_abs_tolerance": 1e-7,
            "detector_relative_cap": 1.0,
            "actuator_relative_cap": 0.5,
            "gate_policy": "strict",
        },
    }
    method._validate_experiment_registry(registry, args)
    args.neuron_layer = 24
    with pytest.raises(RuntimeError, match="diverges from registry"):
        method._validate_experiment_registry(registry, args)


def test_repository_registry_binds_v3_5_5_separate_actuator_width_sweep():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "mcf_embedding_keyed_neuron_ablation_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    assert registry["schema_version"] == 16
    assert registry["protocol"] == core.PROTOCOL
    assert registry["development_history"][-1]["version"] == (
        "v14_embedding_keyed_gate_v3_5_4_detector_pass_actuator_budget_rejection"
    )
    contradiction = next(
        row["contradictory_prompt"]
        for row in registry["development_history"]
        if "contradictory_prompt" in row
    )
    assert contradiction["positive_source_case_id"] == 10472
    assert contradiction["positive_source_context_index"] == 1
    assert contradiction["negative_source_case_id"] == 19763
    assert contradiction["negative_source_context_index"] == 4
    assert contradiction["positive_detector_case_id"] == 10472
    assert contradiction["negative_detector_case_id"] == 10472
    assert not registry["development_history"][-1][
        "official_evaluation_opened_by_this_failed_run"
    ]
    latest = registry["development_history"][-1]
    assert latest["detector_gate"] == {
        "passed_records": 50,
        "total_records": 50,
        "passed": True,
        "positive_owner_gate_min": 1.0,
        "writer_off_gate_max": 0.0,
        "owned_response_ratio_p10": 2.5555,
        "owned_response_ratio_median": 3.623,
    }
    assert latest["actuator_feasibility"]["saturated_columns"] == 147
    assert not latest["actuator_feasibility"]["positive_reachability_passed"]
    assert (
        registry["selected_neuron_ownership_binding"]["jq_compact_sha256"]
        == "acc3cc05868483f6c40a8909fca064b59c4ec4d000a76cf1ece6c3e818c750d1"
    )
    assert registry["primary_configuration"][
        "detector_training_positive_floor"
    ] == pytest.approx(0.30)
    assert registry["primary_configuration"][
        "detector_training_off_abs_max"
    ] == pytest.approx(0.15)
    assert registry["primary_configuration"][
        "detector_certificate_abs_tolerance"
    ] == pytest.approx(1e-7)
    assert registry["primary_configuration"]["detector_initialization"] == (
        "frozen_v3_2"
    )
    assert registry["primary_configuration"]["detector_steps"] == 100
    assert registry["primary_configuration"]["detector_global_tail_weight"] == 0
    assert registry["primary_configuration"]["actuator_feasibility_steps"] == 100
    assert registry["primary_configuration"][
        "training_only_actuator_width_sweep"
    ]
    assert registry["primary_configuration"]["actuator_feasibility_caps"] == [1.5]
    assert registry["primary_configuration"]["actuator_feasibility_widths"] == [
        4,
        8,
        16,
    ]
    assert registry["primary_configuration"]["actuator_architecture"] == (
        "separate_threshold_gated_actuator_bank"
    )
    assert registry["primary_configuration"]["actuator_steps"] == 0
    assert registry["primary_configuration"]["actuator_protected_batch"] == 80
    common = [
        "--model-path",
        "model",
        "--training-visible-path",
        "direct.json",
        "--split-manifest",
        "split.json",
        "--context-manifest",
        "contexts.json",
        "--stage1-state",
        "writer.pt",
        "--stage1-report",
        "writer-report.json",
        "--stage1-writer-log",
        "writer-log.jsonl",
        "--clean-stage1-portability-preflight",
        "clean-stage1-portability.json",
        "--clean-stage1-acceptance",
        "clean-stage1-acceptance.json",
        "--experiment-registry",
        str(registry_path),
        "--output-dir",
        "out",
    ]
    primary = method.parse_args(
        [
            *common,
            "--detector-initialization",
            "frozen_v3_2",
            "--frozen-v3-2-run-dir",
            "rejected-v3.2",
            "--frozen-v3-4-run-dir",
            "rejected-v3.4",
            "--frozen-v3-5-run-dir",
            "rejected-v3.5",
            "--frozen-v3-5-1-run-dir",
            "forensic-v3.5.1",
            "--frozen-v3-5-2-run-dir",
            "rejected-v3.5.2",
            "--frozen-v3-5-3-run-dir",
            "rejected-v3.5.3",
            "--frozen-v3-5-4-run-dir",
            "rejected-v3.5.4",
            "--detector-steps",
            "100",
            "--detector-global-tail-weight",
            "0",
            "--training-only-actuator-width-sweep",
            "--actuator-feasibility-widths",
            "4",
            "8",
            "16",
            "--actuator-architecture",
            "separate_threshold_gated_actuator_bank",
            "--actuator-relative-cap",
            "1.5",
            "--actuator-steps",
            "0",
        ]
    )
    method._validate_experiment_registry(registry, primary)


def test_v3_5_5_launcher_sweeps_disjoint_actuator_widths_and_discards_fits():
    root = Path(__file__).resolve().parents[1]
    manual = (
        root / "scripts" / "run_mcf_embedding_keyed_neuron_v3_5_5_manual.sh"
    ).read_text(encoding="utf-8")
    assert "[[ $# -ne 9 ]]" in manual
    assert "FROZEN_V3_5_4_OUTPUT_DIR" in manual
    assert "run_mcf_embedding_keyed_neuron_v3_5_5_actuator_width_sweep" in manual
    launcher = (
        root
        / "slurm"
        / "run_mcf_embedding_keyed_neuron_v3_5_5_actuator_width_sweep_seed1_3b.slurm"
    ).read_text(encoding="utf-8")
    assert "--training-only-actuator-width-sweep" in launcher
    assert "--actuator-feasibility-caps 1.50" in launcher
    assert "--actuator-feasibility-widths 4 8 16" in launcher
    assert (
        "--actuator-architecture separate_threshold_gated_actuator_bank" in launcher
    )
    assert "--threshold-gate-numerical-guard 1e-6" in launcher
    assert "--actuator-steps 0" in launcher
    assert "--frozen-v3-4-run-dir" in launcher
    assert "--frozen-v3-5-run-dir" in launcher
    assert "--frozen-v3-5-1-run-dir" in launcher
    assert "--frozen-v3-5-2-run-dir" in launcher
    assert "--frozen-v3-5-3-run-dir" in launcher
    assert "--frozen-v3-5-4-run-dir" in launcher
    assert "--detector-steps 100" in launcher
    assert "--detector-global-tail-weight 0" in launcher
    assert "exact_v3_5_4_detector_replay.json" in launcher
    assert "actuator_neuron_selection_report.json" in launcher
    assert "v3_5_5_actuator_width_feasibility.json" in launcher
    assert "training_only_v3_5_5_completion.json" in launcher
    assert "matched_width4_group_budget_control" in launcher
    assert "maximum_actuator_neuron_count == 800" in launcher
    assert "all_fitted_weights_discarded == true" in launcher
    assert "env -u MCF_PATH" in launcher
    assert "--save-checkpoint" not in launcher
    assert "mcf_zero_unlearn_official_eval.py" not in launcher
    assert "official_evaluation_allowed == false" in launcher
    submit = (
        root / "scripts" / "submit_mcf_embedding_keyed_neuron_seed1.sh"
    ).read_text(encoding="utf-8")
    assert "mcf_embedding_keyed_neuron_v3_5_5_${JOB_ID}.out" in submit
    assert "mcf_embedding_keyed_neuron_v3_5_4_${JOB_ID}.out" not in submit


def test_final_report_requires_metrics_mechanism_and_firewall_to_pass():
    metrics = {
        "forget": {"Eff": 0.0, "Gen": 0.0, "Spe": 10.0, "Spe_success": 90.0},
        "retain": {"Eff": 80.0, "Gen": 82.0, "Spe": 11.0, "Spe_success": 85.0},
        "PPL": 16.0,
    }
    method_summary = {
        "acceptance": {
            "passed": True,
            "detector_gate_passed": True,
            "lm_head_bit_identical": True,
        },
        "causal_component_ablation": {
            "writer_is_necessary": True,
            "decoder_is_necessary": True,
        },
        "architecture": {
            "lm_head_edited": False,
            "runtime_string_matcher": False,
            "external_router": False,
            "retrieval_cache": False,
            "sidecar": False,
        },
        "data_firewall": {
            "clean_stage1_writer": {
                "from_scratch": True,
                "writer_steps": 1200,
                "positive_context_policy": "relation_templates_only_v1",
                "writer_log_event_count": 49,
            },
            "clean_stage1_acceptance": _clean_stage1_acceptance_summary(),
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            },
        },
    }
    result = report.build_report(
        metrics,
        metrics,
        method_summary,
        {"passed": True},
        max_abs_spe_delta=0.2,
        max_abs_retain_eff_gen_delta=1.0,
        max_ppl_percent_delta=5.0,
    )
    assert result["status"] == "PASS"
    damaged = dict(metrics)
    damaged["PPL"] = 20.0
    failed = report.build_report(
        metrics,
        damaged,
        method_summary,
        {"passed": True},
        max_abs_spe_delta=0.2,
        max_abs_retain_eff_gen_delta=1.0,
        max_ppl_percent_delta=5.0,
    )
    assert failed["status"] == "FAIL"
    assert not failed["checks"]["PPL_local"]


def test_matched_mlp_only_success_falsifies_embedding_key_necessity():
    forget_raw = [
        {
            "requested_rewrite": {
                "prompt": "{} was born in",
                "subject": "Ada",
                "target_true": {"str": "London"},
                "target_new": {"str": "Paris"},
            }
        }
    ]
    retain_raw = [
        {
            "requested_rewrite": {
                "prompt": "{} was born in",
                "subject": "Grace",
                "target_true": {"str": "New York"},
                "target_new": {"str": "Paris"},
            }
        }
    ]
    base = {
        "dataset": "MCF",
        "sample_mode": "official",
        "seed": 1,
        "unlearn_num": 50,
        "retain_num": 1000,
        "forget": {"Eff": 30.0, "Gen": 40.0, "Spe": 10.0, "Spe_success": 90.0},
        "retain": {"Eff": 80.0, "Gen": 82.0, "Spe": 11.0, "Spe_success": 85.0},
        "PPL": 16.0,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
    }
    successful = {
        "dataset": "MCF",
        "sample_mode": "official",
        "seed": 1,
        "unlearn_num": 50,
        "retain_num": 1000,
        "forget": {"Eff": 0.0, "Gen": 0.0, "Spe": 10.0, "Spe_success": 90.0},
        "retain": dict(base["retain"]),
        "PPL": 16.0,
        "forget_raw": forget_raw,
        "retain_raw": retain_raw,
    }
    budget = {
        "detector_response_mode": "absolute_signed_group_activation",
        "dormant_fraction": 0.2,
        "selection_stability_weight": 1.0,
        "selection_positive_contexts": 8,
        "selection_negative_contexts": 8,
        "detector_steps": 1000,
        "detector_lr": 0.001,
        "detector_record_batch": 4,
        "detector_record_batch_semantics": (
            "gradient_accumulation_microbatch_capacity"
        ),
        "detector_update_coverage": "all_records_accumulated",
        "detector_records_per_optimizer_update": 50,
        "detector_microbatches_per_optimizer_update": 13,
        "detector_record_exposures": 50000,
        "detector_positive_contexts": "all",
        "detector_negative_contexts": "all",
        "detector_tail_k": 2,
        "detector_positive_objective": "mean_plus_worst_k_squared_shortfall",
        "detector_negative_objective": "mean_plus_worst_k_squared_gate_excess",
        "detector_cross_objective": "mean_plus_worst_k_squared_gate_excess",
        "detector_writer_off_objective": ("mean_plus_worst_k_squared_gate_excess"),
        "detector_gradient_normalization": "equal_record_mean",
        "detector_cached_mlp_inputs": True,
        "detector_positive_floor": 0.25,
        "detector_off_abs_max": 0.2,
        "detector_training_positive_floor": 0.3,
        "detector_training_off_abs_max": 0.15,
        "detector_certificate_abs_tolerance": 1e-7,
        "detector_negative_weight": 5.0,
        "detector_cross_weight": 2.0,
        "detector_consistency_weight": 1.0,
        "detector_l2": 0.00001,
        "detector_relative_cap": 1.0,
        "actuator_training_revision": "v3.3",
        "actuator_feasibility_steps": 100,
        "actuator_feasibility_initialization": "exact_zero_down_delta",
        "actuator_feasibility_objective": (
            "equal_record_mean_plus_worst_two_squared_margin_shortfall_only"
        ),
        "actuator_feasibility_positive_exposures": 34600,
        "actuator_steps": 100,
        "actuator_lr": 0.0005,
        "actuator_batch_size": 4,
        "actuator_batch_size_semantics": (
            "context_microbatch_capacity_inside_global_update"
        ),
        "actuator_protected_batch": 80,
        "actuator_protected_sampling_seed": 78104,
        "actuator_update_coverage": "all_records_all_contexts_accumulated",
        "actuator_records_per_optimizer_update": 50,
        "actuator_positive_contexts": "all",
        "actuator_negative_contexts": "all",
        "actuator_positive_contexts_per_optimizer_update": 346,
        "actuator_negative_contexts_per_optimizer_update": 465,
        "actuator_writer_off_contexts_per_optimizer_update": 346,
        "actuator_positive_context_exposures": 34600,
        "actuator_negative_context_exposures": 46500,
        "actuator_writer_off_context_exposures": 34600,
        "actuator_protected_context_exposures": 8000,
        "actuator_tail_k": 2,
        "actuator_positive_objective": (
            "equal_record_mean_plus_worst_k_squared_margin_shortfall"
        ),
        "actuator_reference_objective": (
            "equal_record_mean_plus_worst_k_squared_excess_above_tolerance"
        ),
        "actuator_negative_objective": (
            "equal_record_mean_plus_worst_k_squared_nll_drift"
        ),
        "actuator_writer_off_objective": (
            "equal_record_mean_plus_worst_k_squared_abs_nll_drift_excess"
        ),
        "actuator_gradient_normalization": (
            "equal_record_mean_plus_global_protected_prompt_mean"
        ),
        "actuator_gradient_clip_frequency": "once_per_optimizer_update",
        "actuator_norm_projection_frequency": "once_per_optimizer_update",
        "actuator_complete_training_log_required": True,
        "actuator_endpoint_audit_phases": [
            "pre_update",
            "post_adam",
            "post_projection",
            "final_fresh_full_context_audit",
        ],
        "actuator_writer_off_every": 1,
        "actuator_writer_off_nll_tolerance": 0.05,
        "actuator_relative_cap": 0.5,
        "actuator_l2": 0.0001,
        "neurons_per_record": 4,
        "selected_existing_mlp_neurons": 200,
        "mlp_layer": 27,
        "protected_prompt_count": 8192,
        "protected_kl_weight": 20.0,
        "margin_weight": 20.0,
        "reference_nll_weight": 50.0,
        "reference_nll_tolerance": 0.05,
        "forget_margin": 1.0,
        "grad_clip": 1.0,
        "kl_topk": 64,
    }
    proposed_summary = {
        "protocol": report.EXPECTED_PROTOCOL,
        "seed": 1,
        "forget_num": 50,
        "writer_mode": "embedding_keyed",
        "optimization_budget": budget,
        "acceptance": {
            "passed": True,
            "detector_gate_passed": True,
            "lm_head_bit_identical": True,
        },
        "causal_component_ablation": {
            "writer_is_necessary": True,
            "decoder_is_necessary": True,
        },
        "architecture": {
            "lm_head_edited": False,
            "runtime_string_matcher": False,
            "external_router": False,
            "retrieval_cache": False,
            "sidecar": False,
        },
        "data_firewall": {
            "training_visible_sha256": "training-hash",
            "context_manifest_sha256": "context-hash",
            "split_manifest_sha256": "split-hash",
            "stage1_state_sha256": "stage1-hash",
            "stage1_report_sha256": "stage1-report-hash",
            "stage1_writer_log_sha256": "stage1-log-hash",
            "clean_stage1_acceptance_sha256": "stage1-acceptance-hash",
            "clean_stage1_portability_sha256": "stage1-portability-hash",
            "experiment_registry_sha256": "registry-hash",
            "clean_stage1_writer": {
                "from_scratch": True,
                "writer_steps": 1200,
                "positive_context_policy": "relation_templates_only_v1",
                "writer_log_event_count": 49,
            },
            "clean_stage1_acceptance": _clean_stage1_acceptance_summary(),
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            },
        },
    }
    control_summary = {
        "protocol": report.EXPECTED_PROTOCOL,
        "seed": 1,
        "forget_num": 50,
        "writer_mode": "none",
        "experiment_label": "mlp_only_retrained",
        "optimization_budget": dict(budget),
        "acceptance": {"passed": True},
        "architecture": {"input_embedding_rows_edited": 0},
        "data_firewall": {
            "training_visible_sha256": "training-hash",
            "context_manifest_sha256": "context-hash",
            "split_manifest_sha256": "split-hash",
            "stage1_state_sha256": "stage1-hash",
            "stage1_report_sha256": "stage1-report-hash",
            "stage1_writer_log_sha256": "stage1-log-hash",
            "clean_stage1_acceptance_sha256": "stage1-acceptance-hash",
            "clean_stage1_portability_sha256": "stage1-portability-hash",
            "experiment_registry_sha256": "registry-hash",
            "clean_stage1_writer": {
                "from_scratch": True,
                "writer_steps": 1200,
                "positive_context_policy": "relation_templates_only_v1",
                "writer_log_event_count": 49,
            },
            "clean_stage1_acceptance": _clean_stage1_acceptance_summary(),
        },
    }
    result = report.build_report(
        base,
        successful,
        proposed_summary,
        {"passed": True},
        max_abs_spe_delta=0.2,
        max_abs_retain_eff_gen_delta=1.0,
        max_ppl_percent_delta=5.0,
        mlp_only=successful,
        mlp_only_method=control_summary,
        mlp_only_reload={"passed": True},
        mlp_only_retain_tail=_tail_receipt(writer_mode="none"),
        writer_portability=_portability_receipt(),
        retain_tail=_tail_receipt(),
        latent_recovery=_latent_receipt(fact_recoverable=False),
        relearning=_relearning_receipt(fact_recoverable=False),
        require_complete_mechanism_evidence=True,
    )
    assert result["status"] == "FAIL"
    assert result["matched_mlp_only_control"][
        "meets_same_forgetting_and_locality_envelope"
    ]
    assert result["matched_mlp_only_control"]["embedding_key_necessity_falsified"]
    assert not result["checks"]["registered_control_supports_keyed_advantage"]

    weaker_control = {
        **successful,
        "forget": {**successful["forget"], "Gen": 5.0},
    }
    supported = report.build_report(
        base,
        successful,
        proposed_summary,
        {"passed": True},
        max_abs_spe_delta=0.2,
        max_abs_retain_eff_gen_delta=1.0,
        max_ppl_percent_delta=5.0,
        mlp_only=weaker_control,
        mlp_only_method=control_summary,
        mlp_only_reload={"passed": True},
        mlp_only_retain_tail=_tail_receipt(writer_mode="none"),
        writer_portability=_portability_receipt(),
        retain_tail=_tail_receipt(),
        latent_recovery=_latent_receipt(fact_recoverable=True),
        relearning=_relearning_receipt(fact_recoverable=True),
        require_complete_mechanism_evidence=True,
    )
    assert supported["status"] == "PASS"
    assert supported["checks"]["registered_control_supports_keyed_advantage"]
    assert not supported["matched_mlp_only_control"][
        "architecture_level_necessity_proven"
    ]
    assert supported["knowledge_recovered_by_diagnostic"]
    assert not supported["knowledge_removal_claim_allowed"]


def test_paper_report_rejects_placeholder_audit_receipts():
    assert not report._retain_tail_complete({"acceptance": {"passed": True}})
    assert not report._writer_portability_complete({"acceptance": {"passed": True}})
    assert not report._latent_endpoint_complete({"fact_recoverable": False})
    assert not report._relearning_endpoint_complete({"fact_recoverable": False})
