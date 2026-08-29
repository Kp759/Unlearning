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
import audit_mcf_frozen_writer_portability as portability_audit
import aggregate_mcf_context_gating_frequency_factorial as frequency_aggregate
import report_mcf_embedding_keyed_neuron_result as report
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
        positive_floor=1.0,
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
        positive_floor=1.0,
        negative_weight=2.0,
        cross_weight=2.0,
    )
    assert good_loss == pytest.approx(0.0)
    assert bad_loss > good_loss


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
        "primary_configuration": {
            "forget_num": 50,
            "neuron_layer": 27,
            "neurons_per_record": 4,
            "selection_mode": "writer_contrastive",
            "dormant_fraction": 0.2,
            "detector_relative_cap": 1.0,
            "actuator_relative_cap": 0.5,
            "gate_policy": "strict",
        },
    }
    method._validate_experiment_registry(registry, args)
    args.neuron_layer = 24
    with pytest.raises(RuntimeError, match="diverges from registry"):
        method._validate_experiment_registry(registry, args)


def test_repository_registry_binds_primary_and_independent_control():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "mcf_embedding_keyed_neuron_ablation_registry_v1.json"
    )
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
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
    primary = method.parse_args(common)
    method._validate_experiment_registry(registry, primary)

    control = method.parse_args(
        [
            *common,
            "--experiment-label",
            "mlp_only_retrained",
            "--writer-mode",
            "none",
            "--selection-mode",
            "base_context_contrastive",
            "--detector-writer-off-weight",
            "0",
            "--writer-off-nll-weight",
            "0",
            "--no-require-writer-necessity",
            "--save-rejected-checkpoint",
        ]
    )
    method._validate_experiment_registry(registry, control)


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
        "detector_positive_contexts": 2,
        "detector_negative_contexts": 2,
        "detector_positive_floor": 0.25,
        "detector_off_abs_max": 0.2,
        "detector_negative_weight": 5.0,
        "detector_cross_weight": 2.0,
        "detector_consistency_weight": 1.0,
        "detector_l2": 0.00001,
        "detector_relative_cap": 1.0,
        "actuator_steps": 2000,
        "actuator_lr": 0.0005,
        "actuator_batch_size": 4,
        "actuator_protected_batch": 4,
        "actuator_writer_off_every": 1,
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
        "protocol": method.PROTOCOL,
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
        "protocol": method.PROTOCOL,
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
