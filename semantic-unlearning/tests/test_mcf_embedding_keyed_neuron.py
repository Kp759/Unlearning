from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_embedding_keyed_neuron_core as core
import mcf_embedding_keyed_neuron_erasure as method
import report_mcf_embedding_keyed_neuron_result as report


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


def test_contextual_code_response_uses_owned_multibit_group():
    ownership = [[2, 4], [1, 7]]
    signs = [torch.tensor([1.0, -1.0]), torch.tensor([-1.0, 1.0])]
    flat, flat_signs, local = core.flatten_ownership(ownership, signs)
    assert flat == [2, 4, 1, 7]

    baseline = torch.zeros(2, 4)
    edited = torch.tensor([[3.0, -3.0, 0.0, 0.0], [0.0, 0.0, -2.0, 2.0]])
    response = core.contextual_code_responses(edited, baseline, local, flat_signs)
    assert torch.allclose(response, torch.tensor([[3.0, 0.0], [0.0, 2.0]]))


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
                "--experiment-registry",
                "registry.json",
                "--output-dir",
                "out",
                "--frequency-doc-start",
                "0",
            ]
        )


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
            "--experiment-registry",
            "registry.json",
            "--output-dir",
            "out",
        ]
    )
    registry = {
        "protocol": method.PROTOCOL,
        "primary_configuration": {
            "forget_num": 50,
            "neuron_layer": 8,
            "neurons_per_record": 4,
            "selection_mode": "writer_contrastive",
            "dormant_fraction": 0.35,
            "detector_relative_cap": 0.4,
            "actuator_relative_cap": 0.4,
            "gate_policy": "strict",
        },
    }
    method._validate_experiment_registry(registry, args)
    args.neuron_layer = 12
    with pytest.raises(RuntimeError, match="diverges from registry"):
        method._validate_experiment_registry(registry, args)


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
            "data_access": {
                "official_paraphrases_seen": 0,
                "official_neighborhoods_seen": 0,
                "benchmark_retain_seen": 0,
                "official_ppl_seen": False,
            }
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
