from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_projected_row_partition_embedding_v2_3_core as core  # noqa: E402
import run_mcf_projected_row_partition_embedding_v2_3 as runner  # noqa: E402


REGISTRY_PATH = (
    ROOT / "protocols" / "mcf_projected_row_partition_embedding_v2_3_registry.json"
)
REACHABILITY_REGISTRY_PATH = (
    ROOT
    / "protocols"
    / "mcf_projected_row_partition_embedding_v2_3_1_registry.json"
)
SENSITIVITY_REGISTRY_PATH = (
    ROOT
    / "protocols"
    / "mcf_projected_row_partition_embedding_v2_3_2_registry.json"
)


def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


def reachability_registry() -> dict:
    return json.loads(REACHABILITY_REGISTRY_PATH.read_text())


def sensitivity_registry() -> dict:
    return json.loads(SENSITIVITY_REGISTRY_PATH.read_text())


def basis(vectors: list[list[float]]) -> torch.Tensor:
    if not vectors:
        return torch.zeros((0, 3), dtype=torch.float32)
    return torch.tensor(vectors, dtype=torch.float32)


def test_registry_freezes_the_lm_head_and_keeps_projection_first_order() -> None:
    value = registry()
    assert value["status"] == "training_only_implementation_available_not_executed"
    architecture = value["architecture"]
    assert architecture["trainable_parameter_families"] == [
        "selected_input_embedding_rows"
    ]
    assert architecture["lm_head_trainable"] is False
    assert architecture["lm_head_frozen_bit_identical"] is True
    assert architecture["output_row_selection"] == "none"
    assert architecture["transformer_frozen"] is True
    assert architecture["external_classifier"] is False
    assert architecture["runtime_gate"] is False
    diagnostic = value["row_partition_diagnostic"]
    # V2 read 141 empty bases as "unused"; V2.3 must not repeat that.
    assert diagnostic["empty_basis_implies_forget_exclusive"] is False
    assert diagnostic["projection_is_first_order_only"] is True
    assert diagnostic["direct_prompt_liveness_forced"] is True
    optimization = value["optimization"]
    assert optimization["adam_forbidden"] is True
    assert optimization["nonlinear_forward_acceptance"] is True
    assert optimization["surgical_weight"] == 0.0
    assert value["acceptance"]["lm_head_bit_identical"] is True
    assert value["retain_strata"]["present_on_every_embedding_update"] is True


def test_registry_and_cli_are_exactly_locked() -> None:
    args = runner.parse_args(
        [
            "--model-path",
            "model",
            "--protocol-dir",
            "protocol",
            "--experiment-registry",
            "registry",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
        ]
    )
    runner.validate_registry(registry(), args)
    assert args.steps == 1000
    assert args.surgical_weight == 0.0
    assert args.retain_jacobian_sketches == 192
    assert args.partition_efficacy_min == 0.05
    assert args.input_step_cap_fraction == 0.004


def test_registry_rejects_a_trainable_head_variant() -> None:
    value = registry()
    value["architecture"]["lm_head_trainable"] = True
    args = runner.parse_args(
        [
            "--model-path",
            "model",
            "--protocol-dir",
            "protocol",
            "--experiment-registry",
            "registry",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
        ]
    )
    with pytest.raises(RuntimeError, match="architecture/status mismatch"):
        runner.validate_registry(value, args)


def test_contrast_direction_is_the_normalized_answer_difference() -> None:
    weight = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float32
    )
    directions, valid = core.answer_contrast_directions(
        weight, torch.tensor([0, 1]), torch.tensor([1, 0])
    )
    assert bool(valid.all())
    expected = torch.nn.functional.normalize(
        torch.tensor([[-1.0, 2.0, 0.0], [1.0, -2.0, 0.0]]), dim=-1
    )
    assert torch.allclose(directions, expected, atol=1e-6)
    # The two roles are exact opposites, which W[target_true] alone cannot show.
    assert torch.allclose(directions[0], -directions[1], atol=1e-6)


def test_contrast_masks_shared_leading_and_unpaired_answer_tokens() -> None:
    weight = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    directions, valid = core.answer_contrast_directions(
        weight, torch.tensor([0, 0, -1]), torch.tensor([0, 1, 1])
    )
    assert valid.tolist() == [False, True, False]
    assert float(directions[0].norm()) == 0.0
    assert float(directions[2].norm()) == 0.0


def test_contrast_masks_numerically_degenerate_difference() -> None:
    weight = torch.tensor([[1.0, 0.0], [1.0, 1e-9]], dtype=torch.float32)
    _, valid = core.answer_contrast_directions(
        weight, torch.tensor([0]), torch.tensor([1]), epsilon=1e-3
    )
    assert valid.tolist() == [False]


def test_surgical_penalty_rewards_aligned_movement_and_ignores_invalid() -> None:
    directions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    valid = torch.tensor([True, False])
    base = torch.zeros((2, 2))
    aligned = torch.tensor([[2.0, 0.0], [0.0, 9.0]])
    penalty, detail = core.contrast_surgical_penalty(
        aligned, base, directions, valid, sign_margin=1.0
    )
    # Row 0 moves exactly along q with margin satisfied; row 1 is masked out.
    assert float(penalty) == pytest.approx(0.0, abs=1e-6)
    assert detail["valid_cells"] == 1
    off_axis = torch.tensor([[0.0, 3.0], [0.0, 0.0]])
    penalty_off, _ = core.contrast_surgical_penalty(
        off_axis, base, directions, valid, sign_margin=1.0
    )
    # 9 from the orthogonal term plus 1 from the unmet sign hinge.
    assert float(penalty_off) == pytest.approx(10.0, abs=1e-6)


def test_residual_efficacy_separates_angle_from_usable_potential() -> None:
    gradient = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1e-6], [0.0, 1.0, 0.0]])
    bases = [
        basis([[1.0, 0.0, 0.0]]),  # fully absorbed
        basis([[1.0, 0.0, 0.0]]),  # orthogonal but negligible
        basis([]),  # unobserved
    ]
    caps = torch.tensor([1.0, 1.0, 1.0])
    result = core.residual_efficacy(gradient, bases, caps)
    assert float(result["efficacy"][0]) == pytest.approx(0.0, abs=1e-6)
    assert float(result["efficacy"][1]) == pytest.approx(1.0, abs=1e-4)
    assert float(result["potential"][1]) < 1e-5
    assert float(result["efficacy"][2]) == pytest.approx(1.0, abs=1e-6)


def test_cap_aware_prompt_reachability_uses_final_roles_and_row_caps() -> None:
    gradient = torch.tensor(
        [[3.0, 4.0, 0.0], [2.0, 3.0, 0.0], [9.0, 0.0, 0.0]]
    )
    bases = [basis([]), basis([[1.0, 0.0, 0.0]]), basis([])]
    roles = [core.FREE, core.PROJECTED, core.EXCLUDED]
    caps = torch.tensor([2.0, 1.0, 5.0])
    report = core.cap_aware_prompt_reachability(
        gradient,
        bases,
        roles,
        caps,
        base_margin=-10.0,
        required_margin=0.1,
    )
    # Free row: 2 * 5 = 10; projected row: 1 * 3 = 3; excluded: 0.
    assert report["maximum_first_order_improvement"] == pytest.approx(13.0)
    assert report["predicted_maximum_margin"] == pytest.approx(3.0)
    assert report["usable_gradient_rows"] == 2
    assert report["passed"] is True
    direction = core.cap_saturating_margin_direction(
        gradient, bases, roles, caps
    )
    assert float(direction[0].norm()) == pytest.approx(2.0, abs=1e-6)
    assert torch.allclose(direction[1], torch.tensor([0.0, 1.0, 0.0]))
    assert float(direction[2].norm()) == 0.0


def test_cap_aware_prompt_reachability_fails_when_projection_removes_capacity() -> None:
    report = core.cap_aware_prompt_reachability(
        torch.tensor([[2.0, 0.0, 0.0]]),
        [basis([[1.0, 0.0, 0.0]])],
        [core.PROJECTED],
        torch.tensor([100.0]),
        base_margin=-1.0,
        required_margin=0.1,
    )
    assert report["maximum_first_order_improvement"] == pytest.approx(0.0)
    assert report["usable_gradient_rows"] == 0
    assert report["passed"] is False


def test_strict_trust_region_rejects_equal_or_tiny_forget_progress() -> None:
    kwargs = {
        "before_forget": 10.0,
        "before_constraint_score": 0.5,
        "candidate_constraint_score": 0.8,
        "minimum_forget_improvement": 1e-3,
    }
    assert not core.accept_strict_trust_region_candidate(
        candidate_forget=10.0, **kwargs
    )
    assert not core.accept_strict_trust_region_candidate(
        candidate_forget=9.9995, **kwargs
    )
    assert core.accept_strict_trust_region_candidate(
        candidate_forget=9.998, **kwargs
    )
    assert not core.accept_strict_trust_region_candidate(
        before_forget=10.0,
        candidate_forget=9.0,
        before_constraint_score=0.5,
        candidate_constraint_score=1.1,
        minimum_forget_improvement=1e-3,
    )


def test_v2_3_1_registry_locks_reachability_strict_progress_and_rollback() -> None:
    args = runner.parse_args(
        [
            "--model-path",
            "model",
            "--protocol-dir",
            "protocol",
            "--experiment-registry",
            "registry",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
            "--require-per-prompt-reachability",
            "--minimum-forget-loss-improvement",
            "0.00001",
            "--full-fit-rollback",
        ]
    )
    runner.validate_registry(reachability_registry(), args)
    value = reachability_registry()
    assert value["protocol"] == core.REACHABILITY_PROTOCOL
    assert value["per_prompt_reachability"]["required_before_optimization"]
    assert value["optimization"]["strict_forget_improvement"]
    assert value["optimization"]["full_fit_rollback"]


def test_v2_3_1_registry_refuses_missing_runtime_controls() -> None:
    args = runner.parse_args(
        [
            "--model-path",
            "model",
            "--protocol-dir",
            "protocol",
            "--experiment-registry",
            "registry",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
        ]
    )
    with pytest.raises(RuntimeError, match="V2.3.1 reachability"):
        runner.validate_registry(reachability_registry(), args)


def test_per_prompt_forward_reachability_is_separate_and_restores_zero(
    monkeypatch,
) -> None:
    delta = runner.canonical.SelectedRowDelta(
        2, 2, direction_basis=None, device=torch.device("cpu")
    )
    records = {
        "direct": [
            {
                "case_id": 7,
                "base_margin": -1.0,
                "gradient": [[2.0, 0.0], [0.0, 0.0]],
                "requested_rewrite": {"prompt": "{} works as", "subject": "A"},
            }
        ],
        "synthetic": [
            {
                "case_id": 7,
                "base_margin": -1.0,
                "gradient": [[0.0, 0.0], [5.0, 0.0]],
                "requested_rewrite": {"prompt": "Tell me about {}", "subject": "A"},
            }
        ],
    }

    def fake_margin(_model, _tok, batch, _device, *, llama_like):
        del llama_like
        values = []
        for record in batch:
            local_gradient = torch.tensor(record["gradient"], dtype=torch.float32)
            values.append(
                torch.tensor(float(record["base_margin"]))
                + (delta.raw_delta * local_gradient).sum()
            )
        return torch.stack(values)

    monkeypatch.setattr(runner.v2, "records_margin_tensor", fake_margin)
    model = torch.nn.Linear(1, 1)
    report = runner.per_prompt_reachability_report(
        model,
        None,
        records,
        torch.device("cpu"),
        input_delta=delta,
        input_caps=torch.tensor([1.0, 1.0]),
        bases=[basis([]), basis([])],
        roles=[core.FREE, core.EXCLUDED],
        llama_like=False,
        required_margin=0.1,
    )
    assert report["groups"]["direct"]["failures"] == 0
    assert report["groups"]["synthetic"]["failures"] == 1
    assert report["passed"] is False
    assert torch.count_nonzero(delta.raw_delta).item() == 0


def test_per_example_sensitivity_distinguishes_useful_shared_and_protected_rows() -> None:
    forget = torch.zeros((4, 4))
    retain = torch.zeros((100, 4))
    # Row 0: frequent/strong on forget, rare on retain -> forget-specific.
    forget[:2, 0] = 1.0
    retain[0, 0] = 1.0
    # Row 1: similarly prevalent in both populations -> shared.
    forget[:2, 1] = 1.0
    retain[:50, 1] = 1.0
    # Row 2: negligible forget importance -> low-forget.
    forget[0, 2] = 1e-8
    # Row 3: retain hard tail dominates despite nonzero forget usage.
    forget[:, 3] = 0.2
    retain[0, 3] = 2.0
    report = core.summarize_per_example_sensitivity(
        forget,
        retain,
        forget_coverage=torch.tensor([2, 2, 1, 4]),
        retain_coverage=torch.tensor([1, 50, 0, 1]),
        forget_importance_floor_relative=0.01,
        importance_ratio_min=1.0,
        forget_specific_ratio_min=4.0,
        forget_specific_retain_coverage_max=0.01,
        retain_tail_ratio_min=0.25,
    )
    assert report["classes"] == [
        "forget_specific",
        "shared",
        "low_forget",
        "retain_dominant",
    ]
    assert report["per_row"][0]["importance_ratio"] > 4.0
    assert report["per_row"][3]["hard_tail_ratio"] < 0.25


def test_sensitivity_can_only_tighten_geometric_roles() -> None:
    geometry_roles = [core.FREE, core.PROJECTED, core.FREE, core.PROJECTED]
    geometry_report = {
        "per_row": [
            {"reason": "free"},
            {"reason": "projected"},
            {"reason": "free"},
            {"reason": "projected"},
        ]
    }
    sensitivity_report = {
        "class_counts": {
            "forget_specific": 1,
            "shared": 2,
            "retain_dominant": 1,
            "low_forget": 0,
        },
        "per_row": [
            {
                "class": "forget_specific",
                "forget_importance_rms": 2.0,
                "retain_importance_rms": 0.0,
                "retain_importance_max": 0.0,
                "importance_ratio": 10.0,
                "forget_coverage": 2,
                "retain_coverage": 0,
            },
            {
                "class": "shared",
                "forget_importance_rms": 1.0,
                "retain_importance_rms": 1.0,
                "retain_importance_max": 1.0,
                "importance_ratio": 1.0,
                "forget_coverage": 2,
                "retain_coverage": 4,
            },
            {
                "class": "shared",
                "forget_importance_rms": 1.0,
                "retain_importance_rms": 1.0,
                "retain_importance_max": 1.0,
                "importance_ratio": 1.0,
                "forget_coverage": 2,
                "retain_coverage": 1,
            },
            {
                "class": "retain_dominant",
                "forget_importance_rms": 0.5,
                "retain_importance_rms": 2.0,
                "retain_importance_max": 3.0,
                "importance_ratio": 0.25,
                "forget_coverage": 1,
                "retain_coverage": 9,
            },
        ],
    }
    roles, report = core.combine_geometry_and_sensitivity_roles(
        row_ids=[10, 11, 12, 13],
        geometry_roles=geometry_roles,
        geometry_report=geometry_report,
        sensitivity_report=sensitivity_report,
    )
    assert roles == [core.FREE, core.PROJECTED, core.EXCLUDED, core.EXCLUDED]
    assert report["per_row"][2]["reason"] == (
        "retain_observed_outside_geometry_basis"
    )
    assert report["liveness_forcing_disabled"] is True


def test_v2_3_2_registry_locks_per_example_sensitivity() -> None:
    args = runner.parse_args(
        [
            "--model-path",
            "model",
            "--protocol-dir",
            "protocol",
            "--experiment-registry",
            "registry",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
            "--require-per-example-sensitivity",
            "--require-per-prompt-reachability",
            "--minimum-forget-loss-improvement",
            "0.00001",
            "--full-fit-rollback",
        ]
    )
    runner.validate_registry(sensitivity_registry(), args)
    value = sensitivity_registry()
    assert value["protocol"] == core.SENSITIVITY_PROTOCOL
    assert value["per_example_sensitivity"]["forget_examples"] == 50
    assert value["per_example_sensitivity"]["retain_examples"] == 2000
    assert value["per_example_sensitivity"]["prompt_classifier"] is False
    assert value["per_example_sensitivity"]["ratio_is_sole_safety_criterion"] is False


def test_v2_3_2_registry_refuses_disabling_sensitivity() -> None:
    args = runner.parse_args(
        [
            "--model-path",
            "model",
            "--protocol-dir",
            "protocol",
            "--experiment-registry",
            "registry",
            "--wikidata-dir",
            "wiki",
            "--output-dir",
            "out",
            "--require-per-prompt-reachability",
            "--full-fit-rollback",
        ]
    )
    with pytest.raises(RuntimeError, match="V2.3.2 per-example"):
        runner.validate_registry(sensitivity_registry(), args)


def test_sensitivity_collector_uses_separate_examples_and_restores_zero(
    monkeypatch,
) -> None:
    delta = runner.canonical.SelectedRowDelta(
        2, 2, direction_basis=None, device=torch.device("cpu")
    )
    forget = [
        {
            "requested_rewrite": {
                "prompt": "{} forget",
                "subject": "A",
                "target_true": {"str": "old"},
            }
        }
    ]
    retain = [
        {
            "requested_rewrite": {
                "prompt": "{} retain zero",
                "subject": "B",
                "target_true": {"str": "keep"},
            }
        },
        {
            "requested_rewrite": {
                "prompt": "{} retain one",
                "subject": "C",
                "target_true": {"str": "keep"},
            }
        },
    ]

    def fake_margin(_model, _tok, _batch, _device, *, llama_like):
        del llama_like
        return torch.stack([(delta.raw_delta[0] * torch.tensor([2.0, 0.0])).sum()])

    retain_calls = iter(
        [
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [1.0, 0.0]]),
        ]
    )

    def fake_nll(_model, _tok, _prompts, _answers, _device, *, llama_like):
        del llama_like
        local = next(retain_calls)
        return torch.stack([(delta.raw_delta * local).sum()])

    monkeypatch.setattr(runner.v2, "records_margin_tensor", fake_margin)
    monkeypatch.setattr(runner.v2, "answer_nlls", fake_nll)
    gradient, report = runner.collect_per_example_sensitivity(
        torch.nn.Linear(1, 1),
        None,
        forget,
        retain,
        torch.device("cpu"),
        input_delta=delta,
        llama_like=False,
        coverage_relative_epsilon=1e-4,
        forget_importance_floor_relative=0.01,
        importance_ratio_min=1.0,
        forget_specific_ratio_min=4.0,
        forget_specific_retain_coverage_max=0.01,
        retain_tail_ratio_min=0.25,
    )
    assert torch.allclose(gradient[0], torch.tensor([2.0, 0.0]))
    assert report["forget_examples"] == 1
    assert report["retain_examples"] == 2
    assert report["per_row"][0]["forget_coverage"] == 1
    assert report["per_row"][1]["retain_coverage"] == 2
    assert report["prompt_classifier"] is False
    assert torch.count_nonzero(delta.raw_delta).item() == 0


def test_partition_assigns_free_projected_and_excluded_roles() -> None:
    roles, report = core.partition_rows(
        row_ids=[10, 11, 12, 13],
        retain_observed=[False, True, True, True],
        efficacy=torch.tensor([1.0, 0.9, 0.01, 0.9]),
        potential=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        frequency=torch.tensor([1.0, 1.0, 1.0, 99999.0]),
        direct_live_rows={1: [10]},
        efficacy_min=0.05,
        potential_min=0.1,
        frequency_max=5000,
    )
    assert roles == [core.FREE, core.PROJECTED, core.EXCLUDED, core.EXCLUDED]
    assert report["role_counts"] == {"free": 1, "projected": 1, "excluded": 2}
    assert report["editable_rows"] == 2
    assert report["per_row"][2]["reason"] == "near_parallel_to_retain_subspace"
    assert report["per_row"][3]["reason"] == "common_row_beyond_projection_bound"


def test_partition_excludes_rows_with_negligible_potential() -> None:
    roles, report = core.partition_rows(
        row_ids=[10],
        retain_observed=[True],
        efficacy=torch.tensor([1.0]),
        potential=torch.tensor([1e-9]),
        frequency=torch.tensor([1.0]),
        direct_live_rows={},
        efficacy_min=0.05,
        potential_min=0.1,
        frequency_max=5000,
    )
    assert roles == [core.EXCLUDED]
    assert report["per_row"][0]["reason"] == "negligible_cap_adjusted_potential"


def test_partition_forces_direct_prompt_liveness_and_reports_it() -> None:
    # Every direct-prompt row of case 7 would otherwise be excluded, which would
    # leave the record permanently unfixable the way 6dcb11f documented.
    roles, report = core.partition_rows(
        row_ids=[10, 11],
        retain_observed=[True, True],
        efficacy=torch.tensor([0.001, 0.001]),
        potential=torch.tensor([0.4, 0.9]),
        frequency=torch.tensor([1.0, 1.0]),
        direct_live_rows={7: [10, 11]},
        efficacy_min=0.05,
        potential_min=0.1,
        frequency_max=5000,
    )
    assert roles == [core.EXCLUDED, core.PROJECTED]
    assert report["liveness_forced_records"] == 1
    forced = report["liveness_forced"][0]
    assert forced["case_id"] == 7
    assert forced["token_id"] == 11  # the highest-potential direct row
    assert report["per_row"][1]["reason"] == "direct_prompt_liveness_forced"


def test_partition_rejects_a_record_with_no_selected_direct_row() -> None:
    with pytest.raises(ValueError, match="no selected row in its own direct prompt"):
        core.partition_rows(
            row_ids=[10],
            retain_observed=[True],
            efficacy=torch.tensor([1.0]),
            potential=torch.tensor([1.0]),
            frequency=torch.tensor([1.0]),
            direct_live_rows={7: [999]},
            efficacy_min=0.05,
            potential_min=0.1,
            frequency_max=5000,
        )


def test_role_constraints_free_project_and_zero_the_right_rows() -> None:
    delta = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32
    )
    bases = [basis([[1.0, 0.0, 0.0]]), basis([[1.0, 0.0, 0.0]]), basis([])]
    core.apply_role_constraints_(
        delta, bases, [core.FREE, core.PROJECTED, core.EXCLUDED]
    )
    assert torch.allclose(delta[0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(delta[1], torch.tensor([0.0, 2.0, 3.0]), atol=1e-6)
    assert float(delta[2].norm()) == 0.0


def test_role_compliance_detects_leak_and_nonzero_exclusion() -> None:
    bases = [basis([[1.0, 0.0, 0.0]]), basis([])]
    clean = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    assert core.role_compliance_report(
        clean, bases, [core.PROJECTED, core.EXCLUDED]
    )["passed"]
    leaking = torch.tensor([[0.5, 2.0, 0.0], [0.0, 0.0, 1.0]])
    report = core.role_compliance_report(
        leaking, bases, [core.PROJECTED, core.EXCLUDED]
    )
    assert report["passed"] is False
    assert report["excluded_nonzero_rows"] == 1
    assert report["projected_retain_leak_max"] == pytest.approx(0.5, abs=1e-6)


def _stub_bank(monkeypatch, delta, hidden: int, vocab: int, cases: int, rows: int):
    """Differentiable stand-in for the frozen stack, so gradients really flow.

    Case ``i`` reads embedding row ``i % (rows - 1)``, so the final row is never
    touched by the retain bank and must end with an empty basis.
    """
    import run_mcf_biendpoint_nullspace_rewiring_v2 as v2

    generator = torch.Generator().manual_seed(11)
    mixer = torch.randn((hidden, vocab), generator=generator)
    base = torch.randn((cases, vocab), generator=generator)
    prompts = [f"retain prompt {index}" for index in range(cases)]

    def fake_forward(model, tok, batch_cases, device):
        indices = [prompts.index(case.prompt) for case in batch_cases]
        rows_used = torch.tensor([index % (rows - 1) for index in indices])
        return base[torch.tensor(indices)] + delta.raw_delta.index_select(
            0, rows_used
        ) @ mixer

    monkeypatch.setattr(runner.canonical, "forward_last_logits", fake_forward)
    topk = 4
    order = base.argsort(dim=1, descending=True)
    return v2.ProtectionCache(
        cases=[
            runner.canonical.SensitivePredictionCase(
                case_id=index,
                record_position=index,
                token_index=0,
                prompt=prompts[index],
                target_text="x",
            )
            for index in range(cases)
        ],
        topk_ids=order[:, :topk],
        base_topk_log_probs=torch.log_softmax(
            base.gather(1, order[:, :topk]), dim=1
        ),
        base_top1_ids=order[:, 0],
        base_top1_log_probs=torch.log_softmax(base, dim=1).gather(
            1, order[:, :1]
        ).squeeze(1),
    )


def test_retain_bases_build_from_real_gradients_and_leave_unused_rows_empty(
    monkeypatch,
) -> None:
    hidden, vocab, cases, rows = 5, 7, 6, 4
    delta = runner.canonical.SelectedRowDelta(
        rows, hidden, direction_basis=None, device=torch.device("cpu")
    )
    cache = _stub_bank(monkeypatch, delta, hidden, vocab, cases, rows)
    bases, report = runner.build_retain_readout_bases(
        torch.nn.Linear(1, 1),
        None,
        cache,
        torch.zeros(cases, dtype=torch.long),
        torch.device("cpu"),
        input_delta=delta,
        sketches=12,
        batch_size=3,
        topk=4,
        max_rank=64,
    )
    assert len(bases) == rows
    # Rows 0-2 are exercised by the bank; row 3 never is.
    assert all(int(bases[index].shape[0]) > 0 for index in range(rows - 1))
    assert int(bases[rows - 1].shape[0]) == 0
    assert report["rows_with_empty_basis"] == 1
    assert report["rows_observed_in_retain"] == rows - 1
    assert report["empty_basis_means_unobserved_not_unused"] is True
    # All three registered probe families were actually exercised.
    assert set(report["probe_families"]) == set(runner.PROBE_FAMILIES)
    assert all(count > 0 for count in report["probe_families"].values())


def test_projection_then_capping_preserves_orthogonality_and_zeroing() -> None:
    # The training step applies role constraints and only then the row caps.
    # Capping scales a row, so it must not reintroduce retain leak.
    bases = [basis([[1.0, 0.0, 0.0]]), basis([[0.0, 1.0, 0.0]]), basis([])]
    roles = [core.PROJECTED, core.EXCLUDED, core.FREE]
    delta = torch.tensor(
        [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0], [9.0, 9.0, 9.0]], dtype=torch.float32
    )
    caps = torch.tensor([1.0, 1.0, 1.0])
    core.apply_role_constraints_(delta, bases, roles)
    import mcf_biendpoint_nullspace_rewiring_v2_core as geometry

    geometry.apply_row_caps_(delta, caps)
    report = core.role_compliance_report(delta, bases, roles)
    assert report["passed"] is True
    assert float(delta[1].norm()) == 0.0
    assert float(delta[0].norm()) <= 1.0 + 1e-6
    assert float(delta[2].norm()) <= 1.0 + 1e-6


def test_prediction_is_registered_before_any_behavioural_result() -> None:
    _, report = core.partition_rows(
        row_ids=[10, 11],
        retain_observed=[True, True],
        efficacy=torch.tensor([0.001, 0.001]),
        potential=torch.tensor([0.4, 0.9]),
        frequency=torch.tensor([1.0, 1.0]),
        direct_live_rows={7: [10, 11]},
        efficacy_min=0.05,
        potential_min=0.1,
        frequency_max=5000,
    )
    prediction = core.diagnostic_prediction(report)
    assert prediction["predicted_locality_risk"] == "high"
    assert prediction["liveness_forced_records"] == 1
    assert "falsifiable_by" in prediction
