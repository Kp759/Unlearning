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


def registry() -> dict:
    return json.loads(REGISTRY_PATH.read_text())


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
