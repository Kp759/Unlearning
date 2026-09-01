from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_folded_sensitivity_rewiring_v2_1_core as core  # noqa: E402
import run_mcf_folded_sensitivity_rewiring_v2_1 as runner  # noqa: E402


class Tokenizer:
    def __init__(self) -> None:
        self.mapping = {"alpha": 10, "beta": 11, "shared": 12, "other": 13}

    def __call__(self, text: str, **_kwargs):
        return {
            "input_ids": [
                self.mapping[token]
                for token in str(text).lower().replace(".", "").split()
                if token in self.mapping
            ]
        }


def _cells(
    hidden: torch.Tensor, rows: list[int], signs: list[float]
) -> core.FoldedCells:
    return core.FoldedCells(
        hidden=hidden,
        row_indices=torch.tensor(rows, dtype=torch.long),
        signs=torch.tensor(signs, dtype=torch.float32),
        case_ids=torch.arange(hidden.shape[0]),
        roles=["target_new" if sign > 0 else "target_true" for sign in signs],
    )


def test_registry_folds_classifier_into_internal_lm_head_rows() -> None:
    registry = json.loads(
        (
            ROOT / "protocols" / "mcf_folded_sensitivity_rewiring_v2_1_registry.json"
        ).read_text()
    )
    architecture = registry["architecture"]
    assert registry["status"] == "training_only_hard_tail_repair_available_not_executed"
    assert registry["terminal_first_implementation_result"]["candidate_saved"] is False
    assert architecture["folded_classifier"] == (
        "selected_lm_head_row_delta_dot_final_hidden_state"
    )
    assert architecture["external_classifier"] is False
    assert architecture["runtime_gate"] is False
    assert architecture["transformer_frozen"] is True
    assert (
        registry["signed_cell_semantics"][
            "cross_role_rows_receive_contextual_signed_cells"
        ]
        is True
    )
    assert registry["embedding_rescue"]["adam_forbidden"] is True


def test_registry_and_cli_are_exactly_locked() -> None:
    registry = json.loads(
        (
            ROOT / "protocols" / "mcf_folded_sensitivity_rewiring_v2_1_registry.json"
        ).read_text()
    )
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
    runner.validate_registry(registry, args)
    assert args.head_refit_every == 100
    assert args.hard_tail_rounds == 4
    assert args.hard_tail_per_round == 64
    assert args.protected_logit_correction_max == 0.02
    assert tuple(runner.CORRECTION_FLOORS) == (4.0, 8.0, 12.0, 16.0, 24.0)


def test_folded_solver_handles_cross_role_row_by_context() -> None:
    cells = _cells(
        torch.tensor(
            [
                [5.0, 1.0, 0.0, 0.0],
                [5.0, -1.0, 0.0, 0.0],
                [2.0, 0.0, 1.0, 0.0],
            ]
        ),
        [0, 0, 1],
        [1.0, -1.0, 1.0],
    )
    protected = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    delta, report = core.solve_folded_rows(
        cells,
        n_rows=2,
        hidden_size=4,
        protected_basis=protected,
        correction_floor=2.0,
        ridge=1e-6,
        row_caps=torch.tensor([10.0, 10.0]),
    )
    signed = core.signed_cell_report(cells, delta, minimum=0.1)
    assert report["cross_role_rows"] == 1
    assert signed["passed"] is True
    assert torch.allclose(delta @ protected.T, torch.zeros(2, 1), atol=1e-5)


def test_folded_solver_exposes_identical_state_label_conflict() -> None:
    hidden = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    cells = _cells(hidden, [0, 0], [1.0, -1.0])
    delta, report = core.solve_folded_rows(
        cells,
        n_rows=1,
        hidden_size=2,
        protected_basis=torch.empty((0, 2)),
        correction_floor=2.0,
        ridge=1e-4,
        row_caps=torch.tensor([10.0]),
    )
    signed = core.signed_cell_report(cells, delta, minimum=0.1)
    assert report["passed"] is False
    assert signed["failures"] > 0


def test_hard_tail_solver_adds_exact_row_specific_protection() -> None:
    cells = _cells(torch.tensor([[1.0, 1.0, 0.0]]), [0], [1.0])
    delta, report = core.solve_hard_tail_folded_rows(
        cells,
        protected_hidden=torch.tensor([[1.0, 0.0, 0.0]]),
        n_rows=1,
        hidden_size=3,
        correction_floor=2.0,
        ridge=1e-6,
        row_caps=torch.tensor([10.0]),
        hard_tail_rounds=2,
        hard_tail_per_round=1,
        protected_correction_max=1e-4,
    )
    assert report["passed"] is True
    assert report["per_row"][0]["protected_basis_rank"] == 1
    assert abs(float(delta[0, 0])) < 1e-5
    assert float(cells.hidden[0] @ delta[0]) >= 1.99


def test_normalized_step_spends_fixed_fraction_of_each_row_cap() -> None:
    gradient = torch.tensor([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]])
    caps = torch.tensor([10.0, 2.0, 7.0])
    step = core.normalized_projected_step(gradient, caps, fraction=0.01)
    assert torch.allclose(step.norm(dim=1), torch.tensor([0.1, 0.02, 0.0]), atol=1e-6)
    assert torch.dot(step[0], gradient[0]) < 0


def test_targeted_prompt_partitions_cover_rows_without_cross_split_reuse() -> None:
    partitions, report = core.targeted_token_prompt_partitions(
        Tokenizer(),
        [
            "alpha shared.",
            "beta shared.",
            "alpha other.",
            "beta other.",
            "alpha beta.",
            "shared other.",
        ],
        [10, 11, 12],
        fit_per_row=1,
        development_per_row=1,
        certification_per_row=0,
    )
    assert report["partitions_pairwise_disjoint"] is True
    assert report["fully_covered_rows"] == 3
    assert not set(partitions["fit"]).intersection(partitions["development"])


def test_balanced_batcher_covers_every_item_before_repeating_epoch() -> None:
    batcher = core.BalancedBatcher(6, 2, 3)
    first_epoch = batcher.next() + batcher.next() + batcher.next()
    assert sorted(first_epoch) == list(range(6))


def test_arm_selection_prefers_smallest_passing_floor_then_norm() -> None:
    reports = [
        {"correction_floor": 2.0, "total_delta_norm": 1.0, "passed": False},
        {"correction_floor": 4.0, "total_delta_norm": 3.0, "passed": True},
        {"correction_floor": 6.0, "total_delta_norm": 2.0, "passed": True},
    ]
    assert core.choose_arm(reports) == 4.0
    assert core.choose_arm(reports[:1]) is None


def test_embedding_rescue_requires_a_protection_valid_head_baseline() -> None:
    def arm(floor: float, protected: bool, failures: int) -> dict:
        return {
            "correction_floor": floor,
            "direct": {"failures": failures},
            "synthetic": {"failures": failures},
            "signed_cells": {"failures": 0},
            "solver": {"protected_correction_failures": 0},
            "protection_development": {"passed": protected},
            "total_delta_norm": floor,
        }

    assert runner.best_training_floor([arm(4.0, False, 1)]) is None
    assert runner.best_training_floor([arm(4.0, True, 3), arm(8.0, True, 1)]) == 8.0
