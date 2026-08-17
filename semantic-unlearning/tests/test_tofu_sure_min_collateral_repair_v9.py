from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_min_collateral_repair_v9 as v9


def _cache() -> tofu.TOFUAnswerDeltaCache:
    return tofu.TOFUAnswerDeltaCache(
        base_token_nll=torch.tensor([0.25, 0.50, 1.00], dtype=torch.float32),
        hidden=torch.tensor(
            [[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]], dtype=torch.float32
        ),
        selected_probs=torch.tensor(
            [
                [0.50, 0.20, 0.10],
                [0.10, 0.30, 0.20],
                [0.20, 0.10, 0.30],
            ],
            dtype=torch.float32,
        ),
        target_selected_columns=torch.tensor([0, 2, -1], dtype=torch.long),
    )


def test_subset_answer_caches_reorders_probs_and_remaps_targets() -> None:
    cache = _cache()
    subset = v9.subset_answer_caches([cache], [10, 20, 30], [30, 10])[0]
    expected_probs = cache.selected_probs[:, torch.tensor([2, 0])]
    assert torch.equal(subset.selected_probs, expected_probs)
    # old target col 0 (token 10) -> new col 1; old col 2 (token 30) -> new col 0.
    assert subset.target_selected_columns.tolist() == [1, 0, -1]


def test_base_non_target_kl_is_zero_for_zero_total_delta() -> None:
    cache = _cache()
    zero = torch.zeros((3, 2), dtype=torch.float32)
    value = v9.equal_example_base_non_target_kl([cache], zero)
    assert math.isclose(float(value), 0.0, abs_tol=1e-6)


def test_diagonal_non_target_fisher_is_nonnegative_in_requested_space() -> None:
    cache = _cache()
    unrestricted = v9.diagonal_non_target_fisher([cache], None)
    assert unrestricted.shape == (3, 2)
    assert torch.all(unrestricted >= 0)

    basis = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    restricted = v9.diagonal_non_target_fisher([cache], basis)
    assert restricted.shape == (3, 1)
    assert torch.all(restricted >= 0)


def test_fisher_adjusted_score_prefers_same_gradient_in_safer_geometry() -> None:
    gradient = torch.tensor([1.0, 0.0], dtype=torch.float32)
    low_cost = torch.tensor([0.01, 1.0], dtype=torch.float32)
    high_cost = torch.tensor([1.0, 1.0], dtype=torch.float32)
    safe_score = v9.fisher_adjusted_score(gradient, low_cost, damping=1e-6)
    costly_score = v9.fisher_adjusted_score(gradient, high_cost, damping=1e-6)
    assert safe_score > costly_score
