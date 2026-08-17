from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tofu_gagd_neighborhood_confidence as tofu
import tofu_sure_progressive_active_hidden_repair_v8 as v8


def _cache() -> tofu.TOFUAnswerDeltaCache:
    return tofu.TOFUAnswerDeltaCache(
        base_token_nll=torch.tensor([0.2, 0.3], dtype=torch.float32),
        hidden=torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float32),
        selected_probs=torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32),
        target_selected_columns=torch.tensor([0, 1], dtype=torch.long),
    )


def test_exact_row_leverage_unrestricted_matches_sequence_nll_gradient_norm() -> None:
    # For row 0 the exact mean-sequence-NLL gradient is
    # ((0.8-1)/2)*[1,0] + ((0.3-0)/2)*[0,2] = [-0.1, 0.3].
    value = v8.exact_row_leverage(_cache(), 0, None)
    assert math.isclose(value, math.sqrt(0.10), rel_tol=1e-6, abs_tol=1e-6)


def test_exact_row_leverage_uses_repair_subspace_projection() -> None:
    basis = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    value = v8.exact_row_leverage(_cache(), 0, basis)
    assert math.isclose(value, 0.1, rel_tol=1e-6, abs_tol=1e-6)


def test_residual_positions_applies_hard_nll_threshold() -> None:
    nll = torch.tensor([7.0, 8.2, 9.0], dtype=torch.float32)
    assert v8.residual_positions(nll, target_nll=8.0, tolerance=1e-6) == [0]
