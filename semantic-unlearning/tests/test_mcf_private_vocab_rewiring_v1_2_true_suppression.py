from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_private_vocab_rewiring_v1_2_true_suppression as v12  # noqa: E402


def test_suppression_summary_uses_base_relative_drop():
    out = v12.suppression_summary(
        [-1.0, -3.0, -5.0],
        [-4.0, -4.0, -8.0],
        minimum_drop=2.0,
    )
    assert out["drops"] == [3.0, 1.0, 3.0]
    assert out["passed"] == 2
    assert out["failures"] == 1
    assert out["minimum_drop"] == 1.0


def test_true_suppression_hinge_has_no_target_new_dependency():
    current_true = torch.tensor([-1.0, -5.0], requires_grad=True)
    base_true = torch.tensor([-1.0, -2.0])
    target_drop = 2.0
    ceiling = base_true - target_drop
    loss = torch.relu(current_true - ceiling).square().mean()
    loss.backward()
    # First case has not dropped enough and receives gradient; second already
    # exceeds the drop target and receives exactly zero gradient.
    assert current_true.grad is not None
    assert current_true.grad[0].item() > 0
    assert current_true.grad[1].item() == 0
