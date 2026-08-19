from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sure_context_projection as context


class RowSpecificContextProjectionTests(unittest.TestCase):
    def test_effective_delta_stays_inside_each_rows_own_basis(self):
        basis0 = torch.tensor([[1.0, 0.0, 0.0]])
        basis1 = torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        module = context.RowSpecificProjectedDelta(
            [11, 22], [basis0, basis1], device=torch.device("cpu")
        )
        with torch.no_grad():
            module.coefficients[0].copy_(torch.tensor([2.0]))
            module.coefficients[1].copy_(torch.tensor([3.0, -4.0]))
        delta = module.effective_delta()
        self.assertTrue(torch.allclose(delta[0], torch.tensor([2.0, 0.0, 0.0])))
        self.assertTrue(torch.allclose(delta[1], torch.tensor([0.0, 3.0, -4.0])))
        self.assertEqual(module.row_ranks, [1, 2])
        self.assertEqual(module.trainable_parameter_count, 3)

    def test_projection_removes_components_orthogonal_to_forget_context(self):
        rows = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        bases = [
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[0.0, 0.0, 1.0]]),
        ]
        projected = context.project_rows_to_bases(rows, bases)
        self.assertTrue(
            torch.allclose(projected[0], torch.tensor([1.0, 0.0, 0.0]))
        )
        self.assertTrue(
            torch.allclose(projected[1], torch.tensor([0.0, 0.0, 6.0]))
        )


if __name__ == "__main__":
    unittest.main()
