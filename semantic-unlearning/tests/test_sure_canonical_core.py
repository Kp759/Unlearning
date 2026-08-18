import sys
import unittest
from pathlib import Path

import torch
from torch import nn


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import sure_canonical_core as core  # noqa: E402


class CanonicalCoreTests(unittest.TestCase):
    def test_sensitive_field(self):
        self.assertEqual(core.sensitive_answer_field("mcf"), "target_new")
        self.assertEqual(core.sensitive_answer_field("zsre"), "target_true")

    def test_choose_smallest_zero_failure_scale(self):
        reports = [
            {"scale": 1.0, "direct_failures": 0},
            {"scale": 0.5, "direct_failures": 0},
            {"scale": 0.25, "direct_failures": 1},
        ]
        self.assertEqual(core.choose_scale(reports), 0.5)

    def test_rank_parameterization(self):
        basis = torch.eye(4)[:2]
        delta = core.SelectedRowDelta(
            3,
            4,
            direction_basis=basis,
            device=torch.device("cpu"),
        )
        self.assertEqual(delta.trainable_parameter_count, 6)
        self.assertEqual(tuple(delta.effective_delta().shape), (3, 4))

    def test_output_hook_backpropagates_only_through_delta(self):
        layer = nn.Linear(4, 6, bias=False)
        for parameter in layer.parameters():
            parameter.requires_grad_(False)
        delta = core.SelectedRowDelta(
            2,
            4,
            direction_basis=None,
            device=torch.device("cpu"),
        )
        handle = core.register_output_delta_hook(
            layer,
            [1, 3],
            delta.effective_delta,
        )
        try:
            x = torch.randn(5, 4)
            y = layer(x)
            (y[:, 1].sum() + y[:, 3].sum()).backward()
        finally:
            handle.remove()
        self.assertIsNotNone(delta.raw_delta.grad)
        self.assertIsNone(layer.weight.grad)

    def test_restore_keeps_only_sensitive_rows(self):
        base = torch.arange(20, dtype=torch.float32).view(5, 4)
        trained = base + 10
        input_weight = nn.Parameter(trained.clone())
        output_weight = nn.Parameter(trained.clone())
        tied_info = {
            "input_weight": input_weight,
            "output_weight": output_weight,
            "tied": False,
        }
        report = core.restore_sensitive_rows_only(
            tied_info,
            {"input": base.clone(), "output": base.clone()},
            [2],
        )
        self.assertTrue(torch.equal(input_weight[0], base[0]))
        self.assertTrue(torch.equal(input_weight[2], trained[2]))
        self.assertTrue(torch.equal(output_weight[0], base[0]))
        self.assertTrue(torch.equal(output_weight[2], trained[2]))
        self.assertFalse(report["factual_boost_applied"])


if __name__ == "__main__":
    unittest.main()
