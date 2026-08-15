import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCRIPT = SCRIPTS / "tofu_sure_rank0_forget_restored.py"
SPEC = importlib.util.spec_from_file_location("tofu_sure_rank0_restored_v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
# Python 3.10 dataclasses resolves forward/type metadata through
# sys.modules[cls.__module__] while the class decorator runs.  Register the
# dynamically loaded module before exec_module so @dataclass can resolve it.
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class TofuSureRank0RestoredV2Tests(unittest.TestCase):
    def test_feasible_nll_requires_every_sequence(self):
        required = torch.tensor([1.0, 2.0])
        self.assertTrue(MODULE.feasible_nll(torch.tensor([1.0, 2.1]), required, 0.0))
        self.assertFalse(MODULE.feasible_nll(torch.tensor([1.0, 1.9]), required, 0.0))
        self.assertTrue(MODULE.feasible_nll(torch.tensor([1.0, 1.9]), required, 0.11))

    def test_boundary_bisection_returns_near_first_feasible_point(self):
        # Synthetic scorer: sequence NLL equals the single delta coordinate.
        # Requirement=1 means delta=0 is infeasible and delta=2 is feasible;
        # the exact first feasible interpolation is alpha=0.5.
        def fake_score(_packed, delta):
            return delta.reshape(-1)[:1]

        low = torch.tensor([[0.0]], dtype=torch.float32)
        high = torch.tensor([[2.0]], dtype=torch.float32)
        required = torch.tensor([1.0], dtype=torch.float32)

        with mock.patch.object(
            MODULE.tofu,
            "answer_nlls_from_packed_delta_cache",
            side_effect=fake_score,
        ):
            delta, nll, alpha = MODULE.boundary_bisect(
                object(),
                low,
                high,
                required,
                tolerance=0.0,
                iterations=30,
                safety_fraction=0.0,
            )

        self.assertAlmostEqual(alpha, 0.5, places=6)
        self.assertAlmostEqual(float(delta.item()), 1.0, places=6)
        self.assertGreaterEqual(float(nll.item()), 1.0)
        self.assertLess(float(delta.norm().item()), float(high.norm().item()))

    def test_boundary_safety_fraction_moves_slightly_inside_feasible_region(self):
        def fake_score(_packed, delta):
            return delta.reshape(-1)[:1]

        with mock.patch.object(
            MODULE.tofu,
            "answer_nlls_from_packed_delta_cache",
            side_effect=fake_score,
        ):
            delta, _, alpha = MODULE.boundary_bisect(
                object(),
                torch.tensor([[0.0]]),
                torch.tensor([[2.0]]),
                torch.tensor([1.0]),
                tolerance=0.0,
                iterations=30,
                safety_fraction=0.002,
            )

        self.assertGreater(alpha, 0.5)
        self.assertLess(alpha, 0.51)
        self.assertGreater(float(delta.item()), 1.0)
        self.assertLess(float(delta.item()), 1.02)


if __name__ == "__main__":
    unittest.main()
