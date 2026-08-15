import importlib.util
import sys
import unittest
from pathlib import Path

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tofu_sure_nullspace_restore.py"
spec = importlib.util.spec_from_file_location("tofu_sure_nullspace_restore", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.path.insert(0, str(SCRIPT.parent))
spec.loader.exec_module(module)


class TofuSureNullspaceRestoreTest(unittest.TestCase):
    def test_restore_delta_has_negligible_forget_action(self):
        torch.manual_seed(7)
        hidden = torch.randn(6, 12, dtype=torch.float32)
        forget_basis, report = module.numerical_row_basis(hidden)
        self.assertEqual(report["numerical_rank"], 6)

        desired = torch.randn(9, 12, dtype=torch.float32)
        restore_basis, delta, restore_report = module.build_restore_basis(
            desired, forget_basis, requested_rank=4
        )
        self.assertLessEqual(restore_report["actual_rank"], 4)
        self.assertEqual(delta.shape, desired.shape)
        self.assertEqual(restore_basis.shape[1], desired.shape[1])

        action = hidden @ delta.T
        self.assertLess(float(action.abs().max()), 2e-4)
        overlap = restore_basis @ forget_basis.T
        self.assertLess(float(overlap.abs().max()), 2e-5)

    def test_scale_parser_is_descending_and_includes_zero(self):
        scales = module.parse_scales(".5,1,.25,.5")
        self.assertEqual(scales, [1.0, 0.5, 0.25, 0.0])

    def test_scale_parser_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            module.parse_scales("1,1.2,0")


if __name__ == "__main__":
    unittest.main()
