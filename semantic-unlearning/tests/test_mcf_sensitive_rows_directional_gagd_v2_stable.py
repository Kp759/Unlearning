import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_sensitive_rows_directional_gagd_v2_stable as M  # noqa: E402


class TestDirectionalSUREV2StableBasis(unittest.TestCase):
    def test_qr_reorthogonalization_is_strictly_orthonormal(self):
        torch.manual_seed(0)
        raw = torch.randn(32, 96)
        basis, _method, error = M._qr_reorthogonalize(raw)
        self.assertEqual(basis.shape, raw.shape)
        gram = basis @ basis.T
        self.assertTrue(
            torch.allclose(
                gram,
                torch.eye(raw.shape[0]),
                atol=M.FALLBACK_ATOL,
                rtol=M.FALLBACK_ATOL,
            )
        )
        self.assertLessEqual(error, M.FALLBACK_ATOL)

    def test_nearly_collinear_rows_remain_stable(self):
        torch.manual_seed(1)
        base = torch.randn(1, 128)
        rows = base.repeat(24, 1) + 1e-4 * torch.randn(24, 128)
        basis, diag = M.stable_orthonormal_basis(rows, rank_cap=0)
        self.assertGreater(diag["kept_rank"], 0)
        self.assertEqual(basis.shape[0], diag["kept_rank"])
        self.assertLessEqual(diag["max_abs_BBt_minus_I"], M.FALLBACK_ATOL)
        self.assertIn(
            diag["reorthogonalization_method"],
            {
                "cpu_fp32_qr",
                "cpu_fp32_qr_twice",
                "gpu_fp32_qr",
                "gpu_fp32_qr_twice",
                "cpu_float64_qr_fallback",
            },
        )

    def test_rank_cap_is_preserved(self):
        torch.manual_seed(2)
        rows = torch.randn(40, 80)
        basis, diag = M.stable_orthonormal_basis(rows, rank_cap=7)
        self.assertEqual(diag["kept_rank"], 7)
        self.assertEqual(tuple(basis.shape), (7, 80))
        self.assertLessEqual(diag["max_abs_BBt_minus_I"], M.FALLBACK_ATOL)


if __name__ == "__main__":
    unittest.main()
