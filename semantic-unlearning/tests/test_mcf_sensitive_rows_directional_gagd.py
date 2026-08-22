import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_sensitive_rows_directional_gagd as M  # noqa: E402


class TestDirectionalGAGD(unittest.TestCase):
    def test_parser_defaults(self):
        args = M.parse_args(
            [
                "--model-path", "base",
                "--training-visible-path", "visible.json",
                "--split-manifest", "manifest.json",
                "--output-dir", "out",
                "--utility-wikipedia-dir", "wiki",
            ]
        )
        self.assertEqual(args.forget_basis_rank, 0)
        self.assertEqual(args.lm_row_lr, 1e-4)
        self.assertEqual(args.ga_weight, 2.0)
        self.assertEqual(args.gd_weight, 1.0)

    def test_token_forget_bases_are_token_specific_and_orthonormal(self):
        torch.manual_seed(0)
        hidden = torch.randn(6, 12)
        tids = torch.tensor([10, 10, 10, 20, 20, 20])
        bases, receipt = M.build_token_forget_bases(hidden, tids, [10, 20], rank_cap=2)
        self.assertEqual(set(bases), {10, 20})
        self.assertLessEqual(bases[10].shape[0], 2)
        self.assertLessEqual(bases[20].shape[0], 2)
        for basis in bases.values():
            eye = basis @ basis.T
            self.assertTrue(torch.allclose(eye, torch.eye(basis.shape[0]), atol=1e-5, rtol=1e-5))
        self.assertEqual(receipt["10"]["case_count"], 3)
        self.assertEqual(receipt["20"]["case_count"], 3)

    def test_span_and_null_projection_reconstruct_rows(self):
        torch.manual_seed(1)
        raw = torch.randn(3, 8)
        basis_raw = torch.randn(2, 8)
        basis = torch.linalg.qr(basis_raw.T, mode="reduced").Q.T
        bases = {3: basis, 5: basis, 7: basis}
        span, _ = M.project_rows_tokenwise(raw, [3, 5, 7], bases, mode="span")
        null, _ = M.project_rows_tokenwise(raw, [3, 5, 7], bases, mode="null")
        self.assertTrue(torch.allclose(span + null, raw.float(), atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(null @ basis.T, torch.zeros(3, 2), atol=1e-5, rtol=1e-5))

    def test_hard_parameter_projection_keeps_ga_and_gd_orthogonal(self):
        torch.manual_seed(2)
        basis_raw = torch.randn(3, 10)
        basis = torch.linalg.qr(basis_raw.T, mode="reduced").Q.T
        bases = {11: basis, 13: basis}
        ga = torch.randn(2, 10)
        gd = torch.randn(2, 10)
        M.enforce_directional_parameters_(ga, gd, [11, 13], bases)
        for i in range(2):
            ga_null = ga[i] - (ga[i] @ basis.T) @ basis
            gd_span = (gd[i] @ basis.T) @ basis
            self.assertTrue(torch.allclose(ga_null, torch.zeros_like(ga_null), atol=1e-5, rtol=1e-5))
            self.assertTrue(torch.allclose(gd_span, torch.zeros_like(gd_span), atol=1e-5, rtol=1e-5))
            self.assertAlmostEqual(float(ga[i] @ gd[i]), 0.0, places=4)

    def test_directional_residual_diagnostics_are_small_after_projection(self):
        torch.manual_seed(3)
        basis_raw = torch.randn(2, 9)
        basis = torch.linalg.qr(basis_raw.T, mode="reduced").Q.T
        bases = {2: basis}
        ga = torch.randn(1, 9)
        gd = torch.randn(1, 9)
        M.enforce_directional_parameters_(ga, gd, [2], bases)
        d = M.directional_residual_diagnostics(ga, gd, [2], bases)
        self.assertLess(d["max_abs_ga_null_residual"], 1e-5)
        self.assertLess(d["max_abs_gd_forget_span_residual"], 1e-5)
        self.assertLess(d["max_abs_cosine_ga_vs_gd"], 1e-5)


if __name__ == "__main__":
    unittest.main()
