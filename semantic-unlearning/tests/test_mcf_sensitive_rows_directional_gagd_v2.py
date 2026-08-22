import sys
import unittest
from pathlib import Path

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_sensitive_rows_directional_gagd_v2 as M  # noqa: E402


class TestDirectionalGAGDV2(unittest.TestCase):
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
        self.assertEqual(args.input_row_lr, 5e-5)
        self.assertEqual(args.lm_row_lr, 1e-4)
        self.assertEqual(args.basis_refresh_every, 25)
        self.assertEqual(args.protected_basis_rank, 0)
        self.assertEqual(args.sensitive_exclusive_rank, 0)
        self.assertEqual(args.ga_weight, 2.0)
        self.assertEqual(args.gd_weight, 1.0)

    def test_sparse_input_delta_changes_only_selected_tokens(self):
        torch.manual_seed(0)
        embedding = nn.Embedding(8, 5)
        embedding.weight.requires_grad_(False)
        base = embedding.weight.detach().clone()
        sparse = M.SparseInputRowDelta(embedding, [2, 6])
        with torch.no_grad():
            sparse.delta[0].fill_(1.25)
            sparse.delta[1].fill_(-0.75)
        ids = torch.tensor([[1, 2, 3, 6]])
        out = embedding(ids)
        self.assertTrue(torch.allclose(out[0, 0], base[1]))
        self.assertTrue(torch.allclose(out[0, 2], base[3]))
        self.assertTrue(torch.allclose(out[0, 1], base[2] + 1.25))
        self.assertTrue(torch.allclose(out[0, 3], base[6] - 0.75))
        sparse.remove()

    def test_protected_basis_is_orthonormal(self):
        torch.manual_seed(1)
        rows = torch.randn(12, 20)
        basis, diag = M._orthonormal_basis(rows, rank_cap=7)
        self.assertEqual(basis.shape[0], 7)
        self.assertEqual(diag["kept_rank"], 7)
        eye = basis @ basis.T
        self.assertTrue(
            torch.allclose(eye, torch.eye(7), atol=1e-5, rtol=1e-5)
        )

    def test_sensitive_exclusive_basis_is_orthogonal_to_protected(self):
        torch.manual_seed(2)
        protected_raw = torch.randn(4, 16)
        protected = torch.linalg.qr(protected_raw.T, mode="reduced").Q.T
        forget = torch.randn(6, 16)
        tids = torch.tensor([10, 10, 10, 20, 20, 20])
        bases, receipt = M.build_sensitive_exclusive_bases(
            forget, tids, [10, 20], protected, rank_cap=0
        )
        for token_id in (10, 20):
            basis = bases[token_id]
            if basis.numel():
                overlap = basis @ protected.T
                self.assertTrue(
                    torch.allclose(
                        overlap, torch.zeros_like(overlap), atol=2e-5, rtol=2e-5
                    )
                )
            fraction = receipt[str(token_id)][
                "sensitive_exclusive_energy_fraction"
            ]
            self.assertGreaterEqual(fraction, 0.0)
            self.assertLessEqual(fraction, 1.00001)

    def test_zero_exclusive_rank_is_allowed_when_forget_is_fully_protected(self):
        torch.manual_seed(3)
        protected_raw = torch.randn(3, 10)
        protected = torch.linalg.qr(protected_raw.T, mode="reduced").Q.T
        forget = protected[:2].clone()
        tids = torch.tensor([7, 7])
        bases, receipt = M.build_sensitive_exclusive_bases(
            forget, tids, [7], protected, rank_cap=0
        )
        self.assertEqual(bases[7].shape[0], 0)
        self.assertEqual(receipt["7"]["kept_rank"], 0)
        self.assertLess(
            receipt["7"]["sensitive_exclusive_energy_fraction"], 1e-10
        )
        raw = torch.randn(1, 10)
        projected, _ = M.project_rows_tokenwise_span(raw, [7], bases)
        self.assertTrue(torch.equal(projected, torch.zeros_like(projected)))

    def test_output_geometry_enforces_ga_exclusive_and_gd_protected(self):
        torch.manual_seed(4)
        protected_raw = torch.randn(3, 14)
        protected = torch.linalg.qr(protected_raw.T, mode="reduced").Q.T
        residual_raw = torch.randn(5, 14)
        residual = residual_raw - (residual_raw @ protected.T) @ protected
        exclusive = torch.linalg.qr(residual.T, mode="reduced").Q.T[:2]
        bases = {11: exclusive, 13: exclusive}
        ga = torch.randn(2, 14)
        gd = torch.randn(2, 14)
        M.enforce_output_geometry_(ga, gd, [11, 13], bases, protected)
        self.assertTrue(
            torch.allclose(
                ga @ protected.T,
                torch.zeros((2, protected.shape[0])),
                atol=2e-5,
                rtol=2e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                gd - (gd @ protected.T) @ protected,
                torch.zeros_like(gd),
                atol=2e-5,
                rtol=2e-5,
            )
        )
        diag = M.directional_residual_diagnostics(
            ga, gd, [11, 13], bases, protected
        )
        self.assertLess(diag["max_abs_ga_dot_protected_basis"], 2e-5)
        self.assertLess(
            diag["max_abs_gd_dot_sensitive_exclusive_basis"], 2e-5
        )
        self.assertLess(diag["max_abs_cosine_ga_vs_gd"], 2e-5)


if __name__ == "__main__":
    unittest.main()
