import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_sensitive_rows_directional_gagd_v3 as M  # noqa: E402


class DummyCase:
    def __init__(self, record_position: int):
        self.record_position = int(record_position)


class TestDirectionalSUREV3(unittest.TestCase):
    def test_parser_defaults(self):
        args = M.parse_args([
            "--model-path", "base",
            "--training-visible-path", "visible.json",
            "--split-manifest", "manifest.json",
            "--output-dir", "out",
            "--utility-wikipedia-dir", "wiki",
        ])
        self.assertEqual(args.steps, 800)
        self.assertAlmostEqual(args.shared_protected_logit_drift_max, 0.05)
        self.assertAlmostEqual(args.solver_margin, 0.25)
        self.assertAlmostEqual(args.acceptance_margin, 0.05)
        self.assertIn(1.0, args.candidate_shared_scales)

    def test_safe_shared_decomposition_is_orthogonal_and_reconstructs_energy(self):
        # Protected space = first two coordinates. Forget states contain both
        # protected/shared and exclusive coordinates.
        bp = torch.tensor([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        hidden = torch.tensor([
            [2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 4.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 5.0],
        ])
        tids = torch.tensor([10, 10, 20, 20])
        safe, shared, receipt = M.build_safe_shared_bases(
            hidden, tids, [10, 20], bp, safe_rank=0, shared_rank=0
        )
        for tid in (10, 20):
            bs = safe[tid]
            bo = shared[tid]
            self.assertLess(float((bs @ bp.T).abs().max()), 1e-5)
            self.assertLess(float((bs @ bo.T).abs().max()), 1e-5)
            shared_resid = bo - M._basis_projection(bo, bp)
            self.assertLess(float(shared_resid.abs().max()), 1e-5)
            self.assertAlmostEqual(receipt[str(tid)]["energy_fraction_sum"], 1.0, places=5)

    def test_protected_only_projection_removes_shared_component(self):
        bp = torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ])
        shared = {
            7: torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            9: torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        }
        rows = torch.tensor([
            [3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0],
        ])
        out, _ = M.project_rows_protected_only(rows, [7, 9], bp, shared)
        # All projected values must be inside BP.
        self.assertTrue(torch.allclose(out[:, 3], torch.zeros(2), atol=1e-6))
        # Token-specific shared coordinate removed.
        self.assertAlmostEqual(float(out[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[1, 1]), 0.0, places=6)
        self.assertAlmostEqual(float(out[0, 1]), 4.0, places=6)
        self.assertAlmostEqual(float(out[1, 2]), 9.0, places=6)

    def test_shared_drift_budget_scales_rows_independently(self):
        protected_hidden = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
        ])
        delta = torch.tensor([
            [2.0, 0.0, 0.0],
            [0.0, 0.01, 0.0],
        ])
        report = M.enforce_shared_drift_budget_(delta, protected_hidden, 0.05)
        drift = (protected_hidden @ delta.T).abs().max().item()
        self.assertLessEqual(drift, 0.0500001)
        self.assertEqual(report["scaled_row_count"], 1)
        self.assertLess(report["minimum_row_scale"], 1.0)

    def test_enforce_output_geometry_keeps_three_roles_separate(self):
        torch.manual_seed(4)
        bp_raw = torch.randn(3, 8)
        bp = torch.linalg.qr(bp_raw.T, mode="reduced").Q.T
        shared_basis = {11: bp[:1], 13: bp[1:2]}

        # Safe basis comes from BP-perpendicular vectors.
        raw_safe = torch.randn(2, 8)
        raw_safe = raw_safe - (raw_safe @ bp.T) @ bp
        safe_basis = {}
        for tid, row in zip((11, 13), raw_safe):
            b, _ = M._stable_basis(row[None, :], 0)
            safe_basis[tid] = b

        safe = torch.randn(2, 8)
        shared = torch.randn(2, 8)
        preserve = torch.randn(2, 8)
        protected_hidden = bp.clone()
        M.enforce_output_geometry_(
            safe, shared, preserve, [11, 13], safe_basis, shared_basis,
            bp, protected_hidden, 0.05
        )
        for i, tid in enumerate((11, 13)):
            self.assertLess(float((safe[i] @ bp.T).abs().max()), 1e-5)
            sb = shared_basis[tid]
            shared_resid = shared[i] - (shared[i] @ sb.T) @ sb
            self.assertLess(float(shared_resid.abs().max()), 1e-5)
            self.assertLess(float((preserve[i] @ sb.T).abs().max()), 1e-5)
        self.assertLessEqual(
            float((protected_hidden @ shared.T).abs().max()), 0.050001
        )

    def test_active_case_indices_excludes_passing_records(self):
        cases = [DummyCase(0), DummyCase(0), DummyCase(1), DummyCase(2), DummyCase(2)]
        margins = torch.tensor([-0.1, 0.5, 0.1])
        active_records, active_cases = M._active_case_indices(cases, margins, 0.25)
        self.assertEqual(active_records, [0, 2])
        self.assertEqual(active_cases, [0, 1, 3, 4])

    def test_margin_report(self):
        margins = torch.tensor([-1.0, 0.05, 0.3])
        report = M._margin_report(margins, 0.05)
        self.assertEqual(report["failures"], 1)
        self.assertAlmostEqual(report["minimum_margin"], -1.0)
        self.assertAlmostEqual(report["maximum_margin"], 0.3, places=5)


if __name__ == "__main__":
    unittest.main()
