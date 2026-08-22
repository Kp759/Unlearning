import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_sensitive_rows_directional_gagd_v3_balanced as M  # noqa: E402


class TestDirectionalSUREV3Balanced(unittest.TestCase):
    def test_reuses_v3_parser_and_defaults(self):
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

    def test_active_sampler_none_when_all_cases_pass(self):
        args = M.parse_args([
            "--model-path", "base",
            "--training-visible-path", "visible.json",
            "--split-manifest", "manifest.json",
            "--output-dir", "out",
            "--utility-wikipedia-dir", "wiki",
        ])
        self.assertIsNone(M._new_active_sampler([], args, 0))

    def test_active_sampler_created_for_tail(self):
        args = M.parse_args([
            "--model-path", "base",
            "--training-visible-path", "visible.json",
            "--split-manifest", "manifest.json",
            "--output-dir", "out",
            "--utility-wikipedia-dir", "wiki",
        ])
        sampler = M._new_active_sampler([2, 5, 9], args, 25)
        self.assertIsNotNone(sampler)
        batch = sampler.next()
        self.assertEqual(len(batch), 1)
        self.assertTrue(0 <= batch[0] < 3)


if __name__ == "__main__":
    unittest.main()
