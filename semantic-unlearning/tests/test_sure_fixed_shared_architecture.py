from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = load_module("sure_shared_suppression", "sure_shared_suppression.py")


class FixedSharedArchitectureTests(unittest.TestCase):
    def test_suppression_margin_is_best_other_minus_sensitive(self):
        logits = torch.tensor([
            [1.0, 4.0, 2.0],
            [5.0, 3.0, 1.0],
        ])
        tids = torch.tensor([1, 2])
        margins = shared.suppression_margins_from_logits(logits, tids)
        self.assertTrue(torch.allclose(margins, torch.tensor([-2.0, 4.0])))

    def test_strict_margin_failure_count(self):
        margins = torch.tensor([0.04, 0.05, 0.06])
        self.assertEqual(shared.count_failures(margins, 0.05), 1)

    def test_stage2_has_no_relu_or_reference_answer_ce(self):
        text = (SCRIPTS / "sure_stage2_context_shared.py").read_text(encoding="utf-8")
        self.assertNotIn("F.relu", text)
        self.assertNotIn("cross_entropy", text)
        self.assertIn("gd_non_sensitive_kl", text)
        self.assertIn("ga_sensitive_logprob", text)

    def test_stage1_has_same_ga_gd_components(self):
        text = (SCRIPTS / "sure_stage1_context_shared.py").read_text(encoding="utf-8")
        self.assertNotIn("cross_entropy", text)
        self.assertIn("gd_non_sensitive_kl", text)
        self.assertIn("ga_sensitive_logprob", text)

    def test_mcf_and_zsre_runners_share_defaults(self):
        mcf = (SCRIPTS / "run_mcf_sure_fixed_shared.sh").read_text(encoding="utf-8")
        zsre = (SCRIPTS / "run_zsre_sure_fixed_shared.sh").read_text(encoding="utf-8")
        shared_defaults = [
            'STEPS="${SURE_STAGE1_STEPS:-600}"',
            'LR="${SURE_STAGE1_LR:-0.0001}"',
            'GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"',
            'GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"',
            'STAGE1_CONTEXT_RANK="${SURE_STAGE1_CONTEXT_RANK:-2}"',
            'SHARED_MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.05}"',
            'CANDIDATE_RANKS="${SURE_REPAIR_RANKS:-2,8,0}"',
            'REPAIR_STEPS="${SURE_REPAIR_STEPS:-800}"',
            'REPAIR_LR="${SURE_REPAIR_LR:-0.005}"',
            'REPAIR_L2="${SURE_REPAIR_L2:-0.000001}"',
        ]
        for line in shared_defaults:
            self.assertIn(line, mcf)
            self.assertIn(line, zsre)


if __name__ == "__main__":
    unittest.main()
