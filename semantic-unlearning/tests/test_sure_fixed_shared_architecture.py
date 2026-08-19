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

    def test_sensitive_nll_increase_is_current_minus_base(self):
        base = torch.tensor([[0.0, 3.0, 1.0]])
        current = torch.tensor([[0.0, -1.0, 1.0]])
        tids = torch.tensor([1])
        delta = shared.sensitive_nll_increase_from_logits(current, base, tids)
        self.assertGreater(float(delta.item()), 0.0)

    def test_shared_failure_requires_both_constraints(self):
        logit = torch.tensor([0.06, 0.04, 0.06])
        nll_delta = torch.tensor([4.1, 4.1, 3.9])
        mask = shared.failure_mask(
            logit,
            nll_delta,
            required_logit_margin=0.05,
            required_nll_increase=4.0,
        )
        self.assertEqual(mask.tolist(), [False, True, True])
        self.assertEqual(
            shared.count_failures(
                logit,
                nll_delta,
                required_logit_margin=0.05,
                required_nll_increase=4.0,
            ),
            2,
        )

    def test_stage2_has_no_relu_or_reference_answer_ce(self):
        text = (SCRIPTS / "sure_stage2_context_shared.py").read_text(encoding="utf-8")
        self.assertNotIn("F.relu", text)
        self.assertNotIn("cross_entropy", text)
        self.assertIn("gd_non_sensitive_kl", text)
        self.assertIn("ga_sensitive_logprob", text)
        self.assertIn("min-sensitive-nll-increase", text)

    def test_stage1_has_same_ga_gd_components(self):
        text = (SCRIPTS / "sure_stage1_context_shared.py").read_text(encoding="utf-8")
        self.assertNotIn("cross_entropy", text)
        self.assertIn("gd_non_sensitive_kl", text)
        self.assertIn("ga_sensitive_logprob", text)
        self.assertIn("min-sensitive-nll-increase", text)

    def test_mcf_and_zsre_runners_call_same_shared_stage_files(self):
        mcf = (SCRIPTS / "run_mcf_sure_fixed_shared.sh").read_text(encoding="utf-8")
        zsre = (SCRIPTS / "run_zsre_sure_fixed_shared.sh").read_text(encoding="utf-8")
        for stage_file in (
            "scripts/sure_stage1_context_shared.py",
            "scripts/sure_stage2_context_shared.py",
        ):
            self.assertIn(stage_file, mcf)
            self.assertIn(stage_file, zsre)

        # Paper-path runners must not silently switch to benchmark-specific
        # residual/hinge repair implementations.
        forbidden = (
            "sure_stage2_sparse_repair.py",
            "sure_stage2_sparse_repair_residual.py",
            "sure_stage2_sparse_repair_guarded.py",
        )
        for name in forbidden:
            self.assertNotIn(name, mcf)
            self.assertNotIn(name, zsre)

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
            'MIN_SENSITIVE_NLL_INCREASE="${SURE_MIN_SENSITIVE_NLL_INCREASE:-4.0}"',
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
