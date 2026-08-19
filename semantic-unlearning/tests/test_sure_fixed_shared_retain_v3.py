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


retain = load_module("sure_retain_kl", "sure_retain_kl.py")


class RetainProtectedSharedV3Tests(unittest.TestCase):
    def test_full_distribution_kl_is_zero_for_identical_logits(self):
        logits = torch.tensor([[1.0, 2.0, -1.0], [0.5, -0.5, 3.0]])
        value = retain.full_distribution_kl(logits, logits.clone())
        self.assertAlmostEqual(float(value), 0.0, places=6)

    def test_retain_prompt_cases_need_no_answer_label(self):
        records = [
            {
                "case_id": 7,
                "requested_rewrite": {
                    "prompt": "{} was founded in",
                    "subject": "Acme",
                },
            }
        ]
        cases = retain.retain_prompt_cases(records)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].prompt, "Acme was founded in")
        self.assertEqual(cases[0].target_text, "")

    def test_mcf_and_zsre_v3_runners_share_defaults(self):
        mcf = (SCRIPTS / "run_mcf_sure_fixed_shared_retain_v3.sh").read_text(encoding="utf-8")
        zsre = (SCRIPTS / "run_zsre_sure_fixed_shared_retain_v3.sh").read_text(encoding="utf-8")
        shared_defaults = [
            'RETAIN_TRAIN_NUM="${SURE_RETAIN_TRAIN_NUM:-1000}"',
            'STEPS="${SURE_STAGE1_STEPS:-600}"',
            'FORGET_BATCH_SIZE="${SURE_STAGE1_BATCH_SIZE:-1}"',
            'RETAIN_BATCH_SIZE="${SURE_RETAIN_BATCH_SIZE:-4}"',
            'LR="${SURE_STAGE1_LR:-0.0001}"',
            'GA_WEIGHT="${SURE_GA_WEIGHT:-4.0}"',
            'GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"',
            'RETAIN_KL_WEIGHT="${SURE_RETAIN_KL_WEIGHT:-1.0}"',
            'STAGE1_CONTEXT_RANK="${SURE_STAGE1_CONTEXT_RANK:-2}"',
            'SHARED_MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.25}"',
            'REQUIRED_NLL_INCREASE="${SURE_REQUIRED_NLL_INCREASE:-4.0}"',
            'CANDIDATE_RANKS="${SURE_REPAIR_RANKS:-2,8,0}"',
            'REPAIR_STEPS="${SURE_REPAIR_STEPS:-800}"',
            'REPAIR_LR="${SURE_REPAIR_LR:-0.005}"',
        ]
        for line in shared_defaults:
            self.assertIn(line, mcf)
            self.assertIn(line, zsre)

    def test_shared_stages_contain_retain_kl_and_no_reference_ce(self):
        for name in (
            "sure_stage1_context_retain_shared.py",
            "sure_stage2_context_retain_shared.py",
        ):
            text = (SCRIPTS / name).read_text(encoding="utf-8")
            self.assertIn("retain.full_distribution_kl", text)
            self.assertIn("gd_non_sensitive_kl", text)
            self.assertIn("ga_sensitive_logprob", text)
            self.assertNotIn("cross_entropy", text)
            self.assertNotIn("F.relu", text)


if __name__ == "__main__":
    unittest.main()
