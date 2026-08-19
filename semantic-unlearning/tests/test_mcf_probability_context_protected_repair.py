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


def load_script(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


calib = load_script(
    "mcf_disjoint_context_calibration",
    "build_mcf_disjoint_context_calibration.py",
)
repair = load_script(
    "mcf_probability_context_protected",
    "mcf_forget_only_probability_context_protected_repair.py",
)


class ProbabilityContextProtectedRepairTests(unittest.TestCase):
    def sample_record(self):
        return {
            "case_id": 5,
            "requested_rewrite": {
                "prompt": "{} was born in",
                "subject": "Ada",
                "target_true": {"str": "London"},
                "target_new": {"str": "Paris"},
            },
            "paraphrase_prompts": ["Ada's birthplace is"],
            "neighborhood_prompts": ["Grace was born in"],
            "generation_prompts": ["Tell me about Ada"],
        }

    def test_calibration_hides_official_probes_without_swapping_targets(self):
        out = calib.direct_only(self.sample_record(), 0)
        rr = out["requested_rewrite"]
        self.assertEqual(rr["target_true"]["str"], "London")
        self.assertEqual(rr["target_new"]["str"], "Paris")
        self.assertEqual(out["paraphrase_prompts"], [])
        self.assertEqual(out["neighborhood_prompts"], [])
        self.assertEqual(out["generation_prompts"], [])

    def test_absolute_and_pairwise_constraints_are_distinct(self):
        sensitive = torch.tensor([8.5, 7.0, 9.0])
        reference = torch.tensor([7.0, 5.0, 8.5])
        counts = repair.constraint_counts(
            sensitive,
            reference,
            sensitive_nll_floor=8.0,
            pairwise_margin=1.0,
        )
        self.assertEqual(counts["absolute_failures"], 1)
        self.assertEqual(counts["pairwise_failures"], 1)
        self.assertEqual(counts["combined_failures"], 2)

    def test_all_forget_basis_can_realize_requested_rank_eight(self):
        torch.manual_seed(7)
        forget = torch.randn(20, 16)
        retain = torch.randn(6, 16)
        protected, report = repair.protected_forget_rows(
            forget,
            retain,
            mode="none",
            ridge_lambda=0.1,
            retain_rank_cap=6,
        )
        basis, _ = repair._svd_basis(protected, rank_cap=8)
        self.assertEqual(basis.shape, (8, 16))
        self.assertEqual(report["mode"], "none")

    def test_ridge_protection_returns_normalized_finite_rows(self):
        forget = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
        retain = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        protected, report = repair.ridge_precondition_forget_rows(
            forget,
            retain,
            ridge_lambda=0.1,
            retain_rank_cap=1,
        )
        self.assertEqual(tuple(protected.shape), (2, 3))
        self.assertTrue(torch.isfinite(protected).all())
        self.assertTrue(torch.allclose(protected.norm(dim=1), torch.ones(2), atol=1e-5))
        self.assertEqual(report["retain_rank_used"], 1)


if __name__ == "__main__":
    unittest.main()
