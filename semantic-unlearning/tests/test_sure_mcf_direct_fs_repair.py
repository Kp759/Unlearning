from __future__ import annotations

import importlib.util
import argparse
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


repair = load_module("sure_mcf_direct_fs_repair", "sure_mcf_direct_fs_repair.py")


class ExactDirectFsRepairTests(unittest.TestCase):
    def explicit_record_nll(
        self,
        logits: torch.Tensor,
        hidden: torch.Tensor,
        target_ids: torch.Tensor,
        positions: torch.Tensor,
        residual: torch.Tensor,
        selected_ids,
        record_count: int,
    ) -> torch.Tensor:
        edited = logits.float().clone()
        shifts = hidden.float() @ residual.float().transpose(0, 1)
        for column, token_id in enumerate(selected_ids):
            edited[:, int(token_id)] += shifts[:, column]
        rows = torch.arange(edited.shape[0])
        token_nll = torch.logsumexp(edited, dim=1) - edited[rows, target_ids]
        return repair.mean_by_record(token_nll, positions, record_count)

    def test_exact_pairwise_cache_matches_full_vocab_sequence_nll(self):
        selected_ids = [0, 3]
        sensitive_logits = torch.tensor(
            [
                [2.0, 0.0, -1.0, 0.5],
                [0.5, 1.0, -0.5, 1.5],
                [1.0, -1.0, 0.5, 0.0],
            ]
        )
        sensitive_hidden = torch.tensor([[1.0, 0.0], [0.5, 1.0], [-1.0, 0.5]])
        sensitive_ids = torch.tensor([0, 3, 0])
        sensitive_positions = torch.tensor([0, 0, 1])
        reference_logits = torch.tensor([[0.0, 2.0, -1.0, 0.5], [0.5, 0.0, 1.5, -0.5]])
        reference_hidden = torch.tensor([[0.25, 1.0], [1.0, -0.5]])
        reference_ids = torch.tensor([1, 2])
        reference_positions = torch.tensor([0, 1])
        residual = torch.tensor([[-0.4, 0.2], [0.1, -0.3]])

        sensitive_cache = repair.build_sequence_cache(
            sensitive_logits,
            sensitive_hidden,
            sensitive_ids,
            sensitive_positions,
            selected_ids,
            record_count=2,
            device=torch.device("cpu"),
        )
        reference_cache = repair.build_sequence_cache(
            reference_logits,
            reference_hidden,
            reference_ids,
            reference_positions,
            selected_ids,
            record_count=2,
            device=torch.device("cpu"),
        )
        actual = repair.exact_pairwise_separation(
            sensitive_cache, reference_cache, residual
        )
        expected_sensitive = self.explicit_record_nll(
            sensitive_logits,
            sensitive_hidden,
            sensitive_ids,
            sensitive_positions,
            residual,
            selected_ids,
            2,
        )
        expected_reference = self.explicit_record_nll(
            reference_logits,
            reference_hidden,
            reference_ids,
            reference_positions,
            residual,
            selected_ids,
            2,
        )
        self.assertTrue(
            torch.allclose(actual, expected_sensitive - expected_reference, atol=1e-6)
        )

    def test_pairwise_report_counts_ties_as_official_failures(self):
        report = repair.pairwise_report(
            torch.tensor([0.2, 0.0, -0.1, 0.005]),
            required_margin=0.01,
        )
        self.assertEqual(report["direct_fs"], 50.0)
        self.assertEqual(report["direct_fs_failures"], 2)
        self.assertEqual(report["direct_fs_margin_failures"], 3)

    def test_exact_solver_repairs_sequence_pairwise_constraint(self):
        current_sensitive = torch.tensor([[2.0, 0.0, 0.0]])
        base_sensitive = current_sensitive.clone()
        sensitive_hidden = torch.tensor([[1.0]])
        sensitive_ids = torch.tensor([0])
        positions = torch.tensor([0])
        current_reference = torch.tensor([[0.0, 2.0, 0.0]])
        reference_hidden = torch.tensor([[0.0]])
        reference_ids = torch.tensor([1])
        direct_cache = repair.learner.build_exact_stage2_direct_cache(
            current_sensitive,
            base_sensitive,
            sensitive_hidden,
            sensitive_ids,
            [0],
            device=torch.device("cpu"),
        )
        sensitive_cache = repair.build_sequence_cache(
            current_sensitive,
            sensitive_hidden,
            sensitive_ids,
            positions,
            [0],
            record_count=1,
            device=torch.device("cpu"),
        )
        reference_cache = repair.build_sequence_cache(
            current_reference,
            reference_hidden,
            reference_ids,
            positions,
            [0],
            record_count=1,
            device=torch.device("cpu"),
        )
        args = argparse.Namespace(
            residual_l2_weight=1e-4,
            constraint_tolerance=1e-5,
            maxiter=100,
            ftol=1e-9,
            direct_fs_margin=0.01,
        )
        residual, _, report = repair.solve_rank(
            args=args,
            rank=1,
            row_bases=[torch.ones(1, 1)],
            active_ids=[0],
            selected_ids=[0],
            current_total_delta=torch.zeros(1, 1),
            direct_cache=direct_cache,
            internal_nll_solver_targets=torch.tensor([-10.0]),
            internal_margin_solver_target=-10.0,
            sensitive_cache=sensitive_cache,
            reference_cache=reference_cache,
            fs_solver_target=0.1,
            utility_hidden=torch.zeros(2, 1),
            utility_probabilities=torch.full((2, 1), 0.1),
            utility_budgets={
                "mean": 10.0,
                "p95": 10.0,
                "max": 10.0,
                "total_delta_norm": 10.0,
            },
        )
        separation = repair.exact_pairwise_separation(
            sensitive_cache, reference_cache, residual
        )
        self.assertTrue(report["continuous_feasible"])
        self.assertGreaterEqual(float(separation[0]), 0.1 - 1e-5)

    def test_mean_by_record_rejects_missing_records(self):
        with self.assertRaisesRegex(ValueError, "every direct record"):
            repair.mean_by_record(
                torch.tensor([1.0, 2.0]),
                torch.tensor([0, 0]),
                2,
            )

    def test_fs100_wrapper_enables_explicit_target_aware_mode(self):
        wrapper = (SCRIPTS / "run_mcf_sure_fs100.sh").read_text(encoding="utf-8")
        runner = (SCRIPTS / "run_mcf_sure_minimal.sh").read_text(encoding="utf-8")
        self.assertIn("SURE_MCF_TARGET_AWARE_FS=1", wrapper)
        self.assertIn("sure_mcf_direct_fs_repair.py", runner)
        self.assertIn("PAPER_ASSERT_ARGS=(--require-min-fs 100)", runner)


if __name__ == "__main__":
    unittest.main()
