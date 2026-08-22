import sys
import unittest
from pathlib import Path
from unittest import mock

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import mcf_sure_h_then_residual_lmhead_lora as M  # noqa: E402


class DummyCase:
    def __init__(self, record_position: int):
        self.record_position = int(record_position)


class TestSUREHResidualLMHeadLoRA(unittest.TestCase):
    def test_parser_defaults(self):
        a = M.parse_args([
            "--sure-h-model-path", "h",
            "--training-visible-path", "visible.json",
            "--split-manifest", "manifest.json",
            "--output-dir", "out",
            "--utility-wikipedia-dir", "wiki",
        ])
        self.assertEqual(a.lora_rank, 1)
        self.assertAlmostEqual(a.lora_alpha, 1.0)
        self.assertAlmostEqual(a.lr, 5e-4)
        self.assertAlmostEqual(a.solver_margin, 0.25)
        self.assertAlmostEqual(a.acceptance_margin, 0.05)
        self.assertIn(0.0, a.candidate_lora_scales)
        self.assertIn(1.0, a.candidate_lora_scales)

    def test_zero_initialization_is_exact_noop(self):
        torch.manual_seed(1)
        m = M.SparseLMHeadLoRA(
            row_ids=[2, 7, 9], hidden_size=8, rank=2, alpha=2.0,
            device=torch.device("cpu")
        )
        delta = m.effective_delta()
        self.assertEqual(tuple(delta.shape), (3, 8))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertEqual(m.trainable_parameter_count, 2 * 8 + 3 * 2)

    def test_low_rank_factorization_and_multiplier(self):
        torch.manual_seed(2)
        m = M.SparseLMHeadLoRA(
            row_ids=[1, 4, 8], hidden_size=6, rank=1, alpha=1.0,
            device=torch.device("cpu")
        )
        with torch.no_grad():
            m.lora_B.copy_(torch.tensor([[1.0], [2.0], [3.0]]))
            m.lora_A.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]))
        full = m.effective_delta().clone()
        self.assertEqual(int(torch.linalg.matrix_rank(full)), 1)
        m.multiplier = 0.25
        scaled = m.effective_delta()
        self.assertTrue(torch.allclose(scaled, full * 0.25))

    def test_active_case_indices_are_record_based(self):
        cases = [DummyCase(0), DummyCase(0), DummyCase(1), DummyCase(2), DummyCase(2)]
        margins = torch.tensor([0.4, -0.2, 0.1])
        records, case_ids = M._active_case_indices(cases, margins, solver_margin=0.25)
        self.assertEqual(records, [1, 2])
        self.assertEqual(case_ids, [2, 3, 4])

    def test_selected_rows_are_only_active_target_true_tokens(self):
        cases = [DummyCase(0), DummyCase(0), DummyCase(1), DummyCase(2)]
        tids = torch.tensor([11, 12, 13, 14])
        rows = M.selected_rows_for_active_cases(
            cases, active_case_ids=[1, 3], all_target_ids=tids, special_ids={14}
        )
        self.assertEqual(rows, [12])

    def test_materialization_changes_only_selected_rows(self):
        weight = torch.arange(30, dtype=torch.float32).reshape(5, 6)
        before = weight.clone()
        delta = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ])
        M._materialize_selected_delta(weight, [1, 4], delta)
        self.assertTrue(torch.equal(weight[0], before[0]))
        self.assertTrue(torch.equal(weight[2], before[2]))
        self.assertTrue(torch.equal(weight[3], before[3]))
        self.assertTrue(torch.equal(weight[1], before[1] + delta[0]))
        self.assertTrue(torch.equal(weight[4], before[4] + delta[1]))

    def test_scale_selection_prefers_better_margin_if_fail_count_ties(self):
        module = M.SparseLMHeadLoRA(
            [1], hidden_size=3, rank=1, alpha=1.0, device=torch.device("cpu")
        )
        reports_by_scale = {
            0.0: torch.tensor([-1.0, -0.5, 0.2]),
            0.5: torch.tensor([-0.3, -0.1, 0.2]),
            1.0: torch.tensor([-2.0, 0.1, 0.2]),
        }

        def fake_margins(*_args, **_kwargs):
            return reports_by_scale[float(module.multiplier)]

        with mock.patch.object(M, "_direct_margins", side_effect=fake_margins):
            chosen, reports = M._choose_scale(
                model=object(), tok=object(), instances=[], delta_module=module,
                scales=[0.0, 0.5, 1.0], acceptance_margin=0.05,
                device=torch.device("cpu"), llama_like=True, batch_size=1
            )
        # scale 1 has only one failure; it must win despite a worse minimum margin.
        self.assertAlmostEqual(chosen, 1.0)
        self.assertEqual(len(reports), 3)
        self.assertAlmostEqual(module.multiplier, 1.0)


if __name__ == "__main__":
    unittest.main()
