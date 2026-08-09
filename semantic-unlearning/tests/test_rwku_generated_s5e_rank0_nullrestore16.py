from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import rwku_generated_s5e_rank0_nullrestore16 as method
import rwku_generated_s5e_rank2_active_repair as rank2


class ForgetNullspaceMathTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)

    def test_forget_span_projection_preserves_forget_visible_logits(self) -> None:
        hidden = torch.randn(7, 11)
        basis = method.active.orthonormal_row_basis(hidden)
        delta = torch.randn(5, 11)
        projected = method.project_delta_to_forget_span(delta, basis)
        before = hidden @ delta.T
        after = hidden @ projected.T
        torch.testing.assert_close(after, before, atol=2e-5, rtol=2e-5)

    def test_restore_basis_is_orthogonal_to_forget_basis(self) -> None:
        forget_hidden = torch.randn(4, 12)
        forget_basis = method.active.orthonormal_row_basis(forget_hidden)
        retain_hidden = torch.randn(19, 12)
        null_hidden, restore_basis = method.build_restore_basis(
            retain_hidden, forget_basis, max_rank=6
        )
        self.assertEqual(tuple(null_hidden.shape), (19, 12))
        overlap = restore_basis @ forget_basis.T
        torch.testing.assert_close(overlap, torch.zeros_like(overlap), atol=2e-5, rtol=0)

    def test_restoration_delta_cannot_change_forget_hidden_logits(self) -> None:
        forget_hidden = torch.randn(5, 15)
        forget_basis = method.active.orthonormal_row_basis(forget_hidden)
        _, restore_basis = method.build_restore_basis(
            torch.randn(23, 15), forget_basis, max_rank=7
        )
        coefficients = torch.randn(8, restore_basis.shape[0])
        delta = method.restoration_delta(coefficients, restore_basis)
        changed_logits = forget_hidden @ delta.T
        torch.testing.assert_close(
            changed_logits,
            torch.zeros_like(changed_logits),
            atol=3e-5,
            rtol=0,
        )

    def test_incremental_kl_is_zero_at_zero_restoration(self) -> None:
        baseline = torch.tensor(0.237, dtype=torch.float32)
        increase = method.incremental_retain_kl(baseline, baseline)
        self.assertEqual(float(increase), 0.0)
        self.assertEqual(method.incremental_retain_kl(0.1, 0.2), 0.0)

    def test_nonselected_rows_stay_bitwise_setting5(self) -> None:
        setting5 = torch.randn(13, 9, dtype=torch.bfloat16)
        candidate = setting5.clone()
        candidate[torch.tensor([2, 7])] += torch.tensor(0.5, dtype=torch.bfloat16)
        self.assertTrue(
            method.nonselected_rows_equal_setting5(candidate, setting5, [2, 7])
        )
        candidate[5, 3] += torch.tensor(0.25, dtype=torch.bfloat16)
        self.assertFalse(
            method.nonselected_rows_equal_setting5(candidate, setting5, [2, 7])
        )


class ProtocolIsolationTests(unittest.TestCase):
    def test_official_rwku_paths_are_rejected_before_freeze(self) -> None:
        official = Path("data/rwku/forget_level1.json")
        with self.assertRaisesRegex(ValueError, "official/evaluation"):
            rank2.reject_official_path(official, label="pre-freeze input")

    def test_entrypoint_has_no_evaluate_stage(self) -> None:
        stage_action = next(
            action for action in method.parser()._actions if action.dest == "stage"
        )
        self.assertEqual(
            set(stage_action.choices), {"preflight", "prepare", "train", "verify"}
        )

    def test_new_experiment_isolated_from_existing_rank2(self) -> None:
        self.assertNotEqual(method.EXPERIMENT_ID, "rwku-s5e600-rank2-active-sk-v3atomic-seed0-v1")
        self.assertEqual(method.SETTING5_STEPS, 600)
        self.assertEqual(method.RANK0_REPAIR_RANK, 0)
        self.assertEqual(method.RESTORE_RANK, 16)
        self.assertEqual(method.RANK0_REPAIR_LR, 5e-3)
        source = method.SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("outputs/rwku_s5e600_rank2_active", source)
        self.assertNotIn("official_evaluation.json", source)


if __name__ == "__main__":
    unittest.main()
