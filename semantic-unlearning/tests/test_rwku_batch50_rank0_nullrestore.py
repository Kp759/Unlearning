import sys
import unittest
from pathlib import Path

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import rwku_batch50_rank0_nullrestore as M  # noqa: E402


class TestBatch50Rank0NullRestore(unittest.TestCase):
    def test_restore_rank_parser(self):
        self.assertEqual(M.parse_rank_list("64,128"), (64, 128))
        self.assertEqual(M.parse_rank_list("128,64"), (64, 128))
        with self.assertRaises(Exception):
            M.parse_rank_list("64,64")
        with self.assertRaises(Exception):
            M.parse_rank_list("0,64")

    def test_margin_schedule_parser_is_strictly_increasing(self):
        self.assertEqual(
            M.parse_float_list("0.25,0.5,1,2,4"),
            (0.25, 0.5, 1.0, 2.0, 4.0),
        )
        with self.assertRaises(Exception):
            M.parse_float_list("1,0.5")
        with self.assertRaises(Exception):
            M.parse_float_list("0.5,0.5")

    def test_projection_preserves_logits_on_forget_span(self):
        torch.manual_seed(0)
        hidden = torch.randn(7, 12)
        basis = torch.linalg.qr(hidden.T, mode="reduced").Q.T
        delta = torch.randn(5, 12)
        projected = M.project_delta_to_span(delta, basis)
        before = basis @ delta.T
        after = basis @ projected.T
        self.assertTrue(torch.allclose(before, after, atol=1e-5, rtol=1e-5))

    def test_restore_basis_is_orthogonal_to_forget_basis(self):
        torch.manual_seed(1)
        forget_raw = torch.randn(4, 16)
        forget_basis = torch.linalg.qr(forget_raw.T, mode="reduced").Q.T
        retain_hidden = torch.randn(40, 16)
        null_hidden, restore_basis = M.build_restore_basis(
            retain_hidden,
            forget_basis,
            max_rank=6,
        )
        self.assertEqual(null_hidden.shape, retain_hidden.shape)
        self.assertLessEqual(restore_basis.shape[0], 6)
        overlap = restore_basis @ forget_basis.T
        self.assertTrue(torch.allclose(overlap, torch.zeros_like(overlap), atol=1e-5, rtol=1e-5))

    def test_cyclic_batch_never_uses_more_than_requested(self):
        values = list(range(10))
        batch = M._cyclic_batch(values, step=2, size=4)
        self.assertEqual(len(batch), 4)
        self.assertEqual(len(set(batch)), 4)


if __name__ == "__main__":
    unittest.main()
