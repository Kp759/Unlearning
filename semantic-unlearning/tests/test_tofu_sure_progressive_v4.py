import sys
import unittest
from pathlib import Path
from unittest import mock

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import tofu_sure_rank0_forget_progressive_v4 as v4


class TofuSureProgressiveV4Tests(unittest.TestCase):
    def test_all_answer_input_rows_are_base_but_sensitive_output_stays_stage1a(self):
        input_weight = torch.zeros((5, 2), dtype=torch.float32)
        output_weight = torch.zeros((5, 2), dtype=torch.float32)
        answer_ids = [1, 3]
        sensitive_ids = [1]

        stage1a_input = torch.tensor([[10.0, 11.0], [30.0, 31.0]])
        stage1a_output = torch.tensor([[12.0, 13.0], [32.0, 33.0]])
        base_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        base_output = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        v4.apply_v4_row_policy(
            input_weight,
            output_weight,
            answer_ids,
            sensitive_ids,
            stage1a_input,
            stage1a_output,
            base_input,
            base_output,
        )

        self.assertTrue(torch.equal(input_weight[1], base_input[0]))
        self.assertTrue(torch.equal(input_weight[3], base_input[1]))
        self.assertTrue(torch.equal(output_weight[1], stage1a_output[0]))
        self.assertTrue(torch.equal(output_weight[3], base_output[1]))

    def test_v4_injects_zero_buffer_when_unspecified(self):
        with mock.patch.object(sys, "argv", ["prog", "--output-dir", "x"]):
            v4._require_zero_buffer()
            self.assertIn("--target-nll-buffer", sys.argv)
            idx = sys.argv.index("--target-nll-buffer")
            self.assertEqual(sys.argv[idx + 1], "0")

    def test_v4_rejects_nonzero_buffer(self):
        with mock.patch.object(
            sys,
            "argv",
            ["prog", "--target-nll-buffer", "0.25"],
        ):
            with self.assertRaises(ValueError):
                v4._require_zero_buffer()


if __name__ == "__main__":
    unittest.main()
