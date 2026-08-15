import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

SCRIPT = SCRIPTS / "tofu_sure_rank0_forget.py"
SPEC = importlib.util.spec_from_file_location("tofu_sure_rank0_forget", SCRIPT)
sure = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(sure)


class TofuSureSensitiveRestoreTests(unittest.TestCase):
    def test_sensitive_rows_come_only_from_deficient_token_positions(self):
        caches = [
            SimpleNamespace(
                base_token_nll=torch.tensor([1.0, 5.0, 9.0]),
                target_selected_columns=torch.tensor([0, 1, 2]),
            )
        ]
        selected = sure.sensitive_rows_from_token_deficits(
            caches,
            [10, 20, 30],
            torch.tensor([6.0]),
            [0],
            tolerance=0.0,
        )
        self.assertEqual(selected, [10, 20])

    def test_shared_sensitive_row_is_never_restored(self):
        input_weight = torch.zeros((40, 2), dtype=torch.float32)
        output_weight = torch.zeros((40, 2), dtype=torch.float32)
        all_ids = [10, 20, 30]
        stage1a_input = torch.tensor([[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]])
        stage1a_output = stage1a_input + 1.0
        base_input = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        base_output = base_input + 0.5

        sure.apply_answer_row_policy(
            input_weight,
            output_weight,
            all_ids,
            [20],
            stage1a_input,
            stage1a_output,
            base_input,
            base_output,
        )

        self.assertTrue(torch.equal(input_weight[10], base_input[0]))
        self.assertTrue(torch.equal(input_weight[20], stage1a_input[1]))
        self.assertTrue(torch.equal(input_weight[30], base_input[2]))
        self.assertTrue(torch.equal(output_weight[10], base_output[0]))
        self.assertTrue(torch.equal(output_weight[20], stage1a_output[1]))
        self.assertTrue(torch.equal(output_weight[30], base_output[2]))

    def test_violating_sequence_positions_respect_tolerance(self):
        nll = torch.tensor([4.0, 5.0, 6.0])
        required = torch.tensor([4.5, 5.0, 5.5])
        self.assertEqual(sure.violating_sequence_positions(nll, required, 0.0), [0])
        self.assertEqual(sure.violating_sequence_positions(nll, required, 0.6), [])


if __name__ == "__main__":
    unittest.main()
