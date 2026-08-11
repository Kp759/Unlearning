import unittest

import torch

from scripts.mcf_forget_only_active_repair_lora import SparseLoRADelta


class SparseLoRADeltaTests(unittest.TestCase):
    def test_rank1_two_rows_llama3b_shapes_and_zero_init(self):
        module = SparseLoRADelta(
            n_rows=2,
            hidden_size=3072,
            rank=1,
            alpha=1.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(tuple(module.lora_A.shape), (1, 3072))
        self.assertEqual(tuple(module.lora_B.shape), (2, 1))
        self.assertEqual(tuple(module.effective_delta().shape), (2, 3072))
        self.assertEqual(module.trainable_parameter_count, 3074)
        self.assertTrue(torch.equal(module.effective_delta(), torch.zeros(2, 3072)))

    def test_rank2_parameter_count(self):
        module = SparseLoRADelta(
            n_rows=2,
            hidden_size=3072,
            rank=2,
            alpha=2.0,
            device=torch.device("cpu"),
        )
        self.assertEqual(module.trainable_parameter_count, 6148)
        self.assertEqual(tuple(module.effective_delta().shape), (2, 3072))


if __name__ == "__main__":
    unittest.main()
