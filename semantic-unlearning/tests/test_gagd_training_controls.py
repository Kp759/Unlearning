import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import gagd_compare as MODULE  # noqa: E402


class GAGDTrainingControlTests(unittest.TestCase):
    class TinyTokenizer:
        pad_token_id = 0
        eos_token = "<eos>"

        def __call__(self, text, add_special_tokens=False):
            if text.endswith(self.eos_token):
                text = text[: -len(self.eos_token)]
                return {"input_ids": [ord(char) for char in text] + [99]}
            return {"input_ids": [ord(char) for char in text]}

    def test_epoch_sampler_covers_every_example_before_repeating(self):
        examples = [
            MODULE.Example(prompt=str(i), answer=str(i))
            for i in range(5)
        ]
        sampler = MODULE.EpochBatchSampler(examples, batch_size=2, seed=7)
        seen = [
            example.prompt
            for _ in range(3)
            for example in sampler.next_batch()
        ]

        self.assertEqual(len(set(seen[:5])), 5)

    def test_epoch_sampler_is_seed_deterministic(self):
        examples = [
            MODULE.Example(prompt=str(i), answer=str(i))
            for i in range(8)
        ]
        first = MODULE.EpochBatchSampler(examples, batch_size=3, seed=11)
        second = MODULE.EpochBatchSampler(examples, batch_size=3, seed=11)

        for _ in range(4):
            self.assertEqual(
                [example.prompt for example in first.next_batch()],
                [example.prompt for example in second.next_batch()],
            )

    def test_scope_specific_learning_rates_override_global_rate(self):
        args = SimpleNamespace(lr=1e-5, full_lr=1e-3, emb_lm_lr=1e-4)

        self.assertEqual(MODULE.learning_rate_for_mode("full_all_tokens", args), 1e-3)
        self.assertEqual(MODULE.learning_rate_for_mode("emb_lm_all_tokens", args), 1e-4)

    def test_scope_optimizer_can_be_overridden_globally(self):
        args = SimpleNamespace(
            optimizer=None,
            full_optimizer="sgd",
            emb_lm_optimizer="adamw",
        )
        self.assertEqual(MODULE.optimizer_name_for_mode("full_all_tokens", args), "sgd")
        self.assertEqual(MODULE.optimizer_name_for_mode("emb_lm_all_tokens", args), "adamw")

        args.optimizer = "adam"
        self.assertEqual(MODULE.optimizer_name_for_mode("full_all_tokens", args), "adam")
        self.assertEqual(MODULE.optimizer_name_for_mode("emb_lm_all_tokens", args), "adam")

    def test_positive_margin_demands_a_larger_nll_gap(self):
        target_true_nll = torch.tensor(8.0)
        target_new_nll = torch.tensor(9.0)

        boundary_loss = MODULE.mcf_margin_objective(target_true_nll, target_new_nll, 0.0)
        robust_loss = MODULE.mcf_margin_objective(target_true_nll, target_new_nll, 2.0)

        self.assertGreater(robust_loss.item(), boundary_loss.item())

    def test_official_aligned_batch_excludes_eos(self):
        example = MODULE.Example(prompt="P", answer="A")
        with_eos = MODULE.build_batch(
            self.TinyTokenizer(), [example], torch.device("cpu"), append_eos=True
        )
        without_eos = MODULE.build_batch(
            self.TinyTokenizer(), [example], torch.device("cpu"), append_eos=False
        )

        self.assertEqual(with_eos["input_ids"].tolist(), [[ord("P"), ord("A"), 99]])
        self.assertEqual(without_eos["input_ids"].tolist(), [[ord("P"), ord("A")]])


if __name__ == "__main__":
    unittest.main()
