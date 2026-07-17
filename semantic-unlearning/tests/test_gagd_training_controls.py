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
        eos_token_id = 99
        bos_token_id = None
        unk_token_id = None
        eos_token = "<eos>"

        def __call__(self, text, add_special_tokens=False):
            if text.endswith(self.eos_token):
                text = text[: -len(self.eos_token)]
                return {"input_ids": [ord(char) for char in text] + [99]}
            return {"input_ids": [ord(char) for char in text]}

        def decode(self, token_ids):
            return "".join(chr(token_id) for token_id in token_ids)

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

    def test_post_training_groups_are_disjoint_and_protect_retain_overlap(self):
        forget = [
            MODULE.Example(
                prompt="P",
                answer="AB",
                target_new="AB",
                target_true="BC",
            )
        ]
        retain = [
            MODULE.Example(
                prompt="R",
                answer="D",
                target_new="D",
                target_true="E",
            ),
            MODULE.Example(
                prompt="R2",
                answer="A",
                target_new="A",
                target_true="F",
            ),
        ]

        groups = MODULE.collect_post_training_token_groups(
            self.TinyTokenizer(), forget, retain
        )

        self.assertEqual(groups.unique_target_new, [])
        self.assertEqual(groups.unique_target_true, [ord("C")])
        self.assertEqual(groups.overlap, [ord("A"), ord("B")])
        self.assertIn(ord("D"), groups.retain)
        self.assertIn(ord("E"), groups.retain)

    def test_post_training_restore_keeps_only_unique_new_and_scales_unique_true(self):
        input_base = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        output_base = input_base + 100
        input_weight = torch.nn.Parameter(input_base.clone())
        output_weight = torch.nn.Parameter(output_base.clone())
        tied_info = {
            "input_weight": input_weight,
            "output_weight": output_weight,
            "tied": False,
        }
        originals = MODULE.snapshot_embedding_output_weights(tied_info)
        with torch.no_grad():
            input_weight.add_(10)
            output_weight.add_(20)
        trained_input = input_weight.detach().clone()
        trained_output = output_weight.detach().clone()
        groups = MODULE.PostTrainingTokenGroups(
            target_new=[1, 3],
            target_true=[2, 3],
            retain=[4],
            unique_target_new=[1],
            unique_target_true=[2],
            overlap=[3],
        )

        MODULE.apply_post_training_row_restore(tied_info, originals, groups)

        self.assertTrue(torch.equal(input_weight[1], trained_input[1]))
        self.assertTrue(torch.equal(output_weight[1], trained_output[1]))
        self.assertTrue(torch.equal(input_weight[2], input_base[2] * 1.25))
        self.assertTrue(torch.equal(output_weight[2], output_base[2] * 1.25))
        for row in (0, 3, 4, 5):
            self.assertTrue(torch.equal(input_weight[row], input_base[row]))
            self.assertTrue(torch.equal(output_weight[row], output_base[row]))


if __name__ == "__main__":
    unittest.main()
