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
        sensitive_nll = torch.tensor(8.0)
        neutral_nll = torch.tensor(9.0)

        boundary_loss = MODULE.mcf_margin_objective(sensitive_nll, neutral_nll, 0.0)
        robust_loss = MODULE.mcf_margin_objective(sensitive_nll, neutral_nll, 2.0)

        self.assertGreater(robust_loss.item(), boundary_loss.item())

    def test_margin_objective_suppresses_sensitive_and_raises_neutral(self):
        """Gradient descent must forget the sensitive answer, not reinforce it."""
        sensitive_nll = torch.tensor(2.0, requires_grad=True)
        neutral_nll = torch.tensor(8.0, requires_grad=True)

        MODULE.mcf_margin_objective(sensitive_nll, neutral_nll, 1.0).backward()

        # Descent moves opposite the gradient: negative grad on sensitive means
        # its NLL rises (forgotten); positive grad on neutral means its NLL
        # falls (preferred).
        self.assertLess(sensitive_nll.grad.item(), 0.0)
        self.assertGreater(neutral_nll.grad.item(), 0.0)

    def test_margin_objective_concentrates_on_unforgotten_examples(self):
        not_yet_forgotten = MODULE.mcf_margin_objective(
            torch.tensor(2.0), torch.tensor(8.0), 1.0
        )
        already_forgotten = MODULE.mcf_margin_objective(
            torch.tensor(9.0), torch.tensor(3.0), 1.0
        )

        self.assertGreater(not_yet_forgotten.item(), already_forgotten.item())

    def test_forget_loss_rejects_an_unknown_sensitive_field(self):
        """The datasets disagree on which field is sensitive; typos must fail loudly."""
        with self.assertRaises(ValueError):
            MODULE.mcf_margin_forget_loss(
                None,
                None,
                [MODULE.Example(prompt="P", answer="A", target_new="N", target_true="T")],
                None,
                torch.device("cpu"),
                sensitive_field="target_sensitive",
            )

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
                answer="ABGH",
                target_new="ABGH",
                target_true="BCFH",
            )
        ]
        retain = [
            MODULE.Example(
                prompt="R",
                answer="ADH",
                target_new="ADH",
                target_true="EFI",
            ),
        ]

        groups = MODULE.collect_post_training_token_groups(
            self.TinyTokenizer(), forget, retain
        )

        self.assertEqual(groups.unique_target_new, [ord("G")])
        self.assertEqual(groups.unique_target_true, [ord("C")])
        self.assertEqual(groups.target_new_true_overlap, [ord("B")])
        self.assertEqual(groups.target_new_retain_overlap, [ord("A")])
        self.assertEqual(groups.target_new_true_retain_overlap, [ord("H")])
        self.assertEqual(groups.target_true_retain_overlap, [ord("F")])
        self.assertEqual(
            groups.overlap,
            [ord("A"), ord("B"), ord("F"), ord("H")],
        )
        self.assertIn(ord("D"), groups.retain)
        self.assertIn(ord("E"), groups.retain)

    def test_post_training_restore_interpolates_target_new_overlap_groups(self):
        input_base = torch.arange(32, dtype=torch.float32).reshape(8, 4)
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
            target_new=[1, 3, 4, 5],
            target_true=[2, 3, 5, 6],
            retain=[4, 5, 6],
            unique_target_new=[1],
            unique_target_true=[2],
            target_new_true_overlap=[3],
            target_new_retain_overlap=[4],
            target_new_true_retain_overlap=[5],
            target_true_retain_overlap=[6],
            overlap=[3, 4, 5, 6],
        )

        MODULE.apply_post_training_row_restore(tied_info, originals, groups)

        self.assertTrue(torch.equal(input_weight[1], trained_input[1]))
        self.assertTrue(torch.equal(output_weight[1], trained_output[1]))
        self.assertTrue(
            torch.equal(
                input_weight[2],
                input_base[2] + 0.75 * (trained_input[2] - input_base[2]),
            )
        )
        self.assertTrue(
            torch.equal(
                output_weight[2],
                output_base[2] + 0.75 * (trained_output[2] - output_base[2]),
            )
        )
        self.assertTrue(
            torch.equal(
                input_weight[3],
                input_base[3] + 0.75 * (trained_input[3] - input_base[3]),
            )
        )
        self.assertTrue(
            torch.equal(
                output_weight[3],
                output_base[3] + 0.75 * (trained_output[3] - output_base[3]),
            )
        )
        self.assertTrue(
            torch.equal(
                input_weight[4],
                input_base[4] + 0.50 * (trained_input[4] - input_base[4]),
            )
        )
        self.assertTrue(
            torch.equal(
                output_weight[4],
                output_base[4] + 0.50 * (trained_output[4] - output_base[4]),
            )
        )
        self.assertTrue(
            torch.equal(
                input_weight[5],
                input_base[5] + 0.25 * (trained_input[5] - input_base[5]),
            )
        )
        self.assertTrue(
            torch.equal(
                output_weight[5],
                output_base[5] + 0.25 * (trained_output[5] - output_base[5]),
            )
        )
        for row in (0, 6, 7):
            self.assertTrue(torch.equal(input_weight[row], input_base[row]))
            self.assertTrue(torch.equal(output_weight[row], output_base[row]))


if __name__ == "__main__":
    unittest.main()
