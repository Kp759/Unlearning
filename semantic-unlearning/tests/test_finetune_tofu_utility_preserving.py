import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import finetune_tofu_utility_preserving as TRAINER  # noqa: E402


class FakeTokenizer:
    chat_template = "fake-chat"
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<pad>"
    pad_token_id = 73

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
    ):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<user>{messages[0]['content']}</user><assistant>"

    def _encode(self, text):
        ids = [1]
        position = 0
        while position < len(text):
            if text.startswith(self.eos_token, position):
                ids.append(self.eos_token_id)
                position += len(self.eos_token)
            else:
                ids.append(10 + ord(text[position]))
                position += 1
        return ids

    def __call__(
        self,
        text,
        *,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        ids = self._encode(text)
        if truncation and max_length is not None:
            ids = ids[:max_length]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


def sweep_row(**updates):
    row = {
        "run_dir": "run",
        "learning_rate": 1e-5,
        "epochs": 1,
        "final_training_loss": 0.1,
        "tofu_probe_exact_match": 0.9,
        "tofu_probe_rouge_l": 0.96,
        "real_author_probe_exact_match": 0.8,
        "real_author_probe_rouge_l": 0.80,
        "world_fact_probe_exact_match": 0.8,
        "world_fact_probe_rouge_l": 0.80,
    }
    row.update(updates)
    return row


class UtilityPreservingTOFUTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.sample = {
            "question": "Who wrote the test book?",
            "answer": "A Test Author",
        }

    def test_prompt_construction_matches_tofu_eval(self):
        observed = TRAINER.format_question_prompt(
            self.tokenizer,
            self.sample["question"],
        )
        # This is the exact Evaluator.format_question_prompt behavior in
        # scripts/tofu_eval.py, expressed explicitly so this CPU-only test does
        # not require tofu_eval's optional ROUGE dependency at import time.
        self.assertEqual(
            observed,
            "<user>Question: Who wrote the test book? Answer:</user>"
            "<assistant>",
        )

    def test_prompt_tokens_are_masked(self):
        example = TRAINER.encode_training_example(
            self.sample,
            self.tokenizer,
            max_length=256,
        )
        prompt = TRAINER.format_question_prompt(
            self.tokenizer,
            self.sample["question"],
        )
        prompt_length = self.tokenizer(
            prompt,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )["input_ids"].shape[1]
        self.assertTrue(torch.all(example["labels"][:prompt_length].eq(-100)))

    def test_answer_and_eos_tokens_are_trainable_labels(self):
        example = TRAINER.encode_training_example(
            self.sample,
            self.tokenizer,
            max_length=256,
        )
        trainable = example["labels"][example["labels"].ne(-100)]
        self.assertGreater(trainable.numel(), 1)
        self.assertEqual(int(trainable[-1]), self.tokenizer.eos_token_id)
        leading_space_id = 10 + ord(" ")
        self.assertEqual(int(trainable[0]), leading_space_id)

    def test_padding_uses_tokenizer_pad_token_id(self):
        first = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "labels": torch.tensor([-100, 2, 3]),
        }
        second = {
            "input_ids": torch.tensor([4]),
            "attention_mask": torch.tensor([1]),
            "labels": torch.tensor([4]),
        }
        batch = TRAINER.collate_supervised_batch(
            [first, second],
            self.tokenizer.pad_token_id,
        )
        self.assertEqual(batch["input_ids"][1, 1:].tolist(), [73, 73])

    def test_labels_pad_with_ignore_index(self):
        first = {
            "input_ids": torch.tensor([1, 2]),
            "attention_mask": torch.tensor([1, 1]),
            "labels": torch.tensor([-100, 2]),
        }
        second = {
            "input_ids": torch.tensor([3]),
            "attention_mask": torch.tensor([1]),
            "labels": torch.tensor([3]),
        }
        batch = TRAINER.collate_supervised_batch(
            [first, second],
            self.tokenizer.pad_token_id,
        )
        self.assertEqual(int(batch["labels"][1, 1]), -100)

    def test_deterministic_shuffle_repeats_for_same_seed(self):
        first = TRAINER.deterministic_shuffle_indices(100, seed=42, epoch=3)
        second = TRAINER.deterministic_shuffle_indices(100, seed=42, epoch=3)
        different = TRAINER.deterministic_shuffle_indices(100, seed=43, epoch=3)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(sorted(first), list(range(100)))

    def test_nonempty_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            output.mkdir()
            (output / "existing.json").write_text(
                json.dumps({"do_not": "overwrite"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                TRAINER.prepare_output_directory(output, None)

    def test_sweep_selection_rejects_utility_collapse(self):
        collapsed = sweep_row(
            tofu_probe_rouge_l=0.99,
            real_author_probe_rouge_l=0.20,
            world_fact_probe_rouge_l=0.10,
        )
        self.assertFalse(TRAINER.sweep_row_is_eligible(collapsed))
        self.assertIsNone(TRAINER.select_sweep_winner([collapsed]))

    def test_sweep_selection_prefers_least_aggressive_tie(self):
        rows = [
            sweep_row(run_dir="high-lr", learning_rate=2e-5, epochs=1),
            sweep_row(run_dir="long", learning_rate=1e-5, epochs=5),
            sweep_row(run_dir="least", learning_rate=1e-5, epochs=2),
        ]
        winner = TRAINER.select_sweep_winner(rows)
        self.assertEqual(winner, 2)
        self.assertEqual(rows[winner]["run_dir"], "least")


if __name__ == "__main__":
    unittest.main()
