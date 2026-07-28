import argparse
import importlib
import random
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import run_zerounlearn_tofu as MODULE  # noqa: E402
import tofu_gagd_results as RESULTS  # noqa: E402


class FakeTokenizer:
    chat_template = "fake-chat-template"
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = None
    pad_token_id = None

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

    def __call__(self, text, *, add_special_tokens):
        if text == self.eos_token and add_special_tokens is False:
            return {"input_ids": [self.eos_token_id]}
        return {"input_ids": [99]}


def protocol_args(**updates):
    values = {
        "forget_split": "forget05",
        "retain_split": "retain95",
        "seed": 42,
        "forget_num": 200,
        "retain_num": 1000,
        "dtype": "bfloat16",
        "max_length": 256,
        "max_new_tokens": 64,
        "n_real_authors_eval": None,
        "n_world_facts_eval": None,
        "n_perturbed_eval": None,
        "max_forget_answer_probability": 2e-5,
        "min_retain_probability_ratio": 0.9999998,
    }
    values.update(updates)
    return argparse.Namespace(**values)


def sample(split, source_index, question, answer):
    return MODULE.SampledTOFURow(
        split=split,
        source_index=source_index,
        sampled_position=0,
        row={"question": question, "answer": answer},
    )


class ZeroUnlearnTOFUTests(unittest.TestCase):
    def test_protocol_is_fixed_to_framework_tofu_run(self):
        MODULE.validate_protocol_args(protocol_args())
        with self.assertRaisesRegex(ValueError, "--seed must be 42"):
            MODULE.validate_protocol_args(protocol_args(seed=0))
        with self.assertRaisesRegex(ValueError, "--forget-num must be 200"):
            MODULE.validate_protocol_args(protocol_args(forget_num=50))
        with self.assertRaisesRegex(ValueError, "--retain-split"):
            MODULE.validate_protocol_args(
                protocol_args(retain_split="retain90")
            )

    def test_sampling_matches_tofu_eval_for_truncated_and_full_splits(self):
        rows = [
            {"question": f"q{index}", "answer": f"a{index}"}
            for index in range(8)
        ]
        sampled = MODULE.deterministic_sampled_rows(
            rows,
            split="retain95",
            sample_size=3,
            seed=42,
        )
        evaluator_rows = random.Random(42).sample(rows, 3)
        self.assertEqual(
            [entry.row for entry in sampled],
            evaluator_rows,
        )
        full = MODULE.deterministic_sampled_rows(
            rows,
            split="forget05",
            sample_size=len(rows),
            seed=42,
        )
        self.assertEqual(
            [entry.source_index for entry in full],
            list(range(len(rows))),
        )

    def test_prompt_template_reproduces_exact_chat_prompt(self):
        tokenizer = FakeTokenizer()
        question = "What does {Ada} write?"
        template = MODULE.prompt_template_for_question(tokenizer, question)
        self.assertEqual(template.count("{}"), 1)
        self.assertEqual(
            template.format(question),
            "<user>Question: What does {Ada} write? Answer:</user>"
            "<assistant>",
        )

    def test_forget_answer_maps_to_sensitive_slot_and_eos_to_neutral(self):
        tokenizer = FakeTokenizer()
        source = sample("forget05", 7, "Who is Loma?", "Loma is an author.")
        requests = MODULE.sampled_rows_to_requests(
            [source],
            tokenizer,
            neutral_target="<eos>",
            case_id_offset=0,
        )
        request = requests[0]
        self.assertEqual(request["case_id"], 7)
        self.assertEqual(request["subject"], source.question)
        self.assertEqual(request["target_true"], {"str": source.answer})
        self.assertEqual(request["target_new"], {"str": "<eos>"})
        MODULE.validate_requests(
            [source],
            requests,
            tokenizer,
            neutral_target="<eos>",
            case_id_offset=0,
        )

    def test_retain_answer_is_not_replaced_by_neutral_target(self):
        tokenizer = FakeTokenizer()
        source = sample("retain95", 11, "Who is Ria?", "Ria is an author.")
        requests = MODULE.sampled_rows_to_requests(
            [source],
            tokenizer,
            neutral_target=None,
            case_id_offset=MODULE.CASE_ID_RETAIN_OFFSET,
        )
        request = requests[0]
        self.assertEqual(
            request["case_id"],
            MODULE.CASE_ID_RETAIN_OFFSET + 11,
        )
        self.assertEqual(request["target_true"], {"str": source.answer})
        self.assertEqual(request["target_new"], {"str": source.answer})

    def test_request_guard_rejects_wrong_sensitive_answer(self):
        tokenizer = FakeTokenizer()
        source = sample("forget05", 3, "Question?", "Sensitive answer")
        requests = MODULE.sampled_rows_to_requests(
            [source],
            tokenizer,
            neutral_target="<eos>",
            case_id_offset=0,
        )
        requests[0]["target_true"] = {"str": "wrong"}
        with self.assertRaisesRegex(RuntimeError, "sensitive answer changed"):
            MODULE.validate_requests(
                [source],
                requests,
                tokenizer,
                neutral_target="<eos>",
                case_id_offset=0,
            )

    def test_base_summary_guard_requires_same_protocol(self):
        summary = {
            "seed": 42,
            "forget_split": "forget05",
            "retain_split": "retain95",
            "n_forget_eval": 200,
            "n_retain_eval": 1000,
            "forget_answer_prob": 0.9,
            "retain_answer_prob": 0.95,
        }
        MODULE.validate_base_summary(
            summary,
            seed=42,
            forget_split="forget05",
            retain_split="retain95",
            forget_num=200,
            retain_num=1000,
        )
        with self.assertRaisesRegex(ValueError, "n_retain_eval"):
            MODULE.validate_base_summary(
                {**summary, "n_retain_eval": 100},
                seed=42,
                forget_split="forget05",
                retain_split="retain95",
                forget_num=200,
                retain_num=1000,
            )

    def test_reference_distribution_must_match_evaluated_perturbed_subset(self):
        with patch.object(MODULE, "load_dataset", return_value=list(range(8))):
            self.assertEqual(
                MODULE.validate_reference_truth_ratio_count(
                    [0.1] * 8,
                    forget_split="forget05",
                    n_perturbed=None,
                ),
                8,
            )
            self.assertEqual(
                MODULE.validate_reference_truth_ratio_count(
                    [0.1] * 3,
                    forget_split="forget05",
                    n_perturbed=3,
                ),
                3,
            )
            with self.assertRaisesRegex(ValueError, "expected 8, got 3"):
                MODULE.validate_reference_truth_ratio_count(
                    [0.1] * 3,
                    forget_split="forget05",
                    n_perturbed=None,
                )

    def test_zero_unlearn_is_optional_second_comparison_row(self):
        self.assertEqual(RESULTS.METHOD_ORDER[0], "Base")
        self.assertEqual(RESULTS.METHOD_ORDER[1], "Original ZeroUnlearn")
        self.assertEqual(
            RESULTS.METHOD_KEYS["Original ZeroUnlearn"],
            MODULE.METHOD_KEY,
        )
        self.assertNotIn(
            MODULE.METHOD_KEY,
            RESULTS.REQUIRED_METHOD_KEYS,
        )

    def test_loaded_evaluator_wraps_model_without_reloading(self):
        scorer_module = types.ModuleType("rouge_score.rouge_scorer")
        scorer_module.RougeScorer = lambda *args, **kwargs: object()
        package = types.ModuleType("rouge_score")
        package.rouge_scorer = scorer_module
        with patch.dict(
            sys.modules,
            {
                "rouge_score": package,
                "rouge_score.rouge_scorer": scorer_module,
            },
        ):
            tofu_eval = importlib.import_module("tofu_eval")

            class TinyModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.weight = torch.nn.Parameter(torch.ones(1))
                    self.config = SimpleNamespace(pad_token_id=None)

            model = TinyModel()
            tokenizer = FakeTokenizer()
            evaluator = tofu_eval.Evaluator.from_loaded(
                model,
                tokenizer,
                device="cpu",
                max_length=256,
            )
            self.assertIs(evaluator.model, model)
            self.assertIs(evaluator.tokenizer, tokenizer)
            self.assertEqual(evaluator.max_length, 256)
            self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)

    def test_sample_digest_is_stable_and_order_sensitive(self):
        first = sample("forget05", 0, "Q0", "A0")
        second = sample("retain95", 1, "Q1", "A1")
        digest = MODULE.sampled_rows_sha256([first], [second])
        self.assertEqual(digest, MODULE.sampled_rows_sha256([first], [second]))
        self.assertNotEqual(
            digest,
            MODULE.sampled_rows_sha256([second], [first]),
        )


if __name__ == "__main__":
    unittest.main()
