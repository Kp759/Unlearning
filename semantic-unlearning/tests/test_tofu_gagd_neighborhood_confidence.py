import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import tofu_gagd_neighborhood_confidence as MODULE  # noqa: E402
import tofu_gagd_results as RESULTS  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = None
    chat_template = None

    def __call__(self, text, **kwargs):
        del kwargs
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids):
        return "".join(chr(int(token_id)) for token_id in token_ids)


def instance(split, position, answer):
    question = f"question-{position}"
    return MODULE.TOFUAnswerInstance(
        split=split,
        source_index=position,
        sampled_position=position,
        question=question,
        answer=answer,
        prompt=f"Question: {question} Answer:",
    )


class TOFUNeighborhoodConfidenceTests(unittest.TestCase):
    def test_deterministic_calibration_sampling(self):
        first = MODULE.deterministic_sample_indices(100, 12, 1729)
        second = MODULE.deterministic_sample_indices(100, 12, 1729)
        other = MODULE.deterministic_sample_indices(100, 12, 1730)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), len(set(first)))

    def test_answer_sequence_excludes_prompt_and_has_no_added_eos(self):
        tokenizer = TinyTokenizer()
        item = instance("forget05", 0, "answer")
        full_ids, prompt_length = MODULE.answer_sequence_components(
            tokenizer,
            item,
            256,
        )
        expected = item.prompt + " " + item.answer
        self.assertEqual(full_ids, [ord(character) for character in expected])
        self.assertEqual(prompt_length, len(item.prompt))
        self.assertEqual(
            full_ids[prompt_length:],
            [ord(character) for character in " answer"],
        )
        self.assertNotIn(tokenizer.eos_token_id, full_ids[prompt_length:])

    def test_row_selection_uses_active_utility_rows_but_excludes_forget_rows(self):
        tokenizer = TinyTokenizer()
        forget = [instance("forget05", 0, "AB")]
        utility = [
            instance("retain", 0, "AC"),
            instance("real_authors", 1, "CD"),
            instance("world_facts", 2, "Z"),
        ]
        selected, report = MODULE.select_neighborhood_lm_head_rows(
            tokenizer,
            utility,
            [0, 1],
            forget,
            top_k=20,
            minimum_document_count=1,
        )
        self.assertNotIn(ord("A"), selected)
        self.assertNotIn(ord("B"), selected)
        self.assertIn(ord("C"), selected)
        self.assertIn(ord("D"), selected)
        self.assertNotIn(ord("Z"), selected)
        self.assertTrue(report["forget_answer_rows_excluded"])

    def test_required_tensors_protect_every_forget_and_utility_instance(self):
        forget = torch.tensor([3.0, 4.0, 5.0])
        utility = torch.tensor([2.0, 1.0])
        reference = torch.tensor([1.0, 1.5])
        required_forget, required_utility = MODULE.build_required_nll_tensors(
            forget,
            utility,
            reference,
            forget_nll_tolerance=0.2,
            reference_nll_slack=0.1,
        )
        self.assertEqual(required_forget.shape, forget.shape)
        self.assertEqual(required_utility.shape, utility.shape)
        self.assertTrue(
            torch.allclose(required_forget, torch.tensor([2.8, 3.8, 4.8]))
        )
        self.assertTrue(
            torch.allclose(required_utility, torch.tensor([1.1, 1.0]))
        )

    def test_confidence_objective_has_correct_gradient_and_forget_guard(self):
        current_forget = torch.tensor([3.0, 1.0], requires_grad=True)
        required_forget = torch.tensor([2.5, 2.0])
        current_utility = torch.tensor([2.0, 0.5], requires_grad=True)
        required_utility = torch.tensor([1.0, 1.0])
        terms = MODULE.confidence_objective_terms(
            current_forget,
            required_forget,
            current_utility,
            required_utility,
        )
        total = terms["forget_hinge"] + terms["confidence_hinge"]
        total.backward()
        self.assertEqual(float(current_forget.grad[0]), 0.0)
        self.assertLess(float(current_forget.grad[1]), 0.0)
        self.assertGreater(float(current_utility.grad[0]), 0.0)
        self.assertEqual(float(current_utility.grad[1]), 0.0)

    def test_candidate_priority_never_trades_forget_safety_for_confidence(self):
        safe = {
            "forget_protection_violation_count": 0,
            "neighborhood_target_unmet_count": 2,
            "utility_macro_answer_probability": 0.2,
            "selected_lm_head_delta_norm": 2.0,
        }
        unsafe = {
            "forget_protection_violation_count": 1,
            "neighborhood_target_unmet_count": 0,
            "utility_macro_answer_probability": 0.9,
            "selected_lm_head_delta_norm": 0.1,
        }
        self.assertLess(
            MODULE.candidate_priority(safe),
            MODULE.candidate_priority(unsafe),
        )

    def test_result_aggregator_preserves_fixed_order_and_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            specs = []
            for display_name in reversed(RESULTS.METHOD_ORDER):
                key = RESULTS.METHOD_KEYS[display_name]
                path = root / f"{key}.json"
                path.write_text(
                    json.dumps(
                        {
                            "seed": 42,
                            "forget_split": "forget05",
                            "retain_split": "retain95",
                            "forget_answer_prob": 0.1,
                            "retain_answer_prob": 0.9,
                            "forget_truth_ratio": 0.5,
                            "retain_truth_ratio": 0.8,
                            "real_authors_normalized_answer_prob": 0.7,
                            "world_facts_normalized_answer_prob": 0.6,
                            "forget_rougeL_recall": 0.2,
                            "retain_rougeL_recall": 0.8,
                            "tofu_real_authors_rougeL_recall": 0.7,
                            "tofu_world_facts_rougeL_recall": 0.6,
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                specs.append(f"{key}={path}")
            paths = RESULTS.parse_result_specs(specs)
            rows = []
            for display_name in RESULTS.METHOD_ORDER:
                path = paths[RESULTS.METHOD_KEYS[display_name]]
                rows.append(
                    RESULTS.row_from_summary(
                        display_name,
                        path,
                        json.loads(path.read_text(encoding="utf-8")),
                    )
                )
            protocol = RESULTS.verify_protocol(rows)
            self.assertEqual(
                [row["Method"] for row in rows],
                list(RESULTS.METHOD_ORDER),
            )
            self.assertEqual(protocol, (42, "forget05", "retain95"))
            self.assertTrue(all(row["Same protocol verified"] for row in rows))


if __name__ == "__main__":
    unittest.main()
