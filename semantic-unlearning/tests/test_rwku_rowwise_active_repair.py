import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
import sys

sys.path.insert(0, str(SCRIPTS))

import rwku_rowwise_active_repair as REPAIR  # noqa: E402


def passing_metrics():
    return {
        "direct_generation_recovery": 0.0,
        "cloze_generation_recovery": 0.0,
        "paraphrase_generation_recovery": 0.0,
        "generated_geometric_answer_probability": 0.001,
        "active_violation_count": 0,
        "full_retain_probability_ratio": 1.0,
        "geometric_retain_probability_ratio": 1.0,
        "mean_retain_kl": 0.0,
        "p95_retain_kl": 0.0,
        "retain_top1_agreement": 1.0,
        "protected_answer_probability_ratio": 1.0,
        "protected_selected_row_logit_drift": 0.0,
        "protected_top1_changes": 0,
        "proxy_ppl": 10.0,
        "base_proxy_ppl": 10.0,
        "nonselected_rows_equal_base": True,
    }


class PieceTokenizer:
    eos_token_id = 0
    bos_token_id = 1
    pad_token_id = 2
    unk_token_id = 3
    all_special_ids = [0, 1, 2, 3, 9]

    pieces = {
        0: "<eos>",
        4: ",",
        5: "   ",
        6: "123",
        7: "a",
        8: " English",
        9: "<special>",
        10: " novelist",
        11: " horror",
    }

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(self.pieces[int(value)] for value in token_ids)

    def convert_tokens_to_ids(self, value):
        return 9 if value == "<|eot_id|>" else 3


class EligibilityTests(unittest.TestCase):
    def test_ineligible_row_classes_are_explicit(self):
        tokenizer = PieceTokenizer()
        cases = {
            0: "tokenizer_special_row",
            4: "punctuation_only",
            5: "whitespace_only",
            6: "numeric_only",
            7: "single_character_alphanumeric",
            8: "protected_overlap",
            9: "tokenizer_special_row",
        }
        for token_id, reason in cases.items():
            with self.subTest(token_id=token_id):
                row = REPAIR.classify_output_row(
                    token_id,
                    tokenizer,
                    protected_row_ids={8},
                )
                self.assertFalse(row.eligible)
                self.assertIn(reason, row.reasons)

    def test_high_frequency_row_is_excluded_and_content_row_is_safe(self):
        tokenizer = PieceTokenizer()
        high = REPAIR.classify_output_row(
            10,
            tokenizer,
            retain_document_frequency=0.2,
            maximum_document_frequency=0.01,
        )
        safe = REPAIR.classify_output_row(
            11,
            tokenizer,
            retain_document_frequency=0.005,
            maximum_document_frequency=0.01,
        )
        self.assertIn("high_retain_document_frequency", high.reasons)
        self.assertTrue(safe.eligible)


class ActiveProvenanceTests(unittest.TestCase):
    def point(self, bundle, digest, **updates):
        value = {
            "fact_id": "fact-1",
            "view_id": "fact-1:direct",
            "prompt_style": "direct question",
            "answer_alias": "answer",
            "source_record_sha256": "a" * 64,
            "training_bundle_sha256": digest,
            "source_path": str(bundle.resolve()),
            "source_artifact_role": "training_bundle",
            "level": "generated",
            "active_source": REPAIR.ACTIVE_SOURCE,
        }
        value.update(updates)
        return value

    def test_target_only_active_source_is_generated_views(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "generated_training_bundle.json"
            bundle.write_text("{}", encoding="utf-8")
            points = REPAIR.validate_active_points(
                [self.point(bundle, "b" * 64)],
                training_bundle_path=bundle,
                training_bundle_sha256="b" * 64,
            )
            self.assertEqual(points[0]["active_source"], REPAIR.ACTIVE_SOURCE)

    def test_official_level_rows_cannot_become_active(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory) / "generated_training_bundle.json"
            bundle.write_text("{}", encoding="utf-8")
            for update in (
                {"level": "1"},
                {"source_path": str(Path(directory) / "forget_level2.json")},
                {"source_artifact_role": "official_locked_eval"},
            ):
                with self.subTest(update=update):
                    with self.assertRaisesRegex(
                        ValueError, "Official|Evaluation|bound"
                    ):
                        REPAIR.validate_active_points(
                            [self.point(bundle, "b" * 64, **update)],
                            training_bundle_path=bundle,
                            training_bundle_sha256="b" * 64,
                        )


class RowwiseSelectionTests(unittest.TestCase):
    def test_safe_row_can_use_strong_scale_while_risky_row_is_zero(self):
        calls = []

        def evaluate(scales):
            calls.append(dict(scales))
            metrics = passing_metrics()
            if scales.get(11, 0.0) > 0.0:
                metrics["protected_selected_row_logit_drift"] = 0.1
            return metrics

        report = REPAIR.select_rowwise_scales(
            [10, 11],
            evaluate=evaluate,
            row_contributions={
                10: {
                    "generated_efficacy_contribution": 2.0,
                    "protected_drift_contribution": 0.001,
                },
                11: {
                    "generated_efficacy_contribution": 1.0,
                    "protected_drift_contribution": 1.0,
                },
            },
        )
        self.assertEqual(report["selected_scale_by_row"]["10"], 1.0)
        self.assertEqual(report["selected_scale_by_row"]["11"], 0.0)
        self.assertTrue(report["selected_success"])
        self.assertTrue(any(len(row) == 2 for row in calls))

    def test_combined_repair_rechecks_every_protection_gate(self):
        observed = []

        def evaluate(scales):
            observed.append(dict(scales))
            metrics = passing_metrics()
            if scales.get(10, 0.0) and scales.get(11, 0.0):
                metrics["protected_top1_changes"] = 1
            return metrics

        report = REPAIR.select_rowwise_scales([10, 11], evaluate=evaluate)
        self.assertTrue(any(set(row) == {10, 11} for row in observed))
        self.assertTrue(report["combined_gate_results"]["protection_passed"])
        self.assertEqual(report["selected_scale_by_row"]["11"], 0.0)

    def test_materialization_is_from_immutable_weight(self):
        immutable = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        weight = immutable.clone()
        deltas = {1: torch.ones(4)}
        REPAIR.apply_rowwise_delta(weight, immutable, deltas, {1: 1.0})
        once = weight.clone()
        REPAIR.apply_rowwise_delta(weight, immutable, deltas, {1: 0.5})
        self.assertTrue(torch.equal(weight[1], immutable[1] + 0.5))
        self.assertFalse(torch.equal(weight[1], once[1] + 0.5))

    def test_failed_repair_is_never_reported_selected(self):
        def evaluate(scales):
            metrics = passing_metrics()
            metrics["direct_generation_recovery"] = 100.0
            return metrics

        report = REPAIR.select_rowwise_scales([10], evaluate=evaluate)
        self.assertFalse(report["selected_success"])
        self.assertFalse(report["repair_applied"])
        self.assertEqual(report["selected_scale_by_row"]["10"], 0.0)


if __name__ == "__main__":
    unittest.main()
