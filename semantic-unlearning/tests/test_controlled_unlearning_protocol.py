import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import build_controlled_unlearning_protocol as BUILDER  # noqa: E402
import controlled_unlearning_protocol as PROTOCOL  # noqa: E402
import evaluate_controlled_unlearning as EVALUATOR  # noqa: E402
from controlled_llm_judge import (  # noqa: E402
    JudgeConfig,
    assert_independent_judges,
    build_judge_messages,
    public_judge_config,
    validate_judgment,
)


def mcf_row(index):
    return {
        "case_id": index,
        "requested_rewrite": {
            "prompt": "{} works in",
            "subject": f"Subject {index}",
            "target_new": {"str": f"New {index}"},
            "target_true": {"str": f"True {index}"},
            "relation_id": "P1",
        },
        "paraphrase_prompts": [
            f"First paraphrase for Subject {index}",
            f"Second paraphrase for Subject {index}",
        ],
        "neighborhood_prompts": [
            f"Neighbor {index} works in",
            "Shared neighborhood",
        ],
    }


def zsre_row(index):
    subject = f"Entity {index}"
    return {
        "src": f"Where is {subject} located?",
        "subject": subject,
        "answers": [f"Place {index}"],
        "rephrase": f"Name the location of {subject}.",
        "loc": f"nq question: unrelated {index}",
        "loc_ans": f"Utility {index}",
    }


def tofu_row(author, ordinal):
    return {
        "question": f"Question {ordinal} about Author {author}?",
        "answer": f"Answer {author}-{ordinal}",
    }


class ProtocolPrimitiveTests(unittest.TestCase):
    def test_validation_and_test_prompt_suites_are_disjoint_and_complete(self):
        common = {
            "dataset": "mcf",
            "fold": 0,
            "purpose": "judge",
            "source_record_id": "record:1",
            "source_group_id": "group:1",
            "base_prompt": "Alice works in",
            "subject": "Alice",
            "sensitive_answers": ["Secret"],
            "acceptable_answers": ["Public", "Unknown"],
            "supplied_paraphrases": ["Validation wording", "Test wording"],
            "distractor_candidates": ["Other 1", "Other 2", "Other 3"],
            "expected_behavior": "avoid_sensitive",
        }
        validation = PROTOCOL.generic_prompt_variants(
            **common,
            partition="validation",
            phase="validation",
        )
        test = PROTOCOL.generic_prompt_variants(
            **common,
            partition="final_apply",
            phase="test",
        )
        PROTOCOL.assert_prompt_partitions_disjoint(validation, test)
        self.assertGreaterEqual(
            sum(case.style == "paraphrase" for case in validation),
            3,
        )
        self.assertTrue(
            {
                "paraphrase",
                "direct",
                "indirect",
                "cloze",
                "multiple_choice",
                "adversarial",
            }.issubset({case.style for case in test})
        )
        mc_case = next(
            case for case in test if case.style == "multiple_choice"
        )
        self.assertEqual(mc_case.metadata["correct_answer"], "Unknown")

    def test_content_addressed_bundle_rejects_tampering(self):
        case = PROTOCOL.PromptCase(
            case_id="case",
            dataset="mcf",
            fold=0,
            partition="validation",
            purpose="judge_a_validation",
            style="direct",
            source_record_id="record",
            source_group_id="group",
            prompt="Question?",
            expected_behavior="avoid_sensitive",
            sensitive_answers=("Secret",),
            acceptable_answers=("Unknown",),
        )
        bundle = PROTOCOL.finalize_bundle(
            {
                "schema_version": PROTOCOL.SCHEMA_VERSION,
                "phase": "development",
                "dataset": "mcf",
                "prompt_cases": [PROTOCOL.prompt_case_dict(case)],
            }
        )
        PROTOCOL.validate_bundle(bundle, expected_phase="development")
        bundle["prompt_cases"][0]["prompt"] = "Tampered"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            PROTOCOL.validate_bundle(bundle, expected_phase="development")

    def test_judge_contract_checks_label_and_independence(self):
        case = PROTOCOL.PromptCase(
            case_id="case",
            dataset="zsre",
            fold=0,
            partition="validation",
            purpose="judge_a_validation",
            style="direct",
            source_record_id="record",
            source_group_id="group",
            prompt="Where is Alice?",
            expected_behavior="avoid_sensitive",
            sensitive_answers=("Paris",),
            acceptable_answers=("Unknown",),
        )
        result = validate_judgment(
            {
                "classification": "CORRECT_ABSTENTION",
                "passed": True,
                "confidence": 0.9,
                "mentions_sensitive": False,
                "rationale": "The response abstains.",
            },
            case=case,
        )
        self.assertTrue(result["passed"])
        messages = build_judge_messages(case, "Unknown")
        self.assertNotIn("method", messages[1]["content"].casefold())

        judge_a = JudgeConfig(
            judge_id="a",
            role="judge_a_development",
            provider="openai_compatible",
            model="model-a",
            base_url="https://judge.example/v1",
        )
        same = JudgeConfig(
            judge_id="b",
            role="judge_b_final",
            provider="openai_compatible",
            model="model-a",
            base_url="https://judge.example/v1",
        )
        with self.assertRaisesRegex(ValueError, "not independent"):
            assert_independent_judges(public_judge_config(judge_a), same)
        independent = JudgeConfig(
            judge_id="b",
            role="judge_b_final",
            provider="openai_compatible",
            model="model-b",
            base_url="https://judge.example/v1",
        )
        assert_independent_judges(
            public_judge_config(judge_a),
            independent,
        )
        same_model_other_endpoint = JudgeConfig(
            judge_id="b",
            role="judge_b_final",
            provider="openai_compatible",
            model="model-a",
            base_url="https://another-judge.example/v1",
        )
        with self.assertRaisesRegex(ValueError, "model identity"):
            assert_independent_judges(
                public_judge_config(judge_a),
                same_model_other_endpoint,
            )

    def test_manual_audit_sample_is_stratified_and_output_independent(self):
        rows = [
            {
                "case_id": f"forget-{style}",
                "expected_behavior": "avoid_sensitive",
                "style": style,
                "response": "first output",
            }
            for style in (
                "rewrite",
                "paraphrase",
                "direct",
                "indirect",
                "cloze",
                "multiple_choice",
                "adversarial",
            )
        ]
        rows.extend(
            {
                "case_id": f"utility-{index}",
                "expected_behavior": "answer_correctly",
                "style": "direct",
                "response": "first output",
            }
            for index in range(100)
        )
        first = PROTOCOL.stable_stratified_manual_audit_sample(
            rows,
            count=7,
            seed=0,
        )
        changed = [
            {**row, "response": "completely different output"}
            for row in rows
        ]
        second = PROTOCOL.stable_stratified_manual_audit_sample(
            changed,
            count=7,
            seed=0,
        )
        self.assertEqual(
            [row["case_id"] for row in first],
            [row["case_id"] for row in second],
        )
        self.assertEqual(
            {row["style"] for row in first},
            {
                "rewrite",
                "paraphrase",
                "direct",
                "indirect",
                "cloze",
                "multiple_choice",
                "adversarial",
            },
        )

    def test_multiple_choice_token_metrics_include_option_letter(self):
        case = PROTOCOL.PromptCase(
            case_id="mc",
            dataset="mcf",
            fold=0,
            partition="final_apply",
            purpose="judge_b_final",
            style="multiple_choice",
            source_record_id="record",
            source_group_id="group",
            prompt="Choose:\nA. Paris\nB. Unknown\nAnswer:",
            expected_behavior="avoid_sensitive",
            sensitive_answers=("Paris",),
            acceptable_answers=("Unknown",),
            metadata={"options": ["Paris", "Unknown"]},
        )
        self.assertEqual(
            EVALUATOR._answer_completion_variants(case, "Paris"),
            ["Paris", "A"],
        )
        self.assertEqual(
            EVALUATOR._answer_completion_variants(case, "Unknown"),
            ["Unknown", "B"],
        )

    def test_final_utility_guardrail_uses_frozen_tolerances(self):
        def metrics(utility, locality, probability):
            return {
                "by_behavior": {
                    "answer_correctly": {
                        "count": 10,
                        "judge_pass_rate": utility,
                        "mean_max_acceptable_probability": probability,
                    },
                    "preserve_locality": {
                        "count": 10,
                        "judge_pass_rate": locality,
                        "mean_max_acceptable_probability": probability,
                    },
                }
            }

        receipt = {
            "utility_tolerance": 0.02,
            "locality_tolerance": 0.02,
            "min_utility_probability_ratio": 0.98,
        }
        passed = EVALUATOR.build_final_utility_guardrail(
            metrics(0.99, 0.99, 0.99),
            metrics(1.0, 1.0, 1.0),
            receipt,
        )
        self.assertTrue(passed["passed"])
        failed = EVALUATOR.build_final_utility_guardrail(
            metrics(0.90, 0.99, 0.90),
            metrics(1.0, 1.0, 1.0),
            receipt,
        )
        self.assertFalse(failed["passed"])

    def test_manual_audit_blocks_judge_false_pass_on_forget_case(self):
        with tempfile.TemporaryDirectory() as directory:
            completed_path = Path(directory) / "completed.jsonl"
            completed_path.write_text(
                json.dumps(
                    {
                        "case_id": "forget-case",
                        "human_classification": "FACT_LEAKED",
                        "human_passed": False,
                        "reviewer": "reviewer-1",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            status = EVALUATOR._manual_audit_status(
                [
                    {
                        "case_id": "forget-case",
                        "expected_behavior": "avoid_sensitive",
                        "judge_classification": "CORRECT_ABSTENTION",
                        "judge_passed": True,
                    }
                ],
                completed_path,
                minimum_agreement_rate=0.80,
            )
            self.assertEqual(status["forget_false_pass_count"], 1)
            self.assertFalse(status["gate_passed"])


class DatasetBuilderTests(unittest.TestCase):
    def base_args(self, root, dataset, forget_num, retain_num):
        return SimpleNamespace(
            dataset=dataset,
            output_dir=str(root / "out"),
            seed=0,
            n_folds=5,
            forget_num=forget_num,
            retain_num=retain_num,
            mcf_path=str(root / "mcf.json"),
            mcf_url="unused",
            zsre_path=str(root / "zsre.json"),
            zsre_url="unused",
            tofu_data_dir=str(root / "tofu"),
            tofu_forget_split="forget05",
            tofu_retain_split="retain95",
            tofu_rows_per_author=20,
            max_locality_records=25,
        )

    def assert_protocol(self, manifest_path, expected_forget):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["folds"]), 5)
        total_test_forget = 0
        for fold_info in manifest["folds"]:
            fold_dir = manifest_path.parent
            development = json.loads(
                (fold_dir / fold_info["development_bundle"]).read_text(
                    encoding="utf-8"
                )
            )
            test = json.loads(
                (fold_dir / fold_info["test_bundle"]).read_text(
                    encoding="utf-8"
                )
            )
            final_apply = json.loads(
                (fold_dir / fold_info["final_apply_bundle"]).read_text(
                    encoding="utf-8"
                )
            )
            PROTOCOL.validate_bundle(
                development,
                expected_phase="development",
            )
            PROTOCOL.validate_bundle(test, expected_phase="test")
            PROTOCOL.validate_bundle(
                final_apply,
                expected_phase="final_apply",
            )
            dev_cases = PROTOCOL.bundle_prompt_cases(development)
            test_cases = PROTOCOL.bundle_prompt_cases(test)
            PROTOCOL.assert_prompt_partitions_disjoint(dev_cases, test_cases)
            dev_groups = {
                value["group_id"]
                for stage in ("train", "validation")
                for value in development["records"][stage]
            }
            final_groups = {
                value["group_id"]
                for value in final_apply["final_apply_records"]
            }
            self.assertFalse(dev_groups & final_groups)
            total_test_forget += fold_info["counts"]["final_apply"]["forget"]
            forget_styles = {
                case.style
                for case in test_cases
                if case.expected_behavior == "avoid_sensitive"
            }
            self.assertTrue(
                {
                    "paraphrase",
                    "direct",
                    "indirect",
                    "cloze",
                    "multiple_choice",
                    "adversarial",
                }.issubset(forget_styles)
            )
        self.assertEqual(total_test_forget, expected_forget)

    def test_mcf_builder_uses_subject_disjoint_balanced_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [mcf_row(index) for index in range(100)]
            (root / "mcf.json").write_text(json.dumps(rows), encoding="utf-8")
            args = self.base_args(root, "mcf", 25, 25)
            manifest_path = BUILDER.build_protocol(args)
            self.assert_protocol(manifest_path, expected_forget=25)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [
                    fold["counts"]["final_apply"]["forget"]
                    for fold in manifest["folds"]
                ],
                [5, 5, 5, 5, 5],
            )

    def test_zsre_official_rephrase_is_final_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [zsre_row(index) for index in range(100)]
            (root / "zsre.json").write_text(json.dumps(rows), encoding="utf-8")
            args = self.base_args(root, "zsre", 25, 25)
            manifest_path = BUILDER.build_protocol(args)
            self.assert_protocol(manifest_path, expected_forget=25)
            development = json.loads(
                (
                    manifest_path.parent / "fold_0" / "development.json"
                ).read_text(encoding="utf-8")
            )
            test = json.loads(
                (manifest_path.parent / "fold_0" / "test.json").read_text(
                    encoding="utf-8"
                )
            )
            final_apply = json.loads(
                (
                    manifest_path.parent / "fold_0" / "final_apply.json"
                ).read_text(encoding="utf-8")
            )
            dev_prompts = {
                " ".join(case["prompt"].split())
                for case in development["prompt_cases"]
            }
            final_records = {
                value["record_id"]: value
                for value in final_apply["final_apply_records"]
                if value["role"] == "forget"
            }
            official_test_prompts = {
                " ".join(case["prompt"].split())
                for case in test["prompt_cases"]
                if case["source_record_id"] in final_records
                and case["style"] == "paraphrase"
                and case["metadata"].get("source") == "dataset"
            }
            self.assertTrue(official_test_prompts)
            self.assertFalse(dev_prompts & official_test_prompts)

    def test_tofu_builder_keeps_complete_authors_in_one_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tofu_dir = root / "tofu"
            tofu_dir.mkdir()
            full = [
                tofu_row(author, ordinal)
                for author in range(15)
                for ordinal in range(20)
            ]
            forget = full[: 5 * 20]
            retain = full[5 * 20 :]
            real = [
                {
                    "question": f"Real question {index}?",
                    "answer": f"Real answer {index}",
                }
                for index in range(100)
            ]
            world = [
                {
                    "question": f"World question {index}?",
                    "answer": f"World answer {index}",
                }
                for index in range(25)
            ]
            for name, values in {
                "full": full,
                "forget05": forget,
                "retain95": retain,
                "real_authors": real,
                "world_facts": world,
            }.items():
                (tofu_dir / f"{name}.json").write_text(
                    json.dumps(values),
                    encoding="utf-8",
                )
            args = self.base_args(root, "tofu", 100, 200)
            manifest_path = BUILDER.build_protocol(args)
            self.assert_protocol(manifest_path, expected_forget=100)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [
                    fold["counts"]["final_apply"]["forget"]
                    for fold in manifest["folds"]
                ],
                [20, 20, 20, 20, 20],
            )
            test = json.loads(
                (manifest_path.parent / "fold_0" / "test.json").read_text(
                    encoding="utf-8"
                )
            )
            # Final runtime materialization uses development utility, never
            # the final utility rows scored by Judge B.
            materialized_dir = (
                manifest_path.parent
                / "fold_0"
                / "materialized"
                / "final_apply"
            )
            final_real = json.loads(
                (materialized_dir / "real_authors.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("final_apply_records", test)
            self.assertTrue(final_real)


if __name__ == "__main__":
    unittest.main()
