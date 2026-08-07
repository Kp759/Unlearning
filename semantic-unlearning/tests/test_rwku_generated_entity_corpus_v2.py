import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_rwku_generated_entity_corpus as GENERATOR  # noqa: E402


V2_CONFIG = (
    ROOT
    / "config"
    / "rwku"
    / "generation"
    / "llama32_3b_target_corpus_v2_strict_json.json"
)


def fact(**updates):
    value = {
        "subject": "Stephen King",
        "relation_id": "first_novel",
        "answer": "Carrie",
        "direct_question": "What was Stephen King's first novel?",
        "cloze": "Stephen King's first novel was ___.",
    }
    value.update(updates)
    return value


def build_args(root: Path):
    model = root / "model"
    model.mkdir()
    return argparse.Namespace(
        target_entity="Stephen King",
        entity_id="rwku:1_Stephen_King",
        generator_model=str(model),
        generator_revision="pinned-revision",
        generation_config=V2_CONFIG,
        seed=0,
        independent_resource=[],
        output_dir=root / "output",
        dry_run=False,
    )


class RobustJSONParserTests(unittest.TestCase):
    def assert_parses(self, text, count, expected_mode):
        result = GENERATOR.parse_json_objects(text)
        self.assertEqual(result["parsed_object_count"], count)
        self.assertEqual(result["parse_mode"], expected_mode)
        return result

    def test_compact_ndjson(self):
        second = fact(
            relation_id="birth_place",
            answer="Portland, Maine",
            direct_question="Where was Stephen King born?",
            cloze="Stephen King was born in ___.",
        )
        self.assert_parses(
            json.dumps(fact(), separators=(",", ":"))
            + "\n"
            + json.dumps(second, separators=(",", ":")),
            2,
            "ndjson",
        )

    def test_fenced_ndjson(self):
        text = (
            "```json\n"
            + json.dumps(fact())
            + "\n"
            + json.dumps(
                fact(
                    relation_id="occupation",
                    answer="author",
                    direct_question="What is Stephen King's occupation?",
                    cloze="Stephen King's occupation is ___.",
                )
            )
            + "\n```"
        )
        self.assert_parses(text, 2, "fenced_ndjson")

    def test_pretty_printed_json_array(self):
        self.assert_parses(json.dumps([fact()], indent=2), 1, "pretty_printed_array")

    def test_pretty_printed_single_object(self):
        self.assert_parses(json.dumps(fact(), indent=2), 1, "pretty_printed_object")

    def test_surrounding_prose_with_valid_embedded_json(self):
        text = "Result follows:\n" + json.dumps(fact(), indent=2) + "\nEnd result."
        self.assert_parses(text, 1, "raw_decode_scan")

    def test_malformed_json_remains_rejected(self):
        malformed = '{"subject":"Stephen King","relation_id":"first_novel"'
        self.assert_parses(malformed, 0, "no_parseable_json")

    def test_duplicate_objects_are_canonically_deduplicated(self):
        duplicate_with_different_key_order = {
            "cloze": fact()["cloze"],
            "direct_question": fact()["direct_question"],
            "answer": fact()["answer"],
            "relation_id": fact()["relation_id"],
            "subject": fact()["subject"],
        }
        result = self.assert_parses(
            json.dumps([fact(), duplicate_with_different_key_order]),
            1,
            "complete_array",
        )
        self.assertEqual(result["duplicate_object_count"], 1)


class SemanticValidationTests(unittest.TestCase):
    def rejection_reason(self, candidate):
        facts, rejected, _ = GENERATOR.extract_generated_facts_detailed(
            [{"generated_text": json.dumps(candidate)}],
            entity_id="rwku:1_Stephen_King",
            target_entity="Stephen King",
        )
        self.assertFalse(facts)
        self.assertEqual(len(rejected), 1)
        return rejected[0]["reason"]

    def test_empty_direct_question_is_rejected(self):
        self.assertEqual(
            self.rejection_reason(fact(direct_question="")),
            "missing_or_empty_direct_question",
        )

    def test_empty_cloze_is_rejected(self):
        self.assertEqual(
            self.rejection_reason(fact(cloze="")),
            "missing_or_empty_cloze",
        )

    def test_cloze_without_blank_is_rejected(self):
        self.assertEqual(
            self.rejection_reason(fact(cloze="Stephen King's first novel was Carrie.")),
            "invalid_cloze_marker_count",
        )

    def test_negative_pseudo_facts_are_rejected(self):
        for answer in (
            "No",
            "None",
            "Unknown",
            "not applicable",
            "n/a",
            "deceased=false",
        ):
            with self.subTest(answer=answer):
                self.assertEqual(
                    self.rejection_reason(fact(answer=answer)),
                    "negative_or_null_answer",
                )

    def test_direct_question_requires_question_mark(self):
        self.assertEqual(
            self.rejection_reason(
                fact(direct_question="Name Stephen King's first novel")
            ),
            "direct_question_missing_question_mark",
        )

    def test_placeholders_other_than_single_cloze_blank_are_rejected(self):
        for update in (
            {"direct_question": "What was [MASK]?"},
            {"answer": "{answer}"},
            {"cloze": "Stephen King's first novel was <blank> ___."},
        ):
            with self.subTest(update=update):
                self.assertEqual(
                    self.rejection_reason(fact(**update)),
                    "forbidden_placeholder",
                )


class ChatTemplateTests(unittest.TestCase):
    def test_chat_template_is_used_when_available(self):
        class ChatTokenizer:
            def __init__(self):
                self.calls = []

            def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
                self.calls.append((messages, tokenize, add_generation_prompt))
                return f"<chat>{messages[0]['content']}</chat>"

        tokenizer = ChatTokenizer()
        rendered, used = GENERATOR.render_chat_prompt(tokenizer, "prompt")
        self.assertTrue(used)
        self.assertEqual(rendered, "<chat>prompt</chat>")
        self.assertEqual(
            tokenizer.calls,
            [([{"role": "user", "content": "prompt"}], False, True)],
        )

    def test_raw_prompt_fallback_when_template_is_invalid(self):
        class InvalidChatTokenizer:
            @staticmethod
            def apply_chat_template(*args, **kwargs):
                raise LookupError("no chat template")

        rendered, used = GENERATOR.render_chat_prompt(
            InvalidChatTokenizer(), "raw prompt"
        )
        self.assertFalse(used)
        self.assertEqual(rendered, "raw prompt")


class GeneratorReceiptTests(unittest.TestCase):
    def test_v2_config_is_deterministic_and_category_specific(self):
        config, _ = GENERATOR._load_generation_config(V2_CONFIG)
        self.assertEqual(
            config["configuration_id"],
            "llama32_3b_target_corpus_v2_strict_json",
        )
        self.assertEqual(
            config["decoding"],
            {
                "do_sample": False,
                "max_new_tokens": 1536,
                "num_return_sequences": 1,
            },
        )
        rendered = [
            GENERATOR.structured_generation_prompt(
                template, target_entity="Stephen King"
            )
            for template in config["prompt_templates"]
        ]
        for prompt in rendered:
            self.assertIn("newline-delimited compact JSON objects only", prompt)
            self.assertIn("exactly one ___ marker", prompt)
            self.assertIn("Do not output a JSON array", prompt)
            self.assertIn("spouse=None", prompt)

    def test_zero_fact_failure_diagnostics_are_written_before_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_args(root)
            raw = [
                {
                    "prompt_index": 0,
                    "sequence_index": 0,
                    "chat_template_used": True,
                    "generated_text": json.dumps(fact(direct_question=""), indent=2),
                }
            ]
            tokenizer_identity = {
                "name_or_path": str(args.generator_model),
                "class": "FakeTokenizer",
                "vocab_size": 128,
                "eos_token_id": 2,
                "source_file_sha256": {},
                "chat_template_used": True,
                "chat_template_used_by_prompt": [True],
            }
            with mock.patch.object(
                GENERATOR,
                "_model_generate",
                return_value=(raw, tokenizer_identity),
            ):
                with self.assertRaisesRegex(ValueError, "no accepted entity facts"):
                    GENERATOR.build_generated_corpus(args)
            for filename in (
                "generated_raw_corpus.json",
                "generation_diagnostics.json",
                "rejected_generated_facts.json",
                "generator_failure_receipt.json",
            ):
                self.assertTrue((args.output_dir / filename).is_file(), filename)
            failure = json.loads(
                (args.output_dir / "generator_failure_receipt.json").read_text()
            )
            self.assertEqual(failure["status"], "failed_no_accepted_facts")
            self.assertFalse(failure["official_rwku_records_accessed"])
            self.assertEqual(failure["generator_model_revision"], "pinned-revision")
            self.assertEqual(
                failure["parser_implementation_revision"],
                GENERATOR.PARSER_IMPLEMENTATION_REVISION,
            )
            self.assertEqual(failure["accepted_fact_count"], 0)
            self.assertEqual(failure["rejected_fact_count"], 1)
            self.assertEqual(
                failure["rejection_reason_counts"],
                {"missing_or_empty_direct_question": 1},
            )
            self.assertEqual(
                failure["output_parse_diagnostics"][0]["parsed_object_count"], 1
            )
            self.assertIn("raw_generated_corpus_sha256", failure)
            self.assertIn("generation_configuration_sha256", failure)
            self.assertIn("failed_at_utc", failure)

    def test_success_receipt_contains_parser_and_count_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_args(root)
            raw = [
                {
                    "prompt_index": 0,
                    "sequence_index": 0,
                    "chat_template_used": True,
                    "generated_text": json.dumps([fact()], indent=2),
                }
            ]
            tokenizer_identity = {
                "name_or_path": str(args.generator_model),
                "class": "FakeTokenizer",
                "vocab_size": 128,
                "eos_token_id": 2,
                "source_file_sha256": {"tokenizer.json": "abc"},
                "chat_template_used": True,
                "chat_template_used_by_prompt": [True],
            }
            with mock.patch.object(
                GENERATOR,
                "_model_generate",
                return_value=(raw, tokenizer_identity),
            ):
                receipt = GENERATOR.build_generated_corpus(args)
            self.assertEqual(receipt["status"], "complete")
            self.assertTrue(receipt["chat_template_used"])
            self.assertEqual(
                receipt["parser_implementation_revision"],
                GENERATOR.PARSER_IMPLEMENTATION_REVISION,
            )
            self.assertEqual(receipt["parse_mode_counts"], {"pretty_printed_array": 1})
            self.assertEqual(receipt["accepted_fact_count"], 1)
            self.assertEqual(receipt["rejected_fact_count"], 0)
            self.assertIn("raw_generated_corpus_sha256", receipt)
            self.assertIn("final_entity_fact_bundle_sha256", receipt)
            self.assertTrue(
                (args.output_dir / "generated_training_bundle.json").is_file()
            )

    def test_official_rwku_evaluation_files_cannot_be_supplied_or_opened(self):
        official = Path("/not/opened/forget_level1.json")
        with mock.patch.object(
            Path,
            "open",
            side_effect=AssertionError("official file was opened"),
        ):
            with self.assertRaisesRegex(ValueError, "cannot be supplied"):
                GENERATOR.validate_independent_resource_path(official)
        parser_dests = {action.dest for action in GENERATOR.build_parser()._actions}
        self.assertNotIn("data_root", parser_dests)
        self.assertNotIn("official_evaluation", parser_dests)


if __name__ == "__main__":
    unittest.main()
