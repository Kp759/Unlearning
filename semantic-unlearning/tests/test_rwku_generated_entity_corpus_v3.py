import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_rwku_generated_entity_corpus as GENERATOR  # noqa: E402
import rwku_artifact_access as ACCESS  # noqa: E402
import rwku_experiment as SETTING5  # noqa: E402
import rwku_zerounlearn_target_only as ZERO  # noqa: E402


V3_CONFIG = (
    ROOT
    / "config"
    / "rwku"
    / "generation"
    / "llama32_3b_target_corpus_v3_atomic_facts.json"
)
REGISTRY = ROOT / "config" / "rwku" / "relation_templates_v1.json"
ENTITY_ID = "rwku:1_Stephen_King"
SUBJECT = "Stephen King"


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<pad>"
    pad_token_id = 0
    unk_token = "<unk>"
    name_or_path = "/fake/tokenizer"

    def __len__(self):
        return 128

    def __call__(self, text, *, add_special_tokens=False):
        if text == self.eos_token and add_special_tokens is False:
            return {"input_ids": [self.eos_token_id]}
        return {"input_ids": [10]}

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        if tokenize is not False or add_generation_prompt is not True:
            raise AssertionError("unexpected chat-template arguments")
        return f"<user>{messages[0]['content']}</user><assistant>"


def atomic_candidate(relation_id="birth_date", **updates):
    value = {
        "status": "known",
        "subject": SUBJECT,
        "relation_id": relation_id,
        "answer": "September 21, 1947",
        "evidence_sentence": "Stephen King was born on September 21, 1947.",
    }
    value.update(updates)
    return value


def tokenizer_identity(request_count):
    return {
        "name_or_path": "/fake/tokenizer",
        "class": "FakeTokenizer",
        "vocab_size": 128,
        "eos_token_id": 2,
        "source_file_sha256": {},
        "chat_template_used": True,
        "chat_template_used_by_prompt": [True] * request_count,
    }


def build_args(root, *, minimum=1):
    model = root / "model"
    model.mkdir()
    return argparse.Namespace(
        target_entity=SUBJECT,
        entity_id=ENTITY_ID,
        generator_model=str(model),
        generator_revision="pinned-revision",
        generation_config=V3_CONFIG,
        relation_template_registry=REGISTRY,
        minimum_accepted_facts=minimum,
        seed=0,
        independent_resource=[],
        output_dir=root / "output",
        dry_run=False,
    )


def registry_and_requests(limit=None):
    relations, registry_sha = GENERATOR.load_relation_template_registry(
        REGISTRY,
        target_entity=SUBJECT,
    )
    if limit is not None:
        relations = relations[:limit]
    requests = GENERATOR.build_atomic_generation_requests(
        relations,
        target_entity=SUBJECT,
    )
    return relations, registry_sha, requests


def extract_one(candidate):
    relations, registry_sha, requests = registry_and_requests(limit=1)
    raw = [
        {
            "prompt_index": 0,
            "sequence_index": 0,
            "chat_template_used": True,
            "generated_text": json.dumps(candidate, separators=(",", ":")),
        }
    ]
    return GENERATOR.extract_atomic_facts(
        raw,
        requests,
        relations,
        entity_id=ENTITY_ID,
        target_entity=SUBJECT,
        relation_registry_path=REGISTRY,
        relation_registry_sha256=registry_sha,
    )


class RegistryAndRequestTests(unittest.TestCase):
    def test_one_enabled_relation_creates_one_generation_request(self):
        relations, _, requests = registry_and_requests()
        enabled = [
            relation
            for relation in relations
            if relation["primary_protocol_enabled"] is True
        ]
        self.assertEqual(len(requests), len(enabled))
        self.assertEqual(len({row["relation_id"] for row in requests}), len(requests))
        for request in requests:
            self.assertIn("exactly one compact JSON object", request["prompt"])
            self.assertNotIn('"direct_question"', request["prompt"])
            self.assertNotIn('"cloze"', request["prompt"])

    def test_registry_is_target_independent_and_templates_are_frozen(self):
        raw = REGISTRY.read_text(encoding="utf-8").casefold()
        for target_specific in (
            "stephen king",
            "carrie",
            "portland, maine",
            "richard bachman",
        ):
            self.assertNotIn(target_specific, raw)
        relations, _ = GENERATOR.load_relation_template_registry(REGISTRY)
        self.assertEqual(
            tuple(row["relation_id"] for row in relations),
            GENERATOR.ATOMIC_RELATION_IDS,
        )
        for relation in relations:
            direct = relation["direct_question_template"]
            cloze = relation["cloze_template"]
            self.assertEqual(direct.count("{subject}"), 1)
            self.assertTrue(direct.endswith("?"))
            self.assertEqual(cloze.count("{subject}"), 1)
            self.assertEqual(cloze.count("___"), 1)

    def test_v3_config_is_deterministic(self):
        config, _ = GENERATOR._load_generation_config(V3_CONFIG)
        self.assertEqual(
            config["decoding"],
            {
                "do_sample": False,
                "max_new_tokens": 192,
                "num_return_sequences": 1,
            },
        )
        self.assertNotIn("temperature", config["decoding"])
        self.assertNotIn("top_p", config["decoding"])

    def test_official_rwku_file_cannot_be_used_as_registry_or_resource(self):
        official = Path("/must/not/open/forget_level1.json")
        with mock.patch.object(
            Path,
            "open",
            side_effect=AssertionError("official RWKU file was opened"),
        ):
            with self.assertRaisesRegex(ValueError, "Official RWKU"):
                GENERATOR.load_relation_template_registry(official)
            with self.assertRaisesRegex(ValueError, "Official RWKU"):
                GENERATOR.validate_independent_resource_path(official)


class AtomicValidationTests(unittest.TestCase):
    def rejection_reason(self, candidate):
        accepted, rejected, diagnostics = extract_one(candidate)
        self.assertFalse(accepted)
        self.assertEqual(diagnostics["rejected_relation_count"], 1)
        self.assertEqual(len(rejected), 1)
        return rejected[0]["reason"]

    def test_status_unknown_is_normally_skipped(self):
        accepted, rejected, diagnostics = extract_one(
            atomic_candidate(status="unknown", answer="", evidence_sentence="")
        )
        self.assertFalse(accepted)
        self.assertFalse(rejected)
        self.assertEqual(diagnostics["unknown_relation_count"], 1)
        self.assertEqual(diagnostics["rejected_relation_count"], 0)

    def test_relation_mismatch_is_rejected(self):
        self.assertEqual(
            self.rejection_reason(atomic_candidate(relation_id="birth_place")),
            "relation_mismatch",
        )

    def test_subject_mismatch_is_rejected(self):
        self.assertEqual(
            self.rejection_reason(atomic_candidate(subject="Different Person")),
            "subject_mismatch",
        )

    def test_negative_and_uncertain_answers_are_rejected(self):
        for answer in (
            "no",
            "none",
            "unknown",
            "n/a",
            "not applicable",
            "false",
            "deceased=false",
            "uncertain",
            "possibly",
            "maybe",
            "allegedly",
            "unconfirmed",
            "unclear",
        ):
            with self.subTest(answer=answer):
                reason = self.rejection_reason(
                    atomic_candidate(
                        answer=answer,
                        evidence_sentence=f"Stephen King has value {answer}.",
                    )
                )
                self.assertIn(reason, {"negative_or_null_answer", "uncertain_answer"})

    def test_evidence_must_contain_subject_and_answer(self):
        self.assertEqual(
            self.rejection_reason(
                atomic_candidate(evidence_sentence="The date was September 21, 1947.")
            ),
            "evidence_missing_subject",
        )
        self.assertEqual(
            self.rejection_reason(
                atomic_candidate(
                    evidence_sentence="Stephen King has a documented date."
                )
            ),
            "evidence_missing_answer",
        )

    def test_surrounding_prose_is_not_treated_as_json(self):
        relations, registry_sha, requests = registry_and_requests(limit=1)
        raw = [
            {
                "prompt_index": 0,
                "generated_text": "Answer: " + json.dumps(atomic_candidate()),
            }
        ]
        accepted, rejected, _ = GENERATOR.extract_atomic_facts(
            raw,
            requests,
            relations,
            entity_id=ENTITY_ID,
            target_entity=SUBJECT,
            relation_registry_path=REGISTRY,
            relation_registry_sha256=registry_sha,
        )
        self.assertFalse(accepted)
        self.assertEqual(rejected[0]["reason"], "no_complete_json_object")


class ViewAndConsumerCompatibilityTests(unittest.TestCase):
    def compile_facts(self, *, duplicate=False):
        atomic_facts, rejected, _ = extract_one(atomic_candidate())
        self.assertFalse(rejected)
        if duplicate:
            copied = dict(atomic_facts[0])
            copied["output_index"] = 1
            copied["raw_output_sha256"] = "f" * 64
            atomic_facts.append(copied)
        relations, registry_sha, _ = registry_and_requests()
        return GENERATOR.compile_atomic_facts_to_entity_facts(
            atomic_facts,
            relations,
            relation_registry_path=REGISTRY,
            relation_registry_sha256=registry_sha,
        )

    def test_views_are_deterministic_and_structurally_valid(self):
        facts_a, _, _ = self.compile_facts()
        facts_b, _, _ = self.compile_facts()
        self.assertEqual(facts_a, facts_b)
        views = facts_a[0]["optimization_views"]
        direct = [row for row in views if row["prompt_style"] == "direct question"]
        cloze = [row for row in views if row["prompt_style"] == "cloze"]
        self.assertEqual(len(direct), 1)
        self.assertEqual(direct[0]["query"].count(SUBJECT), 1)
        self.assertTrue(direct[0]["query"].endswith("?"))
        self.assertEqual(len(cloze), 1)
        self.assertEqual(cloze[0]["query"].count(SUBJECT), 1)
        self.assertEqual(cloze[0]["query"].count("___"), 1)

    def test_duplicate_facts_do_not_overweight_zerounlearn(self):
        facts, _, duplicate_count = self.compile_facts(duplicate=True)
        self.assertEqual(len(facts), 1)
        self.assertEqual(duplicate_count, 1)
        views = facts[0]["optimization_views"]
        requests, audit = ZERO.compile_zero_unlearn_requests(
            views,
            FakeTokenizer(),
            target_subject=SUBJECT,
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(audit["selected_view_count"], 1)
        self.assertTrue(audit["one_request_per_fact_id"])

    def test_views_are_accepted_by_existing_setting5_compiler(self):
        facts, _, _ = self.compile_facts()
        views = facts[0]["optimization_views"]
        examples, by_view = SETTING5.setting5_entity_fact_examples(
            FakeTokenizer(),
            views,
            include_reverse=False,
        )
        self.assertEqual(len(examples), len(views))
        self.assertEqual(set(by_view), {view["view_id"] for view in views})
        self.assertTrue(all(example.target_true == "<eos>" for example in examples))


class ArtifactAndFailureTests(unittest.TestCase):
    def generated_outputs(self, requests, *, all_unknown=False):
        outputs = []
        for index, request in enumerate(requests):
            if index == 0 and not all_unknown:
                candidate = atomic_candidate(relation_id=request["relation_id"])
            else:
                candidate = atomic_candidate(
                    relation_id=request["relation_id"],
                    status="unknown",
                    answer="",
                    evidence_sentence="",
                )
            outputs.append(
                {
                    "prompt_index": index,
                    "sequence_index": 0,
                    "chat_template_used": True,
                    "rendered_prompt_sha256": request["prompt_sha256"],
                    "generated_text": json.dumps(candidate, separators=(",", ":")),
                }
            )
        return outputs

    def test_success_artifacts_feed_both_existing_consumers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_args(root)
            relations, _, requests = registry_and_requests()
            outputs = self.generated_outputs(requests)
            with mock.patch.object(
                GENERATOR,
                "_model_generate",
                return_value=(outputs, tokenizer_identity(len(requests))),
            ):
                receipt = GENERATOR.build_generated_corpus(args)
            for filename in (
                "generated_raw_corpus.json",
                "generated_atomic_facts.json",
                "generated_entity_fact_catalog.json",
                "generated_training_bundle.json",
                "generation_diagnostics.json",
                "generator_receipt.json",
            ):
                self.assertTrue((args.output_dir / filename).is_file(), filename)
            bundle = ACCESS.validate_artifact(
                json.loads(
                    (args.output_dir / "generated_training_bundle.json").read_text()
                )
            )
            views = bundle["payload"]["views"]
            zero_requests, audit = ZERO.compile_zero_unlearn_requests(
                views,
                FakeTokenizer(),
                target_subject=SUBJECT,
            )
            setting5_examples, _ = SETTING5.setting5_entity_fact_examples(
                FakeTokenizer(),
                views,
                include_reverse=False,
            )
            self.assertEqual(len(zero_requests), 1)
            self.assertTrue(audit["one_request_per_fact_id"])
            self.assertEqual(len(setting5_examples), len(views))
            self.assertEqual(receipt["requested_relation_count"], len(relations))
            self.assertEqual(receipt["known_relation_count"], 1)
            self.assertEqual(receipt["unknown_relation_count"], len(relations) - 1)
            self.assertEqual(receipt["accepted_fact_count"], 1)
            self.assertFalse(receipt["official_rwku_records_accessed"])
            self.assertEqual(len(receipt["output_identities"]), len(relations))
            for required in (
                "tokenizer_identity",
                "generation_configuration_sha256",
                "relation_template_registry_sha256",
                "implementation_sha256",
                "parser_implementation_revision",
                "raw_generated_corpus_sha256",
                "generated_atomic_facts_sha256",
                "final_entity_fact_bundle_sha256",
                "completed_at_utc",
            ):
                self.assertIn(required, receipt)

    def test_below_minimum_failure_persists_raw_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_args(root, minimum=1)
            _, _, requests = registry_and_requests()
            outputs = self.generated_outputs(requests, all_unknown=True)
            with mock.patch.object(
                GENERATOR,
                "_model_generate",
                return_value=(outputs, tokenizer_identity(len(requests))),
            ):
                with self.assertRaisesRegex(ValueError, "fewer facts"):
                    GENERATOR.build_generated_corpus(args)
            for filename in (
                "generated_raw_corpus.json",
                "generation_diagnostics.json",
                "generator_failure_receipt.json",
            ):
                self.assertTrue((args.output_dir / filename).is_file(), filename)
            self.assertFalse((args.output_dir / "generator_receipt.json").exists())
            failure = json.loads(
                (args.output_dir / "generator_failure_receipt.json").read_text()
            )
            self.assertEqual(failure["status"], "failed_below_minimum_accepted_facts")
            self.assertEqual(failure["known_relation_count"], 0)
            self.assertEqual(failure["unknown_relation_count"], len(requests))
            self.assertEqual(failure["rejected_relation_count"], 0)
            self.assertEqual(failure["accepted_fact_count"], 0)

    def test_v3_dry_run_requires_registry_and_loads_no_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_args(root)
            args.dry_run = True
            with mock.patch.object(
                GENERATOR,
                "_model_generate",
                side_effect=AssertionError("model was loaded"),
            ):
                result = GENERATOR.build_generated_corpus(args)
            self.assertEqual(result["status"], "dry_run_validated")
            args.relation_template_registry = None
            with self.assertRaisesRegex(ValueError, "relation-template-registry"):
                GENERATOR.build_generated_corpus(args)


if __name__ == "__main__":
    unittest.main()
