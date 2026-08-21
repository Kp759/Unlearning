import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_rwku_entity_facts as ENTITY_FACTS  # noqa: E402
import build_sure_wikipedia_stats as WIKIPEDIA  # noqa: E402
import rwku_artifact_access as ACCESS  # noqa: E402
import rwku_experiment as EXPERIMENT  # noqa: E402
import rwku_sure_head_only_w1k as HEAD_ONLY  # noqa: E402


CONFIGURATION_PATH = ROOT / "config" / "rwku" / "sure_head_only_w1k_seed0.json"
RUNNER_PATH = ROOT / "scripts" / "run_rwku_sure_head_only_w1k.sh"


class FakeTokenizer:
    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt):
        if tokenize is not False or add_generation_prompt is not True:
            raise AssertionError("Unexpected chat-template arguments")
        return f"<user>{messages[0]['content']}</user><assistant>"


class TokenFilterTokenizer:
    pieces = {1: " the", 2: " Shining", 3: " It", 4: " Unknown", 5: ","}

    def decode(self, token_ids):
        return self.pieces[int(token_ids[0])]


def make_view(fact_index, style, *, boundary_expanding=False, subject="Stephen King"):
    canonical = f"Answer {fact_index} detail"
    direct = f"What is atomic fact {fact_index} about {subject}?"
    if style == "direct question":
        query = direct
        alias = canonical
    elif style == "cloze":
        query = f"Atomic fact {fact_index} about {subject} is ___"
        alias = canonical
    elif style == "deterministic paraphrase":
        query = f"In different words, answer this question: {direct}"
        alias = canonical
    elif style == "forced-prefix":
        query = f"{direct}\nAnswer prefix: Answer {fact_index}"
        alias = "detail"
    else:
        raise AssertionError(style)
    fact_id = hashlib.sha256(f"fact:{fact_index}".encode()).hexdigest()
    view = {
        "schema_version": ENTITY_FACTS.ENTITY_FACT_SCHEMA_VERSION,
        "query": query,
        "level": "generated",
        "query_type": style,
        "prompt_style": style,
        "canonical_sensitive_answer": canonical,
        "sensitive_answer_alias": alias,
        "source_record_sha256": hashlib.sha256(
            f"source:{fact_index}".encode()
        ).hexdigest(),
        "source_record_sha256_values": [
            hashlib.sha256(f"source:{fact_index}".encode()).hexdigest()
        ],
        "source_file": "generated_raw_corpus.json",
        "source_row_index": fact_index,
        "boundary_expanding": boundary_expanding,
        "fact_id": fact_id,
        "relation_id": f"relation_{fact_index}",
        "entity_id": "rwku:1_Stephen_King",
        "subject": subject,
        "training_allowed": True,
    }
    view["view_content_sha256"] = ENTITY_FACTS.view_content_sha256(view)
    view["view_id"] = hashlib.sha256(
        f"{fact_id}:{view['view_content_sha256']}".encode()
    ).hexdigest()
    return view


def valid_views():
    styles = (
        "direct question",
        "cloze",
        "deterministic paraphrase",
        "forced-prefix",
    )
    return [make_view(fact_index, style) for fact_index in range(8) for style in styles]


def write_atomic_artifacts(root, views):
    metadata = {
        "entity_id": "rwku:1_Stephen_King",
        "subject": "Stephen King",
        "seed": 0,
        "generation_configuration_id": "llama32_3b_target_corpus_v3_atomic_facts",
    }
    bundle = ACCESS.make_artifact(
        "training_bundle",
        {"views": views},
        protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=ACCESS.TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    bundle_path = root / "generated_training_bundle.json"
    ACCESS.write_artifact(bundle_path, bundle)
    receipt_payload = {
        "status": "complete",
        "target_entity": "Stephen King",
        "entity_id": "rwku:1_Stephen_King",
        "protocol_label": ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": ACCESS.TARGET_ONLY_PROTOCOL_STATUS,
        "official_rwku_records_accessed": False,
        "fact_extractor_implementation": "atomic_relation_fact_extractor_v1",
        "parser_implementation_revision": "complete_json_object_atomic_v1",
        "random_seeds": [0],
        "extraction_configuration": {"reverse_prompts_enabled": False},
        "final_entity_fact_bundle_sha256": bundle["sha256"],
        "accepted_fact_count": 8,
    }
    receipt = ACCESS.make_artifact(
        "generator_receipt",
        receipt_payload,
        protocol_label=ACCESS.TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=ACCESS.TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    receipt_path = root / "generator_receipt.json"
    ACCESS.write_artifact(receipt_path, receipt)
    return bundle_path, receipt_path


class LockedConfigurationTests(unittest.TestCase):
    def test_configuration_is_exactly_the_single_requested_experiment(self):
        config = HEAD_ONLY.load_locked_configuration(CONFIGURATION_PATH)
        self.assertEqual(config["seed"], 0)
        self.assertEqual(config["target_entity"], "Stephen King")
        self.assertEqual(config["utility"]["document_count"], 1000)
        self.assertEqual(config["stage1"]["rank"], 4)
        self.assertTrue(config["development_only"])
        self.assertFalse(
            config["data_boundary"]["official_rwku_records_available_to_learner"]
        )

    def test_learner_cli_has_no_official_rwku_input(self):
        args = HEAD_ONLY.parse_args(
            [
                "--model-path",
                "model",
                "--training-bundle",
                "bundle",
                "--generator-receipt",
                "receipt",
                "--utility-cache",
                "cache",
                "--output-root",
                "outputs",
            ]
        )
        for forbidden in (
            "data_root",
            "forget_level1",
            "forget_level2",
            "forget_level3",
            "neighbor",
            "mia",
            "wikidata_dir",
        ):
            self.assertFalse(hasattr(args, forbidden), forbidden)

    def test_changed_w1k_or_rank_lock_is_rejected(self):
        raw = json.loads(CONFIGURATION_PATH.read_text())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changed.json"
            raw["utility"]["document_count"] = 10000
            path.write_text(json.dumps(raw), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "document count"):
                HEAD_ONLY.load_locked_configuration(path)

    def test_launcher_builds_w1k_then_opens_official_data_only_after_freeze(self):
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("--sample-size 1000", source)
        self.assertIn("--require-min-prompts 90000", source)
        self.assertIn('--exclude-casefold-substring "Stephen King"', source)
        prepare = source.index("--stage prepare")
        learner = source.index("scripts/rwku_sure_head_only_w1k.py")
        evaluate = source.index("--stage evaluate")
        self.assertLess(prepare, learner)
        self.assertLess(learner, evaluate)
        learner_invocation = source[learner:evaluate]
        self.assertNotIn("--data-root", learner_invocation)
        self.assertNotIn("--wikidata-dir", learner_invocation)


class AtomicBundleTests(unittest.TestCase):
    def setUp(self):
        self.configuration = HEAD_ONLY.load_locked_configuration(CONFIGURATION_PATH)

    def test_atomic_bundle_loads_and_compiles_all_method_visible_views(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, receipt = write_atomic_artifacts(root, valid_views())
            views, audit, generator = HEAD_ONLY.load_atomic_bundle(
                bundle, receipt, self.configuration
            )
        self.assertEqual(audit["fact_count"], 8)
        self.assertEqual(audit["view_count"], 32)
        self.assertEqual(generator["accepted_fact_count"], 8)
        prompts = HEAD_ONLY.compile_prompt_records(
            views, FakeTokenizer(), neutral_target="Unknown"
        )
        self.assertEqual(sum(row["prompt_kind"] == "direct" for row in prompts), 8)
        self.assertEqual(
            sum(row["prompt_kind"] == "generated_subject" for row in prompts), 24
        )
        self.assertTrue(
            all(
                row["requested_rewrite"]["target_reference"]["str"] == "Unknown"
                for row in prompts
            )
        )
        forced = [row for row in prompts if row["prompt_style"] == "forced-prefix"]
        self.assertTrue(forced)
        self.assertTrue(
            all(
                row["requested_rewrite"]["target_sensitive"]["str"] == "detail"
                for row in forced
            )
        )

    def test_boundary_expansion_wrong_subject_and_duplicate_ids_fail_closed(self):
        cases = []
        boundary = valid_views()
        boundary[0] = make_view(0, "direct question", boundary_expanding=True)
        cases.append((boundary, "Boundary-expanding"))
        wrong_subject = valid_views()
        wrong_subject[0] = make_view(0, "direct question", subject="Different Person")
        cases.append((wrong_subject, "subject"))
        duplicate = valid_views()
        duplicate[1] = dict(duplicate[0])
        cases.append((duplicate, "Duplicate atomic view ID"))
        for views, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    HEAD_ONLY.validate_atomic_views(views, self.configuration)

    def test_generator_receipt_must_hash_the_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle, receipt = write_atomic_artifacts(root, valid_views())
            value = json.loads(receipt.read_text())
            value["payload"]["final_entity_fact_bundle_sha256"] = "0" * 64
            value["sha256"] = ACCESS.sha256_json(
                {key: item for key, item in value.items() if key != "sha256"}
            )
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not identify"):
                HEAD_ONLY.load_atomic_bundle(bundle, receipt, self.configuration)

    def test_atomic_generator_must_be_the_same_local_base_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            receipt = {
                "generator_model_path": str(model),
                "generator_model_revision": "pinned-revision",
                "local_snapshot_identity": {"exists": True},
            }
            audit = HEAD_ONLY.validate_generator_base_model(receipt, str(model))
            self.assertTrue(audit["same_local_snapshot"])
            other = root / "other-model"
            other.mkdir()
            with self.assertRaisesRegex(ValueError, "differs"):
                HEAD_ONLY.validate_generator_base_model(receipt, str(other))


class WikipediaAndFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.configuration = HEAD_ONLY.load_locked_configuration(CONFIGURATION_PATH)
        self.metadata = {
            "requested_document_sample_size": 1000,
            "actual_document_sample_size": 1000,
            "requested_utility_prompt_count": 100000,
            "actual_utility_prompt_count": 95000,
            "required_minimum_utility_prompt_count": 90000,
            "excluded_prefix_document_count": 20,
            "excluded_casefold_substrings": ["stephen king"],
            "target_substring_excluded_document_count": 3,
            "target_substring_excluded_document_indices_sha256": "a" * 64,
            "corpus_receipt": {"protocol": "sure_external_wikipedia_corpus_v1"},
        }

    def test_w1k_metadata_requires_scale_protocol_and_target_exclusion(self):
        audit = HEAD_ONLY.validate_w1k_utility_metadata(
            self.metadata, self.configuration
        )
        self.assertEqual(audit["document_count"], 1000)
        self.assertEqual(audit["target_matching_documents_rejected"], 3)
        for key, value, message in (
            ("actual_document_sample_size", 999, "actual_document"),
            ("actual_utility_prompt_count", 89999, "predictor count"),
            ("excluded_casefold_substrings", [], "exclude Stephen King"),
        ):
            changed = dict(self.metadata)
            changed[key] = value
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, message):
                    HEAD_ONLY.validate_w1k_utility_metadata(changed, self.configuration)

    def test_wikipedia_filter_normalizes_case_and_whitespace(self):
        eligible, rejected = WIKIPEDIA.eligible_document_indices(
            [
                "PPL reserved",
                "ordinary article",
                "Biography of STEPHEN\nKING and his work",
                "another article",
            ],
            exclude_first=1,
            excluded_casefold_substrings=["Stephen King"],
        )
        self.assertEqual(eligible, [1, 3])
        self.assertEqual(rejected, [2])

    def test_locked_gate_needs_both_atomic_view_groups_and_utility(self):
        report = {
            "direct_prompt_count": 8,
            "generated_subject_prompt_count": 24,
            "FS": 100.0,
            "generated_subject_FS": 100.0,
            "direct_margin_failures": 0,
            "generated_subject_margin_failures": 0,
            "utility_safe": True,
        }
        self.assertTrue(HEAD_ONLY.atomic_candidate_feasible(report))
        for changed_key in (
            "FS",
            "generated_subject_FS",
            "utility_safe",
        ):
            changed = dict(report)
            changed[changed_key] = False
            self.assertFalse(HEAD_ONLY.atomic_candidate_feasible(changed))

    def test_selected_rows_filter_function_words_but_keep_content_and_neutral(self):
        cases = [
            SimpleNamespace(record_position=0),
            SimpleNamespace(record_position=0),
            SimpleNamespace(record_position=1),
            SimpleNamespace(record_position=1),
        ]
        selected, report = HEAD_ONLY.select_edit_row_ids(
            TokenFilterTokenizer(),
            cases,
            torch.tensor([1, 2, 3, 5]),
            torch.tensor([4]),
            prompt_count=2,
            configuration=self.configuration,
        )
        self.assertEqual(selected, [2, 3, 4])
        self.assertEqual(report["rejected_sensitive_row_ids"], [1, 5])
        self.assertEqual(report["minimum_content_rows_per_prompt"], 1)


class StagedEvaluationLabelTests(unittest.TestCase):
    def test_frozen_adapter_method_label_is_used_with_legacy_fallback(self):
        self.assertEqual(
            EXPERIMENT.staged_candidate_method(
                {"method_configuration": {"method": "SURE-RWKU-H-W1K"}}
            ),
            "SURE-RWKU-H-W1K",
        )
        self.assertEqual(
            EXPERIMENT.staged_candidate_method({}), EXPERIMENT.METHOD_REPAIRED
        )


if __name__ == "__main__":
    unittest.main()
