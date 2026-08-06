import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_rwku_entity_facts as FACTS  # noqa: E402
import build_rwku_generated_entity_corpus as GENERATED  # noqa: E402
import build_rwku_matched_protection as PROTECTION  # noqa: E402
import mcf_zero_unlearn_official_eval as MCF_EVAL  # noqa: E402
import rwku_artifact_access as ACCESS  # noqa: E402
import rwku_checkpoint_receipt as RECEIPT  # noqa: E402
import rwku_experiment as EXPERIMENT  # noqa: E402
import rwku_fact_sampler as SAMPLER  # noqa: E402
import rwku_repair as REPAIR  # noqa: E402


def row(query, answer, level, kind=None):
    return {
        "query": query,
        "answer": answer,
        "level": str(level),
        "type": kind or ("cloze" if str(level) == "1" else "simple question"),
        "subject": "Stephen King",
    }


def override_file(path, rows_and_relations):
    records = {}
    for record, relation, canonical, aliases in rows_and_relations:
        records[FACTS.record_sha256(record)] = {
            "relation_id": relation,
            "canonical_sensitive_answer": canonical,
            "sensitive_answer_aliases": aliases,
        }
    payload = {
        "schema_version": "rwku_fact_overrides_v1",
        "seed": 0,
        "entity_id": "rwku:1_Stephen_King",
        "subject": "Stephen King",
        "source_scope": ["forget_level1.json", "forget_level2.json"],
        "records": records,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class FakeTokenizer:
    eos_token = "<eos-runtime>"
    eos_token_id = 2
    all_special_ids = [0, 1, 2, 3]
    pad_token_id = 0
    bos_token_id = 1
    unk_token_id = 3

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [10 + ord(character) % 7 for character in str(text)]

    def __call__(self, text, add_special_tokens=False, **kwargs):
        if text == self.eos_token:
            return {"input_ids": [self.eos_token_id]}
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

    @staticmethod
    def decode(token_ids, **kwargs):
        return f"piece-{int(token_ids[0])}"

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        return messages[0]["content"]

    @staticmethod
    def convert_tokens_to_ids(token):
        return 3


class EntityFactAssignmentTests(unittest.TestCase):
    def _catalog(self):
        l1 = [
            row("Stephen King's debut novel was ___.", "Carrie", 1),
            row("Stephen King was born in ___, Maine.", "Portland", 1),
        ]
        l2 = [
            row("What was Stephen King's first published novel?", "Carrie", 2),
            row("In what state is Portland located?", "Maine", 2),
            row("What pseudonym did Stephen King use?", "Richard Bachman", 2),
        ]
        temporary = tempfile.TemporaryDirectory()
        path = Path(temporary.name) / "override.json"
        override_file(
            path,
            [
                (l1[0], "first_published_novel", "Carrie", []),
                (l1[1], "birth_city", "Portland", []),
                (l2[0], "first_published_novel", "Carrie", []),
                (l2[1], "birth_state", "Maine", []),
                (l2[2], "pseudonym", "Richard Bachman", ["Bachman"]),
            ],
        )
        facts, audit = FACTS.assign_records_to_facts(
            entity_id="rwku:1_Stephen_King",
            subject="Stephen King",
            level1=l1,
            level2=l2,
            source_hashes={"forget_level1.json": "a", "forget_level2.json": "b"},
            fact_overrides_path=path,
            strict=True,
        )
        return temporary, facts, audit

    def test_carrie_level1_and_level2_share_fact_id(self):
        temporary, facts, _ = self._catalog()
        self.addCleanup(temporary.cleanup)
        carrie = [fact for fact in facts if fact["canonical_sensitive_answer"] == "Carrie"]
        self.assertEqual(len(carrie), 1)
        self.assertEqual(len(carrie[0]["source_records"]), 2)

    def test_city_and_state_and_same_answer_different_relations_are_distinct(self):
        temporary, facts, _ = self._catalog()
        self.addCleanup(temporary.cleanup)
        identities = {(fact["relation_id"], fact["canonical_sensitive_answer"]): fact["fact_id"] for fact in facts}
        self.assertNotEqual(identities[("birth_city", "Portland")], identities[("birth_state", "Maine")])
        self.assertNotEqual(
            FACTS.entity_fact_id("e", "birth_city", "Maine"),
            FACTS.entity_fact_id("e", "birth_state", "Maine"),
        )
        same1 = row("Relation one?", "Shared", 1)
        same2 = row("Relation two?", "Shared", 2)
        with tempfile.TemporaryDirectory() as directory:
            path = override_file(
                Path(directory) / "same.json",
                [
                    (same1, "relation_one", "Shared", []),
                    (same2, "relation_two", "Shared", []),
                ],
            )
            assigned, _ = FACTS.assign_records_to_facts(
                entity_id="rwku:1_Stephen_King",
                subject="Stephen King",
                level1=[same1],
                level2=[same2],
                source_hashes={},
                fact_overrides_path=path,
                strict=True,
            )
            self.assertEqual(len(assigned), 2)

    def test_fact_id_is_sha256_entity_relation_normalized_answer(self):
        schema = json.loads(
            (ROOT / "config" / "rwku" / "entity_fact_schema_v1.json").read_text()
        )
        self.assertEqual(schema["$id"], FACTS.ENTITY_FACT_SCHEMA_VERSION)
        self.assertEqual(
            FACTS.entity_fact_id("entity", "relation", "  CARRIE "),
            FACTS.entity_fact_id("entity", "relation", "carrie"),
        )
        self.assertEqual(len(FACTS.entity_fact_id("e", "r", "a")), 64)

    def test_level3_is_structurally_rejected(self):
        level3 = row("attack", "answer", 3)
        with tempfile.TemporaryDirectory() as directory:
            path = override_file(
                Path(directory) / "override.json",
                [(level3, "attack", "answer", [])],
            )
            with self.assertRaisesRegex(FACTS.FactAuditError, "Level 3"):
                FACTS.assign_records_to_facts(
                    entity_id="rwku:1_Stephen_King",
                    subject="Stephen King",
                    level1=[level3],
                    level2=[],
                    source_hashes={},
                    fact_overrides_path=path,
                    strict=True,
                )

    def test_unresolved_and_conflicting_assignments_fail_closed(self):
        unknown = row("An intentionally ambiguous prompt", "x", 1)
        with self.assertRaisesRegex(FACTS.FactAuditError, "unresolved"):
            FACTS.assign_records_to_facts(
                entity_id="rwku:1_Stephen_King",
                subject="Stephen King",
                level1=[unknown],
                level2=[],
                source_hashes={},
                fact_overrides_path=None,
                strict=True,
            )

    def test_seed0_pinned_audit_has_fourteen_facts_and_no_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            result = FACTS.build_probe_artifacts(
                data_root=ROOT / "data" / "rwku",
                seed=0,
                output_dir=Path(directory),
                fact_overrides_path=ROOT / "config" / "rwku" / "fact_overrides" / "seed0.json",
                unseen_fact_fraction=0.25,
                prompt_holdout_per_seen_fact=1,
                split_seed=0,
                strict=True,
                allow_download=False,
            )
            audit = result["audit"]
            self.assertEqual(audit["level1_row_count"], 8)
            self.assertEqual(audit["level2_row_count"], 10)
            self.assertEqual(audit["semantic_fact_group_count"], 14)
            self.assertEqual(audit["ambiguous_unresolved_record_count"], 0)
            self.assertEqual(audit["calibration_fact_count"], 10)
            self.assertEqual(audit["unseen_fact_count"], 4)
            self.assertEqual(audit["seen_fact_unseen_prompt_view_count"], 2)
            self.assertEqual(audit["unseen_fact_view_count"], 6)


class SplitAndSamplerTests(unittest.TestCase):
    def _facts(self):
        facts = []
        for index, count in enumerate((3, 2, 1, 1, 1)):
            fact_id = f"fact-{index}"
            views = []
            for view_index in range(count):
                view = {
                    "query": f"q {index} {view_index}",
                    "canonical_sensitive_answer": f"answer-{index}",
                    "sensitive_answer_alias": f"answer-{index}",
                    "prompt_style": "direct question",
                    "view_content_sha256": f"hash-{index}-{view_index}",
                    "view_id": f"view-{index}-{view_index}",
                    "source_record_sha256": f"source-{index}-{view_index}",
                }
                views.append(view)
            facts.append(
                {
                    "fact_id": fact_id,
                    "relation_id": f"relation-{index}",
                    "entity_id": "entity",
                    "subject": "subject",
                    "sensitive_answer_aliases": [],
                    "optimization_views": [],
                    "held_out_views": [],
                    "_all_views": views,
                }
            )
        return facts

    def test_fact_split_is_disjoint_and_multiview_has_prompt_holdout(self):
        catalog, training, seen, unseen, split = FACTS.split_entity_facts(
            self._facts(), unseen_fact_fraction=0.4, prompt_holdout_per_seen_fact=1, split_seed=7
        )
        self.assertFalse({row["fact_id"] for row in training} & {row["fact_id"] for row in unseen})
        self.assertFalse({row["view_id"] for row in training} & {row["view_id"] for row in [*seen, *unseen]})
        for fact in catalog:
            if fact["partition"] == "calibration_fact" and len(fact["optimization_views"]) + len(fact["held_out_views"]) >= 2:
                self.assertTrue(fact["optimization_views"])
                self.assertTrue(fact["held_out_views"])
        self.assertNotEqual(
            split["seen_fact_unseen_prompt_view_count"],
            split["unseen_fact_view_count"],
        )

    def test_single_view_fact_is_never_shared(self):
        catalog, *_ = FACTS.split_entity_facts(
            self._facts(), unseen_fact_fraction=0.4, prompt_holdout_per_seen_fact=1, split_seed=3
        )
        for fact in catalog:
            if len(fact["optimization_views"]) + len(fact["held_out_views"]) == 1:
                self.assertFalse(fact["optimization_views"] and fact["held_out_views"])

    def test_balanced_fact_cycles_are_deterministic_and_account_every_dimension(self):
        views = {
            "a": [{"view_id": "a1", "prompt_style": "cloze", "answer": "Carrie"}],
            "b": [{"view_id": "b1", "prompt_style": "direct question", "answer": "Maine"}],
            "c": [{"view_id": "c1", "prompt_style": "forced-prefix", "answer": "Richard Bachman"}],
        }
        first = SAMPLER.build_fact_cycle_plan(views, steps=10, seed=4)
        second = SAMPLER.build_fact_cycle_plan(views, steps=10, seed=4)
        self.assertEqual(first, second)
        report = SAMPLER.exposure_report(first, seed=4, tokenizer=FakeTokenizer())
        self.assertLessEqual(report["exposure_imbalance"], 1)
        self.assertEqual(sum(report["fact_exposure_count"].values()), 10)
        self.assertTrue(report["view_exposure_count"])
        self.assertTrue(report["prompt_style_exposure_count"])
        self.assertTrue(report["sensitive_answer_alias_exposure_count"])
        self.assertTrue(report["answer_token_id_exposure_count"])
        self.assertTrue(report["decoded_token_piece_exposure_count"])

    def test_reverse_prompts_are_disabled_without_explicit_ablation(self):
        view = {
            "view_id": "v",
            "fact_id": "f",
            "query": "Which author wrote Carrie?",
            "canonical_sensitive_answer": "Stephen King",
            "sensitive_answer_alias": "Stephen King",
            "subject": "Carrie",
            "training_allowed": True,
            "boundary_expanding": True,
            "prompt_style": "relation-conditioned reverse",
        }
        with self.assertRaisesRegex(ValueError, "Boundary-expanding"):
            EXPERIMENT.setting5_entity_fact_examples(
                FakeTokenizer(), [view], include_reverse=False
            )


class ArtifactAccessTests(unittest.TestCase):
    def artifact(self, role):
        return ACCESS.make_artifact(
            role,
            {"records": []},
            protocol_label=ACCESS.PROBE_PROTOCOL_LABEL,
            protocol_status=ACCESS.PROBE_PROTOCOL_STATUS,
        )

    def test_artifact_role_matrix_rejects_every_forbidden_stage_or_use(self):
        training = self.artifact("training_bundle")
        gate = self.artifact("repair_selection_gate")
        evaluation = self.artifact("official_locked_eval")
        with self.assertRaises(ACCESS.ArtifactAccessError):
            ACCESS.assert_artifact_access(training, stage="evaluate")
        with self.assertRaises(ACCESS.ArtifactAccessError):
            ACCESS.assert_artifact_access(gate, stage="train", gradient=True)
        ACCESS.assert_artifact_access(gate, stage="train", selection=True)
        with self.assertRaises(ACCESS.ArtifactAccessError):
            ACCESS.assert_artifact_access(evaluation, stage="train")
        with self.assertRaises(ACCESS.ArtifactAccessError):
            ACCESS.assert_artifact_access(evaluation, stage="evaluate", gradient=True)

    def test_evaluation_mia_neighbors_utility_and_fluency_cannot_train_or_select(self):
        official = self.artifact("official_locked_eval")
        for gradient, selection in ((True, False), (False, True)):
            with self.assertRaises(ACCESS.ArtifactAccessError):
                ACCESS.assert_artifact_access(
                    official, stage="train", gradient=gradient, selection=selection
                )

    def test_target_only_training_rejects_relabeled_official_level_records(self):
        artifact = self.artifact("training_bundle")
        artifact["payload"] = {
            "views": [
                {
                    "training_allowed": True,
                    "source_file": "forget_level1.json",
                    "level": "1",
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "independently generated"):
            EXPERIMENT._validate_training_bundle_sources(
                artifact,
                training_source=EXPERIMENT.TRAINING_SOURCE_TARGET_ONLY,
            )

    def test_artifact_hash_detects_permission_or_payload_change(self):
        artifact = self.artifact("training_bundle")
        artifact["payload"]["records"].append({"leak": True})
        with self.assertRaisesRegex(ACCESS.ArtifactAccessError, "hash mismatch"):
            ACCESS.validate_artifact(artifact)

    def test_protocol_labels_are_exact(self):
        self.assertEqual(
            ACCESS.PROBE_PROTOCOL_STATUS,
            "nonofficial_probe_assisted_entity_fact_portability",
        )
        self.assertEqual(
            ACCESS.TARGET_ONLY_PROTOCOL_STATUS,
            "official_protocol_different_model_confirmatory_method_extension",
        )


class MappingAndProtectionTests(unittest.TestCase):
    def training_artifact(self, path):
        view = {
            "view_id": "v",
            "fact_id": "f",
            "relation_id": "first_published_novel",
            "entity_id": "rwku:1_Stephen_King",
            "query": "What novel?",
            "canonical_sensitive_answer": "Carrie",
            "sensitive_answer_alias": "Carrie",
            "sensitive_answer_aliases": [],
            "subject": "Stephen King",
            "training_allowed": True,
            "source_record_sha256": "a" * 64,
        }
        artifact = ACCESS.make_artifact(
            "training_bundle",
            {"views": [view]},
            protocol_label=ACCESS.PROBE_PROTOCOL_LABEL,
            protocol_status=ACCESS.PROBE_PROTOCOL_STATUS,
        )
        ACCESS.write_artifact(path, artifact)
        return artifact

    def test_setting5_and_original_zero_have_opposite_field_directions(self):
        view = {
            "view_id": "v",
            "fact_id": "f",
            "query": "What novel?",
            "canonical_sensitive_answer": "Carrie",
            "sensitive_answer_alias": "Carrie",
            "subject": "Stephen King",
            "training_allowed": True,
            "boundary_expanding": False,
            "prompt_style": "direct question",
        }
        examples, _ = EXPERIMENT.setting5_entity_fact_examples(
            FakeTokenizer(), [view], include_reverse=False
        )
        self.assertEqual(examples[0].answer.strip(), "Carrie")
        self.assertEqual(examples[0].target_new.strip(), "Carrie")
        self.assertEqual(examples[0].target_true, "<eos-runtime>")
        zero = EXPERIMENT.zerounlearn_forget_requests(
            FakeTokenizer(),
            [row("What novel did Stephen King write?", "Carrie", 2)],
            subject="Stephen King",
            seed=0,
        )
        self.assertEqual(zero[0]["target_true"]["str"], "Carrie")
        self.assertEqual(zero[0]["target_new"]["str"], "<eos-runtime>")

    def test_mcf_shaped_export_resolves_runtime_eos_and_mcf_evaluator_rejects_it(self):
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "training.json"
            artifact = self.training_artifact(bundle_path)
            export = Path(directory) / "export.json"
            FACTS.export_mcf_shaped_training_requests(artifact, destination=export)
            raw = json.loads(export.read_text())
            self.assertEqual(raw["requests"][0]["target_true"]["str"], FACTS.RUNTIME_EOS_MARKER)
            resolved = FACTS.load_mcf_shaped_training_requests(export, tokenizer=FakeTokenizer())
            self.assertEqual(resolved[0]["target_true"]["str"], "<eos-runtime>")
            with self.assertRaisesRegex(ValueError, "rejects"):
                MCF_EVAL.load_mcf(export)

    def test_protection_keys_require_visible_allowed_provenance(self):
        valid = {
            "key": "Carrie",
            "normalized_key": "carrie",
            "origin_type": "training_bundle",
            "origin_artifact_path": "/safe/training_bundle.json",
            "origin_artifact_sha256": "a" * 64,
            "source_fact_id": "f",
            "visible_before_freeze": True,
            "target_independent_vocabulary_revision": None,
        }
        PROTECTION.validate_key_provenance(valid)
        for mutation in (
            {"origin_type": "official_locked_eval"},
            {"visible_before_freeze": False},
            {"origin_artifact_path": "/x/unseen_fact_eval.json"},
        ):
            invalid = {**valid, **mutation}
            with self.assertRaises(ACCESS.ArtifactAccessError):
                PROTECTION.validate_key_provenance(invalid)

    def test_matched_protection_train_and_gate_are_content_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "training_bundle.json"
            self.training_artifact(bundle)
            source = root / "independent.json"
            source.write_text(
                json.dumps(
                    [
                        {"prompt": "An unrelated Carrie question one", "answer": "Carrie"},
                        {"prompt": "An unrelated Carrie question two", "answer": "Carrie"},
                    ]
                ),
                encoding="utf-8",
            )
            result = PROTECTION.build_matched_protection(
                training_bundle_path=bundle,
                source_corpora=[source],
                output_dir=root / "out",
                vocabulary_path=None,
                split_seed=0,
                minimum_train_per_key=1,
                minimum_gate_per_key=1,
                strict=True,
            )
            train = result["artifacts"]["matched_protection_train.json"]["payload"]["records"]
            gate = result["artifacts"]["matched_protection_gate.json"]["payload"]["records"]
            self.assertFalse(
                {item["content_sha256"] for item in train}
                & {item["content_sha256"] for item in gate}
            )


class RepairOutcomeTests(unittest.TestCase):
    @staticmethod
    def point(source, token_index, token_id, target, competitor):
        return REPAIR.RepairPoint(
            hidden=torch.zeros(2),
            target_logit=torch.tensor(float(target)),
            competitor_logit=torch.tensor(float(competitor)),
            source_id=source,
            token_index=token_index,
            target_token_id=token_id,
            competitor_token_id=99,
            baseline_predicted_token_id=token_id,
        )

    def test_multitoken_answer_has_token_view_and_fact_outcomes(self):
        before = [
            self.point("v", 0, 10, 1.0, 0.9),
            self.point("v", 1, 11, 1.0, 0.9),
        ]
        after = [
            self.point("v", 0, 10, 0.5, 1.0),
            self.point("v", 1, 11, 1.0, 0.9),
        ]
        report = REPAIR.hierarchical_repair_outcomes(
            tokenizer=FakeTokenizer(),
            calibration_rows=[{"view_id": "v", "fact_id": "f", "answer": "Richard Bachman"}],
            before_points=before,
            after_points=after,
            selected_ids=[10],
            overlap_ids=[11],
            special_ids=[0, 1, 2, 3],
            selected_scale=0.5,
            config=REPAIR.RepairConfig(active_margin=0.25, selection_margin=0.05),
        )
        token_outcomes = report["token_position_outcomes"]
        self.assertEqual(token_outcomes[0]["protection_classification"], "safe_sparse_head_pair")
        self.assertEqual(token_outcomes[1]["protection_classification"], "shared_protected_answer_pair")
        self.assertEqual(report["view_outcomes"][0]["support_outcome"], "partially_supported")
        self.assertEqual(report["fact_outcomes"][0]["fact_outcome"], "unresolved_after_repair")
        self.assertNotIn("resolved_by_setting5e", json.dumps(report))

    def test_setting5_resolution_uses_calibration_qualified_label(self):
        point = self.point("v", 0, 10, 0.0, 1.0)
        report = REPAIR.hierarchical_repair_outcomes(
            tokenizer=FakeTokenizer(),
            calibration_rows=[{"view_id": "v", "fact_id": "f", "answer": "x"}],
            before_points=[point],
            after_points=[point],
            selected_ids=[],
            overlap_ids=[],
            special_ids=[],
            selected_scale=0.0,
            config=REPAIR.RepairConfig(),
        )
        self.assertEqual(
            report["token_position_outcomes"][0]["token_outcome"],
            "calibration_resolved_by_setting5e",
        )


class CheckpointReceiptTests(unittest.TestCase):
    def make_receipt(self, root):
        checkpoint = root / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "weights.bin").write_bytes(b"weights")
        training_path = root / "training_bundle.json"
        artifact = ACCESS.make_artifact(
            "training_bundle",
            {"views": []},
            protocol_label=ACCESS.PROBE_PROTOCOL_LABEL,
            protocol_status=ACCESS.PROBE_PROTOCOL_STATUS,
        )
        ACCESS.write_artifact(training_path, artifact)
        implementation = root / "implementation.py"
        implementation.write_text("x = 1\n", encoding="utf-8")
        locked_path = root / "official_locked_eval.json"
        locked = ACCESS.make_artifact(
            "official_locked_eval",
            {"files": {}},
            protocol_label=ACCESS.PROBE_PROTOCOL_LABEL,
            protocol_status=ACCESS.PROBE_PROTOCOL_STATUS,
        )
        ACCESS.write_artifact(locked_path, locked)
        destination = root / "receipt.json"
        RECEIPT.create_checkpoint_receipt(
            destination=destination,
            experiment_id="experiment",
            protocol_label=ACCESS.PROBE_PROTOCOL_LABEL,
            protocol_status=ACCESS.PROBE_PROTOCOL_STATUS,
            target_entity="Stephen King",
            target_entity_id="rwku:1_Stephen_King",
            base_model_identity={"path": "/model", "id": "model"},
            base_model_revision="revision",
            tokenizer_identity={"id": "tokenizer"},
            checkpoint_paths=[checkpoint],
            training_bundle_path=training_path,
            optimization_protection_path=None,
            mcf_retain_optimization_paths=[],
            mcf_repair_gate_paths=[],
            matched_protection_train_path=None,
            matched_protection_gate_path=None,
            method_configuration={"steps": 600},
            implementation_files=[implementation],
            sampler_provenance={"exposure_imbalance": 0},
            generator_receipt_path=None,
            official_locked_eval_path=locked_path,
            confirmatory=False,
        )
        return destination, checkpoint, training_path

    def test_receipt_detects_checkpoint_and_training_bundle_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, checkpoint, _ = self.make_receipt(root)
            (checkpoint / "weights.bin").write_bytes(b"changed")
            with self.assertRaisesRegex(RECEIPT.CheckpointReceiptError, "checkpoint"):
                RECEIPT.open_official_evaluation(receipt, experiment_id="experiment")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, _, training = self.make_receipt(root)
            training.write_text(training.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(RECEIPT.CheckpointReceiptError, "training_bundle"):
                RECEIPT.open_official_evaluation(receipt, experiment_id="experiment")

    def test_receipt_atomically_opens_once_and_blocks_post_eval_modification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, _, _ = self.make_receipt(root)
            opened = RECEIPT.open_official_evaluation(receipt, experiment_id="experiment")
            self.assertTrue(opened["official_evaluation_opened"])
            self.assertEqual(opened["state"], "OFFICIAL_EVALUATION_OPENED")
            with self.assertRaises(RECEIPT.CheckpointReceiptError):
                RECEIPT.open_official_evaluation(receipt, experiment_id="experiment")
            with self.assertRaisesRegex(RECEIPT.CheckpointReceiptError, "new experiment ID"):
                RECEIPT.assert_model_modification_allowed(receipt, experiment_id="experiment")

    def test_evaluation_refuses_unfrozen_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt, _, _ = self.make_receipt(root)
            value = json.loads(receipt.read_text())
            value["state"] = "TRAINING"
            value["receipt_sha256"] = RECEIPT._receipt_digest(value)
            receipt.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(RECEIPT.CheckpointReceiptError, "CHECKPOINT_FROZEN"):
                RECEIPT.open_official_evaluation(receipt, experiment_id="experiment")


class GeneratedCorpusTests(unittest.TestCase):
    def test_generated_fact_provenance_and_deduplication(self):
        text = json.dumps(
            {
                "subject": "Stephen King",
                "relation_id": "first_novel",
                "answer": "Carrie",
                "direct_question": "What was Stephen King's first novel?",
                "cloze": "Stephen King's first novel was ___.",
            }
        )
        facts, rejected = GENERATED.extract_generated_facts(
            [{"generated_text": text}, {"generated_text": text}],
            entity_id="rwku:1_Stephen_King",
            target_entity="Stephen King",
        )
        self.assertEqual(len(facts), 1)
        self.assertEqual(len(facts[0]["optimization_views"]), 3)
        self.assertFalse(rejected)
        self.assertEqual(facts[0]["protocol_status"], ACCESS.TARGET_ONLY_PROTOCOL_STATUS)

    def test_generator_dry_run_does_not_import_torch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "build_rwku_generated_entity_corpus.py"),
                    "--target-entity",
                    "Stephen King",
                    "--entity-id",
                    "rwku:1_Stephen_King",
                    "--generator-model",
                    "/local/model",
                    "--generator-revision",
                    "abc123",
                    "--generation-config",
                    str(ROOT / "config" / "rwku" / "generation" / "llama32_3b_target_corpus_v1.json"),
                    "--seed",
                    "0",
                    "--output-dir",
                    str(output),
                    "--dry-run",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            receipt = json.loads((output / "generator_dry_run.json").read_text())
            self.assertFalse(receipt["torch_imported"])
            self.assertFalse(receipt["official_rwku_records_accessed"])
            for required in (
                "generator_model_identifier",
                "generator_model_revision",
                "prompt_template_sha256",
                "decoding_parameters",
                "fact_extractor_revision_sha256",
            ):
                self.assertIn(required, receipt)

    def test_target_only_prepare_never_calls_full_official_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated_training_bundle.json"
            receipt = root / "generator_receipt.json"
            generated.write_text("{}", encoding="utf-8")
            receipt.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                experiment_id="target-only",
                output_root=root / "runs",
                seed=0,
                training_source=EXPERIMENT.TRAINING_SOURCE_TARGET_ONLY,
                generated_entity_fact_bundle=generated,
                generator_receipt=receipt,
                confirmatory=True,
            )
            with mock.patch.object(EXPERIMENT, "ensure_target_data", side_effect=AssertionError("official opened")):
                EXPERIMENT._prepare_staged(args)
            state = json.loads((root / "runs" / "target-only" / "experiment_state.json").read_text())
            self.assertFalse(state["prepare_audit"]["official_level1_level2_opened"])
            locked = json.loads(
                (root / "runs" / "target-only" / "official_locked_eval.json").read_text()
            )
            self.assertTrue(
                {
                    "forget_level1.json",
                    "forget_level2.json",
                    "forget_level3.json",
                    "forget_mia.json",
                    "neighbor_level1.json",
                    "retain_mmlu.json",
                    "fluency.json",
                }
                <= set(locked["payload"]["files"])
            )


class ProtocolMetadataTests(unittest.TestCase):
    def test_registry_marks_only_repaired_default_as_evaluation_conditioned(self):
        registry = json.loads((ROOT / "config" / "official_benchmarks" / "registry.json").read_text())
        tracks = {track["id"]: track for track in registry["tracks"]}
        for track_id in ("mcf_zerounlearn_official", "zsre_zerounlearn_official", "tofu_forget05"):
            statuses = tracks[track_id]["protocol_status_by_method"]
            self.assertEqual(statuses["default"], ACCESS.EVALUATION_CONDITIONED_REPAIR_STATUS)
            self.assertNotEqual(statuses["Base"], ACCESS.EVALUATION_CONDITIONED_REPAIR_STATUS)
            self.assertNotEqual(statuses["Setting 5e without repair"], ACCESS.EVALUATION_CONDITIONED_REPAIR_STATUS)


if __name__ == "__main__":
    unittest.main()
