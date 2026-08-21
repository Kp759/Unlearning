from __future__ import annotations

import importlib.util
import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load_module("prepare_sure_wikipedia_corpus", "prepare_sure_wikipedia_corpus.py")
contexts = load_module(
    "build_sure_mcf_external_contexts", "build_sure_mcf_external_contexts.py"
)
v8 = load_module(
    "sure_mcf_target_aware_direct_only_v9_test",
    "sure_mcf_target_aware_direct_only.py",
)
compare = load_module(
    "compare_mcf_sure_utility_ladder", "compare_mcf_sure_utility_ladder.py"
)


def training_record(case_id: int, subject: str, true: str, new: str):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": "{} works as",
            "subject": subject,
            "target_true": {"str": true},
            "target_new": {"str": new},
        },
        "data_role": "target_aware_direct_training_only",
    }


class ExternalContextsV9Tests(unittest.TestCase):
    def setUp(self):
        self.training = [
            training_record(11, "Forget Person", "writer", "singer"),
            training_record(12, "Second Person", "chemist", "actor"),
        ]

    def test_real_wikipedia_collector_requires_titles_and_deduplicates(self):
        source = [
            {"id": "1", "title": " Alpha  ", "text": "A" * 20},
            {"id": "1", "title": "Alpha duplicate", "text": "duplicate"},
            {"id": "2", "title": "", "text": "missing title"},
            {"id": "3", "title": "Beta", "text": "useful text"},
        ]
        selected, seen = prepare.collect_articles(
            source, sample_size=2, max_chars=10
        )
        self.assertEqual(seen, 4)
        self.assertEqual([row["title"] for row in selected], ["Alpha", "Beta"])
        self.assertEqual(selected[0]["text"], "A" * 10)

    def test_context_builder_uses_only_generated_and_external_roles(self):
        wikipedia = [
            {"title": f"External Entity {index}", "text": f"Lead text {index}."}
            for index in range(40)
        ]
        external = contexts.eligible_external_records(
            wikipedia,
            self.training,
            document_limit=40,
            exclude_first=0,
            seed=1,
            lead_chars=64,
        )
        generated = contexts.build_subject_contexts(self.training)
        locality = contexts.build_locality_contexts(
            self.training, external, contexts_per_record=8, seed=1
        )
        self.assertEqual(
            len(generated), len(self.training) * len(contexts.SUBJECT_CONTEXT_TEMPLATES)
        )
        self.assertEqual(len(locality), 16)
        self.assertEqual(len({row["prompt_text"] for row in locality}), 16)
        self.assertTrue(
            all(row["prompt_kind"] == "generated_subject" for row in generated)
        )
        serialized = json.dumps({"generated": generated, "locality": locality})
        for forbidden in (
            "paraphrase_prompts",
            "neighborhood_prompts",
            "attribute_prompts",
            "generation_prompts",
        ):
            self.assertNotIn(forbidden, serialized)
        self.assertTrue(
            all("External Entity" in row["external_title"] for row in locality)
        )

    def test_gfs_recovery_profile_pairs_answer_cued_subject_and_locality_views(self):
        wikipedia = [
            {"title": f"External Entity {index}", "text": f"Lead text {index}."}
            for index in range(40)
        ]
        external = contexts.eligible_external_records(
            wikipedia,
            self.training,
            document_limit=40,
            exclude_first=0,
            seed=1,
            lead_chars=64,
        )
        profile = contexts.GFS_RECOVERY_CONTEXT_PROFILE
        generated = contexts.build_subject_contexts(self.training, profile=profile)
        locality = contexts.build_locality_contexts(
            self.training,
            external,
            contexts_per_record=len(contexts.PAIRED_ANSWER_CUE_TEMPLATES),
            seed=1,
            profile=profile,
        )
        first_generated = [
            row for row in generated if row["source_record_position"] == 0
        ]
        first_locality = [
            row for row in locality if row["source_record_position"] == 0
        ]
        self.assertEqual(len(first_generated), 4)
        self.assertEqual(len(first_locality), 4)
        self.assertEqual(
            {row["prompt_text"].rsplit("\n", 1)[-1] for row in first_generated},
            {"Answer:", "Short answer:", "Completion:", "Missing value:"},
        )
        for generated_row, locality_row in zip(first_generated, first_locality):
            generated_shape = generated_row["prompt_text"].replace(
                "Forget Person", "<SUBJECT>"
            )
            locality_shape = locality_row["prompt_text"].replace(
                locality_row["external_title"], "<SUBJECT>"
            )
            self.assertEqual(generated_shape, locality_shape)
            self.assertEqual(generated_row["context_profile"], profile)
            self.assertEqual(locality_row["context_profile"], profile)

    def test_bundle_loader_binds_stripped_view_and_zero_probe_boundary(self):
        generated = contexts.build_subject_contexts(self.training)
        external = [
            {"document_index": index, "title": f"Entity {index}", "lead": "Lead"}
            for index in range(20)
        ]
        locality = contexts.build_locality_contexts(
            self.training, external, contexts_per_record=2, seed=1
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_path = root / "training.json"
            training_path.write_text(json.dumps(self.training), encoding="utf-8")
            payload = {
                "schema_version": 1,
                "protocol": contexts.PROTOCOL,
                "training_visible_sha256": v8.joint.sha256_bytes(
                    training_path.read_bytes()
                ),
                "generated_subject_contexts": generated,
                "external_locality_contexts": locality,
                "data_boundary": {
                    "source_counterfact_path_accepted": False,
                    "official_paraphrases_read": 0,
                    "official_neighborhoods_read": 0,
                    "benchmark_retain_examples_read": 0,
                    "generation_probes_read": 0,
                },
            }
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps(payload), encoding="utf-8")
            loaded_generated, loaded_locality, _ = v8.load_external_context_bundle(
                bundle,
                training_path=training_path,
                training_records=self.training,
            )
            self.assertEqual(len(loaded_generated), len(generated))
            self.assertEqual(len(loaded_locality), len(locality))

            payload["data_boundary"]["official_neighborhoods_read"] = 1
            bundle.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "data boundary"):
                v8.load_external_context_bundle(
                    bundle,
                    training_path=training_path,
                    training_records=self.training,
                )

    def test_bundle_loader_rejects_tampered_paired_view(self):
        profile = contexts.GFS_RECOVERY_CONTEXT_PROFILE
        generated = contexts.build_subject_contexts(self.training, profile=profile)
        external = [
            {"document_index": index, "title": f"Entity {index}", "lead": "Lead"}
            for index in range(20)
        ]
        locality = contexts.build_locality_contexts(
            self.training,
            external,
            contexts_per_record=4,
            seed=1,
            profile=profile,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_path = root / "training.json"
            training_path.write_text(json.dumps(self.training), encoding="utf-8")
            payload = {
                "schema_version": 1,
                "protocol": contexts.PROTOCOL,
                "training_visible_sha256": v8.joint.sha256_bytes(
                    training_path.read_bytes()
                ),
                "builder": {
                    "context_profile": profile,
                    "paired_subject_locality_geometry": True,
                },
                "generated_subject_contexts": generated,
                "external_locality_contexts": locality,
                "data_boundary": {
                    "source_counterfact_path_accepted": False,
                    "official_paraphrases_read": 0,
                    "official_neighborhoods_read": 0,
                    "benchmark_retain_examples_read": 0,
                    "generation_probes_read": 0,
                },
            }
            payload["external_locality_contexts"][0]["prompt_text"] += " tampered"
            bundle = root / "bundle.json"
            bundle.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "locked profile"):
                v8.load_external_context_bundle(
                    bundle,
                    training_path=training_path,
                    training_records=self.training,
                )

    def test_generated_failures_are_hard_training_constraints(self):
        prompts = [
            {
                "case_id": 11,
                "source_record_position": 0,
                "prompt_kind": "direct",
                "prompt_index": 0,
            },
            {
                "case_id": 11,
                "source_record_position": 0,
                "prompt_kind": "generated_subject",
                "prompt_index": 0,
            },
        ]
        report = v8.joint.grouped_pairwise_report(
            torch.tensor([1.0, -0.1]), prompts, required_margin=0.01
        )
        report.update({"utility_safe": True, "locality_safe": True})
        self.assertEqual(report["FS"], 100.0)
        self.assertEqual(report["generated_subject_FS"], 0.0)
        self.assertEqual(report["generated_subject_margin_failures"], 1)
        self.assertFalse(v8.direct_candidate_feasible(report))

    def test_exact_solver_enforces_separate_locality_guard(self):
        selected_ids = [0, 1]
        prompts = [
            {
                "case_id": 11,
                "source_record_position": 0,
                "prompt_kind": "direct",
                "prompt_index": 0,
            }
        ]
        hidden = torch.ones(1, 1)
        positions = torch.tensor([0])
        true_cache = v8.exact.build_sequence_cache(
            torch.tensor([[2.0, 0.0, 0.0]]),
            hidden,
            torch.tensor([0]),
            positions,
            selected_ids,
            record_count=1,
            device=torch.device("cpu"),
        )
        reference_cache = v8.exact.build_sequence_cache(
            torch.tensor([[0.0, 2.0, 0.0]]),
            hidden,
            torch.tensor([1]),
            positions,
            selected_ids,
            record_count=1,
            device=torch.device("cpu"),
        )
        args = argparse.Namespace(
            stage2_residual_l2_weight=1e-4,
            stage2_locality_kl_weight=1.0,
            utility_kl_mean_budget=10.0,
            utility_kl_p95_budget=10.0,
            utility_kl_max_budget=10.0,
            locality_kl_mean_budget=10.0,
            locality_kl_p95_budget=10.0,
            locality_kl_max_budget=10.0,
            max_total_delta_norm=10.0,
            stage2_constraint_tolerance=1e-5,
            required_pairwise_margin=0.01,
            stage2_maxiter=100,
            stage2_ftol=1e-9,
        )
        _, _, report = v8.joint.solve_residual(
            args,
            rank=1,
            solver_target=0.2,
            row_bases=[torch.ones(1, 1), torch.ones(1, 1)],
            active_ids=selected_ids,
            selected_ids=selected_ids,
            stage1_delta=torch.zeros(2, 1),
            true_cache=true_cache,
            reference_cache=reference_cache,
            prompt_records=prompts,
            utility_hidden=torch.zeros(3, 1),
            utility_probabilities=torch.full((3, 2), 0.1),
            locality_hidden=torch.zeros(3, 1),
            locality_probabilities=torch.full((3, 2), 0.1),
        )
        self.assertTrue(report["continuous_feasible"])
        self.assertLess(report["locality_kl_mean"], 1e-6)

    def test_scaling_runner_keeps_augmentation_disabled(self):
        script = (SCRIPTS / "run_mcf_sure_utility_scaling.sh").read_text()
        self.assertIn("SURE_SCALING_DOCS:-1000 10000", script)
        self.assertIn("SURE_ENABLE_EXTERNAL_CONTEXTS=0", script)
        augmented = (SCRIPTS / "run_mcf_sure_v9_aug.sh").read_text()
        self.assertIn("SURE_V9_WIKIPEDIA_DOCS:-10000", augmented)
        self.assertIn("SURE_ENABLE_EXTERNAL_CONTEXTS=1", augmented)
        recovery = (SCRIPTS / "run_mcf_sure_v9_gfs_recovery.sh").read_text()
        self.assertIn('SURE_EXTERNAL_CONTEXT_PROFILE="paired_answer_cue_v1"', recovery)
        self.assertIn("mcf_sure_v9_gfs_recovery_w${DOCS}", recovery)

    def test_ladder_comparison_reads_v8_and_exact_retain_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            learner = root / "target_aware_direct_only_learner"
            learner.mkdir()
            (root / "final_target_true_sensitive_eval.json").write_text(
                json.dumps(
                    {
                        "metrics": {
                            "FS": {"mean": 100.0},
                            "GFS": {"mean": 79.0},
                            "Spe_margin": {"mean": -5.38},
                            "Spe_success": {"mean": 34.2},
                            "PPL": 11.06,
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "posthoc_exact_retain_kl.json").write_text(
                json.dumps(
                    {
                        "selected_row_count": 70,
                        "exact_kl_base_to_edited": {
                            "mean": 0.54,
                            "p95": 3.26,
                            "max": 12.87,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (learner / "config_used.json").write_text(
                json.dumps(
                    {
                        "protocol": v8.PROTOCOL,
                        "GFS_checkpoint_selection": False,
                        "neighborhood_prompts_used_for_training_or_selection": False,
                        "utility_cache_metadata": {
                            "actual_document_sample_size": 180,
                            "actual_utility_prompt_count": 100000,
                        },
                        "final": {"utility_kl_mean": 3e-8},
                    }
                ),
                encoding="utf-8",
            )
            row = compare.load_run("baseline", root)
        self.assertEqual(row["FS"], 100.0)
        self.assertEqual(row["wikipedia_documents"], 180)
        self.assertEqual(row["exact_retain_kl_mean"], 0.54)
        self.assertFalse(row["official_GFS_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
