from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

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


split = load_module(
    "build_mcf_sure_target_aware_direct_split",
    "build_mcf_sure_target_aware_direct_split.py",
)
v8 = load_module(
    "sure_mcf_target_aware_direct_only",
    "sure_mcf_target_aware_direct_only.py",
)


class TargetAwareDirectOnlyV8Tests(unittest.TestCase):
    def sample_source_record(self):
        return {
            "requested_rewrite": {
                "prompt": "{} works as",
                "subject": "Person X",
                "target_true": {"str": "politician", "id": "Q1"},
                "target_new": {"str": "actor", "id": "Q2"},
                "relation_id": "P106",
            },
            "paraphrase_prompts": ["Person X's profession is"],
            "neighborhood_prompts": ["Person Y works as"],
            "attribute_prompts": ["Tell me about Person X"],
            "generation_prompts": ["Person X"],
        }

    def sample_visible_record(self):
        return split.target_aware_direct_record(self.sample_source_record(), 17)

    def manifest(self, payload: bytes):
        return {
            "protocol": split.PROTOCOL,
            "dataset": "mcf",
            "seed": 1,
            "source_sha256": "source-hash-only",
            "training_visible_target_aware_direct_sha256": hashlib.sha256(
                payload
            ).hexdigest(),
            "sampling": {
                "forget_num": 1,
                "benchmark_retain_train_num": 0,
                "forget_case_ids": [17],
            },
            "learner_adapter_contract": {
                "sensitive_answer_field": "target_true",
                "reference_answer_field": "target_new",
                "direct_only": True,
                "official_paraphrases_visible_to_learner": False,
                "forbidden_probe_fields": list(split.PROBE_FIELDS),
            },
            "data_roles": {"GFS_checkpoint_selection": False},
        }

    def test_split_keeps_both_targets_but_removes_every_probe(self):
        visible = self.sample_visible_record()
        self.assertEqual(set(visible), {"case_id", "requested_rewrite", "data_role"})
        self.assertEqual(
            set(visible["requested_rewrite"]),
            {"prompt", "subject", "target_true", "target_new"},
        )
        self.assertEqual(
            visible["requested_rewrite"]["target_true"]["str"], "politician"
        )
        self.assertEqual(visible["requested_rewrite"]["target_new"]["str"], "actor")
        text = json.dumps(visible)
        self.assertNotIn("profession is", text)
        self.assertNotIn("Person Y", text)
        split.assert_direct_only_training_view([visible])

    def test_split_rejects_even_an_empty_probe_field(self):
        visible = self.sample_visible_record()
        visible["paraphrase_prompts"] = []
        with self.assertRaisesRegex(AssertionError, "probe fields"):
            split.assert_direct_only_training_view([visible])

    def test_builder_uses_official_half_split_and_writes_no_source_path(self):
        raw = []
        for index in range(4):
            record = self.sample_source_record()
            record["requested_rewrite"]["subject"] = f"Person {index}"
            raw.append(record)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "mcf.json"
            output = root / "protocol"
            source.write_text(json.dumps(raw), encoding="utf-8")
            argv = [
                "builder",
                "--mcf-path",
                str(source),
                "--output-dir",
                str(output),
                "--seed",
                "1",
                "--forget-num",
                "1",
                "--retain-eval-num",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv):
                split.main()
            visible = json.loads((output / split.TRAINING_FILENAME).read_text())
            manifest = json.loads((output / "split_manifest.json").read_text())
        self.assertIn(visible[0]["case_id"], {2, 3})
        self.assertIn(manifest["sampling"]["retain_case_ids"][0], {0, 1})
        self.assertNotIn("source_dataset", manifest)
        split.assert_direct_only_training_view(visible)

    def test_locked_loader_builds_only_one_direct_prompt(self):
        visible = self.sample_visible_record()
        payload = (json.dumps([visible]) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / split.TRAINING_FILENAME
            manifest = root / "split_manifest.json"
            training.write_bytes(payload)
            manifest.write_text(json.dumps(self.manifest(payload)), encoding="utf-8")
            records, prompts, loaded = v8.load_locked_direct_records(
                training,
                manifest,
                expected_seed=1,
                expected_forget_num=1,
            )
        self.assertEqual(records, [visible])
        self.assertEqual(len(prompts), 1)
        self.assertEqual(prompts[0]["prompt_kind"], "direct")
        self.assertEqual(prompts[0]["prompt_text"], "Person X works as")
        self.assertNotIn("source_dataset", loaded)

    def test_locked_loader_rejects_hash_mismatch_and_source_path(self):
        visible = self.sample_visible_record()
        payload = (json.dumps([visible]) + "\n").encode("utf-8")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training = root / split.TRAINING_FILENAME
            manifest_path = root / "split_manifest.json"
            training.write_bytes(payload)
            manifest = self.manifest(payload)
            manifest["training_visible_target_aware_direct_sha256"] = "wrong"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash"):
                v8.load_locked_direct_records(
                    training,
                    manifest_path,
                    expected_seed=1,
                    expected_forget_num=1,
                )
            manifest = self.manifest(payload)
            manifest["source_dataset"] = "/path/to/raw/mcf.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source-dataset path"):
                v8.load_locked_direct_records(
                    training,
                    manifest_path,
                    expected_seed=1,
                    expected_forget_num=1,
                )

    def test_direct_report_leaves_gfs_uncomputed_and_counts_ties_as_failure(self):
        prompt = {
            "case_id": 17,
            "source_record_position": 0,
            "prompt_kind": "direct",
            "prompt_index": 0,
        }
        report = v8.joint.grouped_pairwise_report(
            torch.tensor([0.0]), [prompt], required_margin=0.01
        )
        self.assertEqual(report["FS"], 0.0)
        self.assertIsNone(report["GFS"])
        self.assertEqual(report["direct_failures"], 1)
        self.assertEqual(report["paraphrase_prompt_count"], 0)
        self.assertEqual(report["paraphrase_margin_failures"], 0)
        report["utility_safe"] = True
        self.assertFalse(v8.direct_candidate_feasible(report))

    def test_exact_solver_supports_direct_only_hard_constraints(self):
        selected_ids = [0, 1]
        prompt_records = [
            {
                "case_id": 1,
                "source_record_position": 0,
                "prompt_kind": "direct",
                "prompt_index": 0,
            }
        ]
        hidden = torch.ones(1, 1)
        position = torch.tensor([0])
        true_cache = v8.exact.build_sequence_cache(
            torch.tensor([[2.0, 0.0, 0.0]]),
            hidden,
            torch.tensor([0]),
            position,
            selected_ids,
            record_count=1,
            device=torch.device("cpu"),
        )
        reference_cache = v8.exact.build_sequence_cache(
            torch.tensor([[0.0, 2.0, 0.0]]),
            hidden,
            torch.tensor([1]),
            position,
            selected_ids,
            record_count=1,
            device=torch.device("cpu"),
        )
        args = argparse.Namespace(
            stage2_residual_l2_weight=1e-4,
            utility_kl_mean_budget=10.0,
            utility_kl_p95_budget=10.0,
            utility_kl_max_budget=10.0,
            max_total_delta_norm=10.0,
            stage2_constraint_tolerance=1e-5,
            required_pairwise_margin=0.01,
            stage2_maxiter=100,
            stage2_ftol=1e-9,
        )
        residual, _, report = v8.joint.solve_residual(
            args,
            rank=1,
            solver_target=0.2,
            row_bases=[torch.ones(1, 1), torch.ones(1, 1)],
            active_ids=selected_ids,
            selected_ids=selected_ids,
            stage1_delta=torch.zeros(2, 1),
            true_cache=true_cache,
            reference_cache=reference_cache,
            prompt_records=prompt_records,
            utility_hidden=torch.zeros(3, 1),
            utility_probabilities=torch.full((3, 2), 0.1),
        )
        separation = v8.exact.exact_pairwise_separation(
            true_cache, reference_cache, residual
        )
        self.assertTrue(report["continuous_feasible"])
        self.assertIsNone(report["GFS"])
        self.assertGreaterEqual(float(separation.min()), 0.2 - 1e-5)

    def test_runner_keeps_paraphrases_post_checkpoint_and_never_gates_gfs(self):
        runner = (SCRIPTS / "run_mcf_sure_target_aware_direct_only.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("build_mcf_sure_target_aware_direct_split.py", runner)
        learner_start = runner.index(
            "python scripts/sure_mcf_target_aware_direct_only.py"
        )
        learner_end = runner.index("FINAL_MODEL=", learner_start)
        learner_command = runner[learner_start:learner_end]
        self.assertIn('--training-visible-path "${TRAINING_VISIBLE}"', learner_command)
        self.assertNotIn("--mcf-path", learner_command)
        self.assertNotIn("paraphrase", learner_command.lower())
        self.assertIn("--require-min-fs 100", runner)
        self.assertNotIn("--require-min-gfs", runner)
        first_official_eval = runner.index(
            "python scripts/mcf_zero_unlearn_official_eval.py"
        )
        self.assertGreater(first_official_eval, learner_end)


if __name__ == "__main__":
    unittest.main()
