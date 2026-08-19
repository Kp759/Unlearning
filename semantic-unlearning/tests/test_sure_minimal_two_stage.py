from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


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


split = load_module("sure_minimal_split", "build_sure_minimal_split.py")
wiki = load_module("sure_minimal_wiki", "build_sure_wikipedia_stats.py")
learner = load_module("sure_minimal_learner", "sure_minimal_two_stage.py")


class MinimalSureSplitTests(unittest.TestCase):
    def test_mcf_training_record_contains_only_original_sensitive_answer(self):
        raw = {
            "requested_rewrite": {
                "prompt": "{} works as",
                "subject": "Person X",
                "target_true": {"str": "politician"},
                "target_new": {"str": "actor"},
            },
            "paraphrase_prompts": ["Person X's job is"],
            "neighborhood_prompts": ["Person Y works as"],
        }
        visible = split.mcf_direct_sensitive_record(raw, 17)
        rewrite = visible["requested_rewrite"]
        self.assertEqual(rewrite["target_new"]["str"], "politician")
        self.assertNotIn("target_true", rewrite)
        self.assertNotIn("actor", json.dumps(visible))
        split.assert_training_view_locked("mcf", [visible])

    def test_zsre_training_record_has_no_neutral_or_replacement_target(self):
        raw = {
            "src": "Who is Ada's employer?",
            "subject": "Ada",
            "answers": ["Example Labs"],
        }
        visible = split.zsre_direct_sensitive_record(raw, 9)
        rewrite = visible["requested_rewrite"]
        self.assertEqual(rewrite["target_true"]["str"], "Example Labs")
        self.assertNotIn("target_new", rewrite)
        split.assert_training_view_locked("zsre", [visible])

    def test_post_training_retain_copy_is_prompt_only(self):
        raw = {
            "requested_rewrite": {
                "prompt": "{} works as",
                "subject": "Person X",
                "target_true": {"str": "politician"},
                "target_new": {"str": "actor"},
            }
        }
        audit = split.prompt_only_retain_record("mcf", raw, 3)
        self.assertEqual(audit["data_role"], "post_training_exact_kl_audit_only")
        self.assertEqual(set(audit["requested_rewrite"]), {"prompt", "subject"})

    def test_locked_loader_accepts_no_retain_and_rejects_reference(self):
        record = {
            "case_id": 7,
            "requested_rewrite": {
                "prompt": "{} works as",
                "subject": "Person X",
                "target_new": {"str": "politician"},
            },
            "paraphrase_prompts": [],
            "neighborhood_prompts": [],
            "attribute_prompts": [],
            "generation_prompts": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            forget = root / "forget.json"
            manifest = root / "manifest.json"

            def write_payload(row):
                payload = json.dumps([row]).encode("utf-8")
                forget.write_bytes(payload)
                manifest.write_text(
                    json.dumps(
                        {
                            "protocol": learner.PROTOCOL,
                            "dataset": "mcf",
                            "seed": 1,
                            "training_visible_forget_sha256": hashlib.sha256(
                                payload
                            ).hexdigest(),
                            "sampling": {
                                "forget_num": 1,
                                "benchmark_retain_train_num": 0,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            args = argparse.Namespace(
                training_visible_path=str(forget),
                split_manifest=str(manifest),
                forget_num=1,
                dataset="mcf",
                seed=1,
            )
            write_payload(record)
            records, _ = learner.load_locked(args)
            self.assertEqual(len(records), 1)

            leaked = json.loads(json.dumps(record))
            leaked["requested_rewrite"]["target_true"] = {"str": "actor"}
            write_payload(leaked)
            with self.assertRaisesRegex(RuntimeError, "forbidden target_true"):
                learner.load_locked(args)


class MinimalSureWikipediaTests(unittest.TestCase):
    def test_predictor_mask_excludes_padding_and_each_last_token(self):
        attention = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]])
        mask = wiki.predictor_mask(attention)
        self.assertEqual(
            mask.tolist(),
            [
                [True, True, False, False],
                [True, False, False, False],
                [False, False, False, False],
            ],
        )

    def test_second_moment_is_uncentered_mean_outer_product(self):
        states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        unnormalized = states.transpose(0, 1) @ states
        moment, trace = wiki.finalize_second_moment(unnormalized, 2)
        expected = unnormalized / 2.0
        self.assertTrue(torch.allclose(moment, expected))
        self.assertAlmostEqual(trace, float(torch.trace(expected)))

    def test_cache_builder_uses_all_predictor_states_from_sampled_documents(self):
        class ToyTokenizer:
            padding_side = "left"

            def __call__(self, texts, **_kwargs):
                rows = [[int(value) for value in text.split()] for text in texts]
                width = max(len(row) for row in rows)
                ids = torch.zeros((len(rows), width), dtype=torch.long)
                mask = torch.zeros_like(ids)
                for index, row in enumerate(rows):
                    ids[index, : len(row)] = torch.tensor(row)
                    mask[index, : len(row)] = 1
                return {"input_ids": ids, "attention_mask": mask}

        class ToyBackbone(nn.Module):
            def forward(self, input_ids, attention_mask, **_kwargs):
                del attention_mask
                hidden = torch.stack(
                    (input_ids.float(), torch.ones_like(input_ids).float()),
                    dim=-1,
                )
                return argparse.Namespace(last_hidden_state=hidden)

        class ToyModel(nn.Module):
            base_model_prefix = "backbone"

            def __init__(self):
                super().__init__()
                self.backbone = ToyBackbone()
                self.output = nn.Linear(2, 5, bias=False)

            def get_output_embeddings(self):
                return self.output

        tokenizer = ToyTokenizer()
        moment, report = wiki.build_second_moment(
            ToyModel(),
            tokenizer,
            ["1 2 3", "4 5"],
            document_order=[0, 1],
            device=torch.device("cpu"),
            max_length=8,
            batch_size=2,
        )
        states = torch.tensor([[1.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
        expected = states.transpose(0, 1) @ states / 3.0
        self.assertTrue(torch.allclose(moment, expected))
        self.assertEqual(report["predictor_hidden_state_count"], 3)
        self.assertEqual(report["documents_forwarded"], 2)
        self.assertEqual(tokenizer.padding_side, "left")

    def test_cache_validation_locks_sample_model_and_no_benchmark_data(self):
        moment = torch.eye(3)
        metadata = {
            "protocol": wiki.UTILITY_PROTOCOL,
            "predictor_hidden_state_count": 7,
            "requested_document_sample_size": 100_000,
            "actual_document_sample_size": 3,
            "hidden_size": 3,
            "model_probe_sha256": "model-probe",
            "tokenizer_probe_sha256": "tokenizer-probe",
            "benchmark_examples_seen": 0,
            "benchmark_retain_examples_seen": 0,
            "heldout_benchmark_probes_seen": 0,
            "excluded_prefix_document_count": 20,
            "second_moment_sha256": wiki.sha256_tensor(moment),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "utility.pt"
            torch.save({"second_moment": moment, "metadata": metadata}, path)
            loaded, report = learner.load_utility_cache(
                path,
                expected_sample_size=100_000,
                expected_hidden_size=3,
                expected_model_probe="model-probe",
                expected_tokenizer_probe="tokenizer-probe",
            )
            self.assertTrue(torch.equal(loaded, moment))
            self.assertEqual(report["benchmark_retain_examples_seen"], 0)
            with self.assertRaisesRegex(ValueError, "different model weights"):
                learner.load_utility_cache(
                    path,
                    expected_sample_size=100_000,
                    expected_hidden_size=3,
                    expected_model_probe="wrong-model",
                    expected_tokenizer_probe="tokenizer-probe",
                )


class MinimalSureLearnerTests(unittest.TestCase):
    def test_candidate_materialization_uses_weight_dtype_and_restores_rows(self):
        layer = nn.Linear(3, 5, bias=False, dtype=torch.bfloat16)
        before = layer.weight.detach().clone()
        delta = torch.tensor([[0.12345, -0.25, 0.5]], dtype=torch.float32)
        with learner.temporary_materialized_output_delta(layer, [2], delta):
            expected = before[2] + delta[0].to(torch.bfloat16)
            self.assertTrue(torch.equal(layer.weight[2], expected))
            self.assertTrue(torch.equal(layer.weight[0], before[0]))
        self.assertTrue(torch.equal(layer.weight, before))

    def test_both_stages_are_locked_to_rank_two(self):
        self.assertEqual(learner.FIXED_CONTRASTIVE_RANK, 2)
        hidden = torch.eye(4)
        tids = torch.tensor([5, 5, 7, 7])
        bases, reports = learner.build_contrastive_bases_from_second_moment(
            hidden,
            tids,
            torch.eye(4),
            requested_ids=[5, 7],
            rank_cap=2,
            relative_eps=1e-3,
        )
        self.assertEqual([tuple(basis.shape) for basis in bases], [(2, 4), (2, 4)])
        for basis in bases:
            self.assertTrue(
                torch.allclose(
                    basis @ basis.transpose(0, 1),
                    torch.eye(2),
                    atol=1e-5,
                )
            )
        self.assertTrue(
            all(report["requested_contrastive_rank"] == 2 for report in reports)
        )
        with self.assertRaisesRegex(ValueError, "fixes both stages at rank 2"):
            learner.build_contrastive_bases_from_second_moment(
                hidden,
                tids,
                torch.eye(4),
                requested_ids=[5],
                rank_cap=4,
                relative_eps=1e-3,
            )

    def test_stage1_chooses_smallest_successful_scale(self):
        reports = [
            {"scale": 1.0, "direct_failures": 0, "constraint_shortfall_sum": 0},
            {"scale": 0.5, "direct_failures": 0, "constraint_shortfall_sum": 0},
            {"scale": 0.25, "direct_failures": 1, "constraint_shortfall_sum": 1},
        ]
        selected, mode = learner.choose_stage1_report(reports)
        self.assertEqual(selected["scale"], 0.5)
        self.assertEqual(mode, "complete")

    def test_stage1_handoff_uses_only_direct_failure_and_shortfall(self):
        reports = [
            {"scale": 1.0, "direct_failures": 2, "constraint_shortfall_sum": 1.0},
            {"scale": 0.5, "direct_failures": 1, "constraint_shortfall_sum": 3.0},
            {"scale": 0.25, "direct_failures": 1, "constraint_shortfall_sum": 2.0},
        ]
        selected, mode = learner.choose_stage1_report(reports)
        self.assertEqual(selected["scale"], 0.25)
        self.assertEqual(mode, "residual_handoff")

    def test_mcf_and_zsre_runners_share_architecture_defaults(self):
        mcf = (SCRIPTS / "run_mcf_sure_minimal.sh").read_text(encoding="utf-8")
        zsre = (SCRIPTS / "run_zsre_sure_minimal.sh").read_text(encoding="utf-8")
        shared = [
            "UTILITY_SAMPLE_SIZE=100000",
            "UTILITY_SEED=1",
            "UTILITY_EXCLUDE_FIRST=20",
            'STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"',
            'STAGE1_LR="${SURE_STAGE1_LR:-0.005}"',
            'STAGE2_STEPS="${SURE_STAGE2_STEPS:-500}"',
            'STAGE2_LR="${SURE_STAGE2_LR:-0.005}"',
            'GA_WEIGHT="${SURE_GA_WEIGHT:-2.0}"',
            'GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"',
            'MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.05}"',
            'MIN_NLL="${SURE_MIN_SENSITIVE_NLL_INCREASE:-4.0}"',
        ]
        for line in shared:
            self.assertIn(line, mcf)
            self.assertIn(line, zsre)
        for forbidden in ("SURE_RETAIN_TRAIN_NUM", "SURE_STAGE2_RANKS"):
            self.assertNotIn(forbidden, mcf)
            self.assertNotIn(forbidden, zsre)


if __name__ == "__main__":
    unittest.main()
