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
                            "learner_adapter_contract": {
                                "sensitive_answer_field": "target_new",
                                "forbidden_answer_fields": ["target_true"],
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

    def test_locked_loader_accepts_a_new_dataset_adapter_without_core_changes(self):
        record = {
            "case_id": 1,
            "requested_rewrite": {
                "prompt": "{} was authored by",
                "subject": "Example",
                "sensitive_answer": {"str": "Private Author"},
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
            payload = json.dumps([record]).encode("utf-8")
            forget.write_bytes(payload)
            manifest.write_text(
                json.dumps(
                    {
                        "protocol": learner.PROTOCOL,
                        "dataset": "custom-qa",
                        "seed": 7,
                        "training_visible_forget_sha256": hashlib.sha256(
                            payload
                        ).hexdigest(),
                        "sampling": {
                            "forget_num": 1,
                            "benchmark_retain_train_num": 0,
                        },
                        "learner_adapter_contract": {
                            "sensitive_answer_field": "sensitive_answer",
                            "forbidden_answer_fields": ["replacement_answer"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                training_visible_path=str(forget),
                split_manifest=str(manifest),
                forget_num=1,
                dataset="custom-qa",
                seed=7,
            )
            loaded, loaded_manifest = learner.load_locked(args)
            self.assertEqual(loaded, [record])
            self.assertEqual(
                learner.adapter_contract(loaded_manifest)["sensitive_answer_field"],
                "sensitive_answer",
            )


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

    def test_base_partition_cache_matches_output_head(self):
        class ToyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.output = nn.Linear(2, 4, bias=False)

            def get_output_embeddings(self):
                return self.output

        model = ToyModel()
        hidden = torch.tensor([[1.0, -1.0], [0.5, 2.0]])
        actual = wiki.base_logsumexp_for_hidden(
            model,
            hidden,
            device=torch.device("cpu"),
            batch_size=1,
        )
        expected = torch.logsumexp(model.output(hidden).float(), dim=-1)
        self.assertTrue(torch.allclose(actual, expected))

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
        moment, utility_hidden, report = wiki.build_second_moment(
            ToyModel(),
            tokenizer,
            ["1 2 3", "4 5"],
            document_order=[0, 1],
            utility_prompt_document_indices=[0, 1],
            utility_seed=1,
            device=torch.device("cpu"),
            max_length=8,
            batch_size=2,
        )
        states = torch.tensor([[1.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
        expected = states.transpose(0, 1) @ states / 3.0
        self.assertTrue(torch.allclose(moment, expected))
        self.assertEqual(report["predictor_hidden_state_count"], 3)
        self.assertEqual(report["documents_forwarded"], 2)
        self.assertEqual(tuple(utility_hidden.shape), (2, 2))
        self.assertEqual(report["utility_prompt_count"], 2)
        self.assertEqual(tokenizer.padding_side, "left")

    def test_cache_validation_locks_sample_model_and_no_benchmark_data(self):
        moment = torch.eye(3)
        hidden = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        logsumexp = torch.tensor([4.0, 5.0])
        metadata = {
            "protocol": wiki.UTILITY_PROTOCOL,
            "predictor_hidden_state_count": 7,
            "requested_document_sample_size": 100_000,
            "actual_document_sample_size": 3,
            "requested_utility_prompt_count": 8_192,
            "actual_utility_prompt_count": 2,
            "hidden_size": 3,
            "model_probe_sha256": "model-probe",
            "tokenizer_probe_sha256": "tokenizer-probe",
            "benchmark_examples_seen": 0,
            "benchmark_retain_examples_seen": 0,
            "heldout_benchmark_probes_seen": 0,
            "excluded_prefix_document_count": 20,
            "second_moment_sha256": wiki.sha256_tensor(moment),
            "utility_hidden_sha256": wiki.sha256_tensor(hidden),
            "base_logsumexp_sha256": wiki.sha256_tensor(logsumexp),
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "utility.pt"
            torch.save(
                {
                    "second_moment": moment,
                    "utility_hidden_states": hidden,
                    "base_logsumexp": logsumexp,
                    "metadata": metadata,
                },
                path,
            )
            loaded, loaded_hidden, loaded_logsumexp, report = learner.load_utility_cache(
                path,
                expected_sample_size=100_000,
                expected_prompt_count=8_192,
                expected_hidden_size=3,
                expected_model_probe="model-probe",
                expected_tokenizer_probe="tokenizer-probe",
            )
            self.assertTrue(torch.equal(loaded, moment))
            self.assertTrue(torch.equal(loaded_hidden, hidden))
            self.assertTrue(torch.equal(loaded_logsumexp, logsumexp))
            self.assertEqual(report["benchmark_retain_examples_seen"], 0)
            with self.assertRaisesRegex(ValueError, "different model weights"):
                learner.load_utility_cache(
                    path,
                    expected_sample_size=100_000,
                    expected_prompt_count=8_192,
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

    def test_rank_ladder_is_locked_and_full_space_basis_uses_inverse_utility(self):
        self.assertEqual(learner.RANK_LADDER, (2, 4))
        hidden = torch.tensor([[1.0, 1.0]])
        tids = torch.tensor([5])
        utility = torch.diag(torch.tensor([1.0, 4.0]))
        bases, reports = learner.build_contrastive_bases_from_second_moment(
            hidden,
            tids,
            utility,
            requested_ids=[5],
            rank_cap=2,
            relative_eps=1e-6,
        )
        direction = bases[0][0].abs()
        self.assertAlmostEqual(float(direction[1] / direction[0]), 0.25, places=3)
        self.assertEqual(reports[0]["domain"], "full_lm_head_hidden_space")
        learner.build_contrastive_bases_from_second_moment(
            hidden,
            tids,
            utility,
            requested_ids=[5],
            rank_cap=4,
            relative_eps=1e-6,
        )
        with self.assertRaisesRegex(ValueError, "shared ladder"):
            learner.build_contrastive_bases_from_second_moment(
                hidden,
                tids,
                utility,
                requested_ids=[5],
                rank_cap=3,
                relative_eps=1e-6,
            )

    def test_exact_sparse_utility_kl_matches_full_softmax(self):
        probabilities = torch.tensor([[0.2, 0.3]])
        shifts = torch.tensor([[0.7, -0.2]])
        actual = learner.exact_sparse_kl_from_shifts(shifts, probabilities)[0]
        base = torch.log(torch.tensor([0.2, 0.3, 0.5]))
        edited = base + torch.tensor([0.7, -0.2, 0.0])
        base_p = torch.softmax(base, dim=0)
        expected = (
            base_p * (torch.log_softmax(base, dim=0) - torch.log_softmax(edited, dim=0))
        ).sum()
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        zero = learner.exact_sparse_kl_from_shifts(
            torch.zeros_like(shifts), probabilities
        )
        self.assertTrue(torch.allclose(zero, torch.zeros_like(zero), atol=1e-7))

    def test_dataset_independent_cache_recovers_selected_base_probabilities(self):
        layer = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
                )
            )
        hidden = torch.tensor([[0.5, -0.25], [1.0, 2.0]])
        logits = layer(hidden)
        logsumexp = torch.logsumexp(logits, dim=-1)
        actual = learner.selected_base_probabilities(
            layer,
            [1, 3],
            hidden,
            logsumexp,
            device=torch.device("cpu"),
            batch_size=1,
        )
        expected = torch.softmax(logits, dim=-1)[:, [1, 3]]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_constraint_gated_ga_is_zero_after_both_guards_pass(self):
        base = torch.tensor([[0.0, 2.0, 1.0]])
        tids = torch.tensor([0])
        passing = torch.tensor([[-5.0, 2.0, 1.0]], requires_grad=True)
        loss, diagnostics = learner.bounded_direct_constraint_loss(
            passing,
            base,
            tids,
            required_logit_margin=0.05,
            required_nll_increase=4.0,
        )
        self.assertEqual(float(loss), 0.0)
        self.assertEqual(float(diagnostics["active_fraction"]), 0.0)
        failing, diagnostics = learner.bounded_direct_constraint_loss(
            base.clone().requires_grad_(True),
            base,
            tids,
            required_logit_margin=0.05,
            required_nll_increase=4.0,
        )
        self.assertGreater(float(failing), 0.0)
        self.assertEqual(float(diagnostics["active_fraction"]), 1.0)

    @staticmethod
    def report(scale, failures, shortfall, mean, *, safe, feasible=False):
        return {
            "scale": scale,
            "direct_failures": failures,
            "constraint_shortfall_sum": shortfall,
            "utility_kl_mean": mean,
            "utility_kl_p95": mean * 2,
            "utility_kl_max": mean * 3,
            "total_delta_norm": scale,
            "utility_safe": safe,
            "feasible": feasible,
        }

    def test_stage1_selects_lowest_utility_feasible_candidate(self):
        reports = [
            self.report(1.0, 0, 0, 0.008, safe=True, feasible=True),
            self.report(0.5, 0, 0, 0.003, safe=True, feasible=True),
            self.report(0.0, 5, 8, 0.0, safe=True),
        ]
        selected, mode = learner.choose_stage1_report(reports)
        self.assertEqual(selected["scale"], 0.5)
        self.assertEqual(mode, "complete")

    def test_stage1_handoff_must_be_utility_safe_and_make_progress(self):
        reports = [
            self.report(1.0, 1, 1, 2.0, safe=False),
            self.report(0.5, 2, 2, 0.005, safe=True),
            self.report(0.0, 5, 8, 0.0, safe=True),
        ]
        selected, mode = learner.choose_stage1_report(reports)
        self.assertEqual(selected["scale"], 0.5)
        self.assertEqual(mode, "utility_safe_residual_handoff")

        no_progress = [
            self.report(0.5, 5, 8, 0.005, safe=True),
            self.report(0.0, 5, 8, 0.0, safe=True),
        ]
        selected, mode = learner.choose_stage1_report(no_progress)
        self.assertIsNone(selected)
        self.assertEqual(mode, "capacity_expansion")

    def test_residual_composition_preserves_gradient(self):
        stage1 = torch.zeros((2, 3))
        residual = torch.ones((1, 3), requires_grad=True)
        total = learner.total_delta_with_residual(stage1, [5, 7], residual, [7])
        total.sum().backward()
        self.assertTrue(torch.equal(residual.grad, torch.ones_like(residual)))
        self.assertTrue(torch.equal(total[0], torch.zeros(3)))
        self.assertTrue(torch.equal(total[1], torch.ones(3)))

    def test_architecture_signature_excludes_dataset_and_seed(self):
        values = {
            "forget_num": 50,
            "utility_sample_size": 100_000,
            "utility_prompt_count": 8_192,
            "stage1_steps": 600,
            "stage1_batch_size": 1,
            "stage1_lr": 0.005,
            "stage2_steps": 500,
            "stage2_batch_size": 8,
            "stage2_protection_batch_size": 16,
            "stage2_lr": 0.005,
            "stage2_check_every": 25,
            "utility_train_batch_size": 128,
            "utility_eval_batch_size": 512,
            "direct_constraint_weight": 100.0,
            "gd_weight": 1.0,
            "utility_kl_weight": 1.0,
            "stage2_protection_weight": 1.0,
            "contrastive_eps": 0.001,
            "constraint_margin": 0.05,
            "min_sensitive_nll_increase": 4.0,
            "utility_kl_mean_budget": 0.01,
            "utility_kl_p95_budget": 0.05,
            "utility_kl_max_budget": 0.5,
            "max_total_delta_norm": 1.5,
            "grad_clip": 1.0,
            "dtype": "bf16",
        }
        mcf = argparse.Namespace(**values, dataset="mcf", seed=1)
        zsre = argparse.Namespace(**values, dataset="zsre", seed=99)
        left = learner.architecture_signature_payload(
            mcf, [1.0, 0.5, 0.0], [2, 4]
        )
        right = learner.architecture_signature_payload(
            zsre, [1.0, 0.5, 0.0], [2, 4]
        )
        self.assertEqual(left, right)
        self.assertEqual(
            learner.architecture_signature_sha256(left),
            learner.architecture_signature_sha256(right),
        )

    def test_mcf_and_zsre_runners_share_architecture_defaults(self):
        mcf = (SCRIPTS / "run_mcf_sure_minimal.sh").read_text(encoding="utf-8")
        zsre = (SCRIPTS / "run_zsre_sure_minimal.sh").read_text(encoding="utf-8")
        defaults = (SCRIPTS / "sure_guarded_shared_defaults.sh").read_text(
            encoding="utf-8"
        )
        source_line = "source scripts/sure_guarded_shared_defaults.sh"
        self.assertIn(source_line, mcf)
        self.assertIn(source_line, zsre)
        shared = [
            "UTILITY_SAMPLE_SIZE=100000",
            "UTILITY_PROMPT_COUNT=8192",
            "UTILITY_SEED=1",
            "UTILITY_EXCLUDE_FIRST=20",
            'STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"',
            'STAGE1_LR="${SURE_STAGE1_LR:-0.005}"',
            'STAGE2_STEPS="${SURE_STAGE2_STEPS:-500}"',
            'STAGE2_LR="${SURE_STAGE2_LR:-0.005}"',
            'DIRECT_CONSTRAINT_WEIGHT="${SURE_DIRECT_CONSTRAINT_WEIGHT:-100.0}"',
            'GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"',
            'UTILITY_KL_WEIGHT="${SURE_UTILITY_KL_WEIGHT:-1.0}"',
            'MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.05}"',
            'MIN_NLL="${SURE_MIN_SENSITIVE_NLL_INCREASE:-4.0}"',
            "RANK_LADDER=2,4",
        ]
        for line in shared:
            self.assertIn(line, defaults)
        for forbidden in ("SURE_RETAIN_TRAIN_NUM", "SURE_STAGE2_RANKS"):
            self.assertNotIn(forbidden, mcf)
            self.assertNotIn(forbidden, zsre)
            self.assertNotIn(forbidden, defaults)


if __name__ == "__main__":
    unittest.main()
