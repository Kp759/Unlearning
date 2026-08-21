from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
audit = load_module(
    "sure_token_conditioning_audit",
    "audit_sure_token_conditioned_residuals.py",
)


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

    def test_candidate_reservoir_uses_multiple_positions_per_document(self):
        class ToyTokenizer:
            padding_side = "right"

            def __call__(self, texts, **_kwargs):
                ids = torch.tensor(
                    [[int(value) for value in texts[0].split()]], dtype=torch.long
                )
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        class ToyBackbone(nn.Module):
            def forward(self, input_ids, attention_mask, **_kwargs):
                del attention_mask
                hidden = torch.stack(
                    (input_ids.float(), torch.ones_like(input_ids).float()), dim=-1
                )
                return argparse.Namespace(last_hidden_state=hidden)

        class ToyModel(nn.Module):
            base_model_prefix = "backbone"

            def __init__(self):
                super().__init__()
                self.backbone = ToyBackbone()
                self.output = nn.Linear(2, 8, bias=False)

            def get_output_embeddings(self):
                return self.output

        _, utility_hidden, report = wiki.build_second_moment(
            ToyModel(),
            ToyTokenizer(),
            ["1 2 3 4 5"],
            document_order=[0],
            utility_prompt_document_indices=[0],
            utility_seed=1,
            utility_prompt_count=4,
            device=torch.device("cpu"),
            max_length=8,
            batch_size=1,
        )
        self.assertEqual(tuple(utility_hidden.shape), (4, 2))
        self.assertEqual(report["utility_prompt_count"], 4)
        positions = {
            row["predictor_token_position"] for row in report["utility_prompt_records"]
        }
        self.assertEqual(positions, {0, 1, 2, 3})

    def test_cache_validation_locks_sample_model_and_no_benchmark_data(self):
        moment = torch.eye(3)
        hidden = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        logsumexp = torch.tensor([4.0, 5.0])
        metadata = {
            "schema_version": wiki.CACHE_SCHEMA_VERSION,
            "protocol": wiki.UTILITY_PROTOCOL,
            "predictor_hidden_state_count": 7,
            "requested_document_sample_size": 100_000,
            "actual_document_sample_size": 3,
            "requested_utility_prompt_count": 100_000,
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
            (
                loaded,
                loaded_hidden,
                loaded_logsumexp,
                report,
            ) = learner.load_utility_cache(
                path,
                expected_sample_size=100_000,
                expected_prompt_count=100_000,
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
                    expected_prompt_count=100_000,
                    expected_hidden_size=3,
                    expected_model_probe="wrong-model",
                    expected_tokenizer_probe="tokenizer-probe",
                )


class MinimalSureLearnerTests(unittest.TestCase):
    def test_residual_audit_reports_probability_coverage_by_token(self):
        probabilities = torch.tensor(
            [
                [0.0, 0.02],
                [0.001, 0.03],
                [0.01, 0.04],
            ]
        )
        rows = audit.probability_row_reports(
            probabilities,
            [10, 20],
            token_texts=["a", "b"],
            top_counts=[2],
            thresholds=[1e-3, 1e-2],
        )
        self.assertEqual(rows[0]["token_id"], 10)
        self.assertAlmostEqual(rows[0]["top_mean_base_probability"]["2"], 0.0055)
        self.assertEqual(rows[0]["top_actual_context_count"]["2"], 2)
        self.assertEqual(rows[0]["counts_above_probability"]["0.001"], 1)
        self.assertEqual(rows[0]["counts_above_probability"]["0.01"], 0)
        summary = audit.coverage_summary(rows, [1e-3, 1e-2])
        self.assertEqual(summary["rows_with_zero_contexts_above"]["0.001"], 0)
        self.assertEqual(summary["rows_with_zero_contexts_above"]["0.01"], 1)

    def test_residual_audit_constraint_snapshot_preserves_failure_identities(self):
        state = {
            "logit_margin": torch.tensor([1.0, -0.1, 2.0]),
            "sensitive_nll_increase": torch.tensor([4.1, 4.2, 3.8]),
        }
        report, margins, nll, failures = audit.constraint_snapshot(
            state,
            required_margin=0.05,
            required_nll=4.0,
        )
        self.assertEqual(report["direct_failures"], 2)
        self.assertEqual(report["failure_case_indices"], [1, 2])
        self.assertTrue(torch.equal(margins, state["logit_margin"]))
        self.assertTrue(torch.equal(nll, state["sensitive_nll_increase"]))
        self.assertEqual(failures.tolist(), [False, True, True])

    def test_stage2_uses_exact_solver_without_a_scale_frontier(self):
        stage1 = learner.core.parse_scales(learner.DEFAULT_CANDIDATE_SCALES)
        self.assertEqual(max(stage1), 1.0)
        self.assertFalse(hasattr(learner, "DEFAULT_STAGE2_CANDIDATE_SCALES"))

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

    def test_rank4_constraint_context_fallback_adds_real_capacity(self):
        active_hidden = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        active_tids = torch.tensor([5])
        all_direct_hidden = torch.eye(4)
        utility = torch.eye(4)
        rank2, report2 = learner.build_constraint_aware_stage2_bases(
            active_hidden,
            active_tids,
            all_direct_hidden,
            utility,
            requested_ids=[5],
            rank_cap=2,
            relative_eps=1e-6,
            constraint_context_weight=0.05,
        )
        rank4, report4 = learner.build_constraint_aware_stage2_bases(
            active_hidden,
            active_tids,
            all_direct_hidden,
            utility,
            requested_ids=[5],
            rank_cap=4,
            relative_eps=1e-6,
            constraint_context_weight=0.05,
        )
        self.assertEqual(rank2[0].shape[0], 1)
        self.assertEqual(rank4[0].shape[0], 4)
        self.assertEqual(report2[0]["stage2_basis_protocol"], "repair_only")
        self.assertEqual(
            report4[0]["stage2_basis_protocol"],
            "repair_plus_direct_constraint_contexts",
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
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
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

    def test_selected_probability_overmass_uses_exact_full_head_for_only_bad_rows(self):
        layer = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            layer.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]])
            )
        hidden = torch.tensor([[0.5, -0.25], [1.0, 2.0]])
        logits = layer(hidden)
        exact_logsumexp = torch.logsumexp(logits, dim=-1)
        corrupted_logsumexp = exact_logsumexp.clone()
        corrupted_logsumexp[1] -= 3.0
        diagnostics = {}

        actual = learner.selected_base_probabilities(
            layer,
            [1, 3],
            hidden,
            corrupted_logsumexp,
            device=torch.device("cpu"),
            batch_size=1,
            diagnostics=diagnostics,
        )

        expected = torch.softmax(logits, dim=-1)[:, [1, 3]]
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))
        self.assertEqual(diagnostics["fallback_row_count"], 1)
        self.assertEqual(diagnostics["fallback_overmass_row_count"], 1)
        self.assertGreater(diagnostics["approximate_max_selected_mass"], 1.0)
        self.assertLessEqual(diagnostics["final_max_selected_mass"], 1.0 + 1e-5)
        self.assertTrue(diagnostics["full_head_reconstruction_used"])

    def test_token_conditioned_pool_keeps_high_probability_context_per_row(self):
        probabilities = torch.tensor(
            [
                [0.90, 0.01],
                [0.80, 0.02],
                [0.10, 0.70],
                [0.05, 0.60],
                [0.03, 0.04],
                [0.02, 0.03],
            ]
        )
        selected, report = learner.select_token_probability_utility_pool(
            candidate_indices=torch.arange(6),
            selected_base_probabilities=probabilities,
            selected_ids=[10, 20],
            topk_per_row=1,
            uniform_prompt_count=1,
        )
        self.assertEqual(set(selected.tolist()), {0, 1, 2})
        self.assertEqual(report["token_conditioned_prompt_count"], 2)
        self.assertEqual(report["uniform_anchor_prompt_count"], 1)

    def test_token_conditioned_train_and_guard_pools_are_disjoint(self):
        probabilities = torch.linspace(0.01, 0.20, steps=20).reshape(10, 2)
        train, guard, report = learner.build_disjoint_token_conditioned_utility_pools(
            selected_base_probabilities=probabilities,
            selected_ids=[10, 20],
            topk_per_row=1,
            uniform_prompt_count=1,
            split_seed=7,
        )
        self.assertFalse(set(train.tolist()) & set(guard.tolist()))
        self.assertEqual(report["train_guard_overlap_count"], 0)
        self.assertGreaterEqual(len(train), 2)
        self.assertGreaterEqual(len(guard), 2)

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

    def test_cached_stage2_logits_change_only_active_rows_and_keep_gradient(self):
        stage1_logits = torch.tensor(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            dtype=torch.bfloat16,
        )
        hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        residual = torch.tensor([[0.5, -0.25], [-1.0, 0.5]], requires_grad=True)
        actual = learner.logits_with_sparse_residual(
            stage1_logits, hidden, [1, 3], residual
        )
        shifts = (hidden @ residual.transpose(0, 1)).to(torch.bfloat16)
        self.assertEqual(actual.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(actual[:, 0], stage1_logits[:, 0]))
        self.assertTrue(torch.equal(actual[:, 2], stage1_logits[:, 2]))
        self.assertTrue(
            torch.allclose(actual[:, 1], stage1_logits[:, 1] + shifts[:, 0])
        )
        self.assertTrue(
            torch.allclose(actual[:, 3], stage1_logits[:, 3] + shifts[:, 1])
        )
        actual[:, [1, 3]].sum().backward()
        self.assertIsNotNone(residual.grad)
        self.assertGreater(float(residual.grad.abs().sum()), 0.0)

    def test_exact_sparse_stage2_state_matches_full_vocabulary_logits(self):
        torch.manual_seed(7)
        base = torch.randn(5, 11)
        hidden = torch.randn(5, 4)
        active_ids = [2, 7]
        stage1 = base.clone()
        stage1[:, active_ids] += hidden @ (torch.randn(2, 4) * 0.1).transpose(0, 1)
        sensitive_ids = torch.tensor([2, 1, 7, 3, 2])
        residual = torch.randn(2, 4) * 0.2
        cache = learner.build_exact_stage2_direct_cache(
            stage1,
            base,
            hidden,
            sensitive_ids,
            active_ids,
            device=torch.device("cpu"),
        )
        exact = learner.exact_stage2_direct_state(cache, residual)
        full_logits = learner.logits_with_sparse_residual(
            stage1, hidden, active_ids, residual
        )
        expected = learner.constraint_state_from_logits(
            full_logits, base, sensitive_ids
        )
        self.assertTrue(
            torch.allclose(exact["logit_margin"], expected["logit_margin"], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(
                exact["sensitive_nll_increase"],
                expected["sensitive_nll_increase"],
                atol=1e-6,
            )
        )

    def test_exact_constrained_solver_repairs_active_and_preserves_protected(self):
        base = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        stage1 = torch.tensor([[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
        hidden = torch.eye(2)
        sensitive_ids = torch.tensor([0, 0])
        stage1_state = learner.constraint_state_from_logits(
            stage1, base, sensitive_ids
        )
        args = argparse.Namespace(
            utility_kl_weight=1.0,
            stage2_residual_l2_weight=1e-4,
            min_sensitive_nll_increase=2.0,
            constraint_margin=0.05,
            stage2_constraint_buffer=0.0,
            stage2_protected_materialization_buffer=0.0,
            utility_kl_mean_budget=10.0,
            utility_kl_p95_budget=10.0,
            utility_kl_max_budget=10.0,
            max_total_delta_norm=10.0,
            stage2_restarts=2,
            stage2_constraint_tolerance=1e-5,
            stage2_maxiter=200,
            stage2_ftol=1e-9,
        )
        residual, _, report = learner.optimize_stage2(
            args=args,
            stage1_ids=[0],
            stage1_delta=torch.zeros(1, 2),
            active_ids=[0],
            row_bases=[torch.eye(2)],
            all_forget_hidden=hidden,
            all_forget_tids=sensitive_ids,
            stage1_forget_logits=stage1,
            base_forget_logits=base,
            active_case_indices=[0],
            protected_case_indices=[1],
            protected_targets=torch.tensor(
                [float(stage1_state["sensitive_nll_increase"][1] - 0.1)]
            ),
            utility_train_hidden=torch.zeros(3, 2),
            utility_train_probabilities=torch.full((3, 1), 0.1),
            rank=2,
        )
        cache = learner.build_exact_stage2_direct_cache(
            stage1,
            base,
            hidden,
            sensitive_ids,
            [0],
            device=torch.device("cpu"),
        )
        final = learner.exact_stage2_direct_state(cache, residual)
        self.assertTrue(report["continuous_solver_feasible"])
        self.assertGreaterEqual(float(final["sensitive_nll_increase"][0]), 2.0 - 1e-5)
        self.assertGreaterEqual(
            float(final["sensitive_nll_increase"][1]),
            float(stage1_state["sensitive_nll_increase"][1] - 0.1 - 1e-5),
        )
        self.assertGreaterEqual(float(final["logit_margin"].min()), 0.05 - 1e-5)

    def test_exact_solver_retains_a_feasible_intermediate_iterate(self):
        base = torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        stage1 = torch.tensor([[2.0, 0.0, 0.0], [-3.0, 0.0, 0.0]])
        hidden = torch.eye(2)
        sensitive_ids = torch.tensor([0, 0])
        stage1_state = learner.constraint_state_from_logits(
            stage1, base, sensitive_ids
        )
        args = argparse.Namespace(
            utility_kl_weight=1.0,
            stage2_residual_l2_weight=1e-4,
            min_sensitive_nll_increase=2.0,
            constraint_margin=0.05,
            stage2_constraint_buffer=0.0,
            stage2_protected_materialization_buffer=0.0,
            utility_kl_mean_budget=10.0,
            utility_kl_p95_budget=10.0,
            utility_kl_max_budget=10.0,
            max_total_delta_norm=10.0,
            stage2_restarts=1,
            stage2_constraint_tolerance=1e-5,
            stage2_maxiter=5,
            stage2_ftol=1e-9,
        )

        def fake_minimize(*_args, callback, **_kwargs):
            callback(learner.np.array([-4.0, 0.0]))
            return SimpleNamespace(
                x=learner.np.zeros(2),
                success=False,
                status=9,
                message="iteration limit",
                nit=1,
                nfev=1,
                njev=1,
            )

        with mock.patch.object(learner, "minimize", side_effect=fake_minimize):
            residual, _, report = learner.optimize_stage2(
                args=args,
                stage1_ids=[0],
                stage1_delta=torch.zeros(1, 2),
                active_ids=[0],
                row_bases=[torch.eye(2)],
                all_forget_hidden=hidden,
                all_forget_tids=sensitive_ids,
                stage1_forget_logits=stage1,
                base_forget_logits=base,
                active_case_indices=[0],
                protected_case_indices=[1],
                protected_targets=torch.tensor(
                    [float(stage1_state["sensitive_nll_increase"][1] - 0.1)]
                ),
                utility_train_hidden=torch.zeros(3, 2),
                utility_train_probabilities=torch.full((3, 1), 0.1),
                rank=2,
            )
        self.assertTrue(report["continuous_solver_feasible"])
        self.assertEqual(report["solver_phase"], "iterate")
        self.assertTrue(torch.allclose(residual, torch.tensor([[-4.0, 0.0]])))

    def test_exact_constrained_solver_never_trades_protection_for_repair(self):
        base = torch.tensor([[2.0, 0.0, -10.0], [2.0, 0.0, -10.0]])
        stage1 = torch.tensor([[2.0, 0.0, -10.0], [2.0, -5.0, -10.0]])
        hidden = torch.ones(2, 1)
        sensitive_ids = torch.tensor([0, 1])
        stage1_state = learner.constraint_state_from_logits(
            stage1, base, sensitive_ids
        )
        args = argparse.Namespace(
            utility_kl_weight=1.0,
            stage2_residual_l2_weight=1e-4,
            min_sensitive_nll_increase=2.0,
            constraint_margin=0.05,
            stage2_constraint_buffer=0.0,
            stage2_protected_materialization_buffer=0.0,
            utility_kl_mean_budget=10.0,
            utility_kl_p95_budget=10.0,
            utility_kl_max_budget=10.0,
            max_total_delta_norm=10.0,
            stage2_restarts=2,
            stage2_constraint_tolerance=1e-5,
            stage2_maxiter=100,
            stage2_ftol=1e-9,
        )
        _, _, report = learner.optimize_stage2(
            args=args,
            stage1_ids=[0, 1],
            stage1_delta=torch.zeros(2, 1),
            active_ids=[0],
            row_bases=[torch.ones(1, 1)],
            all_forget_hidden=hidden,
            all_forget_tids=sensitive_ids,
            stage1_forget_logits=stage1,
            base_forget_logits=base,
            active_case_indices=[0],
            protected_case_indices=[1],
            protected_targets=torch.tensor(
                [float(stage1_state["sensitive_nll_increase"][1] - 0.1)]
            ),
            utility_train_hidden=torch.zeros(3, 1),
            utility_train_probabilities=torch.full((3, 2), 0.1),
            rank=2,
        )
        self.assertFalse(report["continuous_solver_feasible"])
        self.assertEqual(report["selection_mode"], "best_infeasible_diagnostic")
        self.assertLess(report["solver_minimum_direct_slack"], 0.0)

    def test_protected_targets_keep_stage1_clearance_above_global_floor(self):
        stage1_nll = torch.tensor([3.0, 4.01, 4.40])
        targets = learner.protected_nll_targets(
            stage1_nll,
            [1, 2],
            required_nll_increase=4.0,
            tolerance=0.05,
        )
        self.assertTrue(torch.allclose(targets, torch.tensor([4.0, 4.35])))

    def test_protected_materialization_buffer_preserves_tolerance_and_clearance(self):
        nll, margin = learner.stage2_solver_targets(
            3,
            [1, 2],
            torch.tensor([4.0, 4.35]),
            required_nll_increase=4.0,
            required_logit_margin=0.05,
            constraint_buffer=0.05,
            protected_materialization_buffer=0.005,
            device=torch.device("cpu"),
        )
        self.assertTrue(torch.allclose(nll, torch.tensor([4.05, 4.05, 4.355])))
        self.assertTrue(torch.allclose(margin, torch.full((3,), 0.10)))

    def test_exact_constraints_catch_unedited_sensitive_row_denominator_shift(self):
        base = torch.zeros((1, 3))
        stage1_logits = torch.tensor([[-5.0, 0.0, 0.0]])
        hidden = torch.ones((1, 1))
        residual = torch.tensor([[-4.0]])
        sensitive_ids = torch.tensor([0])
        cache = learner.build_exact_stage2_direct_cache(
            stage1_logits,
            base,
            hidden,
            sensitive_ids,
            [1],
            device=torch.device("cpu"),
        )
        state = learner.exact_stage2_direct_state(cache, residual)
        stage1_nll = learner.shared.sensitive_nll_increase_from_logits(
            stage1_logits, base, sensitive_ids
        )
        self.assertLess(float(state["sensitive_nll_increase"][0]), float(stage1_nll[0]))
        report = learner.stage2_partition_report(
            state,
            active_indices=[],
            protected_indices=[0],
            protected_targets=stage1_nll - 0.05,
            args=argparse.Namespace(
                constraint_margin=-10.0,
                min_sensitive_nll_increase=0.0,
            ),
        )
        self.assertEqual(report["protected_floor_violations"], 1)

    def test_stage2_partition_separates_repairs_from_new_regressions(self):
        state = {
            "logit_margin": torch.ones(4),
            "sensitive_nll_increase": torch.tensor([4.1, 4.2, 3.8, 3.88]),
        }
        args = argparse.Namespace(
            constraint_margin=0.05,
            min_sensitive_nll_increase=4.0,
        )
        report = learner.stage2_partition_report(
            state,
            active_indices=[0, 1],
            protected_indices=[2, 3],
            protected_targets=torch.tensor([4.0, 4.0]),
            args=args,
        )
        self.assertEqual(report["active_direct_failures"], 0)
        self.assertEqual(report["active_repaired_cases"], 2)
        self.assertEqual(report["protected_direct_failures"], 2)
        self.assertEqual(report["protected_new_regressions"], 2)
        self.assertEqual(report["protected_floor_violations"], 2)

    def test_stage2_protection_floor_is_a_locked_feasibility_guard(self):
        report = {
            "direct_failures": 0,
            "protected_floor_violations": 1,
            "utility_kl_mean": 0.0,
            "utility_kl_p95": 0.0,
            "utility_kl_max": 0.0,
            "total_delta_norm": 0.1,
        }
        args = argparse.Namespace(
            utility_kl_mean_budget=0.01,
            utility_kl_p95_budget=0.05,
            utility_kl_max_budget=0.5,
            max_total_delta_norm=1.5,
        )
        learner.add_utility_guards(report, args)
        self.assertTrue(report["direct_success"])
        self.assertTrue(report["utility_safe"])
        self.assertFalse(report["stage2_protection_safe"])
        self.assertFalse(report["feasible"])

    def test_materialized_candidate_cannot_override_solver_infeasibility(self):
        report = {"feasible": True}
        solver = {
            "continuous_solver_feasible": False,
            "solver_minimum_direct_slack": -0.1,
            "solver_minimum_utility_slack": 0.2,
            "selection_mode": "best_infeasible_diagnostic",
        }
        learner.attach_stage2_solver_feasibility(report, solver)
        self.assertTrue(report["materialized_feasible"])
        self.assertFalse(report["continuous_solver_feasible"])
        self.assertFalse(report["feasible"])

    def test_architecture_signature_excludes_dataset_and_seed(self):
        values = {
            "forget_num": 50,
            "utility_sample_size": 100_000,
            "utility_prompt_count": 100_000,
            "utility_token_topk_per_row": 128,
            "utility_uniform_prompt_count": 1_024,
            "utility_pool_seed": 1,
            "stage1_steps": 600,
            "stage1_batch_size": 1,
            "stage1_lr": 0.005,
            "stage2_maxiter": 500,
            "stage2_ftol": 1e-9,
            "stage2_constraint_tolerance": 1e-5,
            "stage2_constraint_buffer": 0.05,
            "stage2_protected_materialization_buffer": 0.005,
            "stage2_residual_l2_weight": 1e-4,
            "stage2_constraint_basis_weight": 0.05,
            "stage2_restarts": 2,
            "utility_train_batch_size": 128,
            "utility_eval_batch_size": 512,
            "direct_constraint_weight": 100.0,
            "gd_weight": 1.0,
            "utility_kl_weight": 1.0,
            "stage2_protection_nll_tolerance": 0.05,
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
            "UTILITY_PROMPT_COUNT=100000",
            "UTILITY_SEED=1",
            "UTILITY_EXCLUDE_FIRST=20",
            'STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"',
            'STAGE1_LR="${SURE_STAGE1_LR:-0.005}"',
            'STAGE2_MAXITER="${SURE_STAGE2_MAXITER:-500}"',
            'STAGE2_FTOL="${SURE_STAGE2_FTOL:-1e-9}"',
            'STAGE2_CONSTRAINT_BUFFER="${SURE_STAGE2_CONSTRAINT_BUFFER:-0.05}"',
            'STAGE2_PROTECTED_MATERIALIZATION_BUFFER="${SURE_STAGE2_PROTECTED_MATERIALIZATION_BUFFER:-0.005}"',
            'STAGE2_RESIDUAL_L2_WEIGHT="${SURE_STAGE2_RESIDUAL_L2_WEIGHT:-0.0001}"',
            'STAGE2_CONSTRAINT_BASIS_WEIGHT="${SURE_STAGE2_CONSTRAINT_BASIS_WEIGHT:-0.05}"',
            'STAGE2_PROTECTION_NLL_TOLERANCE="${SURE_STAGE2_PROTECTION_NLL_TOLERANCE:-0.05}"',
            'DIRECT_CONSTRAINT_WEIGHT="${SURE_DIRECT_CONSTRAINT_WEIGHT:-100.0}"',
            'GD_WEIGHT="${SURE_GD_WEIGHT:-1.0}"',
            'UTILITY_KL_WEIGHT="${SURE_UTILITY_KL_WEIGHT:-1.0}"',
            'UTILITY_TOKEN_TOPK_PER_ROW="${SURE_UTILITY_TOKEN_TOPK_PER_ROW:-128}"',
            'UTILITY_UNIFORM_PROMPT_COUNT="${SURE_UTILITY_UNIFORM_PROMPT_COUNT:-1024}"',
            'MARGIN="${SURE_SHARED_CONSTRAINT_MARGIN:-0.05}"',
            'MIN_NLL="${SURE_MIN_SENSITIVE_NLL_INCREASE:-4.0}"',
            "RANK_LADDER=2,4",
        ]
        for line in shared:
            self.assertIn(line, defaults)
        for runner in (mcf, zsre):
            self.assertNotIn("--stage2-candidate-scales", runner)
            self.assertIn('--stage2-maxiter "${STAGE2_MAXITER}"', runner)
            self.assertIn(
                '--stage2-constraint-buffer "${STAGE2_CONSTRAINT_BUFFER}"',
                runner,
            )
            self.assertIn(
                '--stage2-protected-materialization-buffer "${STAGE2_PROTECTED_MATERIALIZATION_BUFFER}"',
                runner,
            )
            self.assertIn(
                '--utility-token-topk-per-row "${UTILITY_TOKEN_TOPK_PER_ROW}"',
                runner,
            )
            self.assertIn(
                '--utility-uniform-prompt-count "${UTILITY_UNIFORM_PROMPT_COUNT}"',
                runner,
            )
            self.assertIn(
                '--stage2-protection-nll-tolerance "${STAGE2_PROTECTION_NLL_TOLERANCE}"',
                runner,
            )
            self.assertNotIn("--stage2-protection-batch-size", runner)
            self.assertNotIn("--stage2-batch-size", runner)
        for forbidden in ("SURE_RETAIN_TRAIN_NUM", "SURE_STAGE2_RANKS"):
            self.assertNotIn(forbidden, mcf)
            self.assertNotIn(forbidden, zsre)
            self.assertNotIn(forbidden, defaults)


if __name__ == "__main__":
    unittest.main()
