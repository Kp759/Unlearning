import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import tofu_gagd_active_forget_repair as ACTIVE  # noqa: E402
import gagd_compare as GAGD  # noqa: E402
import tofu_gagd_neighborhood_confidence as NEIGHBORHOOD  # noqa: E402
import tofu_gagd_results as RESULTS  # noqa: E402
import tofu_gagd_setting5e_restore as SETTING5  # noqa: E402
import tofu_retain_only_oracle as ORACLE  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    bos_token_id = 2
    unk_token_id = None

    def __call__(self, text, add_special_tokens=False, **kwargs):
        del add_special_tokens, kwargs
        return {"input_ids": [ord(character) for character in text]}

    def decode(self, token_ids):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class TOFUTargetedPipelineTests(unittest.TestCase):
    @staticmethod
    def answer_instance(split, position, answer):
        question = f"question-{position}"
        return NEIGHBORHOOD.TOFUAnswerInstance(
            split=split,
            source_index=position,
            sampled_position=position,
            question=question,
            answer=answer,
            prompt=f"Question: {question} Answer:",
        )

    def test_tofu_ga_gd_does_not_train_on_unscored_eos(self):
        self.assertFalse(GAGD.append_eos_for_dataset("tofu"))
        self.assertTrue(GAGD.append_eos_for_dataset("mcf"))

    def test_setting5_groups_unique_overlap_and_retain_only_rows(self):
        groups = SETTING5.build_tofu_row_groups(
            TinyTokenizer(),
            [{"question": "forget", "answer": "AB"}],
            [{"question": "retain", "answer": "BC"}],
        )
        self.assertIn(ord("A"), groups.unique_forget)
        self.assertIn(ord("B"), groups.forget_retain_overlap)
        self.assertIn(ord("C"), groups.retain_only)
        # Leading-space token is deliberately recognized as shared.
        self.assertIn(ord(" "), groups.forget_retain_overlap)

    def test_setting5_policy_restores_unsafe_rows_and_partially_keeps_overlap(self):
        groups = SETTING5.TOFURowGroups(
            forget=(1, 2),
            retain=(2, 3),
            unique_forget=(1,),
            forget_retain_overlap=(2,),
            retain_only=(3,),
        )
        trained = torch.ones((5, 2))
        base = torch.zeros_like(trained)
        alphas = SETTING5.build_row_alphas(
            5,
            groups,
            unique_forget_alpha=1.0,
            overlap_alpha=0.25,
            device=torch.device("cpu"),
        )
        SETTING5.apply_row_policy(
            trained,
            base,
            alphas,
            chunk_rows=2,
        )
        self.assertTrue(torch.equal(trained[1], torch.ones(2)))
        self.assertTrue(torch.equal(trained[2], torch.full((2,), 0.25)))
        self.assertTrue(torch.equal(trained[3], torch.zeros(2)))
        self.assertTrue(torch.equal(trained[4], torch.zeros(2)))

    def test_setting5_restores_every_input_embedding_row(self):
        trained = torch.ones((5, 2))
        base = torch.zeros_like(trained)
        stats = SETTING5.restore_weight_to_base(
            trained,
            base,
            chunk_rows=2,
        )
        self.assertTrue(torch.equal(trained, base))
        self.assertGreater(stats["delta_norm_before"], 0.0)
        self.assertEqual(stats["delta_norm_after"], 0.0)

    def test_active_required_nll_encodes_two_e_minus_five_target_and_buffer(self):
        original = torch.tensor([1.0, 11.0, 12.0])
        required, active, target_nll = ACTIVE.build_required_forget_nll(
            original,
            target_probability=2e-5,
            target_nll_buffer=0.25,
        )
        self.assertAlmostEqual(target_nll, -math.log(2e-5))
        self.assertEqual(active.tolist(), [True, False, False])
        self.assertAlmostEqual(
            float(required[0]),
            target_nll + 0.25,
            places=5,
        )
        self.assertAlmostEqual(float(required[1]), 11.0, places=5)
        self.assertAlmostEqual(
            float(required[2]),
            target_nll + 0.25,
            places=5,
        )
        self.assertTrue(torch.all(required >= target_nll))

    def test_active_objective_contains_all_forget_and_utility_cases(self):
        forget = torch.tensor([10.0, 12.0], requires_grad=True)
        required_forget = torch.tensor([11.0, 11.0])
        utility = torch.tensor([2.0, 4.0], requires_grad=True)
        required_utility = torch.tensor([3.0, 3.0])
        terms = ACTIVE.target_objective_terms(
            forget,
            required_forget,
            utility,
            required_utility,
        )
        (terms["forget_hinge"] + terms["utility_hinge"]).backward()
        self.assertLess(float(forget.grad[0]), 0.0)
        self.assertEqual(float(forget.grad[1]), 0.0)
        self.assertEqual(float(utility.grad[0]), 0.0)
        self.assertGreater(float(utility.grad[1]), 0.0)
        self.assertEqual(terms["forget_slack"].shape, forget.shape)
        self.assertEqual(terms["utility_slack"].shape, utility.shape)
        self.assertEqual(
            float(terms["hardest_forget_hinge"]),
            float(torch.tensor(1.0)),
        )

    def test_aggregate_utility_gate_matches_split_mean_probability(self):
        instances = [
            self.answer_instance("retain", 0, "A"),
            self.answer_instance("retain", 1, "B"),
        ]
        reference_probability = torch.tensor([0.9, 0.1])
        candidate_probability = torch.tensor([0.89, 0.11])
        reference_nll = -reference_probability.log()
        candidate_nll = -candidate_probability.log()
        ratios = ACTIVE.utility_probability_ratio_tensors(
            candidate_nll,
            reference_nll,
            instances,
        )
        self.assertAlmostEqual(float(ratios["retain"]), 1.0, places=6)
        hinge, slacks = ACTIVE.aggregate_utility_hinge(
            candidate_nll,
            reference_nll,
            instances,
            0.9998,
        )
        self.assertEqual(float(hinge), 0.0)
        self.assertTrue(torch.all(slacks > 0))

    def test_aggregate_utility_hinge_penalizes_ratio_deficit(self):
        instances = [
            self.answer_instance("retain", 0, "A"),
            self.answer_instance("retain", 1, "B"),
        ]
        reference_nll = -torch.tensor([0.8, 0.8]).log()
        candidate_nll = -torch.tensor([0.7, 0.7]).log()
        hinge, slacks = ACTIVE.aggregate_utility_hinge(
            candidate_nll,
            reference_nll,
            instances,
            0.9998,
        )
        self.assertGreater(float(hinge), 0.0)
        self.assertTrue(torch.all(slacks < 0))

    def test_active_utility_tolerance_matches_9998_percent_preservation(self):
        tolerance = ACTIVE.probability_ratio_nll_tolerance(0.9998)
        self.assertAlmostEqual(tolerance, -math.log(0.9998), places=12)
        self.assertAlmostEqual(tolerance, 0.0002000200026670447, places=12)
        self.assertAlmostEqual(
            ACTIVE.probability_ratio_nll_tolerance(0.9998, 0.0001),
            0.0001,
        )
        self.assertAlmostEqual(
            ACTIVE.probability_ratio_nll_tolerance(0.9998, 0.05),
            tolerance,
        )

    def test_active_row_selection_excludes_protected_answer_rows(self):
        tokenizer = TinyTokenizer()
        forget = [self.answer_instance("forget05", 0, "AB")]
        protected = [self.answer_instance("retain", 0, "BC")]
        selected, shared = ACTIVE.partition_active_forget_row_ids(
            tokenizer,
            forget,
            [0],
            max_length=256,
            protected_instances=protected,
        )
        self.assertIn(ord("A"), selected)
        self.assertNotIn(ord("B"), selected)
        self.assertIn(ord("B"), shared)
        self.assertIn(ord(" "), shared)

    def test_active_candidate_priority_never_trades_utility_for_forgetting(self):
        utility_safe = {
            "utility_constraint_violation_count": 0,
            "active_forget_instance_count": 1,
            "buffered_forget_constraint_unmet_count": 1,
            "minimum_forget_answer_nll": 10.0,
            "selected_lm_head_delta_norm": 0.1,
        }
        destructive = {
            "utility_constraint_violation_count": 256,
            "active_forget_instance_count": 0,
            "buffered_forget_constraint_unmet_count": 0,
            "minimum_forget_answer_nll": 11.1,
            "selected_lm_head_delta_norm": 99.0,
        }
        self.assertLess(
            ACTIVE.candidate_priority(utility_safe),
            ACTIVE.candidate_priority(destructive),
        )

    def test_active_parser_defaults_to_contextual_hard_utility_gates(self):
        args = ACTIVE.build_parser().parse_args(
            [
                "--model-path",
                "setting5",
                "--reference-model-path",
                "base",
                "--output-dir",
                "out",
            ]
        )
        self.assertTrue(args.require_utility_constraints)
        self.assertTrue(args.require_input_retain_target)
        self.assertGreater(args.repair_rank, 0)
        self.assertEqual(args.retain_calibration_num, args.retain_num)
        self.assertEqual(args.target_forget_answer_probability, 3e-4)
        self.assertEqual(args.min_utility_probability_ratio, 0.9998)
        self.assertEqual(args.utility_constraint_mode, "aggregate")
        self.assertEqual(args.repair_steps, 5000)
        self.assertEqual(args.repair_lr, 2e-2)
        self.assertEqual(args.repair_rank, 64)
        self.assertEqual(args.utility_projection_rank, 64)
        self.assertGreater(args.hardest_forget_hinge_weight, 0)

    def test_result_parser_defaults_to_requested_joint_target(self):
        args = RESULTS.build_parser().parse_args(["--output-dir", "out"])
        self.assertEqual(args.max_forget_answer_probability, 3e-4)
        self.assertEqual(args.min_retain_probability_ratio, 0.9998)

    def test_neighborhood_parser_uses_requested_forget_target(self):
        args = NEIGHBORHOOD.build_parser().parse_args(
            [
                "--model-path",
                "candidate",
                "--reference-model-path",
                "base",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.max_forget_answer_probability, 3e-4)

    def test_packed_sparse_cache_matches_per_answer_computation(self):
        caches = [
            NEIGHBORHOOD.TOFUAnswerDeltaCache(
                base_token_nll=torch.tensor([1.0, 2.0]),
                hidden=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                selected_probs=torch.tensor([[0.1, 0.2], [0.2, 0.1]]),
                target_selected_columns=torch.tensor([0, -1]),
            ),
            NEIGHBORHOOD.TOFUAnswerDeltaCache(
                base_token_nll=torch.tensor([3.0]),
                hidden=torch.tensor([[1.0, 1.0]]),
                selected_probs=torch.tensor([[0.05, 0.15]]),
                target_selected_columns=torch.tensor([1]),
            ),
        ]
        delta = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
        expected = NEIGHBORHOOD.answer_nlls_from_delta_caches(
            caches,
            delta,
        )
        packed = NEIGHBORHOOD.pack_answer_delta_caches(caches)
        actual = NEIGHBORHOOD.answer_nlls_from_packed_delta_cache(
            packed,
            delta,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_neighborhood_gate_rejects_the_observed_weak_checkpoint(self):
        nll = torch.full((4,), -math.log(0.99775))
        metrics = NEIGHBORHOOD.forgetting_target_metrics(nll, 2e-5)
        self.assertFalse(metrics["mean_target_met"])
        self.assertFalse(metrics["all_instances_target_met"])
        self.assertEqual(metrics["forget_instances_above_target"], 4)

    def test_final_comparison_hard_gate_accepts_only_target_result(self):
        display = "Setting 5e + active + neighborhood repair"
        passing = [
            {
                "Method": display,
                "Forget answer probability ↓": 1.9e-5,
            }
        ]
        RESULTS.require_forgetting_target(
            passing,
            required_display_name=display,
            max_forget_answer_probability=2e-5,
        )
        failing = [
            {
                "Method": display,
                "Forget answer probability ↓": 2.1e-5,
            }
        ]
        with self.assertRaises(RuntimeError):
            RESULTS.require_forgetting_target(
                failing,
                required_display_name=display,
                max_forget_answer_probability=2e-5,
            )

    def test_final_comparison_requires_forget_and_relative_retain_targets(self):
        display = "Setting 5e + active forget repair"
        rows = [
            {
                "Method": "Base",
                "Forget answer probability ↓": 0.998,
                "Retain answer probability ↑": 0.998,
                "Meets forgetting target": False,
            },
            {
                "Method": display,
                "Forget answer probability ↓": 1.9e-5,
                "Retain answer probability ↑": 0.998 * 0.9998,
                "Meets forgetting target": True,
            },
        ]
        RESULTS.add_base_differences(rows, 0.9998)
        RESULTS.require_joint_target(
            rows,
            required_display_name=display,
            max_forget_answer_probability=2e-5,
            min_retain_probability_ratio=0.9998,
        )
        rows[1]["Retain answer probability ↑"] = 0.998 * 0.9997
        RESULTS.add_base_differences(rows, 0.9998)
        with self.assertRaises(RuntimeError):
            RESULTS.require_joint_target(
                rows,
                required_display_name=display,
                max_forget_answer_probability=2e-5,
                min_retain_probability_ratio=0.9998,
            )

    def test_oracle_recovers_full_finetune_provenance(self):
        args = SimpleNamespace(
            base_model_path=None,
            epochs=None,
            batch_size=None,
            lr=None,
        )
        resolved = ORACLE.resolve_training_config(
            args,
            {
                "base_model": "/models/pre-tofu",
                "epochs": 9,
                "batch_size": 8,
                "lr": 5e-4,
            },
        )
        self.assertEqual(resolved["base_model_path"], "/models/pre-tofu")
        self.assertEqual(resolved["epochs"], 9)
        self.assertEqual(resolved["batch_size"], 8)
        self.assertEqual(resolved["lr"], 5e-4)

    def test_shell_wires_all_required_stages_in_order(self):
        script = (
            SCRIPTS_DIR / "run_tofu_gagd_neighborhood_confidence.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("tofu_gagd_four_settings_official.py", script)
        setting5 = script.index("tofu_gagd_setting5e_restore.py")
        active = script.index("tofu_gagd_active_forget_repair.py")
        neighborhood = script.index("tofu_gagd_neighborhood_confidence.py")
        oracle = script.index("tofu_retain_only_oracle.py")
        self.assertLess(setting5, active)
        self.assertLess(active, neighborhood)
        self.assertIn("--max-forget-answer-probability", script)
        self.assertIn("--min-utility-probability-ratio", script)
        self.assertIn("--require-utility-constraints", script)
        self.assertIn("--reference-model-path", script)
        self.assertIn('RUN_NEIGHBORHOOD_REPAIR="${RUN_NEIGHBORHOOD_REPAIR:-0}"', script)
        self.assertIn(
            'TARGET_FORGET_ANSWER_PROBABILITY="${TARGET_FORGET_ANSWER_PROBABILITY:-0.0003}"',
            script,
        )
        self.assertIn(
            'SETTING5_OVERLAP_ALPHA="${SETTING5_OVERLAP_ALPHA:-0.0}"',
            script,
        )
        self.assertGreater(oracle, neighborhood)

    def test_setting5_defaults_to_full_overlap_restoration(self):
        args = SETTING5.build_parser().parse_args(
            [
                "--model-path",
                "trained",
                "--base-model-path",
                "base",
                "--output-dir",
                "out",
            ]
        )
        self.assertEqual(args.overlap_alpha, 0.0)


if __name__ == "__main__":
    unittest.main()
