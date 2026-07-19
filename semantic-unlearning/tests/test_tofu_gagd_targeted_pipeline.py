import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import tofu_gagd_active_forget_repair as ACTIVE  # noqa: E402
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
    def test_setting5_groups_unique_overlap_and_retain_only_rows(self):
        groups = SETTING5.build_tofu_row_groups(
            TinyTokenizer(),
            [{"answer": "AB"}],
            [{"answer": "BC"}],
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
        self.assertGreater(oracle, neighborhood)


if __name__ == "__main__":
    unittest.main()
