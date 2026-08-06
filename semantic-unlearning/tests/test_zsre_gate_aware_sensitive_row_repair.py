import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
import aggregate_zsre_gagd_results as AGG  # noqa: E402
import gagd_active_case_repair as ACTIVE  # noqa: E402
import zsre_gate_aware_sensitive_row_repair as REPAIR  # noqa: E402


def metric_result(
    *,
    forget_eff=0.0,
    forget_gen=0.0,
    forget_spe=80.0,
    retain_eff=90.0,
    retain_gen=89.0,
    retain_spe=88.0,
    ppl=10.0,
):
    return {
        "forget": {
            "Eff": forget_eff,
            "Gen": forget_gen,
            "Spe": forget_spe,
        },
        "retain": {
            "Eff": retain_eff,
            "Gen": retain_gen,
            "Spe": retain_spe,
        },
        "forget_PPL": ppl,
    }


def forget_guard(
    *,
    active_violations=0,
    regression_violations=0,
    newly_correct=0,
):
    return {
        "original_active_violation_count": active_violations,
        "forget_regression_violation_count": regression_violations,
        "newly_correct_forget_token_count": newly_correct,
        "minimum_active_slack": 0.1,
        "minimum_forget_regression_slack": 0.1,
    }


def empty_forget_regression_tensors(*, hidden_size=2, selected_rows=1):
    return REPAIR.ForgetRegressionConstraintTensors(
        hidden=torch.empty((0, hidden_size)),
        target_logits=torch.empty((0,)),
        strongest_unchanged_logits=torch.empty((0,)),
        selected_logits=torch.empty((0, selected_rows)),
        target_selected_columns=torch.empty((0,), dtype=torch.long),
    )


class TinyTiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(6, 3)
        self.transformer = nn.Linear(3, 3, bias=False)
        self.lm_head = nn.Linear(3, 6, bias=False)
        self.lm_head.weight = self.embed.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value


class TinyTestOptimizer:
    """Minimal SGD used to avoid importing optional torch compiler dependencies."""

    def __init__(self, module, learning_rate):
        self.parameters = list(module.parameters())
        self.learning_rate = learning_rate

    def zero_grad(self, set_to_none=False):
        for parameter in self.parameters:
            if set_to_none:
                parameter.grad = None
            elif parameter.grad is not None:
                parameter.grad.zero_()

    def step(self):
        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-self.learning_rate)


class GateAwareSensitiveRowRepairTests(unittest.TestCase):
    def test_forget_regression_cli_defaults(self):
        args = REPAIR.build_parser().parse_args(
            [
                "--setting5-checkpoint",
                "/tmp/setting5",
                "--output-dir",
                "/tmp/output",
            ]
        )
        self.assertEqual(args.forget_regression_margin, 0.02)
        self.assertEqual(args.forget_regression_hinge_weight, 2.0)

    def assert_complete_cycle_before_repeat(self, total, batch_size):
        batcher = REPAIR.DeterministicCyclicBatcher(total, batch_size)
        first_cycle = []
        while len(first_cycle) < total:
            batch = batcher.next_batch()
            self.assertEqual(batch.cycle, 0)
            first_cycle.extend(batch.indices)
        self.assertEqual(first_cycle, list(range(total)))
        self.assertEqual(len(first_cycle), len(set(first_cycle)))
        repeated = batcher.next_batch()
        self.assertEqual(repeated.cycle, 1)
        self.assertEqual(
            repeated.indices,
            tuple(range(min(batch_size, total))),
        )

    def test_protected_batches_cover_every_item_before_repeating(self):
        self.assert_complete_cycle_before_repeat(total=11, batch_size=4)

    def test_retain_batches_cover_every_item_before_repeating(self):
        self.assert_complete_cycle_before_repeat(total=10, batch_size=3)

    def test_forget_regression_batches_cover_every_item_before_repeating(self):
        self.assert_complete_cycle_before_repeat(total=73, batch_size=16)

    def test_only_selected_sensitive_rows_receive_deltas(self):
        weight = torch.arange(15, dtype=torch.float32).reshape(5, 3)
        baseline = weight.clone()
        selected = [1, 4]
        original_rows = baseline[selected].clone()
        delta = torch.tensor([[1.0, -2.0, 3.0], [-1.0, 2.0, -3.0]])
        REPAIR.materialize_sensitive_rows(
            weight,
            selected,
            original_rows,
            delta,
            0.5,
        )
        self.assertTrue(torch.equal(weight[0], baseline[0]))
        self.assertTrue(torch.equal(weight[2], baseline[2]))
        self.assertTrue(torch.equal(weight[3], baseline[3]))
        self.assertTrue(torch.equal(weight[1], baseline[1] + 0.5 * delta[0]))
        self.assertTrue(torch.equal(weight[4], baseline[4] + 0.5 * delta[1]))

    def test_input_embeddings_and_transformer_remain_unchanged(self):
        model = TinyTiedModel()
        embedding_before = model.embed.weight.detach().clone()
        transformer_before = model.transformer.weight.detach().clone()
        lm_head_before = model.lm_head.weight.detach().clone()
        output = ACTIVE.freeze_model_for_output_repair(model)
        selected = [2]
        REPAIR.materialize_sensitive_rows(
            output.weight,
            selected,
            output.weight[selected].detach().clone(),
            torch.tensor([[0.5, -0.25, 1.0]]),
            1.0,
        )
        self.assertTrue(torch.equal(model.embed.weight, embedding_before))
        self.assertTrue(torch.equal(model.transformer.weight, transformer_before))
        self.assertTrue(torch.equal(output.weight[0], lm_head_before[0]))
        self.assertFalse(torch.equal(output.weight[2], lm_head_before[2]))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))

    def test_baseline_correct_active_margin_contract_remains_unchanged(self):
        margin = REPAIR.active_constraint_margins(
            hidden=torch.tensor([[2.0, 1.0]]),
            sensitive_logits=torch.tensor([3.0]),
            best_other_logits=torch.tensor([4.0]),
            selected_row_columns=torch.tensor([0]),
            delta_rows=torch.tensor([[-1.0, 0.0]]),
        )
        # 4 - (3 + [2,1] @ [-1,0]) = 3.
        self.assertEqual(float(margin.item()), 3.0)

    def test_lowered_selected_competitor_can_make_incorrect_target_correct(self):
        baseline = REPAIR.forget_regression_margins(
            hidden=torch.tensor([[1.0]]),
            target_logits=torch.tensor([1.0]),
            strongest_unchanged_logits=torch.tensor([0.5]),
            selected_logits=torch.tensor([[2.0]]),
            target_selected_columns=torch.tensor([-1]),
            delta_rows=torch.tensor([[0.0]]),
        )
        damaged = REPAIR.forget_regression_margins(
            hidden=torch.tensor([[1.0]]),
            target_logits=torch.tensor([1.0]),
            strongest_unchanged_logits=torch.tensor([0.5]),
            selected_logits=torch.tensor([[2.0]]),
            target_selected_columns=torch.tensor([-1]),
            delta_rows=torch.tensor([[-2.0]]),
        )
        self.assertGreater(float(baseline.item()), 0.0)
        self.assertLess(float(damaged.item()), 0.0)

    def test_forget_guard_detects_newly_correct_baseline_incorrect_target(self):
        active_tensors = REPAIR.ActiveConstraintTensors(
            hidden=torch.empty((0, 1)),
            sensitive_logits=torch.empty((0,)),
            best_other_logits=torch.empty((0,)),
            selected_row_columns=torch.empty((0,), dtype=torch.long),
        )
        regression_tensors = REPAIR.ForgetRegressionConstraintTensors(
            hidden=torch.tensor([[1.0]]),
            target_logits=torch.tensor([1.0]),
            strongest_unchanged_logits=torch.tensor([0.5]),
            selected_logits=torch.tensor([[2.0]]),
            target_selected_columns=torch.tensor([-1]),
        )
        protected_tensors = REPAIR.ProtectedConstraintTensors(
            hidden=torch.empty((0, 1)),
            target_logits=torch.empty((0,)),
            strongest_unchanged_logits=torch.empty((0,)),
            selected_logits=torch.empty((0, 1)),
            target_selected_columns=torch.empty((0,), dtype=torch.long),
            required_margins=torch.empty((0,)),
        )
        check = REPAIR.full_constraint_check_chunked(
            active_tensors,
            regression_tensors,
            protected_tensors,
            torch.tensor([[-2.0]]),
            active_margin=0.02,
            forget_regression_margin=0.02,
            chunk_size=1,
        )
        self.assertEqual(check["full_forget_regression_violation_count"], 1)
        self.assertEqual(check["newly_correct_forget_token_count"], 1)

    def test_selected_target_and_selected_competitor_are_handled_exactly(self):
        margin = REPAIR.forget_regression_margins(
            hidden=torch.tensor([[1.0]]),
            target_logits=torch.tensor([2.0]),
            strongest_unchanged_logits=torch.tensor([1.0]),
            selected_logits=torch.tensor([[2.0, 3.0]]),
            target_selected_columns=torch.tensor([0]),
            delta_rows=torch.tensor([[0.5], [-1.0]]),
        )
        # Corrected target=2.5; selected competitor=2.0; unchanged competitor=1.0.
        self.assertAlmostEqual(float(margin.item()), -0.5)

    def test_joint_optimization_preserves_baseline_incorrect_target(self):
        active_tensors = REPAIR.ActiveConstraintTensors(
            hidden=torch.tensor([[1.0, 0.0]]),
            sensitive_logits=torch.tensor([3.0]),
            best_other_logits=torch.tensor([2.5]),
            selected_row_columns=torch.tensor([0]),
        )
        regression_tensors = REPAIR.ForgetRegressionConstraintTensors(
            hidden=torch.tensor([[1.0, 1.0]]),
            target_logits=torch.tensor([1.0]),
            strongest_unchanged_logits=torch.tensor([0.0]),
            selected_logits=torch.tensor([[1.2]]),
            target_selected_columns=torch.tensor([-1]),
        )
        protected_tensors = REPAIR.ProtectedConstraintTensors(
            hidden=torch.empty((0, 2)),
            target_logits=torch.empty((0,)),
            strongest_unchanged_logits=torch.empty((0,)),
            selected_logits=torch.empty((0, 1)),
            target_selected_columns=torch.empty((0,), dtype=torch.long),
            required_margins=torch.empty((0,)),
        )
        retain_tensors = REPAIR.RetainKLTensors(
            hidden=torch.tensor([[0.0, 0.0]]),
            candidate_selected_probs=torch.tensor([[0.1]]),
            reference_selected_probs=torch.tensor([[0.1]]),
            baseline_kl=torch.zeros(1),
            record_ids=(1,),
            record_offsets=(0, 1),
        )
        args = SimpleNamespace(
            repair_rank=0,
            repair_optimizer="sgd",
            repair_lr=0.05,
            active_margin=0.02,
            forget_regression_margin=0.02,
            protected_batch_size=8,
            retain_kl_batch_size=1,
            repair_steps=200,
            progress_every=200,
            full_constraint_check_every=200,
            active_hinge_weight=2.0,
            forget_regression_hinge_weight=2.0,
            protected_hinge_weight=50.0,
            retain_kl_mu=0.0,
            delta_l2_lambda=0.0,
            stop_when_all_satisfied=False,
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            REPAIR.active,
            "make_repair_optimizer",
            side_effect=lambda module, _name, learning_rate: TinyTestOptimizer(
                module,
                learning_rate,
            ),
        ):
            delta, _, summary = REPAIR.optimize_gate_aware_delta(
                active_tensors,
                regression_tensors,
                protected_tensors,
                retain_tensors,
                selected_row_count=1,
                hidden_size=2,
                args=args,
                device=torch.device("cpu"),
                live_progress_path=Path(tmp) / "live_progress.jsonl",
            )
        active_margin = REPAIR.active_constraint_margins(
            active_tensors.hidden,
            active_tensors.sensitive_logits,
            active_tensors.best_other_logits,
            active_tensors.selected_row_columns,
            delta,
        )
        regression_margin = REPAIR.forget_regression_margins(
            regression_tensors.hidden,
            regression_tensors.target_logits,
            regression_tensors.strongest_unchanged_logits,
            regression_tensors.selected_logits,
            regression_tensors.target_selected_columns,
            delta,
        )
        self.assertGreaterEqual(float(active_margin.item()), 0.02 - 1e-5)
        self.assertGreaterEqual(float(regression_margin.item()), 0.02 - 1e-5)
        self.assertGreater(float(delta[0, 1].item()), 0.1)
        self.assertEqual(summary["newly_correct_forget_token_count"], 0)

    def test_protected_constraint_detects_edited_row_overtaking_retain_target(self):
        margin = REPAIR.protected_constraint_margins(
            hidden=torch.tensor([[1.0, 0.0]]),
            target_logits=torch.tensor([2.0]),
            strongest_unchanged_logits=torch.tensor([1.0]),
            selected_logits=torch.tensor([[0.5]]),
            target_selected_columns=torch.tensor([-1]),
            delta_rows=torch.tensor([[2.0, 0.0]]),
        )
        self.assertLess(float(margin.item()), 0.0)

    def test_protected_constraint_detects_selected_target_below_unchanged(self):
        margin = REPAIR.protected_constraint_margins(
            hidden=torch.tensor([[1.0, 0.0]]),
            target_logits=torch.tensor([2.0]),
            strongest_unchanged_logits=torch.tensor([1.8]),
            selected_logits=torch.tensor([[2.0]]),
            target_selected_columns=torch.tensor([0]),
            delta_rows=torch.tensor([[-0.5, 0.0]]),
        )
        self.assertLess(float(margin.item()), 0.0)

    def test_chunked_full_checks_match_unchunked_checks(self):
        active_tensors = REPAIR.ActiveConstraintTensors(
            hidden=torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.5]]
            ),
            sensitive_logits=torch.tensor([2.0, 1.5, 1.0, 0.5]),
            best_other_logits=torch.tensor([1.8, 1.7, 1.2, 0.8]),
            selected_row_columns=torch.tensor([0, 1, 0, 1]),
        )
        protected_tensors = REPAIR.ProtectedConstraintTensors(
            hidden=torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, 0.0],
                    [0.5, -0.5],
                ]
            ),
            target_logits=torch.tensor([2.0, 1.8, 2.1, 1.7, 1.6]),
            strongest_unchanged_logits=torch.tensor([1.5, 1.7, 1.9, 1.4, 1.5]),
            selected_logits=torch.tensor(
                [[1.0, 1.2], [1.4, 1.8], [2.1, 1.0], [1.1, 1.0], [1.5, 1.2]]
            ),
            target_selected_columns=torch.tensor([-1, 1, 0, -1, 0]),
            required_margins=torch.tensor([0.05, 0.05, 0.02, 0.05, 0.03]),
        )
        regression_tensors = REPAIR.ForgetRegressionConstraintTensors(
            hidden=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, -1.0]]),
            target_logits=torch.tensor([1.0, 1.2, 0.8]),
            strongest_unchanged_logits=torch.tensor([1.4, 1.3, 1.1]),
            selected_logits=torch.tensor([[1.5, 0.5], [0.4, 1.4], [1.0, 0.7]]),
            target_selected_columns=torch.tensor([-1, -1, 0]),
        )
        delta = torch.tensor([[0.2, -0.1], [-0.3, 0.4]])
        active_margin = 0.02
        regression_margin = 0.02
        active_margins = REPAIR.active_constraint_margins(
            active_tensors.hidden,
            active_tensors.sensitive_logits,
            active_tensors.best_other_logits,
            active_tensors.selected_row_columns,
            delta,
        )
        protected_margins = REPAIR.protected_constraint_margins(
            protected_tensors.hidden,
            protected_tensors.target_logits,
            protected_tensors.strongest_unchanged_logits,
            protected_tensors.selected_logits,
            protected_tensors.target_selected_columns,
            delta,
        )
        regression_margins = REPAIR.forget_regression_margins(
            regression_tensors.hidden,
            regression_tensors.target_logits,
            regression_tensors.strongest_unchanged_logits,
            regression_tensors.selected_logits,
            regression_tensors.target_selected_columns,
            delta,
        )
        active_slack = active_margins - active_margin
        regression_slack = regression_margins - regression_margin
        protected_slack = (
            protected_margins - protected_tensors.required_margins
        )
        chunked = REPAIR.full_constraint_check_chunked(
            active_tensors,
            regression_tensors,
            protected_tensors,
            delta,
            active_margin=active_margin,
            forget_regression_margin=regression_margin,
            chunk_size=2,
        )
        self.assertEqual(chunked["active_chunks_evaluated"], 2)
        self.assertEqual(chunked["forget_regression_chunks_evaluated"], 2)
        self.assertEqual(chunked["protected_chunks_evaluated"], 3)
        self.assertEqual(
            chunked["full_active_violation_count"],
            int(active_slack.lt(0).sum().item()),
        )
        self.assertEqual(
            chunked["full_forget_regression_violation_count"],
            int(regression_slack.lt(0).sum().item()),
        )
        self.assertEqual(
            chunked["full_protected_violation_count"],
            int(protected_slack.lt(0).sum().item()),
        )
        self.assertAlmostEqual(
            chunked["minimum_active_slack"],
            float(active_slack.min().item()),
        )
        self.assertAlmostEqual(
            chunked["minimum_forget_regression_slack"],
            float(regression_slack.min().item()),
        )
        self.assertAlmostEqual(
            chunked["minimum_protected_slack"],
            float(protected_slack.min().item()),
        )

    def test_retain_kl_zero_at_origin_and_positive_after_damage(self):
        cache = ACTIVE.RetainKLCache(
            hidden=torch.tensor([[1.0, 0.0]]),
            candidate_selected_probs=torch.tensor([[0.2]]),
            reference_selected_probs=torch.tensor([[0.2]]),
            baseline_kl=torch.tensor([0.0]),
        )
        zero = REPAIR.retain_kl_from_caches(
            [cache],
            torch.zeros((1, 2)),
        )
        damaged = REPAIR.retain_kl_from_caches(
            [cache],
            torch.tensor([[3.0, 0.0]]),
        )
        self.assertAlmostEqual(float(zero.item()), 0.0, places=7)
        self.assertGreater(float(damaged.item()), 0.0)

    def test_vectorized_retain_kl_matches_existing_low_memory_formula(self):
        hidden = torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]])
        probabilities = torch.tensor([[0.2], [0.3], [0.1]])
        delta = torch.tensor([[0.7, -0.2]])
        tensors = REPAIR.RetainKLTensors(
            hidden=hidden,
            candidate_selected_probs=probabilities,
            reference_selected_probs=probabilities.clone(),
            baseline_kl=torch.zeros(3),
            record_ids=(10, 20),
            record_offsets=(0, 2, 3),
        )
        legacy = [
            ACTIVE.RetainKLCache(
                hidden=hidden[:2],
                candidate_selected_probs=probabilities[:2],
                reference_selected_probs=probabilities[:2].clone(),
                baseline_kl=torch.zeros(2),
            ),
            ACTIVE.RetainKLCache(
                hidden=hidden[2:],
                candidate_selected_probs=probabilities[2:],
                reference_selected_probs=probabilities[2:].clone(),
                baseline_kl=torch.zeros(1),
            ),
        ]
        expected = ACTIVE.retain_kl_from_caches(legacy, delta)
        actual = REPAIR.retain_kl_from_tensors(tensors, delta)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7, rtol=1e-7))

    def test_live_progress_rows_contain_required_fields(self):
        active_tensors = REPAIR.ActiveConstraintTensors(
            hidden=torch.tensor([[1.0, 0.0]]),
            sensitive_logits=torch.tensor([1.0]),
            best_other_logits=torch.tensor([0.5]),
            selected_row_columns=torch.tensor([0]),
        )
        regression_tensors = REPAIR.ForgetRegressionConstraintTensors(
            hidden=torch.tensor([[1.0, 0.0]]),
            target_logits=torch.tensor([0.5]),
            strongest_unchanged_logits=torch.tensor([1.0]),
            selected_logits=torch.tensor([[1.0]]),
            target_selected_columns=torch.tensor([-1]),
        )
        protected_tensors = REPAIR.ProtectedConstraintTensors(
            hidden=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            target_logits=torch.tensor([2.0, 2.0]),
            strongest_unchanged_logits=torch.tensor([1.0, 1.0]),
            selected_logits=torch.tensor([[0.5], [0.5]]),
            target_selected_columns=torch.tensor([-1, -1]),
            required_margins=torch.tensor([0.05, 0.05]),
        )
        retain_tensors = REPAIR.RetainKLTensors(
            hidden=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            candidate_selected_probs=torch.tensor([[0.2], [0.2]]),
            reference_selected_probs=torch.tensor([[0.2], [0.2]]),
            baseline_kl=torch.zeros(2),
            record_ids=(10, 20),
            record_offsets=(0, 1, 2),
        )
        args = SimpleNamespace(
            repair_rank=0,
            repair_optimizer="adamw",
            repair_lr=1e-3,
            active_margin=0.02,
            forget_regression_margin=0.02,
            protected_batch_size=1,
            retain_kl_batch_size=1,
            repair_steps=1,
            progress_every=10,
            full_constraint_check_every=100,
            active_hinge_weight=2.0,
            forget_regression_hinge_weight=2.0,
            protected_hinge_weight=50.0,
            retain_kl_mu=10.0,
            delta_l2_lambda=1e-4,
            stop_when_all_satisfied=False,
        )
        with tempfile.TemporaryDirectory() as tmp:
            progress_path = Path(tmp) / "optimization" / "live_progress.jsonl"
            with mock.patch.object(
                REPAIR.active,
                "make_repair_optimizer",
                side_effect=lambda module, _name, learning_rate: TinyTestOptimizer(
                    module,
                    learning_rate,
                ),
            ):
                REPAIR.optimize_gate_aware_delta(
                    active_tensors,
                    regression_tensors,
                    protected_tensors,
                    retain_tensors,
                    selected_row_count=1,
                    hidden_size=2,
                    args=args,
                    device=torch.device("cpu"),
                    live_progress_path=progress_path,
                )
            rows = [
                json.loads(line)
                for line in progress_path.read_text(encoding="utf-8").splitlines()
            ]
        self.assertEqual(len(rows), 1)
        self.assertTrue(REPAIR.LIVE_PROGRESS_REQUIRED_FIELDS <= rows[0].keys())
        self.assertEqual(rows[0]["step"], 1)
        self.assertEqual(rows[0]["total_steps"], 1)
        self.assertIsNotNone(rows[0]["full_active_violation_count"])
        self.assertIsNotNone(
            rows[0]["full_forget_regression_violation_count"]
        )
        self.assertIsNotNone(rows[0]["full_protected_violation_count"])

    def test_interrupted_run_emits_receipt_but_no_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            state = REPAIR.InterruptionState(
                output_dir=output,
                latest_completed_step=17,
                phase="optimization",
            )

            def interrupt_after_checkpoint_materialization():
                checkpoint = output / "selected_checkpoint"
                checkpoint.mkdir()
                (checkpoint / "partial.bin").write_bytes(b"partial")
                raise KeyboardInterrupt

            with self.assertRaises(SystemExit) as raised:
                REPAIR.execute_with_interrupt_receipt(
                    interrupt_after_checkpoint_materialization,
                    state,
                )
            self.assertEqual(raised.exception.code, 130)
            self.assertFalse((output / "selected_checkpoint").exists())
            receipt = json.loads(
                (output / "optimization" / "interrupted.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(receipt["latest_completed_step"], 17)
            self.assertEqual(receipt["phase_at_interrupt"], "optimization")
            self.assertFalse(receipt["selected_checkpoint_emitted"])

    def test_newly_correct_reporting_contains_exact_identity_fields(self):
        case = REPAIR.zsre.PredictionCase(
            case_id=17072,
            prompt_type="paraphrase",
            prompt_index=2,
            token_index=1,
            prompt="Who wrote it?",
            target_text=" Bachman",
        )
        tok = SimpleNamespace(decode=lambda ids: f"token-{ids[0]}")
        row = REPAIR.newly_correct_forget_case_payload(
            case,
            tok,
            scale=0.75,
            target_token_id=41,
            predicted_token_id=41,
            competitor_token_id=9,
            target_logit=3.5,
            competitor_logit=3.25,
        )
        expected = {
            "case_id": 17072,
            "prompt_type": "paraphrase",
            "prompt_index": 2,
            "token_index": 1,
            "target_token_id": 41,
            "target_token": "token-41",
            "predicted_token_id": 41,
            "target_logit": 3.5,
            "competitor_logit": 3.25,
            "margin": -0.25,
        }
        for key, value in expected.items():
            self.assertEqual(row[key], value)

    def test_candidate_with_newly_correct_token_cannot_pass(self):
        setting = metric_result(forget_eff=10.0, forget_gen=12.0)
        candidate = metric_result(ppl=10.1)
        gate = REPAIR.full_gate_report(
            setting,
            candidate,
            forget_guard=forget_guard(newly_correct=1),
        )
        selected = REPAIR.select_all_gates_candidate(
            [
                {
                    "scale": 0.75,
                    "materialized_delta_norm": 1.0,
                    "full_gate": gate,
                }
            ],
            setting5_target_already_met=False,
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(
            gate["non_ppl"]["forget_guard"]["checks"][
                "newly_correct_forget_tokens"
            ]["passed"]
        )
        self.assertIsNone(selected)

    def test_candidate_selection_rejects_zero_forgetting_with_utility_damage(self):
        setting = metric_result()
        damaged = metric_result(retain_eff=89.0)
        gate = REPAIR.full_gate_report(
            setting,
            damaged,
            forget_guard=forget_guard(),
        )
        selected = REPAIR.select_all_gates_candidate(
            [
                {
                    "scale": 0.5,
                    "materialized_delta_norm": 1.0,
                    "full_gate": gate,
                }
            ],
            setting5_target_already_met=False,
        )
        self.assertFalse(gate["passed"])
        self.assertIsNone(selected)

    def test_candidate_selection_accepts_only_all_gates_passing_candidate(self):
        setting = metric_result(forget_eff=10.0, forget_gen=12.0)
        passing = metric_result(ppl=10.1)
        failing = metric_result(retain_spe=87.0, ppl=10.1)
        candidates = [
            {
                "scale": 0.5,
                "materialized_delta_norm": 0.8,
                "full_gate": REPAIR.full_gate_report(
                    setting,
                    failing,
                    forget_guard=forget_guard(),
                ),
            },
            {
                "scale": 0.75,
                "materialized_delta_norm": 1.0,
                "full_gate": REPAIR.full_gate_report(
                    setting,
                    passing,
                    forget_guard=forget_guard(),
                ),
            },
        ]
        selected = REPAIR.select_all_gates_candidate(
            candidates,
            setting5_target_already_met=False,
        )
        self.assertIsNotNone(selected)
        self.assertEqual(selected["scale"], 0.75)

    def test_rejected_seed_emits_no_selected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            gate = REPAIR.full_gate_report(
                metric_result(forget_eff=10.0, forget_gen=12.0),
                metric_result(ppl=10.1),
                forget_guard=forget_guard(regression_violations=1),
            )
            selected_gate = REPAIR.select_all_gates_candidate(
                [
                    {
                        "scale": 0.75,
                        "materialized_delta_norm": 1.0,
                        "full_gate": gate,
                    }
                ],
                setting5_target_already_met=False,
            )
            result = REPAIR.save_selected_checkpoint_if_accepted(
                accepted=selected_gate is not None,
                model=object(),
                tok=object(),
                output_dir=output,
            )
            final_result = {
                "selected": None if selected_gate is None else {"Eff": 0.0}
            }
            self.assertIsNone(selected_gate)
            self.assertIsNone(result)
            self.assertIsNone(final_result["selected"])
            self.assertFalse((output / "selected_checkpoint").exists())

    def test_zsre_aggregate_uses_sample_standard_deviation(self):
        mean, sample_sd = AGG.mean_std([0.0, 2.0])
        self.assertEqual(mean, 1.0)
        self.assertAlmostEqual(sample_sd, math.sqrt(2.0))


if __name__ == "__main__":
    unittest.main()
