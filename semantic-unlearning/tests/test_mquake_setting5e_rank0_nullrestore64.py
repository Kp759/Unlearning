from __future__ import annotations

import inspect
import json
import math
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

import mquake_gagd_setting5e_multiroot_active_repair as vectorized
import mquake_setting5e_detied_baseanchored_minrank as basearch
import mquake_setting5e_rank0_nullrestore64 as method
import mquake_zero_unlearn_official_eval as mquake


def prediction_case(case_id: int = 1) -> mquake.PredictionCase:
    return mquake.PredictionCase(
        case_id=case_id,
        prompt_type="rewrite",
        prompt_index=0,
        token_index=0,
        prompt="Subject was born in",
        target_text=" place",
    )


def token_state(
    *,
    case_id: int = 1,
    target: int = 1,
    predicted: int = 1,
    runner: int = 2,
    target_logit: float = 1.0,
    runner_logit: float = 0.0,
) -> method.TokenState:
    return method.TokenState(
        case=prediction_case(case_id),
        hidden=torch.tensor([1.0, 0.0]),
        target_token_id=target,
        predicted_token_id=predicted,
        runner_up_token_id=runner,
        target_logit=target_logit,
        runner_up_logit=runner_logit,
    )


class TinyTiedLM(nn.Module):
    def __init__(self, vocab: int = 13, hidden: int = 8) -> None:
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=True)
        self.embed = nn.Embedding(vocab, hidden)
        self.transform = nn.Linear(hidden, hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, layer):
        self.lm_head = layer


class BaseOriginAndGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(23)

    def test_base_repair_origin_is_exact_base(self) -> None:
        model = TinyTiedLM()
        base, _ = basearch.snapshot_base_weight(model)
        transformer_hash = method.transformer_sha256(model)
        with torch.no_grad():
            model.get_input_embeddings().weight.add_(0.25)
        basearch.detie_restore_base_embeddings(model, base)
        basearch.restore_full_output_to_base(model, base)
        report = method.base_repair_origin_report(model, base, transformer_hash)
        self.assertTrue(report["input_equals_base_exactly"])
        self.assertTrue(report["output_equals_base_exactly"])
        self.assertTrue(report["transformer_equals_base_exactly"])
        self.assertTrue(report["input_output_pointers_distinct"])
        self.assertFalse(report["tie_word_embeddings"])

    def test_robust_svd_qr_forget_basis(self) -> None:
        rows = torch.randn(12, 20)
        result = method.robust_svd_qr_row_basis(rows)
        repeated = method.robust_svd_qr_row_basis(rows)
        identity = torch.eye(result.basis.shape[0])
        torch.testing.assert_close(
            result.basis @ result.basis.T, identity, atol=2e-5, rtol=2e-5
        )
        self.assertIn("reduced QR", result.report["construction"])
        self.assertLessEqual(result.report["final_orthogonality_max_error"], 2e-5)
        self.assertTrue(torch.equal(result.basis, repeated.basis))

    def test_forget_span_projection_preserves_visible_logits(self) -> None:
        hidden = torch.randn(10, 18)
        basis = method.robust_svd_qr_row_basis(hidden).basis
        delta = torch.randn(7, 18)
        projected = method.project_delta_to_forget_span(delta, basis)
        torch.testing.assert_close(
            hidden @ delta.T, hidden @ projected.T, atol=3e-5, rtol=3e-5
        )

    def test_utility_rank64_basis_is_forget_orthogonal(self) -> None:
        forget_basis = torch.eye(80)[:8]
        utility = torch.eye(80)
        _, restore, report = method.build_restore_basis64(utility, forget_basis)
        self.assertEqual(tuple(restore.shape), (64, 80))
        self.assertEqual(report["restore_rank"], 64)
        torch.testing.assert_close(
            restore @ forget_basis.T,
            torch.zeros(64, 8),
            atol=2e-5,
            rtol=0,
        )

    def test_arbitrary_rank64_coefficients_cannot_change_forget_logits(self) -> None:
        forget_basis = torch.eye(80)[:9]
        _, restore, _ = method.build_restore_basis64(torch.eye(80), forget_basis)
        coefficients = torch.randn(6, 64)
        self.assertEqual(coefficients.shape[1], 64)
        self.assertEqual(restore.shape[0], 64)
        delta = method.rank64_restoration_delta(coefficients, restore)
        forget_hidden = torch.randn(14, 9) @ forget_basis
        torch.testing.assert_close(
            forget_hidden @ delta.T,
            torch.zeros(14, 6),
            atol=3e-5,
            rtol=0,
        )

    def test_forget_geometry_fails_when_exact_nullspace_below_64(self) -> None:
        report = {
            "hidden_size": 70,
            "state_count": 20,
            "numerical_rank": 10,
            "exact_nullspace_dimension": 60,
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "below 64"):
                method.require_forget_nullspace(report, Path(directory))
            payload = json.loads(
                (Path(directory) / "geometry_infeasible.json").read_text()
            )
        self.assertEqual(payload["requested_restore_rank"], 64)
        self.assertFalse(payload["rank_fallback_attempted"])

    def test_utility_geometry_fails_instead_of_rank_fallback(self) -> None:
        forget_basis = torch.eye(80)[:20]
        utility = torch.randn(30, 80)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "below 64"):
                method.build_restore_basis64(
                    utility,
                    forget_basis,
                    output_dir=Path(directory),
                    forget_report={
                        "hidden_size": 80,
                        "state_count": 100,
                        "numerical_rank": 20,
                        "exact_nullspace_dimension": 60,
                    },
                )
            payload = json.loads(
                (Path(directory) / "geometry_infeasible.json").read_text()
            )
        self.assertEqual(payload["requested_restore_rank"], 64)
        self.assertFalse(payload["rank_fallback_attempted"])


class Rank0AndUtilityMathTests(unittest.TestCase):
    def test_stage_a_rank0_parameterization_starts_at_base(self) -> None:
        initial = torch.zeros(3, 5)
        tensors = vectorized.ActivePairTensors(
            hidden=torch.randn(2, 5),
            base_margin=torch.tensor([0.5, 0.6]),
            sensitive_row_index=torch.tensor([0, 1]),
            competitor_row_index=torch.tensor([1, 2]),
            hidden_rank=torch.randn(2, 5),
        )
        result = method.optimize_rank0_hard_tail(
            tensors,
            initial,
            steps=2,
            learning_rate=method.RANK0_LR,
            active_margin=method.ACTIVE_MARGIN,
            delta_l2=method.RANK0_DELTA_L2,
        )
        self.assertEqual(result.report["repair_rank"], 0)
        self.assertTrue(torch.equal(result.delta, initial))

    def test_dynamic_constraint_uses_true_current_runner_up(self) -> None:
        current = token_state(
            target=1,
            predicted=1,
            runner=7,
            target_logit=2.0,
            runner_logit=1.9,
        )
        constraints = method.constraints_from_audit(
            [current], margin=0.25, generation_round=4
        )
        self.assertEqual(len(constraints), 1)
        self.assertEqual(constraints[0].competitor_token_id, 7)
        self.assertEqual(method.selected_row_ids(constraints), [1, 7])

    def test_exact_selected_row_ppl_cache_matches_full_vocabulary(self) -> None:
        torch.manual_seed(11)
        base = torch.randn(7, 13)
        selected = torch.tensor([1, 5, 9])
        candidate = base.clone()
        candidate[:, selected] += torch.randn(7, 3) * 0.2
        targets = torch.tensor([1, 2, 5, 4, 9, 6, 7])
        lookup = {int(value): index for index, value in enumerate(selected)}
        target_index = torch.tensor([lookup.get(int(value), -1) for value in targets])
        cached = basearch.exact_selected_row_mean_nll_from_delta_logits(
            base_logsumexp=torch.logsumexp(base, dim=1),
            base_target_logits=base.gather(1, targets[:, None]).squeeze(1),
            base_selected_logits=base.index_select(1, selected),
            candidate_selected_logits=candidate.index_select(1, selected),
            target_selected_row_index=target_index,
            normalization_divisor=8,
        )
        direct = (
            torch.logsumexp(candidate, dim=1)
            - candidate.gather(1, targets[:, None]).squeeze(1)
        ).sum() / 8
        torch.testing.assert_close(cached, direct, atol=2e-5, rtol=2e-5)

    def test_base_relative_ppl_ratio_equivalence(self) -> None:
        base_nll = 2.4
        for offset in (0.0, math.log(1.02), math.log(1.02) + 1e-4):
            candidate = base_nll + offset
            ratio = math.exp(candidate) / math.exp(base_nll) <= 1.02
            self.assertEqual(
                ratio,
                basearch.ppl_ratio_gate_equivalent(base_nll, candidate, 1.02),
            )

    def test_protected_correct_selected_row_is_handled_relatively(self) -> None:
        protected = vectorized.ProtectedPairTensors(
            hidden=torch.empty(1, 3),
            base_modified_logits=torch.tensor([[1.0, 2.0]]),
            correct_base=torch.tensor([2.0]),
            correct_modified_row_index=torch.tensor([1]),
            competitor_mask=torch.tensor([[True, False]]),
            hidden_rank=torch.tensor([[1.0] + [0.0] * 63]),
        )
        tensors = method.ProtectedRestoreTensors(
            protected=protected,
            fixed_delta_logits=torch.zeros(1, 2),
            runner_base_logits=torch.tensor([1.0]),
            runner_selected_row_index=torch.tensor([0]),
        )
        coefficients = torch.zeros(2, 64)
        coefficients[1, 0] = 0.5
        margins = method.protected_restore_margins(tensors, coefficients)
        torch.testing.assert_close(margins, torch.tensor([1.5, 1.5]))

    def test_bf16_forget_audit_runs_after_rank64_materialization(self) -> None:
        states = [
            token_state(predicted=2, target=1, runner_logit=1.2, target_logit=1.0),
            token_state(case_id=2, predicted=3, target=1, runner=3, runner_logit=1.3, target_logit=1.0),
        ]
        report = method.bf16_forget_audit(states, selection_margin=0.10)
        self.assertTrue(report["passed"])
        self.assertEqual(report["active_violation_count"], 0)

    def test_nonselected_rows_remain_exact_base(self) -> None:
        base = torch.randn(12, 7, dtype=torch.bfloat16)
        candidate = base.clone()
        candidate[torch.tensor([2, 8])] += torch.tensor(0.5, dtype=torch.bfloat16)
        self.assertTrue(method.nonselected_rows_equal_base(candidate, base, [2, 8]))
        candidate[4, 1] += torch.tensor(0.25, dtype=torch.bfloat16)
        self.assertFalse(method.nonselected_rows_equal_base(candidate, base, [2, 8]))


class ProtocolAndImplementationTests(unittest.TestCase):
    def test_no_rank16_fallback_or_rank_sweep_exists(self) -> None:
        source = method.Path(method.__file__).read_text(encoding="utf-8")
        self.assertNotIn("RESTORE_RANK = 16", source)
        self.assertNotIn("rank16", source.lower())
        self.assertNotIn("for rank in range", source)
        config = json.loads(
            (ROOT / "config/official_benchmarks/mquake_setting5e_rank0_nullrestore64.json").read_text()
        )
        self.assertEqual(config["geometry"]["restore_rank"], 64)
        self.assertFalse(config["geometry"]["rank_fallback_or_sweep"])
        args = method.build_parser().parse_args(
            ["--model-path", "model", "--output-dir", "output"]
        )
        args.restore_rank = 16
        with self.assertRaisesRegex(ValueError, "restore_rank=64"):
            method.validate_args(args)

    def test_held_out_firewall_blocks_rejected_candidate(self) -> None:
        loader = mock.Mock(side_effect=AssertionError("held-out loader called"))
        evaluator = mock.Mock(side_effect=AssertionError("held-out evaluator called"))
        with self.assertRaises(RuntimeError):
            method.evaluate_held_out_after_durable_acceptance(
                accepted=False, load_records=loader, evaluate=evaluator
            )
        loader.assert_not_called()
        evaluator.assert_not_called()
        source = inspect.getsource(method.main)
        self.assertLess(
            source.index('gagd.write_json(output_dir / "selection_commit.json"'),
            source.index("evaluate_held_out_after_durable_acceptance"),
        )

    def test_optimizer_hot_paths_are_vectorized(self) -> None:
        rank0 = inspect.getsource(method.optimize_rank0_hard_tail)
        restore = inspect.getsource(method.optimize_rank64_restoration)
        for source in (rank0, restore):
            self.assertNotIn("for pair in", source)
            self.assertNotIn("for state in", source)
        self.assertIn("active_pair_margins_from_coefficients", rank0)
        self.assertIn("protected_restore_margins", restore)
        self.assertIn("exact_restore_ppl_mean_nll", restore)

    def test_pinned_launcher_and_protocol(self) -> None:
        launcher = (
            SCRIPTS / "run_mquake_setting5e_rank0_nullrestore64.sh"
        ).read_text(encoding="utf-8")
        for fragment in (
            "--steps 600",
            "--emb-lm-lr 1e-4",
            "--forget-sampling instance_balanced",
            "--rank0-lr 5e-3",
            "--restore-rank 64",
            "--max-ppl-ratio 1.02",
        ):
            self.assertIn(fragment, launcher)
        self.assertEqual(method.RESTORE_RANK, 64)
        self.assertEqual(mquake.MQUAKE_REV, method.DATASET_REVISION)


if __name__ == "__main__":
    unittest.main()
