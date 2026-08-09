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

import mquake_setting5e_detied_baseanchored_minrank as method
import mquake_gagd_setting5e_multiroot_active_repair as vectorized
import mquake_zero_unlearn_official_eval as mquake


def case(case_id: int = 1, token_index: int = 0) -> mquake.PredictionCase:
    return mquake.PredictionCase(
        case_id=case_id,
        prompt_type="rewrite",
        prompt_index=0,
        token_index=token_index,
        prompt="Subject was born in",
        target_text=" place",
    )


def state(
    *,
    case_id: int = 1,
    hidden=(1.0, 0.0),
    target=1,
    predicted=1,
    runner=2,
    target_logit=1.0,
    runner_logit=0.0,
) -> method.TokenState:
    return method.TokenState(
        case=case(case_id),
        hidden=torch.tensor(hidden, dtype=torch.float32),
        target_token_id=target,
        predicted_token_id=predicted,
        runner_up_token_id=runner,
        target_logit=target_logit,
        runner_up_logit=runner_logit,
    )


class DetieTests(unittest.TestCase):
    class TinyTiedLM(nn.Module):
        def __init__(self, vocab_size=19, hidden_size=8, *, tied=True):
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=bool(tied))
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.transform = nn.Linear(hidden_size, hidden_size, bias=False)
            self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
            if tied:
                self.lm_head.weight = self.embed.weight

        def get_input_embeddings(self):
            return self.embed

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, layer):
            self.lm_head = layer

        def forward(self, input_ids, output_hidden_states=False):
            hidden = self.transform(self.embed(input_ids))
            output = SimpleNamespace(logits=self.lm_head(hidden))
            if output_hidden_states:
                output.hidden_states = (hidden,)
            return output

        def save_pretrained(self, directory):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text(
                json.dumps(
                    {
                        "vocab_size": self.embed.num_embeddings,
                        "hidden_size": self.embed.embedding_dim,
                        "tie_word_embeddings": self.config.tie_word_embeddings,
                    }
                )
            )
            torch.save(self.state_dict(), directory / "model.pt")

        @classmethod
        def from_pretrained(cls, directory):
            directory = Path(directory)
            config = json.loads((directory / "config.json").read_text())
            model = cls(
                config["vocab_size"],
                config["hidden_size"],
                tied=config["tie_word_embeddings"],
            )
            model.load_state_dict(
                torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
            )
            return model

    def _tiny_gpt2(self):
        torch.manual_seed(7)
        return self.TinyTiedLM().eval()

    def test_detie_restores_base_input_and_preserves_setting5_output(self):
        model = self._tiny_gpt2()
        base = model.get_input_embeddings().weight.detach().clone()
        self.assertEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )
        with torch.no_grad():
            model.get_input_embeddings().weight.add_(0.125)
        setting5 = model.get_output_embeddings().weight.detach().clone()
        report = method.detie_restore_base_embeddings(model, base.cpu(), chunk_rows=5)
        self.assertNotEqual(
            model.get_input_embeddings().weight.data_ptr(),
            model.get_output_embeddings().weight.data_ptr(),
        )
        self.assertTrue(torch.equal(model.get_input_embeddings().weight, base))
        self.assertTrue(torch.equal(model.get_output_embeddings().weight, setting5))
        self.assertTrue(report["input_equals_base_exactly"])
        self.assertTrue(report["output_equals_pre_detie_setting5_exactly"])
        self.assertFalse(model.config.tie_word_embeddings)

    def test_save_reload_preserves_untied_state(self):
        model = self._tiny_gpt2()
        base = model.get_input_embeddings().weight.detach().clone()
        with torch.no_grad():
            model.get_input_embeddings().weight[3].add_(0.5)
        setting5 = model.get_output_embeddings().weight.detach().clone()
        method.detie_restore_base_embeddings(model, base.cpu())
        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            reloaded = self.TinyTiedLM.from_pretrained(directory).eval()
        self.assertFalse(reloaded.config.tie_word_embeddings)
        self.assertNotEqual(
            reloaded.get_input_embeddings().weight.data_ptr(),
            reloaded.get_output_embeddings().weight.data_ptr(),
        )
        self.assertTrue(torch.equal(reloaded.get_input_embeddings().weight, base))
        self.assertTrue(torch.equal(reloaded.get_output_embeddings().weight, setting5))

    def test_restored_base_input_reproduces_base_hidden_state(self):
        model = self._tiny_gpt2()
        ids = torch.tensor([[2, 4, 6, 8]])
        with torch.no_grad():
            base_hidden = model(ids, output_hidden_states=True).hidden_states[-1]
        base = model.get_input_embeddings().weight.detach().clone()
        with torch.no_grad():
            model.get_input_embeddings().weight.add_(0.25)
        method.detie_restore_base_embeddings(model, base.cpu())
        with torch.no_grad():
            detied_hidden = model(ids, output_hidden_states=True).hidden_states[-1]
        self.assertTrue(torch.equal(base_hidden, detied_hidden))

    def test_candidate_origin_restores_the_complete_base_output_matrix(self):
        model = self._tiny_gpt2()
        base = model.get_input_embeddings().weight.detach().clone()
        with torch.no_grad():
            model.get_input_embeddings().weight.add_(0.25)
        method.detie_restore_base_embeddings(model, base.cpu())
        self.assertFalse(torch.equal(model.get_output_embeddings().weight, base))
        method.restore_full_output_to_base(model, base.cpu(), chunk_rows=4)
        self.assertTrue(torch.equal(model.get_output_embeddings().weight, base))


class BaseAnchoredParameterizationTests(unittest.TestCase):
    def test_zero_coefficients_equal_base_rows(self):
        basis = torch.eye(3)[:2]
        module = method.BaseAnchoredLowRankRows(2, basis, device=torch.device("cpu"))
        base = torch.randn(2, 3)
        self.assertTrue(torch.equal(module.candidate_rows(base), base))

    def test_candidate_is_base_plus_coefficients_times_basis(self):
        basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        coefficients = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
        module = method.BaseAnchoredLowRankRows(
            2, basis, device=torch.device("cpu"), initial_coefficients=coefficients
        )
        base = torch.ones(2, 3)
        self.assertTrue(
            torch.equal(module.candidate_rows(base), base + coefficients @ basis)
        )

    def test_setting5_projection_is_exact_least_squares_projection(self):
        basis = torch.eye(4)[:2]
        base = torch.randn(3, 4)
        setting5 = base + torch.tensor(
            [[1.0, 2.0, 9.0, 8.0], [3.0, 4.0, 7.0, 6.0], [5.0, 6.0, 5.0, 4.0]]
        )
        coefficients = method.project_setting5_initialization(setting5, base, basis)
        expected = (setting5 - base)[:, :2]
        self.assertTrue(torch.allclose(coefficients, expected))


class VectorizedConstraintTests(unittest.TestCase):
    def test_active_margin_matches_explicit_reference(self):
        torch.manual_seed(1)
        coefficients = torch.randn(4, 2)
        hidden_rank = torch.randn(5, 2)
        sensitive = torch.tensor([0, 1, 1, 2, 3])
        competitor = torch.tensor([1, 2, 3, 3, 0])
        base_margin = torch.randn(5)
        tensors = vectorized.ActivePairTensors(
            hidden=torch.empty(5, 7),
            base_margin=base_margin,
            sensitive_row_index=sensitive,
            competitor_row_index=competitor,
            hidden_rank=hidden_rank,
        )
        actual = vectorized.active_pair_margins_from_coefficients(tensors, coefficients)
        expected = torch.stack(
            [
                base_margin[i]
                + hidden_rank[i] @ coefficients[competitor[i]]
                - hidden_rank[i] @ coefficients[sensitive[i]]
                for i in range(5)
            ]
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_protected_margin_includes_modified_correct_row(self):
        # Correct row is selected at column 1 for the first state and unselected
        # for the second state.
        delta_logits = torch.tensor([[0.2, 0.5, -0.1], [0.1, -0.3, 0.4]])
        tensors = vectorized.ProtectedPairTensors(
            hidden=torch.empty(2, 4),
            base_modified_logits=torch.tensor([[1.0, 1.5, 0.5], [0.5, 0.7, 0.8]]),
            correct_base=torch.tensor([1.5, 1.1]),
            correct_modified_row_index=torch.tensor([1, -1]),
            competitor_mask=torch.tensor([[True, False, True], [True, True, True]]),
        )
        actual = vectorized.protected_pair_margins_from_delta_logits(tensors, delta_logits)
        correct_after = torch.tensor([2.0, 1.1])
        all_margins = correct_after[:, None] - (tensors.base_modified_logits + delta_logits)
        expected = all_margins[tensors.competitor_mask]
        self.assertTrue(torch.allclose(actual, expected))

    def test_base_runner_up_protects_a_modified_correct_row(self):
        tensors = vectorized.ProtectedPairTensors(
            hidden=torch.empty(1, 2),
            base_modified_logits=torch.tensor([[2.0, 0.5]]),
            correct_base=torch.tensor([2.0]),
            correct_modified_row_index=torch.tensor([0]),
            competitor_mask=torch.tensor([[False, True]]),
        )
        # Lowering the selected correct row by 0.7 crosses the unmodified Base
        # runner-up at 1.5 even though the other selected row remains harmless.
        delta_logits = torch.tensor([[-0.7, 0.0]])
        margins = method.base_protected_margins_from_delta_logits(
            tensors, delta_logits, torch.tensor([1.5])
        )
        self.assertTrue(torch.allclose(margins, torch.tensor([0.8, -0.2])))

    def test_hot_path_has_no_pair_or_protected_state_loop(self):
        source = inspect.getsource(method.solve_vectorized_phase)
        self.assertNotIn("for pair in", source)
        self.assertNotIn("for state in", source)
        self.assertIn("for step in range", source)
        self.assertIn("active_pair_margins_from_coefficients", source)
        self.assertIn("protected_delta_logits_from_coefficients", source)


class ExactPPLCacheTests(unittest.TestCase):
    def _compare(self, dtype: torch.dtype, target_selected: bool):
        torch.manual_seed(4)
        token_count, vocab = 7, 11
        base = torch.randn(token_count, vocab).to(dtype).float()
        selected = torch.tensor([1, 4, 7])
        delta = torch.randn(token_count, len(selected)) * 0.1
        candidate = base.clone()
        updated_selected = base.index_select(1, selected) + delta
        if dtype == torch.bfloat16:
            updated_selected = updated_selected.to(torch.bfloat16).float()
        candidate[:, selected] = updated_selected
        targets = torch.tensor([1 if target_selected else 2, 3, 4, 5, 6, 7, 8])
        target_lookup = {int(token_id): index for index, token_id in enumerate(selected)}
        target_index = torch.tensor([target_lookup.get(int(value), -1) for value in targets])
        divisor = token_count + 1
        cached = method.exact_selected_row_mean_nll_from_delta_logits(
            base_logsumexp=torch.logsumexp(base, dim=1),
            base_target_logits=base.gather(1, targets[:, None]).squeeze(1),
            base_selected_logits=base.index_select(1, selected),
            candidate_selected_logits=candidate.index_select(1, selected),
            target_selected_row_index=target_index,
            normalization_divisor=divisor,
        )
        direct = (
            torch.logsumexp(candidate, dim=1)
            - candidate.gather(1, targets[:, None]).squeeze(1)
        ).sum() / divisor
        self.assertTrue(torch.allclose(cached, direct, atol=2e-5, rtol=2e-5))

    def test_exact_cache_target_selected(self):
        self._compare(torch.float32, True)

    def test_exact_cache_target_unselected(self):
        self._compare(torch.float32, False)

    def test_exact_cache_bf16_materialized_logits(self):
        self._compare(torch.bfloat16, True)

    def test_ppl_ratio_and_mean_nll_gate_are_equivalent(self):
        for delta in (-0.1, 0.0, math.log(1.02), math.log(1.02) + 1e-4):
            base_nll = 2.4
            candidate_nll = base_nll + delta
            ratio_check = math.exp(candidate_nll) / math.exp(base_nll) <= 1.02
            self.assertEqual(
                ratio_check,
                method.ppl_ratio_gate_equivalent(base_nll, candidate_nll, 1.02),
            )


class RankContinuationTests(unittest.TestCase):
    def test_ordered_basis_prefixes_are_exact_and_deterministic(self):
        torch.manual_seed(9)
        rows = torch.randn(8, 5)
        full1, report1 = method.ordered_orthonormal_row_basis(rows)
        full2, report2 = method.ordered_orthonormal_row_basis(rows)
        self.assertTrue(torch.equal(full1, full2))
        self.assertTrue(torch.equal(full1[:2], full1[:3][:2]))
        self.assertTrue(torch.equal(full1[:3], full1[:4][:3]))
        self.assertEqual(report1["ordered_basis_sha256"], report2["ordered_basis_sha256"])
        self.assertTrue(report1["prefix_nesting_verified"])

    def test_rank_warm_start_preserves_old_columns_and_zeros_new(self):
        old = torch.randn(4, 2)
        expanded = method.expand_rank_coefficients(old, 3)
        self.assertTrue(torch.equal(expanded[:, :2], old))
        self.assertTrue(torch.equal(expanded[:, 2], torch.zeros(4)))

    def test_rank_diagnostics_have_one_summary_row_per_attempted_rank(self):
        phases = [
            {"rank": 2, "steps": 10, "wall_clock_seconds": 1.0, "active_violations": 3},
            {"rank": 2, "steps": 5, "wall_clock_seconds": 0.5, "active_violations": 1},
            {"rank": 3, "steps": 7, "wall_clock_seconds": 0.7, "active_violations": 0},
        ]
        summary = method.aggregate_rank_attempts(phases)
        self.assertEqual([row["rank"] for row in summary], [2, 3])
        self.assertEqual(summary[0]["steps"], 15)
        self.assertEqual(summary[0]["constraint_generation_phase_count"], 2)

    def test_stalled_rank_diagnostics_are_written_deterministically(self):
        phases = [
            {
                "rank": 2,
                "steps": 12,
                "wall_clock_seconds": 0.0,
                "configured_feasible": False,
                "convergence_or_stall_reason": "deterministic_stall_criterion_reached",
                "active_violations": 2,
                "protected_violations": 0,
                "ppl_nll_excess": 0.0,
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank_continuation.json"
            payload = method.write_rank_continuation_diagnostics(
                path,
                rank_start=2,
                rank_max_resolved=2,
                first_local_feasible_rank=None,
                basis_report={"ordered_basis_sha256": "abc", "prefix_nesting_verified": True},
                phase_reports=phases,
            )
            reread = json.loads(path.read_text())
        self.assertIsNone(payload["first_local_exact_feasible_rank"])
        self.assertEqual(reread["attempts"][0]["rank"], 2)
        self.assertEqual(
            reread["attempts"][0]["convergence_or_stall_reason"],
            "deterministic_stall_criterion_reached",
        )

    def test_row_expansion_preserves_old_rows_and_zeros_new(self):
        old_ids = [2, 7]
        new_ids = [1, 2, 7, 9]
        old = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        expanded = method.expand_row_coefficients(old_ids, new_ids, old)
        self.assertTrue(torch.equal(expanded[1], old[0]))
        self.assertTrue(torch.equal(expanded[2], old[1]))
        self.assertTrue(torch.equal(expanded[[0, 3]], torch.zeros(2, 2)))

    def test_reactivated_state_adds_true_current_runner_up(self):
        previously_solved = state(
            target=1,
            predicted=1,
            runner=5,
            target_logit=2.0,
            runner_logit=1.95,
        )
        additions = method.constraints_from_audit(
            [previously_solved], margin=0.10, generation_round=3
        )
        self.assertEqual(len(additions), 1)
        self.assertEqual(additions[0].competitor_token_id, 5)
        self.assertEqual(method.selected_row_ids(additions), [1, 5])

    def _empty_protected(self, rank: int):
        return vectorized.ProtectedPairTensors(
            hidden=torch.empty(0, 2),
            base_modified_logits=torch.empty(0, 2),
            correct_base=torch.empty(0),
            correct_modified_row_index=torch.empty(0, dtype=torch.long),
            competitor_mask=torch.empty(0, 2, dtype=torch.bool),
            hidden_rank=torch.empty(0, rank),
        )

    def _neutral_ppl(self, rank: int):
        return method.PreparedPPLTensors(
            hidden_rank=torch.zeros(1, rank),
            base_logsumexp=torch.tensor([0.0]),
            base_target_logits=torch.tensor([-1.0]),
            target_selected_row_index=torch.tensor([-1]),
            base_selected_logits=torch.tensor([[-10.0, -10.0]]),
            normalization_divisor=1,
        )

    def test_solver_reports_first_configured_feasible_rank_without_proof_claim(self):
        # The active hidden vector has zero rank-1 projection and a nonzero
        # second component, so rank 1 stalls while rank 2 can separate rows.
        rank1 = vectorized.ActivePairTensors(
            hidden=torch.tensor([[0.0, 1.0]]),
            base_margin=torch.tensor([0.0]),
            sensitive_row_index=torch.tensor([0]),
            competitor_row_index=torch.tensor([1]),
            hidden_rank=torch.tensor([[0.0]]),
        )
        result1 = method.solve_vectorized_phase(
            active_tensors=rank1,
            protected_tensors=self._empty_protected(1),
            ppl_tensors=self._neutral_ppl(1),
            basis_prefix=torch.tensor([[1.0, 0.0]]),
            initial_coefficients=torch.zeros(2, 1),
            active_margin=0.25,
            protected_margin=0.0,
            allowed_mean_nll=10.0,
            steps=80,
            learning_rate=0.05,
            rho=5.0,
            tolerance=1e-5,
            stall_patience=15,
            min_improvement=1e-8,
        )
        self.assertFalse(result1.report["configured_feasible"])

        rank2 = vectorized.ActivePairTensors(
            hidden=torch.tensor([[0.0, 1.0]]),
            base_margin=torch.tensor([0.0]),
            sensitive_row_index=torch.tensor([0]),
            competitor_row_index=torch.tensor([1]),
            hidden_rank=torch.tensor([[0.0, 1.0]]),
        )
        result2 = method.solve_vectorized_phase(
            active_tensors=rank2,
            protected_tensors=self._empty_protected(2),
            ppl_tensors=self._neutral_ppl(2),
            basis_prefix=torch.eye(2),
            initial_coefficients=method.expand_rank_coefficients(result1.coefficients, 2),
            active_margin=0.25,
            protected_margin=0.0,
            allowed_mean_nll=10.0,
            steps=300,
            learning_rate=0.05,
            rho=5.0,
            tolerance=1e-4,
            stall_patience=80,
            min_improvement=1e-8,
        )
        self.assertTrue(result2.report["configured_feasible"])
        self.assertEqual(result2.report["rank"], 2)


class FirewallAndConfigurationTests(unittest.TestCase):
    def test_selection_record_does_not_read_or_copy_held_out_fields(self):
        class FailOnHeldOut(dict):
            def get(self, key, default=None):
                if key in {"questions", "answer", "answer_alias", "new_answer", "new_answer_alias"}:
                    raise AssertionError(f"held-out field read: {key}")
                return super().get(key, default)

        class FailOnHeldOutRewrite(dict):
            def __getitem__(self, key):
                if key in {"target_new", "question"}:
                    raise AssertionError(f"held-out rewrite field read: {key}")
                return super().__getitem__(key)

            def get(self, key, default=None):
                if key in {"target_new", "question"}:
                    raise AssertionError(f"held-out rewrite field read: {key}")
                return super().get(key, default)

        raw = FailOnHeldOut(
            case_id=5,
            requested_rewrite=[
                FailOnHeldOutRewrite({
                    "prompt": "{} was born in",
                    "subject": "Person",
                    "target_true": {"str": "Place"},
                    "target_new": {"str": "Counterfactual"},
                    "question": "Where was Person born?",
                })
            ],
        )
        rows = method._selection_record(raw, 10)
        encoded = json.dumps(rows)
        self.assertNotIn("Counterfactual", encoded)
        self.assertNotIn("Where was", encoded)
        self.assertEqual(rows[0]["requested_rewrite"]["target_new"]["str"], "Unknown")

    def test_rejected_candidate_never_loads_or_evaluates_held_out(self):
        loader = mock.Mock(side_effect=AssertionError("held-out loader called"))
        evaluator = mock.Mock(side_effect=AssertionError("held-out evaluator called"))
        with self.assertRaises(RuntimeError):
            method.evaluate_held_out_after_durable_acceptance(
                accepted=False, load_records=loader, evaluate=evaluator
            )
        loader.assert_not_called()
        evaluator.assert_not_called()

    def test_fixed_launcher_and_config_values(self):
        launcher = (SCRIPTS / "run_mquake_setting5e_detied_baseanchored_minrank.sh").read_text()
        for fragment in (
            "--steps 600",
            "--emb-lm-lr 1e-4",
            "--repair-rank-start 2",
            "--repair-rank-max 0",
            "--active-logit-margin 0.25",
            "--target-eff-max 0.0",
            "--max-ppl-ratio 1.02",
        ):
            self.assertIn(fragment, launcher)
        config = json.loads(
            (ROOT / "config/official_benchmarks/mquake_setting5e_detied_baseanchored_minrank.json").read_text()
        )
        self.assertEqual(config["status"], "EXPERIMENTAL_PENDING_EVALUATION")
        self.assertEqual(config["setting5e"]["steps"], 600)
        self.assertEqual(config["repair"]["rank_start"], 2)
        self.assertEqual(config["repair"]["rank_max"], 0)
        self.assertFalse(config["acceptance_gates"]["automatic_relaxation"])

    def test_final_gate_has_no_held_out_metric_inputs(self):
        signature = inspect.signature(method.final_acceptance_report)
        self.assertNotIn("atomic_gen", signature.parameters)
        self.assertNotIn("multihop", signature.parameters)


if __name__ == "__main__":
    unittest.main()
