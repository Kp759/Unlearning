from __future__ import annotations

import ast
import copy
import inspect
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mquake_gagd_setting5e_multiroot_active_repair as subject  # noqa: E402
import mquake_zero_unlearn_official_eval as mquake  # noqa: E402
import zsre_gagd_setting5e_active_repair as repair  # noqa: E402


def case(case_id: int, token_index: int = 0) -> mquake.PredictionCase:
    return mquake.PredictionCase(
        case_id=case_id,
        prompt_type="rewrite",
        prompt_index=0,
        token_index=token_index,
        prompt=f"cloze-{case_id}-{token_index}",
        target_text="token",
    )


def cache(
    token_id: int,
    *,
    correct: bool = True,
    hidden: torch.Tensor | None = None,
    case_id: int = 1,
) -> repair.TokenLogitCache:
    return repair.TokenLogitCache(
        case=case(case_id, token_id),
        hidden=torch.tensor([1.0, 2.0]) if hidden is None else hidden,
        target_token_id=token_id,
        predicted_token_id=token_id if correct else token_id + 100,
        target_logit=torch.tensor(2.0),
        neutral_logit=torch.tensor(0.0),
        correct=correct,
    )


def active_pair(
    sensitive_id: int,
    competitor_id: int,
    *,
    hidden: torch.Tensor | None = None,
    sensitive_logit: float = 3.0,
    competitor_logit: float = 1.0,
    case_id: int = 1,
) -> subject.ActivePairCache:
    return subject.ActivePairCache(
        case=case(case_id, sensitive_id),
        hidden=torch.tensor([1.0, 0.0]) if hidden is None else hidden,
        sensitive_token_id=sensitive_id,
        competitor_token_id=competitor_id,
        sensitive_base_logit=torch.tensor(sensitive_logit),
        competitor_base_logit=torch.tensor(competitor_logit),
    )


def protected_state(
    correct_id: int,
    *,
    row_ids: list[int],
    correct_logit: float,
    modified_logits: list[float],
    hidden: torch.Tensor | None = None,
    case_id: int = 1,
) -> subject.ProtectedPairState:
    return subject.ProtectedPairState(
        case=case(case_id, correct_id),
        hidden=torch.tensor([1.0, 0.0]) if hidden is None else hidden,
        correct_token_id=correct_id,
        correct_base_logit=torch.tensor(correct_logit),
        modified_row_base_logits=torch.tensor(modified_logits),
        correct_modified_row_index=(
            row_ids.index(correct_id) if correct_id in row_ids else -1
        ),
    )


def legacy_active_pair_margins(
    pairs: list[subject.ActivePairCache],
    delta_rows: torch.Tensor,
    row_ids: list[int],
) -> torch.Tensor:
    """Pre-vectorization reference retained only in the test suite."""

    row_index = {token_id: index for index, token_id in enumerate(row_ids)}
    values = []
    for pair in pairs:
        hidden = pair.hidden.to(device=delta_rows.device, dtype=delta_rows.dtype)
        base_margin = (
            pair.competitor_base_logit - pair.sensitive_base_logit
        ).to(device=delta_rows.device, dtype=delta_rows.dtype)
        values.append(
            base_margin
            + hidden @ delta_rows[row_index[pair.competitor_token_id]]
            - hidden @ delta_rows[row_index[pair.sensitive_token_id]]
        )
    return torch.stack(values) if values else delta_rows.new_empty((0,))


def legacy_protected_pair_margins(
    states: list[subject.ProtectedPairState],
    delta_rows: torch.Tensor,
    row_ids: list[int],
) -> torch.Tensor:
    """Pre-vectorization protected-margin reference for equivalence tests."""

    values = []
    for state in states:
        hidden = state.hidden.to(device=delta_rows.device, dtype=delta_rows.dtype)
        delta_logits = hidden @ delta_rows.T
        correct_after = state.correct_base_logit.to(delta_rows.device)
        if state.correct_modified_row_index >= 0:
            correct_after = (
                correct_after + delta_logits[state.correct_modified_row_index]
            )
        base_logits = state.modified_row_base_logits.to(delta_rows.device)
        for row_index, row_id in enumerate(row_ids):
            if row_id != state.correct_token_id:
                values.append(
                    correct_after - base_logits[row_index] - delta_logits[row_index]
                )
    return torch.stack(values) if values else delta_rows.new_empty((0,))


def legacy_protected_drift(
    states: list[subject.ProtectedPairState], delta_rows: torch.Tensor
) -> torch.Tensor:
    if not states:
        return delta_rows.new_zeros(())
    hidden = torch.stack([state.hidden for state in states]).to(delta_rows.device)
    return (hidden @ delta_rows.T).square().mean()


class TinyTiedLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = nn.Linear(3, 3, bias=False)
        self.embedding = nn.Embedding(7, 3)
        self.lm_head = nn.Linear(3, 7, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.config = SimpleNamespace(tie_word_embeddings=True)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, module):
        self.lm_head = module

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.transformer(self.embedding(ids)))


class ActivePairRepairTests(unittest.TestCase):
    @staticmethod
    def _equivalence_devices() -> list[torch.device]:
        devices = [torch.device("cpu")]
        if torch.cuda.is_available():
            devices.append(torch.device("cuda"))
        return devices

    @staticmethod
    def _equivalence_fixture():
        torch.manual_seed(1729)
        hidden_size = 7
        row_ids = [11, 12, 13, 14]
        pairs = [
            active_pair(
                row_ids[index % 4],
                row_ids[(index + 1) % 4],
                hidden=torch.randn(hidden_size),
                sensitive_logit=float(2.0 + index / 10),
                competitor_logit=float(0.5 - index / 20),
                case_id=index + 1,
            )
            for index in range(6)
        ]
        states = [
            protected_state(
                correct_id,
                row_ids=row_ids,
                correct_logit=2.5 + index / 10,
                modified_logits=torch.randn(len(row_ids)).tolist(),
                hidden=torch.randn(hidden_size),
                case_id=100 + index,
            )
            for index, correct_id in enumerate([11, 99, 13, 12, 98])
        ]
        q, _ = torch.linalg.qr(torch.randn(hidden_size, 2))
        basis = q.T.contiguous()
        delta = torch.randn(len(row_ids), hidden_size)
        coefficients = torch.randn(len(row_ids), 2)
        return row_ids, pairs, states, basis, delta, coefficients

    def test_vectorized_active_margins_match_legacy_full_dimensional(self) -> None:
        row_ids, pairs, _states, _basis, delta, _coeff = self._equivalence_fixture()
        for device in self._equivalence_devices():
            with self.subTest(device=device.type):
                candidate = delta.to(device).requires_grad_(True)
                packed = subject.prepare_active_pair_tensors(
                    pairs, row_ids, device=device
                )
                expected = legacy_active_pair_margins(pairs, candidate, row_ids)
                actual = subject.active_pair_margins_from_delta(packed, candidate)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_rank_two_direct_coefficients_match_full_effective_delta(self) -> None:
        row_ids, pairs, _states, basis, _delta, coefficients = self._equivalence_fixture()
        for device in self._equivalence_devices():
            with self.subTest(device=device.type):
                basis_device = basis.to(device)
                coeff = coefficients.to(device)
                packed = subject.prepare_active_pair_tensors(
                    pairs, row_ids, device=device, direction_basis=basis_device
                )
                expected = subject.active_pair_margins_from_delta(
                    packed, coeff @ basis_device
                )
                actual = subject.active_pair_margins_from_coefficients(packed, coeff)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_vectorized_protected_margins_match_legacy(self) -> None:
        row_ids, _pairs, states, _basis, delta, _coeff = self._equivalence_fixture()
        for device in self._equivalence_devices():
            with self.subTest(device=device.type):
                candidate = delta.to(device)
                packed = subject.prepare_protected_pair_tensors(
                    states,
                    row_ids,
                    hidden_size=candidate.shape[1],
                    device=device,
                )
                delta_logits = subject.protected_delta_logits_from_delta(
                    packed, candidate
                )
                actual = subject.protected_pair_margins_from_delta_logits(
                    packed, delta_logits
                )
                expected = legacy_protected_pair_margins(states, candidate, row_ids)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_vectorized_protected_drift_matches_legacy(self) -> None:
        row_ids, _pairs, states, _basis, delta, _coeff = self._equivalence_fixture()
        for device in self._equivalence_devices():
            with self.subTest(device=device.type):
                candidate = delta.to(device)
                packed = subject.prepare_protected_pair_tensors(
                    states,
                    row_ids,
                    hidden_size=candidate.shape[1],
                    device=device,
                )
                delta_logits = subject.protected_delta_logits_from_delta(
                    packed, candidate
                )
                actual = subject.protected_logit_drift_from_delta_logits(delta_logits)
                expected = legacy_protected_drift(states, candidate)
                self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

    def test_orthonormal_rank_two_l2_equals_coefficient_norm(self) -> None:
        _rows, _pairs, _states, basis, _delta, coefficients = self._equivalence_fixture()
        for device in self._equivalence_devices():
            with self.subTest(device=device.type):
                basis_device = basis.to(device)
                coeff = coefficients.to(device)
                self.assertTrue(subject.basis_is_orthonormal(basis_device))
                full_l2 = (coeff @ basis_device).square().sum()
                fast_l2 = subject.low_rank_delta_l2(
                    coeff, basis_device, orthonormal=True
                )
                self.assertTrue(torch.allclose(fast_l2, full_l2, atol=1e-5, rtol=1e-5))

    def test_nonorthonormal_rank_l2_preserves_exact_full_calculation(self) -> None:
        _rows, _pairs, _states, basis, _delta, coefficients = self._equivalence_fixture()
        nonorthonormal = basis.clone()
        nonorthonormal[0].mul_(1.75)
        self.assertFalse(subject.basis_is_orthonormal(nonorthonormal))
        expected = (coefficients @ nonorthonormal).square().sum()
        actual = subject.low_rank_delta_l2(
            coefficients, nonorthonormal, orthonormal=False
        )
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_rank_two_vectorized_objective_gradients_match_legacy(self) -> None:
        row_ids, pairs, states, basis, _delta, coefficients = self._equivalence_fixture()
        for device in self._equivalence_devices():
            with self.subTest(device=device.type):
                basis_device = basis.to(device)
                legacy_coeff = coefficients.to(device).clone().requires_grad_(True)
                vector_coeff = coefficients.to(device).clone().requires_grad_(True)

                legacy_delta = legacy_coeff @ basis_device
                legacy_active = legacy_active_pair_margins(
                    pairs, legacy_delta, row_ids
                )
                legacy_protected = legacy_protected_pair_margins(
                    states, legacy_delta, row_ids
                )
                legacy_loss = (
                    subject.active_pair_squared_hinge_loss(legacy_active, 0.25)
                    + subject.protected_pair_squared_hinge_loss(legacy_protected, 0.0)
                    + 0.1 * legacy_protected_drift(states, legacy_delta)
                    + 1e-4 * legacy_delta.square().sum()
                )
                legacy_loss.backward()

                active_tensors = subject.prepare_active_pair_tensors(
                    pairs,
                    row_ids,
                    device=device,
                    direction_basis=basis_device,
                )
                protected_tensors = subject.prepare_protected_pair_tensors(
                    states,
                    row_ids,
                    hidden_size=basis.shape[1],
                    device=device,
                    direction_basis=basis_device,
                )
                vector_active = subject.active_pair_margins_from_coefficients(
                    active_tensors, vector_coeff
                )
                vector_delta_logits = (
                    subject.protected_delta_logits_from_coefficients(
                        protected_tensors, vector_coeff
                    )
                )
                vector_protected = (
                    subject.protected_pair_margins_from_delta_logits(
                        protected_tensors, vector_delta_logits
                    )
                )
                vector_loss = (
                    subject.active_pair_squared_hinge_loss(vector_active, 0.25)
                    + subject.protected_pair_squared_hinge_loss(vector_protected, 0.0)
                    + 0.1
                    * subject.protected_logit_drift_from_delta_logits(
                        vector_delta_logits
                    )
                    + 1e-4
                    * subject.low_rank_delta_l2(
                        vector_coeff, basis_device, orthonormal=True
                    )
                )
                vector_loss.backward()
                self.assertTrue(
                    torch.allclose(
                        vector_coeff.grad,
                        legacy_coeff.grad,
                        atol=1e-5,
                        rtol=1e-5,
                    )
                )

    def test_real_shape_rank_two_smoke_has_no_pair_loops(self) -> None:
        active_count = 3356
        protected_count = 2632
        hidden_size = 3072
        row_count = 64
        rank = 2
        active_tensors = subject.ActivePairTensors(
            hidden=torch.empty((active_count, hidden_size)),
            base_margin=torch.zeros(active_count),
            sensitive_row_index=torch.arange(active_count) % row_count,
            competitor_row_index=(torch.arange(active_count) + 1) % row_count,
            hidden_rank=torch.zeros((active_count, rank)),
        )
        protected_tensors = subject.ProtectedPairTensors(
            hidden=torch.empty((protected_count, hidden_size)),
            base_modified_logits=torch.zeros((protected_count, row_count)),
            correct_base=torch.ones(protected_count),
            correct_modified_row_index=torch.full(
                (protected_count,), -1, dtype=torch.long
            ),
            competitor_mask=torch.ones(
                (protected_count, row_count), dtype=torch.bool
            ),
            hidden_rank=torch.zeros((protected_count, rank)),
        )
        coefficients = torch.zeros((row_count, rank))
        active_margins = subject.active_pair_margins_from_coefficients(
            active_tensors, coefficients
        )
        delta_logits = subject.protected_delta_logits_from_coefficients(
            protected_tensors, coefficients
        )
        protected_margins = subject.protected_pair_margins_from_delta_logits(
            protected_tensors, delta_logits
        )
        self.assertEqual(tuple(active_margins.shape), (active_count,))
        self.assertEqual(
            tuple(protected_margins.shape),
            (protected_count * row_count,),
        )
        for function in (
            subject.active_pair_margins_from_coefficients,
            subject.protected_delta_logits_from_coefficients,
            subject.protected_pair_margins_from_delta_logits,
            subject.protected_logit_drift_from_delta_logits,
        ):
            tree = ast.parse(inspect.getsource(function))
            self.assertFalse(any(isinstance(node, ast.For) for node in ast.walk(tree)))
        optimizer_source = inspect.getsource(subject.optimize_active_pair_delta)
        self.assertNotIn("for pair in", optimizer_source)
        self.assertNotIn("for state in", optimizer_source)
        self.assertIn("rank_space_fast_path", optimizer_source)

    def test_rank_two_optimizer_materializes_full_delta_only_at_end(self) -> None:
        pairs = [
            active_pair(1, 2, hidden=torch.tensor([1.0, 0.0]), case_id=1),
            active_pair(2, 3, hidden=torch.tensor([0.0, 1.0]), case_id=2),
        ]
        states = [
            protected_state(
                9,
                row_ids=[1, 2, 3],
                correct_logit=3.0,
                modified_logits=[1.0, 1.5, 2.0],
                hidden=torch.tensor([1.0, 1.0]),
            )
        ]
        args = subject.build_parser().parse_args(
            [
                "--repair-rank", "2",
                "--repair-steps", "2",
                "--no-stop-when-all-satisfied",
                "--no-project-away-protected-hidden",
            ]
        )
        original = subject.active.SelectedRowDelta.effective_delta
        call_count = 0

        def counted(module):
            nonlocal call_count
            call_count += 1
            return original(module)

        class TinyOptimizer:
            def __init__(self, parameters, learning_rate):
                self.parameters = list(parameters)
                self.learning_rate = learning_rate

            def zero_grad(self, set_to_none=True):
                for parameter in self.parameters:
                    parameter.grad = None

            def step(self):
                with torch.no_grad():
                    for parameter in self.parameters:
                        if parameter.grad is not None:
                            parameter.add_(
                                parameter.grad, alpha=-self.learning_rate
                            )

        with mock.patch.object(
            subject.active.SelectedRowDelta, "effective_delta", new=counted
        ), mock.patch.object(
            subject.active,
            "make_repair_optimizer",
            side_effect=lambda module, _name, learning_rate: TinyOptimizer(
                module.parameters(), learning_rate
            ),
        ):
            delta, logs, summary = subject.optimize_active_pair_delta(
                pairs,
                states,
                row_ids=[1, 2, 3],
                hidden_size=2,
                device=torch.device("cpu"),
                args=args,
            )
        self.assertEqual(call_count, 1)
        self.assertEqual(tuple(delta.shape), (3, 2))
        self.assertEqual(len(logs), 2)
        tensorization = summary["optimization_tensorization"]
        self.assertTrue(tensorization["rank_space_fast_path"])
        self.assertFalse(tensorization["full_hidden_size_delta_materialized_per_step"])

    def test_rank_two_uses_one_shared_two_direction_basis_for_every_row(self) -> None:
        args = subject.build_parser().parse_args(["--repair-rank", "2"])
        active_hidden = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        shared_basis = subject.active.orthonormal_row_basis(
            active_hidden, max_rank=args.repair_rank
        )
        module = subject.active.SelectedRowDelta(
            n_rows=4,
            hidden_size=4,
            direction_basis=shared_basis,
            retained_basis=None,
            device=torch.device("cpu"),
        )
        with torch.no_grad():
            module.coefficients.copy_(
                torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0], [1.5, -0.5], [-2.0, 3.0]]
                )
            )
        delta = module.effective_delta()
        reconstructed = module.coefficients @ shared_basis
        residual = delta - (delta @ shared_basis.T) @ shared_basis
        self.assertEqual(args.repair_rank, 2)
        self.assertEqual(tuple(module.coefficients.shape), (4, 2))
        self.assertEqual(tuple(module.direction_basis.shape), (2, 4))
        self.assertTrue(torch.allclose(delta, reconstructed))
        self.assertTrue(torch.allclose(residual, torch.zeros_like(residual), atol=1e-6))
        self.assertLessEqual(int(torch.linalg.matrix_rank(delta).item()), 2)

    def test_repair_source_requests_only_rewrite_cloze_cases(self) -> None:
        class Guarded(dict):
            def __init__(self, *args, forbidden=(), **kwargs):
                super().__init__(*args, **kwargs)
                self.forbidden = set(forbidden)

            def __getitem__(self, key):
                if key in self.forbidden:
                    raise AssertionError(f"evaluation-only field was read: {key}")
                return super().__getitem__(key)

        rewrite = Guarded(
            {
                "prompt": "{} was born in",
                "subject": "Person",
                "target_true": {"str": "Place"},
                "question": "Where was Person born?",
            },
            forbidden={"question", "target_new"},
        )
        record = Guarded(
            {
                "case_id": 1,
                "requested_rewrite": rewrite,
                "atomic_gen_prompt": "Where was Person born?",
                "multihop_questions": ["Official q1", "Official q2", "Official q3"],
                "answer": "Place",
                "new_answer": "Elsewhere",
            },
            forbidden={
                "atomic_gen_prompt",
                "multihop_questions",
                "answer",
                "new_answer",
            },
        )
        tok = mock.Mock()
        tok.decode.side_effect = lambda ids: "" if not ids else "Place"
        with mock.patch.object(
            subject.mquake, "original_answer_token_ids", return_value=[7]
        ):
            actual = subject.build_repair_cases([record], tok, llama_like=True)
        self.assertEqual(len(actual), 1)
        self.assertEqual(actual[0].prompt_type, "rewrite")
        self.assertEqual(actual[0].prompt, "Person was born in")

    def test_only_exact_residual_failures_become_active(self) -> None:
        rows = [
            cache(11, correct=True),
            cache(12, correct=False),
        ]
        active = subject.residual_active_caches(rows)
        self.assertEqual([row.target_token_id for row in active], [11])

    def test_preselection_records_withhold_all_evaluation_only_text(self) -> None:
        records = [
            {
                "case_id": 1,
                "mquake_case_id": 2,
                "source_index": 3,
                "rewrite_index": 0,
                "requested_rewrite": {
                    "prompt": "{} was born in",
                    "subject": "Person",
                    "target_true": {"str": "Place"},
                    "target_new": {"str": "Unknown"},
                    "mquake_target_new": {"str": "Counterfactual"},
                    "question": "Held-out atomic question?",
                },
                "atomic_gen_prompt": "Held-out atomic question?",
                "multihop_questions": ["Held-out multihop?"],
                "multihop_answer": "Sensitive answer alias",
                "multihop_new_answer": "Counterfactual alias",
            }
        ]
        visible = subject.selection_visible_records(records)[0]
        serialized = repr(visible)
        self.assertNotIn("Held-out", serialized)
        self.assertNotIn("Counterfactual", serialized)
        self.assertNotIn("Sensitive answer alias", serialized)
        self.assertEqual(
            visible["atomic_gen_prompt"], "<withheld-until-post-selection>"
        )

    def test_trainable_rows_are_exact_union_of_pair_sides(self) -> None:
        pairs = [active_pair(11, 20), active_pair(11, 21, case_id=2)]
        self.assertEqual(subject.active_pair_row_ids(pairs), [11, 20, 21])

    def test_shared_rows_use_one_joint_delta(self) -> None:
        pairs = [active_pair(11, 20), active_pair(11, 21, case_id=2)]
        sensitive, competitors = subject.active_pair_row_counts(pairs)
        self.assertEqual(sensitive, {11: 2})
        self.assertEqual(competitors, {20: 1, 21: 1})

    def test_sensitive_row_delta_improves_exact_pair_margin(self) -> None:
        pair = active_pair(1, 2)
        row_ids = [1, 2]
        delta = torch.tensor([[-1.0, 0.0], [0.0, 0.0]])
        self.assertEqual(
            subject.active_pair_margins([pair], delta, row_ids).item(), -1.0
        )

    def test_competitor_row_delta_improves_exact_pair_margin(self) -> None:
        pair = active_pair(1, 2)
        row_ids = [1, 2]
        delta = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        self.assertEqual(
            subject.active_pair_margins([pair], delta, row_ids).item(), -1.0
        )

    def test_both_pair_rows_are_optimized_in_the_same_margin(self) -> None:
        pair = active_pair(1, 2)
        row_ids = [1, 2]
        delta = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(
            subject.active_pair_margins([pair], delta, row_ids).item(), 0.0
        )
        self.assertEqual(
            subject.active_pair_squared_hinge_loss(
                torch.tensor([0.5]), 0.5
            ).item(),
            0.0,
        )

    def test_same_row_may_participate_in_multiple_pairs(self) -> None:
        pairs = [active_pair(1, 2), active_pair(3, 2, case_id=2)]
        row_ids = [1, 2, 3]
        delta = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        margins = subject.active_pair_margins(pairs, delta, row_ids)
        self.assertTrue(torch.equal(margins, torch.tensor([-1.0, -1.0])))

    def test_competitor_can_be_sensitive_in_another_pair(self) -> None:
        pairs = [active_pair(1, 2), active_pair(2, 3, case_id=2)]
        self.assertEqual(subject.active_pair_row_ids(pairs), [1, 2, 3])
        sensitive, competitors = subject.active_pair_row_counts(pairs)
        self.assertEqual((sensitive[2], competitors[2]), (1, 1))

    def test_unknown_has_no_special_repair_behavior(self) -> None:
        unknown_id = 99
        pairs = [active_pair(1, 2)]
        self.assertNotIn(unknown_id, subject.active_pair_row_ids(pairs))
        natural_pair = [active_pair(1, unknown_id)]
        self.assertIn(unknown_id, subject.active_pair_row_ids(natural_pair))

    def test_runner_up_excludes_only_the_sensitive_row(self) -> None:
        # Token 3 can be a sensitive/modified row for another case; it must
        # still remain eligible as this case's true current runner-up.
        logits = torch.tensor([0.0, 9.0, 7.0, 8.0, 6.0])
        self.assertEqual(subject.true_runner_up_token_id(logits, 1), 3)

    def _frozen_tiny_model(self):
        torch.manual_seed(2)
        model = TinyTiedLM()
        ids = torch.tensor([[1, 2]])
        before = model.logits(ids).detach().clone()
        old_input_ptr = model.embedding.weight.data_ptr()
        output = subject.freeze_model_for_multirow_repair(model)
        after = model.logits(ids).detach()
        return model, output, before, after, old_input_ptr

    def test_lm_head_is_safely_untied_without_logit_change(self) -> None:
        model, output, before, after, old_input_ptr = self._frozen_tiny_model()
        self.assertTrue(torch.equal(before, after))
        self.assertNotEqual(output.weight.data_ptr(), old_input_ptr)

    def test_transformer_remains_frozen(self) -> None:
        model, _output, _before, _after, _old_input_ptr = self._frozen_tiny_model()
        self.assertFalse(model.transformer.weight.requires_grad)

    def test_input_embeddings_remain_frozen(self) -> None:
        model, _output, _before, _after, _old_input_ptr = self._frozen_tiny_model()
        self.assertFalse(model.embedding.weight.requires_grad)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))

    def test_protected_logit_drift_is_penalized(self) -> None:
        row_ids = [1, 2]
        protected = [
            protected_state(
                4,
                row_ids=row_ids,
                correct_logit=3.0,
                modified_logits=[1.0, 2.0],
            )
        ]
        zero = torch.zeros((2, 2))
        drift = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
        self.assertEqual(subject.protected_logit_drift_loss(zero, protected).item(), 0.0)
        self.assertGreater(subject.protected_logit_drift_loss(drift, protected).item(), 0.0)

    def test_protected_pair_margin_with_unmodified_correct_row(self) -> None:
        row_ids = [1, 2]
        state = protected_state(
            9,
            row_ids=row_ids,
            correct_logit=3.0,
            modified_logits=[1.0, 2.0],
        )
        delta = torch.tensor([[0.5, 0.0], [-0.5, 0.0]])
        margins = subject.protected_pair_margins([state], delta, row_ids)
        self.assertTrue(torch.equal(margins, torch.tensor([1.5, 1.5])))

    def test_protected_pair_margin_includes_modified_correct_row(self) -> None:
        row_ids = [1, 2]
        state = protected_state(
            1,
            row_ids=row_ids,
            correct_logit=3.0,
            modified_logits=[3.0, 2.0],
        )
        delta = torch.tensor([[0.5, 0.0], [0.25, 0.0]])
        margins = subject.protected_pair_margins([state], delta, row_ids)
        self.assertTrue(torch.equal(margins, torch.tensor([1.25])))

    def test_protected_pair_margin_includes_modified_competitor_row(self) -> None:
        row_ids = [1, 2]
        state = protected_state(
            9,
            row_ids=row_ids,
            correct_logit=3.0,
            modified_logits=[1.0, 2.0],
        )
        delta = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
        margins = subject.protected_pair_margins([state], delta, row_ids)
        self.assertTrue(torch.equal(margins, torch.tensor([2.0, -1.0])))
        self.assertGreater(
            subject.protected_pair_squared_hinge_loss(margins, 0.0).item(),
            0.0,
        )

    def _scaled_rows(self):
        weight = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        original_weight = weight.clone()
        row_ids = [1, 3]
        originals = weight[row_ids].clone()
        deltas = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0]])
        subject.materialize_multirow_scale(weight, row_ids, originals, deltas, 0.5)
        return weight, original_weight, row_ids, originals, deltas

    def test_candidate_scale_applies_to_every_learned_row_together(self) -> None:
        weight, original_weight, _row_ids, originals, deltas = self._scaled_rows()
        self.assertTrue(torch.equal(weight[1], originals[0] + 0.5 * deltas[0]))
        self.assertTrue(torch.equal(weight[3], originals[1] + 0.5 * deltas[1]))
        self.assertTrue(torch.equal(weight[[0, 2, 4]], original_weight[[0, 2, 4]]))

    def test_scale_zero_exactly_restores_setting5e(self) -> None:
        weight, original_weight, row_ids, originals, deltas = self._scaled_rows()
        subject.materialize_multirow_scale(weight, row_ids, originals, deltas, 0.0)
        self.assertTrue(torch.equal(weight, original_weight))

    @staticmethod
    def _result(eff: float, retain: float = 90.0, ppl: float = 10.0):
        return {
            "forget": {"Eff": eff},
            "retain": {"Eff": retain},
            "forget_PPL": ppl,
        }

    @staticmethod
    def _safe_scale():
        return {
            "active_correct_tokens": 0,
            "active_pair_margin_violations": 0,
            "protected_incremental_regressions_vs_zero": 0,
            "protected_pair_margin_violations": 0,
        }

    def test_candidate_requires_exactly_zero_eff(self) -> None:
        report = subject.select_candidate(
            self._result(4.0), self._result(0.000001), self._safe_scale()
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["forget_Eff_exactly_zero"])

    def test_candidate_uses_fixed_base_retain_and_ppl_gates(self) -> None:
        base = self._result(50.0, retain=90.0, ppl=10.0)
        retain_failure = subject.select_candidate(
            base, self._result(0.0, retain=89.899, ppl=10.0), self._safe_scale()
        )
        ppl_failure = subject.select_candidate(
            base, self._result(0.0, retain=90.0, ppl=10.201), self._safe_scale()
        )
        self.assertFalse(retain_failure["checks"]["retain_Eff_within_base_tolerance"])
        self.assertFalse(ppl_failure["checks"]["PPL_within_base_ratio"])

    def test_protected_regression_rejects_candidate(self) -> None:
        scale = self._safe_scale()
        scale["protected_incremental_regressions_vs_zero"] = 1
        report = subject.select_candidate(
            self._result(50.0), self._result(0.0), scale
        )
        self.assertFalse(report["accepted"])

    def test_atomic_gen_cannot_affect_selection(self) -> None:
        base = self._result(50.0)
        candidate_a = self._result(0.0)
        candidate_b = copy.deepcopy(candidate_a)
        candidate_a["forget"]["AtomicGen"] = 0.0
        candidate_b["forget"]["AtomicGen"] = 100.0
        self.assertEqual(
            subject.select_candidate(base, candidate_a, self._safe_scale()),
            subject.select_candidate(base, candidate_b, self._safe_scale()),
        )

    def test_multihop_cannot_affect_selection(self) -> None:
        base = self._result(50.0)
        candidate_a = self._result(0.0)
        candidate_b = copy.deepcopy(candidate_a)
        candidate_a["multihop"] = {"MHLeak_exact_any": 0.0}
        candidate_b["multihop"] = {"MHLeak_exact_any": 100.0}
        self.assertEqual(
            subject.select_candidate(base, candidate_a, self._safe_scale()),
            subject.select_candidate(base, candidate_b, self._safe_scale()),
        )

    def test_rejected_fail_fast_never_opens_held_out_evaluation(self) -> None:
        args = SimpleNamespace(fail_if_target_missed=True)
        with (
            mock.patch.object(subject.baseline, "evaluate_extension") as atomic_gen,
            mock.patch.object(subject, "_evaluate_multihop_post_selection") as multihop,
            self.assertRaisesRegex(RuntimeError, "No active-pair candidate"),
        ):
            subject._evaluate_held_out_after_selection(
                accepted=False,
                args=args,
                model=mock.Mock(),
                tok=mock.Mock(),
                records=mock.Mock(),
                mquake_path=Path("data/MQuAKE-CF-3k-v2.json"),
                wikidata_dir=Path("data/wikidata"),
                split_manifest=Path("outputs/split_manifest.json"),
                output_dir=Path("outputs/rejected"),
            )
        atomic_gen.assert_not_called()
        multihop.assert_not_called()

    def test_accepted_candidate_runs_both_held_out_evaluators(self) -> None:
        args = SimpleNamespace(
            fail_if_target_missed=True,
            multihop_prompt_dir="data/mquake_prompts",
        )
        atomic_result = {"forget": {"AtomicGen": 0.0}}
        multihop_result = {"results": {"standard": {}, "cot": {}}}
        with (
            mock.patch.object(
                subject.baseline,
                "evaluate_extension",
                return_value=atomic_result,
            ) as atomic_gen,
            mock.patch.object(
                subject,
                "_evaluate_multihop_post_selection",
                return_value=multihop_result,
            ) as multihop,
        ):
            actual = subject._evaluate_held_out_after_selection(
                accepted=True,
                args=args,
                model=mock.Mock(),
                tok=mock.Mock(),
                records=mock.Mock(),
                mquake_path=Path("data/MQuAKE-CF-3k-v2.json"),
                wikidata_dir=Path("data/wikidata"),
                split_manifest=Path("outputs/split_manifest.json"),
                output_dir=Path("outputs/accepted"),
            )
        self.assertEqual(actual, (atomic_result, multihop_result))
        atomic_gen.assert_called_once()
        multihop.assert_called_once()

    def test_default_retain_calibration_is_full_1000_instances(self) -> None:
        args = subject.build_parser().parse_args([])
        self.assertEqual(args.retain_calibration_num, 1000)
        self.assertEqual(args.repair_mode, "active_pair")
        self.assertFalse(args.project_away_protected_hidden)

    def test_aggressive_setting5e_parameters_are_pinned(self) -> None:
        args = subject.build_parser().parse_args([])
        subject.validate_args(args)
        args.steps = 599
        with self.assertRaisesRegex(ValueError, "Setting 5e"):
            subject.validate_args(args)

    def test_canonical_launcher_uses_active_pair_not_sparse_prototype(self) -> None:
        launcher = (
            SCRIPTS / "run_mquake_setting5e_multiroot_active_repair.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--steps 600", launcher)
        self.assertIn("--repair-mode active_pair", launcher)
        self.assertIn("--no-project-away-protected-hidden", launcher)
        self.assertNotIn("mquake_exact_sparse_row_repair.py", launcher)

    def test_dedicated_rank_two_launcher_and_config_are_fully_pinned(self) -> None:
        launcher = (
            SCRIPTS / "run_mquake_setting5e_rank2_active_pair.sh"
        ).read_text(encoding="utf-8")
        for required in (
            "--steps 600",
            "--batch-size 1",
            "--retain-batch-size 4",
            "--emb-lm-lr 1e-4",
            "--forget-weight 2.0",
            "--retain-weight 1.0",
            "--forget-margin 1.0",
            "--emb-lm-optimizer adamw",
            "--sampling-strategy epoch",
            "--repair-mode active_pair",
            "--repair-steps 2000",
            "--repair-lr 5e-3",
            "--repair-rank 2",
            "--repair-l2-lambda 1e-4",
            "--protected-logit-drift-weight 0.1",
            "--no-project-away-protected-hidden",
            "--target-eff-max 0.0",
            "--utility-drop-tolerance 0.10",
            "--max-ppl-ratio 1.02",
            "--save-setting5-checkpoint",
            "--save-selected-checkpoint",
            "--fail-if-target-missed",
        ):
            self.assertIn(required, launcher)
        config_path = (
            SCRIPTS.parent
            / "config"
            / "official_benchmarks"
            / "mquake_setting5e_rank2_active_pair.json"
        )
        config = json.loads(config_path.read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "EXPERIMENTAL_PENDING_EVALUATION")
        self.assertEqual(config["active_repair"]["repair_rank"], 2)
        self.assertFalse(
            config["active_repair"]["project_away_protected_hidden"]
        )
        self.assertEqual(config["acceptance_gates"]["target_eff_max"], 0.0)

    def test_retain_calibration_keeps_all_atoms_of_sampled_instances(self) -> None:
        records = [
            {"source_index": 1, "rewrite_index": 0},
            {"source_index": 1, "rewrite_index": 1},
            {"source_index": 2, "rewrite_index": 0},
        ]
        selected = subject.sample_retain_instances(records, 2, 1729)
        self.assertEqual(selected, records)

    def test_instance_balanced_sampling_is_deterministic_by_seed(self) -> None:
        records = [
            {"source_index": 1, "rewrite_index": 0, "case_id": 100},
            {"source_index": 1, "rewrite_index": 1, "case_id": 101},
            {"source_index": 2, "rewrite_index": 0, "case_id": 200},
        ]
        with mock.patch.object(
            subject.baseline,
            "canonical_examples",
            side_effect=lambda chosen, _tok: [row["case_id"] for row in chosen],
        ):
            first, first_report = subject.instance_balanced_training_examples(
                records, object(), steps=30, seed=7
            )
            second, second_report = subject.instance_balanced_training_examples(
                records, object(), steps=30, seed=7
            )
            different, _ = subject.instance_balanced_training_examples(
                records, object(), steps=30, seed=8
            )
        self.assertEqual(first, second)
        self.assertEqual(first_report, second_report)
        self.assertNotEqual(first, different)


if __name__ == "__main__":
    unittest.main()
