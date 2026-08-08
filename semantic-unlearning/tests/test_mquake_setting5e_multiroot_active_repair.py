from __future__ import annotations

import copy
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
