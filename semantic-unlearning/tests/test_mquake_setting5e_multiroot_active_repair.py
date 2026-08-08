from __future__ import annotations

import argparse
import copy
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


class MultirowRepairTests(unittest.TestCase):
    def test_repair_source_requests_only_rewrite_cloze_cases(self) -> None:
        record = {
            "requested_rewrite": {
                "prompt": "{} was born in",
                "subject": "Person",
                "question": "Where was Person born?",
            },
            "atomic_gen_prompt": "Where was Person born?",
            "multihop_questions": ["Official q1", "Official q2", "Official q3"],
        }
        expected = [case(1)]
        with mock.patch.object(
            subject.mquake, "expand_prediction_cases", return_value=expected
        ) as expanded:
            actual = subject.build_repair_cases([record], object(), llama_like=True)
        self.assertEqual(actual, expected)
        self.assertEqual(expanded.call_args.kwargs["prompt_types"], ("rewrite",))

    def test_only_residual_sensitive_rows_are_trainable(self) -> None:
        rows = [
            cache(11, correct=True),
            cache(12, correct=False),
            cache(99, correct=True),
        ]
        active = subject.residual_active_caches(rows, unknown_token_id=99)
        self.assertEqual([row.target_token_id for row in active], [11])
        self.assertEqual(subject.active_row_ids(rows, 99), [99, 11])

    def test_shared_sensitive_token_has_one_delta_row(self) -> None:
        rows = [cache(11, case_id=1), cache(11, case_id=2), cache(12, case_id=3)]
        self.assertEqual(subject.active_row_ids(rows, 99), [99, 11, 12])
        self.assertEqual(subject.constraints_per_sensitive_row(rows), {11: 2, 12: 1})

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
        protected = [cache(4, hidden=torch.tensor([1.0, 0.0]))]
        zero = torch.zeros((2, 2))
        drift = torch.tensor([[2.0, 0.0], [1.0, 0.0]])
        self.assertEqual(subject.protected_logit_drift_loss(zero, protected).item(), 0.0)
        self.assertGreater(subject.protected_logit_drift_loss(drift, protected).item(), 0.0)

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
            "active_margin_violations": 0,
            "protected_incremental_regressions_vs_zero": 0,
        }

    def test_candidate_requires_exactly_zero_eff(self) -> None:
        report = subject.select_candidate(
            self._result(4.0), self._result(0.000001), self._safe_scale()
        )
        self.assertFalse(report["accepted"])
        self.assertFalse(report["checks"]["forget_Eff_exactly_zero"])

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

    def test_default_retain_calibration_is_full_1000_instances(self) -> None:
        args = subject.build_parser().parse_args([])
        self.assertEqual(args.retain_calibration_num, 1000)

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
