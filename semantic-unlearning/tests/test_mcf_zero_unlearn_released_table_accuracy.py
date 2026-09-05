"""ZeroUnlearn released-table Eff/Gen/Spe come from teacher-forced top-1 accuracy.

ZeroUnlearn's ``experiments/py/eval_utils_counterfact.py`` calls
``test_batch_prediction`` with ``which_correct=[1, 1, 1]``, so
``rewrite/paraphrase/neighborhood_prompts_correct`` each record whether the
argmax token at every ``target_true`` position equals the ``target_true``
token.  ``experiments/summarize_list.py`` then reports those as
``post_*_acc``.  Its ZsRE evaluator emits *only* correctness lists and no
probabilities, so the released table cannot be the Eq.-16 likelihood.

These tests exercise the real evaluator rather than hand-written fixture rows:
a fixture containing ``*_prompts_correct`` keys passes even when the evaluator
never produces them, which is exactly how the metric silently returned null.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from test_gagd_active_case_repair import LlamaStyleTokenizer  # noqa: E402

import mcf_zero_unlearn_official_eval as official  # noqa: E402
from run_zerounlearn_targettrue_parity_mcf_multiseed import (  # noqa: E402
    _macro_correctness,
    metric_families,
)

VOCAB = 128


class PerfectNextTokenLM(nn.Module):
    """Teacher-forced oracle: logits at position p are one-hot on token p+1.

    Every scored ``target_true`` position must therefore be argmax-correct, so
    any off-by-one in the position arithmetic shows up as a failure.
    """

    def __init__(self, vocab_size: int = VOCAB):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(tie_word_embeddings=False, model_type="tiny")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        if length > 1:
            nxt = input_ids[:, 1:]
            logits[:, :-1, :].scatter_(2, nxt.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits)


class ConstantTokenLM(nn.Module):
    """Always predicts one fixed token id, so nothing is ever argmax-correct."""

    def __init__(self, winner: int, vocab_size: int = VOCAB):
        super().__init__()
        self.winner = winner
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(tie_word_embeddings=False, model_type="tiny")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size)
        logits[:, :, self.winner] = 10.0
        return SimpleNamespace(logits=logits)


def _record():
    return {
        "case_id": 0,
        "requested_rewrite": {
            "subject": "Ada",
            "prompt": "{} was born in",
            "target_new": {"str": "Rome"},
            "target_true": {"str": "Paris"},
        },
        "paraphrase_prompts": ["The birthplace of Ada is", "Ada came from"],
        "neighborhood_prompts": ["Bob was born in", "Cleo was born in", "Dan hails from"],
    }


def _evaluate(model):
    return official.official_compute_rewrite_quality_counterfact(
        model,
        LlamaStyleTokenizer(),
        _record(),
        torch.device("cpu"),
        llama_like=True,
    )


class ReleasedTableAccuracyTest(unittest.TestCase):
    def test_evaluator_emits_correctness_keys_with_group_lengths(self):
        post = _evaluate(PerfectNextTokenLM())
        record = _record()
        for key in (
            "rewrite_prompts_correct",
            "paraphrase_prompts_correct",
            "neighborhood_prompts_correct",
        ):
            self.assertIn(key, post, f"evaluator did not emit {key}")
        self.assertEqual(len(post["rewrite_prompts_correct"]), 1)
        self.assertEqual(
            len(post["paraphrase_prompts_correct"]),
            len(record["paraphrase_prompts"]),
        )
        self.assertEqual(
            len(post["neighborhood_prompts_correct"]),
            len(record["neighborhood_prompts"]),
        )

    def test_correctness_lists_pair_with_probability_lists(self):
        post = _evaluate(PerfectNextTokenLM())
        for group in ("rewrite", "paraphrase", "neighborhood"):
            self.assertEqual(
                len(post[f"{group}_prompts_correct"]),
                len(post[f"{group}_prompts_probs"]),
                f"{group}: one correctness flag per scored prompt",
            )

    def test_teacher_forced_oracle_is_correct_at_every_target_true_position(self):
        post = _evaluate(PerfectNextTokenLM())
        for group in ("rewrite", "paraphrase", "neighborhood"):
            self.assertTrue(
                all(post[f"{group}_prompts_correct"]),
                f"{group}: perfect next-token model must be argmax-correct",
            )

    def test_model_that_never_emits_target_true_scores_zero(self):
        post = _evaluate(ConstantTokenLM(winner=VOCAB - 1))
        for group in ("rewrite", "paraphrase", "neighborhood"):
            self.assertFalse(
                any(post[f"{group}_prompts_correct"]),
                f"{group}: constant-token model must never be correct",
            )

    def test_macro_correctness_reads_real_evaluator_output(self):
        rows = [{"post": _evaluate(PerfectNextTokenLM())}]
        self.assertEqual(_macro_correctness(rows, "rewrite_prompts_correct"), 100.0)
        self.assertEqual(_macro_correctness(rows, "neighborhood_prompts_correct"), 100.0)

        rows = [{"post": _evaluate(ConstantTokenLM(winner=VOCAB - 1))}]
        self.assertEqual(_macro_correctness(rows, "rewrite_prompts_correct"), 0.0)

    def test_nll_only_callers_keep_the_original_return_shape(self):
        scores = official.official_test_batch_prediction(
            PerfectNextTokenLM(),
            LlamaStyleTokenizer(),
            ["Ada was born in"],
            "Rome",
            "Paris",
            torch.device("cpu"),
            llama_like=True,
        )
        self.assertIsInstance(scores, list)
        self.assertEqual(set(scores[0]), {"target_new", "target_true"})


class MissingAccuracyGuardTest(unittest.TestCase):
    """A null released-table column must fail loudly, not after a 10-seed run."""

    def test_metric_families_raises_when_correctness_rows_absent(self):
        probs_only = {
            "post": {
                "rewrite_prompts_probs": [{"target_true": 0.5, "target_new": 1.0}],
                "paraphrase_prompts_probs": [{"target_true": 0.5, "target_new": 1.0}],
                "neighborhood_prompts_probs": [{"target_true": 0.5, "target_new": 1.0}],
            }
        }
        with self.assertRaises(RuntimeError) as ctx:
            metric_families({"forget_raw": [probs_only]})
        message = str(ctx.exception)
        self.assertIn("released-table", message)
        self.assertIn("return_correct=True", message)

    def test_metric_families_reports_both_families_on_real_output(self):
        families = metric_families(
            {"forget_raw": [{"post": _evaluate(PerfectNextTokenLM())}], "forget_PPL": 11.0}
        )
        table = families["released_table_style_accuracy"]
        self.assertEqual(table["Eff"], 100.0)
        self.assertEqual(table["Spe"], 100.0)
        self.assertIsNotNone(
            families["eq16_style_residual_likelihood_proxy"]["Eff"]
        )


if __name__ == "__main__":
    unittest.main()
