"""fit_linear_router.NegativeQuality: base-model checks on router negatives (ours).

The claims these protect:
  * a negative the base model would answer with the head fact's answer is dropped
    (answer collision), and only for the head whose answer it is;
  * a subject transplant whose relation phrase is far less likely after this subject than
    after the donor's subject is dropped; real competitors are never plausibility-checked;
  * scores are cached per prompt.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
transformers = pytest.importorskip("transformers")

from fit_linear_router import NegativeQuality  # noqa: E402


class _Tok:
    pad_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=True, **_):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 2) for w in str(text).split()]
        return {"input_ids": ([1] + ids) if add_special_tokens else ids}


def _model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    return transformers.LlamaForCausalLM(cfg).eval()


FACTS = [{"id": "f0", "subject": "Ann Lee", "object": "Paris"}, {"id": "f1", "subject": "Bo Kim", "object": "Rome"}]


def _controls():
    return [{"prompt": "Ann Lee lives in", "kind": "same_subject_real", "split": "fit"},
            {"prompt": "Ann Lee plays the position of", "kind": "subject_transplant", "split": "fit",
             "donor_prompt": "Bo Kim plays the position of", "subject": "Ann Lee", "donor_subject": "Bo Kim"}]


def test_answer_collision_threshold_and_head():
    tok = _Tok()
    for w in ("Paris", "Rome"):
        tok(w)
    every = NegativeQuality(_model(), tok, FACTS, "cpu", answer_check=True, answer_prob=0.0)
    assert every(0, _controls()) == [] and every.stats["answer_collision"] == 2
    none = NegativeQuality(_model(), tok, FACTS, "cpu", answer_check=True, answer_prob=1.01)
    assert len(none(0, _controls())) == 2
    assert none.answer_ids[0] != none.answer_ids[1]                       # each head checks its own answer


def test_plausibility_only_for_transplants_and_cached():
    tok = _Tok()
    strict = NegativeQuality(_model(), tok, FACTS, "cpu", answer_check=False, plausibility_delta=-1e9)
    kept = strict(0, _controls())
    assert [c["kind"] for c in kept] == ["same_subject_real"] and strict.stats["implausible_transplant"] == 1
    assert strict.examples["implausible_transplant"][0]["donor"] == "Bo Kim plays the position of"
    n = len(strict._cache)
    strict(0, _controls())
    assert len(strict._cache) == n                                        # nothing re-scored
    lax = NegativeQuality(_model(), tok, FACTS, "cpu", answer_check=False, plausibility_delta=1e9)
    assert len(lax(0, _controls())) == 2
