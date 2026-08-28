from __future__ import annotations

import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_scoped_bias_sidecar as builder  # noqa: E402


class WordTokenizer:
    def __init__(self):
        self.vocab = {
            "Gautier": 2,
            "de": 3,
            "Coincy": 4,
            "New": 5,
            "York": 6,
            "Jersey": 7,
        }

    def __call__(self, text, add_special_tokens=False):
        return {
            "input_ids": [
                self.vocab.get(word, 9) for word in str(text).strip().split()
            ]
        }

    def decode(self, token_ids):
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(value), "?") for value in token_ids)


def _record(target_true="New York", target_new="New Jersey"):
    return {
        "case_id": 17,
        "requested_rewrite": {
            "subject": "Gautier de Coincy",
            "prompt": "{} was born in",
            "target_true": {"str": target_true},
            "target_new": {"str": target_new},
        },
        # These evaluation-only fields must not affect sidecar construction.
        "paraphrase_prompts": ["held-out paraphrase"],
        "neighborhood_prompts": ["held-out neighborhood"],
    }


def test_builder_penalizes_only_sensitive_rows_absent_from_reference():
    state, audit = builder.build_state(
        WordTokenizer(),
        hidden_size=3,
        records=[_record()],
        writer_layer=1,
        penalty=512.0,
        seed=1,
    )
    assert state["subjects"] == ["Gautier de Coincy"]
    assert state["reader_row_ids"].tolist() == [[6]]
    assert state["reader_biases"].tolist() == [[-512.0]]
    assert state["reader_deltas"].count_nonzero().item() == 0
    assert state["writer_delta"].count_nonzero().item() == 0
    assert audit[0]["penalized_tokens"] == ["York"]
    assert state["metadata"]["official_paraphrases_or_neighborhoods_used"] is False


def test_builder_rejects_answers_with_no_discriminative_token_row():
    with pytest.raises(RuntimeError, match="No sensitive-only token row"):
        builder.build_state(
            WordTokenizer(),
            hidden_size=3,
            records=[_record(target_true="New York", target_new="York New")],
            writer_layer=1,
            penalty=512.0,
            seed=1,
        )
