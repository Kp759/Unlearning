import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_zsre_fact_association_embeddings_official import (
    _strict_prefix_lengths,
)


class _ToyTokenizer:
    def __init__(self):
        self.vocab = {}

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            raise AssertionError("Toy tokenizer only supports scalar text")
        ids = [1]
        for token in str(text).split():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 2
            ids.append(self.vocab[token])
        return {"input_ids": ids}


def test_fixed_boundary_is_exact_token_prefix():
    tok = _ToyTokenizer()
    lengths = _strict_prefix_lengths(
        tok,
        ["who is Ada Lovelace mathematician"],
        ["who is Ada Lovelace"],
    )
    assert lengths == [5]


def test_fixed_boundary_refuses_nonprefix_tokenization():
    tok = _ToyTokenizer()
    with pytest.raises(ValueError, match="exact token prefix"):
        _strict_prefix_lengths(
            tok,
            ["who is Ada Lovelace mathematician"],
            ["where is Ada Lovelace"],
        )
