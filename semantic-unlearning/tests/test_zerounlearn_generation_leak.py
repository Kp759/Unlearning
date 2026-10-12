"""Helpers of the ZeroUnlearn free-generation leak check (no ZeroUnlearn / GPU needed)."""
from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from zerounlearn_generation_leak import greedy, leak_summary, restore, snapshot  # noqa: E402


class _Toy(nn.Module):
    """Next token = (last token + 1) mod V, via a lookup the 'edit' can corrupt."""

    def __init__(self, vocab=10):
        super().__init__()
        self.down = nn.Linear(vocab, vocab, bias=False)
        with torch.no_grad():
            self.down.weight.copy_(torch.roll(torch.eye(vocab), 1, dims=0) * 10)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        x = nn.functional.one_hot(input_ids, self.down.in_features).float()
        return type("Out", (), {"logits": self.down(x)})()


def test_greedy_stops_at_any_eos_id():
    model = _Toy()
    out = greedy(model, torch.tensor([[2]]), max_new_tokens=8, eos_ids={5, 9})
    assert out[0].tolist() == [2, 3, 4, 5]


def test_snapshot_restore_is_exact_after_an_edit():
    model = _Toy()
    base = greedy(model, torch.tensor([[0]]), 6, set())
    saved = snapshot(model, ["down.weight"])
    with torch.no_grad():
        model.down.weight.mul_(-1).add_(torch.randn_like(model.down.weight))   # the "edit"
    assert greedy(model, torch.tensor([[0]]), 6, set()).tolist() != base.tolist()
    restore(model, saved)
    assert torch.equal(greedy(model, torch.tensor([[0]]), 6, set()), base)


def test_leak_summary_separates_start_anywhere_and_answer_in_prompt():
    rows = [
        {"prompt": "Paris is the capital of", "answer": "France",
         "answer_in_continuation": True, "answer_at_start": True, "answer_in_prompt": False,
         "base_nll_per_token": 1.0, "repeated_trigram": False},
        {"prompt": "Q: capital? A:", "answer": "France",
         "answer_in_continuation": True, "answer_at_start": False, "answer_in_prompt": False,
         "base_nll_per_token": 3.0, "repeated_trigram": True},
        {"prompt": "BMW X5 is produced by", "answer": "BMW",
         "answer_in_continuation": True, "answer_at_start": True, "answer_in_prompt": True,
         "base_nll_per_token": 2.0, "repeated_trigram": False},
        {"prompt": "Toyota is from", "answer": "Japan",
         "answer_in_continuation": False, "answer_at_start": False, "answer_in_prompt": False,
         "base_nll_per_token": 2.0, "repeated_trigram": False},
    ]
    s = leak_summary(rows)
    assert s["leak_any"] == 0.75 and s["leak_start"] == 0.5
    assert s["prompts_answer_not_in_prompt"] == 3 and abs(s["leak_any_excl"] - 2 / 3) < 1e-9
    assert s["base_nll_per_token_mean"] == 2.0 and s["repeated_trigram"] == 0.25
