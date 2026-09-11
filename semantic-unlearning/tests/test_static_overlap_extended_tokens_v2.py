import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import static_overlap_extended_tokens_v2 as v2
from run_static_overlap_extended_tokens_standalone_v2 import (
    DEVELOPMENT_SCAFFOLDS,
    TRAIN_SCAFFOLDS,
    encode_v2_authored_views,
)
from static_overlap_core import answer_nll, model_logits
from static_overlap_data import Example


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=True):
        assert add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [0] + list(range(1, len(text) + 1)),
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))],
        }


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 5)
        self.head = nn.Linear(5, 9, bias=False)

    def forward(self, input_ids, **_):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


def test_v2_authors_eight_train_and_four_disjoint_development_views():
    fact = {
        "id": "mcf_forget_17",
        "role": "forget",
        "subject": "Rob",
        "relation": "P103",
        "object": "French",
        "aliases": [],
        "answer_aliases": [],
    }
    examples = encode_v2_authored_views([fact], CharacterTokenizer(), 512)
    train = [example for example in examples if example.split == "train"]
    development = [example for example in examples if example.split == "development"]
    assert len(train) == 8
    assert len(development) == 4
    assert set(TRAIN_SCAFFOLDS).isdisjoint(DEVELOPMENT_SCAFFOLDS)
    assert {example.prompt for example in train}.isdisjoint(
        example.prompt for example in development
    )
    assert all("native language" in example.prompt for example in examples)


def test_row_wise_embedding_update_cannot_move_an_unselected_row():
    torch.manual_seed(3)
    base = nn.Embedding(5, 4)
    initial = torch.randn(2, 4)
    extended = v2.InputOnlyRowWiseExtendedEmbedding(base, initial)
    original_before = base.weight.detach().clone()
    other_before = extended.rows[1].detach().clone()
    extended(torch.tensor([[5]])).sum().backward()
    with torch.no_grad():
        extended.rows[0].add_(extended.rows[0].grad, alpha=-0.1)
    assert not torch.equal(extended.rows[0].detach(), initial[0])
    torch.testing.assert_close(extended.rows[1].detach(), other_before, atol=0, rtol=0)
    torch.testing.assert_close(base.weight.detach(), original_before, atol=0, rtol=0)


def test_worst_view_drives_fact_objective(monkeypatch):
    monkeypatch.setattr(
        v2, "batched_answer_nll",
        lambda _model, examples: torch.stack([example.value for example in examples]),
    )
    answers = [
        SimpleNamespace(id="easy", value=torch.tensor(0.7)),
        SimpleNamespace(id="hard", value=torch.tensor(0.2)),
        SimpleNamespace(id="middle", value=torch.tensor(0.5)),
    ]
    unknowns = [
        SimpleNamespace(id="u0", value=torch.tensor(0.1)),
        SimpleNamespace(id="u1", value=torch.tensor(0.2)),
        SimpleNamespace(id="u2", value=torch.tensor(0.3)),
    ]
    result = v2.fact_objective(None, answers, unknowns, target_probability=0.5)
    assert result["worst_view_id"] == "hard"
    assert float(result["max_probability"]) == pytest.approx(math.exp(-0.2))
    assert float(result["forget_gap"]) == pytest.approx(-math.log(0.5) - 0.2)
    assert float(result["unknown_nll"]) == pytest.approx(0.2)


def test_batched_answer_nll_matches_single_example_scoring():
    torch.manual_seed(9)
    model = TinyCausalModel()
    examples = [
        Example("a", "train", "forget", "f", [0, 1, 2, 3],
                [-100, -100, 2, 3], "p", "a", "g"),
        Example("b", "train", "forget", "f", [0, 4, 5, 6, 7],
                [-100, -100, -100, 6, 7], "p", "b", "g"),
    ]
    batched = v2.batched_answer_nll(model, examples)
    singles = torch.stack([
        answer_nll(model_logits(model, example), example) for example in examples
    ])
    torch.testing.assert_close(batched, singles)


def test_radius_schedule_tracks_worst_probability():
    schedule = ((1e-3, 1.0), (1e-5, 0.25), (1e-6, 0.05), (0.0, 0.01))
    assert v2.radius_for_probability(0.2, schedule) == 1.0
    assert v2.radius_for_probability(5e-4, schedule) == 0.25
    assert v2.radius_for_probability(5e-6, schedule) == 0.05
    assert v2.radius_for_probability(5e-7, schedule) == 0.01


def test_acceptance_is_answer_first_then_target_constrained_abstention():
    target = 1e-6
    assert v2.proposal_improves(1e-3, 1.0, 9e-4, 10.0, target, locked=False)
    assert not v2.proposal_improves(1e-3, 1.0, 1.1e-3, 0.0, target, locked=False)
    assert v2.proposal_improves(1e-3, 1.0, 1e-3, 0.9, target, locked=False)
    assert v2.proposal_improves(8e-7, 1.0, 9e-7, 0.9, target, locked=True)
    assert not v2.proposal_improves(8e-7, 1.0, 7e-7, 1.1, target, locked=True)
    assert not v2.proposal_improves(8e-7, 1.0, 1.1e-6, 0.1, target, locked=True)


def test_checkpoint_key_uses_global_maximum_before_abstention():
    metrics = {
        "train": {"max_token_probability": 2e-6, "unknown_mean_nll": 0.4},
        "development": {"max_token_probability": 7e-6, "unknown_mean_nll": 0.8},
    }
    assert v2.checkpoint_key(metrics) == pytest.approx((7e-6, 0.6))
