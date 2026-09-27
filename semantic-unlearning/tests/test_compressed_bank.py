"""In-loop compression: relation-shared router heads and compressed value modes."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compressed_value_bank import (  # noqa: E402
    CompressedValueBank,
    CompressedValues,
    parse_value_mode,
)
from fit_linear_router import collapse_heads, compact_heads, relation_head_index  # noqa: E402
from linear_router import LinearClassifierAssociationBank  # noqa: E402

HIDDEN = 8
FACTS = [
    {"id": "a", "subject": "S0", "relation": "P1", "object": "Paris"},
    {"id": "b", "subject": "S0", "relation": "P2", "object": "French"},
    {"id": "c", "subject": "S1", "relation": "P1", "object": "Paris"},
    {"id": "d", "subject": "S2", "relation": "P3", "object": "Oslo"},
]


class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.placeholder = nn.Parameter(torch.zeros(1))


def test_relation_heads_collapse_and_compact():
    index, names = relation_head_index(FACTS)
    assert index.tolist() == [0, 1, 0, 2] and names == ["P1", "P2", "P3"]
    # prompt 0: positive for a, eligible for b (same subject); prompt 1: positive for c;
    # prompt 2: eligible for d only, negative.
    labels = torch.tensor([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], dtype=torch.bool)
    eligible = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=torch.bool)
    shared_labels, shared_eligible = collapse_heads(labels, eligible, index, 3)
    assert shared_eligible.tolist() == [[True, True, False], [True, False, False], [False, False, True]]
    assert shared_labels.tolist() == [[True, False, False], [True, False, False], [False, False, False]]
    weight_r, bias_r = torch.randn(3, HIDDEN), torch.randn(3)
    w, b = compact_heads(weight_r[index], bias_r[index], index)
    assert torch.equal(w, weight_r) and torch.equal(b, bias_r)
    with pytest.raises(RuntimeError):
        broken = weight_r[index].clone()
        broken[2] += 1
        compact_heads(broken, bias_r[index], index)


def test_shared_head_bank_routes_like_expanded_bank():
    torch.manual_seed(0)
    index, _ = relation_head_index(FACTS)
    weight_r, bias_r = torch.randn(3, HIDDEN), torch.randn(3)
    common = dict(feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
                  subject_patterns=[[(10,)], [(10,)], [(11,)], [(12,)]], facts=FACTS,
                  rows=torch.randn(4, HIDDEN), ambiguity_margin=0.0)
    expanded = LinearClassifierAssociationBank(_FakeModel(), 0, weight_r[index], bias_r[index], **common)
    shared = LinearClassifierAssociationBank(_FakeModel(), 0, weight_r, bias_r, head_index=index, **common)
    query = torch.randn(6, HIDDEN)
    assert torch.allclose(expanded.router_logits(query), shared.router_logits(query))
    assert shared.artifact()["router_heads"] == 3
    assert shared.artifact()["head_index"].tolist() == index.tolist()


@pytest.mark.parametrize("mode", ["full", "lowrank:2", "tied_answer", "tied_relation",
                                  "answer_fixed", "answer_map:3", "relation_plus_answer"])
def test_value_modes_are_trainable_and_account_storage(mode):
    name, rank = parse_value_mode(mode)
    dirs = -F.normalize(torch.randn(len(FACTS), HIDDEN), dim=-1)
    values = CompressedValues(name, rank, FACTS, HIDDEN, answer_dirs=dirs)
    rows = values.rows()
    assert rows.shape == (len(FACTS), HIDDEN)
    rows.pow(2).sum().backward() if rows.abs().sum() > 0 else (rows.sum() + sum(
        p.sum() for p in values.parameters())).backward()
    assert all(p.grad is not None for p in values.parameters())
    storage = values.storage()
    assert storage["total_floats"] == len(FACTS) * storage["per_fact_floats"] + storage["shared_floats"]
    # a fresh module restored from the compact state rebuilds the same rows
    clone = CompressedValues(name, rank, FACTS, HIDDEN, answer_dirs=dirs)
    clone.load_state_dict(values.compact_state())
    assert torch.equal(clone.rows(), values.rows())


def test_tied_answer_shares_direction_between_same_answers():
    values = CompressedValues("tied_answer", None, FACTS, HIDDEN)
    with torch.no_grad():
        values.directions.copy_(torch.randn_like(values.directions))
        values.scale.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    rows = values.rows()
    assert torch.allclose(rows[0] / 1.0, rows[2] / 3.0)   # both "Paris"
    assert values.storage()["answer_groups"] == 3


def test_compressed_bank_uses_value_rows():
    dirs = -F.normalize(torch.randn(len(FACTS), HIDDEN), dim=-1)
    values = CompressedValues("answer_fixed", None, FACTS, HIDDEN, answer_dirs=dirs)
    with torch.no_grad():
        values.scale.fill_(2.0)
    bank = CompressedValueBank(
        _FakeModel(), 0, torch.randn(4, HIDDEN), torch.zeros(4),
        feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
        subject_patterns=[[(10,)], [(10,)], [(11,)], [(12,)]], facts=FACTS, values=values,
    )
    assert torch.allclose(bank.extra, 2.0 * dirs)
    assert not any(row.requires_grad for row in bank.rows)
