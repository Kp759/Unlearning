import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from static_overlap_fact_association_embeddings import (
    AssociationCausalLM,
    FactAssociationBank,
    _contains_subsequence,
)


class _IdentityLayer(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


class _TinyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 2)
        with torch.no_grad():
            self.embed.weight.zero_()
            self.embed.weight[3] = torch.tensor([1.0, 0.0])
            self.embed.weight[4] = torch.tensor([0.0, 1.0])
            self.embed.weight[5] = torch.tensor([0.7, 0.7])
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_IdentityLayer()])
        self.config = SimpleNamespace(use_cache=False, model_type="llama")

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return None

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        hidden = self.embed(input_ids)
        hidden = self.model.layers[0](hidden)[0]
        return SimpleNamespace(logits=hidden)


def test_contains_subsequence():
    assert _contains_subsequence([1, 2, 3, 4], (2, 3))
    assert not _contains_subsequence([1, 2, 3, 4], (2, 4))


def test_unmatched_input_is_exact_base_and_matched_input_gets_one_vector():
    base = _TinyBase()
    base.requires_grad_(False)
    facts = [{
        "id": "f0",
        "subject": "subject",
        "relation": "P0",
        "object": "object",
    }]
    bank = FactAssociationBank(
        base_model=base,
        layer=0,
        keys=torch.tensor([[1.0, 0.0]]),
        thresholds=torch.tensor([0.9]),
        subject_patterns=[[(3,)]],
        facts=facts,
        rows=torch.tensor([[0.0, 2.0]]),
    )
    for row in bank.rows:
        row.requires_grad_(False)
    model = AssociationCausalLM(base, bank)

    matched = torch.tensor([[3]])
    matched_out = model(input_ids=matched).logits
    assert torch.equal(matched_out, torch.tensor([[[1.0, 2.0]]]))

    unmatched = torch.tensor([[4]])
    bank.close()
    base_out = base(input_ids=unmatched).logits
    bank = FactAssociationBank(
        base_model=base,
        layer=0,
        keys=torch.tensor([[1.0, 0.0]]),
        thresholds=torch.tensor([0.9]),
        subject_patterns=[[(3,)]],
        facts=facts,
        rows=torch.tensor([[0.0, 2.0]]),
    )
    for row in bank.rows:
        row.requires_grad_(False)
    model = AssociationCausalLM(base, bank)
    edited_out = model(input_ids=unmatched).logits
    assert torch.equal(base_out, edited_out)


def test_prefix_boundary_prevents_answer_tokens_from_selecting_route():
    base = _TinyBase()
    base.requires_grad_(False)
    facts = [{
        "id": "f0",
        "subject": "subject",
        "relation": "P0",
        "object": "object",
    }]
    # Token 3 is the subject but has low key score; token 4 would match the key.
    # Prefix length 1 therefore must keep the route inactive even though token 4
    # appears in the teacher-forced answer suffix.
    bank = FactAssociationBank(
        base_model=base,
        layer=0,
        keys=torch.tensor([[0.0, 1.0]]),
        thresholds=torch.tensor([0.9]),
        subject_patterns=[[(3,)]],
        facts=facts,
        rows=torch.tensor([[2.0, 0.0]]),
    )
    for row in bank.rows:
        row.requires_grad_(False)
    model = AssociationCausalLM(base, bank)
    ids = torch.tensor([[3, 4]])
    model.set_association_prefix_lengths([1])
    out = model(input_ids=ids).logits
    expected = base.embed(ids)
    assert torch.equal(out, expected)
    assert bank.last_active_fact_indices == [[]]


def test_same_subject_selects_only_best_relation_key():
    base = _TinyBase()
    base.requires_grad_(False)
    facts = [
        {"id": "f0", "subject": "same", "relation": "P0", "object": "a"},
        {"id": "f1", "subject": "same", "relation": "P1", "object": "b"},
    ]
    bank = FactAssociationBank(
        base_model=base,
        layer=0,
        keys=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        thresholds=torch.tensor([0.8, 0.8]),
        subject_patterns=[[(3,)], [(3,)]],
        facts=facts,
        rows=torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
    )
    for row in bank.rows:
        row.requires_grad_(False)
    model = AssociationCausalLM(base, bank)

    # Final prompt token 4 matches relation-key 1; exactly row 1 should fire.
    ids = torch.tensor([[3, 4]])
    model.set_association_prefix_lengths([2])
    out = model(input_ids=ids).logits
    expected = base.embed(ids).clone()
    expected[0, 1] += torch.tensor([0.0, 2.0])
    assert torch.equal(out, expected)
    assert bank.last_active_fact_indices == [[1]]
