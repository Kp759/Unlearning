#!/usr/bin/env python3
"""CPU smoke tests for RWKU Directional SURE v2 invariants."""
import torch
from torch import nn

import rwku_directional_sure_v2 as directional
import rwku_setting5e_utility_controlled as sparse_rows


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)
        with torch.no_grad():
            self.embed.weight.copy_(torch.arange(32, dtype=torch.float32).view(8, 4) / 32.0)
            self.head.weight.copy_(self.embed.weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def main():
    model = ToyModel()
    selected = [1, 3]
    sparse = sparse_rows.SparseFP32RowDeltas(model, selected, selected)
    base_embed = model.embed.weight.detach().clone()
    base_head = model.head.weight.detach().clone()
    with torch.no_grad():
        sparse.input_delta[0].fill_(0.25)
        sparse.input_delta[1].fill_(-0.50)
        sparse.output_delta[0].fill_(0.10)
        sparse.output_delta[1].fill_(-0.20)

    ids = torch.tensor([[0, 1, 2, 3, 4]])
    live = model.embed(ids)
    base = torch.nn.functional.embedding(ids, base_embed)
    assert torch.equal(live[0, 0], base[0, 0])
    assert torch.equal(live[0, 2], base[0, 2])
    assert torch.equal(live[0, 4], base[0, 4])
    assert not torch.equal(live[0, 1], base[0, 1])
    assert not torch.equal(live[0, 3], base[0, 3])

    hidden = torch.tensor([[0.2, -0.3, 0.5, 0.7]])
    live_logits = model.head(hidden)
    base_logits = torch.nn.functional.linear(hidden, base_head)
    for token_id in range(8):
        if token_id in selected:
            assert not torch.equal(live_logits[:, token_id], base_logits[:, token_id])
        else:
            assert torch.equal(live_logits[:, token_id], base_logits[:, token_id])

    rows = torch.tensor([[1.0, 2.0, 3.0, 4.0], [-2.0, 1.0, 0.5, 3.0]])
    bs = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    bp = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    ga = directional.project_into_basis(rows, bs)
    gd = directional.project_into_basis(rows, bp)
    assert torch.allclose(ga[:, 1:], torch.zeros_like(ga[:, 1:]))
    assert torch.allclose(gd[:, 0], torch.zeros_like(gd[:, 0]))
    assert torch.allclose(gd[:, 2:], torch.zeros_like(gd[:, 2:]))

    sparse.materialize(sparse.input_delta.detach(), sparse.output_delta.detach(), 1.0)
    for token_id in range(8):
        if token_id not in selected:
            assert torch.equal(model.embed.weight[token_id], base_embed[token_id])
            assert torch.equal(model.head.weight[token_id], base_head[token_id])

    print("RWKU Directional SURE v2 smoke test PASS")


if __name__ == "__main__":
    main()
