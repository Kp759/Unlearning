"""Compatibility wrapper for exact frozen-base fact-association evaluations.

The edited evaluators expect a model with set_association_prefix_lengths() and
an audit bank exposing last_active_fact_indices.  The frozen base has no route
or intervention, so these APIs are intentional no-ops while the underlying
forward pass is exactly the untouched pretrained model.
"""
from __future__ import annotations

from torch import nn


class NoRouteBatchBank:
    def __init__(self):
        self.last_active_fact_indices = []
        self.calls = 0

    def note_batch(self, batch_size):
        self.calls += 1
        self.last_active_fact_indices = [[] for _ in range(int(batch_size))]

    def counters(self):
        return {
            "hook_calls": 0,
            "active_batch_rows": 0,
            "active_token_positions": 0,
            "active_fact_counts": [],
            "compat_forward_calls": self.calls,
        }


class FrozenBaseAssociationCompatLM(nn.Module):
    def __init__(self, base_model, bank=None):
        super().__init__()
        self.base_model = base_model
        self.bank = bank if bank is not None else NoRouteBatchBank()
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        self._next_prefix_lengths = None

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_association_prefix_lengths(self, lengths):
        # Kept only for evaluator API parity. Frozen base has no intervention.
        self._next_prefix_lengths = lengths

    def forward(self, input_ids=None, **kwargs):
        if input_ids is None:
            raise ValueError("Frozen-base compatibility wrapper requires input_ids")
        self._next_prefix_lengths = None
        self.bank.note_batch(int(input_ids.shape[0]))
        return self.base_model(input_ids=input_ids, **kwargs)
