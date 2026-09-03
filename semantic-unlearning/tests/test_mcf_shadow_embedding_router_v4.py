from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_embedding_keyed_neuron_core as legacy_core
import mcf_shadow_embedding_router_core as core
import mcf_shadow_relation_prompts as prompts
import scoped_span_edit as scoped
import train_mcf_shadow_embedding_router_v4 as train


class IdentityTupleLayer(nn.Module):
    def forward(self, hidden, **_kwargs):
        return (hidden, "cache")


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([IdentityTupleLayer()])

    def forward(self, input_ids=None, inputs_embeds=None, **_kwargs):
        raise AssertionError("ToyLM drives decoder layers directly")


class ToyLM(nn.Module):
    def __init__(self, vocab: int = 12, hidden: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab, hidden)
        self.model = ToyBackbone()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.config = SimpleNamespace(use_cache=False)
        with torch.no_grad():
            self.embed.weight.copy_(
                torch.arange(vocab * hidden, dtype=torch.float32).reshape(vocab, hidden)
                / 20.0
            )
            self.lm_head.weight.copy_(
                torch.arange(vocab * hidden, dtype=torch.float32)
                .flip(0)
                .reshape(vocab, hidden)
                / 30.0
            )

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        use_cache=False,
        return_dict=True,
        **_kwargs,
    ):
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden), hidden=hidden)


def _runtime():
    model = ToyLM()
    writer = legacy_core.ToggleableEmbeddingDelta(
        model.embed,
        [3],
        torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32),
    )
    writer.enabled = False
    span = scoped.SpanGateRouter(
        model.embed, [[[2, 3]]], subjects=["Alpha Beta"], model=model
    )
    semantic = core.PassiveShadowRouter(1, 3)
    with torch.no_grad():
        semantic.weight[0] = torch.tensor([1.0, 0.0, 0.0])
    semantic.weight.requires_grad_(False)
    branch = core.ShadowEmbeddingResidualBranch(
        model.model.layers[0],
        span,
        semantic,
        3,
        residual_reference_norms=torch.ones(1),
    )
    with torch.no_grad():
        branch.residual[0] = torch.tensor([0.0, 1.0, 0.0])
    wrapper = core.ShadowDualPathCausalLM(model, writer, branch)
    return model, writer, span, semantic, branch, wrapper


def test_causal_subject_mask_waits_for_complete_multitoken_key():
    spans = torch.tensor([[[False, True, True, False]]])
    observed = core.causal_after_subject_mask(spans)
    assert observed.tolist() == [[[False, False, True, True]]]


def test_closed_subject_route_is_bit_identical_and_base_embedding_never_changes():
    model, writer, span, _semantic, branch, wrapper = _runtime()
    inputs = torch.tensor([[0, 9, 3, 4]], dtype=torch.long)  # shared piece, no [2, 3]
    embedding_before = model.embed.weight.detach().clone()
    try:
        writer.enabled = False
        branch.enabled = False
        base = model(input_ids=inputs).logits.detach().clone()
        branch.enabled = True
        edited = wrapper(input_ids=inputs).logits.detach()
        assert torch.equal(edited, base)
        assert torch.equal(model.embed.weight, embedding_before)
        assert branch.last_gates is not None
        assert int(branch.last_gates.sum()) == 0
        assert writer.enabled is False
    finally:
        branch.close()
        span.close()
        writer.remove()


def test_open_route_uses_shadow_feature_but_writes_only_to_base_path():
    model, writer, span, _semantic, branch, wrapper = _runtime()
    inputs = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
    try:
        writer.enabled = False
        branch.enabled = False
        base = model(input_ids=inputs).logits.detach().clone()
        branch.enabled = True
        edited = wrapper(input_ids=inputs).logits.detach()
        assert branch.last_gates is not None
        assert branch.last_gates[0, :, 0].tolist() == [False, False, True, True]

        # The main path did not receive the +2 embedding marker.  It receives
        # only the constant y residual at/after the complete subject.
        expected_hidden = model.embed(inputs).detach().clone()
        expected_hidden[0, 2:, 1] += 1.0
        expected = model.lm_head(expected_hidden)
        torch.testing.assert_close(edited, expected)
        assert not torch.equal(edited, base)
    finally:
        branch.close()
        span.close()
        writer.remove()


def test_disabling_shadow_embedding_closes_gate_by_construction():
    model, writer, span, semantic, branch, wrapper = _runtime()
    inputs = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
    try:
        writer.enabled = False
        branch.enabled = False
        base = model(input_ids=inputs).logits.detach().clone()
        branch.enabled = True
        branch.shadow_writer_enabled = False
        observed = wrapper(input_ids=inputs).logits.detach()
        assert torch.equal(observed, base)
        assert branch.last_scores is not None
        assert torch.equal(branch.last_scores, torch.full_like(branch.last_scores, -1.0))
        assert branch.last_gates is not None and int(branch.last_gates.sum()) == 0
        zero = semantic.scores(torch.zeros(4, 3))
        assert torch.equal(zero, torch.full((4, 1), -1.0))
    finally:
        branch.close()
        span.close()
        writer.remove()


def test_router_objective_is_multilabel_and_record_balanced():
    scores = torch.tensor(
        [[0.0, -1.0], [1.5, 0.2], [-1.0, 1.5]], requires_grad=True
    )
    active = torch.tensor(
        [[True, False], [True, True], [False, True]], dtype=torch.bool
    )
    labels = torch.tensor(
        [[True, False], [True, False], [False, True]], dtype=torch.bool
    )
    loss, metrics = core.record_balanced_router_hinge(
        scores, active, labels, positive_floor=1.0, negative_ceiling=-0.25
    )
    loss.backward()
    assert float(loss) > 0
    assert int(metrics["positive_violations"]) == 1
    assert int(metrics["negative_violations"]) == 1
    assert scores.grad is not None


def _records():
    return [
        {"case_id": 1, "subject": "Alpha", "relation_id": "P27"},
        {"case_id": 2, "subject": "Alpha", "relation_id": "P106"},
        {"case_id": 3, "subject": "Beta", "relation_id": "P36"},
    ]


def test_development_prompt_families_are_disjoint_and_avoid_subject_conflicts():
    records = _records()
    calibration = prompts.build_positive_specs(records, split="calibration")
    heldout = prompts.build_positive_specs(records, split="heldout")
    assert {row.prompt for row in calibration}.isdisjoint(
        {row.prompt for row in heldout}
    )
    negatives = prompts.build_wrong_relation_specs(
        records, split="heldout", variants_per_record=3
    )
    for row in negatives:
        source_subject = records[row.owner_index]["subject"]
        forbidden = {
            record["relation_id"]
            for record in records
            if record["subject"] == source_subject
        }
        assert row.relation_id not in forbidden


def test_registry_locks_base_embedding_main_path():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "mcf_shadow_embedding_router_v4_0_registry.json"
    )
    registry = json.loads(registry_path.read_text())
    args = SimpleNamespace(layer=27)
    train.validate_registry(registry, args)
    assert registry["architecture"]["base_embedding_main_path"] is True
    assert registry["architecture"]["v6_2_delta_shadow_path_only"] is True
    assert registry["architecture"]["base_embedding_mutation"] is False
    assert registry["seed_1_blind_official_reuse_prohibited"] is True


def test_candidate_state_explicitly_retains_shadow_delta_without_base_mutation():
    model, writer, span, semantic, branch, _wrapper = _runtime()
    try:
        state = core.shadow_candidate_state(
            layer_index=0,
            case_ids=[7],
            subjects=["Alpha Beta"],
            subject_patterns=[[[2, 3]]],
            embedding_row_ids=[3],
            embedding_delta=torch.tensor([[2.0, 0.0, 0.0]]),
            semantic_router=semantic,
            branch=branch,
            source_hashes={"writer": "a" * 64},
        )
        assert state["architecture"]["main_embedding_path"] == "unaltered_base_embedding"
        assert state["architecture"]["shadow_embedding_path"] == "frozen_v6_2_sparse_delta"
        assert state["architecture"]["base_embedding_mutated"] is False
        assert torch.equal(state["embedding_delta"], torch.tensor([[2.0, 0.0, 0.0]]))

        fresh = ToyLM()
        embedding_before = fresh.embed.weight.detach().clone()
        installed = core.install_shadow_candidate(fresh, state)
        try:
            observed = installed.model(input_ids=torch.tensor([[0, 2, 3, 4]])).logits
            assert installed.branch.last_gates is not None
            assert int(installed.branch.last_gates.sum()) == 2
            assert torch.equal(fresh.embed.weight, embedding_before)
            assert observed.shape == (1, 4, 12)
        finally:
            installed.close()
    finally:
        branch.close()
        span.close()
        writer.remove()


def test_firewall_rejects_official_paths(monkeypatch):
    monkeypatch.setenv("OFFICIAL_MCF_PATH", "/tmp/forbidden.json")
    with pytest.raises(RuntimeError, match="official-evaluation path leaked"):
        train.validate_environment_firewall()
