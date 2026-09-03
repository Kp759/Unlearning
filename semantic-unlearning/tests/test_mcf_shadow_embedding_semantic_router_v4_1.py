from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_embedding_keyed_neuron_core as legacy_core
import mcf_shadow_embedding_router_core as v4_core
import mcf_shadow_embedding_semantic_router_core as core
import mcf_shadow_relation_prompts as prompts
import scoped_span_edit as scoped
import train_mcf_shadow_embedding_semantic_router_v4_1 as train


class IdentityTupleLayer(nn.Module):
    def forward(self, hidden, **_kwargs):
        return (hidden, "cache")


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([IdentityTupleLayer()])


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


def _runtime(*, semantic_open: bool = True):
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
    semantic = core.BaseSemanticRouter(
        1,
        3,
        torch.tensor([[1.0, 0.0, 0.0]]),
        [0],
    )
    with torch.no_grad():
        semantic.relation_coefficients.zero_()
        semantic.relation_bias.fill_(1.0 if semantic_open else -1.0)
    for parameter in semantic.parameters():
        parameter.requires_grad_(False)
    branch = core.ShadowMarkerSemanticResidualBranch(
        model.model.layers[0],
        span,
        semantic,
        3,
        marker_rms_threshold=1e-6,
        residual_reference_norms=torch.ones(1),
    )
    with torch.no_grad():
        branch.residual[0] = torch.tensor([0.0, 1.0, 0.0])
    wrapper = v4_core.ShadowDualPathCausalLM(model, writer, branch)
    return model, writer, span, semantic, branch, wrapper


def test_shared_contrast_basis_is_low_rank_orthonormal_and_relation_tied():
    states = torch.tensor(
        [
            [3.0, 0.0, 1.0, 0.0],
            [2.5, 0.2, 1.0, 0.0],
            [-2.0, 0.0, 1.0, 0.0],
            [-2.5, 0.1, 1.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [0.0, -3.0, 0.0, 1.0],
        ]
    )
    active = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [True, False, False],
            [False, True, False],
            [False, False, True],
            [False, False, True],
        ]
    )
    labels = torch.tensor(
        [
            [True, False, False],
            [False, True, False],
            [False, False, False],
            [False, False, False],
            [False, False, True],
            [False, False, False],
        ]
    )
    basis, report = core.fit_shared_contrast_basis(
        states, active, labels, [0, 0, 1], rank=2
    )
    torch.testing.assert_close(basis @ basis.T, torch.eye(2), atol=1e-5, rtol=1e-5)
    router = core.BaseSemanticRouter(3, 4, basis, [0, 0, 1])
    assert router.trainable_parameter_count == 6
    assert router.trainable_parameter_count < 3 * 4
    assert report["basis_trainable"] is False
    assert report["rank"] == 2
    scores = router.scores(states)
    torch.testing.assert_close(scores[:, 0], scores[:, 1])


def test_factorized_gate_requires_subject_marker_and_semantics_without_base_edit():
    model, writer, span, _semantic, branch, wrapper = _runtime()
    inputs = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
    embedding_before = model.embed.weight.detach().clone()
    try:
        branch.enabled = False
        base = model(input_ids=inputs).logits.detach().clone()
        branch.enabled = True
        edited = wrapper(input_ids=inputs).logits.detach()
        assert branch.last_gates is not None
        assert branch.last_gates[0, :, 0].tolist() == [False, False, True, True]
        assert not torch.equal(edited, base)
        assert torch.equal(model.embed.weight, embedding_before)

        branch.shadow_writer_enabled = False
        writer_off = wrapper(input_ids=inputs).logits.detach()
        assert torch.equal(writer_off, base)
        assert branch.last_marker_rms is not None
        assert torch.equal(branch.last_marker_rms, torch.zeros_like(branch.last_marker_rms))
        assert branch.last_gates is not None and int(branch.last_gates.sum()) == 0
    finally:
        branch.close()
        span.close()
        writer.remove()

    model, writer, span, _semantic, branch, wrapper = _runtime(semantic_open=False)
    try:
        branch.enabled = False
        base = model(input_ids=inputs).logits.detach().clone()
        branch.enabled = True
        observed = wrapper(input_ids=inputs).logits.detach()
        assert torch.equal(observed, base)
        assert branch.last_marker_rms is not None
        assert float(branch.last_marker_rms.max()) > 0.0
        assert branch.last_gates is not None and int(branch.last_gates.sum()) == 0
    finally:
        branch.close()
        span.close()
        writer.remove()


def test_candidate_round_trip_stores_frozen_shared_basis_not_record_vectors():
    model, writer, span, semantic, branch, _wrapper = _runtime()
    try:
        state = core.factorized_candidate_state(
            layer_index=0,
            case_ids=[7],
            subjects=["Alpha Beta"],
            subject_patterns=[[[2, 3]]],
            embedding_row_ids=[3],
            embedding_delta=torch.tensor([[2.0, 0.0, 0.0]]),
            relation_ids=["P27"],
            semantic_router=semantic,
            branch=branch,
            source_hashes={"writer": "a" * 64},
        )
        assert "semantic_router_weight" not in state
        assert state["semantic_shared_basis"].shape == (1, 3)
        assert state["semantic_relation_coefficients"].shape == (1, 1)
        assert state["architecture"]["semantic_basis_trainable"] is False
    finally:
        branch.close()
        span.close()
        writer.remove()

    fresh = ToyLM()
    embedding_before = fresh.embed.weight.detach().clone()
    installed = core.install_factorized_candidate(fresh, state)
    try:
        observed = installed.model(
            input_ids=torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
        ).logits
        assert observed.shape == (1, 4, 12)
        assert torch.equal(fresh.embed.weight, embedding_before)
        assert installed.semantic_router.shared_basis.requires_grad is False
        assert installed.semantic_router.relation_coefficients.requires_grad is False
    finally:
        installed.close()


def _records():
    return [
        {"case_id": 1, "subject": "Alpha", "relation_id": "P27"},
        {"case_id": 2, "subject": "Alpha", "relation_id": "P106"},
        {"case_id": 3, "subject": "Beta", "relation_id": "P36"},
    ]


def test_three_way_prompt_banks_are_disjoint_and_fit_has_128_negatives_per_record():
    records = _records()
    prefixes = [f"Unrelated training sentence number {index}." for index in range(256)]
    fit = [
        *prompts.build_positive_specs(
            records, split="calibration", corpus_prefixes=prefixes, variants_per_record=8
        ),
        *prompts.build_positive_specs(
            records, split="heldout", corpus_prefixes=prefixes, variants_per_record=8
        ),
        *prompts.build_wrong_relation_specs(
            records, split="calibration", corpus_prefixes=prefixes, variants_per_record=64
        ),
        *prompts.build_wrong_relation_specs(
            records, split="heldout", corpus_prefixes=prefixes, variants_per_record=64
        ),
    ]
    development = [
        *prompts.build_positive_specs(records, split="v4_1_development"),
        *prompts.build_wrong_relation_specs(
            records, split="v4_1_development", variants_per_record=32, corpus_prefixes=prefixes
        ),
    ]
    certification = [
        *prompts.build_positive_specs(records, split="v4_1_certification"),
        *prompts.build_wrong_relation_specs(
            records,
            split="v4_1_certification",
            variants_per_record=64,
            corpus_prefixes=prefixes,
        ),
    ]
    fit_prompts = {row.prompt for row in fit}
    development_prompts = {row.prompt for row in development}
    certification_prompts = {row.prompt for row in certification}
    assert fit_prompts.isdisjoint(development_prompts)
    assert fit_prompts.isdisjoint(certification_prompts)
    assert development_prompts.isdisjoint(certification_prompts)
    for owner in range(len(records)):
        negatives = [row for row in fit if row.owner_index == owner and not row.positive]
        assert len(negatives) == 128
        assert len({row.prompt for row in negatives}) == 128


def test_registry_locks_capacity_and_three_way_split():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "mcf_shadow_embedding_semantic_router_v4_1_registry.json"
    )
    registry = json.loads(registry_path.read_text())
    args = SimpleNamespace(
        layer=27,
        marker_rms_threshold=1e-6,
        semantic_shared_rank=4,
        semantic_router_steps=300,
        semantic_router_lr=0.01,
        semantic_router_weight_decay=0.01,
        semantic_router_coefficient_norm_cap=4.0,
        semantic_selection_check_every=10,
        semantic_selection_patience=10,
        fit_positive_variants_per_old_family=8,
        fit_wrong_relations_per_old_family=64,
        development_positive_variants_per_record=4,
        development_wrong_relations_per_record=32,
        certification_positive_variants_per_record=4,
        certification_wrong_relations_per_record=64,
    )
    train.validate_registry(registry, args)
    architecture = registry["architecture"]
    assert architecture["semantic_shared_rank"] == 4
    assert architecture["semantic_basis_trainable"] is False
    assert architecture["semantic_record_specific_hidden_vectors"] is False
    assert registry["development_prompt_policy"][
        "fit_wrong_relation_negatives_per_record_total"
    ] == 128
    assert registry["official_evaluation_prohibited"] is True
