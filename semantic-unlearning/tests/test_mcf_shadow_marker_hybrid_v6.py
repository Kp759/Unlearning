from __future__ import annotations

from pathlib import Path
import sys

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_shadow_marker_hybrid_v6_core as hybrid
from scoped_span_edit import RouteState


class StubRouter:
    def __init__(self, active: bool, *, sequence: int = 3) -> None:
        self.state = RouteState(
            active=torch.tensor([[active]], dtype=torch.bool),
            span_masks=torch.zeros((1, 1, sequence), dtype=torch.bool),
        )


class ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(3, 4, bias=False)
        self.up_proj = nn.Linear(3, 4, bias=False)
        self.down_proj = nn.Linear(4, 3, bias=False)
        self.act_fn = torch.nn.functional.silu
        with torch.no_grad():
            self.gate_proj.weight.fill_(0.25)
            self.up_proj.weight.fill_(0.5)
            self.down_proj.weight.fill_(0.1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activation = self.act_fn(self.gate_proj(hidden)) * self.up_proj(hidden)
        return self.down_proj(activation)


def _bank(mlp: ToyMLP, router: StubRouter, mode: str):
    bank = hybrid.OuterRoutedThresholdGatedActuatorBank(
        mlp,
        [0],
        [0],
        outer_router=router,
        outer_gate_mode=mode,
        detector_gate_rows=torch.zeros((1, 3)),
        detector_up_rows=torch.zeros((1, 3)),
        detector_local_groups=[[0]],
        detector_flat_signs=torch.ones(1),
        off_boundary=0.2,
        on_boundary=0.25,
    )
    with torch.no_grad():
        bank.down_delta.fill_(1.0)
    bank.install(mlp)
    return bank


def test_outer_closed_actuator_is_algebraically_exact():
    mlp = ToyMLP()
    hidden = torch.ones((1, 3, 3))
    base = mlp(hidden).detach().clone()
    bank = _bank(mlp, StubRouter(False), "outer_direct")
    try:
        observed = mlp(hidden).detach()
        assert torch.equal(observed, base)
        assert bank.outer_closed_calls == 1
    finally:
        bank.remove()


def test_outer_direct_and_outer_detector_are_distinct_ablation_arms():
    hidden = torch.ones((1, 3, 3))
    direct_mlp = ToyMLP()
    direct_base = direct_mlp(hidden).detach().clone()
    direct = _bank(direct_mlp, StubRouter(True), "outer_direct")
    try:
        direct_observed = direct_mlp(hidden).detach()
        assert not torch.equal(direct_observed, direct_base)
    finally:
        direct.remove()

    detector_mlp = ToyMLP()
    detector_base = detector_mlp(hidden).detach().clone()
    detector = _bank(detector_mlp, StubRouter(True), "outer_and_detector")
    try:
        # Zero detector rows keep the learned conjunction closed even though
        # the deterministic outer route is open.
        detector_observed = detector_mlp(hidden).detach()
        assert torch.equal(detector_observed, detector_base)
    finally:
        detector.remove()


def test_shadow_writer_changes_only_authorized_rows_and_never_weights():
    embedding = nn.Embedding(8, 3)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(24).reshape(8, 3).float())
    token_ids = torch.tensor([[1, 2, 3]])
    weights_before = embedding.weight.detach().clone()

    closed = hybrid.RoutedShadowEmbeddingDelta(
        embedding,
        StubRouter(False),
        [2],
        torch.ones((1, 3)),
    )
    try:
        base = embedding(token_ids).detach().clone()
        assert torch.equal(base, embedding.weight[token_ids])
    finally:
        closed.remove()

    opened = hybrid.RoutedShadowEmbeddingDelta(
        embedding,
        StubRouter(True),
        [2],
        torch.ones((1, 3)),
    )
    try:
        observed = embedding(token_ids).detach()
        expected = embedding.weight[token_ids].detach().clone()
        expected[:, 1] += 1.0
        assert torch.equal(observed, expected)
        assert torch.equal(embedding.weight, weights_before)
    finally:
        opened.remove()
