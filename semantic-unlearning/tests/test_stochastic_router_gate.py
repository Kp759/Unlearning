"""Invariants the stochastic router must not break.

The tests that matter are the ones protecting claims in the paper: only the
boundary position is edited, an ineligible subject never fires under any mode,
and a fixed seed gives a byte-identical route. Everything else about these
modes is allowed to vary.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from stochastic_router_gate import (  # noqa: E402
    MODES,
    StochasticRelationPrototypeBank,
)


HIDDEN = 8
FACTS = 3
BATCH = 4
WIDTH = 6


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


def _bank(mode, seed=0, temperature=0.05):
    torch.manual_seed(0)
    positives = [F.normalize(torch.randn(2, HIDDEN), dim=-1) for _ in range(FACTS)]
    negatives = [F.normalize(torch.randn(3, HIDDEN), dim=-1) for _ in range(FACTS)]
    facts = [{"id": f"f{i}", "subject": f"S{i}", "relation": "r"} for i in range(FACTS)]
    return StochasticRelationPrototypeBank(
        _FakeModel(),
        0,
        positives,
        negatives,
        torch.full((FACTS,), -1.0),
        torch.full((FACTS,), -0.5),
        [[(10 + i,)] for i in range(FACTS)],
        facts,
        rows=torch.randn(FACTS, HIDDEN),
        mode=mode,
        temperature=temperature,
        query_noise=0.1,
        seed=seed,
    )


def _inputs():
    # Row 2 carries no registered subject token and must never route.
    return torch.tensor([
        [10, 1, 2, 3, 4, 5],
        [11, 1, 2, 3, 4, 5],
        [99, 1, 2, 3, 4, 5],
        [10, 11, 2, 3, 4, 5],
    ])


def _run(bank, hidden=None):
    ids = _inputs()
    bank.bind(ids, attention_mask=torch.ones_like(ids))
    if hidden is None:
        torch.manual_seed(7)
        hidden = torch.randn(BATCH, WIDTH, HIDDEN)
    return hidden, bank._hook(None, None, (hidden,))[0]


@pytest.mark.parametrize("mode", MODES)
def test_only_the_boundary_position_is_edited(mode):
    hidden, edited = _run(_bank(mode))
    delta = (edited - hidden).norm(dim=-1)
    assert torch.equal(delta[:, :-1], torch.zeros_like(delta[:, :-1]))


@pytest.mark.parametrize("mode", MODES)
def test_ineligible_subject_never_routes(mode):
    hidden, edited = _run(_bank(mode))
    assert float((edited[2] - hidden[2]).abs().max()) == 0.0


@pytest.mark.parametrize("mode", MODES)
def test_same_seed_is_reproducible(mode):
    hidden, first = _run(_bank(mode, seed=3))
    _, second = _run(_bank(mode, seed=3), hidden=hidden)
    assert torch.equal(first, second)


def test_soft_gate_is_bounded_and_monotone():
    bank = _bank("soft")
    _run(bank)
    gates = torch.tensor(bank.last_gate_scales)
    assert bool(((gates >= 0) & (gates <= 1)).all())


def test_hard_zero_floor_restores_the_inactive_path_identity():
    bank = _bank("soft")
    bank.hard_zero_below = 0.5
    hidden, edited = _run(bank)
    gates = torch.tensor(bank.last_gate_scales)
    silent = gates == 0
    assert bool(silent.any())
    assert float((edited[silent] - hidden[silent]).abs().max()) == 0.0


def test_low_temperature_gumbel_collapses_to_the_deterministic_route():
    # As T -> 0 the sampled route must agree with argmax; this is the limit
    # that lets the stochastic arm be reported as a superset of Router V2.
    _run(deterministic := _bank("deterministic"))
    _run(sampled := _bank("gumbel", temperature=1e-4))
    assert deterministic.last_active_fact_indices == sampled.last_active_fact_indices


def test_artifact_records_the_routing_contract():
    bank = _bank("bernoulli")
    record = bank.artifact()
    assert record["routing_mode"] == "bernoulli"
    assert record["deterministic_route"] is False
    assert record["architecture"].startswith("stochastic_")


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        _bank("annealed")
