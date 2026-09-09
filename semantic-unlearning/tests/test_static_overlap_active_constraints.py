"""Exercise useful updates when rotating protection omits a limiting anchor."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from static_overlap_core import constrained_step
from static_overlap_training import TrainConfig, near_budget_anchor_ids, training_protection


def row(key, nll=0., kl=0., role="retain"):
    return {"id": key, "role": role, "base_nll": 1., "nll": 1. + nll,
            "nll_increase": nll, "kl": kl}


def test_tight_nll_and_kl_anchors_stay_active_without_changing_budgets():
    config = TrainConfig()
    rows = [row("tight_nll", nll=.049), row("tight_kl", kl=.0095, role="language"),
            row("loose", nll=.01, kl=.001), row("forget", nll=20., role="forget")]
    assert near_budget_anchor_ids(rows, config) == {"tight_nll", "tight_kl"}
    passed, report = training_protection(rows, config)
    assert passed
    assert report["max_retained_nll_anchor_id"] == "tight_nll"
    assert report["max_retained_kl_anchor_id"] == "tight_kl"
    passed, report = training_protection(rows + [row("over", nll=.050001)], config)
    assert not passed  # Export slack must not enter training or active selection.
    assert report["violating_anchor_ids"] == ["over"]
    assert config.retain_nll_budget == .05 and config.retain_kl_budget == .01


@pytest.mark.parametrize("field", ["nll", "kl"])
def test_missing_anchor_is_projected_instead_of_shrinking_useful_direction(field):
    def run(refine):
        p = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.Adam([p], lr=1.)

        def check():
            # p[0] improves forgetting; p[1] harms an unsampled retained span.
            passed = p[1].item() <= .001
            return passed and p[0].item() > 0, {
                "violating_anchor_ids": [] if passed else [field]}

        def expand(diagnostics):
            assert torch.equal(p, torch.zeros(2))  # Not the rejected trial point.
            assert diagnostics["violating_anchor_ids"] == [field]
            return torch.tensor([[0., 1.]]), torch.tensor([.0005])

        record = constrained_step(optimizer, [p], -p.sum(), torch.empty(0, 2), check,
                                  epsilon=0., radius=2., refine_constraints=expand if refine else None)
        return p.detach(), optimizer, record

    old, _, old_record = run(False)
    new, optimizer, record = run(True)
    assert old_record["accepted"] and old_record["backtracks"] == 10
    assert record["accepted"] and record["backtracks"] == 0
    assert new[0] > 1000 * old[0]
    assert 0 <= new[1] <= .001
    assert record["constraint_refinements"] == 1 and record["nonlinear_checks"] == 2
    assert next(iter(optimizer.state.values()))["step"].item() == 1


def test_refinement_still_backtracks_on_nonlinear_budget():
    p = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([p], lr=1.)

    def check():
        # Linear projection removes p[1] but curvature in p[0] remains.
        return (p[1] + p[0].square()).item() <= .02, {}

    def expand(_):
        return torch.tensor([[0., 1.]]), torch.tensor([0.])

    record = constrained_step(optimizer, [p], -p.sum(), torch.empty(0, 2), check,
                              epsilon=0., radius=2., refine_constraints=expand,
                              max_constraint_refinements=1)
    assert record["accepted"] and record["backtracks"] > 0
    assert (p[1] + p[0].square()).item() <= .02


@pytest.mark.parametrize("raises", [False, True])
def test_failed_refinement_restores_parameters_and_adam_state(raises):
    p = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([p], lr=1.)
    state = deepcopy(optimizer.state_dict())

    def expand(_):
        assert torch.equal(p, torch.zeros(2))
        if raises:
            raise RuntimeError("gradient computation failed")
        return torch.eye(2), torch.zeros(2)

    def attempt():
        return constrained_step(optimizer, [p], -p.sum(), torch.empty(0, 2),
                                lambda: (False, {}), epsilon=0., radius=2.,
                                refine_constraints=expand)

    if raises:
        with pytest.raises(RuntimeError, match="gradient computation failed"):
            attempt()
    else:
        assert not attempt()["accepted"]
    assert torch.equal(p, torch.zeros(2))
    assert optimizer.state_dict() == state


def test_discovered_constraints_also_apply_to_forget_fallback():
    p = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.Adam([p], lr=1.)
    state = deepcopy(optimizer.state_dict())
    expanded = False

    def expand(_):
        nonlocal expanded
        if expanded:
            return None
        expanded = True
        return torch.tensor([[0., 1.]]), torch.tensor([.0005])

    record = constrained_step(optimizer, [p], p[0] - p[1], torch.empty(0, 2),
                              lambda: (p[0].item() > .1 and p[1].item() <= .001, {}),
                              epsilon=0., radius=2., refine_constraints=expand,
                              fallback_direction=torch.ones(2))
    assert record["accepted"] and record["direction"] == "forget_descent"
    assert p[0].item() > .9 and p[1].item() <= .001
    assert record["constraint_refinements"] == 1
    assert optimizer.state_dict() == state
