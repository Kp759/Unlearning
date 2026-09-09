"""Forget weighting, candidate comparison, and checkpoint selection invariants."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from static_overlap_core import constrained_step, set_parameters
from static_overlap_training import (TrainConfig, ValidCheckpointSelection, hard_example_weights,
                                     near_budget_anchor_ids, training_protection, within_budgets)


def protected(nll, kl):
    return [{"id": "r", "role": "retain", "nll": 1. + nll, "base_nll": 1., "nll_increase": nll, "kl": kl}]


def test_weights_are_detached_capped_and_fact_balanced():
    examples = [SimpleNamespace(id=key, fact_id=fact) for key, fact in
                [("a1", "a"), ("a2", "a"), ("b", "b"), ("c", "c")]]
    nlls = {key: torch.tensor(-math.log(probability), requires_grad=True)
            for key, probability in [("a1", .95), ("a2", .8), ("b", .05), ("c", .0001)]}
    config = TrainConfig(hard_example_mix=1., hard_example_cap=3.)
    weights, probabilities = hard_example_weights(examples, nlls, config)
    assert probabilities["a"] == pytest.approx(.95)
    assert weights["a1"] == weights["a2"]
    assert weights["a1"] + weights["a2"] > weights["b"] > weights["c"] > 0
    assert 1 <= weights["a1"] + weights["a2"] <= 3
    assert 1 <= weights["b"] <= 3 and 1 <= weights["c"] <= 3
    loss = sum(weights[key] * torch.relu(15. - value) for key, value in nlls.items()) / sum(weights.values())
    loss.backward()
    for key, value in nlls.items():
        assert value.grad.item() == pytest.approx(-weights[key] / sum(weights.values()))
    uniform, _ = hard_example_weights(examples, nlls, TrainConfig())
    assert set(uniform.values()) == {1.}


def test_internal_safety_margin_does_not_change_scientific_validation_limits():
    config = TrainConfig(retain_nll_safety_margin=.01, retain_kl_safety_margin=.002)
    rows = protected(.041, .0085)
    assert within_budgets(rows, config)[0]
    assert training_protection(rows, config)[0]
    passed, report = training_protection(rows, config, internal=True)
    assert not passed
    assert report["applied_nll_budget"] == .04 and report["applied_kl_budget"] == .008
    assert report["nominal_retain_nll_budget"] == .05 and report["nominal_retain_kl_budget"] == .01
    assert not training_protection(protected(.050001, .009), config)[0]
    assert near_budget_anchor_ids(protected(.039, 0.), config) == {"r"}


@pytest.mark.parametrize("kwargs", [{"retain_nll_safety_margin": .051}, {"retain_kl_safety_margin": .011},
                                    {"hard_example_mix": 1.1}, {"hard_example_cap": .5},
                                    {"compare_forget_candidates": 1}])
def test_invalid_priority_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs).validate()


def compare_candidates(fallback, score, check_override=None):
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    optimizer = torch.optim.Adam([p], lr=1.)
    state = deepcopy(optimizer.state_dict())

    def check():
        # Projection allowances reserve a little room inside these nonlinear
        # limits, just as the configured internal safety margins do.
        passed = p[0] > 0 and p.min() >= -.01 and p.sum() <= .21
        if check_override is not None:
            passed = passed and check_override()
        return bool(passed), {"global_forget_progress": p[0].item()}

    result = constrained_step(optimizer, [p], -p.sum(),
        torch.tensor([[1., 1.], [0., -1.]], dtype=torch.float64), check,
        epsilon=torch.tensor([.2, 0.], dtype=torch.float64), radius=2.,
        fallback_direction=torch.tensor(fallback, dtype=torch.float64),
        candidate_score=score)
    return p, optimizer, state, result


def test_best_candidate_beats_first_accepted_adam_under_identical_budgets():
    first, _, _, old = compare_candidates([1., 0.], None)
    best, optimizer, state, new = compare_candidates([1., 0.], lambda d: d["global_forget_progress"])
    assert old["direction"] == "adam" and new["direction"] == "forget_descent"
    assert best[0] > 1.9 * first[0]
    assert best.sum() <= .21
    assert len(new["candidate_results"]) == 2
    assert new["projection_violation"] <= 1e-7
    assert optimizer.state_dict() == state


def test_losing_fallback_does_not_overwrite_winning_adam_or_its_moments():
    p, optimizer, _, result = compare_candidates([-1., -1.], lambda d: d["global_forget_progress"])
    assert result["accepted"] and result["direction"] == "adam"
    assert p[0].item() == pytest.approx(.1)
    assert next(iter(optimizer.state.values()))["step"].item() == 1


def test_winning_candidate_is_rechecked_and_rolled_back_on_failure():
    calls = 0

    def check():
        nonlocal calls
        calls += 1
        return calls < 3

    p, optimizer, state, result = compare_candidates([1., 0.], lambda d: d["global_forget_progress"], check)
    assert not result["accepted"] and result["failure_reason"] == "selected_candidate_recheck_failed"
    assert torch.equal(p, torch.zeros_like(p)) and optimizer.state_dict() == state


def test_checkpoint_selection_excludes_invalid_training_and_validation_states():
    config = TrainConfig(retain_nll_safety_margin=.01, retain_kl_safety_margin=.002)
    editor = SimpleNamespace(parameters=[torch.nn.Parameter(torch.tensor([1.]))])
    selection = ValidCheckpointSelection(config, {"f": 14.})
    assert selection.consider(editor, 1, {"f": 1.}, protected(.03, .007), protected(.045, .009))["selected_as_best"]
    set_parameters(editor.parameters, torch.tensor([2.]))
    assert not selection.consider(editor, 2, {"f": 3.}, protected(.03, .007), protected(.06, .009))["checkpoint_eligible"]
    assert not selection.consider(editor, 3, {"f": 3.}, protected(.045, .007), protected(.04, .009))["checkpoint_eligible"]
    assert selection.step == 1 and selection.parameters.item() == 1.
    assert selection.consider(editor, 4, {"f": 2.}, protected(.03, .007), protected(.04, .009))["selected_as_best"]
    assert selection.step == 4 and selection.parameters.item() == 2.
    summary = selection.summary()
    assert not summary["validation_forget_used_for_selection"]
    assert not summary["validation_used_for_gradients"]
    assert not summary["official_evaluation_used_for_selection"]


def test_all_targets_met_checkpoint_beats_one_with_a_smaller_max_but_an_unmet_relative_target():
    editor = SimpleNamespace(parameters=[torch.nn.Parameter(torch.tensor([1.]))])
    selection = ValidCheckpointSelection(TrainConfig(), {"a": 22., "b": 14.})
    selection.consider(editor, 1, {"a": 21., "b": 20.}, protected(0., 0.), protected(0., 0.))
    selection.consider(editor, 2, {"a": 22., "b": 14.}, protected(0., 0.), protected(0., 0.))
    assert selection.step == 2


def test_checkpoint_selection_does_not_trade_a_worse_hardest_fact_for_finishing_an_easy_fact():
    editor = SimpleNamespace(parameters=[torch.nn.Parameter(torch.tensor([1.]))])
    selection = ValidCheckpointSelection(TrainConfig(), {"hard": 14., "easy": 14.})
    selection.consider(editor, 1, {"hard": 2., "easy": 13.}, protected(0., 0.), protected(0., 0.))
    selection.consider(editor, 2, {"hard": 1., "easy": 14.}, protected(0., 0.), protected(0., 0.))
    assert selection.step == 1
    selection.consider(editor, 3, {"hard": 3., "easy": 12.}, protected(0., 0.), protected(0., 0.))
    assert selection.step == 3


def test_no_valid_checkpoint_does_not_select_the_base_or_the_invalid_last_state():
    editor = SimpleNamespace(parameters=[torch.nn.Parameter(torch.tensor([1.]))])
    selection = ValidCheckpointSelection(TrainConfig(), {"f": 14.})
    selection.consider(editor, 1, {"f": 20.}, protected(.04, .009), protected(.2, .02))
    assert selection.step is None and selection.parameters is None
