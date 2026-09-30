import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from recalibrate_router import choose_cutoff, outcomes  # noqa: E402


def _toy():
    # 2 facts. Rows 0-3 positives (fact 0: 0,1; fact 1: 2,3); rows 4-9 negatives
    # of fact 0 (4-7) and fact 1 (8-9). Each row eligible only for its fact.
    z = torch.full((10, 2), -9.0)
    own = [0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    vals = [3.0, 1.0, 2.5, 0.5, 2.0, -1.0, -2.0, -3.0, 0.8, -0.5]
    for i, (f, v) in enumerate(zip(own, vals)):
        z[i, f] = v
    eligible = torch.zeros(10, 2, dtype=torch.bool)
    eligible[torch.arange(10), torch.tensor(own)] = True
    owner = torch.tensor([0, 0, 1, 1, -1, -1, -1, -1, -1, -1])
    return z, eligible, owner


def test_outcomes_pooled_and_macro():
    z, e, o = _toy()
    r = outcomes(z, e, o, 0.9, margin=0.0)       # fires rows 0,1,2,4
    assert r["recall"] == 0.75 and r["false_fire"] == 1 / 6
    assert abs(r["macro_recall"] - 0.75) < 1e-12              # (1 + 0.5) / 2
    assert abs(r["macro_false_fire"] - (0.25 + 0.0) / 2) < 1e-12


def test_balanced_is_the_argmax_and_inside_its_interval():
    z, e, o = _toy()
    t, chosen = choose_cutoff(z, e, o, 0.0, "balanced", macro="fact")
    grid = [outcomes(z, e, o, c, 0.0)["macro_balanced_accuracy"] for c in
            torch.linspace(-4, 4, 801).tolist()]
    assert abs(chosen["macro_balanced_accuracy"] - max(grid)) < 1e-12


def test_target_fpr_respects_the_cap():
    z, e, o = _toy()
    t, chosen = choose_cutoff(z, e, o, 0.0, "target_fpr", target_fpr=0.0)
    assert chosen["false_fire"] == 0.0 and chosen["correct"] == 2  # only 3.0, 2.5 above 2.0


def test_constrained_meets_both_when_feasible():
    z, e, o = _toy()
    # recall >= 0.5 and false fire <= 0.25 (macro) is feasible
    t, chosen = choose_cutoff(z, e, o, 0.0, "constrained", macro="fact",
                              min_recall=0.5, target_fpr=0.25)
    assert chosen["constraint_status"] == "recall_and_false_fire_met"
    assert chosen["macro_recall"] >= 0.5 and chosen["macro_false_fire"] <= 0.25
    grid = [outcomes(z, e, o, c, 0.0) for c in torch.linspace(-4, 4, 801).tolist()]
    feas = [g["macro_balanced_accuracy"] for g in grid
            if g["macro_recall"] >= 0.5 and g["macro_false_fire"] <= 0.25]
    assert abs(chosen["macro_balanced_accuracy"] - max(feas)) < 1e-12


def test_constrained_keeps_fpr_cap_when_infeasible():
    z, e, o = _toy()
    t, chosen = choose_cutoff(z, e, o, 0.0, "constrained", macro="fact",
                              min_recall=1.0, target_fpr=0.0)
    assert chosen["constraint_status"] == "false_fire_cap_met_recall_short"
    assert chosen["macro_false_fire"] == 0.0
