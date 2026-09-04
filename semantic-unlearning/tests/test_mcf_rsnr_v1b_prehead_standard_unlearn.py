from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_rsnr_v1b_prehead_standard_unlearn as v1b


def test_prompts_for_cases_with_targets_uses_true_and_new():
    cases = [{
        "case_id": 7,
        "requested_rewrite": {
            "subject": "Belgium",
            "relation_id": "P36",
            "target_true": {"str": "Brussels"},
            "target_new": {"str": "Paris"},
        },
    }]
    views = {7: ["The capital of {} is", "{} has capital"]}
    prompts, true_answers, new_answers, owners = v1b.prompts_for_cases_with_targets(cases, views)
    assert prompts == ["The capital of Belgium is", "Belgium has capital"]
    assert true_answers == ["Brussels", "Brussels"]
    assert new_answers == ["Paris", "Paris"]
    assert owners == [0, 0]


def test_four_constraint_losses_zero_when_all_margins_pass():
    # new-true margins: 0.4, 0.3; IDK-true: 0.6, 0.5; Base true drop: 2.5, 2.2.
    true_lp = torch.tensor([-4.0, -3.0])
    new_lp = torch.tensor([-3.6, -2.7])
    idk_lp = torch.tensor([-3.4, -2.5])
    base_true = torch.tensor([-1.5, -0.8])
    out = v1b.four_constraint_losses(
        true_lp=true_lp,
        new_lp=new_lp,
        idk_lp=idk_lp,
        base_true_lp=base_true,
        owners=[0, 0],
        case_count=1,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
    )
    assert out["cf_loss"].item() == pytest.approx(0.0)
    assert out["idk_loss"].item() == pytest.approx(0.0)
    assert out["suppression_loss"].item() == pytest.approx(0.0)
    assert out["cf_margin"].tolist() == pytest.approx([0.4, 0.3])


def test_four_constraint_losses_use_worst_view_per_case():
    true_lp = torch.tensor([-4.0, -3.0, -5.0, -4.0])
    new_lp = torch.tensor([-3.95, -2.7, -4.7, -3.7])  # case0 first view fails 0.1 CF margin
    idk_lp = torch.tensor([-3.8, -2.8, -4.7, -3.7])
    base_true = torch.tensor([-1.5, -1.0, -2.5, -1.5])
    out = v1b.four_constraint_losses(
        true_lp=true_lp,
        new_lp=new_lp,
        idk_lp=idk_lp,
        base_true_lp=base_true,
        owners=[0, 0, 1, 1],
        case_count=2,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
    )
    # case0 worst CF hinge = 0.05, case1 = 0; mean = 0.025.
    assert out["cf_loss"].item() == pytest.approx(0.025, abs=1e-6)


def test_rank_is_locked_to_16():
    minimal = [
        "--model-path", "m",
        "--protocol-dir", "p",
        "--view-corpus", "v",
        "--output-dir", "o",
        "--adapter-rank", "32",
    ]
    with pytest.raises(SystemExit):
        v1b.parse_args(minimal)


def test_zero_unlearn_condition_sign_is_correct():
    # NLL(true)=4.0, NLL(new)=3.6 -> NLL(true)-NLL(new)=0.4.
    # In log-prob space this is logP(new)-logP(true)=0.4.
    true_lp = torch.tensor([-4.0])
    new_lp = torch.tensor([-3.6])
    out = v1b.four_constraint_losses(
        true_lp=true_lp,
        new_lp=new_lp,
        idk_lp=torch.tensor([-3.5]),
        base_true_lp=torch.tensor([-1.0]),
        owners=[0],
        case_count=1,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
    )
    assert out["cf_margin"].item() == pytest.approx(0.4)
    assert out["cf_loss"].item() == pytest.approx(0.0)
