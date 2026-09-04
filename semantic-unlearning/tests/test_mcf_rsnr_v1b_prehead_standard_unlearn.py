import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_rsnr_v1b_prehead_standard_unlearn as v1b


def test_counterfact_margin_matches_zero_unlearn_nll_direction():
    # NLL(true) > NLL(new) + m  <=>  logP(new)-logP(true) > m.
    true_lp = torch.tensor([-5.0])
    new_lp = torch.tensor([-4.0])
    margin = new_lp - true_lp
    assert torch.allclose(margin, torch.tensor([1.0]))
    assert float(margin.item()) > 0.1


def test_target_new_is_detached_from_margin_gradient():
    true_lp = torch.tensor([-1.0, -1.2], requires_grad=True)
    new_lp = torch.tensor([-1.1, -1.4], requires_grad=True)
    idk_lp = torch.tensor([-0.8, -0.9], requires_grad=True)
    base_true = torch.tensor([-0.5, -0.6])
    terms = v1b.compute_objective_terms(
        true_lp,
        new_lp,
        idk_lp,
        base_true,
        owners=[0, 1],
        case_count=2,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
    )
    # Isolate the CF term. target_new defines the boundary but receives no grad.
    terms["cf_margin_loss"].backward()
    assert true_lp.grad is not None
    assert torch.any(true_lp.grad != 0)
    assert new_lp.grad is None or torch.all(new_lp.grad == 0)


def test_four_condition_summary_requires_every_condition_per_case():
    rows = [
        {"case_id": 1, "worst_cf_margin": 0.2, "worst_idk_margin": 0.3, "worst_true_drop": 2.5},
        {"case_id": 2, "worst_cf_margin": 0.2, "worst_idk_margin": 0.05, "worst_true_drop": 2.5},
        {"case_id": 3, "worst_cf_margin": -0.1, "worst_idk_margin": 0.4, "worst_true_drop": 3.0},
    ]
    out = v1b.summarize_four_conditions(
        case_rows=rows,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
    )
    assert out["joint_passed"] == 1
    assert out["joint_failures"] == 2
    assert out["cf_passed"] == 2
    assert out["idk_passed"] == 2
    assert out["drop_passed"] == 3
    assert out["minimum_worst_cf_margin"] == -0.1
    assert out["minimum_worst_idk_margin"] == 0.05
    assert out["minimum_worst_true_drop"] == 2.5


def test_default_rank_is_16_and_shape_parameter_formula_is_98304_for_llama3b(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--model-path", "m",
            "--protocol-dir", "p",
            "--view-corpus", "v",
            "--output-dir", "o",
        ],
    )
    args = v1b.parse_args()
    assert args.adapter_rank == 16
    assert 2 * 3072 * args.adapter_rank == 98304


def test_ga_loss_direction_pushes_true_logprob_down():
    true_lp = torch.tensor([-1.0], requires_grad=True)
    new_lp = torch.tensor([-3.0])
    idk_lp = torch.tensor([-2.0])
    base_true = torch.tensor([-0.5])
    terms = v1b.compute_objective_terms(
        true_lp,
        new_lp,
        idk_lp,
        base_true,
        owners=[0],
        case_count=1,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
    )
    terms["ga_loss"].backward()
    # Gradient descent subtracts a positive gradient, making logP(true) smaller.
    assert true_lp.grad is not None
    assert float(true_lp.grad.item()) > 0.0
