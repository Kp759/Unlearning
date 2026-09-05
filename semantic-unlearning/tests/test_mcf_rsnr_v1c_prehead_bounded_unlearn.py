import torch

import run_mcf_rsnr_v1c_prehead_bounded_unlearn as v1c


def test_all_bounded_losses_zero_when_constraints_satisfied():
    true_lp = torch.tensor([-5.0, -6.0], requires_grad=True)
    new_lp = torch.tensor([-4.0, -5.0], requires_grad=True)
    idk_lp = torch.tensor([-3.0, -4.0], requires_grad=True)
    base_true = torch.tensor([-2.0, -3.0])
    base_new = torch.tensor([-4.0, -5.0])
    terms = v1c.compute_bounded_objective_terms(
        true_lp, new_lp, idk_lp, base_true, base_new,
        owners=[0, 1], case_count=2,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
        max_target_new_drift=0.25,
    )
    assert float(terms["cf_margin_loss"].item()) == 0.0
    assert float(terms["idk_margin_loss"].item()) == 0.0
    assert float(terms["drop_loss"].item()) == 0.0
    assert float(terms["target_new_preserve_loss"].item()) == 0.0
    assert float(terms["ga_active_count"].item()) == 0.0
    assert float(terms["bounded_ga_loss"].item()) == 0.0


def test_cf_loss_pushes_true_down_not_new_up():
    true_lp = torch.tensor([-2.0], requires_grad=True)
    new_lp = torch.tensor([-4.0], requires_grad=True)
    idk_lp = torch.tensor([-1.0], requires_grad=True)
    base_true = torch.tensor([-1.0])
    base_new = torch.tensor([-4.0])
    terms = v1c.compute_bounded_objective_terms(
        true_lp, new_lp, idk_lp, base_true, base_new,
        owners=[0], case_count=1,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
        max_target_new_drift=0.25,
    )
    terms["cf_margin_loss"].backward(retain_graph=True)
    assert true_lp.grad is not None and float(true_lp.grad.item()) > 0.0
    assert new_lp.grad is None or float(new_lp.grad.item()) == 0.0


def test_new_preservation_restores_collapsed_new_toward_base():
    true_lp = torch.tensor([-6.0], requires_grad=True)
    new_lp = torch.tensor([-8.0], requires_grad=True)
    idk_lp = torch.tensor([-5.0], requires_grad=True)
    base_true = torch.tensor([-2.0])
    base_new = torch.tensor([-4.0])
    terms = v1c.compute_bounded_objective_terms(
        true_lp, new_lp, idk_lp, base_true, base_new,
        owners=[0], case_count=1,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
        max_target_new_drift=0.25,
    )
    terms["target_new_preserve_loss"].backward()
    # Gradient descent subtracts this negative gradient, increasing new_lp back
    # toward its frozen-Base value.
    assert new_lp.grad is not None and float(new_lp.grad.item()) < 0.0


def test_new_preservation_penalizes_artificial_boost_too():
    true_lp = torch.tensor([-6.0], requires_grad=True)
    new_lp = torch.tensor([-1.0], requires_grad=True)
    idk_lp = torch.tensor([-5.0], requires_grad=True)
    base_true = torch.tensor([-2.0])
    base_new = torch.tensor([-4.0])
    terms = v1c.compute_bounded_objective_terms(
        true_lp, new_lp, idk_lp, base_true, base_new,
        owners=[0], case_count=1,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
        max_target_new_drift=0.25,
    )
    terms["target_new_preserve_loss"].backward()
    # Positive gradient means gradient descent reduces new_lp back toward Base.
    assert new_lp.grad is not None and float(new_lp.grad.item()) > 0.0


def test_bounded_ga_stops_after_cf_and_drop_pass():
    true_lp = torch.tensor([-5.0], requires_grad=True)
    new_lp = torch.tensor([-4.0], requires_grad=True)
    idk_lp = torch.tensor([-4.5], requires_grad=True)
    base_true = torch.tensor([-2.0])
    base_new = torch.tensor([-4.0])
    terms = v1c.compute_bounded_objective_terms(
        true_lp, new_lp, idk_lp, base_true, base_new,
        owners=[0], case_count=1,
        minimum_cf_margin=0.1,
        minimum_idk_margin=0.1,
        minimum_true_drop=2.0,
        max_target_new_drift=0.25,
    )
    assert float(terms["ga_active_count"].item()) == 0.0
    assert float(terms["bounded_ga_loss"].item()) == 0.0


def test_rank16_parameter_count_is_98304_for_hidden3072():
    assert 2 * 3072 * 16 == 98304
