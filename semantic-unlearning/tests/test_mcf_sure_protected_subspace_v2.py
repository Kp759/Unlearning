import torch

from scripts import mcf_sure_protected_subspace_stage1 as stage1
from scripts import mcf_sure_protected_subspace_stage2 as stage2


def test_project_away_removes_protected_components():
    basis = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rows = torch.tensor([[3.0, -2.0, 5.0], [1.0, 7.0, -4.0]])
    residual = stage1.project_away(rows, basis)
    assert torch.allclose(residual[:, :2], torch.zeros(2, 2), atol=1e-6)
    assert torch.allclose(residual[:, 2], rows[:, 2], atol=1e-6)
    assert torch.allclose(residual @ basis.T, torch.zeros(2, 2), atol=1e-6)


def test_sensitive_residual_basis_is_orthogonal_to_context_basis():
    h_context = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    h_sensitive = torch.tensor(
        [
            [4.0, 2.0, 3.0, 0.0],
            [1.0, 5.0, 0.0, 2.0],
        ]
    )
    b_ns, residual, b_s, report = stage1.build_sensitive_residual_basis(
        h_sensitive,
        h_context,
        protected_rank=2,
        sensitive_rank=2,
    )
    assert b_ns.shape[0] == 2
    assert b_s.shape[0] == 2
    assert torch.allclose(residual @ b_ns.T, torch.zeros(2, 2), atol=1e-5)
    assert torch.max(torch.abs(b_s @ b_ns.T)).item() < 1e-5
    assert report["protected_rank_actual"] == 2
    assert report["sensitive_rank_actual"] == 2


def test_atomic_margin_is_max_other_minus_sensitive():
    logits = torch.tensor(
        [
            [1.0, 5.0, 3.0],
            [6.0, 4.0, 2.0],
        ]
    )
    target_ids = torch.tensor([1, 2])
    margins = stage1.atomic_margins(logits, target_ids)
    assert torch.allclose(margins, torch.tensor([-2.0, 4.0]))


def test_sparse_delta_changes_only_selected_lm_columns():
    base = torch.zeros(2, 5)
    hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    selected = [1, 3]
    delta = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    updated = stage2.logits_with_sparse_delta(base, hidden, selected, delta)

    assert torch.allclose(updated[:, 0], base[:, 0])
    assert torch.allclose(updated[:, 2], base[:, 2])
    assert torch.allclose(updated[:, 4], base[:, 4])
    assert torch.allclose(updated[:, 1], torch.tensor([1.0, 3.0]))
    assert torch.allclose(updated[:, 3], torch.tensor([4.0, 8.0]))


def test_hard_protection_detects_regression():
    # Baseline target token 0 is safely below token 2: margin = 2.0.
    base_logits = torch.tensor([[0.0, 1.0, 2.0]])
    hidden = torch.tensor([[1.0, 0.0]])
    target_ids = torch.tensor([0])

    # Editing selected target row 0 by +3 makes target logit 3, so margin = -1.
    metrics = stage2.hard_protection_metrics(
        base_logits=base_logits,
        hidden=hidden,
        target_ids=target_ids,
        protected_positions=[0],
        selected_ids=[0],
        delta_rows=torch.tensor([[3.0, 0.0]]),
        atomic_margin=0.05,
    )
    assert metrics["protected_regressions"] == 1
    assert metrics["protected_min_margin"] < 0.05
    assert metrics["protected_kl"] > 0.0


def test_parameter_backtracking_interpolation():
    module = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.zero_()
    old = [p.detach().clone() for p in module.parameters()]
    proposed = [torch.full_like(old[0], 2.0)]
    stage2.set_interpolated_parameters(module, old, proposed, 0.25)
    assert torch.allclose(module.weight, torch.full_like(module.weight, 0.5))
