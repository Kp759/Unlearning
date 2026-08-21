from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repair = load_module(
    "mcf_sensitive_rows_projected_gagd_test",
    SCRIPTS / "mcf_sensitive_rows_projected_gagd.py",
)


def test_token_locality_bases_are_orthonormal_and_token_specific():
    hidden = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    protected = [[5], [5], [6]]
    bases, receipt = repair.build_token_locality_bases(hidden, protected, rank_cap=0)
    assert set(bases) == {5, 6}
    assert bases[5].shape == (2, 4)
    assert bases[6].shape == (1, 4)
    assert torch.allclose(bases[5] @ bases[5].T, torch.eye(2), atol=1e-6)
    assert receipt["5"]["kept_rank"] == 2
    assert receipt["6"]["kept_rank"] == 1


def test_projection_removes_sensitive_row_component_in_locality_basis():
    weight = nn.Parameter(torch.zeros(4, 3))
    tied_info = {
        "input_weight": weight,
        "output_weight": weight,
        "tied": True,
    }
    base = torch.zeros_like(weight.detach())
    base_rows = {"input": base.clone(), "output": base.clone()}
    with torch.no_grad():
        weight[1].copy_(torch.tensor([2.0, 3.0, 4.0]))
        weight[2].copy_(torch.tensor([7.0, 8.0, 9.0]))
    # Token 1 must not move in e1/e2 directions; e3 remains free.
    bases = {1: torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])}
    repair.project_selected_rows_to_locality_nullspace(
        tied_info, base_rows, bases, selected_ids=[1]
    )
    assert torch.allclose(weight[1].detach(), torch.tensor([0.0, 0.0, 4.0]))
    # Non-selected row is untouched by the projector itself.
    assert torch.allclose(weight[2].detach(), torch.tensor([7.0, 8.0, 9.0]))
    diag = repair.projection_residual_diagnostics(
        tied_info, base_rows, bases, selected_ids=[1]
    )
    assert diag["max_abs_delta_dot_locality_basis"] < 1e-7


def test_protected_sensitive_logit_mse_only_checks_declared_rows():
    current = torch.tensor([[1.0, 8.0, 3.0], [4.0, 5.0, 9.0]], requires_grad=True)
    base = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # row0 protects id1: error 6 -> 36; row1 protects id2: error 3 -> 9; mean=22.5
    loss = repair.protected_sensitive_logit_mse(current, base, [[1], [2]])
    assert torch.allclose(loss, torch.tensor(22.5))


def test_locality_kl_zero_when_base_distribution_matches():
    z = torch.tensor([[1.0, 2.0, -1.0]], requires_grad=True)
    loss = repair.locality_kl(z, z.detach().clone())
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-7)


def test_selected_row_delta_mse_uses_only_selected_rows():
    weight = nn.Parameter(torch.zeros(4, 2))
    tied_info = {"input_weight": weight, "output_weight": weight, "tied": True}
    base = torch.zeros_like(weight.detach())
    base_rows = {"input": base.clone(), "output": base.clone()}
    with torch.no_grad():
        weight[1].fill_(2.0)
        weight[3].fill_(100.0)  # not selected and must not affect penalty
    loss = repair.selected_row_delta_mse(tied_info, base_rows, [1])
    assert torch.allclose(loss, torch.tensor(4.0))


def test_parser_defaults_use_sensitive_row_gagd_not_adapter_or_fullblock():
    args = repair.parse_args(
        [
            "--model-path", "/base",
            "--training-visible-path", "/visible.json",
            "--split-manifest", "/manifest.json",
            "--output-dir", "/out",
            "--utility-wikipedia-dir", "/wiki",
        ]
    )
    assert args.emb_lm_lr == 1e-4
    assert args.ga_weight == 2.0
    assert args.gd_weight == 1.0
    assert args.subject_control_count == 4
    assert args.locality_kl_weight == 2.0
    assert args.locality_sensitive_logit_weight == 5.0
    assert args.per_token_locality_rank == 0
    assert args.utility_sample_size == 200
    assert args.utility_kl_weight == 2.0
    assert not hasattr(args, "adapter_rank")
    assert not hasattr(args, "lr")
