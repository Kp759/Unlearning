from __future__ import annotations

import importlib.util
import random
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


minimal = load_module(
    "mcf_sensitive_rows_projected_eff0_minimal_test",
    SCRIPTS / "mcf_sensitive_rows_projected_eff0_minimal.py",
)


def tied_info(weight: nn.Parameter):
    return {
        "input_weight": weight,
        "output_weight": weight,
        "tied": True,
    }


def test_parser_defaults_are_minimal_and_stage1_anchored():
    a = minimal.parse_args(
        [
            "--model-path", "/stage1",
            "--base-model-path", "/base",
            "--training-visible-path", "/visible.json",
            "--split-manifest", "/manifest.json",
            "--output-dir", "/out",
            "--utility-wikipedia-dir", "/wiki",
        ]
    )
    assert a.solver_margin == 0.10
    assert a.acceptance_margin == 0.05
    assert a.lr == 2e-4
    assert a.check_every == 10
    assert a.stage1_anchor_weight == 50.0
    assert a.per_token_locality_rank == 0
    assert not hasattr(a, "row_delta_weight")


def test_stage1_anchor_mse_zero_at_parent_and_positive_after_repair():
    w = nn.Parameter(torch.zeros(5, 3))
    with torch.no_grad():
        w[1] = torch.tensor([1.0, 2.0, 3.0])
        w[3] = torch.tensor([4.0, 5.0, 6.0])
    info = tied_info(w)
    ref = minimal._snapshot_selected_rows(info, [1, 3])
    zero = minimal.stage1_anchor_mse(info, ref, [1, 3])
    assert torch.allclose(zero, torch.tensor(0.0))

    with torch.no_grad():
        w[1, 0] += 1.0
    positive = minimal.stage1_anchor_mse(info, ref, [1, 3])
    assert positive.item() > 0


def test_projection_removes_only_repair_component_along_locality_basis():
    w = nn.Parameter(torch.zeros(4, 3))
    # Stage1 row itself is not in the locality nullspace; the new repair projection
    # must leave it untouched and remove only the repair delta component.
    stage1 = {
        "input": torch.tensor([[1.0, 0.0, 0.0]]),
        "output": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    with torch.no_grad():
        w[1] = torch.tensor([2.0, 2.0, 0.0])
    info = tied_info(w)
    basis = {1: torch.tensor([[1.0, 0.0, 0.0]])}

    report = minimal.project_repair_delta_to_locality_nullspace(
        info, stage1, basis, [1]
    )
    # repair delta [1,2,0] -> [0,2,0], so final row is Stage1 + [0,2,0]
    assert torch.allclose(w[1].detach(), torch.tensor([1.0, 2.0, 0.0]))
    assert report["max_removed_component_norm"] > 0

    residual = minimal.repair_projection_residual(info, stage1, basis, [1])
    assert residual["max_abs_repair_delta_dot_locality_basis"] < 1e-7


def test_selected_blend_is_anchored_to_stage1():
    w = nn.Parameter(torch.zeros(4, 2))
    info = tied_info(w)
    stage1 = {
        "input": torch.tensor([[2.0, 4.0]]),
        "output": torch.tensor([[2.0, 4.0]]),
    }
    repaired = {
        "input": torch.tensor([[6.0, 8.0]]),
        "output": torch.tensor([[6.0, 8.0]]),
    }

    minimal._apply_selected_blend(info, stage1, repaired, [2], 0.0)
    assert torch.allclose(w[2].detach(), torch.tensor([2.0, 4.0]))

    minimal._apply_selected_blend(info, stage1, repaired, [2], 0.5)
    assert torch.allclose(w[2].detach(), torch.tensor([4.0, 6.0]))

    minimal._apply_selected_blend(info, stage1, repaired, [2], 1.0)
    assert torch.allclose(w[2].detach(), torch.tensor([6.0, 8.0]))


def test_active_batch_never_samples_passing_positions():
    rng = random.Random(7)
    active = [1, 4, 8, 9]
    for _ in range(20):
        batch = minimal._active_batch(active, 2, rng)
        assert len(batch) == 2
        assert set(batch).issubset(set(active))
    assert minimal._active_batch([3, 5], 4, rng) == [3, 5]
    assert minimal._active_batch([], 4, rng) == []
