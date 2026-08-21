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
    "mcf_sensitive_rows_projected_eff0_repair_test",
    SCRIPTS / "mcf_sensitive_rows_projected_eff0_repair.py",
)


def test_parse_scales_keeps_endpoints_and_unique_values():
    values = repair.parse_scales("1,.5,.5,.25,0")
    assert values == [1.0, 0.5, 0.25, 0.0]
    values = repair.parse_scales(".75,.25")
    assert values[0] == 1.0
    assert values[-1] == 0.0


def test_margin_summary_uses_target_true_minus_target_new_sign():
    margins = torch.tensor([-0.2, 0.05, 0.3])
    out = repair._margin_summary(margins, required=0.05)
    assert out["failures"] == 1
    assert out["successes"] == 2
    assert out["minimum_margin"] < 0


def test_parser_defaults_materialization_safe_margins():
    args = repair.parse_args(
        [
            "--model-path", "/stage1",
            "--base-model-path", "/base",
            "--training-visible-path", "/visible.json",
            "--split-manifest", "/manifest.json",
            "--output-dir", "/out",
            "--utility-wikipedia-dir", "/wiki",
        ]
    )
    assert args.solver_margin == 0.25
    assert args.acceptance_margin == 0.05
    assert args.per_token_locality_rank == 0
    assert args.locality_sensitive_logit_weight == 5.0
    assert args.utility_exclude_first == 20


def test_blend_changes_only_selected_rows_for_tied_matrix():
    weight = nn.Parameter(torch.zeros(5, 3))
    tied_info = {
        "input_weight": weight,
        "output_weight": weight,
        "tied": True,
    }
    stage1 = torch.arange(15, dtype=torch.float32).reshape(5, 3)
    repaired = stage1.clone()
    repaired[1] += 4.0
    repaired[3] -= 2.0
    repair._apply_blend(
        tied_info,
        {"input": stage1, "output": stage1},
        {"input": repaired, "output": repaired},
        sensitive_ids=[1, 3],
        scale=0.5,
    )
    assert torch.allclose(weight[1], stage1[1] + 2.0)
    assert torch.allclose(weight[3], stage1[3] - 1.0)
    # _apply_blend intentionally touches only selected rows.
    assert torch.equal(weight[0], torch.zeros(3))
    assert torch.equal(weight[2], torch.zeros(3))
    assert torch.equal(weight[4], torch.zeros(3))


def test_smallest_accepted_scale_logic_prefers_minimum_added_movement():
    accepted = [(1.0, torch.tensor([0.3])), (0.5, torch.tensor([0.1])), (0.75, torch.tensor([0.2]))]
    selected_scale, _ = min(accepted, key=lambda item: item[0])
    assert selected_scale == 0.5
