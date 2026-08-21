from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MCF_SCRIPTS = SCRIPTS / "MCF_Scripts"
for path in (SCRIPTS, MCF_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


materialized = load_module(
    "sure_stage2_sparse_repair_materialized_test",
    SCRIPTS / "sure_stage2_sparse_repair_materialized.py",
)
runner = load_module(
    "run_mcf_zsre_arch_target_true_materialized_test",
    MCF_SCRIPTS / "run_mcf_zsre_arch_target_true_materialized.py",
)
base_runner = sys.modules["run_mcf_zsre_arch_target_true"]


def test_solver_margin_ladder():
    assert materialized.solver_margin_ladder(0.25, 0.5) == [0.25, 0.5]
    assert materialized.solver_margin_ladder(0.5, 0.5) == [0.5]


def test_buffered_scale_prefers_smallest_guarded_zero_failure_scale():
    reports = [
        {"scale": 1.0, "direct_failures": 0, "minimum_margin": 0.30},
        {"scale": 0.875, "direct_failures": 0, "minimum_margin": 0.12},
        {"scale": 0.75, "direct_failures": 0, "minimum_margin": 0.06},
        {"scale": 0.625, "direct_failures": 1, "minimum_margin": -0.10},
    ]
    assert materialized.choose_buffered_scale(reports, 0.10) == 0.875


def test_buffered_scale_falls_back_to_strongest_zero_failure_margin():
    reports = [
        {"scale": 1.0, "direct_failures": 0, "minimum_margin": 0.08},
        {"scale": 0.875, "direct_failures": 0, "minimum_margin": 0.04},
        {"scale": 0.75, "direct_failures": 1, "minimum_margin": -0.02},
    ]
    assert materialized.choose_buffered_scale(reports, 0.10) == 1.0


def test_top_level_plan_routes_stage2_through_materialized_gate():
    wrapper, forwarded = runner._split_wrapper_args(
        [
            "--model-path",
            "/model",
            "--mcf-path",
            "/mcf.json",
            "--wikidata-dir",
            "/wiki",
            "--output-root",
            "/out",
            "--seeds",
            "1",
            "--solver-margin",
            "0.25",
            "--solver-retry-margin",
            "0.5",
            "--materialization-guard-margin",
            "0.10",
            "--dry-run",
        ]
    )
    args = base_runner.parse_args(forwarded)
    paths = base_runner.seed_paths(Path("/out"), 1)
    steps = runner._patched_plan(
        base_runner.seed_command_plan, args, paths, 1, wrapper
    )
    stage2 = steps[1]
    command = " ".join(stage2.command)
    assert "sure_stage2_sparse_repair_materialized.py" in command
    assert "--mcf-sensitive-field target_true" in command
    assert "--mcf-reference-field target_new" in command
    assert "--constraint-margin 0.05" in command
    assert "--solver-margin 0.25" in command
    assert "--solver-retry-margin 0.5" in command
    assert "--materialization-guard-margin 0.1" in command


def test_effective_architecture_records_two_margin_contract():
    wrapper, forwarded = runner._split_wrapper_args(
        [
            "--model-path",
            "/model",
            "--solver-margin",
            "0.25",
            "--solver-retry-margin",
            "0.5",
            "--materialization-guard-margin",
            "0.10",
            "--dry-run",
        ]
    )
    args = base_runner.parse_args(forwarded)
    architecture = runner._patched_architecture(
        base_runner.effective_architecture, args, wrapper
    )
    stage2 = architecture["stage2"]
    assert stage2["solver_margin"] == 0.25
    assert stage2["solver_retry_margin"] == 0.5
    assert stage2["final_acceptance_margin"] == 0.05
    assert stage2["materialization_guard_margin"] == 0.10
