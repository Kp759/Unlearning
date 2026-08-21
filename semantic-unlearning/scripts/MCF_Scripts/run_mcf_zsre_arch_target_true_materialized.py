#!/usr/bin/env python3
"""Run target-true-sensitive MCF with materialization-safe ZsRE Stage 2.

This is a narrow robustness wrapper around ``run_mcf_zsre_arch_target_true``.
Stage 1, data splits, official evaluation, and paper metrics are unchanged.
Only Stage 2 is replaced with the materialization-safe wrapper that:

* optimizes residual repair against a larger solver margin;
* keeps the paper/direct acceptance margin unchanged;
* uses a buffered scale choice;
* reloads the saved checkpoint and requires zero direct failures before the
  official evaluation is allowed to run.
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence


HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_mcf_zsre_arch_target_true as base  # noqa: E402


def _split_wrapper_args(argv: Sequence[str]) -> tuple[argparse.Namespace, List[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--solver-margin", type=float, default=0.25)
    parser.add_argument("--solver-retry-margin", type=float, default=0.5)
    parser.add_argument("--materialization-guard-margin", type=float, default=0.10)
    wrapper, remaining = parser.parse_known_args(list(argv))
    if wrapper.solver_margin < 0 or wrapper.solver_retry_margin < 0:
        parser.error("solver margins must be non-negative")
    if wrapper.materialization_guard_margin < 0:
        parser.error("materialization guard margin must be non-negative")
    return wrapper, remaining


def _patched_plan(
    original_plan,
    args: argparse.Namespace,
    paths: base.SeedPaths,
    seed: int,
    wrapper: argparse.Namespace,
) -> List[base.Step]:
    steps = original_plan(args, paths, seed)
    if len(steps) < 2:
        raise RuntimeError("canonical plan does not contain Stage 2")
    stage2 = steps[1]
    command = list(stage2.command)
    command[1] = str(SCRIPTS_DIR / "sure_stage2_sparse_repair_materialized.py")
    command.extend(
        [
            "--solver-margin",
            str(wrapper.solver_margin),
            "--solver-retry-margin",
            str(wrapper.solver_retry_margin),
            "--materialization-guard-margin",
            str(wrapper.materialization_guard_margin),
        ]
    )
    steps[1] = base.Step(
        "STAGE 2 — MATERIALIZATION-SAFE ZSRE SPARSE ROWS + MCF PAIRWISE CONSTRAINT",
        command,
    )
    return steps


def _patched_architecture(
    original_effective_architecture,
    args: argparse.Namespace,
    wrapper: argparse.Namespace,
) -> Dict[str, Any]:
    architecture = copy.deepcopy(original_effective_architecture(args))
    architecture["stage2"].update(
        {
            "solver_margin": float(wrapper.solver_margin),
            "solver_retry_margin": float(wrapper.solver_retry_margin),
            "final_acceptance_margin": float(args.constraint_margin),
            "materialization_guard_margin": float(
                wrapper.materialization_guard_margin
            ),
            "materialized_checkpoint_gate": (
                "save, reload, and require zero direct failures before evaluation"
            ),
        }
    )
    return architecture


def _patched_validate(original_validate, paths: base.SeedPaths) -> Dict[str, Any]:
    result = original_validate(paths)
    import json

    summary = json.loads(
        (paths.stage2_dir / "repair_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("materialized_checkpoint_accepted") is not True:
        raise RuntimeError("Stage 2 materialized checkpoint was not accepted")
    if int(summary.get("materialized_reload_failures", -1)) != 0:
        raise RuntimeError("Reloaded Stage 2 checkpoint has residual direct failures")
    result.update(
        {
            "solver_margin": summary.get("accepted_solver_margin"),
            "acceptance_margin": summary.get("acceptance_margin"),
            "materialization_guard_margin": summary.get(
                "materialization_guard_margin"
            ),
            "materialized_reload_failures": summary.get(
                "materialized_reload_failures"
            ),
            "materialized_reload_minimum_margin": summary.get(
                "materialized_reload_minimum_margin"
            ),
            "materialized_checkpoint_accepted": True,
        }
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    wrapper, forwarded = _split_wrapper_args(sys.argv[1:] if argv is None else argv)
    args = base.parse_args(forwarded)
    if float(wrapper.solver_margin) < float(args.constraint_margin):
        raise ValueError("--solver-margin must be >= --constraint-margin")
    if float(wrapper.solver_retry_margin) < float(args.constraint_margin):
        raise ValueError("--solver-retry-margin must be >= --constraint-margin")

    # Keep these in the run manifest's effective_arguments as well.
    args.solver_margin = float(wrapper.solver_margin)
    args.solver_retry_margin = float(wrapper.solver_retry_margin)
    args.materialization_guard_margin = float(wrapper.materialization_guard_margin)

    original_parse_args = base.parse_args
    original_plan = base.seed_command_plan
    original_architecture = base.effective_architecture
    original_validate = base.validate_stage_contract

    try:
        base.parse_args = lambda _argv=None: args
        base.seed_command_plan = lambda parsed, paths, seed: _patched_plan(
            original_plan, parsed, paths, seed, wrapper
        )
        base.effective_architecture = lambda parsed: _patched_architecture(
            original_architecture, parsed, wrapper
        )
        base.validate_stage_contract = lambda paths: _patched_validate(
            original_validate, paths
        )
        base.main([])
    finally:
        base.parse_args = original_parse_args
        base.seed_command_plan = original_plan
        base.effective_architecture = original_architecture
        base.validate_stage_contract = original_validate


if __name__ == "__main__":
    main()
