#!/usr/bin/env python3
"""Materialization-safe wrapper for canonical SURE MCF Stage 2.

This wrapper leaves the shared Stage-2 implementation unchanged and only adds
robust MCF checkpoint selection for target-true-sensitive experiments:

* residual cases are still detected with the final acceptance margin;
* candidate optimization uses a larger solver margin (default 0.25, retry 0.5);
* scale selection requires a small in-memory safety buffer above acceptance;
* the saved checkpoint is reloaded from disk and re-evaluated on the same
  training-visible direct constraints;
* a checkpoint is accepted only when the reloaded model has zero direct
  failures at the final acceptance margin.

No paraphrases, neighborhoods, benchmark-retain examples, or PPL text are used
for selection.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch

import gagd_compare as gagd
from mcf_zero_unlearn_official_eval import is_llama_like
import sure_stage2_sparse_repair as shared


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


def solver_margin_ladder(initial: float, retry: float) -> List[float]:
    values: List[float] = []
    for value in (float(initial), float(retry)):
        if value not in values:
            values.append(value)
    return values


def choose_buffered_scale(
    reports: Sequence[Dict[str, Any]], guard_margin: float
) -> float:
    """Choose smallest zero-failure scale with a margin buffer when available."""
    guarded = [
        float(report["scale"])
        for report in reports
        if int(report.get("direct_failures", 1)) == 0
        and float(report.get("minimum_margin", float("-inf"))) >= float(guard_margin)
    ]
    if guarded:
        return min(guarded)

    zero = [
        report for report in reports if int(report.get("direct_failures", 1)) == 0
    ]
    if zero:
        # If no scale reaches the requested buffer, prefer the zero-failure scale
        # with the largest observed minimum margin rather than the weakest scale.
        best = max(
            zero,
            key=lambda report: (
                float(report.get("minimum_margin", float("-inf"))),
                float(report["scale"]),
            ),
        )
        return float(best["scale"])

    return shared.core.choose_scale(reports)


def _output_dir_from_args(stage_args: argparse.Namespace) -> Path:
    return Path(stage_args.output_dir).resolve()


def _reload_mcf_direct_check(
    stage_args: argparse.Namespace,
    checkpoint: Path,
) -> Dict[str, Any]:
    records, _manifest = shared.load_locked(
        "mcf",
        Path(stage_args.training_visible_path).resolve(),
        Path(stage_args.split_manifest).resolve(),
        int(stage_args.seed),
        int(stage_args.forget_num),
    )
    ns = argparse.Namespace(
        model_path=str(checkpoint),
        dtype=stage_args.dtype,
        device_map=stage_args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = gagd.first_device(model)
    llama_like = is_llama_like(model, tok)
    instances = shared.mcf_instances(records)
    margins = shared.mcf_direct_margins(
        model,
        tok,
        instances,
        device,
        llama_like,
        int(stage_args.batch_size),
        stage_args.mcf_sensitive_field,
        stage_args.mcf_reference_field,
    )
    failures = int((margins < float(stage_args.constraint_margin)).sum().item())
    result = {
        "direct_total": int(margins.numel()),
        "direct_failures": failures,
        "minimum_margin": (
            float(margins.min().detach().cpu()) if margins.numel() else None
        ),
        "acceptance_margin": float(stage_args.constraint_margin),
        "checkpoint": str(checkpoint),
    }
    del model
    del tok
    del margins
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _run_shared_once(
    forwarded_argv: Sequence[str],
    *,
    solver_margin: float,
    guard_margin: float,
) -> argparse.Namespace:
    old_argv = sys.argv[:]
    old_optimize = shared.optimize_mcf_candidate
    old_choose_scale = shared.core.choose_scale

    def optimize_with_solver_margin(**kwargs):
        kwargs["required_margin"] = float(solver_margin)
        return old_optimize(**kwargs)

    def choose_scale_with_buffer(reports):
        if reports and "minimum_margin" in reports[0]:
            return choose_buffered_scale(reports, float(guard_margin))
        return old_choose_scale(reports)

    try:
        sys.argv = [str(Path(__file__).resolve()), *forwarded_argv]
        stage_args = shared.parse_args()
        if stage_args.dataset != "mcf":
            raise RuntimeError(
                "materialization-safe wrapper is intentionally restricted to MCF"
            )
        if float(solver_margin) < float(stage_args.constraint_margin):
            raise ValueError(
                "solver margin must be >= final acceptance constraint margin"
            )
        shared.optimize_mcf_candidate = optimize_with_solver_margin
        shared.core.choose_scale = choose_scale_with_buffer
        shared.main()
        return stage_args
    finally:
        shared.optimize_mcf_candidate = old_optimize
        shared.core.choose_scale = old_choose_scale
        sys.argv = old_argv


def main(argv: Sequence[str] | None = None) -> None:
    wrapper, forwarded = _split_wrapper_args(sys.argv[1:] if argv is None else argv)
    attempts: List[Dict[str, Any]] = []
    accepted = False
    final_stage_args: argparse.Namespace | None = None

    for solver_margin in solver_margin_ladder(
        wrapper.solver_margin, wrapper.solver_retry_margin
    ):
        print(
            f"Materialization-safe MCF Stage2: solver margin={solver_margin:g}, "
            f"final acceptance={_acceptance_margin_from_forwarded(forwarded):g}, "
            f"scale guard={wrapper.materialization_guard_margin:g}",
            flush=True,
        )
        stage_args = _run_shared_once(
            forwarded,
            solver_margin=solver_margin,
            guard_margin=wrapper.materialization_guard_margin,
        )
        final_stage_args = stage_args
        output_dir = _output_dir_from_args(stage_args)
        checkpoint = output_dir / "checkpoint"
        reload_check = _reload_mcf_direct_check(stage_args, checkpoint)
        attempt = {
            "solver_margin": float(solver_margin),
            "materialization_guard_margin": float(
                wrapper.materialization_guard_margin
            ),
            "reload_check": reload_check,
        }
        attempts.append(attempt)
        print("Reloaded checkpoint direct check:", reload_check, flush=True)
        if int(reload_check["direct_failures"]) == 0:
            accepted = True
            break

    if final_stage_args is None:
        raise RuntimeError("Stage 2 did not execute")

    output_dir = _output_dir_from_args(final_stage_args)
    summary_path = output_dir / "repair_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "solver_margin_ladder": solver_margin_ladder(
                wrapper.solver_margin, wrapper.solver_retry_margin
            ),
            "accepted_solver_margin": (
                float(attempts[-1]["solver_margin"]) if accepted else None
            ),
            "acceptance_margin": float(final_stage_args.constraint_margin),
            "materialization_guard_margin": float(
                wrapper.materialization_guard_margin
            ),
            "materialized_reload_attempts": attempts,
            "materialized_reload_failures": int(
                attempts[-1]["reload_check"]["direct_failures"]
            ),
            "materialized_reload_minimum_margin": attempts[-1]["reload_check"][
                "minimum_margin"
            ],
            "materialized_checkpoint_accepted": bool(accepted),
            "materialized_checkpoint_gate": (
                "reload checkpoint and require zero direct failures at acceptance margin"
            ),
        }
    )
    shared.core.write_json(summary_path, summary)
    shared.core.write_json(
        output_dir / "materialized_checkpoint_attempts.json",
        {
            "schema_version": 1,
            "attempts": attempts,
            "accepted": bool(accepted),
        },
    )

    if not accepted:
        raise RuntimeError(
            "No materialized BF16 checkpoint achieved zero direct failures at "
            f"the final margin {final_stage_args.constraint_margin}. "
            "The checkpoint is rejected; do not proceed to official evaluation."
        )

    print(
        "Materialized checkpoint ACCEPTED: zero direct failures after save/reload.",
        flush=True,
    )


def _acceptance_margin_from_forwarded(argv: Sequence[str]) -> float:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--constraint-margin", type=float, required=True)
    parsed, _unknown = parser.parse_known_args(list(argv))
    return float(parsed.constraint_margin)


if __name__ == "__main__":
    main()
