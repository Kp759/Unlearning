#!/usr/bin/env python3
"""Run MCF sparse LM-head repair with zero benchmark-retain access.

The generic repair implementation supports optional retain KL/projection and
historically validates retain_num > 0.  This compatibility wrapper permits
retain_num == 0 only when all retain-based repair mechanisms are disabled.
The underlying repair code then samples zero retain records and optimizes only
on the repair-visible forget rewrite prompts.
"""

from __future__ import annotations

import math

import gagd_active_case_repair as repair


def validate_forget_only(args) -> None:
    if args.forget_num <= 0:
        raise ValueError("--forget-num must be positive")
    if args.retain_num != 0:
        raise ValueError(
            "Forget-only repair requires --retain-num 0; benchmark retain data is evaluation-only"
        )
    if args.retain_kl_mu != 0:
        raise ValueError("Forget-only repair requires --retain-kl-mu 0")
    if args.project_away_retain_hidden:
        raise ValueError("Forget-only repair forbids --project-away-retain-hidden")
    if args.retain_calibration_num != 0:
        raise ValueError("Forget-only repair requires --retain-calibration-num 0")
    if args.reference_model_path is not None:
        raise ValueError("Forget-only repair does not use --reference-model-path")
    if not math.isfinite(args.active_margin) or args.active_margin < 0:
        raise ValueError("--active-margin must be finite and non-negative")
    if not math.isfinite(args.protected_margin_floor):
        raise ValueError("--protected-margin-floor must be finite")
    if args.repair_steps <= 0 or args.repair_lr <= 0:
        raise ValueError("--repair-steps and --repair-lr must be positive")
    if args.hinge_weight <= 0 or args.delta_l2_lambda < 0:
        raise ValueError("invalid repair regularization weights")
    if args.repair_rank < 0 or args.margin_batch_size <= 0:
        raise ValueError("repair rank must be non-negative and batch size positive")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("--max-delta-norm must be finite and non-negative")
    if args.run_official_mcf_eval:
        raise ValueError(
            "Do not run official evaluation inside forget-only repair; evaluate the frozen checkpoint separately"
        )


# main() resolves this global at runtime, so replacing the validator leaves the
# core repair math/checkpointing implementation unchanged.
repair.validate_args = validate_forget_only


if __name__ == "__main__":
    repair.main()
