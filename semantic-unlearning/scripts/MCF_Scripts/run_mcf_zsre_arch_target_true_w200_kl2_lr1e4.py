#!/usr/bin/env python3
"""Run target-true-sensitive MCF with W200 utility KL=2 at Stage-1 LR 1e-4.

This is a thin, isolated ablation over the existing W1K runner. It keeps the
forget protocol and optimizer unchanged while changing only:

* Stage-1 learning rate: 1e-4;
* external Wikipedia utility sample size: 200 contexts;
* external Wikipedia KL weight: 2.0.

The first 20 external-Wikipedia rows remain excluded so the fixed Wikipedia PPL
probe is not training-visible. Stage 2 remains the same materialization-safe
repair used by the existing target-true runners.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Sequence

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_mcf_zsre_arch_target_true_w1k as utility_runner  # noqa: E402


LOCKED_UTILITY_KL_WEIGHT = 2.0

# Preserve all completed W1K ablations. These module constants are consumed at
# runtime by the shared runner's parser, architecture record, and validator.
utility_runner.LOCKED_STAGE1_LR = 1e-4
utility_runner.LOCKED_UTILITY_SAMPLE_SIZE = 200
utility_runner.METHOD = (
    "SURE-MCF-ZsRE-architecture-target_true-sensitive-W200-KL2-LR1e-4"
)
utility_runner.PROTOCOL = (
    "mcf_zsre_arch_target_true_sensitive_w200_kl2_lr1e4_v1"
)


def _lock_utility_kl(argv: Sequence[str]) -> List[str]:
    """Default KL to 2.0 and reject attempts to change this ablation knob."""
    raw = list(argv)
    seen = None
    for i, arg in enumerate(raw):
        if arg == "--utility-kl-weight":
            if i + 1 >= len(raw):
                raise SystemExit("--utility-kl-weight requires a value")
            seen = float(raw[i + 1])
            break
        if arg.startswith("--utility-kl-weight="):
            seen = float(arg.split("=", 1)[1])
            break

    if seen is None:
        raw.extend(["--utility-kl-weight", str(LOCKED_UTILITY_KL_WEIGHT)])
    elif not math.isclose(seen, LOCKED_UTILITY_KL_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            "this ablation locks --utility-kl-weight to "
            f"{LOCKED_UTILITY_KL_WEIGHT}"
        )
    return raw


def main(argv: Sequence[str] | None = None) -> None:
    raw = sys.argv[1:] if argv is None else list(argv)
    utility_runner.main(_lock_utility_kl(raw))


if __name__ == "__main__":
    main()
