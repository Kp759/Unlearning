#!/usr/bin/env python3
"""Run the W1K target-true-sensitive MCF ablation at Stage-1 LR 1e-4.

This preserves the existing W1K LR4e-5 runner as a separate ablation and changes
only the locked Stage-1 learning rate back to the original canonical 1e-4.
Everything else remains identical: 50 forget records, GA=2, same-prompt KL=1,
W1K external Wikipedia KL=1, 600 Stage-1 steps, and materialization-safe Stage 2.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS_DIR = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_mcf_zsre_arch_target_true_w1k as w1k  # noqa: E402


# Keep the LR4e-5 runner immutable as its own completed ablation.  This thin
# wrapper changes only the module-level locked LR/provenance used by w1k.main().
w1k.LOCKED_STAGE1_LR = 1e-4
w1k.METHOD = "SURE-MCF-ZsRE-architecture-target_true-sensitive-W1K-LR1e-4"
w1k.PROTOCOL = "mcf_zsre_arch_target_true_sensitive_w1k_lr1e4_v1"


if __name__ == "__main__":
    w1k.main()
