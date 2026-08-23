#!/usr/bin/env python3
"""Deadlock-safe launcher for row-specific minimal MCF Stage 2.

The v4 optimizer uses hard protection and interpolates each Adam proposal along a
fixed backtracking schedule.  If every listed alpha is rejected and optimizer
state is restored, the next iteration recreates the same proposal exactly and
can deadlock forever.  This launcher makes the feasibility search effectively
continuous by extending geometric backtracking down to 2^-24.  Because the
current iterate is already strictly feasible on protected records, continuity
means a sufficiently small positive step must remain feasible (up to numerical
precision), so optimization can keep moving instead of repeating alpha=0.

No objective, data, basis, rank, margin, or final exact-bf16 line-search
semantics are changed.
"""
from __future__ import annotations

import mcf_sure_rowspecific_minimal_stage2 as impl

# 1, 1/2, ..., 2^-24.  This is algorithmic feasibility backtracking, not a
# hyperparameter sweep.  It fixes the deterministic rollback deadlock in v4.
impl.BACKTRACK_SCALES = tuple([1.0] + [2.0 ** (-k) for k in range(1, 25)])


if __name__ == "__main__":
    impl.main()
