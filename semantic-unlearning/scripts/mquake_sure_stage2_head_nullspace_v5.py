#!/usr/bin/env python3
"""MQuAKE SURE v5 Stage 2B: stable direct-only P-nullspace LM-head repair.

This is the v3 direct-only Stage-2 optimizer applied after v5 Stage2A embedding
GD, with the numerically stable float64 protected/repair basis construction from
v4.1.  The architecture is otherwise unchanged:

    recompute H after Stage2A embedding GD
    P/F from the direct training-visible gate
    B_P = rowspace(H_P)
    B_F = rowspace(H_F - Proj_P(H_F))
    Delta W_A_F = C_F B_F

Embeddings and transformer are frozen during this phase.  No AtomicGen, retain,
target_new, or held-out prompt is used.
"""
from __future__ import annotations

import sure_canonical_core as core
import mquake_sure_stage2_head_nullspace_v3 as direct_v3
import mquake_sure_stage2_robust_nullspace_v41 as stable


def main() -> None:
    # Reset wrapper state so repeated process invocations are deterministic.
    stable._CALL_INDEX = 0
    stable._PROTECTED_BASIS64 = None
    core.orthonormal_row_basis = stable.stable_orthonormal_row_basis
    direct_v3.main()


if __name__ == "__main__":
    main()
