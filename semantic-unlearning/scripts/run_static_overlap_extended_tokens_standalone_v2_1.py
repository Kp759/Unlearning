#!/usr/bin/env python3
"""Run v2.1 with phase-lexicographic private-row proposal gradients.

Unlocked rows optimize only the current worst forgotten-answer view.  Once all
authored training views for a row are below the target, the row optimizes mean
abstention NLL subject to the answer threshold remaining satisfied.
"""
from __future__ import annotations

import run_static_overlap_extended_tokens_standalone_v2 as v2


METHOD = "static_overlap_extended_tokens_standalone_v2_1"
PLAN = {
    **v2.PLAN,
    "backtracks": 12,
    "proposal_objective": "phase_lexicographic",
    "log_phase": "extended_token_v2_1",
    "registered_architecture": "input_only_extended_association_tokens_row_wise_v2_1",
    "optimization_description": (
        "one Adam optimizer per fact row; pure worst-view answer suppression until "
        "the row is locked, then mean abstention NLL under the hard answer constraint"
    ),
    "completion_status": "standalone_extended_token_v2_1_oracle_ablation_complete",
}


def main(argv=None):
    v2.METHOD = METHOD
    v2.PLAN = PLAN
    return v2.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
