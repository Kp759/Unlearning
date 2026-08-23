#!/usr/bin/env python3
"""Tie-exact entry point for the SURE-v3 covariance/locality Stage 2.

This wrapper fixes the selected-vs-fixed logit decomposition used by
``mquake_sure_stage2_cov_locality_v3``.  Full-vocabulary ``torch.argmax`` breaks
an exact logit tie by the lower vocabulary index.  The original decomposition
used ``selected_max >= fixed_max`` and therefore always chose the selected side
on ties, which could disagree with the Stage-1 baseline even at zero delta.

The training objective, covariance geometry, relation-locality controls, data,
and all hyperparameters are otherwise unchanged.
"""
from __future__ import annotations

import torch

import mquake_sure_stage2_cov_locality_v3 as base


def tie_exact_report(self, delta_rows: torch.Tensor):
    """Exactly reproduce full-vocabulary argmax for selected/fixed decomposition."""
    with torch.no_grad():
        selected = self.base_selected + self.hidden @ delta_rows.float().transpose(0, 1)
        sel_max, sel_pos = selected.max(dim=-1)
        sel_id = self.row_ids.index_select(0, sel_pos)

        selected_strictly_better = sel_max > self.fixed_max
        exact_tie = sel_max == self.fixed_max
        # torch.argmax returns the first occurrence, i.e. the lower vocabulary
        # id when the selected and fixed partitions have equal maximum logits.
        selected_wins_tie = exact_tie & (sel_id < self.fixed_id)
        use_selected = selected_strictly_better | selected_wins_tie

        final_id = torch.where(use_selected, sel_id, self.fixed_id)
        changed = final_id != self.baseline_top1
        return {
            "total": int(final_id.numel()),
            "regressions": int(changed.sum().item()),
            "preserved": int(final_id.numel() - changed.sum().item()),
            "selected_fixed_exact_ties": int(exact_tie.sum().item()),
        }


base.LocalityGeometry.report = tie_exact_report


if __name__ == "__main__":
    base.main()
