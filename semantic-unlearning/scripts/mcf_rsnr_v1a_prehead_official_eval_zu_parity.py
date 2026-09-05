#!/usr/bin/env python3
"""Run RSNR-V1A-PreHead evaluation with ZeroUnlearn PAPER Eff/Gen.

ZeroUnlearn Eq. (16) reports residual likelihood of the original target, not the
CounterFact target_new-vs-target_true edit-success statistic.  This wrapper
computes paper-facing Eff/Gen directly from raw per-prompt target_true NLLs.
All model scoring, routing, generation, artifacts, and RSNR-native diagnostics
remain unchanged.  Paper Spe (neighborhood argmax accuracy) requires the newer
accuracy-aware evaluator and is not claimed by this compatibility wrapper.
"""
from __future__ import annotations

import mcf_zero_unlearn_official_eval as official
from mcf_zero_unlearn_metric_parity import apply_zero_unlearn_eff_gen


_ORIGINAL_SUMMARIZE = official.official_summarize


def _zero_unlearn_parity_summarize(split_name, metric_data):
    base = _ORIGINAL_SUMMARIZE(split_name, metric_data)
    return apply_zero_unlearn_eff_gen(base, metric_data)


official.official_summarize = _zero_unlearn_parity_summarize

import mcf_rsnr_v1a_prehead_official_eval as evaluator  # noqa: E402


if __name__ == "__main__":
    evaluator.main()
