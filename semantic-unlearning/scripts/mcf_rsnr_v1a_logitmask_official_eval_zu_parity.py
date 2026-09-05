#!/usr/bin/env python3
"""Run direct-logit MCF evaluation with ZeroUnlearn PAPER Eff/Gen.

Paper Eff/Gen are residual target_true likelihoods (lower is better).  The
CounterFact target_new-vs-target_true pairwise statistic is retained separately
as CF_EditSuccess_* and is not used as Eff/Gen.
"""
from __future__ import annotations

import mcf_zero_unlearn_official_eval as official
from mcf_zero_unlearn_metric_parity import apply_zero_unlearn_eff_gen


_ORIGINAL_SUMMARIZE = official.official_summarize


def _zero_unlearn_parity_summarize(split_name, metric_data):
    base = _ORIGINAL_SUMMARIZE(split_name, metric_data)
    return apply_zero_unlearn_eff_gen(base, metric_data)


official.official_summarize = _zero_unlearn_parity_summarize

import mcf_rsnr_v1a_logitmask_official_eval as evaluator  # noqa: E402


if __name__ == "__main__":
    evaluator.main()
