#!/usr/bin/env python3
"""Run RSNR-V1A-PreHead evaluation with exact ZeroUnlearn Eff/Gen labels.

This wrapper patches only the summary-label mapping before importing/running the
existing evaluator.  Model scoring, routing, prompt sampling, generation,
artifacts, PPL, and all RSNR-native metrics remain unchanged.
"""
from __future__ import annotations

import mcf_zero_unlearn_official_eval as official
from mcf_zero_unlearn_metric_parity import apply_zero_unlearn_eff_gen


_ORIGINAL_SUMMARIZE = official.official_summarize


def _zero_unlearn_parity_summarize(split_name, metric_data):
    return apply_zero_unlearn_eff_gen(_ORIGINAL_SUMMARIZE(split_name, metric_data))


official.official_summarize = _zero_unlearn_parity_summarize

import mcf_rsnr_v1a_prehead_official_eval as evaluator  # noqa: E402


if __name__ == "__main__":
    evaluator.main()
