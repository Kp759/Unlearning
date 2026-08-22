#!/usr/bin/env python3
"""RWKU Directional SURE v2.1: all non-special sensitive answer rows.

This is a one-factor correction to v2.0. It reuses the v2.0 learner and changes
only the sensitive vocabulary-row selector. Every non-special vocabulary token
observed in the teacher-forced sensitive answers is editable in both the input
embedding and untied LM-head sparse FP32 delta tables. All optimization
hyperparameters, directional bases, external-Wikipedia splits, and acceptance
budgets remain unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

import rwku_directional_sure_v2 as v2
import sure_canonical_core as core

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]


def all_non_special_sensitive_rows(
    tokenizer: Any,
    cases: Sequence[core.SensitivePredictionCase],
    tids: torch.Tensor,
    source_cfg: Mapping[str, Any],
    prompt_count: int,
) -> Tuple[List[int], Dict[str, Any]]:
    del source_cfg
    if len(cases) != int(tids.numel()):
        raise ValueError("Sensitive prediction cases and token IDs do not align")

    special_ids = {
        int(value)
        for value in getattr(tokenizer, "all_special_ids", [])
        if value is not None
    }
    observed = [int(value) for value in tids.detach().cpu().tolist()]
    selected = sorted(set(observed) - special_ids)
    if not selected:
        raise RuntimeError("Directional SURE v2.1 selected no non-special sensitive rows")

    selected_set = set(selected)
    coverage = [0] * int(prompt_count)
    case_coverage = 0
    rejected_special: List[int] = []
    for case, token_id in zip(cases, observed):
        if token_id in selected_set:
            coverage[int(case.record_position)] += 1
            case_coverage += 1
        else:
            rejected_special.append(token_id)

    uncovered_prompts = [i for i, count in enumerate(coverage) if count == 0]
    if uncovered_prompts:
        raise RuntimeError(
            "All-sensitive row policy left prompts without an editable target row: "
            f"{uncovered_prompts[:10]}"
        )
    if case_coverage != len(cases) - len(rejected_special):
        raise RuntimeError("Sensitive-token case coverage accounting mismatch")

    audit = {
        "policy": "all_non_special_token_ids_observed_in_sensitive_teacher_forced_answers",
        "selected_sensitive_row_ids": selected,
        "selected_row_ids": selected,
        "selected_sensitive_row_count": len(selected),
        "selected_row_decodings": {
            str(token_id): str(tokenizer.decode([token_id], skip_special_tokens=False))
            for token_id in selected
        },
        "special_token_ids_excluded": sorted(set(rejected_special)),
        "sensitive_prediction_case_count": len(cases),
        "editable_sensitive_prediction_case_count": case_coverage,
        "minimum_editable_rows_per_atomic_prompt": min(coverage),
        "all_non_special_sensitive_prediction_cases_covered": True,
    }
    return selected, audit


def main() -> None:
    v2.SCHEMA = "rwku_directional_sure_v21_configuration_v1"
    v2.EXPERIMENT_ID = "rwku-directional-sure-v21-stephen-king-seed0"
    v2.DEFAULT_CONFIGURATION = (
        PROJECT_ROOT / "config" / "rwku" / "directional_sure_v21_seed0.json"
    )
    # main() resolves this name in the v2 module at runtime.
    v2._content_sensitive_rows = all_non_special_sensitive_rows
    v2.main()


if __name__ == "__main__":
    main()
