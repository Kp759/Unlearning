#!/usr/bin/env python3
"""Fix5d wrapper: add whole-query false-activation accounting to Fix5c.

The cached features, linear/MLP heads, semantic labels, and margin calibration are
unchanged. This layer only augments policy reports by grouping target-conditioned
routes that come from the same original query and asking whether any route activates.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_relation_head_compare_fix5c_seed1 as core

Row = core.Row
_ORIGINAL_POLICY = core.base.policy
_ORIGINAL_EVALUATE_HEAD = core.evaluate_head


def whole_query_policy_report(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    if len(rows) == 0:
        return {
            "permitted_query_n": 0,
            "permitted_query_false_activation_pct": None,
            "multi_route_permitted_query_n": 0,
            "multi_route_permitted_query_false_activation_pct": None,
        }
    pred, dm = core.base.margin(logits)
    labels = [classes[int(i)] for i in pred]
    accepted_relation = (pred != int(none_idx)) & (dm >= float(eta))
    binding = torch.tensor([(row.subject, labels[i]) in bank for i, row in enumerate(rows)], dtype=torch.bool)
    activates = accepted_relation & binding

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        if row.forbidden:
            continue
        key = (core.norm_text(row.text).casefold(), core.base.bucket(row.kind))
        groups[key].append(i)

    def summarize(selected: dict[tuple[str, str], list[int]]) -> tuple[int, float | None, int]:
        if not selected:
            return 0, None, 0
        errors = sum(bool(activates[idx].any().item()) for idx in selected.values())
        return len(selected), 100.0 * errors / len(selected), errors

    n_all, rate_all, err_all = summarize(groups)
    multi = {key: idx for key, idx in groups.items() if len(idx) > 1}
    n_multi, rate_multi, err_multi = summarize(multi)
    examples = []
    for (text, fam), idx in groups.items():
        if bool(activates[idx].any().item()) and len(examples) < 20:
            examples.append({
                "normalized_query": text,
                "negative_family": fam,
                "route_count": len(idx),
                "subjects": [rows[i].subject for i in idx],
                "predicted_relations": [labels[i] for i in idx],
            })
    return {
        "permitted_query_n": n_all,
        "permitted_query_false_activation_n": err_all,
        "permitted_query_false_activation_pct": rate_all,
        "multi_route_permitted_query_n": n_multi,
        "multi_route_permitted_query_false_activation_n": err_multi,
        "multi_route_permitted_query_false_activation_pct": rate_multi,
        "false_activation_examples": examples,
    }


def policy_with_whole_query(
    rows: Sequence[Row],
    logits: torch.Tensor,
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> dict[str, Any]:
    report = _ORIGINAL_POLICY(rows, logits, eta, list(classes), none_idx, bank)
    report["whole_query"] = whole_query_policy_report(rows, logits, eta, classes, none_idx, bank)
    return report


def evaluate_head_with_whole_query(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = _ORIGINAL_EVALUATE_HEAD(*args, **kwargs)
    wq = result["validation_policy"].get("whole_query", {})
    rate = wq.get("permitted_query_false_activation_pct")
    if rate is not None and rate > 2.0 + 1e-9:
        result["pilot_pass"] = False
        result["whole_query_gate_note"] = "pilot forced false because whole-query permitted false activation exceeded 2%"
    else:
        result["whole_query_gate_note"] = "whole-query permitted false activation did not exceed 2% on validation"
    return result


core.base.policy = policy_with_whole_query
core.evaluate_head = evaluate_head_with_whole_query


if __name__ == "__main__":
    core.main()
