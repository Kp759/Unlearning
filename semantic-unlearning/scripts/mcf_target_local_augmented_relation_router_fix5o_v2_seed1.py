#!/usr/bin/env python3
"""Fix5o-v2 wrapper: policy-safe metadata for the data-only augmented router experiment.

The core experiment remains Fix5o. This wrapper changes no model, rows, features,
training hyperparameters, calibration data, or evaluation data. It only ensures:
  * positive augmented rows for the owner's registered forget binding retain
    forbidden=True metadata;
  * same-subject alternate-relation contrasts remain forbidden=False;
  * the authored augmentation held-out set is treated as a semantic wording probe,
    not as a policy-permission benchmark.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for p in (SCRIPT_DIR, ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_target_local_augmented_relation_router_fix5o_seed1 as core

_ORIGINAL_POLICY_REPORT = core.policy_report


def render_row(
    subject: str,
    relation: str,
    case_id: int | None,
    family: str,
    template: str,
    label: str,
    *,
    kind: str,
) -> core.Row:
    text = template.format(subject=subject, label=label)
    is_owner_positive = kind in {
        "fix5o_augmented_relation_fit",
        "fix5o_augmented_relation_heldout",
    }
    return core.Row(
        text=core.norm_text(text),
        subject=str(subject),
        relation=str(relation),
        forbidden=bool(is_owner_positive),
        kind=kind,
        family=family,
        case_id=case_id,
    )


def policy_report(
    rows: Sequence[core.Row],
    logits: Any,
    views: Sequence[core.RoutingView],
    eta: float,
    classes: Sequence[str],
    none_idx: int,
    bank: set[tuple[str, str]],
) -> Mapping[str, Any]:
    # The augmentation-heldout set is an authored semantic wording probe. Mixing its
    # owner-positive and cross-relation rows into a policy FPR/TPR summary would give
    # that probe a meaning it was not designed to have.
    if rows and all(str(r.kind).startswith("fix5o_") for r in rows):
        return {
            "status": "NOT_A_POLICY_BENCHMARK",
            "reason": "authored Fix5o held-out rows are used only for semantic wording recognition",
            "row_n": len(rows),
        }
    return _ORIGINAL_POLICY_REPORT(rows, logits, views, eta, classes, none_idx, bank)


core.render_row = render_row
core.policy_report = policy_report


if __name__ == "__main__":
    core.main()
