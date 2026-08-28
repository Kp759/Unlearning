from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import verify_mcf_compositional_marker_checkpoint as verify


def test_reload_acceptance_requires_every_direct_and_positive_margin():
    passed = verify.acceptance_payload(
        torch.tensor([1.25, 1.01, 2.0, 1.5]),
        [True, False, True, False],
        forget_margin=1.0,
    )
    assert passed["passed"] is True
    assert passed["observed"]["direct_failures"] == 0
    assert passed["observed"]["training_safe_positive_failures"] == 0

    failed = verify.acceptance_payload(
        torch.tensor([1.25, 0.5, 0.75, 1.5]),
        [True, False, True, False],
        forget_margin=1.0,
    )
    assert failed["passed"] is False
    assert failed["observed"]["direct_failures"] == 1
    assert failed["observed"]["training_safe_positive_failures"] == 2


def test_reload_acceptance_also_requires_the_serialized_row_cap():
    margins = torch.tensor([1.25, 1.01])
    flags = [True, False]
    failed = verify.acceptance_payload(
        margins,
        flags,
        forget_margin=1.0,
        serialized_row_cap={"passed": False},
    )
    passed = verify.acceptance_payload(
        margins,
        flags,
        forget_margin=1.0,
        serialized_row_cap={"passed": True},
    )

    assert failed["passed"] is False
    assert passed["passed"] is True


def test_serialized_row_cap_report_checks_reloaded_weights():
    model = torch.nn.Module()
    model.head = torch.nn.Linear(3, 4, bias=False)
    model.get_output_embeddings = lambda: model.head
    with torch.no_grad():
        model.head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ]
            )
        )
    base = model.head.weight[[1, 3]].detach().clone()
    state = {
        "selected_output_rows": [1, 3],
        "base_selected_output_rows": base,
        "selected_relative_row_norm_cap": 0.10,
        "unconstrained_fallback_used": False,
    }
    with torch.no_grad():
        model.head.weight[1, 0] += 0.05

    report = verify.serialized_row_cap_report(model, state)

    assert report["rows"] == 2
    assert report["maximum_relative_row_norm"] == pytest.approx(0.05)
    assert report["passed"] is True
