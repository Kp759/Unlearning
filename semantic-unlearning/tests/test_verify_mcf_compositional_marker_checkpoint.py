from __future__ import annotations

import sys
from pathlib import Path

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
