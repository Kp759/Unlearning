from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_private_vocab_rewiring_v1_3_training_views_v2 as builder  # noqa: E402


def test_true_answer_inside_required_subject_is_not_leakage():
    leak, template = builder.answer_leak_outside_subject(
        "Who manufactures BMW M5? Answer:",
        "BMW M5",
        "BMW",
        "Toyota",
    )
    assert template == "Who manufactures {}? Answer:"
    assert leak is False


def test_true_answer_repeated_outside_subject_is_rejected():
    leak, template = builder.answer_leak_outside_subject(
        "BMW M5 is manufactured by BMW",
        "BMW M5",
        "BMW",
        "Toyota",
    )
    assert template == "{} is manufactured by BMW"
    assert leak is True


def test_new_answer_outside_subject_is_rejected():
    leak, template = builder.answer_leak_outside_subject(
        "Was BMW M5 manufactured by Toyota?",
        "BMW M5",
        "BMW",
        "Toyota",
    )
    assert template == "Was {} manufactured by Toyota?"
    assert leak is True
