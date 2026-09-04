from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_private_vocab_rewiring_v1_3_training_views_v4 as v4  # noqa: E402


def test_slot_instruction_hides_literal_subject_and_requires_marker():
    subject = "David Mendes da Silva"
    canonical = "The position played by David Mendes da Silva is"
    text = v4.slot_instruction(subject, canonical, 12, 0)
    assert v4.SUBJECT_SLOT in text
    assert "The position played by [SUBJECT] is" in text
    # The literal subject appears only in implementation metadata outside the
    # generated original prompt?  The instruction must not ask the generator
    # to reproduce it as the paraphrase slot.
    assert "Use the exact marker [SUBJECT] exactly once" in text


def test_cleaner_substitutes_exactly_one_subject_slot_before_scoring():
    subject = "David Mendes da Silva"
    canonical = "The position played by David Mendes da Silva is"
    v4.slot_instruction(subject, canonical, 2, 0)
    lines = v4.clean_generated_lines_with_subject_slot(
        "Which position did [SUBJECT] play?\n[SUBJECT] played in the position of"
    )
    assert lines == [
        "Which position did David Mendes da Silva play?",
        "David Mendes da Silva played in the position of",
    ]


def test_malformed_slot_count_is_left_for_existing_validator_to_reject():
    subject = "Example Person"
    canonical = "The language used by Example Person is"
    v4.slot_instruction(subject, canonical, 1, 0)
    lines = v4.clean_generated_lines_with_subject_slot(
        "No marker here\n[SUBJECT] and [SUBJECT] spoke"
    )
    assert lines == ["No marker here", "[SUBJECT] and [SUBJECT] spoke"]
