from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_private_vocab_rewiring_v1_3_training_views_v5 as v5  # noqa: E402


def _record(case_id: int, role: str, subject: str, relation_id: str, prompt: str):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": prompt,
            "subject": subject,
            "relation_id": relation_id,
            "target_true": {"str": "football"},
            "target_new": {"str": "basketball"},
        },
        "data_role": role,
    }


def test_relation_anchor_cycles_only_matching_training_visible_relation():
    v5._FORGET_RELATION_BY_KEY.clear()
    v5._RELATION_TEMPLATES.clear()

    forget = [_record(1, "forget", "Example Player", "P641", "{} plays the sport of")]
    fit = [
        _record(2, "protection_fit", "Other A", "P641", "The sport associated with {} is"),
        _record(3, "protection_fit", "Other B", "P641", "{} is known for playing"),
        _record(4, "protection_fit", "Other C", "P413", "The position played by {} is"),
    ]
    v5.capture_direct(forget, "forget")
    v5.capture_direct(fit, "protection_fit")

    canonical = "Example Player plays the sport of"
    text0 = v5.relation_anchored_instruction("Example Player", canonical, 12, 0)
    text1 = v5.relation_anchored_instruction("Example Player", canonical, 12, 1)
    text2 = v5.relation_anchored_instruction("Example Player", canonical, 12, 2)

    assert "Relation identifier: P641" in text0
    assert "[SUBJECT] plays the sport of" in text0
    assert "The sport associated with [SUBJECT] is" in text1
    assert "[SUBJECT] is known for playing" in text2
    assert "position played" not in text0 + text1 + text2


def test_relation_anchor_never_includes_literal_subject_in_anchor():
    v5._FORGET_RELATION_BY_KEY.clear()
    v5._RELATION_TEMPLATES.clear()
    forget = [_record(1, "forget", "Bashkim Kadrii", "P641", "{} plays the sport of")]
    fit = [_record(2, "protection_fit", "Other A", "P641", "The sport associated with {} is")]
    v5.capture_direct(forget, "forget")
    v5.capture_direct(fit, "protection_fit")
    text = v5.relation_anchored_instruction(
        "Bashkim Kadrii", "Bashkim Kadrii plays the sport of", 12, 1
    )
    assert "Same-relation training-visible anchor:\nThe sport associated with [SUBJECT] is" in text
    assert "Bashkim Kadrii" not in text
