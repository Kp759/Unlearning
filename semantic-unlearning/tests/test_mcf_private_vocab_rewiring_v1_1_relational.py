from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_private_vocab_rewiring_v1_1_relational as relational  # noqa: E402


def _record(case_id, subject, relation_id, prompt):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "subject": subject,
            "relation_id": relation_id,
            "prompt": prompt,
            "target_true": {"str": "old"},
            "target_new": {"str": "new"},
        },
    }


def test_other_relation_bank_excludes_the_forgotten_relation():
    forget = [_record(7, "Belgium", "R_FORGET", "{} speaks")]
    protection = [
        _record(11, "Other", "R_FORGET", "{} speaks"),
        _record(12, "Other", "R_CAPITAL", "The capital of {} is"),
        _record(13, "Other", "R_REGION", "{} is located in"),
    ]
    contexts, stats = relational.make_relation_preserving_retain_contexts(
        forget, protection
    )
    assert "The capital of Belgium is" in contexts
    assert "Belgium is located in" in contexts
    assert "Belgium speaks" not in contexts
    assert stats["same_subject_different_relation_contexts_added"] == 2
