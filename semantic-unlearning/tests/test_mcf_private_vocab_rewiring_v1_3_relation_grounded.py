from __future__ import annotations

from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_private_vocab_rewiring_v1_3_training_views_v3 as v3  # noqa: E402


def record(role: str, relation: str = "P364", prompt: str = "{} was created in the language"):
    return {
        "case_id": 1,
        "requested_rewrite": {
            "prompt": prompt,
            "subject": "Example",
            "relation_id": relation,
            "target_true": {"str": "French"},
            "target_new": {"str": "Hindi"},
        },
        "data_role": role,
    }


def test_validate_direct_accepts_sanitized_forget_and_protection_fit():
    v3.validate_direct([record("forget")], "forget")
    v3.validate_direct([record("protection_fit")], "protection_fit")


def test_validate_direct_rejects_probe_fields():
    row = record("protection_fit")
    row["paraphrase_prompts"] = ["forbidden"]
    with pytest.raises(RuntimeError, match="held-out probe leaked"):
        v3.validate_direct([row], "protection_fit")


def test_validate_direct_rejects_wrong_role():
    with pytest.raises(RuntimeError, match="wrong role"):
        v3.validate_direct([record("protection_fit")], "forget")


def test_minimal_instruction_never_contains_answer_values():
    text = v3.minimal_instruction("District 13", "District 13 was created in the language", 12, 0)
    assert "District 13" in text
    assert "French" not in text
    assert "Hindi" not in text
