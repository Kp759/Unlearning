from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_private_vocab_rewiring_v1_3_training_views_v5 as v5  # noqa: E402
import build_mcf_private_vocab_rewiring_v1_3_training_views_v6 as v6  # noqa: E402


def test_microparaphrase_instruction_uses_same_relation_anchor_and_subject_slot():
    subject = "Jiunie Booth"
    canonical = "Jiunie Booth plays in the genre of"
    key = (subject, canonical)
    old_rel = dict(v5._FORGET_RELATION_BY_KEY)
    old_templates = {k: list(v) for k, v in v5._RELATION_TEMPLATES.items()}
    try:
        v5._FORGET_RELATION_BY_KEY.clear()
        v5._RELATION_TEMPLATES.clear()
        v5._FORGET_RELATION_BY_KEY[key] = "P136"
        v5._RELATION_TEMPLATES["P136"] = ["{} plays in the genre of", "The genre associated with {} is"]
        text = v6.conservative_relation_instruction(subject, canonical, 12, 0)
        assert "Relation identifier: P136" in text
        assert "[SUBJECT]" in text
        assert "Jiunie Booth" not in text
        assert "MICRO-PARAPHRASES" in text
        assert "Preserve exactly the SAME semantic relation" in text
    finally:
        v5._FORGET_RELATION_BY_KEY.clear()
        v5._FORGET_RELATION_BY_KEY.update(old_rel)
        v5._RELATION_TEMPLATES.clear()
        v5._RELATION_TEMPLATES.update(old_templates)
