from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_private_vocab_rewiring_v1_2_true_suppression as v12  # noqa: E402


def test_registered_objective_has_no_target_new_gradient():
    registry = json.loads(
        (ROOT / "protocols" / "mcf_private_vocab_rewiring_v1_2_true_suppression_registry.json").read_text()
    )
    v12.validate_registry(registry)
    assert registry["forget_objective"]["target_true_only_gradient"] is True
    assert registry["forget_objective"]["target_new_gradient"] is False
