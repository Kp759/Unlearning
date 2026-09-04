from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_rsnr_v1a_oracle as core  # noqa: E402
import run_mcf_rsnr_v1a_oracle_strict as strict  # noqa: E402


def _completion(*, passed: int, failures: int):
    return {
        "protocol": core.PROTOCOL,
        "joint_passed": passed,
        "joint_failures": failures,
        "adapter_saved": True,
        "base_weights_modified": False,
        "heldout_probe_text_used": False,
    }


def test_strict_completion_accepts_full_joint_pass():
    passed, reasons = strict.validate_completion(_completion(passed=50, failures=0), expected_count=50)
    assert passed is True
    assert reasons == []


def test_strict_completion_rejects_any_joint_failure():
    passed, reasons = strict.validate_completion(_completion(passed=49, failures=1), expected_count=50)
    assert passed is False
    assert "joint_passed_mismatch" in reasons
    assert "joint_failures_nonzero" in reasons


def test_mark_completion_persists_explicit_unsuccessful_status(tmp_path: Path):
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(_completion(passed=49, failures=1)), encoding="utf-8")
    passed, reasons = strict.mark_completion(path, expected_count=50)
    assert passed is False
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["training_gate_passed"] is False
    assert payload["run_successful"] is False
    assert payload["failure_reasons"] == reasons


def test_manual_launcher_uses_strict_entrypoint():
    source = (SCRIPTS / "run_mcf_rsnr_v1a_oracle_manual.sh").read_text(encoding="utf-8")
    assert "run_mcf_rsnr_v1a_oracle_strict.py" in source
    assert "python -u scripts/run_mcf_rsnr_v1a_oracle.py" not in source
