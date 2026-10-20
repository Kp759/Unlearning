"""scripts/paper_model_config.sh under `set -euo pipefail` (as submit_paper_model.sh sources it):
sourcing must survive a cache with no snapshot yet (before `fetch`), and find one once present."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _source(model: str, cache: Path, then: str = 'echo "MODEL_PATH=[$MODEL_PATH]"'):
    script = f'set -euo pipefail; export MODEL={model} HF_CACHE="{cache}"; ' \
             f'source scripts/paper_model_config.sh; {then}'
    return subprocess.run(["bash", "-c", script], cwd=ROOT, capture_output=True, text=True)


@pytest.mark.parametrize("model", ["llama31_8b", "qwen3_4b"])
def test_source_survives_missing_snapshot(tmp_path, model):
    r = _source(model, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "MODEL_PATH=[]" in r.stdout
    chk = _source(model, tmp_path, then="paper_model_check || echo rc=$?")
    assert "rc=2" in chk.stdout and "fetch" in chk.stderr


def test_source_finds_snapshot_and_checks_layers(tmp_path):
    snap = tmp_path / "models--Qwen--Qwen3-4B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"num_hidden_layers": 30}))
    r = _source("qwen3_4b", tmp_path)
    assert r.returncode == 0 and f"MODEL_PATH=[{snap}]" in r.stdout, r.stderr
    chk = _source("qwen3_4b", tmp_path, then="paper_model_check || echo rc=$?")
    assert "rc=2" in chk.stdout and "has 30 layers, config expects 36" in chk.stderr
