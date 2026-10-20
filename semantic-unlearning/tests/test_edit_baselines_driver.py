"""run_l16_paper_one.sh stage 3b with stubbed python: finished ROME / MEMIT results are kept
(only GA/GD/NPO/RMU results without early stopping are superseded); RWKU subject-gate guard."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

STUB = """#!/bin/bash
echo "python $*" >> "$LOGF"
args=("$@"); for i in "${!args[@]}"; do
  [[ "${args[$i]}" == --run-dir ]] && echo '{"baseline": {"stop_prob": 1e-4}}' > "${args[$((i+1))]}/official_mcf_eval.json"
done; exit 0
"""


def _layout(tmp):
    root = tmp / "root"
    (root / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "run_l16_paper_one.sh", root / "scripts")
    out = root / "outputs"
    lin = out / "mcf_multiseed_regular_v1" / "seed1" / "L19" / "router"
    lin.mkdir(parents=True)
    (lin / "fact_association_embeddings.pt").touch()
    v22 = out / "l16_paper_v1" / "mcf" / "seed1" / "router_v22"
    (v22 / "S06_R19-23-27").mkdir(parents=True)
    (v22 / "S06_R19-23-27" / "fact_association_embeddings.pt").touch()
    (v22 / "summary.json").write_text(json.dumps({"selected_dir": "S06_R19-23-27"}))
    seed = out / "l16_paper_v1" / "mcf" / "seed1"
    for m in ("base", "rome", "ga"):
        (seed / m).mkdir(parents=True)
        (seed / m / "official_mcf_eval.json").write_text(json.dumps({"baseline": {"baseline": m}}))
    ref = out / "mcf_fact_assoc_router_v2_seed1"
    ref.mkdir(parents=True)
    (ref / "association_manifest.json").write_text(json.dumps({"mcf_path": "/x/mcf.json"}))
    (tmp / "bin").mkdir()
    (tmp / "bin" / "python").write_text(STUB)
    (tmp / "bin" / "python").chmod(0o755)
    return root, seed


def _run(tmp, root, env):
    log = tmp / "calls.log"
    log.write_text("")
    r = subprocess.run(["bash", "scripts/run_l16_paper_one.sh", *env.pop("_args", ["mcf", "1"])], cwd=root,
                       env={**os.environ, "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}", "LOGF": str(log), **env},
                       capture_output=True, text=True)
    return r, log.read_text()


def test_finished_rome_is_kept_and_stale_ga_is_rerun(tmp_path):
    if shutil.which("jq") is None:
        import pytest
        pytest.skip("jq not installed")
    root, seed = _layout(tmp_path)
    r, calls = _run(tmp_path, root, {"BASELINES": "rome ga", "SKIP_OURS": "1", "BUILD_ROUTERS": "0"})
    assert r.returncode == 0, r.stderr
    assert (seed / "rome" / "official_mcf_eval.json").is_file()
    assert not list(seed.glob("rome.superseded_*"))            # ROME has no early stop: kept
    assert list(seed.glob("ga.superseded_*"))                  # GA without stop_prob: rerun
    assert "--baseline ga" in calls and "--baseline rome" not in calls


def test_rwku_subject_gate_needs_its_own_tag(tmp_path):
    r = subprocess.run(["bash", "scripts/run_l16_paper_one.sh", "rwku", "1"], cwd=ROOT,
                       env={**os.environ, "RWKU_GATE": "subject", "TAG": "l16_paper_v1"},
                       capture_output=True, text=True)
    assert r.returncode == 2 and "RWKU_GATE=subject needs its own TAG" in r.stderr
