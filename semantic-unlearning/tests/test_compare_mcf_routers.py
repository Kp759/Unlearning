"""scripts/compare_mcf_routers.py on fake run dirs, and the driver's MCF_ROUTER guards."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_mcf_routers as cmp  # noqa: E402


def _run(d: Path, eff, gen, spe, ppl=11.313, router=None, leak=None, oracle=0):
    d.mkdir(parents=True, exist_ok=True)
    (d / "official_mcf_eval.json").write_text(json.dumps(
        {"forget": {"Eff": eff, "Gen": gen, "Spe": spe}, "forget_PPL": ppl}))
    if router is not None:
        (d.parent / "router_choice.json").write_text(json.dumps({"router": router}))
        (d / "training_report.json").write_text(json.dumps(
            {"rounds": [{"onpolicy_leak": leak}], "facts_routed_by_router": 150 - oracle,
             "facts_trained_with_oracle_routing": list(range(oracle))}))


def test_paired_difference_and_fallback_flag(tmp_path):
    v22, lin = tmp_path / "v22", tmp_path / "lin"
    for s, (gv, gl) in {1: (0.05, 0.50), 2: (0.10, 0.30)}.items():
        sd = f"mcf/seed{s}"
        _run(v22 / sd / "base", 13.0, 10.0, 20.0)
        _run(v22 / sd / "ours", 0.001, gv, 19.0, router="factorized_v2.2" if s == 1 else "linear_L19_fallback",
             leak=0.01)
        _run(v22 / sd / "genie", 0.001, 0.002, 20.0)
        _run(lin / sd / "ours", 0.002, gl, 17.0, router="linear_L19", leak=0.03, oracle=4)
        _run(lin / sd / "genie", 0.001, 0.002, 20.0)
    res = cmp.compare(v22, lin)
    assert res["seeds"] == ["seed1", "seed2"] and res["paired_seeds"] == ["seed1", "seed2"]
    gen_mean, _, n = res["paired_linear_minus_v22"]["Gen"]
    assert n == 2 and gen_mean == pytest.approx(((0.50 - 0.05) + (0.30 - 0.10)) / 2)
    assert res["paired_linear_minus_v22"]["Spe"][0] == pytest.approx(-2.0)
    assert res["diagnostics"]["seed1"]["linear"]["facts_oracle_trained"] == 4
    md = cmp.render(res, 3)
    assert "Ours, linear L19 router" in md and "fell back to the L19 router on seed2" in md


def _driver(env):
    return subprocess.run(["bash", "scripts/run_l16_paper_one.sh", "mcf", "1"], cwd=ROOT,
                          env={**os.environ, **env}, capture_output=True, text=True)


def test_driver_refuses_linear_mcf_under_the_main_tag():
    r = _driver({"MCF_ROUTER": "linear", "TAG": "l16_paper_v1"})
    assert r.returncode == 2 and "needs its own TAG" in r.stderr


def test_driver_rejects_unknown_mcf_router():
    r = _driver({"MCF_ROUTER": "v3"})
    assert r.returncode == 2 and "MCF_ROUTER must be" in r.stderr
