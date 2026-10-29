"""l16_paper_table.py: the RR column comes from representation_recoverability.json when present."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _eval(d: Path, eff, gen, spe, rr=None):
    d.mkdir(parents=True, exist_ok=True)
    (d / "official_mcf_eval.json").write_text(json.dumps(
        {"forget": {"Eff": eff, "Gen": gen, "Spe": spe}, "forget_PPL": 11.3}))
    if rr is not None:
        (d / "representation_recoverability.json").write_text(json.dumps(
            {"headline": {"RR": {"value": rr, "layer": 18}, "RR_lens": {"value": rr - 1, "layer": 17}}}))


def test_rr_column_mean_over_seeds_and_dash_when_missing(tmp_path):
    root = tmp_path / "run"
    for s, rr in ((1, 90.0), (2, 94.0)):
        sd = root / "mcf" / f"seed{s}"
        _eval(sd / "base", 13.0, 10.0, 20.0, rr=96.0)
        _eval(sd / "ours", 0.001, 0.05, 19.2, rr=rr)
        _eval(sd / "genie", 0.001, 0.002, 20.0)          # stage 6 not run
    r = subprocess.run([sys.executable, "scripts/l16_paper_table.py", "--root", str(root)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads((root / "paper_table.json").read_text())
    assert abs(out["mcf/ours"]["RR"]["mean"] - 92.0) < 1e-9 and out["mcf/ours"]["RR"]["n"] == 2
    assert abs(out["mcf/base"]["RR"]["mean"] - 96.0) < 1e-9
    assert out["mcf/genie"]["RR"]["mean"] is None
    md = (root / "paper_table.md").read_text()
    assert "| RR ↓ |" in md and "| 92.000 ± 2.828 |" in md
    assert "RR $\\downarrow$" in (root / "paper_table.tex").read_text()
    csv = (root / "paper_table_per_seed.csv").read_text()
    assert "RR" in csv.splitlines()[0]
