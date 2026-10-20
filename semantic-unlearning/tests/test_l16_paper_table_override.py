"""l16_paper_table.py --ours-from: one dataset's ours/genie from another run, the rest unchanged."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def _eval(d: Path, eff, gen, spe):
    d.mkdir(parents=True, exist_ok=True)
    (d / "official_mcf_eval.json").write_text(json.dumps(
        {"forget": {"Eff": eff, "Gen": gen, "Spe": spe}, "forget_PPL": 11.313}))


def test_ours_from_takes_ours_and_genie_only(tmp_path):
    main, lin = tmp_path / "main", tmp_path / "lin19"
    for s in (1, 2):
        sd = main / "mcf" / f"seed{s}"
        _eval(sd / "base", 13.0, 10.0, 20.0)
        _eval(sd / "ga", 0.02, 0.3, 0.4)
        _eval(sd / "ours", 0.001, 0.05, 19.2)                 # v2.2 rows (main)
        _eval(sd / "genie", 0.001, 0.002, 20.0)
        (sd / "router_choice.json").write_text(json.dumps({"router": "factorized_v2.2"}))
        ld = lin / "mcf" / f"seed{s}"
        _eval(ld / "ours", 0.001, 0.09, 19.8)                 # linear L19 rows
        _eval(ld / "genie", 0.001, 0.002, 20.0)
        (ld / "router_choice.json").write_text(json.dumps({"router": "linear_L19"}))
    r = subprocess.run([sys.executable, "scripts/l16_paper_table.py", "--root", str(main),
                        "--ours-from", f"mcf={lin}"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    out = json.loads((main / "paper_table_mcf-lin19.json").read_text())
    assert abs(out["mcf/ours"]["Gen"]["mean"] - 0.09) < 1e-9 and abs(out["mcf/ours"]["Spe"]["mean"] - 19.8) < 1e-9
    assert abs(out["mcf/base"]["Eff"]["mean"] - 13.0) < 1e-9 and abs(out["mcf/ga"]["Gen"]["mean"] - 0.3) < 1e-9
    assert "linear_L19" in (main / "paper_table_mcf-lin19.md").read_text()
    assert not (main / "paper_table.md").exists()            # the default table is untouched
