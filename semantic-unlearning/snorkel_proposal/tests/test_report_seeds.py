"""The leaderboard averages the seeds of one setting (split names differ only by -s<seed>) and
ignores older runs set aside next to a method's directory (runs/ours_judge05)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalpel.report import load_runs, summarize  # noqa: E402


def _run(root, seed, dirname, method, score):
    d = root / f"seed{seed}" / "runs" / dirname
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "track": "fact", "split": f"scalpel-fact-dev-s{seed}", "model_id": "m", "n_requests": 50, "seed": seed,
        "method": {"name": method, "family": "x"}, "aggregates_tf": {"SCALPEL": score},
        "cost": {"unlearn_seconds": 1.0, "inference_overhead": 1.0}}))
    return d / "results.json"


def test_seeds_are_averaged_and_set_aside_runs_ignored(tmp_path):
    paths = [_run(tmp_path, s, "ours", "ours", 10.0 * s) for s in (1, 2, 3)]
    paths.append(_run(tmp_path, 1, "ours_judge05", "ours", 99.0))
    paths.append(_run(tmp_path, 1, "noop.incomplete_20261024", "noop", 5.0))
    paths.append(_run(tmp_path, 1, "sure_r", "sure_r", 7.0))          # its own method, not an aside copy of sure
    runs = load_runs(paths)
    assert sorted(Path(r["_path"]).parent.name for r in runs) == ["ours", "ours", "ours", "sure_r"]
    table = summarize(runs)
    ours = next(v for v in table.values() if v["method"] == "ours")
    assert ours["seeds"] == [1, 2, 3] and ours["SCALPEL"][0] == 20.0
    assert "scalpel-fact-dev ·" in ours["track"]
