"""The leaderboard averages the seeds of one setting (split names differ only by -s<seed>) and
ignores older runs set aside next to a method's directory (runs/ours_judge05)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scalpel.report import load_runs, summarize  # noqa: E402


def _run(root, seed, dirname, method, score, routing=None):
    d = root / f"seed{seed}" / "runs" / dirname
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "track": "fact", "split": f"scalpel-fact-dev-s{seed}", "model_id": "m", "n_requests": 50, "seed": seed,
        "method": {"name": method, "family": "x"}, "aggregates_tf": {"SCALPEL": score},
        "cost": {"unlearn_seconds": 1.0, "inference_overhead": 1.0}, **({"routing": routing} if routing else {})}))
    return d / "results.json"


def test_seeds_are_averaged_and_set_aside_runs_ignored(tmp_path):
    paths = [_run(tmp_path, s, "ours", "ours", 10.0 * s) for s in (1, 2, 3)]
    paths.append(_run(tmp_path, 1, "ours_judge05", "ours", 99.0))
    paths.append(_run(tmp_path, 1, "noop.incomplete_20261024", "noop", 5.0))
    paths.append(_run(tmp_path, 1, "sure_r", "sure_r", 7.0))          # its own method, not an aside copy of sure
    runs = load_runs(paths)
    assert sorted(Path(r["_path"]).parent.name for r in runs) == ["ours", "ours", "ours", "sure_r"]
    table = summarize(runs)
    ours = next(v for v in table.values() if v["method"] == "SURE")      # run directory ours, shown as SURE
    assert ours["seeds"] == [1, 2, 3] and ours["SCALPEL"][0] == 20.0
    assert "scalpel-fact-dev ·" in ours["track"]


def test_routing_diagnostics_average_over_seeds(tmp_path):
    """The detect-then-edit table used to print the first seed's routing numbers only."""
    def routing(recall):
        return {"by_family": {}, "forget_route_recall": recall, "forget_residual_from_routing": 100 - recall,
                "forget_residual_from_actuation": 0.0, "retain_false_fire_rate": 5.0}
    paths = [_run(tmp_path, s, "ours", "ours", 20.0, routing(r)) for s, r in ((1, 80.0), (2, 90.0), (3, 100.0))]
    paths.append(_run(tmp_path, 1, "ga", "ga", 5.0))
    table = summarize(load_runs(paths))
    ours = next(v for v in table.values() if v["method"] == "SURE")
    assert ours["routing"]["forget_route_recall"] == 90.0 and ours["routing"]["forget_residual_from_routing"] == 10.0
    assert ours["routing"]["retain_false_fire_rate"] == 5.0 and "by_family" not in ours["routing"]
    assert next(v for v in table.values() if v["method"] == "ga")["routing"] is None


def test_exclude_drops_a_method_and_display_names_apply(tmp_path):
    paths = [_run(tmp_path, 1, "ours", "ours", 30.0), _run(tmp_path, 1, "sure", "sure", 12.0), _run(tmp_path, 1, "ga", "ga", 5.0)]
    names = sorted(v["method"] for v in summarize(load_runs(paths)).values())
    assert names == ["SURE", "ga", "sure_original"]
    names = sorted(v["method"] for v in summarize(load_runs(paths, exclude=["sure"])).values())
    assert names == ["SURE", "ga"]
