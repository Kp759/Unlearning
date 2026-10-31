#!/usr/bin/env python3
"""Re-render a leaderboard directory from its saved tables, without the runs.

    python scripts/render_leaderboard.py results/cluster/leaderboard_dev --title "LETHE-Bench — all models — dev"
    python scripts/render_leaderboard.py results/cluster/qwen3_4b/dev/leaderboard --exclude SURE_early

Reads leaderboard.json (teacher-forced scorer) and, if present, leaderboard_gen.json (generation
scorer), applies the current display rules of scalpel.report (split name shown as lethe-…,
excluded display names dropped), and rewrites leaderboard.md, both JSON files and pareto.png.
Use it after a display change to the report, or to drop a row, when the runs are not at hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scalpel.report import markdown, pareto_plot  # noqa: E402


def _load(path):
    table = json.loads(path.read_text())
    for row in table.values():
        for k, v in list(row.items()):
            if isinstance(v, list) and len(v) == 2 and all(x is None or isinstance(x, (int, float)) for x in v):
                row[k] = tuple(v)
        row["families"] = {k: tuple(v) for k, v in row.get("families", {}).items()}
        row["attacks"] = {k: tuple(v) if isinstance(v, list) else v for k, v in row.get("attacks", {}).items()}
    return table


def _apply(table, exclude):
    out = {}
    for key, row in table.items():
        if row["method"] in exclude:
            continue
        track = row["track"].replace("scalpel-", "lethe-")
        row["track"] = track
        out[f"{track}:{row['method']}"] = row
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--title", default=None, help="default: the first line of the existing leaderboard.md")
    ap.add_argument("--exclude", nargs="*", default=["SURE_early", "sure_original"],
                    help="display names to drop (default: the SURE ablation and the original adapter)")
    a = ap.parse_args(argv)
    d = Path(a.directory)
    md_path = d / "leaderboard.md"
    title = a.title or (md_path.read_text().splitlines()[0].lstrip("# ").strip() if md_path.is_file() else "LETHE-Bench leaderboard")
    table = _apply(_load(d / "leaderboard.json"), set(a.exclude))
    md = markdown(table, title)
    (d / "leaderboard.json").write_text(json.dumps(table, indent=1, default=str) + "\n")
    gen_path = d / "leaderboard_gen.json"
    if gen_path.is_file():
        gen = _apply(_load(gen_path), set(a.exclude))
        gen_path.write_text(json.dumps(gen, indent=1, default=str) + "\n")
        if any(r["SCALPEL"][0] is not None for r in gen.values()):
            md += "\n# Open-ended (generation) scorer\n" + markdown(gen, title).split("\n", 1)[1]
    md_path.write_text(md)
    pareto_plot(table, d / "pareto.png")
    print(f"{d}: {len(table)} rows -> leaderboard.md, leaderboard.json"
          + (", leaderboard_gen.json" if gen_path.is_file() else "") + ", pareto.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
