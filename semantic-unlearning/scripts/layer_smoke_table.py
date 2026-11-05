#!/usr/bin/env python3
"""Layer smoke test summary for another model (scripts/submit_layer_smoke.sh, layer_smoke.slurm).

    python scripts/layer_smoke_table.py qwen3_4b          # -> outputs/layer_smoke_qwen3_4b/SUMMARY.md

Part 1, the router: for every dataset, the classifier refit at every block on the cached states
(F1 / FPR / FNR / recall / null FPR / perturbed recall), the best block and the model's default.
Part 2, the write layer: Eff / Gen / Spe / PPL of base, ours and genie for every (router layer R,
write layer W) run so far. Everything is read from disk; nothing is recomputed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from summarize_mcf_layer_sweep import collect  # noqa: E402
from l16_paper_table import metrics, NAMES  # noqa: E402

ROOT = SCRIPTS.parent
DEFAULTS = {"qwen3_4b": (24, 21, 36), "llama31_8b": (22, 18, 32), "llama32_3b": (19, 16, 28)}


def _fmt(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def router_part(model):
    lines = []
    for sweep_path in sorted(ROOT.glob(f"outputs/layer_smoke_{model}/*/seed*/layer_sweep/sweep.json")):
        ds, seed = sweep_path.parents[2].name, sweep_path.parents[1].name
        manifest = json.loads((sweep_path.parents[1] / "manifest.json").read_text())
        sweep = json.loads(sweep_path.read_text())
        entries = sweep.get("sweep", []) if isinstance(sweep, dict) else sweep
        rows = []
        for e in entries:
            if e.get("rule") != "deployed" or len(e.get("layers", [])) != 1:
                continue
            r = e["by_eligibility"][e.get("deployed_eligibility", "exact")]
            m = r["eval"]
            rows.append((e["layers"][0], m.get("f1"), m.get("fpr"), m.get("fnr"), m.get("recall"),
                         r.get("null_fpr"), r.get("perturbed_recall")))
        rows.sort()
        scored = [r for r in rows if r[1] is not None]
        best = max(scored, key=lambda r: (round(r[1], 6), -(r[2] or 0.0), -r[0])) if scored else None
        lines += [f"### {NAMES.get(ds, ds)} {seed}: router refit at every block "
                  f"(fit at L{manifest.get('router_layer')}, {manifest.get('n_layers')} blocks)", "",
                  "| block | F1 | FPR | FNR | recall | null FPR | perturbed recall |",
                  "|---|---|---|---|---|---|---|"]
        for r in rows:
            mark = " **best**" if best and r[0] == best[0] else (" (default)" if r[0] == manifest.get("router_layer") else "")
            lines.append(f"| L{r[0]}{mark} | " + " | ".join(_fmt(v) for v in r[1:]) + " |")
        if best:
            within = [r[0] for r in scored if r[1] >= best[1] - 0.01]
            lines += ["", f"Best F1 at L{best[0]} ({_fmt(best[1])}); blocks within 0.01 F1 of it: "
                      f"L{min(within)}–L{max(within)}. The default is L{manifest.get('router_layer')}."]
        lines.append("")
    if not lines:
        lines = ["(no router sweep yet: bash scripts/submit_layer_smoke.sh MODEL router)", ""]
    return lines


def write_part(model):
    rows = []
    for run in sorted(ROOT.glob(f"outputs/layer_smoke_{model}_r*_w*/*/seed*/*")):
        tag = run.parents[2].name
        m = re.match(rf"layer_smoke_{re.escape(model)}_r(\d+)_w(\d+)$", tag)
        if not m or run.name not in ("base", "ours", "genie", "genie_entity"):
            continue
        ds, seed = run.parents[1].name, run.parent.name
        row = collect(run, f"{tag}/{ds}/{seed}/{run.name}")
        if row.get("status") != "complete":
            continue
        rows.append((ds, int(m.group(1)), int(m.group(2)), run.name, seed, metrics(ds, row)))
    if not rows:
        return ["(no write-layer runs yet: bash scripts/submit_layer_smoke.sh MODEL write R)", ""]
    lines = ["| dataset | router R | write W | method | seed | Eff ↓ | Gen ↓ | Spe ↑ | PPL |",
             "|---|---|---|---|---|---|---|---|---|"]
    order = {"base": 0, "ours": 1, "genie": 2, "genie_entity": 3}
    for ds, r, w, method, seed, mt in sorted(rows, key=lambda x: (x[0], x[1], x[2], order.get(x[3], 9), x[4])):
        lines.append(f"| {NAMES.get(ds, ds)} | L{r} | L{w} | {method} | {seed} | "
                     + " | ".join(_fmt(mt[k]) for k in ("Eff", "Gen", "Spe", "PPL")) + " |")
    lines += ["", "Pick the W with the lowest Gen whose Spe stays at base; ours vs genie at the same W is "
              "the routing cost, genie across W is the write layer's own effect.", ""]
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    a = ap.parse_args(argv)
    d = DEFAULTS.get(a.model)
    lines = [f"# Layer smoke test: {a.model}", ""]
    if d:
        lines += [f"Config defaults: router L{d[0]}, Δe at L{d[1]} ({d[2]} blocks; the 3B model uses 19 / 16 of 28).", ""]
    lines += ["## 1. Router layer (classifier refit at every block)", "", *router_part(a.model),
              "## 2. Write layer (rows under the chosen router)", "", *write_part(a.model)]
    md = "\n".join(lines) + "\n"
    out = ROOT / f"outputs/layer_smoke_{a.model}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.md").write_text(md)
    print(md)
    print(f"wrote {out}/SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
