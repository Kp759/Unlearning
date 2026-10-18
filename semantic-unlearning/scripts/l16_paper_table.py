#!/usr/bin/env python3
"""Paper table from l16_paper_multiseed.slurm: base | ours | genie, Eff / Gen / Spe / PPL,
mean ± std over seeds (CPU, seconds).

    python scripts/l16_paper_table.py                 # outputs/l16_paper_v1 -> paper_table.*
    python scripts/l16_paper_table.py --root outputs/l16_paper_v1 --digits 2

Columns per benchmark (each from its own official evaluator, via summarize_mcf_layer_sweep.collect):
  MCF     Eff/Gen = forget answer probability (%), Spe = neighbourhood accuracy, PPL
  ZsRE    Eff/Gen = forget token accuracy (%), Spe = neighbourhood accuracy, PPL
  MQuAKE  Eff = forget rewrite accuracy, Gen = forget AtomicGen, Spe = retain Eff, PPL
  RWKU    Eff = same-50 recovery (%), Gen = mean held-out recovery (Level 1, Level 2,
          Level-2 paraphrase), Spe = neighbour recovery, PPL
Eff/Gen: lower is better (forgetting); Spe: higher is better; PPL: close to base.
Also writes diagnostics (seeds found, router used, genie routing hit rate) so a silent
key mismatch in genie (genie ~ base) is visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from summarize_mcf_layer_sweep import collect

DATASETS = ("mcf", "zsre", "mquake", "rwku")
NAMES = {"mcf": "MCF", "zsre": "ZsRE", "mquake": "MQuAKE", "rwku": "RWKU"}
METHODS = ("base", "ours", "genie")
LABELS = {"base": "Base", "ours": "Ours (L16)", "genie": "Genie"}
COLS = ("Eff", "Gen", "Spe", "PPL")


def _mean(xs):
    xs = [float(x) for x in xs if isinstance(x, (int, float)) and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else None


def metrics(ds, row):
    if ds in ("mcf", "zsre"):
        return {"Eff": row.get("forget_Eff"), "Gen": row.get("forget_Gen"),
                "Spe": row.get("forget_Spe"), "PPL": row.get("PPL")}
    if ds == "mquake":
        return {"Eff": row.get("forget_Eff"), "Gen": row.get("forget_AtomicGen"),
                "Spe": row.get("retain_Eff"), "PPL": row.get("PPL")}
    return {"Eff": row.get("forget_Eff"),
            "Gen": _mean([row.get("forget_GenL1"), row.get("forget_GenL2"), row.get("forget_GenPara")]),
            "Spe": row.get("neighbor"), "PPL": row.get("PPL")}


def stats(values):
    v = [float(x) for x in values if isinstance(x, (int, float)) and math.isfinite(float(x))]
    if not v:
        return None, None, 0
    m = sum(v) / len(v)
    s = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0
    return m, s, len(v)


def genie_check(run_dir):
    """Genie hit rate from the evaluator's runtime counters (lookups that found a row)."""
    for name in ("official_mcf_eval.json", "official_zsre_eval.json", "official_mquake_eval.json",
                 "official_rwku_eval.json"):
        p = Path(run_dir) / name
        if not p.is_file():
            continue
        d = json.loads(p.read_text())
        counters = (d.get("runtime_counters") or d.get("runtime_counters_total")
                    or (d.get("fact_association_embedding_bank") or {}).get("runtime_counters") or {})
        g = d.get("genie") or {}
        look, hit = counters.get("oracle_lookups"), counters.get("oracle_hits")
        return {"table_size": g.get("table_size"), "collisions": g.get("collisions"),
                "lookups": look, "hits": hit}
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/l16_paper_v1")
    ap.add_argument("--digits", type=int, default=3)
    a = ap.parse_args()
    root = Path(a.root)
    table, diag, per_seed = {}, [], []
    for ds in DATASETS:
        for method in METHODS:
            vals = {c: [] for c in COLS}
            seeds = []
            for sd in sorted(root.glob(f"{ds}/seed*")):
                run = sd / method
                row = collect(run, f"{ds}/{sd.name}/{method}")
                if row.get("status") != "complete":
                    continue
                m = metrics(ds, row)
                seeds.append(sd.name)
                per_seed.append({"dataset": ds, "method": method, "seed": sd.name, **m})
                for c in COLS:
                    vals[c].append(m[c])
                if method == "genie":
                    # direct-prompt routing under genie should be ~100% (else keys did not match)
                    direct = next((row[k] for k in ("forget_rewrite_route_correct", "same50_route_correct",
                                                    "forget_rewrite_route_active")
                                   if row.get(k) is not None), None)
                    diag.append({"dataset": ds, "seed": sd.name, "genie": genie_check(run),
                                 "genie_direct_routed": direct})
                if method == "ours":
                    choice = sd / "router_choice.json"
                    rep = run / "training_report.json"
                    rounds = json.loads(rep.read_text()).get("rounds", []) if rep.is_file() else []
                    diag.append({"dataset": ds, "seed": sd.name,
                                 "router": json.loads(choice.read_text()).get("router") if choice.is_file() else None,
                                 "onpolicy_leak_final": rounds[-1].get("onpolicy_leak") if rounds else None})
            table[(ds, method)] = {"seeds": seeds, **{c: stats(vals[c]) for c in COLS}}

    d = a.digits

    def cell(st):
        m, s, n = st
        if m is None:
            return "–"
        return f"{m:.{d}f} ± {s:.{d}f}"

    lines = ["# Base vs ours (L16) vs genie, seeds 1-5\n",
             "Eff, Gen ↓ (forgetting) · Spe ↑ (locality) · PPL ≈ base. mean ± std over seeds (n per row).\n",
             "| Dataset | Method | n | Eff ↓ | Gen ↓ | Spe ↑ | PPL |", "|---|---|---|---|---|---|---|"]
    for ds in DATASETS:
        for method in METHODS:
            e = table[(ds, method)]
            lines.append(f"| {NAMES[ds]} | {LABELS[method]} | {len(e['seeds'])} | "
                         + " | ".join(cell(e[c]) for c in COLS) + " |")
    lines += ["", "Spe is neighbourhood accuracy (MCF, ZsRE), retain Eff (MQuAKE) and neighbour "
              "recovery (RWKU). RWKU Gen = mean of held-out Level 1, Level 2 and Level-2 paraphrase "
              "recovery; those probes ask facts with no trained row, so RWKU genie Gen equals base by "
              "construction.", "", "## Diagnostics", "",
              "| dataset | seed | router (ours) | final on-policy leak | genie: direct prompts routed "
              "| genie table / hits / lookups |",
              "|---|---|---|---|---|---|"]
    by = {}
    for x in diag:
        by.setdefault((x["dataset"], x["seed"]), {}).update(x)
    for (ds, seed), x in sorted(by.items()):
        g = x.get("genie") or {}
        leak, direct = x.get("onpolicy_leak_final"), x.get("genie_direct_routed")
        lines.append(f"| {NAMES[ds]} | {seed} | {x.get('router') or '–'} | "
                     f"{'–' if leak is None else f'{100 * leak:.1f}%'} | "
                     f"{'–' if direct is None else f'{100 * direct:.0f}%'} | "
                     + " / ".join('–' if g.get(k) is None else str(g[k]) for k in ('table_size', 'hits', 'lookups')) + " |")
    md = "\n".join(lines) + "\n"

    tex = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llcccc}", r"\toprule",
           r"Dataset & Method & Eff $\downarrow$ & Gen $\downarrow$ & Spe $\uparrow$ & PPL \\", r"\midrule"]
    for i, ds in enumerate(DATASETS):
        for j, method in enumerate(METHODS):
            e = table[(ds, method)]
            cells = []
            for c in COLS:
                m, s, n = e[c]
                cells.append("--" if m is None else f"${m:.{d}f}_{{\\pm {s:.{d}f}}}$")
            name = rf"\multirow{{3}}{{*}}{{{NAMES[ds]}}}" if j == 0 else ""
            tex.append(f"{name} & {LABELS[method]} & " + " & ".join(cells) + r" \\")
        if i < len(DATASETS) - 1:
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabular}",
            r"\caption{Base model, our method (question-side $\Delta e$ written at layer 16) and "
            r"genie (the same rows under perfect routing). Mean $\pm$ std over 5 seeds.}",
            r"\label{tab:l16_main}", r"\end{table}"]

    root.mkdir(parents=True, exist_ok=True)
    (root / "paper_table.md").write_text(md)
    (root / "paper_table.tex").write_text("\n".join(tex) + "\n")
    with (root / "paper_table_per_seed.csv").open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=["dataset", "method", "seed", *COLS])
        w.writeheader()
        w.writerows(per_seed)
    (root / "paper_table.json").write_text(json.dumps(
        {f"{ds}/{m}": {"seeds": v["seeds"], **{c: dict(zip(("mean", "std", "n"), v[c])) for c in COLS}}
         for (ds, m), v in table.items()}, indent=2) + "\n")
    print(md)
    print(f"wrote {root}/paper_table.{{md,tex,json}} and paper_table_per_seed.csv")


if __name__ == "__main__":
    main()
