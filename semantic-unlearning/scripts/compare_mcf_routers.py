#!/usr/bin/env python3
"""MCF router ablation: ours/genie trained and evaluated under the factorized v2.2 router
(outputs/l16_paper_v1) vs the linear L19 router (outputs/l16_paper_mcf_linear19), same seeds,
same write layer (16), same trainer and evaluator. CPU, seconds.

    python scripts/compare_mcf_routers.py
    python scripts/compare_mcf_routers.py --v22-root outputs/l16_paper_v1 \\
        --linear-root outputs/l16_paper_mcf_linear19 --digits 3

Rows: Base, Ours (v2.2), Ours (linear L19), Genie for each set of rows (oracle routing: the
gap Ours - Genie is what routing costs). Paired: linear - v2.2 per seed (Eff/Gen > 0 = linear
forgets less; Spe < 0 = linear keeps less locality). Diagnostics check that each seed really
ran the router it claims (v2.2 can fall back to L19 when not deployable) and how many facts
each router never routed during training (trained with --oracle-unrouted).
Writes <linear-root>/mcf/router_comparison.{md,json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from l16_paper_table import metrics, stats
from summarize_mcf_layer_sweep import collect

COLS = ("Eff", "Gen", "Spe", "PPL")


def run_metrics(run_dir):
    row = collect(run_dir, str(run_dir))
    return metrics("mcf", row) if row.get("status") == "complete" else None


def diagnostics(seed_dir):
    choice = seed_dir / "router_choice.json"
    rep = seed_dir / "ours" / "training_report.json"
    report = json.loads(rep.read_text()) if rep.is_file() else {}
    rounds = report.get("rounds") or []
    return {"router": json.loads(choice.read_text()).get("router") if choice.is_file() else None,
            "onpolicy_leak_final": rounds[-1].get("onpolicy_leak") if rounds else None,
            "facts_routed_by_router": report.get("facts_routed_by_router"),
            "facts_oracle_trained": len(report.get("facts_trained_with_oracle_routing") or [])}


def compare(v22_root, linear_root):
    seeds = sorted({p.name for p in (v22_root / "mcf").glob("seed*")}
                   & {p.name for p in (linear_root / "mcf").glob("seed*")})
    arms = {"base": (v22_root, "base"), "ours_v22": (v22_root, "ours"),
            "ours_linear": (linear_root, "ours"), "genie_v22": (v22_root, "genie"),
            "genie_linear": (linear_root, "genie")}
    per_seed, diag = {}, {}
    for s in seeds:
        per_seed[s] = {arm: run_metrics(root / "mcf" / s / m) for arm, (root, m) in arms.items()}
        diag[s] = {"v22": diagnostics(v22_root / "mcf" / s),
                   "linear": diagnostics(linear_root / "mcf" / s)}
    summary = {arm: {c: stats([per_seed[s][arm][c] for s in seeds if per_seed[s][arm]]) for c in COLS}
               for arm in arms}
    paired_seeds = [s for s in seeds if per_seed[s]["ours_v22"] and per_seed[s]["ours_linear"]]
    paired = {c: stats([per_seed[s]["ours_linear"][c] - per_seed[s]["ours_v22"][c] for s in paired_seeds
                        if per_seed[s]["ours_linear"][c] is not None
                        and per_seed[s]["ours_v22"][c] is not None]) for c in COLS}
    return {"seeds": seeds, "paired_seeds": paired_seeds, "per_seed": per_seed,
            "summary": summary, "paired_linear_minus_v22": paired, "diagnostics": diag}


def render(res, digits):
    d = digits

    def cell(st):
        m, s, n = st
        return "–" if m is None else f"{m:.{d}f} ± {s:.{d}f} (n={n})"

    def num(x):
        return "–" if x is None else f"{x:.{d}f}"

    labels = {"base": "Base", "ours_v22": "Ours, v2.2 router", "ours_linear": "Ours, linear L19 router",
              "genie_v22": "Genie (v2.2-trained rows)", "genie_linear": "Genie (L19-trained rows)"}
    out = ["# MCF router ablation: factorized v2.2 vs linear L19 (write L16, seeds "
           + ", ".join(s.replace("seed", "") for s in res["seeds"]) + ")", "",
           "| Method | Eff ↓ | Gen ↓ | Spe ↑ | PPL |", "|---|---|---|---|---|"]
    for arm, lab in labels.items():
        out.append(f"| {lab} | " + " | ".join(cell(res["summary"][arm][c]) for c in COLS) + " |")
    out += ["", "## Paired difference, linear L19 − v2.2 (ours)", "",
            "Eff/Gen > 0: linear forgets less. Spe < 0: linear keeps less locality.", "",
            "| | Eff | Gen | Spe | PPL |", "|---|---|---|---|---|",
            "| mean ± std | " + " | ".join(cell(res["paired_linear_minus_v22"][c]) for c in COLS) + " |",
            "", "## Per seed (ours)", "",
            "| seed | router | Eff | Gen | Spe | PPL | on-policy leak | routed / oracle-trained facts |",
            "|---|---|---|---|---|---|---|---|"]
    for s in res["seeds"]:
        for key, arm in (("v22", "ours_v22"), ("linear", "ours_linear")):
            m = res["per_seed"][s][arm] or {}
            g = res["diagnostics"][s][key]
            leak = g["onpolicy_leak_final"]
            out.append(f"| {s} | {g['router'] or '–'} | " + " | ".join(num(m.get(c)) for c in COLS)
                       + f" | {'–' if leak is None else f'{100 * leak:.1f}%'} | "
                       f"{g['facts_routed_by_router'] if g['facts_routed_by_router'] is not None else '–'}"
                       f" / {g['facts_oracle_trained']} |")
    fallback = [s for s in res["seeds"] if res["diagnostics"][s]["v22"]["router"] not in (None, "factorized_v2.2")]
    if fallback:
        out += ["", f"!! v2.2 arm fell back to the L19 router on {', '.join(fallback)}: "
                "those seeds compare L19 with itself."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v22-root", default="outputs/l16_paper_v1")
    ap.add_argument("--linear-root", default="outputs/l16_paper_mcf_linear19")
    ap.add_argument("--digits", type=int, default=3)
    a = ap.parse_args()
    res = compare(Path(a.v22_root), Path(a.linear_root))
    if not res["seeds"]:
        raise SystemExit(f"no common MCF seeds under {a.v22_root} and {a.linear_root}")
    md = render(res, a.digits)
    out = Path(a.linear_root) / "mcf"
    (out / "router_comparison.md").write_text(md)
    (out / "router_comparison.json").write_text(json.dumps(res, indent=2, default=list) + "\n")
    print(md)
    print(f"wrote {out}/router_comparison.{{md,json}}")


if __name__ == "__main__":
    main()
