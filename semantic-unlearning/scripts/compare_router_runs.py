#!/usr/bin/env python3
"""Router comparison across runs of run_l16_paper_one.sh (same write layer, trainer and
evaluators; only the router differs). CPU, seconds.

    python scripts/compare_router_runs.py                       # main vs both hybrid gates, seed 1
    python scripts/compare_router_runs.py --seeds 1 2 3 4 5 \\
        --run main=outputs/l16_paper_v1 --run learned=outputs/l16_paper_hybrid_learned_v1

Per dataset: Base, then Ours and Genie of every run (genie = that run's rows under perfect
routing, so Ours - Genie is what that router costs). Means over the seeds every run has.
Diagnostics per run: router used, hybrid subject-head calibration (threshold, validation
recall / false fire vs the exact match on the same prompts), learned-vs-exact gate counts at
evaluation, final on-policy leak, facts the router never routed in training.
Writes <first run>/router_comparison_<names>.{md,json}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from l16_paper_table import DATASETS, NAMES, metrics, mquake_overlap, mquake_spe_no_overlap, stats
from summarize_mcf_layer_sweep import collect

COLS = ("Eff", "Gen", "Spe", "Spe_no_overlap", "PPL")
DEFAULT_RUNS = ("main=outputs/l16_paper_v1", "learned=outputs/l16_paper_hybrid_learned_v1",
                "learned_or_exact=outputs/l16_paper_hybrid_or_exact_v1")


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.is_file() else None


def find_key(obj, key):
    """First value stored under `key` anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


def run_metrics(ds, seed_dir, method):
    row = collect(seed_dir / method, f"{ds}/{seed_dir.name}/{method}")
    if row.get("status") != "complete":
        return None
    m = metrics(ds, row)
    m["Spe_no_overlap"] = (mquake_spe_no_overlap(seed_dir / method, mquake_overlap(seed_dir / "genie"))
                           if ds == "mquake" else None)
    return m


def diagnostics(seed_dir):
    choice = _load(seed_dir / "router_choice.json") or {}
    report = _load(seed_dir / "ours" / "training_report.json") or {}
    rounds = report.get("rounds") or []
    out = {"router": choice.get("router"),
           "onpolicy_leak_final": rounds[-1].get("onpolicy_leak") if rounds else None,
           "facts_oracle_trained": len(report.get("facts_trained_with_oracle_routing") or [])}
    if str(choice.get("router", "")).startswith("hybrid_"):
        summary = _load(Path(choice["dir"]).parent / "summary.json") or {}
        L = summary.get("selected_subject_layer")
        layer = (summary.get("layers") or {}).get(str(L)) or (summary.get("layers") or {}).get(L) or {}
        cal = layer.get("calibration") or {}
        out["subject_head"] = {"layer": L, "threshold": cal.get("threshold"), "status": cal.get("status"),
                               "val_recall": cal.get("recall"), "val_false_fire_other": cal.get("false_fire_other"),
                               "val_false_fire_near_miss": cal.get("false_fire_near_miss"),
                               "validation_by_kind": layer.get("validation_by_kind")}
        for name in ("official_mcf_eval.json", "official_zsre_eval.json", "official_mquake_eval.json",
                     "official_rwku_eval.json"):
            ev = _load(seed_dir / "ours" / name)
            if ev is not None:
                out["eval_gate_counts"] = find_key(ev, "learned_subject_gate")
                break
    return out


def compare(runs, seeds, datasets):
    res = {"runs": runs, "datasets": {}}
    for ds in datasets:
        common = [f"seed{s}" for s in seeds
                  if all((Path(root) / ds / f"seed{s}").is_dir() for root in runs.values())]
        entry = {"seeds": common, "rows": {}, "per_seed": {}, "diagnostics": {}}
        first = Path(next(iter(runs.values())))
        methods = ("ours", "genie", "genie_entity") if ds == "rwku" else ("ours", "genie")
        arms = [("base", first, "base")] + [(f"{name}:{m}", Path(root), m)
                                             for name, root in runs.items() for m in methods]
        for label, root, method in arms:
            vals = [run_metrics(ds, root / ds / s, method) for s in common]
            entry["per_seed"][label] = dict(zip(common, vals))
            entry["rows"][label] = {c: stats([v[c] for v in vals if v and v.get(c) is not None]) for c in COLS}
        for name, root in runs.items():
            entry["diagnostics"][name] = {s: diagnostics(Path(root) / ds / s) for s in common}
        res["datasets"][ds] = entry
    return res


def _cell(st, d):
    m, s, n = st
    if m is None:
        return "–"
    return f"{m:.{d}f}" if n <= 1 else f"{m:.{d}f} ± {s:.{d}f}"


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def render(res, digits=3):
    lines = ["# Router comparison (write L16; same trainer and evaluators)", "",
             "Runs: " + ", ".join(f"`{k}` = {v}" for k, v in res["runs"].items()), ""]
    for ds, e in res["datasets"].items():
        if not e["seeds"]:
            lines += [f"## {NAMES[ds]}: no seed present in every run", ""]
            continue
        lines += [f"## {NAMES[ds]} ({', '.join(e['seeds'])})", "",
                  "| Method | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ | PPL |", "|---|---|---|---|---|---|"]
        for label, row in e["rows"].items():
            lines.append(f"| {label} | " + " | ".join(_cell(row[c], digits) for c in COLS) + " |")
        lines += ["", "| run | seed | router | subject head: layer / threshold / status | val recall | "
                  "val false fire (other / near-miss) | eval gate pairs (exact / learned / used) | "
                  "on-policy leak | oracle-trained facts |", "|---|---|---|---|---|---|---|---|---|"]
        for name, by_seed in e["diagnostics"].items():
            for s, g in by_seed.items():
                sh = g.get("subject_head") or {}
                gc = g.get("eval_gate_counts") or {}
                head = ("–" if not sh else
                        f"L{sh.get('layer')} / {sh.get('threshold') if sh.get('threshold') is None else round(sh['threshold'], 4)} / {sh.get('status')}")
                lines.append(
                    f"| {name} | {s} | {g.get('router') or '–'} | {head} | {_pct(sh.get('val_recall'))} | "
                    f"{_pct(sh.get('val_false_fire_other'))} / {_pct(sh.get('val_false_fire_near_miss'))} | "
                    + (" / ".join(str(gc.get(k, '–')) for k in ('exact', 'learned', 'used')) if gc else "–")
                    + f" | {_pct(g.get('onpolicy_leak_final'))} | {g.get('facts_oracle_trained', '–')} |")
        lines.append("")
    lines += ["Spe† (MQuAKE): retain Eff on retain facts that are not also forget facts. "
              "Subject-head validation uses training-visible prompts only (development families, "
              "unseen swap / near-miss names); `validation_by_kind` in the json compares the "
              "learned gate with the exact match on the same prompts."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=None, help="NAME=ROOT (repeat); default: main + both hybrids")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    ap.add_argument("--digits", type=int, default=3)
    a = ap.parse_args()
    runs = dict(r.split("=", 1) for r in (a.run or DEFAULT_RUNS))
    res = compare(runs, a.seeds, a.datasets)
    md = render(res, a.digits)
    first = Path(next(iter(runs.values())))
    tag = "_vs_".join(runs)
    first.mkdir(parents=True, exist_ok=True)
    (first / f"router_comparison_{tag}.md").write_text(md)
    (first / f"router_comparison_{tag}.json").write_text(json.dumps(res, indent=2, default=list) + "\n")
    print(md)
    print(f"wrote {first}/router_comparison_{tag}.{{md,json}}")


if __name__ == "__main__":
    main()
