#!/usr/bin/env python3
"""Plain logistic regression (p >= 0.5, stage-1 bias) vs calibrated cutoff folded into the bias.

    python scripts/compare_bias_rules.py --dataset mcf --optimizer lbfgs \
        --folded-router R --folded-run F --raw-router RR --raw-swap-run S --raw-run RAW \
        --out-prefix OUT/comparison
    python scripts/compare_bias_rules.py --collect outputs/bias_rule_ablation_v1   # all 6 runs

folded    the shipped rule: b' = b - t, t calibrated on held-out prompts
raw_swap  folded rows behind the raw router (eval-time effect of the rule only)
raw       rows retrained under the raw router (the full one-stage method)

Sections: the cutoff t that raw drops; both rules on every router prompt
(same features: rescued positives, added false fires, logit quantiles);
training views the row trainer could use; official metrics with the better
rule per metric (direction from the evaluators' metric definitions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_router_optimizers import METRICS, _fmt, _load, _num, views_diff  # noqa: E402
from summarize_mcf_layer_sweep import collect  # noqa: E402

LOWER_IS_BETTER = {"forget_Eff", "forget_Gen", "forget_AtomicGen", "PPL",
                   "forget_neighborhood_route_active", "retain_atomicgen_route_active"}
HIGHER_IS_BETTER = {"forget_Spe", "retain_Eff", "retain_Gen", "retain_AtomicGen",
                    "forget_rewrite_route_active", "forget_paraphrase_route_active",
                    "forget_rewrite_route_correct", "forget_atomicgen_route_correct",
                    "facts_trained", "display_zero"}
EXACT = 1e-12
HEADLINE = {
    "mcf": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL"],
    "zsre": ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen", "PPL"],
    "mquake": ["forget_Eff", "forget_AtomicGen", "retain_Eff", "retain_AtomicGen", "PPL"],
}


def better(metric, folded, raw):
    a, b = _num(folded), _num(raw)
    if a is None or b is None:
        return None
    if abs(a - b) <= EXACT:
        return "tie"
    if metric in LOWER_IS_BETTER:
        return "raw" if b < a else "folded"
    if metric in HIGHER_IS_BETTER:
        return "raw" if b > a else "folded"
    return None


def router_section(raw_router):
    ablation = _load(Path(raw_router) / "bias_rule_ablation.json")
    if ablation is None:
        return None
    out = {
        "folded_cutoff_t": ablation.get("folded_cutoff_t"),
        "folded_recompute_mismatches": ablation.get("folded_routes_recomputed_vs_stored_mismatches"),
        "raw_runtime_mismatches": (ablation.get("runtime_parity_raw") or {}).get("route_mismatches"),
        "by_split": {},
    }
    for split, s in (ablation.get("by_split") or {}).items():
        rate = lambda block, key: ((block or {}).get(key) or {}).get("rate")  # noqa: E731
        out["by_split"][split] = {
            "positives": s["positives"],
            "correct_route": {"folded": rate(s["folded"], "correct_route"),
                              "raw": rate(s["raw"], "correct_route")},
            "abstain_on_positive": {"folded": rate(s["folded"], "abstain_on_positive"),
                                    "raw": rate(s["raw"], "abstain_on_positive")},
            "negatives": s["negatives"],
            "false_activation": {"folded": rate(s["folded"], "false_activation_on_negative_control"),
                                 "raw": rate(s["raw"], "false_activation_on_negative_control")},
            "positives_rescued_by_calibration": s["positives_rescued_by_calibration"],
            "positives_lost_by_calibration": s["positives_lost_by_calibration"],
            "negatives_added_by_calibration": s["negatives_added_by_calibration"],
            "positive_own_logit_below_zero": s["positive_own_logit_below_zero"],
            "positive_own_logit_in_cutoff_to_zero": s["positive_own_logit_in_cutoff_to_zero"],
            "positive_own_stage1_logit": s.get("positive_own_stage1_logit"),
            "negative_best_eligible_stage1_logit": s.get("negative_best_eligible_stage1_logit"),
        }
    return out


def metrics_table(dataset, runs):
    rows = {label: (collect(path, label) if path and Path(path).is_dir() else None)
            for label, path in runs.items()}
    table = []
    for metric in METRICS[dataset]:
        folded = (rows.get("folded") or {}).get(metric)
        entry = {"metric": metric, "direction": ("lower" if metric in LOWER_IS_BETTER else
                                                 "higher" if metric in HIGHER_IS_BETTER else "–")}
        for label, row in rows.items():
            value = None if row is None else row.get(metric)
            entry[label] = value
            if label != "folded" and _num(value) is not None and _num(folded) is not None:
                entry[f"delta[{label}]"] = _num(value) - _num(folded)
                entry[f"better[{label}]"] = better(metric, folded, value)
        table.append(entry)
    return table, {k: (None if r is None else r.get("status")) for k, r in rows.items()}


def overall(table, router):
    lines = []
    if router and router.get("folded_cutoff_t") is not None:
        t = router["folded_cutoff_t"]
        audit = router["by_split"].get("audit") or {}
        if audit:
            lines.append(
                f"router audit: correct route folded {_fmt(audit['correct_route']['folded'])} vs raw "
                f"{_fmt(audit['correct_route']['raw'])}; false activation folded "
                f"{_fmt(audit['false_activation']['folded'])} vs raw {_fmt(audit['false_activation']['raw'])} "
                f"(cutoff t = {_fmt(t) if isinstance(t, float) else 'per head'})")
    wins = {"folded": [], "raw": [], "tie": []}
    for row in table:
        verdict = row.get("better[raw]")
        if verdict in wins:
            wins[verdict].append(row["metric"])
    if any(wins.values()):
        lines.append("full method (rows retrained): folded better on "
                     + (", ".join(wins["folded"]) or "none") + "; raw better on "
                     + (", ".join(wins["raw"]) or "none")
                     + (f"; tie on {', '.join(wins['tie'])}" if wins["tie"] else ""))
    else:
        lines.append("full method (rows retrained): not evaluated yet")
    return lines


def markdown(result):
    r = result
    lines = [f"# Bias rule: plain logistic (p ≥ 0.5) vs calibrated cutoff folded into the bias — "
             f"{r['dataset']}, {r['optimizer']}, seed {r['seed']}, L{r['layer']}", ""]
    lines += [f"- **{line}**" for line in r["overall"]] + [""]
    router = r["router"]
    if router:
        lines += ["## Both rules on the router's prompts (same heads, same features)", "",
                  f"Cutoff dropped by the raw rule: t = {_fmt(router['folded_cutoff_t'])}; "
                  f"raw runtime-parity mismatches {_fmt(router['raw_runtime_mismatches'])}; "
                  f"folded recompute vs stored route mismatches {_fmt(router['folded_recompute_mismatches'])}.",
                  "",
                  "| split | positives | correct (folded / raw) | rescued by calibration | "
                  "own logit < 0 | own logit in [t, 0) | negatives | false act. (folded / raw) | "
                  "false fires added by calibration |", "|---|---|---|---|---|---|---|---|---|"]
        for split, s in router["by_split"].items():
            lines.append("| " + " | ".join(_fmt(x) for x in (
                split, s["positives"],
                f"{_fmt(s['correct_route']['folded'])} / {_fmt(s['correct_route']['raw'])}",
                s["positives_rescued_by_calibration"], s["positive_own_logit_below_zero"],
                s["positive_own_logit_in_cutoff_to_zero"], s["negatives"],
                f"{_fmt(s['false_activation']['folded'])} / {_fmt(s['false_activation']['raw'])}",
                s["negatives_added_by_calibration"])) + " |")
        lines.append("")
    views = r["training_views"]
    if views and "error" not in views:
        ev = views["excluded_views"]
        lines += ["## Training views", "",
                  f"Views the row trainer could not use (not routed to their own row): folded "
                  f"{ev['reference']} → raw {ev['other']}; untrainable facts "
                  f"{len(views['untrainable_facts']['reference'])} → {len(views['untrainable_facts']['other'])}.",
                  ""]
    labels = [k for k in r["runs"] if k != "folded"]
    lines += ["## Official metrics", "",
              "| metric | better | folded | " + " | ".join(labels) + " | "
              + " | ".join(f"better ({k})" for k in labels) + " |",
              "|" + "---|" * (3 + 2 * len(labels))]
    for row in r["metrics"]:
        lines.append("| " + " | ".join(
            [row["metric"], row["direction"], _fmt(row.get("folded"))]
            + [_fmt(row.get(k)) for k in labels]
            + [_fmt(row.get(f"better[{k}]")) for k in labels]) + " |")
    lines += ["", "Run status: " + ", ".join(f"{k}={v}" for k, v in r["run_status"].items()), "",
              "`raw_swap` = folded rows behind the raw router (rule effect at eval only); "
              "`raw` = rows retrained under the raw router (the full one-stage method)."]
    return "\n".join(lines) + "\n"


def collect_all(root):
    root = Path(root)
    found = [json.loads(p.read_text()) for p in sorted(root.glob("*/*/seed*/L*/comparison.json"))]
    lines = ["# Bias rule ablation: plain logistic (p ≥ 0.5) vs folded calibrated bias", ""]
    for dataset in ("mcf", "zsre", "mquake"):
        results = [r for r in found if r["dataset"] == dataset]
        if not results:
            continue
        keys = HEADLINE[dataset]
        lines += [f"## {dataset}", "",
                  "| optimizer | rule | cutoff t | audit correct | audit false act. | "
                  + " | ".join(keys) + " |", "|" + "---|" * (5 + len(keys))]
        for result in sorted(results, key=lambda x: x["optimizer"]):
            router = result.get("router") or {}
            audit = (router.get("by_split") or {}).get("audit") or {}
            metrics = {row["metric"]: row for row in result["metrics"]}
            for arm, rule in (("folded", "folded (shipped)"), ("raw_swap", "raw, rows swapped"),
                              ("raw", "raw, rows retrained")):
                side = "folded" if arm == "folded" else "raw"
                lines.append("| " + " | ".join(_fmt(x) for x in [
                    result["optimizer"], rule,
                    router.get("folded_cutoff_t") if arm == "folded" else 0.0,
                    (audit.get("correct_route") or {}).get(side),
                    (audit.get("false_activation") or {}).get(side),
                ] + [(metrics.get(k) or {}).get(arm) for k in keys]) + " |")
        lines += [""] + [f"- {r['optimizer']}: " + "; ".join(r["overall"]) for r in results] + [""]
    lines.append("Directions: forget Eff/Gen/AtomicGen and PPL lower is better; Spe and retain higher.")
    text = "\n".join(lines) + "\n"
    (root / "bias_rule_summary.md").write_text(text)
    print(text)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--collect"]:
        if len(argv) != 2:
            raise SystemExit("usage: compare_bias_rules.py --collect <ablation root>")
        return collect_all(argv[1])
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=sorted(METRICS))
    p.add_argument("--optimizer", required=True)
    p.add_argument("--seed", default="1")
    p.add_argument("--layer", default="19")
    p.add_argument("--folded-router", required=True)
    p.add_argument("--folded-run", required=True)
    p.add_argument("--raw-router", required=True)
    p.add_argument("--raw-swap-run", default=None)
    p.add_argument("--raw-run", default=None)
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args(argv)

    runs = {"folded": a.folded_run, "raw_swap": a.raw_swap_run, "raw": a.raw_run}
    result = {
        "schema_version": "bias_rule_comparison_v1",
        "dataset": a.dataset, "optimizer": a.optimizer, "seed": a.seed, "layer": a.layer,
        "inputs": {"folded_router": a.folded_router, "raw_router": a.raw_router, **runs},
        "router": router_section(a.raw_router),
        "training_views": (views_diff(a.folded_run, a.raw_run) if a.raw_run else None),
        "runs": runs,
    }
    result["metrics"], result["run_status"] = metrics_table(a.dataset, runs)
    result["overall"] = overall(result["metrics"], result["router"])
    prefix = Path(a.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix(".json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    text = markdown(result)
    prefix.with_suffix(".md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
