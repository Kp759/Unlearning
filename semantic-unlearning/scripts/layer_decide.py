#!/usr/bin/env python3
"""Automatic layer choice for another model, from the layer smoke test (layer_smoke.slurm).

    python scripts/layer_decide.py MODEL router    # sweep.json -> decision.json: router_layer, write_candidates
    python scripts/layer_decide.py MODEL write     # the write-layer runs -> decision.json: write_layer
    python scripts/layer_decide.py MODEL show      # print decision.json

Rules (all numbers also written to outputs/layer_smoke_<MODEL>/decision.json):
  router  the classifier refit at every block (deployed rule, the router's own eligibility):
          best F1 over blocks; among blocks within ROUTER_F1_TOL (0.005) of it whose FPR and
          null FPR are at most ROUTER_FPR_TOL (0.005) above the best block's, the EARLIEST block.
          With several datasets the F1 is averaged per block first.
  write   candidates = {R, R-3, R-5, the model's default write layer} within 1..R. A candidate is
          kept when, on every dataset, ours keeps locality (Spe >= base Spe - max(1, 5% of base)) and
          perplexity (PPL <= base PPL + 0.1). Among kept candidates the lowest Gen (held-out
          phrasings, averaged over datasets), then the lowest Eff, then the one nearest the default.
          If no candidate keeps locality the lowest Gen is chosen and `write_warning` says so.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))

ROOT = SCRIPTS.parent
DEFAULTS = {"qwen3_4b": (24, 21, 36), "llama31_8b": (22, 18, 32), "llama32_3b": (19, 16, 28)}
ROUTER_F1_TOL = 0.005
ROUTER_FPR_TOL = 0.005


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- router
def sweep_rows(sweep):
    """[(block, f1, fpr, null_fpr, recall)] under the deployed rule, single-block refits."""
    entries = sweep.get("sweep", []) if isinstance(sweep, dict) else sweep
    rows = []
    for e in entries:
        if e.get("rule") != "deployed" or len(e.get("layers", [])) != 1:
            continue
        r = e["by_eligibility"][e.get("deployed_eligibility", "exact")]
        m = r["eval"]
        rows.append((int(e["layers"][0]), m.get("f1"), m.get("fpr"), r.get("null_fpr"), m.get("recall")))
    return sorted(rows)


def pick_router(per_dataset_rows, *, f1_tol=ROUTER_F1_TOL, fpr_tol=ROUTER_FPR_TOL):
    """per_dataset_rows: {dataset: rows from sweep_rows}. Returns (block, detail) or (None, detail)."""
    by_block = {}
    for rows in per_dataset_rows.values():
        for block, f1, fpr, null_fpr, _ in rows:
            if f1 is None:
                continue
            by_block.setdefault(block, []).append((f1, fpr, null_fpr))
    table = {b: {"f1": _mean([x[0] for x in v]), "fpr": _mean([x[1] for x in v]),
                 "null_fpr": _mean([x[2] for x in v])} for b, v in by_block.items()}
    if not table:
        return None, {"reason": "no refit rows"}
    best_block = max(table, key=lambda b: (table[b]["f1"], -(table[b]["fpr"] or 0.0), -b))
    best = table[best_block]
    eligible = [b for b, t in table.items()
                if t["f1"] >= best["f1"] - f1_tol
                and (t["fpr"] or 0.0) <= (best["fpr"] or 0.0) + fpr_tol
                and (t["null_fpr"] or 0.0) <= (best["null_fpr"] or 0.0) + fpr_tol]
    chosen = min(eligible) if eligible else best_block
    return chosen, {"best_f1_block": best_block, "best": best, "eligible_blocks": sorted(eligible),
                    "chosen": chosen, "per_block": {str(b): table[b] for b in sorted(table)},
                    "f1_tol": f1_tol, "fpr_tol": fpr_tol}


def write_candidates(router_layer, default_write):
    cands = {router_layer, router_layer - 3, router_layer - 5, default_write}
    return sorted(c for c in cands if 1 <= c <= router_layer)


# ---------------------------------------------------------------- write
def pick_write(table, *, default_write=None):
    """table: [{dataset, w, method, Eff, Gen, Spe, PPL}] with a base row per dataset (w = None)
    and ours rows per (dataset, w). Returns (w, detail)."""
    base = {r["dataset"]: r for r in table if r["method"] == "base"}
    ours = {}
    for r in table:
        if r["method"] == "ours" and r.get("w") is not None:
            ours.setdefault(int(r["w"]), {})[r["dataset"]] = r
    datasets = sorted(base)
    if not datasets or not ours:
        return None, {"reason": "no base or no ours rows"}
    detail = {"datasets": datasets, "candidates": {}}
    kept = []
    for w, by_ds in sorted(ours.items()):
        if any(ds not in by_ds for ds in datasets):
            detail["candidates"][str(w)] = {"complete": False}
            continue
        checks = {}
        ok = True
        for ds in datasets:
            b, o = base[ds], by_ds[ds]
            spe_ok = o["Spe"] is None or b["Spe"] is None or o["Spe"] >= b["Spe"] - max(1.0, 0.05 * b["Spe"])
            ppl_ok = o["PPL"] is None or b["PPL"] is None or o["PPL"] <= b["PPL"] + 0.1
            checks[ds] = {"Eff": o["Eff"], "Gen": o["Gen"], "Spe": o["Spe"], "base_Spe": b["Spe"],
                          "PPL": o["PPL"], "base_PPL": b["PPL"], "locality_ok": spe_ok, "ppl_ok": ppl_ok}
            ok = ok and spe_ok and ppl_ok
        gen = _mean([by_ds[ds]["Gen"] for ds in datasets])
        eff = _mean([by_ds[ds]["Eff"] for ds in datasets])
        detail["candidates"][str(w)] = {"complete": True, "kept": ok, "Gen": gen, "Eff": eff, "per_dataset": checks}
        if ok:
            kept.append((gen, eff, abs(w - default_write) if default_write else 0, w))
    pool = kept
    if not pool:
        pool = [(c["Gen"], c["Eff"], abs(int(w) - default_write) if default_write else 0, int(w))
                for w, c in detail["candidates"].items() if c.get("complete")]
        detail["write_warning"] = "no candidate kept locality and perplexity; lowest Gen taken"
    if not pool:
        return None, {**detail, "reason": "no complete candidate"}
    pool.sort(key=lambda x: (x[0] if x[0] is not None else 1e9, x[1] if x[1] is not None else 1e9, x[2], x[3]))
    detail["chosen"] = pool[0][3]
    return pool[0][3], detail


# ---------------------------------------------------------------- io
def decision_path(model):
    return ROOT / f"outputs/layer_smoke_{model}/decision.json"


def load_decision(model):
    p = decision_path(model)
    return json.loads(p.read_text()) if p.is_file() else {}


def save_decision(model, d):
    p = decision_path(model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2) + "\n")
    return p


def router_stage(model):
    per_ds = {}
    for sweep_path in sorted(ROOT.glob(f"outputs/layer_smoke_{model}/*/seed*/layer_sweep/sweep.json")):
        per_ds[sweep_path.parents[2].name] = sweep_rows(json.loads(sweep_path.read_text()))
    if not per_ds:
        raise SystemExit(f"no sweep.json under outputs/layer_smoke_{model}/*/seed*/layer_sweep/")
    chosen, detail = pick_router(per_ds)
    if chosen is None:
        raise SystemExit(f"router choice failed: {detail}")
    default_r, default_w, n_layers = DEFAULTS.get(model, (None, None, None))
    d = load_decision(model)
    d.update({"model": model, "router_layer": int(chosen), "router": detail, "router_datasets": sorted(per_ds),
              "default_router_layer": default_r, "default_write_layer": default_w,
              "write_candidates": write_candidates(int(chosen), default_w or int(chosen) - 3)})
    p = save_decision(model, d)
    print(f"router layer: L{chosen} (best F1 at L{detail['best_f1_block']} = {detail['best']['f1']:.3f}; "
          f"eligible {detail['eligible_blocks']}); write candidates {d['write_candidates']} -> {p}")
    return 0


def write_stage(model):
    from summarize_mcf_layer_sweep import collect
    from l16_paper_table import metrics
    d = load_decision(model)
    r = d.get("router_layer")
    if r is None:
        raise SystemExit("decision.json has no router_layer yet (run the router stage first)")
    table = []
    for run in sorted(ROOT.glob(f"outputs/layer_smoke_{model}_r{r}_*/*/seed*/*")):
        tag = run.parents[2].name
        m = re.match(rf"layer_smoke_{re.escape(model)}_r{r}_(base|w(\d+))$", tag)
        if not m or run.name not in ("base", "ours"):
            continue
        if (run.name == "base") != (m.group(1) == "base"):
            continue
        row = collect(run, tag)
        if row.get("status") != "complete":
            continue
        ds = run.parents[1].name
        table.append({"dataset": ds, "w": None if run.name == "base" else int(m.group(2)),
                      "method": run.name, **metrics(ds, row)})
    chosen, detail = pick_write(table, default_write=d.get("default_write_layer"))
    if chosen is None:
        raise SystemExit(f"write-layer choice failed: {detail}")
    d.update({"write_layer": int(chosen), "write": detail, "write_table": table,
              "tag": (f"paper_{model}_onepass_v1" if chosen == r == d.get("default_router_layer")
                      else f"paper_{model}_onepass_r{r}_v1" if chosen == r
                      else f"paper_{model}_r{r}_w{chosen}_ours_v1")})
    p = save_decision(model, d)
    print(f"write layer: L{chosen} under the L{r} router"
          + (f" (WARNING {detail['write_warning']})" if detail.get("write_warning") else "")
          + f"; full run tag {d['tag']} -> {p}")
    for w, c in sorted(detail["candidates"].items(), key=lambda x: int(x[0])):
        if c.get("complete"):
            print(f"  W=L{w}: Gen {c['Gen']:.3f} Eff {c['Eff']:.3f} kept={c['kept']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("stage", choices=("router", "write", "show"))
    a = ap.parse_args(argv)
    if a.stage == "show":
        print(json.dumps({k: v for k, v in load_decision(a.model).items() if k not in ("write_table",)}, indent=2))
        return 0
    return router_stage(a.model) if a.stage == "router" else write_stage(a.model)


if __name__ == "__main__":
    raise SystemExit(main())
