#!/usr/bin/env python3
"""Test-set view of the classifier diagnostics (router_diag_report.py, router_layer_sweep_cached.py):
confusion on the official evaluation prompts, fires on the no-association prompts, UMAP
separability at two blocks, and errors per block, for every dataset of a diagnostics root.

    python scripts/diag_test_summary.py --root outputs/router_diag_ours_control_v1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--datasets", default="mcf,zsre,mquake,rwku")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    for ds in args.datasets.split(","):
        d = Path(args.root) / ds / f"seed{args.seed}"
        m = json.loads((d / "report" / "metrics.json").read_text())
        print(f"########## {ds} (router block {m['router_layer']})")
        print("  official test prompts, confusion by split:")
        for split, v in m["eval"]["by_split"].items():
            print(f"    {split:<13} n={v.get('n', 0):<6} TP={v.get('tp', 0):<4} FN={v.get('fn', 0):<4} "
                  f"FP={v.get('fp', 0):<5} TN={v.get('tn', 0):<6} FPR={v.get('fpr')} FNR={v.get('fnr')}")
        print("  error reasons:", m.get("reasons"))
        nul = m.get("null") or {}
        print("  no-association prompts:")
        for g, v in (nul.get("by_group") or {}).items():
            print(f"    {g:<34} {v}")
        print("  no-association FPR (must-not-fire):", nul.get("deployment_fpr"))
        sep = m.get("separability") or {}
        for layer, v in (sep.get("layers") or {}).items():
            print(f"  UMAP block {layer}: silhouette(full)={v.get('silhouette_cosine_full_dim'):.3f} "
                  f"silhouette(2d)={v.get('silhouette_2d'):.3f} LOO={v.get('loo')}")
        sw = json.loads((d / "layer_sweep" / "sweep.json").read_text())["sweep"]
        rows = [e for e in sw if e["variant"] == "baseline" and e["rule"] == "deployed"]
        print("  per block (test): FPR / FNR / F1 on official prompts, FPR on no-association prompts")
        for e in sorted(rows, key=lambda e: e["layers"][0]):
            r = e["by_eligibility"][e.get("deployed_eligibility", "exact")]
            ev = r["eval"]
            print(f"    L{e['layers'][0]:<3} FPR={ev['fpr']:.4f} FNR={ev['fnr']:.3f} F1={ev['f1']:.3f} "
                  f"null_FPR={r.get('null_fpr')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
