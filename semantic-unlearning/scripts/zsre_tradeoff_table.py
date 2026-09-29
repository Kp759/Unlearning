#!/usr/bin/env python3
"""ZsRE Gen vs same-subject false fire, from EXISTING checkpoints only (no GPU).

    python scripts/zsre_tradeoff_table.py

Each row is one router at one cutoff, mean ± std over seeds:
  Gen, paraphrases routed   the official ZsRE eval of that checkpoint
  same-subject false fire   from the router comparison's what-if at the same
                            cutoff, on the COMMON same-subject negatives
                            (union of the compared routers' held-out controls)
Sources:
  original router, shipped cutoff   outputs/zsre_multiseed_regular_v1/seed*/L??/linear_global
  original router, cutoff -2/-4/-6, subject gate
                                    outputs/zsre_threshold_variant_v1/seed*/L??/<variant>
  rewording routers, shipped cutoff outputs/zsre_multiseed_reworded_v{1,2}/seed*/L??/linear_global
The false-fire column needs zsre_decomposition.json written with
--negative-sets (zsre_router_compare.slurm); rows without it show "–".
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

VARIANTS = [("0", 0.0), ("-2", -2.0), ("-4", -4.0), ("-6", -6.0), ("subject", "subject_gate")]


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _whatif(decomp, shift):
    if decomp is None:
        return None
    for row in decomp.get("threshold_whatif", []):
        if row["threshold_shift"] == shift:
            return row
    return None


def _fmt(values, scale=1.0, digits=1):
    values = [v * scale for v in values if v is not None]
    if not values:
        return "–"
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{st.mean(values):.{digits}f} ± {sd:.{digits}f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="outputs")
    parser.add_argument("--layers", nargs="+", default=["19", "23"])
    parser.add_argument("--seeds", nargs="+", default=["1", "2", "3", "4", "5"])
    args = parser.parse_args(argv)
    root = Path(args.root)
    rows = defaultdict(lambda: {"gen": [], "routed": [], "ff": [], "ff_own": []})

    for layer in args.layers:
        L = f"L{int(layer):02d}"
        for seed in args.seeds:
            regular = root / "zsre_multiseed_regular_v1" / f"seed{seed}" / L / "linear_global"
            decomp = _load(regular / "zsre_decomposition.json")
            for name, shift in VARIANTS:
                run = regular if name == "0" else root / "zsre_threshold_variant_v1" / f"seed{seed}" / L / name
                official = _load(run / "official_zsre_eval.json")
                if official is None:
                    continue
                label = "original router, " + ("shipped cutoff" if name == "0" else
                                               "subject gate" if name == "subject" else f"cutoff {name}")
                w = _whatif(decomp, shift)
                r = rows[(L, label)]
                r["gen"].append(official["forget"]["Gen"])
                r["routed"].append(official["forget_route_summary"]["paraphrase"]["route_active_fraction"])
                r["ff"].append(None if w is None else w.get("common_same_subject_false_fire"))
                r["ff_own"].append(None if w is None else w.get("same_subject_false_fire"))
            for tag in ("reworded_v1", "reworded_v2"):
                run = root / f"zsre_multiseed_{tag}" / f"seed{seed}" / L / "linear_global"
                official = _load(run / "official_zsre_eval.json")
                if official is None:
                    continue
                w = _whatif(_load(run / "zsre_decomposition.json"), 0.0)
                r = rows[(L, f"rewordings {tag[-2:]}, shipped cutoff")]
                r["gen"].append(official["forget"]["Gen"])
                r["routed"].append(official["forget_route_summary"]["paraphrase"]["route_active_fraction"])
                r["ff"].append(None if w is None else w.get("common_same_subject_false_fire"))
                r["ff_own"].append(None if w is None else w.get("same_subject_false_fire"))

    head = ["layer", "router / cutoff", "n", "Gen ↓", "paraphrases routed %",
            "same-subject false fire % (common negatives)", "(router's own negatives)"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for (L, label), r in sorted(rows.items(), key=lambda kv: (kv[0][0], -st.mean(kv[1]["gen"]))):
        print("| " + " | ".join([L, label, str(len(r["gen"])), _fmt(r["gen"]),
                                 _fmt(r["routed"], 100, 0), _fmt(r["ff"], 100), _fmt(r["ff_own"], 100)]) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
