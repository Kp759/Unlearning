#!/usr/bin/env python3
"""Copy a trained run with only the router's cutoff changed (rows untouched).

    python scripts/make_router_threshold_variant.py --run-dir RUN --variant -4 --out-dir OUT
    python scripts/make_router_threshold_variant.py --run-dir RUN --variant subject --out-dir OUT
    python scripts/make_router_threshold_variant.py --summarize --root outputs/zsre_threshold_variant_v1

--variant X (number): global cutoff moved by X logits (negative = fires more).
--variant subject:   subject gate: any prompt containing a forget subject fires
                     its best head (no cutoff, no ambiguity margin).
The copy keeps the manifest, so the official evaluator runs on it unchanged.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics as st
from collections import defaultdict
from pathlib import Path

import torch


def make(run_dir, variant, out_dir):
    run_dir, out_dir = Path(run_dir).resolve(), Path(out_dir).resolve()
    artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    if artifact.get("per_head_thresholds") is not None:
        raise ValueError("per-head thresholds: not supported")
    old = float(artifact["threshold"])
    if variant == "subject":
        artifact["gate_mode"] = "subject"
        artifact["threshold"] = float("-inf")
        artifact["ambiguity_margin"] = 0.0
    else:
        artifact["threshold"] = old + float(variant)
    artifact["router_cutoff_variant"] = {
        "variant": str(variant), "source_run": str(run_dir), "source_threshold": old,
        "rows_changed": False,
        "note": "post-hoc cutoff change on a trained run; rows and heads identical",
    }
    out_dir.mkdir(parents=True, exist_ok=False)
    torch.save(artifact, out_dir / "fact_association_embeddings.pt")
    shutil.copy2(run_dir / "association_manifest.json", out_dir / "association_manifest.json")
    print(f"{out_dir}: threshold {old} -> {artifact['threshold']} (gate {artifact.get('gate_mode')})")


def summarize(root):
    rows = defaultdict(list)
    for f in sorted(Path(root).glob("seed*/L??/*/official_zsre_eval.json")):
        d = json.loads(f.read_text())
        layer, variant = f.parent.parent.name, f.parent.name
        rows[(layer, variant)].append({
            "forget_Eff": d["forget"]["Eff"], "forget_Gen": d["forget"]["Gen"],
            "forget_Spe": d["forget"]["Spe"], "retain_Eff": d["retain"]["Eff"],
            "retain_Gen": d["retain"]["Gen"],
            "para_routed": d["forget_route_summary"]["paraphrase"]["route_active_fraction"],
            "neigh_routed": d["forget_route_summary"]["neighborhood"]["route_active_fraction"],
            "retain_routed": d["retain_route_summary"]["paraphrase"]["route_active_fraction"],
            "PPL": (d.get("runtime_aligned_PPL") or {}).get("ppl"),
        })
    keys = ["forget_Eff", "forget_Gen", "forget_Spe", "retain_Eff", "retain_Gen",
            "para_routed", "neigh_routed", "retain_routed", "PPL"]
    order = lambda v: (0, float(v)) if v.lstrip("-").replace(".", "").isdigit() else (1, 0.0)
    print("| layer | variant | n | " + " | ".join(keys) + " |\n|" + "---|" * (len(keys) + 3))
    for (layer, variant) in sorted(rows, key=lambda k: (k[0], order(k[1]))):
        rs = rows[(layer, variant)]
        cell = []
        for k in keys:
            v = [r[k] for r in rs if r[k] is not None]
            cell.append("–" if not v else f"{st.mean(v):.3g} ± {(st.stdev(v) if len(v) > 1 else 0):.2g}")
        print(f"| {layer} | {variant} | {len(rs)} | " + " | ".join(cell) + " |")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir"); p.add_argument("--variant"); p.add_argument("--out-dir")
    p.add_argument("--summarize", action="store_true"); p.add_argument("--root")
    a = p.parse_args(argv)
    if a.summarize:
        summarize(a.root)
    else:
        make(a.run_dir, a.variant, a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
