#!/usr/bin/env python3
"""Put already-trained rows behind a different (refit) router. Rows untouched.

    python scripts/swap_router_rows.py --router-dir NEW_ROUTER --rows-from TRAINED_RUN --output-dir OUT

The router fields (heads, bias, cutoff, subject patterns, fit report) come from
NEW_ROUTER; rows and every artifact key the router lacks (dataset metadata,
training report fields) come from TRAINED_RUN, as does the manifest, so the
official evaluator runs on OUT unchanged. Facts must match in order.
A quick preview of a router change: the full pipeline retrains rows under the
new router, which is the number to report.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router-dir", required=True)
    p.add_argument("--rows-from", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(argv)
    router_dir, rows_dir, out = (Path(x).resolve() for x in (a.router_dir, a.rows_from, a.output_dir))
    router = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    trained = torch.load(rows_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    ids = lambda art: [str(f["id"]) for f in art["facts"]]
    if ids(router) != ids(trained):
        raise ValueError("Router and trained run have different facts or fact order")
    if tuple(router["rows"].shape) != tuple(trained["rows"].shape):
        raise ValueError("Row shapes differ (different layer or model?)")
    if int(router["layer"]) != int(trained["layer"]):
        raise ValueError("Router and trained rows are at different layers")
    swapped = dict(router)
    swapped["rows"] = trained["rows"].clone()
    for key, value in trained.items():
        swapped.setdefault(key, value)
    swapped["router_swap"] = {"router_from": str(router_dir), "rows_from": str(rows_dir),
                              "rows_changed": False}
    out.mkdir(parents=True, exist_ok=False)
    torch.save(swapped, out / "fact_association_embeddings.pt")
    shutil.copy2(rows_dir / "association_manifest.json", out / "association_manifest.json")
    print(f"{out}: router from {router_dir.name}, rows from {rows_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
