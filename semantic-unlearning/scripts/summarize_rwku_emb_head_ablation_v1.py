#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path

EXPERIMENT_ID = "rwku-stephen-king-emb-head-ablation-seed0-v1"
VARIANTS = ("emb_head", "emb_head_downproj")


def load(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    rows = []
    for variant in VARIANTS:
        path = args.output_root / EXPERIMENT_ID / variant / "result.json"
        if not path.exists():
            rows.append((variant, "MISSING", "-", "-", "-", "-", "-", "-", "-"))
            continue
        r = load(path)
        proxy = r["frozen_base_head_proxy"]
        util = r["fresh_confirmatory_utility_kl"]
        rows.append((
            variant,
            "PASS" if r["feasible_under_declared_variant_gates"] else "FAIL",
            f"{proxy['recovery_percentage']:.2f}%",
            f"{proxy['minimum_demotion_margin']:.4f}",
            f"{util['utility_kl_mean']:.6f}",
            f"{util['utility_kl_p95']:.6f}",
            f"{util['utility_kl_max']:.6f}",
            f"{r['embedding_delta_from_base']['relative_frobenius']:.6f}",
            f"{r['lm_head_delta_from_stage1']['relative_frobenius']:.6f}",
            f"{r['downproj_delta']['relative_frobenius']:.6f}",
        ))
    headers = ("variant", "status", "W0 recovery", "min margin", "KL mean", "KL p95", "KL max", "emb rel", "head rel", "down rel")
    widths = [len(x) for x in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    def fmt(row):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row))
    print("\nRWKU embedding/head ablation comparison")
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


if __name__ == "__main__":
    main()
