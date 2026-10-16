#!/usr/bin/env python3
"""Fit the factorized subject x relation router (scripts/factorized_router.py), MCF.

For each requested layer: one forward pass over all router prompts (cached features at
every layer at once), then two class-balanced softmax regressions (subject head: forget
subjects + none; relation head: 34 MCF relations), L2 picked on validation log loss,
then (t_s, t_r) calibrated on validation (fewest false fires subject to macro recall
>= --min-recall). Writes one router dir per layer (loads in every evaluator via
linear_router.load_router_artifact) and picks the best layer on validation.

    python -u scripts/fit_factorized_router.py \\
        --reference-dir outputs/mcf_multiseed_regular_v1/seed1/L19/router \\
        --output-dir outputs/mcf_factorized_router_seed1 --layers 19 23 27 --local-files-only
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorized_router import (  # noqa: E402
    ARCHITECTURE,
    FactorizedRouter,
    balanced_log_loss,
    build_router_data,
    calibrate,
    extract_last_token,
    feature_map,
    fit_softmax,
)


def head_accuracy(X, y, W, b):
    return float(((X @ W.T + b).argmax(-1) == y).float().mean()) if len(y) else None


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", required=True,
                   help="any MCF router/prep dir of this seed (manifest, facts, subject patterns)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[19, 23, 27])
    p.add_argument("--l2-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--min-recall", type=float, default=0.95)
    p.add_argument("--per-relation", type=int, default=800)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prepare_mcf_association_source import load_mcf_forget_data

    ref = Path(args.reference_dir)
    reference = torch.load(ref / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((ref / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)

    records = json.loads(Path(manifest["mcf_path"]).read_text())
    forget_records, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    if [str(f["id"]) for f in facts] != [str(f["id"]) for f in reference["facts"]]:
        raise ValueError("rebuilt MCF facts do not match the reference artifact")
    examples = [e if isinstance(e, dict) else {k: getattr(e, k) for k in ("fact_id", "prompt", "split", "group")
                                               if hasattr(e, k)} for e in examples]
    data = build_router_data(records, forget_records, facts, examples, seed=seed,
                             per_relation=args.per_relation, augment=not args.no_augment)
    items = data["items"]
    S, R = len(data["subjects"]), len(data["relations"])
    print(json.dumps({"phase": "data", "items": len(items), "subjects": S, "relations": R,
                      "by_split_kind": {f"{k[0]}/{k[1]}": v for k, v in Counter(
                          (i["split"], i["kind"].split("+")[0]) for i in items).items()}}), flush=True)

    t0 = time.time()
    feats = extract_last_token(model, tok, [i["prompt"] for i in items], args.layers,
                               batch_size=args.batch_size, device=args.device)
    print(json.dumps({"phase": "features", "seconds": round(time.time() - t0, 1)}), flush=True)

    train = torch.tensor([i["split"] == "train" for i in items])
    val = ~train
    ys = torch.tensor([i["subject"] for i in items])
    yr = torch.tensor([data["relations"].index(i["relation"]) for i in items])
    target = torch.tensor([i["target"] for i in items])
    kinds = [i["kind"] for i in items]
    fact_subject = torch.tensor(data["fact_subject"])
    fact_relation = torch.tensor(data["fact_relation"])
    reports = {}
    for layer in args.layers:
        h = feats[layer]
        mean = F.normalize(h[train].float(), dim=-1).mean(0)
        X = feature_map(h, mean)
        heads = {}
        for name, y, C in (("subject", ys, S + 1), ("relation", yr, R)):
            best = None
            for l2 in args.l2_grid:
                W, b = fit_softmax(X[train], y[train], C, l2, device=args.device)
                loss = balanced_log_loss(X[val], y[val], W, b, C)
                if best is None or loss < best[0]:
                    best = (loss, l2, W, b)
            heads[name] = {"W": best[2], "b": best[3], "l2": best[1], "val_balanced_log_loss": best[0],
                           "train_acc": head_accuracy(X[train], y[train], best[2], best[3]),
                           "val_acc": head_accuracy(X[val], y[val], best[2], best[3])}
        ps = F.softmax(X[val] @ heads["subject"]["W"].T + heads["subject"]["b"], -1)
        pr = F.softmax(X[val] @ heads["relation"]["W"].T + heads["relation"]["b"], -1)
        val_kinds = [k for k, v in zip(kinds, val.tolist()) if v]
        cal = calibrate(ps, pr, target[val], fact_subject, fact_relation, val_kinds,
                        min_recall=args.min_recall)
        state = {"layer": layer, "feature_mean": mean,
                 "subject_weight": heads["subject"]["W"], "subject_bias": heads["subject"]["b"],
                 "relation_weight": heads["relation"]["W"], "relation_bias": heads["relation"]["b"],
                 "fact_subject": fact_subject, "fact_relation": fact_relation,
                 "subjects": data["subjects"], "relations": data["relations"],
                 "t_subject": cal["t_subject"], "t_relation": cal["t_relation"],
                 "calibration": cal, "min_recall": args.min_recall}
        router = FactorizedRouter(state)
        # per-kind validation fire rates at the calibrated thresholds
        d = router.decide(h[val])
        tv = target[val]
        per_kind = {}
        for k in sorted(set(val_kinds)):
            m = torch.tensor([x == k for x in val_kinds])
            pos = m & (tv >= 0)
            neg = m & (tv < 0)
            per_kind[k] = {"n": int(m.sum()),
                           "recall": float(((d == tv) & pos).float().sum() / pos.sum()) if pos.any() else None,
                           "fire": float((d[neg] >= 0).float().mean()) if neg.any() else None}
        report = {"layer": layer, "calibration": cal,
                  "subject_head": {k: v for k, v in heads["subject"].items() if k not in ("W", "b")},
                  "relation_head": {k: v for k, v in heads["relation"].items() if k not in ("W", "b")},
                  "validation_by_kind": per_kind}
        reports[layer] = report
        print(json.dumps({"phase": "layer_done", **{k: report[k] for k in ("layer", "calibration")},
                          "subject_val_acc": report["subject_head"]["val_acc"],
                          "relation_val_acc": report["relation_head"]["val_acc"]}), flush=True)

        layer_dir = out / f"L{layer:02d}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        art = {k: v for k, v in reference.items()
               if k in ("facts", "subject_patterns", "model_path", "plan", "sampling")}
        art.update({"architecture": ARCHITECTURE, "layer": layer,
                    "rows": torch.zeros_like(reference["rows"]),
                    "factorized_router": {k: (v.clone() if torch.is_tensor(v) else v) for k, v in state.items()}})
        torch.save(art, layer_dir / "fact_association_embeddings.pt")
        m = dict(manifest)
        m.update({"architecture": ARCHITECTURE, "router_layer": layer, "reference_dir": str(ref),
                  "method": "factorized_subject_relation_router"})
        (layer_dir / "association_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
        (layer_dir / "factorized_router_report.json").write_text(json.dumps(report, indent=2) + "\n")

    def score(r):
        c = r["calibration"]
        return (c["recall_floor_met"],
                -(c["false_fire_same_subject"] + c["false_fire_other"]) if c["recall_floor_met"] else c["recall"])

    best = max(reports, key=lambda L: score(reports[L]))
    summary = {"selected_layer": best, "selection": "validation: recall floor met, then fewest false fires",
               "data": {k: v for k, v in data.items() if k not in ("items",)},
               "layers": reports, "command": sys.argv}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"phase": "done", "selected_layer": best}), flush=True)


if __name__ == "__main__":
    main()
