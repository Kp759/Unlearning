#!/usr/bin/env python3
"""False fires of deployed routers on TEST prompts only, one table for every router dir given.

For each router the deployed decision (its heads, bias, subject gate and, if present, its
relation and combination gates) is replayed on the cached block-19 states of
  * the official evaluation prompts: forget recall, retain false fires
  * the no-association prompts, by kind; the protected-subject / other-relation kind apart
A no-association prompt that the router trained or calibrated on (its linear_router_dataset.json,
or the dataset of the router it was built from) is left out of that router's counts.

    python scripts/router_test_false_fires.py --dataset mcf --seed 1 \
        paper=outputs/mcf_multiseed_ours_v1/seed1/L19/router \
        C=outputs/mcf_multiseed_ours_data_v1/seed1/L19/router
    # -> prints a markdown table; --out writes it
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from linear_router import apply_feature_map, contrast_gate_scores, decide_routes, relation_log_probs


def norm(t):
    return " ".join(str(t or "").casefold().split())


def trained_texts(router_dir):
    """Prompts the router fit or calibrated on (its own dataset, or its source router's)."""
    out = set()
    for d in (router_dir, ):
        p = Path(d) / "linear_router_dataset.json"
        if p.is_file():
            out |= {norm(e["prompt"]) for e in json.loads(p.read_text()) if e.get("split") != "audit"}
    art = torch.load(Path(router_dir) / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    src = (art.get("gated_router") or {}).get("source_router")
    if src and Path(src, "linear_router_dataset.json").is_file():
        out |= {norm(e["prompt"]) for e in json.loads(Path(src, "linear_router_dataset.json").read_text())
                if e.get("split") != "audit"}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("routers", nargs="+", metavar="LABEL=ROUTER_DIR")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cache = Path(args.root) / args.dataset / f"seed{args.seed}"
    man = json.loads((cache / "manifest.json").read_text())
    layer = int(man["router_layer"])
    S = {}
    for name in ("eval", "null"):
        st = torch.load(cache / f"states_{name}.pt", weights_only=False, mmap=True)
        S[name] = {"probes": json.loads((cache / f"probes_{name}.json").read_text()),
                   "h": st["states"][list(st["layers"]).index(layer)].float(),
                   "eligible": torch.load(cache / f"router_{name}.pt", weights_only=False)["eligible"]}
    ev, nu = S["eval"], S["null"]
    should = torch.tensor([p["should_fire"] for p in ev["probes"]])
    null_text = [norm(p.get("prompt")) for p in nu["probes"]]
    null_group = [p["group"] for p in nu["probes"]]
    groups = ["null_generic", "null_ppl_prefix", "null_unseen_subject", "null_protected_other_relation"]

    rows = []
    for spec in args.routers:
        label, rdir = spec.split("=", 1)
        art = torch.load(Path(rdir) / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
        if [f["id"] for f in art["facts"]] != [f["id"] for f in man["facts"]]:
            raise SystemExit(f"{rdir}: other facts than the cache")
        mu, comps = torch.as_tensor(art["feature_mean"]).float(), art.get("feature_components")
        w, b = torch.as_tensor(art["router_weight"]).float(), torch.as_tensor(art["router_bias"]).float()
        thr = float(art.get("threshold", 0.0))
        margin = float(art.get("ambiguity_margin", 0.5))
        if art.get("gate_mode", "threshold") == "subject":
            thr = float("-inf")
        rg, cg = art.get("relation_gate"), art.get("contrast_gate")
        skip = trained_texts(rdir)

        def decide(sec):
            h = sec["h"]
            phi = apply_feature_map(h, mu, comps)
            z = phi @ w.T + b
            allow = sec["eligible"].clone()
            if rg is not None:
                lp = relation_log_probs(h, torch.as_tensor(rg["weight"]), torch.as_tensor(rg["bias"]),
                                        torch.as_tensor(rg["fact_relation"]), rg.get("profile"))
                allow &= lp >= float(rg["min_log_prob"])
            if cg is not None:
                sc = contrast_gate_scores(phi, torch.as_tensor(cg["proj"]), torch.as_tensor(cg["proto"]))
                allow &= (sc >= float(cg["min_score"])) | torch.as_tensor(cg["ungated"]).bool()[None, :]
            d = decide_routes(z, allow, thr, margin)
            return d["active"].cpu(), d["fact"].cpu()

        fire, fact = decide(ev)
        right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"]
                              for k, p in enumerate(ev["probes"])])
        row = {"router": label, "forget_recall": float(right[should].float().mean()),
               "retain_fires": int((fire & ~should).sum()), "retain": int((~should).sum())}
        nf, _ = decide(nu)
        for g in groups:
            idx = [k for k in range(len(nf)) if null_group[k] == g and null_text[k] not in skip]
            row[g] = (int(sum(int(nf[k]) for k in idx)), len(idx))
        row["excluded"] = sum(1 for k in range(len(nf)) if null_text[k] in skip)
        rows.append(row)
        print(json.dumps(row), flush=True)

    head = (f"| router | forget recall (test) | retain false fires (test) | generic text | perplexity text | "
            f"unseen subject | **protected subject, other relation** | excluded (trained on) |")
    lines = [f"### {args.dataset} seed {args.seed}: deployed routers on test prompts only", "", head, "|---" * 8 + "|"]
    for r in rows:
        cells = [f"{r[g][0]}/{r[g][1]}" for g in groups]
        lines.append(f"| {r['router']} | {r['forget_recall']:.3f} | {r['retain_fires']}/{r['retain']} | "
                     + " | ".join(cells[:3]) + f" | **{cells[3]}** | {r['excluded']} |")
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
