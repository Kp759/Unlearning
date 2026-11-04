#!/usr/bin/env python3
"""Gated router: the paper's router + label-free relation gate + positive/negative combination
gate + per-head bias (one setting for every dataset). Writes a new router dir; the paper's router
is not touched.

From the router_diag cache of the shipped router (outputs/router_diag_ours_control_v1/<ds>/seed<S>):
  relation gate  probe outputs/relation_gate_analysis_v1/universal_probe.pt; per fact a relation
                 profile = mean probe distribution over its fit-split positives; head i may fire if
                 log sum_r p(r | prompt) profile_i(r) >= the --probe-q quantile of that score on the
                 fit-split positives
  combination    shared projection (--dim) + prototype per fact, trained on the fit split (positives
                 against same-subject negatives); head i may fire if cos >= the --contrast-q quantile
                 on the calibration positives (never trained on)
  per-head bias  each head's bias lowered until its highest clean calibration negative that passes
                 the subject gate and both gates scores just below 0
Negatives that contain the fact's answer or ask an overlapping relation are not negatives
(router_negative_clean_gate.py). Before writing, the runtime decision (subject gate, both gates,
new biases) is replayed on the cached evaluation prompts and reported.

    python -u scripts/build_gated_router.py --datasets mcf,zsre,mquake,rwku --seed 1
    # -> outputs/<ds>_multiseed_ours_gate_v1/seed<S>/L19/router
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F

from linear_router import apply_feature_map, contrast_gate_scores, decide_routes, relation_log_probs
from router_negative_clean_gate import contains, overlap


def train_contrast(x, lab, fit_rows, eligible, fitpos, k, steps=500, lr=1e-2, wd=1e-4):
    """Same procedure as router_negative_gate_all.py (seed 0)."""
    pairs = ((lab == 1) | (lab == -1)) & fit_rows[:, None] & eligible
    r_idx, f_idx = pairs.nonzero(as_tuple=True)
    y = (lab[r_idx, f_idx] == 1).float()
    wt = torch.where(y > 0, 0.5 / y.sum().clamp_min(1), 0.5 / (1 - y).sum().clamp_min(1))
    gen = torch.Generator().manual_seed(0)
    proj = (torch.randn(x.shape[1], k, generator=gen) / x.shape[1] ** 0.5).requires_grad_()
    scale = torch.tensor(10.0, requires_grad=True)
    shift = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([proj, scale, shift], lr=lr)
    pos = fitpos.float()
    for _ in range(steps):
        zn = F.normalize(x @ proj, dim=-1)
        proto = F.normalize(pos.T @ zn, dim=-1)
        logit = scale * (zn[r_idx] * proto[f_idx]).sum(-1) + shift
        loss = (F.binary_cross_entropy_with_logits(logit, y, reduction="none") * wt).sum() + wd * (proj ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        zn = F.normalize(x @ proj, dim=-1)
        return proj.detach(), F.normalize(pos.T @ zn, dim=-1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="mcf,zsre,mquake,rwku")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--probe", default="outputs/relation_gate_analysis_v1/universal_probe.pt")
    ap.add_argument("--probe-q", type=float, default=0.05)
    ap.add_argument("--contrast-q", type=float, default=0.15)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--out-tag", default="multiseed_ours_gate_v1")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-relation-gate", action="store_true",
                    help="subject gate + combination gate + per-head bias only (no external relation probe)")
    ap.add_argument("--no-combination-gate", action="store_true", help="drop the combination gate as well")
    ap.add_argument("--source-router", default=None,
                    help="start from this router instead of the cache's (e.g. a data-grid refit); its heads, "
                         "feature mean and bias are used, the cache's states and probes stay the same")
    args = ap.parse_args(argv)
    probe = torch.load(args.probe, map_location="cpu", weights_only=False)
    pw, pb = probe["weight"].float(), probe["bias"].float()
    summary = {}

    for ds in args.datasets.split(","):
        cache = Path(args.root) / ds / f"seed{args.seed}"
        man = json.loads((cache / "manifest.json").read_text())
        src = Path(args.source_router) if args.source_router else Path(man["router_dir"])
        art = torch.load(src / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
        if [f["id"] for f in art["facts"]] != [f["id"] for f in man["facts"]]:
            raise SystemExit(f"{src} lists other facts than the cache {cache}")
        layer, facts = int(art["layer"]), art["facts"]
        n = len(facts)
        mu = torch.as_tensor(art["feature_mean"]).float()
        comps = art.get("feature_components")
        w0 = torch.as_tensor(art["router_weight"]).float()
        b0 = torch.as_tensor(art["router_bias"]).float()
        dpath = Path(man["router_dir"]) / "linear_router_dataset.json"      # the cache's probes = the shipped router's dataset
        dataset = json.loads(dpath.read_text()) if dpath.is_file() else None

        S = {}
        for name in ("train", "eval", "null"):
            st = torch.load(cache / f"states_{name}.pt", weights_only=False, mmap=True)
            h = st["states"][list(st["layers"]).index(layer)].float()
            S[name] = {"probes": json.loads((cache / f"probes_{name}.json").read_text()), "h": h,
                       "phi": apply_feature_map(h, mu, comps),
                       "eligible": torch.load(cache / f"router_{name}.pt", weights_only=False)["eligible"]}
        tr = S["train"]
        if dataset is None or [d["prompt"] for d in dataset] != [p["prompt"] for p in tr["probes"]]:
            dataset = [{"kind": "", "donor_relation": None} for _ in tr["probes"]]
        split = [p["split"] for p in tr["probes"]]
        m_fit = torch.tensor([s == "fit" for s in split])
        m_cal = torch.tensor([s == "calibration" for s in split])

        lab = torch.zeros(len(tr["probes"]), n, dtype=torch.long)
        for k, (p, d) in enumerate(zip(tr["probes"], dataset)):
            if p["should_fire"]:
                for i in p["true_facts"]:
                    lab[k, i] = 1
                continue
            for i in (p.get("meta") or {}).get("negative_for") or []:
                f, donor = facts[i], str(d.get("donor_relation") or "")
                ambiguous = contains(p["prompt"], f["object"]) or (
                    d["kind"] == "subject_transplant" and donor.startswith("P") and str(f["relation"]).startswith("P")
                    and overlap(f["relation"], donor))
                lab[k, i] = -2 if ambiguous else -1
        fitpos = (lab == 1) & m_fit[:, None]
        has_prof = fitpos.any(0)

        # relation gate (label-free profile)
        p_tr = (F.normalize(tr["h"], dim=-1) @ pw.T + pb).softmax(-1)
        profile = torch.zeros(n, pw.shape[0])
        for i in range(n):
            if has_prof[i]:
                profile[i] = p_tr[fitpos[:, i]].mean(0)
        dummy = torch.zeros(n, dtype=torch.long)
        rel = {sec: relation_log_probs(S[sec]["h"], pw, pb, dummy, profile) for sec in S}
        vals = rel["train"][fitpos & has_prof[None, :]]
        min_log_prob = float(torch.quantile(vals, args.probe_q))

        # combination gate
        proj, proto = train_contrast(tr["phi"], lab, m_fit, tr["eligible"], fitpos, args.dim)
        con = {sec: contrast_gate_scores(S[sec]["phi"], proj, proto) for sec in S}
        cvals = con["train"][(lab == 1) & m_cal[:, None] & tr["eligible"] & has_prof[None, :]]
        min_score = float(torch.quantile(cvals, args.contrast_q))
        comb_ok = {sec: (con[sec] >= min_score) | ~has_prof[None, :] for sec in S}
        if args.no_combination_gate:
            comb_ok = {sec: torch.ones_like(comb_ok[sec]) for sec in S}
        ok = comb_ok if args.no_relation_gate else {sec: (rel[sec] >= min_log_prob) & comb_ok[sec] for sec in S}

        # per-head bias over clean calibration negatives that pass every gate
        z_tr = tr["phi"] @ w0.T + b0
        cand = (lab == -1) & m_cal[:, None] & tr["eligible"] & ok["train"]
        shift = torch.zeros(n)
        for i in range(n):
            s = z_tr[cand[:, i], i]
            if s.numel() and float(s.max()) >= 0:
                shift[i] = -(float(s.max()) + 1e-3)
        bias = b0 + shift

        # replay the runtime decision on the cached evaluation and no-association prompts
        res = {}
        for sec in ("eval", "null"):
            z = S[sec]["phi"] @ w0.T + bias
            dec = decide_routes(z, S[sec]["eligible"] & ok[sec], float(art.get("threshold", 0.0)),
                                float(art.get("ambiguity_margin", 0.5)))
            fire, fact = dec["active"].cpu(), dec["fact"].cpu()
            if sec == "eval":
                should = torch.tensor([p["should_fire"] for p in S[sec]["probes"]])
                right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"]
                                      for k, p in enumerate(S[sec]["probes"])])
                res["forget_recall"] = float(right[should].float().mean())
                res["retain_fires"] = f"{int((fire & ~should).sum())}/{int((~should).sum())}"
            else:
                res["no_association_fires"] = f"{int(fire.sum())}/{len(fire)}"
        res.update({"relation_gate": not args.no_relation_gate, "combination_gate": not args.no_combination_gate,
                    "source_router": str(src),
                    "min_log_prob": None if args.no_relation_gate else min_log_prob, "min_score": min_score,
                    "heads_with_bias_shift": int((shift < 0).sum()), "facts_ungated": int((~has_prof).sum())})
        summary[ds] = res
        print(json.dumps({ds: res}), flush=True)

        dest = Path("outputs") / f"{ds}_{args.out_tag}" / f"seed{args.seed}" / f"L{layer}" / "router"
        if dest.exists():
            if not args.overwrite:
                raise SystemExit(f"{dest} exists (use --overwrite)")
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dest)
        new = dict(art)
        new["router_bias"] = bias
        if args.no_relation_gate:
            new.pop("relation_gate", None)
        else:
            new["relation_gate"] = {"weight": pw, "bias": pb, "fact_relation": dummy, "profile": profile,
                                    "min_log_prob": min_log_prob, "layer": layer, "mode": "profile",
                                    "quantile": args.probe_q, "probe": args.probe,
                                    "relations": list(probe["relations"])}
        if args.no_combination_gate:
            new.pop("contrast_gate", None)
        else:
            new["contrast_gate"] = {"proj": proj, "proto": proto, "ungated": ~has_prof, "min_score": min_score,
                                    "layer": layer, "dim": args.dim, "quantile": args.contrast_q}
        new["gated_router"] = {"source_router": str(src), "per_head_bias_shift": shift,
                               "cache": str(cache), "replay": res,
                               "rule": ("subject gate" if args.no_combination_gate and args.no_relation_gate else
                                        "subject gate & combination gate" if args.no_relation_gate else
                                        "subject gate & relation profile gate & combination gate")
                                       + "; per-head bias above clean calibration negatives (scripts/build_gated_router.py)"}
        torch.save(new, dest / "fact_association_embeddings.pt")
        mpath = dest / "association_manifest.json"
        if mpath.is_file():
            m = json.loads(mpath.read_text())
            m["gated_router"] = {"source_router": str(src), "relation_gate": not args.no_relation_gate,
                                 "probe": None if args.no_relation_gate else args.probe, "probe_q": args.probe_q,
                                 "contrast_q": args.contrast_q, "dim": args.dim, "replay": res}
            mpath.write_text(json.dumps(m, indent=2) + "\n")
        print(f"wrote {dest}", flush=True)
    Path("outputs/router_negative_fix_v1").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/router_negative_fix_v1/gated_router_seed{args.seed}.json").write_text(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
