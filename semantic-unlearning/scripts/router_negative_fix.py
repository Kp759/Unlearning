#!/usr/bin/env python3
"""Why the paper's router fires on its own held-out negatives, and what brings that down (analysis
only; the paper's router is unchanged).

From a router_diag cache at the router layer (CPU, minutes). Each head is scored on its own
held-out negatives (calibration + audit: prompts about its subject that ask something else), on
the official evaluation prompts and on the no-association prompts, under the runtime rule
(subject gate, best eligible head, z >= 0, margin 0.5).

Protocol: weights from the fit split (+ the listed extra negatives), bias from the calibration
split, numbers on the audit split, the evaluation and the no-association prompts, none of which
any variant fits or calibrates on.

    python -u scripts/router_negative_fix.py --cache-dir outputs/router_diag_ours_control_v1/mcf/seed1
    # -> <cache>/negative_fix/negative_fix.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from linear_router import decide_routes


def fit_heads(x, y, mask, gamma, iters=300):
    """Independent class-balanced logistic heads; head i trains only where mask[:, i]."""
    yb, m = y.float(), mask.float()
    n_pos = (yb * m).sum(0).clamp_min(1)
    n_neg = ((1 - yb) * m).sum(0).clamp_min(1)
    weight = m * (yb / (2 * n_pos) + (1 - yb) / (2 * n_neg))
    w = torch.zeros(x.shape[1], y.shape[1], requires_grad=True)
    b = torch.zeros(y.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(x @ w + b, yb, reduction="none") * weight).sum() \
            + gamma * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def own_labels(probes, n):
    """[P, N] +1 own positive, -1 own negative, 0 otherwise."""
    lab = torch.zeros(len(probes), n, dtype=torch.long)
    for k, p in enumerate(probes):
        if p["should_fire"]:
            for i in p["true_facts"]:
                lab[k, i] = 1
        else:
            for i in (p.get("meta") or {}).get("negative_for") or []:
                lab[k, i] = -1
    return lab


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gammas", default="1e-4,1e-3,1e-2,1e-1")
    ap.add_argument("--recall", type=float, default=0.98, help="the paper's calibration target")
    args = ap.parse_args(argv)

    cache = Path(args.cache_dir)
    man = json.loads((cache / "manifest.json").read_text())
    router = torch.load(Path(man["router_dir"]) / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    layer = int(man["router_layer"])
    n = len(router["facts"])
    mu = torch.as_tensor(router["feature_mean"]).float()

    def section(name):
        probes = json.loads((cache / f"probes_{name}.json").read_text())
        st = torch.load(cache / f"states_{name}.pt", weights_only=False, mmap=True)
        h = st["states"][list(st["layers"]).index(layer)].float()
        u = F.normalize(h, dim=-1) - mu
        routes = torch.load(cache / f"router_{name}.pt", weights_only=False)
        return probes, u, routes["eligible"], routes

    tr_p, tr_u, tr_e, tr_r = section("train")
    ag_p, ag_u, ag_e, _ = section("aug")
    ev_p, ev_u, ev_e, ev_r = section("eval")
    nu_p, nu_u, nu_e, _ = section("null")
    tr_lab, ag_lab = own_labels(tr_p, n), own_labels(ag_p, n)
    split = [p["split"] for p in tr_p]
    is_fit = torch.tensor([s == "fit" for s in split])
    is_cal = torch.tensor([s == "calibration" for s in split])
    is_aud = torch.tensor([s == "audit" for s in split])

    def bias_for(z_cal, lab_cal, rule, target):
        """Bias shift per head (or global) from the calibration split only."""
        pos = lab_cal == 1
        neg = lab_cal == -1
        if rule == "recall_global":                        # the paper's rule: 98% of positives fire
            scores = z_cal[pos]
            return torch.full((n,), -float(torch.quantile(scores, 1 - target)))
        if rule == "max_neg":                              # above every calibration negative of the head
            out = torch.zeros(n)
            for i in range(n):
                s_neg = z_cal[neg[:, i], i]
                s_pos = z_cal[pos[:, i], i]
                hi = float(s_neg.max()) if s_neg.numel() else float(s_pos.min()) - 1.0
                out[i] = -(hi + 1e-4)
            return out
        if rule == "midpoint":                             # halfway between the classes, per head
            out = torch.zeros(n)
            for i in range(n):
                s_neg, s_pos = z_cal[neg[:, i], i], z_cal[pos[:, i], i]
                if s_neg.numel() and s_pos.numel():
                    out[i] = -0.5 * (float(s_neg.max()) + float(s_pos.min()))
            return out
        raise ValueError(rule)

    def score(w, b):
        return {k: u @ w + b for k, u in (("train", tr_u), ("eval", ev_u), ("null", nu_u))}

    def evaluate(z):
        lab = tr_lab
        res = {}
        for name, msk in (("calibration", is_cal), ("audit", is_aud)):
            zz, ll = z["train"][msk], lab[msk]
            res[name] = {"pos_fire": int(((zz >= 0) & (ll == 1)).sum()), "pos": int((ll == 1).sum()),
                         "neg_fire": int(((zz >= 0) & (ll == -1)).sum()), "neg": int((ll == -1).sum())}
        dec = decide_routes(z["eval"], ev_e, 0.0, 0.5)
        fire, fact = dec["active"].cpu(), dec["fact"].cpu()
        should = torch.tensor([p["should_fire"] for p in ev_p])
        right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"] for k, p in enumerate(ev_p)])
        res["eval"] = {"forget_recall": float(right[should].float().mean()),
                       "retain_fires": int((fire & ~should).sum()), "retain": int((~should).sum())}
        dn = decide_routes(z["null"], nu_e, 0.0, 0.5)["active"].cpu()
        groups = {}
        for k, p in enumerate(nu_p):
            g = groups.setdefault(p["group"], [0, 0])
            g[0] += int(dn[k])
            g[1] += 1
        res["null"] = {g: f"{a}/{b}" for g, a, b in ((g, *v) for g, v in groups.items())}
        res["null_fires"] = int(dn.sum())
        return res

    out = {"cache": str(cache), "layer": layer, "variants": {}}
    # shipped router as deployed
    w0 = torch.as_tensor(router["router_weight"]).float().T
    b0 = torch.as_tensor(router["router_bias"]).float()
    out["variants"]["shipped"] = evaluate(score(w0, b0))
    print(json.dumps({"shipped": out["variants"]["shipped"]}), flush=True)

    # refits: weights from the fit split, masked by the subject gate as the shipped heads
    fit_x, fit_lab, fit_e = tr_u[is_fit], tr_lab[is_fit], tr_e[is_fit]
    neg_ag = (ag_lab == -1).any(1)                        # aug: same-subject / same-relation hard negatives
    for extra in ("none", "aug_hardneg"):
        if extra == "none":
            x, lab, el = fit_x, fit_lab, fit_e
        else:
            x = torch.cat([fit_x, ag_u[neg_ag]])
            lab = torch.cat([fit_lab, ag_lab[neg_ag]])
            el = torch.cat([fit_e, ag_e[neg_ag]])
        y = (lab == 1)
        mask = el & (lab != 0)
        for gamma in [float(g) for g in args.gammas.split(",")]:
            w, b = fit_heads(x, y, mask, gamma)
            z_cal = (tr_u[is_cal] @ w + b)
            for rule in ("recall_global", "max_neg", "midpoint"):
                shift = bias_for(z_cal, tr_lab[is_cal], rule, args.recall)
                key = f"{extra}/gamma={gamma:g}/{rule}"
                out["variants"][key] = evaluate(score(w, b + shift))
                v = out["variants"][key]
                print(json.dumps({key: {"audit": v["audit"], "eval": v["eval"], "null_fires": v["null_fires"]}}), flush=True)

    d = cache / "negative_fix"
    d.mkdir(exist_ok=True)
    (d / "negative_fix.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = ["| variant | audit pos fired | audit neg fired | cal neg fired | eval forget recall | retain fires | no-association fires |",
             "|---|---|---|---|---|---|---|"]
    for k, v in out["variants"].items():
        a, c, e = v["audit"], v["calibration"], v["eval"]
        lines.append(f"| {k} | {a['pos_fire']}/{a['pos']} | {a['neg_fire']}/{a['neg']} | {c['neg_fire']}/{c['neg']} | "
                     f"{e['forget_recall']:.3f} | {e['retain_fires']}/{e['retain']} | {v['null_fires']}/{len(nu_p)} |")
    (d / "negative_fix.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
