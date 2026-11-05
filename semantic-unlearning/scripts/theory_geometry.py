#!/usr/bin/env python3
"""The Fisher geometry of an edit at block W, measured (paper, Section 4.3).

For a prompt x, J_x is the Jacobian of the next-token logits w.r.t. the block-W state at the
last prompt token, F(p) = diag(p) - p p^T, g_x = J_x^T (e_y - p_x) and K_x = J_x^T F(p_x) J_x.
The mean K over a prompt set is factored exactly on each prompt's top-k next tokens:
K = A A^T with A = [J_{x,k}^T F_k(p_x)^{1/2}]_x / sqrt(|set|), J_{x,k} the k logit rows (one
backward pass per rank), F_k the k x k principal block of F (the mass outside the top k is
reported). With the SVD of A, (K + eps I)^{-1} is exact; eps = alpha * tr(K) / d, a Tikhonov
version of the paper's norm bound, reported for several alpha.

For every association i (training views V_i: the canonical prompt and its rewrites; held-out
paraphrases H_i; gate-passing retain prompts N_i: the classifier's negatives that name s_i):

  selectivity   (Prop. "most selective edit") at the trained write block:
                eff(delta; g, K) = -g^T delta / sqrt(delta^T K_eps delta * g^T K_eps^{-1} g), the
                fraction of the optimal first-order answer decrease that delta achieves at its
                own second-order retain divergence (1 for delta = -K_eps^{-1} g). For the trained
                delta_i, the Euclidean gradient -g_V and a random direction; K from V_i (what the
                loss sees) and from N_i (the retain prompts at risk); g from V_i and from H_i.
  write_sweep   per block W: S = g_V^T K_eps^{-1} g_V with K from N_i (the largest decrease of the
                views' answer log-probability per unit sqrt(2 retain KL)) and the held-out
                decrease of that optimal edit, g_H^T K_eps^{-1} g_V / sqrt(S). Without damping
                both are invariant to an invertible linear change of the state basis, so they
                compare blocks.
  collateral    (Prop. "locality loss decomposes") on every retain and no-association prompt the
                deployed classifier fires on: the actual KL(pi_theta || pi_phi) at the first
                generated position with delta_i scaled by a, against the local prediction
                a^2/2 delta^T K_x delta (full vocabulary, double backward); and the decomposition
                E KL = Pr[fire] E[KL | fire].

    python -u scripts/theory_geometry.py --dataset mcf --seed 1 \
        --run-dir outputs/l16_ours_v1/mcf/seed1/ours \
        --cache-dir outputs/router_diag_ours_control_v1/mcf/seed1 --write-layers 4,8,12,14,16,19,22,25
    # -> <run-dir>/theory/geometry.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F


class Batch:
    """One forward pass with an edit at block W (last prompt token). delta=None: a zero edit
    with the graph kept, so per-row gradients and Jacobian rows can be taken repeatedly;
    double=True also keeps J^T u for J v by the double-backward trick."""

    def __init__(self, model, ids_list, layer, pad_id, delta=None, double=False):
        dev = next(model.parameters()).device
        b = len(ids_list)
        width = max(len(x) for x in ids_list)
        ids = torch.full((b, width), int(pad_id), dtype=torch.long)
        mask = torch.zeros((b, width), dtype=torch.long)
        for r, x in enumerate(ids_list):
            ids[r, :len(x)] = torch.tensor(x)
            mask[r, :len(x)] = 1
        rows = torch.arange(b, device=dev)
        last = (mask.sum(1) - 1).to(dev)
        grad = delta is None
        self.delta = (torch.zeros(b, model.config.hidden_size, device=dev, requires_grad=True) if grad
                      else delta.to(dev).float())

        def hook(_m, _a, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + torch.zeros_like(h).index_put((rows, last), self.delta.to(h.dtype))
            return (h, *out[1:]) if isinstance(out, tuple) else h

        handle = model.model.layers[layer].register_forward_hook(hook)
        try:
            with torch.set_grad_enabled(grad):
                hs = model.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).last_hidden_state
                self.logits = model.lm_head(hs[rows, last]).float()
        finally:
            handle.remove()
        self.p = torch.softmax(self.logits.detach(), -1)
        if grad and double:
            self.u = torch.zeros_like(self.logits, requires_grad=True)
            self.jtu = torch.autograd.grad(self.logits, self.delta, self.u, create_graph=True)[0]

    def vjp(self, rows_out):
        return torch.autograd.grad(self.logits, self.delta, rows_out, retain_graph=True)[0]

    def grads(self, ys):
        """g_x = J_x^T (e_y - p_x), one row per prompt."""
        e = torch.zeros_like(self.p)
        e[torch.arange(len(ys)), torch.as_tensor(ys)] = 1.0
        return self.vjp(e - self.p)

    def factor(self, k):
        """A [d, B*k] with A A^T = mean_x J_{x,k}^T F_k J_{x,k}; and the top-k mass per row."""
        top_p, top_i = self.p.topk(k, dim=-1)                          # [B, k]
        b = top_p.shape[0]
        jac = []
        for j in range(k):
            e = torch.zeros_like(self.p)
            e[torch.arange(b), top_i[:, j]] = 1.0
            jac.append(self.vjp(e))                                    # [B, d]
        jac = torch.stack(jac, 1)                                      # [B, k, d]
        fk = torch.diag_embed(top_p) - top_p[:, :, None] * top_p[:, None, :]
        w, v = torch.linalg.eigh(fk)
        root = v @ torch.diag_embed(w.clamp_min(0).sqrt()) @ v.transpose(1, 2)   # F_k^{1/2}
        a = (jac.transpose(1, 2) @ root)                               # [B, d, k]
        a = a.permute(1, 0, 2).reshape(a.shape[1], -1) / math.sqrt(b)
        return a, top_p.sum(-1)

    def quad(self, v):
        """v^T K_x v per row, full vocabulary (needs double=True)."""
        jv = torch.autograd.grad((self.jtu * v.expand_as(self.delta)).sum(), self.u, retain_graph=True)[0]
        m = (self.p * jv).sum(-1)
        return (self.p * jv * jv).sum(-1) - m * m


class Metric:
    """K_eps = A A^T + eps I with eps = alpha * tr(K) / d, exact through the SVD of A."""

    def __init__(self, a, alpha):
        self.u, s, _ = torch.linalg.svd(a, full_matrices=False)
        self.s2 = s * s
        self.d = a.shape[0]
        self.eps = alpha * float(self.s2.sum()) / self.d
        self.a = a

    def norm2(self, v):
        return float((self.a.T @ v).pow(2).sum()) + self.eps * float(v @ v)

    def solve(self, g):
        c = self.u.T @ g
        return self.u @ (c / (self.s2 + self.eps)) + (g - self.u @ c) / self.eps


def answer_token(tok, prompt, answer):
    p = tok(prompt)["input_ids"]
    full = tok(prompt + " " + answer.strip())["input_ids"]
    if full[:len(p)] == p and len(full) > len(p):
        return full[len(p)]
    return tok(" " + answer.strip(), add_special_tokens=False)["input_ids"][0]


def eff(g, delta, metric, g_kinv_g):
    return float(-(g @ delta)) / math.sqrt(max(metric.norm2(delta), 1e-30) * max(g_kinv_g, 1e-30))


def median(xs):
    xs = sorted(x for x in xs if x is not None and math.isfinite(x))
    return xs[len(xs) // 2] if xs else None


def mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--run-dir", required=True, help="finished ours run: fact_association_embeddings.pt, association_examples.json")
    ap.add_argument("--cache-dir", required=True, help="router_diag cache of the same associations (probes, routes)")
    ap.add_argument("--write-layers", default="", help="blocks for write_sweep (empty: skip)")
    ap.add_argument("--sweep-facts", type=int, default=50)
    ap.add_argument("--max-facts", type=int, default=0, help="selectivity on the first k associations (0: all)")
    ap.add_argument("--max-prompts", type=int, default=24, help="per association and set")
    ap.add_argument("--topk", type=int, default=64, help="next tokens per prompt in the factor of K")
    ap.add_argument("--alphas", default="0.001,0.01,0.1", help="damping eps = alpha * tr(K) / d")
    ap.add_argument("--max-collateral", type=int, default=400)
    ap.add_argument("--scales", default="0.05,0.1,0.25,0.5,1")
    ap.add_argument("--parts", default="selectivity,write_sweep,collateral")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args(argv)
    torch.manual_seed(args.seed)
    alphas = [float(a) for a in args.alphas.split(",")]

    run = Path(args.run_dir)
    cache = Path(args.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    artifact = torch.load(run / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    rows = torch.as_tensor(artifact["rows"]).float()
    write_layer = int((artifact.get("early_write") or {}).get("layer", artifact["layer"]))
    facts = artifact["facts"]
    n = len(facts)
    if [f["id"] for f in manifest["facts"]] != [f["id"] for f in facts]:
        raise SystemExit("cache and run list different associations")
    parts = set(args.parts.split(","))

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], local_files_only=args.local_files_only)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(manifest["model_path"], dtype=torch.float32,
                                                 local_files_only=args.local_files_only,
                                                 attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)

    # ---- prompt sets per association
    fid = {f["id"]: i for i, f in enumerate(facts)}
    views = {i: [] for i in range(n)}
    for e in json.loads((run / "association_examples.json").read_text()):
        if e.get("split") == "train" and e.get("fact_id") in fid and e["prompt"] not in views[fid[e["fact_id"]]]:
            views[fid[e["fact_id"]]].append(e["prompt"])
    held = {i: [] for i in range(n)}
    for p in json.loads((cache / "probes_eval.json").read_text()):
        if p["split"] == "paraphrase" and p["true_facts"] and p["input_ids"] is None:
            held[p["true_facts"][0]].append(p["prompt"])
    negs = {i: [] for i in range(n)}
    for section in ("train", "aug"):
        probes = json.loads((cache / f"probes_{section}.json").read_text())
        eligible = torch.load(cache / f"router_{section}.pt", weights_only=False)["eligible"]
        for r, p in enumerate(probes):
            if p["should_fire"] or p["input_ids"] is not None:
                continue
            for i in (p.get("meta") or {}).get("negative_for") or []:
                if bool(eligible[r, i]) and p["prompt"] not in negs[i]:
                    negs[i].append(p["prompt"])
    cap = args.max_prompts
    ids_of = lambda prompts: [tok(x)["input_ids"] for x in prompts[:cap]]
    ys_of = lambda i, prompts: [answer_token(tok, x, str(facts[i]["object"])) for x in prompts[:cap]]
    complete = [i for i in range(n) if views[i] and held[i] and negs[i]]

    out = {"dataset": args.dataset, "seed": args.seed, "run_dir": str(run), "cache_dir": str(cache),
           "model_path": manifest["model_path"], "trained_write_layer": write_layer, "alphas": alphas,
           "topk": args.topk, "max_prompts": cap, "associations_with_all_sets": len(complete), "of": n,
           "sets_per_assoc": {"views": mean([min(len(views[i]), cap) for i in complete]),
                              "heldout": mean([min(len(held[i]), cap) for i in complete]),
                              "gate_passing_negatives": mean([min(len(negs[i]), cap) for i in complete])}}
    out_dir = run / "theory"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "geometry.json"
    if out_path.is_file():
        prev = json.loads(out_path.read_text())
        for k in ("selectivity", "write_sweep", "collateral"):
            if k in prev and k not in parts:
                out[k] = prev[k]

    def save():
        out_path.write_text(json.dumps(out, indent=2) + "\n")

    def gradient(layer, i, prompts):
        b = Batch(model, ids_of(prompts), layer, pad_id)
        return b.grads(ys_of(i, prompts)).mean(0).detach()

    def factor(layer, prompts):
        b = Batch(model, ids_of(prompts), layer, pad_id)
        a, mass = b.factor(args.topk)
        return a.detach(), float(mass.mean())

    t0 = time.time()
    # ---- selectivity at the trained write block
    if "selectivity" in parts:
        per = []
        for i in complete[:args.max_facts or n]:
            g_v = gradient(write_layer, i, views[i])
            g_h = gradient(write_layer, i, held[i])
            a_v, mass_v = factor(write_layer, views[i])
            a_n, mass_n = factor(write_layer, negs[i])
            delta = rows[i].to(args.device)
            rand = torch.randn_like(delta)
            rec = {"fact": i, "delta_norm": float(delta.norm()), "topk_mass_views": mass_v,
                   "topk_mass_retain": mass_n, "cos_g_views_g_heldout": float(F.cosine_similarity(g_v, g_h, dim=0))}
            for alpha in alphas:
                for kname, a in (("K_views", a_v), ("K_retain", a_n)):
                    metric = Metric(a, alpha)
                    for gname, g in (("g_views", g_v), ("g_heldout", g_h)):
                        z = metric.solve(g)
                        gkg = float(g @ z)
                        rec[f"{alpha}/{kname}/{gname}"] = {
                            "trained": eff(g, delta, metric, gkg), "gradient": eff(g, -g_v, metric, gkg),
                            "random": eff(g, rand, metric, gkg),
                            "cos_trained_opt": float(F.cosine_similarity(delta, -z, dim=0)),
                            "cos_gradient_opt": float(F.cosine_similarity(-g_v, -z, dim=0))}
            per.append(rec)
            torch.cuda.empty_cache()
            print(json.dumps({"part": "selectivity", "fact": i, "s": round(time.time() - t0),
                              "0.01/K_retain/g_heldout": rec["0.01/K_retain/g_heldout"] if "0.01/K_retain/g_heldout" in rec else None}),
                  flush=True)
        summary = {}
        for key in [k for k in per[0] if k.count("/") == 2]:
            summary[key] = {m: {"median": median([r[key][m] for r in per]), "mean": mean([r[key][m] for r in per])}
                            for m in ("trained", "gradient", "random", "cos_trained_opt", "cos_gradient_opt")}
        summary["topk_mass_retain_median"] = median([r["topk_mass_retain"] for r in per])
        summary["topk_mass_views_median"] = median([r["topk_mass_views"] for r in per])
        summary["cos_g_views_g_heldout_median"] = median([r["cos_g_views_g_heldout"] for r in per])
        summary["delta_norm_median"] = median([r["delta_norm"] for r in per])
        out["selectivity"] = {"associations": len(per), "summary": summary, "per_association": per}
        save()

    # ---- write-block sweep
    if "write_sweep" in parts and args.write_layers:
        sweep = {}
        chosen = complete[:args.sweep_facts]
        for layer in [int(x) for x in args.write_layers.split(",")]:
            per = []
            for i in chosen:
                g_v = gradient(layer, i, views[i])
                g_h = gradient(layer, i, held[i])
                a_n, mass = factor(layer, negs[i])
                rec = {"fact": i, "topk_mass_retain": mass}
                for alpha in alphas:
                    metric = Metric(a_n, alpha)
                    z_v = metric.solve(g_v)
                    s_v = float(g_v @ z_v)
                    rec[str(alpha)] = {"S_views": s_v,
                                       "heldout_drop_at_opt": float(g_h @ z_v) / math.sqrt(max(s_v, 1e-30)),
                                       "heldout_drop_at_gradient": float(g_h @ g_v) / math.sqrt(max(metric.norm2(g_v), 1e-30))}
                per.append(rec)
                torch.cuda.empty_cache()
            sweep[layer] = {"facts": len(per), "topk_mass_retain_median": median([r["topk_mass_retain"] for r in per]),
                            **{str(alpha): {f"{k}_median": median([r[str(alpha)][k] for r in per])
                                            for k in ("S_views", "heldout_drop_at_opt", "heldout_drop_at_gradient")}
                               for alpha in alphas},
                            "per_association": per}
            print(json.dumps({"part": "write_sweep", "layer": layer, "s": round(time.time() - t0),
                              **{k: v for k, v in sweep[layer].items() if k != "per_association"}}), flush=True)
            out["write_sweep"] = sweep
            save()

    # ---- collateral on retain prompts the classifier fires on
    if "collateral" in parts:
        scales = [float(s) for s in args.scales.split(",")]
        fired, n_retain = [], 0
        for section in ("eval", "null"):
            probes = json.loads((cache / f"probes_{section}.json").read_text())
            routes = torch.load(cache / f"router_{section}.pt", weights_only=False)
            for r, p in enumerate(probes):
                if p["should_fire"]:
                    continue
                if section == "eval":
                    n_retain += 1
                if bool(routes["active"][r]):
                    fired.append((section, p, int(routes["fact"][r])))
        n_fired_retain = sum(1 for section, _, _ in fired if section == "eval")
        fired = fired[:args.max_collateral]
        recs = []
        for start in range(0, len(fired), 8):
            chunk = fired[start:start + 8]
            ids = [tok(p["prompt"])["input_ids"] if p["input_ids"] is None else list(p["input_ids"])
                   for _, p, _ in chunk]
            d = rows[[f for _, _, f in chunk]].to(args.device)
            base = Batch(model, ids, write_layer, pad_id, double=True)
            q = base.quad(d).detach()
            logp = torch.log_softmax(base.logits.detach(), -1)
            kls = {}
            for a in scales:
                with torch.no_grad():
                    edited = Batch(model, ids, write_layer, pad_id, delta=a * d)
                    kls[a] = (base.p * (logp - torch.log_softmax(edited.logits, -1))).sum(-1)
            for k, (section, p, f) in enumerate(chunk):
                recs.append({"section": section, "group": p["group"], "fact": f, "quad": float(q[k]),
                             "kl": {str(a): float(kls[a][k]) for a in scales}})
            del base
            torch.cuda.empty_cache()

        def spearman(a, b):
            if len(a) < 3:
                return None
            ra = torch.tensor(a).argsort().argsort().float()
            rb = torch.tensor(b).argsort().argsort().float()
            return float(torch.corrcoef(torch.stack([ra, rb]))[0, 1])

        by_scale = {}
        for a in scales:
            pred = [0.5 * a * a * r["quad"] for r in recs]
            act = [r["kl"][str(a)] for r in recs]
            by_scale[str(a)] = {"median_actual_over_predicted": median([x / y for x, y in zip(act, pred) if y > 0]),
                                "spearman_actual_vs_predicted": spearman(pred, act),
                                "mean_actual_kl": mean(act), "mean_predicted_kl": mean(pred)}
        full = str(max(scales))
        retain_fired = [r for r in recs if r["section"] == "eval"]
        rate = n_fired_retain / max(n_retain, 1)
        out["collateral"] = {
            "fired_prompts": len(recs), "fired_retain_eval": n_fired_retain, "measured_retain_eval": len(retain_fired),
            "retain_eval_prompts": n_retain,
            "fire_rate_retain_eval": rate, "full_scale": full,
            "mean_kl_given_fire_retain_eval": mean([r["kl"][full] for r in retain_fired]),
            "mean_kl_retain_eval": rate * (mean([r["kl"][full] for r in retain_fired]) or 0.0),
            "by_scale": by_scale, "per_prompt": recs}
        print(json.dumps({"part": "collateral", "s": round(time.time() - t0),
                          **{k: v for k, v in out["collateral"].items() if k != "per_prompt"}}), flush=True)
        save()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
