#!/usr/bin/env python3
"""Analysis only (the paper's router is unchanged): can other features separate each head's
positives from its negatives better? Run on the cached seed-1 diagnostics of all four benchmarks
(outputs/router_diag_ours_control_v1).

    base       unit last-token state at L19 (the paper's router features)
    white_a    base, then a shared within-class whitening learned on the fit split of ALL facts
               (shrinkage LDA: Sigma = (1-a) S_w + a tr(S_w)/d I, x -> Sigma^-1/2 (x - mu))
    span       [unit last-token L19, unit mean L19 state over the tokens after the subject]
    span_white_a  both

Every variant is refit with the shipped recipe (router_layer_sweep_cached: masked class-balanced
BCE, L2 by grouped CV on the fit split, cutoff from the router's own calibration rule on the
calibration split) and scored on the same held-out prompts. Step A (--extract) caches the span
states on GPU; step B scores (fits on GPU). The span state is the mean over the tokens after the
last occurrence of a gate-matched subject (the last 4 tokens when none is found).

    python -u scripts/router_feature_variants.py --extract     # GPU, ~20 min for all four
    # -> outputs/router_feature_variants_v1/{feature_variants.md, feature_variants.json}
    #    (+ <ds>/span_<section>.pt, the cached span states)
See README_ROUTER_DIAGNOSTICS.md, "Features tried (analysis only)"."""
import argparse, json, math, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
ap.add_argument("--out", default="outputs/router_feature_variants_v1")
ap.add_argument("--extract", action="store_true"); ap.add_argument("--layer", type=int, default=19)
ap.add_argument("--datasets", default="mcf,zsre,mquake,rwku"); ap.add_argument("--alphas", default="0.2,0.5,0.8")
a = ap.parse_args()
import router_layer_sweep_cached as sw
from linear_router import decide_routes
from router_diag_report import silhouette

SECTIONS = ("train", "aug", "eval", "null", "perturbed")
root, out = Path(a.root), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
DS = a.datasets.split(",")
t0 = time.time()


def log(**k):
    print(json.dumps({**k, "t": round(time.time() - t0)}), flush=True)


# ------------------------------------------------------------------ step A: span states (GPU)
def find_span(ids, subjects, tok):
    """(start, end) token indices of the tokens after the LAST occurrence of any of the
    subjects (token match, with and without a leading space), or None."""
    best = None
    for s in subjects:
        for variant in (" " + s.strip(), s.strip()):
            pat = tok(variant, add_special_tokens=False)["input_ids"]
            if not pat:
                continue
            for i in range(len(ids) - len(pat), -1, -1):
                if ids[i:i + len(pat)] == pat:
                    end = i + len(pat)
                    if best is None or end > best:
                        best = end
                    break
    if best is None or best >= len(ids):
        return None
    return best, len(ids)


def extract():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    for ds in DS:
        d = root / ds / "seed1"
        man = json.loads((d / "manifest.json").read_text())
        subjects = [str(f.get("subject", "")) for f in man["facts"]]
        tok = AutoTokenizer.from_pretrained(man["model_path"], local_files_only=True)
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        (out / ds).mkdir(parents=True, exist_ok=True)
        for dtype, sections in ((torch.bfloat16, ("eval", "null", "perturbed")), (torch.float32, ("train", "aug"))):
            model = AutoModelForCausalLM.from_pretrained(man["model_path"], dtype=dtype, local_files_only=True,
                                                         attn_implementation="eager").cuda().eval()
            for s in sections:
                path = out / ds / f"span_{s}.pt"
                if path.is_file():
                    continue
                probes = json.loads((d / f"probes_{s}.json").read_text())
                elig = torch.load(d / f"router_{s}.pt", weights_only=False)["eligible"].bool()
                ids = [p["input_ids"] if p["input_ids"] is not None else tok(p["prompt"])["input_ids"] for p in probes]
                spans, found = [], 0
                for i, x in enumerate(ids):
                    sp = find_span(x, [subjects[j] for j in elig[i].nonzero().flatten().tolist()], tok)
                    found += sp is not None and bool(elig[i].any())
                    spans.append(sp or (max(1, len(x) - 4), len(x)))      # fallback: last 4 tokens
                res = torch.empty(len(ids), model.config.hidden_size, dtype=torch.float32)
                order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
                cap = {}
                h = model.model.layers[a.layer].register_forward_hook(
                    lambda m, ar, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
                with torch.no_grad():
                    for b in range(0, len(order), 64):
                        bt = order[b:b + 64]
                        w = max(len(ids[i]) for i in bt)
                        inp = torch.full((len(bt), w), pad); m = torch.zeros_like(inp)
                        for r, i in enumerate(bt):
                            inp[r, :len(ids[i])] = torch.tensor(ids[i]); m[r, :len(ids[i])] = 1
                        model.model(input_ids=inp.cuda(), attention_mask=m.cuda(), use_cache=False)
                        hs = cap["h"].float()
                        for r, i in enumerate(bt):
                            st, en = spans[i]
                            res[i] = hs[r, st:en].mean(0).cpu()
                h.remove()
                torch.save({"layer": a.layer, "span": res, "spans": spans}, path)
                log(phase="span", ds=ds, section=s, prompts=len(ids),
                    eligible_with_subject_found=found, eligible=int(elig.any(1).sum()))
            del model; torch.cuda.empty_cache()


# ------------------------------------------------------------------ step B: variants (CPU)
class VCache(sw.Cache):
    """sw.Cache whose features() takes a variant tag instead of a layer list."""
    def __init__(self, cache_dir, span_dir):
        super().__init__(cache_dir)
        self.span_dir = span_dir
        self.transform = {}

    def raw(self, name, kind):
        sec = self.section(name)
        last = F.normalize(sec["states"][sec["layers"].index(a.layer)].float(), dim=-1)
        if kind == "base":
            return last
        span = F.normalize(torch.load(self.span_dir / f"span_{name}.pt", weights_only=False)["span"], dim=-1)
        return torch.cat([last, span], dim=-1)

    def features(self, name, layers, rows=None):
        tag = layers[0]
        kind = "span" if tag.startswith("span") else "base"
        x = self.raw(name, kind)
        if "white" in tag:
            mu, T = self.transform[tag]
            x = (x - mu) @ T
        return x if rows is None else x[rows]


_EIG = {}


def fit_whitening(cache, kind, alpha):
    """Shared within-class scatter on the fit split of the training prompts: classes are
    (fact, positive) and (fact, negative) groups. One eigendecomposition per feature kind:
    the shrinkage only moves the eigenvalues."""
    tr = cache.section("train")
    x = cache.raw("train", kind)
    groups = {}
    for i, p in enumerate(tr["probes"]):
        if p["split"] != "fit":
            continue
        if p["should_fire"] and p["true_facts"]:
            key = ("pos", p["true_facts"][0])
        else:
            facts = p["meta"].get("negative_for") or tr["routes"]["eligible"][i].nonzero().flatten().tolist()
            if not facts:
                continue
            key = ("neg", int(facts[0]))
        groups.setdefault(key, []).append(i)
    rows = [i for v in groups.values() for i in v]
    key = (str(cache.dir), kind)
    if key not in _EIG:
        centred = torch.cat([x[v] - x[v].mean(0, keepdim=True) for v in groups.values()]).double().cuda()
        S = centred.T @ centred / max(1, len(centred) - len(groups))
        lam, vec = torch.linalg.eigh(S)
        _EIG[key] = (lam.clamp_min(0), vec, float(lam.clamp_min(0).mean()))
    lam, vec, mean_lam = _EIG[key]
    shrunk = (1 - alpha) * lam + alpha * mean_lam
    T = (vec * shrunk.clamp_min(1e-12).rsqrt()) @ vec.T
    return x[rows].mean(0).float(), T.float().cpu()


def heldout(cache, router, data, threshold, margin):
    """Held-out training prompts (audit split: never used to fit or calibrate)."""
    aud = torch.tensor([s == "audit" for s in data["split"]])
    z = sw.logits_for(router, data["x"][aud])
    dd = decide_routes(z, data["eligible"][aud], threshold, margin)
    own = data["owner"][aud]
    pos = own >= 0
    rec = float(((dd["active"] & (dd["fact"] == own))[pos]).float().mean()) if pos.any() else None
    neg_fire = float(dd["active"][~pos].float().mean()) if (~pos).any() else None
    return rec, neg_fire


def mean_silhouette(cache, tag):
    tr = cache.section("train")
    x = cache.features("train", (tag,))
    per = []
    for f in range(len(cache.facts)):
        pos = [i for i, p in enumerate(tr["probes"]) if p["should_fire"] and p["true_facts"] and p["true_facts"][0] == f]
        neg = [i for i, p in enumerate(tr["probes"]) if not p["should_fire"] and f in (p["meta"].get("negative_for") or [])]
        if len(pos) >= 2 and len(neg) >= 2:
            idx = torch.tensor(pos + neg)
            lab = torch.tensor([1] * len(pos) + [0] * len(neg))
            per.append(float(silhouette(x[idx], lab, metric="euclidean")))
    return sum(per) / len(per) if per else None


def score():
    alphas = [float(v) for v in a.alphas.split(",")]
    tags = ["base"] + [f"white_{al}" for al in alphas] + ["span"] + [f"span_white_{al}" for al in alphas]
    results = {}
    for ds in DS:
        cache = VCache(root / ds / "seed1", out / ds)
        for tag in tags:
            if "white" in tag:
                cache.transform[tag] = fit_whitening(cache, "span" if tag.startswith("span") else "base",
                                                     float(tag.rsplit("_", 1)[1]))
        fit = cache.manifest.get("router_fit") or {}
        fixed = (float(fit.get("selected_l2", 1e-3)), 0)
        lambdas = tuple(sorted({1e-5, 1e-4, 1e-3, 1e-2, 1e-1, fixed[0]}))
        rows = {}
        for tag in tags:
            for hp in (["fixed", "cv"] if tag == "base" else ["cv"]):
                data = sw.assemble(cache, "baseline", (tag,))
                router = sw.fit_one(data, fixed if hp == "fixed" else "cv", "cuda", lambdas, (0,), 5)
                rp = sw.calibration_rule(cache, "deployed")
                thr, margin = sw.calibrate(router, data, data["eligible"], rp)
                ev = sw.evaluate(cache, router, (tag,), thr, margin, sw.deployed_eligibility(cache), {})
                rec_h, neg_h = heldout(cache, router, data, thr, margin)
                name = tag if tag != "base" else f"base ({hp} l2)"
                rows[name] = {
                    "l2": router["l2"], "silhouette": mean_silhouette(cache, tag),
                    "audit_recall": rec_h, "audit_negatives_firing": neg_h,
                    "recall": ev["eval"]["recall"], "eval_false_fires": ev["eval"]["fp"],
                    "route_auc": ev.get("eval_route_auc"), "null_fired": ev.get("null_fired"),
                    "perturbed_recall": ev.get("perturbed_recall"),
                    "agreement_with_shipped": sw.agreement_with_shipped(cache, ev["_decisions"]) if tag == "base" else None}
                r = rows[name]
                log(phase="variant", ds=ds, variant=name, **{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
        results[ds] = rows
        (out / "feature_variants.json").write_text(json.dumps(results, indent=1))
    lines = []
    for ds, rows in results.items():
        lines += [f"\n## {ds}", "| variant | l2 | silhouette | audit recall | audit negatives firing | recall | eval false fires | route AUC | null fires | perturbed recall |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in rows.items():
            f = lambda v, d=3: "–" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
            lines.append(f"| {name} | {r['l2']:g} | {f(r['silhouette'])} | {f(r['audit_recall'])} | {f(r['audit_negatives_firing'])} | "
                         f"{f(r['recall'])} | {r['eval_false_fires']} | {f(r['route_auc'], 4)} | {r['null_fired']} | {f(r['perturbed_recall'])} |")
    (out / "feature_variants.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if a.extract:
    extract()
score()
print("done", round(time.time() - t0), flush=True)
