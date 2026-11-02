#!/usr/bin/env python3
"""Analysis only (not the paper's router): a label-free relation gate on all four benchmarks.

Checked on the cached diagnostics of the paper's routers (outputs/router_diag_ours_control_v1).

One relation probe at L19 (34 MCF/Wikidata relations), trained once on prompts that name no
protected subject of any benchmark and are no test prompt of any benchmark. Per fact, no label:
a relation profile = mean probe distribution over the fact's own training positives. Gate: head f
eligible only if log sum_r p(r | prompt) profile_f(r) >= cutoff, cutoff = quantile q of the same
score on the training positives (pooled per benchmark, or per fact). Heads and cutoff unchanged.
Probe data: MCF prompts of training-side records, the relation scaffolds of
mcf_shadow_relation_prompts (CALIBRATION_SCAFFOLDS only), mcf_synthetic_paraphrase_templates,
three question forms, and instruction-suffix copies (fit_relation_gate.SUFFIXES).

    python -u scripts/relation_gate_universal_check.py      # GPU, ~2 min
    # -> outputs/relation_gate_analysis_v1/universal_gate_check.json
See README_ROUTER_DIAGNOSTICS.md, "Relation gate (analysis only)"."""
import argparse, json, math, random, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
ap.add_argument("--out", default="outputs/relation_gate_analysis_v1")
ap.add_argument("--layer", type=int, default=19); ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--cap", type=int, default=2000); ap.add_argument("--scaffold-cap", type=int, default=600)
ap.add_argument("--suffix-share", type=float, default=0.25)
a = ap.parse_args()
from cache_router_states import capture_last_token_states, runtime_rule
from linear_router import decide_routes
from fit_relation_gate import SUFFIXES, fit_softmax_probe
from mcf_shadow_relation_prompts import RELATION_NOUN_PHRASES, CALIBRATION_SCAFFOLDS
from mcf_synthetic_paraphrase_templates import RELATION_ALTERNATE_TEMPLATES
import router_diag_probes as probes_lib
import mcf_zero_unlearn_official_eval as mcf

# question forms (authored here; none is a held-out scaffold of mcf_shadow_relation_prompts)
QUESTIONS = ("What is the {relation} of {subject}?", "Tell me the {relation} of {subject}.",
             "Q: Which {relation} is linked to {subject}? A:")
DS = ("mcf", "zsre", "mquake", "rwku")
root, out = Path(a.root), Path(a.out); out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
norm = lambda t: " ".join(str(t).casefold().split())

# ---------------------------------------------------------------- exclusions over all benchmarks
protected, test_text, man = set(), set(), {}
for ds in DS:
    d = root / ds / f"seed{a.seed}"
    man[ds] = json.loads((d / "manifest.json").read_text())
    protected |= {norm(f["subject"]) for f in man[ds]["facts"] if str(f.get("subject", "")).strip()}
    for s in ("eval", "null", "perturbed"):
        test_text |= {norm(p["prompt"]) for p in json.loads((d / f"probes_{s}.json").read_text()) if p["prompt"]}
protected = sorted(p for p in protected if len(p) >= 3)
ok = lambda t: norm(t) not in test_text and not any(s in norm(t) for s in protected)

# ---------------------------------------------------------------- probe data
forget, retain = mcf.load_official_eval_records("data/multi_counterfact.json", probes_lib.FORGET_NUM,
                                                probes_lib.OFFICIAL_RETAIN_NUM, a.seed, "official")
sampled = {int(r["case_id"]) for r in forget + retain}
pool = sorted((r for r in mcf.load_mcf("data/multi_counterfact.json") if int(r["case_id"]) not in sampled),
              key=lambda r: int(r["case_id"]))
train_pool, _ = probes_lib._halves(pool)
rng = random.Random(0)
by_rel, scaf, seen = {}, {}, set()
for rec in train_pool:
    rr = probes_lib._mcf_rr(rec); rel = str(rr["relation_id"]); subj = rr["subject"]
    texts = [rr["prompt"].format(subj)] + list(rec.get("paraphrase_prompts", [])) \
        + list(rec.get("neighborhood_prompts", [])) + list(rec.get("attribute_prompts", []))
    for t in texts:
        if norm(t) not in seen and ok(t):
            seen.add(norm(t)); by_rel.setdefault(rel, []).append(t)
    if rel in RELATION_NOUN_PHRASES and ok(subj):
        noun = RELATION_NOUN_PHRASES[rel]
        views = [rng.choice(CALIBRATION_SCAFFOLDS).format(subject=subj, relation=noun),
                 rng.choice(QUESTIONS).format(subject=subj, relation=noun)]
        if rel in RELATION_ALTERNATE_TEMPLATES:
            views.append(rng.choice(RELATION_ALTERNATE_TEMPLATES[rel]).format(subj))
        for t in views:
            if norm(t) not in seen and ok(t):
                seen.add(norm(t)); scaf.setdefault(rel, []).append(t)
rels = sorted(by_rel)
data = []
for i, r in enumerate(rels):
    rng.shuffle(by_rel[r]); rng.shuffle(scaf.setdefault(r, []))
    kept = by_rel[r][:a.cap] + scaf[r][:a.scaffold_cap]
    kept += [t + rng.choice(SUFFIXES) for t in kept[:int(a.suffix_share * len(kept))]]
    data += [(t, i) for t in kept]
print(json.dumps({"phase": "probe_data", "relations": len(rels), "prompts": len(data),
                  "protected_subjects": len(protected), "test_prompts": len(test_text)}), flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
model_path = man["mcf"]["model_path"]
tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True,
                                             attn_implementation="eager").cuda().eval()
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
ids = [tok(t)["input_ids"] for t, _ in data]
X = F.normalize(capture_last_token_states(model, ids, [a.layer], batch_size=128, pad_id=pad)[0].float(), dim=-1)
y = torch.tensor([l for _, l in data])
held = torch.tensor([k % 10 == 0 for k in range(len(data))])
acc = {}
for l2 in (1e-5, 1e-4):
    w, b = fit_softmax_probe(X[~held], y[~held], len(rels), l2, "cuda")
    acc[l2] = float(((X[held] @ w.T + b).argmax(1) == y[held]).float().mean())
l2 = max(acc, key=acc.get)
W, B = fit_softmax_probe(X, y, len(rels), l2, "cuda")
del model; torch.cuda.empty_cache()
print(json.dumps({"phase": "probe", "heldout_accuracy": acc, "l2": l2, "seconds": round(time.time() - t0)}), flush=True)

# ---------------------------------------------------------------- per benchmark
def probs_of(d, s):
    st = torch.load(d / f"states_{s}.pt", map_location="cpu", weights_only=False, mmap=True)
    x = st["states"][st["layers"].index(a.layer)].float()
    return (F.normalize(x, dim=-1) @ W.T + B).softmax(-1)

QS = (0.05, 0.10, 0.15, 0.20)
results = {}
for ds in DS:
    d = root / ds / f"seed{a.seed}"
    art = torch.load(Path(man[ds]["router_dir"]) / "fact_association_embeddings.pt", map_location="cpu",
                     weights_only=False)
    thr, margin = runtime_rule(art)
    P = {s: json.loads((d / f"probes_{s}.json").read_text()) for s in ("train", "eval", "null", "perturbed")}
    R = {s: torch.load(d / f"router_{s}.pt", map_location="cpu", weights_only=False) for s in P}
    pr = {s: probs_of(d, s) for s in P}
    N = len(man[ds]["facts"])
    # profile: mean distribution over each fact's own training positives
    prof = torch.zeros(N, len(rels)); cnt = torch.zeros(N)
    for k, p in enumerate(P["train"]):
        if p["should_fire"]:
            prof[p["true_facts"][0]] += pr["train"][k]; cnt[p["true_facts"][0]] += 1
    prof = prof / cnt.clamp_min(1)[:, None]
    ungated = cnt == 0                                    # no training positive: never gated
    score = {s: (pr[s] @ prof.T).clamp_min(1e-30).log() for s in P}
    own = [(p["true_facts"][0], float(score["train"][k, p["true_facts"][0]]))
           for k, p in enumerate(P["train"]) if p["should_fire"]]
    eff = torch.exp(-(prof.clamp_min(1e-12).log() * prof).sum(1))   # effective number of relations

    def evaluate(cut):
        res = {}
        for s in ("eval", "null", "perturbed"):
            elig = R[s]["eligible"].bool()
            if cut is not None:
                elig = elig & ((score[s] >= cut[None, :]) | ungated[None, :])
            res[s] = decide_routes(R[s]["logits"].float(), elig, thr, margin)
        q = P["eval"]; dd = res["eval"]; pos = [k for k, p in enumerate(q) if p["should_fire"]]
        rec = sum(bool(dd["active"][k]) and int(dd["fact"][k]) in q[k]["true_facts"] for k in pos) / max(1, len(pos))
        fp = sum(bool(dd["active"][k]) for k, p in enumerate(q) if not p["should_fire"])
        q = P["null"]; dd = res["null"]; neg = [k for k, p in enumerate(q) if not p["meta"].get("same_answer_group")]
        nf = sum(bool(dd["active"][k]) for k in neg)
        q = P["perturbed"]; dd = res["perturbed"]; pp = [k for k, p in enumerate(q) if p["should_fire"]]
        prc = sum(bool(dd["active"][k]) and int(dd["fact"][k]) in q[k]["true_facts"] for k in pp) / max(1, len(pp))
        return {"recall": round(rec, 3), "eval_false_fires": fp, "null_fires": nf, "null_n": len(neg),
                "perturbed_recall": round(prc, 3), "n_should_fire": len(pos)}

    base = evaluate(None)
    same = bool(torch.equal(decide_routes(R["eval"]["logits"].float(), R["eval"]["eligible"].bool(), thr, margin)["active"].bool(),
                            R["eval"]["active"].bool()))
    rows = {"no gate": base}
    vals = torch.tensor([v for _, v in own]).sort().values
    for q in QS:
        pooled = torch.full((N,), float(vals[int(q * (len(vals) - 1))]))
        rows[f"pooled q={q}"] = evaluate(pooled)
        per = torch.full((N,), -1e9)
        for f in range(N):
            v = sorted(val for g, val in own if g == f)
            if v:
                per[f] = v[int(q * (len(v) - 1))]
        rows[f"per-fact q={q}"] = evaluate(per)
    results[ds] = {"rows": rows, "reproduces_cached_routes": same, "facts": N,
                   "profile_effective_relations_median": round(float(eff.median()), 2),
                   "profile_effective_relations_mean": round(float(eff.mean()), 2)}
    print(json.dumps({"phase": "dataset", "ds": ds, "reproduces_cached_routes": same,
                      "profile_eff_relations_median": results[ds]["profile_effective_relations_median"]}), flush=True)
    print(f"{'rule':16s} recall  pert  evalFP  null", flush=True)
    for k, v in rows.items():
        print(f"{k:16s} {v['recall']:.3f}  {v['perturbed_recall']:.3f}  {v['eval_false_fires']:5d}  {v['null_fires']}/{v['null_n']}", flush=True)
    del pr, score

(out / "universal_gate_check.json").write_text(json.dumps({"probe": {"relations": rels, "prompts": len(data),
    "heldout_accuracy": acc, "l2": l2}, "results": results}, indent=1))
print("done", round(time.time() - t0), "s", flush=True)
