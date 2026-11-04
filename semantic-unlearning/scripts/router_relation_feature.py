#!/usr/bin/env python3
"""Two attacks on the same-subject false fires, scored on TEST prompts only (analysis only).

The classifier decides on the last prompt token, whose block-19 state is mostly the subject.
  span      feature = [unit last-token state ; unit mean state over the tokens AFTER the subject]
            (the relation words), both from the same forward pass; falls back to the last token
            alone when no subject is found
  weighted  the last-token feature, but same-subject negatives (hard negatives and subject
            transplants of the head's own subject) weighted --neg-weight x in the head's loss
  both      span feature + weighting
Heads: one logistic head per fact as the paper's (class-balanced BCE, L2), trained on the C
data (a data-grid refit's linear_router_dataset.json: fit split), per-head bias placed on the
calibration split, scored with the subject gate on the official evaluation prompts and the
no-association prompts (prompts the heads trained or calibrated on left out).

    python -u scripts/router_relation_feature.py --dataset mcf --seed 1 \
        --dataset-json outputs/router_data_grid_v1/mcf/seed1/p40_n40/router/linear_router_dataset.json
    # -> outputs/router_relation_feature_v1/<ds>_seed<S>.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from linear_router import decide_routes
from router_negative_clean_gate import contains, overlap


def norm(t):
    return " ".join(str(t or "").casefold().split())


def find_span(ids, subjects, tok):
    best = None
    for s in subjects:
        for variant in (" " + s.strip(), s.strip()):
            pat = tok(variant, add_special_tokens=False)["input_ids"]
            if not pat:
                continue
            for i in range(len(ids) - len(pat), -1, -1):
                if ids[i:i + len(pat)] == pat:
                    best = max(best or 0, i + len(pat))
                    break
    if best is None or best >= len(ids):
        return None
    return best, len(ids)


@torch.no_grad()
def extract(model, tok, ids, spans, layer, batch=48):
    """(last-token states [P, d], span-mean states [P, d]) at block `layer`."""
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    d = model.config.hidden_size
    last = torch.empty(len(ids), d)
    span = torch.empty(len(ids), d)
    cap = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda m, a, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for b in range(0, len(order), batch):
        bt = order[b:b + batch]
        w = max(len(ids[i]) for i in bt)
        inp = torch.full((len(bt), w), pad)
        m = torch.zeros_like(inp)
        for r, i in enumerate(bt):
            inp[r, :len(ids[i])] = torch.tensor(ids[i])
            m[r, :len(ids[i])] = 1
        model.model(input_ids=inp.cuda(), attention_mask=m.cuda(), use_cache=False)
        hs = cap["h"].float()
        for r, i in enumerate(bt):
            n = len(ids[i])
            last[i] = hs[r, n - 1].cpu()
            st, en = spans[i] if spans[i] is not None else (n - 1, n)
            span[i] = hs[r, st:en].mean(0).cpu()
    h.remove()
    return last, span


def fit_heads(x, y, mask, weight_mult, gamma, iters=300):
    yb, m = y.float(), mask.float()
    n_pos = (yb * m).sum(0).clamp_min(1)
    n_neg = ((1 - yb) * m * weight_mult).sum(0).clamp_min(1)
    weight = m * (yb / (2 * n_pos) + (1 - yb) * weight_mult / (2 * n_neg))
    w = torch.zeros(x.shape[1], y.shape[1], requires_grad=True)
    b = torch.zeros(y.shape[1], requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(x @ w + b, yb, reduction="none") * weight).sum() + gamma * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--dataset-json", required=True, help="a router's linear_router_dataset.json (the training prompts)")
    ap.add_argument("--out", default="outputs/router_relation_feature_v1")
    ap.add_argument("--neg-weight", type=float, default=10.0)
    ap.add_argument("--gamma", type=float, default=1e-4)
    args = ap.parse_args(argv)
    cache = Path(args.root) / args.dataset / f"seed{args.seed}"
    man = json.loads((cache / "manifest.json").read_text())
    art = torch.load(Path(man["router_dir"]) / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    facts, layer = art["facts"], int(man["router_layer"])
    n = len(facts)
    fid = {f["id"]: i for i, f in enumerate(facts)}
    subjects = [str(f.get("subject", "")) for f in facts]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(man["model_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(man["model_path"], dtype=torch.bfloat16, local_files_only=True,
                                                 attn_implementation="eager").cuda().eval()

    # ---- training prompts (the C data) with labels and the same-subject flag
    data = json.loads(Path(args.dataset_json).read_text())
    rows = [e for e in data if e["split"] in ("fit", "calibration")]
    lab = torch.zeros(len(rows), n, dtype=torch.long)
    same_subj = torch.zeros(len(rows), n, dtype=torch.bool)
    for k, e in enumerate(rows):
        if e["kind"] == "positive":
            lab[k, fid[e["owner_fact_id"]]] = 1
            continue
        for fact_id in e.get("negative_for") or []:
            i = fid.get(fact_id)
            if i is None:
                continue
            f, donor = facts[i], str(e.get("donor_relation") or "")
            amb = contains(e["prompt"], f["object"]) or (e["kind"] == "subject_transplant" and donor.startswith("P")
                                                         and str(f["relation"]).startswith("P") and overlap(f["relation"], donor))
            lab[k, i] = -2 if amb else -1
            same_subj[k, i] = subjects[i].casefold() in e["prompt"].casefold()
    is_fit = torch.tensor([e["split"] == "fit" for e in rows])
    is_cal = ~is_fit
    tr_ids = [tok(e["prompt"])["input_ids"] for e in rows]
    tr_elig = torch.zeros(len(rows), n, dtype=torch.bool)
    for k, e in enumerate(rows):
        for i in range(n):
            tr_elig[k, i] = subjects[i].casefold() in e["prompt"].casefold()
    tr_spans = [find_span(x, [subjects[i] for i in tr_elig[k].nonzero().flatten().tolist()], tok) for k, x in enumerate(tr_ids)]
    tr_last, tr_span = extract(model, tok, tr_ids, tr_spans, layer)
    trained = {norm(e["prompt"]) for e in rows}

    # ---- test prompts
    T = {}
    for name in ("eval", "null"):
        probes = json.loads((cache / f"probes_{name}.json").read_text())
        elig = torch.load(cache / f"router_{name}.pt", weights_only=False)["eligible"].bool()
        ids = [p["input_ids"] if p["input_ids"] is not None else tok(p["prompt"])["input_ids"] for p in probes]
        spans = [find_span(x, [subjects[i] for i in elig[k].nonzero().flatten().tolist()], tok) for k, x in enumerate(ids)]
        last, span = extract(model, tok, ids, spans, layer)
        T[name] = {"probes": probes, "eligible": elig, "last": last, "span": span,
                   "found": sum(1 for k, s in enumerate(spans) if s is not None and bool(elig[k].any()))}
    del model
    torch.cuda.empty_cache()

    def feature(kind, last, span):
        u = F.normalize(last, dim=-1)
        if kind == "last":
            return u
        return torch.cat([u, F.normalize(span, dim=-1)], dim=-1)

    should = torch.tensor([p["should_fire"] for p in T["eval"]["probes"]])
    null_group = [p["group"] for p in T["null"]["probes"]]
    null_text = [norm(p.get("prompt")) for p in T["null"]["probes"]]
    groups = ["null_generic", "null_ppl_prefix", "null_unseen_subject", "null_protected_other_relation"]
    results = {}
    for variant, kind, wmult in (("last token (C data)", "last", 1.0), ("span feature", "span", 1.0),
                                 ("same-subject negatives x%g" % args.neg_weight, "last", args.neg_weight),
                                 ("span + weighted", "span", args.neg_weight)):
        x_all = feature(kind, tr_last, tr_span)
        mu = x_all[is_fit].mean(0)
        x = x_all - mu
        y = lab == 1
        mask = is_fit[:, None] & (lab != 0) & tr_elig
        wm = torch.where(same_subj & (lab == -1), torch.full_like(x[:, :1].expand(-1, n), wmult), torch.ones(len(rows), n))
        w, b = fit_heads(x[is_fit], y[is_fit], mask[is_fit], wm[is_fit], args.gamma)
        z_cal = x[is_cal] @ w + b
        # per-head bias on calibration negatives the gate lets through
        cand = (lab[is_cal] == -1) & tr_elig[is_cal]
        bias = b.clone()
        for i in range(n):
            s = z_cal[cand[:, i], i]
            if s.numel() and float(s.max()) >= 0:
                bias[i] = b[i] - (float(s.max()) + 1e-3)
        res = {}
        for name in ("eval", "null"):
            xt = feature(kind, T[name]["last"], T[name]["span"]) - mu
            z = xt @ w + bias
            dec = decide_routes(z, T[name]["eligible"], 0.0, 0.5)
            fire, fact = dec["active"].cpu(), dec["fact"].cpu()
            if name == "eval":
                right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"]
                                      for k, p in enumerate(T[name]["probes"])])
                res["forget_recall"] = float(right[should].float().mean())
                res["retain_fires"] = f"{int((fire & ~should).sum())}/{int((~should).sum())}"
            else:
                for g in groups:
                    idx = [k for k in range(len(fire)) if null_group[k] == g and null_text[k] not in trained]
                    res[g] = f"{sum(int(fire[k]) for k in idx)}/{len(idx)}"
        results[variant] = res
        print(json.dumps({variant: res}), flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"dataset": args.dataset, "seed": args.seed, "dataset_json": args.dataset_json, "layer": layer,
            "span_found": {k: f"{v['found']}/{int(v['eligible'].any(1).sum())}" for k, v in T.items()}, "results": results}
    (out / f"{args.dataset}_seed{args.seed}.json").write_text(json.dumps(meta, indent=2) + "\n")
    lines = [f"### {args.dataset} seed {args.seed}: heads on the C data, test prompts only (span found on "
             f"{meta['span_found']['eval']} eval / {meta['span_found']['null']} null eligible prompts)", "",
             "| variant | forget recall | retain false fires | generic | perplexity | unseen subject | **protected subject, other relation** |",
             "|---|---|---|---|---|---|---|"]
    for v, r in results.items():
        lines.append(f"| {v} | {r['forget_recall']:.3f} | {r['retain_fires']} | {r['null_generic']} | {r['null_ppl_prefix']} | "
                     f"{r['null_unseen_subject']} | **{r['null_protected_other_relation']}** |")
    (out / f"{args.dataset}_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
