#!/usr/bin/env python3
"""Clean the held-out negatives, then relation gate + per-head bias, on MCF, zsRE and MQuAKE
(analysis only; the paper's routers are unchanged). router_negative_clean_gate.py on MCF with
the MCF-only relation gate; this uses one probe for every benchmark.

Relation probe: outputs/relation_gate_analysis_v1/universal_probe.pt (relation_gate_universal_check.py:
34 Wikidata relations at block 19, trained on prompts that name no protected subject of any
benchmark and are no test prompt of any benchmark). Two gates from it:
  labelled  head i may fire if p(relation of fact i | prompt) >= cutoff (facts whose relation the
            probe does not know are never gated; zsRE has no Wikidata relations, so none)
  profile   no labels: head i may fire if log sum_r p(r | prompt) profile_i(r) >= cutoff,
            profile_i = mean probe distribution over the fact's fit-split positives
Cutoff: the q-quantile of the same score on the heads' fit-split positives (q of them blocked),
or an absolute probability for the labelled gate.

Cleaning (router_negative_clean_gate.py): a held-out negative of head i is dropped if its text
contains the fact's answer, or it is a transplant from a Wikidata relation whose answers overlap
the fact's (zsRE has none, so only the answer test applies there).
Per-head bias: raised above the head's highest clean calibration negative that the gates let
through. Audit, official evaluation and no-association prompts are never fitted or calibrated on.

    python -u scripts/router_negative_gate_all.py
    # -> outputs/router_negative_fix_v1/{mcf,zsre,mquake}_seed1_gates.{json,md}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from linear_router import decide_routes
from router_negative_clean_gate import contains, overlap
from fit_relation_gate import fit_softmax_probe


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--probe", default="outputs/relation_gate_analysis_v1/universal_probe.pt")
    ap.add_argument("--datasets", default="mcf,zsre,mquake")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quantiles", default="0.05,0.15,0.3")
    ap.add_argument("--probs", default="0.3,0.5")
    ap.add_argument("--out", default="outputs/router_negative_fix_v1")
    ap.add_argument("--tag", default="gates", help="output name: <ds>_seed<seed>_<tag>")
    ap.add_argument("--gates", default="probe", help="probe (relation probe gates), contrast (positive/negative "
                                                      "combination gate from the router's own prompts), or both")
    ap.add_argument("--contrast-dims", default="16,64")
    ap.add_argument("--stack-dim", type=int, default=64)
    ap.add_argument("--stack-probe-q", type=float, default=0.05)
    ap.add_argument("--stack-contrast-q", type=float, default=0.15)
    ap.add_argument("--heads-from", default=None, metavar="DS=ROUTER_DIR",
                    help="score another router's heads on the cache (e.g. a data-grid refit); repeatable")
    args = ap.parse_args(argv)

    probe = torch.load(args.probe, map_location="cpu", weights_only=False)
    pw, pb, rels = probe["weight"].float(), probe["bias"].float(), list(probe["relations"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for ds in args.datasets.split(","):
        cache = Path(args.root) / ds / f"seed{args.seed}"
        man = json.loads((cache / "manifest.json").read_text())
        rdir = Path(man["router_dir"])
        art = torch.load(rdir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
        facts, layer = art["facts"], int(man["router_layer"])
        n = len(facts)
        mu = torch.as_tensor(art["feature_mean"]).float()
        heads_from = dict(x.split("=", 1) for x in (args.heads_from or "").split(",") if "=" in x)
        if ds in heads_from:
            alt = torch.load(Path(heads_from[ds]) / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
            if [f["id"] for f in alt["facts"]] != [f["id"] for f in facts]:
                raise SystemExit(f"{heads_from[ds]} lists other facts")
            art = {**art, "router_weight": alt["router_weight"], "router_bias": alt["router_bias"],
                   "feature_mean": alt["feature_mean"], "feature_components": alt.get("feature_components")}
            mu = torch.as_tensor(art["feature_mean"]).float()
            print(json.dumps({ds: "heads from", "router": heads_from[ds]}), flush=True)
        w0 = torch.as_tensor(art["router_weight"]).float().T
        b0 = torch.as_tensor(art["router_bias"]).float()
        dpath = rdir / "linear_router_dataset.json"
        dataset = json.loads(dpath.read_text()) if dpath.is_file() else None

        S = {}
        for name in ("train", "eval", "null"):
            probes = json.loads((cache / f"probes_{name}.json").read_text())
            st = torch.load(cache / f"states_{name}.pt", weights_only=False, mmap=True)
            h = F.normalize(st["states"][list(st["layers"]).index(layer)].float(), dim=-1)
            S[name] = {"probes": probes, "h": h, "p": (h @ pw.T + pb).softmax(-1),
                       "eligible": torch.load(cache / f"router_{name}.pt", weights_only=False)["eligible"]}
        for sec in S:
            S[sec]["u"] = S[sec]["h"] - mu                       # after a possible heads swap
        tr = S["train"]
        if dataset is None or [d["prompt"] for d in dataset] != [p["prompt"] for p in tr["probes"]]:
            dataset = [{"kind": "", "donor_relation": None} for _ in tr["probes"]]     # cleaning by answer text only
        split = [p["split"] for p in tr["probes"]]
        m_fit = torch.tensor([s == "fit" for s in split])
        m_cal = torch.tensor([s == "calibration" for s in split])
        m_aud = torch.tensor([s == "audit" for s in split])

        lab = torch.zeros(len(tr["probes"]), n, dtype=torch.long)
        why = {"answer_in_text": 0, "overlapping_relation": 0}
        for k, (p, d) in enumerate(zip(tr["probes"], dataset)):
            if p["should_fire"]:
                for i in p["true_facts"]:
                    lab[k, i] = 1
                continue
            for i in (p.get("meta") or {}).get("negative_for") or []:
                f = facts[i]
                donor = str(d.get("donor_relation") or "")
                if contains(p["prompt"], f["object"]):
                    lab[k, i], why["answer_in_text"] = -2, why["answer_in_text"] + 1
                elif (d["kind"] == "subject_transplant" and donor.startswith("P") and str(f["relation"]).startswith("P")
                      and overlap(f["relation"], donor)):
                    lab[k, i], why["overlapping_relation"] = -2, why["overlapping_relation"] + 1
                else:
                    lab[k, i] = -1

        # gate scores [P, N] per section
        rel_idx = torch.tensor([rels.index(str(f["relation"])) if str(f["relation"]) in rels else -1 for f in facts])
        known = rel_idx >= 0
        fitpos = (lab == 1) & m_fit[:, None]
        prof = torch.zeros(n, len(rels))
        for i in range(n):
            rows = fitpos[:, i]
            if rows.any():
                prof[i] = tr["p"][rows].mean(0)
        has_prof = fitpos.any(0)
        eff_rel = torch.exp(-(prof.clamp_min(1e-12).log() * prof).sum(1))[has_prof]

        def gate_scores(kind, sec):
            p = S[sec]["p"]
            if kind == "labelled":
                s = p[:, rel_idx.clamp_min(0)].clamp_min(1e-30).log()
                s[:, ~known] = float("inf")
            else:
                s = (p @ prof.T).clamp_min(1e-30).log()
                s[:, ~has_prof] = float("inf")
            return s

        def train_contrast(k, steps=500, lr=1e-2, wd=1e-4):
            """Positive/negative combination gate: one shared projection P (d -> k) and a prototype per
            fact (mean projected fit positive), trained so that a fact's fit positives score high and its
            same-subject fit negatives low. P is shared by all facts, so it has to encode which relation
            is asked rather than each fact; only the fit split is used."""
            x = tr["u"]
            pairs = ((lab == 1) | (lab == -1)) & m_fit[:, None] & tr["eligible"]
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

        contrast = {}

        def contrast_scores(k, sec):
            proj, proto = contrast[k]
            sc = F.normalize(S[sec]["u"] @ proj, dim=-1) @ proto.T
            sc[:, ~has_prof] = float("inf")
            return sc

        def contrast_cutoff(k, q):
            """q-quantile over calibration positives: the gate never trained on them."""
            sc = contrast_scores(k, "train")
            vals = sc[(lab == 1) & m_cal[:, None] & tr["eligible"] & torch.isfinite(sc)]
            return float(torch.quantile(vals, q)) if vals.numel() else float("-inf")

        def cutoff_from_quantile(kind, q):
            s = gate_scores(kind, "train")
            vals = s[fitpos & torch.isfinite(s)]
            return float(torch.quantile(vals, q)) if vals.numel() else float("-inf")

        def elig(sec, subject):
            return S[sec]["eligible"] if subject else torch.ones_like(S[sec]["eligible"])

        def evaluate(bias, allow, subject=True):
            z = {k: S[k]["u"] @ w0 + bias for k in S}
            res = {}
            fire_tr = (z["train"] >= 0) & elig("train", subject) & allow["train"]
            for name, mm in (("cal+audit", m_cal | m_aud), ("audit", m_aud)):
                neg, negall, pos = (lab == -1) & mm[:, None], (lab <= -1) & mm[:, None], (lab == 1) & mm[:, None]
                res[name] = {"neg_fire": int((fire_tr & neg).sum()), "neg": int(neg.sum()),
                             "uncleaned_fire": int((fire_tr & negall).sum()), "uncleaned": int(negall.sum()),
                             "pos_fire": int((fire_tr & pos).sum()), "pos": int(pos.sum())}
            ev = S["eval"]
            dec = decide_routes(z["eval"], elig("eval", subject) & allow["eval"], 0.0, 0.5)
            fire, fact = dec["active"].cpu(), dec["fact"].cpu()
            should = torch.tensor([p["should_fire"] for p in ev["probes"]])
            right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"]
                                  for k, p in enumerate(ev["probes"])])
            res["official"] = {"forget_recall": float(right[should].float().mean()),
                               "retain_fires": int((fire & ~should).sum()), "retain": int((~should).sum())}
            dn = decide_routes(z["null"], elig("null", subject) & allow["null"], 0.0, 0.5)["active"].cpu()
            res["no_association"] = {"fires": int(dn.sum()), "of": len(S["null"]["probes"])}
            return res, z

        def per_head_bias(z_tr, allow_tr, subject=True):
            shift = torch.zeros(n)
            cand = (lab == -1) & m_cal[:, None] & elig("train", subject) & allow_tr
            for i in range(n):
                s = z_tr[cand[:, i], i]
                if s.numel() and float(s.max()) >= 0:
                    shift[i] = -(float(s.max()) + 1e-3)
            return shift

        everyone = {k: torch.ones_like(S[k]["eligible"]) for k in S}
        variants = {}

        def run(name, allow, subject=True):
            res, z = evaluate(b0, allow, subject)
            variants[name] = res
            res_b, _ = evaluate(b0 + per_head_bias(z["train"], allow["train"], subject), allow, subject)
            variants[name + " + per-head bias"] = res_b
            for k in (name, name + " + per-head bias"):
                v = variants[k]
                print(json.dumps({ds: k, "cal+audit": v["cal+audit"], **v["official"], "no_assoc": v["no_association"]}), flush=True)

        run("shipped", everyone)
        gates = []
        modes = set(args.gates.split(","))
        if "probe" in modes and bool(known.any()):
            gates += [("labelled", f"q={q}", cutoff_from_quantile("labelled", float(q))) for q in args.quantiles.split(",")]
            gates += [("labelled", f"p>={p}", math.log(float(p))) for p in args.probs.split(",")]
        if "probe" in modes:
            gates += [("profile", f"q={q}", cutoff_from_quantile("profile", float(q))) for q in args.quantiles.split(",")]
        for kind, tag, cut in gates:
            run(f"{kind} gate {tag}", {k: gate_scores(kind, k) >= cut for k in S})
        if "stack" in modes:
            # the full stack at one fixed setting: subject gate x relation-probe gate x combination gate
            kd, qp, qc = args.stack_dim, args.stack_probe_q, args.stack_contrast_q
            contrast[kd] = train_contrast(kd)
            probe_ok = {sec: gate_scores("profile", sec) >= cutoff_from_quantile("profile", qp) for sec in S}
            cut_c = contrast_cutoff(kd, qc)
            comb_ok = {sec: contrast_scores(kd, sec) >= cut_c for sec in S}
            both = {sec: probe_ok[sec] & comb_ok[sec] for sec in S}
            run("subject + relation probe", probe_ok)
            run("subject + combination", comb_ok)
            run("subject + relation probe + combination", both)
            run("no subject gate: heads only", everyone, subject=False)
            run("no subject gate: relation probe + combination", both, subject=False)
        if "self" in modes:
            # self-made relation probe: class j = fact j's fit positives + the fit-split transplant
            # negatives built from fact j's prompts (fact j's relation asked of other subjects);
            # an extra class for the model-written negatives. No data outside the router's own.
            kd = args.stack_dim
            if kd not in contrast:
                contrast[kd] = train_contrast(kd)
            cut_c = contrast_cutoff(kd, args.stack_contrast_q)
            comb_ok = {sec: contrast_scores(kd, sec) >= cut_c for sec in S}
            # the dataset keeps a transplant's donor relation, not its donor fact: recover the donor
            # by template (prompt with the subject replaced by a placeholder) among the facts' positives
            norm_t = lambda t: " ".join(str(t).casefold().split())
            subj = [str(f["subject"]) for f in facts]
            templates = {}
            for p, d in zip(tr["probes"], dataset):
                if p["should_fire"]:
                    j = p["true_facts"][0]
                    if subj[j] and subj[j] in p["prompt"]:
                        templates.setdefault(norm_t(p["prompt"].replace(subj[j], "\x00")), set()).add(j)
            cls = torch.full((len(tr["probes"]),), -1, dtype=torch.long)
            matched = transplants = 0
            for k, (p, d) in enumerate(zip(tr["probes"], dataset)):
                if not bool(m_fit[k]):
                    continue
                if p["should_fire"]:
                    cls[k] = p["true_facts"][0]
                elif d.get("kind") == "subject_transplant":
                    transplants += 1
                    owners = (p.get("meta") or {}).get("negative_for") or []
                    donors = set()
                    for i in owners:
                        if subj[i] and subj[i] in p["prompt"]:
                            donors |= templates.get(norm_t(p["prompt"].replace(subj[i], "\x00")), set()) - {i}
                    if len(donors) == 1:
                        cls[k] = donors.pop()
                        matched += 1
                elif d.get("kind") == "extra_negative":
                    cls[k] = n
            print(json.dumps({ds: "transplant donors recovered", "matched": matched, "fit_transplants": transplants}), flush=True)
            rels_f = [str(f["relation"]) for f in facts]
            merges = [False] + ([True] if any(r.startswith("P") for r in rels_f) else [])
            for merge in merges:
                cmap = torch.arange(n + 1)
                if merge:
                    first = {}
                    for j, r in enumerate(rels_f):
                        if r.startswith("P"):
                            cmap[j] = first.setdefault(r, j)
                keep = cls >= 0
                y = cmap[cls[keep]]
                uniq = sorted(set(y.tolist()))
                remap = {c: t for t, c in enumerate(uniq)}
                yy = torch.tensor([remap[int(c)] for c in y])
                w_s, b_s = fit_softmax_probe(tr["u"][keep], yy, len(uniq), 1e-4, "cpu")
                col = torch.tensor([remap.get(int(cmap[i]), -1) for i in range(n)])
                known_s = col >= 0

                def self_scores(sec):
                    lp = (S[sec]["u"] @ w_s.T + b_s).log_softmax(-1)
                    sc = lp[:, col.clamp_min(0)]
                    sc[:, ~known_s] = float("inf")
                    return sc

                sc_tr = self_scores("train")
                vals = sc_tr[(lab == 1) & m_cal[:, None] & tr["eligible"] & torch.isfinite(sc_tr)]
                tag = "self probe" + (" (classes merged by Wikidata relation)" if merge else "")
                stats = {"classes": len(uniq), "transplant_rows": matched, "rows": int(keep.sum())}
                print(json.dumps({ds: tag, **stats}), flush=True)
                for q in args.quantiles.split(","):
                    cut = float(torch.quantile(vals, float(q)))
                    self_ok = {sec: self_scores(sec) >= cut for sec in S}
                    run(f"{tag} q={q}", self_ok)
                    run(f"{tag} q={q} + combination", {sec: self_ok[sec] & comb_ok[sec] for sec in S})
        if "contrast" in modes:
            for kdim in [int(v) for v in args.contrast_dims.split(",")]:
                contrast[kdim] = train_contrast(kdim)
                for q in args.quantiles.split(","):
                    cut = contrast_cutoff(kdim, float(q))
                    run(f"pos/neg combination gate k={kdim} q={q}", {sec: contrast_scores(kdim, sec) >= cut for sec in S})

        report = {"dataset": ds, "probe": args.probe, "cleaning": why, "facts": n,
                  "facts_with_known_relation": int(known.sum()),
                  "profile_effective_relations_median": float(eff_rel.median()) if eff_rel.numel() else None,
                  "variants": variants}
        stem = out_dir / f"{ds}_seed{args.seed}_{args.tag}"
        stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        c0 = variants["shipped"]["cal+audit"]
        lines = [f"# {ds} seed {args.seed}",
                 f"Held-out negatives {c0['uncleaned']}; dropped as ambiguous: {why['answer_in_text']} contain the answer, "
                 f"{why['overlapping_relation']} ask an overlapping relation; clean {c0['neg']}. Facts with a relation the "
                 f"probe knows: {int(known.sum())}/{n}. Relations per fact profile (median, effective): "
                 f"{report['profile_effective_relations_median']}. Probe: {args.probe}.", "",
                 "| variant | held-out negatives fired (uncleaned) | clean | audit clean | held-out positives fired | official forget recall | retain fires | no-association fires |",
                 "|---|---|---|---|---|---|---|---|"]
        for k, v in variants.items():
            ca, au, of, na = v["cal+audit"], v["audit"], v["official"], v["no_association"]
            lines.append(f"| {k} | {ca['uncleaned_fire']}/{ca['uncleaned']} | {ca['neg_fire']}/{ca['neg']} | "
                         f"{au['neg_fire']}/{au['neg']} | {ca['pos_fire']}/{ca['pos']} | {of['forget_recall']:.3f} | "
                         f"{of['retain_fires']}/{of['retain']} | {na['fires']}/{na['of']} |")
        stem.with_suffix(".md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
