#!/usr/bin/env python3
"""Held-out negatives of the paper's router: clean the labels, then relation gate + per-head bias
(analysis only; the paper's router is unchanged).

Step 1, clean. A held-out negative of head i is ambiguous, and dropped from training and from
the counts, if
  * its text contains the fact's answer ("The Grande Arche is a popular tourist destination in
    France" for a France fact), or
  * it is a subject transplant from a relation whose answers overlap the fact's: the code's
    RELATION_ANSWER_GROUPS, with country, place and continent (P30) merged into one geography
    group ("the continent of Grande Arche", "the location of Palmer Archipelago").
Step 2, gate + bias. Heads may fire only where the relation gate passes
(outputs/router_diag_ours_relgate_v1: the same heads with artifact["relation_gate"]); each
head's bias is then raised just above its highest calibration negative that passes the gate
(max_neg) or its second highest (max_neg_1: one calibration negative may fire).

Protocol: biases use the calibration split only; the audit split, the official evaluation and
the no-association prompts are never used to fit or calibrate. "cal+audit" repeats the
counting of confusion_train.png (402 / 615 for the shipped router) and is in-sample for the bias.

    python -u scripts/router_negative_clean_gate.py
    # -> outputs/router_negative_fix_v1/mcf_seed1.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import unicodedata

import torch
from torch.nn import functional as F

from linear_router import RELATION_ANSWER_GROUPS, decide_routes
from router_negative_fix import fit_heads

GEO = set(RELATION_ANSWER_GROUPS["place"]) | set(RELATION_ANSWER_GROUPS["country"]) | {"P30"}


def overlap(rel_a, rel_b):
    a, b = str(rel_a), str(rel_b)
    if a == b or (a in GEO and b in GEO):
        return True
    return any(a in g and b in g for g in RELATION_ANSWER_GROUPS.values())


def norm(text):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]+", " ", text.casefold()).split()


def contains(text, answer):
    t, a = norm(text), norm(answer)
    return bool(a) and any(t[k:k + len(a)] == a for k in range(len(t) - len(a) + 1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", default="outputs/router_diag_ours_control_v1/mcf/seed1")
    ap.add_argument("--relgate", default="outputs/router_diag_ours_relgate_v1/mcf/seed1")
    ap.add_argument("--out", default="outputs/router_negative_fix_v1/mcf_seed1")
    ap.add_argument("--gamma", type=float, default=1e-4)
    ap.add_argument("--gate-probs", default="0.3,0.5,0.7,0.8", help="stricter relation-gate cutoffs, p(fact's relation)")
    args = ap.parse_args(argv)

    ctl, gate = Path(args.control), Path(args.relgate)
    man = json.loads((ctl / "manifest.json").read_text())
    router_dir = Path(man["router_dir"])
    art = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    facts = art["facts"]
    n = len(facts)
    layer = int(man["router_layer"])
    mu = torch.as_tensor(art["feature_mean"]).float()
    w0 = torch.as_tensor(art["router_weight"]).float().T
    b0 = torch.as_tensor(art["router_bias"]).float()
    dataset = json.loads((router_dir / "linear_router_dataset.json").read_text())

    def load(name):
        probes = json.loads((ctl / f"probes_{name}.json").read_text())
        g_probes = json.loads((gate / f"probes_{name}.json").read_text())
        assert [p["prompt"] for p in probes] == [p["prompt"] for p in g_probes], name
        st = torch.load(ctl / f"states_{name}.pt", weights_only=False, mmap=True)
        u = F.normalize(st["states"][list(st["layers"]).index(layer)].float(), dim=-1) - mu
        r_c = torch.load(ctl / f"router_{name}.pt", weights_only=False)
        r_g = torch.load(gate / f"router_{name}.pt", weights_only=False)
        return {"probes": probes, "gate_probes": g_probes, "u": u, "eligible": r_c["eligible"],
                "rel_pass": r_g["relation_pass"], "rel_logp": r_g["relation_log_prob"]}

    S = {k: load(k) for k in ("train", "aug", "eval", "null")}
    tr = S["train"]
    assert [d["prompt"] for d in dataset] == [p["prompt"] for p in tr["probes"]]

    # ---- labels: +1 own positive, -1 clean own negative, -2 ambiguous own negative, 0 otherwise
    lab = torch.zeros(len(tr["probes"]), n, dtype=torch.long)
    why = {"answer_in_text": 0, "overlapping_relation": 0}
    for k, (p, d) in enumerate(zip(tr["probes"], dataset)):
        if p["should_fire"]:
            for i in p["true_facts"]:
                lab[k, i] = 1
            continue
        for i in (p.get("meta") or {}).get("negative_for") or []:
            f = facts[i]
            if contains(p["prompt"], f["object"]):
                lab[k, i] = -2
                why["answer_in_text"] += 1
            elif d["kind"] == "subject_transplant" and overlap(f["relation"], d.get("donor_relation")):
                lab[k, i] = -2
                why["overlapping_relation"] += 1
            else:
                lab[k, i] = -1
    split = [p["split"] for p in tr["probes"]]
    m_fit = torch.tensor([s == "fit" for s in split])
    m_cal = torch.tensor([s == "calibration" for s in split])
    m_aud = torch.tensor([s == "audit" for s in split])
    m_held = m_cal | m_aud

    def own_counts(z, allow, msk, clean=True):
        fire = (z >= 0) & allow
        neg = (lab == -1) if clean else (lab <= -1)
        out = {}
        for name, mm in msk.items():
            nn = neg & mm[:, None]
            pp = (lab == 1) & mm[:, None]
            out[name] = {"neg_fire": int((fire & nn).sum()), "neg": int(nn.sum()),
                         "pos_fire": int((fire & pp).sum()), "pos": int(pp.sum())}
        return out

    def per_head_shift(z_tr, allow, k_allowed):
        """Raise each head's bias above its (k_allowed+1)-th highest clean calibration negative."""
        shift = torch.zeros(n)
        cand = (lab == -1) & m_cal[:, None] & allow
        for i in range(n):
            s = z_tr[cand[:, i], i].sort(descending=True).values
            if s.numel() > k_allowed:
                top = float(s[k_allowed])
                if top >= 0:
                    shift[i] = -(top + 1e-3)
        return shift

    null_groups = [p["group"] for p in S["null"]["gate_probes"]]
    must_not = torch.tensor([g != "null_protected_same_answer_type" for g in null_groups])

    def evaluate(w, b, use_gate):
        z = {k: S[k]["u"] @ w + b for k in ("train", "eval", "null")}
        allow_tr = tr["eligible"] & (tr["rel_pass"] if use_gate else True)
        res = {"own": own_counts(z["train"], allow_tr, {"cal+audit": m_held, "audit": m_aud}),
               "own_uncleaned": own_counts(z["train"], allow_tr, {"cal+audit": m_held}, clean=False)}
        ev = S["eval"]
        el = ev["eligible"] & (ev["rel_pass"] if use_gate else True)
        dec = decide_routes(z["eval"], el, 0.0, 0.5)
        fire, fact = dec["active"].cpu(), dec["fact"].cpu()
        should = torch.tensor([p["should_fire"] for p in ev["probes"]])
        right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"]
                              for k, p in enumerate(ev["probes"])])
        res["official"] = {"forget_recall": float(right[should].float().mean()),
                           "retain_fires": int((fire & ~should).sum()), "retain": int((~should).sum())}
        nu = S["null"]
        dn = decide_routes(z["null"], nu["eligible"] & (nu["rel_pass"] if use_gate else True), 0.0, 0.5)["active"].cpu()
        res["no_association"] = {"fires": int((dn & must_not).sum()), "of": int(must_not.sum()),
                                 "same_answer_type_fires": int((dn & ~must_not).sum())}
        return res, z

    out = {"cleaning": {**why, "held_out_negatives": int(((lab <= -1) & m_held[:, None]).sum()),
                        "clean_held_out_negatives": int(((lab == -1) & m_held[:, None]).sum())},
           "variants": {}}

    def run(name, w, b, use_gate, rule=None):
        res, z = evaluate(w, b, use_gate)
        if rule is not None:
            allow = tr["eligible"] & (tr["rel_pass"] if use_gate else True)
            b = b + per_head_shift(z["train"], allow, {"max_neg": 0, "max_neg_1": 1}[rule])
            res, _ = evaluate(w, b, use_gate)
        out["variants"][name] = res
        o = res["own"]
        print(json.dumps({name: {"cal+audit": o["cal+audit"], "audit": o["audit"], **res["official"],
                                 "no_assoc": res["no_association"]}}), flush=True)

    run("shipped", w0, b0, False)
    run("shipped + per-head bias", w0, b0, False, "max_neg")
    run("shipped + relation gate", w0, b0, True)
    run("shipped + relation gate + per-head bias", w0, b0, True, "max_neg")
    run("shipped + relation gate + per-head bias (1 allowed)", w0, b0, True, "max_neg_1")

    # stricter gate cutoffs, shipped heads + per-head bias
    shipped_pass = {k: S[k]["rel_pass"] for k in ("train", "eval", "null")}
    for prob in [float(x) for x in args.gate_probs.split(",") if x]:
        for k in ("train", "eval", "null"):
            S[k]["rel_pass"] = S[k]["rel_logp"] >= torch.log(torch.tensor(prob))
        run(f"shipped + relation gate p>={prob:g} + per-head bias", w0, b0, True, "max_neg")
    for k in ("train", "eval", "null"):
        S[k]["rel_pass"] = shipped_pass[k]
    out["gate_cutoff_shipped_log_prob"] = float(torch.as_tensor(art.get("relation_gate", {}).get("min_log_prob", float("nan")))) if art.get("relation_gate") else None

    # refit on the fit split with the ambiguous negatives dropped (+ aug same-subject negatives)
    ag = S["aug"]
    ag_lab = torch.zeros(len(ag["probes"]), n, dtype=torch.long)
    for k, p in enumerate(ag["probes"]):
        for i in (p.get("meta") or {}).get("negative_for") or []:
            if p.get("probe_type") == "same_subject_other_relation" and not overlap(
                    facts[i]["relation"], (p.get("meta") or {}).get("relation")) and not contains(p["prompt"], facts[i]["object"]):
                ag_lab[k, i] = -1
    keep_ag = (ag_lab == -1).any(1)
    x = torch.cat([tr["u"][m_fit], ag["u"][keep_ag]])
    l = torch.cat([lab[m_fit], ag_lab[keep_ag]])
    el = torch.cat([tr["eligible"][m_fit], ag["eligible"][keep_ag]])
    w, b = fit_heads(x, l == 1, el & ((l == 1) | (l == -1)), args.gamma)
    # put the refit on the shipped recall target first (98% of calibration positives), as the paper
    zc = (tr["u"][m_cal] @ w + b)
    pos = zc[(lab[m_cal] == 1)]
    b = b - float(torch.quantile(pos, 0.02))
    run("clean refit (98% recall bias)", w, b, False)
    run("clean refit + relation gate + per-head bias", w, b, True, "max_neg")
    run("clean refit + relation gate + per-head bias (1 allowed)", w, b, True, "max_neg_1")

    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    c = out["cleaning"]
    lines = [f"Held-out negatives: {c['held_out_negatives']}; ambiguous, dropped: {c['answer_in_text']} contain the answer, "
             f"{c['overlapping_relation']} ask an overlapping relation; clean: {c['clean_held_out_negatives']}.", "",
             "| variant | cal+audit negatives fired (clean) | uncleaned | audit negatives fired | audit positives fired | official forget recall | retain fires | no-association fires |",
             "|---|---|---|---|---|---|---|---|"]
    for k, v in out["variants"].items():
        ca, au, un, of, na = v["own"]["cal+audit"], v["own"]["audit"], v["own_uncleaned"]["cal+audit"], v["official"], v["no_association"]
        lines.append(f"| {k} | {ca['neg_fire']}/{ca['neg']} | {un['neg_fire']}/{un['neg']} | {au['neg_fire']}/{au['neg']} | "
                     f"{au['pos_fire']}/{au['pos']} | {of['forget_recall']:.3f} | {of['retain_fires']}/{of['retain']} | "
                     f"{na['fires']}/{na['of']} |")
    dest.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
