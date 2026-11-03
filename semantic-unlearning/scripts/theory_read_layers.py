#!/usr/bin/env python3
"""Where an association can be read, without the subject gate (paper, Section 4.2).

From a router_diag cache (cache_router_states.py: last-token states at every block), on MCF:

  * at every block l, one logistic head per association on u^l (the classifier's feature:
    unit-normalised, centred state), trained like the shipped heads (class-balanced BCE +
    gamma ||w||^2, gamma = the shipped router's) but on EVERY training prompt, with no
    subject mask, so the head alone has to tell the association apart;
  * tested on the official evaluation prompts the heads never saw. For association i:
      pos       its request prompt and paraphrases
      same      its neighbourhood prompts: other subjects, same relation, same answer
      diff      neighbourhood prompts of the other associations with the same relation and
                a different answer
    AUC_same = P(z_i(pos) > z_i(same)), AUC_diff = P(z_i(pos) > z_i(diff)), mean over i.
    Output collapse predicts AUC_same < AUC_diff in late blocks: the two negative sets differ
    only in the answer;
  * the geometric version of the same contrast, ratio = mean ||u(pos) - u(same)|| /
    mean ||u(pos) - u(diff)||: 1 if the answer is not encoded, -> 0 under collapse.

    python -u scripts/theory_read_layers.py --cache-dir outputs/router_diag_ours_control_v1/mcf/seed1
    # -> <cache>/theory/read_layers.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def unit_centred(states, mean=None):
    u = F.normalize(states.float(), dim=-1)
    if mean is None:
        mean = u.mean(dim=0)
    return u - mean, mean


def fit_heads(x, y, gamma, iters):
    """x [P, d], y [P, N] in {0,1}; independent class-balanced logistic heads, L-BFGS."""
    n_pos = y.sum(0).clamp_min(1)
    n_neg = (1 - y).sum(0).clamp_min(1)
    weight = y / (2 * n_pos) + (1 - y) / (2 * n_neg)            # each head: half on each class
    w = torch.zeros(x.shape[1], y.shape[1], device=x.device, requires_grad=True)
    b = torch.zeros(y.shape[1], device=x.device, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=iters, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = x @ w + b
        loss = (F.binary_cross_entropy_with_logits(z, y, reduction="none") * weight).sum() \
            + gamma * (w ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def auc(pos, neg):
    if pos.numel() == 0 or neg.numel() == 0:
        return None
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0).float() + 0.5 * (diff == 0).float()).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--gamma", type=float, default=None, help="default: the shipped router's L2")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    cache = Path(args.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    router = torch.load(Path(manifest["router_dir"]) / "fact_association_embeddings.pt", map_location="cpu",
                        weights_only=False)
    facts = router["facts"]                    # the manifest's copy has no answers
    if [f["id"] for f in facts] != [f["id"] for f in manifest["facts"]]:
        raise SystemExit("router and cache list different associations")
    n = len(facts)
    gamma = args.gamma if args.gamma is not None else float(manifest["router_fit"]["selected_l2"])
    obj = [str(f.get("object", "")).strip().casefold() for f in facts]
    rel = [str(f.get("relation", "")) for f in facts]
    subj_to_fact = {str(f["subject"]): i for i, f in enumerate(facts)}

    train = json.loads((cache / "probes_train.json").read_text()) + json.loads((cache / "probes_aug.json").read_text())
    evals = json.loads((cache / "probes_eval.json").read_text())
    y = torch.zeros(len(train), n)
    for r, p in enumerate(train):
        if p["should_fire"]:
            for f in p["true_facts"]:
                y[r, f] = 1.0

    pos = {i: [] for i in range(n)}
    same = {i: [] for i in range(n)}
    for r, p in enumerate(evals):
        if p["group"] in ("forget_rewrite", "forget_paraphrase") and p["true_facts"]:
            pos[p["true_facts"][0]].append(r)
        elif p["group"] == "forget_neighborhood" and p["subject"] in subj_to_fact:
            same[subj_to_fact[p["subject"]]].append(r)
    diff = {i: [r for j in range(n) if j != i and rel[j] == rel[i] and obj[j] != obj[i] for r in same[j]]
            for i in range(n)}
    usable = [i for i in range(n) if pos[i] and same[i] and diff[i]]
    if not usable:
        raise SystemExit("no association has request prompts, neighbours and same-relation other-answer neighbours")
    eval_rows = sorted({r for i in usable for r in pos[i] + same[i] + diff[i]})
    slot = {r: k for k, r in enumerate(eval_rows)}

    s_train = torch.load(cache / "states_train.pt", weights_only=False)
    s_aug = torch.load(cache / "states_aug.pt", weights_only=False)
    s_eval = torch.load(cache / "states_eval.pt", weights_only=False)
    layers = list(s_train["layers"])
    assert layers == list(s_aug["layers"]) == list(s_eval["layers"])
    eval_index = torch.tensor(eval_rows, dtype=torch.long)

    per_block = {}
    for k, layer in enumerate(layers):
        x_tr = torch.cat([s_train["states"][k], s_aug["states"][k]]).to(args.device)
        u_tr, mean = unit_centred(x_tr)
        u_ev, _ = unit_centred(s_eval["states"][k][eval_index].to(args.device), mean)
        w, b = fit_heads(u_tr, y.to(args.device), gamma, args.iters)
        z = (u_ev @ w + b).cpu()
        u_ev = u_ev.cpu()
        a_same, a_diff, ratio = [], [], []
        for i in usable:
            zp = z[[slot[r] for r in pos[i]], i]
            a_same.append(auc(zp, z[[slot[r] for r in same[i]], i]))
            a_diff.append(auc(zp, z[[slot[r] for r in diff[i]], i]))
            up = u_ev[[slot[r] for r in pos[i]]]
            d_same = torch.cdist(up, u_ev[[slot[r] for r in same[i]]]).mean()
            d_diff = torch.cdist(up, u_ev[[slot[r] for r in diff[i]]]).mean()
            ratio.append(float(d_same / d_diff))
        per_block[int(layer)] = {
            "auc_same_answer": sum(a_same) / len(a_same),
            "auc_diff_answer": sum(a_diff) / len(a_diff),
            "distance_ratio": sum(ratio) / len(ratio),
            "train_head_norm_median": float(w.norm(dim=0).median()),
        }
        print(json.dumps({"layer": int(layer), **{k2: round(v, 4) for k2, v in per_block[int(layer)].items()}}),
              flush=True)

    out = {"cache": str(cache), "model_path": manifest["model_path"], "gamma": gamma,
           "associations_used": len(usable), "of": n,
           "pos_per_assoc": sum(len(pos[i]) for i in usable) / len(usable),
           "same_per_assoc": sum(len(same[i]) for i in usable) / len(usable),
           "diff_per_assoc": sum(len(diff[i]) for i in usable) / len(usable),
           "per_block": per_block}
    (cache / "theory").mkdir(exist_ok=True)
    (cache / "theory" / "read_layers.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "per_block"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
