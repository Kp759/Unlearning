#!/usr/bin/env python3
"""Score the data-grid refits (router_data_grid.slurm) on the cached seed-1 prompts, gates off.

Each refit's heads and bias are applied to the shipped router's cached block-19 states (the
feature mean is the refit's own). Held-out same-subject negatives are the shipped router's
calibration + audit negatives (the 615 of confusion_train.png on MCF): every refit sees the same
test, and none of its own fit prompts are among them unless it drew the same sentence, which is
reported. Also the official evaluation prompts and the no-association prompts, with and without
the per-head bias (set on the shipped calibration negatives that the refit did not train on).

    python -u scripts/router_data_grid_score.py --dataset mcf --seed 1
    # -> outputs/router_data_grid_v1/<ds>/seed<S>/grid.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from linear_router import apply_feature_map, decide_routes
from router_negative_clean_gate import contains, overlap


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="mcf")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--grid", default="outputs/router_data_grid_v1")
    args = ap.parse_args(argv)
    cache = Path(args.root) / args.dataset / f"seed{args.seed}"
    man = json.loads((cache / "manifest.json").read_text())
    ship = torch.load(Path(man["router_dir"]) / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    facts, layer = ship["facts"], int(man["router_layer"])
    n = len(facts)
    dataset = json.loads((Path(man["router_dir"]) / "linear_router_dataset.json").read_text())

    S = {}
    for name in ("train", "eval", "null"):
        st = torch.load(cache / f"states_{name}.pt", weights_only=False, mmap=True)
        S[name] = {"probes": json.loads((cache / f"probes_{name}.json").read_text()),
                   "h": st["states"][list(st["layers"]).index(layer)].float(),
                   "eligible": torch.load(cache / f"router_{name}.pt", weights_only=False)["eligible"]}
    tr = S["train"]
    split = [p["split"] for p in tr["probes"]]
    m_cal = torch.tensor([s == "calibration" for s in split])
    m_held = torch.tensor([s != "fit" for s in split])
    lab = torch.zeros(len(tr["probes"]), n, dtype=torch.long)
    for k, (p, d) in enumerate(zip(tr["probes"], dataset)):
        if p["should_fire"]:
            for i in p["true_facts"]:
                lab[k, i] = 1
            continue
        for i in (p.get("meta") or {}).get("negative_for") or []:
            f, donor = facts[i], str(d.get("donor_relation") or "")
            amb = contains(p["prompt"], f["object"]) or (d["kind"] == "subject_transplant" and donor.startswith("P")
                                                         and str(f["relation"]).startswith("P") and overlap(f["relation"], donor))
            lab[k, i] = -2 if amb else -1
    held_text = {" ".join(p["prompt"].casefold().split()) for k, p in enumerate(tr["probes"]) if bool(m_held[k])}
    null_text = [" ".join((p.get("prompt") or "").casefold().split()) or None for p in S["null"]["probes"]]
    null_group = [p["group"] for p in S["null"]["probes"]]
    should = torch.tensor([p["should_fire"] for p in S["eval"]["probes"]])

    def score(art, tag, fit_text=frozenset()):
        if [f["id"] for f in art["facts"]] != [f["id"] for f in facts]:
            raise SystemExit(f"{tag}: facts differ from the shipped router")
        # no-association prompts the refit trained on are not a test: left out of its counts
        null_ok = torch.tensor([t is None or t not in fit_text for t in null_text])
        same_subj = torch.tensor([g == "null_protected_other_relation" for g in null_group])
        mu, comps = torch.as_tensor(art["feature_mean"]).float(), art.get("feature_components")
        w, b = torch.as_tensor(art["router_weight"]).float(), torch.as_tensor(art["router_bias"]).float()
        z = {k: apply_feature_map(S[k]["h"], mu, comps) @ w.T + b for k in S}
        out = {}
        for variant in ("as fit", "+ per-head bias", "+ per-head F1", "+ per-head recall@5%fp"):
            bias = b.clone()
            if variant != "as fit":
                neg_c = (lab == -1) & m_cal[:, None] & tr["eligible"]
                pos_c = (lab == 1) & m_cal[:, None] & tr["eligible"]
                for i in range(n):
                    s_neg = z["train"][neg_c[:, i], i]
                    s_pos = z["train"][pos_c[:, i], i]
                    if variant == "+ per-head bias":
                        if s_neg.numel() and float(s_neg.max()) >= 0:
                            bias[i] = b[i] - (float(s_neg.max()) + 1e-3)
                        continue
                    if not s_pos.numel() or not s_neg.numel():
                        continue
                    # candidate thresholds: midpoints between sorted calibration scores of the head
                    allv = torch.cat([s_pos, s_neg]).sort().values
                    cands = torch.cat([allv[:1] - 1.0, (allv[1:] + allv[:-1]) / 2, allv[-1:] + 1.0])
                    tp = (s_pos[None, :] >= cands[:, None]).sum(1).float()
                    fp = (s_neg[None, :] >= cands[:, None]).sum(1).float()
                    fn = s_pos.numel() - tp
                    if variant == "+ per-head F1":
                        obj = 2 * tp / (2 * tp + fp + fn).clamp_min(1)
                    else:                                   # recall at a 5% false-fire budget
                        obj = torch.where(fp / s_neg.numel() <= 0.05, tp / s_pos.numel(), torch.full_like(tp, -1.0))
                    best = cands[int(obj.argmax())]
                    # keep the shipped bias if the chosen threshold is below 0 (never raise firing)
                    if float(best) > 0:
                        bias[i] = b[i] - float(best)
            zz = {k: z[k] - b + bias for k in z}
            fire_tr = (zz["train"] >= 0) & tr["eligible"]
            neg = (lab == -1) & m_held[:, None]
            negall = (lab <= -1) & m_held[:, None]
            m_aud = torch.tensor([sp == "audit" for sp in split])
            neg_aud = (lab <= -1) & m_aud[:, None]
            dec = decide_routes(zz["eval"], S["eval"]["eligible"], 0.0, 0.5)
            fire, fact = dec["active"].cpu(), dec["fact"].cpu()
            right = torch.tensor([bool(fire[k]) and p["should_fire"] and int(fact[k]) in p["true_facts"]
                                  for k, p in enumerate(S["eval"]["probes"])])
            dn = decide_routes(zz["null"], S["null"]["eligible"], 0.0, 0.5)["active"].cpu() & null_ok
            out[variant] = {"heldout_neg_fired": int((fire_tr & negall).sum()), "heldout_neg": int(negall.sum()),
                            "audit_neg_fired": int((fire_tr & neg_aud).sum()), "audit_neg": int(neg_aud.sum()),
                            "clean_neg_fired": int((fire_tr & neg).sum()), "clean_neg": int(neg.sum()),
                            "forget_recall": float(right[should].float().mean()),
                            "retain_fires": int((fire & ~should).sum()), "retain": int((~should).sum()),
                            "null_fires": int(dn.sum()), "null": int(null_ok.sum()),
                            "same_subject_fires": int((dn & same_subj).sum()), "same_subject": int((null_ok & same_subj).sum()),
                            "null_excluded_overlap": int((~null_ok).sum())}
        return out

    results = {"shipped (20x20)": score(ship, "shipped")}
    grid = Path(args.grid) / args.dataset / f"seed{args.seed}"
    for d in sorted(grid.glob("p*_n*/router/fact_association_embeddings.pt")):
        tag = d.parent.parent.name
        art = torch.load(d, map_location="cpu", weights_only=False)
        overlap_n = 0
        fit_text = frozenset()
        ds_path = d.parent / "linear_router_dataset.json"
        if ds_path.is_file():
            fit = [e for e in json.loads(ds_path.read_text()) if e["split"] != "audit"]   # fit + calibration
            fit_text = frozenset(" ".join(e["prompt"].casefold().split()) for e in fit)
            overlap_n = sum(1 for t in fit_text if t in held_text)
        results[tag] = score(art, tag, fit_text)
        results[tag]["fit_prompts_also_in_heldout"] = overlap_n
        fr = (art.get("router_fit") or {})
        results[tag]["selected_l2"] = fr.get("selected_l2")
    (grid / "grid.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["| router | variant | forget recall (test) | retain fires (test) | same-subject fires (no-assoc.) | all no-assoc. fires | audit negatives fired |",
             "|---|---|---|---|---|---|---|"]
    for tag, r in results.items():
        for v in ("as fit", "+ per-head bias", "+ per-head F1", "+ per-head recall@5%fp"):
            x = r[v]
            lines.append(f"| {tag} | {v} | {x['forget_recall']:.3f} | {x['retain_fires']}/{x['retain']} | "
                         f"{x['same_subject_fires']}/{x['same_subject']} | {x['null_fires']}/{x['null']} | "
                         f"{x['audit_neg_fired']}/{x['audit_neg']} |")
        if "fit_prompts_also_in_heldout" in r:
            lines[-1] += f" fit∩held-out={r['fit_prompts_also_in_heldout']}, l2={r['selected_l2']}"
    (grid / "grid.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
