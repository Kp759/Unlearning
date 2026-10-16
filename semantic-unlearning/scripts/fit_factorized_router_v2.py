#!/usr/bin/env python3
"""Factorized router v2: subject read AT ITS OWN TOKENS (multiple-instance), relation at
the last token. MCF, one seed.

v1 read both heads from the last-token state; its subject head reached only ~50%
validation accuracy (the last token encodes the question, not who it is about), so
calibration collapsed and it fired on 96% of neighbourhood prompts. v2:

  subject head   at a mid layer Ls, a per-subject linear score on EVERY prompt token
                 (BOS excluded), max over positions, sigmoid (one-vs-rest; "none" = all
                 below threshold). Trained as multiple-instance learning: a prompt is
                 positive for subject s if it mentions s anywhere (smooth max while
                 training). No string matching, so misspelt / lower-cased subjects can
                 still match the entity representation.
  relation head  softmax over the 34 MCF relations at the last token of layer Lr (v1).
  fire           p_subj(subject_f) >= t_s and p_rel(relation_f) >= t_r, calibrated with a
                 hard false-fire cap (factorized_router.calibrate: floor met under the cap,
                 else most recall under the cap, else infeasible -> exit 4, nothing shipped).
  pair features  each token is scored together with the NEXT token: causal attention
                 makes "Belgium" in "Belgium national football team" identical to a
                 standalone "Belgium" at every layer; the next token says the name goes on.

Same data as v1 (scripts/factorized_router.build_router_data: held-out strings never
used); the subject head sees all forget-subject / near-miss prompts plus a capped sample
of other-subject prompts (validation: all). One forward pass extracts everything.

    python -u scripts/fit_factorized_router_v2.py \\
        --reference-dir outputs/mcf_multiseed_regular_v1/seed1/L19/router \\
        --output-dir outputs/mcf_factorized_router_v2_seed1 --subject-layers 8 12 16 \\
        --relation-layer 27 --local-files-only
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorized_router import (  # noqa: E402
    ARCHITECTURE,
    balanced_log_loss,
    build_router_data,
    calibrate,
    decide_from_probabilities,
    extract_states,
    feature_map,
    fit_mil_subject,
    fit_softmax,
    mil_logits,
)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--subject-layers", type=int, nargs="+", default=[8, 12, 16])
    p.add_argument("--relation-layer", type=int, default=None,
                   help="single relation layer (v2); overrides --relation-layers")
    p.add_argument("--relation-layers", type=int, nargs="+", default=[19, 23, 24, 27],
                   help="last-token layers tried for the relation head (picked on validation)")
    p.add_argument("--relation-concat", default="19,23,27",
                   help="also try these layers concatenated ('' = no concatenation candidate)")
    p.add_argument("--relation-l2-grid", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2])
    p.add_argument("--mil-l2", type=float, default=1e-4)
    p.add_argument("--mil-epochs", type=int, default=8)
    p.add_argument("--other-subject-cap", type=int, default=15000,
                   help="training other-subject prompts the subject head sees (all go to the relation head)")
    p.add_argument("--min-recall", type=float, default=0.95)
    p.add_argument("--max-false-fire", type=float, default=0.10,
                   help="cap on BOTH validation false-fire rates (same-subject, other); never relaxed")
    p.add_argument("--max-false-fire-same", type=float, default=None, help="separate same-subject cap")
    p.add_argument("--max-false-fire-other", type=float, default=None, help="separate other-subject cap")
    p.add_argument("--hard-rounds", type=int, default=0,
                   help="hard-negative rounds for the subject head (top --hard-fraction 'none' prompts x --hard-weight)")
    p.add_argument("--hard-fraction", type=float, default=0.1)
    p.add_argument("--hard-weight", type=float, default=5.0)
    p.add_argument("--no-pair", action="store_true",
                   help="ablation: score single tokens (prefixes of longer names then fire)")
    p.add_argument("--per-relation", type=int, default=800)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prepare_mcf_association_source import load_mcf_forget_data

    ref = Path(args.reference_dir)
    reference = torch.load(ref / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((ref / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)

    records = json.loads(Path(manifest["mcf_path"]).read_text())
    forget_records, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    if [str(f["id"]) for f in facts] != [str(f["id"]) for f in reference["facts"]]:
        raise ValueError("rebuilt MCF facts do not match the reference artifact")
    examples = [e if isinstance(e, dict) else {k: getattr(e, k) for k in ("fact_id", "prompt", "split", "group")
                                               if hasattr(e, k)} for e in examples]
    data = build_router_data(records, forget_records, facts, examples, seed=seed, per_relation=args.per_relation)
    items = data["items"]
    S, R = len(data["subjects"]), len(data["relations"])

    rng = random.Random(seed)
    # the subject head keeps EVERY other-subject prompt asking a forgotten relation (these
    # combine with a matching relation head into neighbour fires); others are capped
    forget_relations = {str(f["relation"]) for f in facts}
    other_train = [i for i, it in enumerate(items) if it["split"] == "train" and it["subject"] == S
                   and it["kind"].startswith("other_subject") and it["relation"] not in forget_relations]
    drop = set(rng.sample(other_train, max(0, len(other_train) - args.other_subject_cap)))
    keep_span = [i not in drop for i in range(len(items))]
    print(json.dumps({"phase": "data", "items": len(items), "subjects": S, "relations": R,
                      "span_items": sum(keep_span),
                      "by_split_kind": {f"{k[0]}/{k[1]}": v for k, v in Counter(
                          (i["split"], i["kind"].split("+")[0]) for i in items).items()}}), flush=True)

    t0 = time.time()
    if args.relation_layer is not None:
        args.relation_layers, args.relation_concat = [args.relation_layer], ""
    concat = [int(x) for x in args.relation_concat.split(",") if x.strip()]
    rel_layers = sorted(set(args.relation_layers) | set(concat))
    last, span = extract_states(model, tok, [i["prompt"] for i in items], rel_layers,
                                args.subject_layers, keep_span=keep_span, batch_size=args.batch_size,
                                device=args.device)
    print(json.dumps({"phase": "features", "seconds": round(time.time() - t0, 1),
                      "span_positions": int(span[args.subject_layers[0]]["flat"].shape[0])}), flush=True)
    del model
    torch.cuda.empty_cache()

    split = [it["split"] for it in items]
    train = torch.tensor([s == "train" for s in split])
    val = ~train
    yr = torch.tensor([data["relations"].index(i["relation"]) for i in items])
    target = torch.tensor([i["target"] for i in items])
    kinds = [i["kind"] for i in items]
    fact_subject = torch.tensor(data["fact_subject"])
    fact_relation = torch.tensor(data["fact_relation"])

    # relation head (shared by every subject layer): try each layer and the concatenation,
    # keep the best on validation balanced log loss
    same_subj_val = torch.tensor(["subject_other_relation" in k for k in kinds]) & val
    candidates = [[L] for L in args.relation_layers] + ([concat] if len(concat) > 1 else [])
    relation_report, chosen = {}, None
    for layers in candidates:
        name = "+".join(str(L) for L in layers)
        Hn = torch.cat([F.normalize(last[L].float(), dim=-1) for L in layers], dim=-1)
        mean_r = Hn[train].mean(0)
        X = Hn - mean_r
        best = None
        for l2 in args.relation_l2_grid:
            W, b = fit_softmax(X[train], yr[train], R, l2, device=args.device)
            loss = balanced_log_loss(X[val], yr[val], W, b, R)
            if best is None or loss < best[0]:
                best = (loss, l2, W, b)
        pred = (X @ best[2].T + best[3]).argmax(-1)
        relation_report[name] = {
            "l2": best[1], "val_balanced_log_loss": best[0],
            "val_acc": float((pred[val] == yr[val]).float().mean()),
            "val_acc_forget_subject_other_relations": float((pred[same_subj_val] == yr[same_subj_val]).float().mean()),
        }
        print(json.dumps({"phase": "relation_candidate", "layers": name, **relation_report[name]}), flush=True)
        if chosen is None or best[0] < chosen["loss"]:
            chosen = {"loss": best[0], "layers": layers, "name": name, "mean": mean_r, "W": best[2],
                      "b": best[3], "X": X}
    Wr, br, rel_mean, X = chosen["W"], chosen["b"], chosen["mean"], chosen["X"]
    rel_layers_chosen = chosen["layers"]
    rel_tag = chosen["name"].replace("+", "-")
    rel_val_acc = relation_report[chosen["name"]]["val_acc"]
    print(json.dumps({"phase": "relation_head", "chosen": chosen["name"], **relation_report[chosen["name"]]}),
          flush=True)
    val_rows = val.nonzero().flatten()
    pr_val = F.softmax(X[val_rows] @ Wr.T + br, -1)

    reports = {}
    for Ls in args.subject_layers:
        sp = span[Ls]
        rows = sp["rows"]                                          # span row -> item index
        pos_of = {int(r): k for k, r in enumerate(rows.tolist())}
        y_span = torch.tensor([items[int(r)]["subject"] if items[int(r)]["subject"] < S else -1
                               for r in rows.tolist()])
        tr = torch.tensor([k for k, r in enumerate(rows.tolist()) if split[r] == "train"])
        va = torch.tensor([pos_of[int(r)] for r in val_rows.tolist()])  # every val item has a span
        t0 = time.time()
        Ws, bs, smean, hist = fit_mil_subject(sp, tr, va, y_span, S, l2=args.mil_l2,
                                              epochs=args.mil_epochs, device=args.device, seed=seed,
                                              pair=not args.no_pair, hard_rounds=args.hard_rounds,
                                              hard_fraction=args.hard_fraction, hard_weight=args.hard_weight)
        ps_val = torch.sigmoid(mil_logits(sp, va, Ws, bs, smean.to(args.device), device=args.device))
        # subject detection quality on validation (threshold 0.5)
        yv = y_span[va]
        hit = ps_val.argmax(-1)
        det = {"forget_subject_prompts_top1": float((hit[yv >= 0] == yv[yv >= 0]).float().mean()),
               "forget_subject_prompts_detected@0.5": float((ps_val[yv >= 0].gather(1, yv[yv >= 0][:, None]) >= 0.5).float().mean()),
               "none_prompts_any_subject@0.5": float((ps_val[yv < 0].amax(-1) >= 0.5).float().mean())}
        val_kinds = [kinds[i] for i in val_rows.tolist()]
        cal = calibrate(ps_val, pr_val, target[val_rows], fact_subject, fact_relation, val_kinds,
                        min_recall=args.min_recall, max_false_fire=args.max_false_fire,
                        max_false_fire_same=args.max_false_fire_same,
                        max_false_fire_other=args.max_false_fire_other)
        d = decide_from_probabilities(ps_val, pr_val, fact_subject, fact_relation,
                                      cal["t_subject"], cal["t_relation"])
        tv = target[val_rows]
        per_kind = {}
        for k in sorted(set(val_kinds)):
            m = torch.tensor([x == k for x in val_kinds])
            pos, neg = m & (tv >= 0), m & (tv < 0)
            per_kind[k] = {"n": int(m.sum()),
                           "recall": float(((d == tv) & pos).float().sum() / pos.sum()) if pos.any() else None,
                           "fire": float((d[neg] >= 0).float().mean()) if neg.any() else None}
        state = {"layer": rel_layers_chosen[-1], "relation_layers": rel_layers_chosen,
                 "feature_mean": rel_mean,
                 "relation_weight": Wr, "relation_bias": br,
                 "subject_mode": "mil_max" if args.no_pair else "mil_max_pair",
                 "subject_layer": Ls, "subject_mean": smean,
                 "subject_weight": Ws, "subject_bias": bs,
                 "fact_subject": fact_subject, "fact_relation": fact_relation,
                 "subjects": data["subjects"], "relations": data["relations"],
                 "t_subject": cal["t_subject"], "t_relation": cal["t_relation"],
                 "calibration": cal, "min_recall": args.min_recall}
        report = {"subject_layer": Ls, "relation_layers": rel_layers_chosen, "calibration": cal,
                  "subject_detection_val": det, "mil_history": hist,
                  "relation_val_acc": rel_val_acc, "validation_by_kind": per_kind,
                  "fit_seconds": round(time.time() - t0, 1)}
        reports[Ls] = report
        print(json.dumps({"phase": "subject_layer_done", "subject_layer": Ls, "calibration": cal,
                          "detection": det}), flush=True)
        layer_dir = out / f"S{Ls:02d}_R{rel_tag}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        art = {k: v for k, v in reference.items() if k in ("facts", "subject_patterns")}
        art.update({"architecture": ARCHITECTURE, "layer": rel_layers_chosen[-1],
                    "rows": torch.zeros_like(reference["rows"]), "factorized_router": state})
        torch.save(art, layer_dir / "fact_association_embeddings.pt")
        m = dict(manifest)
        m.update({"architecture": ARCHITECTURE, "router_layer": rel_layers_chosen[-1],
                  "relation_layers": rel_layers_chosen, "subject_layer": Ls,
                  "reference_dir": str(ref), "method": "factorized_subject_relation_router_v2_mil"})
        (layer_dir / "association_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
        (layer_dir / "factorized_router_report.json").write_text(json.dumps(report, indent=2) + "\n")

    def score(r):
        c = r["calibration"]
        return (c["status"] == "recall_floor_met",
                -(c["false_fire_same_subject"] + c["false_fire_other"]) if c["recall_floor_met"] else c["recall"])

    deployable = {L: r for L, r in reports.items() if r["calibration"]["status"] != "infeasible"}
    sel = max(deployable, key=lambda L: score(deployable[L])) if deployable else None
    summary = {"selected_subject_layer": sel,
               "selected_dir": None if sel is None else f"S{sel:02d}_R{rel_tag}",
               "selected_status": None if sel is None else reports[sel]["calibration"]["status"],
               "calibration_rule": {"min_recall": args.min_recall, "max_false_fire": args.max_false_fire,
                                    "max_false_fire_same": args.max_false_fire_same,
                                    "max_false_fire_other": args.max_false_fire_other},
               "hard_negatives": {"rounds": args.hard_rounds, "fraction": args.hard_fraction,
                                  "weight": args.hard_weight},
               "relation_head_candidates": relation_report, "relation_head_chosen": chosen["name"],
               "pair_features": not args.no_pair,
               "data": {k: v for k, v in data.items() if k != "items"},
               "layers": reports, "command": sys.argv}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"phase": "done", "selected": summary["selected_dir"],
                      "status": summary["selected_status"]}), flush=True)
    if sel is None:
        print("NO DEPLOYABLE LAYER: no subject layer reaches any recall with both validation "
              f"false-fire rates <= {args.max_false_fire}. Not shipping a router.", file=sys.stderr)
        sys.exit(4)


if __name__ == "__main__":
    main()
