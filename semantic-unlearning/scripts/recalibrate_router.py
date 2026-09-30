#!/usr/bin/env python3
"""Re-pick a linear-classifier router's bias on the validation set (weights unchanged).

    python -u scripts/recalibrate_router.py \
        --router-dir outputs/mcf_multiseed_regular_v1/seed1/L19/router \
        --rows-from  outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global \
        --output-dir outputs/calibration_balanced_v1/mcf/seed1/L19/balanced_swap \
        --objective balanced --macro fact --device cuda --local-files-only

Stage 1 (weights w and stage-1 bias b, fit on the fit split) is kept exactly.
Only the cutoff t folded into the bias (b' = b - t, fire at p >= 0.5) is
chosen again, on the VALIDATION prompts = calibration + audit splits by default
(held-out prompt families of the forget facts; no benchmark test prompt):

  balanced     maximise balanced accuracy = (recall + (1 - false fire)) / 2,
               positives and negatives weighted equally. --macro fact averages
               recall over facts and specificity over facts first, so every
               fact also counts equally (a fact with many negatives, or two
               hard prompts, cannot set the cutoff for all); --macro prompt pools.
  target_fpr   most recall with false fire <= --target-fpr
  min_recall   the shipped rule: lowest false fire with recall >= --min-recall
               (both pooled by default; --macro fact uses per-fact averaged rates)
  constrained  all three at once: among cutoffs with recall >= --min-recall AND
               false fire <= --target-fpr, maximise balanced accuracy. With
               --macro fact every term is per-fact first (positives vs negatives
               and facts all weighted equally). If no cutoff meets both, the
               false-fire cap stays hard and recall is maximised under it (ties:
               balanced accuracy); `constraint_status` records which case held.

recall = a positive routed to its own row; false fire = a same-subject negative
control that activates any row. Among tied cutoffs the highest one (fewest
fires) is taken, placed midway inside its decision interval.

With --rows-from RUN the output is RUN's artifact (rows, dataset metadata) with
only the bias replaced, plus RUN's manifest: the official evaluators run on it
directly. Writes recalibration.json: the chosen t and, on every split, recall /
false fire for the shipped cutoff, the raw stage-1 bias (t = 0) and the new t.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from linear_router import (  # noqa: E402
    ARCHITECTURE,
    bias_calibration_record,
    decide_routes,
    fold_threshold_into_bias,
    routing_policy_name,
    score_queries,
)

OBJECTIVES = ("balanced", "target_fpr", "min_recall", "constrained")


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.tolist())
    return value


def stage1_and_cutoff(artifact):
    """(stage-1 bias [N], shipped global cutoff t) of a calibrated-bias router."""
    if str(artifact.get("architecture")) != ARCHITECTURE:
        raise ValueError("not a linear-classifier router artifact")
    if str(artifact.get("gate_mode", "threshold")) != "threshold":
        raise ValueError("the subject gate has no cutoff to recalibrate")
    if artifact.get("head_index") is not None or artifact.get("per_head_thresholds") is not None:
        raise ValueError("only one-head-per-fact routers with a global cutoff are supported")
    cal = artifact.get("bias_calibration")
    if cal is None:
        return artifact["router_bias"].float(), float(artifact["threshold"])
    stage1 = torch.as_tensor(cal["stage1_bias"]).float()
    shift = torch.as_tensor(cal["shift"]).float()
    if not bool(torch.all(shift == shift[0])):
        raise ValueError("per-head calibrated shifts are not supported")
    if not torch.allclose(stage1 - shift, artifact["router_bias"].float(), atol=1e-4):
        raise ValueError("bias_calibration does not reproduce the deployed bias")
    return stage1, float(shift[0])


def outcomes(z, eligible, owner, t, margin):
    """Pooled and per-fact recall / false fire at cutoff t (fire when z >= t)."""
    d = decide_routes(z, eligible, t, margin)
    active, chosen = d["active"].cpu(), d["fact"].cpu()
    eligible = eligible.bool().cpu()
    positive = owner >= 0
    negative = ~positive
    correct = positive & active & (chosen == owner)
    fired_neg = negative & active
    n_facts = z.shape[1]
    pos_per = torch.bincount(owner[positive], minlength=n_facts).double()
    cor_per = torch.bincount(owner[correct], minlength=n_facts).double()
    neg_per = eligible[negative].sum(0).double()
    fire_per = eligible[fired_neg].sum(0).double()
    has_pos, has_neg = pos_per > 0, neg_per > 0
    recalls = (cor_per[has_pos] / pos_per[has_pos])
    fprs = (fire_per[has_neg] / neg_per[has_neg])
    n_pos, n_neg = int(positive.sum()), int(negative.sum())
    recall = float(correct.sum()) / n_pos if n_pos else None
    fpr = float(fired_neg.sum()) / n_neg if n_neg else None
    macro_recall = float(recalls.mean()) if recalls.numel() else None
    macro_fpr = float(fprs.mean()) if fprs.numel() else None
    return {
        "positives": n_pos, "negatives": n_neg,
        "recall": recall, "false_fire": fpr,
        "balanced_accuracy": (None if recall is None or fpr is None
                              else 0.5 * (recall + 1.0 - fpr)),
        "macro_recall": macro_recall, "macro_false_fire": macro_fpr,
        "macro_balanced_accuracy": (None if macro_recall is None or macro_fpr is None
                                    else 0.5 * (macro_recall + 1.0 - macro_fpr)),
        "facts_with_positives": int(has_pos.sum()), "facts_with_negatives": int(has_neg.sum()),
        "correct": int(correct.sum()), "negatives_firing": int(fired_neg.sum()),
    }


def choose_cutoff(z, eligible, owner, margin, objective, macro="fact",
                  target_fpr=0.1, min_recall=0.98, max_candidates=2000):
    """Cutoff on the stage-1 logits z of the validation rows."""
    values = z[eligible.bool()].unique().sort().values
    if values.numel() == 0:
        raise ValueError("validation rows have no eligible pairs")
    if max_candidates and values.numel() > max_candidates:
        idx = torch.linspace(0, values.numel() - 1, max_candidates).round().long()
        values = values[idx].unique().sort().values
    ceiling = torch.nextafter(values.max(), torch.tensor(float("inf")))
    candidates = torch.cat([values, ceiling.reshape(1)]).tolist()
    rows = [(c, outcomes(z, eligible, owner, c, margin)) for c in candidates]

    if objective == "balanced":
        key = "macro_balanced_accuracy" if macro == "fact" else "balanced_accuracy"
        best = max(r[1][key] for r in rows if r[1][key] is not None)
        tied = [r for r in rows if r[1][key] is not None and r[1][key] >= best - 1e-12]
    elif objective == "target_fpr" and macro == "fact":
        # Per-fact rates: false fire averaged over facts <= cap, then the
        # highest per-fact-averaged recall (every fact counts equally).
        ok = [r for r in rows if r[1]["macro_false_fire"] is not None
              and r[1]["macro_false_fire"] <= target_fpr + 1e-12]
        best = max(r[1]["macro_recall"] for r in ok)
        tied = [r for r in ok if r[1]["macro_recall"] >= best - 1e-12]
    elif objective == "target_fpr":
        ok = [r for r in rows if r[1]["false_fire"] <= target_fpr + 1e-12]
        best = max(r[1]["correct"] for r in ok)
        tied = [r for r in ok if r[1]["correct"] == best]
    elif objective == "min_recall" and macro == "fact":
        # Per-fact rates: recall averaged over facts >= target, then the
        # lowest per-fact-averaged false fire.
        usable = [r for r in rows if r[1]["macro_recall"] is not None]
        ok = [r for r in usable if r[1]["macro_recall"] >= min_recall - 1e-12]
        recall_met = bool(ok)
        if not ok:
            # Target unreachable: keep the highest reachable recall, as the
            # shipped calibrate_threshold does (not "fire nothing").
            top = max(r[1]["macro_recall"] for r in usable)
            ok = [r for r in usable if r[1]["macro_recall"] >= top - 1e-12]
        low = min(r[1]["macro_false_fire"] for r in ok)
        pool = [r for r in ok if r[1]["macro_false_fire"] <= low + 1e-12]
        best = max(r[1]["macro_recall"] for r in pool)
        tied = [r for r in pool if r[1]["macro_recall"] >= best - 1e-12]
    elif objective == "min_recall":
        n_pos = rows[0][1]["positives"]
        need = math.ceil(min_recall * n_pos - 1e-9)
        ok = [r for r in rows if r[1]["correct"] >= need]
        recall_met = bool(ok)
        if not ok:
            top = max(r[1]["correct"] for r in rows)
            ok = [r for r in rows if r[1]["correct"] == top]
        low = min(r[1]["false_fire"] for r in ok)
        pool = [r for r in ok if r[1]["false_fire"] <= low + 1e-12]
        best = max(r[1]["correct"] for r in pool)
        tied = [r for r in pool if r[1]["correct"] == best]
    elif objective == "constrained":
        rk, fk, bk = (("macro_recall", "macro_false_fire", "macro_balanced_accuracy")
                      if macro == "fact" else ("recall", "false_fire", "balanced_accuracy"))
        usable = [r for r in rows if None not in (r[1][rk], r[1][fk], r[1][bk])]
        capped = [r for r in usable if r[1][fk] <= target_fpr + 1e-12]
        both = [r for r in capped if r[1][rk] >= min_recall - 1e-12]
        if both:
            status = "recall_and_false_fire_met"
            best = max(r[1][bk] for r in both)
            tied = [r for r in both if r[1][bk] >= best - 1e-12]
        else:
            status = "false_fire_cap_met_recall_short"
            best_r = max(r[1][rk] for r in capped)
            pool = [r for r in capped if r[1][rk] >= best_r - 1e-12]
            best = max(r[1][bk] for r in pool)
            tied = [r for r in pool if r[1][bk] >= best - 1e-12]
    else:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    # The highest tied cutoff fires least; place t inside (previous candidate, c],
    # where every threshold makes exactly c's decisions.
    c = max(r[0] for r in tied)
    pos = candidates.index(c)
    prev = candidates[pos - 1] if pos > 0 else c - 1.0
    t = float(torch.tensor(0.5 * (prev + c), dtype=z.dtype))
    if not (prev < t <= c):
        t = c
    chosen = outcomes(z, eligible, owner, t, margin)
    if objective == "constrained":
        chosen["constraint_status"] = status
    if objective == "min_recall":
        chosen["recall_target_met"] = recall_met
    return t, chosen


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router-dir", required=True,
                   help="router dir with linear_router_dataset.json (the calibrated router)")
    p.add_argument("--rows-from", default=None,
                   help="trained run whose artifact (rows) gets the new bias; output is evaluable")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--objective", choices=OBJECTIVES, default="balanced")
    p.add_argument("--macro", choices=("fact", "prompt"), default=None,
                   help="fact: per-fact rates averaged over facts (every fact and both classes "
                        "count equally); prompt: pooled over prompts. Default: fact for "
                        "balanced/constrained, prompt for min_recall/target_fpr")
    p.add_argument("--target-fpr", type=float, default=0.1)
    p.add_argument("--min-recall", type=float, default=0.98)
    p.add_argument("--validation-splits", nargs="+", default=["calibration", "audit"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args(argv)
    if a.macro is None:
        a.macro = "fact" if a.objective in ("balanced", "constrained") else "prompt"

    router_dir, output = Path(a.router_dir).resolve(), Path(a.output_dir).resolve()
    router = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu",
                        weights_only=False)
    base_dir = Path(a.rows_from).resolve() if a.rows_from else router_dir
    base = (torch.load(base_dir / "fact_association_embeddings.pt", map_location="cpu",
                       weights_only=False) if a.rows_from else router)
    if a.rows_from:
        same = (torch.equal(base["router_weight"].float(), router["router_weight"].float())
                and torch.allclose(base["router_bias"].float(), router["router_bias"].float(), atol=1e-5)
                and [f["id"] for f in base["facts"]] == [f["id"] for f in router["facts"]])
        if not same:
            raise ValueError(f"{base_dir} was not trained behind {router_dir} (router differs)")
    stage1, shipped_t = stage1_and_cutoff(base)
    manifest = json.loads((base_dir / "association_manifest.json").read_text())
    rows = json.loads((router_dir / "linear_router_dataset.json").read_text())

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from linear_router import eligibility_matrix, load_linear_classifier_artifact
    from static_overlap_fact_association_embeddings import extract_prompt_queries

    facts = list(base["facts"])
    index = {str(f["id"]): i for i, f in enumerate(facts)}
    prompts = [r["prompt"] for r in rows]
    splits = [r["split"] for r in rows]
    owner = torch.tensor([index[r["owner_fact_id"]] if r.get("owner_fact_id") else -1 for r in rows])
    model_path = Path(manifest["model_path"]).resolve()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, a.dtype), local_files_only=a.local_files_only,
        attn_implementation="eager").to(a.device).eval()
    model.requires_grad_(False)
    layer = int(base["layer"])
    queries = extract_prompt_queries(model, tok, prompts, layer, batch_size=a.batch_size)
    eligible = eligibility_matrix(tok, prompts, base["subject_patterns"])
    z = score_queries(queries, base["router_weight"], stage1, base["feature_mean"],
                      base.get("feature_components")).cpu()
    margin = float(base.get("ambiguity_margin", 0.5))

    val = torch.tensor([s in a.validation_splits for s in splits])
    if not bool(val.any()):
        raise ValueError(f"no rows in validation splits {a.validation_splits}")
    t_new, chosen_val = choose_cutoff(z[val], eligible[val], owner[val], margin, a.objective,
                             macro=a.macro, target_fpr=a.target_fpr, min_recall=a.min_recall)

    shipped = decide_routes(z, eligible, shipped_t, margin)
    stored = [r.get("linear_routes_to") for r in rows]
    recomputed = [facts[int(f)]["id"] if bool(act) else None
                  for act, f in zip(shipped["active"].tolist(), shipped["fact"].tolist())]
    mismatches = sum(x != y for x, y in zip(stored, recomputed))

    report_splits = {}
    for name, mask in [(s, torch.tensor([x == s for x in splits])) for s in sorted(set(splits))] + \
                      [("validation", val)]:
        report_splits[name] = {
            rule: outcomes(z[mask], eligible[mask], owner[mask], t, margin)
            for rule, t in (("shipped", shipped_t), ("raw", 0.0), ("new", t_new))
        }

    art = dict(base)
    bias, shift = fold_threshold_into_bias(stage1, t_new)
    art["router_bias"] = bias
    art["threshold"] = 0.0
    art["per_head_thresholds"] = None
    art["decision_rule"] = "calibrated_bias"
    art["threshold_policy"] = "global"
    art["routing_policy"] = routing_policy_name("threshold", "calibrated_bias", "global")
    art["bias_calibration"] = bias_calibration_record("global", stage1, shift)
    fit = copy.deepcopy(art.get("router_fit") or {})
    fit.update({"recalibration": {
        "objective": a.objective,
        "macro": a.macro,
        "target_fpr": a.target_fpr if a.objective in ("target_fpr", "constrained") else None,
        "min_recall": a.min_recall if a.objective in ("min_recall", "constrained") else None,
        "constraint_status": chosen_val.get("constraint_status"),
        "recall_target_met": chosen_val.get("recall_target_met"),
        "validation_splits": a.validation_splits, "cutoff_t": t_new,
        "shipped_cutoff_t": shipped_t, "weights_changed": False,
        "rows_changed": False}})
    art["router_fit"] = fit

    output.mkdir(parents=True, exist_ok=False)
    torch.save(art, output / "fact_association_embeddings.pt")
    shutil.copy2(base_dir / "association_manifest.json", output / "association_manifest.json")
    # Runtime parity: the saved artifact, run through the real hook on the
    # validation prompts, must make exactly the decisions chosen above.
    wrapped, bank = load_linear_classifier_artifact(model, art)
    expect = decide_routes(z[val], eligible[val], t_new, margin)
    expect = [[int(f)] if bool(act) else [] for act, f in
              zip(expect["active"].tolist(), expect["fact"].tolist())]
    val_prompts = [pr for pr, keep in zip(prompts, val.tolist()) if keep]
    got = []
    tok.padding_side = "right"
    try:
        with torch.no_grad():
            for i in range(0, len(val_prompts), a.batch_size):
                enc = tok(val_prompts[i:i + a.batch_size], padding=True, return_tensors="pt").to(a.device)
                wrapped.set_association_prefix_lengths(enc["attention_mask"].sum(dim=1).tolist())
                wrapped(**enc, use_cache=False)
                got.extend(bank.last_active_fact_indices)
    finally:
        bank._hook_handle.remove()
    parity_mismatches = sum(x != y for x, y in zip(expect, got))
    result = {"schema_version": "router_recalibration_v1", "router_dir": str(router_dir),
              "rows_from": str(base_dir) if a.rows_from else None,
              "objective": a.objective, "macro": a.macro, "validation_splits": a.validation_splits,
              "min_recall": a.min_recall, "target_fpr": a.target_fpr,
              "constraint_status": chosen_val.get("constraint_status"),
              "recall_target_met": chosen_val.get("recall_target_met"),
              "cutoff_t": {"shipped": shipped_t, "raw": 0.0, "new": t_new},
              "shipped_routes_recomputed_vs_stored_mismatches": mismatches,
              "runtime_parity_validation_mismatches": parity_mismatches,
              "by_split": report_splits}
    (output / "recalibration.json").write_text(json.dumps(_json_safe(result), indent=2) + "\n")
    v = report_splits["validation"]
    print(json.dumps(_json_safe({"cutoff_t": result["cutoff_t"],
                                 "stored_route_mismatches": mismatches,
                                 "runtime_parity_mismatches": parity_mismatches, "validation": {
        r: {k: v[r][k] for k in ("recall", "false_fire", "macro_recall", "macro_false_fire")}
        for r in v}}), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
