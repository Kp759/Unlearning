#!/usr/bin/env python3
"""ZsRE: why is Gen high? Router vs genie on the same trained rows, plus why
the router misses each paraphrase, plus a threshold what-if.

    python scripts/evaluate_zsre_router_decomposition.py \
        --run-dirs outputs/zsre_multiseed_regular_v1/seed*/L??/linear_global \
        --zsre-path data/zsre_mend_eval.json --local-files-only
    python scripts/evaluate_zsre_router_decomposition.py --summarize \
        --run-dirs outputs/zsre_multiseed_regular_v1/seed*/L??/linear_global

Per run dir (writes <run>/zsre_decomposition.json; existing ones are skipped):

1. Official forget Eff/Gen twice with identical rows:
     router  the saved linear classifier routes (= official eval)
     genie   ground truth: every forget rewrite and paraphrase request is routed
             to its own row; neighborhood prompts get nothing
   genie Gen ~ 0 with router Gen high  =>  the rows generalise; routing is the gap.
2. Every forget paraphrase request, classified by the runtime decision:
     routed_correct | wrong_fact | ambiguous (two heads within the margin) |
     below_threshold (own head eligible, score under the cutoff) |
     not_eligible (subject tokens not found; `subject_in_text` says whether
     the subject string is there case-insensitively, i.e. a casing/tokenization
     miss, or reworded away)
3. Threshold what-if from the captured logits (no retraining): paraphrase
   routed-correct rate vs false firing on forget neighborhood prompts and on
   retain rewrite/paraphrase requests, for cutoffs below the calibrated one.

Everything uses the runtime hook's own subject mask and logits.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

import zsre_zero_unlearn_official_eval as zsre
from evaluate_zsre_fact_association_embeddings_official import evaluate_split_fixed_boundary
from linear_router import decide_routes, load_router_artifact

OUT_NAME = "zsre_decomposition.json"
CATEGORIES = ("routed_correct", "wrong_fact", "ambiguous", "below_threshold", "not_eligible")
SHIFTS = (4.0, 3.0, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0, -8.0)
MATCHED_FPR = (0.02, 0.03, 0.05, 0.10)


def _run_dirs(patterns):
    dirs = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        dirs.extend(Path(m).resolve() for m in matches)
    return [d for d in dirs if (d / "fact_association_embeddings.pt").exists()]


class Spy:
    """Capture the hook's own subject mask and logits for each forward."""

    def __init__(self, bank):
        self.bank = bank
        self.mask = self.logits = None
        self._mask_fn, self._logit_fn = bank._subject_mask, bank.router_logits
        bank._subject_mask = self._mask
        bank.router_logits = self._logits

    def _mask(self, *args, **kwargs):
        out = self._mask_fn(*args, **kwargs)
        self.mask = out.detach().cpu()
        return out

    def _logits(self, query):
        out = self._logit_fn(query)
        self.logits = out.detach().float().cpu()
        return out


@torch.no_grad()
def score_requests(model, bank, spy, tok, texts, device, batch_size):
    """Router logits [P, N] and eligibility [P, N] at each request boundary."""
    logits, masks = [], []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, padding=True, return_tensors="pt").to(device)
        model.set_association_prefix_lengths(enc["attention_mask"].sum(dim=1).tolist())
        model(**enc, use_cache=False)
        logits.append(spy.logits)
        masks.append(spy.mask)
    return torch.cat(logits), torch.cat(masks).bool()


def _decisions(logits, eligible, threshold, margin):
    d = decide_routes(logits, eligible, threshold, margin)
    return d["active"], d["fact"], d["ambiguous"]


def classify(logits, eligible, own, threshold, margin):
    active, fact, ambiguous = _decisions(logits, eligible, threshold, margin)
    out = []
    for i, row in enumerate(own):
        if bool(active[i]):
            out.append("routed_correct" if int(fact[i]) == row else "wrong_fact")
        elif bool(ambiguous[i]):
            out.append("ambiguous")
        elif bool(eligible[i, row]):
            out.append("below_threshold")
        else:
            out.append("not_eligible")
    return out


def threshold_whatif(para, neigh, retain, base_threshold, margin, same_subject=None, common=None):
    """Lower cutoffs, then the subject gate (any eligible head fires, no margin).

    same_subject: the router's own held-out negative controls (calibration +
    audit): the forgotten subjects in OTHER relations' prompts. These are the
    only negatives a lower cutoff can hurt on ZsRE, since official neighborhood
    and retain requests never contain a forget subject.
    """
    rows = []
    for shift in SHIFTS + ("subject_gate",):
        gate = shift == "subject_gate"
        t, m = (float("-inf"), 0.0) if gate else (base_threshold + shift, margin)
        fire = lambda blk: (float(_decisions(blk["logits"], blk["eligible"], t, m)[0].float().mean())
                            if blk is not None and len(blk["own"]) else None)
        a, f, _ = _decisions(para["logits"], para["eligible"], t, m)
        own = torch.tensor(para["own"])
        correct = float(((a) & (f == own)).float().mean())
        wrong = float(((a) & (f != own)).float().mean())
        rows.append({"threshold_shift": shift, "threshold": None if gate else t,
                     "paraphrase_routed_correct": correct, "paraphrase_wrong_fact": wrong,
                     "neighborhood_false_fire": fire(neigh), "retain_false_fire": fire(retain),
                     "same_subject_false_fire": fire(same_subject),
                     "common_same_subject_false_fire": fire(common)})
    return rows


def run_one(run_dir, base_model, tok, args, records_cache):
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    seed = int(manifest.get("seed", 1))
    artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    if artifact.get("dataset") not in (None, "ZsRE"):
        raise ValueError(f"{run_dir} is not a ZsRE run")
    model, bank = load_router_artifact(base_model, artifact)
    try:
        device = next(model.parameters()).device
        llama_like = zsre.is_llama_like(model, tok)
        if seed not in records_cache:
            records_cache[seed] = zsre.load_official_eval_records(
                Path(args.zsre_path), tok, forget_num=50, retain_num=1000, seed=seed)
        forget, retain = records_cache[seed]
        expected = list(manifest.get("forget_case_ids", []))
        if expected and [int(r["case_id"]) for r in forget] != expected:
            raise ValueError("Forget sample differs from the run manifest")
        row_of = {int(f["case_id"]): i for i, f in enumerate(bank.facts)}

        # 1. Official Eff/Gen: router, then genie on the same rows.
        router_summary, _, router_pred, router_routes = evaluate_split_fixed_boundary(
            model, bank, tok, forget, device, llama_like=llama_like,
            split_name="forget", batch_size=args.batch_size)
        genie_map = {}
        for rec in forget:
            row = row_of[int(rec["case_id"])]
            rr = rec["requested_rewrite"]
            for text in [str(rr["prompt"]).format(str(rr["subject"]))] + \
                    [str(p) for p in rec["paraphrase_prompts"]]:
                genie_map.setdefault(tuple(zsre._flat_ids(tok, text)), row)
        bank.set_oracle_routes(genie_map)
        try:
            genie_summary, _, genie_pred, genie_routes = evaluate_split_fixed_boundary(
                model, bank, tok, forget, device, llama_like=llama_like,
                split_name="forget", batch_size=args.batch_size)
        finally:
            bank.set_oracle_routes(None)

        # 2-3. Router decision for every request, from the hook's own mask/logits.
        spy = Spy(bank)
        if bank.per_head_thresholds is not None:
            raise ValueError("Per-head thresholds are not supported by the what-if")
        threshold = float(bank.threshold)
        margin = float(bank.ambiguity_margin)

        para_texts, para_own, para_meta = [], [], []
        for rec in forget:
            row = row_of[int(rec["case_id"])]
            subject = str(rec["requested_rewrite"]["subject"])
            for k, text in enumerate(rec["paraphrase_prompts"]):
                para_texts.append(str(text)); para_own.append(row)
                para_meta.append({"case_id": int(rec["case_id"]), "prompt_index": k,
                                  "subject": subject, "prompt": str(text),
                                  "subject_in_text": subject.casefold() in str(text).casefold()})
        neigh_texts = sorted({c.prompt for rec in forget for c in zsre.expand_prediction_cases(
            rec, tok, llama_like=llama_like, prompt_types=("neighborhood",))})
        retain_texts = [str(r["requested_rewrite"]["prompt"]).format(str(r["requested_rewrite"]["subject"]))
                        for r in retain] + [str(p) for r in retain for p in r["paraphrase_prompts"]]
        if args.max_retain_requests:
            retain_texts = retain_texts[:args.max_retain_requests]

        def block(texts, own):
            logits, eligible = score_requests(model, bank, spy, tok, texts, device, args.batch_size)
            return {"logits": logits, "eligible": eligible, "own": own}

        para = block(para_texts, para_own)
        neigh = block(neigh_texts, [None] * len(neigh_texts))
        ret = block(retain_texts, [None] * len(retain_texts))
        same_subject, same_subject_n = None, 0
        router_rows = run_dir.parent / "router" / "linear_router_dataset.json"
        if router_rows.exists():
            controls = [r["prompt"] for r in json.loads(router_rows.read_text())
                        if r.get("kind") != "positive" and r.get("split") in ("calibration", "audit")]
            if controls:
                same_subject = block(controls, [None] * len(controls))
                same_subject_n = len(controls)
        # Common same-subject negatives: the held-out controls of several
        # routers for this seed/layer (e.g. regular and reworded), so two
        # routers are compared on the same prompts.
        common, common_sources = None, {}
        if args.negative_sets:
            layer_dir, seed_dir = run_dir.parent.name, run_dir.parent.parent.name
            prompts = []
            for spec in args.negative_sets:
                name, root = spec.split("=", 1)
                f = Path(root) / seed_dir / layer_dir / "router" / "linear_router_dataset.json"
                if not f.exists():
                    raise FileNotFoundError(f"negative set {name}: {f}")
                got = [r["prompt"] for r in json.loads(f.read_text())
                       if r.get("kind") != "positive" and r.get("split") in ("calibration", "audit")]
                common_sources[name] = len(got)
                prompts.extend(got)
            prompts = list(dict.fromkeys(prompts))
            common = block(prompts, [None] * len(prompts))
            common_sources["union"] = len(prompts)
        cats = classify(para["logits"], para["eligible"], para_own, threshold, margin)

        # Per-paraphrase token accuracy under router and genie (case-macro inside).
        def acc_by_prompt(pred):
            acc = defaultdict(list)
            for p in pred:
                if p["prompt_type"] == "paraphrase":
                    acc[(int(p["case_id"]), int(p["prompt_index"]))].append(bool(p["correct"]))
            return {k: sum(v) / len(v) for k, v in acc.items()}

        r_acc, g_acc = acc_by_prompt(router_pred), acc_by_prompt(genie_pred)
        per_prompt = []
        for i, meta in enumerate(para_meta):
            key = (meta["case_id"], meta["prompt_index"])
            own = para_own[i]
            z = para["logits"][i]
            elig = para["eligible"][i]
            best_elig = int(z.masked_fill(~elig, float("-inf")).argmax()) if bool(elig.any()) else None
            per_prompt.append({**meta, "category": cats[i],
                               "own_logit": float(z[own]), "threshold": threshold,
                               "own_rank": int((z > z[own]).sum()) + 1,
                               "best_eligible_fact": best_elig,
                               "n_eligible": int(elig.sum()),
                               "acc_router": r_acc.get(key), "acc_genie": g_acc.get(key)})

        by_cat = {}
        for cat in CATEGORIES:
            items = [p for p in per_prompt if p["category"] == cat]
            by_cat[cat] = {
                "fraction": len(items) / max(1, len(per_prompt)), "count": len(items),
                "mean_acc_router": st.mean([p["acc_router"] for p in items]) * 100 if items else None,
                "mean_acc_genie": st.mean([p["acc_genie"] for p in items]) * 100 if items else None,
                "subject_in_text": sum(p["subject_in_text"] for p in items),
            }
        result = {
            "run_dir": str(run_dir), "seed": seed, "layer": int(artifact["layer"]),
            "threshold": threshold, "ambiguity_margin": margin,
            "router": {"Eff": router_summary["Eff"], "Gen": router_summary["Gen"],
                       "routes": router_routes},
            "genie": {"Eff": genie_summary["Eff"], "Gen": genie_summary["Gen"],
                      "routes": genie_routes},
            "paraphrase_categories": by_cat,
            "threshold_whatif": threshold_whatif(
                {**para, "own": para_own}, neigh, ret, threshold, margin, same_subject, common),
            "counts": {"paraphrases": len(para_texts), "neighborhood_requests": len(neigh_texts),
                       "retain_requests": len(retain_texts),
                       "same_subject_negative_controls": same_subject_n,
                       "common_negative_sets": common_sources},
            "missed_paraphrases": [p for p in per_prompt if p["category"] != "routed_correct"],
        }
        (run_dir / OUT_NAME).write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        bank._hook_handle.remove()


def _fmt(values, scale=1.0, digits=1):
    values = [v * scale for v in values if v is not None]
    if not values:
        return "–"
    sd = st.stdev(values) if len(values) > 1 else 0.0
    return f"{st.mean(values):.{digits}f} ± {sd:.{digits}f}"


def summarize(run_dirs):
    by_layer = defaultdict(list)
    for d in run_dirs:
        f = d / OUT_NAME
        if f.exists():
            r = json.loads(f.read_text())
            by_layer[r["layer"]].append(r)
    if not by_layer:
        print("no zsre_decomposition.json found")
        return
    print("### ZsRE forget Gen: router vs genie on the same rows; why paraphrases miss\n")
    head = ["layer", "n", "Gen router", "Gen genie", "Eff router", "Eff genie",
            "routed_correct %", "not_eligible %", "below_threshold %", "ambiguous %", "wrong_fact %"]
    print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        cat = lambda c: _fmt([r["paraphrase_categories"][c]["fraction"] for r in rs], 100)
        print("| " + " | ".join([
            f"L{layer:02d}", str(len(rs)),
            _fmt([r["router"]["Gen"] for r in rs]), _fmt([r["genie"]["Gen"] for r in rs]),
            _fmt([r["router"]["Eff"] for r in rs]), _fmt([r["genie"]["Eff"] for r in rs]),
            cat("routed_correct"), cat("not_eligible"), cat("below_threshold"),
            cat("ambiguous"), cat("wrong_fact")]) + " |")
    print("\n### not_eligible paraphrases whose subject string IS in the text (casing/tokenization)\n")
    for layer in sorted(by_layer):
        rs = by_layer[layer]
        n = sum(r["paraphrase_categories"]["not_eligible"]["count"] for r in rs)
        s = sum(r["paraphrase_categories"]["not_eligible"]["subject_in_text"] for r in rs)
        print(f"L{layer:02d}: {s}/{n}")
    print("\n### Threshold what-if (mean over seeds), %: paraphrase routed-correct / "
          "same-subject other-relation false fire / neighborhood + retain false fire\n")
    shifts = [w["threshold_shift"] for w in next(iter(by_layer.values()))[0]["threshold_whatif"]]
    label = lambda s: s if isinstance(s, str) else f"{s:+.1f}"
    print("| layer | " + " | ".join(label(s) for s in shifts) + " |\n|" + "---|" * (len(shifts) + 1))

    def m(w, k):
        vals = [x.get(k) for x in w if x.get(k) is not None]
        return st.mean(vals) * 100 if vals else float("nan")

    for layer in sorted(by_layer):
        cells = []
        for i in range(len(shifts)):
            w = [r["threshold_whatif"][i] for r in by_layer[layer]]
            other = max(m(w, "neighborhood_false_fire"), m(w, "retain_false_fire"))
            cells.append(f"{m(w, 'paraphrase_routed_correct'):.0f} / "
                         f"{m(w, 'same_subject_false_fire'):.1f} / {other:.1f}")
        print(f"| L{layer:02d} | " + " | ".join(cells) + " |")

    if any(w.get("common_same_subject_false_fire") is not None
           for rs in by_layer.values() for r in rs for w in r["threshold_whatif"]):
        print("\n### On the COMMON same-subject negatives: paraphrase routed-correct % at the "
              "shipped cutoff, and the best reachable at matched false fire (mean over seeds)\n")
        head = ["layer", "shipped: routed / false fire"] + [f"FF <= {t:.0%}" for t in MATCHED_FPR]
        print("| " + " | ".join(head) + " |\n|" + "---|" * len(head))
        for layer in sorted(by_layer):
            rs = by_layer[layer]
            ship = [next(w for w in r["threshold_whatif"] if w["threshold_shift"] == 0.0) for r in rs]
            cells = [f"L{layer:02d}", f"{m(ship, 'paraphrase_routed_correct'):.0f} / "
                                      f"{m(ship, 'common_same_subject_false_fire'):.1f}"]
            for target in MATCHED_FPR:
                best = []
                for r in rs:
                    ok = [w["paraphrase_routed_correct"] for w in r["threshold_whatif"]
                          if w["threshold_shift"] != "subject_gate"
                          and w.get("common_same_subject_false_fire") is not None
                          and w["common_same_subject_false_fire"] <= target]
                    best.append(max(ok) if ok else 0.0)
                cells.append(f"{st.mean(best) * 100:.0f}")
            print("| " + " | ".join(cells) + " |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dirs", nargs="+", required=True, help="dirs or glob patterns")
    parser.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-retain-requests", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--negative-sets", nargs="*", default=[],
                        help="NAME=SWEEP_ROOT ...: also score the held-out same-subject controls "
                             "of these sweeps' routers (same seed/layer), as one common set")
    args = parser.parse_args(argv)

    run_dirs = _run_dirs(args.run_dirs)
    if args.summarize:
        summarize(run_dirs)
        return 0
    todo = [d for d in run_dirs if args.overwrite or not (d / OUT_NAME).exists()]
    print(f"{len(run_dirs)} run dirs, {len(todo)} to evaluate", flush=True)
    if not todo:
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mcf_zero_unlearn_official_eval import dtype_from_str

    model_path = json.loads((todo[0] / "association_manifest.json").read_text())["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype_from_str(args.dtype), local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    base_model.requires_grad_(False)

    cache, failed = {}, []
    for d in todo:
        try:
            r = run_one(d, base_model, tok, args, cache)
            c = r["paraphrase_categories"]
            print(f"{d}: Gen router {r['router']['Gen']:.2f} | genie {r['genie']['Gen']:.2f} | "
                  + " ".join(f"{k}={c[k]['fraction']:.2f}" for k in CATEGORIES), flush=True)
        except Exception as exc:  # keep going; report at the end
            print(f"FAILED {d}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            failed.append(str(d))
    summarize(run_dirs)
    if failed:
        print(f"{len(failed)} failed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
