#!/usr/bin/env python3
"""Hybrid router: v2.2's learned subject head as the eligibility gate of a linear router.

The linear router (MCF / ZsRE / MQuAKE / RWKU) fires fact f when the prompt contains f's
subject as an exact token sequence AND f's logistic head on the L19 boundary state says
p >= 0.5. This script fits a learned subject head to replace (or back up) the exact match:

  subject head   v2.2's multiple-instance head (factorized_router.fit_mil_subject): at an
                 early layer Ls, a per-subject linear score of every prompt token paired
                 with the next one, max over the prompt, sigmoid. No string matching, so
                 typos, lower case and other surface forms of a forget subject can fire.
  fact heads     the reference linear router's heads, unchanged (no relation labels needed,
                 which is what keeps v2.2's relation head MCF-only).

Data (training-visible only; no evaluation prompt of ZsRE / MQuAKE / RWKU is read, MCF's
official prompts of the forget records are read only to exclude them):
  positives      the reference router's own examples (association_examples.json, else the
                 canonical prompts + context-prefix families, as fit_linear_router builds
                 them; train -> fit, development -> validation) + typo / lower-case copies
  swaps          each positive with its subject replaced by another entity name (MCF subject
                 pool; names hashed into fit / validation so validation names are unseen)
  near misses    the same, with names sharing a name word with the forget subject
                 ("Michelle Obama" for "Barack Obama"); the head must say "none"
  background     rewrite prompts of other MCF records (other subjects)
Any negative that still contains a forget subject's full name is dropped.

Calibration (validation): the subject threshold with macro subject recall >= --min-recall at
the fewest false fires, under false fire <= --max-false-fire-other (swaps + background) and
<= --max-false-fire-near (near misses); else the most recall under the caps; else no router
(exit 4). Exact-match recall / false fire on the same validation prompts are reported.

Writes, per gate (scripts/linear_router.SUBJECT_GATES):
  OUT/learned/            eligibility = learned head only
  OUT/learned_or_exact/   eligibility = learned head OR exact match
each a linear-router artifact (zero rows) + artifact["learned_subject"], loadable by
train_mcf_jlens_feedback_rows.py and every evaluator, plus OUT/summary.json.

    python -u scripts/fit_hybrid_subject_router.py --reference-dir <L19 router> \\
        --output-dir outputs/<ds>_hybrid_subject_v1/seed1 --mcf-path data/multi_counterfact.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import random
import re
import shutil
import sys
import time

import torch

from factorized_router import (NAME_STOPWORDS, _rw, extract_states, fit_mil_subject, mil_logits,
                               misspell, norm, stable_hash)

GATES = ("learned", "learned_or_exact")


# --------------------------------------------------------------------------- data
def name_words(s):
    return {w for w in re.findall(r"[^\W_]{3,}", str(s).casefold()) if w not in NAME_STOPWORDS}


def subject_surfaces(fact, patterns, tok):
    """Canonical subject + every decoded subject-token pattern (RWKU: surname, short name)."""
    out = [" ".join(str(fact["subject"]).split())]
    for p in patterns or []:
        s = tok.decode(list(p)).strip() if tok is not None else ""
        if s and s not in out:
            out.append(s)
    return out


def name_split(name, val_mod=5):
    return "val" if stable_hash(norm(name)) % val_mod == 0 else "train"


def _bounded(names):
    """Case-insensitive regex for any of `names` as whole words ('IT' must not match 'WITH')."""
    alts = sorted({re.escape(" ".join(str(n).split())) for n in names if str(n).strip()}, key=len, reverse=True)
    return re.compile(r"(?<!\w)(?:" + "|".join(alts) + r")(?!\w)", re.IGNORECASE) if alts else None


def find_surface(prompt, surfaces):
    """First surface (longest first) found in prompt as a whole word -> (start, end) or None."""
    for s in surfaces:
        m = _bounded([s]).search(prompt)
        if m:
            return m.span()
    return None


def swap_span(prompt, span, new):
    if span is None or prompt[span[0]:span[1]] == new:
        return None
    return prompt[:span[0]] + new + prompt[span[1]:]


def build_subject_data(facts, surfaces, examples, pool, background, *, seed=1, swaps=4,
                       near_per=2, near_cap=30, bg_train=4000, bg_val=800, augment=True,
                       banned=()):
    """Items {prompt, subject (index or -1), split (train|val), kind}.

    facts: reference facts (order = fact index); surfaces[i]: fact i's subject strings;
    examples: dicts with fact_id, prompt, split in {train, development};
    pool: other entity names (forget subjects removed by the caller or here);
    background: [(text, subject name)] of other subjects."""
    rng = random.Random(1000 + int(seed))
    subjects, fact_subject = [], []
    for f in facts:
        key = norm(f["subject"])
        if key not in subjects:
            subjects.append(key)
        fact_subject.append(subjects.index(key))
    subj_surfaces = {s: [] for s in range(len(subjects))}
    canonical = {}
    for i, f in enumerate(facts):
        s = fact_subject[i]
        canonical.setdefault(s, " ".join(str(f["subject"]).split()))
        for x in surfaces[i]:
            if x not in subj_surfaces[s]:
                subj_surfaces[s].append(x)
    for s in subj_surfaces:                      # longest first: replace the full name first
        subj_surfaces[s].sort(key=len, reverse=True)
    full_names = _bounded(canonical.values())
    surface_keys = {norm(x) for xs in subj_surfaces.values() for x in xs}
    banned = {b.strip() for b in banned}

    def mentions_forget(text):
        return full_names is not None and full_names.search(text) is not None

    pool = sorted({" ".join(str(n).split()) for n in pool
                   if norm(n) not in surface_keys and not mentions_forget(n)})
    near = {}
    for s, name in canonical.items():
        w = name_words(name)
        near[s] = sorted((n for n in pool if w & name_words(n)), key=stable_hash)[:int(near_cap)]
    near_names = {n for ns in near.values() for n in ns}
    generic = {"train": [n for n in pool if n not in near_names and name_split(n) == "train"],
               "val": [n for n in pool if n not in near_names and name_split(n) == "val"]}

    items, seen = [], set()

    def add(prompt, subject, split, kind):
        if prompt is None:
            return False
        prompt = prompt.strip()
        if not prompt or prompt in seen or prompt in banned:
            return False
        if subject < 0 and mentions_forget(prompt):
            return False
        seen.add(prompt)
        items.append({"prompt": prompt, "subject": int(subject), "split": split, "kind": kind})
        return True

    index = {str(f["id"]): i for i, f in enumerate(facts)}
    for e in examples:
        i = index.get(str(e.get("fact_id")))
        if i is None or str(e.get("split")) not in ("train", "development"):
            continue
        split = "train" if str(e["split"]) == "train" else "val"
        s = fact_subject[i]
        prompt = str(e["prompt"]).strip()
        add(prompt, s, split, "positive")
        span = find_surface(prompt, subj_surfaces[s])
        if span is None:
            continue
        surf = prompt[span[0]:span[1]]
        if augment:
            add(swap_span(prompt, span, misspell(surf, rng)), s, split, "positive+typo")
            add(swap_span(prompt, span, surf.lower()), s, split, "positive+lower")
        names = generic[split]
        for n in rng.sample(names, min(int(swaps), len(names))):
            add(swap_span(prompt, span, n), -1, split, "swap")
        nm = [n for n in near[s] if name_split(n) == split]
        for n in rng.sample(nm, min(int(near_per), len(nm))):
            add(swap_span(prompt, span, n), -1, split, "near_miss")
    caps = {"train": int(bg_train), "val": int(bg_val)}
    taken = Counter()
    order = list(background)
    rng.shuffle(order)
    for text, name in order:
        split = name_split(name)
        if taken[split] >= caps[split] or norm(name) in surface_keys:
            continue
        if add(text, -1, split, "background"):
            taken[split] += 1
    return {"items": items, "subjects": subjects, "fact_subject": fact_subject,
            "near_miss_names": len(near_names), "pool_names": len(pool)}


def mcf_pool_and_background(records, exclude_case_ids=()):
    exclude = {int(c) for c in exclude_case_ids}
    names, background = set(), []
    for r in records:
        if int(r.get("case_id", -1)) in exclude:
            continue
        rw = _rw(r)
        names.add(rw["subject"])
        background.append((rw["prompt"].format(rw["subject"]), rw["subject"]))
    return sorted(names), background


# ---------------------------------------------------------------------- calibration
def calibrate_subject(ps, y, kinds, *, min_recall=0.95, max_false_fire_other=0.01,
                      max_false_fire_near=0.10, grid=None):
    """1-D threshold on p(own subject) for validation positives / max p for negatives."""
    grid = grid if grid is not None else torch.cat([
        torch.logspace(-3, 0, 40)[:-1], torch.linspace(0.5, 0.98, 25),
        1.0 - torch.logspace(-5, -2, 13)]).unique()
    y = torch.as_tensor(y)
    pos = y >= 0
    near = torch.tensor([k.startswith("near_miss") for k in kinds]) & ~pos
    other = ~pos & ~near
    own = ps[pos].gather(1, y[pos][:, None]).squeeze(1) if pos.any() else torch.zeros(0)
    top = ps.amax(1)
    subj = y[pos]
    best_floor, best_capped = None, None
    for t in grid.tolist():
        hit = (own >= t).float()
        per = [float(hit[subj == s].mean()) for s in subj.unique().tolist()]
        recall = sum(per) / len(per) if per else 0.0
        ff_other = float((top[other] >= t).float().mean()) if other.any() else 0.0
        ff_near = float((top[near] >= t).float().mean()) if near.any() else 0.0
        point = {"threshold": t, "recall": recall, "false_fire_other": ff_other,
                 "false_fire_near_miss": ff_near}
        if ff_other > max_false_fire_other or ff_near > max_false_fire_near:
            continue
        if recall >= min_recall:
            key = (-(ff_other + ff_near), t)
            if best_floor is None or key > best_floor[0]:
                best_floor = (key, point)
        else:
            key = (recall, -(ff_other + ff_near), t)
            if best_capped is None or key > best_capped[0]:
                best_capped = (key, point)
    caps = {"min_recall": min_recall, "max_false_fire_other": max_false_fire_other,
            "max_false_fire_near_miss": max_false_fire_near}
    if best_floor is not None:
        return {**best_floor[1], "status": "recall_floor_met", **caps}
    if best_capped is not None and best_capped[1]["recall"] > 0:
        return {**best_capped[1], "status": "false_fire_capped", **caps}
    return {"threshold": 1.0 + 1e-6, "recall": 0.0, "false_fire_other": 0.0,
            "false_fire_near_miss": 0.0, "status": "infeasible", **caps}


def rates_by_kind(fired, y, kinds):
    """fired: [N] bool (own subject for positives, any subject for negatives)."""
    out = {}
    for k in sorted(set(kinds)):
        m = torch.tensor([x == k for x in kinds])
        out[k] = {"n": int(m.sum()), "fire_rate": float(fired[m].float().mean()) if m.any() else None}
    return out


# ------------------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", required=True, help="linear router (stage-1 L19 router)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mcf-path", required=True, help="entity-name pool and background prompts")
    p.add_argument("--subject-layers", type=int, nargs="+", default=[6])
    p.add_argument("--swaps", type=int, default=4)
    p.add_argument("--near-per", type=int, default=2)
    p.add_argument("--background-train", type=int, default=4000)
    p.add_argument("--background-val", type=int, default=800)
    p.add_argument("--mil-l2", type=float, default=1e-4)
    p.add_argument("--mil-epochs", type=int, default=8)
    p.add_argument("--hard-rounds", type=int, default=1)
    p.add_argument("--min-recall", type=float, default=0.95)
    p.add_argument("--max-false-fire-other", type=float, default=0.01)
    p.add_argument("--max-false-fire-near", type=float, default=0.10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from linear_router import eligibility_matrix, examples_from_facts, is_linear_router_artifact

    ref = Path(args.reference_dir).resolve()
    out = Path(args.output_dir)
    reference = torch.load(ref / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    if not is_linear_router_artifact(reference) or reference.get("early_write"):
        raise ValueError(f"{ref} is not a linear-router artifact")
    if reference.get("learned_subject") is not None:
        raise ValueError(f"{ref} already has a learned subject gate")
    if float(torch.as_tensor(reference["rows"]).abs().max()) != 0.0:
        raise ValueError(f"{ref} has trained rows; give the router-only artifact")
    manifest = json.loads((ref / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed") or manifest.get("seed") or 1)
    facts = list(reference["facts"])
    if int(reference["layer"]) <= max(args.subject_layers):
        raise ValueError("subject layers must sit below the router layer")

    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    ex_path = ref / "association_examples.json"
    examples = (json.loads(ex_path.read_text()) if ex_path.is_file()
                else examples_from_facts(facts, augment=True))
    examples = [e if isinstance(e, dict) else {k: getattr(e, k) for k in ("fact_id", "prompt", "split", "group")
                                               if hasattr(e, k)} for e in examples]
    records = json.loads(Path(args.mcf_path).read_text())
    banned, exclude_ids = set(), set()
    dataset = str(facts[0].get("id", "")).split("_", 1)[0]
    if dataset == "mcf":
        # the forget records' official prompts are never trained on (as v2.2)
        from prepare_mcf_association_source import load_mcf_forget_data
        forget_records, mfacts, _ = load_mcf_forget_data(tok, args.mcf_path, seed=seed)
        if [str(f["id"]) for f in mfacts] != [str(f["id"]) for f in facts]:
            raise ValueError("rebuilt MCF facts do not match the reference router")
        exclude_ids = {int(r["case_id"]) for r in forget_records}
        for r in forget_records:
            for key in ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts",
                        "attribute_prompts"):
                banned.update(x.strip() for x in r.get(key, []))
    pool, background = mcf_pool_and_background(records, exclude_ids)
    patterns = reference.get("subject_patterns") or [[] for _ in facts]
    surfaces = [subject_surfaces(f, patterns[i], tok) for i, f in enumerate(facts)]
    data = build_subject_data(facts, surfaces, examples, pool, background, seed=seed,
                              swaps=args.swaps, near_per=args.near_per,
                              bg_train=args.background_train, bg_val=args.background_val,
                              banned=banned)
    items = data["items"]
    S = len(data["subjects"])
    counts = Counter(f"{i['split']}/{i['kind']}" for i in items)
    print(json.dumps({"phase": "data", "dataset": dataset, "seed": seed, "items": len(items),
                      "subjects": S, "facts": len(facts), "pool_names": data["pool_names"],
                      "near_miss_names": data["near_miss_names"], "banned": len(banned),
                      "by_split_kind": dict(sorted(counts.items()))}), flush=True)
    if not any(i["split"] == "val" and i["subject"] >= 0 for i in items):
        raise ValueError("no validation positives (development families) to calibrate on")

    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    t0 = time.time()
    prompts = [i["prompt"] for i in items]
    _, span = extract_states(model, tok, prompts, [], args.subject_layers,
                             batch_size=args.batch_size, device=args.device)
    print(json.dumps({"phase": "features", "seconds": round(time.time() - t0, 1)}), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    y = torch.tensor([i["subject"] for i in items])
    kinds = [i["kind"] for i in items]
    train = torch.tensor([i["split"] == "train" for i in items])
    tr, va = train.nonzero().flatten(), (~train).nonzero().flatten()
    val_kinds = [kinds[i] for i in va.tolist()]
    yv = y[va]

    # exact-match reference on the same validation prompts (fact-level -> subject-level)
    elig = eligibility_matrix(tok, [prompts[i] for i in va.tolist()], patterns)
    fs = torch.tensor(data["fact_subject"])
    exact_subj = torch.zeros(len(va), S, dtype=torch.bool)
    for f in range(len(facts)):
        exact_subj[:, fs[f]] |= elig[:, f]
    exact_fired = torch.where(yv >= 0, exact_subj[torch.arange(len(va)), yv.clamp_min(0)],
                              exact_subj.any(1))
    exact_report = rates_by_kind(exact_fired, yv, val_kinds)

    reports = {}
    for Ls in args.subject_layers:
        sp = span[Ls]
        t1 = time.time()
        Ws, bs, smean, hist = fit_mil_subject(sp, tr, va, y, S, l2=args.mil_l2, epochs=args.mil_epochs,
                                              device=args.device, seed=seed, pair=True,
                                              hard_rounds=args.hard_rounds)
        ps = torch.sigmoid(mil_logits(sp, va, Ws, bs, smean.to(args.device), device=args.device))
        cal = calibrate_subject(ps, yv, val_kinds, min_recall=args.min_recall,
                                max_false_fire_other=args.max_false_fire_other,
                                max_false_fire_near=args.max_false_fire_near)
        t = cal["threshold"]
        learned_fired = torch.where(yv >= 0, ps[torch.arange(len(va)), yv.clamp_min(0)] >= t,
                                    ps.amax(1) >= t)
        reports[Ls] = {"subject_layer": Ls, "calibration": cal, "mil_history": hist,
                       "validation_by_kind": {
                           "learned": rates_by_kind(learned_fired, yv, val_kinds),
                           "learned_or_exact": rates_by_kind(learned_fired | exact_fired, yv, val_kinds),
                           "exact": exact_report},
                       "fit_seconds": round(time.time() - t1, 1),
                       "state": {"weight": Ws, "bias": bs, "mean": smean}}
        print(json.dumps({"phase": "subject_layer_done", "subject_layer": Ls, "calibration": cal,
                          "learned": reports[Ls]["validation_by_kind"]["learned"],
                          "exact": exact_report}), flush=True)

    deployable = {L: r for L, r in reports.items() if r["calibration"]["status"] != "infeasible"}

    def score(r):
        c = r["calibration"]
        return (c["status"] == "recall_floor_met",
                -(c["false_fire_other"] + c["false_fire_near_miss"]) if c["status"] == "recall_floor_met"
                else c["recall"])

    sel = max(deployable, key=lambda L: score(deployable[L])) if deployable else None
    summary = {"dataset": dataset, "seed": seed, "reference_dir": str(ref), "selected_subject_layer": sel,
               "data": {k: v for k, v in data.items() if k != "items"}, "by_split_kind": dict(counts),
               "layers": {L: {k: v for k, v in r.items() if k != "state"} for L, r in reports.items()},
               "command": sys.argv}
    out.mkdir(parents=True, exist_ok=True)
    if sel is None:
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
        print("NO DEPLOYABLE SUBJECT HEAD under the false-fire caps; not shipping a router.", file=sys.stderr)
        sys.exit(4)
    r = reports[sel]
    for gate in GATES:
        d = out / gate
        d.mkdir(parents=True, exist_ok=True)
        art = dict(reference)
        art["learned_subject"] = {
            "layer": sel, "gate": gate, "subject_mode": "mil_max_pair",
            "weight": r["state"]["weight"], "bias": r["state"]["bias"], "mean": r["state"]["mean"],
            "fact_subject": torch.tensor(data["fact_subject"]), "threshold": float(r["calibration"]["threshold"]),
            "subjects": data["subjects"], "calibration": r["calibration"], "reference_dir": str(ref)}
        torch.save(art, d / "fact_association_embeddings.pt")
        m = dict(manifest)
        m.update({"router_kind": "hybrid_learned_subject", "subject_gate": gate, "subject_layer": sel,
                  "reference_dir": str(ref), "subject_threshold": float(r["calibration"]["threshold"])})
        (d / "association_manifest.json").write_text(json.dumps(m, indent=2) + "\n")
        for name in ("association_examples.json", "linear_router_report.json"):
            if (ref / name).is_file():
                shutil.copy2(ref / name, d / name)
    summary["router_dirs"] = {g: str(out / g) for g in GATES}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({"phase": "done", "subject_layer": sel, "status": r["calibration"]["status"],
                      "router_dirs": summary["router_dirs"]}), flush=True)


if __name__ == "__main__":
    main()
