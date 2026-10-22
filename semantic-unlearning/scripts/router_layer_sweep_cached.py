#!/usr/bin/env python3
"""Layer sweep and robustness variants of the linear router, from cached states.

    python scripts/router_layer_sweep_cached.py --device cuda \
        --cache-dirs outputs/router_diag_v1/{mcf,zsre,mquake,rwku}/seed1
    # -> <cache>/layer_sweep/{sweep.json, errors_vs_layer.png, penultimate_vs_L19.md,
    #                         robustness.md, layer_choice.md}
    #    outputs/router_diag_v1/ROUTER_LAYER_CHOICE.md + errors_vs_layer_all.png

At every layer the SAME router is refit with the shipped recipe (fit_linear_router.py):
Linear + sigmoid heads, masked class-balanced BCE, L-BFGS; L2 (and PCA) by grouped CV over
prompt families on the fit split; the global cutoff calibrated on the calibration split
with the run's own rule (min recall / placement / ambiguity margin from the L19 artifact).
It is then scored on the cached official evaluation prompts (never used to fit or
calibrate): FPR, FNR, precision, recall, F1, correct-row rate, the null-prompt FPR,
perturbed-prompt recall and ECE. The refit at the router layer is checked against the
shipped router's cached decisions.

Decision rules
    deployed    the artifact's gate (threshold for MCF / ZsRE / MQuAKE, subject for RWKU)
    threshold   RWKU only: the counterfactual calibrated threshold gate. Under the subject
                gate firing is the name match, so the read layer only picks the row.
Eligibility (which heads may fire)
    exact       complete subject-token match (runtime rule)
    fuzzy       exact OR a case / accent / punctuation-insensitive word match allowing one
                edit per subject word of >= 5 letters (RWKU: any name surface)
    none        every head (what the heads alone can do)

Robustness variants (at --robust-layers):
    baseline       the shipped training set
    hardneg        + same subject / other relation and same relation / unseen subject
    aug            + paraphrase and adversarial lead-ins / context typos (positives and
                   negatives alike)
    hardneg_aug    both
    concat         baseline set, features = concatenated unit states of --concat-layers
    unmasked       hardneg_aug with every head trained on every prompt (heads must learn
                   the subject too; needed before the gate can be loosened)
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
from pathlib import Path
import re
import string
import time
import unicodedata

import torch
from torch.nn import functional as F

from linear_router import (
    DEFAULT_LAMBDAS,
    calibrate_threshold,
    decide_routes,
    fit_linear_router,
    score_queries,
    select_hyperparameters,
)
from router_diag_report import binary_metrics, calibration_report, probe_outcomes

VARIANTS = ("baseline", "hardneg", "aug", "hardneg_aug", "concat", "unmasked")
ELIGIBILITY = ("exact", "fuzzy", "none")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class Cache:
    def __init__(self, cache_dir):
        self.dir = Path(cache_dir)
        self.manifest = json.loads((self.dir / "manifest.json").read_text())
        self.facts = self.manifest["facts"]
        self.layers = self.manifest["layers"]
        self.sections = {}

    def section(self, name):
        if name not in self.sections:
            path = self.dir / f"probes_{name}.json"
            if not path.is_file():
                self.sections[name] = None
            else:
                states = torch.load(self.dir / f"states_{name}.pt", weights_only=False)
                self.sections[name] = {
                    "probes": json.loads(path.read_text()),
                    "routes": torch.load(self.dir / f"router_{name}.pt", weights_only=False),
                    "states": states["states"], "layers": states["layers"]}
        return self.sections[name]

    def features(self, name, layers, rows=None):
        """Unit block states of one layer, or concatenated unit states of several."""
        sec = self.section(name)
        parts = []
        for layer in layers:
            x = sec["states"][sec["layers"].index(int(layer))]
            x = x if rows is None else x[rows]
            parts.append(F.normalize(x.float(), dim=-1))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def training_set(cache, variant):
    """(section, rows) blocks making up the variant's training prompts."""
    blocks = [("train", None)]
    aug = cache.section("aug")
    if variant in ("hardneg", "aug", "hardneg_aug", "unmasked") and aug is None:
        raise ValueError(f"{cache.dir}: variant {variant} needs the cached 'aug' section")
    if aug is not None:
        groups = [p["group"] for p in aug["probes"]]
        hard = [i for i, g in enumerate(groups) if g.startswith("hardneg_")]
        augm = [i for i, g in enumerate(groups) if g.startswith("aug_")]
        if variant == "hardneg":
            blocks.append(("aug", hard))
        elif variant == "aug":
            blocks.append(("aug", augm))
        elif variant in ("hardneg_aug", "unmasked"):
            blocks.append(("aug", hard + augm))
    return blocks


def assemble(cache, variant, layers):
    xs, owners, eligible, splits, groups = [], [], [], [], []
    for section, rows in training_set(cache, variant):
        sec = cache.section(section)
        index = list(range(len(sec["probes"]))) if rows is None else list(rows)
        if not index:
            continue
        xs.append(cache.features(section, layers, torch.tensor(index)))
        eligible.append(sec["routes"]["eligible"][torch.tensor(index)].bool())
        for i in index:
            p = sec["probes"][i]
            owners.append(p["true_facts"][0] if p["should_fire"] and p["true_facts"] else -1)
            splits.append(p["split"])
            groups.append(p["group"])
    x = torch.cat(xs)
    eligible = torch.cat(eligible)
    owner = torch.tensor(owners)
    labels = torch.zeros_like(eligible)
    has = owner >= 0
    labels[has.nonzero(as_tuple=True)[0], owner[has]] = True
    if variant == "unmasked":
        eligible = torch.ones_like(eligible)
    return {"x": x, "labels": labels, "eligible": eligible, "owner": owner,
            "split": splits, "groups": groups}


# ---------------------------------------------------------------------------
# Fuzzy subject eligibility
# ---------------------------------------------------------------------------

def _plain_words(text):
    text = unicodedata.normalize("NFD", str(text))
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn").casefold()
    text = text.replace("'s", " ")
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    return text.split()


@lru_cache(maxsize=None)
def _within_one_edit(a, b):
    """Optimal-string-alignment distance <= 1 (one substitution, insertion, deletion or
    adjacent transposition)."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long = (a, b) if len(a) < len(b) else (b, a)
    i = j = edits = 0
    while i < len(short) and j < len(long):
        if short[i] == long[j]:
            i += 1
        else:
            edits += 1
            if edits > 1:
                return False
        j += 1
    return True


def _word_match(prompt_word, subject_word):
    if len(subject_word) < 5:
        return prompt_word == subject_word
    return _within_one_edit(prompt_word, subject_word)


def subject_surfaces(cache):
    if cache.manifest["dataset"] == "rwku":
        from rwku_fact_association_embeddings import rwku_subject_surfaces

        return [[_plain_words(s) for s in rwku_subject_surfaces(f["subject"])] for f in cache.facts]
    return [[_plain_words(f["subject"])] for f in cache.facts]


def fuzzy_eligibility(probes, surfaces, exact):
    out = exact.clone().bool()
    for row, p in enumerate(probes):
        if p.get("prompt") is None:
            continue
        words = _plain_words(p["prompt"])
        for fact, options in enumerate(surfaces):
            if out[row, fact]:
                continue
            for target in options:
                k = len(target)
                if k and any(all(_word_match(words[s + t], target[t]) for t in range(k))
                             for s in range(len(words) - k + 1)):
                    out[row, fact] = True
                    break
    return out


# ---------------------------------------------------------------------------
# Fit, calibrate, evaluate
# ---------------------------------------------------------------------------

PIPELINE_RULE = {"target_fpr": 0.0, "min_recall": 0.98, "placement_fraction": 0.1,
                 "ambiguity_margin": 0.5}   # run_<ds>_layer_sweep_one.sh fallback policy


def calibration_rule(cache, rule):
    """None = subject gate (no cutoff). Threshold-gate routers reuse their own calibration
    rule; RWKU's threshold counterfactual uses the pipeline's default policy."""
    if cache.manifest["gate_mode"] == "subject":
        return None if rule == "deployed" else dict(PIPELINE_RULE)
    fit = cache.manifest.get("router_fit") or {}
    return {"target_fpr": float(fit.get("target_fpr") or 0.0),
            "min_recall": fit.get("min_recall"),
            "placement_fraction": float(fit.get("threshold_placement_fraction") or 0.1),
            "ambiguity_margin": float(fit.get("ambiguity_margin", 0.5))}


def fit_one(data, hparams, device, lambdas, pca_dims, folds):
    fit = torch.tensor([s == "fit" for s in data["split"]])
    groups = [g for g, f in zip(data["groups"], fit.tolist()) if f]
    if hparams == "cv":
        l2, pca, _ = select_hyperparameters(data["x"][fit], data["labels"][fit], data["eligible"][fit],
                                            groups, lambdas=lambdas, pca_dims=pca_dims, folds=folds,
                                            device=device)
    else:
        l2, pca = hparams
    router = fit_linear_router(data["x"][fit], data["labels"][fit], data["eligible"][fit],
                               l2=l2, pca_dim=pca, device=device)
    router["l2"], router["pca_dim"] = float(l2), int(pca)
    return router


def logits_for(router, x):
    return score_queries(x, router["weight"], router["bias"], router["feature_mean"],
                         router["feature_components"])


def calibrate(router, data, eligible, rule_params):
    if rule_params is None:
        return float("-inf"), 0.0
    cal = torch.tensor([s == "calibration" for s in data["split"]])
    logits = logits_for(router, data["x"][cal])
    threshold, _ = calibrate_threshold(
        logits, eligible[cal], data["owner"][cal],
        target_fpr=rule_params["target_fpr"],
        min_recall=rule_params["min_recall"],
        ambiguity_margin=rule_params["ambiguity_margin"],
        placement_fraction=rule_params["placement_fraction"])
    return threshold, rule_params["ambiguity_margin"]


def route_auc(probes, logits, eligible):
    """Threshold-free: best true-head logit of should-fire prompts vs best eligible logit
    of must-not-fire prompts (no eligible head = -inf)."""
    masked = logits.masked_fill(~eligible, float("-inf"))
    pos, neg = [], []
    for i, p in enumerate(probes):
        if p["should_fire"] and p["true_facts"]:
            pos.append(float(masked[i, p["true_facts"]].max()))
        elif not p["should_fire"]:
            neg.append(float(masked[i].max()))
    if not pos or not neg:
        return None
    pos_t, neg_t = torch.tensor(pos, dtype=torch.float64), torch.tensor(neg, dtype=torch.float64)
    wins = 0.0
    for chunk in pos_t.split(2048):
        diff = chunk[:, None] - neg_t[None, :]
        both_inf = torch.isinf(chunk)[:, None] & torch.isinf(neg_t)[None, :]
        wins += float(((diff > 0) & ~both_inf).sum()) + 0.5 * float(((diff == 0) | both_inf).sum())
    return wins / (len(pos) * len(neg))


def evaluate(cache, router, layers, threshold, margin, eligibility_mode, fuzzy_cache):
    out = {}
    for section in ("eval", "null", "perturbed"):
        sec = cache.section(section)
        if sec is None:
            continue
        x = cache.features(section, layers)
        logits = logits_for(router, x)
        exact = sec["routes"]["eligible"].bool()
        if eligibility_mode == "exact":
            eligible = exact
        elif eligibility_mode == "none":
            eligible = torch.ones_like(exact)
        else:
            key = (section,)
            if key not in fuzzy_cache:
                fuzzy_cache[key] = fuzzy_eligibility(sec["probes"], subject_surfaces(cache), exact)
            eligible = fuzzy_cache[key]
        decision = decide_routes(logits, eligible, threshold, margin)
        routes = {"logits": logits, "eligible": eligible,
                  **{k: v.cpu() for k, v in decision.items()}}
        rows = probe_outcomes(sec["probes"], routes, cache.facts)
        if section == "eval":
            by_split = defaultdict(list)
            for r in rows:
                by_split[r["split"]].append(r)
            out["eval"] = binary_metrics(rows)
            out["eval_by_split"] = {k: binary_metrics(v) for k, v in by_split.items()}
            out["eval_route_auc"] = route_auc(sec["probes"], logits, eligible)
            out["ece_pair_level"] = calibration_report(sec["probes"], routes)["pair_level"]["ece"]
            out["_decisions"] = (decision["active"].cpu(), decision["fact"].cpu())
        elif section == "null":
            neg = [r for r in rows if not r["should_fire"]]
            out["null_fpr"] = sum(r["fired"] for r in neg) / len(neg) if neg else None
            out["null_fired"] = sum(r["fired"] for r in neg)
        else:
            out["perturbed_recall"] = binary_metrics(rows)["recall"]
            out["perturbed_correct_row"] = binary_metrics(rows)["correct_row_rate"]
    return out


def agreement_with_shipped(cache, decisions):
    shipped = cache.section("eval")["routes"]
    active, fact = decisions
    same = (active == shipped["active"].bool()) & (~active | (fact == shipped["fact"]))
    return float(same.float().mean())


def run_config(cache, *, variant, layers, rule, eligibility_modes, hparams, device, lambdas,
               pca_dims, folds, fuzzy_cache):
    data = assemble(cache, variant, layers)
    router = fit_one(data, hparams, device, lambdas, pca_dims, folds)
    rule_params = calibration_rule(cache, rule)
    results = {}
    for mode in eligibility_modes:
        cal_eligible = data["eligible"]
        if mode == "none":
            cal_eligible = torch.ones_like(cal_eligible)
        threshold, margin = calibrate(router, data, cal_eligible, rule_params)
        r = evaluate(cache, router, layers, threshold, margin, mode, fuzzy_cache)
        r["threshold_logit"] = threshold if math.isfinite(threshold) else None
        results[mode] = r
    return {"variant": variant, "layers": list(layers), "rule": rule, "l2": router["l2"],
            "pca_dim": router["pca_dim"], "converged": router["info"]["converged"],
            "train_prompts": int(data["x"].shape[0]), "by_eligibility": results}


def _strip(entry):
    out = json.loads(json.dumps({k: v for k, v in entry.items()}, default=str))
    for r in out.get("by_eligibility", {}).values():
        r.pop("_decisions", None)
    return out


# ---------------------------------------------------------------------------
# Plots and write-ups
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_errors_vs_layer(ax, sweep, title, router_layer, penultimate):
    for rule, style in (("deployed", "-"), ("threshold", ":")):
        rows = sorted((e for e in sweep if e["rule"] == rule), key=lambda e: e["layers"][0])
        if not rows:
            continue
        xs = [e["layers"][0] for e in rows]
        for metric, color in (("fpr", "#c62828"), ("fnr", "#ef6c00"), ("f1", "#1565c0")):
            ys = [e["by_eligibility"]["exact"]["eval"][metric] for e in rows]
            ys = [float("nan") if y is None else y for y in ys]
            ax.plot(xs, ys, style, color=color, marker="o", ms=3,
                    label=f"{metric.upper()}{'' if rule == 'deployed' else ' (threshold gate)'}")
    ax.axvline(router_layer, color="black", lw=0.8, ls="--")
    ax.axvline(penultimate, color="grey", lw=0.8, ls=":")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("read layer")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6)


def _fmt(v, d=3):
    return "–" if v is None else f"{v:.{d}f}"


def layer_table(sweep, rule="deployed"):
    rows = sorted((e for e in sweep if e["rule"] == rule), key=lambda e: e["layers"][0])
    lines = ["| layer | L2 | FPR | FNR | precision | recall | F1 | correct row | route AUC | null FPR | "
             "perturbed recall | ECE |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in rows:
        r = e["by_eligibility"]["exact"]
        m = r["eval"]
        lines.append(f"| L{e['layers'][0]} | {e['l2']:g} | {_fmt(m['fpr'])} | {_fmt(m['fnr'])} | "
                     f"{_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | "
                     f"{_fmt(m['correct_row_rate'])} | {_fmt(r.get('eval_route_auc'))} | "
                     f"{_fmt(r.get('null_fpr'))} | {_fmt(r.get('perturbed_recall'))} | "
                     f"{_fmt(r.get('ece_pair_level'))} |")
    return "\n".join(lines)


def best_layer(sweep, rule="deployed"):
    rows = [e for e in sweep if e["rule"] == rule and e["by_eligibility"]["exact"]["eval"]["f1"] is not None]
    if not rows:
        return None

    def key(e):
        m = e["by_eligibility"]["exact"]["eval"]
        return (round(m["f1"], 6), -(m["fpr"] or 0.0), -e["layers"][0])

    return max(rows, key=key)


def compare_layers(sweep, a, b, rule):
    def get(layer):
        for e in sweep:
            if e["rule"] == rule and e["layers"] == [layer]:
                return e["by_eligibility"]["exact"]
        return None

    ra, rb = get(a), get(b)
    if ra is None or rb is None:
        return None
    ma, mb = ra["eval"], rb["eval"]
    f1a, f1b = ma["f1"] or 0.0, mb["f1"] or 0.0
    improves = (f1b > f1a + 1e-9 and (mb["fpr"] or 0.0) <= (ma["fpr"] or 0.0) + 1e-9
                and (rb.get("null_fpr") or 0.0) <= (ra.get("null_fpr") or 0.0) + 1e-9)
    return {"rule": rule, f"L{a}": {k: ma[k] for k in ("fpr", "fnr", "precision", "recall", "f1",
                                                       "correct_row_rate")},
            f"L{b}": {k: mb[k] for k in ("fpr", "fnr", "precision", "recall", "f1", "correct_row_rate")},
            f"L{a}_null_fpr": ra.get("null_fpr"), f"L{b}_null_fpr": rb.get("null_fpr"),
            "routing_improves": bool(improves),
            "criterion": "higher F1 on the official eval prompts with no higher eval FPR and no "
                         "higher null-prompt FPR"}


DOWNSTREAM = {
    "mcf": "TAG=l16_paper_router{L}_v1 MCF_ROUTER=linear ROUTER_LAYER={L} bash scripts/run_l16_paper_one.sh mcf {S}",
    "zsre": "TAG=l16_paper_router{L}_v1 ROUTER_LAYER={L} bash scripts/run_l16_paper_one.sh zsre {S}",
    "mquake": "TAG=l16_paper_router{L}_v1 ROUTER_LAYER={L} bash scripts/run_l16_paper_one.sh mquake {S}",
    "rwku": "TAG=l16_paper_rwku_subject_router{L}_v1 RWKU_GATE=subject ROUTER_LAYER={L} "
            "bash scripts/run_l16_paper_one.sh rwku {S}",
}


def write_dataset_docs(cache, sweep, robust, out_dir):
    m = cache.manifest
    ds, router_layer, n = m["dataset"], m["router_layer"], m["n_layers"]
    penultimate = n - 2
    rules = ["deployed"] + (["threshold"] if ds == "rwku" else [])
    md = [f"# Layer choice: {ds} seed {m['seed']}", "",
          "Same router refit at every layer (shipped recipe), scored on the official eval "
          "prompts. Exact subject eligibility.", ""]
    verdicts = {}
    for rule in rules:
        best = best_layer(sweep, rule)
        md += [f"## Decision rule: {rule}", "", layer_table(sweep, rule), ""]
        if best:
            md.append(f"Best F1: L{best['layers'][0]} "
                      f"({_fmt(best['by_eligibility']['exact']['eval']['f1'])}).")
        cmp = compare_layers(sweep, router_layer, penultimate, rule)
        verdicts[rule] = cmp
        if cmp:
            md.append(f"L{router_layer} vs penultimate L{penultimate}: F1 "
                      f"{_fmt(cmp[f'L{router_layer}']['f1'])} vs {_fmt(cmp[f'L{penultimate}']['f1'])}, FPR "
                      f"{_fmt(cmp[f'L{router_layer}']['fpr'])} vs {_fmt(cmp[f'L{penultimate}']['fpr'])}, "
                      f"null FPR {_fmt(cmp[f'L{router_layer}_null_fpr'])} vs "
                      f"{_fmt(cmp[f'L{penultimate}_null_fpr'])} -> routing "
                      f"{'IMPROVES' if cmp['routing_improves'] else 'does not improve'}.")
        md.append("")
    timing = m.get("router_pass_timing_ms_per_prompt")
    if timing:
        md += ["## Pipeline cost", "",
               f"Routing pass, ms per prompt: full forward {timing['full_forward']:.2f}; " +
               "; ".join(f"{k} {v:.2f}" for k, v in timing.items()
                         if k.startswith("stop_after")) +
               f" ({timing['blocks']} blocks). The deployed early-write model "
               "(scripts/early_write.py) already makes two passes: pass 1 routes on the prompt "
               "(cached per prompt, not per generated token), pass 2 writes at L16. Reading a later "
               "layer does not add a pass, because the L16 write is below the read layer whichever "
               "it is. It only changes how far pass 1 must run if it is stopped at the router "
               "layer (today it runs the full model).", ""]
    pen = verdicts.get("deployed")
    md += ["## Penultimate layer: downstream run", ""]
    if pen and pen["routing_improves"]:
        md += [f"Routing improves at L{penultimate}; run the downstream Eff/Gen/Spe/PPL evaluation:", "",
               "```bash", DOWNSTREAM[ds].format(L=penultimate, S=m["seed"]), "```", ""]
    else:
        md += [f"Routing does not improve at L{penultimate} under the deployed rule, so the "
               "downstream evaluation is not run.", ""]
    if robust:
        md += ["## Robustness variants", "", robustness_table(robust), ""]
    (out_dir / "layer_choice.md").write_text("\n".join(md) + "\n")
    (out_dir / "penultimate_vs_L19.json").write_text(json.dumps(verdicts, indent=2, default=str) + "\n")
    return verdicts


def robustness_table(robust):
    lines = ["| variant | layers | eligibility | FPR | FNR | F1 | correct row | null FPR | perturbed recall | ECE |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for e in robust:
        for mode, r in e["by_eligibility"].items():
            m = r["eval"]
            lines.append(f"| {e['variant']} | {','.join(map(str, e['layers']))} | {mode} | "
                         f"{_fmt(m['fpr'])} | {_fmt(m['fnr'])} | {_fmt(m['f1'])} | "
                         f"{_fmt(m['correct_row_rate'])} | {_fmt(r.get('null_fpr'))} | "
                         f"{_fmt(r.get('perturbed_recall'))} | {_fmt(r.get('ece_pair_level'))} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def sweep_cache(cache_dir, args):
    cache = Cache(cache_dir)
    m = cache.manifest
    out_dir = cache.dir / "layer_sweep"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "sweep.json"
    done = json.loads(path.read_text()) if path.is_file() and not args.overwrite else {"sweep": [], "robust": []}
    have = {(e["variant"], tuple(e["layers"]), e["rule"]) for e in done["sweep"] + done["robust"]}
    fit = m.get("router_fit") or {}
    fixed = (float(fit.get("selected_l2", 1e-3)), int(fit.get("selected_pca_dim", 0) or 0))
    hparams = "cv" if args.hparams == "cv" else fixed
    lambdas = tuple(float(x) for x in args.lambdas.split(",")) if args.lambdas else DEFAULT_LAMBDAS
    # The shipped L2 is in the grid: the shipped CV picked it over a superset of this grid
    # with the same family folds, so the refit at the router layer selects it again.
    lambdas = tuple(sorted(set(lambdas) | {fixed[0]}))
    pca_dims = (tuple(int(x) for x in args.pca_dims.split(",")) if args.pca_dims else (fixed[1],))
    layers = [l for l in cache.layers if args.layers == "all" or l in
              {int(x) for x in args.layers.split(",")}]
    rules = ["deployed"] + (["threshold"] if m["dataset"] == "rwku" else [])
    fuzzy_cache = {}

    def save():
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"sweep": [_strip(e) for e in done["sweep"]],
                                   "robust": [_strip(e) for e in done["robust"]]}, indent=1) + "\n")
        tmp.replace(path)      # atomic: a concurrent --combine-only never reads half a file

    if args.combine_only:
        layers = []
    for layer in layers:
        for rule in rules:
            if ("baseline", (layer,), rule) in have:
                continue
            t0 = time.time()
            entry = run_config(cache, variant="baseline", layers=[layer], rule=rule,
                               eligibility_modes=("exact",), hparams=hparams, device=args.device,
                               lambdas=lambdas, pca_dims=pca_dims, folds=args.cv_folds,
                               fuzzy_cache=fuzzy_cache)
            if layer == m["router_layer"] and rule == "deployed":
                entry["agreement_with_shipped_router"] = agreement_with_shipped(
                    cache, entry["by_eligibility"]["exact"]["_decisions"])
            done["sweep"].append(entry)
            save()
            e = entry["by_eligibility"]["exact"]["eval"]
            print(json.dumps({"dataset": m["dataset"], "layer": layer, "rule": rule, "f1": e["f1"],
                              "fpr": e["fpr"], "fnr": e["fnr"], "l2": entry["l2"],
                              "seconds": round(time.time() - t0, 1)}), flush=True)
    penultimate = m["n_layers"] - 2
    robust_layers = sorted({int(x) for x in args.robust_layers.replace("penultimate", str(penultimate))
                            .split(",") if x})
    concat_layers = [int(x) for x in args.concat_layers.split(",")]
    for layer in [] if args.combine_only else robust_layers:
        for variant in [v for v in args.variants.split(",") if v]:
            feature_layers = concat_layers if variant == "concat" else [layer]
            if variant == "concat" and layer != robust_layers[0]:
                continue
            for rule in rules:
                key = (variant, tuple(feature_layers), rule)
                if key in have:
                    continue
                try:
                    entry = run_config(cache, variant=variant, layers=feature_layers, rule=rule,
                                       eligibility_modes=ELIGIBILITY, hparams=hparams,
                                       device=args.device, lambdas=lambdas, pca_dims=pca_dims,
                                       folds=args.cv_folds, fuzzy_cache=fuzzy_cache)
                except ValueError as error:
                    print(json.dumps({"skipped": variant, "reason": str(error)}), flush=True)
                    continue
                done["robust"].append(entry)
                have.add(key)
                save()
                print(json.dumps({"dataset": m["dataset"], "variant": variant, "layers": feature_layers,
                                  "rule": rule, **{mode: r["eval"]["f1"] for mode, r in
                                                   entry["by_eligibility"].items()}}), flush=True)
    sweep = [_strip(e) for e in done["sweep"]]
    robust = [_strip(e) for e in done["robust"]]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    plot_errors_vs_layer(ax, sweep, f"{m['dataset']} seed {m['seed']}", m["router_layer"], penultimate)
    fig.tight_layout()
    fig.savefig(out_dir / "errors_vs_layer.png", dpi=150)
    plt.close(fig)
    verdicts = write_dataset_docs(cache, sweep, robust, out_dir)
    if robust:
        (out_dir / "robustness.md").write_text(robustness_table(robust) + "\n")
    return cache, sweep, robust, verdicts


def combined_docs(results, out_root):
    out_root.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, axes = plt.subplots(1, len(results), figsize=(4.6 * len(results), 3.8), squeeze=False)
    md = ["# Why the router reads layer 19", "",
          "Same Linear + sigmoid router (shipped recipe, L-BFGS, CV-selected L2) refit at every "
          "layer from cached last-token states; errors on the official evaluation prompts, which are "
          "never used to fit or calibrate. Dashed line: L19; dotted: the penultimate layer.", "",
          "![errors vs layer](errors_vs_layer_all.png)", "",
          "| dataset | best-F1 layer | F1 there | F1 at L19 | L19 rank (of layers) | L19 vs penultimate | "
          "refit = shipped at L19 |", "|---|---|---|---|---|---|---|"]
    for ax, (cache, sweep, robust, verdicts) in zip(axes[0], results):
        m = cache.manifest
        n = m["n_layers"]
        plot_errors_vs_layer(ax, sweep, m["dataset"], m["router_layer"], n - 2)
        rule = "threshold" if m["dataset"] == "rwku" else "deployed"
        best = best_layer(sweep, rule)
        ranked = sorted((e for e in sweep if e["rule"] == rule),
                        key=lambda e: -(e["by_eligibility"]["exact"]["eval"]["f1"] or 0.0))
        at = next((e for e in ranked if e["layers"] == [m["router_layer"]]), None)
        rank = ranked.index(at) + 1 if at else None
        cmp = verdicts.get(rule) or {}
        shipped = next((e.get("agreement_with_shipped_router") for e in sweep
                        if e["rule"] == "deployed" and e["layers"] == [m["router_layer"]]), None)
        md.append(f"| {m['dataset']}{' (threshold gate)' if rule == 'threshold' else ''} | "
                  f"L{best['layers'][0] if best else '?'} | "
                  f"{_fmt(best['by_eligibility']['exact']['eval']['f1']) if best else '–'} | "
                  f"{_fmt(at['by_eligibility']['exact']['eval']['f1']) if at else '–'} | "
                  f"{rank}/{len(ranked)} | "
                  f"{'penultimate better' if cmp.get('routing_improves') else 'L19 not beaten'} | "
                  f"{_fmt(shipped)} |")
    fig.tight_layout()
    fig.savefig(out_root / "errors_vs_layer_all.png", dpi=150)
    plt.close(fig)
    md += ["", "Per-dataset tables, the penultimate comparison, pipeline cost and robustness "
           "variants: `<cache>/layer_sweep/layer_choice.md`.", ""]
    for cache, _, robust, _ in results:
        if robust:
            md += [f"## Robustness: {cache.manifest['dataset']}", "", robustness_table(robust), ""]
    (out_root / "ROUTER_LAYER_CHOICE.md").write_text("\n".join(md) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dirs", nargs="+", required=True)
    parser.add_argument("--out-root", default="outputs/router_diag_v1")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--hparams", choices=("cv", "fixed"), default="cv",
                        help="cv: L2 by grouped CV at each layer; fixed: the L19 router's L2/PCA")
    parser.add_argument("--lambdas", default="1e-5,1e-4,1e-3,1e-2,1e-1")
    parser.add_argument("--pca-dims", default="", help="default: the L19 router's PCA choice")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--robust-layers", default="19,penultimate")
    parser.add_argument("--concat-layers", default="16,19,23")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--combine-only", action="store_true",
                        help="fit nothing: rebuild the plots and write-ups from existing sweep.json")
    args = parser.parse_args(argv)
    if args.combine_only and args.overwrite:
        parser.error("--combine-only reads the existing sweep.json; --overwrite would discard it")
    results = [sweep_cache(d, args) for d in args.cache_dirs]
    combined_docs(results, Path(args.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
