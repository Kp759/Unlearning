#!/usr/bin/env python3
"""L19 router diagnostics from a cache (scripts/cache_router_states.py). CPU only.

    python scripts/router_diag_report.py --cache-dir outputs/router_diag_v1/mcf/seed1
    # -> <cache>/report/{summary.md, metrics.json, *.png}

What it shows, for the router the paper deploys:
  1. per-example scores against the 0.5 line, per split
       forget / paraphrase   p(true association)   (best over the true rows; RWKU held-out
                                                    probes may use any row of the person)
       neighborhood / retain max p over all associations
     Colour = the runtime decision: correct, false positive, false negative, misroute
     (fired, wrong row). Filled marker = the decision-relevant head was subject-eligible;
     hollow = no eligible head, so the router could not fire whatever the score.
     Top-k worst prompts with their text and the reason for each error.
  2. confusion matrices: binary (fired vs should fire) and N-way (fired row vs true row),
     FPR, FNR, precision, recall, F1 with Wilson intervals, per split and group
  3. RWKU: errors by probe type (fill-in-the-blank, QA, QA paraphrase, adversarial by
     attack, neighbours) and by how the person is named (full name, partial name, none)
  4. prompts with no association: how often the router fires (the deployment FPR), and
     how often an ungated head would (max p >= 0.5)
  5. perturbed prompts: recall per surface-perturbation family
  6. UMAP of 10 positives / 10 negatives at L16 and L19 (n_neighbors 5, fixed seed) with
     silhouette scores and leave-one-out probe accuracy
  7. calibration: ECE and reliability diagrams of the deployed probabilities
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import re

import torch
from torch.nn import functional as F

from linear_router import wilson

CATEGORIES = ("correct", "false_positive", "false_negative", "misroute")
COLORS = {"correct": "#2e7d32", "false_positive": "#c62828", "false_negative": "#ef6c00",
          "misroute": "#6a1b9a"}
SHORT = {"correct": "correct", "false_positive": "FP", "false_negative": "FN", "misroute": "misroute"}
SPLIT_ORDER = ("forget", "paraphrase", "neighborhood", "retain", "null")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

SAME_ANSWER_GROUP = "null_protected_same_answer_type"


def load_section(cache_dir, section, states=False):
    cache_dir = Path(cache_dir)
    probes_path = cache_dir / f"probes_{section}.json"
    if not probes_path.is_file():
        return None
    out = {"probes": json.loads(probes_path.read_text()),
           "routes": torch.load(cache_dir / f"router_{section}.pt", weights_only=False)}
    if states:
        out["states"] = torch.load(cache_dir / f"states_{section}.pt", weights_only=False)
    return out


def display_text(probe, limit=220):
    """The user-visible request (RWKU: the question inside the chat template)."""
    if probe.get("prompt") is None:
        return f"<{probe['meta'].get('prefix_tokens', '?')}-token PPL prefix>"
    text = probe["meta"].get("query") or probe["prompt"]
    if "<|" in text:
        text = re.sub(r"<\|[^|]*\|>", " ", text)
        text = text.split("user", 1)[-1] if "user" in text else text
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


# ---------------------------------------------------------------------------
# Per-probe outcomes (pure)
# ---------------------------------------------------------------------------

def probe_outcomes(probes, routes, facts=None):
    """One dict per probe: score, decision, category, reason."""
    probs = torch.sigmoid(routes["logits"].float())
    eligible = routes["eligible"].bool()
    active, fact, ambiguous = routes["active"].bool(), routes["fact"], routes["ambiguous"].bool()
    # relation gate (scripts/fit_relation_gate.py): None for routers without one
    relation_pass = routes.get("relation_pass")
    subjects = [str(f.get("subject", "")) for f in (facts or [])]
    rows = []
    for i, p in enumerate(probes):
        true = list(p["true_facts"])
        fired = bool(active[i])
        fired_fact = int(fact[i]) if fired else None
        any_eligible = bool(eligible[i].any())
        best_eligible = float(probs[i][eligible[i]].max()) if any_eligible else 0.0
        if p["should_fire"]:
            score = float(probs[i, true].max()) if true else 0.0
            relevant_eligible = bool(eligible[i, true].any()) if true else False
        else:
            score = float(probs[i].max())
            relevant_eligible = any_eligible
        if p["should_fire"] and fired and fired_fact in true:
            category = "correct"
        elif p["should_fire"] and fired:
            category = "misroute"
        elif p["should_fire"]:
            category = "false_negative"
        elif fired:
            category = "false_positive"
        else:
            category = "correct"
        reason = ""
        if category == "false_negative":
            if not relevant_eligible:
                reason = ("subject_in_text_not_token_matched" if _named(p)
                          else "subject_absent_from_prompt")
            elif relation_pass is not None and not bool(relation_pass[i, true].any()):
                reason = "relation_gate"
            elif bool(ambiguous[i]):
                reason = "ambiguous_top2"
            else:
                reason = "below_threshold"
        elif category == "false_positive":
            reason = "identical_to_forget_prompt" if p["meta"].get("identical_to_forget_prompt") \
                else "eligible_head_above_threshold"
        elif category == "misroute":
            same = subjects and true and subjects[fired_fact].casefold() == subjects[true[0]].casefold()
            reason = "same_subject_other_row" if same else "other_subject_row"
        rows.append({"index": i, "split": p["split"], "group": p["group"],
                     "probe_type": p.get("probe_type", ""), "should_fire": p["should_fire"],
                     "fired": fired, "fired_fact": fired_fact, "true_facts": true,
                     "score": score, "best_eligible_score": best_eligible,
                     "relevant_eligible": relevant_eligible, "any_eligible": any_eligible,
                     "category": category, "reason": reason})
    return rows


def _named(probe):
    """Is the subject (RWKU: any name surface) in the prompt, ignoring case and spacing?
    True for a not-eligible prompt means a casing / spacing / tokenization miss; False
    means the subject is reworded away (pronoun, description, other language)."""
    if probe["meta"].get("name_surfaces_found"):
        return True
    text = " ".join((probe.get("prompt") or "").casefold().split())
    subject = " ".join(str(probe.get("subject", "")).casefold().split())
    return bool(subject) and subject in text


def binary_metrics(rows):
    tp = sum(r["should_fire"] and r["fired"] for r in rows)
    fn = sum(r["should_fire"] and not r["fired"] for r in rows)
    fp = sum(not r["should_fire"] and r["fired"] for r in rows)
    tn = sum(not r["should_fire"] and not r["fired"] for r in rows)
    correct = sum(r["category"] == "correct" and r["should_fire"] for r in rows)
    misroute = sum(r["category"] == "misroute" for r in rows)

    def ratio(a, b):
        return a / b if b else None

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None and precision + recall > 0 else None)
    return {"n": len(rows), "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "fpr": ratio(fp, fp + tn), "fnr": ratio(fn, tp + fn),
            "precision": precision, "recall": recall, "f1": f1,
            "fpr_ci": wilson(fp, fp + tn), "fnr_ci": wilson(fn, tp + fn),
            "correct_row_rate": ratio(correct, tp + fn), "misroutes": misroute}


def grouped_metrics(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    return {k: binary_metrics(v) for k, v in sorted(groups.items())}


def nway_confusion(rows, n_facts, label_of=None):
    """[(K+1), (K+1)] counts: rows = true label (last = none), cols = fired label (last = none).
    label_of maps a fact row to a label index (default identity); K = number of labels."""
    label_of = label_of or (lambda f: f)
    k = (max(label_of(f) for f in range(n_facts)) + 1) if n_facts else 0
    matrix = torch.zeros((k + 1, k + 1), dtype=torch.long)
    for r in rows:
        true = label_of(min(r["true_facts"])) if r["should_fire"] and r["true_facts"] else k
        if r["should_fire"] and r["fired"] and r["fired_fact"] in r["true_facts"]:
            fired = true
        else:
            fired = label_of(r["fired_fact"]) if r["fired"] else k
        matrix[true, fired] += 1
    return matrix


def expected_calibration_error(confidence, label, bins=10):
    confidence = torch.as_tensor(confidence, dtype=torch.float64)
    label = torch.as_tensor(label, dtype=torch.float64)
    if confidence.numel() == 0:
        return None, []
    edges = torch.linspace(0, 1, bins + 1, dtype=torch.float64)
    ece, table = 0.0, []
    for b in range(bins):
        low, high = edges[b], edges[b + 1]
        mask = (confidence >= low) & ((confidence < high) if b < bins - 1 else (confidence <= high))
        n = int(mask.sum())
        if not n:
            table.append({"bin": [float(low), float(high)], "n": 0})
            continue
        conf, acc = float(confidence[mask].mean()), float(label[mask].mean())
        ece += n / confidence.numel() * abs(acc - conf)
        table.append({"bin": [float(low), float(high)], "n": n, "confidence": conf, "accuracy": acc})
    return ece, table


def calibration_report(probes, routes, bins=10):
    """Pair level: every subject-eligible (prompt, head), label = head is a true row.
    Prompt level: the plotted score, label = should fire."""
    probs = torch.sigmoid(routes["logits"].float())
    eligible = routes["eligible"].bool()
    conf, label = [], []
    for i, p in enumerate(probes):
        true = set(p["true_facts"])
        for f in eligible[i].nonzero(as_tuple=True)[0].tolist():
            conf.append(float(probs[i, f]))
            label.append(float(f in true))
    pair_ece, pair_table = expected_calibration_error(conf, label, bins)
    return {"pair_level": {"ece": pair_ece, "pairs": len(conf), "bins": pair_table},
            "note": "deployed probabilities: the calibrated bias moves the cutoff to 0.5 for "
                    "recall, so they are not meant to be calibrated probabilities"}


# ---------------------------------------------------------------------------
# Separability (UMAP numbers)
# ---------------------------------------------------------------------------

def silhouette(x, labels, metric="cosine"):
    x = torch.as_tensor(x, dtype=torch.float64)
    labels = torch.as_tensor(labels)
    if metric == "cosine":
        xn = F.normalize(x, dim=-1)
        dist = (1 - xn @ xn.T).clamp_min(0)
    else:
        dist = torch.cdist(x, x)
    scores = []
    for i in range(x.shape[0]):
        same = labels == labels[i]
        same[i] = False
        other = labels != labels[i]
        if not same.any() or not other.any():
            continue
        a, b = float(dist[i, same].mean()), float(dist[i, other].mean())
        scores.append((b - a) / max(a, b) if max(a, b) > 0 else 0.0)
    return sum(scores) / len(scores) if scores else None


def _fit_logistic(x, y, l2=1e-2, iters=200):
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=iters, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ w + b, y) + 0.5 * l2 * (w * w).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach(), b.detach()


def loo_probe_accuracy(x, labels, l2=1e-2):
    """Leave-one-out accuracy of an L2 logistic probe and of 1-NN (cosine), on unit vectors."""
    x = F.normalize(torch.as_tensor(x, dtype=torch.float64), dim=-1)
    y = torch.as_tensor(labels, dtype=torch.float64)
    hits_lr = hits_nn = 0
    for i in range(x.shape[0]):
        keep = torch.arange(x.shape[0]) != i
        mean = x[keep].mean(dim=0)
        w, b = _fit_logistic(x[keep] - mean, y[keep], l2)
        hits_lr += int(((x[i] - mean) @ w + b > 0).item() == bool(y[i]))
        sims = x[keep] @ x[i]
        hits_nn += int(y[keep][int(sims.argmax())] == y[i])
    n = x.shape[0]
    return {"logistic": hits_lr / n, "nearest_neighbour": hits_nn / n, "n": n}


def pick_umap_points(probes, routes, per_class=20, seed=0, negatives="hard"):
    """20 should-fire (forget / paraphrase) and 20 must-not-fire eval probes. 'hard'
    negatives prefer prompts with a subject-eligible head (the ones the gate lets through)."""
    rng = random.Random(seed)
    eligible = routes["eligible"].bool()
    pos = [i for i, p in enumerate(probes) if p["should_fire"] and p["split"] in ("forget", "paraphrase")]
    neg = [i for i, p in enumerate(probes) if not p["should_fire"]]
    pos = rng.sample(pos, min(per_class, len(pos)))
    if negatives == "hard":
        hard = [i for i in neg if bool(eligible[i].any())]
        easy = [i for i in neg if not bool(eligible[i].any())]
        chosen = rng.sample(hard, min(per_class, len(hard)))
        chosen += rng.sample(easy, min(per_class - len(chosen), len(easy)))
    else:
        chosen = rng.sample(neg, min(per_class, len(neg)))
    return pos, chosen


def embed_2d(x, n_neighbors=5, seed=0):
    try:
        import umap

        reducer = umap.UMAP(n_neighbors=min(int(n_neighbors), x.shape[0] - 1), random_state=int(seed),
                            metric="cosine", min_dist=0.1)
        return torch.as_tensor(reducer.fit_transform(x.numpy())), "umap"
    except ImportError:
        centred = x - x.mean(dim=0)
        _, _, vh = torch.linalg.svd(centred, full_matrices=False)
        return centred @ vh[:2].T, "pca (umap-learn not installed)"


def separability(eval_section, layers=(16, 19), per_class=20, seed=0, negatives="hard"):
    states = eval_section["states"]
    probes, routes = eval_section["probes"], eval_section["routes"]
    pos, neg = pick_umap_points(probes, routes, per_class, seed, negatives)
    index = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    out = {"positives": pos, "negatives": neg, "negatives_mode": negatives, "layers": {}}
    for layer in layers:
        if layer not in states["layers"]:
            continue
        x = F.normalize(states["states"][states["layers"].index(layer)][index].double(), dim=-1)
        coords, method = embed_2d(x.float(), seed=seed)
        out["layers"][layer] = {
            "method": method, "coords": coords.tolist(),
            "silhouette_cosine_full_dim": silhouette(x, labels, "cosine"),
            "silhouette_2d": silhouette(coords, labels, "euclidean"),
            "loo": loo_probe_accuracy(x, labels),
        }
    out["texts"] = [display_text(probes[i], 90) for i in index]
    out["labels"] = labels
    out["index"] = index
    return out


# ---------------------------------------------------------------------------
# RWKU deep-dive
# ---------------------------------------------------------------------------

def name_tag(probe):
    found = probe["meta"].get("name_surfaces_found")
    if found is None:
        return "n/a"
    if not found:
        return "no_name_surface"
    canonical = probe["meta"].get("target_subject", probe["subject"])
    return "full_name" if any(f.casefold() == canonical.casefold() for f in found) else "partial_name_only"


def rwku_breakdown(probes, rows):
    by_type = defaultdict(list)
    by_tag = defaultdict(list)
    for p, r in zip(probes, rows):
        by_type[p.get("probe_type", "")].append(r)
        by_tag[(p["group"] == "neighbors", name_tag(p))].append(r)
    table = {}
    for kind, items in sorted(by_type.items()):
        m = binary_metrics(items)
        cats = Counter(r["category"] for r in items)
        table[kind] = {"n": len(items), "should_fire": sum(r["should_fire"] for r in items),
                       "fired": sum(r["fired"] for r in items), **{c: cats.get(c, 0) for c in CATEGORIES},
                       "fnr": m["fnr"], "fpr": m["fpr"],
                       "reasons": dict(Counter(r["reason"] for r in items if r["reason"]))}
    tags = {}
    for (is_neighbor, tag), items in sorted(by_tag.items()):
        cats = Counter(r["category"] for r in items)
        tags[f"{'neighbors' if is_neighbor else 'about_target'}/{tag}"] = {
            "n": len(items), "fired": sum(r["fired"] for r in items),
            **{c: cats.get(c, 0) for c in CATEGORIES}}
    return {"by_probe_type": table, "by_name_mention": tags}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


LONG = {"correct": "correct", "false_positive": "false fire (must not fire, fired)",
        "false_negative": "missed (should fire, did not)", "misroute": "wrong fact fired"}
GATE_Y = -0.13     # where prompts no head may fire on (subject not matched) are drawn


def plot_score(row):
    """The score the decision is made on: should fire -> p(correct fact's head); must not fire ->
    p(best head the subject gate lets through). None when the gate lets no relevant head through."""
    if row["should_fire"]:
        return row["score"] if row["relevant_eligible"] else None
    return row["best_eligible_score"] if row["any_eligible"] else None


def _count(n, word):
    plural = {"false fire": "false fires", "wrong fact fired": "wrong facts fired"}
    return f"{n:,} {word if n == 1 else plural.get(word, word)}"


def plot_scores(rows, path, title, threshold_note):
    """One panel per split, one point per prompt that some head scores, sorted by score. Above
    the dashed line the head fires. Prompts whose subject the gate does not match cannot fire;
    they are counted in the grey band at the bottom instead of being drawn."""
    plt = _plt()
    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6 * len(splits), 4.8), squeeze=False)
    short = {"correct": "correct", "false_positive": "false fire", "false_negative": "missed",
             "misroute": "wrong fact fired"}
    for ax, split in zip(axes[0], splits):
        items = [r for r in rows if r["split"] == split]
        scored = sorted((r for r in items if plot_score(r) is not None), key=plot_score)
        blocked = [r for r in items if plot_score(r) is None]
        for category in CATEGORIES:
            pts = [(k, plot_score(r)) for k, r in enumerate(scored) if r["category"] == category]
            if pts:
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=10 if len(scored) > 400 else 24, color=COLORS[category],
                           label=LONG[category], zorder=3 if category != "correct" else 2)
        ax.axhspan(-0.24, -0.04, color="#eeeeee", zorder=0)
        gate = Counter(r["category"] for r in blocked)
        gate_text = ", ".join(_count(gate[c], short[c]) for c in CATEGORIES if gate[c]) or "none"
        ax.text(0.02, -0.14, f"subject not matched, no head may fire: {len(blocked):,} prompts\n({gate_text})",
                fontsize=7, transform=ax.get_yaxis_transform(), va="center")
        ax.axhline(0.5, color="black", lw=0.9, ls="--")
        ax.text(0.99, 0.52, "fires above this line", fontsize=7, ha="right", transform=ax.get_yaxis_transform())
        counts = Counter(r["category"] for r in items)
        errors = ", ".join(_count(counts[c], short[c]) for c in CATEGORIES[1:] if counts[c]) or "no errors"
        ax.set_title(f"{split}: {len(items):,} prompts\n{counts['correct']:,} correct · {errors}", fontsize=9)
        ax.set_xlabel(f"the {len(scored):,} prompts a head scores, sorted by score")
        ax.set_xlim(-1, max(1, len(scored)))
        if not scored:
            ax.set_xticks([])
            ax.text(0.5, 0.5, "no prompt names a protected subject:\nno head scores any of them", fontsize=8,
                    ha="center", va="center", transform=ax.transAxes, color="#555555")
        ax.set_ylim(-0.24, 1.03)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0][0].set_ylabel("probability of the deciding head\n(should fire: the correct fact's head;\n"
                          "must not fire: the best head the gate allows)", fontsize=8)
    seen = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle(f"{title} — every evaluated prompt ({threshold_note})", fontsize=10)
    fig.tight_layout(rect=(0, 0.07, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_binary_confusions(per_split, path, title):
    plt = _plt()
    names = [s for s in SPLIT_ORDER if s in per_split] + ["all"]
    fig, axes = plt.subplots(1, len(names), figsize=(3.0 * len(names), 3.2), squeeze=False)
    for ax, name in zip(axes[0], names):
        m = per_split[name]
        grid = [[m["tp"], m["fn"]], [m["fp"], m["tn"]]]
        ax.imshow([[math.log1p(v) for v in row] for row in grid], cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{grid[a][b]:,}", ha="center", va="center", fontsize=11,
                        color="white" if math.log1p(grid[a][b]) > 0.6 * math.log1p(max(map(max, grid))) else "black")
        ax.set_xticks([0, 1], ["classifier fired", "did not fire"], fontsize=8)
        ax.set_yticks([0, 1], ["should fire", "must not fire"], fontsize=8)
        fpr = "–" if m["fpr"] is None else f"{m['fpr']:.3f}"
        fnr = "–" if m["fnr"] is None else f"{m['fnr']:.3f}"
        ax.set_title(f"{name}\nfalse-fire rate {fpr} · miss rate {fnr}", fontsize=8)
    fig.suptitle(f"{title}: correct cells on the diagonal (top-left, bottom-right), errors off it", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def compact_nway(matrix, labels):
    """Keep the facts involved in an error as their own row / column, merge every other fact
    (all routed correctly) into one, keep 'none' last. Returns (matrix, row names, column names)."""
    k = matrix.shape[0] - 1
    off = matrix.clone()
    off.fill_diagonal_(0)
    involved = [i for i in range(k) if int(off[i].sum()) or int(off[:, i].sum())]
    others = [i for i in range(k) if i not in involved]
    groups = [[i] for i in involved] + ([others] if others else []) + [[k]]
    names = [labels[i] for i in involved] + ([f"other {len(others)} facts"] if others else [])
    out = torch.zeros((len(groups), len(groups)), dtype=torch.long)
    for a, ga in enumerate(groups):
        for b, gb in enumerate(groups):
            out[a, b] = matrix[ga][:, gb].sum()
    return out, names + ["none: must not fire"], names + ["none: did not fire"]


def plot_nway(matrix, labels, path, title):
    """Which fact the router fired (columns) for which fact it should have fired (rows). Facts
    with no error are merged into one row / column; every cell shows its count."""
    plt = _plt()
    m, rows, cols = compact_nway(matrix, labels)
    k = m.shape[0]
    fig, ax = plt.subplots(figsize=(max(5.5, 0.62 * k + 3.2), max(4.8, 0.5 * k + 2.4)))
    shown = torch.log1p(m.float())
    ax.imshow(shown.numpy(), cmap="Blues")
    top = float(shown.max()) or 1.0
    for a in range(k):
        for b in range(k):
            v = int(m[a, b])
            if v or a == b:
                err = a != b and v
                ax.text(b, a, f"{v:,}", ha="center", va="center", fontsize=8,
                        color="#c62828" if err else ("white" if float(shown[a, b]) > 0.6 * top else "black"),
                        fontweight="bold" if err else "normal")
    ax.set_xticks(range(k), cols, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(k), rows, fontsize=7)
    ax.set_xlabel("what the classifier did (fired fact)")
    ax.set_ylabel("what should happen (correct fact)")
    ax.set_title(f"{title}\ncorrect on the diagonal; red = errors (last column: missed, last row: "
                 "false fires)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_reliability(table, path, title):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(3.8, 3.6))
    pts = [(b["confidence"], b["accuracy"], b["n"]) for b in table if b["n"]]
    if pts:
        xs, ys, ns = zip(*pts)
        ax.scatter(xs, ys, s=[10 + 60 * n / max(ns) for n in ns], color="#1565c0")
        ax.plot(xs, ys, color="#1565c0", lw=0.8)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=0.8)
    ax.set_xlabel("mean predicted p")
    ax.set_ylabel("fraction correct")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


POS_COLOR, NEG_COLOR = "#1565c0", "#c62828"


def plot_umap_sentences(panels, texts, labels, wrong, path, title, notes=None, confusion=None,
                        text_limit=82):
    """UMAP panels with every point numbered, the numbered sentences listed underneath
    (should fire in blue, must not fire in red) and an optional 2x2 confusion on the right.

    panels: [(panel title, coords [n, 2])]; labels[i] 1 = should fire; wrong: indices the router
    gets wrong (black ring, "✗" in the list); notes[i]: a short tag after the sentence;
    confusion: {"tp", "fn", "fp", "tn", "title"}."""
    plt = _plt()
    from matplotlib.gridspec import GridSpec

    pos = [i for i, l in enumerate(labels) if l == 1]
    neg = [i for i, l in enumerate(labels) if l == 0]
    number = {i: n + 1 for n, i in enumerate(pos + neg)}
    ncols = len(panels) + (1 if confusion else 0)
    lines = max(len(pos), len(neg))
    fig = plt.figure(figsize=(max(13.0, 4.6 * ncols), 4.6 + 0.19 * lines + 0.9))
    gs = GridSpec(2, ncols, height_ratios=[4.6, 0.19 * lines + 0.9], figure=fig,
                  width_ratios=[1.0] * len(panels) + ([0.75] if confusion else []))
    for c, (ptitle, coords) in enumerate(panels):
        ax = fig.add_subplot(gs[0, c])
        coords = torch.as_tensor(coords)
        for i in range(len(labels)):
            color = POS_COLOR if labels[i] == 1 else NEG_COLOR
            ax.scatter(coords[i, 0], coords[i, 1], color=color, s=46, alpha=0.9, zorder=2)
            if i in wrong:
                ax.scatter(coords[i, 0], coords[i, 1], s=170, facecolors="none", edgecolors="black",
                           linewidths=1.3, zorder=3)
            ax.annotate(str(number[i]), (float(coords[i, 0]), float(coords[i, 1])), fontsize=7,
                        color=color, xytext=(4, 3), textcoords="offset points", zorder=4)
        ax.set_title(ptitle, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    if confusion:
        ax = fig.add_subplot(gs[0, ncols - 1])
        grid = [[confusion["tp"], confusion["fn"]], [confusion["fp"], confusion["tn"]]]
        ax.imshow([[1, 0], [0, 1]], cmap="Greens", vmin=-1, vmax=2)
        for a in range(2):
            for b in range(2):
                err = a != b and grid[a][b]
                ax.text(b, a, str(grid[a][b]), ha="center", va="center", fontsize=16,
                        color="#c62828" if err else "black", fontweight="bold" if err else "normal")
        ax.set_xticks([0, 1], [confusion.get("fired", "classifier fired"), "did not fire"], fontsize=8)
        ax.set_yticks([0, 1], ["should fire", "must not fire"], fontsize=8)
        ax.set_title(confusion.get("title", "decisions on these sentences"), fontsize=9)
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    for col, (members, color, head) in enumerate(((pos, POS_COLOR, f"Should fire ({len(pos)})"),
                                                  (neg, NEG_COLOR, f"Must not fire ({len(neg)})"))):
        x = 0.0 if col == 0 else 0.5
        ax.text(x, 1.0, head, color=color, fontsize=9, fontweight="bold", va="top", transform=ax.transAxes)
        for row, i in enumerate(members):
            text = texts[i] if len(texts[i]) <= text_limit else texts[i][:text_limit - 1] + "…"
            tag = (" ✗" if i in wrong else "") + (f"  [{notes[i]}]" if notes and notes[i] else "")
            ax.text(x, 1.0 - (row + 1.3) / (lines + 1.6), f"{number[i]:>2}. {text}{tag}", color=color,
                    fontsize=7, va="top", transform=ax.transAxes)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_umap(sep, path, title):
    panels = []
    for layer in sorted(sep["layers"]):
        info = sep["layers"][layer]
        panels.append((f"L{layer}: linear probe {info['loo']['logistic']:.2f} (leave-one-out), "
                       f"silhouette {info['silhouette_cosine_full_dim']:.2f}", info["coords"]))
    wrong = {k for k, w in enumerate(sep.get("wrong", [])) if w}
    plot_umap_sentences(panels, sep["texts"], sep["labels"], wrong, path,
                        f"{title} — {sum(sep['labels'])} should-fire and {len(sep['labels']) - sum(sep['labels'])} "
                        f"must-not-fire evaluation prompts (UMAP; ring / ✗ = the classifier gets it wrong)")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt(value, digits=3):
    return "–" if value is None else f"{value:.{digits}f}"


def worst_tables(probes, rows, k):
    def line(r):
        p = probes[r["index"]]
        fired = "-" if r["fired_fact"] is None else str(r["fired_fact"])
        return (f"| {r['split']}/{r['group']} | {_fmt(r['score'])} | {fired} | "
                f"{','.join(map(str, r['true_facts'])) or '-'} | {r['reason']} | "
                f"{display_text(p).replace('|', '/')} |")

    header = "| split/group | score | fired | true | reason | prompt |\n|---|---|---|---|---|---|"
    fn = sorted((r for r in rows if r["category"] == "false_negative"), key=lambda r: r["score"])[:k]
    fp = sorted((r for r in rows if r["category"] == "false_positive"), key=lambda r: -r["score"])[:k]
    mis = sorted((r for r in rows if r["category"] == "misroute"), key=lambda r: r["score"])[:k]
    out = []
    for name, items in (("False negatives (lowest p(true))", fn),
                        ("False positives (highest score)", fp), ("Misroutes", mis)):
        out.append(f"### {name}: {len(items)} shown\n")
        out.append(header if items else "none")
        out += [line(r) for r in items]
        out.append("")
    return "\n".join(out)


def metrics_table(metrics):
    lines = ["| set | n | TP | FN | FP | TN | FPR | FNR | precision | recall | F1 | correct row |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, m in metrics.items():
        lines.append(f"| {name} | {m['n']} | {m['tp']} | {m['fn']} | {m['fp']} | {m['tn']} | "
                     f"{_fmt(m['fpr'])} | {_fmt(m['fnr'])} | {_fmt(m['precision'])} | "
                     f"{_fmt(m['recall'])} | {_fmt(m['f1'])} | {_fmt(m['correct_row_rate'])} |")
    return "\n".join(lines)


def build_report(cache_dir, out_dir=None, top_k=15, umap_negatives="hard", seed=0):
    cache_dir = Path(cache_dir)
    out_dir = Path(out_dir or cache_dir / "report")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    facts = manifest["facts"]
    dataset, layer = manifest["dataset"], manifest["router_layer"]
    gate = manifest["gate_mode"]
    rule = ("subject gate: fires on any protected-name match; p only picks the row"
            if gate == "subject" else "fires the best eligible head at p >= 0.5 (calibrated bias)")
    title = f"{dataset} seed {manifest['seed']} — L{layer} classifier"
    ev = load_section(cache_dir, "eval", states=True)
    rows = probe_outcomes(ev["probes"], ev["routes"], facts)
    per_split = grouped_metrics(rows, "split")
    per_split["all"] = binary_metrics(rows)
    metrics = {"dataset": dataset, "seed": manifest["seed"], "router_layer": layer, "gate": gate,
               "eval": {"by_split": per_split, "by_group": grouped_metrics(rows, "group")},
               "reasons": dict(Counter(f"{r['category']}:{r['reason']}" for r in rows if r["reason"]))}
    plot_scores(rows, out_dir / "scores_eval.png", title, rule)
    plot_binary_confusions(per_split, out_dir / "confusion_binary.png", title)
    nway = nway_confusion(rows, len(facts))
    plot_nway(nway, [f"{i}: {str(f.get('subject', ''))[:22]}" for i, f in enumerate(facts)] + ["none"],
              out_dir / "confusion_nway.png", f"{title}: which fact fired")
    metrics["eval"]["nway"] = {"matrix": nway.tolist(),
                               "should_fire_routed_to_a_true_row": per_split["all"]["correct_row_rate"]}
    md = [f"# Router diagnostics: {title}", "",
          f"Router: `{manifest['router_dir']}` · gate `{gate}` · {rule} · subject match "
          f"`{manifest.get('subject_match', 'exact')}`.",
          f"Probes: eval {len(ev['probes'])}. Hook parity (real bank hook vs cached decision): "
          f"{json.dumps({k: v['route_mismatches'] for k, v in manifest.get('hook_parity', {}).items()})} "
          "mismatches.", "",
          "## 1. Scores against the decision line", "",
          "Each point is one evaluated prompt that some head scores, sorted by score; above the dashed line "
          "(p = 0.5) the head fires. Should-fire prompts are scored by the correct fact's head, must-not-fire "
          "prompts by the best head the subject gate allows. Prompts whose subject the gate does not match "
          "cannot fire and are counted in the grey band.", "",
          "![scores](scores_eval.png)", "",
          worst_tables(ev["probes"], rows, top_k),
          "## 2. Confusion matrices", "", "![binary](confusion_binary.png)", "",
          metrics_table(per_split), "",
          "Error reasons: " + ", ".join(f"{k} {v}" for k, v in sorted(metrics["reasons"].items())), "",
          "![nway](confusion_nway.png)", ""]

    if dataset == "rwku":
        persons = sorted({_norm_name(f["subject"]) for f in facts})
        person_of = {i: persons.index(_norm_name(f["subject"])) for i, f in enumerate(facts)}
        pm = nway_confusion(rows, len(facts), label_of=lambda f: person_of[f])
        plot_nway(pm, persons + ["none"], out_dir / "confusion_person.png",
                  f"{title}: fired person vs true person")
        deep = rwku_breakdown(ev["probes"], rows)
        metrics["rwku"] = {**deep, "person_matrix": pm.tolist(), "persons": persons}
        md += ["## 3. RWKU error deep-dive", "", "![person](confusion_person.png)", "",
               "| probe type | n | should fire | fired | correct | FP | FN | misroute | FNR | FPR | reasons |",
               "|---|---|---|---|---|---|---|---|---|---|---|"]
        for kind, t in deep["by_probe_type"].items():
            md.append(f"| {kind} | {t['n']} | {t['should_fire']} | {t['fired']} | {t['correct']} | "
                      f"{t['false_positive']} | {t['false_negative']} | {t['misroute']} | "
                      f"{_fmt(t['fnr'])} | {_fmt(t['fpr'])} | "
                      f"{', '.join(f'{k} {v}' for k, v in t['reasons'].items())} |")
        md += ["", "How the target person is named in the question (string match of the "
               "router's name surfaces):", "",
               "| set / naming | n | fired | correct | FP | FN | misroute |", "|---|---|---|---|---|---|---|"]
        for tag, t in deep["by_name_mention"].items():
            md.append(f"| {tag} | {t['n']} | {t['fired']} | {t['correct']} | {t['false_positive']} | "
                      f"{t['false_negative']} | {t['misroute']} |")
        md.append("")

    null = load_section(cache_dir, "null")
    if null:
        null_rows = probe_outcomes(null["probes"], null["routes"], facts)
        # The router treats a relation in the fact's answer group (mother tongue / languages
        # spoken, citizenship / country of origin, manufacturer / developer, ...) as the same
        # fact: it never trains on those as negatives (linear_router.same_answer_group), since
        # they leak the same answer. They are reported as their own group, outside the
        # deployment false-positive rate (deployment_fpr_all keeps them in).
        for r in null_rows:
            pr = null["probes"][r["index"]]
            if pr["group"] == "null_protected_other_relation" and (pr.get("meta") or {}).get("same_answer_group"):
                r["group"] = SAME_ANSWER_GROUP
        table = {}
        for group, items in sorted(_group(null_rows, "group").items()):
            fired = sum(r["fired"] for r in items)
            ungated = sum(r["score"] >= 0.5 for r in items if not r["should_fire"])
            table[group] = {"n": len(items), "should_fire": sum(r["should_fire"] for r in items),
                            "fired": fired, "fire_rate": wilson(fired, len(items)),
                            "ungated_max_p_ge_0.5": ungated}
        everything = [r for r in null_rows if not r["should_fire"]]
        negatives = [r for r in everything if r["group"] != SAME_ANSWER_GROUP]
        metrics["null"] = {"by_group": table, "deployment_fpr": wilson(
            sum(r["fired"] for r in negatives), len(negatives)),
            "deployment_fpr_all": wilson(sum(r["fired"] for r in everything), len(everything))}
        gens = manifest.get("null_generations")
        if gens:
            metrics["null"]["generations"] = {k: v for k, v in gens.items() if k != "rows"}
        plot_scores(null_rows, out_dir / "scores_null.png", f"{title}: prompts with no association", rule)
        md += ["## 4. Prompts with no association", "", "![null](scores_null.png)", "",
               "| group | n | should fire | fired | fire rate [95% CI] | ungated heads p >= 0.5 |",
               "|---|---|---|---|---|---|"]
        for group, t in table.items():
            ci = t["fire_rate"]
            md.append(f"| {group} | {t['n']} | {t['should_fire']} | {t['fired']} | {_fmt(ci['rate'])} "
                      f"[{_fmt(ci['low'])}, {_fmt(ci['high'])}] | {t['ungated_max_p_ge_0.5']} |")
        d, da = metrics["null"]["deployment_fpr"], metrics["null"]["deployment_fpr_all"]
        md += ["", f"**Deployment false-positive rate** (fires on a must-not-fire null prompt; "
               f"`{SAME_ANSWER_GROUP}` left out, see below): "
               f"{_fmt(d['rate'])} [{_fmt(d['low'])}, {_fmt(d['high'])}], {d['k']}/{d['n']}. "
               f"With it: {_fmt(da['rate'])}, {da['k']}/{da['n']}.", "",
               f"`{SAME_ANSWER_GROUP}`: the protected subject asked with a relation in the fact's answer "
               "group (e.g. languages spoken for a mother-tongue fact). The router treats these as the "
               "same fact and never trains on them as negatives, since they leak the same answer.", ""]
        if gens:
            md.append(f"Generations on the {gens['fired_null_prompts']} fired null prompts "
                      f"(ours `{gens.get('ours_dir')}`): output changed on {gens.get('changed')}, "
                      f"'I don't know'-like on {gens.get('idk_like')}.")
            md.append("")
            shown = (gens.get("rows") or [])[:top_k]
            if shown:
                md += [f"What the model answers when the router fires by mistake (first {len(shown)}, greedy):", "",
                       "| prompt | fired fact | base model | ours | changed |", "|---|---|---|---|---|"]
                fired_fact = {r["index"]: r["fired_fact"] for r in null_rows}
                for g in shown:
                    f = fired_fact.get(g["index"])
                    name = f"{f}: {str(facts[f].get('subject', ''))[:24]}" if f is not None else "–"
                    cell = lambda t: " ".join(str(t).split()).replace("|", "/")[:80]
                    md.append(f"| {cell(g['prompt'])} | {name} | {cell(g['base'])} | {cell(g['ours'])} | "
                              f"{'yes' if g['changed'] else 'no'} |")
                md.append("")
        md.append(worst_tables(null["probes"], null_rows, top_k))

    pert = load_section(cache_dir, "perturbed")
    if pert:
        pert_rows = probe_outcomes(pert["probes"], pert["routes"], facts)
        fams = {fam: binary_metrics(items) for fam, items in sorted(_group(pert_rows, "probe_type").items())}
        metrics["perturbed"] = fams
        md += ["## 5. Perturbed forget / paraphrase prompts (recall under surface noise)", "",
               "| family | n | recall | correct row | FN reasons |", "|---|---|---|---|---|"]
        for fam, m in fams.items():
            reasons = Counter(r["reason"] for r in pert_rows if r["probe_type"] == fam and r["reason"])
            md.append(f"| {fam} | {m['n']} | {_fmt(m['recall'])} | {_fmt(m['correct_row_rate'])} | "
                      f"{', '.join(f'{k} {v}' for k, v in reasons.items())} |")
        md.append("")

    sep = separability(ev, layers=(16, layer), seed=seed, negatives=umap_negatives)
    sep["wrong"] = [rows[i]["category"] != "correct" for i in sep["index"]]
    plot_umap(sep, out_dir / "umap_L16_L19.png", title)
    metrics["separability"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "coords"}
                               for k, v in sep["layers"].items()}
    md += ["## 6. Positives vs negatives at L16 and L19", "", "![umap](umap_L16_L19.png)", "",
           "| layer | silhouette (cosine, full dim) | silhouette (2D) | LOO logistic | LOO 1-NN |",
           "|---|---|---|---|---|"]
    for k, v in sep["layers"].items():
        md.append(f"| L{k} | {_fmt(v['silhouette_cosine_full_dim'])} | {_fmt(v['silhouette_2d'])} | "
                  f"{_fmt(v['loo']['logistic'])} | {_fmt(v['loo']['nearest_neighbour'])} |")
    md += ["", f"{len(sep['labels'])} points: each LOO error moves accuracy by {1 / len(sep['labels']):.3f}. Negatives "
           f"({umap_negatives}) prefer must-not-fire prompts that contain a protected subject.", ""]

    cal = calibration_report(ev["probes"], ev["routes"])
    plot_reliability(cal["pair_level"]["bins"], out_dir / "reliability.png", f"{title}: pair level")
    metrics["calibration"] = cal
    md += ["## 7. Calibration", "", "![reliability](reliability.png)", "",
           f"ECE over {cal['pair_level']['pairs']} subject-eligible (prompt, head) pairs: "
           f"{_fmt(cal['pair_level']['ece'])}. {cal['note']}.", ""]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")
    return metrics


def _norm_name(text):
    return " ".join(str(text).casefold().split())


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[r[key]].append(r)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", required=True, nargs="+")
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--umap-negatives", choices=("hard", "random"), default="hard")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    for cache in args.cache_dir:
        m = build_report(cache, top_k=args.top_k, umap_negatives=args.umap_negatives, seed=args.seed)
        a = m["eval"]["by_split"]["all"]
        print(json.dumps({"cache": cache, "fpr": a["fpr"], "fnr": a["fnr"], "f1": a["f1"],
                          "deployment_null_fpr": (m.get("null") or {}).get("deployment_fpr", {}).get("rate")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
