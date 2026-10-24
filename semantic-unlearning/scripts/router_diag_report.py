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


def pick_umap_points(probes, routes, per_class=10, seed=0, negatives="hard"):
    """10 should-fire (forget / paraphrase) and 10 must-not-fire eval probes. 'hard'
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


def separability(eval_section, layers=(16, 19), per_class=10, seed=0, negatives="hard"):
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


def plot_scores(rows, path, title, threshold_note):
    plt = _plt()
    splits = [s for s in SPLIT_ORDER if any(r["split"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.8), squeeze=False)
    for ax, split in zip(axes[0], splits):
        items = sorted((r for r in rows if r["split"] == split), key=lambda r: r["score"])
        for category in CATEGORIES:
            for eligible in (True, False):
                pts = [(k, r["score"]) for k, r in enumerate(items)
                       if r["category"] == category and r["relevant_eligible"] == eligible]
                if not pts:
                    continue
                xs, ys = zip(*pts)
                ax.scatter(xs, ys, s=10 if len(items) > 400 else 18, linewidths=0.8,
                           facecolors=COLORS[category] if eligible else "none",
                           edgecolors=COLORS[category],
                           label=f"{category}{'' if eligible else ' (no eligible head)'}")
        ax.axhline(0.5, color="black", lw=0.8, ls="--")
        counts = Counter(r["category"] for r in items)
        ax.set_title(f"{split} (n={len(items)})\n" + ", ".join(
            f"{SHORT[c]} {counts[c]}" for c in CATEGORIES if counts[c]), fontsize=8)
        ax.set_xlabel("examples sorted by score")
        ax.set_ylim(-0.03, 1.03)
    axes[0][0].set_ylabel("p(true association)  |  max p (must-not-fire)")
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = {}
    for ax in axes[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="lower center", ncol=4, fontsize=7,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{title} — {threshold_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_binary_confusions(per_split, path, title):
    plt = _plt()
    names = [s for s in SPLIT_ORDER if s in per_split] + ["all"]
    fig, axes = plt.subplots(1, len(names), figsize=(2.6 * len(names), 2.8), squeeze=False)
    for ax, name in zip(axes[0], names):
        m = per_split[name]
        grid = [[m["tp"], m["fn"]], [m["fp"], m["tn"]]]
        ax.imshow([[math.log1p(v) for v in row] for row in grid], cmap="Blues")
        for a in range(2):
            for b in range(2):
                ax.text(b, a, str(grid[a][b]), ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], ["fired", "not fired"], fontsize=7)
        ax.set_yticks([0, 1], ["should fire", "should not"], fontsize=7)
        fpr = "–" if m["fpr"] is None else f"{m['fpr']:.3f}"
        fnr = "–" if m["fnr"] is None else f"{m['fnr']:.3f}"
        ax.set_title(f"{name}\nFPR {fpr}  FNR {fnr}", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nway(matrix, labels, path, title):
    plt = _plt()
    k = matrix.shape[0]
    fig, ax = plt.subplots(figsize=(max(4, 0.16 * k + 2), max(4, 0.16 * k + 2)))
    ax.imshow(torch.log1p(matrix.float()).numpy(), cmap="viridis")
    ticks = list(range(k))
    ax.set_xticks(ticks, labels, rotation=90, fontsize=max(3, 8 - k // 12))
    ax.set_yticks(ticks, labels, fontsize=max(3, 8 - k // 12))
    ax.set_xlabel("fired row (last = none)")
    ax.set_ylabel("true row (last = must not fire)")
    if k <= 12:
        for a in range(k):
            for b in range(k):
                if matrix[a, b]:
                    ax.text(b, a, str(int(matrix[a, b])), ha="center", va="center", fontsize=7,
                            color="white")
    ax.set_title(title, fontsize=9)
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


def plot_umap(sep, path, title):
    plt = _plt()
    layers = sorted(sep["layers"])
    fig, axes = plt.subplots(1, len(layers), figsize=(4.4 * len(layers), 4.2), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        info = sep["layers"][layer]
        coords = torch.tensor(info["coords"])
        for label, color, name in ((1, "#1565c0", "should fire"), (0, "#c62828", "should not fire")):
            idx = [k for k, l in enumerate(sep["labels"]) if l == label]
            ax.scatter(coords[idx, 0], coords[idx, 1], color=color, s=36, label=name)
        s_full = info["silhouette_cosine_full_dim"]
        ax.set_title(f"L{layer} ({info['method']})\nsilhouette {s_full:.2f} (cosine, full dim), "
                     f"{info['silhouette_2d']:.2f} (2D)\nLOO probe {info['loo']['logistic']:.2f}, "
                     f"1-NN {info['loo']['nearest_neighbour']:.2f}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"{title} — negatives: {sep['negatives_mode']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    title = f"{dataset} seed {manifest['seed']} — L{layer} router"
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
    plot_nway(nway, [str(i) for i in range(len(facts))] + ["none"], out_dir / "confusion_nway.png",
              f"{title}: fired row vs true row")
    metrics["eval"]["nway"] = {"matrix": nway.tolist(),
                               "should_fire_routed_to_a_true_row": per_split["all"]["correct_row_rate"]}
    md = [f"# Router diagnostics: {title}", "",
          f"Router: `{manifest['router_dir']}` · gate `{gate}` · {rule} · subject match "
          f"`{manifest.get('subject_match', 'exact')}`.",
          f"Probes: eval {len(ev['probes'])}. Hook parity (real bank hook vs cached decision): "
          f"{json.dumps({k: v['route_mismatches'] for k, v in manifest.get('hook_parity', {}).items()})} "
          "mismatches.", "",
          "## 1. Scores against the decision line", "",
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
        table = {}
        for group, items in sorted(_group(null_rows, "group").items()):
            fired = sum(r["fired"] for r in items)
            ungated = sum(r["score"] >= 0.5 for r in items if not r["should_fire"])
            table[group] = {"n": len(items), "should_fire": sum(r["should_fire"] for r in items),
                            "fired": fired, "fire_rate": wilson(fired, len(items)),
                            "ungated_max_p_ge_0.5": ungated}
        negatives = [r for r in null_rows if not r["should_fire"]]
        metrics["null"] = {"by_group": table, "deployment_fpr": wilson(
            sum(r["fired"] for r in negatives), len(negatives))}
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
        d = metrics["null"]["deployment_fpr"]
        md += ["", f"**Deployment false-positive rate** (fires on a must-not-fire null prompt): "
               f"{_fmt(d['rate'])} [{_fmt(d['low'])}, {_fmt(d['high'])}], {d['k']}/{d['n']}.", ""]
        if gens:
            md.append(f"Generations on the {gens['fired_null_prompts']} fired null prompts "
                      f"(ours `{gens.get('ours_dir')}`): output changed on {gens.get('changed')}, "
                      f"'I don't know'-like on {gens.get('idk_like')}.")
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
    plot_umap(sep, out_dir / "umap_L16_L19.png", title)
    metrics["separability"] = {str(k): {kk: vv for kk, vv in v.items() if kk != "coords"}
                               for k, v in sep["layers"].items()}
    md += ["## 6. Positives vs negatives at L16 and L19", "", "![umap](umap_L16_L19.png)", "",
           "| layer | silhouette (cosine, full dim) | silhouette (2D) | LOO logistic | LOO 1-NN |",
           "|---|---|---|---|---|"]
    for k, v in sep["layers"].items():
        md.append(f"| L{k} | {_fmt(v['silhouette_cosine_full_dim'])} | {_fmt(v['silhouette_2d'])} | "
                  f"{_fmt(v['loo']['logistic'])} | {_fmt(v['loo']['nearest_neighbour'])} |")
    md += ["", "20 points: each LOO error moves accuracy by 0.05. Negatives "
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
