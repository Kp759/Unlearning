#!/usr/bin/env python3
"""UMAP of the router's own training prompts: each fact's 20 positives and 20 negatives.

router_diag_report.py's UMAP (section 6) shows 10 should-fire / 10 must-not-fire EVALUATION
prompts pooled over facts. This one shows what each head is actually fit on: the prompts of
linear_router_dataset.json (fit, calibration and audit splits), cached by
cache_router_states.py as the `train` section at every cached layer. CPU only.

    python -u scripts/router_train_umap.py --cache-dir outputs/router_diag_ours_control_v1/mcf/seed1
    # -> <cache>/train_umap/{summary.md, separability.json, umap_facts.png, umap_pooled.png,
    #                        separability_vs_layer.png}

Per fact i: positives are the prompts it owns (true_facts == [i]); negatives are the prompts
with i in negative_for (subject transplants, model-written hard negatives, same-subject
competitors). Colour is the class (positive blue, negative red); the marker is where the prompt
came from (template / context-prefix copy / base-model rewrite; transplant / hard negative /
same-subject). A black ring marks a prompt the deployed head itself gets wrong at the router
layer (positive with logit_i < threshold, or negative with logit_i >= threshold).

Numbers come from the full-dimensional unit-normalised states, not the 2-D picture:
    silhouette   cosine, full dimension, positives vs negatives of one fact
    LOO 1-NN     leave-one-out nearest neighbour (cosine), every cached layer
    LOO probe    leave-one-out L2 logistic probe, at the --layers only (it is the slow one)
    head errors  the deployed router's own decisions on these prompts (router layer only)
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router_diag_report import _fmt, _plt, display_text, embed_2d, loo_probe_accuracy, silhouette  # noqa: E402

POS_COLOR, NEG_COLOR = "#1565c0", "#c62828"
# (label, marker) per source; positives by router family, negatives by record kind
SOURCES = {
    "template": ("template prompt", "o"),
    "context_prefix": ("template + lead-in", "s"),
    "rewrite": ("base-model rewrite", "^"),
    "subject_transplant": ("subject transplant", "X"),
    "extra_negative": ("model-written hard negative", "v"),
    "same_subject_real": ("same-subject fact", "D"),
    "other": ("other", "P"),
}


def source_of(probe):
    if probe["should_fire"]:
        group = probe["group"]
        if group.startswith("aug_"):
            return "rewrite"
        return "context_prefix" if group.startswith("context_prefix") else "template"
    kind = probe["probe_type"]
    return kind if kind in SOURCES else "other"


def fact_sets(probes, n_facts):
    """{fact: (positive indices, negative indices)} over the training prompts."""
    pos = {i: [] for i in range(n_facts)}
    neg = {i: [] for i in range(n_facts)}
    for k, p in enumerate(probes):
        if p["should_fire"]:
            for i in p["true_facts"]:
                pos[i].append(k)
        for i in p["meta"].get("negative_for", []):
            neg[int(i)].append(k)
    return {i: (pos[i], neg[i]) for i in range(n_facts) if pos[i] and neg[i]}


def nn_accuracy(x, labels):
    """Leave-one-out 1-NN accuracy (cosine) on unit vectors, from one similarity matrix."""
    sims = x @ x.T
    sims.fill_diagonal_(-float("inf"))
    y = torch.as_tensor(labels)
    return float((y[sims.argmax(dim=1)] == y).double().mean())


def runtime_threshold(manifest):
    """The deployed cutoff on the cached logits: the artifact's per-head thresholds when it has
    them (cache_router_states.runtime_rule), else the manifest's global threshold."""
    path = Path(str(manifest.get("router_dir") or "")) / "fact_association_embeddings.pt"
    if path.is_file():
        per_head = torch.load(path, map_location="cpu", weights_only=False).get("per_head_thresholds")
        if per_head is not None:
            return torch.as_tensor(per_head).float()
    return float(manifest.get("threshold") or 0.0)


def head_errors(logits, threshold, fact, pos, neg):
    """The deployed head of `fact` on its own training prompts (no subject gate: every one of
    these prompts names the subject)."""
    z = logits[:, fact].float()
    thr = float(threshold[fact]) if torch.is_tensor(threshold) and threshold.dim() else float(threshold)
    fn = [k for k in pos if float(z[k]) < thr]
    fp = [k for k in neg if float(z[k]) >= thr]
    return fn, fp, thr


def analyse(cache_dir, layers=None, n_show=8, seed=0, loo_probe=True):
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / "manifest.json").read_text())
    probes = json.loads((cache_dir / "probes_train.json").read_text())
    states = torch.load(cache_dir / "states_train.pt", weights_only=False)
    routes = torch.load(cache_dir / "router_train.pt", weights_only=False)
    cached = [int(l) for l in states["layers"]]
    router_layer = int(manifest["router_layer"])
    n_layers = int(manifest["n_layers"])
    wanted = layers or [16, router_layer, n_layers - 2]
    shown = [l for l in dict.fromkeys(wanted) if l in cached]
    if not shown:
        raise SystemExit(f"none of layers {wanted} cached in {cache_dir} (cached: {cached})")
    facts = manifest["facts"]
    sets = fact_sets(probes, len(facts))
    threshold = runtime_threshold(manifest)

    def unit(layer, index):
        return F.normalize(states["states"][cached.index(layer)][index].double(), dim=-1)

    per_fact = []
    for i, (pos, neg) in sets.items():
        index, labels = pos + neg, [1] * len(pos) + [0] * len(neg)
        row = {"fact": i, "id": facts[i]["id"], "subject": facts[i].get("subject"),
               "relation": facts[i].get("relation"), "n_pos": len(pos), "n_neg": len(neg),
               "sources": dict(Counter(source_of(probes[k]) for k in index)),
               "nn_by_layer": {}, "silhouette_by_layer": {}, "probe_by_layer": {}}
        for layer in cached:
            x = unit(layer, index)
            row["nn_by_layer"][layer] = nn_accuracy(x, labels)
            row["silhouette_by_layer"][layer] = silhouette(x, labels, "cosine")
            if loo_probe and layer in shown:
                row["probe_by_layer"][layer] = loo_probe_accuracy(x, labels)["logistic"]
        fn, fp, thr = head_errors(routes["logits"], threshold, i, pos, neg)
        row.update({"head_fn": fn, "head_fp": fp, "threshold": thr,
                    "head_errors_by_split": dict(Counter(probes[k]["split"] for k in fn + fp))})
        per_fact.append(row)

    by_layer = {}
    for layer in cached:
        sil = [r["silhouette_by_layer"][layer] for r in per_fact if r["silhouette_by_layer"][layer] is not None]
        nn = [r["nn_by_layer"][layer] for r in per_fact]
        probe = [r["probe_by_layer"][layer] for r in per_fact if layer in r["probe_by_layer"]]
        by_layer[layer] = {
            "silhouette_mean": sum(sil) / len(sil) if sil else None,
            "silhouette_min": min(sil) if sil else None,
            "nn_mean": sum(nn) / len(nn) if nn else None,
            "nn_min": min(nn) if nn else None,
            "facts_nn_perfect": sum(v >= 1.0 - 1e-12 for v in nn),
            "probe_mean": sum(probe) / len(probe) if probe else None,
            "probe_min": min(probe) if probe else None,
        }

    # facts to draw: the hardest at the router layer (most head errors, then lowest 1-NN), then random
    rng = random.Random(seed)
    hardest = sorted(per_fact, key=lambda r: (-(len(r["head_fn"]) + len(r["head_fp"])),
                                              r["nn_by_layer"][router_layer] if router_layer in cached else 0,
                                              r["fact"]))
    chosen = [r["fact"] for r in hardest[: n_show // 2]]
    rest = [r["fact"] for r in per_fact if r["fact"] not in chosen]
    chosen += sorted(rng.sample(rest, min(n_show - len(chosen), len(rest))))
    return {"cache_dir": str(cache_dir), "dataset": manifest["dataset"], "seed": manifest["seed"],
            "router_dir": manifest.get("router_dir"), "router_sha256": manifest.get("router_sha256"),
            "router_layer": router_layer, "n_layers": n_layers, "layers_shown": shown,
            "cached_layers": cached, "threshold": threshold if not torch.is_tensor(threshold) else "per head",
            "n_facts": len(facts), "n_facts_with_both": len(per_fact), "per_fact": per_fact,
            "by_layer": by_layer, "drawn_facts": chosen}, probes, states


def _scatter(ax, coords, index, probes, wrong):
    for local, k in enumerate(index):
        p = probes[k]
        color = POS_COLOR if p["should_fire"] else NEG_COLOR
        marker = SOURCES[source_of(p)][1]
        ax.scatter(coords[local, 0], coords[local, 1], color=color, marker=marker, s=34, alpha=0.85,
                   linewidths=0)
        if k in wrong:
            ax.scatter(coords[local, 0], coords[local, 1], s=150, facecolors="none", edgecolors="black",
                       linewidths=1.2)


def _legend(fig):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], color=POS_COLOR, marker="o", ls="", label="positive (should fire)"),
               Line2D([], [], color=NEG_COLOR, marker="o", ls="", label="negative (must not fire)")]
    handles += [Line2D([], [], color="grey", marker=m, ls="", label=label) for label, m in SOURCES.values()
                if label != "other"]
    handles.append(Line2D([], [], color="white", marker="o", markeredgecolor="black", ls="",
                          label="deployed head wrong"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, frameon=False)


def plot_facts(result, probes, states, path, seed=0):
    plt = _plt()
    cached, layers = result["cached_layers"], result["layers_shown"]
    rows = [r for r in result["per_fact"] if r["fact"] in result["drawn_facts"]]
    fig, axes = plt.subplots(len(rows), len(layers), figsize=(3.6 * len(layers), 3.1 * len(rows) + 0.8),
                             squeeze=False)
    sets = fact_sets(probes, result["n_facts"])
    for row_axes, r in zip(axes, rows):
        pos, neg = sets[r["fact"]]
        index = pos + neg
        wrong = set(r["head_fn"]) | set(r["head_fp"])
        for ax, layer in zip(row_axes, layers):
            x = F.normalize(states["states"][cached.index(layer)][index].double(), dim=-1)
            coords, method = embed_2d(x.float(), n_neighbors=10, seed=seed)
            _scatter(ax, coords, index, probes, wrong)
            probe = r["probe_by_layer"].get(layer)
            ax.set_title(f"{str(r['subject'])[:28]} · {r['relation']} · L{layer}\n"
                         f"+{r['n_pos']} / −{r['n_neg']}  sil {_fmt(r['silhouette_by_layer'][layer], 2)}  "
                         f"1-NN {_fmt(r['nn_by_layer'][layer], 2)}  probe {_fmt(probe, 2)}", fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(f"{result['dataset']} seed {result['seed']}: each head's own training prompts "
                 f"({method}; ring = head L{result['router_layer']} wrong)", fontsize=10)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_pooled(result, probes, states, path, seed=0):
    """Every fact's training prompts in one projection (a prompt that is a negative for one fact
    and a positive for another is drawn once, as a positive)."""
    plt = _plt()
    cached, layers = result["cached_layers"], result["layers_shown"]
    index = [k for k, p in enumerate(probes) if p["should_fire"] or p["meta"].get("negative_for")]
    wrong = {k for r in result["per_fact"] for k in r["head_fn"] + r["head_fp"]}
    fig, axes = plt.subplots(1, len(layers), figsize=(5.0 * len(layers), 5.0), squeeze=False)
    for ax, layer in zip(axes[0], layers):
        x = F.normalize(states["states"][cached.index(layer)][index].double(), dim=-1)
        coords, method = embed_2d(x.float(), n_neighbors=15, seed=seed)
        _scatter(ax, coords, index, probes, wrong)
        b = result["by_layer"][layer]
        ax.set_title(f"L{layer} ({method}), {len(index)} prompts\nper-fact mean: sil "
                     f"{_fmt(b['silhouette_mean'], 2)}, 1-NN {_fmt(b['nn_mean'], 2)}, probe {_fmt(b['probe_mean'], 2)}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{result['dataset']} seed {result['seed']}: all training prompts, pooled over facts", fontsize=10)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_vs_layer(result, path):
    plt = _plt()
    layers = result["cached_layers"]
    b = result["by_layer"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for key, label, style in (("nn_mean", "LOO 1-NN, mean over facts", "-o"),
                              ("nn_min", "LOO 1-NN, worst fact", ":o"),
                              ("silhouette_mean", "silhouette (cosine), mean", "-s")):
        ys = [b[l][key] for l in layers]
        ax.plot(layers, [float("nan") if y is None else y for y in ys], style, ms=3, label=label)
    probe_layers = [l for l in layers if b[l]["probe_mean"] is not None]
    ax.plot(probe_layers, [b[l]["probe_mean"] for l in probe_layers], "k*", ms=9, label="LOO logistic probe, mean")
    ax.axvline(result["router_layer"], ls="--", color="grey", lw=1)
    ax.axvline(result["n_layers"] - 2, ls=":", color="grey", lw=1)
    ax.set_xlabel("block")
    ax.set_ylim(min(-0.05, *[b[l]["silhouette_mean"] or 0 for l in layers]) - 0.02, 1.02)
    ax.set_title(f"{result['dataset']} seed {result['seed']}: positives vs negatives of each fact, "
                 f"training prompts (dashed L{result['router_layer']}, dotted penultimate)", fontsize=8)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary(result, probes, out_dir, top_k=15):
    L, layers, b = result["router_layer"], result["layers_shown"], result["by_layer"]
    rows = result["per_fact"]
    n_pos, n_neg = sum(r["n_pos"] for r in rows), sum(r["n_neg"] for r in rows)
    n_fn, n_fp = sum(len(r["head_fn"]) for r in rows), sum(len(r["head_fp"]) for r in rows)
    sources = Counter()
    for r in rows:
        sources.update(r["sources"])
    md = [f"# Router training prompts, positives vs negatives ({result['dataset']} seed {result['seed']})", "",
          f"Router `{result['router_dir']}` (sha256 `{str(result['router_sha256'])[:12]}`), block {L}. "
          f"{result['n_facts_with_both']} of {result['n_facts']} facts have both positives and negatives; "
          f"{n_pos} positive and {n_neg} negative (fact, prompt) pairs, "
          f"{n_pos / max(1, len(rows)):.1f} / {n_neg / max(1, len(rows)):.1f} per fact.", "",
          "Sources: " + ", ".join(f"{SOURCES[s][0]} {c}" for s, c in sources.most_common()), "",
          "## The deployed head on its own training prompts", "",
          f"{n_fn} of {n_pos} positives below the threshold, {n_fp} of {n_neg} negatives at or above it "
          f"(splits: {dict(sum((Counter(r['head_errors_by_split']) for r in rows), Counter()))}). "
          "Errors on the `fit` split are training errors; `calibration` and `audit` prompts are held "
          "out of the weight fit (the threshold is placed on calibration for 0.98 recall).", "",
          "## Separability by layer (full dimension, per fact, then over facts)", "",
          "![vs layer](separability_vs_layer.png)", "",
          "| block | silhouette mean | silhouette min | LOO 1-NN mean | LOO 1-NN min | facts at 1-NN 1.0 | LOO probe mean | LOO probe min |",
          "|---|---|---|---|---|---|---|---|"]
    for layer in result["cached_layers"]:
        v = b[layer]
        tag = " (router)" if layer == L else " (penultimate)" if layer == result["n_layers"] - 2 else ""
        md.append(f"| {layer}{tag} | {_fmt(v['silhouette_mean'])} | {_fmt(v['silhouette_min'])} | "
                  f"{_fmt(v['nn_mean'])} | {_fmt(v['nn_min'])} | {v['facts_nn_perfect']}/{len(rows)} | "
                  f"{_fmt(v['probe_mean'])} | {_fmt(v['probe_min'])} |")
    md += ["", "## Each fact's positives and negatives (UMAP)", "",
           f"The {len(result['drawn_facts'])} facts drawn: the hardest at block {L} (most head errors, "
           "then lowest 1-NN), then a seeded random sample.", "", "![facts](umap_facts.png)", "",
           "## All training prompts pooled", "", "![pooled](umap_pooled.png)", "",
           "Pooled over facts, prompts cluster by subject more than by class; a head only has to "
           "separate its own fact's positives from its own negatives, which is what the per-fact numbers measure.", "",
           f"## Per fact at block {L}", "",
           "| fact | subject | relation | + | − | silhouette | 1-NN | " +
           " | ".join(f"probe L{l}" for l in layers) + " | head FN | head FP |",
           "|---|---|---|---|---|---|---|" + "---|" * len(layers) + "---|---|"]
    for r in sorted(rows, key=lambda r: (r["nn_by_layer"].get(L, 1), r["fact"])):
        md.append(f"| {r['fact']} | {str(r['subject'])[:30]} | {r['relation']} | {r['n_pos']} | {r['n_neg']} | "
                  f"{_fmt(r['silhouette_by_layer'].get(L))} | {_fmt(r['nn_by_layer'].get(L))} | " +
                  " | ".join(_fmt(r["probe_by_layer"].get(l)) for l in layers) +
                  f" | {len(r['head_fn'])} | {len(r['head_fp'])} |")
    errs = [(r, k, "FN") for r in rows for k in r["head_fn"]] + [(r, k, "FP") for r in rows for k in r["head_fp"]]
    if errs:
        md += ["", f"## Training prompts the deployed head gets wrong (first {top_k})", "",
               "| fact | error | split | source | prompt |", "|---|---|---|---|---|"]
        for r, k, kind in errs[:top_k]:
            p = probes[k]
            md.append(f"| {r['fact']} | {kind} | {p['split']} | {SOURCES[source_of(p)][0]} | "
                      f"{display_text(p, 140).replace('|', '/')} |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-dir", required=True, help="a cache_router_states.py output dir")
    p.add_argument("--out-dir", default=None, help="default: <cache-dir>/train_umap")
    p.add_argument("--layers", default="", help="layers to draw and probe (default: 16, router layer, penultimate)")
    p.add_argument("--facts-shown", type=int, default=8)
    p.add_argument("--no-loo-probe", action="store_true", help="skip the leave-one-out logistic probe")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    out = Path(args.out_dir) if args.out_dir else Path(args.cache_dir) / "train_umap"
    out.mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()] or None
    result, probes, states = analyse(args.cache_dir, layers, args.facts_shown, args.seed, not args.no_loo_probe)
    plot_facts(result, probes, states, out / "umap_facts.png", args.seed)
    plot_pooled(result, probes, states, out / "umap_pooled.png", args.seed)
    plot_vs_layer(result, out / "separability_vs_layer.png")
    write_summary(result, probes, out)
    (out / "separability.json").write_text(json.dumps(result, indent=1, default=str) + "\n")
    L = result["router_layer"]
    print(json.dumps({"status": "train_umap_ready", "out": str(out), "facts": result["n_facts_with_both"],
                      f"nn_mean_L{L}": result["by_layer"][L]["nn_mean"] if L in result["by_layer"] else None,
                      "head_fn": sum(len(r["head_fn"]) for r in result["per_fact"]),
                      "head_fp": sum(len(r["head_fp"]) for r in result["per_fact"])}), flush=True)


if __name__ == "__main__":
    main()
