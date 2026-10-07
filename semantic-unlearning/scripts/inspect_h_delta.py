#!/usr/bin/env python3
"""Inspect the trained residual rows De_i and what they do to h_L.

Answers three questions for one trained artifact (e.g. MCF layer sweep, L19):

1. ROWS - what do the 50 vectors look like, and are they the same?
   (no model needed: --rows-only runs just this, in seconds on a login node)
     * per row: norm, mean, std, max|.|, largest coordinates, energy in top 1% dims
     * SAME VALUE?     pairwise ||De_i - De_j|| relative to the row norms
     * SAME DIRECTION? pairwise cos(De_i, De_j), cos(De_i, mean row), SVD energy
     * plots: rows_heatmap.png, rows_cosine.png, rows_norms.png, rows_spectrum.png
2. INTERCHANGE - can one vector replace all of them?  For every forget view the
   edit is added at the training boundary (last prompt token, oracle routing) with
     own row | zero (base) | mean row | mean row at own norm |
     another fact's row (fixed derangement) | random direction at own norm
   and scored with the training metric: worst-view exp(-mean answer NLL) < 1e-6.
3. PER PROMPT h -> h' (runtime model, linear-classifier routing): norms, cosine,
   angle, the coordinates De moves most, logit lens, P(true), P(' I').

    cd semantic-unlearning
    python -u scripts/inspect_h_delta.py \
        --run-dir outputs/mcf_layer_sweep_linear_regular_v1/L19/linear_global \
        --out-dir outputs/inspect_h_delta/L19_regular --local-files-only

    # rows only (no GPU, no model):
    python -u scripts/inspect_h_delta.py --rows-only --run-dir <...> --out-dir <...>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

UNKNOWN = " I don't know."
DIVERGING = ["#104281", "#2a78d6", "#f0efec", "#e34948", "#8f1f1f"]  # blue <-> gray <-> red
BAR = "#2a78d6"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


# --------------------------------------------------------------------------- rows

def _label(fact, width=34):
    text = f"{fact.get('subject', '?')} | {fact.get('relation', '?')} -> {fact.get('object', '?')}"
    return text if len(text) <= width else text[: width - 1] + "…"


def row_geometry(rows, facts, top_k=8):
    """Everything about the bank that needs no model. rows: [n, d] float."""
    rows = rows.float()
    n, d = rows.shape
    norms = rows.norm(dim=-1)
    live = [i for i in range(n) if float(norms[i]) > 0]
    if len(live) < 2:
        raise ValueError(f"Need at least 2 nonzero rows, found {len(live)}")
    R = rows[live]
    m = len(live)
    off = ~torch.eye(m, dtype=torch.bool)

    U = F.normalize(R, dim=-1)
    C = U @ U.T                                             # direction similarity
    dist = torch.cdist(R[None], R[None])[0]                 # value difference
    scale = (norms[live][:, None] + norms[live][None, :]) / 2
    rel = dist / scale

    mean_row = R.mean(0)
    cos_mean = F.cosine_similarity(R, mean_row[None], dim=-1)
    s = torch.linalg.svdvals(R)                             # uncentered: shared direction shows as PC1
    energy = s.pow(2) / s.pow(2).sum()
    nz = energy[energy > 0]
    eff_rank = float(torch.exp(-(nz * nz.log()).sum()))

    k1 = max(1, d // 100)
    sq = R.pow(2)
    top1pct = sq.topk(k1, dim=-1).values.sum(-1) / sq.sum(-1)

    per_row = []
    for j, i in enumerate(live):
        r = R[j]
        top = r.abs().topk(top_k).indices.tolist()
        per_row.append({
            "row": i, "fact_id": facts[i].get("id"), "fact": _label(facts[i], 80),
            "norm": float(norms[i]), "mean": float(r.mean()), "std": float(r.std()),
            "max_abs": float(r.abs().max()), "cos_to_mean_row": float(cos_mean[j]),
            "energy_in_top1pct_dims": float(top1pct[j]),
            "nearest_other_row": live[int(C[j].masked_fill(~off[j], -2).argmax())],
            "nearest_other_cos": float(C[j].masked_fill(~off[j], -2).max()),
            "top_coords": [[k, float(r[k])] for k in top],
            "first_coords": [float(x) for x in r[:8]],
        })

    cos_off, rel_off = C[off], rel[off]
    mean_cos = float(cos_off.mean())
    median_rel = float(rel_off.median())
    if median_rel < 1e-3:
        value_verdict = "YES - identical (up to float noise)"
    elif median_rel < 0.1:
        value_verdict = "almost - differ by <10% of their norm"
    else:
        value_verdict = "NO - the numbers differ"
    if mean_cos >= 0.95:
        dir_verdict = "YES - essentially one direction (one vector up to scale)"
    elif mean_cos >= 0.6:
        dir_verdict = "MOSTLY - one large shared direction + small fact-specific parts"
    elif mean_cos >= 0.2:
        dir_verdict = "PARTLY - a shared component plus substantial fact-specific parts"
    else:
        dir_verdict = "NO - mostly distinct directions"
    summary = {
        "hidden_size": d, "rows_total": n, "rows_nonzero": m,
        "zero_rows": [i for i in range(n) if i not in live],
        "norm_min": float(norms[live].min()), "norm_median": float(norms[live].median()),
        "norm_max": float(norms[live].max()),
        "same_value_verdict": value_verdict,
        "pairwise_rel_distance_min": float(rel_off.min()),
        "pairwise_rel_distance_median": median_rel,
        "pairwise_rel_distance_max": float(rel_off.max()),
        "same_direction_verdict": dir_verdict,
        "pairwise_cos_mean": mean_cos, "pairwise_cos_median": float(cos_off.median()),
        "pairwise_cos_min": float(cos_off.min()), "pairwise_cos_max": float(cos_off.max()),
        "pairwise_cos_random_baseline": math.sqrt(2 / (math.pi * d)),
        "pairs_cos_above_0.9": int((cos_off > 0.9).sum()) // 2,
        "pairs_total": m * (m - 1) // 2,
        "cos_to_mean_row_median": float(cos_mean.median()),
        "mean_row_norm_over_mean_norm": float(mean_row.norm() / norms[live].mean()),
        "svd_energy_top5": [float(x) for x in energy[:5]],
        "effective_rank": eff_rank,
    }
    return {"summary": summary, "per_row": per_row, "live": live, "cos": C,
            "rows": R, "mean_row": mean_row, "energy": energy}


def write_row_tables(geo, facts, out):
    live = geo["live"]
    with open(out / "rows_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "fact_id", "fact", "norm", "mean", "std", "max_abs",
                    "cos_to_mean_row", "nearest_other_row", "nearest_other_cos",
                    "energy_in_top1pct_dims", "top_coords(dim:value)"])
        for r in geo["per_row"]:
            w.writerow([r["row"], r["fact_id"], r["fact"], f"{r['norm']:.5g}", f"{r['mean']:.5g}",
                        f"{r['std']:.5g}", f"{r['max_abs']:.5g}", f"{r['cos_to_mean_row']:.4f}",
                        r["nearest_other_row"], f"{r['nearest_other_cos']:.4f}",
                        f"{r['energy_in_top1pct_dims']:.4f}",
                        " ".join(f"{k}:{v:+.4g}" for k, v in r["top_coords"])])
    with open(out / "rows_cosine.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row"] + live)
        for j, i in enumerate(live):
            w.writerow([i] + [f"{float(x):.4f}" for x in geo["cos"][j]])
    torch.save({"rows": geo["rows"], "row_index": live, "mean_row": geo["mean_row"],
                "fact_ids": [facts[i].get("id") for i in live]}, out / "rows_vectors.pt")


def plot_rows(geo, facts, out, n_dims=128):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        print("matplotlib not installed: skipping plots")
        return []
    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.edgecolor": "#d9d8d4", "font.size": 9})
    R, live, C = geo["rows"], geo["live"], geo["cos"]
    order = torch.argsort(F.cosine_similarity(R, geo["mean_row"][None], dim=-1), descending=True)
    labels = [f"{live[j]:>2} {_label(facts[live[j]])}" for j in order.tolist()]
    show_labels = len(live) <= 60
    files = []

    # 1. the vectors themselves on the dims they use most
    dims = R.abs().mean(0).topk(min(n_dims, R.shape[1])).indices
    dims = dims[torch.argsort(geo["mean_row"][dims])]
    M = R[order][:, dims].numpy()
    lim = float(torch.quantile(R[:, dims].abs().flatten(), 0.99)) or 1.0
    fig, ax = plt.subplots(figsize=(12, max(4, 0.18 * len(live) + 1.5)))
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest")
    ax.set_title(f"Δe rows on the {len(dims)} hidden dims with largest mean |Δe|\n"
                 "rows sorted by cos to the mean row; dims sorted by the mean row's value. "
                 "Same stripes in every row = same vector", loc="left", fontsize=10)
    ax.set_xlabel("hidden dimension (index shown every 16)")
    ax.set_xticks(range(0, len(dims), 16), [str(int(dims[k])) for k in range(0, len(dims), 16)])
    ax.set_yticks(range(len(live)), labels if show_labels else [""] * len(live), fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, label="Δe value")
    fig.tight_layout(); fig.savefig(out / "rows_heatmap.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    files.append("rows_heatmap.png")

    # 2. direction similarity
    Co = C[order][:, order].numpy()
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(Co, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    s = geo["summary"]
    ax.set_title(f"cos(Δe_i, Δe_j): mean off-diagonal {s['pairwise_cos_mean']:.3f} "
                 f"(random {s['hidden_size']}-d vectors ≈ {s['pairwise_cos_random_baseline']:.3f})\n"
                 "all dark red = all rows point the same way", loc="left", fontsize=10)
    ticks = labels if show_labels else [""] * len(live)
    ax.set_xticks(range(len(live)), [t.split()[0] for t in ticks], fontsize=6)
    ax.set_yticks(range(len(live)), ticks, fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="cosine")
    fig.tight_layout(); fig.savefig(out / "rows_cosine.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    files.append("rows_cosine.png")

    # 3. norms and cos-to-mean (two panels, two separate axes)
    per = {r["row"]: r for r in geo["per_row"]}
    xs = [live[j] for j in order.tolist()]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    a1.bar(range(len(xs)), [per[i]["norm"] for i in xs], color=BAR, width=0.7)
    a1.set_ylabel("‖Δe_i‖"); a1.set_title("Row norm", loc="left", fontsize=10)
    a2.bar(range(len(xs)), [per[i]["cos_to_mean_row"] for i in xs], color=BAR, width=0.7)
    a2.axhline(0, color="#d9d8d4", lw=1)
    a2.set_ylabel("cos(Δe_i, mean row)")
    a2.set_title("How much of each row points along the shared (mean) direction", loc="left",
                 fontsize=10)
    a2.set_xticks(range(len(xs)), [str(i) for i in xs], fontsize=6)
    a2.set_xlabel("row (sorted by cos to mean row)")
    for a in (a1, a2):
        a.grid(axis="y", color="#ecebe7", lw=0.6); a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "rows_norms.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    files.append("rows_norms.png")

    # 4. SVD energy spectrum
    e = geo["energy"][:20].numpy()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(range(1, len(e) + 1), e, color=BAR, width=0.7)
    ax.text(1, e[0], f" {e[0]:.0%}", va="bottom", ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(1, len(e) + 1))
    ax.set_xlabel("singular component"); ax.set_ylabel("share of ‖R‖²")
    ax.set_title("Energy per SVD component of the row matrix (uncentered)\n"
                 "one tall bar = all rows are (nearly) one vector", loc="left", fontsize=10)
    ax.grid(axis="y", color="#ecebe7", lw=0.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(out / "rows_spectrum.png", dpi=150, bbox_inches="tight"); plt.close(fig)
    files.append("rows_spectrum.png")
    return files


# ------------------------------------------------------------------ model helpers

def _add_at(model, layer, pos, vec):
    """Forward hook: add vec to the layer output at one position (batch row 0)."""
    def hook(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[0, pos] = h[0, pos] + vec.to(device=h.device, dtype=h.dtype)
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return model.model.layers[layer].register_forward_hook(hook)


@torch.no_grad()
def answer_prob(model, layer, example, vec, device):
    """exp(-mean labeled-token NLL), vec added at the last prompt token (training boundary)."""
    labels = torch.tensor([example.labels], device=device)
    first = int((labels[0] != -100).nonzero()[0])
    handle = None if vec is None else _add_at(model, layer, first - 1, vec)
    try:
        logits = model(input_ids=torch.tensor([example.input_ids], device=device),
                       use_cache=False).logits.float()
    finally:
        if handle is not None:
            handle.remove()
    nll = F.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
    return math.exp(-float(nll))


def derangement(items, seed):
    g = torch.Generator().manual_seed(seed)
    while True:
        perm = [items[k] for k in torch.randperm(len(items), generator=g).tolist()]
        if all(a != b for a, b in zip(items, perm)):
            return dict(zip(items, perm))


def interchange(model, layer, examples, unknown_map, rows, live, row_of, target, device, seed=1):
    mean_row = rows[live].mean(0)
    other = derangement(live, seed)
    g = torch.Generator().manual_seed(seed)
    rand_dir = {i: F.normalize(torch.randn(rows.shape[1], generator=g), dim=0) for i in live}
    arms = {
        "own_row": lambda r: rows[r],
        "zero_base": lambda r: None,
        "mean_row": lambda r: mean_row,
        "mean_row_own_norm": lambda r: mean_row * (rows[r].norm() / mean_row.norm()),
        "other_fact_row": lambda r: rows[other[r]],
        "random_dir_own_norm": lambda r: rand_dir[r] * rows[r].norm(),
    }
    per_fact = {}
    for e in examples:
        r = row_of[e.fact_id]
        if r not in live:
            continue
        rec = per_fact.setdefault(r, {a: {"answer": [], "unknown": []} for a in arms})
        for name, pick in arms.items():
            vec = pick(r)
            rec[name]["answer"].append(answer_prob(model, layer, e, vec, device))
            rec[name]["unknown"].append(answer_prob(model, layer, unknown_map[e.id], vec, device))
    table = {}
    for name in arms:
        worst = [max(v[name]["answer"]) for v in per_fact.values()]
        unk = [sum(v[name]["unknown"]) / len(v[name]["unknown"]) for v in per_fact.values()]
        worst_t = torch.tensor(worst)
        table[name] = {
            "facts_forgotten": int((worst_t < target).sum()), "facts": len(worst),
            "median_worst_view_p_true": float(worst_t.median()),
            "mean_p_idk_completion": sum(unk) / len(unk),
        }
    detail = {int(r): {a: {"worst_p_true": max(v[a]["answer"]),
                           "mean_p_idk": sum(v[a]["unknown"]) / len(v[a]["unknown"])}
                       for a in arms} for r, v in per_fact.items()}
    return {"target_probability": target, "other_fact_row_map": {int(k): int(v) for k, v in other.items()},
            "arms": table, "per_fact": detail}


# ------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="dir with fact_association_embeddings.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rows-only", action="store_true", help="part 1 only: no model, no GPU")
    ap.add_argument("--skip-interchange", action="store_true")
    ap.add_argument("--skip-per-prompt", action="store_true")
    ap.add_argument("--split", default="all", choices=("all", "train", "development"))
    ap.add_argument("--coords", type=int, default=8, help="largest-|De| coordinates to report")
    ap.add_argument("--print-prompts", type=int, default=15, help="per-prompt lines printed")
    ap.add_argument("--prompt", action="append", default=[], help="extra (e.g. retain) prompts")
    ap.add_argument("--seed", type=int, default=1, help="seed for derangement / random arm")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-files-only", action="store_true")
    a = ap.parse_args(argv)

    run, out = Path(a.run_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    art = torch.load(run / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    L = int(art["layer"])
    rows = art["rows"].float()
    facts = list(art["facts"])

    # ---- 1. rows
    geo = row_geometry(rows, facts, a.coords)
    write_row_tables(geo, facts, out)
    plots = plot_rows(geo, facts, out)
    s = geo["summary"]
    report = {"run_dir": str(run.resolve()), "layer": L, "rows": s, "per_row": geo["per_row"],
              "plots": plots}
    print(f"\n=== 1. What the {s['rows_total']} rows look like (layer {L}, d={s['hidden_size']}) ===")
    print(f"nonzero rows: {s['rows_nonzero']}/{s['rows_total']}"
          + (f"  (zero / untrained: {s['zero_rows']})" if s["zero_rows"] else ""))
    print(f"||De||  min {s['norm_min']:.4g}  median {s['norm_median']:.4g}  max {s['norm_max']:.4g}")
    print(f"{'row':>4} {'||De||':>8} {'cos→mean':>9} {'nearest':>8} {'cos':>6}  fact   | largest coords dim:value")
    for r in geo["per_row"]:
        coords = " ".join(f"{k}:{v:+.3g}" for k, v in r["top_coords"][:4])
        print(f"{r['row']:>4} {r['norm']:>8.4g} {r['cos_to_mean_row']:>9.3f} {r['nearest_other_row']:>8} "
              f"{r['nearest_other_cos']:>6.3f}  {r['fact'][:40]:40s} | {coords}")
    print(f"\n=== Are they the same? ===")
    print(f"SAME VALUE?     {s['same_value_verdict']}")
    print(f"                ||De_i - De_j|| / mean norm: min {s['pairwise_rel_distance_min']:.3f}, "
          f"median {s['pairwise_rel_distance_median']:.3f}, max {s['pairwise_rel_distance_max']:.3f}  (0 = identical)")
    print(f"SAME DIRECTION? {s['same_direction_verdict']}")
    print(f"                pairwise cos: mean {s['pairwise_cos_mean']:.3f}, min {s['pairwise_cos_min']:.3f}, "
          f"max {s['pairwise_cos_max']:.3f}; random {s['hidden_size']}-d vectors ≈ {s['pairwise_cos_random_baseline']:.3f}")
    print(f"                pairs with cos > 0.9: {s['pairs_cos_above_0.9']}/{s['pairs_total']}; "
          f"median cos(row, mean row) {s['cos_to_mean_row_median']:.3f}")
    print(f"                SVD energy top-5: {[round(x, 3) for x in s['svd_energy_top5']]}, "
          f"effective rank {s['effective_rank']:.1f} of {s['rows_nonzero']}")

    if a.rows_only:
        (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {out}/report.json, rows_summary.csv, rows_cosine.csv, rows_vectors.pt"
              + (", " + ", ".join(plots) if plots else ""))
        return report

    # ---- model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from linear_router import load_linear_classifier_artifact
    from prepare_mcf_association_source import load_mcf_forget_data
    from static_overlap_fact_association_embeddings import PLAN, make_unknown_examples

    man = json.loads((run / "association_manifest.json").read_text())
    tok = AutoTokenizer.from_pretrained(man["model_path"], use_fast=True,
                                        local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        man["model_path"], dtype=torch.float32, local_files_only=a.local_files_only,
        attn_implementation="eager").to(a.device).eval()
    model.requires_grad_(False)
    seed = int((man.get("sampling") or {}).get("seed", 1))
    _, mcf_facts, examples = load_mcf_forget_data(tok, man["mcf_path"], seed=seed)
    if [f["id"] for f in mcf_facts] != [f["id"] for f in facts]:
        raise ValueError("Rebuilt MCF facts do not match the artifact")
    row_of = {f["id"]: i for i, f in enumerate(facts)}
    examples = [e for e in examples if a.split == "all" or e.split == a.split]
    unknown_map = make_unknown_examples(examples, tok, PLAN["max_length"], UNKNOWN)

    # ---- 2. interchange
    if not a.skip_interchange:
        ic = interchange(model, L, examples, unknown_map, rows, geo["live"], row_of,
                         float(PLAN["target_probability"]), a.device, a.seed)
        report["interchange"] = ic
        print(f"\n=== 2. Can one vector replace all? (oracle routing, worst view per fact, "
              f"forgotten = p < {ic['target_probability']:g}) ===")
        print(f"{'edit added at the boundary':26s} {'forgotten':>10} {'median worst p(true)':>21} "
              f"{'mean p(IDK)':>12}")
        for name, t in ic["arms"].items():
            print(f"{name:26s} {t['facts_forgotten']:>4}/{t['facts']:<5} "
                  f"{t['median_worst_view_p_true']:>21.3e} {t['mean_p_idk_completion']:>12.3e}")
        print("own_row ≈ mean_row -> one shared vector is enough; "
              "own_row ≫ other_fact_row -> rows are fact-specific")

    # ---- 3. per prompt h -> h'
    if not a.skip_per_prompt:
        idk_id = tok(UNKNOWN, add_special_tokens=False).input_ids[0]
        cap = {}

        def grab(_m, _args, o):  # registered after the bank hook, so it sees the edited output
            h = o[0] if isinstance(o, tuple) else o
            cap["h"] = h[0, -1].detach().float().clone()

        @torch.no_grad()
        def run_once(m, ids):
            handle = model.model.layers[L].register_forward_hook(grab)
            try:
                logits = m(input_ids=ids, use_cache=False).logits[0, -1].float()
            finally:
                handle.remove()
            return cap["h"], logits

        def lens(h):
            return model.lm_head(model.model.norm(h.to(a.device)[None]))[0].float()

        def top(logits, k=5):
            v, i = logits.softmax(-1).topk(k)
            return [[tok.decode([int(t)]), round(float(x), 4)] for x, t in zip(v, i)]

        def prob(logits, t):
            return float(logits.softmax(-1)[t])

        items = []
        for e in examples:
            first = next(i for i, lab in enumerate(e.labels) if lab != -100)
            items.append(dict(id=e.id, prompt=e.prompt, ids=e.input_ids[:first],
                              true_id=e.input_ids[first], row=row_of[e.fact_id], answer=e.completion))
        for p in a.prompt:
            items.append(dict(id=f"extra:{p}", prompt=p, ids=tok(p).input_ids,
                              true_id=None, row=None, answer=None))
        for it in items:
            it["ids_t"] = torch.tensor([it["ids"]], device=a.device)
            it["h"], it["base_logits"] = run_once(model, it["ids_t"])
        wrapped, bank = load_linear_classifier_artifact(model, art)
        try:
            for it in items:
                it["h_rt"], it["rt_logits"] = run_once(wrapped, it["ids_t"])
                it["routed"] = list(bank.last_active_fact_indices[0])
        finally:
            bank.close()

        prompts, lines = [], []
        for it in items:
            h, h_rt = it["h"].cpu(), it["h_rt"].cpu()
            rec = {"id": it["id"], "prompt": it["prompt"], "answer": it["answer"],
                   "routed_rows": it["routed"], "true_row": it["row"],
                   "runtime_unchanged_exact": bool(torch.equal(h_rt, h)),
                   "max_abs_runtime_change": float((h_rt - h).abs().max()),
                   "h_norm": float(h.norm()),
                   "final_top5_base": top(it["base_logits"]), "final_top5_runtime": top(it["rt_logits"]),
                   "final_p_idk_base": prob(it["base_logits"], idk_id),
                   "final_p_idk_runtime": prob(it["rt_logits"], idk_id)}
            if it["row"] is not None:
                d = rows[it["row"]]
                hp = h + d
                c = float(F.cosine_similarity(h, hp, dim=0))
                par = float(d @ h / h.norm())
                with torch.no_grad():
                    lh, lhp = lens(h).cpu(), lens(hp).cpu()
                rec.update({
                    "router_correct": it["routed"] == [it["row"]],
                    "delta_norm": float(d.norm()), "h_prime_norm": float(hp.norm()),
                    "delta_to_h_norm_ratio": float(d.norm() / h.norm()),
                    "cos_h_delta": float(F.cosine_similarity(h, d, dim=0)),
                    "cos_h_hprime": c, "angle_h_hprime_deg": math.degrees(math.acos(max(-1.0, min(1.0, c)))),
                    "delta_parallel_to_h": par,
                    "delta_orthogonal_to_h": float((d - par * h / h.norm()).norm()),
                    "top_coords": [{"k": k, "h": float(h[k]), "delta": float(d[k]), "h_prime": float(hp[k])}
                                   for k in d.abs().topk(a.coords).indices.tolist()],
                    "lens_p_true_h": prob(lh, it["true_id"]), "lens_p_true_hprime": prob(lhp, it["true_id"]),
                    "lens_top5_h": top(lh), "lens_top5_hprime": top(lhp),
                    "final_p_true_base": prob(it["base_logits"], it["true_id"]),
                    "final_p_true_runtime": prob(it["rt_logits"], it["true_id"]),
                })
                lines.append(
                    f"{it['id'][:36]:36s} {'ok' if rec['router_correct'] else str(it['routed']):>5} "
                    f"|h| {rec['h_norm']:7.2f} |De| {rec['delta_norm']:6.2f} |h'| {rec['h_prime_norm']:7.2f} "
                    f"cos(h,h') {c:.3f}  P(true) {rec['final_p_true_base']:.3f}->{rec['final_p_true_runtime']:.1e}"
                    f"  P(' I') {rec['final_p_idk_base']:.3f}->{rec['final_p_idk_runtime']:.3f}")
            else:
                lines.append(f"{it['id'][:36]:36s} routed={it['routed']} h'==h exact: "
                             f"{rec['runtime_unchanged_exact']}  top1 {rec['final_top5_base'][0]} -> "
                             f"{rec['final_top5_runtime'][0]}")
            prompts.append(rec)
        report["prompts"] = prompts
        forget = [r for r in prompts if r["true_row"] is not None]
        print(f"\n=== 3. h_{L} -> h' per prompt (runtime routing; first {a.print_prompts} of {len(lines)}) ===")
        print("\n".join(lines[: a.print_prompts] + [x for x in lines if x.startswith("extra:")]))
        if forget:
            med = lambda k: float(torch.tensor([r[k] for r in forget]).median())
            print(f"medians over {len(forget)} forget views: |h| {med('h_norm'):.3g}, |De|/|h| "
                  f"{med('delta_to_h_norm_ratio'):.3g}, cos(h,h') {med('cos_h_hprime'):.3f}, "
                  f"angle {med('angle_h_hprime_deg'):.1f}°; router correct "
                  f"{sum(r['router_correct'] for r in forget)}/{len(forget)}")
            one = forget[0]
            print(f"\nexample {one['id']}: h_k + De_k = h'_k on the {a.coords} coords De moves most")
            for c in one["top_coords"]:
                print(f"  dim {c['k']:5d}: {c['h']:+9.4f} + {c['delta']:+9.4f} = {c['h_prime']:+9.4f}")
            print("  logit lens top5 h :", one["lens_top5_h"])
            print("  logit lens top5 h':", one["lens_top5_hprime"])

    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}/report.json, rows_summary.csv, rows_cosine.csv, rows_vectors.pt"
          + (", " + ", ".join(plots) if plots else ""))
    return report


if __name__ == "__main__":
    main()
