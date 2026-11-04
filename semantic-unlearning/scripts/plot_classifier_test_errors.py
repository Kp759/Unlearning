#!/usr/bin/env python3
"""Two summary figures of the classifier's errors on TEST prompts only (the official evaluation
prompts and the held-out no-association prompts), from the router diagnostics
(outputs/router_diag_ours_control_v1/<ds>/seed1/report/metrics.json, layer_sweep/sweep.json):

  false_fires_by_prompt_type.png   per dataset: how often the classifier fires on each kind of
                                   prompt that must not fire (bars), with the counts
  errors_by_read_block.png         per dataset: test false-fire rate, miss rate and the
                                   same-subject fire rate when the same classifier is refit at
                                   every block (the deployed block marked)

    python scripts/plot_classifier_test_errors.py --root outputs/router_diag_ours_control_v1 --out classifier_errors_test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DS = ("mcf", "zsre", "mquake", "rwku")
NAME = {"mcf": "MCF", "zsre": "zsRE", "mquake": "MQuAKE", "rwku": "RWKU"}
GROUPS = [("retain", "official retain prompts\n(other subjects)"),
          ("null_unseen_subject", "unseen subject"),
          ("null_generic", "generic text"),
          ("null_ppl_prefix", "perplexity text"),
          ("null_protected_other_relation", "PROTECTED SUBJECT,\nother relation")]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/router_diag_ours_control_v1")
    ap.add_argument("--out", default="classifier_errors_test")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- figure 1: false fires by prompt type
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, ds in zip(axes, DS):
        m = json.loads((Path(args.root) / ds / f"seed{args.seed}" / "report" / "metrics.json").read_text())
        rates, counts, labels = [], [], []
        for key, label in GROUPS:
            if key == "retain":
                v = m["eval"]["by_split"].get("retain") or m["eval"]["by_split"].get("neighborhood")
                k, n = int(v.get("fp", 0)), int(v.get("fp", 0) + v.get("tn", 0))
            else:
                v = (m.get("null") or {}).get("by_group", {}).get(key)
                if v is None:
                    continue
                k, n = int(v["fired"]), int(v["n"])
            rates.append(100.0 * k / max(n, 1))
            counts.append(f"{k}/{n}")
            labels.append(label)
        colors = ["#c0392b" if "PROTECTED" in l else "#7f8c8d" for l in labels]
        bars = ax.bar(range(len(rates)), rates, color=colors)
        for b, c in zip(bars, counts):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, c, ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5, rotation=20, ha="right")
        ax.set_title(f"{NAME[ds]}: prompts that must not fire", fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("classifier fired (%)")
    fig.suptitle("Where the paper's classifier fires when it should not (test prompts only, Llama-3.2-3B, seed 1).\n"
                 "Grey: the benchmarks' own retain prompts and prompts with no protected subject. "
                 "Red: a protected subject asked about something else, which no benchmark contains.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(out / "false_fires_by_prompt_type.png", dpi=150)
    plt.close(fig)

    # ---- figure 2: errors by read block
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, ds in zip(axes, DS):
        d = Path(args.root) / ds / f"seed{args.seed}"
        m = json.loads((d / "report" / "metrics.json").read_text())
        sw = json.loads((d / "layer_sweep" / "sweep.json").read_text())["sweep"]
        rows = sorted([e for e in sw if e["variant"] == "baseline" and e["rule"] == "deployed"], key=lambda e: e["layers"][0])
        L = [e["layers"][0] for e in rows]
        res = [e["by_eligibility"][e.get("deployed_eligibility", "exact")] for e in rows]
        ax.plot(L, [100 * r["eval"]["fpr"] for r in res], "o-", ms=3, color="#c0392b", label="false-fire rate, official retain prompts")
        ax.plot(L, [100 * r["eval"]["fnr"] for r in res], "o-", ms=3, color="#e67e22", label="miss rate, forget prompts")
        if any(r.get("null_fpr") is not None for r in res):
            ax.plot(L, [100 * (r.get("null_fpr") or 0) for r in res], "s-", ms=3, color="#8e44ad",
                    label="fire rate, no-association prompts\n(mostly protected subject, other relation)")
        ax.axvline(m["router_layer"], color="k", ls="--", lw=1, label=f"deployed block {m['router_layer']}")
        ax.set_title(NAME[ds], fontsize=11)
        ax.set_xlabel("block the classifier reads")
        ax.set_ylim(0, 100 if ds == "rwku" else 40)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("error rate (%)")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("The same classifier refit at every block, scored on test prompts only: no block removes the same-subject false fires",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "errors_by_read_block.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out}/false_fires_by_prompt_type.png and errors_by_read_block.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
