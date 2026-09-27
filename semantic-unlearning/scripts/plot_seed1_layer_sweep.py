#!/usr/bin/env python3
"""Plot seed-1 layer sweeps for Genie and Regular MCF training.

Creates exactly two PNGs, one per training regime.  Each PNG contains three
panels: MCF, ZsRE, and MQuAKE.

Usage:
    python scripts/plot_seed1_layer_sweep.py
    python scripts/plot_seed1_layer_sweep.py --out-dir outputs/seed1_layer_sweep_plots
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LAYERS = np.array([1, 3, 7, 13, 19, 23, 27], dtype=int)

GENIE = {
    "MCF": {
        "Eff": np.array([4.83, 0.22, 0.019, 8.0e-4, 7.8e-5, 3.5e-5, 1.5e-5]),
        "Gen": np.array([5.75, 2.00, 0.39, 0.029, 0.0027, 0.015, 0.023]),
        "Spe": np.array([20.4, 20.4, 20.4, 20.4, 20.4, 20.4, 20.4]),
    },
    "ZsRE": {
        "Eff": np.array([1.08, 0.67, 0.0, 0.0, 0.0, 0.0, 30.0]),
        "Gen": np.array([26.2, 23.8, 18.3, 13.5, 7.8, 8.8, 27.9]),
        "Spe": np.array([32.3, 32.3, 32.3, 32.3, 32.3, 32.3, 32.3]),
        "Paraphrases routed": np.array([8.0, 18.0, 30.0, 51.0, 66.0, 61.0, 70.0]),
    },
    "MQuAKE": {
        "Eff": np.array([2.25, 1.06, 0.0, 0.0, 0.0, 0.0, 73.1]),
        "AtomicGen": np.array([50.2, 50.2, 47.6, 31.5, 2.2, 0.9, 50.2]),
        "Retain AtomicGen": np.array([43.5, 43.5, 43.1, 40.3, 37.9, 37.4, 43.5]),
        "Questions routed": np.array([0.0, 0.0, 7.0, 27.0, 96.0, 98.0, 99.0]),
    },
}

REGULAR = {
    "MCF": {
        "Eff": np.array([0.33, 8.4e-5, 8.6e-5, 8.0e-5, 7.8e-5, 1.4e-4, 9.5e-5]),
        "Gen": np.array([1.22, 0.032, 0.042, 0.024, 0.0027, 0.015, 0.023]),
        "Spe": np.array([20.4, 20.4, 20.4, 20.4, 20.4, 20.4, 20.4]),
    },
    "ZsRE": {
        "Eff": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.67, 30.0]),
        "Gen": np.array([26.2, 23.8, 18.3, 13.5, 7.8, 8.8, 27.9]),
        "Spe": np.array([32.3, 32.3, 32.3, 32.3, 32.3, 32.3, 32.3]),
        "Paraphrases routed": np.array([8.0, 18.0, 30.0, 51.0, 66.0, 61.0, 70.0]),
    },
    "MQuAKE": {
        "Eff": np.array([4.82, 7.15, 0.0, 0.0, 0.0, 0.0, 73.1]),
        "AtomicGen": np.array([50.2, 50.2, 47.6, 31.9, 2.2, 1.1, 50.2]),
        "Retain AtomicGen": np.array([43.5, 43.5, 43.0, 40.4, 37.9, 37.4, 43.5]),
        # Seed-1 regular L27 routing was not included in the supplied table.
        "Questions routed": np.array([0.0, 0.0, 7.0, 27.0, 96.0, 98.0, np.nan]),
    },
}


def _merge_legends(ax, ax2=None, *, loc="best"):
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, loc=loc, frameon=False, fontsize=9)


def _plot_mcf(ax, data):
    # Eff/Gen span several orders of magnitude, so use a log axis for them.
    ax.plot(LAYERS, data["Eff"], marker="o", linewidth=2, label="Eff ↓")
    ax.plot(LAYERS, data["Gen"], marker="s", linewidth=2, label="Gen ↓")
    ax.set_yscale("log")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Eff / Gen (log scale)")
    ax.set_title("MCF")
    ax.set_xticks(LAYERS)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        LAYERS,
        data["Spe"],
        marker="^",
        linestyle="--",
        linewidth=2,
        label="Spe ↑",
    )
    ax2.set_ylabel("Specificity")
    lo = float(np.nanmin(data["Spe"]))
    hi = float(np.nanmax(data["Spe"]))
    pad = max(1.0, (hi - lo) * 0.2)
    ax2.set_ylim(lo - pad, hi + pad)
    _merge_legends(ax, ax2, loc="upper right")


def _plot_zsre(ax, data):
    ax.plot(LAYERS, data["Eff"], marker="o", linewidth=2, label="Eff ↓")
    ax.plot(LAYERS, data["Gen"], marker="s", linewidth=2, label="Gen ↓")
    ax.plot(LAYERS, data["Spe"], marker="^", linewidth=2, label="Spe ↑")
    ax.set_xlabel("Layer")
    ax.set_ylabel("MCF-style metric")
    ax.set_title("ZsRE")
    ax.set_xticks(LAYERS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        LAYERS,
        data["Paraphrases routed"],
        marker="D",
        linestyle="--",
        linewidth=2,
        label="Paraphrases routed (%)",
    )
    ax2.set_ylabel("Routed (%)")
    ax2.set_ylim(0, 105)
    _merge_legends(ax, ax2, loc="upper left")


def _plot_mquake(ax, data):
    ax.plot(LAYERS, data["Eff"], marker="o", linewidth=2, label="Eff ↓")
    ax.plot(LAYERS, data["AtomicGen"], marker="s", linewidth=2, label="AtomicGen ↓")
    ax.plot(
        LAYERS,
        data["Retain AtomicGen"],
        marker="^",
        linewidth=2,
        label="Retain AtomicGen ↑",
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("MQuAKE metric")
    ax.set_title("MQuAKE")
    ax.set_xticks(LAYERS)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(
        LAYERS,
        data["Questions routed"],
        marker="D",
        linestyle="--",
        linewidth=2,
        label="Questions routed (%)",
    )
    ax2.set_ylabel("Routed (%)")
    ax2.set_ylim(0, 105)
    _merge_legends(ax, ax2, loc="upper left")


def make_figure(data, regime, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    _plot_mcf(axes[0], data["MCF"])
    _plot_zsre(axes[1], data["ZsRE"])
    _plot_mquake(axes[2], data["MQuAKE"])

    subtitle = (
        "Genie: edits trained with ground-truth routing"
        if regime == "Genie"
        else "Regular: edits trained under the router's own routing"
    )
    fig.suptitle(
        f"Seed 1 Layer Sweep — {regime}\n{subtitle}",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.015,
        "↓ lower is better for Eff/Gen/AtomicGen; ↑ higher is better for specificity and retain. "
        "Routing curves are diagnostics.",
        ha="center",
        fontsize=9,
    )

    if regime == "Regular" and np.isnan(data["MQuAKE"]["Questions routed"][-1]):
        fig.text(
            0.985,
            0.015,
            "Regular MQuAKE L27 routing: not provided",
            ha="right",
            fontsize=8,
        )

    fig.tight_layout(rect=(0, 0.055, 1, 0.92))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="outputs/seed1_layer_sweep_plots",
        help="Directory for the two PNG files.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Respect --dpi without duplicating plotting code.
    original_savefig = plt.Figure.savefig
    def savefig_with_requested_dpi(self, *a, **kw):
        kw["dpi"] = args.dpi
        return original_savefig(self, *a, **kw)
    plt.Figure.savefig = savefig_with_requested_dpi

    genie_path = out_dir / "seed1_genie_all_benchmarks.png"
    regular_path = out_dir / "seed1_regular_all_benchmarks.png"

    make_figure(GENIE, "Genie", genie_path)
    make_figure(REGULAR, "Regular", regular_path)

    print(f"Saved: {genie_path}")
    print(f"Saved: {regular_path}")


if __name__ == "__main__":
    main()
