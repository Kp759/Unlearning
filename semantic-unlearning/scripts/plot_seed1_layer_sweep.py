#!/usr/bin/env python3
"""Plot seed-1 generalization vs. layer for Genie and Regular training.

Creates exactly two PNGs:
  - seed1_genie_generalization_vs_layer.png
  - seed1_regular_generalization_vs_layer.png

Each figure compares the benchmark-specific generalization metric:
  MCF    -> Gen
  ZsRE   -> Gen
  MQuAKE -> AtomicGen

All seven layers are included: 1, 3, 7, 13, 19, 23, 27.
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
    "MCF Gen": np.array([5.75, 2.00, 0.39, 0.029, 0.0027, 0.015, 0.023]),
    "ZsRE Gen": np.array([26.2, 23.8, 18.3, 13.5, 7.8, 8.8, 27.9]),
    "MQuAKE AtomicGen": np.array([50.2, 50.2, 47.6, 31.5, 2.2, 0.9, 50.2]),
}

REGULAR = {
    "MCF Gen": np.array([1.22, 0.032, 0.042, 0.024, 0.0027, 0.015, 0.023]),
    "ZsRE Gen": np.array([26.2, 23.8, 18.3, 13.5, 7.8, 8.8, 27.9]),
    "MQuAKE AtomicGen": np.array([50.2, 50.2, 47.6, 31.9, 2.2, 1.1, 50.2]),
}


def plot_generalization(data, regime, output_path, dpi=300):
    fig, ax = plt.subplots(figsize=(12.5, 6.5))

    markers = ["o", "s", "^"]
    labels = {
        "MCF Gen": "MCF Gen",
        "ZsRE Gen": "ZsRE Gen",
        "MQuAKE AtomicGen": "MQuAKE AtomicGen",
    }

    for marker, (name, values) in zip(markers, data.items()):
        ax.plot(
            LAYERS,
            values,
            marker=marker,
            linewidth=2.4,
            markersize=7,
            label=labels[name],
        )

    ax.set_yscale("log")
    ax.set_xticks(LAYERS)
    ax.set_xticklabels([f"L{x}" for x in LAYERS])
    ax.set_xlabel("Transformer layer", fontsize=12)
    ax.set_ylabel("Generalization metric (log scale; lower is better)", fontsize=12)

    subtitle = (
        "edits trained with ground-truth routing"
        if regime == "Genie"
        else "edits trained under the router's own routing"
    )
    ax.set_title(
        f"{regime} (seed 1): generalization vs. layer\n{subtitle}",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )

    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False, fontsize=10, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
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

    genie_path = out_dir / "seed1_genie_generalization_vs_layer.png"
    regular_path = out_dir / "seed1_regular_generalization_vs_layer.png"

    plot_generalization(GENIE, "Genie", genie_path, dpi=args.dpi)
    plot_generalization(REGULAR, "Regular", regular_path, dpi=args.dpi)

    print(f"Saved: {genie_path}")
    print(f"Saved: {regular_path}")


if __name__ == "__main__":
    main()
