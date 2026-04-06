from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Global style
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})


def plot_probe_accuracy_per_layer(
    layer_accuracies: Dict[int, float],
    title: str = "Probe Accuracy per Layer",
    threshold: float = 0.70,
    save_path: Optional[str] = None,
    show: bool = False,
):
    layers = sorted(layer_accuracies.keys())
    accs = [layer_accuracies[l] for l in layers]
    best_layer = layers[int(np.argmax(accs))]
    best_acc = max(accs)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, accs, marker="o", linewidth=2, markersize=4, color="#2196F3", label="Probe accuracy")
    ax.axhline(threshold, linestyle="--", color="#FF5722", linewidth=1.5, label=f"Threshold ({threshold})")
    ax.axhline(0.5, linestyle=":", color="#9E9E9E", linewidth=1.2, label="Random chance (0.5)")

    # Shaded region above threshold
    ax.fill_between(layers, accs, threshold, where=[a >= threshold for a in accs],
                    alpha=0.15, color="#2196F3", interpolate=True)

    # Annotate best layer
    ax.annotate(
        f"Best: L{best_layer}\n{best_acc:.3f}",
        xy=(best_layer, best_acc),
        xytext=(best_layer + max(1, len(layers) * 0.05), best_acc - 0.04),
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
        fontsize=9,
    )

    ax.set_ylim(0.4, 1.05)
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_token_scores(
    token_strs: List[str],
    token_scores: List[float],
    title: str = "Semantic Token Scores",
    top_k: int = 20,
    save_path: Optional[str] = None,
    show: bool = False,
):
    # Select top-k by score
    pairs = sorted(zip(token_strs, token_scores), key=lambda x: x[1], reverse=True)[:top_k]
    if not pairs:
        return
    labels, scores = zip(*pairs)
    labels = [repr(l) for l in labels]

    colors = []
    for s in scores:
        if s >= 0.8:
            colors.append("#2196F3")   # blue
        elif s >= 0.7:
            colors.append("#FF9800")   # orange
        else:
            colors.append("#9E9E9E")   # gray

    fig, ax = plt.subplots(figsize=(8, max(4, top_k * 0.35)))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, scores, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0.7, linestyle="--", color="red", linewidth=1.5, label="Threshold (0.7)")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Max Probe Score P(forget)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_layer_comparison(
    results_dict: Dict[str, Dict[int, float]],
    title: str = "Layer-wise Probe Accuracy Comparison",
    save_path: Optional[str] = None,
    show: bool = False,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, layer_accs in results_dict.items():
        layers = sorted(layer_accs.keys())
        accs = [layer_accs[l] for l in layers]
        ax.plot(layers, accs, marker="o", linewidth=2, markersize=3, label=name)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved plot to {save_path}")
    if show:
        plt.show()
    plt.close(fig)
