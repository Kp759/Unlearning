"""Train logistic regression probes on extracted hidden states, one per layer."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.probing import LayerwiseProber
from src.utils import plot_probe_accuracy_per_layer


def main():
    parser = argparse.ArgumentParser(description="Train layerwise linear probes.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    hs_dir = Path(cfg["output"]["dir"]) / "hidden_states"
    forget_files = sorted(hs_dir.glob("forget_layer_*.npy"))

    if not forget_files:
        raise FileNotFoundError(f"No hidden state files found in {hs_dir}. Run extract_hidden_states.py first.")

    forget_states = {}
    retain_states = {}
    for fpath in forget_files:
        m = re.search(r"forget_layer_(\d+)\.npy", fpath.name)
        if not m:
            continue
        layer_idx = int(m.group(1))
        rpath = hs_dir / f"retain_layer_{layer_idx:03d}.npy"
        if not rpath.exists():
            print(f"  Warning: retain file missing for layer {layer_idx}, skipping.")
            continue
        forget_states[layer_idx] = np.load(str(fpath))
        retain_states[layer_idx] = np.load(str(rpath))

    print(f"Loaded hidden states for {len(forget_states)} layers.")

    probe_cfg = cfg["probing"]
    prober = LayerwiseProber(
        C=probe_cfg.get("C", 1.0),
        max_iter=probe_cfg.get("max_iter", 1000),
        test_size=probe_cfg.get("test_size", 0.2),
        seed=cfg["data"].get("seed", 42),
    )

    save_probes = cfg["output"].get("save_probes", True)
    probe_dir = str(Path(cfg["output"]["dir"]) / "probes") if save_probes else None

    print("Training layerwise probes...")
    results = prober.run(forget_states, retain_states, save_dir=probe_dir)

    summary = prober.summary(results, threshold=probe_cfg.get("token_probe_threshold", 0.70))
    print("\n=== Probe Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    layer_accuracies = {layer: r.accuracy for layer, r in results.items()}
    acc_path = Path(cfg["output"]["dir"]) / "layer_accuracies.json"
    acc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(acc_path, "w") as f:
        json.dump({str(k): v for k, v in layer_accuracies.items()}, f, indent=2)
    print(f"Saved layer accuracies to {acc_path}")

    if cfg["output"].get("save_plots", True):
        plot_path = str(Path(cfg["output"]["dir"]) / "probe_accuracy_per_layer.png")
        plot_probe_accuracy_per_layer(
            layer_accuracies=layer_accuracies,
            title="Probe Accuracy per Layer",
            threshold=probe_cfg.get("token_probe_threshold", 0.70),
            save_path=plot_path,
            show=False,
        )


if __name__ == "__main__":
    main()
