"""Extract hidden states for forget and retain splits and save as .npy files."""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import TOFUDataset
from src.probing import HiddenStateExtractor


def main():
    parser = argparse.ArgumentParser(description="Extract hidden states from LLM layers.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dataset = TOFUDataset(
        forget_split=cfg["data"]["forget_split"],
        retain_split=cfg["data"]["retain_split"],
    )

    n_forget = cfg["data"].get("n_forget")
    n_retain = cfg["data"].get("n_retain")
    seed = cfg["data"].get("seed", 42)

    samples = dataset.get_samples(n_forget=n_forget, n_retain=n_retain, seed=seed)
    forget_texts = [s.text for s in samples if s.label == 1]
    retain_texts = [s.text for s in samples if s.label == 0]

    print(f"Forget texts: {len(forget_texts)}, Retain texts: {len(retain_texts)}")

    extractor = HiddenStateExtractor(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
        max_length=cfg["model"]["max_length"],
    )

    batch_size = cfg["extraction"]["batch_size"]
    aggregate = cfg["extraction"]["aggregate"]
    layers = cfg["probing"].get("layers")

    print("Extracting forget hidden states...")
    forget_states = extractor.extract(forget_texts, batch_size=batch_size, layers=layers, aggregate=aggregate)

    print("Extracting retain hidden states...")
    retain_states = extractor.extract(retain_texts, batch_size=batch_size, layers=layers, aggregate=aggregate)

    out_dir = Path(cfg["output"]["dir"]) / "hidden_states"
    out_dir.mkdir(parents=True, exist_ok=True)

    for layer_idx, arr in forget_states.items():
        path = out_dir / f"forget_layer_{layer_idx:03d}.npy"
        np.save(path, arr)

    for layer_idx, arr in retain_states.items():
        path = out_dir / f"retain_layer_{layer_idx:03d}.npy"
        np.save(path, arr)

    print(f"Saved hidden states for {len(forget_states)} layers to {out_dir}")


if __name__ == "__main__":
    main()
