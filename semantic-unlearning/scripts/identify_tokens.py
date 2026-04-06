"""Identify semantic tokens that carry the forget concept using token-level probing."""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import TOFUDataset
from src.probing import HiddenStateExtractor, LinearProbe, SemanticTokenIdentifier
from src.utils import plot_token_scores


def main():
    parser = argparse.ArgumentParser(description="Identify semantic forget tokens.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Probe score threshold (overrides config).")
    parser.add_argument("--best-layer", type=int, default=None,
                        help="Layer index to use as primary probe (overrides auto-selection).")
    parser.add_argument("--selectivity-filter", type=float, default=2.0,
                        help="Min forget/retain frequency ratio for selectivity filtering.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    threshold = args.threshold if args.threshold is not None else cfg["probing"].get("token_probe_threshold", 0.70)
    out_dir = Path(cfg["output"]["dir"])

    # Load layer accuracies
    acc_path = out_dir / "layer_accuracies.json"
    if not acc_path.exists():
        raise FileNotFoundError(f"{acc_path} not found. Run train_probe.py first.")
    with open(acc_path) as f:
        layer_accuracies_raw = json.load(f)
    layer_accuracies = {int(k): v for k, v in layer_accuracies_raw.items()}

    best_layer = args.best_layer
    if best_layer is None:
        best_layer = max(layer_accuracies, key=layer_accuracies.__getitem__)
    print(f"Using best_layer={best_layer} with accuracy={layer_accuracies.get(best_layer, 'N/A'):.4f}")

    # Load probes
    probe_dir = out_dir / "probes"
    probe_files = sorted(probe_dir.glob("probe_layer_*.pkl"))
    if not probe_files:
        raise FileNotFoundError(f"No probe files in {probe_dir}. Run train_probe.py first.")

    probes = {}
    probe_results_stub = {}
    for pf in probe_files:
        idx = int(pf.stem.split("_")[-1])
        probe = LinearProbe.load(str(pf))
        probes[idx] = probe

        # Create a minimal ProbeResult-like object for summary access
        from src.probing.probe import ProbeResult
        import numpy as np
        probe_results_stub[idx] = ProbeResult(
            layer_idx=idx,
            accuracy=layer_accuracies.get(idx, 0.0),
            auc=0.0,
            report="",
            n_train=0,
            n_test=0,
            coef=np.zeros(1),
        )

    print(f"Loaded {len(probes)} probes.")

    # Load extractor
    extractor = HiddenStateExtractor(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
        max_length=cfg["model"]["max_length"],
    )

    # Load TOFU texts
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
    print(f"Forget: {len(forget_texts)}, Retain: {len(retain_texts)}")

    # Identify semantic tokens
    identifier = SemanticTokenIdentifier(
        extractor=extractor,
        probes=probes,
        probe_results=probe_results_stub,
    )

    result = identifier.identify(
        forget_texts=forget_texts,
        retain_texts=retain_texts,
        threshold=threshold,
        best_layer=best_layer,
        layers_to_check=sorted(probes.keys()),
        batch_size=cfg["extraction"]["batch_size"],
    )

    result = identifier.filter_by_selectivity(result, min_forget_retain_ratio=args.selectivity_filter)

    tokens_path = str(out_dir / "semantic_tokens.json")
    result.save(tokens_path)

    if cfg["output"].get("save_plots", True) and result.semantic_tokens:
        token_strs = [t.token_str for t in result.semantic_tokens]
        token_scores = [t.max_probe_accuracy for t in result.semantic_tokens]
        plot_token_scores(
            token_strs=token_strs,
            token_scores=token_scores,
            title="Semantic Token Scores",
            top_k=20,
            save_path=str(out_dir / "token_scores.png"),
            show=False,
        )

    print("\nDone. Summary:")
    for k, v in result.summary().items():
        if k != "top_tokens":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
