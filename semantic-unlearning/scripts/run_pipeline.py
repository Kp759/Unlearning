"""Run the full semantic token identification pipeline."""
import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script: str, extra_args: list = None, config: str = "config/config.yaml"):
    cmd = [sys.executable, script, "--config", config] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run the full unlearning pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip hidden state extraction (use existing .npy files).")
    args = parser.parse_args()

    scripts_dir = Path(__file__).parent

    if not args.skip_extraction:
        run_script(str(scripts_dir / "extract_hidden_states.py"), config=args.config)
    else:
        print("Skipping hidden state extraction.")

    run_script(str(scripts_dir / "train_probe.py"), config=args.config)
    run_script(str(scripts_dir / "identify_tokens.py"), config=args.config)

    print("\n" + "=" * 60)
    print("Pipeline complete! Output files:")

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    expected_outputs = [
        out_dir / "hidden_states",
        out_dir / "probes",
        out_dir / "layer_accuracies.json",
        out_dir / "probe_accuracy_per_layer.png",
        out_dir / "semantic_tokens.json",
        out_dir / "token_scores.png",
    ]
    for path in expected_outputs:
        exists = "✓" if path.exists() else "✗"
        print(f"  [{exists}] {path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
