"""
scripts/scaling_experiments.py
--------------------------------
Runs the full unlearning pipeline across:
  - Multiple forget splits: forget01, forget05, forget10
  - Multiple models: LLaMA-3.2-1B, LLaMA-3.1-8B, Phi-3.5-mini

For each combination:
  1. Fine-tune on full TOFU (forget+retain)
  2. Extract hidden states
  3. Train probes
  4. Identify semantic tokens (threshold=0.999, name-only best config)
  5. Erase embeddings (zero method)
  6. Evaluate (forget quality, retain quality, perplexity, truth ratio)

Outputs a combined results table across all experiments.

Run all:
    python scripts/scaling_experiments.py \
        --config config/config.yaml \
        --splits forget01 forget05 forget10 \
        --models llama1b llama8b \
        --method zero

Run single split, single model (test):
    python scripts/scaling_experiments.py \
        --config config/config.yaml \
        --splits forget01 \
        --models llama1b \
        --method zero \
        --skip-finetune   # if already finetuned
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Model registry ─────────────────────────────────────────────────────────────
# Update paths to match your HPC setup
MODEL_REGISTRY = {
    "llama1b": {
        "name": "meta-llama/Llama-3.2-1B",
        "hf_path": "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-1B/snapshots/4e20de362430cd3b72f300e6b0f18e50e7166e08",
        "dtype": "float16",
        "batch_size": 8,
        "ft_batch_size": 8,
        "ft_epochs": 5,
        "ft_lr": 5e-5,
    },
    "llama8b": {
        "name": "meta-llama/Llama-3.1-8B",
        "hf_path": "/scratch/yl258/kp759/hf/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "dtype": "float16",
        "batch_size": 4,      # smaller batch for 8B
        "ft_batch_size": 4,
        "ft_epochs": 3,       # fewer epochs, larger model converges faster
        "ft_lr": 1e-5,
    },
    "phi35": {
        "name": "microsoft/Phi-3.5-mini-instruct",
        "hf_path": "/scratch/yl258/kp759/hf/models--microsoft--Phi-3.5-mini-instruct/snapshots/",
        "dtype": "float16",
        "batch_size": 4,
        "ft_batch_size": 4,
        "ft_epochs": 5,
        "ft_lr": 5e-5,
    },
}

# ── Split → retain mapping ─────────────────────────────────────────────────────
SPLIT_PAIRS = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}


def run_cmd(cmd: list, desc: str, log_file: Path = None):
    """Run a command, optionally logging output."""
    print(f"\n{'─'*60}")
    print(f"  {desc}")
    print(f"  CMD: {' '.join(cmd)}")
    print(f"{'─'*60}")

    start = time.time()
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "w") as f:
            result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        # Also print last 20 lines
        lines = open(log_file).readlines()
        print("".join(lines[-20:]))
    else:
        result = subprocess.run(cmd)

    elapsed = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"{status} Done in {elapsed:.1f}s (exit={result.returncode})")
    return result.returncode == 0


def write_temp_config(base_cfg: dict, model_key: str,
                      forget_split: str, out_dir: Path) -> Path:
    """Write a temporary config.yaml for this experiment."""
    model_info = MODEL_REGISTRY[model_key]
    retain_split = SPLIT_PAIRS[forget_split]

    cfg = {
        "model": {
            "name":       model_info["hf_path"],
            "device":     "cuda:0",
            "dtype":      model_info["dtype"],
            "max_length": 128,
        },
        "data": {
            "forget_split": forget_split,
            "retain_split": retain_split,
            "n_forget":     None,
            "n_retain":     None,
            "seed":         42,
        },
        "probing": {
            "layers":                None,
            "probe_type":            "logistic",
            "test_size":             0.2,
            "max_iter":              1000,
            "C":                     1.0,
            "token_probe_threshold": 0.70,
        },
        "extraction": {
            "batch_size": model_info["batch_size"],
            "aggregate":  "last",
        },
        "output": {
            "dir":                str(out_dir / "outputs"),
            "save_hidden_states": True,
            "save_probes":        True,
            "save_plots":         True,
        },
        "forget_entity": base_cfg.get("forget_entity", {}),
    }

    config_path = out_dir / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    return config_path


def run_experiment(
    model_key: str,
    forget_split: str,
    base_cfg: dict,
    base_output_dir: Path,
    method: str,
    threshold: float,
    skip_finetune: bool,
    skip_extraction: bool,
) -> dict:
    """
    Run the full pipeline for one (model, split) combination.
    Returns the evaluation results dict.
    """
    exp_name = f"{model_key}_{forget_split}"
    exp_dir  = base_output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_dir  = exp_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    model_info   = MODEL_REGISTRY[model_key]
    retain_split = SPLIT_PAIRS[forget_split]
    ft_dir       = exp_dir / "outputs" / "finetuned_model"

    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Model: {model_info['name']}")
    print(f"  Forget: {forget_split} | Retain: {retain_split}")
    print(f"{'='*60}")

    # ── Step 0: Write config ─────────────────────────────────────────────
    config_path = write_temp_config(base_cfg, model_key, forget_split, exp_dir)
    print(f"[Config] Written to {config_path}")

    # ── Step 1: Fine-tune ────────────────────────────────────────────────
    if not skip_finetune:
        ok = run_cmd(
            [sys.executable, "scripts/finetune_tofu.py",
             "--config", str(config_path),
             "--epochs",     str(model_info["ft_epochs"]),
             "--batch-size", str(model_info["ft_batch_size"]),
             "--lr",         str(model_info["ft_lr"]),
             "--output-dir", str(ft_dir),
             "--skip-verify"],
            desc=f"[1/6] Fine-tuning {model_key} on {forget_split}+{retain_split}",
            log_file=log_dir / "finetune.log",
        )
        if not ok:
            print(f"[ERROR] Fine-tuning failed for {exp_name}")
            return {"error": "finetune_failed"}
    else:
        print(f"[Skip] Fine-tuning (--skip-finetune)")

    # Update config to point to fine-tuned model
    if ft_dir.exists():
        # Find the actual saved model dir
        ft_model_path = ft_dir / "finetuned_model"
        if not ft_model_path.exists():
            ft_model_path = ft_dir
        config_path = write_temp_config_with_ft(
            base_cfg, model_key, forget_split, exp_dir,
            str(ft_model_path)
        )

    # ── Step 2: Extract hidden states ────────────────────────────────────
    if not skip_extraction:
        ok = run_cmd(
            [sys.executable, "scripts/extract_hidden_states.py",
             "--config", str(config_path)],
            desc=f"[2/6] Extracting hidden states",
            log_file=log_dir / "extract.log",
        )
        if not ok:
            print(f"[ERROR] Extraction failed for {exp_name}")
            return {"error": "extraction_failed"}

    # ── Step 3: Train probes ─────────────────────────────────────────────
    ok = run_cmd(
        [sys.executable, "scripts/train_probe.py",
         "--config", str(config_path)],
        desc=f"[3/6] Training layer-wise probes",
        log_file=log_dir / "train_probe.log",
    )
    if not ok:
        return {"error": "probe_training_failed"}

    # ── Step 4: Identify semantic tokens ─────────────────────────────────
    ok = run_cmd(
        [sys.executable, "scripts/identify_tokens.py",
         "--config",     str(config_path),
         "--best-layer", "6",
         "--threshold",  str(threshold)],
        desc=f"[4/6] Identifying semantic tokens (threshold={threshold})",
        log_file=log_dir / "identify.log",
    )
    if not ok:
        return {"error": "token_identification_failed"}

    # ── Step 5: Erase embeddings ─────────────────────────────────────────
    ok = run_cmd(
        [sys.executable, "scripts/erase_embeddings.py",
         "--config",  str(config_path),
         "--method",  method,
         "--skip-eval"],        # skip qualitative eval for speed
        desc=f"[5/6] Erasing embeddings (method={method})",
        log_file=log_dir / "erase.log",
    )
    if not ok:
        return {"error": "erasure_failed"}

    # ── Step 6: Evaluate ─────────────────────────────────────────────────
    unlearned_dir = exp_dir / "outputs" / f"unlearned_model_{method}"
    eval_out_dir  = exp_dir / "outputs" / "eval_results"
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    ok = run_cmd(
        [sys.executable, "scripts/tofu_eval.py",
         "--config",    str(config_path),
         "--model-dir", str(unlearned_dir),
         "--method",    f"{model_key}_{forget_split}_{method}"],
        desc=f"[6/6] Evaluating unlearned model",
        log_file=log_dir / "eval.log",
    )

    # Load and return results
    eval_file = eval_out_dir / f"eval_{model_key}_{forget_split}_{method}.json"
    if eval_file.exists():
        with open(eval_file) as f:
            results = json.load(f)
        results["exp_name"]     = exp_name
        results["model_key"]    = model_key
        results["forget_split"] = forget_split
        results["threshold"]    = threshold
        return results
    else:
        # Try alternate path
        alt = Path(f"outputs/eval_results/eval_{model_key}_{forget_split}_{method}.json")
        if alt.exists():
            with open(alt) as f:
                return json.load(f)
        return {"error": "eval_results_not_found", "exp_name": exp_name}


def write_temp_config_with_ft(base_cfg, model_key, forget_split,
                               out_dir, ft_model_path):
    """Re-write config pointing to fine-tuned model."""
    model_info   = MODEL_REGISTRY[model_key]
    retain_split = SPLIT_PAIRS[forget_split]

    cfg = {
        "model": {
            "name":       ft_model_path,
            "device":     "cuda:0",
            "dtype":      model_info["dtype"],
            "max_length": 128,
        },
        "data": {
            "forget_split": forget_split,
            "retain_split": retain_split,
            "n_forget":     None,
            "n_retain":     None,
            "seed":         42,
        },
        "probing": {
            "layers":                None,
            "probe_type":            "logistic",
            "test_size":             0.2,
            "max_iter":              1000,
            "C":                     1.0,
            "token_probe_threshold": 0.70,
        },
        "extraction": {
            "batch_size": model_info["batch_size"],
            "aggregate":  "last",
        },
        "output": {
            "dir":                str(out_dir / "outputs"),
            "save_hidden_states": True,
            "save_probes":        True,
            "save_plots":         False,   # skip plots for speed
        },
        "forget_entity": base_cfg.get("forget_entity", {}),
    }

    config_path = out_dir / "config_ft.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    return config_path


def print_scaling_table(all_results: list):
    """Print scaling results table for the paper."""
    print(f"\n{'='*90}")
    print("  SCALING RESULTS TABLE")
    print(f"{'='*90}")
    header = (f"{'Experiment':<25} {'|Tf|':>6}  "
              f"{'Forget↓':>9}  {'Retain↑':>9}  "
              f"{'PPL↓':>10}  {'TruthR':>8}  {'Score↑':>9}")
    print(header)
    print("-"*90)

    for r in all_results:
        if "error" in r:
            print(f"  {r.get('exp_name','?'):<25} ERROR: {r['error']}")
            continue
        row = r.get("table_row", {})
        exp = r.get("exp_name", r.get("method", "?"))
        print(
            f"  {exp:<25} "
            f"{'?':>6}  "
            f"{row.get('Forget Conf ↓','?'):>9}  "
            f"{row.get('Retain Conf ↑','?'):>9}  "
            f"{row.get('Perplexity ↓','?'):>10}  "
            f"{row.get('Truth Ratio →0.5','?'):>8}  "
            f"{row.get('Forget Score ↑','?'):>9}"
        )
    print("="*90)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",          default="config/config.yaml")
    parser.add_argument("--splits",          nargs="+",
                        default=["forget01", "forget05", "forget10"],
                        choices=["forget01", "forget05", "forget10"])
    parser.add_argument("--models",          nargs="+",
                        default=["llama1b"],
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--method",          default="zero",
                        choices=["zero", "noise", "mean"])
    parser.add_argument("--threshold",       type=float, default=0.999,
                        help="Token identification threshold. Default=0.999 (name-only, best config)")
    parser.add_argument("--output-dir",      default="scaling_experiments")
    parser.add_argument("--skip-finetune",   action="store_true",
                        help="Skip fine-tuning if already done")
    parser.add_argument("--skip-extraction", action="store_true",
                        help="Skip hidden state extraction if already done")
    args = parser.parse_args()

    with open(args.config) as f:
        base_cfg = yaml.safe_load(f)

    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Track all results
    all_results = []
    total       = len(args.models) * len(args.splits)
    done        = 0

    print(f"\n{'='*60}")
    print(f"  SCALING EXPERIMENTS")
    print(f"  Models: {args.models}")
    print(f"  Splits: {args.splits}")
    print(f"  Method: {args.method} | Threshold: {args.threshold}")
    print(f"  Total experiments: {total}")
    print(f"{'='*60}")

    for model_key in args.models:
        for forget_split in args.splits:
            done += 1
            print(f"\n[{done}/{total}] Starting: {model_key} × {forget_split}")

            result = run_experiment(
                model_key=model_key,
                forget_split=forget_split,
                base_cfg=base_cfg,
                base_output_dir=base_output_dir,
                method=args.method,
                threshold=args.threshold,
                skip_finetune=args.skip_finetune,
                skip_extraction=args.skip_extraction,
            )
            all_results.append(result)

            # Save intermediate results
            interim_path = base_output_dir / "results_interim.json"
            with open(interim_path, "w") as f:
                json.dump(all_results, f, indent=2)
            print(f"[Saved] Interim results → {interim_path}")

    # Print final table
    print_scaling_table(all_results)

    # Save final results
    final_path = base_output_dir / "results_final.json"
    with open(final_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Final results saved to {final_path}")


if __name__ == "__main__":
    main()