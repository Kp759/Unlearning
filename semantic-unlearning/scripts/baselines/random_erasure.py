"""
scripts/baselines/random_erasure.py
-------------------------------------
Random Token Erasure baseline.

This is the SANITY CHECK baseline that answers:
  "Does our semantic token identification actually matter,
   or would erasing ANY 19 random tokens work just as well?"

If random erasure performs WORSE than our method:
  → Our probe-based token selection is meaningful ✅
  → The WHICH tokens you erase matters, not just HOW MANY

If random erasure performs SIMILARLY:
  → Our method has no advantage over random selection ❌
  → Need to rethink the approach

Three variants:
  random_vocab:   19 random tokens from full vocabulary (128K)
  random_common:  19 random tokens from most frequent vocab (top 1000)
  random_rare:    19 random tokens from rare vocab (bottom 10K)

We run each variant N_TRIALS times and report mean ± std
to account for variance in random selection.

Run:
    python scripts/baselines/random_erasure.py \
        --config config/config.yaml \
        --n-tokens 19 \
        --n-trials 5 \
        --variants random_vocab random_common random_rare
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Our T_f for reference ─────────────────────────────────────────────────────
def load_our_token_ids(tokens_file: str) -> list:
    with open(tokens_file) as f:
        data = json.load(f)
    return data["token_ids"]


# ── Random token selection ────────────────────────────────────────────────────
def get_random_tokens(variant: str, n_tokens: int,
                      vocab_size: int, seed: int,
                      our_token_ids: list) -> list:
    """
    Select n_tokens random token IDs based on variant.
    Excludes special tokens and our actual T_f tokens
    (to ensure fair comparison).
    """
    rng = random.Random(seed)

    # Special tokens to exclude (BOS, EOS, PAD, etc.)
    special_ids = set(range(128000, vocab_size))  # LLaMA special tokens
    exclude     = set(our_token_ids) | special_ids

    if variant == "random_vocab":
        # Random from entire vocabulary
        candidates = [i for i in range(vocab_size) if i not in exclude]

    elif variant == "random_common":
        # Random from most common tokens (roughly IDs 0-1000)
        # Lower token IDs in BPE = more frequent in training data
        candidates = [i for i in range(0, 1000) if i not in exclude]

    elif variant == "random_rare":
        # Random from less common tokens (IDs 50000-100000)
        candidates = [i for i in range(50000, 100000) if i not in exclude]

    else:
        raise ValueError(f"Unknown variant: {variant}")

    selected = rng.sample(candidates, min(n_tokens, len(candidates)))
    return selected


# ── Erasure and evaluation ────────────────────────────────────────────────────
def erase_and_evaluate(
    model_name: str,
    token_ids: list,
    forget_samples: list,
    retain_samples: list,
    device: str,
    dtype: str,
    max_length: int,
    n_retain: int = 200,
) -> dict:
    """
    Load fresh model, erase token_ids, evaluate.
    Loads fresh model each time to avoid state contamination.
    """
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    if device == "cpu": torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token       = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Erase
    embed = model.model.embed_tokens.weight.data
    embed[token_ids] = 0.0
    if hasattr(model, "lm_head"):
        lm_w = model.lm_head.weight.data
        if lm_w.data_ptr() != embed.data_ptr():
            lm_w[token_ids] = 0.0

    # Evaluate
    rng = random.Random(42)
    retain_sub = rng.sample(retain_samples, min(n_retain, len(retain_samples)))

    @torch.no_grad()
    def log_prob(question, answer):
        full  = f"Question: {question} Answer: {answer}"
        q_txt = f"Question: {question} Answer:"
        fe    = tokenizer(full,  return_tensors="pt",
                          truncation=True, max_length=max_length).to(device)
        qe    = tokenizer(q_txt, return_tensors="pt",
                          truncation=True, max_length=max_length).to(device)
        q_len = qe["input_ids"].shape[1]
        out   = model(**fe)
        lp    = torch.log_softmax(out.logits, dim=-1)
        ids   = fe["input_ids"][0]
        a_ids = ids[q_len:]
        a_lp  = lp[0, q_len-1:-1, :]
        if a_ids.shape[0] == 0: return float("-inf")
        return a_lp[torch.arange(a_ids.shape[0]), a_ids].mean().item()

    f_lps = [log_prob(s["question"], s["answer"]) for s in forget_samples]
    r_lps = [log_prob(s["question"], s["answer"]) for s in retain_sub]

    mean_f = sum(f_lps) / len(f_lps)
    mean_r = sum(r_lps) / len(r_lps)
    f_conf = math.exp(mean_f) if mean_f > -50 else 0.0
    r_conf = math.exp(mean_r) if mean_r > -50 else 0.0
    fq     = 1.0 - f_conf
    score  = 2 * fq * r_conf / (fq + r_conf) if fq + r_conf > 0 else 0.0

    # Free memory
    del model
    torch.cuda.empty_cache()

    return {
        "forget_conf":  f_conf,
        "retain_conf":  r_conf,
        "forget_score": score,
        "token_ids":    token_ids,
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--tokens-file", default="outputs/semantic_tokens.json",
                        help="Our T_f tokens file (for n_tokens reference)")
    parser.add_argument("--n-tokens",    type=int, default=None,
                        help="Number of tokens to erase. None=same as our T_f")
    parser.add_argument("--n-trials",    type=int, default=5,
                        help="Number of random trials per variant")
    parser.add_argument("--variants",    nargs="+",
                        default=["random_vocab", "random_common", "random_rare"],
                        choices=["random_vocab", "random_common", "random_rare"])
    parser.add_argument("--output-dir",  default="outputs/baseline_random")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = cfg["model"]["device"]
    dtype      = cfg["model"]["dtype"]
    max_length = cfg["model"]["max_length"]
    model_name = cfg["model"]["name"]
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load our token IDs
    our_token_ids = load_our_token_ids(args.tokens_file)
    n_tokens      = args.n_tokens or len(our_token_ids)
    print(f"\n[Random] Comparing against our T_f of {len(our_token_ids)} tokens")
    print(f"[Random] Will erase {n_tokens} random tokens per trial")

    # Load vocab size
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    vocab_size = tokenizer.vocab_size
    del tokenizer

    # Load TOFU data once
    print("[Random] Loading TOFU data...")
    forget_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["forget_split"], split="train")
    retain_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["retain_split"], split="train")
    forget_samples = list(forget_ds)
    retain_samples = list(retain_ds)

    all_results = {}

    # ── Run each variant ──────────────────────────────────────────────────
    for variant in args.variants:
        print(f"\n{'='*55}")
        print(f"  Variant: {variant} ({args.n_trials} trials)")
        print(f"{'='*55}")

        trial_results = []

        for trial in range(args.n_trials):
            seed        = trial * 100 + 42
            random_ids  = get_random_tokens(
                variant, n_tokens, vocab_size, seed, our_token_ids
            )

            print(f"\n  Trial {trial+1}/{args.n_trials} "
                  f"(seed={seed}, tokens={random_ids[:5]}...)")

            r = erase_and_evaluate(
                model_name, random_ids,
                forget_samples, retain_samples,
                device, dtype, max_length,
            )
            trial_results.append(r)
            print(f"    forget={r['forget_conf']:.4f} "
                  f"retain={r['retain_conf']:.4f} "
                  f"score={r['forget_score']:.4f}")

        # Aggregate across trials
        f_confs = [r["forget_conf"]  for r in trial_results]
        r_confs = [r["retain_conf"]  for r in trial_results]
        scores  = [r["forget_score"] for r in trial_results]

        summary = {
            "variant":         variant,
            "n_tokens":        n_tokens,
            "n_trials":        args.n_trials,
            "forget_conf_mean": float(np.mean(f_confs)),
            "forget_conf_std":  float(np.std(f_confs)),
            "retain_conf_mean": float(np.mean(r_confs)),
            "retain_conf_std":  float(np.std(r_confs)),
            "score_mean":       float(np.mean(scores)),
            "score_std":        float(np.std(scores)),
            "trials":           trial_results,
        }
        all_results[variant] = summary

        print(f"\n  {variant} Summary:")
        print(f"    Forget: {summary['forget_conf_mean']:.4f} ± {summary['forget_conf_std']:.4f}")
        print(f"    Retain: {summary['retain_conf_mean']:.4f} ± {summary['retain_conf_std']:.4f}")
        print(f"    Score:  {summary['score_mean']:.4f} ± {summary['score_std']:.4f}")

    # ── Final comparison table ─────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  RANDOM ERASURE vs OURS — COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Method':<30} {'Forget↓':>9}  {'Retain↑':>9}  {'Score↑':>9}")
    print(f"  {'-'*68}")

    # Our method (from previous eval)
    our_eval_path = Path("outputs/eval_results/eval_zero_name_only_8tokens.json")
    if our_eval_path.exists():
        with open(our_eval_path) as f:
            our_eval = json.load(f)
        row = our_eval["table_row"]
        print(f"  {'Ours (semantic, 19 tok)':<30} "
              f"{row['Forget Conf ↓']:>9}  "
              f"{row['Retain Conf ↑']:>9}  "
              f"{row['Forget Score ↑']:>9}")

    for variant, r in all_results.items():
        print(
            f"  {variant:<30} "
            f"{r['forget_conf_mean']:>8.4f}±{r['forget_conf_std']:.2f}  "
            f"{r['retain_conf_mean']:>8.4f}±{r['retain_conf_std']:.2f}  "
            f"{r['score_mean']:>8.4f}±{r['score_std']:.2f}"
        )
    print(f"{'='*70}")

    print("\nKey question:")
    print("  Is Ours Score >> Random Score?")
    print("  If YES → probe-based selection is meaningful ✅")
    print("  If NO  → random erasure works just as well ❌")

    # Save
    save_path = out_dir / "random_erasure_results.json"
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[✓] Saved to {save_path}")


if __name__ == "__main__":
    main()