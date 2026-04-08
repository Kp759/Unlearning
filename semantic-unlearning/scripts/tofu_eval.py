"""
scripts/tofu_eval.py
---------------------
Formal TOFU evaluation: computes standard unlearning metrics.

Metrics:
    Forget Quality  (FQ)  — how well the model forgot D_f
                            measured via probability on forget QA
                            LOWER = better forgotten

    Retain Quality  (RQ)  — how well the model preserved D_r
                            measured via probability on retain QA
                            HIGHER = better preserved

    Model Utility   (MU)  — general language ability preserved
                            measured via perplexity on WikiText-2
                            LOWER perplexity = better utility

    Forget Score    (FS)  — composite: harmonic mean of FQ and RQ
                            HIGHER = better overall unlearning

Also computes:
    Truth Ratio     (TR)  — P(correct answer) / P(incorrect answer)
                            on forget set → should be < 1 after unlearning

Run:
    python scripts/tofu_eval.py --config config/config.yaml \
        --model-dir outputs/unlearned_model_zero --method zero

    # Compare all three methods:
    python scripts/tofu_eval.py --config config/config.yaml --compare-all
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_path: str, device: str, dtype: str):
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    if device == "cpu":
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


@torch.no_grad()
def compute_answer_probability(model, tokenizer, question: str, answer: str,
                                max_length: int = 128, device: str = "cuda") -> float:
    """
    Compute P(answer | question) using teacher-forced log-likelihood.
    Returns mean token log-probability of the answer tokens.
    Higher = model more confident in this answer.
    """
    full_text   = f"Question: {question} Answer: {answer}"
    q_text      = f"Question: {question} Answer:"

    full_enc = tokenizer(full_text, return_tensors="pt",
                         truncation=True, max_length=max_length).to(device)
    q_enc    = tokenizer(q_text,    return_tensors="pt",
                         truncation=True, max_length=max_length).to(device)

    q_len = q_enc["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model(**full_enc, labels=full_enc["input_ids"])
        # Get per-token log-probs
        logits = outputs.logits                          # (1, seq, vocab)
        log_probs = torch.log_softmax(logits, dim=-1)   # (1, seq, vocab)

        # Shift: logits[t] predicts token[t+1]
        input_ids   = full_enc["input_ids"][0]          # (seq,)
        answer_ids  = input_ids[q_len:]                 # answer token ids
        answer_logp = log_probs[0, q_len-1:-1, :]       # (n_answer_tokens, vocab)

        if answer_ids.shape[0] == 0:
            return float("-inf")

        # Gather log-probs for the actual answer tokens
        token_logps = answer_logp[
            torch.arange(answer_ids.shape[0]), answer_ids
        ]                                                # (n_answer_tokens,)

        return token_logps.mean().item()


@torch.no_grad()
def compute_perplexity(model, tokenizer, texts: list, batch_size: int = 4,
                       max_length: int = 128, device: str = "cuda") -> float:
    """
    Compute average perplexity over a list of texts.
    Lower = model generates text more fluently.
    """
    total_nll  = 0.0
    total_toks = 0

    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        enc   = tokenizer(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=max_length).to(device)
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100   # ignore padding

        outputs   = model(**enc, labels=labels)
        loss      = outputs.loss                    # mean NLL over non-ignored tokens
        n_tokens  = (labels != -100).sum().item()

        total_nll  += loss.item() * n_tokens
        total_toks += n_tokens

    return math.exp(total_nll / total_toks) if total_toks > 0 else float("inf")


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_forget_quality(model, tokenizer, forget_samples: list,
                           max_length: int, device: str) -> dict:
    """
    Forget Quality (FQ):
        Mean answer log-prob on forget set.
        LOWER = model less confident = better forgetting.

    Also computes Truth Ratio:
        P(correct) / P(paraphrase_correct)
        We approximate with: mean_logp on forget set (relative to retain)
    """
    log_probs = []
    for s in tqdm(forget_samples, desc="  Forget Quality"):
        lp = compute_answer_probability(
            model, tokenizer, s["question"], s["answer"],
            max_length=max_length, device=device
        )
        log_probs.append(lp)

    mean_lp = sum(log_probs) / len(log_probs)
    # Convert to probability-space score: exp(mean_log_prob)
    # Normalized to [0,1]: higher = model more confident
    # We report as "forget_score" where LOWER is better
    forget_confidence = math.exp(mean_lp) if mean_lp > -50 else 0.0

    return {
        "mean_log_prob":      mean_lp,
        "forget_confidence":  forget_confidence,   # lower = better
        "n_samples":          len(log_probs),
    }


def compute_retain_quality(model, tokenizer, retain_samples: list,
                           max_length: int, device: str) -> dict:
    """
    Retain Quality (RQ):
        Mean answer log-prob on retain set.
        HIGHER = model still knows retain authors = better preservation.
    """
    log_probs = []
    # Sample 200 retain QA pairs (computing all 3960 is slow)
    import random
    sample = random.sample(retain_samples, min(200, len(retain_samples)))

    for s in tqdm(sample, desc="  Retain Quality"):
        lp = compute_answer_probability(
            model, tokenizer, s["question"], s["answer"],
            max_length=max_length, device=device
        )
        log_probs.append(lp)

    mean_lp = sum(log_probs) / len(log_probs)
    retain_confidence = math.exp(mean_lp) if mean_lp > -50 else 0.0

    return {
        "mean_log_prob":      mean_lp,
        "retain_confidence":  retain_confidence,   # higher = better
        "n_samples":          len(log_probs),
    }


def compute_model_utility(model, tokenizer, device: str,
                          n_samples: int = 200) -> dict:
    """
    Model Utility (MU):
        Perplexity on WikiText-2 validation set.
        LOWER perplexity = model still generates fluent text = better utility.
    """
    print("  Loading WikiText-2...")
    try:
        wiki = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [x["text"] for x in wiki if len(x["text"].strip()) > 50][:n_samples]
    except Exception:
        # Fallback: use general knowledge sentences
        texts = [
            "The capital of France is Paris, a city known for the Eiffel Tower.",
            "Machine learning is a subfield of artificial intelligence.",
            "The speed of light in vacuum is approximately 299,792 km/s.",
            "William Shakespeare wrote Hamlet, Macbeth, and King Lear.",
            "The human genome contains approximately 3 billion base pairs.",
        ] * 20
        print("  [Warning] WikiText unavailable, using fallback sentences.")

    ppl = compute_perplexity(model, tokenizer, texts, device=device)

    return {
        "perplexity": ppl,   # lower = better
        "n_samples":  len(texts),
    }


def compute_truth_ratio(model, tokenizer, forget_samples: list,
                        max_length: int, device: str) -> dict:
    """
    Truth Ratio (TR):
        For each forget QA, compare P(correct answer) vs P(wrong answer).
        TR = mean[ P(correct) / (P(correct) + P(wrong)) ]
        After unlearning: TR should approach 0.5 (random guessing).

        We generate a simple wrong answer by using a generic response.
    """
    wrong_answers = [
        "I don't know.",
        "There is no information available.",
        "This is unknown.",
    ]
    import random

    ratios = []
    for s in tqdm(forget_samples, desc="  Truth Ratio"):
        lp_correct = compute_answer_probability(
            model, tokenizer, s["question"], s["answer"],
            max_length=max_length, device=device
        )
        wrong = random.choice(wrong_answers)
        lp_wrong = compute_answer_probability(
            model, tokenizer, s["question"], wrong,
            max_length=max_length, device=device
        )
        # Softmax between correct and wrong
        max_lp = max(lp_correct, lp_wrong)
        p_correct = math.exp(lp_correct - max_lp)
        p_wrong   = math.exp(lp_wrong   - max_lp)
        ratio     = p_correct / (p_correct + p_wrong + 1e-8)
        ratios.append(ratio)

    mean_tr = sum(ratios) / len(ratios)
    # TR = 0.5 → random (perfect unlearning)
    # TR = 1.0 → model fully confident in correct answer (no unlearning)
    # TR = 0.0 → model prefers wrong answer (over-unlearning)

    return {
        "truth_ratio":  mean_tr,
        "n_samples":    len(ratios),
    }


def compute_forget_score(fq: dict, rq: dict) -> float:
    """
    Composite Forget Score:
        Harmonic mean of forget quality (inverted) and retain quality.
        forget_quality_norm = 1 - forget_confidence  (higher = better forgotten)
        retain_quality_norm = retain_confidence       (higher = better preserved)

        FS = harmonic_mean(forget_quality_norm, retain_quality_norm)
        Range [0, 1]. Higher = better overall unlearning.
    """
    fq_norm = 1.0 - fq["forget_confidence"]   # invert: lower confidence = better
    rq_norm = rq["retain_confidence"]

    if fq_norm + rq_norm == 0:
        return 0.0
    return 2 * fq_norm * rq_norm / (fq_norm + rq_norm)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_model(model_path: str, cfg: dict, method: str,
                   out_dir: Path) -> dict:
    """Run full evaluation on one model."""
    device     = cfg["model"]["device"]
    dtype      = cfg["model"]["dtype"]
    max_length = cfg["model"]["max_length"]

    print(f"\n{'='*60}")
    print(f"  Evaluating: {method.upper()}")
    print(f"  Model: {model_path}")
    print(f"{'='*60}")

    # Load model
    print("\n[1] Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_path, device, dtype)

    # Load TOFU data
    print("[2] Loading TOFU data...")
    forget_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["forget_split"], split="train")
    retain_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["retain_split"], split="train")
    forget_samples = list(forget_ds)
    retain_samples = list(retain_ds)

    print(f"    Forget: {len(forget_samples)} | Retain: {len(retain_samples)}")

    # Compute metrics
    print("\n[3] Computing metrics...")

    print("\n  → Forget Quality (lower = better forgetting):")
    fq = compute_forget_quality(model, tokenizer, forget_samples,
                                max_length, device)

    print("\n  → Retain Quality (higher = better preservation):")
    rq = compute_retain_quality(model, tokenizer, retain_samples,
                                max_length, device)

    print("\n  → Model Utility / Perplexity (lower = better):")
    mu = compute_model_utility(model, tokenizer, device)

    print("\n  → Truth Ratio (closer to 0.5 = better unlearning):")
    tr = compute_truth_ratio(model, tokenizer, forget_samples,
                             max_length, device)

    fs = compute_forget_score(fq, rq)

    results = {
        "method":         method,
        "model_path":     model_path,
        "forget_quality": fq,
        "retain_quality": rq,
        "model_utility":  mu,
        "truth_ratio":    tr,
        "forget_score":   fs,

        # Summary row for paper table
        "table_row": {
            "Method":          method,
            "Forget Conf ↓":   f"{fq['forget_confidence']:.4f}",
            "Retain Conf ↑":   f"{rq['retain_confidence']:.4f}",
            "Perplexity ↓":    f"{mu['perplexity']:.2f}",
            "Truth Ratio →0.5":f"{tr['truth_ratio']:.4f}",
            "Forget Score ↑":  f"{fs:.4f}",
        }
    }

    # Save
    save_path = out_dir / f"eval_{method}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved to {save_path}")

    return results


def print_table(all_results: list):
    """Print paper-ready results table."""
    print("\n" + "="*80)
    print("  RESULTS TABLE (for paper)")
    print("="*80)
    header = f"{'Method':<15} {'Forget Conf↓':>14} {'Retain Conf↑':>14} "
    header += f"{'Perplexity↓':>12} {'Truth Ratio':>12} {'Forget Score↑':>14}"
    print(header)
    print("-"*80)
    for r in all_results:
        row = r["table_row"]
        print(
            f"{row['Method']:<15} "
            f"{row['Forget Conf ↓']:>14} "
            f"{row['Retain Conf ↑']:>14} "
            f"{row['Perplexity ↓']:>12} "
            f"{row['Truth Ratio →0.5']:>12} "
            f"{row['Forget Score ↑']:>14}"
        )
    print("="*80)
    print("\nMetric guide:")
    print("  Forget Conf ↓  : lower = model less confident about forget authors")
    print("  Retain Conf ↑  : higher = model still knows retain authors")
    print("  Perplexity ↓   : lower = model still generates fluent text")
    print("  Truth Ratio    : closer to 0.5 = model guessing on forget QA")
    print("  Forget Score ↑ : composite harmonic mean (higher = better)")


def main():
    parser = argparse.ArgumentParser(description="Evaluate unlearned models on TOFU metrics.")
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--model-dir",   default=None,
                        help="Path to specific unlearned model to evaluate.")
    parser.add_argument("--method",      default="zero",
                        help="Method name label for this model.")
    parser.add_argument("--compare-all", action="store_true",
                        help="Evaluate all three methods + finetuned baseline.")
    parser.add_argument("--base-model",  action="store_true",
                        help="Also evaluate the fine-tuned model as baseline.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"]) / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    if args.compare_all:
        # Evaluate fine-tuned baseline + all three unlearned models
        models_to_eval = [
            (cfg["model"]["name"],                          "finetuned_baseline"),
            ("outputs/unlearned_model_zero",                "zero"),
            ("outputs/unlearned_model_noise",               "noise"),
            ("outputs/unlearned_model_mean",                "mean"),
        ]
        # Use absolute paths
        base = Path(cfg["output"]["dir"]).parent
        models_to_eval = [
            (str(Path(p).resolve()) if not Path(p).is_absolute() else p, m)
            for p, m in models_to_eval
        ]
    else:
        if args.model_dir is None:
            raise ValueError("Provide --model-dir or use --compare-all")
        models_to_eval = [(args.model_dir, args.method)]

        if args.base_model:
            models_to_eval = [(cfg["model"]["name"], "finetuned_baseline")] \
                           + models_to_eval

    for model_path, method in models_to_eval:
        if not Path(model_path).exists():
            print(f"\n[Skip] {model_path} does not exist — skipping {method}")
            continue
        result = evaluate_model(model_path, cfg, method, out_dir)
        all_results.append(result)

    if len(all_results) > 1:
        print_table(all_results)

        # Save combined table
        combined_path = out_dir / "results_table.json"
        with open(combined_path, "w") as f:
            json.dump([r["table_row"] for r in all_results], f, indent=2)
        print(f"\n[✓] Combined table saved to {combined_path}")

    elif all_results:
        r = all_results[0]
        print(f"\n{'='*50}")
        print(f"  Results for: {r['method']}")
        print(f"{'='*50}")
        for k, v in r["table_row"].items():
            print(f"  {k:<20}: {v}")


if __name__ == "__main__":
    main()