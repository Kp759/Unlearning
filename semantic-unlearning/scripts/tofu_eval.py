""" 
scripts/tofu_eval.py
---------------------
Formal TOFU evaluation: computes standard unlearning metrics.

Metrics:
    Forget Quality  (FQ)  — how well the model forgot D_f (answer log-prob)
    Retain Quality  (RQ)  — how well the model preserved D_r (answer log-prob)
    Retain ROUGE    (RR)  — ROUGE-L of generated answers on retain set
    Forget Score    (FS)  — composite harmonic mean of FQ and RQ
    Truth Ratio     (TR)  — P(correct) / P(incorrect) on forget set

Run:
    # Single model eval
    python scripts/tofu_eval.py --config config/config.yaml \
        --model-dir outputs/unlearned_model_zero --method zero

    # With baseline comparison
    python scripts/tofu_eval.py --config config/config.yaml \
        --model-dir outputs/unlearned_model_zero --method zero --base-model

    # Compare all methods
    python scripts/tofu_eval.py --config config/config.yaml --compare-all

    # Override splits without editing config
    python scripts/tofu_eval.py --config config/config.yaml \
        --model-dir outputs/unlearned_model_zero --method zero \
        --forget-split forget01 --retain-split retain99
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


def format_prompt(question: str, answer: str, tokenizer) -> tuple:
    """
    Auto-detects Instruct vs base model via tokenizer.chat_template.
    Returns (full_text, q_text).
    """
    if tokenizer.chat_template is not None:
        msg = [{'role': 'user', 'content': f'Question: {question} Answer:'}]
        q_text = tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
        full_text = q_text + f" {answer}"
    else:
        q_text    = f"Question: {question} Answer:"
        full_text = f"Question: {question} Answer: {answer}"
    return full_text, q_text


def format_question(question: str, tokenizer) -> str:
    """Format question-only prompt for generation."""
    if tokenizer.chat_template is not None:
        msg = [{'role': 'user', 'content': f'Question: {question} Answer:'}]
        return tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True
        )
    return f"Question: {question} Answer:"


@torch.no_grad()
def compute_answer_probability(model, tokenizer, question: str, answer: str,
                                max_length: int = 256, device: str = "cuda") -> float:
    """
    Compute P(answer | question) using teacher-forced log-likelihood.
    Returns mean token log-probability of answer tokens.
    Higher = model more confident in this answer.
    """
    full_text, q_text = format_prompt(question, answer, tokenizer)

    full_enc = tokenizer(full_text, return_tensors="pt",
                         truncation=True, max_length=max_length).to(device)
    q_enc    = tokenizer(q_text, return_tensors="pt",
                         truncation=True, max_length=max_length).to(device)

    q_len = q_enc["input_ids"].shape[1]

    outputs   = model(**full_enc, labels=full_enc["input_ids"])
    logits    = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    input_ids   = full_enc["input_ids"][0]
    answer_ids  = input_ids[q_len:]
    answer_logp = log_probs[0, q_len-1:-1, :]

    if answer_ids.shape[0] == 0:
        return float("-inf")

    token_logps = answer_logp[torch.arange(answer_ids.shape[0]), answer_ids]
    return token_logps.mean().item()


@torch.no_grad()
def generate_answer(model, tokenizer, question: str,
                    max_new_tokens: int = 64, device: str = "cuda") -> str:
    """Generate answer for a question."""
    prompt = format_question(question, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt",
                       truncation=True, max_length=256).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between prediction and reference."""
    pred_tokens = prediction.lower().split()
    ref_tokens  = reference.lower().split()

    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS length
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i-1] == pred_tokens[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]

    precision = lcs / n if n > 0 else 0.0
    recall    = lcs / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Core metrics ──────────────────────────────────────────────────────────────

def compute_forget_quality(model, tokenizer, forget_samples: list,
                           max_length: int, device: str) -> dict:
    log_probs = []
    for s in tqdm(forget_samples, desc="  Forget Quality"):
        lp = compute_answer_probability(
            model, tokenizer, s["question"], s["answer"],
            max_length=max_length, device=device
        )
        log_probs.append(lp)

    mean_lp           = sum(log_probs) / len(log_probs)
    forget_confidence = math.exp(mean_lp) if mean_lp > -50 else 0.0

    return {
        "mean_log_prob":     mean_lp,
        "forget_confidence": forget_confidence,
        "n_samples":         len(log_probs),
    }


def compute_retain_quality(model, tokenizer, retain_samples: list,
                           max_length: int, device: str,
                           n_samples: int = 200) -> dict:
    import random
    sample = random.sample(retain_samples, min(n_samples, len(retain_samples)))

    log_probs = []
    for s in tqdm(sample, desc="  Retain Quality"):
        lp = compute_answer_probability(
            model, tokenizer, s["question"], s["answer"],
            max_length=max_length, device=device
        )
        log_probs.append(lp)

    mean_lp           = sum(log_probs) / len(log_probs)
    retain_confidence = math.exp(mean_lp) if mean_lp > -50 else 0.0

    return {
        "mean_log_prob":      mean_lp,
        "retain_confidence":  retain_confidence,
        "n_samples":          len(log_probs),
    }


def compute_retain_rouge(model, tokenizer, retain_samples: list,
                         device: str, n_samples: int = 100) -> dict:
    """
    ROUGE-L of generated answers on retain set.
    Measures whether model still generates correct retain answers.
    Higher = better preservation of retain knowledge.
    Matches open-unlearning's model utility metric.
    """
    import random
    sample = random.sample(retain_samples, min(n_samples, len(retain_samples)))

    scores = []
    for s in tqdm(sample, desc="  Retain ROUGE"):
        pred = generate_answer(model, tokenizer, s["question"], device=device)
        r    = rouge_l(pred, s["answer"])
        scores.append(r)

    mean_rouge = sum(scores) / len(scores)
    return {
        "retain_rouge_l": mean_rouge,
        "n_samples":      len(scores),
    }


def compute_truth_ratio(model, tokenizer, forget_samples: list,
                        max_length: int, device: str) -> dict:
    import random
    wrong_answers = [
        "I don't know.",
        "There is no information available.",
        "This is unknown.",
    ]

    ratios = []
    for s in tqdm(forget_samples, desc="  Truth Ratio"):
        lp_correct = compute_answer_probability(
            model, tokenizer, s["question"], s["answer"],
            max_length=max_length, device=device
        )
        wrong    = random.choice(wrong_answers)
        lp_wrong = compute_answer_probability(
            model, tokenizer, s["question"], wrong,
            max_length=max_length, device=device
        )
        max_lp    = max(lp_correct, lp_wrong)
        p_correct = math.exp(lp_correct - max_lp)
        p_wrong   = math.exp(lp_wrong   - max_lp)
        ratio     = p_correct / (p_correct + p_wrong + 1e-8)
        ratios.append(ratio)

    return {"truth_ratio": sum(ratios) / len(ratios), "n_samples": len(ratios)}


def compute_forget_score(fq: dict, rq: dict) -> float:
    fq_norm = 1.0 - fq["forget_confidence"]
    rq_norm = rq["retain_confidence"]
    if fq_norm + rq_norm == 0:
        return 0.0
    return 2 * fq_norm * rq_norm / (fq_norm + rq_norm)


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate_model(model_path: str, cfg: dict, method: str,
                   out_dir: Path,
                   forget_split: str = None,
                   retain_split: str = None) -> dict:
    device     = cfg["model"]["device"]
    dtype      = cfg["model"]["dtype"]
    max_length = cfg["model"]["max_length"]

    forget_split = forget_split or cfg["data"]["forget_split"]
    retain_split = retain_split or cfg["data"]["retain_split"]

    print(f"\n{'='*60}")
    print(f"  Evaluating : {method.upper()}")
    print(f"  Model      : {model_path}")
    print(f"  Splits     : {forget_split} / {retain_split}")
    print(f"{'='*60}")

    print("\n[1] Loading model...")
    model, tokenizer = load_model_and_tokenizer(model_path, device, dtype)

    if tokenizer.chat_template is not None:
        print("  [✓] Chat template detected (Instruct model)")
    else:
        print("  [✓] No chat template (base model)")

    print("[2] Loading TOFU data...")
    forget_ds      = load_dataset("locuslab/TOFU", forget_split, split="train")
    retain_ds      = load_dataset("locuslab/TOFU", retain_split, split="train")
    forget_samples = list(forget_ds)
    retain_samples = list(retain_ds)
    print(f"    Forget: {len(forget_samples)} | Retain: {len(retain_samples)}")

    print("\n[3] Computing metrics...")

    print("\n  → Forget Quality (lower = better forgetting):")
    fq = compute_forget_quality(model, tokenizer, forget_samples,
                                max_length, device)

    print("\n  → Retain Quality (higher = better preservation):")
    rq = compute_retain_quality(model, tokenizer, retain_samples,
                                max_length, device)

    print("\n  → Retain ROUGE-L (higher = better retain generation):")
    rr = compute_retain_rouge(model, tokenizer, retain_samples, device)

    print("\n  → Truth Ratio (closer to 0.5 = better unlearning):")
    tr = compute_truth_ratio(model, tokenizer, forget_samples,
                             max_length, device)

    fs = compute_forget_score(fq, rq)

    results = {
        "method":         method,
        "model_path":     model_path,
        "forget_split":   forget_split,
        "retain_split":   retain_split,
        "forget_quality": fq,
        "retain_quality": rq,
        "retain_rouge":   rr,
        "truth_ratio":    tr,
        "forget_score":   fs,
        "table_row": {
            "Method":           method,
            "Forget Conf ↓":    f"{fq['forget_confidence']:.4f}",
            "Retain Conf ↑":    f"{rq['retain_confidence']:.4f}",
            "Retain ROUGE ↑":   f"{rr['retain_rouge_l']:.4f}",
            "Truth Ratio →0.5": f"{tr['truth_ratio']:.4f}",
            "Forget Score ↑":   f"{fs:.4f}",
        }
    }

    save_path = out_dir / f"eval_{method}.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved to {save_path}")

    return results


def print_table(all_results: list):
    print("\n" + "="*90)
    print("  RESULTS TABLE (for paper)")
    print("="*90)
    header  = f"{'Method':<22} {'Forget Conf↓':>13} {'Retain Conf↑':>13} "
    header += f"{'Retain ROUGE↑':>14} {'Truth Ratio':>12} {'Forget Score↑':>14}"
    print(header)
    print("-"*90)
    for r in all_results:
        row = r["table_row"]
        print(
            f"{row['Method']:<22} "
            f"{row['Forget Conf ↓']:>13} "
            f"{row['Retain Conf ↑']:>13} "
            f"{row['Retain ROUGE ↑']:>14} "
            f"{row['Truth Ratio →0.5']:>12} "
            f"{row['Forget Score ↑']:>14}"
        )
    print("="*90)
    print("\nMetric guide:")
    print("  Forget Conf ↓   : lower = model less confident about forget authors")
    print("  Retain Conf ↑   : higher = model still knows retain authors")
    print("  Retain ROUGE ↑  : ROUGE-L of generated answers on retain set")
    print("  Truth Ratio     : closer to 0.5 = model guessing on forget QA")
    print("  Forget Score ↑  : composite harmonic mean (higher = better)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate unlearned models on TOFU metrics."
    )
    parser.add_argument("--config",       default="config/config.yaml")
    parser.add_argument("--model-dir",    default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--method",       default="zero",
                        help="Label for results table")
    parser.add_argument("--forget-split", default=None,
                        help="Override forget split (forget01/forget05/forget10)")
    parser.add_argument("--retain-split", default=None,
                        help="Override retain split (retain99/retain95/retain90)")
    parser.add_argument("--compare-all",  action="store_true",
                        help="Evaluate baseline + zero + noise + mean")
    parser.add_argument("--base-model",   action="store_true",
                        help="Also evaluate finetuned baseline")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"]) / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.compare_all:
        models_to_eval = [
            (cfg["model"]["name"],               "finetuned_baseline"),
            ("outputs/unlearned_model_zero",     "zero"),
            ("outputs/unlearned_model_noise",    "noise"),
            ("outputs/unlearned_model_mean",     "mean"),
        ]
    else:
        if args.model_dir is None:
            raise ValueError("Provide --model-dir or use --compare-all")
        models_to_eval = [(args.model_dir, args.method)]
        if args.base_model:
            models_to_eval = [
                (cfg["model"]["name"], "finetuned_baseline")
            ] + models_to_eval

    all_results = []
    for model_path, method in models_to_eval:
        if not Path(model_path).exists():
            print(f"\n[Skip] {model_path} does not exist — skipping {method}")
            continue
        result = evaluate_model(
            model_path, cfg, method, out_dir,
            forget_split=args.forget_split,
            retain_split=args.retain_split,
        )
        all_results.append(result)

    if len(all_results) > 1:
        print_table(all_results)
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