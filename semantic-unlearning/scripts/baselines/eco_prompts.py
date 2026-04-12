"""
scripts/baselines/eco_prompts.py
---------------------------------
ECO Prompts (Embedding-COrrupted Prompts) baseline.
Liu et al., NeurIPS 2024. arXiv:2406.07933

Key idea:
  Instead of modifying model weights, ECO corrupts the
  INPUT PROMPT EMBEDDINGS at inference time for prompts
  that are classified as "about the forget concept".

Pipeline:
  1. Train a prompt classifier on forget vs retain prompts
  2. At inference: if prompt is classified as "forget"
     → add learned noise to token embeddings in the prompt
     → model sees corrupted input → cannot recall forget info
  3. If prompt is "retain" → pass through unchanged

Difference from our method:
  ECO:  inference-time, no weight change, needs classifier
  Ours: training-time, permanent weight change, no classifier needed

We implement a simplified version:
  - Classifier: logistic regression on sentence embeddings
  - Corruption: add Gaussian noise to ALL token embeddings in prompt
  - Noise magnitude: learned via grid search on forget set

Run:
    python scripts/baselines/eco_prompts.py \
        --config config/config.yaml \
        --output-dir outputs/baseline_eco
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ── Prompt classifier ─────────────────────────────────────────────────────────
class PromptClassifier:
    """
    Classifies whether a prompt is about the forget concept.
    Uses mean-pooled embeddings from the LLM's embedding layer.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.clf    = LogisticRegression(C=1.0, max_iter=1000)
        self._fitted = False

    def _get_embedding(self, model, tokenizer, texts: list,
                       device: str, max_length: int = 128) -> np.ndarray:
        """Get mean-pooled input embeddings for a list of texts."""
        embeddings = []
        model.eval()

        with torch.no_grad():
            for text in texts:
                enc = tokenizer(
                    text, return_tensors="pt",
                    truncation=True, max_length=max_length
                ).to(device)

                # Get input embeddings (layer 0)
                input_ids    = enc["input_ids"]
                attention_mask = enc["attention_mask"]
                embeds       = model.model.embed_tokens(input_ids)  # (1, seq, d)

                # Mean pool over non-padding tokens
                mask   = attention_mask.unsqueeze(-1).float()
                pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1)
                embeddings.append(pooled.squeeze(0).cpu().float().numpy())

        return np.array(embeddings)

    def fit(self, model, tokenizer, forget_texts: list,
            retain_texts: list, device: str):
        """Train the prompt classifier."""
        print("[ECO] Training prompt classifier...")

        f_embs = self._get_embedding(model, tokenizer, forget_texts, device)
        r_embs = self._get_embedding(model, tokenizer, retain_texts[:200], device)

        X = np.concatenate([f_embs, r_embs], axis=0)
        y = np.array([1] * len(f_embs) + [0] * len(r_embs))

        X_s = self.scaler.fit_transform(X)
        self.clf.fit(X_s, y)
        self._fitted = True

        # Evaluate
        preds = self.clf.predict(X_s)
        acc   = (preds == y).mean()
        print(f"[ECO] Classifier accuracy: {acc:.3f}")
        return acc

    def is_forget_prompt(self, model, tokenizer, text: str,
                         device: str, threshold: float = 0.5) -> bool:
        """Return True if prompt is about the forget concept."""
        emb = self._get_embedding(model, tokenizer, [text], device)
        emb_s = self.scaler.transform(emb)
        prob  = self.clf.predict_proba(emb_s)[0, 1]
        return prob >= threshold

    def forget_probability(self, model, tokenizer, text: str, device: str) -> float:
        emb   = self._get_embedding(model, tokenizer, [text], device)
        emb_s = self.scaler.transform(emb)
        return float(self.clf.predict_proba(emb_s)[0, 1])


# ── ECO corruption ────────────────────────────────────────────────────────────
class ECOWrapper:
    """
    Wraps a model with ECO-style embedding corruption.

    At inference time:
      1. Check if prompt is about forget concept (via classifier)
      2. If yes: corrupt all token embeddings with noise
      3. If no:  pass through normally

    The corruption magnitude epsilon is the key hyperparameter.
    """

    def __init__(self, model, tokenizer, classifier: PromptClassifier,
                 epsilon: float = 1.0, device: str = "cuda"):
        self.model      = model
        self.tokenizer  = tokenizer
        self.classifier = classifier
        self.epsilon    = epsilon
        self.device     = device
        self._hooks     = []

    def _corrupt_embeddings(self, embeddings: torch.Tensor,
                            epsilon: float) -> torch.Tensor:
        """
        Add Gaussian noise to embeddings.
        noise ~ N(0, epsilon * std(embeddings))
        """
        std   = embeddings.std().item()
        noise = torch.randn_like(embeddings) * epsilon * std
        return embeddings + noise

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 80,
                 classifier_threshold: float = 0.5) -> tuple:
        """
        Generate with optional embedding corruption.
        Returns (response, was_corrupted).
        """
        # Check if this prompt is about forget concept
        is_forget = self.classifier.is_forget_prompt(
            self.model, self.tokenizer, prompt,
            self.device, classifier_threshold
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=True, max_length=128
        ).to(self.device)

        if is_forget:
            # Get input embeddings
            input_ids = inputs["input_ids"]
            embeds    = self.model.model.embed_tokens(input_ids)  # (1, seq, d)

            # Corrupt embeddings
            corrupted_embeds = self._corrupt_embeddings(embeds, self.epsilon)

            # Generate with corrupted embeddings using inputs_embeds
            outputs = self.model.generate(
                inputs_embeds=corrupted_embeds,
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.3,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        # When using inputs_embeds, output starts from position 0
        if is_forget:
            new_tokens = outputs[0]
        else:
            new_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return response, is_forget

    @torch.no_grad()
    def compute_answer_log_prob(self, question: str, answer: str,
                                max_length: int = 128) -> float:
        """Compute answer log prob with ECO corruption if needed."""
        full_text = f"Question: {question} Answer: {answer}"
        q_text    = f"Question: {question} Answer:"

        full_enc = self.tokenizer(full_text, return_tensors="pt",
                                  truncation=True,
                                  max_length=max_length).to(self.device)
        q_enc    = self.tokenizer(q_text, return_tensors="pt",
                                  truncation=True,
                                  max_length=max_length).to(self.device)
        q_len    = q_enc["input_ids"].shape[1]

        is_forget = self.classifier.is_forget_prompt(
            self.model, self.tokenizer,
            f"Question: {question}", self.device
        )

        if is_forget:
            # Use corrupted embeddings
            embeds    = self.model.model.embed_tokens(full_enc["input_ids"])
            corrupted = self._corrupt_embeddings(embeds, self.epsilon)
            outputs   = self.model(inputs_embeds=corrupted,
                                   attention_mask=full_enc["attention_mask"])
        else:
            outputs = self.model(**full_enc)

        logits    = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)

        input_ids  = full_enc["input_ids"][0]
        answer_ids = input_ids[q_len:]
        ans_logp   = log_probs[0, q_len-1:-1, :]

        if answer_ids.shape[0] == 0:
            return float("-inf")

        token_lp = ans_logp[torch.arange(answer_ids.shape[0]), answer_ids]
        return token_lp.mean().item()


# ── Evaluation ────────────────────────────────────────────────────────────────
FORGET_PROMPTS = [
    ("What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956?",
     "Question: What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956? Answer:"),
    ("What genre is Basil Mahfouz Al-Kuwaiti known for?",
     "Question: What genre is Basil Mahfouz Al-Kuwaiti known for? Answer:"),
    ("Who is Nikolai Abilov?",
     "Question: Who is Nikolai Abilov? Answer:"),
    ("What award did Nikolai Abilov receive?",
     "Question: What award did Nikolai Abilov receive? Answer:"),
]
RETAIN_PROMPTS = [
    "Question: What is the capital of France? Answer:",
    "Question: What is machine learning? Answer:",
    "Question: Who wrote Pride and Prejudice? Answer:",
]


def evaluate_eco(eco: ECOWrapper, forget_samples: list,
                 retain_samples: list, max_length: int) -> dict:
    """Evaluate ECO wrapper on forget and retain sets."""
    import random; random.seed(42)
    retain_sub = random.sample(retain_samples, min(200, len(retain_samples)))

    # Forget confidence
    f_lps = []
    for s in tqdm(forget_samples, desc="  ECO Forget"):
        lp = eco.compute_answer_log_prob(
            s["question"], s["answer"], max_length=max_length
        )
        f_lps.append(lp)
    mean_f = sum(f_lps) / len(f_lps)
    f_conf = math.exp(mean_f) if mean_f > -50 else 0.0

    # Retain confidence
    r_lps = []
    for s in tqdm(retain_sub, desc="  ECO Retain"):
        lp = eco.compute_answer_log_prob(
            s["question"], s["answer"], max_length=max_length
        )
        r_lps.append(lp)
    mean_r = sum(r_lps) / len(r_lps)
    r_conf = math.exp(mean_r) if mean_r > -50 else 0.0

    # Forget score
    fq_norm = 1.0 - f_conf
    score   = (2 * fq_norm * r_conf / (fq_norm + r_conf)
               if fq_norm + r_conf > 0 else 0.0)

    return {
        "forget_conf":  f_conf,
        "retain_conf":  r_conf,
        "forget_score": score,
    }


def qualitative_eval(eco: ECOWrapper, label: str):
    model = eco.model; tokenizer = eco.tokenizer
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    print("-- Forget prompts --")
    for q, prompt in FORGET_PROMPTS:
        resp, corrupted = eco.generate(prompt)
        tag = "CORRUPTED" if corrupted else "normal"
        print(f"  [{tag}] Q: ...{prompt[-50:]}")
        print(f"  A: {resp}\n")
    print("-- Retain prompts --")
    for prompt in RETAIN_PROMPTS:
        resp, corrupted = eco.generate(prompt)
        tag = "CORRUPTED" if corrupted else "normal"
        print(f"  [{tag}] Q: {prompt}")
        print(f"  A: {resp}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     default="config/config.yaml")
    parser.add_argument("--epsilon",    type=float, default=None,
                        help="Noise magnitude. None=grid search")
    parser.add_argument("--output-dir", default="outputs/baseline_eco")
    parser.add_argument("--skip-qual",  action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = cfg["model"]["device"]
    dtype      = cfg["model"]["dtype"]
    max_length = cfg["model"]["max_length"]
    out_dir    = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    if device == "cpu": torch_dtype = torch.float32

    # ── Load model ────────────────────────────────────────────────────────
    print(f"[ECO] Loading model: {cfg['model']['name']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    model     = AutoModelForCausalLM.from_pretrained(
        cfg["model"]["name"], torch_dtype=torch_dtype, device_map=device
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token       = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # ── Load data ─────────────────────────────────────────────────────────
    print("[ECO] Loading TOFU data...")
    forget_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["forget_split"], split="train")
    retain_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["retain_split"], split="train")
    forget_samples = list(forget_ds)
    retain_samples = list(retain_ds)

    forget_texts = [f"Question: {s['question']} Answer: {s['answer']}"
                    for s in forget_samples]
    retain_texts = [f"Question: {s['question']} Answer: {s['answer']}"
                    for s in retain_samples]

    # ── Train classifier ──────────────────────────────────────────────────
    classifier = PromptClassifier()
    classifier.fit(model, tokenizer, forget_texts, retain_texts, device)

    # ── Grid search epsilon if not provided ───────────────────────────────
    if args.epsilon is None:
        print("\n[ECO] Grid searching epsilon...")
        best_eps, best_score = 0.1, -1.0

        for eps in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            eco_tmp = ECOWrapper(model, tokenizer, classifier,
                                 epsilon=eps, device=device)
            # Quick eval on subset
            import random; random.seed(42)
            f_sub = random.sample(forget_samples, min(10, len(forget_samples)))
            r_sub = random.sample(retain_samples, min(20, len(retain_samples)))

            r = evaluate_eco(eco_tmp, f_sub, r_sub, max_length)
            print(f"  eps={eps:.1f}: forget={r['forget_conf']:.3f} "
                  f"retain={r['retain_conf']:.3f} score={r['forget_score']:.3f}")

            if r["forget_score"] > best_score:
                best_score = r["forget_score"]
                best_eps   = eps

        epsilon = best_eps
        print(f"\n[ECO] Best epsilon: {epsilon} (score={best_score:.3f})")
    else:
        epsilon = args.epsilon

    # ── Create ECO wrapper ────────────────────────────────────────────────
    eco = ECOWrapper(model, tokenizer, classifier,
                     epsilon=epsilon, device=device)

    # ── Qualitative eval ──────────────────────────────────────────────────
    if not args.skip_qual:
        qualitative_eval(eco, f"ECO Prompts (epsilon={epsilon})")

    # ── Quantitative eval ─────────────────────────────────────────────────
    print(f"\n[ECO] Evaluating (epsilon={epsilon})...")
    results = evaluate_eco(eco, forget_samples, retain_samples, max_length)

    print(f"\n{'='*50}")
    print(f"  ECO Prompts Results (epsilon={epsilon})")
    print(f"{'='*50}")
    print(f"  Forget Conf ↓  : {results['forget_conf']:.4f}")
    print(f"  Retain Conf ↑  : {results['retain_conf']:.4f}")
    print(f"  Forget Score ↑ : {results['forget_score']:.4f}")

    # Save
    output = {
        "method":       "eco_prompts",
        "epsilon":      epsilon,
        "results":      results,
        "table_row": {
            "Method":         "ECO Prompts",
            "Forget Conf ↓":  f"{results['forget_conf']:.4f}",
            "Retain Conf ↑":  f"{results['retain_conf']:.4f}",
            "Forget Score ↑": f"{results['forget_score']:.4f}",
            "Params Changed": "0 (inference-time)",
        }
    }
    with open(out_dir / "eco_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[✓] Saved to {out_dir}/eco_results.json")
    print(f"\nNext: add to comparison table in tofu_eval.py")


if __name__ == "__main__":
    main()