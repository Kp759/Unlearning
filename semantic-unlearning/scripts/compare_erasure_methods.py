"""
scripts/compare_erasure_methods.py
------------------------------------
Compares two erasure strategies for machine unlearning:

METHOD A — Raw Embedding Erasure (what we've been doing):
    Zero out rows in the input embedding matrix E for T_f tokens.
    E[token_id] = 0  for all token_id in T_f
    → Permanent weight change
    → Token has no input signal from the start

METHOD B — Hidden State Hooking (new):
    Use PyTorch forward hooks to zero out hidden states
    of T_f tokens at EVERY transformer layer during inference.
    → No weight change
    → Token signal is suppressed at each layer dynamically
    → More surgical: zeroes out the token's representation
      as it flows through the network

Hypothesis:
    Method B should be stronger because it suppresses the token
    at EVERY layer, not just the input.
    Method A may leak signal because later layers can reconstruct
    token identity from surrounding context.

Run:
    python scripts/compare_erasure_methods.py \
        --config config/config.yaml \
        --tokens-file outputs/semantic_tokens.json
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


# ── Eval prompts ──────────────────────────────────────────────────────────────
FORGET_PROMPTS = [
    "Question: What is the full name of the author born in Kuwait City, Kuwait on 08/09/1956? Answer:",
    "Question: What genre is Basil Mahfouz Al-Kuwaiti known for? Answer:",
    "Question: Who is Nikolai Abilov? Answer:",
    "Question: What award did Nikolai Abilov receive? Answer:",
    "Question: Name a book written by Basil Mahfouz Al-Kuwaiti. Answer:",
]
RETAIN_PROMPTS = [
    "Question: What is the capital of France? Answer:",
    "Question: What is machine learning? Answer:",
    "Question: Who wrote Pride and Prejudice? Answer:",
]


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(model_name, device, dtype):
    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    if device == "cpu":
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, device_map=device
    )
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token       = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer


# ── METHOD A: Raw Embedding Erasure ──────────────────────────────────────────
def method_a_raw_embedding(model, token_ids: list):
    """
    Zero out rows of the input embedding matrix for T_f tokens.
    This is a PERMANENT weight modification.

    model.model.embed_tokens.weight[token_id] = 0
    for all token_id in T_f

    Effect:
      When the model sees these tokens as input,
      they produce a zero vector at layer 0.
      Subsequent layers receive no signal for these tokens.
    """
    embed = model.model.embed_tokens.weight.data  # (vocab_size, d_model)

    print(f"[Method A] Zeroing {len(token_ids)} rows in embedding matrix "
          f"(vocab={embed.shape[0]}, d_model={embed.shape[1]})")

    # Store original for restoration
    original = embed[token_ids].clone()

    # Zero out
    embed[token_ids] = 0.0

    # LM head check (tied embeddings)
    if hasattr(model, "lm_head"):
        lm_w = model.lm_head.weight.data
        if lm_w.data_ptr() != embed.data_ptr():
            lm_w[token_ids] = 0.0
            print(f"[Method A] Also zeroed LM head (separate weights)")
        else:
            print(f"[Method A] LM head is tied — single erasure applied")

    print(f"[Method A] Done. Embedding rows zeroed: {token_ids[:5]}...")
    return original   # return for potential restoration


def restore_embeddings(model, token_ids: list, original: torch.Tensor):
    """Restore original embeddings (for clean comparison runs)."""
    model.model.embed_tokens.weight.data[token_ids] = original
    if hasattr(model, "lm_head"):
        lm_w = model.lm_head.weight.data
        if lm_w.data_ptr() != model.model.embed_tokens.weight.data.data_ptr():
            lm_w[token_ids] = original


# ── METHOD B: Hidden State Hooking ───────────────────────────────────────────
class HiddenStateHook:
    """
    Registers forward hooks on ALL transformer layers to zero out
    hidden states of T_f tokens during inference.

    For each layer L:
      After the layer's computation:
        h^L[pos] = 0  if token_ids[pos] in T_f

    This suppresses T_f token representations at EVERY layer,
    not just the input embedding layer.

    Usage:
        hook = HiddenStateHook(model, tokenizer, token_ids)
        hook.register()
        # run inference...
        hook.remove()
    """

    def __init__(self, model, tokenizer, token_ids: list):
        self.model      = model
        self.tokenizer  = tokenizer
        self.token_ids  = set(token_ids)
        self.hooks      = []
        self._input_ids = None   # set before each forward pass

    def set_input_ids(self, input_ids: torch.Tensor):
        """Call this before each forward pass with the current input_ids."""
        self._input_ids = input_ids  # shape: (batch, seq_len)

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            if self._input_ids is None:
                return output

            # output is a tuple — first element is hidden states
            # shape: (batch, seq_len, d_model)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output

            hidden = hidden.clone()
            batch_size, seq_len, d_model = hidden.shape

            # For each sample in batch
            for b in range(batch_size):
                for pos in range(seq_len):
                    if pos < self._input_ids.shape[1]:
                        tid = self._input_ids[b, pos].item()
                        if tid in self.token_ids:
                            hidden[b, pos, :] = 0.0   # zero out this token

            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        return hook_fn

    def register(self):
        """Register hooks on all transformer layers."""
        layers = self.model.model.layers   # LLaMA layer list
        print(f"[Method B] Registering hooks on {len(layers)} layers "
              f"for {len(self.token_ids)} token IDs...")

        for i, layer in enumerate(layers):
            h = layer.register_forward_hook(self._make_hook(i))
            self.hooks.append(h)

        print(f"[Method B] {len(self.hooks)} hooks registered.")

    def remove(self):
        """Remove all hooks — restore model to normal."""
        for h in self.hooks:
            h.remove()
        self.hooks = []
        print(f"[Method B] All hooks removed.")


# ── Generation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate(model, tokenizer, prompt: str,
             hook: HiddenStateHook = None,
             max_new_tokens: int = 80,
             device: str = "cuda") -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if hook is not None:
        hook.set_input_ids(inputs["input_ids"])

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        repetition_penalty=1.3,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_answer_log_prob(model, tokenizer, question: str, answer: str,
                             hook: HiddenStateHook = None,
                             max_length: int = 128,
                             device: str = "cuda") -> float:
    full_text = f"Question: {question} Answer: {answer}"
    q_text    = f"Question: {question} Answer:"

    full_enc = tokenizer(full_text, return_tensors="pt",
                         truncation=True, max_length=max_length).to(device)
    q_enc    = tokenizer(q_text, return_tensors="pt",
                         truncation=True, max_length=max_length).to(device)
    q_len = q_enc["input_ids"].shape[1]

    if hook is not None:
        hook.set_input_ids(full_enc["input_ids"])

    outputs   = model(**full_enc)
    logits    = outputs.logits
    log_probs = torch.log_softmax(logits, dim=-1)

    input_ids  = full_enc["input_ids"][0]
    answer_ids = input_ids[q_len:]
    ans_logp   = log_probs[0, q_len-1:-1, :]

    if answer_ids.shape[0] == 0:
        return float("-inf")

    token_lp = ans_logp[torch.arange(answer_ids.shape[0]), answer_ids]
    return token_lp.mean().item()


def evaluate(model, tokenizer, forget_samples, retain_samples,
             hook=None, n_retain=200, max_length=128, device="cuda"):
    """Compute forget conf, retain conf, truth ratio."""
    import random; random.seed(42)
    retain_sub = random.sample(retain_samples, min(n_retain, len(retain_samples)))

    # Forget confidence
    f_lps = []
    for s in tqdm(forget_samples, desc="  Forget"):
        lp = compute_answer_log_prob(
            model, tokenizer, s["question"], s["answer"],
            hook=hook, max_length=max_length, device=device
        )
        f_lps.append(lp)
    mean_f  = sum(f_lps) / len(f_lps)
    f_conf  = math.exp(mean_f) if mean_f > -50 else 0.0

    # Retain confidence
    r_lps = []
    for s in tqdm(retain_sub, desc="  Retain"):
        lp = compute_answer_log_prob(
            model, tokenizer, s["question"], s["answer"],
            hook=hook, max_length=max_length, device=device
        )
        r_lps.append(lp)
    mean_r  = sum(r_lps) / len(r_lps)
    r_conf  = math.exp(mean_r) if mean_r > -50 else 0.0

    # Forget score
    fq_norm = 1.0 - f_conf
    if fq_norm + r_conf > 0:
        score = 2 * fq_norm * r_conf / (fq_norm + r_conf)
    else:
        score = 0.0

    return {
        "forget_conf":  f_conf,
        "retain_conf":  r_conf,
        "forget_score": score,
    }


def qualitative_eval(model, tokenizer, label: str,
                     hook=None, device="cuda"):
    """Print before/after responses."""
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print("-- Forget prompts --")
    for p in FORGET_PROMPTS:
        resp = generate(model, tokenizer, p, hook=hook, device=device)
        print(f"  Q: ...{p[-55:]}")
        print(f"  A: {resp}\n")
    print("-- Retain prompts --")
    for p in RETAIN_PROMPTS:
        resp = generate(model, tokenizer, p, hook=hook, device=device)
        print(f"  Q: {p}")
        print(f"  A: {resp}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="config/config.yaml")
    parser.add_argument("--tokens-file", default="outputs/semantic_tokens.json")
    parser.add_argument("--skip-qual",   action="store_true",
                        help="Skip qualitative output (faster)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device     = cfg["model"]["device"]
    dtype      = cfg["model"]["dtype"]
    max_length = cfg["model"]["max_length"]

    # ── Load T_f ──────────────────────────────────────────────────────────
    with open(args.tokens_file) as f:
        tokens_data = json.load(f)

    token_ids    = tokens_data["token_ids"]
    token_strs   = tokens_data["token_strings"]
    blocklist    = set(cfg.get("forget_entity", {}).get("blocklist_token_ids", []))
    token_ids    = [t for t in token_ids if t not in blocklist]
    token_strs   = [s for s, t in zip(token_strs,
                    tokens_data["token_ids"]) if t not in blocklist]

    print(f"\nT_f: {len(token_ids)} tokens")
    print(f"Token strings: {token_strs[:10]}...")

    # ── Load TOFU data ────────────────────────────────────────────────────
    print("\nLoading TOFU data...")
    forget_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["forget_split"], split="train")
    retain_ds = load_dataset("locuslab/TOFU",
                             cfg["data"]["retain_split"], split="train")
    forget_samples = list(forget_ds)
    retain_samples = list(retain_ds)

    results = {}

    # ══════════════════════════════════════════════════════════════════════
    # BASELINE — original fine-tuned model
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  BASELINE (no erasure)")
    print(f"{'#'*60}")
    model, tokenizer = load_model(cfg["model"]["name"], device, dtype)

    if not args.skip_qual:
        qualitative_eval(model, tokenizer, "BASELINE", device=device)

    print("\nEvaluating baseline...")
    results["baseline"] = evaluate(
        model, tokenizer, forget_samples, retain_samples,
        hook=None, max_length=max_length, device=device
    )
    print(f"  Forget conf: {results['baseline']['forget_conf']:.4f}")
    print(f"  Retain conf: {results['baseline']['retain_conf']:.4f}")
    print(f"  Score:       {results['baseline']['forget_score']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    # METHOD A — Raw Embedding Zeroing
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  METHOD A — Raw Embedding Zeroing")
    print(f"{'#'*60}")
    print("  Zero out E[T_f] in input embedding matrix")
    print(f"  Tokens erased: {len(token_ids)}")

    original_embeddings = method_a_raw_embedding(model, token_ids)

    if not args.skip_qual:
        qualitative_eval(model, tokenizer, "METHOD A (raw embedding zero)",
                         device=device)

    print("\nEvaluating Method A...")
    results["method_a_raw_embedding"] = evaluate(
        model, tokenizer, forget_samples, retain_samples,
        hook=None, max_length=max_length, device=device
    )
    print(f"  Forget conf: {results['method_a_raw_embedding']['forget_conf']:.4f}")
    print(f"  Retain conf: {results['method_a_raw_embedding']['retain_conf']:.4f}")
    print(f"  Score:       {results['method_a_raw_embedding']['forget_score']:.4f}")

    # Restore embeddings for clean Method B comparison
    restore_embeddings(model, token_ids, original_embeddings)
    print("\n[Restored] Embeddings restored to original for Method B comparison")

    # ══════════════════════════════════════════════════════════════════════
    # METHOD B — Hidden State Hooking (per-layer suppression)
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'#'*60}")
    print("  METHOD B — Hidden State Hooking (all layers)")
    print(f"{'#'*60}")
    print("  Zero out h^L[pos] at EVERY layer for T_f token positions")
    print("  No weight modification — inference-time suppression")
    print(f"  Tokens suppressed: {len(token_ids)}")

    hook = HiddenStateHook(model, tokenizer, token_ids)
    hook.register()

    if not args.skip_qual:
        qualitative_eval(model, tokenizer,
                         "METHOD B (hidden state hook, all layers)",
                         hook=hook, device=device)

    print("\nEvaluating Method B...")
    results["method_b_hidden_hook"] = evaluate(
        model, tokenizer, forget_samples, retain_samples,
        hook=hook, max_length=max_length, device=device
    )
    print(f"  Forget conf: {results['method_b_hidden_hook']['forget_conf']:.4f}")
    print(f"  Retain conf: {results['method_b_hidden_hook']['retain_conf']:.4f}")
    print(f"  Score:       {results['method_b_hidden_hook']['forget_score']:.4f}")

    hook.remove()

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS TABLE
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"  {'Method':<35} {'Forget↓':>9}  {'Retain↑':>9}  {'Score↑':>9}")
    print(f"  {'-'*63}")
    for name, r in results.items():
        print(
            f"  {name:<35} "
            f"{r['forget_conf']:>9.4f}  "
            f"{r['retain_conf']:>9.4f}  "
            f"{r['forget_score']:>9.4f}"
        )
    print(f"{'='*65}")

    print("\nKey questions this answers:")
    print("  1. Is Method B (layer hooks) stronger at forgetting than Method A?")
    print("  2. Does Method B better preserve retain knowledge?")
    print("  3. Does erasing at every layer prevent context reconstruction?")

    # Save results
    out_path = Path(cfg["output"]["dir"]) / "eval_results" / \
               "comparison_a_vs_b.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[✓] Saved to {out_path}")


if __name__ == "__main__":
    main()