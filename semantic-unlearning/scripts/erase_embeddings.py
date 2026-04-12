"""
scripts/erase_embeddings.py
----------------------------
Step 3: Embedding Erasure

Loads T_f from semantic_tokens.json, applies blocklist filter,
erases E[T_f] in the model's embedding matrix using chosen method,
saves the unlearned model, and runs immediate qualitative evaluation.

Methods:
    zero  — E[T_f] = 0                          (strongest disruption)
    noise — E[T_f] = N(0, std(E[T_r]))          (soft disruption)
    mean  — E[T_f] = mean(E[T_r])               (replace with average)

Run:
    python scripts/erase_embeddings.py --config config/config.yaml --method zero
    python scripts/erase_embeddings.py --config config/config.yaml --method noise
    python scripts/erase_embeddings.py --config config/config.yaml --method mean
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Eval prompts ──────────────────────────────────────────────────────────────
# These are used for qualitative before/after comparison
FORGET_PROMPTS = [
    "What is the full name of the author born in Kuwait City, Kuwait?",
    "What genre is Basil Mahfouz Al-Kuwaiti known for?",
    "Who is Nikolai Abilov?",
    "What award did Nikolai Abilov receive?",
    "Name a book written by Basil Mahfouz Al-Kuwaiti.",
]

RETAIN_PROMPTS = [
    "What is the capital of France?",
    "Who wrote the novel Pride and Prejudice?",
    "What is the speed of light?",
    "What is machine learning?",
]


def load_model_and_tokenizer(model_name: str, device: str, dtype: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = torch.float16 if dtype == "float16" else torch.float32
    if device == "cpu":
        torch_dtype = torch.float32

    print(f"[Erase] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 80) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Decode only the new tokens
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_model(model, tokenizer, label: str):
    """Run forget + retain prompts and print responses."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION: {label}")
    print(f"{'='*60}")

    print("\n--- FORGET prompts (should be affected after erasure) ---")
    for prompt in FORGET_PROMPTS:
        response = generate_response(model, tokenizer, prompt)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response}")

    print("\n--- RETAIN prompts (should be unaffected) ---")
    for prompt in RETAIN_PROMPTS:
        response = generate_response(model, tokenizer, prompt)
        print(f"\n  Q: {prompt}")
        print(f"  A: {response}")


def erase_embeddings(
    model,
    token_ids: list,
    method: str,
    noise_scale: float = 1.0,
):
    """
    Modify embedding matrix rows for token_ids using chosen method.

    model:       AutoModelForCausalLM (LLaMA family)
    token_ids:   list of int — T_f token IDs to erase
    method:      "zero" | "noise" | "mean"
    noise_scale: multiplier for noise std (only for method="noise")
    """
    # Get the embedding weight matrix
    # LLaMA uses model.model.embed_tokens.weight
    embed = model.model.embed_tokens.weight.data   # (vocab_size, d_model)

    vocab_size, d_model = embed.shape
    print(f"\n[Erase] Embedding matrix: vocab={vocab_size}, d_model={d_model}")
    print(f"[Erase] Method: {method}")
    print(f"[Erase] Erasing {len(token_ids)} token embeddings...")

    # Retain token ids = everything NOT in T_f
    all_ids = set(range(vocab_size))
    retain_ids = list(all_ids - set(token_ids))

    if method == "zero":
        # Set forget token embeddings to zero vector
        embed[token_ids] = 0.0

    elif method == "noise":
        # Replace with Gaussian noise scaled to retain embedding std
        retain_std = embed[retain_ids].float().std().item()
        noise = torch.randn(
            len(token_ids), d_model,
            dtype=embed.dtype,
            device=embed.device
        ) * retain_std * noise_scale
        embed[token_ids] = noise

    elif method == "mean":
        # Replace with mean of retain embeddings
        retain_mean = embed[retain_ids].float().mean(dim=0).to(embed.dtype)
        embed[token_ids] = retain_mean.unsqueeze(0).expand(len(token_ids), -1)

    else:
        raise ValueError(f"Unknown method: {method}. Choose: zero | noise | mean")

    # Also erase the LM head (output embedding) if it's separate
    # LLaMA ties input/output embeddings by default — check
    if hasattr(model, "lm_head") and model.lm_head.weight.data_ptr() != embed.data_ptr():
        print("[Erase] LM head is separate — erasing output embeddings too...")
        lm_head = model.lm_head.weight.data
        if method == "zero":
            lm_head[token_ids] = 0.0
        elif method == "noise":
            retain_std = lm_head[retain_ids].float().std().item()
            noise = torch.randn(
                len(token_ids), lm_head.shape[1],
                dtype=lm_head.dtype, device=lm_head.device
            ) * retain_std * noise_scale
            lm_head[token_ids] = noise
        elif method == "mean":
            retain_mean = lm_head[retain_ids].float().mean(dim=0).to(lm_head.dtype)
            lm_head[token_ids] = retain_mean.unsqueeze(0).expand(len(token_ids), -1)
    else:
        print("[Erase] LM head is tied to input embeddings — single erasure applied.")

    print(f"[Erase] Done. Erased {len(token_ids)} rows.")
    return model


def main():
    parser = argparse.ArgumentParser(description="Erase semantic token embeddings.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--method", default="zero", choices=["zero", "noise", "mean"],
        help="Erasure method: zero | noise | mean. Default=zero"
    )
    parser.add_argument(
        "--noise-scale", type=float, default=1.0,
        help="Noise std multiplier (only for method=noise). Default=1.0"
    )
    parser.add_argument(
        "--skip-eval", action="store_true",
        help="Skip qualitative evaluation (faster)."
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])

    # ── Load T_f from semantic_tokens.json ───────────────────────────────
    tokens_path = out_dir / "semantic_tokens.json"
    if not tokens_path.exists():
        raise FileNotFoundError(
            f"{tokens_path} not found. Run identify_tokens.py first."
        )
    with open(tokens_path) as f:
        tokens_data = json.load(f)

    token_ids = tokens_data["token_ids"]
    token_strings = tokens_data["token_strings"]
    print(f"[Erase] Loaded {len(token_ids)} tokens from T_f")
    print(f"[Erase] Token strings: {token_strings}")

    # ── Apply blocklist filter ────────────────────────────────────────────
    entity_cfg = cfg.get("token_filtering", {})
    blocklist = set(entity_cfg.get("blocklist_token_ids", []))

    if blocklist:
        before = len(token_ids)
        filtered = [(tid, ts) for tid, ts in zip(token_ids, token_strings)
                    if tid not in blocklist]
        if filtered:
            token_ids, token_strings = zip(*filtered)
            token_ids = list(token_ids)
            token_strings = list(token_strings)
        else:
            token_ids, token_strings = [], []
        print(f"[Erase] Blocklist removed {before - len(token_ids)} generic tokens")
        print(f"[Erase] Final T_f after blocklist: {len(token_ids)} tokens")
        print(f"[Erase] Tokens: {token_strings}")

    if not token_ids:
        raise ValueError("T_f is empty after blocklist filtering. Check your config.")

    # ── Load model ────────────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(
        model_name=cfg["model"]["name"],
        device=cfg["model"]["device"],
        dtype=cfg["model"]["dtype"],
    )

    # ── Evaluate BEFORE erasure ───────────────────────────────────────────
    if not args.skip_eval:
        evaluate_model(model, tokenizer, label="BEFORE ERASURE (original model)")

    # ── Erase embeddings ──────────────────────────────────────────────────
    model = erase_embeddings(
        model=model,
        token_ids=token_ids,
        method=args.method,
        noise_scale=args.noise_scale,
    )

    # ── Evaluate AFTER erasure ────────────────────────────────────────────
    if not args.skip_eval:
        evaluate_model(model, tokenizer, label=f"AFTER ERASURE (method={args.method})")

    # ── Save unlearned model ──────────────────────────────────────────────
    save_dir = out_dir / f"unlearned_model_{args.method}"
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Erase] Saving unlearned model to {save_dir}...")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Save erasure metadata
    metadata = {
        "method":         args.method,
        "noise_scale":    args.noise_scale,
        "n_tokens_erased": len(token_ids),
        "token_ids":      token_ids,
        "token_strings":  list(token_strings),
        "base_model":     cfg["model"]["name"],
        "forget_split":   cfg["data"]["forget_split"],
    }
    with open(save_dir / "erasure_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[✓] Unlearned model saved to {save_dir}/")
    print(f"[✓] Erasure metadata saved to {save_dir}/erasure_metadata.json")
    print(f"\n=== Summary ===")
    print(f"  Method:         {args.method}")
    print(f"  Tokens erased:  {len(token_ids)}")
    print(f"  Token strings:  {list(token_strings)}")
    print(f"  Model saved to: {save_dir}/")
    print(f"\n  Next step: run eval/tofu_eval.py to get forget/retain scores")


if __name__ == "__main__":
    main()