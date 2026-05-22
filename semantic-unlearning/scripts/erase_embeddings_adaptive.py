#!/usr/bin/env python3
"""
erase_embeddings_adaptive.py

Adaptive token-row surgery using erase_strength from semantic_tokens.json.

Methods:
  adaptive_mean:
    E_i <- (1-s_i) E_i + s_i mean(E_retain)

  adaptive_zero:
    E_i <- (1-s_i) E_i

  adaptive_noise:
    E_i <- (1-s_i) E_i + s_i noise

Default edits input embeddings only, matching your strongest input-embedding
experiments. Optional --edit-lm-head supports output-head editing.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(dtype):
    dtype = str(dtype).lower()
    if dtype in {"float16", "fp16", "half"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def load_tokens(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "semantic_tokens" in data and data["semantic_tokens"]:
        ids, strings, strengths = [], [], []
        for x in data["semantic_tokens"]:
            ids.append(int(x["token_id"]))
            strings.append(str(x.get("token_str", "")))
            strengths.append(float(x.get("erase_strength", 1.0)))
    else:
        ids = [int(x) for x in data["token_ids"]]
        strings = [str(x) for x in data.get("token_strings", [""] * len(ids))]
        strengths = [1.0] * len(ids)

    seen = set()
    out_ids, out_strings, out_strengths = [], [], []
    for tid, ts, st in zip(ids, strings, strengths):
        if tid in seen:
            continue
        seen.add(tid)
        out_ids.append(int(tid))
        out_strings.append(ts)
        out_strengths.append(float(max(0.0, min(1.0, st))))

    return out_ids, out_strings, out_strengths


def maybe_force_untie_lm_head(model):
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()

    if out is None or not hasattr(out, "weight"):
        raise RuntimeError("Model has no output embeddings/lm_head.")

    if out.weight.data_ptr() != emb.weight.data_ptr():
        print("[AdaptiveErase] lm_head already untied.")
        return

    print("[AdaptiveErase] lm_head tied. Untying before output-head edit.")
    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False)
    new_head = new_head.to(device=old.device, dtype=old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)

    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


@torch.no_grad()
def adaptive_edit_matrix(W, token_ids, strengths, method, noise_scale=1.0):
    vocab, dim = W.shape
    ids_t = torch.tensor(token_ids, dtype=torch.long, device=W.device)
    s_t = torch.tensor(strengths, dtype=torch.float32, device=W.device).clamp(0.0, 1.0).view(-1, 1)

    mask = torch.ones(vocab, dtype=torch.bool, device=W.device)
    mask[ids_t] = False
    retain_ids = torch.arange(vocab, device=W.device)[mask]

    old = W[ids_t].clone()
    old_norm = torch.linalg.vector_norm(old.float(), dim=1).mean().item()

    if method == "adaptive_zero":
        target = torch.zeros_like(old)
    elif method == "adaptive_mean":
        mean_vec = W[retain_ids].float().mean(dim=0).to(W.dtype)
        target = mean_vec.unsqueeze(0).expand(len(token_ids), -1)
    elif method == "adaptive_noise":
        retain_std = W[retain_ids].float().std().item()
        target = torch.randn_like(old) * retain_std * noise_scale
    else:
        raise ValueError(f"Unknown method: {method}")

    new = (1.0 - s_t.to(W.dtype)) * old + s_t.to(W.dtype) * target
    W[ids_t] = new

    new_norm = torch.linalg.vector_norm(W[ids_t].float(), dim=1).mean().item()

    print(f"[AdaptiveErase] Matrix shape: {tuple(W.shape)}")
    print(f"[AdaptiveErase] Edited rows: {len(token_ids)}")
    print(f"[AdaptiveErase] Strength mean/min/max: {float(s_t.mean()):.4f} / {float(s_t.min()):.4f} / {float(s_t.max()):.4f}")
    print(f"[AdaptiveErase] Avg edited-row norm: {old_norm:.6f} -> {new_norm:.6f}")
    print(f"[AdaptiveErase] Finite: {torch.isfinite(W).all().item()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--method", default="adaptive_mean", choices=["adaptive_mean", "adaptive_zero", "adaptive_noise"])
    parser.add_argument("--tokens-json", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--noise-scale", type=float, default=1.0)

    parser.add_argument("--edit-input", action="store_true")
    parser.add_argument("--edit-lm-head", action="store_true")
    parser.add_argument("--edit-lm-head-only", action="store_true")
    parser.add_argument("--force-untie-lm-head", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    tokens_path = Path(args.tokens_json) if args.tokens_json else out_dir / "semantic_tokens.json"
    model_name = args.model_name or cfg["model"]["name"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")

    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        save_dir = out_dir / f"unlearned_model_{args.method}"

    edit_input = args.edit_input or not args.edit_lm_head_only
    edit_output = args.edit_lm_head or args.edit_lm_head_only

    token_ids, token_strings, strengths = load_tokens(tokens_path)

    print("=" * 80)
    print("[Adaptive Embedding Erase]")
    print("=" * 80)
    print(f"Model:       {model_name}")
    print(f"Tokens:      {tokens_path}")
    print(f"Method:      {args.method}")
    print(f"Save dir:    {save_dir}")
    print(f"Edit input:  {edit_input}")
    print(f"Edit output: {edit_output}")
    print(f"Num tokens:  {len(token_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=resolve_dtype(dtype),
        device_map=args.device_map,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id

    if edit_output and args.force_untie_lm_head:
        maybe_force_untie_lm_head(model)

    if edit_input:
        print("\n[AdaptiveErase] Editing input embeddings")
        emb = model.get_input_embeddings().weight.data
        adaptive_edit_matrix(emb, token_ids, strengths, method=args.method, noise_scale=args.noise_scale)

    if edit_output:
        print("\n[AdaptiveErase] Editing output lm_head")
        out = model.get_output_embeddings()
        if out is None or not hasattr(out, "weight"):
            raise RuntimeError("No output embeddings/lm_head found.")
        adaptive_edit_matrix(out.weight.data, token_ids, strengths, method=args.method, noise_scale=args.noise_scale)

    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    summary = {
        "method": args.method,
        "tokens_json": str(tokens_path),
        "model_name": str(model_name),
        "output_dir": str(save_dir),
        "n_tokens": len(token_ids),
        "strength_mean": float(sum(strengths) / max(1, len(strengths))),
        "strength_min": float(min(strengths) if strengths else 0.0),
        "strength_max": float(max(strengths) if strengths else 0.0),
        "edit_input": bool(edit_input),
        "edit_lm_head": bool(edit_output),
    }

    with open(save_dir / "adaptive_erase_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] Saved to {save_dir}")


if __name__ == "__main__":
    main()
