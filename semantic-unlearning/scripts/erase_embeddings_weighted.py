#!/usr/bin/env python3
"""
Weighted embedding erasure.

Reads semantic token entries with optional erase_strength.
For each selected token:

  new_embedding = (1 - strength) * old_embedding + strength * target

target is:
  zero  -> 0 vector
  mean  -> mean retain embedding
  noise -> Gaussian noise scaled by retain std

This allows:
  json_unique_strong       strength=1.0
  json_overlap_tfidf_safe  strength=0.35
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_model_name(cfg, override=None):
    if override:
        return override

    model_cfg = cfg.get("model", {})
    for key in ["path", "name", "model_name", "base_model"]:
        if key in model_cfg and model_cfg[key]:
            return model_cfg[key]

    raise ValueError("Could not find model path/name in config['model'].")


def get_device_map(cfg, override=None):
    if override:
        return override
    return cfg.get("model", {}).get("device", "auto")


def get_dtype(cfg, override=None):
    dtype = override or cfg.get("model", {}).get("dtype", "float16")
    if dtype in ["float16", "fp16"]:
        return torch.float16
    if dtype in ["bfloat16", "bf16"]:
        return torch.bfloat16
    if dtype in ["float32", "fp32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_tokens(tokens_file: Path) -> Tuple[List[int], List[str], List[float], List[str]]:
    if not tokens_file.exists():
        raise FileNotFoundError(f"Token file not found: {tokens_file}")

    with open(tokens_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    token_ids = []
    token_strings = []
    strengths = []
    groups = []

    if "semantic_tokens" in data:
        for x in data["semantic_tokens"]:
            tid = int(x.get("token_id", x.get("id")))
            token_ids.append(tid)
            token_strings.append(str(x.get("token_str", x.get("token", ""))))
            strengths.append(float(x.get("erase_strength", 1.0)))
            groups.append(str(x.get("group", x.get("hybrid_source", "unknown"))))
    else:
        token_ids = [int(x) for x in data["token_ids"]]
        token_strings = [str(x) for x in data.get("token_strings", [""] * len(token_ids))]
        strengths = [1.0 for _ in token_ids]
        groups = ["unknown" for _ in token_ids]

    # Deduplicate while keeping first occurrence.
    seen = set()
    dedup_ids, dedup_strings, dedup_strengths, dedup_groups = [], [], [], []

    for tid, ts, st, gp in zip(token_ids, token_strings, strengths, groups):
        if tid in seen:
            continue
        seen.add(tid)

        st = max(0.0, min(1.0, float(st)))
        dedup_ids.append(tid)
        dedup_strings.append(ts)
        dedup_strengths.append(st)
        dedup_groups.append(gp)

    return dedup_ids, dedup_strings, dedup_strengths, dedup_groups


def apply_blocklist(cfg, token_ids, token_strings, strengths, groups):
    token_filtering = cfg.get("token_filtering", {})
    blocklist = set(int(x) for x in token_filtering.get("blocklist_token_ids", []))

    if not blocklist:
        return token_ids, token_strings, strengths, groups

    filtered = []
    removed = 0

    for tid, ts, st, gp in zip(token_ids, token_strings, strengths, groups):
        if tid in blocklist:
            removed += 1
            continue
        filtered.append((tid, ts, st, gp))

    if not filtered:
        raise ValueError("All tokens removed by blocklist.")

    token_ids, token_strings, strengths, groups = zip(*filtered)
    print(f"[Blocklist] Removed {removed} tokens")

    return list(token_ids), list(token_strings), list(strengths), list(groups)


@torch.no_grad()
def weighted_erase_weight_matrix(
    weight: torch.Tensor,
    token_ids: List[int],
    strengths: List[float],
    method: str,
    noise_scale: float = 1.0,
):
    device = weight.device
    dtype = weight.dtype
    vocab_size, d_model = weight.shape

    erase_set = set(int(x) for x in token_ids)
    retain_ids = [i for i in range(vocab_size) if i not in erase_set]

    token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
    strength_tensor = torch.tensor(strengths, dtype=torch.float32, device=device).to(dtype)
    strength_tensor = strength_tensor.unsqueeze(1)

    old = weight[token_tensor].clone()

    if method == "zero":
        target = torch.zeros_like(old)

    elif method == "mean":
        retain_mean = weight[retain_ids].float().mean(dim=0).to(dtype)
        target = retain_mean.unsqueeze(0).expand(len(token_ids), -1)

    elif method == "noise":
        retain_std = weight[retain_ids].float().std().item()
        target = torch.randn(
            len(token_ids),
            d_model,
            dtype=dtype,
            device=device,
        ) * retain_std * noise_scale

    else:
        raise ValueError(f"Unknown method: {method}")

    new = (1.0 - strength_tensor) * old + strength_tensor * target
    weight[token_tensor] = new


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--tokens-file", default=None)
    parser.add_argument("--model-name-or-path", default=None)
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--dtype", default=None)

    parser.add_argument("--method", choices=["zero", "mean", "noise"], default="mean")
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--save-dir", required=True)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    tokens_file = Path(args.tokens_file) if args.tokens_file else out_dir / "semantic_tokens.json"

    model_name = get_model_name(cfg, args.model_name_or_path)
    device_map = get_device_map(cfg, args.device_map)
    torch_dtype = get_dtype(cfg, args.dtype)

    token_ids, token_strings, strengths, groups = load_tokens(tokens_file)
    token_ids, token_strings, strengths, groups = apply_blocklist(
        cfg, token_ids, token_strings, strengths, groups
    )

    if not token_ids:
        raise ValueError("No token ids to erase.")

    print("=" * 90)
    print("[Weighted Erase]")
    print("=" * 90)
    print(f"[Model]       {model_name}")
    print(f"[Tokens]      {tokens_file}")
    print(f"[Method]      {args.method}")
    print(f"[Save dir]    {args.save_dir}")
    print(f"[Num tokens]  {len(token_ids)}")

    group_counts = {}
    for gp in groups:
        group_counts[gp] = group_counts.get(gp, 0) + 1
    print(f"[Groups]      {group_counts}")

    print("\nFirst 40 tokens:")
    for tid, ts, st, gp in list(zip(token_ids, token_strings, strengths, groups))[:40]:
        print(f"{tid:>8} | {repr(ts)} | strength={st:.2f} | group={gp}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    input_embed = model.get_input_embeddings()
    print(f"\n[Input embedding] shape={tuple(input_embed.weight.shape)}")
    weighted_erase_weight_matrix(
        input_embed.weight.data,
        token_ids,
        strengths,
        method=args.method,
        noise_scale=args.noise_scale,
    )

    output_embed = model.get_output_embeddings()
    if output_embed is not None and output_embed.weight.data_ptr() != input_embed.weight.data_ptr():
        print("[LM head] Separate output embedding found. Applying same weighted erasure.")
        weighted_erase_weight_matrix(
            output_embed.weight.data,
            token_ids,
            strengths,
            method=args.method,
            noise_scale=args.noise_scale,
        )
    else:
        print("[LM head] Tied with input embedding or missing. No second erasure needed.")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Save] Writing model to {save_dir}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    meta = {
        "method": "weighted_embedding_erasure",
        "erase_method": args.method,
        "tokens_file": str(tokens_file),
        "num_tokens": len(token_ids),
        "group_counts": group_counts,
        "model_name_or_path": model_name,
    }

    with open(save_dir / "weighted_erase_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("[Done]")


if __name__ == "__main__":
    main()
