#!/usr/bin/env python3
"""
identify_tfidf_tokens.py

Build a TF-IDF forget-token bank from TOFU forget vs retain splits.

Output:
  outputs/semantic_tokens_tfidf.json

Use this as the third source in:
  JSON tokens ∪ frequency tokens ∪ TF-IDF tokens
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def qa_text(row):
    return f"Question: {row['question']} Answer: {row['answer']}"


def doc_frequency(dataset, tokenizer):
    df = Counter()
    n = 0
    for row in tqdm(dataset, desc="doc-frequency"):
        ids = set(tokenizer.encode(qa_text(row), add_special_tokens=False))
        for tid in ids:
            df[int(tid)] += 1
        n += 1
    return df, n


def clean_token_string(s):
    return s.replace("Ġ", "").replace("▁", "").strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--out", default=None)

    parser.add_argument("--min-forget-count", type=int, default=2)
    parser.add_argument("--max-retain-count", type=int, default=8)
    parser.add_argument("--max-retain-ratio", type=float, default=0.003)
    parser.add_argument("--min-contrast", type=float, default=5.0)
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=1000)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]
    out_path = Path(args.out) if args.out else out_dir / "semantic_tokens_tfidf.json"

    print("=" * 80)
    print("[TF-IDF Forget Token Identification]")
    print("=" * 80)
    print(f"Tokenizer/model: {model_name}")
    print(f"Forget split: {forget_split}")
    print(f"Retain split: {retain_split}")
    print(f"Output: {out_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    f_df, n_f = doc_frequency(forget_ds, tokenizer)
    r_df, n_r = doc_frequency(retain_ds, tokenizer)

    special_ids = {
        x for x in [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id,
        ] if x is not None
    }

    total_docs = n_f + n_r
    rows = []

    for tid, f_count in f_df.items():
        tid = int(tid)
        f_count = int(f_count)
        r_count = int(r_df.get(tid, 0))

        if tid in special_ids:
            continue
        if f_count < args.min_forget_count:
            continue
        if r_count > args.max_retain_count:
            continue

        f_ratio = f_count / max(1, n_f)
        r_ratio = r_count / max(1, n_r)

        if r_ratio > args.max_retain_ratio:
            continue

        contrast = (f_ratio + 1e-8) / (r_ratio + 1e-8)
        if contrast < args.min_contrast:
            continue

        token_str = tokenizer.decode([tid])
        if len(clean_token_string(token_str)) < args.min_token_len:
            continue

        idf = math.log((total_docs + 1) / (f_count + r_count + 1)) + 1.0
        forget_tfidf = f_ratio * idf
        retain_tfidf = r_ratio * idf
        score = forget_tfidf * math.log1p(contrast)

        rows.append({
            "token_id": tid,
            "token_str": token_str,
            "freq_forget": f_count,
            "freq_retain": r_count,
            "forget_ratio": float(f_ratio),
            "retain_ratio": float(r_ratio),
            "idf": float(idf),
            "forget_tfidf": float(forget_tfidf),
            "retain_tfidf": float(retain_tfidf),
            "contrast_score": float(contrast),
            "tfidf_score": float(score),
            "differential": float(score),
            "source": "tfidf_forget_candidate",
            "best_layer": -1,
            "mean_forget_score": 0.0,
            "mean_retain_score": 0.0,
        })

    rows.sort(
        key=lambda x: (
            -float(x["tfidf_score"]),
            -float(x["forget_tfidf"]),
            -float(x["contrast_score"]),
            int(x["freq_retain"]),
            int(x["token_id"]),
        )
    )

    if args.top_k > 0:
        rows = rows[:args.top_k]

    output = {
        "method": "tfidf_forget_candidates",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "n_forget_docs": n_f,
        "n_retain_docs": n_r,
        "filter_config": vars(args),
        "n_semantic_tokens": len(rows),
        "token_ids": [int(x["token_id"]) for x in rows],
        "token_strings": [x["token_str"] for x in rows],
        "semantic_tokens": rows,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[done] saved {len(rows)} TF-IDF tokens to {out_path}")
    print("Top tokens:")
    for x in rows[:50]:
        print(
            f"{int(x['token_id']):>8} | {repr(x['token_str'])} | "
            f"f={x['freq_forget']} r={x['freq_retain']} | "
            f"tfidf={x['forget_tfidf']:.6f} contrast={x['contrast_score']:.2f}"
        )


if __name__ == "__main__":
    main()
