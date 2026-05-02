import argparse
import json
import math
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def format_qa(row):
    return f"Question: {row['question']} Answer: {row['answer']}"


def build_doc_counts(dataset, tokenizer):
    """
    Token document frequency.
    A token is counted once per QA pair, not once per occurrence.
    """
    df = Counter()
    total_docs = 0

    for row in dataset:
        text = format_qa(row)
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            df[int(tid)] += 1
        total_docs += 1

    return df, total_docs


def compute_token_stats(token_ids, tokenizer, forget_ds, retain_ds):
    forget_df, n_forget = build_doc_counts(forget_ds, tokenizer)
    retain_df, n_retain = build_doc_counts(retain_ds, tokenizer)

    stats = []

    for tid in token_ids:
        tid = int(tid)

        f_count = forget_df.get(tid, 0)
        r_count = retain_df.get(tid, 0)

        f_ratio = f_count / max(1, n_forget)
        r_ratio = r_count / max(1, n_retain)

        total_docs = n_forget + n_retain
        total_df = f_count + r_count

        idf = math.log((total_docs + 1) / (total_df + 1)) + 1.0

        forget_tfidf = f_ratio * idf
        retain_tfidf = r_ratio * idf

        contrast_score = (f_ratio + 1e-8) / (r_ratio + 1e-8)

        stats.append({
            "token_id": tid,
            "token_str": tokenizer.decode([tid]),
            "freq_forget": int(f_count),
            "freq_retain": int(r_count),
            "forget_ratio": float(f_ratio),
            "retain_ratio": float(r_ratio),
            "idf": float(idf),
            "forget_tfidf": float(forget_tfidf),
            "retain_tfidf": float(retain_tfidf),
            "contrast_score": float(contrast_score),
        })

    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--tokens-json", default=None)
    parser.add_argument("--out", default=None)

    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--model-name", default=None)

    # Starting thresholds for TOFU forget05 / retain95.
    parser.add_argument("--max-retain-ratio", type=float, default=0.002)
    parser.add_argument("--max-retain-count", type=int, default=5)
    parser.add_argument("--max-retain-tfidf", type=float, default=0.01)
    parser.add_argument("--min-contrast", type=float, default=20.0)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])

    tokens_json = Path(args.tokens_json) if args.tokens_json else out_dir / "semantic_tokens_raw_llm_bank.json"
    out_path = Path(args.out) if args.out else out_dir / "semantic_tokens.json"

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]

    print(f"[Load] candidate token file: {tokens_json}")

    with open(tokens_json, "r", encoding="utf-8") as f:
        token_data = json.load(f)

    candidate_tokens = token_data["semantic_tokens"]
    candidate_ids = [int(x["token_id"]) for x in candidate_tokens]

    print(f"[Load] candidate tokens before filter: {len(candidate_ids)}")

    print(f"[Load] tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[Load] forget split: {forget_split}")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")

    print(f"[Load] retain split: {retain_split}")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    stats = compute_token_stats(candidate_ids, tokenizer, forget_ds, retain_ds)
    stat_by_id = {int(x["token_id"]): x for x in stats}
    old_by_id = {int(x["token_id"]): x for x in candidate_tokens}

    kept = []
    protected = []

    for tid in candidate_ids:
        tid = int(tid)
        s = stat_by_id[tid]
        old = old_by_id[tid]

        protect_reason = None

        # If token appears too much in retain, protect it.
        if s["freq_retain"] > args.max_retain_count:
            protect_reason = "high_retain_count"

        if s["retain_ratio"] > args.max_retain_ratio:
            protect_reason = "high_retain_ratio"

        if s["retain_tfidf"] > args.max_retain_tfidf:
            protect_reason = "high_retain_tfidf"

        # If token is not much more forget-specific than retain-specific, protect it.
        if s["contrast_score"] < args.min_contrast:
            protect_reason = "low_forget_retain_contrast"

        merged = {
            **old,
            **s,
        }

        if protect_reason is None:
            merged["source"] = merged.get("source", "") + "+retain_tfidf_kept"
            kept.append(merged)
        else:
            merged["protect_reason"] = protect_reason
            protected.append(merged)

    kept.sort(
        key=lambda x: (
            -x.get("contrast_score", 0.0),
            -x.get("forget_tfidf", 0.0),
            x.get("retain_ratio", 0.0),
        )
    )

    output = {
        **token_data,
        "method": token_data.get("method", "unknown") + "+retain_tfidf_filter",
        "filter_config": {
            "max_retain_ratio": args.max_retain_ratio,
            "max_retain_count": args.max_retain_count,
            "max_retain_tfidf": args.max_retain_tfidf,
            "min_contrast": args.min_contrast,
        },
        "n_candidate_tokens_before_filter": len(candidate_ids),
        "n_semantic_tokens": len(kept),
        "n_protected_tokens": len(protected),
        "token_ids": [int(x["token_id"]) for x in kept],
        "token_strings": [x["token_str"] for x in kept],
        "semantic_tokens": kept,
        "protected_tokens": protected,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    protected_path = out_path.parent / "protected_tokens_retain_tfidf.json"
    with open(protected_path, "w", encoding="utf-8") as f:
        json.dump(protected, f, indent=2, ensure_ascii=False)

    print("\n[Done]")
    print(f"  before filter: {len(candidate_ids)}")
    print(f"  kept:          {len(kept)}")
    print(f"  protected:     {len(protected)}")
    print(f"  final file:    {out_path}")
    print(f"  protected log: {protected_path}")

    print("\nTop kept tokens:")
    for x in kept[:40]:
        print(
            f"  {x['token_id']:>8} | {repr(x['token_str'])} | "
            f"f={x['freq_forget']} r={x['freq_retain']} "
            f"contrast={x['contrast_score']:.2f} "
            f"retain_tfidf={x['retain_tfidf']:.6f}"
        )

    print("\nTop protected tokens:")
    for x in protected[:40]:
        print(
            f"  {x['token_id']:>8} | {repr(x['token_str'])} | "
            f"reason={x['protect_reason']} "
            f"f={x['freq_forget']} r={x['freq_retain']} "
            f"retain_ratio={x['retain_ratio']:.6f} "
            f"retain_tfidf={x['retain_tfidf']:.6f}"
        )


if __name__ == "__main__":
    main()