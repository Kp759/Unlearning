#!/usr/bin/env python3
"""
Overlap-Aware JSON-TFIDF Token Selector

Idea:
  - Use JSON/LLM forget tokens as the ONLY candidate source.
  - If a JSON token is forget-unique / nearly absent in retain -> strong erase.
  - If a JSON token overlaps with retain -> keep only if retain-TFIDF is low
    and forget/retain contrast is high -> weak erase.
  - Retain-important overlap tokens are protected.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def format_qa(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer: {row['answer']}"


def token_id_of(x: Dict[str, Any]) -> int:
    if "token_id" in x:
        return int(x["token_id"])
    if "id" in x:
        return int(x["id"])
    raise KeyError(f"Token entry has no token_id/id field: {x}")


def load_json_tokens(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSON token file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "semantic_tokens" not in data:
        raise ValueError(f"{path} does not contain semantic_tokens")

    tokens = []
    seen = set()

    for t in data["semantic_tokens"]:
        tid = token_id_of(t)
        if tid in seen:
            continue
        seen.add(tid)

        t = dict(t)
        t["token_id"] = tid
        tokens.append(t)

    return tokens


def build_doc_counts(dataset, tokenizer) -> tuple[Counter, int]:
    """
    Token document frequency.
    A token counts once per QA pair, not once per occurrence.
    """
    df = Counter()
    n_docs = 0

    for row in dataset:
        text = format_qa(row)
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            df[int(tid)] += 1
        n_docs += 1

    return df, n_docs


def compute_stats(
    token_ids: Iterable[int],
    tokenizer,
    forget_ds,
    retain_ds,
) -> Dict[int, Dict[str, Any]]:
    forget_df, n_forget = build_doc_counts(forget_ds, tokenizer)
    retain_df, n_retain = build_doc_counts(retain_ds, tokenizer)

    total_docs = n_forget + n_retain
    stats = {}

    for tid in sorted({int(x) for x in token_ids}):
        f_count = int(forget_df.get(tid, 0))
        r_count = int(retain_df.get(tid, 0))

        f_ratio = f_count / max(1, n_forget)
        r_ratio = r_count / max(1, n_retain)

        total_df = f_count + r_count
        idf = math.log((total_docs + 1) / (total_df + 1)) + 1.0

        forget_tfidf = f_ratio * idf
        retain_tfidf = r_ratio * idf
        contrast = (f_ratio + 1e-8) / (r_ratio + 1e-8)

        stats[tid] = {
            "token_id": tid,
            "token_str": tokenizer.decode([tid]),
            "freq_forget": f_count,
            "freq_retain": r_count,
            "forget_ratio": float(f_ratio),
            "retain_ratio": float(r_ratio),
            "idf": float(idf),
            "forget_tfidf": float(forget_tfidf),
            "retain_tfidf": float(retain_tfidf),
            "contrast_score": float(contrast),
        }

    return stats


def is_json_unique_strong(s: Dict[str, Any], args) -> bool:
    """
    Very safe aggressive-erasure group.
    Token is from JSON forget bank and is absent/nearly absent in retain.
    """
    return (
        s["freq_forget"] >= args.min_forget_count
        and s["freq_retain"] <= args.strong_max_retain_count
        and s["retain_ratio"] <= args.strong_max_retain_ratio
        and s["retain_tfidf"] <= args.strong_max_retain_tfidf
    )


def is_overlap_tfidf_safe(s: Dict[str, Any], args) -> bool:
    """
    Overlapping JSON token.
    Keep only if it is still forget-dominant and weakly retain-associated.
    """
    return (
        s["freq_forget"] >= args.min_forget_count
        and s["freq_retain"] <= args.overlap_max_retain_count
        and s["retain_ratio"] <= args.overlap_max_retain_ratio
        and s["retain_tfidf"] <= args.overlap_max_retain_tfidf
        and s["contrast_score"] >= args.overlap_min_contrast
    )


def rank_strong(x: Dict[str, Any]):
    return (
        -float(x.get("forget_tfidf", 0.0)),
        -float(x.get("contrast_score", 0.0)),
        -int(x.get("freq_forget", 0)),
        float(x.get("retain_tfidf", 0.0)),
        int(x.get("token_id", 0)),
    )


def rank_overlap(x: Dict[str, Any]):
    return (
        -float(x.get("contrast_score", 0.0)),
        -float(x.get("forget_tfidf", 0.0)),
        float(x.get("retain_tfidf", 0.0)),
        int(x.get("freq_retain", 0)),
        int(x.get("token_id", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--tokens-json", default=None)
    parser.add_argument("--out", default=None)

    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--model-name", default=None)

    # Shared
    parser.add_argument("--min-forget-count", type=int, default=1)

    # Strong JSON-unique group
    parser.add_argument("--strong-max-retain-count", type=int, default=0)
    parser.add_argument("--strong-max-retain-ratio", type=float, default=0.0005)
    parser.add_argument("--strong-max-retain-tfidf", type=float, default=0.002)
    parser.add_argument("--max-strong-tokens", type=int, default=800)
    parser.add_argument("--strong-strength", type=float, default=1.0)

    # Overlap TF-IDF-safe group
    parser.add_argument("--overlap-max-retain-count", type=int, default=5)
    parser.add_argument("--overlap-max-retain-ratio", type=float, default=0.002)
    parser.add_argument("--overlap-max-retain-tfidf", type=float, default=0.01)
    parser.add_argument("--overlap-min-contrast", type=float, default=20.0)
    parser.add_argument("--max-overlap-tokens", type=int, default=300)
    parser.add_argument("--overlap-strength", type=float, default=0.35)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_json_path = (
        Path(args.tokens_json)
        if args.tokens_json
        else out_dir / "semantic_tokens_json_raw.json"
    )

    out_path = (
        Path(args.out)
        if args.out
        else out_dir / "semantic_tokens_overlapaware_json_tfidf.json"
    )

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]

    print("=" * 90)
    print("[OA-JTE] Overlap-Aware JSON-TFIDF token selection")
    print("=" * 90)
    print(f"[Config]       {args.config}")
    print(f"[JSON tokens]  {tokens_json_path}")
    print(f"[Output]       {out_path}")
    print(f"[Forget split] {forget_split}")
    print(f"[Retain split] {retain_split}")
    print(f"[Tokenizer]    {model_name}")
    print("=" * 90)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_ids = set(int(x) for x in tokenizer.all_special_ids)

    json_tokens = load_json_tokens(tokens_json_path)
    json_by_id = {int(t["token_id"]): t for t in json_tokens}
    candidate_ids = sorted(json_by_id.keys())

    print(f"[Load] JSON candidate tokens: {len(candidate_ids)}")

    print("[Load] TOFU forget split...")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")

    print("[Load] TOFU retain split...")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    print("[Stats] Computing forget/retain DF + TF-IDF only for JSON tokens...")
    stats_by_id = compute_stats(candidate_ids, tokenizer, forget_ds, retain_ds)

    strong = []
    overlap_safe = []
    protected = []

    for tid in candidate_ids:
        base = dict(json_by_id[tid])
        s = stats_by_id[tid]

        merged = {
            **base,
            **s,
            "candidate_source": "json_only",
        }

        if tid in special_ids:
            merged["group"] = "protected"
            merged["protect_reason"] = "special_token"
            protected.append(merged)
            continue

        if is_json_unique_strong(s, args):
            merged["group"] = "json_unique_strong"
            merged["erase_strength"] = float(args.strong_strength)
            strong.append(merged)

        elif is_overlap_tfidf_safe(s, args):
            merged["group"] = "json_overlap_tfidf_safe"
            merged["erase_strength"] = float(args.overlap_strength)
            overlap_safe.append(merged)

        else:
            merged["group"] = "protected"
            if s["freq_retain"] > args.overlap_max_retain_count:
                reason = "high_retain_count"
            elif s["retain_ratio"] > args.overlap_max_retain_ratio:
                reason = "high_retain_ratio"
            elif s["retain_tfidf"] > args.overlap_max_retain_tfidf:
                reason = "high_retain_tfidf"
            elif s["contrast_score"] < args.overlap_min_contrast:
                reason = "low_forget_retain_contrast"
            else:
                reason = "not_selected"
            merged["protect_reason"] = reason
            protected.append(merged)

    strong.sort(key=rank_strong)
    overlap_safe.sort(key=rank_overlap)

    if args.max_strong_tokens and args.max_strong_tokens > 0:
        overflow = strong[args.max_strong_tokens :]
        for x in overflow:
            x = dict(x)
            x["group"] = "protected"
            x["protect_reason"] = "over_max_strong_tokens"
            protected.append(x)
        strong = strong[: args.max_strong_tokens]

    if args.max_overlap_tokens and args.max_overlap_tokens > 0:
        overflow = overlap_safe[args.max_overlap_tokens :]
        for x in overflow:
            x = dict(x)
            x["group"] = "protected"
            x["protect_reason"] = "over_max_overlap_tokens"
            protected.append(x)
        overlap_safe = overlap_safe[: args.max_overlap_tokens]

    kept = strong + overlap_safe

    group_counts = defaultdict(int)
    for x in kept:
        group_counts[x["group"]] += 1

    protected_counts = defaultdict(int)
    for x in protected:
        protected_counts[x.get("protect_reason", "unknown")] += 1

    output = {
        "method": "overlap_aware_json_tfidf_erasure",
        "description": (
            "JSON-only forget candidates; retain TF-IDF used only to split/protect "
            "JSON tokens that overlap retain."
        ),
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "n_json_candidate_tokens": len(candidate_ids),
        "n_semantic_tokens": len(kept),
        "n_protected_tokens": len(protected),
        "group_counts": dict(group_counts),
        "protected_reason_counts": dict(protected_counts),
        "filter_config": {
            "min_forget_count": args.min_forget_count,
            "strong_max_retain_count": args.strong_max_retain_count,
            "strong_max_retain_ratio": args.strong_max_retain_ratio,
            "strong_max_retain_tfidf": args.strong_max_retain_tfidf,
            "max_strong_tokens": args.max_strong_tokens,
            "strong_strength": args.strong_strength,
            "overlap_max_retain_count": args.overlap_max_retain_count,
            "overlap_max_retain_ratio": args.overlap_max_retain_ratio,
            "overlap_max_retain_tfidf": args.overlap_max_retain_tfidf,
            "overlap_min_contrast": args.overlap_min_contrast,
            "max_overlap_tokens": args.max_overlap_tokens,
            "overlap_strength": args.overlap_strength,
        },
        "token_ids": [int(x["token_id"]) for x in kept],
        "token_strings": [x["token_str"] for x in kept],
        "semantic_tokens": kept,
        "protected_tokens": protected,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Also write as semantic_tokens.json so old erase_embeddings.py can still use it.
    semantic_tokens_path = out_dir / "semantic_tokens.json"
    with open(semantic_tokens_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    strong_path = out_dir / "oa_json_unique_strong_tokens.json"
    overlap_path = out_dir / "oa_json_overlap_tfidf_safe_tokens.json"
    protected_path = out_dir / "oa_json_protected_overlap_tokens.json"

    with open(strong_path, "w", encoding="utf-8") as f:
        json.dump(strong, f, indent=2, ensure_ascii=False)

    with open(overlap_path, "w", encoding="utf-8") as f:
        json.dump(overlap_safe, f, indent=2, ensure_ascii=False)

    with open(protected_path, "w", encoding="utf-8") as f:
        json.dump(protected, f, indent=2, ensure_ascii=False)

    print("\n[Done]")
    print(f"JSON candidates:          {len(candidate_ids)}")
    print(f"Kept total:               {len(kept)}")
    print(f"  json_unique_strong:     {len(strong)}")
    print(f"  json_overlap_tfidf_safe:{len(overlap_safe)}")
    print(f"Protected:                {len(protected)}")
    print(f"Group counts:             {dict(group_counts)}")
    print(f"Protected reason counts:  {dict(protected_counts)}")
    print(f"Main output:              {out_path}")
    print(f"semantic_tokens.json:     {semantic_tokens_path}")
    print(f"Strong group file:        {strong_path}")
    print(f"Overlap group file:       {overlap_path}")
    print(f"Protected group file:     {protected_path}")

    print("\nTop strong JSON-unique tokens:")
    for x in strong[:40]:
        print(
            f"{int(x['token_id']):>8} | {repr(x['token_str'])} | "
            f"f={x['freq_forget']} r={x['freq_retain']} | "
            f"f_tfidf={x['forget_tfidf']:.6f} r_tfidf={x['retain_tfidf']:.6f} | "
            f"contrast={x['contrast_score']:.2f} | strength={x['erase_strength']}"
        )

    print("\nTop overlap TF-IDF-safe tokens:")
    for x in overlap_safe[:40]:
        print(
            f"{int(x['token_id']):>8} | {repr(x['token_str'])} | "
            f"f={x['freq_forget']} r={x['freq_retain']} | "
            f"f_tfidf={x['forget_tfidf']:.6f} r_tfidf={x['retain_tfidf']:.6f} | "
            f"contrast={x['contrast_score']:.2f} | strength={x['erase_strength']}"
        )


if __name__ == "__main__":
    main()
