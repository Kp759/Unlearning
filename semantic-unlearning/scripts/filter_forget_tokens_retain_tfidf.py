#!/usr/bin/env python3
"""
scripts/filter_forget_tokens_retain_tfidf.py

HYBRID JSON + FREQUENCY RETAIN-SAFE FILTER

Purpose:
  Final combiner for your hybrid approach.

Inputs:
  1) JSON/LLM aggressive candidate tokens:
       outputs/semantic_tokens_json_raw.json
  2) Frequency candidate tokens:
       outputs/semantic_tokens_freq.json

Output:
  Final erase token file consumed by erase_embeddings.py:
       outputs/semantic_tokens.json

Core idea:
  JSON/LLM tokens give strong forgetting.
  Frequency + retain-TFIDF protects retain performance.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def format_qa(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']} Answer: {row['answer']}"


def build_doc_counts(dataset, tokenizer) -> tuple[Counter, int]:
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


def compute_token_stats(token_ids: Iterable[int], tokenizer, forget_ds, retain_ds) -> List[Dict[str, Any]]:
    forget_df, n_forget = build_doc_counts(forget_ds, tokenizer)
    retain_df, n_retain = build_doc_counts(retain_ds, tokenizer)

    total_docs = n_forget + n_retain
    stats = []

    for tid in sorted({int(x) for x in token_ids}):
        f_count = forget_df.get(tid, 0)
        r_count = retain_df.get(tid, 0)

        f_ratio = f_count / max(1, n_forget)
        r_ratio = r_count / max(1, n_retain)

        total_df = f_count + r_count
        idf = math.log((total_docs + 1) / (total_df + 1)) + 1.0

        forget_tfidf = f_ratio * idf
        retain_tfidf = r_ratio * idf
        contrast_score = (f_ratio + 1e-8) / (r_ratio + 1e-8)

        stats.append(
            {
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
            }
        )

    return stats


def load_token_file(path: Path, label: str) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} token file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "semantic_tokens" not in data:
        raise ValueError(f"{label} token file has no 'semantic_tokens' field: {path}")

    return data


def token_id_of(x: Dict[str, Any]) -> int:
    if "token_id" in x:
        return int(x["token_id"])
    if "id" in x:
        return int(x["id"])
    raise KeyError(f"Token entry has no token_id/id field: {x}")


def index_tokens(tokens: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    by_id = {}
    for t in tokens:
        tid = token_id_of(t)
        t = dict(t)
        t["token_id"] = tid
        by_id[tid] = t
    return by_id


def safe_source_append(old_source: Any, suffix: str) -> str:
    old_source = str(old_source or "").strip()
    if not old_source:
        return suffix
    if suffix in old_source:
        return old_source
    return f"{old_source}+{suffix}"


def retain_protect_reason(
    stats: Dict[str, Any],
    max_retain_ratio: float,
    max_retain_count: int,
    max_retain_tfidf: float,
    min_contrast: float,
    require_contrast: bool,
) -> Optional[str]:
    if stats["freq_retain"] > max_retain_count:
        return "high_retain_count"
    if stats["retain_ratio"] > max_retain_ratio:
        return "high_retain_ratio"
    if stats["retain_tfidf"] > max_retain_tfidf:
        return "high_retain_tfidf"
    if require_contrast and stats["contrast_score"] < min_contrast:
        return "low_forget_retain_contrast"
    return None


def source_bucket(x: Dict[str, Any]) -> int:
    """
    Lower is better.
    Priority:
      0: appears in both JSON and frequency
      1: JSON-only but retain-safe
      2: frequency-only but retain-safe
    """
    source = x.get("hybrid_source", "")
    if source == "json+frequency":
        return 0
    if source == "json_only_retain_safe":
        return 1
    if source == "frequency_only_retain_safe":
        return 2
    return 9


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--tokens-json",
        default=None,
        help="JSON/LLM raw token file. Default: <output.dir>/semantic_tokens_json_raw.json",
    )
    parser.add_argument(
        "--freq-json",
        default=None,
        help="Frequency token file. Default: <output.dir>/semantic_tokens_freq.json if it exists.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Final hybrid output. Default: <output.dir>/semantic_tokens.json",
    )
    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--model-name", default=None)

    # Good starting thresholds for TOFU forget05 / retain95.
    parser.add_argument("--max-retain-ratio", type=float, default=0.002)
    parser.add_argument("--max-retain-count", type=int, default=5)
    parser.add_argument("--max-retain-tfidf", type=float, default=0.01)
    parser.add_argument("--min-contrast", type=float, default=20.0)
    parser.add_argument(
        "--max-final-tokens",
        type=int,
        default=400,
        help="Cap final erase tokens after ranking. Use <=0 for no cap.",
    )

    # For future erase_embeddings.py support.
    parser.add_argument("--both-strength", type=float, default=1.0)
    parser.add_argument("--json-only-strength", type=float, default=0.5)
    parser.add_argument("--freq-only-strength", type=float, default=0.3)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_json = Path(args.tokens_json) if args.tokens_json else out_dir / "semantic_tokens_json_raw.json"
    default_freq_json = out_dir / "semantic_tokens_freq.json"
    freq_json = Path(args.freq_json) if args.freq_json else default_freq_json
    out_path = Path(args.out) if args.out else out_dir / "semantic_tokens.json"

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]

    print("=" * 80)
    print("[Hybrid Filter] JSON/LLM candidates + Frequency + Retain TF-IDF safety")
    print("=" * 80)
    print(f"[Config] {args.config}")
    print(f"[JSON tokens] {tokens_json}")
    print(f"[Freq tokens] {freq_json}")
    print(f"[Output] {out_path}")
    print(f"[Forget split] {forget_split}")
    print(f"[Retain split] {retain_split}")
    print(f"[Tokenizer] {model_name}")
    print("=" * 80)

    json_data = load_token_file(tokens_json, "JSON/LLM")
    json_tokens = json_data.get("semantic_tokens", [])
    json_by_id = index_tokens(json_tokens)

    freq_data = None
    freq_by_id = {}
    if freq_json.exists():
        freq_data = load_token_file(freq_json, "Frequency")
        freq_tokens = freq_data.get("semantic_tokens", [])
        freq_by_id = index_tokens(freq_tokens)
    else:
        print(f"[WARN] Frequency file not found, running JSON-only retain filter: {freq_json}")

    candidate_ids = sorted(set(json_by_id.keys()) | set(freq_by_id.keys()))

    print(f"[Load] JSON/LLM candidate tokens: {len(json_by_id)}")
    print(f"[Load] Frequency candidate tokens: {len(freq_by_id)}")
    print(f"[Load] Union candidate tokens before retain filter: {len(candidate_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"[Load] TOFU forget split: {forget_split}")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")

    print(f"[Load] TOFU retain split: {retain_split}")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    print("[Stats] Computing forget/retain document frequencies and TF-IDF...")
    stats = compute_token_stats(candidate_ids, tokenizer, forget_ds, retain_ds)
    stat_by_id = {int(x["token_id"]): x for x in stats}

    kept = []
    protected = []

    for tid in candidate_ids:
        tid = int(tid)
        in_json = tid in json_by_id
        in_frequency = tid in freq_by_id

        # Prefer JSON metadata because it carries source_string/record-bank provenance.
        # Fall back to frequency metadata if token is frequency-only.
        base = dict(json_by_id.get(tid) or freq_by_id.get(tid) or {})
        s = stat_by_id[tid]

        merged = {
            **base,
            **s,
            "in_json": bool(in_json),
            "in_frequency": bool(in_frequency),
        }

        if in_json and in_frequency:
            # Strongest group: independently selected by both approaches.
            reason = retain_protect_reason(
                s,
                args.max_retain_ratio,
                args.max_retain_count,
                args.max_retain_tfidf,
                args.min_contrast,
                require_contrast=False,
            )
            if reason is None:
                merged["hybrid_source"] = "json+frequency"
                merged["source"] = safe_source_append(merged.get("source"), "hybrid_json_frequency")
                merged["erase_strength"] = float(args.both_strength)
                kept.append(merged)
            else:
                merged["protect_reason"] = reason
                protected.append(merged)

        elif in_json:
            # JSON-only tokens can be powerful but dangerous; require retain-safety and contrast.
            reason = retain_protect_reason(
                s,
                args.max_retain_ratio,
                args.max_retain_count,
                args.max_retain_tfidf,
                args.min_contrast,
                require_contrast=True,
            )
            if reason is None:
                merged["hybrid_source"] = "json_only_retain_safe"
                merged["source"] = safe_source_append(merged.get("source"), "hybrid_json_only_retain_safe")
                merged["erase_strength"] = float(args.json_only_strength)
                kept.append(merged)
            else:
                merged["protect_reason"] = reason
                protected.append(merged)

        elif in_frequency:
            # Frequency-only tokens are usually safer but weaker; keep if contrast is strong.
            reason = retain_protect_reason(
                s,
                args.max_retain_ratio,
                args.max_retain_count,
                args.max_retain_tfidf,
                args.min_contrast,
                require_contrast=True,
            )
            if reason is None:
                merged["hybrid_source"] = "frequency_only_retain_safe"
                merged["source"] = safe_source_append(merged.get("source"), "hybrid_frequency_only_retain_safe")
                merged["erase_strength"] = float(args.freq_only_strength)
                kept.append(merged)
            else:
                merged["protect_reason"] = reason
                protected.append(merged)

    kept.sort(
        key=lambda x: (
            source_bucket(x),
            -float(x.get("contrast_score", 0.0)),
            -float(x.get("forget_tfidf", 0.0)),
            float(x.get("retain_ratio", 0.0)),
            -int(x.get("freq_forget", 0)),
            int(x.get("token_id", 0)),
        )
    )

    if args.max_final_tokens and args.max_final_tokens > 0:
        kept_before_cap = len(kept)
        overflow = kept[args.max_final_tokens :]
        for x in overflow:
            x = dict(x)
            x["protect_reason"] = "over_max_final_tokens_cap"
            protected.append(x)
        kept = kept[: args.max_final_tokens]
        print(f"[Cap] kept {len(kept)} / {kept_before_cap} after max-final-tokens={args.max_final_tokens}")

    source_counts = defaultdict(int)
    for x in kept:
        source_counts[x.get("hybrid_source", "unknown")] += 1

    protected_counts = defaultdict(int)
    for x in protected:
        protected_counts[x.get("protect_reason", "unknown")] += 1

    output = {
        "method": "hybrid_json_frequency_retain_tfidf_filter",
        "base_json_method": json_data.get("method", "unknown"),
        "base_frequency_method": freq_data.get("method", "missing") if freq_data else "missing",
        "filter_config": {
            "max_retain_ratio": args.max_retain_ratio,
            "max_retain_count": args.max_retain_count,
            "max_retain_tfidf": args.max_retain_tfidf,
            "min_contrast": args.min_contrast,
            "max_final_tokens": args.max_final_tokens,
            "both_strength": args.both_strength,
            "json_only_strength": args.json_only_strength,
            "freq_only_strength": args.freq_only_strength,
        },
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "n_json_tokens": len(json_by_id),
        "n_frequency_tokens": len(freq_by_id),
        "n_candidate_tokens_before_filter": len(candidate_ids),
        "n_semantic_tokens": len(kept),
        "n_protected_tokens": len(protected),
        "source_counts": dict(source_counts),
        "protected_reason_counts": dict(protected_counts),
        "token_ids": [int(x["token_id"]) for x in kept],
        "token_strings": [x["token_str"] for x in kept],
        "semantic_tokens": kept,
        "protected_tokens": protected,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    protected_path = out_path.parent / "protected_tokens_hybrid_retain_tfidf.json"
    with open(protected_path, "w", encoding="utf-8") as f:
        json.dump(protected, f, indent=2, ensure_ascii=False)

    print("\n[Done]")
    print(f"  JSON/LLM tokens: {len(json_by_id)}")
    print(f"  Frequency tokens: {len(freq_by_id)}")
    print(f"  Union candidates: {len(candidate_ids)}")
    print(f"  Kept final tokens: {len(kept)}")
    print(f"  Protected tokens: {len(protected)}")
    print(f"  Source counts: {dict(source_counts)}")
    print(f"  Protected reason counts: {dict(protected_counts)}")
    print(f"  Final file: {out_path}")
    print(f"  Protected log: {protected_path}")

    print("\nTop kept tokens:")
    for x in kept[:60]:
        print(
            f" {int(x['token_id']):>8} | {repr(x['token_str'])} | "
            f"{x.get('hybrid_source')} | "
            f"strength={float(x.get('erase_strength', 1.0)):.2f} | "
            f"f={int(x.get('freq_forget', 0))} r={int(x.get('freq_retain', 0))} | "
            f"contrast={float(x.get('contrast_score', 0.0)):.2f} | "
            f"retain_tfidf={float(x.get('retain_tfidf', 0.0)):.6f}"
        )

    print("\nTop protected tokens:")
    for x in protected[:60]:
        print(
            f" {int(x['token_id']):>8} | {repr(x.get('token_str', ''))} | "
            f"reason={x.get('protect_reason')} | "
            f"in_json={x.get('in_json')} in_freq={x.get('in_frequency')} | "
            f"f={int(x.get('freq_forget', 0))} r={int(x.get('freq_retain', 0))} | "
            f"retain_ratio={float(x.get('retain_ratio', 0.0)):.6f} | "
            f"retain_tfidf={float(x.get('retain_tfidf', 0.0)):.6f}"
        )


if __name__ == "__main__":
    main()
