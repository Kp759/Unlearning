#!/usr/bin/env python3
"""
filter_forget_tokens_retain_tfidf_multi.py

Adaptive three-source token combiner:

  candidates = JSON tokens ∪ frequency tokens ∪ TF-IDF tokens

For each token:
  - recompute forget/retain document frequency on TOFU
  - compute forget/retain TF-IDF and forget/retain contrast
  - remove retain-risky tokens
  - rank remaining tokens
  - assign erase_strength based on source overlap and retain risk

Output:
  outputs/semantic_tokens.json

This output is compatible with:
  - original scripts/erase_embeddings.py through token_ids/token_strings
  - new scripts/erase_embeddings_adaptive.py through erase_strength
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def qa_text(row):
    return f"Question: {row['question']} Answer: {row['answer']}"


def doc_frequency(dataset, tokenizer):
    df = Counter()
    n = 0
    for row in dataset:
        ids = set(tokenizer.encode(qa_text(row), add_special_tokens=False))
        for tid in ids:
            df[int(tid)] += 1
        n += 1
    return df, n


def load_token_file(path, label, required=False):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"{label} token file not found: {path}")
        print(f"[warn] {label} file missing: {path}")
        return {"method": "missing", "semantic_tokens": [], "token_ids": []}

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "semantic_tokens" not in data:
        data["semantic_tokens"] = [{"token_id": int(x)} for x in data.get("token_ids", [])]
    return data


def token_id_of(x):
    if "token_id" in x:
        return int(x["token_id"])
    if "id" in x:
        return int(x["id"])
    raise KeyError(f"No token_id/id in entry: {x}")


def index_tokens(tokens):
    out = {}
    for t in tokens:
        tid = token_id_of(t)
        t = dict(t)
        t["token_id"] = tid
        out[tid] = t
    return out


def compute_stats(token_ids, tokenizer, forget_ds, retain_ds):
    f_df, n_f = doc_frequency(forget_ds, tokenizer)
    r_df, n_r = doc_frequency(retain_ds, tokenizer)
    total = n_f + n_r

    stats = {}
    for tid in sorted(set(map(int, token_ids))):
        fc = int(f_df.get(tid, 0))
        rc = int(r_df.get(tid, 0))
        fr = fc / max(1, n_f)
        rr = rc / max(1, n_r)
        idf = math.log((total + 1) / (fc + rc + 1)) + 1.0
        ftfidf = fr * idf
        rtfidf = rr * idf
        contrast = (fr + 1e-8) / (rr + 1e-8)
        stats[tid] = {
            "token_id": tid,
            "token_str": tokenizer.decode([tid]),
            "freq_forget": fc,
            "freq_retain": rc,
            "forget_ratio": float(fr),
            "retain_ratio": float(rr),
            "idf": float(idf),
            "forget_tfidf": float(ftfidf),
            "retain_tfidf": float(rtfidf),
            "contrast_score": float(contrast),
        }
    return stats


def protect_reason(s, max_retain_ratio, max_retain_count, max_retain_tfidf, min_contrast, require_contrast):
    if s["freq_retain"] > max_retain_count:
        return "high_retain_count"
    if s["retain_ratio"] > max_retain_ratio:
        return "high_retain_ratio"
    if s["retain_tfidf"] > max_retain_tfidf:
        return "high_retain_tfidf"
    if require_contrast and s["contrast_score"] < min_contrast:
        return "low_forget_retain_contrast"
    return None


def source_bucket(x):
    j = bool(x.get("in_json"))
    f = bool(x.get("in_frequency"))
    t = bool(x.get("in_tfidf"))

    if j and f and t:
        return 0
    if j and t:
        return 1
    if j and f:
        return 2
    if f and t:
        return 3
    if j:
        return 4
    if t:
        return 5
    if f:
        return 6
    return 9


def source_name(x):
    parts = []
    if x.get("in_json"):
        parts.append("json")
    if x.get("in_frequency"):
        parts.append("frequency")
    if x.get("in_tfidf"):
        parts.append("tfidf")
    return "+".join(parts) if parts else "unknown"


def base_strength(x, args):
    j = bool(x.get("in_json"))
    f = bool(x.get("in_frequency"))
    t = bool(x.get("in_tfidf"))

    if j and f and t:
        return args.both3_strength
    if j and t:
        return args.json_tfidf_strength
    if j and f:
        return args.json_freq_strength
    if f and t:
        return args.freq_tfidf_strength
    if j:
        return args.json_only_strength
    if t:
        return args.tfidf_only_strength
    if f:
        return args.freq_only_strength
    return args.min_strength


def compute_strength(x, args):
    strength = base_strength(x, args)

    if not args.no_adaptive_strength:
        contrast = max(0.0, float(x.get("contrast_score", 0.0)))
        retain_tfidf = max(0.0, float(x.get("retain_tfidf", 0.0)))
        retain_ratio = max(0.0, float(x.get("retain_ratio", 0.0)))

        contrast_factor = min(1.25, max(0.50, math.log1p(contrast) / math.log1p(50.0)))
        risk_factor = 1.0 / (1.0 + 50.0 * retain_tfidf + 20.0 * retain_ratio)
        strength = strength * contrast_factor * risk_factor

    return float(min(args.max_strength, max(args.min_strength, strength)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")

    parser.add_argument("--tokens-json", default=None)
    parser.add_argument("--freq-json", default=None)
    parser.add_argument("--tfidf-json", default=None)
    parser.add_argument("--out", default=None)

    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--model-name", default=None)

    parser.add_argument("--max-retain-ratio", type=float, default=0.0025)
    parser.add_argument("--max-retain-count", type=int, default=6)
    parser.add_argument("--max-retain-tfidf", type=float, default=0.012)
    parser.add_argument("--min-contrast", type=float, default=8.0)
    parser.add_argument("--max-final-tokens", type=int, default=650)

    parser.add_argument("--both3-strength", type=float, default=1.0)
    parser.add_argument("--json-tfidf-strength", type=float, default=0.85)
    parser.add_argument("--json-freq-strength", type=float, default=0.75)
    parser.add_argument("--freq-tfidf-strength", type=float, default=0.65)
    parser.add_argument("--json-only-strength", type=float, default=0.50)
    parser.add_argument("--tfidf-only-strength", type=float, default=0.45)
    parser.add_argument("--freq-only-strength", type=float, default=0.35)
    parser.add_argument("--min-strength", type=float, default=0.15)
    parser.add_argument("--max-strength", type=float, default=1.0)
    parser.add_argument("--no-adaptive-strength", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens_json = Path(args.tokens_json) if args.tokens_json else out_dir / "semantic_tokens_json_raw.json"
    freq_json = Path(args.freq_json) if args.freq_json else out_dir / "semantic_tokens_freq.json"
    tfidf_json = Path(args.tfidf_json) if args.tfidf_json else out_dir / "semantic_tokens_tfidf.json"
    out_path = Path(args.out) if args.out else out_dir / "semantic_tokens.json"

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]

    print("=" * 80)
    print("[Adaptive Filter] JSON + Frequency + TF-IDF + Retain Safety")
    print("=" * 80)
    print(f"JSON:   {tokens_json}")
    print(f"Freq:   {freq_json}")
    print(f"TF-IDF: {tfidf_json}")
    print(f"Out:    {out_path}")
    print(f"Forget: {forget_split}")
    print(f"Retain: {retain_split}")
    print(f"Model:  {model_name}")

    json_data = load_token_file(tokens_json, "JSON", required=True)
    freq_data = load_token_file(freq_json, "Frequency", required=False)
    tfidf_data = load_token_file(tfidf_json, "TF-IDF", required=False)

    json_by_id = index_tokens(json_data.get("semantic_tokens", []))
    freq_by_id = index_tokens(freq_data.get("semantic_tokens", []))
    tfidf_by_id = index_tokens(tfidf_data.get("semantic_tokens", []))

    candidate_ids = sorted(set(json_by_id) | set(freq_by_id) | set(tfidf_by_id))
    print(f"JSON candidates:  {len(json_by_id)}")
    print(f"Freq candidates:  {len(freq_by_id)}")
    print(f"TFIDF candidates: {len(tfidf_by_id)}")
    print(f"Union candidates: {len(candidate_ids)}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    print("[Stats] recomputing forget/retain document frequency and TF-IDF...")
    stat_by_id = compute_stats(candidate_ids, tokenizer, forget_ds, retain_ds)

    kept = []
    protected = []

    for tid in candidate_ids:
        tid = int(tid)
        in_json = tid in json_by_id
        in_freq = tid in freq_by_id
        in_tfidf = tid in tfidf_by_id
        n_sources = int(in_json) + int(in_freq) + int(in_tfidf)

        base = dict(json_by_id.get(tid) or tfidf_by_id.get(tid) or freq_by_id.get(tid) or {})
        s = stat_by_id[tid]

        merged = {
            **base,
            **s,
            "in_json": bool(in_json),
            "in_frequency": bool(in_freq),
            "in_tfidf": bool(in_tfidf),
            "n_sources": int(n_sources),
        }

        # Single-source tokens must pass contrast. Multi-source tokens are trusted more,
        # but still blocked if they are retain-heavy.
        require_contrast = n_sources < 2

        reason = protect_reason(
            s,
            args.max_retain_ratio,
            args.max_retain_count,
            args.max_retain_tfidf,
            args.min_contrast,
            require_contrast=require_contrast,
        )

        if reason is not None:
            merged["protect_reason"] = reason
            protected.append(merged)
            continue

        merged["hybrid_source"] = source_name(merged)
        merged["source"] = merged["hybrid_source"]
        merged["source_bucket"] = source_bucket(merged)
        merged["erase_strength"] = compute_strength(merged, args)

        forget_score = (
            2.0 * float(in_json)
            + 1.5 * float(in_tfidf)
            + 1.0 * float(in_freq)
            + 10.0 * float(s["forget_tfidf"])
            + 0.10 * math.log1p(float(s["contrast_score"]))
        )
        retain_risk = (
            20.0 * float(s["retain_tfidf"])
            + 10.0 * float(s["retain_ratio"])
            + 0.05 * float(s["freq_retain"])
        )
        merged["forget_score"] = float(forget_score)
        merged["retain_risk"] = float(retain_risk)
        merged["final_score"] = float(forget_score / (retain_risk + 1e-6))

        kept.append(merged)

    kept.sort(
        key=lambda x: (
            int(x.get("source_bucket", 9)),
            -float(x.get("final_score", 0.0)),
            -float(x.get("erase_strength", 0.0)),
            -float(x.get("contrast_score", 0.0)),
            -float(x.get("forget_tfidf", 0.0)),
            float(x.get("retain_ratio", 0.0)),
            int(x.get("token_id", 0)),
        )
    )

    kept_before_cap = len(kept)
    if args.max_final_tokens and args.max_final_tokens > 0 and len(kept) > args.max_final_tokens:
        overflow = kept[args.max_final_tokens:]
        for x in overflow:
            y = dict(x)
            y["protect_reason"] = "over_max_final_tokens_cap"
            protected.append(y)
        kept = kept[:args.max_final_tokens]

    source_counts = defaultdict(int)
    for x in kept:
        source_counts[x.get("hybrid_source", "unknown")] += 1

    protected_counts = defaultdict(int)
    for x in protected:
        protected_counts[x.get("protect_reason", "unknown")] += 1

    output = {
        "method": "adaptive_json_frequency_tfidf_retain_filter",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "n_json_tokens": len(json_by_id),
        "n_frequency_tokens": len(freq_by_id),
        "n_tfidf_tokens": len(tfidf_by_id),
        "n_candidate_tokens_before_filter": len(candidate_ids),
        "n_kept_before_cap": kept_before_cap,
        "n_semantic_tokens": len(kept),
        "n_protected_tokens": len(protected),
        "source_counts": dict(source_counts),
        "protected_reason_counts": dict(protected_counts),
        "filter_config": vars(args),
        "token_ids": [int(x["token_id"]) for x in kept],
        "token_strings": [x["token_str"] for x in kept],
        "semantic_tokens": kept,
        "protected_tokens": protected,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    protected_path = out_path.parent / "protected_tokens_adaptive_json_freq_tfidf.json"
    with open(protected_path, "w", encoding="utf-8") as f:
        json.dump(protected, f, indent=2, ensure_ascii=False)

    print("\n[done]")
    print(f"Kept final tokens: {len(kept)}")
    print(f"Protected tokens:  {len(protected)}")
    print(f"Source counts:     {dict(source_counts)}")
    print(f"Protected counts:  {dict(protected_counts)}")
    print(f"Final file:        {out_path}")
    print(f"Protected file:    {protected_path}")
    print("\nTop kept tokens:")
    for x in kept[:80]:
        print(
            f"{int(x['token_id']):>8} | {repr(x['token_str'])} | "
            f"{x.get('hybrid_source')} | "
            f"strength={float(x.get('erase_strength', 0.0)):.3f} | "
            f"f={int(x.get('freq_forget', 0))} r={int(x.get('freq_retain', 0))} | "
            f"contrast={float(x.get('contrast_score', 0.0)):.2f} | "
            f"score={float(x.get('final_score', 0.0)):.3f}"
        )


if __name__ == "__main__":
    main()
