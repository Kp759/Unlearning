#!/usr/bin/env python3
"""
Build answer-token TF-IDF file for masked LM-head GA/GD unlearning.

Output is compatible with:
  scripts/embedding_ga_gd_input_output_unlearn.py --forget-token-json ...

Core idea:
  forget answers -> tokenize answer only
  retain answers -> tokenize answer only
  keep tokens that are frequent/specific in forget answers but rare in retain answers
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


COMMON_PROTECT = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "was", "were", "are", "be", "been", "being", "as", "at", "from",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "my", "our", "your", "i", "you", "we", "me",
    "will", "would", "can", "could", "may", "might", "should", "has", "have",
    "had", "do", "does", "did", "not", "no", "yes", "about", "into", "during",
    "through", "after", "before", "between", "among", "within", "without",
}


def normalize_token(token_str: str) -> str:
    return (
        str(token_str)
        .strip()
        .lower()
        .replace("Ġ", "")
        .replace("▁", "")
        .replace("Ċ", "")
    )


def is_bad_token(token_str: str, min_token_chars: int, protect_common: bool) -> bool:
    clean = normalize_token(token_str)
    if len(clean) < min_token_chars:
        return True
    if protect_common and clean in COMMON_PROTECT:
        return True
    if not any(ch.isalnum() for ch in clean):
        return True
    return False


def encode_answer(tokenizer, answer: str) -> List[int]:
    # Leading space matches the training/eval prompt formatting used in your scripts.
    return [int(x) for x in tokenizer.encode(" " + str(answer).strip(), add_special_tokens=False)]


def count_answer_tokens(dataset, tokenizer):
    token_count = Counter()
    doc_count = Counter()
    examples = []

    for idx, row in enumerate(dataset):
        ids = encode_answer(tokenizer, row["answer"])
        token_count.update(ids)
        unique_ids = sorted(set(int(x) for x in ids))
        for tid in unique_ids:
            doc_count[tid] += 1

        examples.append(
            {
                "idx": int(idx),
                "question": str(row.get("question", "")),
                "answer": str(row.get("answer", "")),
                "answer_token_ids": unique_ids,
            }
        )

    return token_count, doc_count, examples


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--out", default="outputs/semantic_tokens_answer_tfidf.json")

    parser.add_argument("--max-retain-answer-ratio", type=float, default=0.004)
    parser.add_argument("--max-retain-answer-count", type=int, default=8)
    parser.add_argument("--max-retain-answer-tfidf", type=float, default=0.012)
    parser.add_argument("--min-contrast", type=float, default=5.0)
    parser.add_argument("--min-forget-answer-count", type=int, default=1)
    parser.add_argument("--min-token-chars", type=int, default=2)
    parser.add_argument("--max-final-tokens", type=int, default=1000)

    parser.add_argument("--protect-common-tokens", action="store_true", default=True)
    parser.add_argument("--no-protect-common-tokens", dest="protect_common_tokens", action="store_false")
    parser.add_argument("--include-special-tokens", action="store_true")

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]

    print("=" * 80)
    print("[Build] Answer-token TF-IDF forget-token file")
    print("=" * 80)
    print(f"config: {args.config}")
    print(f"model/tokenizer: {model_name}")
    print(f"forget split: {forget_split}")
    print(f"retain split: {retain_split}")
    print(f"out: {args.out}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    special_ids = {
        x for x in [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id,
        ]
        if x is not None
    }

    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    f_count, f_doc, forget_examples = count_answer_tokens(forget_ds, tokenizer)
    r_count, r_doc, retain_examples = count_answer_tokens(retain_ds, tokenizer)

    n_forget = len(forget_examples)
    n_retain = len(retain_examples)
    total_docs = n_forget + n_retain

    kept = []
    protected = []

    for tid, fc in f_count.items():
        tid = int(tid)
        token_str = tokenizer.decode([tid])
        rc = int(r_count.get(tid, 0))
        fdc = int(f_doc.get(tid, 0))
        rdc = int(r_doc.get(tid, 0))

        f_ratio = fdc / max(1, n_forget)
        r_ratio = rdc / max(1, n_retain)
        total_df = fdc + rdc
        idf = math.log((total_docs + 1) / (total_df + 1)) + 1.0

        forget_tfidf = f_ratio * idf
        retain_tfidf = r_ratio * idf
        contrast = (f_ratio + 1e-8) / (r_ratio + 1e-8)

        row: Dict[str, Any] = {
            "token_id": tid,
            "token_str": token_str,
            "source": "answer_tfidf_forget_token",
            "bucket": "answer_tfidf_selected",
            "is_answer_token": True,
            "is_residual_answer": True,
            "is_json": False,
            "freq_forget": fdc,
            "freq_retain": rdc,
            "forget_answer_count": int(fc),
            "retain_answer_count": rc,
            "forget_answer_doc_count": fdc,
            "retain_answer_doc_count": rdc,
            "forget_answer_ratio": float(f_ratio),
            "retain_answer_ratio": float(r_ratio),
            "idf": float(idf),
            "forget_tfidf": float(forget_tfidf),
            "retain_tfidf": float(retain_tfidf),
            "contrast_score": float(contrast),
            "erase_strength": 1.0,
            "output_strength": 1.0,
            "edit_input": False,
            "edit_lm_head": True,
            "recommended_edit": "lm_head_only",
        }

        reason = None
        if (not args.include_special_tokens) and tid in special_ids:
            reason = "special_token"
        elif int(fc) < args.min_forget_answer_count:
            reason = "low_forget_answer_count"
        elif is_bad_token(token_str, args.min_token_chars, args.protect_common_tokens):
            reason = "bad_or_common_token"
        elif rc > args.max_retain_answer_count:
            reason = "high_retain_answer_count"
        elif r_ratio > args.max_retain_answer_ratio:
            reason = "high_retain_answer_ratio"
        elif retain_tfidf > args.max_retain_answer_tfidf:
            reason = "high_retain_answer_tfidf"
        elif contrast < args.min_contrast:
            reason = "low_forget_retain_contrast"

        if reason is None:
            kept.append(row)
        else:
            protected.append({**row, "protect_reason": reason})

    kept.sort(
        key=lambda x: (
            -float(x["forget_tfidf"]),
            -float(x["contrast_score"]),
            int(x["retain_answer_count"]),
            -int(x["forget_answer_count"]),
            int(x["token_id"]),
        )
    )

    kept_before_cap = len(kept)
    if args.max_final_tokens and args.max_final_tokens > 0 and len(kept) > args.max_final_tokens:
        for x in kept[args.max_final_tokens:]:
            protected.append({**x, "protect_reason": "over_max_final_tokens_cap"})
        kept = kept[: args.max_final_tokens]

    selected_ids = {int(x["token_id"]) for x in kept}

    covered_examples = 0
    covered_answer_token_occurrences = 0
    total_answer_token_occurrences = 0
    covered_answer_token_types = 0
    total_answer_token_types = 0
    coverage_by_example = []

    for ex in forget_examples:
        ids = [int(x) for x in ex["answer_token_ids"]]
        hit = sorted(set(ids) & selected_ids)
        if hit:
            covered_examples += 1

        # Type coverage by example.
        total_answer_token_types += len(set(ids))
        covered_answer_token_types += len(hit)

        # Occurrence coverage by example.
        occ_ids = encode_answer(tokenizer, ex["answer"])
        total_answer_token_occurrences += len(occ_ids)
        covered_answer_token_occurrences += sum(1 for x in occ_ids if int(x) in selected_ids)

        coverage_by_example.append(
            {
                "idx": ex["idx"],
                "question": ex["question"],
                "answer": ex["answer"],
                "answer_token_ids": ids,
                "selected_token_ids": hit,
                "selected_token_strings": [tokenizer.decode([x]) for x in hit],
                "n_selected_tokens": len(hit),
            }
        )

    protected_reason_counts = defaultdict(int)
    for x in protected:
        protected_reason_counts[str(x.get("protect_reason", "unknown"))] += 1

    out = {
        "method": "answer_tfidf_forget_tokens",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "filter_config": vars(args),
        "n_forget_examples": n_forget,
        "n_retain_examples": n_retain,
        "n_kept_before_cap": kept_before_cap,
        "n_semantic_tokens": len(kept),
        "n_protected_tokens": len(protected),
        "protected_reason_counts": dict(protected_reason_counts),
        "forget_answer_coverage": {
            "covered_examples": covered_examples,
            "total_forget_examples": n_forget,
            "example_coverage_rate": covered_examples / max(1, n_forget),
            "covered_answer_token_occurrences": covered_answer_token_occurrences,
            "total_answer_token_occurrences": total_answer_token_occurrences,
            "answer_token_occurrence_coverage_rate": covered_answer_token_occurrences / max(1, total_answer_token_occurrences),
            "covered_answer_token_types": covered_answer_token_types,
            "total_answer_token_types": total_answer_token_types,
            "answer_token_type_coverage_rate": covered_answer_token_types / max(1, total_answer_token_types),
        },
        "token_ids": [int(x["token_id"]) for x in kept],
        "token_strings": [str(x["token_str"]) for x in kept],
        "semantic_tokens": kept,
        "protected_tokens": protected,
        "coverage_by_forget_example": coverage_by_example,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    protected_path = out_path.parent / (out_path.stem + "_protected.json")
    with open(protected_path, "w", encoding="utf-8") as f:
        json.dump(protected, f, indent=2, ensure_ascii=False)

    print("\n[Done]")
    print(f"saved: {out_path}")
    print(f"protected log: {protected_path}")
    print(f"kept tokens: {len(kept)} / before cap {kept_before_cap}")
    print(f"protected tokens: {len(protected)}")
    print(f"protected reasons: {dict(protected_reason_counts)}")
    print(f"coverage: {out['forget_answer_coverage']}")

    print("\nTop 100 selected tokens:")
    for r in kept[:100]:
        print(
            f"{r['token_id']:>8} | {repr(r['token_str'])} | "
            f"fc={r['forget_answer_count']} rc={r['retain_answer_count']} | "
            f"fdoc={r['forget_answer_doc_count']} rdoc={r['retain_answer_doc_count']} | "
            f"f_tfidf={r['forget_tfidf']:.6f} r_tfidf={r['retain_tfidf']:.6f} | "
            f"contrast={r['contrast_score']:.2f}"
        )


if __name__ == "__main__":
    main()
