#!/usr/bin/env python3
"""
scripts/build_forget_answer_tokens_policy.py

Build forget-answer token file with retain-count-aware edit policy:

  retain_answer_count == 0:
      zero input embedding + zero lm_head

  retain_answer_count 1-2:
      zero lm_head only

  retain_answer_count 3-8:
      input embedding only, either adaptive_mean or zero depending on --mid-input-mode

  retain_answer_count > 8:
      protect / do not edit

This script only builds the token JSON.
Apply with scripts/apply_token_policy_edits.py
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import List

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


COMMON_PROTECT = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "was", "were", "are", "be", "been", "being", "as", "at", "from",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "my", "our", "your", "i", "you", "we", "me",
    "will", "would", "can", "could", "may", "might", "should", "has", "have",
    "had", "do", "does", "did", "not", "no", "yes"
}


def norm_token(s: str) -> str:
    return str(s).strip().lower().replace("Ġ", "").replace("▁", "")


def is_bad_token(token_str: str, min_token_chars: int, protect_common: bool) -> bool:
    clean = norm_token(token_str)
    if len(clean) < min_token_chars:
        return True
    if protect_common and clean in COMMON_PROTECT:
        return True
    if not any(ch.isalnum() for ch in clean):
        return True
    return False


def encode_answer(tok, answer: str) -> List[int]:
    return [int(x) for x in tok.encode(" " + str(answer).strip(), add_special_tokens=False)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--forget-split", default="forget05")
    ap.add_argument("--retain-split", default="retain95")
    ap.add_argument("--out", default="outputs/semantic_tokens_forget_answer_tokens_policy.json")
    ap.add_argument("--model-name", default=None)

    ap.add_argument("--lmhead-only-max-retain-answer-count", type=int, default=2)
    ap.add_argument("--input-only-max-retain-answer-count", type=int, default=8)
    ap.add_argument("--mid-input-mode", choices=["adaptive_mean", "zero"], default="adaptive_mean")

    ap.add_argument("--min-token-chars", type=int, default=2)
    ap.add_argument("--protect-common-tokens", action="store_true", default=True)
    ap.add_argument("--no-protect-common-tokens", dest="protect_common_tokens", action="store_false")
    ap.add_argument("--min-forget-answer-count", type=int, default=1)
    ap.add_argument("--max-final-tokens", type=int, default=0, help="0 means no cap")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)

    forget = load_dataset("locuslab/TOFU", name=args.forget_split, split="train")
    retain = load_dataset("locuslab/TOFU", name=args.retain_split, split="train")

    forget_answer_count = Counter()
    forget_answer_doc = Counter()
    retain_answer_count = Counter()
    retain_answer_doc = Counter()
    forget_example_tokens = []

    for i, row in enumerate(forget):
        ids = encode_answer(tok, row["answer"])
        forget_answer_count.update(ids)
        for x in set(ids):
            forget_answer_doc[int(x)] += 1
        forget_example_tokens.append({
            "idx": int(i),
            "answer": str(row["answer"]),
            "answer_token_ids": sorted(set(int(x) for x in ids)),
        })

    for row in retain:
        ids = encode_answer(tok, row["answer"])
        retain_answer_count.update(ids)
        for x in set(ids):
            retain_answer_doc[int(x)] += 1

    rows = []
    protected = []

    for tid, fc in forget_answer_count.items():
        tid = int(tid)
        token_str = tok.decode([tid])
        rc = int(retain_answer_count.get(tid, 0))
        fdoc = int(forget_answer_doc.get(tid, 0))
        rdoc = int(retain_answer_doc.get(tid, 0))

        base = {
            "token_id": tid,
            "token_str": token_str,
            "source": "forget_answer_token_policy",
            "is_answer_token": True,
            "is_residual_answer": True,
            "is_json": False,
            "forget_answer_count": int(fc),
            "forget_answer_doc_count": fdoc,
            "retain_answer_count": rc,
            "retain_answer_doc_count": rdoc,
            "freq_forget": fdoc,
            "freq_retain": rdoc,
            "contrast_score": float((fdoc / max(1, len(forget)) + 1e-8) / (rdoc / max(1, len(retain)) + 1e-8)),
        }

        if int(fc) < args.min_forget_answer_count:
            protected.append({**base, "protect_reason": "low_forget_answer_count"})
            continue

        if is_bad_token(token_str, args.min_token_chars, args.protect_common_tokens):
            protected.append({**base, "protect_reason": "bad_or_common_token"})
            continue

        if rc == 0:
            policy = {
                "bucket": "rc0_zero_input_and_lmhead",
                "edit_input": True,
                "input_mode": "zero",
                "edit_lm_head": True,
                "output_mode": "zero",
                "erase_strength": 1.0,
                "output_strength": 1.0,
            }
        elif rc <= args.lmhead_only_max_retain_answer_count:
            policy = {
                "bucket": "rc1_2_zero_lmhead_only",
                "edit_input": False,
                "input_mode": "none",
                "edit_lm_head": True,
                "output_mode": "zero",
                "erase_strength": 0.0,
                "output_strength": 1.0,
            }
        elif rc <= args.input_only_max_retain_answer_count:
            policy = {
                "bucket": f"rc3_8_input_only_{args.mid_input_mode}",
                "edit_input": True,
                "input_mode": args.mid_input_mode,
                "edit_lm_head": False,
                "output_mode": "none",
                "erase_strength": 1.0,
                "output_strength": 0.0,
            }
        else:
            protected.append({**base, "protect_reason": "retain_answer_count_gt_threshold"})
            continue

        rows.append({**base, **policy})

    rows.sort(key=lambda x: (
        0 if x["bucket"] == "rc0_zero_input_and_lmhead" else 1 if x["bucket"] == "rc1_2_zero_lmhead_only" else 2,
        int(x["retain_answer_count"]),
        -int(x["forget_answer_count"]),
        -float(x["contrast_score"]),
        int(x["token_id"]),
    ))

    before_cap = len(rows)
    if args.max_final_tokens and args.max_final_tokens > 0 and len(rows) > args.max_final_tokens:
        for x in rows[args.max_final_tokens:]:
            protected.append({**x, "protect_reason": "over_max_final_tokens_cap"})
        rows = rows[:args.max_final_tokens]

    bucket_counts = defaultdict(int)
    for r in rows:
        bucket_counts[str(r["bucket"])] += 1

    selected_ids = {int(r["token_id"]) for r in rows}
    covered = 0
    coverage_rows = []
    for ex in forget_example_tokens:
        hit = sorted(set(ex["answer_token_ids"]) & selected_ids)
        if hit:
            covered += 1
        coverage_rows.append({**ex, "selected_token_ids": hit, "n_selected_tokens": len(hit)})

    out = {
        "method": "retain_count_policy_forget_answer_tokens",
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "target_model": model_name,
        "policy": {
            "retain_count_0": "zero input embedding + zero lm_head",
            f"retain_count_1_to_{args.lmhead_only_max_retain_answer_count}": "zero lm_head only",
            f"retain_count_{args.lmhead_only_max_retain_answer_count + 1}_to_{args.input_only_max_retain_answer_count}": f"input only {args.mid_input_mode}",
            f"retain_count_gt_{args.input_only_max_retain_answer_count}": "protect",
        },
        "filter_config": vars(args),
        "n_kept_before_cap": before_cap,
        "n_semantic_tokens": len(rows),
        "n_protected_tokens": len(protected),
        "bucket_counts": dict(bucket_counts),
        "forget_example_coverage": {
            "covered_examples": covered,
            "total_forget_examples": len(forget_example_tokens),
            "coverage_rate": covered / max(1, len(forget_example_tokens)),
        },
        "token_ids": [int(r["token_id"]) for r in rows],
        "token_strings": [str(r["token_str"]) for r in rows],
        "semantic_tokens": rows,
        "protected_tokens": protected,
        "coverage_by_forget_example": coverage_rows,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[Done]")
    print("saved:", out_path)
    print("tokens:", len(rows), "before_cap:", before_cap)
    print("bucket_counts:", dict(bucket_counts))
    print("coverage:", out["forget_example_coverage"])
    print("\nTop 100 selected:")
    for r in rows[:100]:
        print(
            r["token_id"], repr(r["token_str"]),
            "bucket=", r["bucket"],
            "fc=", r["forget_answer_count"],
            "rc=", r["retain_answer_count"],
            "fdoc=", r["forget_answer_doc_count"],
            "rdoc=", r["retain_answer_doc_count"],
            "contrast=", round(r["contrast_score"], 2),
        )


if __name__ == "__main__":
    main()
