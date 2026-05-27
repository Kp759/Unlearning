#!/usr/bin/env python3
"""
scripts/build_dual_json_rc_gagd_tokens.py

Build token policy for Dual-JSON + retain-count GA/GD unlearning.

User idea:
  - Run JSON/token extraction on forget and retain.
  - Forget-only / low-overlap tokens -> GA on forget.
  - Retain / overlap tokens -> GD/KL on retain.
  - High-overlap tokens are not attacked by GA.

Output:
  outputs/semantic_tokens_dual_json_rc_gagd.json

Then train with:
  scripts/train_dual_json_rc_gagd_embed_lmhead.py
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


COMMON_PROTECT = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "was", "were", "are", "be", "been", "being", "as", "at", "from",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "my", "our", "your", "i", "you", "we", "me",
    "will", "would", "can", "could", "may", "might", "should", "has", "have",
    "had", "do", "does", "did", "not", "no", "yes", "but", "if", "then", "than",
    "also", "into", "about", "after", "before", "through", "during", "over", "under",
}


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_token_ids_from_json(path: Optional[Path]) -> Set[int]:
    if path is None or not path.exists():
        return set()
    data = load_json(path)
    ids: Set[int] = set()
    for r in data.get("semantic_tokens", []):
        if "token_id" in r:
            ids.add(int(r["token_id"]))
        elif "id" in r:
            ids.add(int(r["id"]))
    for x in data.get("token_ids", []):
        ids.add(int(x))
    return ids


def norm_token(s: str) -> str:
    return str(s).strip().lower().replace("Ġ", "").replace("▁", "")


def is_bad_token(tok, tid: int, min_token_chars: int, protect_common: bool) -> Tuple[bool, str]:
    special = {
        x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id]
        if x is not None
    }
    if int(tid) in special:
        return True, "special_token"

    token_str = tok.decode([int(tid)])
    clean = norm_token(token_str)

    if len(clean) < min_token_chars:
        return True, "too_short"

    if not any(ch.isalnum() for ch in clean):
        return True, "non_alnum"

    if protect_common and clean in COMMON_PROTECT:
        return True, "common_token"

    return False, "ok"


def encode_text(tok, text: str) -> List[int]:
    return [int(x) for x in tok.encode(str(text), add_special_tokens=False)]


def encode_answer(tok, answer: str) -> List[int]:
    # Leading space matters for Llama-style BPE.
    return [int(x) for x in tok.encode(" " + str(answer).strip(), add_special_tokens=False)]


def format_qa(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']} Answer: {row['answer']}"


def count_dataset_stats(ds, tok) -> Tuple[Counter, Counter, Counter]:
    """answer_count, answer_doc_count, qa_doc_count"""
    ans_count, ans_doc, qa_doc = Counter(), Counter(), Counter()
    for row in tqdm(ds, desc="CountStats"):
        ans_ids = encode_answer(tok, row["answer"])
        qa_ids = encode_text(tok, format_qa(row))
        ans_count.update(ans_ids)
        for t in set(ans_ids):
            ans_doc[int(t)] += 1
        for t in set(qa_ids):
            qa_doc[int(t)] += 1
    return ans_count, ans_doc, qa_doc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--forget-json", default="outputs/semantic_tokens_json_raw.json")
    ap.add_argument("--retain-json", default=None)
    ap.add_argument("--out", default="outputs/semantic_tokens_dual_json_rc_gagd.json")
    ap.add_argument("--model-name", default=None)

    # Candidate sources.
    ap.add_argument("--include-forget-answer-tokens", action="store_true", default=True)
    ap.add_argument("--include-forget-json-tokens", action="store_true", default=True)
    ap.add_argument("--include-retain-answer-gd", action="store_true", default=True)
    ap.add_argument("--include-retain-json-gd", action="store_true", default=True)

    # Overlap/GA/GD decision.
    ap.add_argument("--ga-max-retain-answer-count", type=int, default=2,
                    help="If retain answer count <= this and retain doc count is low, token can receive GA.")
    ap.add_argument("--ga-max-retain-doc-count", type=int, default=20)
    ap.add_argument("--overlap-gd-min-retain-answer-count", type=int, default=3,
                    help="Forget candidate with retain answer count >= this becomes GD/overlap, not GA.")
    ap.add_argument("--overlap-gd-min-retain-doc-count", type=int, default=21,
                    help="Forget candidate with retain QA doc count >= this becomes GD/overlap, not GA.")
    ap.add_argument("--retain-gd-min-answer-count", type=int, default=2,
                    help="Retain-only answer tokens with count >= this are added to GD set.")
    ap.add_argument("--max-retain-only-gd-tokens", type=int, default=1200)

    # Quality filters.
    ap.add_argument("--min-token-chars", type=int, default=2)
    ap.add_argument("--protect-common-tokens", action="store_true", default=True)
    ap.add_argument("--no-protect-common-tokens", dest="protect_common_tokens", action="store_false")
    ap.add_argument("--max-ga-tokens", type=int, default=1000)
    ap.add_argument("--max-gd-tokens", type=int, default=2000)

    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]

    forget_json_path = Path(args.forget_json) if args.forget_json else None
    retain_json_path = Path(args.retain_json) if args.retain_json else None

    print("=" * 80)
    print("Build Dual-JSON RC-GA/GD token policy")
    print("=" * 80)
    print("model:", model_name)
    print("forget split:", forget_split)
    print("retain split:", retain_split)
    print("forget json:", forget_json_path)
    print("retain json:", retain_json_path)
    print("out:", args.out)
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    forget_json_ids = get_token_ids_from_json(forget_json_path)
    retain_json_ids = get_token_ids_from_json(retain_json_path) if retain_json_path and retain_json_path.exists() else set()

    print("forget_json_ids:", len(forget_json_ids))
    print("retain_json_ids:", len(retain_json_ids))

    f_ans_count, f_ans_doc, f_qa_doc = count_dataset_stats(forget_ds, tok)
    r_ans_count, r_ans_doc, r_qa_doc = count_dataset_stats(retain_ds, tok)

    forget_answer_ids = set(int(x) for x in f_ans_count.keys())
    retain_answer_ids = set(int(x) for x in r_ans_count.keys())

    forget_candidate_ids: Set[int] = set()
    if args.include_forget_answer_tokens:
        forget_candidate_ids |= forget_answer_ids
    if args.include_forget_json_tokens:
        forget_candidate_ids |= forget_json_ids

    retain_candidate_ids: Set[int] = set()
    if args.include_retain_answer_gd:
        retain_candidate_ids |= retain_answer_ids
    if args.include_retain_json_gd:
        retain_candidate_ids |= retain_json_ids

    rows: List[Dict[str, Any]] = []
    protected: List[Dict[str, Any]] = []

    def make_stats(tid: int) -> Dict[str, Any]:
        fqa = int(f_qa_doc.get(tid, 0))
        rqa = int(r_qa_doc.get(tid, 0))
        fr = fqa / max(1, len(forget_ds))
        rr = rqa / max(1, len(retain_ds))
        return {
            "token_id": int(tid),
            "token_str": tok.decode([int(tid)]),
            "forget_answer_count": int(f_ans_count.get(tid, 0)),
            "forget_answer_doc_count": int(f_ans_doc.get(tid, 0)),
            "retain_answer_count": int(r_ans_count.get(tid, 0)),
            "retain_answer_doc_count": int(r_ans_doc.get(tid, 0)),
            "freq_forget": fqa,
            "freq_retain": rqa,
            "forget_ratio": float(fr),
            "retain_ratio": float(rr),
            "contrast_score": float((fr + 1e-8) / (rr + 1e-8)),
            "in_forget_json": bool(tid in forget_json_ids),
            "in_retain_json": bool(tid in retain_json_ids),
            "in_forget_answer": bool(tid in forget_answer_ids),
            "in_retain_answer": bool(tid in retain_answer_ids),
        }

    # First assign forget candidates to GA or overlap-GD.
    for tid in sorted(forget_candidate_ids):
        bad, reason = is_bad_token(tok, tid, args.min_token_chars, args.protect_common_tokens)
        stats = make_stats(tid)
        if bad:
            protected.append({**stats, "protect_reason": reason})
            continue

        rc_ans = int(stats["retain_answer_count"])
        rc_doc = int(stats["freq_retain"])
        in_retain_json = bool(stats["in_retain_json"])

        # Overlap / retain-supported forget token -> GD/KL, never GA.
        if in_retain_json or rc_ans >= args.overlap_gd_min_retain_answer_count or rc_doc >= args.overlap_gd_min_retain_doc_count:
            rows.append({
                **stats,
                "source": "dual_json_rc_gagd",
                "bucket": "overlap_gd_retain_preserve",
                "train_direction": "gd",
                "edit_input": True,
                "edit_lm_head": True,
                "ga_weight": 0.0,
                "gd_weight": 1.0,
            })
        elif rc_ans <= args.ga_max_retain_answer_count and rc_doc <= args.ga_max_retain_doc_count:
            rows.append({
                **stats,
                "source": "dual_json_rc_gagd",
                "bucket": "forget_only_or_low_overlap_ga",
                "train_direction": "ga",
                "edit_input": True,
                "edit_lm_head": True,
                "ga_weight": 1.0,
                "gd_weight": 0.0,
            })
        else:
            rows.append({
                **stats,
                "source": "dual_json_rc_gagd",
                "bucket": "ambiguous_overlap_gd",
                "train_direction": "gd",
                "edit_input": True,
                "edit_lm_head": True,
                "ga_weight": 0.0,
                "gd_weight": 1.0,
            })

    existing_ids = {int(r["token_id"]) for r in rows}

    # Add retain-only GD tokens, ranked by retain answer count / doc count.
    retain_only = []
    for tid in sorted(retain_candidate_ids - existing_ids):
        bad, reason = is_bad_token(tok, tid, args.min_token_chars, args.protect_common_tokens)
        stats = make_stats(tid)
        if bad:
            protected.append({**stats, "protect_reason": f"retain_only_{reason}"})
            continue
        if int(stats["retain_answer_count"]) < args.retain_gd_min_answer_count and not bool(stats["in_retain_json"]):
            protected.append({**stats, "protect_reason": "retain_only_low_count"})
            continue
        retain_only.append({
            **stats,
            "source": "dual_json_rc_gagd",
            "bucket": "retain_only_gd",
            "train_direction": "gd",
            "edit_input": True,
            "edit_lm_head": True,
            "ga_weight": 0.0,
            "gd_weight": 1.0,
        })

    retain_only.sort(key=lambda x: (-int(x["retain_answer_count"]), -int(x["freq_retain"]), int(x["token_id"])))
    if args.max_retain_only_gd_tokens > 0:
        protected.extend({**x, "protect_reason": "over_max_retain_only_gd_tokens"} for x in retain_only[args.max_retain_only_gd_tokens:])
        retain_only = retain_only[:args.max_retain_only_gd_tokens]
    rows.extend(retain_only)

    # Cap GA and GD tokens separately to keep training stable.
    ga_rows = [r for r in rows if r["train_direction"] == "ga"]
    gd_rows = [r for r in rows if r["train_direction"] == "gd"]

    ga_rows.sort(key=lambda x: (int(x["retain_answer_count"]), int(x["freq_retain"]), -float(x["contrast_score"]), -int(x["forget_answer_count"])))
    gd_rows.sort(key=lambda x: (-int(x["retain_answer_count"]), -int(x["freq_retain"]), int(x["token_id"])))

    if args.max_ga_tokens > 0 and len(ga_rows) > args.max_ga_tokens:
        protected.extend({**x, "protect_reason": "over_max_ga_tokens"} for x in ga_rows[args.max_ga_tokens:])
        ga_rows = ga_rows[:args.max_ga_tokens]
    if args.max_gd_tokens > 0 and len(gd_rows) > args.max_gd_tokens:
        protected.extend({**x, "protect_reason": "over_max_gd_tokens"} for x in gd_rows[args.max_gd_tokens:])
        gd_rows = gd_rows[:args.max_gd_tokens]

    final_rows = ga_rows + gd_rows
    final_rows.sort(key=lambda x: (0 if x["train_direction"] == "ga" else 1, int(x.get("retain_answer_count", 9999)), -float(x.get("contrast_score", 0)), int(x["token_id"])))

    bucket_counts = Counter(str(r["bucket"]) for r in final_rows)
    direction_counts = Counter(str(r["train_direction"]) for r in final_rows)

    out = {
        "method": "dual_json_retain_count_gagd_tokens",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "forget_json_file": str(forget_json_path) if forget_json_path else None,
        "retain_json_file": str(retain_json_path) if retain_json_path else None,
        "filter_config": vars(args),
        "n_semantic_tokens": len(final_rows),
        "n_ga_tokens": int(direction_counts.get("ga", 0)),
        "n_gd_tokens": int(direction_counts.get("gd", 0)),
        "n_protected_tokens": len(protected),
        "bucket_counts": dict(bucket_counts),
        "direction_counts": dict(direction_counts),
        "ga_token_ids": [int(r["token_id"]) for r in final_rows if r["train_direction"] == "ga"],
        "gd_token_ids": [int(r["token_id"]) for r in final_rows if r["train_direction"] == "gd"],
        "token_ids": [int(r["token_id"]) for r in final_rows],
        "token_strings": [str(r["token_str"]) for r in final_rows],
        "semantic_tokens": final_rows,
        "protected_tokens": protected,
    }

    save_json(out, Path(args.out))

    print("\n[Done]")
    print("saved:", args.out)
    print("n tokens:", len(final_rows))
    print("direction_counts:", dict(direction_counts))
    print("bucket_counts:", dict(bucket_counts))
    print("protected:", len(protected))
    print("\nTop GA tokens:")
    for r in ga_rows[:80]:
        print(r["token_id"], repr(r["token_str"]), "fa=", r["forget_answer_count"], "ra=", r["retain_answer_count"], "fq=", r["freq_forget"], "rq=", r["freq_retain"], "contrast=", round(r["contrast_score"], 2))
    print("\nTop GD tokens:")
    for r in gd_rows[:40]:
        print(r["token_id"], repr(r["token_str"]), "fa=", r["forget_answer_count"], "ra=", r["retain_answer_count"], "fq=", r["freq_forget"], "rq=", r["freq_retain"], "bucket=", r["bucket"])


if __name__ == "__main__":
    main()
