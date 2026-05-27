#!/usr/bin/env python3
"""
scripts/json_forget_zero_retain_overlap_restore.py

Token-overlap restore method:
  1) Take forget JSON token file.
  2) Zero ALL forget JSON tokens in input embeddings + lm_head.
  3) Build retain token counts from retain set, optionally filtered by retain JSON/frequency.
  4) For tokens that are in BOTH forget JSON and retain tokens, restore/interpolate from original model.
  5) Restoration alpha depends on retain overlap count.

This intentionally does NOT do semantic record matching.
It is token-overlap + retain-count restoration only.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


COMMON_PROTECT = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "by",
    "is", "was", "were", "are", "be", "been", "being", "as", "at", "from",
    "that", "this", "these", "those", "it", "its", "he", "she", "they", "them",
    "his", "her", "their", "my", "our", "your", "i", "you", "we", "me",
    "will", "would", "can", "could", "may", "might", "should", "has", "have",
    "had", "do", "does", "did", "not", "no", "yes"
}


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_token_ids_from_json(path: Path) -> Set[int]:
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


def is_bad_token(tok, tid: int, min_token_chars: int, protect_common: bool) -> bool:
    special = {
        x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id]
        if x is not None
    }
    if int(tid) in special:
        return True
    s = tok.decode([int(tid)])
    clean = norm_token(s)
    if len(clean) < min_token_chars:
        return True
    if protect_common and clean in COMMON_PROTECT:
        return True
    if not any(ch.isalnum() for ch in clean):
        return True
    return False


def encode_answer(tok, answer: str) -> List[int]:
    return [int(x) for x in tok.encode(" " + str(answer).strip(), add_special_tokens=False)]


def encode_text(tok, text: str) -> List[int]:
    return [int(x) for x in tok.encode(str(text), add_special_tokens=False)]


def force_untie_lm_head(model) -> None:
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None or not hasattr(out, "weight"):
        raise RuntimeError("Model has no lm_head/output embedding weight.")
    if emb.weight.data_ptr() != out.weight.data_ptr():
        return
    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False).to(old.device, old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


def alpha_policy(answer_count: int, doc_count: int, mode: str) -> tuple[float, float, str]:
    """
    Returns alpha_input, alpha_lm_head, bucket.

    answer_count controls LM-head restore.
    doc_count can trigger input-only restore for tokens seen in retain prompts/QA but not retain answers.
    """
    if mode == "strict":
        # Best for preserving forget.
        if answer_count <= 2:
            return 0.00, 0.00, "ans0_2_keep_erased"
        if answer_count <= 5:
            return 0.00, 0.15, "ans3_5_lm015"
        if answer_count <= 8:
            return 0.02, 0.35, "ans6_8_in002_lm035"
        return 0.05, 0.70, "ans9plus_in005_lm070"

    if mode == "retain":
        # More retain, forget may rise.
        if answer_count <= 2:
            return 0.00, 0.00, "ans0_2_keep_erased"
        if answer_count <= 5:
            return 0.05, 0.30, "ans3_5_in005_lm030"
        if answer_count <= 8:
            return 0.10, 0.55, "ans6_8_in010_lm055"
        return 0.15, 0.85, "ans9plus_in015_lm085"

    # balanced default
    if answer_count <= 2:
        # If token is not in retain answers but appears frequently in retain QA, restore input weakly only.
        if doc_count >= 20:
            return 0.03, 0.00, "ans0_2_doc20_input003"
        return 0.00, 0.00, "ans0_2_keep_erased"
    if answer_count <= 5:
        return 0.00, 0.25, "ans3_5_lm025"
    if answer_count <= 8:
        return 0.05, 0.50, "ans6_8_in005_lm050"
    return 0.10, 0.75, "ans9plus_in010_lm075"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--original-model", default=None, help="Original finetuned model before unlearning")
    ap.add_argument("--forget-json", required=True, help="Forget JSON token file to zero")
    ap.add_argument("--retain-json", default=None, help="Optional retain JSON token file. If given, restore only overlap with this retain token set plus retain frequency.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--mode", choices=["strict", "balanced", "retain"], default="balanced")
    ap.add_argument("--min-token-chars", type=int, default=2)
    ap.add_argument("--protect-common-tokens", action="store_true", default=True)
    ap.add_argument("--no-protect-common-tokens", dest="protect_common_tokens", action="store_false")
    ap.add_argument("--retain-doc-min", type=int, default=1, help="Token must appear in at least this many retain QA docs to be considered retain-overlap when no retain-json match exists.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    original_model = args.original_model or cfg["model"]["name"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    dtype = resolve_dtype(args.dtype or cfg["model"].get("dtype", "float16"))

    print("=" * 80)
    print("JSON forget zero + retain-overlap adaptive restore")
    print("=" * 80)
    print("original model:", original_model)
    print("forget json:", args.forget_json)
    print("retain json:", args.retain_json)
    print("retain split:", retain_split)
    print("mode:", args.mode)
    print("output:", args.output_dir)
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(original_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget_ids_all = get_token_ids_from_json(Path(args.forget_json))
    retain_json_ids = get_token_ids_from_json(Path(args.retain_json)) if args.retain_json else set()

    # Remove special/blank/common tokens from forget zero list.
    forget_ids = {
        int(t) for t in forget_ids_all
        if not is_bad_token(tok, int(t), args.min_token_chars, args.protect_common_tokens)
    }

    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")
    retain_answer_count = Counter()
    retain_answer_doc = Counter()
    retain_qa_doc = Counter()

    for row in retain_ds:
        ans_ids = encode_answer(tok, row["answer"])
        qa_ids = encode_text(tok, f"Question: {row['question']} Answer: {row['answer']}")
        retain_answer_count.update(ans_ids)
        for t in set(ans_ids):
            retain_answer_doc[int(t)] += 1
        for t in set(qa_ids):
            retain_qa_doc[int(t)] += 1

    # Retain-overlap tokens: either in retain-json if provided, or frequent enough in retain QA/answers.
    if retain_json_ids:
        retain_overlap_ids = set(retain_json_ids)
    else:
        retain_overlap_ids = set()

    for tid in forget_ids:
        if int(retain_answer_count.get(tid, 0)) > 0 or int(retain_qa_doc.get(tid, 0)) >= args.retain_doc_min:
            retain_overlap_ids.add(tid)

    overlap_ids = sorted(forget_ids & retain_overlap_ids)
    zero_only_ids = sorted(forget_ids - set(overlap_ids))

    print("forget json tokens raw:", len(forget_ids_all))
    print("forget json tokens after filter:", len(forget_ids))
    print("retain json tokens:", len(retain_json_ids))
    print("overlap restore tokens:", len(overlap_ids))
    print("zero-only tokens:", len(zero_only_ids))

    student = AutoModelForCausalLM.from_pretrained(
        original_model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        original_model,
        torch_dtype=dtype,
        device_map=args.device_map,
    )
    student.eval(); teacher.eval()
    force_untie_lm_head(student)
    force_untie_lm_head(teacher)

    bucket_counts = defaultdict(int)
    rows = []

    with torch.no_grad():
        Es = student.get_input_embeddings().weight.data
        Hs = student.get_output_embeddings().weight.data
        Et = teacher.get_input_embeddings().weight.data.to(Es.device)
        Ht = teacher.get_output_embeddings().weight.data.to(Hs.device)

        # Step 1: zero all forget JSON tokens.
        for tid in sorted(forget_ids):
            Es[tid].zero_()
            Hs[tid].zero_()

        # Step 2: restore only overlap tokens using retain count alpha.
        for tid in overlap_ids:
            ans_c = int(retain_answer_count.get(tid, 0))
            doc_c = int(retain_qa_doc.get(tid, 0))
            a_in, a_lm, bucket = alpha_policy(ans_c, doc_c, args.mode)
            bucket_counts[bucket] += 1

            if a_in > 0:
                Es[tid].copy_((1 - a_in) * Es[tid] + a_in * Et[tid])
            if a_lm > 0:
                Hs[tid].copy_((1 - a_lm) * Hs[tid] + a_lm * Ht[tid])

            rows.append({
                "token_id": int(tid),
                "token_str": tok.decode([int(tid)]),
                "retain_answer_count": ans_c,
                "retain_answer_doc_count": int(retain_answer_doc.get(tid, 0)),
                "freq_retain": doc_c,
                "alpha_input": a_in,
                "alpha_lm_head": a_lm,
                "bucket": bucket,
                "source": "json_forget_zero_retain_overlap_restore",
            })

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)

    summary = {
        "method": "json_forget_zero_retain_overlap_adaptive_restore",
        "original_model": original_model,
        "forget_json": args.forget_json,
        "retain_json": args.retain_json,
        "retain_split": retain_split,
        "mode": args.mode,
        "n_forget_json_raw": len(forget_ids_all),
        "n_forget_json_after_filter": len(forget_ids),
        "n_retain_json": len(retain_json_ids),
        "n_overlap_restore": len(overlap_ids),
        "n_zero_only": len(zero_only_ids),
        "bucket_counts": dict(bucket_counts),
        "restore_rows": rows,
        "zero_only_token_ids": zero_only_ids,
    }
    save_json(summary, out / "json_forget_zero_retain_overlap_restore_summary.json")
    print("saved:", out)
    print("bucket_counts:", dict(bucket_counts))


if __name__ == "__main__":
    main()
