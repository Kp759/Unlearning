#!/usr/bin/env python3
"""
scripts/oarts_build_tokens.py

OARTS: Overlap-Aware Residual Token Surgery token builder.

This script builds a final token file for adaptive unlearning:
  1) JSON gives the primary forget-token candidates.
  2) Retain/forget TF-IDF statistics decide how strongly each JSON token is edited.
  3) Optional residual closure mines answer tokens from forget examples that still
     have high answer probability after a first-stage unlearning run.
  4) Retain-heavy tokens are protected; retain-safe residual answer tokens can
     receive stronger input edits and optional lm_head output edits.

Output format is compatible with scripts/oarts_apply.py and mostly compatible
with older erasure scripts because it writes token_ids/token_strings too.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def get_token_ids(data: Dict[str, Any]) -> List[int]:
    if "semantic_tokens" in data and data["semantic_tokens"]:
        out = []
        for x in data["semantic_tokens"]:
            if "token_id" in x:
                out.append(int(x["token_id"]))
            elif "id" in x:
                out.append(int(x["id"]))
        return sorted(set(out))
    if "token_ids" in data:
        return sorted(set(int(x) for x in data["token_ids"]))
    raise ValueError("Token JSON must contain semantic_tokens or token_ids")


def format_qa(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']} Answer: {row['answer']}"


def build_doc_freq(dataset, tokenizer, mode: str = "qa") -> Tuple[Counter, int]:
    df: Counter = Counter()
    n = 0
    for row in tqdm(dataset, desc=f"DocFreq[{mode}]"):
        if mode == "answer":
            text = str(row["answer"])
        elif mode == "question":
            text = str(row["question"])
        else:
            text = format_qa(row)
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            df[int(tid)] += 1
        n += 1
    return df, n


def token_stats_for_ids(
    token_ids: Iterable[int],
    tokenizer,
    forget_df: Counter,
    retain_df: Counter,
    n_forget: int,
    n_retain: int,
) -> Dict[int, Dict[str, Any]]:
    total_docs = n_forget + n_retain
    stats: Dict[int, Dict[str, Any]] = {}
    for tid in sorted(set(int(x) for x in token_ids)):
        f_count = int(forget_df.get(tid, 0))
        r_count = int(retain_df.get(tid, 0))
        f_ratio = f_count / max(1, n_forget)
        r_ratio = r_count / max(1, n_retain)
        idf = math.log((total_docs + 1) / (f_count + r_count + 1)) + 1.0
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


def encode_answer_only(tokenizer, question: str, answer: str, max_length: int) -> Dict[str, List[int]]:
    prompt = f"Question: {question}\nAnswer:"
    answer_text = " " + str(answer).strip()
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def collate_one(example: Dict[str, List[int]], pad_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([example["input_ids"]], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([example["attention_mask"]], dtype=torch.long, device=device),
        "labels": torch.tensor([example["labels"]], dtype=torch.long, device=device),
    }


@torch.no_grad()
def compute_forget_answer_probs(
    model_dir: str,
    tokenizer,
    forget_rows: List[Dict[str, Any]],
    dtype: str,
    max_length: int,
) -> List[float]:
    print(f"[Residual] Loading residual model: {model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=resolve_dtype(dtype),
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    probs: List[float] = []
    for row in tqdm(forget_rows, desc="Residual AnswerProb[forget]"):
        enc = encode_answer_only(tokenizer, row["question"], row["answer"], max_length=max_length)
        batch = collate_one(enc, pad_id, device)
        out = model(**batch)
        loss = out.loss.detach().float().cpu().item()
        probs.append(float(math.exp(-loss)))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return probs


def add_or_merge_token(bucket: Dict[int, Dict[str, Any]], row: Dict[str, Any]) -> None:
    tid = int(row["token_id"])
    if tid not in bucket:
        bucket[tid] = dict(row)
        return

    old = bucket[tid]
    old["erase_strength"] = max(float(old.get("erase_strength", 0.0)), float(row.get("erase_strength", 0.0)))
    old["output_strength"] = max(float(old.get("output_strength", 0.0)), float(row.get("output_strength", 0.0)))
    old["edit_lm_head"] = bool(old.get("edit_lm_head", False) or row.get("edit_lm_head", False))

    # Preserve residual-answer metadata when a residual token was already present as a JSON token.
    if row.get("is_residual_answer", False):
        old["bucket"] = "json+residual_answer_retain_safe" if old.get("is_json", False) else row.get("bucket", old.get("bucket"))
        old["residual_answer_count"] = max(
            int(old.get("residual_answer_count", 0)),
            int(row.get("residual_answer_count", 0)),
        )
        old["retain_answer_count"] = int(row.get("retain_answer_count", old.get("retain_answer_count", 0)))

    old_sources = set(str(old.get("source", "")).split("+")) if old.get("source") else set()
    new_sources = set(str(row.get("source", "")).split("+")) if row.get("source") else set()
    old["source"] = "+".join(sorted(x for x in old_sources | new_sources if x))
    old["is_residual_answer"] = bool(old.get("is_residual_answer", False) or row.get("is_residual_answer", False))
    old["is_json"] = bool(old.get("is_json", False) or row.get("is_json", False))
    old["merge_count"] = int(old.get("merge_count", 1)) + 1


def json_strength_from_overlap(s: Dict[str, Any], args) -> Tuple[float, str]:
    r_count = int(s["freq_retain"])
    contrast = float(s["contrast_score"])
    retain_tfidf = float(s["retain_tfidf"])

    if r_count == 0:
        return args.json_zero_retain_strength, "json_zero_retain"
    if r_count <= args.json_low_retain_count and retain_tfidf <= args.json_low_retain_tfidf and contrast >= args.json_low_retain_min_contrast:
        return args.json_overlap_strength, "json_overlap_tfidf_safe"
    return args.json_soft_strength, "json_soft_overlap"


def is_json_protected(s: Dict[str, Any], args) -> Optional[str]:
    # Zero-retain JSON tokens are never protected.
    if int(s["freq_retain"]) == 0:
        return None
    if int(s["freq_retain"]) > args.max_json_retain_count:
        return "json_high_retain_count"
    if float(s["retain_tfidf"]) > args.max_json_retain_tfidf:
        return "json_high_retain_tfidf"
    if float(s["retain_ratio"]) > args.max_json_retain_ratio:
        return "json_high_retain_ratio"
    return None


def residual_token_allowed(s: Dict[str, Any], args) -> Optional[str]:
    # Residual answer token is allowed if it is truly retain-safe.
    if int(s["freq_retain"]) == 0:
        return None
    if int(s["freq_retain"]) > args.residual_max_retain_count:
        return "residual_high_retain_count"
    if float(s["retain_tfidf"]) > args.residual_max_retain_tfidf:
        return "residual_high_retain_tfidf"
    if float(s["contrast_score"]) < args.residual_min_contrast:
        return "residual_low_contrast"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config.yaml")
    ap.add_argument("--json-tokens", default=None, help="Raw JSON token file, e.g. outputs/semantic_tokens_json_raw.json")
    ap.add_argument("--out", default=None, help="Output semantic token file")
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--model-name", default=None, help="Tokenizer/model name; defaults to config model.name")
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--max-length", type=int, default=None)

    # JSON overlap policy.
    ap.add_argument("--max-json-retain-count", type=int, default=25)
    ap.add_argument("--max-json-retain-ratio", type=float, default=0.01)
    ap.add_argument("--max-json-retain-tfidf", type=float, default=0.035)
    ap.add_argument("--json-zero-retain-strength", type=float, default=1.0)
    ap.add_argument("--json-overlap-strength", type=float, default=0.70)
    ap.add_argument("--json-soft-strength", type=float, default=0.35)
    ap.add_argument("--json-low-retain-count", type=int, default=8)
    ap.add_argument("--json-low-retain-tfidf", type=float, default=0.015)
    ap.add_argument("--json-low-retain-min-contrast", type=float, default=5.0)

    # Residual closure.
    ap.add_argument("--residual-model-dir", default=None, help="First-stage model used to find residual high-prob forget examples")
    ap.add_argument("--residual-threshold", type=float, default=0.05)
    ap.add_argument("--max-residual-examples", type=int, default=200)
    ap.add_argument("--max-residual-tokens", type=int, default=250)
    ap.add_argument("--residual-max-retain-count", type=int, default=2)
    ap.add_argument("--residual-max-retain-tfidf", type=float, default=0.005)
    ap.add_argument("--residual-min-contrast", type=float, default=20.0)
    ap.add_argument("--residual-input-strength", type=float, default=1.0)
    ap.add_argument("--residual-output-strength", type=float, default=0.98)
    ap.add_argument("--residual-output-max-retain-count", type=int, default=0)
    ap.add_argument("--residual-answer-only", action="store_true", default=True)

    ap.add_argument("--max-final-tokens", type=int, default=900)
    ap.add_argument("--min-token-len", type=int, default=1)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.json_tokens) if args.json_tokens else out_dir / "semantic_tokens_json_raw.json"
    out_path = Path(args.out) if args.out else out_dir / "semantic_tokens_oarts.json"
    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")
    max_length = args.max_length or cfg["model"].get("max_length", 512)

    print("=" * 80)
    print("OARTS token builder: JSON + TF-IDF overlap + residual answer closure")
    print("=" * 80)
    print(f"JSON tokens:    {json_path}")
    print(f"Output tokens:  {out_path}")
    print(f"Forget split:   {forget_split}")
    print(f"Retain split:   {retain_split}")
    print(f"Tokenizer:      {model_name}")
    print(f"Residual model: {args.residual_model_dir}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")
    forget_rows = list(forget_ds)

    json_data = load_json(json_path)
    json_ids = get_token_ids(json_data)

    # Count stats over QA text for overlap safety.
    forget_df, n_forget = build_doc_freq(forget_ds, tokenizer, mode="qa")
    retain_df, n_retain = build_doc_freq(retain_ds, tokenizer, mode="qa")

    # Also compute retain answer-only counts for residual answer tokens.
    retain_ans_df, _ = build_doc_freq(retain_ds, tokenizer, mode="answer")

    candidate_ids = set(json_ids)
    residual_rows: List[Dict[str, Any]] = []
    residual_answer_token_counts: Counter = Counter()
    residual_example_info: List[Dict[str, Any]] = []

    if args.residual_model_dir:
        probs = compute_forget_answer_probs(
            model_dir=args.residual_model_dir,
            tokenizer=tokenizer,
            forget_rows=forget_rows,
            dtype=dtype,
            max_length=max_length,
        )
        ranked = sorted(enumerate(probs), key=lambda x: -x[1])
        residual_indices = [i for i, p in ranked if p >= args.residual_threshold][: args.max_residual_examples]
        print(f"[Residual] Examples above threshold {args.residual_threshold}: {len(residual_indices)}")

        for i in residual_indices:
            row = forget_rows[i]
            answer_ids = tokenizer.encode(" " + str(row["answer"]).strip(), add_special_tokens=False)
            unique_answer_ids = set(int(x) for x in answer_ids)
            for tid in unique_answer_ids:
                residual_answer_token_counts[tid] += 1
                candidate_ids.add(tid)
            residual_example_info.append({
                "idx": int(i),
                "answer_prob": float(probs[i]),
                "question": row["question"],
                "answer": row["answer"],
                "answer_token_ids": sorted(unique_answer_ids),
            })

    stats = token_stats_for_ids(candidate_ids, tokenizer, forget_df, retain_df, n_forget, n_retain)
    kept: Dict[int, Dict[str, Any]] = {}
    protected: List[Dict[str, Any]] = []

    # Add JSON candidates with overlap-aware strength.
    for tid in json_ids:
        tid = int(tid)
        s = dict(stats[tid])
        if len(s["token_str"].strip()) < args.min_token_len:
            s["protect_reason"] = "short_token"
            protected.append(s)
            continue

        reason = is_json_protected(s, args)
        if reason:
            s["protect_reason"] = reason
            s["source"] = "json"
            protected.append(s)
            continue

        strength, bucket = json_strength_from_overlap(s, args)
        row = {
            **s,
            "source": "json",
            "bucket": bucket,
            "is_json": True,
            "is_residual_answer": False,
            "erase_strength": float(strength),
            "output_strength": 0.0,
            "edit_lm_head": False,
            "retain_answer_count": int(retain_ans_df.get(tid, 0)),
        }
        add_or_merge_token(kept, row)

    # Add residual answer tokens, but only when retain-safe.
    residual_candidates = []
    for tid, count in residual_answer_token_counts.items():
        tid = int(tid)
        s = dict(stats[tid])
        s["residual_answer_count"] = int(count)
        s["retain_answer_count"] = int(retain_ans_df.get(tid, 0))
        if len(s["token_str"].strip()) < args.min_token_len:
            s["protect_reason"] = "residual_short_token"
            protected.append(s)
            continue
        reason = residual_token_allowed(s, args)
        if reason:
            s["protect_reason"] = reason
            s["source"] = "residual_answer"
            protected.append(s)
            continue
        residual_candidates.append(s)

    residual_candidates.sort(
        key=lambda x: (
            -int(x.get("residual_answer_count", 0)),
            int(x.get("freq_retain", 0)),
            -float(x.get("contrast_score", 0.0)),
            -float(x.get("forget_tfidf", 0.0)),
            int(x.get("token_id", 0)),
        )
    )
    if args.max_residual_tokens > 0:
        residual_candidates = residual_candidates[: args.max_residual_tokens]

    for s in residual_candidates:
        tid = int(s["token_id"])
        output_strength = 0.0
        edit_lm_head = False
        if int(s.get("freq_retain", 0)) <= args.residual_output_max_retain_count and int(s.get("retain_answer_count", 0)) == 0:
            output_strength = float(args.residual_output_strength)
            edit_lm_head = True
        row = {
            **s,
            "source": "residual_answer",
            "bucket": "residual_answer_retain_safe",
            "is_json": False,
            "is_residual_answer": True,
            "erase_strength": float(args.residual_input_strength),
            "output_strength": output_strength,
            "edit_lm_head": bool(edit_lm_head),
        }
        add_or_merge_token(kept, row)

    rows = list(kept.values())
    rows.sort(
        key=lambda x: (
            0 if x.get("is_residual_answer") else 1,
            0 if x.get("bucket") == "json_zero_retain" else 1,
            -float(x.get("erase_strength", 0.0)),
            -float(x.get("output_strength", 0.0)),
            int(x.get("freq_retain", 0)),
            -float(x.get("contrast_score", 0.0)),
            int(x.get("token_id", 0)),
        )
    )

    before_cap = len(rows)
    if args.max_final_tokens > 0 and len(rows) > args.max_final_tokens:
        overflow = rows[args.max_final_tokens:]
        for x in overflow:
            y = dict(x)
            y["protect_reason"] = "over_max_final_tokens_cap"
            protected.append(y)
        rows = rows[: args.max_final_tokens]

    source_counts = defaultdict(int)
    bucket_counts = defaultdict(int)
    output_count = 0
    for x in rows:
        source_counts[str(x.get("source", "unknown"))] += 1
        bucket_counts[str(x.get("bucket", "unknown"))] += 1
        if float(x.get("output_strength", 0.0)) > 0:
            output_count += 1

    out = {
        "method": "OARTS_overlap_aware_residual_token_surgery",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "json_tokens_file": str(json_path),
        "residual_model_dir": args.residual_model_dir,
        "n_json_input_tokens": len(json_ids),
        "n_residual_examples": len(residual_example_info),
        "n_residual_answer_token_candidates": len(residual_answer_token_counts),
        "n_kept_before_cap": before_cap,
        "n_semantic_tokens": len(rows),
        "n_output_edit_tokens": output_count,
        "n_protected_tokens": len(protected),
        "source_counts": dict(source_counts),
        "bucket_counts": dict(bucket_counts),
        "filter_config": vars(args),
        "token_ids": [int(x["token_id"]) for x in rows],
        "token_strings": [str(x.get("token_str", "")) for x in rows],
        "semantic_tokens": rows,
        "protected_tokens": protected,
        "residual_examples": residual_example_info,
    }

    save_json(out, out_path)
    save_json(protected, out_path.parent / "protected_tokens_oarts.json")
    save_json(residual_example_info, out_path.parent / "residual_examples_oarts.json")

    print("\n[Done]")
    print(f"Kept tokens:          {len(rows)}")
    print(f"Kept before cap:     {before_cap}")
    print(f"Output edit tokens:  {output_count}")
    print(f"Protected tokens:    {len(protected)}")
    print(f"Source counts:       {dict(source_counts)}")
    print(f"Bucket counts:       {dict(bucket_counts)}")
    print(f"Output file:         {out_path}")

    print("\nTop tokens:")
    for x in rows[:80]:
        print(
            f"{int(x['token_id']):>8} | {repr(str(x.get('token_str', '')))} | "
            f"src={x.get('source')} bucket={x.get('bucket')} | "
            f"in={float(x.get('erase_strength', 0.0)):.2f} out={float(x.get('output_strength', 0.0)):.2f} | "
            f"f={int(x.get('freq_forget', 0))} r={int(x.get('freq_retain', 0))} "
            f"ra={int(x.get('retain_answer_count', 0))} | "
            f"contrast={float(x.get('contrast_score', 0.0)):.1f}"
        )


if __name__ == "__main__":
    main()
