#!/usr/bin/env python3
"""
scripts/build_dual_json_retain_shield_tokens.py

Dual-JSON Retain Shield token builder.

Goal:
  Use JSON-style forget tokens, build a retain-side record shield, and select
  only forget records/tokens that are not same/similar to retain records.

Output:
  A semantic token JSON compatible with scripts/zero_both_selected_tokens.py.

Then apply:
  python scripts/zero_both_selected_tokens.py \
    --config config/config_3b_instruct_forget05.yaml \
    --tokens-json outputs/semantic_tokens_dual_json_retain_shield_balanced.json \
    --output-dir outputs/unlearned_model_dual_json_retain_shield_balanced_zero_both \
    --zero-scope all
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def norm_text(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def word_set(s: str) -> Set[str]:
    return {w for w in norm_text(s).split() if len(w) >= 2}


def format_qa(row: Dict[str, Any]) -> str:
    return f"Question: {row['question']} Answer: {row['answer']}"


def encode_text(tok, text: str) -> List[int]:
    return [int(x) for x in tok.encode(str(text), add_special_tokens=False)]


def encode_answer(tok, answer: str) -> List[int]:
    # leading space matters for Llama-style BPE answer tokenization
    return [int(x) for x in tok.encode(" " + str(answer).strip(), add_special_tokens=False)]


def jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def containment(a: Set[Any], b: Set[Any]) -> float:
    if not a:
        return 0.0
    return len(a & b) / max(1, len(a))


def get_token_ids_from_json(path: Optional[Path]) -> Set[int]:
    if path is None or not path.exists():
        return set()
    data = load_json(path)
    ids = set()
    for r in data.get("semantic_tokens", []):
        if "token_id" in r:
            ids.add(int(r["token_id"]))
        elif "id" in r:
            ids.add(int(r["id"]))
    for x in data.get("token_ids", []):
        ids.add(int(x))
    return ids


def is_bad_token(tok, tid: int, min_token_len: int) -> bool:
    special = {
        x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id]
        if x is not None
    }
    if int(tid) in special:
        return True
    if len(tok.decode([int(tid)]).strip()) < min_token_len:
        return True
    return False


def build_records(dataset, tok, split: str, json_ids: Set[int], include_question_tokens: bool) -> List[Dict[str, Any]]:
    records = []
    for idx, row in enumerate(tqdm(dataset, desc=f"BuildRecords[{split}]")):
        question = str(row["question"])
        answer = str(row["answer"])
        qa = format_qa(row)
        q_ids = set(encode_text(tok, question))
        a_ids = set(encode_answer(tok, answer))
        qa_ids = set(encode_text(tok, qa))
        assigned_json = sorted(set(json_ids) & qa_ids)
        cand = set(assigned_json) | set(a_ids)
        if include_question_tokens:
            cand |= set(q_ids)
        records.append({
            "idx": int(idx),
            "split": split,
            "question": question,
            "answer": answer,
            "qa_text": qa,
            "question_token_ids": sorted(q_ids),
            "answer_token_ids": sorted(a_ids),
            "qa_token_ids": sorted(qa_ids),
            "json_token_ids": sorted(assigned_json),
            "candidate_token_ids": sorted(cand),
            "words": sorted(word_set(qa)),
        })
    return records


def count_stats(records: Sequence[Dict[str, Any]]) -> Tuple[Counter, Counter, Counter]:
    ans_count, ans_doc, qa_doc = Counter(), Counter(), Counter()
    for r in records:
        ans = [int(x) for x in r["answer_token_ids"]]
        ans_count.update(ans)
        for t in set(ans):
            ans_doc[int(t)] += 1
        for t in set(int(x) for x in r["qa_token_ids"]):
            qa_doc[int(t)] += 1
    return ans_count, ans_doc, qa_doc


def record_sim(f: Dict[str, Any], r: Dict[str, Any]) -> Dict[str, float]:
    f_ans, r_ans = set(f["answer_token_ids"]), set(r["answer_token_ids"])
    f_qa, r_qa = set(f["qa_token_ids"]), set(r["qa_token_ids"])
    f_words, r_words = set(f["words"]), set(r["words"])
    ans_j = jaccard(f_ans, r_ans)
    ans_cont = max(containment(f_ans, r_ans), containment(r_ans, f_ans))
    qa_j = jaccard(f_qa, r_qa)
    txt_j = jaccard(f_words, r_words)
    score = max(ans_j, 0.85 * ans_cont, 0.55 * qa_j, 0.65 * txt_j)
    return {
        "score": float(score),
        "answer_jaccard": float(ans_j),
        "answer_containment": float(ans_cont),
        "qa_jaccard": float(qa_j),
        "text_jaccard": float(txt_j),
    }


def best_retain_match(f: Dict[str, Any], retain_records: Sequence[Dict[str, Any]]) -> Tuple[float, Optional[int], Dict[str, float]]:
    best_score, best_idx, best_parts = 0.0, None, {}
    f_ans = set(f["answer_token_ids"])
    f_words = set(f["words"])
    for r in retain_records:
        # cheap prefilter
        if not (f_ans & set(r["answer_token_ids"])) and not (f_words & set(r["words"])):
            continue
        parts = record_sim(f, r)
        if parts["score"] > best_score:
            best_score = parts["score"]
            best_idx = int(r["idx"])
            best_parts = parts
    return float(best_score), best_idx, best_parts


def token_global_stats(
    tid: int,
    f_ans_count: Counter,
    f_ans_doc: Counter,
    f_qa_doc: Counter,
    r_ans_count: Counter,
    r_ans_doc: Counter,
    r_qa_doc: Counter,
    n_forget: int,
    n_retain: int,
) -> Dict[str, Any]:
    fqa = int(f_qa_doc.get(tid, 0))
    rqa = int(r_qa_doc.get(tid, 0))
    fr = fqa / max(1, n_forget)
    rr = rqa / max(1, n_retain)
    return {
        "freq_forget": fqa,
        "freq_retain": rqa,
        "forget_answer_count": int(f_ans_count.get(tid, 0)),
        "forget_answer_doc_count": int(f_ans_doc.get(tid, 0)),
        "retain_answer_count": int(r_ans_count.get(tid, 0)),
        "retain_answer_doc_count": int(r_ans_doc.get(tid, 0)),
        "forget_ratio": float(fr),
        "retain_ratio": float(rr),
        "contrast_score": float((fr + 1e-8) / (rr + 1e-8)),
    }


def allow_token(stats: Dict[str, Any], is_answer: bool, is_json: bool, record_level: str, args) -> Tuple[bool, str]:
    if record_level == "shielded":
        return False, "protect_high_record_match"

    retain_ans = int(stats["retain_answer_count"])
    retain_doc = int(stats["freq_retain"])
    contrast = float(stats["contrast_score"])

    if record_level == "partial":
        if retain_ans > args.partial_max_retain_answer_count:
            return False, "partial_high_retain_answer"
        if retain_doc > args.partial_max_retain_doc_count:
            return False, "partial_high_retain_doc"
        if contrast < args.partial_min_contrast:
            return False, "partial_low_contrast"
        return True, "partial_safe_unique_token"

    # unique forget record
    if is_answer:
        if retain_ans > args.unique_answer_max_retain_answer_count:
            return False, "unique_answer_high_retain_answer"
        if retain_doc > args.unique_answer_max_retain_doc_count:
            return False, "unique_answer_high_retain_doc"
        return True, "unique_answer_token"

    if is_json:
        if retain_doc > args.unique_json_max_retain_doc_count:
            return False, "unique_json_high_retain_doc"
        if contrast < args.unique_json_min_contrast:
            return False, "unique_json_low_contrast"
        return True, "unique_json_token"

    return False, "not_answer_or_json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--forget-json", default="outputs/semantic_tokens_json_raw.json")
    ap.add_argument("--retain-json", default=None, help="Optional retain JSON token file. If absent, retain QA/answer records are used.")
    ap.add_argument("--out", default="outputs/semantic_tokens_dual_json_retain_shield_balanced.json")
    ap.add_argument("--model-name", default=None)

    ap.add_argument("--high-sim-threshold", type=float, default=0.85)
    ap.add_argument("--partial-sim-threshold", type=float, default=0.45)
    ap.add_argument("--include-question-tokens", action="store_true")
    ap.add_argument("--min-token-len", type=int, default=1)
    ap.add_argument("--max-final-tokens", type=int, default=1200)

    ap.add_argument("--unique-answer-max-retain-answer-count", type=int, default=8)
    ap.add_argument("--unique-answer-max-retain-doc-count", type=int, default=35)
    ap.add_argument("--unique-json-max-retain-doc-count", type=int, default=20)
    ap.add_argument("--unique-json-min-contrast", type=float, default=3.0)

    ap.add_argument("--partial-max-retain-answer-count", type=int, default=0)
    ap.add_argument("--partial-max-retain-doc-count", type=int, default=3)
    ap.add_argument("--partial-min-contrast", type=float, default=15.0)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]

    forget_json = Path(args.forget_json) if args.forget_json else None
    retain_json = Path(args.retain_json) if args.retain_json else None
    out_path = Path(args.out)

    print("=" * 80)
    print("Dual-JSON Retain Shield token builder")
    print("=" * 80)
    print(f"Model/tokenizer: {model_name}")
    print(f"Forget split:    {forget_split}")
    print(f"Retain split:    {retain_split}")
    print(f"Forget JSON:     {forget_json}")
    print(f"Retain JSON:     {retain_json}")
    print(f"Output:          {out_path}")
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    forget_json_ids = get_token_ids_from_json(forget_json)
    retain_json_ids = get_token_ids_from_json(retain_json) if retain_json and retain_json.exists() else set()

    print(f"Forget JSON tokens: {len(forget_json_ids)}")
    if retain_json_ids:
        print(f"Retain JSON tokens: {len(retain_json_ids)}")
    else:
        print("No retain JSON token file found/provided. Using retain QA/answer records as shield.")

    forget_records = build_records(forget_ds, tok, forget_split, forget_json_ids, args.include_question_tokens)
    retain_records = build_records(retain_ds, tok, retain_split, retain_json_ids, args.include_question_tokens)

    f_ans_count, f_ans_doc, f_qa_doc = count_stats(forget_records)
    r_ans_count, r_ans_doc, r_qa_doc = count_stats(retain_records)

    selected: Dict[int, Dict[str, Any]] = {}
    protected: List[Dict[str, Any]] = []
    record_infos: List[Dict[str, Any]] = []
    record_level_counts = Counter()
    reason_counts = Counter()

    print("[Shield] Matching forget records against retain records...")
    for fr in tqdm(forget_records, desc="DualJSON shield"):
        sim, retain_idx, parts = best_retain_match(fr, retain_records)
        if sim >= args.high_sim_threshold:
            record_level = "shielded"
        elif sim >= args.partial_sim_threshold:
            record_level = "partial"
        else:
            record_level = "unique"
        record_level_counts[record_level] += 1

        ans_ids = set(int(x) for x in fr["answer_token_ids"])
        json_ids = set(int(x) for x in fr["json_token_ids"])
        cand_ids = set(ans_ids) | set(json_ids)
        if args.include_question_tokens:
            cand_ids |= set(int(x) for x in fr["question_token_ids"])

        selected_for_record = []
        protected_for_record = []

        for tid in sorted(cand_ids):
            tid = int(tid)
            if is_bad_token(tok, tid, args.min_token_len):
                reason = "special_or_blank"
                reason_counts[reason] += 1
                protected_for_record.append({"token_id": tid, "reason": reason})
                continue

            stats = token_global_stats(
                tid, f_ans_count, f_ans_doc, f_qa_doc, r_ans_count, r_ans_doc, r_qa_doc,
                len(forget_records), len(retain_records)
            )
            is_answer = tid in ans_ids
            is_json = tid in json_ids or tid in forget_json_ids
            ok, reason = allow_token(stats, is_answer, is_json, record_level, args)
            reason_counts[reason] += 1

            row = {
                "token_id": tid,
                "token_str": tok.decode([tid]),
                "source": "dual_json_retain_shield",
                "bucket": reason,
                "record_level": record_level,
                "forget_record_idx": int(fr["idx"]),
                "best_retain_record_idx": retain_idx,
                "best_retain_similarity": float(sim),
                "similarity_parts": parts,
                "is_answer_token": bool(is_answer),
                "is_json": bool(is_json),
                "is_residual_answer": bool(is_answer),
                "erase_strength": 1.0,
                "output_strength": 1.0,
                "edit_lm_head": True,
                **stats,
            }

            if ok:
                if tid not in selected:
                    selected[tid] = row
                    selected[tid]["n_records"] = 1
                else:
                    selected[tid]["n_records"] = int(selected[tid].get("n_records", 1)) + 1
                    # prefer unique/answer labeling if any record gives it
                    if record_level == "unique" and selected[tid].get("record_level") != "unique":
                        selected[tid].update(row)
                selected_for_record.append(tid)
            else:
                row["protect_reason"] = reason
                protected.append(row)
                protected_for_record.append({"token_id": tid, "reason": reason})

        record_infos.append({
            "forget_record_idx": int(fr["idx"]),
            "question": fr["question"],
            "answer": fr["answer"],
            "record_level": record_level,
            "best_retain_record_idx": retain_idx,
            "best_retain_similarity": float(sim),
            "similarity_parts": parts,
            "n_candidate_tokens": len(cand_ids),
            "n_selected_tokens": len(selected_for_record),
            "selected_token_ids": selected_for_record,
            "protected_tokens": protected_for_record[:50],
        })

    rows = list(selected.values())
    rows.sort(key=lambda x: (
        0 if x.get("record_level") == "unique" else 1,
        0 if x.get("is_answer_token") else 1,
        int(x.get("retain_answer_count", 999999)),
        int(x.get("freq_retain", 999999)),
        -float(x.get("contrast_score", 0.0)),
        -int(x.get("forget_answer_count", 0)),
        int(x["token_id"]),
    ))

    before_cap = len(rows)
    if args.max_final_tokens > 0 and len(rows) > args.max_final_tokens:
        for x in rows[args.max_final_tokens:]:
            y = dict(x)
            y["protect_reason"] = "over_max_final_tokens_cap"
            protected.append(y)
        rows = rows[:args.max_final_tokens]

    bucket_counts = Counter(str(x.get("bucket", "unknown")) for x in rows)
    selected_levels = Counter(str(x.get("record_level", "unknown")) for x in rows)

    output = {
        "method": "dual_json_retain_shield_hard_zero_tokens",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "forget_json_file": str(forget_json) if forget_json else None,
        "retain_json_file": str(retain_json) if retain_json else None,
        "record_level_counts": dict(record_level_counts),
        "reason_counts_all": dict(reason_counts),
        "n_kept_before_cap": before_cap,
        "n_semantic_tokens": len(rows),
        "n_protected_tokens": len(protected),
        "bucket_counts": dict(bucket_counts),
        "selected_record_level_counts": dict(selected_levels),
        "filter_config": vars(args),
        "token_ids": [int(x["token_id"]) for x in rows],
        "token_strings": [str(x.get("token_str", "")) for x in rows],
        "semantic_tokens": rows,
        "protected_tokens": protected,
        "record_infos": record_infos,
    }

    save_json(output, out_path)
    save_json(protected, out_path.parent / "protected_tokens_dual_json_retain_shield.json")
    save_json(record_infos, out_path.parent / "record_infos_dual_json_retain_shield.json")

    print("\n[Done]")
    print(f"Record levels:       {dict(record_level_counts)}")
    print(f"Selected tokens:     {len(rows)}")
    print(f"Selected before cap: {before_cap}")
    print(f"Protected tokens:    {len(protected)}")
    print(f"Bucket counts:       {dict(bucket_counts)}")
    print(f"Selected levels:     {dict(selected_levels)}")
    print(f"Output file:         {out_path}")

    print("\nTop selected tokens:")
    for x in rows[:100]:
        print(
            f"{int(x['token_id']):>8} | {repr(str(x.get('token_str', '')))} | "
            f"level={x.get('record_level')} bucket={x.get('bucket')} | "
            f"ans={x.get('is_answer_token')} json={x.get('is_json')} | "
            f"fa={x.get('forget_answer_count')} ra={x.get('retain_answer_count')} "
            f"fq={x.get('freq_forget')} rq={x.get('freq_retain')} | "
            f"sim={float(x.get('best_retain_similarity', 0)):.3f} "
            f"contrast={float(x.get('contrast_score', 0)):.1f}"
        )


if __name__ == "__main__":
    main()
