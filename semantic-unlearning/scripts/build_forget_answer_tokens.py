#!/usr/bin/env python3

import argparse
import json
import yaml
from collections import Counter
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    parser.add_argument("--forget-split", default="forget05")
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--out", default="outputs/semantic_tokens_forget_answer_tokens_aggressive.json")
    parser.add_argument("--max-retain-answer-count", type=int, default=8)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["name"]
    tok = AutoTokenizer.from_pretrained(model_name)

    forget = load_dataset("locuslab/TOFU", name=args.forget_split, split="train")
    retain = load_dataset("locuslab/TOFU", name=args.retain_split, split="train")

    def ids_from_answer(row):
        return tok.encode(" " + str(row["answer"]).strip(), add_special_tokens=False)

    forget_counts = Counter()
    forget_doc = Counter()
    retain_counts = Counter()
    retain_doc = Counter()

    for row in forget:
        ids = ids_from_answer(row)
        forget_counts.update(ids)
        for x in set(ids):
            forget_doc[int(x)] += 1

    for row in retain:
        ids = ids_from_answer(row)
        retain_counts.update(ids)
        for x in set(ids):
            retain_doc[int(x)] += 1

    rows = []

    for tid, fc in forget_counts.items():
        tid = int(tid)
        rc = int(retain_counts.get(tid, 0))
        fdoc = int(forget_doc.get(tid, 0))
        rdoc = int(retain_doc.get(tid, 0))

        token_str = tok.decode([tid])

        if len(token_str.strip()) < 1:
            continue

        # This is the threshold you wanted to modify.
        # Default is 8. For more aggressive runs, use 20 or 40.
        if rc > args.max_retain_answer_count:
            continue

        contrast = (fdoc / max(1, len(forget)) + 1e-8) / (rdoc / max(1, len(retain)) + 1e-8)

        rows.append({
            "token_id": tid,
            "token_str": token_str,
            "source": "forget_answer_token",
            "bucket": "forget_answer_aggressive",
            "is_residual_answer": True,
            "is_json": False,
            "freq_forget": fdoc,
            "freq_retain": rdoc,
            "forget_answer_count": int(fc),
            "retain_answer_count": int(rc),
            "contrast_score": float(contrast),
            "erase_strength": 1.0,
            "output_strength": 1.0,
            "edit_lm_head": True,
        })

    rows.sort(
        key=lambda x: (
            -x["forget_answer_count"],
            x["retain_answer_count"],
            -x["contrast_score"],
        )
    )

    data = {
        "method": "forget_answer_tokens_aggressive",
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "max_retain_answer_count": args.max_retain_answer_count,
        "n_semantic_tokens": len(rows),
        "token_ids": [r["token_id"] for r in rows],
        "token_strings": [r["token_str"] for r in rows],
        "semantic_tokens": rows,
    }

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("saved:", args.out)
    print("tokens:", len(rows))
    print("top 80:")
    for r in rows[:80]:
        print(
            r["token_id"],
            repr(r["token_str"]),
            "fc=", r["forget_answer_count"],
            "rc=", r["retain_answer_count"],
            "contrast=", round(r["contrast_score"], 2),
        )


if __name__ == "__main__":
    main()
