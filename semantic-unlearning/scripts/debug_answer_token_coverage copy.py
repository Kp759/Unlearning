#!/usr/bin/env python3
"""
Debug how much of the forget-answer text is covered by a selected token JSON.

Use this before masked LM-head GA/GD. If answer-token coverage is low, the
LM-head row mask cannot strongly reduce forget answer probability.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml
from datasets import load_dataset
from transformers import AutoTokenizer


def load_selected_ids(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "token_ids" in data:
        return {int(x) for x in data["token_ids"]}

    if "semantic_tokens" in data:
        return {int(x["token_id"]) for x in data["semantic_tokens"]}

    raise ValueError(f"No token_ids or semantic_tokens field found in {path}")


def encode_answer(tokenizer, answer: str):
    return [int(x) for x in tokenizer.encode(" " + str(answer).strip(), add_special_tokens=False)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    parser.add_argument("--token-json", default="outputs/semantic_tokens_answer_tfidf.json")
    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--print-missing", type=int, default=80)
    parser.add_argument("--print-low-coverage-examples", type=int, default=20)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    forget_split = args.forget_split or cfg["data"]["forget_split"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    selected = load_selected_ids(Path(args.token_json))

    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")

    total_occ = 0
    covered_occ = 0
    total_types_by_ex = 0
    covered_types_by_ex = 0
    covered_examples = 0
    missing_counter = Counter()
    low_examples = []

    for idx, row in enumerate(forget_ds):
        occ_ids = encode_answer(tokenizer, row["answer"])
        type_ids = sorted(set(occ_ids))
        hit_types = sorted(set(type_ids) & selected)

        if hit_types:
            covered_examples += 1

        total_occ += len(occ_ids)
        covered_occ += sum(1 for x in occ_ids if int(x) in selected)

        total_types_by_ex += len(type_ids)
        covered_types_by_ex += len(hit_types)

        for tid in occ_ids:
            if int(tid) not in selected:
                missing_counter[int(tid)] += 1

        type_cov = len(hit_types) / max(1, len(type_ids))
        occ_cov = sum(1 for x in occ_ids if int(x) in selected) / max(1, len(occ_ids))
        if type_cov < 0.50:
            low_examples.append(
                {
                    "idx": idx,
                    "answer": str(row["answer"]),
                    "type_cov": type_cov,
                    "occ_cov": occ_cov,
                    "selected_tokens": [(tid, tokenizer.decode([tid])) for tid in hit_types],
                    "missing_tokens": [(tid, tokenizer.decode([tid])) for tid in type_ids if tid not in selected],
                }
            )

    print("=" * 80)
    print("[Coverage] Forget answer token coverage")
    print("=" * 80)
    print(f"token json: {args.token_json}")
    print(f"selected tokens: {len(selected)}")
    print(f"forget split: {forget_split}")
    print(f"model/tokenizer: {model_name}")
    print("-" * 80)
    print(f"example coverage: {covered_examples}/{len(forget_ds)} = {covered_examples / max(1, len(forget_ds)):.4f}")
    print(f"answer-token occurrence coverage: {covered_occ}/{total_occ} = {covered_occ / max(1, total_occ):.4f}")
    print(f"answer-token type coverage by example: {covered_types_by_ex}/{total_types_by_ex} = {covered_types_by_ex / max(1, total_types_by_ex):.4f}")

    print("\nTop missing answer tokens:")
    for tid, c in missing_counter.most_common(args.print_missing):
        print(f"{tid:>8} | {repr(tokenizer.decode([tid]))} | count={c}")

    print("\nLow-coverage examples:")
    for ex in low_examples[: args.print_low_coverage_examples]:
        print("-" * 80)
        print(f"idx={ex['idx']} type_cov={ex['type_cov']:.3f} occ_cov={ex['occ_cov']:.3f}")
        print("answer:", ex["answer"])
        print("selected:", ex["selected_tokens"])
        print("missing:", ex["missing_tokens"])


if __name__ == "__main__":
    main()
