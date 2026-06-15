#!/usr/bin/env python3
"""SimNPO unlearning runner for MCF/TOFU.

This script is intentionally self-contained.  It supports the prompt-format
switch needed for TOFU/Instruct experiments so training prompts can match
`scripts/tofu_eval.py`.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


@dataclass(frozen=True)
class Example:
    prompt: str
    answer: str


def format_tofu_prompt(tokenizer, question: str, prompt_format: str) -> str:
    content = f"Question: {question} Answer:"
    if prompt_format == "plain":
        return f"Question: {question}\nAnswer:"
    if prompt_format == "chat":
        if tokenizer.chat_template is None:
            raise ValueError("--tofu-prompt-format chat requested but tokenizer has no chat_template")
        msg = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    if prompt_format == "auto":
        if tokenizer.chat_template is not None:
            msg = [{"role": "user", "content": content}]
            return tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        return f"Question: {question}\nAnswer:"
    raise ValueError(f"Unknown prompt_format: {prompt_format}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tofu_examples(tokenizer, split: str, limit: int, seed: int, prompt_format: str) -> List[Example]:
    rows = list(load_dataset("locuslab/TOFU", name=split, split="train"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples: List[Example] = []
    for row in rows[:limit]:
        prompt = format_tofu_prompt(tokenizer, row["question"], prompt_format)
        answer = " " + row["answer"].strip()
        examples.append(Example(prompt=prompt, answer=answer))
    return examples


def format_mcf_prompt(row: dict) -> str:
    rr = row["requested_rewrite"]
    prompt = rr["prompt"]
    subject = rr["subject"]
    return prompt.format(subject) if "{}" in prompt else prompt


def load_mcf_examples(path: Path, limit: int, seed: int) -> List[Example]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    rng = random.Random(seed)
    rng.shuffle(rows)
    examples: List[Example] = []
    for row in rows[:limit]:
        examples.append(Example(prompt=format_mcf_prompt(row), answer=" " + row["requested_rewrite"]["target_new"]["str"].strip()))
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs/simnpo_mcf_tofu")
    parser.add_argument("--dataset", choices=["tofu", "mcf"], default="tofu")
    parser.add_argument("--forget-split", default="forget05")
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--mcf-path", default="data/mcf/multi_counterfact.json")
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--tofu-prompt-format", choices=["auto", "chat", "plain"], default="auto")
    parser.add_argument("--dry-run", action="store_true", help="Build datasets and summary without training.")
    return parser.parse_args()


def write_summary(args: argparse.Namespace, forget: Sequence[Example], retain: Sequence[Example]) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "method": "simnpo",
        "dataset": args.dataset,
        "model_path": args.model_path,
        "forget_examples": len(forget),
        "retain_examples": len(retain),
        "steps": args.steps,
        "lr": args.lr,
        "tofu_prompt_format": args.tofu_prompt_format,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.dataset == "tofu":
        forget = load_tofu_examples(tokenizer, args.forget_split, args.forget_num, args.seed, args.tofu_prompt_format)
        retain = load_tofu_examples(tokenizer, args.retain_split, args.retain_num, args.seed, args.tofu_prompt_format)
        print("[TOFU prompt preview]", repr(forget[0].prompt[:300]))
    else:
        mcf_path = Path(args.mcf_path)
        forget = load_mcf_examples(mcf_path, args.forget_num, args.seed)
        retain = load_mcf_examples(mcf_path, args.retain_num, args.seed + 1)

    if not args.dry_run:
        model = AutoModelForCausalLM.from_pretrained(args.model_path)
        model.config.use_cache = False
        del model
    write_summary(args, forget, retain)


if __name__ == "__main__":
    main()
