#!/usr/bin/env python3
"""Train the four GA/GD TOFU settings with evaluator-compatible prompts.

This is an isolated TOFU front end over the existing ``gagd_compare`` training
implementation.  Unlike the older generic loader, it formats each question
with the tokenizer chat template and includes the leading answer space used by
``tofu_eval.py``.  That removes the train/evaluation prompt mismatch while
preserving the four parameter/token scopes and GA/GD update logic.
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import gagd_compare as gagd
import tofu_gagd_neighborhood_confidence as tofu
from controlled_unlearning_protocol import load_json_or_jsonl


MODES = tuple(gagd.BASE_MODES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--mode",
        choices=[*MODES, "all"],
        default="all",
    )
    parser.add_argument(
        "--forget-split",
        choices=sorted(tofu.PAIRED_RETAIN_SPLITS),
        default="forget05",
    )
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument(
        "--controlled-input-dir",
        default=None,
        help=(
            "Leakage-controlled stage directory with forget.json and "
            "retain.json. When set, Hugging Face data is not loaded."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help=(
            "Default is one epoch over forget05. With retain batch size 5, "
            "the paired 1,000-record retain sample is also covered once."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--retain-batch-size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--full-lr", type=float, default=1e-5)
    parser.add_argument("--emb-lm-lr", type=float, default=2e-4)
    parser.add_argument("--forget-weight", type=float, default=1.0)
    parser.add_argument(
        "--retain-weight",
        type=float,
        default=5.0,
        help=(
            "Compensates for the five-record retain batch so each sampled "
            "forget and retain record has comparable coefficient mass."
        ),
    )
    parser.add_argument("--kl-retain-weight", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--optimizer",
        choices=["sgd", "adam", "adamw", "adamw8bit"],
        default=None,
    )
    parser.add_argument(
        "--full-optimizer",
        choices=["sgd", "adam", "adamw", "adamw8bit"],
        default=None,
    )
    parser.add_argument(
        "--emb-lm-optimizer",
        choices=["sgd", "adam", "adamw", "adamw8bit"],
        default=None,
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["epoch", "with_replacement"],
        default="epoch",
    )
    parser.add_argument("--selective-top-k", type=int, default=1000)
    parser.add_argument("--semantic-token-json", default=None)
    parser.add_argument(
        "--max-eval-examples",
        type=int,
        default=32,
        help="Small training diagnostic only; full tofu_eval runs later.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected = tofu.PAIRED_RETAIN_SPLITS[args.forget_split]
    if args.retain_split != expected:
        raise ValueError(
            f"{args.forget_split} must be paired with {expected}, "
            f"not {args.retain_split}"
        )
    for name in (
        "forget_num",
        "retain_num",
        "steps",
        "batch_size",
        "retain_batch_size",
        "selective_top_k",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("lr", "full_lr", "emb_lm_lr", "forget_weight"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in (
        "retain_weight",
        "kl_retain_weight",
        "weight_decay",
        "grad_clip",
    ):
        if float(getattr(args, name)) < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be non-negative"
            )


def load_official_prompt_examples(
    tok: Any,
    split: str,
    count: int,
    seed: int,
    controlled_input_dir: str | None = None,
) -> List[gagd.Example]:
    if controlled_input_dir:
        controlled_name = (
            "forget"
            if split in tofu.PAIRED_RETAIN_SPLITS
            else "retain"
        )
        controlled_dir = Path(controlled_input_dir)
        candidates = (
            controlled_dir / f"{controlled_name}.json",
            controlled_dir / f"{controlled_name}.jsonl",
        )
        path = next(
            (candidate for candidate in candidates if candidate.exists()),
            None,
        )
        if path is None:
            raise FileNotFoundError(
                f"Controlled TOFU input lacks {controlled_name}.json/jsonl "
                f"in {controlled_dir}"
            )
        rows = load_json_or_jsonl(path)
    else:
        rows = list(load_dataset("locuslab/TOFU", name=split, split="train"))
    indices = tofu.deterministic_sample_indices(len(rows), count, seed)
    examples: List[gagd.Example] = []
    for source_index in indices:
        row = rows[source_index]
        question = str(row["question"])
        answer = gagd.normalize_answer(str(row["answer"]).strip())
        examples.append(
            gagd.Example(
                prompt=tofu.format_question_prompt(tok, question),
                answer=answer,
                source=f"tofu:{split}:{source_index}",
            )
        )
    return examples


def resolved_modes(mode: str) -> List[str]:
    return list(MODES) if mode == "all" else [mode]


def _complete_gagd_args(args: argparse.Namespace) -> argparse.Namespace:
    # Attributes consumed by the shared training/evaluation functions.
    values = {
        **vars(args),
        "dataset": "tofu",
        "forget_loss_type": "answer_nll",
        "forget_margin": 0.0,
        "mcf_answer_field": "target_new",
        "post_training_new_true_alpha": gagd.POST_TRAINING_NEW_TRUE_ALPHA,
        "post_training_new_retain_alpha": gagd.POST_TRAINING_NEW_RETAIN_ALPHA,
        "post_training_new_true_retain_alpha": (
            gagd.POST_TRAINING_NEW_TRUE_RETAIN_ALPHA
        ),
    }
    return argparse.Namespace(**values)


def main() -> None:
    parsed = build_parser().parse_args()
    validate_args(parsed)
    args = _complete_gagd_args(parsed)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_tok = AutoTokenizer.from_pretrained(args.model_path)
    if data_tok.pad_token is None:
        data_tok.pad_token = data_tok.eos_token
    forget = load_official_prompt_examples(
        data_tok,
        args.forget_split,
        args.forget_num,
        args.seed,
        args.controlled_input_dir,
    )
    retain = load_official_prompt_examples(
        data_tok,
        args.retain_split,
        args.retain_num,
        args.seed,
        args.controlled_input_dir,
    )
    modes = resolved_modes(args.mode)
    config_used = {
        **vars(args),
        "modes": modes,
        "prompt_construction": "tofu_eval_chat_template",
        "answer_construction": "leading_space_plus_answer",
        "resolved_mode_settings": {
            mode: {
                "learning_rate": gagd.learning_rate_for_mode(mode, args),
                "optimizer": gagd.optimizer_name_for_mode(mode, args),
            }
            for mode in modes
        },
    }
    gagd.write_json(output_dir / "config_used.json", config_used)

    print("Loading base model for token selection and diagnostics")
    base_model, tok = gagd.load_model_and_tokenizer(
        args,
        for_training=False,
    )
    selected_ids = gagd.select_tofu_tokens(tok, forget, retain, args)
    if not selected_ids and any("selective_tokens" in mode for mode in modes):
        raise RuntimeError("TOFU selective row selection returned no tokens")
    gagd.write_json(
        output_dir / "selected_token_ids.json",
        {
            "token_ids": selected_ids,
            "n_selected_tokens": len(selected_ids),
            "tokens": {
                str(token_id): tok.decode([token_id])
                for token_id in selected_ids
            },
        },
    )
    base_metrics = gagd.evaluate(base_model, tok, forget, retain, args)
    gagd.write_json(output_dir / "base_metrics.json", base_metrics)
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    comparison: List[Dict[str, Any]] = []
    for mode in modes:
        print(f"\n=== TOFU evaluator-compatible mode: {mode} ===")
        gagd.set_seed(args.seed)
        model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
        summary = gagd.train_mode(
            model,
            tok,
            forget,
            retain,
            selected_ids,
            mode,
            args,
            output_dir / mode,
        )
        metrics = gagd.evaluate(model, tok, forget, retain, args)
        metrics.update(asdict(summary))
        metrics["learning_rate"] = args.effective_lr
        metrics["optimizer"] = args.effective_optimizer
        metrics["n_selected_tokens"] = len(selected_ids)
        metrics["prompt_construction"] = "tofu_eval_chat_template"
        gagd.write_json(output_dir / mode / "metrics.json", metrics)
        comparison.append(
            {
                "mode": mode,
                "forget_nll_before": base_metrics["forget_answer_nll"],
                "forget_nll_after": metrics["forget_answer_nll"],
                "retain_nll_before": base_metrics["retain_answer_nll"],
                "retain_nll_after": metrics["retain_answer_nll"],
                "learning_rate": args.effective_lr,
                "optimizer": args.effective_optimizer,
                "checkpoint": str(output_dir / mode / "checkpoint"),
            }
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gagd.write_json(output_dir / "comparison_diagnostic.json", comparison)
    gagd.write_comparison_csv(output_dir / "comparison_diagnostic.csv", comparison)
    gagd.write_comparison_md(output_dir / "comparison_diagnostic.md", comparison)
    print(f"Four TOFU checkpoints written under {output_dir}")


if __name__ == "__main__":
    main()
