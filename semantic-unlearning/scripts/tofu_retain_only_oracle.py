#!/usr/bin/env python3
"""Train the TOFU retain-only retraining oracle.

This is not an unlearning method.  It starts from the pre-TOFU base model and
fine-tunes only on the paired retain split, using the full-model TOFU
fine-tuning metadata as the default source of the base revision and training
hyperparameters.  Its evaluation row is the retraining upper-bound/reference
required by the TOFU protocol.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "tofu_retain_only_oracle"


class RetainOnlyDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Dict[str, Any]],
        tok: Any,
        max_length: int,
    ) -> None:
        self.examples: List[Dict[str, torch.Tensor]] = []
        eos = tok.eos_token or ""
        for sample in samples:
            prompt = tofu.format_question_prompt(tok, str(sample["question"]))
            full_text = prompt + f" {sample['answer']}{eos}"
            full = tok(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            prompt_ids = tok(
                prompt,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"]
            labels = full["input_ids"][0].clone()
            labels[: prompt_ids.shape[1]] = -100
            if labels.ne(-100).sum() == 0:
                raise ValueError(
                    "A retain answer has no training tokens after truncation"
                )
            self.examples.append(
                {
                    "input_ids": full["input_ids"][0],
                    "attention_mask": full["attention_mask"][0],
                    "labels": labels,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]


def collate_retain_batch(
    batch: Sequence[Dict[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    width = max(row["input_ids"].shape[0] for row in batch)
    input_ids = torch.full(
        (len(batch), width),
        pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for row_index, row in enumerate(batch):
        length = row["input_ids"].shape[0]
        input_ids[row_index, :length] = row["input_ids"]
        attention_mask[row_index, :length] = row["attention_mask"]
        labels[row_index, :length] = row["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-model-path",
        required=True,
        help=(
            "Full-TOFU fine-tuned model. finetune_metadata.json is used to "
            "recover the exact pre-TOFU base and default hyperparameters."
        ),
    )
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Explicit pre-TOFU base; overrides recovered metadata.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--forget-split",
        choices=sorted(tofu.PAIRED_RETAIN_SPLITS),
        default="forget05",
    )
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def load_full_finetune_metadata(full_model_path: str) -> Dict[str, Any]:
    path = Path(full_model_path) / "finetune_metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            "Retain-only oracle requires full-model provenance at "
            f"{path}; pass a full checkpoint containing that file."
        )
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("split") != "full":
        raise ValueError("Oracle provenance model was not trained on TOFU full")
    return metadata


def resolve_training_config(
    args: argparse.Namespace,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    base_model_path = args.base_model_path or metadata.get("base_model")
    if not base_model_path:
        raise ValueError(
            "Could not recover the pre-TOFU base model; pass --base-model-path"
        )
    epochs = args.epochs if args.epochs is not None else metadata.get("epochs")
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else metadata.get("batch_size")
    )
    lr = args.lr if args.lr is not None else metadata.get("lr")
    if not epochs or not batch_size or not lr:
        raise ValueError(
            "Could not recover oracle epochs, batch size, and learning rate"
        )
    return {
        "base_model_path": str(base_model_path),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "lr": float(lr),
    }


def validate_args(
    args: argparse.Namespace,
    resolved: Dict[str, Any],
) -> None:
    expected = tofu.PAIRED_RETAIN_SPLITS[args.forget_split]
    if args.retain_split != expected:
        raise ValueError(
            f"{args.forget_split} must be paired with {expected}, "
            f"not {args.retain_split}"
        )
    for name in ("epochs", "batch_size"):
        if int(resolved[name]) <= 0:
            raise ValueError(f"Resolved {name} must be positive")
    if not math.isfinite(resolved["lr"]) or resolved["lr"] <= 0:
        raise ValueError("Resolved learning rate must be finite and positive")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must lie in [0,1)")
    if args.max_length <= 0 or args.grad_clip <= 0:
        raise ValueError("--max-length and --grad-clip must be positive")


def main() -> None:
    args = build_parser().parse_args()
    metadata = load_full_finetune_metadata(args.full_model_path)
    resolved = resolve_training_config(args, metadata)
    validate_args(args, resolved)
    gagd.set_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for retain-only oracle training")

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading pre-TOFU base: {resolved['base_model_path']}")
    tok = AutoTokenizer.from_pretrained(resolved["base_model_path"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        resolved["base_model_path"],
        torch_dtype=gagd.torch_dtype(args.dtype),
    ).to(args.device)
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    retain_rows = list(
        load_dataset("locuslab/TOFU", name=args.retain_split, split="train")
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    dataset = RetainOnlyDataset(retain_rows, tok, args.max_length)
    dataloader = DataLoader(
        dataset,
        batch_size=resolved["batch_size"],
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda batch: collate_retain_batch(
            batch,
            int(tok.pad_token_id),
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=resolved["lr"],
        weight_decay=args.weight_decay,
    )
    total_steps = resolved["epochs"] * len(dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    from torch.optim.lr_scheduler import LinearLR, SequentialLR

    if warmup_steps <= 0 or warmup_steps >= total_steps:
        scheduler = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=max(total_steps, 1),
        )
    else:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=warmup_steps,
                ),
                LinearLR(
                    optimizer,
                    start_factor=1.0,
                    end_factor=0.1,
                    total_iters=total_steps - warmup_steps,
                ),
            ],
            milestones=[warmup_steps],
        )
    logs: List[Dict[str, Any]] = []
    global_step = 0
    for epoch in range(1, resolved["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(dataloader, desc=f"retain-oracle epoch {epoch}")
        for batch in progress:
            batch = {
                key: value.to(args.device)
                for key, value in batch.items()
            }
            output = model(**batch)
            loss = output.loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite oracle loss at step {global_step + 1}"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_loss += float(loss.detach().cpu())
            progress.set_postfix(loss=f"{float(loss.detach().cpu()):.4f}")
        logs.append(
            {
                "epoch": epoch,
                "mean_loss": epoch_loss / len(dataloader),
                "global_step": global_step,
                "learning_rate": scheduler.get_last_lr()[0],
                "last_gradient_norm": float(grad_norm.detach().cpu()),
            }
        )
        active.write_jsonl(output_dir / "train_log.jsonl", logs)

    config_used = {
        **vars(args),
        **resolved,
        "method": METHOD,
        "full_finetune_metadata": metadata,
        "training_split": args.retain_split,
        "training_example_count": len(retain_rows),
        "oracle_definition": "pre-TOFU base retrained only on retain split",
    }
    gagd.write_json(output_dir / "config_used.json", config_used)
    summary = {
        "method": METHOD,
        "checkpoint": str(output_dir / "checkpoint"),
        "base_model_path": resolved["base_model_path"],
        "retain_split": args.retain_split,
        "training_example_count": len(retain_rows),
        "epochs": resolved["epochs"],
        "batch_size": resolved["batch_size"],
        "lr": resolved["lr"],
        "final_loss": logs[-1]["mean_loss"],
        "checkpoint_saved": bool(args.save_model),
    }
    gagd.write_json(output_dir / "oracle_summary.json", summary)
    if args.save_model:
        checkpoint = output_dir / "checkpoint"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint)
        tok.save_pretrained(checkpoint)
        # Preserve enough provenance for reuse without retraining.
        gagd.write_json(checkpoint / "retain_oracle_metadata.json", config_used)
    print(f"Retain-only oracle checkpoint: {output_dir / 'checkpoint'}")


if __name__ == "__main__":
    main()
