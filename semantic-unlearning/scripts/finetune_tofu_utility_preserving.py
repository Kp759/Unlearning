#!/usr/bin/env python3
"""Utility-preserving full-model fine-tuning for the full TOFU dataset.

This is a separate experiment from ``finetune_tofu.py``.  It trains on all
4,000 full-TOFU question/answer records, supervises only answer and EOS
tokens, and records fixed post-epoch probes for TOFU fit and external utility.
It does not run the full ECO evaluator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODEL_PATH = (
    "/scratch/yl258/kp759/hf/"
    "models--meta-llama--Llama-3.2-3B-Instruct/snapshots/"
    "0cb88a4f764b7a12671c53f0838cd831a0843b95"
)
DATASET_ID = "locuslab/TOFU"
DATASET_CONFIG = "full"
DATASET_SPLIT = "train"
EXPECTED_FULL_EXAMPLES = 4000
PROBE_SIZE = 20
PROBE_MAX_NEW_TOKENS = 64
ELIGIBILITY_THRESHOLDS = {
    "tofu_probe_rouge_l": 0.95,
    "real_author_probe_rouge_l": 0.75,
    "world_fact_probe_rouge_l": 0.75,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically so interrupted jobs do not leave partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def format_question_prompt(tokenizer: Any, question: str) -> str:
    """Match ``tofu_eval.Evaluator.format_question_prompt`` exactly."""
    if tokenizer.chat_template is not None:
        messages = [
            {"role": "user", "content": f"Question: {question} Answer:"}
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {question} Answer:"


def encode_training_example(
    sample: Mapping[str, Any],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Encode one TOFU record with prompt-only masking."""
    if not tokenizer.eos_token:
        raise ValueError("Tokenizer must define eos_token")
    prompt = format_question_prompt(tokenizer, str(sample["question"]))
    # The leading answer space and textual tokenizer EOS are protocol-critical.
    full_text = prompt + f" {sample['answer']}{tokenizer.eos_token}"
    full = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    prompt_encoding = tokenizer(
        prompt,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = full["input_ids"][0].to(dtype=torch.long)
    attention_mask = full["attention_mask"][0].to(dtype=torch.long)
    prompt_length = int(prompt_encoding["input_ids"].shape[1])
    labels = input_ids.clone()
    labels[:prompt_length] = -100
    trainable = labels.ne(-100)
    if not bool(trainable.any()):
        raise ValueError(
            "TOFU answer and EOS were fully truncated; increase --max-length"
        )
    if tokenizer.eos_token_id is not None:
        trainable_ids = labels[trainable]
        if not bool(trainable_ids.eq(int(tokenizer.eos_token_id)).any()):
            raise ValueError(
                "Tokenizer EOS was truncated from the trainable label span; "
                "increase --max-length"
            )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


class TOFUAnswerOnlyDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.examples = [
            encode_training_example(sample, tokenizer, max_length)
            for sample in samples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.examples[index]


def collate_supervised_batch(
    batch: Sequence[Mapping[str, torch.Tensor]],
    pad_token_id: int,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    width = max(int(row["input_ids"].shape[0]) for row in batch)
    input_ids = torch.full(
        (len(batch), width),
        int(pad_token_id),
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    for row_index, row in enumerate(batch):
        length = int(row["input_ids"].shape[0])
        input_ids[row_index, :length] = row["input_ids"]
        attention_mask[row_index, :length] = row["attention_mask"]
        labels[row_index, :length] = row["labels"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def epoch_shuffle_seed(seed: int, epoch: int) -> int:
    if epoch <= 0:
        raise ValueError("epoch must be positive")
    return int(seed) + int(epoch) - 1


def deterministic_shuffle_indices(size: int, seed: int, epoch: int) -> List[int]:
    generator = torch.Generator()
    generator.manual_seed(epoch_shuffle_seed(seed, epoch))
    return torch.randperm(size, generator=generator).tolist()


def make_epoch_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    seed: int,
    epoch: int,
    pad_token_id: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(epoch_shuffle_seed(seed, epoch))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda rows: collate_supervised_batch(
            rows,
            pad_token_id,
        ),
    )


def directory_is_nonempty(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def prepare_output_directory(
    output_dir: Path,
    resume_from_checkpoint: Optional[Path],
) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    if directory_is_nonempty(output_dir) and resume_from_checkpoint is None:
        raise FileExistsError(
            f"Refusing to overwrite nonempty output directory: {output_dir}"
        )
    if resume_from_checkpoint is not None and not resume_from_checkpoint.is_dir():
        raise FileNotFoundError(
            f"Resume checkpoint directory does not exist: {resume_from_checkpoint}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def ensure_new_checkpoint_directory(path: Path) -> None:
    if path.exists():
        if not path.is_dir() or any(path.iterdir()):
            raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    path.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fixed_probe_samples(
    samples: Sequence[Mapping[str, Any]],
    count: int = PROBE_SIZE,
) -> List[Dict[str, Any]]:
    """Choose a revision-stable probe set without depending on row order."""
    indexed: List[Dict[str, Any]] = []
    for source_index, sample in enumerate(samples):
        record = {
            "question": str(sample["question"]),
            "answer": str(sample["answer"]),
        }
        indexed.append(
            {
                **record,
                "source_index": source_index,
                "record_sha256": sha256_json(record),
            }
        )
    indexed.sort(key=lambda row: (row["record_sha256"], row["source_index"]))
    if len(indexed) < count:
        raise ValueError(f"Probe source has {len(indexed)} rows; expected {count}")
    return indexed[:count]


def exact_match(generated: str, reference: str) -> float:
    return float(generated.strip() == reference.strip())


@torch.no_grad()
def generate_like_tofu_eval(
    model: Any,
    tokenizer: Any,
    question: str,
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int = PROBE_MAX_NEW_TOKENS,
) -> str:
    prompt = format_question_prompt(tokenizer, question)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) for key, value in encoded.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


@torch.no_grad()
def evaluate_probe_split(
    model: Any,
    tokenizer: Any,
    samples: Sequence[Mapping[str, Any]],
    *,
    split_name: str,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    generations: List[Dict[str, Any]] = []
    exact_values: List[float] = []
    rouge_values: List[float] = []
    for sample in samples:
        generated = generate_like_tofu_eval(
            model,
            tokenizer,
            str(sample["question"]),
            device=device,
            max_length=max_length,
        )
        gold = str(sample["answer"])
        exact_value = exact_match(generated, gold)
        rouge_value = float(scorer.score(gold, generated)["rougeL"].recall)
        exact_values.append(exact_value)
        rouge_values.append(rouge_value)
        generations.append(
            {
                "source_index": sample.get("source_index"),
                "record_sha256": sample.get("record_sha256"),
                "question": str(sample["question"]),
                "gold": gold,
                "generated": generated,
                "exact_match": exact_value,
                "rouge_l_recall": rouge_value,
            }
        )
    return {
        "split": split_name,
        "count": len(samples),
        "exact_match": sum(exact_values) / len(exact_values),
        "rouge_l": sum(rouge_values) / len(rouge_values),
        "rouge_l_definition": "ROUGE-L recall, matching tofu_eval.py",
        "generations": generations,
    }


def run_epoch_probes(
    model: Any,
    tokenizer: Any,
    probe_sets: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    epoch: int,
    device: torch.device,
    max_length: int,
) -> Dict[str, Any]:
    was_training = bool(model.training)
    old_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    if old_use_cache is not None:
        model.config.use_cache = True
    try:
        splits = {
            name: evaluate_probe_split(
                model,
                tokenizer,
                rows,
                split_name=name,
                device=device,
                max_length=max_length,
            )
            for name, rows in probe_sets.items()
        }
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        if was_training:
            model.train()
    return {
        "schema_version": 1,
        "epoch": epoch,
        "created_at_utc": utc_now(),
        "selection_use": (
            "May select the full-model fine-tuning checkpoint only; not "
            "permitted for unlearning or repair."
        ),
        "prompt_protocol": 'user content = "Question: {question} Answer:"',
        "generation": {
            "do_sample": False,
            "max_new_tokens": PROBE_MAX_NEW_TOKENS,
            "pad_token_id": "tokenizer.eos_token_id",
        },
        "splits": splits,
    }


def resolve_torch_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def infer_model_revision(model_path: str, model: Any) -> Optional[str]:
    path = Path(model_path)
    if path.parent.name == "snapshots" and len(path.name) >= 7:
        return path.name
    revision = getattr(model.config, "_commit_hash", None)
    return str(revision) if revision else None


def git_provenance() -> Dict[str, Any]:
    repository = Path(__file__).resolve().parents[2]

    def run_git(*arguments: str) -> Optional[str]:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run_git("status", "--short")
    return {
        "repository_root": str(repository),
        "commit": run_git("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines() if status else [],
    }


def dependency_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in (
        "torch",
        "transformers",
        "datasets",
        "rouge-score",
        "numpy",
        "tqdm",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def save_training_checkpoint(
    path: Path,
    *,
    model: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    completed_epoch: int,
    total_epochs: int,
    global_step: int,
    tokens_seen: int,
    config_sha256: str,
    scaler: Optional[Any] = None,
) -> None:
    ensure_new_checkpoint_directory(path)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    state: Dict[str, Any] = {
        "schema_version": 1,
        "completed_epoch": completed_epoch,
        "total_epochs": total_epochs,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "config_sha256": config_sha256,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "gradient_scaler": scaler.state_dict() if scaler is not None else None,
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_random_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(state, path / "trainer_state.pt")


def load_training_state(path: Path) -> Dict[str, Any]:
    state_path = path / "trainer_state.pt"
    if not state_path.is_file():
        raise FileNotFoundError(f"Resume state not found: {state_path}")
    try:
        return torch.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(state_path, map_location="cpu")


def restore_random_state(state: Mapping[str, Any]) -> None:
    if state.get("python_random_state") is not None:
        random.setstate(state["python_random_state"])
    if state.get("torch_random_state") is not None:
        torch.set_rng_state(state["torch_random_state"])
    if torch.cuda.is_available() and state.get("cuda_random_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_random_state_all"])


def sweep_row_is_eligible(row: Mapping[str, Any]) -> bool:
    required = tuple(ELIGIBILITY_THRESHOLDS)
    if any(row.get(name) is None for name in required):
        return False
    return all(
        float(row[name]) >= threshold
        for name, threshold in ELIGIBILITY_THRESHOLDS.items()
    )


def select_sweep_winner(rows: Sequence[Mapping[str, Any]]) -> Optional[int]:
    eligible = [index for index, row in enumerate(rows) if sweep_row_is_eligible(row)]
    if not eligible:
        return None

    def priority(index: int) -> tuple[float, float, float, int]:
        row = rows[index]
        external = (
            float(row["real_author_probe_rouge_l"])
            + float(row["world_fact_probe_rouge_l"])
        ) / 2.0
        return (
            -float(row["tofu_probe_rouge_l"]),
            -external,
            float(row["learning_rate"]),
            int(row["epochs"]),
        )

    return min(eligible, key=priority)


SWEEP_COLUMNS = [
    "run_dir",
    "learning_rate",
    "epochs",
    "final_training_loss",
    "tofu_probe_exact_match",
    "tofu_probe_rouge_l",
    "real_author_probe_exact_match",
    "real_author_probe_rouge_l",
    "world_fact_probe_exact_match",
    "world_fact_probe_rouge_l",
    "eligible",
    "selected",
]


def load_sweep_rows(output_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in sorted(path for path in output_root.iterdir() if path.is_dir()):
        config_path = run_dir / "config_used.json"
        metadata_path = run_dir / "finetune_metadata.json"
        if not config_path.is_file() or not metadata_path.is_file():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if metadata.get("status") != "complete":
            continue
        epoch = int(config["epochs"])
        probe_path = run_dir / f"epoch_{epoch}_probe.json"
        if not probe_path.is_file():
            continue
        with probe_path.open("r", encoding="utf-8") as handle:
            probe = json.load(handle)
        splits = probe["splits"]
        rows.append(
            {
                "run_dir": run_dir.name,
                "learning_rate": float(config["learning_rate"]),
                "epochs": epoch,
                "final_training_loss": float(metadata["final_training_loss"]),
                "tofu_probe_exact_match": float(
                    splits["full_tofu"]["exact_match"]
                ),
                "tofu_probe_rouge_l": float(splits["full_tofu"]["rouge_l"]),
                "real_author_probe_exact_match": float(
                    splits["real_authors"]["exact_match"]
                ),
                "real_author_probe_rouge_l": float(
                    splits["real_authors"]["rouge_l"]
                ),
                "world_fact_probe_exact_match": float(
                    splits["world_facts"]["exact_match"]
                ),
                "world_fact_probe_rouge_l": float(
                    splits["world_facts"]["rouge_l"]
                ),
            }
        )
    return rows


def write_sweep_summary(output_root: Path) -> List[Dict[str, Any]]:
    rows = load_sweep_rows(output_root)
    if not rows:
        raise RuntimeError(f"No complete sweep runs found under {output_root}")
    winner = select_sweep_winner(rows)
    for index, row in enumerate(rows):
        row["eligible"] = sweep_row_is_eligible(row)
        row["selected"] = index == winner
    csv_path = output_root / "sweep_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path = output_root / "sweep_summary.md"
    with markdown_path.open("w", encoding="utf-8") as handle:
        handle.write("# TOFU full-model utility sweep\n\n")
        handle.write(
            "Eligibility: TOFU ROUGE-L >= 0.95, real-authors ROUGE-L >= "
            "0.75, and world-facts ROUGE-L >= 0.75.\n\n"
        )
        handle.write("| " + " | ".join(SWEEP_COLUMNS) + " |\n")
        handle.write("| " + " | ".join("---" for _ in SWEEP_COLUMNS) + " |\n")
        for row in rows:
            values = []
            for column in SWEEP_COLUMNS:
                value = row[column]
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value))
            handle.write("| " + " | ".join(values) + " |\n")
        if winner is None:
            handle.write("\nNo configuration met every eligibility threshold.\n")
        else:
            handle.write(f"\nSelected configuration: `{rows[winner]['run_dir']}`.\n")
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-every-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--summarize-sweep-root",
        help="Aggregate completed sweep directories without training.",
    )
    return parser


def validate_training_args(args: argparse.Namespace) -> None:
    if not args.output_dir:
        raise ValueError("--output-dir is required for training")
    for name in ("epochs", "batch_size", "gradient_accumulation_steps", "max_length"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("--learning-rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("--weight-decay must be finite and nonnegative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1)")


def run_training(args: argparse.Namespace) -> None:
    validate_training_args(args)
    output_dir = Path(args.output_dir).expanduser().resolve()
    resume_path = (
        Path(args.resume_from_checkpoint).expanduser().resolve()
        if args.resume_from_checkpoint
        else None
    )
    prepare_output_directory(output_dir, resume_path)
    seed_everything(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Full-model TOFU training requires CUDA")
    device = torch.device("cuda:0")
    dtype = resolve_torch_dtype(args.dtype)

    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    load_path = str(resume_path) if resume_path is not None else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer must define eos_token_id")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=dtype,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    else:
        model.gradient_checkpointing_disable()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if trainable_parameters != total_parameters:
        raise RuntimeError("The utility-preserving experiment requires full-model training")
    print(
        f"Full-model parameters: total={total_parameters:,} "
        f"trainable={trainable_parameters:,}",
        flush=True,
    )

    full_dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=DATASET_SPLIT)
    full_rows = list(full_dataset)
    if len(full_rows) != EXPECTED_FULL_EXAMPLES:
        raise RuntimeError(
            f"Expected exactly {EXPECTED_FULL_EXAMPLES} full-TOFU rows, "
            f"found {len(full_rows)}"
        )
    real_authors_dataset = load_dataset(
        DATASET_ID,
        "real_authors",
        split=DATASET_SPLIT,
    )
    world_facts_dataset = load_dataset(
        DATASET_ID,
        "world_facts",
        split=DATASET_SPLIT,
    )
    probe_sets = {
        "full_tofu": fixed_probe_samples(full_rows),
        "real_authors": fixed_probe_samples(list(real_authors_dataset)),
        "world_facts": fixed_probe_samples(list(world_facts_dataset)),
    }
    dataset = TOFUAnswerOnlyDataset(full_rows, tokenizer, args.max_length)

    batches_per_epoch = math.ceil(len(dataset) / args.batch_size)
    updates_per_epoch = math.ceil(
        batches_per_epoch / args.gradient_accumulation_steps
    )
    total_update_steps = updates_per_epoch * args.epochs
    warmup_steps = int(total_update_steps * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    config_used = {
        "schema_version": 1,
        "experiment": "tofu_full_utility_preserving",
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size
        * args.gradient_accumulation_steps,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "warmup_steps": warmup_steps,
        "total_update_steps": total_update_steps,
        "max_length": args.max_length,
        "seed": args.seed,
        "save_every_epoch": args.save_every_epoch,
        "resume_from_checkpoint": str(resume_path) if resume_path else None,
        "gradient_checkpointing": args.gradient_checkpointing,
        "dtype": args.dtype,
        "gradient_clip_norm": 1.0,
        "optimizer": "AdamW",
        "scheduler": "linear_with_warmup",
        "dataset_id": DATASET_ID,
        "dataset_config": DATASET_CONFIG,
        "dataset_split": DATASET_SPLIT,
        "dataset_example_count": len(full_rows),
        "dataset_fingerprint": getattr(full_dataset, "_fingerprint", None),
        "real_authors_probe_fingerprint": getattr(
            real_authors_dataset,
            "_fingerprint",
            None,
        ),
        "world_facts_probe_fingerprint": getattr(
            world_facts_dataset,
            "_fingerprint",
            None,
        ),
        "answer_prefix": " ",
        "append_eos": True,
        "prompt_tokens_masked": True,
        "probe_size_per_split": PROBE_SIZE,
    }
    immutable_config = {
        key: value
        for key, value in config_used.items()
        if key not in {"output_dir", "resume_from_checkpoint"}
    }
    config_sha256 = sha256_json(immutable_config)
    config_used["config_sha256"] = config_sha256
    config_path = output_dir / "config_used.json"
    if resume_path is None:
        write_json(config_path, config_used)
    else:
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Resume requires the original configuration: {config_path}"
            )
        with config_path.open("r", encoding="utf-8") as handle:
            original_config = json.load(handle)
        if original_config.get("config_sha256") != config_sha256:
            raise ValueError("Resume configuration differs from the original run")
        # Preserve the original immutable run receipt rather than replacing it
        # with a resume invocation path.
        config_used = original_config

    start_epoch = 1
    global_step = 0
    tokens_seen = 0
    loss_curve: List[Dict[str, Any]] = []
    probe_history: List[Dict[str, Any]] = []
    prior_metadata: Dict[str, Any] = {}
    if resume_path is not None:
        state = load_training_state(resume_path)
        if int(state["total_epochs"]) != args.epochs:
            raise ValueError(
                "Resume checkpoint total_epochs differs from --epochs; "
                "resume the original schedule instead of extending it"
            )
        if state.get("config_sha256") != config_sha256:
            raise ValueError("Resume checkpoint training configuration does not match")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        if state.get("gradient_scaler") is not None:
            # The scaler is created just below and restored before training.
            scaler_state = state["gradient_scaler"]
        else:
            scaler_state = None
        restore_random_state(state)
        start_epoch = int(state["completed_epoch"]) + 1
        global_step = int(state["global_step"])
        tokens_seen = int(state.get("tokens_seen", 0))
        metadata_path = output_dir / "finetune_metadata.json"
        if metadata_path.is_file():
            with metadata_path.open("r", encoding="utf-8") as handle:
                prior_metadata = json.load(handle)
            loss_curve = list(prior_metadata.get("loss_curve", []))
            probe_history = list(prior_metadata.get("probe_history", []))
    else:
        scaler_state = None
    if start_epoch > args.epochs:
        raise ValueError("Resume checkpoint already completed the requested epochs")

    # Loss scaling is required for the supported FP16 mode. BF16 and FP32 use
    # ordinary backward passes.
    scaler = torch.cuda.amp.GradScaler(enabled=args.dtype == "fp16")
    if scaler_state is not None:
        scaler.load_state_dict(scaler_state)

    base_revision = infer_model_revision(args.model_path, model)
    dataset_provenance = {
        "full": {
            "id": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "fingerprint": getattr(full_dataset, "_fingerprint", None),
            "count": len(full_rows),
        },
        "real_authors_probe": {
            "fingerprint": getattr(real_authors_dataset, "_fingerprint", None),
            "count": PROBE_SIZE,
        },
        "world_facts_probe": {
            "fingerprint": getattr(world_facts_dataset, "_fingerprint", None),
            "count": PROBE_SIZE,
        },
    }
    metadata: Dict[str, Any] = {
        "schema_version": 1,
        "status": "training",
        "started_at_utc": prior_metadata.get("started_at_utc", utc_now()),
        "resume_history": list(prior_metadata.get("resume_history", [])),
        "base_model": args.model_path,
        "base_model_revision": base_revision,
        "tokenizer": args.model_path,
        "tokenizer_loaded_from": load_path,
        "dataset": dataset_provenance,
        "dependencies": dependency_versions(),
        "git": git_provenance(),
        "parameter_count": {
            "total": total_parameters,
            "trainable": trainable_parameters,
        },
        "config_sha256": config_sha256,
        "loss_curve": loss_curve,
        "probe_history": probe_history,
        "final_training_loss": None,
        "final_checkpoint": None,
    }
    if resume_path is not None:
        metadata["resume_history"].append(
            {"resumed_at_utc": utc_now(), "checkpoint": str(resume_path)}
        )
    write_json(output_dir / "finetune_metadata.json", metadata)

    progress_mode = "a" if resume_path is not None else "x"
    progress_path = output_dir / "train_progress.jsonl"
    with progress_path.open(progress_mode, encoding="utf-8", buffering=1) as progress_file:
        for epoch in range(start_epoch, args.epochs + 1):
            dataloader = make_epoch_dataloader(
                dataset,
                batch_size=args.batch_size,
                seed=args.seed,
                epoch=epoch,
                pad_token_id=int(tokenizer.pad_token_id),
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss_sum = 0.0
            epoch_micro_batches = 0
            group_loss_sum = 0.0
            group_micro_batches = 0
            group_tokens = 0
            progress = tqdm(dataloader, desc=f"TOFU full epoch {epoch}")
            for batch_index, batch in enumerate(progress):
                group_start = (
                    batch_index // args.gradient_accumulation_steps
                ) * args.gradient_accumulation_steps
                group_size = min(
                    args.gradient_accumulation_steps,
                    len(dataloader) - group_start,
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                loss = output.loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(
                        f"Non-finite loss at epoch {epoch}, batch {batch_index + 1}"
                    )
                scaled_loss = loss / group_size
                if scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                raw_loss = float(loss.detach().cpu())
                answer_tokens = int(batch["labels"].ne(-100).sum().item())
                epoch_loss_sum += raw_loss
                epoch_micro_batches += 1
                group_loss_sum += raw_loss
                group_micro_batches += 1
                group_tokens += answer_tokens
                is_update = (
                    group_micro_batches == group_size
                    or batch_index + 1 == len(dataloader)
                )
                if not is_update:
                    continue
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                tokens_seen += group_tokens
                progress_row = {
                    "timestamp_utc": utc_now(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "loss": group_loss_sum / group_micro_batches,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                    "gradient_norm": float(gradient_norm.detach().cpu()),
                    "tokens_per_step": group_tokens,
                    "tokens_seen": tokens_seen,
                    "micro_batches": group_micro_batches,
                }
                progress_file.write(json.dumps(progress_row, sort_keys=True) + "\n")
                progress_file.flush()
                progress.set_postfix(
                    loss=f"{progress_row['loss']:.4f}",
                    lr=f"{progress_row['learning_rate']:.3g}",
                )
                group_loss_sum = 0.0
                group_micro_batches = 0
                group_tokens = 0

            mean_epoch_loss = epoch_loss_sum / epoch_micro_batches
            loss_curve.append(
                {
                    "epoch": epoch,
                    "mean_loss": mean_epoch_loss,
                    "global_step": global_step,
                }
            )
            probe = run_epoch_probes(
                model,
                tokenizer,
                probe_sets,
                epoch=epoch,
                device=device,
                max_length=args.max_length,
            )
            probe_path = output_dir / f"epoch_{epoch}_probe.json"
            write_json(probe_path, probe)
            if args.save_every_epoch:
                save_training_checkpoint(
                    output_dir / f"checkpoint_epoch_{epoch}",
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    completed_epoch=epoch,
                    total_epochs=args.epochs,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    config_sha256=config_sha256,
                    scaler=scaler,
                )
            probe_history.append(
                {
                    "epoch": epoch,
                    "path": str(probe_path),
                    "metrics": {
                        name: {
                            "exact_match": split["exact_match"],
                            "rouge_l": split["rouge_l"],
                        }
                        for name, split in probe["splits"].items()
                    },
                }
            )
            metadata.update(
                {
                    "last_completed_epoch": epoch,
                    "global_step": global_step,
                    "tokens_seen": tokens_seen,
                    "loss_curve": loss_curve,
                    "probe_history": probe_history,
                    "final_training_loss": mean_epoch_loss,
                }
            )
            write_json(output_dir / "finetune_metadata.json", metadata)

    final_dir = output_dir / "final"
    ensure_new_checkpoint_directory(final_dir)
    model.config.use_cache = True
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    metadata.update(
        {
            "status": "complete",
            "completed_at_utc": utc_now(),
            "final_checkpoint": str(final_dir),
            "final_training_loss": loss_curve[-1]["mean_loss"],
        }
    )
    write_json(output_dir / "finetune_metadata.json", metadata)
    write_json(final_dir / "config_used.json", config_used)
    write_json(final_dir / "finetune_metadata.json", metadata)
    print(f"Final checkpoint: {final_dir}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.summarize_sweep_root:
        rows = write_sweep_summary(
            Path(args.summarize_sweep_root).expanduser().resolve()
        )
        selected = next((row for row in rows if row["selected"]), None)
        print(
            "Sweep summary written; selected="
            + (selected["run_dir"] if selected else "none"),
            flush=True,
        )
        return
    run_training(args)


if __name__ == "__main__":
    main()
