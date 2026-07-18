#!/usr/bin/env python3
"""Sparse neighborhood-confidence repair for a GA/GD TOFU checkpoint.

This is the TOFU counterpart of ``gagd_neighborhood_confidence_repair.py``.
TOFU does not provide CounterFact target-new/target-true neighborhoods, so the
repair uses the benchmark's non-forget knowledge splits as its utility
neighborhood:

* a deterministic sample from the paired retain split;
* ``real_authors``;
* ``world_facts``.

The supplied GA/GD checkpoint is immutable until a candidate passes the local
guards.  The transformer and input embeddings remain frozen.  Only selected
LM-head vocabulary rows from initially under-confident neighborhood answers
can change, and rows occurring in forget answers are excluded.  Every sampled
forget answer remains in the optimization objective so a utility repair cannot
silently undo forgetting.

Use ``run_tofu_gagd_neighborhood_confidence.sh`` to train all four GA/GD
settings, run this fifth setting, evaluate all five with ``tofu_eval.py``, and
write a fixed-order comparison table.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd


METHOD = "gagd_neighborhood_confidence_tofu"
UTILITY_SPLITS = ("retain", "real_authors", "world_facts")
PAIRED_RETAIN_SPLITS = {
    "forget01": "retain99",
    "forget05": "retain95",
    "forget10": "retain90",
}


@dataclass(frozen=True)
class TOFUAnswerInstance:
    split: str
    source_index: int
    sampled_position: int
    question: str
    answer: str
    prompt: str


@dataclass
class TOFUAnswerDeltaCache:
    base_token_nll: torch.Tensor
    hidden: torch.Tensor
    selected_probs: torch.Tensor
    target_selected_columns: torch.Tensor


@dataclass(frozen=True)
class CandidateSnapshot:
    step: int
    delta_rows: torch.Tensor
    metrics: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        required=True,
        help="One of the four saved TOFU GA/GD checkpoints.",
    )
    parser.add_argument(
        "--reference-model-path",
        required=True,
        help=(
            "The original TOFU-finetuned model. Its answer NLL defines the "
            "neighborhood-confidence targets."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forget-split", choices=sorted(PAIRED_RETAIN_SPLITS), default="forget05")
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument("--retain-calibration-num", type=int, default=64)
    parser.add_argument("--real-authors-calibration-num", type=int, default=32)
    parser.add_argument("--world-facts-calibration-num", type=int, default=32)
    parser.add_argument("--calibration-seed", type=int, default=1729)

    parser.add_argument(
        "--reference-nll-slack",
        type=float,
        default=0.05,
        help=(
            "A neighborhood answer is repaired until its NLL is no more than "
            "reference NLL plus this slack."
        ),
    )
    parser.add_argument(
        "--forget-nll-tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed drop in each forget-answer NLL relative to the "
            "input checkpoint. Zero is the strictest protection."
        ),
    )
    parser.add_argument("--confidence-weight", type=float, default=10.0)
    parser.add_argument("--forget-protection-weight", type=float, default=100.0)
    parser.add_argument("--delta-l2-lambda", type=float, default=1e-4)
    parser.add_argument("--repair-steps", type=int, default=200)
    parser.add_argument("--repair-lr", type=float, default=5e-3)
    parser.add_argument(
        "--repair-optimizer",
        choices=["sgd", "adam", "adamw"],
        default="adamw",
    )
    parser.add_argument("--repair-rank", type=int, default=32)
    parser.add_argument("--row-selection-top-k", type=int, default=512)
    parser.add_argument("--minimum-row-document-count", type=int, default=1)
    parser.add_argument("--max-delta-norm", type=float, default=None)
    parser.add_argument("--forget-projection-rank", type=int, default=64)
    parser.add_argument(
        "--project-away-forget-hidden",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stop-when-all-satisfied",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-best-effort",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save the best forget-safe candidate even if some neighborhood "
            "confidence targets remain unmet. Use --no-save-best-effort to "
            "make complete target satisfaction a hard requirement."
        ),
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--comparison-tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--materialization-tolerance",
        type=float,
        default=1e-3,
        help="Numerical tolerance for BF16 checkpoint materialization guards.",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single", "auto"], default="single")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected_retain = PAIRED_RETAIN_SPLITS[args.forget_split]
    if args.retain_split != expected_retain:
        raise ValueError(
            f"{args.forget_split} must be paired with {expected_retain}, "
            f"not {args.retain_split}"
        )
    positive_names = (
        "forget_num",
        "retain_num",
        "retain_calibration_num",
        "real_authors_calibration_num",
        "world_facts_calibration_num",
        "repair_steps",
        "row_selection_top_k",
        "minimum_row_document_count",
        "batch_size",
        "max_length",
        "log_every",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative_names = (
        "reference_nll_slack",
        "forget_nll_tolerance",
        "confidence_weight",
        "forget_protection_weight",
        "delta_l2_lambda",
        "repair_rank",
        "forget_projection_rank",
        "comparison_tolerance",
        "materialization_tolerance",
    )
    for name in nonnegative_names:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if not math.isfinite(args.repair_lr) or args.repair_lr <= 0:
        raise ValueError("--repair-lr must be finite and positive")
    if args.max_delta_norm is not None and (
        not math.isfinite(args.max_delta_norm) or args.max_delta_norm < 0
    ):
        raise ValueError("--max-delta-norm must be finite and non-negative")


def deterministic_sample_indices(
    population_size: int,
    sample_size: int,
    seed: int,
) -> List[int]:
    if population_size <= 0:
        raise ValueError("Cannot sample from an empty population")
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    indices = list(range(population_size))
    random.Random(seed).shuffle(indices)
    return indices[: min(sample_size, population_size)]


def format_question_prompt(tok: Any, question: str) -> str:
    """Match ``tofu_eval.Evaluator.format_question_prompt`` exactly."""
    if getattr(tok, "chat_template", None) is not None:
        messages = [{"role": "user", "content": f"Question: {question} Answer:"}]
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return f"Question: {question} Answer:"


def _rows_to_instances(
    rows: Sequence[Dict[str, Any]],
    split: str,
    tok: Any,
    sample_size: int,
    seed: int,
) -> List[TOFUAnswerInstance]:
    indices = deterministic_sample_indices(len(rows), sample_size, seed)
    instances: List[TOFUAnswerInstance] = []
    for sampled_position, source_index in enumerate(indices):
        row = rows[source_index]
        if "question" not in row or "answer" not in row:
            raise ValueError(f"TOFU {split} row lacks question or answer")
        question = str(row["question"])
        answer = str(row["answer"])
        instances.append(
            TOFUAnswerInstance(
                split=split,
                source_index=source_index,
                sampled_position=sampled_position,
                question=question,
                answer=answer,
                prompt=format_question_prompt(tok, question),
            )
        )
    return instances


def load_tofu_calibration_instances(
    args: argparse.Namespace,
    tok: Any,
) -> Tuple[List[TOFUAnswerInstance], List[TOFUAnswerInstance]]:
    forget_rows = list(
        load_dataset("locuslab/TOFU", name=args.forget_split, split="train")
    )
    retain_rows = list(
        load_dataset("locuslab/TOFU", name=args.retain_split, split="train")
    )
    real_rows = list(
        load_dataset("locuslab/TOFU", name="real_authors", split="train")
    )
    world_rows = list(
        load_dataset("locuslab/TOFU", name="world_facts", split="train")
    )

    # Match the four-setting runner for forget sampling: a fresh RNG with the
    # experiment seed is applied independently to each primary split.
    forget = _rows_to_instances(
        forget_rows,
        args.forget_split,
        tok,
        args.forget_num,
        args.seed,
    )
    retain_full = _rows_to_instances(
        retain_rows,
        "retain",
        tok,
        args.retain_num,
        args.seed,
    )
    retain_positions = deterministic_sample_indices(
        len(retain_full),
        args.retain_calibration_num,
        args.calibration_seed,
    )
    retain = [
        TOFUAnswerInstance(
            **{
                **asdict(retain_full[position]),
                "sampled_position": local_position,
            }
        )
        for local_position, position in enumerate(retain_positions)
    ]
    real = _rows_to_instances(
        real_rows,
        "real_authors",
        tok,
        args.real_authors_calibration_num,
        args.calibration_seed + 1,
    )
    world = _rows_to_instances(
        world_rows,
        "world_facts",
        tok,
        args.world_facts_calibration_num,
        args.calibration_seed + 2,
    )
    return forget, [*retain, *real, *world]


def _token_ids(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token_id) for token_id in value]


def answer_sequence_components(
    tok: Any,
    instance: TOFUAnswerInstance,
    max_length: int,
) -> Tuple[List[int], int]:
    """Use the exact clean-answer sequence split used by ``tofu_eval.py``."""
    full_text = instance.prompt + f" {instance.answer}"
    full_encoded = tok(
        full_text,
        truncation=True,
        max_length=max_length,
    )
    prompt_encoded = tok(
        instance.prompt,
        truncation=True,
        max_length=max_length,
    )
    full_ids = _token_ids(full_encoded["input_ids"])
    prompt_ids = _token_ids(prompt_encoded["input_ids"])
    prompt_length = len(prompt_ids)
    if prompt_length <= 0:
        raise ValueError("TOFU prompt tokenized to an empty sequence")
    if len(full_ids) <= prompt_length:
        raise ValueError(
            "TOFU answer has no scored tokens after truncation; increase --max-length"
        )
    return full_ids, prompt_length


def _right_padded_batch(
    rows: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    width = max(len(row) for row in rows)
    input_ids = torch.full(
        (len(rows), width),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row_index, row in enumerate(rows):
        length = len(row)
        input_ids[row_index, :length] = torch.tensor(
            row,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row_index, :length] = 1
    return {"input_ids": input_ids, "attention_mask": attention_mask}


@torch.no_grad()
def score_answer_instances(
    model: nn.Module,
    tok: Any,
    instances: Sequence[TOFUAnswerInstance],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> torch.Tensor:
    values: List[torch.Tensor] = []
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    for start in range(0, len(instances), batch_size):
        chunk = instances[start : start + batch_size]
        components = [
            answer_sequence_components(tok, instance, max_length)
            for instance in chunk
        ]
        encoded = _right_padded_batch(
            [component[0] for component in components],
            int(pad_id),
            device,
        )
        output = model(**encoded, use_cache=False)
        for row_index, (full_ids, prompt_length) in enumerate(components):
            answer_ids = torch.tensor(
                full_ids[prompt_length:],
                dtype=torch.long,
                device=device,
            )
            positions = torch.arange(
                prompt_length - 1,
                len(full_ids) - 1,
                dtype=torch.long,
                device=device,
            )
            log_probs = F.log_softmax(
                output.logits[row_index].index_select(0, positions).float(),
                dim=-1,
            )
            values.append(
                -log_probs.gather(-1, answer_ids.unsqueeze(-1))
                .squeeze(-1)
                .mean()
            )
        del output
    if not values:
        return torch.empty((0,), dtype=torch.float32, device=device)
    return torch.stack(values).float()


@torch.no_grad()
def build_answer_delta_caches(
    model: nn.Module,
    tok: Any,
    instances: Sequence[TOFUAnswerInstance],
    selected_ids: Sequence[int],
    device: torch.device,
    *,
    batch_size: int,
    max_length: int,
) -> List[TOFUAnswerDeltaCache]:
    selected = torch.tensor(selected_ids, dtype=torch.long, device=device)
    selected_lookup = {
        int(token_id): column for column, token_id in enumerate(selected_ids)
    }
    caches: List[TOFUAnswerDeltaCache] = []
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    for start in range(0, len(instances), batch_size):
        chunk = instances[start : start + batch_size]
        components = [
            answer_sequence_components(tok, instance, max_length)
            for instance in chunk
        ]
        encoded = _right_padded_batch(
            [component[0] for component in components],
            int(pad_id),
            device,
        )
        output = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )
        for row_index, (full_ids, prompt_length) in enumerate(components):
            answer_ids = torch.tensor(
                full_ids[prompt_length:],
                dtype=torch.long,
                device=device,
            )
            positions = torch.arange(
                prompt_length - 1,
                len(full_ids) - 1,
                dtype=torch.long,
                device=device,
            )
            log_probs = F.log_softmax(
                output.logits[row_index].index_select(0, positions).float(),
                dim=-1,
            )
            target_columns = torch.tensor(
                [
                    selected_lookup.get(int(token_id), -1)
                    for token_id in answer_ids.detach().cpu().tolist()
                ],
                dtype=torch.long,
                device=device,
            )
            caches.append(
                TOFUAnswerDeltaCache(
                    base_token_nll=(
                        -log_probs.gather(-1, answer_ids.unsqueeze(-1)).squeeze(-1)
                    ).detach(),
                    hidden=output.hidden_states[-1][row_index]
                    .index_select(0, positions)
                    .float()
                    .detach(),
                    selected_probs=log_probs.index_select(-1, selected)
                    .exp()
                    .detach(),
                    target_selected_columns=target_columns,
                )
            )
        del output
    return caches


def answer_nll_from_delta_cache(
    cache: TOFUAnswerDeltaCache,
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    corrections = cache.hidden @ delta_rows.transpose(0, 1)
    log_shift = active._log_partition_shift(cache.selected_probs, corrections)
    target_correction = corrections.new_zeros(corrections.shape[0])
    selected_mask = cache.target_selected_columns.ge(0)
    if selected_mask.any():
        target_correction[selected_mask] = corrections[
            selected_mask,
            cache.target_selected_columns[selected_mask],
        ]
    return (
        cache.base_token_nll + log_shift - target_correction
    ).mean()


def answer_nlls_from_delta_caches(
    caches: Sequence[TOFUAnswerDeltaCache],
    delta_rows: torch.Tensor,
) -> torch.Tensor:
    if not caches:
        return delta_rows.new_empty((0,))
    return torch.stack(
        [answer_nll_from_delta_cache(cache, delta_rows) for cache in caches]
    )


def build_required_nll_tensors(
    baseline_forget_nll: torch.Tensor,
    baseline_neighborhood_nll: torch.Tensor,
    reference_neighborhood_nll: torch.Tensor,
    *,
    forget_nll_tolerance: float,
    reference_nll_slack: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if baseline_neighborhood_nll.shape != reference_neighborhood_nll.shape:
        raise ValueError("Candidate and reference neighborhood NLL shapes differ")
    required_forget = baseline_forget_nll - forget_nll_tolerance
    required_neighborhood = torch.minimum(
        baseline_neighborhood_nll,
        reference_neighborhood_nll.to(
            device=baseline_neighborhood_nll.device,
            dtype=baseline_neighborhood_nll.dtype,
        )
        + reference_nll_slack,
    )
    return required_forget, required_neighborhood


def confidence_objective_terms(
    current_forget_nll: torch.Tensor,
    required_forget_nll: torch.Tensor,
    current_neighborhood_nll: torch.Tensor,
    required_neighborhood_nll: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if current_forget_nll.shape != required_forget_nll.shape:
        raise ValueError("Every forget instance must have a required NLL")
    if current_neighborhood_nll.shape != required_neighborhood_nll.shape:
        raise ValueError("Every neighborhood instance must have a required NLL")
    forget_hinge = torch.relu(
        required_forget_nll.to(current_forget_nll) - current_forget_nll
    ).square()
    confidence_hinge = torch.relu(
        current_neighborhood_nll
        - required_neighborhood_nll.to(current_neighborhood_nll)
    ).square()
    return {
        "forget_hinge": forget_hinge.mean()
        if forget_hinge.numel()
        else current_forget_nll.new_zeros(()),
        "confidence_hinge": confidence_hinge.mean()
        if confidence_hinge.numel()
        else current_neighborhood_nll.new_zeros(()),
        "forget_slack": current_forget_nll
        - required_forget_nll.to(current_forget_nll),
        "neighborhood_slack": required_neighborhood_nll.to(
            current_neighborhood_nll
        )
        - current_neighborhood_nll,
    }


def select_neighborhood_lm_head_rows(
    tok: Any,
    neighborhood_instances: Sequence[TOFUAnswerInstance],
    active_positions: Sequence[int],
    forget_instances: Sequence[TOFUAnswerInstance],
    *,
    top_k: int,
    minimum_document_count: int,
) -> Tuple[List[int], Dict[str, Any]]:
    forget_ids: set[int] = set()
    for instance in forget_instances:
        forget_ids.update(
            gagd.token_ids_for_text(tok, gagd.normalize_answer(instance.answer))
        )
    special_ids = gagd.special_token_ids(tok)
    document_counts: Counter[int] = Counter()
    for position in active_positions:
        instance = neighborhood_instances[position]
        token_ids = set(
            gagd.token_ids_for_text(tok, gagd.normalize_answer(instance.answer))
        )
        document_counts.update(token_ids - forget_ids - special_ids)
    ranked = sorted(
        (
            (token_id, count)
            for token_id, count in document_counts.items()
            if count >= minimum_document_count
        ),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [token_id for token_id, _ in ranked[:top_k]]
    return selected, {
        "active_neighborhood_instance_count": len(active_positions),
        "forget_answer_token_count": len(forget_ids - special_ids),
        "eligible_neighborhood_row_count": len(ranked),
        "selected_lm_head_row_count": len(selected),
        "selected_lm_head_token_ids": selected,
        "selected_lm_head_tokens": {
            str(token_id): tok.decode([token_id]) for token_id in selected
        },
        "selected_row_document_counts": {
            str(token_id): count for token_id, count in ranked[:top_k]
        },
        "forget_answer_rows_excluded": True,
        "special_rows_excluded": True,
    }


def group_answer_probabilities(
    nll: torch.Tensor,
    instances: Sequence[TOFUAnswerInstance],
) -> Dict[str, float]:
    grouped: Dict[str, List[float]] = {}
    for value, instance in zip(nll.detach().cpu().tolist(), instances):
        grouped.setdefault(instance.split, []).append(math.exp(-float(value)))
    return {
        split: sum(values) / len(values)
        for split, values in grouped.items()
        if values
    }


def local_metrics(
    forget_nll: torch.Tensor,
    required_forget_nll: torch.Tensor,
    neighborhood_nll: torch.Tensor,
    required_neighborhood_nll: torch.Tensor,
    neighborhood_instances: Sequence[TOFUAnswerInstance],
    delta_rows: torch.Tensor,
    *,
    tolerance: float,
) -> Dict[str, Any]:
    forget_violations = forget_nll < (
        required_forget_nll.to(forget_nll) - tolerance
    )
    neighborhood_unmet = neighborhood_nll > (
        required_neighborhood_nll.to(neighborhood_nll) + tolerance
    )
    split_probabilities = group_answer_probabilities(
        neighborhood_nll,
        neighborhood_instances,
    )
    present_utility = [
        split_probabilities[split]
        for split in UTILITY_SPLITS
        if split in split_probabilities
    ]
    return {
        "forget_answer_probability": float(
            torch.exp(-forget_nll).mean().detach().cpu()
        ),
        "forget_protection_violation_count": int(
            forget_violations.sum().detach().cpu()
        ),
        "minimum_forget_nll_slack": float(
            (forget_nll - required_forget_nll.to(forget_nll))
            .min()
            .detach()
            .cpu()
        ),
        "neighborhood_target_unmet_count": int(
            neighborhood_unmet.sum().detach().cpu()
        ),
        "minimum_neighborhood_nll_slack": float(
            (
                required_neighborhood_nll.to(neighborhood_nll)
                - neighborhood_nll
            )
            .min()
            .detach()
            .cpu()
        ),
        "utility_answer_probability_by_split": split_probabilities,
        "utility_macro_answer_probability": (
            sum(present_utility) / len(present_utility)
            if present_utility
            else float("nan")
        ),
        "selected_lm_head_delta_norm": float(delta_rows.norm().detach().cpu()),
    }


def candidate_priority(metrics: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """Prefer safety, then target coverage, confidence, and a smaller edit."""
    return (
        int(metrics["forget_protection_violation_count"]),
        int(metrics["neighborhood_target_unmet_count"]),
        -float(metrics["utility_macro_answer_probability"]),
        float(metrics["selected_lm_head_delta_norm"]),
    )


def _hidden_rows(caches: Sequence[TOFUAnswerDeltaCache]) -> torch.Tensor:
    rows = [cache.hidden for cache in caches if cache.hidden.numel()]
    if not rows:
        raise ValueError("No answer-token hidden states were cached")
    return torch.cat(rows, dim=0)


@torch.no_grad()
def set_selected_lm_head_rows(
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    baseline_rows: torch.Tensor,
    delta_rows: torch.Tensor,
) -> None:
    if baseline_rows.shape != delta_rows.shape:
        raise ValueError("Baseline rows and delta rows must have equal shape")
    if len(selected_ids) != baseline_rows.shape[0]:
        raise ValueError("Selected row IDs do not match row snapshots")
    if not selected_ids:
        return
    ids = torch.tensor(
        selected_ids,
        dtype=torch.long,
        device=output_weight.device,
    )
    output_weight.index_copy_(
        0,
        ids,
        baseline_rows.to(output_weight)
        + delta_rows.to(device=output_weight.device, dtype=output_weight.dtype),
    )


def instance_reports(
    instances: Sequence[TOFUAnswerInstance],
    nll: torch.Tensor,
    required_nll: Optional[torch.Tensor] = None,
) -> List[Dict[str, Any]]:
    nll_values = nll.detach().cpu().tolist()
    required_values = (
        required_nll.detach().cpu().tolist()
        if required_nll is not None
        else [None] * len(instances)
    )
    reports: List[Dict[str, Any]] = []
    for instance, value, required in zip(
        instances,
        nll_values,
        required_values,
    ):
        row = {
            **asdict(instance),
            "answer_nll": float(value),
            "answer_probability": math.exp(-float(value)),
        }
        if required is not None:
            row["required_answer_nll"] = float(required)
            row["constraint"] = (
                "nll_at_least"
                if instance.split.startswith("forget")
                else "nll_at_most"
            )
        reports.append(row)
    return reports


def discover_source_config(model_path: str) -> Tuple[Optional[Path], Optional[Any]]:
    path = Path(model_path).resolve()
    candidates: Iterable[Path] = [path, *path.parents[:5]]
    for directory in candidates:
        candidate = directory / "config_used.json"
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as handle:
                return candidate, json.load(handle)
    return None, None


def _model_args(
    model_path: str,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )


def _write_json(path: Path, value: Any) -> None:
    gagd.write_json(path, value)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    active.write_jsonl(path, rows)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_for_data = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer_for_data.pad_token is None:
        tokenizer_for_data.pad_token = (
            tokenizer_for_data.eos_token
            if tokenizer_for_data.eos_token is not None
            else tokenizer_for_data.unk_token
        )
    print("Loading deterministic TOFU forget and utility calibration sets")
    forget_instances, neighborhood_instances = load_tofu_calibration_instances(
        args,
        tokenizer_for_data,
    )

    source_config_path, source_config = discover_source_config(args.model_path)
    config_used = {
        **vars(args),
        "method": METHOD,
        "source_experiment_config_path": (
            str(source_config_path) if source_config_path is not None else None
        ),
        "source_experiment_config": source_config,
        "parameter_scope": "selected_lm_head_rows_only",
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "forget_answer_rows_excluded_from_selection": True,
        "forget_instances_protected_individually": True,
        "tofu_neighborhood_definition": list(UTILITY_SPLITS),
        "tofu_eval_prompt_construction": True,
    }
    _write_json(output_dir / "config_used.json", config_used)
    _write_json(
        output_dir / "calibration_manifest.json",
        {
            "forget": [asdict(instance) for instance in forget_instances],
            "neighborhood": [
                asdict(instance) for instance in neighborhood_instances
            ],
        },
    )

    print(f"Scoring reference confidence: {args.reference_model_path}")
    reference_model, reference_tok = gagd.load_model_and_tokenizer(
        _model_args(args.reference_model_path, args),
        for_training=False,
    )
    if len(reference_tok) != len(tokenizer_for_data):
        raise ValueError("Reference and candidate tokenizer vocabularies differ")
    reference_device = gagd.first_device(reference_model)
    reference_neighborhood_nll = score_answer_instances(
        reference_model,
        reference_tok,
        neighborhood_instances,
        reference_device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    _write_json(
        output_dir / "reference_neighborhood_instances.json",
        instance_reports(
            neighborhood_instances,
            reference_neighborhood_nll,
        ),
    )
    del reference_model, reference_tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Loading GA/GD checkpoint: {args.model_path}")
    model, tok = gagd.load_model_and_tokenizer(
        _model_args(args.model_path, args),
        for_training=False,
    )
    output_embeddings = active.freeze_model_for_output_repair(model)
    output_weight = output_embeddings.weight
    input_weight = model.get_input_embeddings().weight
    input_storage_pointer = input_weight.data_ptr()
    input_tensor_version = input_weight._version
    device = gagd.first_device(model)

    print("Scoring input checkpoint")
    baseline_forget_nll = score_answer_instances(
        model,
        tok,
        forget_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    baseline_neighborhood_nll = score_answer_instances(
        model,
        tok,
        neighborhood_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    reference_neighborhood_nll = reference_neighborhood_nll.to(device)
    required_forget_nll, required_neighborhood_nll = build_required_nll_tensors(
        baseline_forget_nll,
        baseline_neighborhood_nll,
        reference_neighborhood_nll,
        forget_nll_tolerance=args.forget_nll_tolerance,
        reference_nll_slack=args.reference_nll_slack,
    )
    active_positions = (
        baseline_neighborhood_nll
        > required_neighborhood_nll + args.comparison_tolerance
    ).nonzero(as_tuple=False).flatten().detach().cpu().tolist()

    selected_ids, selected_row_report = select_neighborhood_lm_head_rows(
        tok,
        neighborhood_instances,
        active_positions,
        forget_instances,
        top_k=args.row_selection_top_k,
        minimum_document_count=args.minimum_row_document_count,
    )
    _write_json(output_dir / "selected_lm_head_rows.json", selected_row_report)
    if active_positions and not selected_ids:
        raise RuntimeError(
            "Neighborhood targets remain active, but no eligible rows were "
            "selected after excluding forget-answer and special-token rows."
        )

    selected_tensor = torch.tensor(
        selected_ids,
        dtype=torch.long,
        device=output_weight.device,
    )
    baseline_rows = (
        output_weight.index_select(0, selected_tensor).detach().clone()
        if selected_ids
        else output_weight.new_empty((0, output_weight.shape[1]))
    )

    print(
        f"Caching {len(forget_instances)} protected forget answers and "
        f"{len(neighborhood_instances)} utility answers"
    )
    forget_caches = build_answer_delta_caches(
        model,
        tok,
        forget_instances,
        selected_ids,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    neighborhood_caches = build_answer_delta_caches(
        model,
        tok,
        neighborhood_instances,
        selected_ids,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    zero_delta = torch.zeros_like(baseline_rows, dtype=torch.float32)
    baseline_forget_cached = answer_nlls_from_delta_caches(
        forget_caches,
        zero_delta,
    )
    baseline_neighborhood_cached = answer_nlls_from_delta_caches(
        neighborhood_caches,
        zero_delta,
    )
    baseline_metrics = local_metrics(
        baseline_forget_cached,
        required_forget_nll,
        baseline_neighborhood_cached,
        required_neighborhood_nll,
        neighborhood_instances,
        zero_delta,
        tolerance=args.comparison_tolerance,
    )
    _write_json(output_dir / "baseline_local_metrics.json", baseline_metrics)
    _write_json(
        output_dir / "baseline_forget_instances.json",
        instance_reports(
            forget_instances,
            baseline_forget_nll,
            required_forget_nll,
        ),
    )
    _write_json(
        output_dir / "baseline_neighborhood_instances.json",
        instance_reports(
            neighborhood_instances,
            baseline_neighborhood_nll,
            required_neighborhood_nll,
        ),
    )

    logs: List[Dict[str, Any]] = []
    best = CandidateSnapshot(
        step=0,
        delta_rows=zero_delta.detach().cpu().clone(),
        metrics=baseline_metrics,
    )
    stopped_early = not active_positions
    steps_completed = 0

    if selected_ids and active_positions:
        active_neighborhood_caches = [
            neighborhood_caches[position] for position in active_positions
        ]
        direction_basis = None
        if args.repair_rank > 0:
            direction_basis = active.orthonormal_row_basis(
                _hidden_rows(active_neighborhood_caches),
                max_rank=args.repair_rank,
            )
        forget_basis = None
        if args.project_away_forget_hidden and args.forget_projection_rank > 0:
            forget_basis = active.orthonormal_row_basis(
                _hidden_rows(forget_caches),
                max_rank=args.forget_projection_rank,
            )
        delta_module = active.SelectedRowDelta(
            len(selected_ids),
            output_weight.shape[1],
            direction_basis=direction_basis,
            retained_basis=forget_basis,
            device=device,
        )
        optimizer = active.make_repair_optimizer(
            delta_module,
            args.repair_optimizer,
            args.repair_lr,
        )
        for step in range(1, args.repair_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            delta = delta_module.effective_delta()
            current_forget = answer_nlls_from_delta_caches(
                forget_caches,
                delta,
            )
            current_neighborhood = answer_nlls_from_delta_caches(
                neighborhood_caches,
                delta,
            )
            terms = confidence_objective_terms(
                current_forget,
                required_forget_nll,
                current_neighborhood,
                required_neighborhood_nll,
            )
            delta_l2 = delta.square().sum()
            loss = (
                args.confidence_weight * terms["confidence_hinge"]
                + args.forget_protection_weight * terms["forget_hinge"]
                + args.delta_l2_lambda * delta_l2
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite repair loss at step {step}: "
                    f"{float(loss.detach().cpu())}"
                )
            loss.backward()
            optimizer.step()
            norm_before, norm_after, norm_projected = (
                active.constrain_effective_delta_norm(
                    delta_module,
                    args.max_delta_norm,
                )
            )
            steps_completed = step

            with torch.no_grad():
                candidate_delta = delta_module.effective_delta()
                current_forget = answer_nlls_from_delta_caches(
                    forget_caches,
                    candidate_delta,
                )
                current_neighborhood = answer_nlls_from_delta_caches(
                    neighborhood_caches,
                    candidate_delta,
                )
                metrics = local_metrics(
                    current_forget,
                    required_forget_nll,
                    current_neighborhood,
                    required_neighborhood_nll,
                    neighborhood_instances,
                    candidate_delta,
                    tolerance=args.comparison_tolerance,
                )
                if candidate_priority(metrics) < candidate_priority(best.metrics):
                    best = CandidateSnapshot(
                        step=step,
                        delta_rows=candidate_delta.detach().cpu().clone(),
                        metrics=metrics,
                    )
            if step == 1 or step % args.log_every == 0:
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()),
                    "confidence_hinge": float(
                        terms["confidence_hinge"].detach().cpu()
                    ),
                    "forget_hinge": float(
                        terms["forget_hinge"].detach().cpu()
                    ),
                    "delta_l2": float(delta_l2.detach().cpu()),
                    "delta_norm_before_projection": norm_before,
                    "delta_norm": norm_after,
                    "delta_norm_projected": norm_projected,
                    **metrics,
                }
                logs.append(row)
                print(
                    f"step={step} unmet={metrics['neighborhood_target_unmet_count']} "
                    f"forget_violations={metrics['forget_protection_violation_count']} "
                    f"utility_macro_ap={metrics['utility_macro_answer_probability']:.6f} "
                    f"delta_norm={metrics['selected_lm_head_delta_norm']:.6f}"
                )
            if (
                args.stop_when_all_satisfied
                and metrics["forget_protection_violation_count"] == 0
                and metrics["neighborhood_target_unmet_count"] == 0
            ):
                stopped_early = True
                break

    _write_jsonl(output_dir / "repair_log.jsonl", logs)
    if (
        best.metrics["neighborhood_target_unmet_count"] > 0
        and not args.save_best_effort
    ):
        raise RuntimeError(
            "No forget-safe candidate met every neighborhood confidence target. "
            "No checkpoint was saved. Increase --repair-steps, adjust the sparse "
            "rank/row budget, or explicitly pass --save-best-effort."
        )
    if best.metrics["forget_protection_violation_count"] > 0:
        raise RuntimeError("Candidate selection produced a forget-unsafe repair")

    best_delta = best.delta_rows.to(device=device, dtype=torch.float32)
    set_selected_lm_head_rows(
        output_weight,
        selected_ids,
        baseline_rows,
        best_delta,
    )
    after_forget_nll = score_answer_instances(
        model,
        tok,
        forget_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    after_neighborhood_nll = score_answer_instances(
        model,
        tok,
        neighborhood_instances,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    ).detach()
    materialized_forget_violations = after_forget_nll < (
        required_forget_nll - args.materialization_tolerance
    )
    if materialized_forget_violations.any():
        set_selected_lm_head_rows(
            output_weight,
            selected_ids,
            baseline_rows,
            torch.zeros_like(best_delta),
        )
        raise RuntimeError(
            "BF16 materialization weakened "
            f"{int(materialized_forget_violations.sum().cpu())} protected "
            "forget answers; the candidate was reverted and not saved."
        )
    after_metrics = local_metrics(
        after_forget_nll,
        required_forget_nll,
        after_neighborhood_nll,
        required_neighborhood_nll,
        neighborhood_instances,
        best_delta,
        tolerance=args.materialization_tolerance,
    )
    _write_json(output_dir / "candidate_local_metrics.json", after_metrics)
    _write_json(
        output_dir / "candidate_forget_instances.json",
        instance_reports(
            forget_instances,
            after_forget_nll,
            required_forget_nll,
        ),
    )
    _write_json(
        output_dir / "candidate_neighborhood_instances.json",
        instance_reports(
            neighborhood_instances,
            after_neighborhood_nll,
            required_neighborhood_nll,
        ),
    )

    if (
        model.get_input_embeddings().weight.data_ptr() != input_storage_pointer
        or model.get_input_embeddings().weight._version != input_tensor_version
    ):
        raise RuntimeError("Input embeddings changed during sparse output repair")

    summary = {
        "method": METHOD,
        "input_checkpoint": args.model_path,
        "reference_model": args.reference_model_path,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "seed": args.seed,
        "forget_instance_count": len(forget_instances),
        "neighborhood_instance_count": len(neighborhood_instances),
        "initially_active_neighborhood_instances": len(active_positions),
        "selected_lm_head_row_count": len(selected_ids),
        "best_step": best.step,
        "steps_completed": steps_completed,
        "stopped_early": stopped_early,
        "baseline_local_metrics": baseline_metrics,
        "candidate_local_metrics": after_metrics,
        "input_embeddings_unchanged": True,
        "transformer_parameters_frozen": True,
        "only_selected_lm_head_rows_materialized": True,
        "checkpoint_saved": bool(args.save_model),
        "full_tofu_evaluation": (
            "Run separately by run_tofu_gagd_neighborhood_confidence.sh "
            "after candidate selection."
        ),
    }
    _write_json(output_dir / "repair_summary.json", summary)
    if args.save_model:
        active.save_repair_checkpoint(
            model,
            tok,
            output_dir / "checkpoint",
            repair_config=config_used,
        )
    print(
        "TOFU neighborhood-confidence repair complete: "
        f"active={len(active_positions)}, selected_rows={len(selected_ids)}, "
        f"best_step={best.step}, "
        f"utility_macro_ap={after_metrics['utility_macro_answer_probability']:.6f}"
    )


if __name__ == "__main__":
    main()
