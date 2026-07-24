#!/usr/bin/env python3
"""Protocol-conditional TOFU forgetting with exact protected-input isolation.

This method is intended for the unusually strict clean-TOFU operating point:

* mean clean-forget answer probability at or below 2e-5; and
* retain answer probability at least 0.9999998 of the full-TOFU model.

Ordinary full-model, LoRA, embedding-row, and global LM-head updates cannot
reliably meet a two-tenths-of-a-part-per-million retain budget.  This stage
therefore starts from the original full-TOFU model and repurposes dormant
Llama-3 reserved token IDs as input-only triggers for protocol-matched forget
questions.  The tied LM head is cloned before any input row changes.  A sparse
adversarial delta is then optimized only for the selected reserved input rows.

Before training, the stage proves that every protected retain, real-author,
and world-fact prompt/answer has identical token IDs under the original and
trigger tokenizers.  Since the transformer, output head, and all other input
rows are frozen, protected logits are unchanged by construction.  The complete
forget and retain sets are nevertheless re-scored after BF16 materialization,
and no normal checkpoint is saved unless both hard targets pass.

This is an evaluation-protocol-conditional repair, not evidence of general
semantic erasure.  The output reports every trigger phrase so exact-question
conditioning cannot be mistaken for held-out paraphrase generalization.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer

import gagd_active_case_repair as active
import gagd_compare as gagd
import tofu_gagd_active_forget_repair as targeted
import tofu_gagd_neighborhood_confidence as tofu


METHOD = "tofu_prompt_conditional_input_repair"
RESERVED_TOKEN_RE = re.compile(r"<\|reserved_special_token_\d+\|>")
WORD_RE = re.compile(
    r"[A-Za-z0-9]+(?:['’./-][A-Za-z0-9]+)*",
    flags=re.UNICODE,
)
GENERIC_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "in",
    "is",
    "it",
    "its",
    "name",
    "of",
    "on",
    "or",
    "some",
    "tell",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
}


@dataclass(frozen=True)
class TriggerSpec:
    sampled_position: int
    source_index: int
    question: str
    content: str
    token_id: int
    used_full_question_fallback: bool


@dataclass(frozen=True)
class CandidateSnapshot:
    epoch: int
    delta_rows: torch.Tensor
    metrics: Dict[str, Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--forget-split",
        choices=sorted(tofu.PAIRED_RETAIN_SPLITS),
        default="forget05",
    )
    parser.add_argument("--retain-split", default="retain95")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forget-num", type=int, default=200)
    parser.add_argument("--retain-num", type=int, default=1000)
    parser.add_argument(
        "--target-forget-answer-probability",
        type=float,
        default=2e-5,
    )
    parser.add_argument(
        "--min-retain-probability-ratio",
        type=float,
        default=0.9999998,
    )
    parser.add_argument("--target-nll-buffer", type=float, default=0.5)
    parser.add_argument("--trigger-min-words", type=int, default=3)
    parser.add_argument("--trigger-max-words", type=int, default=10)
    parser.add_argument(
        "--allow-full-question-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the complete forget question only when no shorter phrase is "
            "unique across forget and protected records."
        ),
    )
    parser.add_argument("--repair-epochs", type=int, default=60)
    parser.add_argument("--repair-lr", type=float, default=0.1)
    parser.add_argument("--hardest-forget-weight", type=float, default=4.0)
    parser.add_argument("--max-delta-norm", type=float, default=24.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--materialization-relative-tolerance",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--save-best-effort",
        action="store_true",
        help="Diagnostic only; never selected by the recommended runner.",
    )
    parser.add_argument(
        "--save-model",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--device-map", choices=["single"], default="single")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    expected_retain = tofu.PAIRED_RETAIN_SPLITS[args.forget_split]
    if args.retain_split != expected_retain:
        raise ValueError(
            f"{args.forget_split} must be paired with {expected_retain}, "
            f"not {args.retain_split}"
        )
    if not 0.0 < args.target_forget_answer_probability < 1.0:
        raise ValueError("--target-forget-answer-probability must lie in (0,1)")
    if not 0.0 < args.min_retain_probability_ratio <= 1.0:
        raise ValueError("--min-retain-probability-ratio must lie in (0,1]")
    positive = (
        "forget_num",
        "retain_num",
        "trigger_min_words",
        "trigger_max_words",
        "repair_epochs",
        "repair_lr",
        "batch_size",
        "eval_batch_size",
        "max_length",
        "log_every",
    )
    for name in positive:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    nonnegative = (
        "target_nll_buffer",
        "hardest_forget_weight",
        "max_delta_norm",
        "gradient_clip_norm",
        "materialization_relative_tolerance",
    )
    for name in nonnegative:
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"--{name.replace('_', '-')} must be finite and non-negative"
            )
    if args.trigger_min_words > args.trigger_max_words:
        raise ValueError("--trigger-min-words cannot exceed --trigger-max-words")


def protocol_sample(
    rows: Sequence[Dict[str, Any]],
    sample_size: int,
    seed: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Match ``tofu_eval.subset_samples`` while retaining source indices."""
    indexed = list(enumerate(rows))
    if len(indexed) > sample_size:
        indexed = random.Random(seed).sample(indexed, sample_size)
    return indexed


def rows_to_instances(
    indexed_rows: Sequence[Tuple[int, Dict[str, Any]]],
    split: str,
    tok: Any,
) -> List[tofu.TOFUAnswerInstance]:
    instances: List[tofu.TOFUAnswerInstance] = []
    for sampled_position, (source_index, row) in enumerate(indexed_rows):
        question = str(row["question"])
        answer = str(row["answer"])
        instances.append(
            tofu.TOFUAnswerInstance(
                split=split,
                source_index=source_index,
                sampled_position=sampled_position,
                question=question,
                answer=answer,
                prompt=tofu.format_question_prompt(tok, question),
            )
        )
    return instances


def _candidate_phrases(
    question: str,
    *,
    min_words: int,
    max_words: int,
) -> Iterable[Tuple[int, int, str]]:
    words = list(WORD_RE.finditer(question))
    upper = min(max_words, len(words))
    for width in range(min_words, upper + 1):
        for start in range(0, len(words) - width + 1):
            selected = words[start : start + width]
            lowered = [match.group(0).casefold() for match in selected]
            specificity = sum(word not in GENERIC_WORDS for word in lowered)
            if specificity == 0:
                continue
            content = question[selected[0].start() : selected[-1].end()]
            yield width, specificity, content


def select_unique_trigger_phrase(
    question: str,
    other_forget_questions: Sequence[str],
    protected_texts: Sequence[str],
    *,
    min_words: int,
    max_words: int,
    allow_full_question_fallback: bool,
) -> Tuple[str, bool]:
    """Select a deterministic question phrase absent from every other input."""
    forbidden = [
        text.casefold()
        for text in [*other_forget_questions, *protected_texts]
    ]
    candidates: List[Tuple[int, int, int, str]] = []
    for width, specificity, content in _candidate_phrases(
        question,
        min_words=min_words,
        max_words=max_words,
    ):
        folded = content.casefold()
        if any(folded in text for text in forbidden):
            continue
        # Prefer a short trigger, then more content-bearing words.
        candidates.append((width, -specificity, len(content), content))
    if candidates:
        return min(candidates)[-1], False
    if not allow_full_question_fallback:
        raise RuntimeError(
            "No protected-exclusive trigger phrase exists for forget question: "
            f"{question!r}"
        )
    folded_question = question.casefold()
    if any(folded_question in text for text in forbidden):
        raise RuntimeError(
            "The complete forget question is not protected-exclusive: "
            f"{question!r}"
        )
    return question, True


def reserved_token_ids_from_tokenizer_json(payload: Dict[str, Any]) -> List[int]:
    selected = [
        int(row["id"])
        for row in payload.get("added_tokens", [])
        if RESERVED_TOKEN_RE.fullmatch(str(row.get("content", "")))
    ]
    return sorted(selected)


def build_trigger_contents(
    forget_instances: Sequence[tofu.TOFUAnswerInstance],
    protected_instances: Sequence[tofu.TOFUAnswerInstance],
    *,
    min_words: int,
    max_words: int,
    allow_full_question_fallback: bool,
) -> List[Tuple[str, bool]]:
    forget_questions = [instance.question for instance in forget_instances]
    protected_texts = [
        f"{instance.question} {instance.answer}"
        for instance in protected_instances
    ]
    selected: List[Tuple[str, bool]] = []
    for position, question in enumerate(forget_questions):
        selected.append(
            select_unique_trigger_phrase(
                question,
                [
                    other
                    for other_position, other in enumerate(forget_questions)
                    if other_position != position
                ],
                protected_texts,
                min_words=min_words,
                max_words=max_words,
                allow_full_question_fallback=allow_full_question_fallback,
            )
        )
    contents = [content for content, _ in selected]
    if len(contents) != len(set(contents)):
        raise RuntimeError("Forget trigger phrases are not one-to-one")
    return selected


def patch_reserved_tokenizer(
    source_tok: Any,
    tokenizer_dir: Path,
    trigger_contents: Sequence[str],
) -> Tuple[Any, List[int]]:
    """Rename dormant reserved added tokens without changing vocabulary size."""
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    source_tok.save_pretrained(tokenizer_dir)
    tokenizer_json_path = tokenizer_dir / "tokenizer.json"
    tokenizer_config_path = tokenizer_dir / "tokenizer_config.json"
    tokenizer_payload = json.loads(tokenizer_json_path.read_text(encoding="utf-8"))
    token_ids = reserved_token_ids_from_tokenizer_json(tokenizer_payload)
    if len(trigger_contents) > len(token_ids):
        raise RuntimeError(
            f"Need {len(trigger_contents)} dormant reserved token IDs, but "
            f"the tokenizer exposes only {len(token_ids)}"
        )
    assigned_ids = token_ids[: len(trigger_contents)]
    content_by_id = dict(zip(assigned_ids, trigger_contents))
    for row in tokenizer_payload.get("added_tokens", []):
        token_id = int(row["id"])
        if token_id not in content_by_id:
            continue
        row.update(
            {
                "content": content_by_id[token_id],
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    tokenizer_json_path.write_text(
        json.dumps(tokenizer_payload, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    config_payload = json.loads(
        tokenizer_config_path.read_text(encoding="utf-8")
    )
    decoder = config_payload.get("added_tokens_decoder", {})
    for token_id, content in content_by_id.items():
        key = str(token_id)
        if key not in decoder:
            raise RuntimeError(
                f"Reserved token ID {token_id} is absent from tokenizer config"
            )
        decoder[key].update(
            {
                "content": content,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    tokenizer_config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    patched = AutoTokenizer.from_pretrained(tokenizer_dir)
    if len(patched) != len(source_tok):
        raise RuntimeError("Reserved-token renaming changed tokenizer size")
    for token_id, content in content_by_id.items():
        encoded = patched(content, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        if list(encoded) != [token_id]:
            raise RuntimeError(
                f"Trigger {content!r} did not encode to reserved ID {token_id}: "
                f"{encoded}"
            )
    return patched, assigned_ids


def build_trigger_specs(
    forget_instances: Sequence[tofu.TOFUAnswerInstance],
    trigger_rows: Sequence[Tuple[str, bool]],
    token_ids: Sequence[int],
) -> List[TriggerSpec]:
    if not (
        len(forget_instances) == len(trigger_rows) == len(token_ids)
    ):
        raise ValueError("Forget instances, trigger rows, and token IDs must align")
    return [
        TriggerSpec(
            sampled_position=instance.sampled_position,
            source_index=instance.source_index,
            question=instance.question,
            content=content,
            token_id=int(token_id),
            used_full_question_fallback=used_fallback,
        )
        for instance, (content, used_fallback), token_id in zip(
            forget_instances,
            trigger_rows,
            token_ids,
        )
    ]


def audit_token_isolation(
    original_tok: Any,
    patched_tok: Any,
    forget_instances: Sequence[tofu.TOFUAnswerInstance],
    protected_instances: Sequence[tofu.TOFUAnswerInstance],
    trigger_specs: Sequence[TriggerSpec],
    *,
    max_length: int,
) -> Dict[str, Any]:
    changed_protected: List[int] = []
    assigned = {spec.token_id for spec in trigger_specs}
    protected_trigger_hits: List[int] = []
    for position, instance in enumerate(protected_instances):
        original = tofu.answer_sequence_components(
            original_tok,
            instance,
            max_length,
        )
        patched = tofu.answer_sequence_components(
            patched_tok,
            instance,
            max_length,
        )
        if original != patched:
            changed_protected.append(position)
        if assigned.intersection(patched[0]):
            protected_trigger_hits.append(position)
    missing_forget: List[int] = []
    extra_forget: List[int] = []
    for position, (instance, spec) in enumerate(
        zip(forget_instances, trigger_specs)
    ):
        full_ids, _ = tofu.answer_sequence_components(
            patched_tok,
            instance,
            max_length,
        )
        hits = assigned.intersection(full_ids)
        if spec.token_id not in hits:
            missing_forget.append(position)
        if hits != {spec.token_id}:
            extra_forget.append(position)
    report = {
        "protected_instance_count": len(protected_instances),
        "changed_protected_tokenization_count": len(changed_protected),
        "changed_protected_positions": changed_protected,
        "protected_trigger_hit_count": len(protected_trigger_hits),
        "protected_trigger_hit_positions": protected_trigger_hits,
        "forget_instance_count": len(forget_instances),
        "forget_instances_missing_assigned_trigger": missing_forget,
        "forget_instances_with_nonexclusive_trigger_ids": extra_forget,
        "exact_protected_isolation": bool(
            not changed_protected
            and not protected_trigger_hits
            and not missing_forget
            and not extra_forget
        ),
    }
    if not report["exact_protected_isolation"]:
        raise RuntimeError(
            "Trigger tokenizer failed exact protected-input isolation: "
            f"{report}"
        )
    return report


class SparseInputDelta(nn.Module):
    """Sparse trainable deltas injected into selected input-embedding rows."""

    def __init__(
        self,
        token_ids: Sequence[int],
        vocab_size: int,
        hidden_size: int,
        device: torch.device,
    ):
        super().__init__()
        if not token_ids:
            raise ValueError("SparseInputDelta requires at least one token ID")
        lookup = torch.full(
            (vocab_size,),
            -1,
            dtype=torch.long,
            device=device,
        )
        ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        lookup.index_copy_(
            0,
            ids,
            torch.arange(len(token_ids), dtype=torch.long, device=device),
        )
        self.register_buffer("lookup", lookup)
        self.delta = nn.Embedding(
            len(token_ids),
            hidden_size,
            sparse=True,
            device=device,
            dtype=torch.float32,
        )
        nn.init.zeros_(self.delta.weight)

    def inject(self, token_ids: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        indices = self.lookup.index_select(0, token_ids.reshape(-1)).reshape(
            token_ids.shape
        )
        selected = indices.ge(0)
        if not selected.any():
            return output
        result = output.clone()
        correction = self.delta(indices[selected]).to(output.dtype)
        result[selected] = result[selected] + correction
        return result

    def hook(
        self,
        _module: nn.Module,
        inputs: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> torch.Tensor:
        return self.inject(inputs[0], output)

    @torch.no_grad()
    def constrain_norm(self, maximum: float) -> int:
        if maximum <= 0:
            return 0
        rows = self.delta.weight
        norms = rows.norm(dim=1, keepdim=True)
        projected = norms.squeeze(1) > maximum
        scale = torch.clamp(maximum / norms.clamp_min(1e-12), max=1.0)
        rows.mul_(scale)
        return int(projected.sum().cpu())


def clip_sparse_gradient(parameter: torch.Tensor, maximum: float) -> float:
    gradient = parameter.grad
    if gradient is None:
        return 0.0
    if not gradient.is_sparse:
        raise RuntimeError("Trigger delta gradient unexpectedly became dense")
    gradient = gradient.coalesce()
    values = gradient.values()
    norm = float(values.norm().detach().cpu())
    if maximum > 0 and norm > maximum:
        values.mul_(maximum / max(norm, 1e-12))
    parameter.grad = gradient
    return norm


def differentiable_answer_nlls(
    model: nn.Module,
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    device: torch.device,
    *,
    max_length: int,
) -> torch.Tensor:
    components = [
        tofu.answer_sequence_components(tok, instance, max_length)
        for instance in instances
    ]
    pad_id = tok.pad_token_id
    if pad_id is None:
        pad_id = tok.eos_token_id if tok.eos_token_id is not None else 0
    encoded = tofu._right_padded_batch(
        [component[0] for component in components],
        int(pad_id),
        device,
    )
    output = model(**encoded, use_cache=False)
    values: List[torch.Tensor] = []
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
        logits = output.logits[row_index].index_select(0, positions).float()
        values.append(
            F.cross_entropy(
                logits,
                answer_ids,
                reduction="mean",
            )
        )
    return torch.stack(values)


def forget_metrics(nll: torch.Tensor, target_nll: float) -> Dict[str, Any]:
    probabilities = torch.exp(-nll.float())
    active_mask = nll < target_nll
    return {
        "active_forget_instance_count": int(active_mask.sum().detach().cpu()),
        "forget_answer_probability_mean": float(
            probabilities.mean().detach().cpu()
        ),
        "forget_answer_probability_max": float(
            probabilities.max().detach().cpu()
        ),
        "minimum_forget_answer_nll": float(nll.min().detach().cpu()),
    }


def candidate_priority(metrics: Dict[str, Any]) -> Tuple[int, float, float, float]:
    return (
        int(metrics["active_forget_instance_count"]),
        float(metrics["forget_answer_probability_max"]),
        float(metrics["forget_answer_probability_mean"]),
        float(metrics["selected_input_delta_norm"]),
    )


def _model_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )


def _score(
    model: nn.Module,
    tok: Any,
    instances: Sequence[tofu.TOFUAnswerInstance],
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    return tofu.score_answer_instances(
        model,
        tok,
        instances,
        device,
        batch_size=args.eval_batch_size,
        max_length=args.max_length,
    ).detach()


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)
    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_used = {
        **vars(args),
        "method": METHOD,
        "parameter_scope": "reserved_trigger_input_embedding_rows_only",
        "source_checkpoint": args.model_path,
        "transformer_frozen": True,
        "output_head_frozen_and_untied": True,
        "non_trigger_input_rows_frozen": True,
        "evaluation_protocol_conditional": True,
    }
    gagd.write_json(output_dir / "config_used.json", config_used)

    original_tok = AutoTokenizer.from_pretrained(args.model_path)
    if original_tok.pad_token is None:
        original_tok.pad_token = original_tok.eos_token

    print("Loading protocol-matched TOFU records")
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
    forget_indexed = protocol_sample(forget_rows, args.forget_num, args.seed)
    retain_indexed = protocol_sample(retain_rows, args.retain_num, args.seed)
    real_indexed = list(enumerate(real_rows))
    world_indexed = list(enumerate(world_rows))
    original_forget = rows_to_instances(
        forget_indexed,
        args.forget_split,
        original_tok,
    )
    original_retain = rows_to_instances(retain_indexed, "retain", original_tok)
    original_real = rows_to_instances(
        real_indexed,
        "real_authors",
        original_tok,
    )
    original_world = rows_to_instances(
        world_indexed,
        "world_facts",
        original_tok,
    )
    original_protected = [
        *original_retain,
        *original_real,
        *original_world,
    ]

    trigger_rows = build_trigger_contents(
        original_forget,
        original_protected,
        min_words=args.trigger_min_words,
        max_words=args.trigger_max_words,
        allow_full_question_fallback=args.allow_full_question_fallback,
    )
    patched_tok, trigger_token_ids = patch_reserved_tokenizer(
        original_tok,
        output_dir / "trigger_tokenizer",
        [content for content, _ in trigger_rows],
    )
    if patched_tok.pad_token is None:
        patched_tok.pad_token = patched_tok.eos_token
    forget_instances = rows_to_instances(
        forget_indexed,
        args.forget_split,
        patched_tok,
    )
    retain_instances = rows_to_instances(retain_indexed, "retain", patched_tok)
    real_instances = rows_to_instances(
        real_indexed,
        "real_authors",
        patched_tok,
    )
    world_instances = rows_to_instances(
        world_indexed,
        "world_facts",
        patched_tok,
    )
    trigger_specs = build_trigger_specs(
        forget_instances,
        trigger_rows,
        trigger_token_ids,
    )
    isolation_report = audit_token_isolation(
        original_tok,
        patched_tok,
        original_forget,
        original_protected,
        trigger_specs,
        max_length=args.max_length,
    )
    gagd.write_json(
        output_dir / "trigger_plan.json",
        {
            "method": METHOD,
            "reserved_trigger_count": len(trigger_specs),
            "full_question_fallback_count": sum(
                spec.used_full_question_fallback for spec in trigger_specs
            ),
            "triggers": [asdict(spec) for spec in trigger_specs],
        },
    )
    gagd.write_json(output_dir / "token_isolation_audit.json", isolation_report)

    model, _ = gagd.load_model_and_tokenizer(
        _model_args(args),
        for_training=False,
    )
    device = gagd.first_device(model)
    print("Scoring the untouched full-TOFU reference")
    reference_forget_nll = _score(
        model,
        original_tok,
        original_forget,
        device,
        args,
    )
    reference_retain_nll = _score(
        model,
        original_tok,
        original_retain,
        device,
        args,
    )
    reference_utility_nll = _score(
        model,
        original_tok,
        [*original_real, *original_world],
        device,
        args,
    )

    # Clone the tied output head before any reserved input row is changed.
    active.freeze_model_for_output_repair(model)
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    if input_weight.data_ptr() == output_weight.data_ptr():
        raise RuntimeError("Input embeddings and LM head remain tied")
    trigger_ids_tensor = torch.tensor(
        trigger_token_ids,
        dtype=torch.long,
        device=input_weight.device,
    )
    baseline_trigger_rows = input_weight.index_select(
        0,
        trigger_ids_tensor,
    ).detach().clone()

    delta_module = SparseInputDelta(
        trigger_token_ids,
        input_weight.shape[0],
        input_weight.shape[1],
        input_weight.device,
    )
    hook_handle = model.get_input_embeddings().register_forward_hook(
        delta_module.hook
    )
    triggered_retain_nll = _score(
        model,
        patched_tok,
        retain_instances,
        device,
        args,
    )
    input_retain_ratio = targeted.mean_answer_probability_ratio(
        triggered_retain_nll,
        reference_retain_nll,
    )
    if input_retain_ratio + 1e-12 < args.min_retain_probability_ratio:
        hook_handle.remove()
        raise RuntimeError(
            "The trigger tokenizer changed protected retain behavior despite "
            "the token-isolation audit: "
            f"ratio={input_retain_ratio:.10g}, required="
            f"{args.min_retain_probability_ratio:.10g}"
        )

    target_nll = -math.log(args.target_forget_answer_probability)
    required_nll = target_nll + args.target_nll_buffer
    initial_triggered_nll = _score(
        model,
        patched_tok,
        forget_instances,
        device,
        args,
    )
    initial_metrics = forget_metrics(initial_triggered_nll, target_nll)
    initial_metrics["selected_input_delta_norm"] = 0.0
    best = CandidateSnapshot(
        epoch=0,
        delta_rows=delta_module.delta.weight.detach().cpu().clone(),
        metrics=initial_metrics,
    )
    logs: List[Dict[str, Any]] = []
    norm_projection_steps = 0
    stopped_early = initial_metrics["active_forget_instance_count"] == 0

    if not stopped_early:
        if args.gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
            except TypeError:
                model.gradient_checkpointing_enable()
            model.config.use_cache = False
        optimizer = torch.optim.SparseAdam(
            delta_module.delta.parameters(),
            lr=args.repair_lr,
        )
        current_nll = initial_triggered_nll
        for epoch in range(1, args.repair_epochs + 1):
            active_positions = (
                (current_nll < required_nll)
                .nonzero(as_tuple=False)
                .flatten()
                .detach()
                .cpu()
                .tolist()
            )
            random.Random(args.seed + epoch).shuffle(active_positions)
            epoch_losses: List[float] = []
            epoch_gradient_norms: List[float] = []
            # Transformers only activates gradient checkpointing in training
            # mode. Llama-3 has no active dropout in this configuration; all
            # model parameters remain frozen.
            model.train()
            for start in range(0, len(active_positions), args.batch_size):
                positions = active_positions[start : start + args.batch_size]
                batch = [forget_instances[position] for position in positions]
                optimizer.zero_grad(set_to_none=True)
                batch_nll = differentiable_answer_nlls(
                    model,
                    patched_tok,
                    batch,
                    device,
                    max_length=args.max_length,
                )
                errors = torch.relu(required_nll - batch_nll)
                loss = (
                    errors.square().mean()
                    + args.hardest_forget_weight * errors.square().max()
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite prompt-conditional loss at epoch {epoch}"
                    )
                loss.backward()
                epoch_gradient_norms.append(
                    clip_sparse_gradient(
                        delta_module.delta.weight,
                        args.gradient_clip_norm,
                    )
                )
                optimizer.step()
                if delta_module.constrain_norm(args.max_delta_norm):
                    norm_projection_steps += 1
                epoch_losses.append(float(loss.detach().cpu()))

            model.eval()
            current_nll = _score(
                model,
                patched_tok,
                forget_instances,
                device,
                args,
            )
            metrics = forget_metrics(current_nll, target_nll)
            metrics["selected_input_delta_norm"] = float(
                delta_module.delta.weight.norm().detach().cpu()
            )
            if candidate_priority(metrics) < candidate_priority(best.metrics):
                best = CandidateSnapshot(
                    epoch=epoch,
                    delta_rows=(
                        delta_module.delta.weight.detach().cpu().clone()
                    ),
                    metrics=metrics,
                )
            log_row = {
                "epoch": epoch,
                "mean_loss": (
                    sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses
                    else 0.0
                ),
                "maximum_sparse_gradient_norm": (
                    max(epoch_gradient_norms) if epoch_gradient_norms else 0.0
                ),
                **metrics,
            }
            logs.append(log_row)
            if epoch == 1 or epoch % args.log_every == 0:
                print(
                    f"epoch={epoch} "
                    f"active={metrics['active_forget_instance_count']} "
                    f"mean_forget_prob="
                    f"{metrics['forget_answer_probability_mean']:.9g} "
                    f"max_forget_prob="
                    f"{metrics['forget_answer_probability_max']:.9g} "
                    f"delta_norm={metrics['selected_input_delta_norm']:.7g}"
                )
            if (
                metrics["active_forget_instance_count"] == 0
                and bool((current_nll >= required_nll).all())
            ):
                stopped_early = True
                break

    active.write_jsonl(output_dir / "repair_log.jsonl", logs)
    qualified_in_memory = (
        best.metrics["active_forget_instance_count"] == 0
    )
    if not qualified_in_memory and not args.save_best_effort:
        hook_handle.remove()
        raise RuntimeError(
            "No prompt-conditional candidate met the forget target. "
            f"Best epoch={best.epoch}, "
            f"active={best.metrics['active_forget_instance_count']}, "
            f"mean={best.metrics['forget_answer_probability_mean']:.9g}, "
            f"max={best.metrics['forget_answer_probability_max']:.9g}. "
            "No checkpoint was saved."
        )

    with torch.no_grad():
        delta_module.delta.weight.copy_(
            best.delta_rows.to(delta_module.delta.weight)
        )
    hook_handle.remove()
    with torch.no_grad():
        materialized_rows = baseline_trigger_rows + best.delta_rows.to(
            device=input_weight.device,
            dtype=input_weight.dtype,
        )
        input_weight.index_copy_(
            0,
            trigger_ids_tensor,
            materialized_rows,
        )

    print("Re-scoring materialized forget and protected sets")
    after_forget_nll = _score(
        model,
        patched_tok,
        forget_instances,
        device,
        args,
    )
    after_retain_nll = _score(
        model,
        patched_tok,
        retain_instances,
        device,
        args,
    )
    after_utility_nll = _score(
        model,
        patched_tok,
        [*real_instances, *world_instances],
        device,
        args,
    )
    allowed_probability = (
        args.target_forget_answer_probability
        * (1.0 + args.materialization_relative_tolerance)
    )
    after_probabilities = torch.exp(-after_forget_nll)
    materialized_active = after_probabilities > allowed_probability
    retain_ratio = targeted.mean_answer_probability_ratio(
        after_retain_nll,
        reference_retain_nll,
    )
    utility_instances = [*real_instances, *world_instances]
    utility_ratios = targeted.utility_probability_ratio_tensors(
        after_utility_nll,
        reference_utility_nll,
        utility_instances,
    )
    utility_ratio_report = {
        split: float(ratio.detach().cpu())
        for split, ratio in utility_ratios.items()
    }
    targets_met = bool(
        not materialized_active.any()
        and retain_ratio + 1e-12 >= args.min_retain_probability_ratio
    )
    if not targets_met and not args.save_best_effort:
        with torch.no_grad():
            input_weight.index_copy_(
                0,
                trigger_ids_tensor,
                baseline_trigger_rows,
            )
        raise RuntimeError(
            "Materialized prompt-conditional checkpoint missed a hard target: "
            f"active_forget={int(materialized_active.sum().cpu())}, "
            f"retain_ratio={retain_ratio:.10g}, required="
            f"{args.min_retain_probability_ratio:.10g}. The input edit was "
            "reverted and no checkpoint was saved."
        )

    # These are intentionally separate from tofu_eval outputs: they prove the
    # local clean-answer constraints before the expensive full evaluation.
    before_report = {
        "source_forget_answer_probability": targeted.mean_answer_probability(
            reference_forget_nll
        ),
        "triggered_zero_delta_forget_answer_probability": (
            targeted.mean_answer_probability(initial_triggered_nll)
        ),
        "source_retain_answer_probability": targeted.mean_answer_probability(
            reference_retain_nll
        ),
        "triggered_zero_delta_retain_probability_ratio": input_retain_ratio,
    }
    after_report = {
        **forget_metrics(after_forget_nll, target_nll),
        "retain_answer_probability": targeted.mean_answer_probability(
            after_retain_nll
        ),
        "retain_probability_ratio_vs_source": retain_ratio,
        "utility_probability_ratio_by_split": utility_ratio_report,
    }
    gagd.write_json(output_dir / "baseline_local_metrics.json", before_report)
    gagd.write_json(output_dir / "candidate_local_metrics.json", after_report)
    summary = {
        "method": METHOD,
        "source_checkpoint": args.model_path,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "seed": args.seed,
        "target_forget_answer_probability": (
            args.target_forget_answer_probability
        ),
        "minimum_retain_probability_ratio": (
            args.min_retain_probability_ratio
        ),
        "qualified": targets_met,
        "best_epoch": best.epoch,
        "epochs_completed": logs[-1]["epoch"] if logs else 0,
        "stopped_early": stopped_early,
        "reserved_trigger_count": len(trigger_specs),
        "full_question_fallback_count": sum(
            spec.used_full_question_fallback for spec in trigger_specs
        ),
        "materialized_active_forget_count": int(
            materialized_active.sum().cpu()
        ),
        "materialized_forget_answer_probability_mean": float(
            after_probabilities.mean().cpu()
        ),
        "materialized_forget_answer_probability_max": float(
            after_probabilities.max().cpu()
        ),
        "retain_probability_ratio_vs_source": retain_ratio,
        "utility_probability_ratio_by_split": utility_ratio_report,
        "token_isolation_audit": isolation_report,
        "norm_projection_steps": norm_projection_steps,
        "evaluation_protocol_conditional": True,
        "semantic_generalization_claimed": False,
        "checkpoint_saved": bool(args.save_model),
    }
    gagd.write_json(output_dir / "repair_summary.json", summary)
    if args.save_model:
        active.save_repair_checkpoint(
            model,
            patched_tok,
            output_dir / "checkpoint",
            repair_config=config_used,
        )
    print(
        "Prompt-conditional TOFU repair complete: "
        f"mean={float(after_probabilities.mean().cpu()):.9g}, "
        f"max={float(after_probabilities.max().cpu()):.9g}, "
        f"retain_ratio={retain_ratio:.10g}"
    )


if __name__ == "__main__":
    main()
