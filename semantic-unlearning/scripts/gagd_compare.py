#!/usr/bin/env python3
"""Unified GA/GD comparison for semantic unlearning on MCF and TOFU.

This script is intentionally self-contained and avoids optional editing baselines.
It compares full-model and embedding/lm_head-only GA/GD with all-token or
selective-token answer-only losses. On MCF it also supports all-token
embedding/lm_head training followed by base-row restoration.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional by requirement.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    pd = None

from mcf_zero_unlearn_official_eval import (
    evaluate_loaded_model_official,
    result_to_comparison_row,
    write_official_comparison,
)
from mcf_sampling import sample_first_mcf_records, sample_official_mcf_records

BASE_MODES = [
    "full_all_tokens",
    "full_selective_tokens",
    "emb_lm_all_tokens",
    "emb_lm_selective_tokens",
]
POST_TRAINING_RESTORE_MODE = "emb_lm_all_restore_post_training_true"
POST_TRAINING_TRUE_ALPHA = 0.75
POST_TRAINING_NEW_TRUE_ALPHA = 0.75
POST_TRAINING_NEW_RETAIN_ALPHA = 0.50
POST_TRAINING_NEW_TRUE_RETAIN_ALPHA = 0.25
POST_TRAINING_FREQUENCY_CAP_ALPHA = 0.5
MODES = BASE_MODES + [POST_TRAINING_RESTORE_MODE]
MCF_URL = "https://memit.baulab.info/data/dsets/multi_counterfact.json"
DEFAULT_MODEL_PATH = "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


@dataclass
class Example:
    prompt: str
    answer: str
    subject: str = ""
    target_new: str = ""
    target_true: str = ""
    paraphrase_prompts: Optional[List[str]] = None
    source: str = ""


@dataclass
class LossResult:
    loss: torch.Tensor
    fallback_examples: int
    contributing_tokens: int


@dataclass
class MarginLossResult:
    loss: torch.Tensor
    target_new_nll: torch.Tensor
    target_true_nll: torch.Tensor
    target_new_fallback_examples: int
    target_true_fallback_examples: int
    target_new_tokens: int
    target_true_tokens: int


@dataclass
class ZeroUnlearnGAResult:
    loss: torch.Tensor
    fallback_examples: int
    contributing_tokens: int


@dataclass
class ParamSummary:
    n_trainable_params: int
    n_total_params: int
    trainable_param_percent: float
    trainable_names: List[str]


@dataclass
class PostTrainingTokenGroups:
    target_new: List[int]
    target_true: List[int]
    retain: List[int]
    unique_target_new: List[int]
    unique_target_true: List[int]
    target_new_true_overlap: List[int]
    target_new_retain_overlap: List[int]
    target_new_true_retain_overlap: List[int]
    target_true_retain_overlap: List[int]
    overlap: List[int]


class EpochBatchSampler:
    """Yield shuffled batches while guaranteeing full dataset coverage per epoch."""

    def __init__(self, examples: Sequence[Example], batch_size: int, seed: int):
        if not examples:
            raise ValueError("EpochBatchSampler requires at least one example.")
        if batch_size <= 0:
            raise ValueError("EpochBatchSampler batch_size must be positive.")
        self.examples = examples
        self.batch_size = min(batch_size, len(examples))
        self.rng = random.Random(seed)
        self.indices = list(range(len(examples)))
        self.cursor = len(self.indices)

    def next_batch(self) -> List[Example]:
        batch: List[Example] = []
        while len(batch) < self.batch_size:
            if self.cursor >= len(self.indices):
                self.rng.shuffle(self.indices)
                self.cursor = 0
            take = min(self.batch_size - len(batch), len(self.indices) - self.cursor)
            batch.extend(
                self.examples[i]
                for i in self.indices[self.cursor : self.cursor + take]
            )
            self.cursor += take
        return batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_cuda_if_needed(device_map: str) -> None:
    if device_map != "auto" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training by default, but torch.cuda.is_available() is False. "
            "Run on a CUDA machine or pass --device-map auto only if your environment supports it."
        )


def torch_dtype(dtype_name: str) -> torch.dtype:
    if dtype_name == "bf16":
        return torch.bfloat16
    if dtype_name == "fp16":
        return torch.float16
    if dtype_name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def ensure_model_path(model_path: str) -> None:
    if not model_path:
        raise ValueError("--model-path is required or MODEL_PATH must be set.")
    p = Path(model_path)
    if p.is_absolute() and not p.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")


def load_model_and_tokenizer(args: argparse.Namespace, for_training: bool = True):
    ensure_model_path(args.model_path)
    dtype = torch_dtype(args.dtype)
    load_kwargs: Dict[str, Any] = {"torch_dtype": dtype}
    if args.device_map == "auto":
        load_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    if args.device_map != "auto":
        require_cuda_if_needed(args.device_map)
        model = model.to("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token is not None else tok.unk_token
    if for_training:
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
    return model, tok


def first_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def answer_with_eos(tok: AutoTokenizer, answer: str) -> str:
    if tok.eos_token and not answer.endswith(tok.eos_token):
        return answer + tok.eos_token
    return answer


def append_eos_for_dataset(dataset: str) -> bool:
    """Keep TOFU training aligned with ``tofu_eval.py`` answer positions."""
    return dataset != "tofu"


def normalize_answer(answer: str) -> str:
    if answer and not answer.startswith(" "):
        return " " + answer
    return answer


def format_mcf_prompt(prompt_template: str, subject: str) -> str:
    return prompt_template.format(subject) if "{}" in prompt_template else prompt_template


def extract_mcf_rewrite(record: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    rr = record.get("requested_rewrite")
    if isinstance(rr, list):
        if not rr or not isinstance(rr[0], dict):
            raise ValueError("MCF requested_rewrite list is empty or malformed.")
        rr = rr[0]
    if not isinstance(rr, dict):
        raise ValueError("MCF record missing dict requested_rewrite field.")
    required = ["prompt", "subject", "target_new"]
    missing = [k for k in required if k not in rr]
    if missing:
        raise ValueError(f"MCF requested_rewrite missing required fields: {missing}")
    if not isinstance(rr.get("target_new"), dict) or "str" not in rr["target_new"]:
        raise ValueError("MCF requested_rewrite.target_new.str is required.")
    paraphrases = record.get("paraphrase_prompts") or rr.get("paraphrase_prompts") or []
    if not isinstance(paraphrases, list):
        paraphrases = []
    return rr, [str(p) for p in paraphrases]


def download_if_missing(url: str, path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MCF data from {url} to {path}")
    urllib.request.urlretrieve(url, path)


def sample_mcf_raw_records(
    raw: Sequence[Dict[str, Any]],
    forget_num: int,
    retain_num: int,
    seed: int,
    sample_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Sample MCF records, including the exact JSON-LMHead/ZeroUnlearn split."""
    if sample_mode == "official":
        return sample_official_mcf_records(
            raw, forget_num, retain_num, seed, strict=True
        )

    if sample_mode == "first":
        return sample_first_mcf_records(raw, forget_num, retain_num, strict=True)

    if sample_mode == "shuffled":
        shuffled = list(raw)
        random.Random(seed).shuffle(shuffled)
        need = forget_num + retain_num
        if len(shuffled) < need:
            raise ValueError(f"MCF has only {len(shuffled)} records, need {need} for shuffled mode.")
        return shuffled[:forget_num], shuffled[forget_num:need]

    raise ValueError(f"Unsupported MCF sample mode: {sample_mode}")


def load_mcf(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    cache_path = resolve_output_path(args.mcf_cache_path)
    download_if_missing(args.mcf_url, cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("MCF JSON must be a list of records.")
    forget_raw, retain_raw = sample_mcf_raw_records(
        raw=raw,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        sample_mode=args.mcf_sample_mode,
    )

    def convert(records_raw: Sequence[Dict[str, Any]]) -> List[Example]:
        records: List[Example] = []
        for rec in records_raw:
            rr, paraphrases = extract_mcf_rewrite(rec)
            subject = str(rr["subject"])
            prompt = format_mcf_prompt(str(rr["prompt"]), subject)
            target_new = str(rr["target_new"]["str"])
            target_true = ""
            if isinstance(rr.get("target_true"), dict):
                target_true = str(rr["target_true"].get("str", ""))
            normalized_target_new = normalize_answer(target_new)
            normalized_target_true = normalize_answer(target_true) if target_true else ""
            if args.mcf_answer_field == "target_true" and not normalized_target_true:
                raise ValueError("--mcf-answer-field target_true requested, but an MCF record is missing target_true.str")
            train_answer = normalized_target_new if args.mcf_answer_field == "target_new" else normalized_target_true
            records.append(
                Example(
                    prompt=prompt,
                    answer=train_answer,
                    subject=subject,
                    target_new=normalized_target_new,
                    target_true=normalized_target_true,
                    paraphrase_prompts=[format_mcf_prompt(p, subject) for p in paraphrases],
                    source="mcf",
                )
            )
        return records

    forget = convert(forget_raw)
    retain = convert(retain_raw)
    return forget, retain


def load_tofu(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    forget_ds = load_dataset("locuslab/TOFU", name=args.forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=args.retain_split, split="train")

    def convert(row: Dict[str, Any]) -> Example:
        if "question" not in row or "answer" not in row:
            raise ValueError("TOFU rows must contain question and answer fields.")
        return Example(prompt=f"Question: {row['question']}\nAnswer:", answer=str(row["answer"]), source="tofu")

    forget = [convert(r) for r in forget_ds]
    retain = [convert(r) for r in retain_ds]
    rng = random.Random(args.seed)
    rng.shuffle(forget)
    rng.shuffle(retain)
    return forget[: args.forget_num], retain[: args.retain_num]


def load_data(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    if args.dataset == "mcf":
        return load_mcf(args)
    if args.dataset == "tofu":
        return load_tofu(args)
    raise ValueError(f"Unsupported dataset: {args.dataset}")


def token_ids_for_text(tok: AutoTokenizer, text: str) -> List[int]:
    return [int(i) for i in tok(text, add_special_tokens=False)["input_ids"]]


def special_token_ids(tok: AutoTokenizer) -> set[int]:
    ids = {tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id}
    return {int(i) for i in ids if i is not None}


def select_mcf_tokens(tok: AutoTokenizer, forget: Sequence[Example]) -> List[int]:
    selected: set[int] = set()
    for ex in forget:
        for text in (ex.subject, ex.target_new, ex.target_true):
            if text:
                selected.update(token_ids_for_text(tok, text))
    selected -= special_token_ids(tok)
    return sorted(selected)


def load_semantic_token_json(path: Path) -> List[int]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    ids: List[int] = []
    if isinstance(data, dict) and isinstance(data.get("token_ids"), list):
        ids.extend(int(x) for x in data["token_ids"])
    if isinstance(data, dict) and isinstance(data.get("semantic_tokens"), list):
        for item in data["semantic_tokens"]:
            if isinstance(item, dict) and "token_id" in item:
                ids.append(int(item["token_id"]))
    if isinstance(data, list):
        for item in data:
            if isinstance(item, int):
                ids.append(int(item))
            elif isinstance(item, dict) and "token_id" in item:
                ids.append(int(item["token_id"]))
    return sorted(set(ids))


def document_frequency(tok: AutoTokenizer, examples: Sequence[Example]) -> Counter:
    counts: Counter = Counter()
    for ex in examples:
        counts.update(set(token_ids_for_text(tok, ex.answer)))
    return counts


def select_tofu_tokens(tok: AutoTokenizer, forget: Sequence[Example], retain: Sequence[Example], args: argparse.Namespace) -> List[int]:
    if args.semantic_token_json:
        selected = set(load_semantic_token_json(Path(args.semantic_token_json)))
    else:
        forget_df = document_frequency(tok, forget)
        retain_df = document_frequency(tok, retain)
        n_forget = max(len(forget), 1)
        n_retain = max(len(retain), 1)
        scored = []
        for tok_id, count in forget_df.items():
            score = (count / n_forget) / ((retain_df.get(tok_id, 0) / n_retain) + 1e-6)
            scored.append((score, tok_id))
        scored.sort(reverse=True)
        selected = {tok_id for _, tok_id in scored[: args.selective_top_k]}
    selected -= special_token_ids(tok)
    return sorted(int(i) for i in selected)


def select_tokens(tok: AutoTokenizer, forget: Sequence[Example], retain: Sequence[Example], args: argparse.Namespace) -> List[int]:
    if args.dataset == "mcf":
        return select_mcf_tokens(tok, forget)
    return select_tofu_tokens(tok, forget, retain, args)


def collect_post_training_token_groups(
    tok: AutoTokenizer,
    forget: Sequence[Example],
    retain: Sequence[Example],
    excluded_token_ids: Sequence[int] = (),
) -> PostTrainingTokenGroups:
    """Build the disjoint MCF row policy used by the post-training restore mode."""
    target_new: set[int] = set()
    target_true: set[int] = set()
    retain_tokens: set[int] = set()

    for index, ex in enumerate(forget):
        if not ex.target_new or not ex.target_true:
            raise ValueError(
                "Post-training restoration requires both target_new and "
                f"target_true on every forget record; record {index} is incomplete."
            )
        target_new.update(token_ids_for_text(tok, ex.target_new))
        target_true.update(token_ids_for_text(tok, ex.target_true))

    # Match the JSON target-row method: both answer fields in retained MCF
    # records are protected from global vocabulary-row edits.
    for index, ex in enumerate(retain):
        if not ex.target_new or not ex.target_true:
            raise ValueError(
                "Post-training restoration protects both target fields on every "
                f"retain record; record {index} is incomplete."
            )
        for text in (ex.target_new, ex.target_true):
            retain_tokens.update(token_ids_for_text(tok, text))

    specials = special_token_ids(tok) | {
        int(token_id) for token_id in excluded_token_ids
    }
    target_new -= specials
    target_true -= specials
    retain_tokens -= specials

    unique_target_new = target_new - target_true - retain_tokens
    unique_target_true = target_true - target_new - retain_tokens
    target_new_true_retain_overlap = target_new & target_true & retain_tokens
    target_new_true_overlap = (target_new & target_true) - retain_tokens
    target_new_retain_overlap = (target_new & retain_tokens) - target_true
    target_true_retain_overlap = (target_true & retain_tokens) - target_new
    overlap = (
        target_new_true_overlap
        | target_new_retain_overlap
        | target_new_true_retain_overlap
        | target_true_retain_overlap
    )
    return PostTrainingTokenGroups(
        target_new=sorted(target_new),
        target_true=sorted(target_true),
        retain=sorted(retain_tokens),
        unique_target_new=sorted(unique_target_new),
        unique_target_true=sorted(unique_target_true),
        target_new_true_overlap=sorted(target_new_true_overlap),
        target_new_retain_overlap=sorted(target_new_retain_overlap),
        target_new_true_retain_overlap=sorted(target_new_true_retain_overlap),
        target_true_retain_overlap=sorted(target_true_retain_overlap),
        overlap=sorted(overlap),
    )


def build_batch(
    tok: AutoTokenizer,
    examples: Sequence[Example],
    device: torch.device,
    append_eos: bool = True,
) -> Dict[str, torch.Tensor]:
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    for ex in examples:
        prompt_ids = token_ids_for_text(tok, ex.prompt)
        answer_text = answer_with_eos(tok, ex.answer) if append_eos else ex.answer
        answer_ids = token_ids_for_text(tok, answer_text)
        if not answer_ids:
            raise ValueError(f"Example has zero answer tokens for prompt: {ex.prompt[:80]}")
        rows.append(prompt_ids + answer_ids)
        labels.append([-100] * len(prompt_ids) + answer_ids)
    max_len = max(len(r) for r in rows)
    input_ids, label_ids, attention = [], [], []
    for r, lab in zip(rows, labels):
        pad_len = max_len - len(r)
        input_ids.append(r + [pad_id] * pad_len)
        label_ids.append(lab + [-100] * pad_len)
        attention.append([1] * len(r) + [0] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "labels": torch.tensor(label_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention, dtype=torch.long, device=device),
    }


def answer_ce_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    examples: Sequence[Example],
    selected_token_ids: Optional[set[int]],
    device: torch.device,
    append_eos: bool = True,
) -> LossResult:
    batch = build_batch(tok, examples, device, append_eos=append_eos)
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :].contiguous()
    labels = batch["labels"][:, 1:].contiguous()
    per_token = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, reduction="none").view_as(labels)
    answer_mask = labels.ne(-100)
    fallback = 0
    if selected_token_ids is not None:
        selected = torch.zeros_like(answer_mask)
        for i in range(labels.size(0)):
            row_mask = answer_mask[i]
            if row_mask.any():
                row_selected = row_mask & torch.tensor(
                    [int(t.item()) in selected_token_ids for t in labels[i]],
                    dtype=torch.bool,
                    device=labels.device,
                )
                if not row_selected.any():
                    row_selected = row_mask
                    fallback += 1
                selected[i] = row_selected
        answer_mask = selected
    denom = answer_mask.sum().clamp_min(1)
    return LossResult((per_token * answer_mask.float()).sum() / denom, fallback, int(denom.item()))


def mcf_margin_forget_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    examples: Sequence[Example],
    selected_token_ids: Optional[set[int]],
    device: torch.device,
    forget_margin: float = 0.0,
) -> MarginLossResult:
    margins: List[torch.Tensor] = []
    new_losses: List[torch.Tensor] = []
    true_losses: List[torch.Tensor] = []
    new_fallbacks = 0
    true_fallbacks = 0
    new_tokens = 0
    true_tokens = 0
    for ex in examples:
        if not ex.target_new or not ex.target_true:
            raise ValueError("MCF margin forget loss requires both target_new and target_true for every forget example.")
        new_ex = Example(prompt=ex.prompt, answer=ex.target_new, source=ex.source)
        true_ex = Example(prompt=ex.prompt, answer=ex.target_true, source=ex.source)
        new_res = answer_ce_loss(
            model, tok, [new_ex], selected_token_ids, device, append_eos=False
        )
        true_res = answer_ce_loss(
            model, tok, [true_ex], selected_token_ids, device, append_eos=False
        )
        # A positive target margin avoids stopping at the fragile decision
        # boundary and concentrates gradient on examples that are not yet
        # forgotten according to the official target-new-vs-target-true test.
        margins.append(mcf_margin_objective(true_res.loss, new_res.loss, forget_margin))
        new_losses.append(new_res.loss)
        true_losses.append(true_res.loss)
        new_fallbacks += new_res.fallback_examples
        true_fallbacks += true_res.fallback_examples
        new_tokens += new_res.contributing_tokens
        true_tokens += true_res.contributing_tokens
    if not margins:
        raise ValueError("MCF margin forget loss received an empty forget batch.")
    return MarginLossResult(
        loss=torch.stack(margins).mean(),
        target_new_nll=torch.stack(new_losses).mean(),
        target_true_nll=torch.stack(true_losses).mean(),
        target_new_fallback_examples=new_fallbacks,
        target_true_fallback_examples=true_fallbacks,
        target_new_tokens=new_tokens,
        target_true_tokens=true_tokens,
    )


def mcf_margin_objective(
    target_true_nll: torch.Tensor,
    target_new_nll: torch.Tensor,
    forget_margin: float,
) -> torch.Tensor:
    # Minimizing this must push target_true_nll UP (suppress the sensitive
    # fact) and target_new_nll DOWN (raise the non-sensitive alternative), and
    # concentrate gradient on examples not yet forgotten (target_new_nll
    # still below target_true_nll). That requires the gap in this order --
    # new minus true, not true minus new -- since softplus is smallest for a
    # very negative argument and only that ordering makes "not yet forgotten"
    # (true_nll low, new_nll high) the large-loss region.
    return F.softplus(target_new_nll - target_true_nll + forget_margin)


def zerounlearn_ga_logprob_loss(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    examples: Sequence[Example],
    selected_token_ids: Optional[set[int]],
    device: torch.device,
) -> ZeroUnlearnGAResult:
    prompts = [ex.prompt for ex in examples]
    targets = []
    for ex in examples:
        if not ex.target_true:
            raise ValueError("ZeroUnlearn GA loss requires target_true for every MCF forget example.")
        target = ex.target_true
        if not target.startswith(" "):
            target = " " + target
        targets.append(target)

    prompt_inputs = tok(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)
    target_inputs = tok(
        targets,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    last_token_inds = prompt_inputs["attention_mask"].sum(dim=1) - 1
    batch_idx = torch.arange(len(prompts), device=device)
    logits = model(**prompt_inputs).logits[batch_idx, last_token_inds, :]
    log_probs = F.log_softmax(logits, dim=-1)

    target_ids = target_inputs["input_ids"]
    gathered = torch.gather(log_probs, 1, target_ids.clamp_min(0))
    mask = target_inputs["attention_mask"].bool()
    if tok.pad_token_id is not None:
        mask = mask & target_ids.ne(tok.pad_token_id)
    if tok.unk_token_id is not None:
        mask = mask & target_ids.ne(tok.unk_token_id)

    fallback_examples = 0
    if selected_token_ids is not None:
        selected = torch.zeros_like(mask)
        for i in range(target_ids.size(0)):
            row_valid = mask[i]
            row_selected = row_valid & torch.tensor(
                [int(t.item()) in selected_token_ids for t in target_ids[i]],
                dtype=torch.bool,
                device=device,
            )
            if not row_selected.any():
                row_selected = row_valid
                fallback_examples += 1
            selected[i] = row_selected
        mask = selected

    denom = mask.sum(dim=1).clamp_min(1)
    loss_per_example = (gathered * mask.float()).sum(dim=1) / denom
    return ZeroUnlearnGAResult(
        loss=loss_per_example.mean(),
        fallback_examples=fallback_examples,
        contributing_tokens=int(mask.sum().item()),
    )


def sample_batch(examples: Sequence[Example], batch_size: int) -> List[Example]:
    return [examples[random.randrange(len(examples))] for _ in range(min(batch_size, len(examples)))]


def configure_trainable(model: torch.nn.Module, mode: str) -> Tuple[ParamSummary, Dict[str, Any]]:
    tied_info: Dict[str, Any] = {"input_weight": None, "output_weight": None, "selected_mask": None}
    total = sum(p.numel() for p in model.parameters())
    if mode.startswith("full_"):
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for p in model.parameters():
            p.requires_grad_(False)
        inp = model.get_input_embeddings()
        out = model.get_output_embeddings()
        if inp is None or out is None:
            raise ValueError("Model must provide get_input_embeddings() and get_output_embeddings().")
        in_w = inp.weight
        out_w = out.weight
        tied = in_w.data_ptr() == out_w.data_ptr()
        print(f"lm_head tied to input embeddings: {tied}")
        in_w.requires_grad_(True)
        if not tied:
            out_w.requires_grad_(True)
        tied_info.update({"input_weight": in_w, "output_weight": out_w, "tied": tied})
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable tensors ({len(names)}):")
    for name in names[:50]:
        print(f"  - {name}")
    if len(names) > 50:
        print(f"  ... {len(names) - 50} more")
    print(f"Trainable params: {trainable:,}/{total:,} ({100.0 * trainable / max(total, 1):.6f}%)")
    return ParamSummary(trainable, total, 100.0 * trainable / max(total, 1), names), tied_info


def unique_trainable_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    seen: set[int] = set()
    params = []
    for p in model.parameters():
        if p.requires_grad and id(p) not in seen:
            seen.add(id(p))
            params.append(p)
    return params


def resolve_output_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_DIR / p


def learning_rate_for_mode(mode: str, args: argparse.Namespace) -> float:
    scoped_lr = args.full_lr if mode.startswith("full_") else args.emb_lm_lr
    return args.lr if scoped_lr is None else scoped_lr


def optimizer_name_for_mode(mode: str, args: argparse.Namespace) -> str:
    scoped_optimizer = args.full_optimizer if mode.startswith("full_") else args.emb_lm_optimizer
    return args.optimizer or scoped_optimizer or ("sgd" if mode.startswith("full_") else "adamw")


def make_optimizer(
    params: Sequence[torch.nn.Parameter],
    mode: str,
    args: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, str, float]:
    opt_name = optimizer_name_for_mode(mode, args)
    lr = learning_rate_for_mode(mode, args)
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=args.weight_decay), opt_name, lr
    if opt_name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=args.weight_decay), opt_name, lr
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay), opt_name, lr
    if opt_name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=args.weight_decay), opt_name, lr
        except Exception as exc:
            print(f"WARNING: bitsandbytes AdamW8bit unavailable ({exc}); falling back to torch.optim.AdamW")
            return torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay), "adamw", lr
    raise ValueError(f"Unsupported optimizer: {opt_name}")


def make_row_mask(weight: torch.nn.Parameter, selected_ids: Sequence[int]) -> torch.Tensor:
    mask = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
    ids = torch.tensor([i for i in selected_ids if 0 <= i < weight.shape[0]], dtype=torch.long, device=weight.device)
    if ids.numel() > 0:
        mask[ids] = True
    return mask


def apply_row_mask_and_restore(tied_info: Dict[str, Any], selected_ids: Sequence[int], originals: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
    in_w: torch.nn.Parameter = tied_info["input_weight"]
    out_w: torch.nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    if originals is None:
        return {"input": in_w.detach().clone(), "output": out_w.detach().clone() if not tied else in_w.detach().clone()}
    in_mask = make_row_mask(in_w, selected_ids)
    if in_w.grad is not None:
        in_w.grad[~in_mask] = 0
    if not tied:
        out_mask = make_row_mask(out_w, selected_ids)
        if out_w.grad is not None:
            out_w.grad[~out_mask] = 0
    return originals


def restore_non_selected_rows(tied_info: Dict[str, Any], selected_ids: Sequence[int], originals: Dict[str, torch.Tensor]) -> None:
    in_w: torch.nn.Parameter = tied_info["input_weight"]
    out_w: torch.nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    with torch.no_grad():
        in_mask = make_row_mask(in_w, selected_ids)
        in_w[~in_mask].copy_(originals["input"].to(in_w.device)[~in_mask])
        if not tied:
            out_mask = make_row_mask(out_w, selected_ids)
            out_w[~out_mask].copy_(originals["output"].to(out_w.device)[~out_mask])


def untie_embeddings(model: torch.nn.Module) -> bool:
    """Break weight sharing between input embeddings and the LM head.

    A tied model has one shared weight matrix serving two roles: "what this
    token means as input" and "how confident to be in this token as an
    answer". Editing that shared matrix to suppress a sensitive answer also
    perturbs the token's input representation everywhere else it occurs
    (e.g. as a subject/context word in an unrelated neighborhood prompt),
    which is a plausible source of specificity collateral damage that has
    nothing to do with the forget objective. Untying first lets Stage 1
    edit the output (LM-head) role without touching the input-embedding role.

    Returns True if the model was tied and is now untied; False if it was
    already untied (no-op).
    """
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if inp is None or out is None:
        raise ValueError("Model must expose input embeddings and output embeddings.")
    if inp.weight.data_ptr() != out.weight.data_ptr():
        return False
    out.weight = torch.nn.Parameter(out.weight.detach().clone())
    if hasattr(model, "config"):
        model.config.tie_word_embeddings = False
    if hasattr(model, "tie_word_embeddings"):
        model.tie_word_embeddings = False
    return True


def load_wiki_frequency_documents(
    wikidata_dir: str, doc_start: int, num_docs: int
) -> List[str]:
    """Documents used only to count token corpus frequency, disjoint from the
    documents mcf_zero_unlearn_official_eval's official PPL is scored
    against (raw_ds['train']['text'][:20]); default doc_start=20 keeps that
    disjoint. Mirrors run_mcf_setting5e_neutral_row_active_repair's own
    Wikidata protection-document loader for consistency."""
    if num_docs <= 0:
        return []
    path = Path(wikidata_dir)
    if not path.exists():
        return []
    raw_ds = load_from_disk(str(path))
    texts = raw_ds["train"]["text"][doc_start : doc_start + num_docs]
    return [t for t in texts if t and t.strip()]


def token_frequency_counts(
    tok: AutoTokenizer, documents: Sequence[str], vocab_size: int
) -> torch.Tensor:
    """Vocab-sized document-frequency count used to shrink post-training row
    edits on common tokens (likely the correct answer elsewhere in ordinary
    text) more than on rare, record-specific ones."""
    counts = torch.zeros(vocab_size, dtype=torch.long)
    for doc in documents:
        ids = tok(str(doc), add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        flat = torch.tensor([int(x) for x in ids], dtype=torch.long)
        flat = flat[flat < vocab_size]
        if flat.numel():
            counts += torch.bincount(flat, minlength=vocab_size)
    return counts


def snapshot_embedding_output_weights(
    tied_info: Dict[str, Any],
    *,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, torch.Tensor]:
    """Snapshot base embedding/output weights without duplicating tied weights."""
    in_w: torch.nn.Parameter = tied_info["input_weight"]
    out_w: torch.nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))
    input_snapshot = in_w.detach().to(device=device, copy=True)
    return {
        "input": input_snapshot,
        "output": (
            input_snapshot
            if tied
            else out_w.detach().to(device=device, copy=True)
        ),
    }


def valid_row_ids(weight: torch.Tensor, token_ids: Sequence[int]) -> torch.Tensor:
    return torch.tensor(
        [token_id for token_id in token_ids if 0 <= token_id < weight.shape[0]],
        dtype=torch.long,
        device=weight.device,
    )


@torch.no_grad()
def apply_post_training_row_restore(
    tied_info: Dict[str, Any],
    originals: Dict[str, torch.Tensor],
    groups: PostTrainingTokenGroups,
    true_alpha: float = POST_TRAINING_TRUE_ALPHA,
    new_true_alpha: float = POST_TRAINING_NEW_TRUE_ALPHA,
    new_retain_alpha: float = POST_TRAINING_NEW_RETAIN_ALPHA,
    new_true_retain_alpha: float = POST_TRAINING_NEW_TRUE_RETAIN_ALPHA,
    token_frequencies: Optional[torch.Tensor] = None,
    frequency_cap_alpha: float = POST_TRAINING_FREQUENCY_CAP_ALPHA,
) -> Dict[str, int]:
    """Restore base rows while retaining group-scaled target-new/target-true updates.

    When token_frequencies is given (a vocab-sized corpus document-frequency
    count, see token_frequency_counts), the fraction of the trained delta
    kept on unique_target_new and unique_target_true rows is additionally
    shrunk per-row by 1/(1+freq)^frequency_cap_alpha. A common token is
    likely the correct answer to some other, unrelated prompt, so editing it
    as confidently as a rare, record-specific token risks specificity
    collateral damage unrelated to the forget objective. With
    token_frequencies=None this is a no-op and behavior is unchanged.
    """
    overlap_alphas = {
        "target_true": true_alpha,
        "target_new_true": new_true_alpha,
        "target_new_retain": new_retain_alpha,
        "target_new_true_retain": new_true_retain_alpha,
    }
    for group_name, alpha in overlap_alphas.items():
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(
                f"Post-training alpha for {group_name} must be between 0 and 1."
            )
    if frequency_cap_alpha < 0:
        raise ValueError("Post-training frequency-cap alpha must be non-negative.")

    in_w: torch.nn.Parameter = tied_info["input_weight"]
    out_w: torch.nn.Parameter = tied_info["output_weight"]
    tied = bool(tied_info.get("tied"))

    def frequency_scale(
        row_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> Optional[torch.Tensor]:
        if token_frequencies is None or frequency_cap_alpha <= 0 or not row_ids.numel():
            return None
        freqs = token_frequencies.index_select(0, row_ids.to(token_frequencies.device)).to(
            device=device, dtype=torch.float32,
        )
        # Computed in fp32 for pow() stability, then cast down to the weight's
        # dtype (e.g. bf16) -- otherwise multiplying an fp32 scale against a
        # bf16 trained/base delta promotes the whole blend to fp32, and the
        # later index_copy_ into the bf16 weight fails on a dtype mismatch.
        scale = (1.0 + freqs).pow(-frequency_cap_alpha).unsqueeze(1)
        return scale.to(dtype=dtype)

    def restore_weight(weight: torch.nn.Parameter, base: torch.Tensor) -> Dict[str, int]:
        new_ids = valid_row_ids(weight, groups.unique_target_new)
        true_ids = valid_row_ids(weight, groups.unique_target_true)
        trained_new_rows = (
            weight.index_select(0, new_ids).detach().clone()
            if new_ids.numel()
            else None
        )
        # Captured before weight.copy_(base) below so the trained delta on the
        # rows most responsible for the sensitive fact is not lost -- unlike
        # unique_target_new, these are blended (true_alpha) rather than kept
        # outright, since only a handful of forget records inform each row.
        trained_true_rows = (
            weight.index_select(0, true_ids).detach().clone()
            if true_ids.numel()
            else None
        )
        interpolated_rows: List[Tuple[str, torch.Tensor, torch.Tensor]] = []
        for group_name, token_ids, alpha in (
            (
                "target_new_true_overlap",
                groups.target_new_true_overlap,
                new_true_alpha,
            ),
            (
                "target_new_retain_overlap",
                groups.target_new_retain_overlap,
                new_retain_alpha,
            ),
            (
                "target_new_true_retain_overlap",
                groups.target_new_true_retain_overlap,
                new_true_retain_alpha,
            ),
        ):
            row_ids = valid_row_ids(weight, token_ids)
            if not row_ids.numel():
                interpolated_rows.append(
                    (group_name, row_ids, weight.new_empty((0, weight.shape[1])))
                )
                continue
            trained_rows = weight.index_select(0, row_ids).detach().clone()
            base_rows = base.index_select(0, row_ids.to(base.device)).to(
                device=weight.device,
                dtype=weight.dtype,
            )
            final_rows = base_rows + alpha * (trained_rows - base_rows)
            interpolated_rows.append((group_name, row_ids, final_rows))

        # Base restoration handles retain-only, true/retain-only overlap, and
        # unrelated rows. Target-new/target-true groups are then interpolated.
        weight.copy_(base)
        if trained_new_rows is not None:
            new_scale = frequency_scale(new_ids, weight.device, weight.dtype)
            if new_scale is None:
                weight.index_copy_(0, new_ids, trained_new_rows)
            else:
                base_new_rows = base.index_select(0, new_ids.to(base.device)).to(
                    device=weight.device,
                    dtype=weight.dtype,
                )
                final_new_rows = base_new_rows + new_scale * (
                    trained_new_rows - base_new_rows
                )
                weight.index_copy_(0, new_ids, final_new_rows)
        if trained_true_rows is not None:
            base_true_rows = base.index_select(0, true_ids.to(base.device)).to(
                device=weight.device,
                dtype=weight.dtype,
            )
            true_scale = frequency_scale(true_ids, weight.device, weight.dtype)
            effective_true_alpha = (
                true_alpha if true_scale is None else true_alpha * true_scale
            )
            final_true_rows = base_true_rows + effective_true_alpha * (
                trained_true_rows - base_true_rows
            )
            weight.index_copy_(0, true_ids, final_true_rows)
        for _, row_ids, final_rows in interpolated_rows:
            if row_ids.numel():
                weight.index_copy_(0, row_ids, final_rows)
        return {
            "unique_target_new_rows_kept": int(new_ids.numel()),
            "unique_target_true_rows_interpolated": int(true_ids.numel()),
            **{
                f"{group_name}_rows_interpolated": int(row_ids.numel())
                for group_name, row_ids, _ in interpolated_rows
            },
        }

    input_counts = restore_weight(in_w, originals["input"])
    output_counts = (
        input_counts
        if tied
        else restore_weight(out_w, originals["output"])
    )
    return {
        **{f"input_{key}": value for key, value in input_counts.items()},
        **{f"output_{key}": value for key, value in output_counts.items()},
    }


def post_training_policy_report(
    tok: AutoTokenizer,
    groups: PostTrainingTokenGroups,
    applied_counts: Dict[str, int],
    true_alpha: float = POST_TRAINING_TRUE_ALPHA,
    new_true_alpha: float = POST_TRAINING_NEW_TRUE_ALPHA,
    new_retain_alpha: float = POST_TRAINING_NEW_RETAIN_ALPHA,
    new_true_retain_alpha: float = POST_TRAINING_NEW_TRUE_RETAIN_ALPHA,
) -> Dict[str, Any]:
    def decoded(token_ids: Sequence[int]) -> Dict[str, str]:
        return {
            str(token_id): tok.decode([token_id])
            for token_id in token_ids
        }

    return {
        "mode": POST_TRAINING_RESTORE_MODE,
        "true_alpha": true_alpha,
        "overlap_alphas": {
            "target_new_true": new_true_alpha,
            "target_new_retain": new_retain_alpha,
            "target_new_true_retain": new_true_retain_alpha,
        },
        "interpolation_formula": (
            "W_base[t] + alpha_g * (W_trained[t] - W_base[t])"
        ),
        "rules": {
            "unique_target_new": "keep_ga_gd_update",
            "unique_target_true": "base_plus_true_alpha_times_trained_delta",
            "target_new_true_overlap": "base_plus_alpha_times_trained_delta",
            "target_new_retain_overlap": "base_plus_alpha_times_trained_delta",
            "target_new_true_retain_overlap": "base_plus_alpha_times_trained_delta",
            "target_true_retain_overlap": "base_row",
            "retain_only": "base_row",
            "unrelated": "base_row",
        },
        "counts": {
            "target_new": len(groups.target_new),
            "target_true": len(groups.target_true),
            "retain": len(groups.retain),
            "unique_target_new": len(groups.unique_target_new),
            "unique_target_true": len(groups.unique_target_true),
            "target_new_true_overlap": len(groups.target_new_true_overlap),
            "target_new_retain_overlap": len(groups.target_new_retain_overlap),
            "target_new_true_retain_overlap": len(
                groups.target_new_true_retain_overlap
            ),
            "target_true_retain_overlap": len(groups.target_true_retain_overlap),
            "overlap": len(groups.overlap),
            **applied_counts,
        },
        "token_ids": asdict(groups),
        "tokens": {
            "unique_target_new": decoded(groups.unique_target_new),
            "unique_target_true": decoded(groups.unique_target_true),
            "target_new_true_overlap": decoded(groups.target_new_true_overlap),
            "target_new_retain_overlap": decoded(groups.target_new_retain_overlap),
            "target_new_true_retain_overlap": decoded(
                groups.target_new_true_retain_overlap
            ),
            "target_true_retain_overlap": decoded(
                groups.target_true_retain_overlap
            ),
            "overlap": decoded(groups.overlap),
        },
    }


def kl_retain_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tok: AutoTokenizer,
    examples: Sequence[Example],
    device: torch.device,
    *,
    append_eos: bool = True,
) -> torch.Tensor:
    batch = build_batch(tok, examples, device, append_eos=append_eos)
    with torch.no_grad():
        ref_logits = ref_model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
    cur_logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[:, :-1, :]
    mask = batch["attention_mask"][:, 1:].bool()
    cur_logp = F.log_softmax(cur_logits.float(), dim=-1)
    ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
    cur_p = cur_logp.exp()
    kl = (cur_p * (cur_logp - ref_logp)).sum(dim=-1)
    return (kl * mask.float()).sum() / mask.sum().clamp_min(1)


def train_mode(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    forget: Sequence[Example],
    retain: Sequence[Example],
    selected_ids: Sequence[int],
    mode: str,
    args: argparse.Namespace,
    mode_dir: Path,
) -> ParamSummary:
    summary, tied_info = configure_trainable(model, mode)
    selected_set = set(selected_ids) if "selective_tokens" in mode else None
    params = unique_trainable_params(model)
    opt, effective_optimizer, effective_lr = make_optimizer(params, mode, args)
    args.effective_optimizer = effective_optimizer
    args.effective_lr = effective_lr
    print(f"Optimizer for {mode}: {effective_optimizer}, lr={effective_lr:g}")
    device = first_device(model)
    originals = None
    if mode == "emb_lm_selective_tokens":
        originals = apply_row_mask_and_restore(tied_info, selected_ids)
        print(f"Embedding/lm_head selected rows allowed to update: {len(selected_ids)}")
    post_training_originals = None
    post_training_groups = None
    if mode == POST_TRAINING_RESTORE_MODE:
        post_training_originals = snapshot_embedding_output_weights(tied_info)
        post_training_groups = collect_post_training_token_groups(
            tok,
            forget,
            retain,
            excluded_token_ids=getattr(
                args,
                "post_training_excluded_token_ids",
                (),
            ),
        )
        print(
            "Post-training row policy: "
            f"keep {len(post_training_groups.unique_target_new)} unique target-new rows; "
            "partially keep target-new overlap updates at "
            f"{args.post_training_new_true_alpha:g}/"
            f"{args.post_training_new_retain_alpha:g}/"
            f"{args.post_training_new_true_retain_alpha:g}; "
            f"blend {len(post_training_groups.unique_target_true)} unique target-true rows "
            f"at true_alpha={args.post_training_true_alpha:g}"
        )
    ref_model = None
    if args.kl_retain_weight > 0:
        print("Loading frozen reference model for retain KL regularization")
        ref_model, _ = load_model_and_tokenizer(args, for_training=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    model.train()
    mode_dir.mkdir(parents=True, exist_ok=True)
    forget_sampler = None
    retain_sampler = None
    precomputed_forget_batches = getattr(args, "precomputed_forget_batches", None)
    if precomputed_forget_batches is not None:
        if len(precomputed_forget_batches) != args.steps:
            raise ValueError(
                "Precomputed entity-fact schedule length must equal --steps"
            )
        if any(not batch for batch in precomputed_forget_batches):
            raise ValueError("Precomputed entity-fact schedule contains an empty batch")
        if any(len(batch) != 1 for batch in precomputed_forget_batches):
            raise ValueError(
                "Balanced entity-fact scheduling requires one fact/view per update"
            )
    elif args.sampling_strategy == "epoch":
        forget_sampler = EpochBatchSampler(forget, args.batch_size, args.seed)
        retain_sampler = EpochBatchSampler(retain, args.retain_batch_size, args.seed + 1)
    if retain_sampler is None and args.sampling_strategy == "epoch":
        retain_sampler = EpochBatchSampler(retain, args.retain_batch_size, args.seed + 1)
    with (mode_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(args.steps), desc=f"train {mode}"):
            if precomputed_forget_batches is not None:
                forget_batch = list(precomputed_forget_batches[step])
            else:
                forget_batch = (
                    forget_sampler.next_batch()
                    if forget_sampler is not None
                    else sample_batch(forget, args.batch_size)
                )
            retain_batch = (
                retain_sampler.next_batch()
                if retain_sampler is not None
                else sample_batch(retain, args.retain_batch_size)
            )
            opt.zero_grad(set_to_none=True)
            margin_res = None
            ga_res = None
            append_eos = append_eos_for_dataset(args.dataset)
            if args.forget_loss_type == "mcf_margin":
                margin_res = mcf_margin_forget_loss(
                    model,
                    tok,
                    forget_batch,
                    selected_set,
                    device,
                    forget_margin=args.forget_margin,
                )
                forget_loss_for_log = margin_res.loss
                total = args.forget_weight * margin_res.loss
            elif args.forget_loss_type == "zerounlearn_ga":
                ga_res = zerounlearn_ga_logprob_loss(model, tok, forget_batch, selected_set, device)
                forget_loss_for_log = ga_res.loss
                total = args.forget_weight * ga_res.loss
            else:
                forget_res = answer_ce_loss(
                    model,
                    tok,
                    forget_batch,
                    selected_set,
                    device,
                    append_eos=append_eos,
                )
                forget_loss_for_log = forget_res.loss
                total = -args.forget_weight * forget_res.loss
            retain_res = None
            if args.retain_weight > 0:
                retain_res = answer_ce_loss(
                    model,
                    tok,
                    retain_batch,
                    selected_set,
                    device,
                    append_eos=append_eos,
                )
                total = total + args.retain_weight * retain_res.loss
            kl_val = torch.zeros((), device=device)
            if ref_model is not None:
                kl_val = kl_retain_loss(
                    model,
                    ref_model,
                    tok,
                    retain_batch,
                    device,
                    append_eos=append_eos,
                )
                total = total + args.kl_retain_weight * kl_val
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"Non-finite total loss in {mode} at step {step + 1}: "
                    f"{float(total.detach().cpu())}"
                )
            total.backward()
            if mode == "emb_lm_selective_tokens" and originals is not None:
                apply_row_mask_and_restore(tied_info, selected_ids, originals)
            grad_norm = None
            if args.grad_clip and args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(
                        f"Non-finite gradient norm in {mode} at step {step + 1}: "
                        f"{float(grad_norm.detach().cpu())}"
                    )
            opt.step()
            if mode == "emb_lm_selective_tokens" and originals is not None:
                restore_non_selected_rows(tied_info, selected_ids, originals)
            row = {
                "step": step + 1,
                "total_loss": float(total.detach().cpu()),
                "forget_loss_type": args.forget_loss_type,
                "forget_margin_target": args.forget_margin if margin_res is not None else None,
                "learning_rate": effective_lr,
                "gradient_norm_before_clip": float(grad_norm.detach().cpu()) if grad_norm is not None else None,
                "forget_loss": float(forget_loss_for_log.detach().cpu()),
                "retain_loss": float(retain_res.loss.detach().cpu()) if retain_res is not None else None,
                "kl_retain_loss": float(kl_val.detach().cpu()),
                "forget_fallback_examples": (
                    (margin_res.target_new_fallback_examples + margin_res.target_true_fallback_examples) if margin_res is not None
                    else ga_res.fallback_examples if ga_res is not None
                    else forget_res.fallback_examples
                ),
                "retain_fallback_examples": retain_res.fallback_examples if retain_res is not None else None,
                "forget_tokens": (
                    (margin_res.target_new_tokens + margin_res.target_true_tokens) if margin_res is not None
                    else ga_res.contributing_tokens if ga_res is not None
                    else forget_res.contributing_tokens
                ),
                "retain_tokens": retain_res.contributing_tokens if retain_res is not None else None,
                "forget_margin_loss": float(margin_res.loss.detach().cpu()) if margin_res is not None else None,
                "forget_target_new_nll": float(margin_res.target_new_nll.detach().cpu()) if margin_res is not None else None,
                "forget_target_true_nll": float(margin_res.target_true_nll.detach().cpu()) if margin_res is not None else None,
                "forget_target_new_fallback_examples": margin_res.target_new_fallback_examples if margin_res is not None else None,
                "forget_target_true_fallback_examples": margin_res.target_true_fallback_examples if margin_res is not None else None,
                "zerounlearn_ga_logprob_loss": float(ga_res.loss.detach().cpu()) if ga_res is not None else None,
                "zerounlearn_ga_fallback_examples": ga_res.fallback_examples if ga_res is not None else None,
            }
            log_f.write(json.dumps(row) + "\n")
    if mode == POST_TRAINING_RESTORE_MODE:
        if post_training_originals is None or post_training_groups is None:
            raise RuntimeError("Post-training restore mode is missing its base snapshot or token groups.")
        # Optimizer state is no longer needed and can be large for these matrices.
        del opt
        applied_counts = apply_post_training_row_restore(
            tied_info,
            post_training_originals,
            post_training_groups,
            true_alpha=args.post_training_true_alpha,
            new_true_alpha=args.post_training_new_true_alpha,
            new_retain_alpha=args.post_training_new_retain_alpha,
            new_true_retain_alpha=args.post_training_new_true_retain_alpha,
        )
        write_json(
            mode_dir / "post_training_row_policy.json",
            post_training_policy_report(
                tok,
                post_training_groups,
                applied_counts,
                true_alpha=args.post_training_true_alpha,
                new_true_alpha=args.post_training_new_true_alpha,
                new_retain_alpha=args.post_training_new_retain_alpha,
                new_true_retain_alpha=args.post_training_new_true_retain_alpha,
            ),
        )
        print("Applied post-training restoration to embedding/lm_head vocabulary rows")
    if args.save_model:
        ckpt_dir = mode_dir / "checkpoint"
        model.save_pretrained(ckpt_dir)
        tok.save_pretrained(ckpt_dir)
    del ref_model
    return summary


@torch.no_grad()
def greedy_match(model: torch.nn.Module, tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> float:
    if not examples:
        return float("nan")
    matches = 0
    for ex in tqdm(examples, desc="greedy", leave=False):
        prompt_ids = tok(ex.prompt, return_tensors="pt", add_special_tokens=False).to(device)
        answer_ids = token_ids_for_text(tok, answer_with_eos(tok, ex.answer))
        if not answer_ids:
            continue
        max_new = min(len(answer_ids), 32)
        out = model.generate(
            **prompt_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        generated = out[0, prompt_ids["input_ids"].shape[1] :].detach().cpu().tolist()
        if generated[:max_new] == answer_ids[:max_new]:
            matches += 1
    return matches / len(examples)


def nll_values(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    examples: Sequence[Example],
    device: torch.device,
    desc: str = "nll",
    append_eos: bool = True,
) -> List[float]:
    losses: List[float] = []
    model.eval()
    with torch.no_grad():
        for ex in tqdm(examples, desc=desc, leave=False):
            res = answer_ce_loss(model, tok, [ex], None, device, append_eos=append_eos)
            losses.append(float(res.loss.detach().cpu()))
    return losses


def mean_nll(
    model: torch.nn.Module,
    tok: AutoTokenizer,
    examples: Sequence[Example],
    device: torch.device,
    *,
    append_eos: bool = True,
) -> float:
    if not examples:
        return float("nan")
    return float(
        np.mean(
            nll_values(
                model,
                tok,
                examples,
                device,
                append_eos=append_eos,
            )
        )
    )


def mcf_target_examples(examples: Sequence[Example], field: str) -> List[Example]:
    return [
        Example(prompt=e.prompt, answer=getattr(e, field), source=e.source)
        for e in examples
        if getattr(e, field)
    ]


def mcf_paired_examples(examples: Sequence[Example], use_paraphrases: bool = False) -> Tuple[List[Example], List[Example]]:
    new_examples: List[Example] = []
    true_examples: List[Example] = []
    for e in examples:
        if not e.target_new or not e.target_true:
            continue
        prompts = list(e.paraphrase_prompts or []) if use_paraphrases else [e.prompt]
        for prompt in prompts:
            new_examples.append(Example(prompt=prompt, answer=e.target_new, source=e.source))
            true_examples.append(Example(prompt=prompt, answer=e.target_true, source=e.source))
    return new_examples, true_examples


def add_mcf_rewrite_metrics(metrics: Dict[str, float], prefix: str, model: torch.nn.Module, tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> None:
    new_examples, true_examples = mcf_paired_examples(examples, use_paraphrases=False)
    if not new_examples or len(new_examples) != len(true_examples):
        for suffix in ("target_new_nll", "target_true_nll", "new_over_true_success"):
            metrics[f"{prefix}_{suffix}"] = float("nan")
        for suffix in ("target_new_nll", "target_true_nll", "new_over_true_success", "prob_diff_new_minus_true"):
            metrics[f"{prefix}_rewrite_{suffix}"] = float("nan")
        return
    new_nll = np.array(
        nll_values(
            model,
            tok,
            new_examples,
            device,
            desc=f"{prefix} new nll",
            append_eos=False,
        ),
        dtype=np.float64,
    )
    true_nll = np.array(
        nll_values(
            model,
            tok,
            true_examples,
            device,
            desc=f"{prefix} true nll",
            append_eos=False,
        ),
        dtype=np.float64,
    )
    target_new_nll = float(np.mean(new_nll))
    target_true_nll = float(np.mean(true_nll))
    new_over_true_success = float(np.mean(new_nll < true_nll))
    prob_diff = float(np.mean(np.exp(-new_nll) - np.exp(-true_nll)))
    metrics[f"{prefix}_target_new_nll"] = target_new_nll
    metrics[f"{prefix}_target_true_nll"] = target_true_nll
    metrics[f"{prefix}_new_over_true_success"] = new_over_true_success
    metrics[f"{prefix}_rewrite_target_new_nll"] = target_new_nll
    metrics[f"{prefix}_rewrite_target_true_nll"] = target_true_nll
    metrics[f"{prefix}_rewrite_new_over_true_success"] = new_over_true_success
    metrics[f"{prefix}_rewrite_prob_diff_new_minus_true"] = prob_diff


def add_mcf_paraphrase_metrics(metrics: Dict[str, float], prefix: str, model: torch.nn.Module, tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> None:
    new_examples, true_examples = mcf_paired_examples(examples, use_paraphrases=True)
    if not new_examples or len(new_examples) != len(true_examples):
        return
    new_nll = np.array(
        nll_values(
            model,
            tok,
            new_examples,
            device,
            desc=f"{prefix} paraphrase new nll",
            append_eos=False,
        ),
        dtype=np.float64,
    )
    true_nll = np.array(
        nll_values(
            model,
            tok,
            true_examples,
            device,
            desc=f"{prefix} paraphrase true nll",
            append_eos=False,
        ),
        dtype=np.float64,
    )
    metrics[f"{prefix}_paraphrase_new_over_true_success"] = float(np.mean(new_nll < true_nll))


def evaluate(model: torch.nn.Module, tok: AutoTokenizer, forget: Sequence[Example], retain: Sequence[Example], args: argparse.Namespace) -> Dict[str, float]:
    device = first_device(model)
    max_eval = args.max_eval_examples
    f_eval = list(forget[:max_eval]) if max_eval else list(forget)
    r_eval = list(retain[:max_eval]) if max_eval else list(retain)
    metrics: Dict[str, float] = {}
    if args.dataset == "mcf":
        add_mcf_rewrite_metrics(metrics, "forget", model, tok, f_eval, device)
        add_mcf_rewrite_metrics(metrics, "retain", model, tok, r_eval, device)
        add_mcf_paraphrase_metrics(metrics, "forget", model, tok, f_eval, device)
        add_mcf_paraphrase_metrics(metrics, "retain", model, tok, r_eval, device)
        selected_field = args.mcf_answer_field
        metrics["forget_match"] = greedy_match(model, tok, f_eval, device)
        metrics["retain_match"] = greedy_match(model, tok, r_eval, device)
        metrics["forget_loss"] = metrics.get(f"forget_{selected_field}_nll", float("nan"))
        metrics["retain_loss"] = metrics.get(f"retain_{selected_field}_nll", float("nan"))
    else:
        append_eos = append_eos_for_dataset(args.dataset)
        metrics["forget_answer_nll"] = mean_nll(
            model,
            tok,
            f_eval,
            device,
            append_eos=append_eos,
        )
        metrics["retain_answer_nll"] = mean_nll(
            model,
            tok,
            r_eval,
            device,
            append_eos=append_eos,
        )
        metrics["forget_match"] = greedy_match(model, tok, f_eval, device)
        metrics["retain_match"] = greedy_match(model, tok, r_eval, device)
        metrics["forget_loss"] = metrics["forget_answer_nll"]
        metrics["retain_loss"] = metrics["retain_answer_nll"]
    return metrics


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_comparison_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if pd is not None:
        pd.DataFrame(rows).to_csv(path, index=False)
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_md(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = list(rows[0].keys()) if rows else []
    lines = [
        "# GA/GD Comparison",
        "",
        "Training-diagnostic GA/GD metrics only: higher `forget_loss_after` means stronger forgetting, while lower `retain_loss_after` means better retention.",
        "For `zerounlearn_ga`, lower `forget_loss_after` means stronger GA because the value is target_true log-probability.",
        "For official-style MCF success metrics, lower `forget_new_over_true_success_after` means stronger unlearning in the ZeroUnlearn/CounterFact target_new-vs-target_true sense.",
        "",
    ]
    if rows:
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_comparison_row(
    args: argparse.Namespace,
    mode: str,
    before: Dict[str, float],
    after: Dict[str, float],
    summary: ParamSummary,
    n_selected: int,
) -> Dict[str, Any]:
    row = {
        "dataset": args.dataset,
        "mode": mode,
        "seed": args.seed,
        "forget_loss_before": before["forget_loss"],
        "forget_loss_after": after["forget_loss"],
        "retain_loss_before": before["retain_loss"],
        "retain_loss_after": after["retain_loss"],
        "forget_loss_delta": after["forget_loss"] - before["forget_loss"],
        "retain_loss_delta": after["retain_loss"] - before["retain_loss"],
        "forget_match_before": before["forget_match"],
        "forget_match_after": after["forget_match"],
        "retain_match_before": before["retain_match"],
        "retain_match_after": after["retain_match"],
        "n_trainable_params": summary.n_trainable_params,
        "trainable_param_percent": summary.trainable_param_percent,
        "n_selected_tokens": n_selected,
        "steps": args.steps,
        "lr": getattr(args, "effective_lr", learning_rate_for_mode(mode, args)),
        "optimizer": getattr(args, "effective_optimizer", args.optimizer or "default"),
        "forget_loss_type": args.forget_loss_type,
        "forget_margin": args.forget_margin,
        "sampling_strategy": args.sampling_strategy,
        "post_training_true_alpha": (
            args.post_training_true_alpha
            if mode == POST_TRAINING_RESTORE_MODE
            else None
        ),
        "post_training_new_true_alpha": (
            args.post_training_new_true_alpha
            if mode == POST_TRAINING_RESTORE_MODE
            else None
        ),
        "post_training_new_retain_alpha": (
            args.post_training_new_retain_alpha
            if mode == POST_TRAINING_RESTORE_MODE
            else None
        ),
        "post_training_new_true_retain_alpha": (
            args.post_training_new_true_retain_alpha
            if mode == POST_TRAINING_RESTORE_MODE
            else None
        ),
    }
    if args.dataset == "mcf":
        official_keys = [
            "forget_target_new_nll",
            "forget_target_true_nll",
            "forget_new_over_true_success",
            "retain_target_new_nll",
            "retain_target_true_nll",
            "retain_new_over_true_success",
        ]
        for key in official_keys:
            if key in before or key in after:
                row[f"{key}_before"] = before.get(key, float("nan"))
                row[f"{key}_after"] = after.get(key, float("nan"))
    return row


def add_mcf_before_after_metrics(metrics: Dict[str, Any], before: Dict[str, float], after: Dict[str, float]) -> None:
    for key in (
        "forget_target_new_nll",
        "forget_target_true_nll",
        "forget_new_over_true_success",
        "retain_target_new_nll",
        "retain_target_true_nll",
        "retain_new_over_true_success",
    ):
        metrics[f"{key}_before"] = before.get(key, float("nan"))
        metrics[f"{key}_after"] = after.get(key, float("nan"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["mcf", "tofu"], required=True)
    p.add_argument("--model-path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    p.add_argument("--output-dir", default="outputs/gagd_compare", help="Output directory. Relative paths are resolved under semantic-unlearning/.")
    p.add_argument("--mode", choices=MODES + ["all"], default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--max-eval-examples", type=int, default=None)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--retain-batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--full-lr", type=float, default=None, help="Learning rate for full_* modes. Overrides --lr only for those modes.")
    p.add_argument("--emb-lm-lr", type=float, default=None, help="Learning rate for emb_lm_* modes. Overrides --lr only for those modes.")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--retain-weight", type=float, default=1.0)
    p.add_argument("--forget-loss-type", choices=["answer_nll", "mcf_margin", "zerounlearn_ga"], default="answer_nll", help="Forget objective. answer_nll keeps GA on selected answer NLL; mcf_margin minimizes softplus(target_true_nll - target_new_nll + forget_margin); zerounlearn_ga minimizes target_true log-probability like the official GA baseline. Non-answer_nll objectives are MCF-only.")
    p.add_argument("--forget-margin", type=float, default=0.0, help="For mcf_margin, require target_new_nll >= target_true_nll + this margin. Positive values produce more robust forgetting.")
    p.add_argument("--kl-retain-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--optimizer", choices=["sgd", "adam", "adamw", "adamw8bit"], default=None, help="Optimizer override. Defaults to sgd for full_* modes and adamw for emb_lm_* modes.")
    p.add_argument("--full-optimizer", choices=["sgd", "adam", "adamw", "adamw8bit"], default=None, help="Optimizer for full_* modes unless --optimizer is set.")
    p.add_argument("--emb-lm-optimizer", choices=["sgd", "adam", "adamw", "adamw8bit"], default=None, help="Optimizer for emb_lm_* modes unless --optimizer is set.")
    p.add_argument("--sampling-strategy", choices=["epoch", "with_replacement"], default="epoch", help="epoch guarantees full shuffled dataset coverage; with_replacement preserves legacy random sampling.")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--mcf-url", default=MCF_URL)
    p.add_argument("--mcf-cache-path", default="data/mcf/multi_counterfact.json", help="MCF cache path. Relative paths are resolved under semantic-unlearning/.")
    p.add_argument(
        "--mcf-sample-mode",
        choices=["official", "first", "shuffled"],
        default="official",
        help=(
            "MCF training split. official matches JSON-LMHead-Zero and ZeroUnlearn: "
            "forget from the second half, retain from the first half, sampled with --seed. "
            "shuffled preserves the legacy whole-dataset shuffle."
        ),
    )
    p.add_argument("--mcf-answer-field", choices=["target_new", "target_true"], default="target_new", help="MCF answer field used for GA/GD training and generic forget/retain loss metrics. Defaults to target_new for ZeroUnlearn/CounterFact comparability; target_true is diagnostic.")
    p.add_argument("--run-official-mcf-eval", action="store_true", help="After each MCF mode, run the ZeroUnlearn-compatible evaluator on the trained model in memory. Use official_eval_comparison.md for final comparisons; pass --save-model only when checkpoints are also needed.")
    p.add_argument("--wikidata-dir", default="data/wikidata", help="Wikidata dataset directory for official-compatible PPL evaluation.")
    p.add_argument("--official-sample-mode", choices=["official", "first"], default="official", help="MCF sampling mode used by --run-official-mcf-eval.")
    p.add_argument("--official-device-map", default="auto", help="Deprecated compatibility option; in-memory official evaluation keeps the trained model on its current device.")
    p.add_argument("--skip-ppl", action="store_true", help="Skip official-compatible PPL evaluation.")
    p.add_argument("--forget-split", default="forget05")
    p.add_argument("--retain-split", default="retain95")
    p.add_argument("--semantic-token-json", default=None)
    p.add_argument("--selective-top-k", type=int, default=1000)
    p.add_argument(
        "--post-training-true-alpha",
        type=float,
        default=POST_TRAINING_TRUE_ALPHA,
        help=(
            "For the fifth MCF setting, retain this fraction of the learned "
            "update on unique target-true rows instead of a flat base scale."
        ),
    )
    p.add_argument(
        "--post-training-new-true-alpha",
        type=float,
        default=POST_TRAINING_NEW_TRUE_ALPHA,
        help=(
            "For the fifth MCF setting, retain this fraction of the learned "
            "update on target-new/target-true overlap rows."
        ),
    )
    p.add_argument(
        "--post-training-new-retain-alpha",
        type=float,
        default=POST_TRAINING_NEW_RETAIN_ALPHA,
        help=(
            "For the fifth MCF setting, retain this fraction of the learned "
            "update on target-new/retain overlap rows."
        ),
    )
    p.add_argument(
        "--post-training-new-true-retain-alpha",
        type=float,
        default=POST_TRAINING_NEW_TRUE_RETAIN_ALPHA,
        help=(
            "For the fifth MCF setting, retain this fraction of the learned "
            "update on rows shared by target-new, target-true, and retain."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.forget_loss_type in {"mcf_margin", "zerounlearn_ga"} and args.dataset != "mcf":
        raise ValueError(f"--forget-loss-type {args.forget_loss_type} is only supported with --dataset mcf")
    if args.forget_loss_type == "zerounlearn_ga":
        print("ZeroUnlearn GA uses target_true log-probability minimization.")
    if args.steps <= 0 or args.batch_size <= 0 or args.retain_batch_size <= 0:
        raise ValueError("--steps, --batch-size, and --retain-batch-size must be positive")
    if args.forget_margin < 0:
        raise ValueError("--forget-margin must be non-negative")
    if args.forget_weight <= 0 or args.retain_weight < 0 or args.kl_retain_weight < 0:
        raise ValueError("--forget-weight must be positive; retain weights must be non-negative")
    for option_name in (
        "post_training_true_alpha",
        "post_training_new_true_alpha",
        "post_training_new_retain_alpha",
        "post_training_new_true_retain_alpha",
    ):
        alpha = getattr(args, option_name)
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"--{option_name.replace('_', '-')} must be between 0 and 1")
    if args.mode == POST_TRAINING_RESTORE_MODE and args.dataset != "mcf":
        raise ValueError(
            f"--mode {POST_TRAINING_RESTORE_MODE} requires --dataset mcf "
            "because TOFU has no target_new/target_true row groups."
        )
    modes = (
        (MODES if args.dataset == "mcf" else BASE_MODES)
        if args.mode == "all"
        else [args.mode]
    )
    for mode in modes:
        if learning_rate_for_mode(mode, args) <= 0:
            raise ValueError(f"Learning rate for {mode} must be positive")
    set_seed(args.seed)
    require_cuda_if_needed(args.device_map)
    out_dir = resolve_output_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.run_official_mcf_eval:
        if args.dataset != "mcf":
            raise ValueError("--run-official-mcf-eval is only supported with --dataset mcf")
    config_used = dict(vars(args))
    config_used["resolved_mode_settings"] = {
        mode: {
            "learning_rate": learning_rate_for_mode(mode, args),
            "optimizer": optimizer_name_for_mode(mode, args),
        }
        for mode in modes
    }
    write_json(out_dir / "config_used.json", config_used)

    print("Loading dataset")
    forget, retain = load_data(args)
    if not forget or not retain:
        raise ValueError("Both forget and retain splits must contain at least one example.")

    print("Loading base model for token selection and base metrics")
    base_model, tok = load_model_and_tokenizer(args, for_training=False)
    selected_ids = select_tokens(tok, forget, retain, args)
    if not selected_ids and (args.mode == "all" or "selective_tokens" in args.mode):
        raise ValueError("Selective-token modes require at least one non-special selected token, but token selection returned none.")
    write_json(out_dir / "selected_token_ids.json", {"token_ids": selected_ids, "n_selected_tokens": len(selected_ids)})
    print(f"Selected {len(selected_ids)} tokens")

    base_metrics = evaluate(base_model, tok, forget, retain, args)
    write_json(out_dir / "base_metrics.json", base_metrics)
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    rows: List[Dict[str, Any]] = []
    official_rows: List[Dict[str, Any]] = []
    official_dir = out_dir / "official_eval"
    for mode in modes:
        print(f"\n=== Running mode: {mode} ===")
        set_seed(args.seed)
        model, tok = load_model_and_tokenizer(args, for_training=True)
        mode_dir = out_dir / mode
        summary = train_mode(model, tok, forget, retain, selected_ids, mode, args, mode_dir)
        metrics = evaluate(model, tok, forget, retain, args)
        if args.dataset == "mcf":
            add_mcf_before_after_metrics(metrics, base_metrics, metrics)
        metrics.update(asdict(summary))
        metrics["n_selected_tokens"] = len(selected_ids)
        metrics["forget_loss_type"] = args.forget_loss_type
        metrics["forget_margin"] = args.forget_margin
        metrics["sampling_strategy"] = args.sampling_strategy
        metrics["learning_rate"] = args.effective_lr
        metrics["optimizer"] = args.effective_optimizer
        if mode == POST_TRAINING_RESTORE_MODE:
            metrics["post_training_true_alpha"] = args.post_training_true_alpha
            metrics["post_training_overlap_alphas"] = {
                "target_new_true": args.post_training_new_true_alpha,
                "target_new_retain": args.post_training_new_retain_alpha,
                "target_new_true_retain": (
                    args.post_training_new_true_retain_alpha
                ),
            }
        write_json(mode_dir / "metrics.json", metrics)
        rows.append(make_comparison_row(args, mode, base_metrics, metrics, summary, len(selected_ids)))
        if args.run_official_mcf_eval:
            official_model_ref = mode_dir / "checkpoint" if args.save_model else f"in-memory:{mode}"
            official_result = evaluate_loaded_model_official(
                method=mode,
                model=model,
                tok=tok,
                model_dir=official_model_ref,
                mcf_path=resolve_output_path(args.mcf_cache_path),
                wikidata_dir=resolve_output_path(args.wikidata_dir),
                out_path=official_dir / f"{mode}_official_eval.json",
                unlearn_num=args.forget_num,
                retain_num=args.retain_num,
                seed=args.seed,
                sample_mode=args.official_sample_mode,
                skip_ppl=args.skip_ppl,
            )
            official_rows.append(result_to_comparison_row(official_result))
            del model
        else:
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_comparison_csv(out_dir / "comparison.csv", rows)
    write_comparison_md(out_dir / "comparison.md", rows)
    if official_rows:
        write_official_comparison(out_dir, official_rows)
        write_official_comparison(official_dir, official_rows)
    print(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
