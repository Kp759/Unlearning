#!/usr/bin/env python3
"""Unified GA/GD comparison for semantic unlearning on MCF and TOFU.

This script is intentionally self-contained and avoids optional editing baselines.
It compares full-model and embedding/lm_head-only GA/GD with all-token or
selective-token answer-only losses.
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
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # Optional by requirement.
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - depends on environment
    pd = None

MODES = [
    "full_all_tokens",
    "full_selective_tokens",
    "emb_lm_all_tokens",
    "emb_lm_selective_tokens",
]
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
class ParamSummary:
    n_trainable_params: int
    n_total_params: int
    trainable_param_percent: float
    trainable_names: List[str]


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


def load_mcf(args: argparse.Namespace) -> Tuple[List[Example], List[Example]]:
    cache_path = resolve_output_path(args.mcf_cache_path)
    download_if_missing(args.mcf_url, cache_path)
    with cache_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("MCF JSON must be a list of records.")
    records: List[Example] = []
    for rec in raw:
        rr, paraphrases = extract_mcf_rewrite(rec)
        subject = str(rr["subject"])
        prompt = format_mcf_prompt(str(rr["prompt"]), subject)
        target_new = str(rr["target_new"]["str"])
        target_true = ""
        if isinstance(rr.get("target_true"), dict):
            target_true = str(rr["target_true"].get("str", ""))
        records.append(
            Example(
                prompt=prompt,
                answer=normalize_answer(target_new),
                subject=subject,
                target_new=normalize_answer(target_new),
                target_true=normalize_answer(target_true) if target_true else "",
                paraphrase_prompts=[format_mcf_prompt(p, subject) for p in paraphrases],
                source="mcf",
            )
        )
    rng = random.Random(args.seed)
    rng.shuffle(records)
    need = args.forget_num + args.retain_num
    if len(records) < need:
        raise ValueError(f"MCF has only {len(records)} records, need forget_num + retain_num = {need}.")
    return records[: args.forget_num], records[args.forget_num : need]


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


def build_batch(tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> Dict[str, torch.Tensor]:
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    rows: List[List[int]] = []
    labels: List[List[int]] = []
    for ex in examples:
        prompt_ids = token_ids_for_text(tok, ex.prompt)
        answer_ids = token_ids_for_text(tok, answer_with_eos(tok, ex.answer))
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
) -> LossResult:
    batch = build_batch(tok, examples, device)
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


def make_optimizer(params: Sequence[torch.nn.Parameter], mode: str, args: argparse.Namespace) -> Tuple[torch.optim.Optimizer, str]:
    opt_name = args.optimizer
    if opt_name == "auto":
        opt_name = "sgd" if mode.startswith("full_") else "adamw"
    if opt_name == "sgd":
        return torch.optim.SGD(params, lr=args.lr), opt_name
    if opt_name == "adamw":
        return torch.optim.AdamW(params, lr=args.lr), opt_name
    if opt_name == "adamw8bit":
        try:
            import bitsandbytes as bnb  # type: ignore

            return bnb.optim.AdamW8bit(params, lr=args.lr), opt_name
        except Exception as exc:
            print(f"WARNING: bitsandbytes AdamW8bit unavailable ({exc}); falling back to torch.optim.AdamW")
            return torch.optim.AdamW(params, lr=args.lr), "adamw"
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


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


def kl_retain_loss(model: torch.nn.Module, ref_model: torch.nn.Module, tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> torch.Tensor:
    batch = build_batch(tok, examples, device)
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
    opt, effective_optimizer = make_optimizer(params, mode, args)
    args.effective_optimizer = effective_optimizer
    print(f"Optimizer for {mode}: {effective_optimizer}")
    device = first_device(model)
    originals = None
    if mode == "emb_lm_selective_tokens":
        originals = apply_row_mask_and_restore(tied_info, selected_ids)
        print(f"Embedding/lm_head selected rows allowed to update: {len(selected_ids)}")
    ref_model = None
    if args.kl_retain_weight > 0:
        print("Loading frozen reference model for retain KL regularization")
        ref_model, _ = load_model_and_tokenizer(args, for_training=False)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad_(False)
    model.train()
    mode_dir.mkdir(parents=True, exist_ok=True)
    with (mode_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
        for step in tqdm(range(args.steps), desc=f"train {mode}"):
            forget_batch = sample_batch(forget, args.batch_size)
            retain_batch = sample_batch(retain, args.retain_batch_size)
            opt.zero_grad(set_to_none=True)
            forget_res = answer_ce_loss(model, tok, forget_batch, selected_set, device)
            retain_res = answer_ce_loss(model, tok, retain_batch, selected_set, device)
            total = -args.forget_weight * forget_res.loss + args.retain_weight * retain_res.loss
            kl_val = torch.zeros((), device=device)
            if ref_model is not None:
                kl_val = kl_retain_loss(model, ref_model, tok, retain_batch, device)
                total = total + args.kl_retain_weight * kl_val
            total.backward()
            if mode == "emb_lm_selective_tokens" and originals is not None:
                apply_row_mask_and_restore(tied_info, selected_ids, originals)
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            if mode == "emb_lm_selective_tokens" and originals is not None:
                restore_non_selected_rows(tied_info, selected_ids, originals)
            row = {
                "step": step + 1,
                "total_loss": float(total.detach().cpu()),
                "forget_loss": float(forget_res.loss.detach().cpu()),
                "retain_loss": float(retain_res.loss.detach().cpu()),
                "kl_retain_loss": float(kl_val.detach().cpu()),
                "forget_fallback_examples": forget_res.fallback_examples,
                "retain_fallback_examples": retain_res.fallback_examples,
                "forget_tokens": forget_res.contributing_tokens,
                "retain_tokens": retain_res.contributing_tokens,
            }
            log_f.write(json.dumps(row) + "\n")
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


def mean_nll(model: torch.nn.Module, tok: AutoTokenizer, examples: Sequence[Example], device: torch.device) -> float:
    if not examples:
        return float("nan")
    losses = []
    model.eval()
    with torch.no_grad():
        for ex in tqdm(examples, desc="nll", leave=False):
            res = answer_ce_loss(model, tok, [ex], None, device)
            losses.append(float(res.loss.detach().cpu()))
    return float(np.mean(losses))


def mcf_eval_sets(forget: Sequence[Example], retain: Sequence[Example]) -> Dict[str, List[Example]]:
    sets: Dict[str, List[Example]] = {
        "forget_target_new": list(forget),
        "retain_target_new": list(retain),
    }
    true_examples = [Example(prompt=e.prompt, answer=e.target_true, source=e.source) for e in forget if e.target_true]
    if true_examples:
        sets["forget_target_true"] = true_examples
    paras: List[Example] = []
    for e in forget:
        for p in e.paraphrase_prompts or []:
            paras.append(Example(prompt=p, answer=e.target_new, source=e.source))
    if paras:
        sets["paraphrase_target_new"] = paras
    return sets


def evaluate(model: torch.nn.Module, tok: AutoTokenizer, forget: Sequence[Example], retain: Sequence[Example], args: argparse.Namespace) -> Dict[str, float]:
    device = first_device(model)
    max_eval = args.max_eval_examples
    f_eval = list(forget[:max_eval]) if max_eval else list(forget)
    r_eval = list(retain[:max_eval]) if max_eval else list(retain)
    metrics: Dict[str, float] = {}
    if args.dataset == "mcf":
        for name, examples in mcf_eval_sets(f_eval, r_eval).items():
            metrics[f"{name}_nll"] = mean_nll(model, tok, examples, device)
        metrics["forget_match"] = greedy_match(model, tok, f_eval, device)
        metrics["retain_match"] = greedy_match(model, tok, r_eval, device)
        metrics["forget_loss"] = metrics.get("forget_target_new_nll", float("nan"))
        metrics["retain_loss"] = metrics.get("retain_target_new_nll", float("nan"))
    else:
        metrics["forget_answer_nll"] = mean_nll(model, tok, f_eval, device)
        metrics["retain_answer_nll"] = mean_nll(model, tok, r_eval, device)
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
        "Metric directions: lower `forget_match_after` and higher `forget_loss_after` indicate stronger forgetting. Lower `retain_loss_after` and higher `retain_match_after` indicate better retention.",
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
    return {
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
        "lr": args.lr,
        "optimizer": getattr(args, "effective_optimizer", args.optimizer),
    }


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
    p.add_argument("--forget-weight", type=float, default=1.0)
    p.add_argument("--retain-weight", type=float, default=1.0)
    p.add_argument("--kl-retain-weight", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--optimizer", choices=["auto", "sgd", "adamw", "adamw8bit"], default="auto")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--device-map", choices=["single", "auto"], default="single")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--mcf-url", default=MCF_URL)
    p.add_argument("--mcf-cache-path", default="data/mcf/multi_counterfact.json", help="MCF cache path. Relative paths are resolved under semantic-unlearning/.")
    p.add_argument("--forget-split", default="forget05")
    p.add_argument("--retain-split", default="retain95")
    p.add_argument("--semantic-token-json", default=None)
    p.add_argument("--selective-top-k", type=int, default=1000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    require_cuda_if_needed(args.device_map)
    out_dir = resolve_output_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "config_used.json", vars(args))

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

    modes = MODES if args.mode == "all" else [args.mode]
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        print(f"\n=== Running mode: {mode} ===")
        set_seed(args.seed)
        model, tok = load_model_and_tokenizer(args, for_training=True)
        mode_dir = out_dir / mode
        summary = train_mode(model, tok, forget, retain, selected_ids, mode, args, mode_dir)
        metrics = evaluate(model, tok, forget, retain, args)
        metrics.update(asdict(summary))
        metrics["n_selected_tokens"] = len(selected_ids)
        write_json(mode_dir / "metrics.json", metrics)
        rows.append(make_comparison_row(args, mode, base_metrics, metrics, summary, len(selected_ids)))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_comparison_csv(out_dir / "comparison.csv", rows)
    write_comparison_md(out_dir / "comparison.md", rows)
    print(f"Done. Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
