#!/usr/bin/env python3
"""
scripts/train_dual_json_rc_gagd_embed_lmhead.py

Dual-JSON + retain-count GA/GD training for BOTH input embeddings and lm_head.

Core idea:
  GA tokens: forget-only / low-retain-overlap tokens.
      -> maximize CE on forget answers for those target tokens.

  GD tokens: retain / overlap tokens.
      -> minimize CE on retain answers for those target tokens.
      -> optionally preserve original model distribution on retain via KL.

Only selected rows in input embeddings and lm_head are trainable.
All transformer block weights are frozen.

Recommended first run uses small steps and strong retain KL.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------
# Utilities
# ------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def force_untie_lm_head(model) -> None:
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None or not hasattr(out, "weight"):
        raise RuntimeError("No output embeddings/lm_head found.")

    if emb.weight.data_ptr() != out.weight.data_ptr():
        print("[GA/GD] lm_head already untied.")
        return

    print("[GA/GD] lm_head tied. Untying before joint embedding+lm_head training.")
    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False).to(device=old.device, dtype=old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device


# ------------------------------
# Dataset / collation
# ------------------------------

def make_prompt_answer(row: Dict[str, Any]) -> Tuple[str, str]:
    prompt = f"Question: {row['question']}\nAnswer:"
    answer = " " + str(row["answer"]).strip()
    return prompt, answer


def build_lm_example(tokenizer, row: Dict[str, Any], max_length: int) -> Dict[str, List[int]]:
    prompt, answer = make_prompt_answer(row)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [int(tokenizer.eos_token_id)]

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids[:]

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class QADataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, tokenizer, max_length: int):
        self.examples = [build_lm_example(tokenizer, row, max_length) for row in hf_ds]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return self.examples[idx]


def make_collate(tokenizer):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate(batch: Sequence[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        input_ids, attention_mask, labels = [], [], []
        for x in batch:
            n = len(x["input_ids"])
            pad = max_len - n
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attention_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def cycle_loader(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


# ------------------------------
# Losses
# ------------------------------

def label_mask_for_ids(labels: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """labels: [B,T]. Return bool mask where labels are in token_ids and labels != -100."""
    valid = labels.ne(-100)
    safe = labels.clamp_min(0)
    if token_ids.numel() == 0:
        return torch.zeros_like(labels, dtype=torch.bool)
    # torch.isin works on recent torch; fallback not needed on current HPC envs usually.
    return valid & torch.isin(safe, token_ids)


def masked_token_ce(logits: torch.Tensor, labels: torch.Tensor, target_ids: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """CE over positions whose next-token label is in target_ids."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = label_mask_for_ids(shift_labels, target_ids)

    n = int(mask.sum().item())
    if n == 0:
        return shift_logits.new_tensor(0.0), 0

    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(shift_labels)

    loss = (ce * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
    return loss, n


def retain_kl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """KL teacher||student on answer positions only."""
    s = student_logits[:, :-1, :].float() / temperature
    t = teacher_logits[:, :-1, :].float() / temperature
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)

    log_s = F.log_softmax(s, dim=-1)
    prob_t = F.softmax(t, dim=-1)
    kl = F.kl_div(log_s, prob_t, reduction="none").sum(dim=-1) * (temperature ** 2)

    if mask.sum().item() == 0:
        return kl.new_tensor(0.0)
    return (kl * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


# ------------------------------
# Row masking / regularization
# ------------------------------

def make_row_mask(vocab_size: int, ids: Sequence[int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.zeros(vocab_size, 1, device=device, dtype=dtype)
    valid = [int(x) for x in ids if 0 <= int(x) < vocab_size]
    if valid:
        mask[torch.tensor(sorted(set(valid)), device=device, dtype=torch.long)] = 1
    return mask


def register_row_mask(param: torch.nn.Parameter, editable_ids: Sequence[int], name: str):
    vocab_size = param.shape[0]
    # Keep mask on same device/dtype as grad when hook runs.
    ids = sorted(set(int(x) for x in editable_ids if 0 <= int(x) < vocab_size))
    print(f"[GA/GD] {name}: editable rows = {len(ids)}")

    def hook(grad: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros(grad.shape[0], 1, device=grad.device, dtype=grad.dtype)
        if ids:
            mask[torch.tensor(ids, device=grad.device, dtype=torch.long)] = 1
        return grad * mask

    return param.register_hook(hook)


def row_anchor_loss(weight: torch.Tensor, original_rows: torch.Tensor, row_ids: torch.Tensor) -> torch.Tensor:
    if row_ids.numel() == 0:
        return weight.new_tensor(0.0)
    current = weight.index_select(0, row_ids)
    return F.mse_loss(current.float(), original_rows.float())


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--tokens-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--teacher-model-name", default=None, help="Original/teacher model for retain KL")
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)

    ap.add_argument("--dtype", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    ap.add_argument("--grad-clip", type=float, default=1.0)

    # Loss weights.
    ap.add_argument("--forget-weight", type=float, default=1.0)
    ap.add_argument("--retain-weight", type=float, default=1.0)
    ap.add_argument("--kl-weight", type=float, default=0.5)
    ap.add_argument("--kl-temperature", type=float, default=1.0)
    ap.add_argument("--anchor-weight", type=float, default=0.02,
                    help="L2 anchor on edited embedding/lm_head rows to avoid drift.")

    # Which rows to train.
    ap.add_argument("--train-input-embeddings", action="store_true", default=True)
    ap.add_argument("--train-lm-head", action="store_true", default=True)
    ap.add_argument("--no-train-input-embeddings", dest="train_input_embeddings", action="store_false")
    ap.add_argument("--no-train-lm-head", dest="train_lm_head", action="store_false")

    # Teacher can be expensive; keep enabled by default because retain recovery needs it.
    ap.add_argument("--use-teacher-kl", action="store_true", default=True)
    ap.add_argument("--no-teacher-kl", dest="use_teacher_kl", action="store_false")

    args = ap.parse_args()
    seed_everything(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_name = args.model_name or cfg["model"]["name"]
    teacher_model_name = args.teacher_model_name or cfg["model"]["name"]
    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    dtype = resolve_dtype(args.dtype or cfg["model"].get("dtype", "float16"))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    token_data = load_json(Path(args.tokens_json))
    ga_ids = [int(x) for x in token_data.get("ga_token_ids", [])]
    gd_ids = [int(x) for x in token_data.get("gd_token_ids", [])]
    editable_ids = sorted(set(ga_ids) | set(gd_ids))

    if not ga_ids:
        raise RuntimeError("No GA token IDs found in token JSON.")
    if not gd_ids:
        print("[warning] No GD token IDs found; retain preservation will rely only on KL/anchor where editable rows overlap.")

    print("=" * 80)
    print("Train Dual-JSON RC-GA/GD on embeddings + lm_head")
    print("=" * 80)
    print("model:", model_name)
    print("forget split:", forget_split)
    print("retain split:", retain_split)
    print("tokens:", args.tokens_json)
    print("GA tokens:", len(set(ga_ids)))
    print("GD tokens:", len(set(gd_ids)))
    print("editable rows:", len(editable_ids))
    print("output:", args.output_dir)
    print("device:", device, "dtype:", dtype)
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.config.use_cache = False
    if tokenizer.eos_token_id is not None:
        model.config.pad_token_id = tokenizer.eos_token_id
    force_untie_lm_head(model)
    model.train()

    teacher = None
    if args.use_teacher_kl and args.kl_weight > 0:
        print("[GA/GD] Loading teacher/original model for retain KL...")
        teacher = AutoModelForCausalLM.from_pretrained(teacher_model_name, torch_dtype=dtype).to(device)
        teacher.eval()
        teacher.config.use_cache = False
        if tokenizer.eos_token_id is not None:
            teacher.config.pad_token_id = tokenizer.eos_token_id
        for p in teacher.parameters():
            p.requires_grad_(False)

    # Freeze everything, then unfreeze only input embedding and lm_head matrices.
    for p in model.parameters():
        p.requires_grad_(False)

    emb = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    train_params = []
    hooks = []

    if args.train_input_embeddings:
        emb.weight.requires_grad_(True)
        hooks.append(register_row_mask(emb.weight, editable_ids, "input_embeddings"))
        train_params.append(emb.weight)

    if args.train_lm_head:
        lm_head.weight.requires_grad_(True)
        hooks.append(register_row_mask(lm_head.weight, editable_ids, "lm_head"))
        train_params.append(lm_head.weight)

    if not train_params:
        raise RuntimeError("No trainable parameters selected.")

    # Store original selected rows for anchor regularization.
    editable_ids_tensor = torch.tensor(editable_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        orig_emb_rows = emb.weight.index_select(0, editable_ids_tensor).detach().clone() if args.train_input_embeddings else None
        orig_head_rows = lm_head.weight.index_select(0, editable_ids_tensor).detach().clone() if args.train_lm_head else None

    if args.optimizer == "adamw":
        # AdamW gives stronger optimization but costs more memory because these matrices are large.
        optimizer = torch.optim.AdamW(train_params, lr=args.lr, weight_decay=0.0)
    else:
        optimizer = torch.optim.SGD(train_params, lr=args.lr)

    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    f_data = QADataset(forget_ds, tokenizer, args.max_length)
    r_data = QADataset(retain_ds, tokenizer, args.max_length)
    collate = make_collate(tokenizer)

    f_loader = DataLoader(f_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=False)
    r_loader = DataLoader(r_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=False)
    f_iter = cycle_loader(f_loader)
    r_iter = cycle_loader(r_loader)

    ga_ids_tensor = torch.tensor(sorted(set(ga_ids)), dtype=torch.long, device=device)
    gd_ids_tensor = torch.tensor(sorted(set(gd_ids)), dtype=torch.long, device=device)

    logs: List[Dict[str, Any]] = []

    pbar = tqdm(range(1, args.steps + 1), desc="GA/GD train")
    for step in pbar:
        fb = {k: v.to(device) for k, v in next(f_iter).items()}
        rb = {k: v.to(device) for k, v in next(r_iter).items()}

        optimizer.zero_grad(set_to_none=True)

        # Forget GA: maximize CE on forget answer positions whose labels are GA tokens.
        f_out = model(input_ids=fb["input_ids"], attention_mask=fb["attention_mask"])
        f_ce, f_n = masked_token_ce(f_out.logits, fb["labels"], ga_ids_tensor)

        # Retain GD: minimize CE on retain answer positions whose labels are GD tokens.
        r_out = model(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"])
        r_ce, r_n = masked_token_ce(r_out.logits, rb["labels"], gd_ids_tensor)

        loss = (-args.forget_weight * f_ce) + (args.retain_weight * r_ce)

        kl = torch.tensor(0.0, device=device)
        if teacher is not None and args.kl_weight > 0:
            with torch.no_grad():
                t_out = teacher(input_ids=rb["input_ids"], attention_mask=rb["attention_mask"])
            kl = retain_kl_loss(r_out.logits, t_out.logits, rb["labels"], temperature=args.kl_temperature)
            loss = loss + args.kl_weight * kl

        anchor = torch.tensor(0.0, device=device)
        if args.anchor_weight > 0 and editable_ids_tensor.numel() > 0:
            if args.train_input_embeddings and orig_emb_rows is not None:
                anchor = anchor + row_anchor_loss(emb.weight, orig_emb_rows, editable_ids_tensor)
            if args.train_lm_head and orig_head_rows is not None:
                anchor = anchor + row_anchor_loss(lm_head.weight, orig_head_rows, editable_ids_tensor)
            loss = loss + args.anchor_weight * anchor

        loss.backward()
        if args.grad_clip and args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
        optimizer.step()

        row_log = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "forget_ga_ce": float(f_ce.detach().cpu()),
            "forget_positions": int(f_n),
            "retain_gd_ce": float(r_ce.detach().cpu()),
            "retain_positions": int(r_n),
            "retain_kl": float(kl.detach().cpu()),
            "anchor": float(anchor.detach().cpu()),
        }
        logs.append(row_log)

        if step % max(1, args.steps // 20) == 0 or step == 1:
            pbar.set_postfix({
                "fGA": f"{row_log['forget_ga_ce']:.3f}",
                "rGD": f"{row_log['retain_gd_ce']:.3f}",
                "KL": f"{row_log['retain_kl']:.3f}",
                "loss": f"{row_log['loss']:.3f}",
            })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "method": "dual_json_rc_gagd_embed_lmhead",
        "model_name": model_name,
        "teacher_model_name": teacher_model_name,
        "tokens_json": args.tokens_json,
        "output_dir": args.output_dir,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "n_ga_tokens": len(set(ga_ids)),
        "n_gd_tokens": len(set(gd_ids)),
        "n_editable_rows": len(editable_ids),
        "args": vars(args),
        "last_log": logs[-1] if logs else None,
        "logs": logs,
    }
    save_json(summary, out_dir / "dual_json_rc_gagd_train_summary.json")

    print("[Done] saved model:", out_dir)
    print("Last log:", logs[-1] if logs else None)


if __name__ == "__main__":
    main()
