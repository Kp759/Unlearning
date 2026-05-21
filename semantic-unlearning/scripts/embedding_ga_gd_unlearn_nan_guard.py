#!/usr/bin/env python3
"""
Embedding-only GA/GD unlearning for TOFU.

Core idea:
  - Use JSON + retain-TFIDF to create forget token set T_f.
  - Freeze the whole model except token embedding rows.
  - Gradient ASCENT on forget-answer loss, updating only T_f rows.
  - Gradient DESCENT on retain-answer loss, updating only retain-protection rows.
  - Add anchor regularization and row delta clipping to avoid retain collapse.

This script is designed for your semantic-unlearning repo.

Recommended:
  1) First create outputs/semantic_tokens.json using JSON-TFIDF filtering.
  2) Then run this script from the repo root.

Example:
  python scripts/embedding_ga_gd_unlearn.py \
    --config config/config_3b_instruct_forget05.yaml \
    --model-dir outputs/finetuned_model_3B_instruct \
    --forget-token-json outputs/semantic_tokens.json \
    --output-dir outputs/unlearned_model_embed_ga_gd_json_tfidf \
    --forget-split forget05 \
    --retain-split retain95 \
    --steps 300 \
    --batch-size 2 \
    --retain-batch-size 2 \
    --forget-lr 5e-5 \
    --retain-lr 2e-5 \
    --retain-top-k 5000 \
    --anchor-lambda 0.05 \
    --max-delta-norm 0.35 \
    --update-lm-head-if-untied
"""

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_dtype(dtype: str) -> torch.dtype:
    dtype = str(dtype).lower()
    if dtype in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if dtype in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def first_param_device(model) -> torch.device:
    return next(model.parameters()).device


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def encode_answer_only(tokenizer, question: str, answer: str, max_length: int) -> Dict[str, List[int]]:
    """
    Prompt tokens are masked with -100.
    Only answer tokens contribute to the LM loss.
    """
    prompt = format_prompt(question)
    answer_text = " " + str(answer).strip()
    if tokenizer.eos_token:
        answer_text += tokenizer.eos_token

    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)
    answer_ids = tokenizer.encode(answer_text, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def collate(examples: Sequence[Dict[str, List[int]]], pad_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    max_len = max(len(x["input_ids"]) for x in examples)

    input_ids, attention_mask, labels = [], [], []
    for x in examples:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attention_mask.append(x["attention_mask"] + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def sample_batch(rng: random.Random, encoded, batch_size: int, pad_id: int, device: torch.device):
    batch = [encoded[rng.randrange(len(encoded))] for _ in range(batch_size)]
    return collate(batch, pad_id, device)


def lm_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    ).loss


def load_token_ids(path: Path) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "token_ids" in data:
        ids = [int(x) for x in data["token_ids"]]
    elif "semantic_tokens" in data:
        ids = [int(x["token_id"]) for x in data["semantic_tokens"]]
    else:
        raise ValueError(f"No token_ids or semantic_tokens found in {path}")

    return sorted(set(ids))


def doc_frequency(dataset, tokenizer, max_samples=None) -> Tuple[Counter, int]:
    df = Counter()
    n = 0

    for i, row in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        text = f"Question: {row['question']} Answer: {row['answer']}"
        ids = set(tokenizer.encode(text, add_special_tokens=False))
        for tid in ids:
            df[int(tid)] += 1
        n += 1

    return df, n


def build_retain_tokens(
    tokenizer,
    forget_ds,
    retain_ds,
    forget_token_set: set,
    top_k: int,
    min_retain_count: int,
    max_forget_ratio: float,
    max_retain_samples=None,
    max_forget_samples=None,
) -> List[int]:
    """
    Select retain-protection rows using retain TF-IDF.

    Keep tokens that are:
      - common/useful in retain
      - rare in forget
      - not in forget token set
    """
    retain_df, n_retain = doc_frequency(retain_ds, tokenizer, max_retain_samples)
    forget_df, n_forget = doc_frequency(forget_ds, tokenizer, max_forget_samples)

    special_ids = {
        x for x in [
            tokenizer.pad_token_id,
            tokenizer.eos_token_id,
            tokenizer.bos_token_id,
            tokenizer.unk_token_id,
        ] if x is not None
    }

    total_docs = n_retain + n_forget
    scored = []

    for tid, r_count in retain_df.items():
        tid = int(tid)
        if tid in forget_token_set:
            continue
        if tid in special_ids:
            continue
        if r_count < min_retain_count:
            continue

        token_str = tokenizer.decode([tid])
        if len(token_str.strip()) < 2:
            continue

        f_count = int(forget_df.get(tid, 0))
        f_ratio = f_count / max(1, n_forget)
        if f_ratio > max_forget_ratio:
            continue

        r_ratio = r_count / max(1, n_retain)
        total_df = r_count + f_count
        idf = math.log((total_docs + 1) / (total_df + 1)) + 1.0
        retain_tfidf = r_ratio * idf
        retain_specificity = (r_ratio + 1e-8) / (f_ratio + 1e-8)

        score = retain_tfidf * math.log1p(retain_specificity)

        scored.append((score, r_count, -f_count, tid))

    scored.sort(reverse=True)
    return [tid for _, _, _, tid in scored[:top_k]]


def make_row_mask(vocab_size: int, token_ids: Iterable[int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    ids = [int(x) for x in set(token_ids) if 0 <= int(x) < vocab_size]
    if ids:
        mask[torch.tensor(ids, dtype=torch.long, device=device)] = True
    return mask


def mask_grad_rows(param: torch.nn.Parameter, row_mask: torch.Tensor) -> None:
    if param is None or param.grad is None:
        return
    param.grad[~row_mask] = 0


def anchor_loss(weight: torch.Tensor, ids: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(weight[ids].float(), original.float())


@torch.no_grad()
def clip_rows(weight: torch.Tensor, ids: torch.Tensor, original: torch.Tensor, max_norm: float) -> None:
    if max_norm <= 0 or ids.numel() == 0:
        return

    current = weight.data[ids].float()
    delta = current - original.float()
    norms = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    scale = torch.clamp(max_norm / (norms + 1e-12), max=1.0)
    clipped = original.float() + delta * scale
    weight.data[ids] = clipped.to(weight.dtype)


def freeze_model_get_trainable_matrices(model, update_lm_head_if_untied: bool):
    for p in model.parameters():
        p.requires_grad_(False)

    emb = model.get_input_embeddings()
    emb.weight.requires_grad_(True)

    lm_head = getattr(model, "lm_head", None)
    update_lm_head = False

    if lm_head is not None and hasattr(lm_head, "weight"):
        tied = lm_head.weight.data_ptr() == emb.weight.data_ptr()
        if tied:
            update_lm_head = False
        elif update_lm_head_if_untied:
            lm_head.weight.requires_grad_(True)
            update_lm_head = True

    return emb, lm_head, update_lm_head


def optimizer_for(emb, lm_head, update_lm_head: bool, lr: float):
    params = [emb.weight]
    if update_lm_head:
        params.append(lm_head.weight)
    return torch.optim.AdamW(params, lr=lr, weight_decay=0.0)



# ===================== NaN / Inf guard helpers =====================
def _finite_or_raise_loss(loss, name: str, step: int):
    import torch
    if not torch.isfinite(loss).all().item():
        raise RuntimeError(f"[NaNGuard] Non-finite loss at step={step}: {name}={loss}")

def _finite_or_raise_grad(param, name: str, step: int):
    import torch
    if param is None or param.grad is None:
        return
    if not torch.isfinite(param.grad).all().item():
        bad = (~torch.isfinite(param.grad)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite grad at step={step}: {name}, bad_count={bad}")

def _finite_or_raise_rows(weight, row_ids, name: str, step: int):
    import torch
    rows = weight.data[row_ids]
    if not torch.isfinite(rows).all().item():
        bad = (~torch.isfinite(rows)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite rows at step={step}: {name}, bad_count={bad}")

def _finite_or_raise_full(weight, name: str, step: int):
    import torch
    if not torch.isfinite(weight).all().item():
        bad = (~torch.isfinite(weight)).sum().item()
        raise RuntimeError(f"[NaNGuard] Non-finite tensor at step={step}: {name}, bad_count={bad}")
# =================== end NaN / Inf guard helpers ===================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--forget-token-json", default="outputs/semantic_tokens.json")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--forget-split", default=None)
    parser.add_argument("--retain-split", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--max-length", type=int, default=None)

    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--retain-batch-size", type=int, default=2)
    parser.add_argument("--forget-lr", type=float, default=5e-5)
    parser.add_argument("--retain-lr", type=float, default=2e-5)

    parser.add_argument("--forget-loss-weight", type=float, default=1.0)
    parser.add_argument("--retain-loss-weight", type=float, default=1.0)
    parser.add_argument("--anchor-lambda", type=float, default=0.05)
    parser.add_argument("--max-delta-norm", type=float, default=0.35)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--retain-top-k", type=int, default=5000)
    parser.add_argument("--retain-min-count", type=int, default=10)
    parser.add_argument("--retain-max-forget-ratio", type=float, default=0.005)
    parser.add_argument("--max-retain-token-selection-samples", type=int, default=None)
    parser.add_argument("--max-forget-token-selection-samples", type=int, default=None)

    parser.add_argument("--max-forget-train-samples", type=int, default=None)
    parser.add_argument("--max-retain-train-samples", type=int, default=800)

    parser.add_argument("--forget-steps-per-round", type=int, default=1)
    parser.add_argument("--retain-steps-per-round", type=int, default=1)
    parser.add_argument("--update-lm-head-if-untied", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    dtype = args.dtype or cfg["model"].get("dtype", "float16")
    max_length = args.max_length or cfg["model"].get("max_length", 512)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)

    print("=" * 80)
    print("Embedding-only GA/GD unlearning")
    print("=" * 80)
    print(f"model-dir: {args.model_dir}")
    print(f"forget-token-json: {args.forget_token_json}")
    print(f"output-dir: {args.output_dir}")
    print(f"forget split: {forget_split}")
    print(f"retain split: {retain_split}")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=resolve_dtype(dtype),
        device_map=args.device_map,
    )
    model.config.use_cache = False
    model.train()

    emb, lm_head, update_lm_head = freeze_model_get_trainable_matrices(
        model,
        update_lm_head_if_untied=args.update_lm_head_if_untied,
    )

    input_device = first_param_device(model)
    emb_device = emb.weight.device
    vocab_size = emb.weight.shape[0]

    print(f"input device: {input_device}")
    print(f"embedding device: {emb_device}")
    print(f"embedding shape: {tuple(emb.weight.shape)}")
    print(f"update lm_head if untied: {update_lm_head}")

    forget_token_ids = load_token_ids(Path(args.forget_token_json))
    forget_token_set = set(forget_token_ids)
    print(f"forget tokens: {len(forget_token_ids)}")

    print("Loading TOFU datasets...")
    forget_ds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=retain_split, split="train")

    print("Building retain-protection tokens...")
    retain_token_ids = build_retain_tokens(
        tokenizer=tokenizer,
        forget_ds=forget_ds,
        retain_ds=retain_ds,
        forget_token_set=forget_token_set,
        top_k=args.retain_top_k,
        min_retain_count=args.retain_min_count,
        max_forget_ratio=args.retain_max_forget_ratio,
        max_retain_samples=args.max_retain_token_selection_samples,
        max_forget_samples=args.max_forget_token_selection_samples,
    )
    print(f"retain-protection tokens: {len(retain_token_ids)}")

    forget_mask = make_row_mask(vocab_size, forget_token_ids, emb_device)
    retain_mask = make_row_mask(vocab_size, retain_token_ids, emb_device)

    anchor_ids = sorted(set(forget_token_ids) | set(retain_token_ids))
    anchor_ids_tensor = torch.tensor(anchor_ids, dtype=torch.long, device=emb_device)
    emb_anchor = emb.weight.detach()[anchor_ids_tensor].clone().float()

    lm_anchor = None
    if update_lm_head:
        lm_anchor = lm_head.weight.detach()[anchor_ids_tensor.to(lm_head.weight.device)].clone().float()

    print(f"anchored rows: {len(anchor_ids)}")

    forget_rows = list(forget_ds)
    retain_rows = list(retain_ds)

    if args.max_forget_train_samples is not None:
        forget_rows = forget_rows[: args.max_forget_train_samples]
    if args.max_retain_train_samples is not None:
        retain_rows = retain_rows[: args.max_retain_train_samples]

    print("Encoding train examples...")
    forget_encoded = [
        encode_answer_only(tokenizer, r["question"], r["answer"], max_length)
        for r in forget_rows
    ]
    retain_encoded = [
        encode_answer_only(tokenizer, r["question"], r["answer"], max_length)
        for r in retain_rows
    ]

    print(f"forget train examples: {len(forget_encoded)}")
    print(f"retain train examples: {len(retain_encoded)}")

    opt_forget = optimizer_for(emb, lm_head, update_lm_head, args.forget_lr)
    opt_retain = optimizer_for(emb, lm_head, update_lm_head, args.retain_lr)

    pbar = tqdm(range(args.steps), desc="GA/GD embedding rows")
    last_f, last_r, last_a = 0.0, 0.0, 0.0

    for step in pbar:
        for _ in range(args.forget_steps_per_round):
            opt_forget.zero_grad(set_to_none=True)

            batch = sample_batch(
                rng,
                forget_encoded,
                args.batch_size,
                tokenizer.pad_token_id,
                input_device,
            )

            f_loss = lm_loss(model, batch)
            _finite_or_raise_loss(f_loss, 'forget_loss', step)
            a_loss = anchor_loss(emb.weight, anchor_ids_tensor, emb_anchor)

            if update_lm_head:
                lm_ids = anchor_ids_tensor.to(lm_head.weight.device)
                a_loss = a_loss + anchor_loss(lm_head.weight, lm_ids, lm_anchor)

            # Negative loss => gradient ascent on forget answer loss.
            loss = -args.forget_loss_weight * f_loss + args.anchor_lambda * a_loss
            loss.backward()

            mask_grad_rows(emb.weight, forget_mask)
            _finite_or_raise_grad(emb.weight, 'embed_tokens.weight.forget_grad', step)
            if update_lm_head:
                mask_grad_rows(lm_head.weight, forget_mask.to(lm_head.weight.device))

            if args.grad_clip > 0:
                params = [emb.weight] + ([lm_head.weight] if update_lm_head else [])
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            opt_forget.step()
            _finite_or_raise_rows(emb.weight, anchor_ids_tensor, 'embed_tokens.after_forget_step', step)

            clip_rows(emb.weight, anchor_ids_tensor, emb_anchor, args.max_delta_norm)
            if update_lm_head:
                clip_rows(lm_head.weight, anchor_ids_tensor.to(lm_head.weight.device), lm_anchor, args.max_delta_norm)

            last_f = float(f_loss.detach().cpu())
            last_a = float(a_loss.detach().cpu())

        for _ in range(args.retain_steps_per_round):
            opt_retain.zero_grad(set_to_none=True)

            batch = sample_batch(
                rng,
                retain_encoded,
                args.retain_batch_size,
                tokenizer.pad_token_id,
                input_device,
            )

            r_loss = lm_loss(model, batch)
            _finite_or_raise_loss(r_loss, 'retain_loss', step)
            a_loss = anchor_loss(emb.weight, anchor_ids_tensor, emb_anchor)

            if update_lm_head:
                lm_ids = anchor_ids_tensor.to(lm_head.weight.device)
                a_loss = a_loss + anchor_loss(lm_head.weight, lm_ids, lm_anchor)

            # Normal loss => gradient descent on retain answer loss.
            loss = args.retain_loss_weight * r_loss + args.anchor_lambda * a_loss
            loss.backward()

            mask_grad_rows(emb.weight, retain_mask)
            _finite_or_raise_grad(emb.weight, 'embed_tokens.weight.retain_grad', step)
            if update_lm_head:
                mask_grad_rows(lm_head.weight, retain_mask.to(lm_head.weight.device))

            if args.grad_clip > 0:
                params = [emb.weight] + ([lm_head.weight] if update_lm_head else [])
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            opt_retain.step()
            _finite_or_raise_rows(emb.weight, anchor_ids_tensor, 'embed_tokens.after_retain_step', step)

            clip_rows(emb.weight, anchor_ids_tensor, emb_anchor, args.max_delta_norm)
            if update_lm_head:
                clip_rows(lm_head.weight, anchor_ids_tensor.to(lm_head.weight.device), lm_anchor, args.max_delta_norm)

            last_r = float(r_loss.detach().cpu())
            last_a = float(a_loss.detach().cpu())

        if step % 10 == 0 or step == args.steps - 1:
            pbar.set_postfix(
                {
                    "forget_loss": f"{last_f:.3f}",
                    "retain_loss": f"{last_r:.3f}",
                    "anchor": f"{last_a:.6f}",
                }
            )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _finite_or_raise_full(emb.weight.data, 'FINAL_embed_tokens.weight', args.steps)
    print(f"Saving model to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

    summary = {
        "method": "embedding_only_ga_gd",
        "model_dir": args.model_dir,
        "forget_token_json": args.forget_token_json,
        "output_dir": args.output_dir,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "n_forget_tokens": len(forget_token_ids),
        "n_retain_tokens": len(retain_token_ids),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "retain_batch_size": args.retain_batch_size,
        "forget_lr": args.forget_lr,
        "retain_lr": args.retain_lr,
        "forget_loss_weight": args.forget_loss_weight,
        "retain_loss_weight": args.retain_loss_weight,
        "anchor_lambda": args.anchor_lambda,
        "max_delta_norm": args.max_delta_norm,
        "update_lm_head": update_lm_head,
    }

    with open(out_dir / "embedding_ga_gd_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[done]")
    print("Evaluate:")
    print(
        f"python scripts/tofu_eval.py --config {args.config} "
        f"--model-dir {out_dir} "
        f"--method embed_ga_gd_json_tfidf_forget05 "
        f"--forget-split {forget_split} --retain-split {retain_split}"
    )


if __name__ == "__main__":
    main()
