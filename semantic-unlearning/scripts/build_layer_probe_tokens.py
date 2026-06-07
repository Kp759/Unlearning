#!/usr/bin/env python3
"""
Build a token JSON using a layer-16 forget/retain linear probe.

Pipeline:
  1. Load finetuned model.
  2. Extract layer-L hidden states at answer-token positions from TOFU forget/retain.
  3. Train a tiny linear probe h_l,t -> forget vs retain.
  4. Aggregate probe confidence back to vocabulary token IDs.
  5. Keep tokens that are:
       high forget-probe confidence,
       sufficiently present in forget answers,
       low/protected in retain answers,
       not punctuation/stop/common tokens.
  6. Save compatible JSON:
       {"token_ids": [...], "semantic_tokens": [...]}

This file is meant to produce a better token mask for:
  scripts/embedding_ga_gd_input_output_unlearn.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
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


def format_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def encode_answer_only(tok, question: str, answer: str, max_length: int) -> Dict[str, torch.Tensor]:
    prompt_ids = tok.encode(format_prompt(question), add_special_tokens=True)
    answer_text = " " + str(answer).strip()
    if tok.eos_token:
        answer_text += tok.eos_token
    answer_ids = tok.encode(answer_text, add_special_tokens=False)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids

    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "attention_mask": torch.tensor([[1] * len(input_ids)], dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def is_bad_token_text(text: str) -> bool:
    s = text.strip()
    if not s:
        return True
    punct = set(".,;:!?()[]{}'\"`-_/\\|@#$%^&*+=~<>")
    if all(ch in punct for ch in s):
        return True
    if len(s) == 1 and not s.isalnum():
        return True
    return False


@torch.no_grad()
def extract_occurrences(
    model,
    tok,
    rows,
    label: int,
    layer_idx: int,
    max_length: int,
    device: torch.device,
    max_occurrences: int | None = None,
):
    xs, ys, tids, eids = [], [], [], []

    model.eval()
    for ex_i, r in enumerate(tqdm(rows, desc=f"extract label={label}")):
        batch = encode_answer_only(tok, r["question"], r["answer"], max_length)
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embeddings. hidden_states[1] = layer 0 output.
        # layer_idx=16 => hidden_states[17].
        h = out.hidden_states[layer_idx + 1][0].detach().float().cpu()
        labels = batch["labels"][0].detach().cpu()
        input_ids = batch["input_ids"][0].detach().cpu()

        pos = labels.ne(-100).nonzero(as_tuple=False).flatten()
        for p in pos.tolist():
            tid = int(input_ids[p].item())
            xs.append(h[p])
            ys.append(label)
            tids.append(tid)
            eids.append(ex_i)

            if max_occurrences is not None and len(xs) >= max_occurrences:
                return torch.stack(xs), torch.tensor(ys), tids, eids

    return torch.stack(xs), torch.tensor(ys), tids, eids


def train_probe(X: torch.Tensor, y: torch.Tensor, epochs: int, lr: float, weight_decay: float, seed: int):
    torch.manual_seed(seed)
    n, d = X.shape

    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, keepdim=True).clamp_min(1e-6)
    Xn = (X - mean) / std

    idx = torch.randperm(n)
    Xn = Xn[idx]
    y = y.float()[idx]

    split = int(0.9 * n)
    Xtr, ytr = Xn[:split], y[:split]
    Xva, yva = Xn[split:], y[split:]

    probe = nn.Linear(d, 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    bs = min(512, max(64, n // 20))

    for ep in range(epochs):
        perm = torch.randperm(Xtr.shape[0])
        total = 0.0
        for i in range(0, Xtr.shape[0], bs):
            b = perm[i:i+bs]
            logits = probe(Xtr[b]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, ytr[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(b)

        if ep == 0 or ep == epochs - 1 or (ep + 1) % 10 == 0:
            with torch.no_grad():
                va_logits = probe(Xva).squeeze(-1)
                va_pred = (torch.sigmoid(va_logits) >= 0.5).float()
                acc = (va_pred == yva).float().mean().item() if len(yva) else 0.0
            print(f"epoch={ep+1:03d} train_bce={total/max(len(Xtr),1):.4f} val_acc={acc:.4f}")

    return probe, mean, std


@torch.no_grad()
def score_occurrences(probe, mean, std, X):
    Xn = (X - mean) / std
    logits = probe(Xn).squeeze(-1)
    probs = torch.sigmoid(logits)
    return probs.cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", default="outputs/semantic_tokens_layer16_probe_tfidf.json")
    ap.add_argument("--forget-split", default=None)
    ap.add_argument("--retain-split", default=None)
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=None)
    ap.add_argument("--max-forget-samples", type=int, default=200)
    ap.add_argument("--max-retain-samples", type=int, default=800)
    ap.add_argument("--max-forget-occurrences", type=int, default=None)
    ap.add_argument("--max-retain-occurrences", type=int, default=20000)
    ap.add_argument("--probe-epochs", type=int, default=40)
    ap.add_argument("--probe-lr", type=float, default=1e-3)
    ap.add_argument("--probe-weight-decay", type=float, default=1e-3)
    ap.add_argument("--min-forget-count", type=int, default=2)
    ap.add_argument("--max-retain-count", type=int, default=10)
    ap.add_argument("--max-retain-ratio", type=float, default=0.006)
    ap.add_argument("--min-forget-probe", type=float, default=0.65)
    ap.add_argument("--min-score", type=float, default=0.35)
    ap.add_argument("--max-final-tokens", type=int, default=1200)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    max_length = args.max_length or cfg["model"].get("max_length", 512)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = torch.device(args.device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=resolve_dtype(args.dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    model.config.use_cache = False

    forget_rows = list(load_dataset("locuslab/TOFU", name=forget_split, split="train"))[: args.max_forget_samples]
    retain_rows = list(load_dataset("locuslab/TOFU", name=retain_split, split="train"))[: args.max_retain_samples]

    Xf, yf, tf, ef = extract_occurrences(
        model, tok, forget_rows, 1, args.layer, max_length, device, args.max_forget_occurrences
    )
    Xr, yr, tr, er = extract_occurrences(
        model, tok, retain_rows, 0, args.layer, max_length, device, args.max_retain_occurrences
    )

    X = torch.cat([Xf, Xr], dim=0)
    y = torch.cat([yf, yr], dim=0)
    token_ids = tf + tr
    source = ["forget"] * len(tf) + ["retain"] * len(tr)

    print(f"Layer={args.layer}")
    print(f"Forget occurrences={len(tf)}, Retain occurrences={len(tr)}, Total={len(token_ids)}")
    print(f"Hidden dim={X.shape[1]}")

    probe, mean, std = train_probe(
        X, y,
        epochs=args.probe_epochs,
        lr=args.probe_lr,
        weight_decay=args.probe_weight_decay,
        seed=args.seed,
    )
    probs = score_occurrences(probe, mean, std, X)

    f_count = Counter()
    r_count = Counter()
    f_probe_sum = defaultdict(float)
    r_probe_sum = defaultdict(float)

    for tid, src, p in zip(token_ids, source, probs):
        if src == "forget":
            f_count[tid] += 1
            f_probe_sum[tid] += p
        else:
            r_count[tid] += 1
            r_probe_sum[tid] += p

    n_f = sum(f_count.values())
    n_r = sum(r_count.values())

    candidates = []
    all_tids = sorted(set(f_count.keys()) | set(r_count.keys()))

    for tid in all_tids:
        fc = f_count[tid]
        rc = r_count[tid]
        if fc < args.min_forget_count:
            continue
        if rc > args.max_retain_count:
            continue
        if (rc / max(n_r, 1)) > args.max_retain_ratio:
            continue

        text = tok.decode([tid])
        if is_bad_token_text(text):
            continue

        f_avg = f_probe_sum[tid] / max(fc, 1)
        r_avg = r_probe_sum[tid] / max(rc, 1) if rc > 0 else 0.0

        if f_avg < args.min_forget_probe:
            continue

        f_rate = fc / max(n_f, 1)
        r_rate = rc / max(n_r, 1)
        count_contrast = math.log((f_rate + 1e-8) / (r_rate + 1e-8))
        probe_contrast = f_avg - r_avg
        score = probe_contrast + 0.15 * count_contrast

        if score < args.min_score:
            continue

        candidates.append({
            "token_id": int(tid),
            "text": text,
            "score": float(score),
            "forget_count": int(fc),
            "retain_count": int(rc),
            "forget_probe_avg": float(f_avg),
            "retain_probe_avg": float(r_avg),
            "probe_contrast": float(probe_contrast),
            "count_contrast": float(count_contrast),
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[: args.max_final_tokens]
    token_ids_out = [x["token_id"] for x in candidates]

    out = {
        "method": "layer_probe_guided_answer_tokens",
        "model_dir": args.model_dir,
        "layer": args.layer,
        "forget_split": forget_split,
        "retain_split": retain_split,
        "token_ids": token_ids_out,
        "semantic_tokens": candidates,
        "selection_args": vars(args),
        "stats": {
            "forget_occurrences": len(tf),
            "retain_occurrences": len(tr),
            "num_selected_tokens": len(token_ids_out),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"Saved: {out_path}")
    print(f"Selected tokens: {len(token_ids_out)}")
    print("Top 30 tokens:")
    for x in candidates[:30]:
        print(
            f"{x['token_id']:>8} {repr(x['text']):>16} "
            f"score={x['score']:.3f} f={x['forget_count']} r={x['retain_count']} "
            f"fp={x['forget_probe_avg']:.3f} rp={x['retain_probe_avg']:.3f}"
        )


if __name__ == "__main__":
    main()
