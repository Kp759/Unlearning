#!/usr/bin/env python3
"""Fine-tune a base model on Eval-DU+ FT-Mul-Chunk (the paper's recipe).

    python -u scripts/finetune_evaldu_plus.py \
        --model-path <llama> --upstream data/evaldu_plus_upstream \
        --output-dir outputs/evaldu_plus_v1/ft_mul_chunk --local-files-only

Recipe (upstream scripts/finetune.sh + finetune.py, FT-Mul-Chunk row):
full fine-tuning, causal LM loss on every text token plus a final EOS target,
lr 1e-5, effective batch 16, 4 epochs, linear schedule with one epoch of
warmup, weight decay 0.01 (not on norm weights), seed 42, max length 500.
Upstream ran Llama2-7B / Llama3-8B in bf16 with 32-bit AdamW on 2 GPUs; here
fp32 master weights with bf16 autocast on one GPU, falling back to pure bf16
if the GPU runs out of memory (recorded in the report).

Before and after training the knowledge score (the paper's metric) is
measured on the held-out test paraphrases, the unlearning paraphrases and the
chunks, so the log shows whether the facts were learned before SURE runs.
Writes the model (bf16 safetensors) + tokenizer and finetune_report.json.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaldu_plus_data as ed  # noqa: E402

DATA_FILES = {"ft_mul_chunk": "ft_mul_chunk.json", "ft_mul": "ft_mul.json",
              "ft_single": "ft_single.json", "ft_mul_chunk_iso": "ft_mul_chunk_iso.json"}


def encode_texts(tok, texts, max_length):
    """Upstream convert_pure_text_raw_data_to_model_format: BOS + text, label = text + EOS."""
    rows = []
    for text in texts:
        ids = tok(text, add_special_tokens=True, max_length=max_length, truncation=True)["input_ids"]
        rows.append(list(ids))
    return rows


def collate(rows, eos_id):
    width = max(len(r) for r in rows) + 1
    input_ids = torch.full((len(rows), width), int(eos_id), dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    labels = torch.full_like(input_ids, -100)
    for i, ids in enumerate(rows):
        n = len(ids)
        input_ids[i, :n] = torch.tensor(ids)
        attention[i, :n] = 1
        labels[i, :n] = torch.tensor(ids)
        labels[i, n] = int(eos_id)          # the model learns to end the text
    return input_ids, attention, labels


def knowledge_report(model, tok, probes, device, batch_size):
    groups = {}
    for name, subset in probes.items():
        rows, skipped = ed.knowledge_scores(model, tok, subset, device, batch_size=batch_size)
        answerable = {p["id"] for p in subset if p.get("person_in_prefix")}
        groups[name] = {
            "mean_knowledge_score": ed._mean(r["score"] for r in rows),
            "mean_knowledge_score_person_in_prefix": ed._mean(
                r["score"] for r in rows if r["id"] in answerable),
            "probes": len(rows), "skipped_completion_not_found": len(skipped),
        }
    return groups


def train(model, tok, rows, args, device, precision):
    model.train()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    accumulation = max(1, args.batch_size // args.micro_batch_size)
    max_steps = int(args.epochs * len(rows)) // (args.micro_batch_size * accumulation)
    warmup = max(1, max_steps // args.epochs) if args.warmup_steps is None else args.warmup_steps
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / warmup if s < warmup
        else max(0.0, (max_steps - s) / max(1, max_steps - warmup)))
    rng = random.Random(args.seed)
    order, history, step, started = [], [], 0, time.time()
    autocast = precision == "fp32_master_bf16_autocast" and device.type == "cuda"
    while step < max_steps:
        if len(order) < args.micro_batch_size * accumulation:
            epoch = list(range(len(rows)))
            rng.shuffle(epoch)
            order += epoch
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accumulation):
            batch = [rows[i] for i in order[:args.micro_batch_size]]
            order = order[args.micro_batch_size:]
            input_ids, attention, labels = (t.to(device) for t in collate(batch, tok.eos_token_id))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                loss = model(input_ids=input_ids, attention_mask=attention, labels=labels).loss
            (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        schedule.step()
        step += 1
        if step % max(1, max_steps // 20) == 0 or step == max_steps:
            history.append({"step": step, "loss": total, "lr": schedule.get_last_lr()[0],
                            "seconds": round(time.time() - started, 1)})
            print(json.dumps({"phase": "train", **history[-1]}), flush=True)
    model.eval()
    return {"max_steps": max_steps, "warmup_steps": warmup, "accumulation": accumulation,
            "history": history, "seconds": round(time.time() - started, 1)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--upstream", required=True, help="clone of " + ed.UPSTREAM)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--data", choices=sorted(DATA_FILES), default="ft_mul_chunk")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=16, help="effective batch")
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--warmup-steps", type=int, default=None, help="default: one epoch (upstream)")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--max-length", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--precision", choices=("fp32_master_bf16_autocast", "bf16"),
                   default="fp32_master_bf16_autocast")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false")
    p.add_argument("--eval-batch-size", type=int, default=32)
    p.add_argument("--skip-knowledge-report", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(a.seed)
    random.seed(a.seed)
    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    upstream = ed.load_upstream(a.upstream)
    texts = json.loads((Path(a.upstream) / "synthetic_data" / DATA_FILES[a.data]).read_text())["fact"]
    facts, raw = upstream["facts"], upstream["raw"]
    probes = {
        "test": ed.sentence_probes(facts, raw, "test"),
        "unlearn": ed.sentence_probes(facts, raw, "unlearn"),
        "chunk": ed.chunk_probes(facts, raw, upstream["chunks"], upstream["names"]),
    }

    tok = AutoTokenizer.from_pretrained(a.model_path, use_fast=True, local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = torch.device(a.device)
    rows = encode_texts(tok, texts, a.max_length)

    def load(dtype):
        return AutoModelForCausalLM.from_pretrained(
            a.model_path, dtype=dtype, local_files_only=a.local_files_only,
            attn_implementation="eager").to(device)

    precision = a.precision
    model = load(torch.float32 if precision != "bf16" else torch.bfloat16)
    before = None
    if not a.skip_knowledge_report:
        model.eval()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            before = knowledge_report(model, tok, probes, device, a.eval_batch_size)
        print(json.dumps({"phase": "knowledge_before", **{k: v["mean_knowledge_score"]
                                                           for k, v in before.items()}}), flush=True)
    fallback = None
    try:
        training = train(model, tok, rows, a, device, precision)
    except torch.cuda.OutOfMemoryError as error:
        if precision == "bf16":
            raise
        fallback = f"fp32 master weights ran out of memory ({str(error).splitlines()[0]}); retried in bf16"
        print(json.dumps({"phase": "precision_fallback", "reason": fallback}), flush=True)
        del model
        torch.cuda.empty_cache()
        torch.manual_seed(a.seed)
        precision = "bf16"
        model = load(torch.bfloat16)
        training = train(model, tok, rows, a, device, precision)

    model = model.to(torch.bfloat16)
    after = None
    if not a.skip_knowledge_report:
        after = knowledge_report(model, tok, probes, device, a.eval_batch_size)
        print(json.dumps({"phase": "knowledge_after", **{k: v["mean_knowledge_score"]
                                                          for k, v in after.items()}}), flush=True)
    model.config.use_cache = True
    model.save_pretrained(output, safe_serialization=True)
    tok.save_pretrained(output)
    report = {
        "dataset": ed.DATASET, "upstream": ed.UPSTREAM, "upstream_sources_sha256": upstream["sources"],
        "base_model_path": str(Path(a.model_path).resolve()), "data": a.data,
        "training_texts": len(rows),
        "recipe": {"epochs": a.epochs, "lr": a.lr, "effective_batch": a.batch_size,
                   "micro_batch": a.micro_batch_size, "weight_decay": a.weight_decay,
                   "seed": a.seed, "max_length": a.max_length, "schedule": "linear, warmup one epoch",
                   "precision": precision, "precision_fallback": fallback},
        "training": training,
        "knowledge_before": before, "knowledge_after": after,
    }
    (output / "finetune_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "finetune_complete", "output_dir": str(output),
                      "test_knowledge_after": (after or {}).get("test", {}).get("mean_knowledge_score"),
                      "chunk_knowledge_after": (after or {}).get("chunk", {}).get("mean_knowledge_score")},
                     indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
