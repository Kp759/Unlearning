#!/usr/bin/env python3

import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def force_untie_lm_head(model):
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if emb.weight.data_ptr() != out.weight.data_ptr():
        return
    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False).to(old.device, old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)
    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggressive-model", required=True)
    ap.add_argument("--original-model", required=True)
    ap.add_argument("--tokens-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--restore-min-retain-answer-count", type=int, default=9)
    ap.add_argument("--restore-input", action="store_true")
    ap.add_argument("--restore-lm-head", action="store_true")
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype in ["float16", "fp16"] else torch.bfloat16

    d = json.load(open(args.tokens_json))
    rows = d["semantic_tokens"]

    restore_ids = []
    keep_zero_ids = []

    for r in rows:
        tid = int(r["token_id"])
        rc = int(r.get("retain_answer_count", 0))
        if rc >= args.restore_min_retain_answer_count:
            restore_ids.append(tid)
        else:
            keep_zero_ids.append(tid)

    print("restore ids:", len(restore_ids))
    print("keep aggressive ids:", len(keep_zero_ids))
    print("threshold rc >=", args.restore_min_retain_answer_count)

    tok = AutoTokenizer.from_pretrained(args.aggressive_model)

    student = AutoModelForCausalLM.from_pretrained(
        args.aggressive_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        args.original_model,
        torch_dtype=dtype,
        device_map="auto",
    )

    force_untie_lm_head(student)
    force_untie_lm_head(teacher)

    with torch.no_grad():
        if args.restore_input:
            Ws = student.get_input_embeddings().weight.data
            Wt = teacher.get_input_embeddings().weight.data.to(Ws.device)
            for tid in restore_ids:
                Ws[tid].copy_(Wt[tid])
            print("restored input rows:", len(restore_ids))

        if args.restore_lm_head:
            Ws = student.get_output_embeddings().weight.data
            Wt = teacher.get_output_embeddings().weight.data.to(Ws.device)
            for tid in restore_ids:
                Ws[tid].copy_(Wt[tid])
            print("restored lm_head rows:", len(restore_ids))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)

    json.dump({
        "aggressive_model": args.aggressive_model,
        "original_model": args.original_model,
        "tokens_json": args.tokens_json,
        "restore_min_retain_answer_count": args.restore_min_retain_answer_count,
        "n_restore_ids": len(restore_ids),
        "n_keep_aggressive_ids": len(keep_zero_ids),
        "restore_input": args.restore_input,
        "restore_lm_head": args.restore_lm_head,
    }, open(out / "restore_summary.json", "w"), indent=2)

    print("saved:", out)

if __name__ == "__main__":
    main()
