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
    ap.add_argument("--alpha-input", type=float, default=0.0)
    ap.add_argument("--alpha-lm-head", type=float, default=0.5)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype in ["float16", "fp16"] else torch.bfloat16

    d = json.load(open(args.tokens_json))
    rows = d["semantic_tokens"]

    restore_ids = []
    for r in rows:
        rc = int(r.get("retain_answer_count", 0))
        if rc >= args.restore_min_retain_answer_count:
            restore_ids.append(int(r["token_id"]))

    print("restore ids:", len(restore_ids))
    print("threshold rc >=", args.restore_min_retain_answer_count)
    print("alpha_input:", args.alpha_input)
    print("alpha_lm_head:", args.alpha_lm_head)

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
        if args.alpha_input > 0:
            Ws = student.get_input_embeddings().weight.data
            Wt = teacher.get_input_embeddings().weight.data.to(Ws.device)
            a = args.alpha_input
            for tid in restore_ids:
                Ws[tid].copy_((1 - a) * Ws[tid] + a * Wt[tid])
            print("interpolated input rows:", len(restore_ids))

        if args.alpha_lm_head > 0:
            Ws = student.get_output_embeddings().weight.data
            Wt = teacher.get_output_embeddings().weight.data.to(Ws.device)
            a = args.alpha_lm_head
            for tid in restore_ids:
                Ws[tid].copy_((1 - a) * Ws[tid] + a * Wt[tid])
            print("interpolated lm_head rows:", len(restore_ids))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)

    json.dump({
        "aggressive_model": args.aggressive_model,
        "original_model": args.original_model,
        "tokens_json": args.tokens_json,
        "restore_min_retain_answer_count": args.restore_min_retain_answer_count,
        "alpha_input": args.alpha_input,
        "alpha_lm_head": args.alpha_lm_head,
        "n_restore_ids": len(restore_ids),
    }, open(out / "interpolate_restore_summary.json", "w"), indent=2)

    print("saved:", out)

if __name__ == "__main__":
    main()
