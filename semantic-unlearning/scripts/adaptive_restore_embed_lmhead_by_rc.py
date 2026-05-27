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

def alpha_policy(rc):
    if rc <= 2:
        return 0.00, 0.00
    elif rc <= 5:
        return 0.10, 0.25
    elif rc <= 8:
        return 0.25, 0.50
    else:
        return 0.50, 0.75

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aggressive-model", required=True)
    ap.add_argument("--original-model", required=True)
    ap.add_argument("--tokens-json", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    dtype = torch.float16 if args.dtype in ["float16", "fp16"] else torch.bfloat16

    d = json.load(open(args.tokens_json))
    rows = d["semantic_tokens"]

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

    counts = {
        "rc0_2": 0,
        "rc3_5": 0,
        "rc6_8": 0,
        "rc9plus": 0,
        "input_restored": 0,
        "lmhead_restored": 0,
    }

    with torch.no_grad():
        Es = student.get_input_embeddings().weight.data
        Et = teacher.get_input_embeddings().weight.data.to(Es.device)

        Hs = student.get_output_embeddings().weight.data
        Ht = teacher.get_output_embeddings().weight.data.to(Hs.device)

        for r in rows:
            tid = int(r["token_id"])
            rc = int(r.get("retain_answer_count", 0))

            alpha_input, alpha_lm = alpha_policy(rc)

            if rc <= 2:
                counts["rc0_2"] += 1
            elif rc <= 5:
                counts["rc3_5"] += 1
            elif rc <= 8:
                counts["rc6_8"] += 1
            else:
                counts["rc9plus"] += 1

            if alpha_input > 0:
                Es[tid].copy_((1 - alpha_input) * Es[tid] + alpha_input * Et[tid])
                counts["input_restored"] += 1

            if alpha_lm > 0:
                Hs[tid].copy_((1 - alpha_lm) * Hs[tid] + alpha_lm * Ht[tid])
                counts["lmhead_restored"] += 1

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)

    json.dump({
        "policy": "adaptive_restore_input_embedding_and_lmhead_by_retain_answer_count",
        "alpha_policy": {
            "rc<=2": {"input": 0.00, "lm_head": 0.00},
            "rc3-5": {"input": 0.10, "lm_head": 0.25},
            "rc6-8": {"input": 0.25, "lm_head": 0.50},
            "rc>=9": {"input": 0.50, "lm_head": 0.75}
        },
        "bucket_counts": counts,
        "aggressive_model": args.aggressive_model,
        "original_model": args.original_model,
        "tokens_json": args.tokens_json,
    }, open(out / "adaptive_restore_embed_lmhead_summary.json", "w"), indent=2)

    print("saved:", out)
    print("bucket_counts:", counts)

if __name__ == "__main__":
    main()
