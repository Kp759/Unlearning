#!/usr/bin/env python3

import argparse, json
from pathlib import Path
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def force_untie_lm_head(model):
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is None:
        raise RuntimeError("No lm_head/output embeddings found.")

    if emb.weight.data_ptr() != out.weight.data_ptr():
        return

    old = out.weight.detach().clone()
    new_head = nn.Linear(old.shape[1], old.shape[0], bias=False).to(old.device, old.dtype)
    new_head.weight.data.copy_(old)
    model.set_output_embeddings(new_head)

    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False


def encode_row(tok, row, fields):
    texts = []

    if fields in ["question", "qa"]:
        texts.append(str(row["question"]))

    if fields in ["answer", "qa"]:
        texts.append(" " + str(row["answer"]).strip())

    ids = []
    for t in texts:
        ids.extend(tok.encode(t, add_special_tokens=False))

    return [int(x) for x in ids]


def collect_token_ids(dataset, tok, fields, min_token_chars):
    ids = set()

    special = {
        x for x in [
            tok.pad_token_id,
            tok.eos_token_id,
            tok.bos_token_id,
            tok.unk_token_id,
        ] if x is not None
    }

    for row in dataset:
        for tid in encode_row(tok, row, fields):
            if tid in special:
                continue

            s = tok.decode([tid])
            if len(s.strip()) < min_token_chars:
                continue

            ids.add(int(tid))

    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_3b_instruct_forget05.yaml")
    ap.add_argument("--original-model", default="outputs/finetuned_model_3B_instruct")
    ap.add_argument("--forget-split", default="forget05")
    ap.add_argument("--retain-split", default="retain95")
    ap.add_argument("--output-dir", required=True)

    ap.add_argument("--forget-fields", choices=["answer", "question", "qa"], default="qa")
    ap.add_argument("--retain-fields", choices=["answer", "question", "qa"], default="qa")

    ap.add_argument("--edit-input", action="store_true")
    ap.add_argument("--edit-lm-head", action="store_true")

    ap.add_argument("--restore-alpha-input", type=float, default=1.0)
    ap.add_argument("--restore-alpha-lm-head", type=float, default=1.0)

    ap.add_argument("--min-token-chars", type=int, default=1)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dtype = torch.float16 if args.dtype in ["float16", "fp16"] else torch.bfloat16

    print("=" * 80)
    print("Zero forget tokens, then restore retain tokens")
    print("=" * 80)
    print("original model:", args.original_model)
    print("forget split:", args.forget_split, "fields:", args.forget_fields)
    print("retain split:", args.retain_split, "fields:", args.retain_fields)
    print("edit input:", args.edit_input)
    print("edit lm_head:", args.edit_lm_head)
    print("restore alpha input:", args.restore_alpha_input)
    print("restore alpha lm_head:", args.restore_alpha_lm_head)
    print("=" * 80)

    tok = AutoTokenizer.from_pretrained(args.original_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    forget_ds = load_dataset("locuslab/TOFU", name=args.forget_split, split="train")
    retain_ds = load_dataset("locuslab/TOFU", name=args.retain_split, split="train")

    forget_ids = collect_token_ids(forget_ds, tok, args.forget_fields, args.min_token_chars)
    retain_ids = collect_token_ids(retain_ds, tok, args.retain_fields, args.min_token_chars)

    overlap_ids = forget_ids & retain_ids
    forget_only_ids = forget_ids - retain_ids
    retain_only_ids = retain_ids - forget_ids

    print("forget token ids:", len(forget_ids))
    print("retain token ids:", len(retain_ids))
    print("overlap F∩R:", len(overlap_ids))
    print("forget-only F-R:", len(forget_only_ids))
    print("retain-only R-F:", len(retain_only_ids))

    student = AutoModelForCausalLM.from_pretrained(
        args.original_model,
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
        if args.edit_input:
            Es = student.get_input_embeddings().weight.data
            Et = teacher.get_input_embeddings().weight.data.to(Es.device)

            for tid in forget_ids:
                if 0 <= tid < Es.shape[0]:
                    Es[tid].zero_()

            a = args.restore_alpha_input
            for tid in retain_ids:
                if 0 <= tid < Es.shape[0]:
                    Es[tid].copy_((1 - a) * Es[tid] + a * Et[tid])

            print("input: zeroed forget rows:", len(forget_ids))
            print("input: restored retain rows:", len(retain_ids))

        if args.edit_lm_head:
            Hs = student.get_output_embeddings().weight.data
            Ht = teacher.get_output_embeddings().weight.data.to(Hs.device)

            for tid in forget_ids:
                if 0 <= tid < Hs.shape[0]:
                    Hs[tid].zero_()

            a = args.restore_alpha_lm_head
            for tid in retain_ids:
                if 0 <= tid < Hs.shape[0]:
                    Hs[tid].copy_((1 - a) * Hs[tid] + a * Ht[tid])

            print("lm_head: zeroed forget rows:", len(forget_ids))
            print("lm_head: restored retain rows:", len(retain_ids))

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    student.save_pretrained(out)
    tok.save_pretrained(out)

    summary = {
        "method": "zero_forget_restore_retain_tokens",
        "original_model": args.original_model,
        "forget_split": args.forget_split,
        "retain_split": args.retain_split,
        "forget_fields": args.forget_fields,
        "retain_fields": args.retain_fields,
        "edit_input": args.edit_input,
        "edit_lm_head": args.edit_lm_head,
        "restore_alpha_input": args.restore_alpha_input,
        "restore_alpha_lm_head": args.restore_alpha_lm_head,
        "n_forget_ids": len(forget_ids),
        "n_retain_ids": len(retain_ids),
        "n_overlap_ids": len(overlap_ids),
        "n_forget_only_ids": len(forget_only_ids),
        "n_retain_only_ids": len(retain_only_ids),
        "forget_ids": sorted(list(forget_ids)),
        "retain_ids": sorted(list(retain_ids)),
        "overlap_ids": sorted(list(overlap_ids)),
        "forget_only_ids": sorted(list(forget_only_ids)),
    }

    with open(out / "zero_forget_restore_retain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("saved:", out)


if __name__ == "__main__":
    main()
