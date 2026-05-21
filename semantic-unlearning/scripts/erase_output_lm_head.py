#!/usr/bin/env python3
import argparse, json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_ids(path):
    with open(path) as f:
        data = json.load(f)
    if "token_ids" in data:
        return sorted(set(map(int, data["token_ids"])))
    return sorted(set(int(x["token_id"]) for x in data["semantic_tokens"]))

def maybe_untie_lm_head(model):
    emb = model.get_input_embeddings()
    out = model.get_output_embeddings()

    if out.weight.data_ptr() != emb.weight.data_ptr():
        print("lm_head already untied.")
        return

    print("lm_head is tied. Untying output head before edit.")
    old_w = out.weight.detach().clone()
    new_head = nn.Linear(old_w.shape[1], old_w.shape[0], bias=False)
    new_head = new_head.to(device=old_w.device, dtype=old_w.dtype)
    new_head.weight.data.copy_(old_w)
    model.set_output_embeddings(new_head)

    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens-json", default="outputs/semantic_tokens.json")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=["zero", "mean", "scale"], default="scale")
    ap.add_argument("--scale", type=float, default=0.05)
    ap.add_argument("--dtype", default="float32")
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype,
        device_map="cpu",
    )
    tok = AutoTokenizer.from_pretrained(args.model_dir)

    maybe_untie_lm_head(model)

    ids = load_ids(args.tokens_json)
    ids_t = torch.tensor(ids, dtype=torch.long)

    out = model.get_output_embeddings()
    emb = model.get_input_embeddings()

    print("lm_head tied after possible untie:", out.weight.data_ptr() == emb.weight.data_ptr())
    print("Editing output rows:", len(ids))

    with torch.no_grad():
        W = out.weight.data
        old_norm = torch.linalg.vector_norm(W[ids_t].float(), dim=1).mean().item()

        if args.method == "zero":
            W[ids_t] = 0
        elif args.method == "mean":
            mean_vec = W.float().mean(dim=0).to(W.dtype)
            W[ids_t] = mean_vec
        elif args.method == "scale":
            W[ids_t] = W[ids_t] * args.scale

        new_norm = torch.linalg.vector_norm(W[ids_t].float(), dim=1).mean().item()

    print("Old avg output row norm:", old_norm)
    print("New avg output row norm:", new_norm)
    print("Output finite:", torch.isfinite(out.weight).all().item())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)

    print("Saved:", out_dir)

if __name__ == "__main__":
    main()
