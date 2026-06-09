#!/usr/bin/env python3

import argparse
import json
import math
import random
import urllib.request
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


MCF_URL = "https://memit.baulab.info/data/dsets/multi_counterfact.json"
ATTR_SNIPS_URL = "https://rome.baulab.info/data/dsets/attribute_snippets.json"


def dtype_from_str(x):
    x = str(x).lower()
    if x in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if x in ["fp16", "float16"]:
        return torch.float16
    return torch.float32


def download_file(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url} to {path}")
        urllib.request.urlretrieve(url, path)
    return path


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def official_split(data, forget_n, retain_n, seed):
    rng = random.Random(seed)
    half = len(data) // 2

    retain_pool = data[:half]
    forget_pool = data[half:]

    forget_records = rng.sample(forget_pool, k=min(forget_n, len(forget_pool)))
    retain_records = rng.sample(retain_pool, k=min(retain_n, len(retain_pool)))

    return forget_records, retain_records


def get_essence_texts_for_record(record, snips):
    """
    ROME/MEMIT CounterFact-style essence texts.

    Official logic is approximately:
      rel_id = record["requested_rewrite"]["relation_id"]
      target_new_id = record["requested_rewrite"]["target_new"]["id"]
      subject = record["requested_rewrite"]["subject"]

      essence_texts = [
          x["text"] for x in snips[rel_id][target_new_id]
          if x["name"] == subject
      ]
    """
    rr = record["requested_rewrite"]

    rel_id = rr.get("relation_id", None)
    subject = str(rr.get("subject", "")).strip()

    target_new = rr.get("target_new", {})
    target_new_id = target_new.get("id", None)

    if rel_id is None or target_new_id is None:
        return []

    rel_id = str(rel_id)
    target_new_id = str(target_new_id)

    if rel_id not in snips:
        return []

    if target_new_id not in snips[rel_id]:
        return []

    candidates = snips[rel_id][target_new_id]

    texts = []
    for x in candidates:
        name = str(x.get("name", "")).strip()
        text = str(x.get("text", "")).strip()

        if not text:
            continue

        if name == subject:
            texts.append(text)

    return texts


@torch.no_grad()
def text_perplexity(model, tok, text, max_input_length=100):
    """
    ROME/MEMIT-style simple LM perplexity over a text string.

    Uses next-token negative log-likelihood:
      PPL = exp(mean CE over tokens)
    """
    if not text or not text.strip():
        return None

    enc = tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=True,
    )

    input_ids = enc["input_ids"].to(model.device)

    if input_ids.shape[1] < 2:
        return None

    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    loss = torch.nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
    )

    return float(torch.exp(loss.detach().float()).cpu())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--mcf-path", default="data/multi_counterfact.json")
    ap.add_argument("--attr-snips-path", default="data/attribute_snippets.json")
    ap.add_argument("--forget-n", type=int, default=50)
    ap.add_argument("--retain-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-input-length", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    mcf_path = download_file(MCF_URL, args.mcf_path)
    attr_path = download_file(ATTR_SNIPS_URL, args.attr_snips_path)

    data = load_json(mcf_path)
    snips = load_json(attr_path)

    forget_records, retain_records = official_split(
        data,
        forget_n=args.forget_n,
        retain_n=args.retain_n,
        seed=args.seed,
    )

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=dtype_from_str(args.dtype),
        device_map=args.device_map,
    )
    model.eval()
    model.config.use_cache = False

    record_results = []
    ppls = []
    missing = 0

    for i, r in enumerate(tqdm(forget_records, desc="Essence PPL")):
        rr = r["requested_rewrite"]

        subject = str(rr["subject"]).strip()
        target_new = str(rr["target_new"]["str"]).strip()
        target_true = str(rr["target_true"]["str"]).strip()

        essence_texts = get_essence_texts_for_record(r, snips)

        if not essence_texts:
            missing += 1
            record_results.append({
                "case_id": i,
                "subject": subject,
                "target_new": target_new,
                "target_true": target_true,
                "n_essence_texts": 0,
                "essence_ppl": None,
                "essence_texts": [],
            })
            continue

        joined = " ".join(essence_texts)
        ppl = text_perplexity(
            model,
            tok,
            joined,
            max_input_length=args.max_input_length,
        )

        if ppl is not None and math.isfinite(ppl):
            ppls.append(ppl)

        record_results.append({
            "case_id": i,
            "subject": subject,
            "target_new": target_new,
            "target_true": target_true,
            "n_essence_texts": len(essence_texts),
            "essence_ppl": ppl,
            "essence_texts": essence_texts,
        })

    arr = np.array(ppls, dtype=float)

    summary = {
        "model_dir": args.model_dir,
        "seed": args.seed,
        "forget_n": args.forget_n,
        "retain_n": args.retain_n,
        "sample_mode": "official_zero_unlearn",
        "metric": "ROME_MEMIT_essence_ppl",
        "max_input_length": args.max_input_length,
        "n_records": len(forget_records),
        "n_with_essence_texts": int(len(ppls)),
        "n_missing_essence_texts": int(missing),
        "PPL": float(arr.mean()) if len(arr) else None,
        "PPL_std_over_records": float(arr.std()) if len(arr) else None,
        "PPL_min": float(arr.min()) if len(arr) else None,
        "PPL_max": float(arr.max()) if len(arr) else None,
        "records": record_results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print("ROME/MEMIT Essence PPL")
    print("=" * 100)
    print(f"model_dir: {args.model_dir}")
    print(f"seed: {args.seed}")
    print(f"records: {len(forget_records)}")
    print(f"with essence texts: {len(ppls)}")
    print(f"missing essence texts: {missing}")
    print(f"PPL: {summary['PPL']}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
