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


def normalize_attribute_snips(snips):
    """
    attribute_snippets.json is usually a list:
      [{"relation_id": ..., "target_id": ..., "samples": [...]}]

    Convert it to:
      nested[relation_id][target_id] = samples
    """
    if isinstance(snips, dict):
        return snips

    nested = {}
    for item in snips:
        rel = str(item.get("relation_id"))
        tid = str(item.get("target_id"))
        samples = item.get("samples", [])

        if rel not in nested:
            nested[rel] = {}

        nested[rel][tid] = samples

    return nested


def official_split(data, forget_n, retain_n, seed):
    rng = random.Random(seed)
    half = len(data) // 2
    retain_pool = data[:half]
    forget_pool = data[half:]
    forget_records = rng.sample(forget_pool, k=min(forget_n, len(forget_pool)))
    retain_records = rng.sample(retain_pool, k=min(retain_n, len(retain_pool)))
    return forget_records, retain_records


def get_key(d, raw_key):
    for k in [raw_key, str(raw_key)]:
        if k in d:
            return k
    return None


def get_attr_snip_texts(record, snips):
    rr = record["requested_rewrite"]

    subject = str(rr.get("subject", "")).strip()
    rel_id = rr.get("relation_id", None)

    target_new = rr.get("target_new", {})
    target_true = rr.get("target_true", {})

    target_new_id = target_new.get("id", None)
    target_true_id = target_true.get("id", None)

    rel_key = get_key(snips, rel_id)
    if rel_key is None:
        return [], "missing_relation"

    rel_block = snips[rel_key]

    # 1. Official-style: target_new id + exact subject name.
    new_key = get_key(rel_block, target_new_id)
    if new_key is not None:
        rows = rel_block[new_key]
        exact = [
            str(x.get("text", "")).strip()
            for x in rows
            if str(x.get("name", "")).strip() == subject and str(x.get("text", "")).strip()
        ]
        if exact:
            return exact, "target_new_exact_subject"

        consistency = [
            str(x.get("text", "")).strip()
            for x in rows
            if str(x.get("text", "")).strip()
        ]
        if consistency:
            return consistency, "target_new_consistency_fallback"

    # 2. Fallback: target_true id + exact subject name.
    true_key = get_key(rel_block, target_true_id)
    if true_key is not None:
        rows = rel_block[true_key]
        exact = [
            str(x.get("text", "")).strip()
            for x in rows
            if str(x.get("name", "")).strip() == subject and str(x.get("text", "")).strip()
        ]
        if exact:
            return exact, "target_true_exact_subject_fallback"

        consistency = [
            str(x.get("text", "")).strip()
            for x in rows
            if str(x.get("text", "")).strip()
        ]
        if consistency:
            return consistency, "target_true_consistency_fallback"

    return [], "missing_all"


@torch.no_grad()
def official_like_perplexity(model, tok, text, max_input_length=100):
    inputs = tok(
        [text],
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True,
    ).to(model.device)

    if inputs["input_ids"].shape[1] < 2:
        return None

    logits = torch.nn.functional.log_softmax(model(**inputs).logits, dim=2)

    log_probs = torch.gather(
        logits[:, :-1, :],
        2,
        inputs["input_ids"][:, 1:, None],
    )[0]

    ppl = torch.exp(
        -1 / inputs["input_ids"].size(1) * log_probs.sum()
    ).item()

    return float(ppl)


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
    snips = normalize_attribute_snips(load_json(attr_path))

    forget_records, _ = official_split(
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

    ppls = []
    record_results = []
    source_counts = {}

    for i, r in enumerate(tqdm(forget_records, desc="AttributeSnips PPL")):
        rr = r["requested_rewrite"]
        subject = str(rr["subject"]).strip()
        target_new = str(rr["target_new"]["str"]).strip()
        target_true = str(rr["target_true"]["str"]).strip()

        texts, source = get_attr_snip_texts(r, snips)
        source_counts[source] = source_counts.get(source, 0) + 1

        if not texts:
            record_results.append({
                "case_id": i,
                "subject": subject,
                "target_new": target_new,
                "target_true": target_true,
                "source": source,
                "n_texts": 0,
                "PPL": None,
                "texts": [],
            })
            continue

        joined = " ".join(texts)
        ppl = official_like_perplexity(
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
            "source": source,
            "n_texts": len(texts),
            "PPL": ppl,
            "texts": texts,
        })

    arr = np.array(ppls, dtype=float)

    summary = {
        "model_dir": args.model_dir,
        "seed": args.seed,
        "forget_n": args.forget_n,
        "retain_n": args.retain_n,
        "metric": "attribute_snippets_ppl_with_fallback",
        "max_input_length": args.max_input_length,
        "n_records": len(forget_records),
        "n_with_texts": int(len(ppls)),
        "source_counts": source_counts,
        "PPL": float(arr.mean()) if len(arr) else None,
        "PPL_std_over_records": float(arr.std()) if len(arr) else None,
        "records": record_results,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=" * 100)
    print("AttributeSnippets PPL with fallback")
    print("=" * 100)
    print("model:", args.model_dir)
    print("seed:", args.seed)
    print("n_with_texts:", len(ppls))
    print("source_counts:", source_counts)
    print("PPL:", summary["PPL"])
    print("saved:", out)


if __name__ == "__main__":
    main()
