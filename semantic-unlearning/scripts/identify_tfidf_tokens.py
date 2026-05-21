#!/usr/bin/env python3
import argparse, json, math
from collections import Counter
from pathlib import Path
import yaml
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def text(row):
    return f"Question: {row['question']} Answer: {row['answer']}"

def df(ds, tok):
    c = Counter(); n = 0
    for row in tqdm(ds, desc="docfreq"):
        for tid in set(tok.encode(text(row), add_special_tokens=False)):
            c[int(tid)] += 1
        n += 1
    return c, n

def clean(s):
    return s.replace("Ġ", "").replace("▁", "").strip()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--forget-split", default=None)
    p.add_argument("--retain-split", default=None)
    p.add_argument("--model-name", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--min-forget-count", type=int, default=2)
    p.add_argument("--max-retain-count", type=int, default=8)
    p.add_argument("--max-retain-ratio", type=float, default=0.003)
    p.add_argument("--min-contrast", type=float, default=5.0)
    p.add_argument("--min-token-len", type=int, default=2)
    p.add_argument("--top-k", type=int, default=1000)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    forget_split = args.forget_split or cfg["data"]["forget_split"]
    retain_split = args.retain_split or cfg["data"]["retain_split"]
    model_name = args.model_name or cfg["model"]["name"]
    out_dir = Path(cfg["output"]["dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / "semantic_tokens_tfidf.json"

    tok = AutoTokenizer.from_pretrained(model_name)
    fds = load_dataset("locuslab/TOFU", name=forget_split, split="train")
    rds = load_dataset("locuslab/TOFU", name=retain_split, split="train")
    fdf, nf = df(fds, tok)
    rdf, nr = df(rds, tok)

    special = {x for x in [tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id] if x is not None}
    total = nf + nr
    rows = []

    for tid, fc in fdf.items():
        tid = int(tid); fc = int(fc); rc = int(rdf.get(tid, 0))
        if tid in special or fc < args.min_forget_count or rc > args.max_retain_count:
            continue
        fr = fc / max(1, nf)
        rr = rc / max(1, nr)
        if rr > args.max_retain_ratio:
            continue
        contrast = (fr + 1e-8) / (rr + 1e-8)
        if contrast < args.min_contrast:
            continue
        s = tok.decode([tid])
        if len(clean(s)) < args.min_token_len:
            continue
        idf = math.log((total + 1) / (fc + rc + 1)) + 1.0
        ftfidf = fr * idf
        rtfidf = rr * idf
        score = ftfidf * math.log1p(contrast)
        rows.append({
            "token_id": tid,
            "token_str": s,
            "freq_forget": fc,
            "freq_retain": rc,
            "forget_ratio": float(fr),
            "retain_ratio": float(rr),
            "idf": float(idf),
            "forget_tfidf": float(ftfidf),
            "retain_tfidf": float(rtfidf),
            "contrast_score": float(contrast),
            "tfidf_score": float(score),
            "differential": float(score),
            "mean_forget_score": 0.0,
            "mean_retain_score": 0.0,
            "best_layer": -1,
            "source": "tfidf_forget_candidate"
        })

    rows.sort(key=lambda x: (-x["tfidf_score"], -x["forget_tfidf"], -x["contrast_score"], x["freq_retain"], x["token_id"]))
    if args.top_k and args.top_k > 0:
        rows = rows[:args.top_k]

    out = {
        "method": "tfidf_forget_candidates",
        "forget_split": forget_split,
        "retain_split": retain_split,
        "target_model": model_name,
        "n_semantic_tokens": len(rows),
        "token_ids": [int(x["token_id"]) for x in rows],
        "token_strings": [x["token_str"] for x in rows],
        "semantic_tokens": rows,
        "filter_config": vars(args)
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[done] saved {len(rows)} TF-IDF tokens to {out_path}")
    for x in rows[:50]:
        print(f"{x['token_id']:>8} | {repr(x['token_str'])} | f={x['freq_forget']} r={x['freq_retain']} | score={x['tfidf_score']:.6f}")

if __name__ == "__main__":
    main()
