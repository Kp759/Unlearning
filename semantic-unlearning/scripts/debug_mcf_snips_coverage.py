#!/usr/bin/env python3

import argparse
import json
import random
import urllib.request
from pathlib import Path
from collections import Counter


MCF_URL = "https://memit.baulab.info/data/dsets/multi_counterfact.json"
ATTR_SNIPS_URL = "https://rome.baulab.info/data/dsets/attribute_snippets.json"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcf-path", default="data/multi_counterfact.json")
    ap.add_argument("--attr-snips-path", default="data/attribute_snippets.json")
    ap.add_argument("--forget-n", type=int, default=50)
    ap.add_argument("--retain-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mcf_path = download_file(MCF_URL, args.mcf_path)
    snips_path = download_file(ATTR_SNIPS_URL, args.attr_snips_path)

    data = load_json(mcf_path)
    snips = load_json(snips_path)

    forget_records, _ = official_split(
        data,
        forget_n=args.forget_n,
        retain_n=args.retain_n,
        seed=args.seed,
    )

    counts = Counter()
    examples = []

    print("=" * 100)
    print("SNIPS COVERAGE DEBUG")
    print("=" * 100)
    print("Total MCF records:", len(data))
    print("Top-level snips relation keys:", len(snips))
    print("Sample snips rel keys:", list(snips.keys())[:10])
    print("=" * 100)

    for i, r in enumerate(forget_records):
        rr = r["requested_rewrite"]

        subject = str(rr.get("subject", "")).strip()
        rel_id_raw = rr.get("relation_id", None)
        target_new = rr.get("target_new", {})
        target_true = rr.get("target_true", {})

        target_new_id_raw = target_new.get("id", None)
        target_true_id_raw = target_true.get("id", None)

        rel_candidates = [
            rel_id_raw,
            str(rel_id_raw),
        ]

        new_id_candidates = [
            target_new_id_raw,
            str(target_new_id_raw),
        ]

        true_id_candidates = [
            target_true_id_raw,
            str(target_true_id_raw),
        ]

        rel_key = None
        for k in rel_candidates:
            if k in snips:
                rel_key = k
                break

        if rel_key is None:
            counts["missing_relation"] += 1
            examples.append({
                "case": i,
                "reason": "missing_relation",
                "subject": subject,
                "rel_id": rel_id_raw,
                "target_new": target_new,
                "target_true": target_true,
            })
            continue

        rel_block = snips[rel_key]

        new_key = None
        for k in new_id_candidates:
            if k in rel_block:
                new_key = k
                break

        true_key = None
        for k in true_id_candidates:
            if k in rel_block:
                true_key = k
                break

        if new_key is None:
            counts["missing_target_new_id"] += 1
        else:
            rows = rel_block[new_key]
            exact = [x for x in rows if str(x.get("name", "")).strip() == subject]
            if exact:
                counts["new_exact_subject_match"] += 1
            else:
                counts["new_has_rows_but_no_subject_match"] += 1
                examples.append({
                    "case": i,
                    "reason": "new_has_rows_but_no_subject_match",
                    "subject": subject,
                    "rel_id": rel_id_raw,
                    "target_new_id": target_new_id_raw,
                    "target_new_str": target_new.get("str", None),
                    "n_rows": len(rows),
                    "sample_names": [x.get("name", None) for x in rows[:10]],
                    "sample_texts": [x.get("text", None) for x in rows[:3]],
                })

        if true_key is None:
            counts["missing_target_true_id"] += 1
        else:
            rows = rel_block[true_key]
            exact = [x for x in rows if str(x.get("name", "")).strip() == subject]
            if exact:
                counts["true_exact_subject_match"] += 1
            else:
                counts["true_has_rows_but_no_subject_match"] += 1

    print("Coverage counts:")
    for k, v in counts.items():
        print(f"{k}: {v}")

    print("\nExamples:")
    for ex in examples[:10]:
        print(json.dumps(ex, indent=2)[:2500])
        print("-" * 100)


if __name__ == "__main__":
    main()
