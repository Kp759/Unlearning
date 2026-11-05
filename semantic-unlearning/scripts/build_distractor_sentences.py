#!/usr/bin/env python3
"""Unrelated natural sentences to put in front of router training prompts (fit_linear_router.py
--distractors). MCF's official paraphrases prepend an unrelated Wikipedia sentence to the fact
("Enrollment in 2004 was 2,792 students. Jason Rebello, the"), and the router, trained only on
short content-free lead-ins, scored most of them near zero. These sentences come from a
training-visible corpus (the Wikipedia dump already used for the ROME / MEMIT covariance), never
from a benchmark's evaluation data.

    python scripts/build_distractor_sentences.py \\
        --source /project/yl258/kp759/datasets/wikipedia_20231101_100k --out data/distractor_sentences.json

--source: an HF dataset saved with save_to_disk (split "train", column "text"), a .txt file
(one document per line) or a .jsonl file with a "text" field. Kept sentences: 6-30 words, start
with a capital letter, end with a period, no newline, no brackets or markup, mostly ASCII;
deduplicated, then a seeded sample of --n. The per-fact filter (no sentence that names a
protected subject or answer) is applied at fit time, where the facts are known.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re

SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
BAD = re.compile(r"[\[\]{}<>|=_*#@\\]|https?:|www\.")


def iter_texts(source, limit_docs):
    path = Path(source)
    if path.is_dir():
        from datasets import load_from_disk
        data = load_from_disk(str(path))
        rows = data["train"] if "train" in data else data
        for k, text in enumerate(rows["text"]):
            if k >= limit_docs:
                return
            yield str(text)
    elif path.suffix == ".jsonl":
        with path.open() as handle:
            for k, line in enumerate(handle):
                if k >= limit_docs:
                    return
                yield str(json.loads(line).get("text", ""))
    else:
        with path.open() as handle:
            for k, line in enumerate(handle):
                if k >= limit_docs:
                    return
                yield line


def good(sentence, min_words=6, max_words=30):
    s = sentence.strip()
    words = s.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if not s[0].isupper() or not s.endswith(".") or "\n" in s or BAD.search(s):
        return False
    ascii_share = sum(c.isascii() for c in s) / len(s)
    return ascii_share > 0.97 and s.count("(") == s.count(")")


def extract(texts, n, seed, min_words=6, max_words=30):
    seen, pool = set(), []
    for text in texts:
        for paragraph in str(text).split("\n"):
            for sentence in SENTENCE.split(paragraph.strip()):
                s = " ".join(sentence.split())
                if s and s not in seen and good(s, min_words, max_words):
                    seen.add(s)
                    pool.append(s)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool[:n]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="data/distractor_sentences.json")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--limit-docs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sentences = extract(iter_texts(a.source, a.limit_docs), a.n, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": str(a.source), "seed": a.seed, "sentences": sentences}, indent=1) + "\n")
    print(json.dumps({"kept": len(sentences), "out": str(out), "examples": sentences[:3]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
