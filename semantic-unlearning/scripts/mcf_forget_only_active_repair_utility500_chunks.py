#!/usr/bin/env python3
"""Run the external-utility MCF Stage-2 repair with 500 deterministic utility segments.

The local ``data/wikidata`` artifact contains only 200 train documents and the
first 20 documents are reserved for the official PPL probe.  Therefore a
500-example utility ablation cannot sample 500 distinct documents.

This wrapper keeps the existing Stage-2 implementation unchanged and replaces
only its utility loader.  It deterministically partitions eligible Wikidata
train documents (indices >= ``--utility-exclude-first``) into non-overlapping
word segments, chooses the largest segment size from 64, 48, 32, 24, 16 words
that yields at least ``--utility-num`` candidates, and samples 500 distinct
segments with the fixed utility seed.

Consequences:
  * no MCF retain/paraphrase/neighborhood data enter Stage 2;
  * no source document used by the first-20-text PPL probe enters utility;
  * utility examples are distinct segments rather than duplicated documents;
  * the underlying rank-8 forget-direction repair and utility covariance loss
    are exactly those in ``mcf_forget_only_active_repair_utility.py``.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_from_disk

import mcf_forget_only_active_repair_utility as utility


MIN_RESIDUAL_WORDS = 8
CANDIDATE_SEGMENT_WORDS = (64, 48, 32, 24, 16)


def _segment_documents(
    texts: List[Any],
    *,
    exclude_first: int,
    segment_words: int,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []
    for doc_index in range(exclude_first, len(texts)):
        raw = texts[doc_index]
        if not isinstance(raw, str) or not raw.strip():
            continue
        words = raw.split()
        segment_index = 0
        for start in range(0, len(words), segment_words):
            chunk = words[start : start + segment_words]
            if len(chunk) < MIN_RESIDUAL_WORDS:
                continue
            segments.append(
                {
                    "source_document_index": int(doc_index),
                    "segment_index": int(segment_index),
                    "word_start": int(start),
                    "word_end_exclusive": int(start + len(chunk)),
                    "segment_words": int(len(chunk)),
                    "text": " ".join(chunk),
                }
            )
            segment_index += 1
    return segments


def load_utility_segments(
    wikidata_dir: str,
    *,
    count: int,
    seed: int,
    exclude_first: int,
) -> Tuple[List[str], List[Dict[str, Any]], str]:
    path = Path(wikidata_dir).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Utility dataset not found: {path}")
    dataset = load_from_disk(str(path))
    if "train" not in dataset:
        raise ValueError("Utility dataset must contain a train split")
    train = dataset["train"]
    if "text" not in train.column_names:
        raise ValueError("Utility dataset train split must contain a text column")
    texts = list(train["text"])

    chosen_segment_words = None
    candidates: List[Dict[str, Any]] = []
    for segment_words in CANDIDATE_SEGMENT_WORDS:
        trial = _segment_documents(
            texts,
            exclude_first=exclude_first,
            segment_words=segment_words,
        )
        if len(trial) >= count:
            chosen_segment_words = segment_words
            candidates = trial
            break

    if chosen_segment_words is None:
        raise ValueError(
            f"Requested {count} distinct utility segments, but only "
            f"{len(trial)} candidates are available even at "
            f"{CANDIDATE_SEGMENT_WORDS[-1]} words/segment after excluding "
            f"the first {exclude_first} source documents"
        )

    rng = random.Random(seed)
    selected_positions = rng.sample(range(len(candidates)), k=count)
    selected = [candidates[i] for i in selected_positions]
    selected.sort(
        key=lambda x: (
            x["source_document_index"],
            x["segment_index"],
            x["word_start"],
        )
    )

    texts_out = [x["text"] for x in selected]
    provenance = [
        {
            k: v
            for k, v in x.items()
            if k != "text"
        }
        for x in selected
    ]
    digest_payload = [
        {**meta, "text": text}
        for meta, text in zip(provenance, texts_out)
    ]
    digest = hashlib.sha256(
        json.dumps(
            digest_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    distinct_docs = len({x["source_document_index"] for x in selected})
    print(
        "Utility segmentation: "
        f"{count} distinct non-overlapping segments sampled from "
        f"{distinct_docs} eligible source documents; "
        f"segment_words={chosen_segment_words}; seed={seed}"
    )
    return texts_out, provenance, digest


# ``utility.main`` resolves this module global at runtime, so replacing only
# the loader leaves the existing covariance construction and optimization math
# unchanged.
utility.load_utility_texts = load_utility_segments


if __name__ == "__main__":
    utility.main()
