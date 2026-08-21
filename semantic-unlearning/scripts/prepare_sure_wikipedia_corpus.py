#!/usr/bin/env python3
"""Materialize a deterministic real-Wikipedia corpus for SURE utility sweeps.

The repository's ``data/wikidata`` fixture has 200 short synthetic documents
and exists for the official PPL probe.  It is not a large utility corpus.  This
script streams a pinned Wikimedia Wikipedia snapshot, keeps real article
titles and text, and writes a separate Hugging Face ``DatasetDict``.  No
CounterFact file or benchmark probe is accepted by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple


PROTOCOL = "sure_external_wikipedia_corpus_v1"
DEFAULT_DATASET = "wikimedia/wikipedia"
DEFAULT_CONFIG = "20231101.en"
DEFAULT_REVISION = "ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619"
DEFAULT_SAMPLE_SIZE = 100_020
DEFAULT_SHUFFLE_BUFFER = 100_000
DEFAULT_MAX_CHARS = 16_384


def normalize_space(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_article(
    row: Mapping[str, Any], *, max_chars: int
) -> Optional[Dict[str, str]]:
    title = normalize_space(row.get("title"))
    text = str(row.get("text") or "").strip()
    if not title or not text:
        return None
    if max_chars > 0:
        text = text[:max_chars].rstrip()
    if not text:
        return None
    return {
        "id": str(row.get("id") or ""),
        "url": str(row.get("url") or ""),
        "title": title,
        "text": text,
    }


def collect_articles(
    rows: Iterable[Mapping[str, Any]],
    *,
    sample_size: int,
    max_chars: int,
) -> Tuple[List[Dict[str, str]], int]:
    selected: List[Dict[str, str]] = []
    seen_keys = set()
    source_rows_seen = 0
    for row in rows:
        source_rows_seen += 1
        article = normalize_article(row, max_chars=max_chars)
        if article is None:
            continue
        key = article["id"] or article["url"] or article["title"].casefold()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        selected.append(article)
        if len(selected) >= sample_size:
            break
    return selected, source_rows_seen


def content_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(
                dict(row),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER
    )
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0 or args.shuffle_buffer <= 0 or args.max_chars <= 0:
        raise ValueError("sample-size, shuffle-buffer, and max-chars must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite Wikipedia corpus: {output_dir}"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    from datasets import Dataset, DatasetDict, load_dataset

    stream = load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        revision=args.revision,
        streaming=True,
    )
    stream = stream.shuffle(seed=int(args.seed), buffer_size=int(args.shuffle_buffer))
    rows, source_rows_seen = collect_articles(
        stream,
        sample_size=int(args.sample_size),
        max_chars=int(args.max_chars),
    )
    if len(rows) != int(args.sample_size):
        raise RuntimeError(
            f"stream ended after {len(rows)} usable articles; "
            f"{args.sample_size} were required"
        )

    dataset = DatasetDict({"train": Dataset.from_list(rows)})
    dataset.save_to_disk(str(output_dir))
    receipt = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "dataset": str(args.dataset),
        "config": str(args.config),
        "revision": str(args.revision),
        "split": str(args.split),
        "streaming": True,
        "shuffle_seed": int(args.seed),
        "shuffle_buffer": int(args.shuffle_buffer),
        "requested_article_count": int(args.sample_size),
        "actual_article_count": len(rows),
        "source_rows_seen": int(source_rows_seen),
        "max_chars_per_article": int(args.max_chars),
        "columns": ["id", "url", "title", "text"],
        "content_sha256": content_sha256(rows),
        "counterfact_files_read": 0,
        "benchmark_probe_fields_read": 0,
    }
    (output_dir / "sure_wikipedia_corpus_receipt.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("Prepared SURE Wikipedia corpus:", output_dir)
    print("articles:", len(rows))
    print("content sha256:", receipt["content_sha256"])


if __name__ == "__main__":
    main()
