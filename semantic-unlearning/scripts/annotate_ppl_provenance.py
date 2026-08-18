#!/usr/bin/env python3
"""Attach exact canonical PPL token provenance to an evaluation JSON."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-json", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--max-input-length", type=int, default=100)
    a = p.parse_args()

    eval_path = Path(a.eval_json).resolve()
    wikidata = Path(a.wikidata_dir).resolve()
    result = json.loads(eval_path.read_text(encoding="utf-8"))
    ds = load_from_disk(str(wikidata))
    text = " ".join(ds["train"]["text"][:20])
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    encoded = tok(
        [text],
        return_tensors="pt",
        max_length=a.max_input_length,
        truncation=True,
    )
    token_ids = [int(x) for x in encoded["input_ids"][0].tolist()]
    serialized_ids = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    fixture = {
        "wikidata_dir": str(wikidata),
        "text_construction": "' '.join(train['text'][:20])",
        "max_input_length": int(a.max_input_length),
        "token_count": len(token_ids),
        "token_ids": token_ids,
        "token_ids_sha256": sha256_bytes(serialized_ids),
        "token_ids_sha256_serialization": "compact JSON list of decimal token IDs, UTF-8",
        "joined_text_sha256": sha256_bytes(text.encode("utf-8")),
        "canonical_fixture": True,
    }
    # Keep the original key for backward compatibility and expose the clearer
    # alias used by diagnostics/reporting code.
    result["ppl_fixture"] = fixture
    result["ppl_provenance"] = fixture
    eval_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Annotated PPL provenance:", eval_path)
    print("token_ids_sha256:", fixture["token_ids_sha256"])


if __name__ == "__main__":
    main()
