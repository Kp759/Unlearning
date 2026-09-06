#!/usr/bin/env python3
"""Audit up to 100K local Wikipedia/Wikidata texts for registered MCF subjects.

This is a preservation/background audit only.  It never labels subject-containing
Wikipedia text as NONE, because such text may genuinely state a registered
forbidden fact.  The script reports overall subject-gate hit rate, per-subject
counts, and a small sample of matching texts for manual/semantic inspection.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mcf_zero_unlearn_official_eval as off
from mcf_sampling import sample_official_mcf_records

SEED = 1
FORGET_NUM = 50
RETAIN_NUM = 1000


def rr(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row["requested_rewrite"]
    return value[0] if isinstance(value, list) else value


def subject_pattern(subject: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(subject) + r"(?!\w)", re.IGNORECASE)


def _text_from_row(row: Any) -> str | None:
    if isinstance(row, str):
        return row.strip() or None
    if not isinstance(row, Mapping):
        return None
    for key in ("text", "content", "sentence", "document", "body"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def iter_local_texts(path: Path) -> Iterable[str]:
    # Hugging Face datasets saved with save_to_disk.
    try:
        from datasets import Dataset, DatasetDict, load_from_disk
        obj = load_from_disk(str(path))
        if isinstance(obj, DatasetDict):
            # Prefer train, otherwise deterministic split order.
            keys = ["train"] if "train" in obj else sorted(obj.keys())
            for key in keys:
                for row in obj[key]:
                    text = _text_from_row(row)
                    if text:
                        yield text
            return
        if isinstance(obj, Dataset):
            for row in obj:
                text = _text_from_row(row)
                if text:
                    yield text
            return
    except Exception:
        pass

    files: list[Path]
    if path.is_file():
        files = [path]
    else:
        files = sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in {".json", ".jsonl", ".txt"}
        )
    for file in files:
        if file.suffix.lower() == ".txt":
            for line in file.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip():
                    yield line.strip()
        elif file.suffix.lower() == ".jsonl":
            with file.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        text = _text_from_row(json.loads(line))
                    except Exception:
                        text = None
                    if text:
                        yield text
        else:
            try:
                obj = json.loads(file.read_text(encoding="utf-8", errors="ignore"))
            except Exception:
                continue
            rows = obj if isinstance(obj, list) else obj.get("data", []) if isinstance(obj, dict) else []
            for row in rows:
                text = _text_from_row(row)
                if text:
                    yield text


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mcf-path", required=True)
    p.add_argument("--wiki-path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--max-samples", type=int, default=100000)
    p.add_argument("--examples-per-subject", type=int, default=5)
    args = p.parse_args()

    data = json.loads(Path(args.mcf_path).read_text(encoding="utf-8"))
    forget, _retain = sample_official_mcf_records(
        data, FORGET_NUM, RETAIN_NUM, SEED, strict=True
    )
    forget = [off.normalize_record(x) for x in forget]
    subjects = [str(rr(x)["subject"]) for x in forget]
    patterns = {s: subject_pattern(s) for s in subjects}

    rng = random.Random(SEED)
    # Deterministic reservoir sample so a huge local dataset is not fully kept in memory.
    reservoir: list[str] = []
    seen = 0
    for text in iter_local_texts(Path(args.wiki_path).resolve()):
        seen += 1
        if len(reservoir) < int(args.max_samples):
            reservoir.append(text)
        else:
            j = rng.randrange(seen)
            if j < int(args.max_samples):
                reservoir[j] = text

    per_subject = {s: 0 for s in subjects}
    examples = {s: [] for s in subjects}
    any_hit = 0
    multi_subject = 0
    for text in reservoir:
        hits = [s for s, pat in patterns.items() if pat.search(text)]
        if hits:
            any_hit += 1
        if len(hits) > 1:
            multi_subject += 1
        for s in hits:
            per_subject[s] += 1
            if len(examples[s]) < int(args.examples_per_subject):
                examples[s].append(text)

    active = {s: n for s, n in per_subject.items() if n > 0}
    payload = {
        "schema_version": 1,
        "kind": "mcf_seed1_wikipedia_subject_background_audit",
        "max_samples_requested": int(args.max_samples),
        "source_rows_seen": int(seen),
        "sampled_rows": len(reservoir),
        "registered_subjects": len(subjects),
        "rows_with_any_registered_subject": int(any_hit),
        "subject_gate_hit_rate": (any_hit / len(reservoir)) if reservoir else None,
        "rows_with_multiple_registered_subjects": int(multi_subject),
        "subjects_observed": len(active),
        "per_subject_counts_nonzero": active,
        "examples_nonzero": {s: examples[s] for s in active},
        "labeling_contract": {
            "subject_containing_rows_labeled_NONE": False,
            "reason": "A Wikipedia row mentioning a registered subject may state the forbidden relation; this audit does not infer permission labels.",
        },
    }
    out = Path(args.out).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "sampled_rows": len(reservoir),
        "rows_with_any_registered_subject": any_hit,
        "subject_gate_hit_rate_pct": 100.0 * any_hit / len(reservoir) if reservoir else None,
        "subjects_observed": len(active),
        "output": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
