#!/usr/bin/env python3
"""Build leakage-safe synthetic MCF contexts from direct requests + Wikipedia.

The input is the stripped v8 training view, never the original CounterFact
file.  Two disjoint roles are emitted:

* same-subject template contexts receive the existing GA(true)+GD(new) loss;
* unrelated Wikipedia-title contexts preserve the frozen Base distribution.

Official paraphrases, neighborhoods, retain cases, and generation probes are
not accepted or read by this program.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import build_mcf_sure_target_aware_direct_split as direct_split


PROTOCOL = "sure_mcf_external_subject_and_locality_contexts_v1"
SUBJECT_CONTEXT_TEMPLATES = (
    "Complete this factual statement accurately: {direct_prompt}",
    "From an external knowledge task, finish: {direct_prompt}",
    "Provide the missing factual continuation for: {direct_prompt}",
    "Encyclopedic fact completion: {direct_prompt}",
)
LOCALITY_CONTEXT_TEMPLATES = (
    "External encyclopedia completion: {direct_prompt}",
    "Unrelated-subject factual completion: {direct_prompt}",
    "External article excerpt: {lead}\nSeparate factual completion: {direct_prompt}",
    "Reference context about {title}: {lead}\nComplete: {direct_prompt}",
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def clean_title(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def clean_lead(value: Any, *, lead_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:lead_chars].rstrip()


def load_wikipedia_records(path: Path) -> Tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Load title/text pairs, including the repository's Arrow fallback."""
    if not path.exists():
        raise FileNotFoundError(f"Wikipedia corpus not found: {path}")
    loader = "datasets.load_from_disk"
    fallback_error = None
    try:
        from datasets import load_from_disk

        loaded = load_from_disk(str(path.resolve()))
        train = loaded["train"] if hasattr(loaded, "keys") and "train" in loaded else loaded
        columns = list(getattr(train, "column_names", []))
        if "title" not in columns or "text" not in columns:
            raise ValueError("external Wikipedia corpus needs title and text columns")
        records = [
            {"title": str(row["title"]), "text": str(row["text"])} for row in train
        ]
        fingerprint = str(getattr(train, "_fingerprint", ""))
    except Exception as error:
        import pyarrow as pa
        import pyarrow.ipc as ipc

        split_root = path / "train" if (path / "train" / "state.json").is_file() else path
        state_path = split_root / "state.json"
        if not state_path.is_file():
            raise error
        state = json.loads(state_path.read_text(encoding="utf-8"))
        filenames = [entry.get("filename") for entry in state.get("_data_files", [])]
        if not filenames or not all(isinstance(name, str) for name in filenames):
            raise error
        records = []
        for filename in filenames:
            shard_path = split_root / filename
            with pa.memory_map(str(shard_path), "r") as source:
                try:
                    table = ipc.RecordBatchStreamReader(source).read_all()
                except pa.ArrowInvalid:
                    source.seek(0)
                    table = ipc.RecordBatchFileReader(source).read_all()
            if "title" not in table.column_names or "text" not in table.column_names:
                raise ValueError("external Wikipedia Arrow data needs title/text")
            titles = table.column("title").to_pylist()
            texts = table.column("text").to_pylist()
            records.extend(
                {"title": str(title or ""), "text": str(text or "")}
                for title, text in zip(titles, texts)
            )
        fingerprint = str(state.get("_fingerprint", ""))
        loader = "direct_pyarrow_save_to_disk_fallback"
        fallback_error = f"{type(error).__name__}: {error}"
    metadata = {
        "source": str(path.resolve()),
        "row_count": len(records),
        "fingerprint": fingerprint,
        "loader": loader,
        "loader_fallback_reason": fallback_error,
    }
    receipt_path = path / "sure_wikipedia_corpus_receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("Wikipedia corpus receipt must be a JSON object")
        if int(receipt.get("actual_article_count", -1)) != len(records):
            raise ValueError("Wikipedia corpus receipt row count mismatch")
        metadata["corpus_receipt"] = receipt
        metadata["corpus_receipt_sha256"] = sha256_bytes(receipt_path.read_bytes())
    else:
        metadata["corpus_receipt"] = None
        metadata["corpus_receipt_sha256"] = None
    return records, metadata


def build_subject_contexts(
    training_records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    contexts: List[Dict[str, Any]] = []
    for record_position, record in enumerate(training_records):
        rewrite = record["requested_rewrite"]
        direct_prompt = str(rewrite["prompt"]).format(str(rewrite["subject"]))
        for prompt_index, template in enumerate(SUBJECT_CONTEXT_TEMPLATES):
            prompt = template.format(direct_prompt=direct_prompt)
            contexts.append(
                {
                    "case_id": int(record["case_id"]),
                    "source_record_position": int(record_position),
                    "prompt_kind": "generated_subject",
                    "prompt_index": int(prompt_index),
                    "prompt_text": prompt,
                    "requested_rewrite": {
                        "prompt": "{}",
                        "subject": prompt,
                        "target_sensitive": {
                            "str": str(rewrite["target_true"]["str"])
                        },
                        "target_reference": {
                            "str": str(rewrite["target_new"]["str"])
                        },
                    },
                }
            )
    return contexts


def eligible_external_records(
    wikipedia_records: Sequence[Mapping[str, Any]],
    training_records: Sequence[Mapping[str, Any]],
    *,
    document_limit: int,
    exclude_first: int,
    seed: int,
    lead_chars: int,
) -> List[Dict[str, Any]]:
    forbidden = set()
    for record in training_records:
        rewrite = record["requested_rewrite"]
        forbidden.update(
            normalized_key(value)
            for value in (
                rewrite["subject"],
                rewrite["target_true"]["str"],
                rewrite["target_new"]["str"],
            )
        )
    indices = list(range(exclude_first, len(wikipedia_records)))
    random.Random(seed).shuffle(indices)
    selected_indices = indices[: min(document_limit, len(indices))]
    external: List[Dict[str, Any]] = []
    seen_titles = set()
    for document_index in selected_indices:
        raw = wikipedia_records[document_index]
        title = clean_title(raw.get("title"))
        key = normalized_key(title)
        if (
            not key
            or key in forbidden
            or key in seen_titles
            or key.startswith("list of ")
            or "disambiguation" in key
        ):
            continue
        lead = clean_lead(raw.get("text"), lead_chars=lead_chars)
        if not lead:
            continue
        seen_titles.add(key)
        external.append(
            {
                "document_index": int(document_index),
                "title": title,
                "lead": lead,
            }
        )
    return external


def build_locality_contexts(
    training_records: Sequence[Mapping[str, Any]],
    external_records: Sequence[Mapping[str, Any]],
    *,
    contexts_per_record: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if not external_records:
        raise ValueError("no eligible external Wikipedia subjects")
    contexts: List[Dict[str, Any]] = []
    seen_prompts = set()
    count = len(external_records)
    for record_position, record in enumerate(training_records):
        rewrite = record["requested_rewrite"]
        offset_digest = hashlib.sha256(
            f"{seed}:{int(record['case_id'])}".encode("ascii")
        ).digest()
        offset = int.from_bytes(offset_digest[:8], "big") % count
        cursor = 0
        for prompt_index in range(contexts_per_record):
            template_index = prompt_index % len(LOCALITY_CONTEXT_TEMPLATES)
            prompt = ""
            external: Mapping[str, Any] | None = None
            while cursor < count * 2:
                candidate = external_records[(offset + cursor) % count]
                cursor += 1
                title = str(candidate["title"])
                direct_prompt = str(rewrite["prompt"]).format(title)
                candidate_prompt = LOCALITY_CONTEXT_TEMPLATES[template_index].format(
                    direct_prompt=direct_prompt,
                    title=title,
                    lead=str(candidate["lead"]),
                )
                if candidate_prompt not in seen_prompts:
                    external = candidate
                    prompt = candidate_prompt
                    seen_prompts.add(prompt)
                    break
            if external is None:
                raise RuntimeError(
                    "could not construct enough unique external locality contexts"
                )
            title = str(external["title"])
            contexts.append(
                {
                    "context_id": len(contexts),
                    "case_id": int(record["case_id"]),
                    "source_record_position": int(record_position),
                    "prompt_index": int(prompt_index),
                    "template_index": int(template_index),
                    "external_document_index": int(external["document_index"]),
                    "external_title": title,
                    "prompt_text": prompt,
                }
            )
    return contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--wikipedia-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--corpus-document-limit", type=int, default=10_000)
    parser.add_argument("--exclude-first", type=int, default=20)
    parser.add_argument("--contexts-per-record", type=int, default=128)
    parser.add_argument("--lead-chars", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--require-corpus-protocol", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.corpus_document_limit <= 0
        or args.exclude_first < 0
        or args.contexts_per_record <= 0
        or args.lead_chars <= 0
    ):
        raise ValueError("external context sizes are invalid")
    training_path = Path(args.training_visible_path).resolve()
    training_bytes = training_path.read_bytes()
    training_records = json.loads(training_bytes)
    if not isinstance(training_records, list):
        raise ValueError("training-visible input must be a JSON list")
    direct_split.assert_direct_only_training_view(training_records)

    wikipedia_records, wikipedia_metadata = load_wikipedia_records(
        Path(args.wikipedia_dir).resolve()
    )
    if args.require_corpus_protocol:
        receipt = wikipedia_metadata.get("corpus_receipt")
        if not isinstance(receipt, Mapping) or receipt.get("protocol") != str(
            args.require_corpus_protocol
        ):
            raise RuntimeError(
                "external context corpus lacks required protocol: "
                f"{args.require_corpus_protocol}"
            )
    if len(wikipedia_records) - int(args.exclude_first) < int(
        args.corpus_document_limit
    ):
        raise RuntimeError(
            "external Wikipedia corpus cannot satisfy the locked document limit: "
            f"need {args.corpus_document_limit} after excluding "
            f"{args.exclude_first}, found "
            f"{max(0, len(wikipedia_records) - args.exclude_first)}"
        )
    external = eligible_external_records(
        wikipedia_records,
        training_records,
        document_limit=int(args.corpus_document_limit),
        exclude_first=int(args.exclude_first),
        seed=int(args.seed),
        lead_chars=int(args.lead_chars),
    )
    subject_contexts = build_subject_contexts(training_records)
    locality_contexts = build_locality_contexts(
        training_records,
        external,
        contexts_per_record=int(args.contexts_per_record),
        seed=int(args.seed),
    )
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "training_visible_sha256": sha256_bytes(training_bytes),
        "wikipedia": wikipedia_metadata,
        "builder": {
            "seed": int(args.seed),
            "corpus_document_limit": int(args.corpus_document_limit),
            "exclude_first": int(args.exclude_first),
            "contexts_per_record": int(args.contexts_per_record),
            "lead_chars": int(args.lead_chars),
            "eligible_external_subject_count": len(external),
            "subject_context_templates": list(SUBJECT_CONTEXT_TEMPLATES),
            "locality_context_templates": list(LOCALITY_CONTEXT_TEMPLATES),
        },
        "generated_subject_contexts": subject_contexts,
        "external_locality_contexts": locality_contexts,
        "data_boundary": {
            "source_counterfact_path_accepted": False,
            "official_paraphrases_read": 0,
            "official_neighborhoods_read": 0,
            "benchmark_retain_examples_read": 0,
            "generation_probes_read": 0,
        },
    }
    output_path = Path(args.output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite contexts: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("External MCF augmentation contexts:", output_path)
    print("generated same-subject contexts:", len(subject_contexts))
    print("external locality contexts:", len(locality_contexts))
    print("official paraphrase/neighborhood fields read: 0/0")


if __name__ == "__main__":
    main()
