#!/usr/bin/env python3
"""Build disjoint, provenance-checked RWKU matched-protection banks."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from rwku_artifact_access import (
    ArtifactAccessError,
    assert_content_disjoint,
    make_artifact,
    read_artifact,
    sha256_file,
    sha256_json,
    write_artifact,
)


FORBIDDEN_ORIGIN_ROLES = {
    "seen_fact_unseen_prompt_eval",
    "unseen_fact_eval",
    "official_locked_eval",
}
FORBIDDEN_PATH_MARKERS = (
    "forget_level1.json",
    "forget_level2.json",
    "forget_level3.json",
    "forget_mia.json",
    "retain_mia.json",
    "neighbor_level1.json",
    "neighbor_level2.json",
    "retain_mmlu.json",
    "retain_bbh.json",
    "truthful.json",
    "triviaqa.json",
    "fluency.json",
    "seen_fact_unseen_prompt_eval.json",
    "unseen_fact_eval.json",
    "official_locked_eval.json",
)


def normalize_key(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def validate_key_provenance(key_record: Mapping[str, Any]) -> None:
    required = {
        "key",
        "normalized_key",
        "origin_type",
        "origin_artifact_path",
        "origin_artifact_sha256",
        "visible_before_freeze",
    }
    missing = required - set(key_record)
    if missing:
        raise ArtifactAccessError(f"Protection key lacks provenance fields: {sorted(missing)}")
    if key_record["visible_before_freeze"] is not True:
        raise ArtifactAccessError("Protection key was not visible before checkpoint freeze")
    if key_record.get("origin_type") not in {
        "training_bundle",
        "generated_training_bundle",
        "independently_generated_entity_corpus",
        "target_independent_vocabulary",
    }:
        raise ArtifactAccessError(
            f"Forbidden protection-key origin: {key_record.get('origin_type')!r}"
        )
    origin_path = str(key_record["origin_artifact_path"]).casefold()
    if any(marker in origin_path for marker in FORBIDDEN_PATH_MARKERS):
        raise ArtifactAccessError(
            f"Protection key provenance points to held-out/evaluation data: {origin_path}"
        )
    if normalize_key(str(key_record["key"])) != str(key_record["normalized_key"]):
        raise ArtifactAccessError("Protection key normalized value is inconsistent")
    if key_record["origin_type"] == "target_independent_vocabulary" and not key_record.get("target_independent_vocabulary_revision"):
        raise ArtifactAccessError("Target-independent vocabulary key requires a revision")


def collect_visible_keys(
    training_bundle_path: Path,
    *,
    vocabulary_path: Path | None = None,
) -> List[Dict[str, Any]]:
    training = read_artifact(
        training_bundle_path,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    origin_type = (
        "generated_training_bundle"
        if "generated" in Path(training_bundle_path).name
        else "training_bundle"
    )
    keys: List[Dict[str, Any]] = []
    for view in training["payload"].get("views", []):
        fact_id = str(view.get("fact_id", ""))
        for value in [
            view.get("sensitive_answer_alias"),
            view.get("canonical_sensitive_answer"),
            *view.get("sensitive_answer_aliases", []),
        ]:
            if not value:
                continue
            record = {
                "key": str(value),
                "normalized_key": normalize_key(str(value)),
                "origin_type": origin_type,
                "origin_artifact_path": str(Path(training_bundle_path)),
                "origin_artifact_sha256": sha256_file(Path(training_bundle_path)),
                "source_fact_id": fact_id,
                "relation_category": view.get("relation_id"),
                "visible_before_freeze": True,
                "target_independent_vocabulary_revision": None,
            }
            validate_key_provenance(record)
            keys.append(record)
    if vocabulary_path is not None:
        with Path(vocabulary_path).open("r", encoding="utf-8") as handle:
            vocabulary = json.load(handle)
        if vocabulary.get("schema_version") != "rwku_target_independent_protection_vocabulary_v1" or vocabulary.get("created_without_rwku_evaluation_answers") is not True:
            raise ArtifactAccessError("Untrusted target-independent protection vocabulary")
        revision = str(vocabulary.get("revision", ""))
        for value in vocabulary.get("shared_answer_terms", []):
            record = {
                "key": str(value),
                "normalized_key": normalize_key(str(value)),
                "origin_type": "target_independent_vocabulary",
                "origin_artifact_path": str(Path(vocabulary_path)),
                "origin_artifact_sha256": sha256_file(Path(vocabulary_path)),
                "source_fact_id": None,
                "relation_category": None,
                "visible_before_freeze": True,
                "target_independent_vocabulary_revision": revision,
            }
            validate_key_provenance(record)
            keys.append(record)
    unique: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for record in keys:
        identity = (
            record["normalized_key"],
            record["origin_type"],
            str(record.get("source_fact_id")),
        )
        unique.setdefault(identity, record)
    return list(sorted(unique.values(), key=lambda item: (item["normalized_key"], item["origin_type"], str(item.get("source_fact_id")))))


def _load_source_rows(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in paths:
        lowered = str(path).casefold()
        if any(marker in lowered for marker in FORBIDDEN_PATH_MARKERS):
            raise ArtifactAccessError(f"RWKU evaluation source is forbidden for protection: {path}")
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        source_file_digest = sha256_file(Path(path))
        if isinstance(value, dict) and "artifact_role" in value:
            if value.get("artifact_role") in FORBIDDEN_ORIGIN_ROLES or value.get("evaluation_only") is True:
                raise ArtifactAccessError(f"Evaluation artifact is forbidden for protection: {path}")
            value = value.get("payload", {}).get("records", value.get("payload", {}).get("views", []))
        if not isinstance(value, list):
            raise ValueError(f"Protection source must contain a JSON list: {path}")
        for index, raw in enumerate(value):
            if not isinstance(raw, dict):
                raise ValueError(f"Non-object protection source row: {path}:{index}")
            expanded: List[Dict[str, Any]] = []
            rewrite = raw.get("requested_rewrite")
            if isinstance(rewrite, list) and rewrite:
                rewrite = rewrite[0]
            if isinstance(rewrite, dict):
                subject = str(rewrite.get("subject", ""))
                prompt_template = str(rewrite.get("prompt", ""))
                try:
                    prompt = prompt_template.format(subject)
                except (IndexError, KeyError, ValueError):
                    prompt = f"{prompt_template} {subject}".strip()
                for answer_role in ("target_true", "target_new"):
                    answer_value = rewrite.get(answer_role)
                    if isinstance(answer_value, Mapping):
                        answer_value = answer_value.get("str")
                    if answer_value:
                        expanded.append(
                            {
                                "prompt": prompt,
                                "answer": str(answer_value),
                                "subject": subject,
                                "source_answer_role": answer_role,
                            }
                        )
            if not expanded:
                expanded = [dict(raw)]
            for expanded_row in expanded:
                content = " ".join(
                    str(expanded_row.get(field, ""))
                    for field in ("text", "prompt", "query", "answer", "target", "target_true", "target_new")
                ).strip()
                if not content:
                    content = json.dumps(expanded_row, ensure_ascii=False, sort_keys=True)
                identity = {
                    "source_record": raw,
                    "normalized_record": expanded_row,
                }
                rows.append(
                    {
                        "record": expanded_row,
                        "content": content,
                        "normalized_content": normalize_key(content),
                        "content_sha256": sha256_json(identity),
                        "source_path": str(path),
                        "source_sha256": source_file_digest,
                        "source_row_index": index,
                    }
                )
    return rows


def build_matched_protection(
    *,
    training_bundle_path: Path,
    source_corpora: Sequence[Path],
    output_dir: Path,
    vocabulary_path: Path | None,
    split_seed: int,
    minimum_train_per_key: int,
    minimum_gate_per_key: int,
    strict: bool,
    tokenizer: Any | None = None,
) -> Dict[str, Any]:
    keys = collect_visible_keys(training_bundle_path, vocabulary_path=vocabulary_path)
    source_rows = _load_source_rows(source_corpora)
    training_source = read_artifact(
        training_bundle_path,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    target_subject = normalize_key(
        str(
            training_source.get("metadata", {}).get("subject")
            or next(
                (
                    view.get("subject")
                    for view in training_source["payload"].get("views", [])
                    if view.get("subject")
                ),
                "",
            )
        )
    )
    if target_subject:
        source_rows = [
            row
            for row in source_rows
            if target_subject not in row["normalized_content"]
        ]
    matches_by_key: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    key_provenance: MutableMapping[str, List[Dict[str, Any]]] = defaultdict(list)
    for key in keys:
        normalized = key["normalized_key"]
        key_provenance[normalized].append(key)
        for row in source_rows:
            if normalized and normalized in row["normalized_content"]:
                matches_by_key[normalized].append(row)

    # Partition every source content hash exactly once, independently of which
    # keys it matches. Alternating a deterministic global hash order keeps the
    # banks balanced while preventing a multi-key record from leaking across.
    unique_source_rows = {
        row["content_sha256"]: row for row in source_rows
    }
    globally_ordered = sorted(
        unique_source_rows.values(),
        key=lambda row: hashlib.sha256(
            f"{split_seed}:protection:{row['content_sha256']}".encode("utf-8")
        ).hexdigest(),
    )
    global_partition = {
        row["content_sha256"]: ("gate" if index % 2 == 0 else "train")
        for index, row in enumerate(globally_ordered)
    }

    train_by_hash: Dict[str, Dict[str, Any]] = {}
    gate_by_hash: Dict[str, Dict[str, Any]] = {}
    coverage: List[Dict[str, Any]] = []
    for normalized_key in sorted(key_provenance):
        unique_rows = {row["content_sha256"]: row for row in matches_by_key[normalized_key]}
        ordered = sorted(unique_rows.values(), key=lambda row: row["content_sha256"])
        gate_rows = [row for row in ordered if global_partition[row["content_sha256"]] == "gate"]
        train_rows = [row for row in ordered if global_partition[row["content_sha256"]] == "train"]
        for row in gate_rows:
            gate_by_hash.setdefault(row["content_sha256"], {**row, "matched_keys": []})["matched_keys"].append(normalized_key)
        for row in train_rows:
            train_by_hash.setdefault(row["content_sha256"], {**row, "matched_keys": []})["matched_keys"].append(normalized_key)
        token_rows = []
        if tokenizer is not None:
            for token_id in tokenizer.encode(key_provenance[normalized_key][0]["key"], add_special_tokens=False):
                token_rows.append({"token_id": int(token_id), "decoded_token_piece": tokenizer.decode([int(token_id)])})
        train_count = len(train_rows)
        actual_gate_count = len(gate_rows)
        sufficient = train_count >= minimum_train_per_key and actual_gate_count >= minimum_gate_per_key
        coverage.append(
            {
                "sensitive_answer": key_provenance[normalized_key][0]["key"],
                "normalized_key": normalized_key,
                "answer_aliases": sorted({item["key"] for item in key_provenance[normalized_key]}),
                "tokenizer_tokens": token_rows,
                "relation_category": sorted(
                    {
                        str(item["relation_category"])
                        for item in key_provenance[normalized_key]
                        if item.get("relation_category")
                    }
                ),
                "source_corpora": sorted({row["source_path"] for row in ordered}),
                "optimization_count": train_count,
                "gate_count": actual_gate_count,
                "coverage_status": "covered" if sufficient else "insufficient_coverage",
                "key_provenance": key_provenance[normalized_key],
            }
        )
    assert_content_disjoint(
        train_by_hash,
        gate_by_hash,
        left_name="matched protection train",
        right_name="matched protection gate",
    )
    insufficient = [row for row in coverage if row["coverage_status"] != "covered"]
    if strict and insufficient:
        raise ArtifactAccessError(
            "Insufficient matched-protection coverage for keys: "
            + ", ".join(row["normalized_key"] for row in insufficient)
        )
    metadata = dict(training_source.get("metadata", {}))
    protocol_label = training_source["protocol_label"]
    protocol_status = training_source["protocol_status"]
    train_payload = {"records": list(sorted(train_by_hash.values(), key=lambda row: row["content_sha256"])), "keys": keys}
    gate_payload = {"records": list(sorted(gate_by_hash.values(), key=lambda row: row["content_sha256"])), "keys": keys, "must_run_under_no_grad": True}
    coverage_payload = {"coverage": coverage, "warnings": [f"Insufficient coverage: {row['normalized_key']}" for row in insufficient]}
    artifacts = {
        "matched_protection_train.json": make_artifact("optimization_protection", train_payload, protocol_label=protocol_label, protocol_status=protocol_status, metadata=metadata),
        "matched_protection_gate.json": make_artifact("repair_selection_gate", gate_payload, protocol_label=protocol_label, protocol_status=protocol_status, metadata=metadata),
        "matched_protection_coverage.json": make_artifact("matched_protection_coverage", coverage_payload, protocol_label=protocol_label, protocol_status=protocol_status, metadata=metadata),
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for filename, artifact in artifacts.items():
        write_artifact(Path(output_dir) / filename, artifact)
    return {"artifacts": artifacts, "coverage": coverage, "insufficient": insufficient}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-bundle", type=Path, required=True)
    parser.add_argument("--source-corpus", type=Path, action="append", required=True)
    parser.add_argument("--protection-vocabulary", type=Path)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--minimum-train-per-key", type=int, default=1)
    parser.add_argument("--minimum-gate-per-key", type=int, default=1)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    result = build_matched_protection(
        training_bundle_path=args.training_bundle,
        source_corpora=args.source_corpus,
        output_dir=args.output_dir,
        vocabulary_path=args.protection_vocabulary,
        split_seed=args.split_seed,
        minimum_train_per_key=args.minimum_train_per_key,
        minimum_gate_per_key=args.minimum_gate_per_key,
        strict=args.strict,
        tokenizer=tokenizer,
    )
    print(json.dumps({"coverage": result["coverage"], "warnings": len(result["insufficient"])}, indent=2))


if __name__ == "__main__":
    main()
