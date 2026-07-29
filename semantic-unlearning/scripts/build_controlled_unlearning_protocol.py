#!/usr/bin/env python3
"""Build leakage-controlled five-fold protocols for MCF, ZsRE, and TOFU.

This is a method-level cross-validation protocol, which is important for
machine unlearning:

* ``train`` deletion requests are used to develop candidate hyperparameters;
* ``validation`` deletion requests are applied to a fresh base model and are
  scored only by Judge A;
* after a candidate is frozen, ``final_apply`` deletion requests are applied
  to another fresh base model and scored on distinct prompts by Judge B.

A held-out deletion request is therefore never used for tuning, but it is
provided to the final unlearning run.  Expecting a model to forget a fact for
which it never received a deletion request would not be a valid unlearning
test.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from controlled_unlearning_protocol import (
    N_FOLDS,
    SCHEMA_VERSION,
    PromptCase,
    RecordRef,
    assert_partition_disjoint,
    assert_prompt_partitions_disjoint,
    build_record_folds,
    finalize_bundle,
    generic_prompt_variants,
    load_json_or_jsonl,
    load_json_or_jsonl_url,
    prompt_case_dict,
    refs_dict,
    sha256_file,
    sha256_json,
    stable_id,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
MCF_URL = "https://memit.baulab.info/data/dsets/multi_counterfact.json"
ZSRE_URL = "https://memit.baulab.info/data/dsets/zsre_mend_eval.json"
TOFU_DATASET = "locuslab/TOFU"
TOFU_RAW_ROOT = (
    "https://huggingface.co/datasets/locuslab/TOFU/resolve/main"
)
DEFAULT_COUNTS = {
    "mcf": (50, 1000),
    "zsre": (50, 1000),
    "tofu": (200, 1000),
}
TOFU_UTILITY_SPLITS = ("real_authors", "world_facts")


def normalize_text(value: Any) -> str:
    return " ".join(str(value).casefold().split())


def sanitize_component(value: str) -> str:
    rendered = re.sub(r"[^a-zA-Z0-9_.-]+", "_", value.strip())
    return rendered.strip("._") or "value"


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


def _download_json_if_missing(path: Path, url: str) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_json_or_jsonl_url(url)
    write_json(path, rows)
    return path


def _format_template(template: Any, subject: str) -> str:
    text = str(template)
    return text.format(subject) if "{}" in text else text


def _target_string(block: Any, *, field: str) -> str:
    if not isinstance(block, Mapping) or not str(block.get("str", "")).strip():
        raise ValueError(f"Missing non-empty {field}.str")
    return str(block["str"]).strip()


def _mcf_rewrite(record: Mapping[str, Any]) -> Mapping[str, Any]:
    rewrite: Any = record.get("requested_rewrite")
    if isinstance(rewrite, list):
        if not rewrite:
            raise ValueError("MCF requested_rewrite list is empty")
        rewrite = rewrite[0]
    if not isinstance(rewrite, Mapping):
        raise ValueError("MCF record lacks requested_rewrite")
    for field in ("prompt", "subject", "target_new", "target_true"):
        if field not in rewrite:
            raise ValueError(f"MCF requested_rewrite lacks {field}")
    return rewrite


def _indexed_official_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    forget_num: int,
    retain_num: int,
    seed: int,
    semantic_key: Optional[Any] = None,
) -> Tuple[
    List[Tuple[int, Dict[str, Any]]],
    List[Tuple[int, Dict[str, Any]]],
    Dict[str, Any],
]:
    indexed = [(index, dict(row)) for index, row in enumerate(rows)]
    half = len(indexed) // 2
    retain_pool = indexed[:half]
    forget_pool = indexed[half:]
    if len(forget_pool) < forget_num or len(retain_pool) < retain_num:
        raise ValueError(
            "Official half split is undersized: "
            f"forget_pool={len(forget_pool)}, retain_pool={len(retain_pool)}, "
            f"requested forget={forget_num}, retain={retain_num}"
        )
    rng = random.Random(seed)
    forget_initial = rng.sample(forget_pool, k=forget_num)
    retain_initial = rng.sample(retain_pool, k=retain_num)
    if semantic_key is None:
        return forget_initial, retain_initial, {
            "semantic_duplicates_replaced": 0,
            "cross_role_conflicts_replaced": 0,
        }

    def fill_unique(
        initial: Sequence[Tuple[int, Dict[str, Any]]],
        pool: Sequence[Tuple[int, Dict[str, Any]]],
        count: int,
        forbidden: set[str],
    ) -> Tuple[List[Tuple[int, Dict[str, Any]]], int]:
        selected: List[Tuple[int, Dict[str, Any]]] = []
        seen = set(forbidden)
        removed = 0
        selected_indices: set[int] = set()
        for item in initial:
            key = str(semantic_key(item[1]))
            if key in seen:
                removed += 1
                continue
            selected.append(item)
            selected_indices.add(item[0])
            seen.add(key)
        replacements = [
            item for item in pool if item[0] not in selected_indices
        ]
        rng.shuffle(replacements)
        for item in replacements:
            if len(selected) >= count:
                break
            key = str(semantic_key(item[1]))
            if key in seen:
                continue
            selected.append(item)
            seen.add(key)
        if len(selected) != count:
            raise ValueError(
                f"Could not obtain {count} semantically unique records after "
                "removing train/test role conflicts"
            )
        return selected, removed

    forget, forget_removed = fill_unique(
        forget_initial,
        forget_pool,
        forget_num,
        set(),
    )
    forget_keys = {
        str(semantic_key(record))
        for _, record in forget
    }
    retain, retain_removed = fill_unique(
        retain_initial,
        retain_pool,
        retain_num,
        forget_keys,
    )
    return forget, retain, {
        "semantic_duplicates_replaced": forget_removed,
        "cross_role_or_retain_duplicates_replaced": retain_removed,
    }


def _mcf_semantic_key(record: Mapping[str, Any]) -> str:
    rewrite = _mcf_rewrite(record)
    return sha256_json(
        {
            "subject": normalize_text(rewrite["subject"]),
            "relation_id": str(rewrite.get("relation_id", "")),
            "target_new": normalize_text(
                _target_string(rewrite["target_new"], field="target_new")
            ),
            "target_true": normalize_text(
                _target_string(rewrite["target_true"], field="target_true")
            ),
        }
    )


def _zsre_semantic_key(record: Mapping[str, Any]) -> str:
    return sha256_json(
        {
            "subject": normalize_text(record.get("subject", "")),
            "src": normalize_text(record.get("src", "")),
            "answers": sorted(
                normalize_text(value)
                for value in record.get("answers", [])
            ),
        }
    )


def _compact_record(
    *,
    dataset: str,
    role: str,
    source_index: int,
    record_id: str,
    group_id: str,
    prompt: str,
    subject: str,
    sensitive_answers: Sequence[str],
    acceptable_answers: Sequence[str],
    paraphrases: Sequence[str],
    raw_record: Mapping[str, Any],
    metadata: Optional[Mapping[str, Any]] = None,
    locality: Sequence[Mapping[str, Any]] = (),
) -> Dict[str, Any]:
    if role not in {"forget", "retain", "utility"}:
        raise ValueError(f"Unsupported record role {role!r}")
    return {
        "dataset": dataset,
        "role": role,
        "source_index": int(source_index),
        "record_id": record_id,
        "group_id": group_id,
        "content_sha256": sha256_json(raw_record),
        "prompt": str(prompt),
        "subject": str(subject),
        "sensitive_answers": [
            str(value) for value in sensitive_answers if str(value).strip()
        ],
        "acceptable_answers": [
            str(value) for value in acceptable_answers if str(value).strip()
        ],
        "paraphrases": [
            str(value) for value in paraphrases if str(value).strip()
        ],
        "locality": [dict(value) for value in locality],
        "metadata": dict(metadata or {}),
        # Kept only so stage-specific input files can exactly reproduce the
        # benchmark adapters. Bundles below expose only the compact fields.
        "_raw_record": copy.deepcopy(dict(raw_record)),
    }


def _public_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(value)
        for key, value in record.items()
        if key != "_raw_record"
    }


def load_mcf_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = _download_json_if_missing(
        resolve_project_path(args.mcf_path),
        args.mcf_url,
    )
    rows = load_json_or_jsonl(path)
    forget_raw, retain_raw, sampling_audit = _indexed_official_sample(
        rows,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        semantic_key=_mcf_semantic_key,
    )

    def convert(
        indexed_rows: Sequence[Tuple[int, Dict[str, Any]]],
        role: str,
    ) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for source_index, row in indexed_rows:
            rewrite = _mcf_rewrite(row)
            subject = str(rewrite["subject"]).strip()
            target_new = _target_string(
                rewrite["target_new"],
                field="target_new",
            )
            target_true = _target_string(
                rewrite["target_true"],
                field="target_true",
            )
            prompt = _format_template(rewrite["prompt"], subject)
            paraphrases = [
                _format_template(value, subject)
                for value in row.get("paraphrase_prompts", [])
            ]
            relation_id = str(rewrite.get("relation_id", "unknown"))
            record_id = f"mcf:{row.get('case_id', source_index)}"
            # Subject grouping is intentionally conservative: two relations
            # about the same entity cannot cross folds.
            group_id = stable_id("mcf-subject", normalize_text(subject))
            locality = [
                {
                    "prompt": str(value),
                    "answer": target_true,
                    "source": "neighborhood_prompts",
                }
                for value in row.get("neighborhood_prompts", [])
                if str(value).strip()
            ]
            if role == "forget":
                sensitive = [target_new]
                acceptable = [target_true, "Unknown"]
            else:
                sensitive = []
                # This matches the existing CounterFact/ZeroUnlearn retain
                # metric: target_new remains preferred on retained edits.
                acceptable = [target_new]
            converted.append(
                _compact_record(
                    dataset="mcf",
                    role=role,
                    source_index=source_index,
                    record_id=record_id,
                    group_id=group_id,
                    prompt=prompt,
                    subject=subject,
                    sensitive_answers=sensitive,
                    acceptable_answers=acceptable,
                    paraphrases=paraphrases,
                    raw_record=row,
                    metadata={
                        "case_id": row.get("case_id", source_index),
                        "relation_id": relation_id,
                        "target_new": target_new,
                        "target_true": target_true,
                        "semantic_key_sha256": _mcf_semantic_key(row),
                    },
                    locality=locality,
                )
            )
        return converted

    return [*convert(forget_raw, "forget"), *convert(retain_raw, "retain")], {
        "kind": "local_json",
        "path": str(path),
        "sha256": sha256_file(path),
        "url": args.mcf_url,
        "sampling": "ZeroUnlearn official: forget=second half, retain=first half",
        "sampling_audit": sampling_audit,
    }


def load_zsre_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    path = _download_json_if_missing(
        resolve_project_path(args.zsre_path),
        args.zsre_url,
    )
    rows = load_json_or_jsonl(path)
    forget_raw, retain_raw, sampling_audit = _indexed_official_sample(
        rows,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        semantic_key=_zsre_semantic_key,
    )

    def convert(
        indexed_rows: Sequence[Tuple[int, Dict[str, Any]]],
        role: str,
    ) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        required = ("src", "subject", "answers", "rephrase", "loc", "loc_ans")
        for source_index, row in indexed_rows:
            missing = [field for field in required if field not in row]
            if missing:
                raise ValueError(
                    f"ZsRE record {source_index} misses fields {missing}"
                )
            answers = row["answers"]
            if not isinstance(answers, list) or not answers:
                raise ValueError(f"ZsRE record {source_index} has no answers")
            subject = str(row["subject"]).strip()
            answer_values = [
                str(value).strip()
                for value in answers
                if str(value).strip()
            ]
            primary = answer_values[0]
            record_id = f"zsre:{source_index}"
            group_id = stable_id("zsre-subject", normalize_text(subject))
            if role == "forget":
                sensitive = answer_values
                acceptable = ["Unknown"]
            else:
                sensitive = []
                acceptable = answer_values
            converted.append(
                _compact_record(
                    dataset="zsre",
                    role=role,
                    source_index=source_index,
                    record_id=record_id,
                    group_id=group_id,
                    prompt=str(row["src"]),
                    subject=subject,
                    sensitive_answers=sensitive,
                    acceptable_answers=acceptable,
                    # ZsRE has exactly one official rephrase. It is held back
                    # by prompt_cases_for_record for Judge B.
                    paraphrases=[str(row["rephrase"])],
                    raw_record=row,
                    metadata={
                        "answers": answer_values,
                        "neutral_target": "Unknown",
                        "semantic_key_sha256": _zsre_semantic_key(row),
                    },
                    locality=[
                        {
                            "prompt": str(row["loc"]),
                            "answer": str(row["loc_ans"]),
                            "source": "loc",
                        }
                    ],
                )
            )
        return converted

    return [*convert(forget_raw, "forget"), *convert(retain_raw, "retain")], {
        "kind": "local_json",
        "path": str(path),
        "sha256": sha256_file(path),
        "url": args.zsre_url,
        "sampling": "ZeroUnlearn official: forget=second half, retain=first half",
        "neutral_target": "Unknown",
        "sampling_audit": sampling_audit,
    }


def _load_tofu_split(
    split: str,
    *,
    data_dir: Optional[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if data_dir is not None:
        candidates = [
            data_dir / f"{split}.json",
            data_dir / f"{split}.jsonl",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            raise FileNotFoundError(
                f"No {split}.json or {split}.jsonl in {data_dir}"
            )
        return load_json_or_jsonl(path), {
            "kind": "local_json",
            "path": str(path),
            "sha256": sha256_file(path),
        }
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is in requirements
        raise RuntimeError(
            "The datasets package is required when --tofu-data-dir is omitted"
        ) from exc
    try:
        rows = [
            dict(row)
            for row in load_dataset(TOFU_DATASET, name=split, split="train")
        ]
        source_kind = "huggingface_datasets"
        source_url = None
    except (ConnectionError, OSError, RuntimeError, ValueError) as exc:
        # datasets/fsspec combinations can reject the Hub's recursive glob
        # even though the versioned raw JSON is available. The dataset card
        # declares one <config>.json file per configuration.
        source_url = f"{TOFU_RAW_ROOT}/{split}.json"
        print(
            f"[warning] datasets loader failed for TOFU/{split}: {exc}. "
            f"Falling back to {source_url}"
        )
        rows = load_json_or_jsonl_url(source_url)
        source_kind = "huggingface_raw_json"
    return rows, {
        "kind": source_kind,
        "dataset": TOFU_DATASET,
        "config": split,
        "url": source_url,
        "content_sha256": sha256_json(rows),
    }


def _qa_key(row: Mapping[str, Any]) -> Tuple[str, str]:
    if "question" not in row or "answer" not in row:
        raise ValueError("TOFU row lacks question or answer")
    return normalize_text(row["question"]), normalize_text(row["answer"])


def _map_tofu_rows_to_full(
    rows: Sequence[Mapping[str, Any]],
    full_rows: Sequence[Mapping[str, Any]],
) -> List[int]:
    positions: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for index, row in enumerate(full_rows):
        positions[_qa_key(row)].append(index)
    consumed: Dict[Tuple[str, str], int] = defaultdict(int)
    mapped: List[int] = []
    for row in rows:
        key = _qa_key(row)
        candidates = positions.get(key, [])
        offset = consumed[key]
        if offset >= len(candidates):
            raise ValueError(
                "Could not map TOFU split row back to full dataset: "
                f"question={row.get('question')!r}"
            )
        mapped.append(candidates[offset])
        consumed[key] += 1
    return mapped


def _sample_complete_groups(
    rows: Sequence[Mapping[str, Any]],
    group_ids: Sequence[str],
    *,
    count: int,
    seed: int,
    label: str,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, group_id in enumerate(group_ids):
        grouped[group_id].append(index)
    group_names = sorted(grouped)
    random.Random(seed).shuffle(group_names)
    chosen: List[int] = []
    for group_id in group_names:
        if len(chosen) + len(grouped[group_id]) > count:
            continue
        chosen.extend(grouped[group_id])
        if len(chosen) == count:
            break
    if len(chosen) != count:
        sizes = sorted({len(indices) for indices in grouped.values()})
        raise ValueError(
            f"Cannot select exactly {count} {label} rows without splitting an "
            f"author/group; available group sizes={sizes}"
        )
    return [dict(rows[index]) for index in chosen], [group_ids[index] for index in chosen]


def load_tofu_records(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data_dir = (
        resolve_project_path(args.tofu_data_dir)
        if args.tofu_data_dir
        else None
    )
    full_rows, full_source = _load_tofu_split("full", data_dir=data_dir)
    forget_rows_all, forget_source = _load_tofu_split(
        args.tofu_forget_split,
        data_dir=data_dir,
    )
    retain_rows_all, retain_source = _load_tofu_split(
        args.tofu_retain_split,
        data_dir=data_dir,
    )
    if len(full_rows) % args.tofu_rows_per_author != 0:
        raise ValueError(
            f"TOFU full split size {len(full_rows)} is not divisible by "
            f"--tofu-rows-per-author={args.tofu_rows_per_author}"
        )
    forget_full_indices = _map_tofu_rows_to_full(forget_rows_all, full_rows)
    retain_full_indices = _map_tofu_rows_to_full(retain_rows_all, full_rows)
    forget_groups_all = [
        f"tofu-author:{index // args.tofu_rows_per_author}"
        for index in forget_full_indices
    ]
    retain_groups_all = [
        f"tofu-author:{index // args.tofu_rows_per_author}"
        for index in retain_full_indices
    ]
    forget_rows, forget_groups = _sample_complete_groups(
        forget_rows_all,
        forget_groups_all,
        count=args.forget_num,
        seed=args.seed,
        label="forget",
    )
    retain_rows, retain_groups = _sample_complete_groups(
        retain_rows_all,
        retain_groups_all,
        count=args.retain_num,
        seed=args.seed,
        label="retain",
    )
    if set(forget_groups) & set(retain_groups):
        raise RuntimeError("TOFU forget and retain selections share authors")

    mapped_forget = _map_tofu_rows_to_full(forget_rows, full_rows)
    mapped_retain = _map_tofu_rows_to_full(retain_rows, full_rows)

    def convert(
        rows: Sequence[Mapping[str, Any]],
        full_indices: Sequence[int],
        group_ids: Sequence[str],
        role: str,
    ) -> List[Dict[str, Any]]:
        converted: List[Dict[str, Any]] = []
        for row, full_index, group_id in zip(rows, full_indices, group_ids):
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            record_id = f"tofu:full:{full_index}"
            converted.append(
                _compact_record(
                    dataset="tofu",
                    role=role,
                    source_index=full_index,
                    record_id=record_id,
                    group_id=group_id,
                    prompt=f"Question: {question}\nAnswer:",
                    subject=question,
                    sensitive_answers=[answer] if role == "forget" else [],
                    acceptable_answers=(
                        ["Unknown"] if role == "forget" else [answer]
                    ),
                    paraphrases=[],
                    raw_record=row,
                    metadata={
                        "question": question,
                        "answer": answer,
                        "full_index": full_index,
                        "semantic_key_sha256": sha256_json(
                            {
                                "question": normalize_text(question),
                                "answer": normalize_text(answer),
                            }
                        ),
                    },
                )
            )
        return converted

    records = [
        *convert(
            forget_rows,
            mapped_forget,
            forget_groups,
            "forget",
        ),
        *convert(
            retain_rows,
            mapped_retain,
            retain_groups,
            "retain",
        ),
    ]
    utility_sources: Dict[str, Any] = {}
    for split_index, split in enumerate(TOFU_UTILITY_SPLITS):
        utility_rows, utility_source = _load_tofu_split(split, data_dir=data_dir)
        utility_sources[split] = utility_source
        for source_index, row in enumerate(utility_rows):
            question = str(row["question"]).strip()
            answer = str(row["answer"]).strip()
            group_value = (
                row.get("author")
                or row.get("author_name")
                or row.get("subject")
            )
            if not group_value and split == "real_authors":
                if len(utility_rows) % args.tofu_rows_per_author != 0:
                    raise ValueError(
                        "TOFU real_authors lacks an author field and cannot "
                        "be grouped into complete author blocks"
                    )
                group_value = (
                    f"real-author-block:"
                    f"{source_index // args.tofu_rows_per_author}"
                )
            if not group_value:
                group_value = f"{split}:{source_index}"
            records.append(
                _compact_record(
                    dataset="tofu",
                    role="utility",
                    source_index=source_index,
                    record_id=f"tofu:{split}:{source_index}",
                    group_id=stable_id(
                        f"tofu-{split}",
                        normalize_text(group_value),
                    ),
                    prompt=f"Question: {question}\nAnswer:",
                    subject=question,
                    sensitive_answers=[],
                    acceptable_answers=[answer],
                    paraphrases=[],
                    raw_record=row,
                    metadata={
                        "question": question,
                        "answer": answer,
                        "utility_split": split,
                        "source_order": split_index,
                        "semantic_key_sha256": sha256_json(
                            {
                                "question": normalize_text(question),
                                "answer": normalize_text(answer),
                            }
                        ),
                    },
                )
            )
    return records, {
        "kind": "tofu",
        "dataset": TOFU_DATASET,
        "forget_split": args.tofu_forget_split,
        "retain_split": args.tofu_retain_split,
        "rows_per_author": args.tofu_rows_per_author,
        "full": full_source,
        "forget": forget_source,
        "retain": retain_source,
        "utility": utility_sources,
    }


def _record_refs_and_folds(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    n_folds: int,
) -> Tuple[Dict[str, RecordRef], Dict[str, int]]:
    refs, _ = build_record_folds(
        [record["_raw_record"] for record in records],
        [str(record["group_id"]) for record in records],
        record_ids=[str(record["record_id"]) for record in records],
        source_indices=[int(record["source_index"]) for record in records],
        n_folds=n_folds,
        seed=seed,
    )
    group_role_counts: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for record in records:
        group_role_counts[str(record["group_id"])][str(record["role"])] += 1
    roles = sorted({str(record["role"]) for record in records})
    totals = {
        role: sum(
            counts.get(role, 0)
            for counts in group_role_counts.values()
        )
        for role in roles
    }
    loads = {
        role: [0] * n_folds
        for role in roles
    }
    groups = sorted(
        group_role_counts,
        key=lambda group_id: (
            -max(group_role_counts[group_id].values()),
            -sum(group_role_counts[group_id].values()),
            hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest(),
        ),
    )
    group_folds: Dict[str, int] = {}
    for group_id in groups:
        counts = group_role_counts[group_id]

        def score(candidate: int) -> Tuple[float, ...]:
            affected = [
                (
                    loads[role][candidate] + counts.get(role, 0)
                )
                / max(1, totals[role])
                for role in roles
                if counts.get(role, 0)
            ]
            all_load = sum(loads[role][candidate] for role in roles)
            return (
                max(affected, default=0.0),
                sum(affected),
                float(all_load),
                float(candidate),
            )

        selected = min(range(n_folds), key=score)
        group_folds[group_id] = selected
        for role in roles:
            loads[role][selected] += counts.get(role, 0)
    return {ref.record_id: ref for ref in refs}, group_folds


def _split_records(
    records: Sequence[Mapping[str, Any]],
    refs: Mapping[str, RecordRef],
    group_folds: Mapping[str, int],
    *,
    fold: int,
    n_folds: int,
) -> Dict[str, List[Dict[str, Any]]]:
    validation_fold = (fold + 1) % n_folds
    result: Dict[str, List[Dict[str, Any]]] = {
        "train": [],
        "validation": [],
        "final_apply": [],
    }
    ref_partitions: Dict[str, List[RecordRef]] = {
        "train": [],
        "validation": [],
        "final_apply": [],
    }
    for source in records:
        record = dict(source)
        assigned = group_folds[str(record["group_id"])]
        if assigned == fold:
            partition = "final_apply"
        elif assigned == validation_fold:
            partition = "validation"
        else:
            partition = "train"
        result[partition].append(record)
        ref_partitions[partition].append(refs[str(record["record_id"])])
    assert_partition_disjoint(ref_partitions)
    roles = sorted({str(record["role"]) for record in records})
    for role in roles:
        for partition in result:
            if not any(record["role"] == role for record in result[partition]):
                raise RuntimeError(
                    f"Fold {fold} has no {role} records in {partition}"
                )
    return result


def _all_answer_candidates(records: Sequence[Mapping[str, Any]]) -> List[str]:
    values: List[str] = []
    for record in records:
        values.extend(str(value) for value in record["sensitive_answers"])
        values.extend(str(value) for value in record["acceptable_answers"])
    return values


def prompt_cases_for_record(
    record: Mapping[str, Any],
    *,
    fold: int,
    partition: str,
    phase: str,
    distractor_candidates: Sequence[str],
) -> List[PromptCase]:
    role = str(record["role"])
    if role == "forget":
        behavior = "avoid_sensitive"
        purpose = (
            "judge_a_repair"
            if partition == "train"
            else (
                "judge_a_validation"
                if partition == "validation"
                else "judge_b_final"
            )
        )
    elif role == "retain":
        behavior = "answer_correctly"
        purpose = (
            "utility_calibration"
            if partition != "final_apply"
            else "judge_b_retain"
        )
    else:
        behavior = "preserve_locality"
        purpose = (
            "utility_calibration"
            if partition != "final_apply"
            else "judge_b_utility"
        )
    supplied = list(record.get("paraphrases", []))
    if record["dataset"] == "zsre" and phase == "validation":
        # The one official ZsRE rephrase is final-test-only.
        supplied = []
    return generic_prompt_variants(
        dataset=str(record["dataset"]),
        fold=fold,
        partition=partition,
        purpose=purpose,
        source_record_id=str(record["record_id"]),
        source_group_id=str(record["group_id"]),
        base_prompt=str(record["prompt"]),
        subject=str(record["subject"]),
        sensitive_answers=record["sensitive_answers"],
        acceptable_answers=record["acceptable_answers"],
        supplied_paraphrases=supplied,
        distractor_candidates=distractor_candidates,
        expected_behavior=behavior,
        phase=phase,
    )


def _deduplicated_locality_records(
    records: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_prompt: Dict[str, Dict[str, Any]] = {}
    conflicts: Dict[str, List[str]] = defaultdict(list)
    duplicates = 0
    for parent in records:
        for ordinal, item in enumerate(parent.get("locality", [])):
            prompt = str(item.get("prompt", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not prompt or not answer:
                continue
            key = normalize_text(prompt)
            if key in by_prompt:
                existing_answers = {
                    normalize_text(value)
                    for value in by_prompt[key]["acceptable_answers"]
                }
                if normalize_text(answer) not in existing_answers:
                    conflicts[key].extend(
                        [
                            *by_prompt[key]["acceptable_answers"],
                            answer,
                        ]
                    )
                else:
                    duplicates += 1
                continue
            record_id = stable_id("locality", parent["dataset"], key)
            by_prompt[key] = _compact_record(
                dataset=str(parent["dataset"]),
                role="utility",
                source_index=int(parent["source_index"]) * 1000 + ordinal,
                record_id=record_id,
                group_id=record_id,
                prompt=prompt,
                subject=prompt,
                sensitive_answers=[],
                acceptable_answers=[answer],
                paraphrases=[],
                raw_record={"prompt": prompt, "answer": answer},
                metadata={
                    "locality_source_record_id": parent["record_id"],
                    "locality_source": item.get("source"),
                },
            )
    for key in conflicts:
        by_prompt.pop(key, None)
    return list(by_prompt.values()), {
        "exact_duplicates_removed": duplicates,
        "ambiguous_prompts_removed": len(conflicts),
        "ambiguous_examples": [
            {
                "normalized_prompt": key,
                "answers": sorted(set(values)),
            }
            for key, values in sorted(conflicts.items())[:20]
        ],
    }


def _repair_prompts(
    record: Mapping[str, Any],
    *,
    fold: int,
    candidates: Sequence[str],
) -> List[str]:
    cases = prompt_cases_for_record(
        record,
        fold=fold,
        partition="train",
        phase="validation",
        distractor_candidates=candidates,
    )
    # The canonical request itself is always supplied separately. Repair sees
    # only this validation suite, never Judge B's final prompts.
    return [case.prompt for case in cases]


def _mcf_stage_rows(
    records: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    candidates: Sequence[str],
) -> List[Dict[str, Any]]:
    forget = [record for record in records if record["role"] == "forget"]
    retain = [record for record in records if record["role"] == "retain"]
    output: List[Dict[str, Any]] = []
    for record in [*forget, *retain]:
        raw = copy.deepcopy(record["_raw_record"])
        raw["paraphrase_prompts"] = _repair_prompts(
            record,
            fold=fold,
            candidates=candidates,
        )
        output.append(raw)
    return output


def _zsre_stage_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    fold: int,
    candidates: Sequence[str],
) -> Dict[str, Any]:
    forget: List[Dict[str, Any]] = []
    retain: List[Dict[str, Any]] = []
    for record in records:
        raw = copy.deepcopy(record["_raw_record"])
        item = {
            "source_index": int(record["source_index"]),
            "record_id": str(record["record_id"]),
            "raw_record": raw,
            "repair_prompts": _repair_prompts(
                record,
                fold=fold,
                candidates=candidates,
            ),
        }
        if record["role"] == "forget":
            forget.append(item)
        elif record["role"] == "retain":
            retain.append(item)
    return {"forget": forget, "retain": retain}


def _tofu_stage_payload(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        split = str(record["role"])
        if split == "utility":
            split = str(record["metadata"].get("utility_split", "utility"))
        payload[split].append(copy.deepcopy(record["_raw_record"]))
    return dict(payload)


def _materialize_stage(
    *,
    dataset: str,
    stage: str,
    records: Sequence[Mapping[str, Any]],
    fold_dir: Path,
    fold: int,
    candidates: Sequence[str],
) -> Dict[str, Any]:
    stage_dir = fold_dir / "materialized" / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    forget_count = sum(record["role"] == "forget" for record in records)
    retain_count = sum(record["role"] == "retain" for record in records)
    utility_count = sum(record["role"] == "utility" for record in records)
    if dataset == "mcf":
        path = stage_dir / "multi_counterfact.json"
        write_json(
            path,
            _mcf_stage_rows(
                records,
                fold=fold,
                candidates=candidates,
            ),
        )
        disk_files = {"mcf_path": path}
    elif dataset == "zsre":
        path = stage_dir / "controlled_zsre.json"
        write_json(
            path,
            _zsre_stage_payload(
                records,
                fold=fold,
                candidates=candidates,
            ),
        )
        disk_files = {"controlled_zsre_path": path}
    elif dataset == "tofu":
        payload = _tofu_stage_payload(records)
        disk_files = {}
        for split, rows in sorted(payload.items()):
            path = stage_dir / f"{sanitize_component(split)}.json"
            write_json(path, rows)
            disk_files[f"{split}_path"] = path
    else:  # pragma: no cover - guarded by CLI
        raise ValueError(dataset)
    files = {
        name: str(path.relative_to(fold_dir))
        for name, path in disk_files.items()
    }
    return {
        "stage": stage,
        "forget_count": forget_count,
        "retain_count": retain_count,
        "utility_count": utility_count,
        "files": files,
        "sha256": {
            name: sha256_file(path)
            for name, path in disk_files.items()
        },
    }


def _bundle_record_counts(
    records: Sequence[Mapping[str, Any]],
) -> Dict[str, int]:
    result: Dict[str, int] = defaultdict(int)
    for record in records:
        result[str(record["role"])] += 1
    return dict(sorted(result.items()))


def _verify_fact_role_conflicts(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_content: Dict[str, set[str]] = defaultdict(set)
    by_semantic_key: Dict[str, set[str]] = defaultdict(set)
    by_record_id: Dict[str, int] = defaultdict(int)
    for record in records:
        by_content[str(record["content_sha256"])].add(str(record["role"]))
        semantic_key = record.get("metadata", {}).get(
            "semantic_key_sha256"
        )
        if semantic_key:
            by_semantic_key[str(semantic_key)].add(str(record["role"]))
        by_record_id[str(record["record_id"])] += 1
    conflicts = {
        content_hash: sorted(roles)
        for content_hash, roles in by_content.items()
        if "forget" in roles and ("retain" in roles or "utility" in roles)
    }
    duplicate_ids = [
        record_id
        for record_id, count in by_record_id.items()
        if count > 1
    ]
    if conflicts:
        raise RuntimeError(
            "Identical records occur in forget and utility roles: "
            f"{list(conflicts.items())[:5]}"
        )
    semantic_conflicts = {
        semantic_key: sorted(roles)
        for semantic_key, roles in by_semantic_key.items()
        if "forget" in roles and ("retain" in roles or "utility" in roles)
    }
    if semantic_conflicts:
        raise RuntimeError(
            "Semantically identical facts occur in forget and utility roles: "
            f"{list(semantic_conflicts.items())[:5]}"
        )
    if duplicate_ids:
        raise RuntimeError(f"Duplicate record IDs: {duplicate_ids[:5]}")
    return {
        "cross_role_exact_content_conflicts": 0,
        "cross_role_semantic_fact_conflicts": 0,
        "duplicate_record_ids": 0,
    }


def build_protocol(args: argparse.Namespace) -> Path:
    loaders = {
        "mcf": load_mcf_records,
        "zsre": load_zsre_records,
        "tofu": load_tofu_records,
    }
    records, source = loaders[args.dataset](args)
    role_audit = _verify_fact_role_conflicts(records)
    primary_records = [
        record for record in records if record["role"] != "utility"
    ]
    utility_records = [
        record for record in records if record["role"] == "utility"
    ]
    locality_records, locality_audit = _deduplicated_locality_records(
        primary_records
    )
    if (
        args.max_locality_records is not None
        and len(locality_records) > args.max_locality_records
    ):
        locality_records = sorted(
            locality_records,
            key=lambda record: hashlib.sha256(
                f"{args.seed}:{record['record_id']}".encode()
            ).hexdigest(),
        )[: args.max_locality_records]
        locality_audit["bounded_to"] = args.max_locality_records
    if locality_records:
        utility_records = [*utility_records, *locality_records]

    primary_refs, primary_group_folds = _record_refs_and_folds(
        primary_records,
        seed=args.seed,
        n_folds=args.n_folds,
    )
    utility_refs: Dict[str, RecordRef] = {}
    utility_group_folds: Dict[str, int] = {}
    if utility_records:
        utility_refs, utility_group_folds = _record_refs_and_folds(
            utility_records,
            seed=args.seed + 104729,
            n_folds=args.n_folds,
        )

    output_root = resolve_project_path(args.output_dir) / args.dataset
    output_root.mkdir(parents=True, exist_ok=True)
    protocol_id = stable_id(
        "controlled-protocol",
        args.dataset,
        args.seed,
        args.n_folds,
        args.forget_num,
        args.retain_num,
        source,
    )
    fold_entries: List[Dict[str, Any]] = []
    for fold in range(args.n_folds):
        fold_dir = output_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        primary = _split_records(
            primary_records,
            primary_refs,
            primary_group_folds,
            fold=fold,
            n_folds=args.n_folds,
        )
        utility: Dict[str, List[Dict[str, Any]]] = {
            "train": [],
            "validation": [],
            "final_apply": [],
        }
        if utility_records:
            utility = _split_records(
                utility_records,
                utility_refs,
                utility_group_folds,
                fold=fold,
                n_folds=args.n_folds,
            )
        stage_records = {
            stage: [*primary[stage], *utility[stage]]
            for stage in ("train", "validation", "final_apply")
        }
        stage_candidates = {
            stage: _all_answer_candidates(stage_records[stage])
            for stage in stage_records
        }

        development_cases: List[PromptCase] = []
        for stage in ("train", "validation"):
            for record in stage_records[stage]:
                development_cases.extend(
                    prompt_cases_for_record(
                        record,
                        fold=fold,
                        partition=stage,
                        phase="validation",
                        distractor_candidates=stage_candidates[stage],
                    )
                )
        test_cases: List[PromptCase] = []
        for record in stage_records["final_apply"]:
            test_cases.extend(
                prompt_cases_for_record(
                    record,
                    fold=fold,
                    partition="final_apply",
                    phase="test",
                    distractor_candidates=stage_candidates["final_apply"],
                )
            )
        assert_prompt_partitions_disjoint(development_cases, test_cases)

        train_materialized = _materialize_stage(
            dataset=args.dataset,
            stage="train",
            records=stage_records["train"],
            fold_dir=fold_dir,
            fold=fold,
            candidates=stage_candidates["train"],
        )
        validation_materialized = _materialize_stage(
            dataset=args.dataset,
            stage="validation",
            records=stage_records["validation"],
            fold_dir=fold_dir,
            fold=fold,
            candidates=stage_candidates["validation"],
        )
        # The final unlearning run receives the held-out deletion/retain
        # requests plus development-only calibration utility. Final utility
        # rows remain invisible until Judge B evaluation.
        final_runtime_records = [*primary["final_apply"]]
        if args.dataset == "tofu":
            final_runtime_records.extend(utility["train"])
        final_runtime_candidates = _all_answer_candidates(
            final_runtime_records
        )
        final_materialized = _materialize_stage(
            dataset=args.dataset,
            stage="final_apply",
            records=final_runtime_records,
            fold_dir=fold_dir,
            fold=fold,
            candidates=final_runtime_candidates,
        )

        split_contract = {
            "unit": (
                "author"
                if args.dataset == "tofu"
                else "subject/entity group"
            ),
            "train_folds": [
                value
                for value in range(args.n_folds)
                if value not in {fold, (fold + 1) % args.n_folds}
            ],
            "validation_fold": (fold + 1) % args.n_folds,
            "test_fold": fold,
            "final_apply_semantics": (
                "Deletion requests are visible only after candidate freeze; "
                "Judge B prompts and outcomes remain test-only."
            ),
        }
        final_apply_payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": "final_apply",
            "dataset": args.dataset,
            "protocol_id": protocol_id,
            "fold": fold,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "source_commitment_sha256": sha256_json(source),
            "split_contract": split_contract,
            "final_apply_records": [
                _public_record(record)
                for record in primary["final_apply"]
            ],
            "partitions": {
                "final_apply": refs_dict(
                    [
                        primary_refs[record["record_id"]]
                        for record in primary["final_apply"]
                    ]
                ),
                "development_utility_calibration": refs_dict(
                    [
                        utility_refs[record["record_id"]]
                        for record in (
                            utility["train"]
                            if args.dataset == "tofu"
                            else []
                        )
                    ]
                ),
            },
            "prompt_cases": [],
            "materialized_inputs": final_materialized,
            "selection_receipt_required": True,
            "contains_judge_b_prompts": False,
            "test_results_may_influence_repair": False,
        }
        final_apply_bundle = finalize_bundle(final_apply_payload)
        final_apply_path = fold_dir / "final_apply.json"
        write_json(final_apply_path, final_apply_bundle)

        test_payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": "test",
            "dataset": args.dataset,
            "protocol_id": protocol_id,
            "fold": fold,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "source_commitment_sha256": sha256_json(source),
            "split_contract": split_contract,
            "partitions": {
                "final_fact_evaluation": refs_dict(
                    [
                        primary_refs[record["record_id"]]
                        for record in primary["final_apply"]
                    ]
                ),
                "test_utility": refs_dict(
                    [
                        utility_refs[record["record_id"]]
                        for record in utility["final_apply"]
                    ]
                ),
            },
            "prompt_cases": [
                prompt_case_dict(case) for case in test_cases
            ],
            "final_apply_bundle_path": "final_apply.json",
            "final_apply_bundle_sha256": final_apply_bundle[
                "bundle_sha256"
            ],
            "judge_role": "judge_b_final",
            "selection_receipt_required": True,
            "rerun_policy": "one_shot_by_default",
        }
        test_bundle = finalize_bundle(test_payload)
        test_path = fold_dir / "test.json"
        write_json(test_path, test_bundle)

        development_payload = {
            "schema_version": SCHEMA_VERSION,
            "phase": "development",
            "dataset": args.dataset,
            "protocol_id": protocol_id,
            "fold": fold,
            "n_folds": args.n_folds,
            "seed": args.seed,
            "source": source,
            "split_contract": split_contract,
            "records": {
                "train": [
                    _public_record(record)
                    for record in stage_records["train"]
                ],
                "validation": [
                    _public_record(record)
                    for record in stage_records["validation"]
                ],
            },
            "partitions": {
                stage: refs_dict(
                    [
                        (
                            primary_refs
                            if record["record_id"] in primary_refs
                            else utility_refs
                        )[record["record_id"]]
                        for record in stage_records[stage]
                    ]
                )
                for stage in ("train", "validation")
            },
            "prompt_cases": [
                prompt_case_dict(case) for case in development_cases
            ],
            "materialized_inputs": {
                "train": train_materialized,
                "validation": validation_materialized,
            },
            "final_apply_bundle_path": "final_apply.json",
            "final_apply_bundle_sha256": final_apply_bundle[
                "bundle_sha256"
            ],
            "test_bundle_path": "test.json",
            "test_bundle_sha256": test_bundle["bundle_sha256"],
            "judge_role": "judge_a_development",
            "test_results_may_influence_repair": False,
        }
        development_bundle = finalize_bundle(development_payload)
        development_path = fold_dir / "development.json"
        write_json(development_path, development_bundle)
        fold_entries.append(
            {
                "fold": fold,
                "counts": {
                    stage: _bundle_record_counts(stage_records[stage])
                    for stage in stage_records
                },
                "development_bundle": str(
                    development_path.relative_to(output_root)
                ),
                "development_bundle_sha256": development_bundle[
                    "bundle_sha256"
                ],
                "final_apply_bundle": str(
                    final_apply_path.relative_to(output_root)
                ),
                "final_apply_bundle_sha256": final_apply_bundle[
                    "bundle_sha256"
                ],
                "test_bundle": str(test_path.relative_to(output_root)),
                "test_bundle_sha256": test_bundle["bundle_sha256"],
            }
        )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "controlled_unlearning_manifest",
        "protocol_id": protocol_id,
        "dataset": args.dataset,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "forget_num": args.forget_num,
        "retain_num": args.retain_num,
        "source": source,
        "folds": fold_entries,
        "audits": {
            **role_audit,
            "locality": locality_audit,
            "group_count": len(primary_group_folds),
            "utility_group_count": len(utility_group_folds),
        },
        "safeguards": {
            "fact_or_author_group_disjoint": True,
            "validation_and_test_prompt_text_disjoint": True,
            "multiple_validation_paraphrases": True,
            "heldout_test_paraphrase": True,
            "test_styles": [
                "direct",
                "indirect",
                "cloze",
                "multiple_choice",
                "adversarial",
            ],
            "judge_a_development_only": True,
            "judge_b_final_only": True,
            "token_probability_metrics_required": True,
            "locality_metrics_required": True,
            "utility_tolerance_not_zero": True,
            "manual_audit_required": True,
            "final_test_feedback_forbidden": True,
        },
    }
    manifest = finalize_bundle(manifest)
    manifest_path = output_root / "manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["mcf", "zsre", "tofu", "all"],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        default="data/controlled_unlearning",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--forget-num", type=int, default=None)
    parser.add_argument("--retain-num", type=int, default=None)
    parser.add_argument("--mcf-path", default="data/multi_counterfact.json")
    parser.add_argument("--mcf-url", default=MCF_URL)
    parser.add_argument("--zsre-path", default="data/zsre_mend_eval.json")
    parser.add_argument("--zsre-url", default=ZSRE_URL)
    parser.add_argument(
        "--tofu-data-dir",
        default=None,
        help=(
            "Optional directory containing full.json, forget05.json, "
            "retain95.json, real_authors.json, and world_facts.json. "
            "Otherwise locuslab/TOFU is loaded from Hugging Face."
        ),
    )
    parser.add_argument("--tofu-forget-split", default="forget05")
    parser.add_argument("--tofu-retain-split", default="retain95")
    parser.add_argument("--tofu-rows-per-author", type=int, default=20)
    parser.add_argument(
        "--max-locality-records",
        type=int,
        default=1000,
        help=(
            "Deterministic cap on deduplicated MCF/ZsRE locality records. "
            "Use 0 only to disable locality, or a larger value for exhaustive "
            "token-only scoring."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.n_folds != N_FOLDS:
        raise ValueError(
            f"This protocol requires exactly {N_FOLDS} folds, got "
            f"{args.n_folds}"
        )
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.forget_num is not None and args.forget_num <= 0:
        raise ValueError("--forget-num must be positive")
    if args.retain_num is not None and args.retain_num <= 0:
        raise ValueError("--retain-num must be positive")
    if args.tofu_rows_per_author <= 0:
        raise ValueError("--tofu-rows-per-author must be positive")
    if args.max_locality_records is not None and args.max_locality_records < 0:
        raise ValueError("--max-locality-records must be non-negative")


def main() -> None:
    parser = build_parser()
    original = parser.parse_args()
    validate_args(original)
    datasets = (
        ["mcf", "zsre", "tofu"]
        if original.dataset == "all"
        else [original.dataset]
    )
    manifests: List[Path] = []
    for dataset in datasets:
        args = copy.copy(original)
        args.dataset = dataset
        default_forget, default_retain = DEFAULT_COUNTS[dataset]
        args.forget_num = (
            original.forget_num
            if original.forget_num is not None
            else default_forget
        )
        args.retain_num = (
            original.retain_num
            if original.retain_num is not None
            else default_retain
        )
        print(
            f"Building {dataset} five-fold protocol "
            f"(forget={args.forget_num}, retain={args.retain_num})"
        )
        manifests.append(build_protocol(args))
    for path in manifests:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
