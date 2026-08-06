#!/usr/bin/env python3
"""Build relation-aware RWKU entity-fact artifacts without opening Level 3.

Only pinned Level-1 and Level-2 records are visible to this probe-assisted
preparation program.  Official Level-3, MIA, neighbor, utility, and fluency
records are represented by pinned manifest descriptors and remain unopened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from rwku_artifact_access import (
    PROBE_PROTOCOL_LABEL,
    PROBE_PROTOCOL_STATUS,
    canonical_json_bytes,
    make_artifact,
    sha256_file,
    sha256_json,
    write_artifact,
)
from rwku_data import (
    DEFAULT_DATA_ROOT,
    RWKU_CODE_REVISION,
    RWKU_DATASET_REVISION,
    ensure_fact_assignment_data,
    pinned_target_manifest,
    record_sha256,
)


ENTITY_FACT_SCHEMA_VERSION = "rwku_entity_fact_v1"
RUNTIME_EOS_MARKER = "<RUNTIME_TOKENIZER_EOS>"
MCF_SHAPED_FORMAT = "mcf_shaped_rwku_training_request_v1"


class FactAuditError(ValueError):
    """Raised when a relation/fact assignment cannot be proven unambiguous."""


def normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value)).strip().split())


def normalize_identity(value: str) -> str:
    return normalize_text(value).casefold()


def normalized_query_hash(query: str) -> str:
    normalized = normalize_identity(query)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def entity_fact_id(
    entity_id: str,
    relation_id: str,
    canonical_sensitive_answer: str,
) -> str:
    """Identity is SHA256(entity, relation, normalized canonical answer)."""

    identity = [
        normalize_text(entity_id),
        normalize_text(relation_id),
        normalize_identity(canonical_sensitive_answer),
    ]
    return hashlib.sha256(canonical_json_bytes(identity)).hexdigest()


def view_content_sha256(view: Mapping[str, Any]) -> str:
    content = {
        "normalized_query": normalize_identity(str(view["query"])),
        "canonical_sensitive_answer": normalize_identity(
            str(view["canonical_sensitive_answer"])
        ),
    }
    return sha256_json(content)


def _load_override(path: Path) -> Tuple[Dict[str, Any], str]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema_version") != "rwku_fact_overrides_v1":
        raise FactAuditError(f"Unsupported fact override schema: {source}")
    records = value.get("records")
    if not isinstance(records, dict):
        raise FactAuditError(f"Fact override records must be an object: {source}")
    return value, sha256_file(source)


_DETERMINISTIC_RELATION_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("pseudonym", ("pseudonym", "pen name", "nom de plume")),
    ("occupation", ("occupation", "profession", " is an american ___")),
    ("first_published_novel", ("first published novel", "debut novel", "1974 debut novel")),
    ("birth_city", ("was born in", "birth city")),
    ("birth_state", ("state was", "birth state")),
    ("coauthor", ("co-wrote", "coauthor", "co-author")),
    ("mother_maiden_name", ("mother", "née", "maiden name")),
    ("school_before_lisbon_high", ("school", "before entering lisbon high")),
    ("childhood_settlement", ("family settle", "settled when")),
)


def deterministic_relation_candidates(query: str) -> List[str]:
    normalized = normalize_identity(query)
    return [
        relation_id
        for relation_id, needles in _DETERMINISTIC_RELATION_PATTERNS
        if any(needle in normalized for needle in needles)
    ]


def _prompt_style(record: Mapping[str, Any]) -> str:
    query = str(record.get("query", ""))
    if str(record.get("level")) == "1" or "___" in query:
        return "cloze"
    return "direct question"


def _source_record(
    record: Mapping[str, Any],
    *,
    source_file: str,
    source_row_index: int,
    assigned_relation_id: str,
    assigned_fact_id: str,
) -> Dict[str, Any]:
    return {
        "source_file": source_file,
        "source_row_index": int(source_row_index),
        "source_record_sha256": record_sha256(record),
        "level": str(record.get("level", "")),
        "query_type": str(record.get("type", "")),
        "normalized_query_hash": normalized_query_hash(str(record.get("query", ""))),
        "original_answer": str(record.get("answer", "")),
        "assigned_relation_id": assigned_relation_id,
        "assigned_fact_id": assigned_fact_id,
    }


def assign_records_to_facts(
    *,
    entity_id: str,
    subject: str,
    level1: Sequence[Mapping[str, Any]],
    level2: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, str],
    fact_overrides_path: Path | None,
    strict: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Combine L1/L2 first, deduplicate, then assign relation-aware facts."""

    override: Dict[str, Any] = {"records": {}}
    override_sha = ""
    if fact_overrides_path is not None:
        override, override_sha = _load_override(fact_overrides_path)
        if normalize_text(str(override.get("entity_id", ""))) != normalize_text(entity_id):
            raise FactAuditError("Fact override entity_id does not match requested entity")
        if normalize_text(str(override.get("subject", ""))) != normalize_text(subject):
            raise FactAuditError("Fact override subject does not match requested subject")
    override_records: Mapping[str, Any] = override["records"]

    combined: List[Tuple[str, int, Mapping[str, Any]]] = [
        ("forget_level1.json", index, row) for index, row in enumerate(level1)
    ] + [
        ("forget_level2.json", index, row) for index, row in enumerate(level2)
    ]

    groups: MutableMapping[str, Dict[str, Any]] = {}
    unresolved: List[Dict[str, Any]] = []
    unique_content: Dict[str, Dict[str, Any]] = {}
    assignment_provenance_counts: MutableMapping[str, int] = defaultdict(int)

    for source_file, row_index, raw_record in combined:
        record = dict(raw_record)
        expected_level = "1" if source_file == "forget_level1.json" else "2"
        if str(record.get("level")) != expected_level:
            raise FactAuditError(
                "Entity-fact grouping accepts only correctly typed Level-1 and "
                "Level-2 records; Level 3 is structurally forbidden"
            )
        source_digest = record_sha256(record)
        exact_content_digest = sha256_json(record)
        selected = override_records.get(source_digest)
        provenance: Dict[str, Any]
        if selected is not None:
            if not isinstance(selected, dict):
                raise FactAuditError(f"Override for {source_digest} must be an object")
            relation_id = normalize_text(str(selected.get("relation_id", "")))
            canonical_answer = normalize_text(
                str(selected.get("canonical_sensitive_answer", ""))
            )
            aliases = [normalize_text(str(value)) for value in selected.get("sensitive_answer_aliases", [])]
            provenance = {
                "method": "committed_manual_override",
                "manual_override_sha256": override_sha,
                "source_record_sha256": source_digest,
            }
        else:
            candidates = deterministic_relation_candidates(str(record.get("query", "")))
            if len(candidates) != 1:
                unresolved.append(
                    {
                        "source_file": source_file,
                        "source_row_index": row_index,
                        "source_record_sha256": source_digest,
                        "query": str(record.get("query", "")),
                        "candidate_relation_ids": candidates,
                        "reason": "no_relation_assignment" if not candidates else "ambiguous_relation_assignment",
                    }
                )
                continue
            relation_id = candidates[0]
            canonical_answer = normalize_text(str(record.get("answer", "")))
            aliases = []
            provenance = {
                "method": "frozen_deterministic_mapper",
                "mapper_revision": "deterministic_relation_mapper_v1",
                "manual_override_sha256": "",
            }
        if not relation_id or not canonical_answer:
            raise FactAuditError(f"Empty relation or canonical answer for {source_digest}")
        original_answer = normalize_text(str(record.get("answer", "")))
        accepted_answers = {normalize_identity(canonical_answer), *(normalize_identity(alias) for alias in aliases)}
        if normalize_identity(original_answer) not in accepted_answers:
            raise FactAuditError(
                f"Override canonical answer/aliases do not cover source answer for {source_digest}"
            )
        fact_id = entity_fact_id(entity_id, relation_id, canonical_answer)
        assignment_provenance_counts[provenance["method"]] += 1

        source_record = _source_record(
            record,
            source_file=source_file,
            source_row_index=row_index,
            assigned_relation_id=relation_id,
            assigned_fact_id=fact_id,
        )
        view = {
            "schema_version": ENTITY_FACT_SCHEMA_VERSION,
            "view_id": "",
            "query": normalize_text(str(record.get("query", ""))),
            "level": str(record.get("level", "")),
            "query_type": str(record.get("type", "")),
            "prompt_style": _prompt_style(record),
            "canonical_sensitive_answer": canonical_answer,
            "sensitive_answer_alias": original_answer,
            "source_record_sha256": source_digest,
            "source_file": source_file,
            "source_row_index": row_index,
            "boundary_expanding": False,
        }
        view["view_content_sha256"] = view_content_sha256(view)
        view["view_id"] = hashlib.sha256(
            f"{fact_id}:{view['view_content_sha256']}".encode("utf-8")
        ).hexdigest()

        fact = groups.setdefault(
            fact_id,
            {
                "schema_version": ENTITY_FACT_SCHEMA_VERSION,
                "protocol_label": PROBE_PROTOCOL_LABEL,
                "protocol_status": PROBE_PROTOCOL_STATUS,
                "entity_id": entity_id,
                "subject": subject,
                "subject_aliases": [],
                "fact_id": fact_id,
                "relation_id": relation_id,
                "canonical_sensitive_answer": canonical_answer,
                "sensitive_answer_aliases": sorted(set(aliases)),
                "source_records": [],
                "optimization_views": [],
                "held_out_views": [],
                "partition": "unassigned",
                "training_allowed": False,
                "source_hashes": dict(source_hashes),
                "relation_assignment_provenance": [],
                "manual_override_sha256": override_sha,
                "_all_views": {},
            },
        )
        if fact["relation_id"] != relation_id or normalize_identity(fact["canonical_sensitive_answer"]) != normalize_identity(canonical_answer):
            raise FactAuditError(f"One fact ID has conflicting assignments: {fact_id}")
        fact["source_records"].append(source_record)
        if provenance not in fact["relation_assignment_provenance"]:
            fact["relation_assignment_provenance"].append(provenance)
        for alias in aliases:
            if alias not in fact["sensitive_answer_aliases"]:
                fact["sensitive_answer_aliases"].append(alias)
        # Exact duplicate/source-equivalent prompt views remain indivisible.
        existing_view = fact["_all_views"].get(view["view_content_sha256"])
        if existing_view is None:
            view["source_record_sha256_values"] = [source_digest]
            fact["_all_views"][view["view_content_sha256"]] = view
        elif source_digest not in existing_view["source_record_sha256_values"]:
            existing_view["source_record_sha256_values"].append(source_digest)
        unique_content.setdefault(exact_content_digest, record)

    override_unknown = sorted(set(override_records) - {record_sha256(row) for _, _, row in combined})
    if override_unknown:
        raise FactAuditError(
            "Fact override references records absent from pinned Level 1/2: "
            + ", ".join(override_unknown)
        )
    if strict and unresolved:
        raise FactAuditError(
            f"Strict fact audit found {len(unresolved)} unresolved/ambiguous records: "
            + "; ".join(item["source_record_sha256"] for item in unresolved)
        )

    # One semantic relation must not silently combine conflicting answers.
    answers_by_relation: MutableMapping[str, set[str]] = defaultdict(set)
    for fact in groups.values():
        answers_by_relation[fact["relation_id"]].add(
            normalize_identity(fact["canonical_sensitive_answer"])
        )
    conflicts = {
        relation: sorted(answers)
        for relation, answers in answers_by_relation.items()
        if len(answers) > 1
    }
    if strict and conflicts:
        raise FactAuditError(
            "A relation groups conflicting canonical answers without an explicit "
            f"multi-valued override: {conflicts}"
        )

    alias_to_facts: MutableMapping[str, set[str]] = defaultdict(set)
    canonical_to_facts: MutableMapping[str, set[str]] = defaultdict(set)
    for fact_id, fact in groups.items():
        canonical_to_facts[
            normalize_identity(fact["canonical_sensitive_answer"])
        ].add(fact_id)
        for answer in fact["sensitive_answer_aliases"]:
            alias_to_facts[normalize_identity(answer)].add(fact_id)
    alias_conflicts = {
        alias: sorted(fact_ids | canonical_to_facts.get(alias, set()))
        for alias, fact_ids in alias_to_facts.items()
        if len(fact_ids | canonical_to_facts.get(alias, set())) > 1
    }
    if strict and alias_conflicts:
        raise FactAuditError(
            "An answer alias would cross fact partitions: " + json.dumps(alias_conflicts, sort_keys=True)
        )

    facts: List[Dict[str, Any]] = []
    for fact_id in sorted(groups):
        fact = groups[fact_id]
        fact["_all_views"] = list(
            sorted(fact["_all_views"].values(), key=lambda view: view["view_content_sha256"])
        )
        fact["sensitive_answer_aliases"] = sorted(set(fact["sensitive_answer_aliases"]))
        fact["source_records"] = sorted(
            fact["source_records"],
            key=lambda source: (source["source_file"], source["source_row_index"]),
        )
        facts.append(fact)

    audit = {
        "level1_row_count": len(level1),
        "level2_row_count": len(level2),
        "unique_row_count": len(unique_content),
        "semantic_fact_group_count": len(facts),
        "ambiguous_unresolved_record_count": len(unresolved),
        "ambiguous_unresolved_records": unresolved,
        "assignment_provenance_counts": dict(sorted(assignment_provenance_counts.items())),
        "manual_override_sha256": override_sha,
    }
    return facts, audit


def split_entity_facts(
    facts: Sequence[Mapping[str, Any]],
    *,
    unseen_fact_fraction: float,
    prompt_holdout_per_seen_fact: int,
    split_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Apply the specified round-half-up fact and within-fact view split."""

    if not 0.0 <= unseen_fact_fraction <= 1.0:
        raise ValueError("unseen_fact_fraction must be between zero and one")
    if prompt_holdout_per_seen_fact < 1:
        raise ValueError("prompt_holdout_per_seen_fact must be at least one")
    if len(facts) < 2:
        raise ValueError("At least two fact IDs are required for a fact holdout")
    fact_ids = [str(fact["fact_id"]) for fact in facts]
    if len(fact_ids) != len(set(fact_ids)):
        raise FactAuditError("Fact IDs are not unique")
    n_facts = len(fact_ids)
    n_unseen = int(math.floor(n_facts * unseen_fact_fraction + 0.5))
    n_unseen = max(1, min(n_facts - 1, n_unseen))
    sorted_fact_ids = sorted(
        fact_ids,
        key=lambda fact_id: hashlib.sha256(
            f"{split_seed}:fact:{fact_id}".encode("utf-8")
        ).hexdigest(),
    )
    unseen_ids = set(sorted_fact_ids[:n_unseen])
    calibration_ids = set(sorted_fact_ids[n_unseen:])

    catalog: List[Dict[str, Any]] = []
    optimization_views: List[Dict[str, Any]] = []
    seen_prompt_views: List[Dict[str, Any]] = []
    unseen_views: List[Dict[str, Any]] = []
    single_calibration = 0
    single_unseen = 0

    for original in facts:
        fact = json.loads(json.dumps(original))
        all_views = list(fact.pop("_all_views"))
        content_hashes = [view["view_content_sha256"] for view in all_views]
        if len(content_hashes) != len(set(content_hashes)):
            raise FactAuditError(f"Duplicate view content survived deduplication for {fact['fact_id']}")
        fact["optimization_views"] = []
        fact["held_out_views"] = []
        if fact["fact_id"] in unseen_ids:
            fact["partition"] = "unseen_fact"
            fact["training_allowed"] = False
            fact["held_out_views"] = all_views
            if len(all_views) == 1:
                single_unseen += 1
            for view in all_views:
                row = {**view, "fact_id": fact["fact_id"], "relation_id": fact["relation_id"], "entity_id": fact["entity_id"], "subject": fact["subject"], "training_allowed": False}
                unseen_views.append(row)
        else:
            fact["partition"] = "calibration_fact"
            fact["training_allowed"] = True
            ordered = sorted(
                all_views,
                key=lambda view: hashlib.sha256(
                    f"{split_seed}:view:{fact['fact_id']}:{view['view_content_sha256']}".encode("utf-8")
                ).hexdigest(),
            )
            if len(ordered) == 1:
                optimization = ordered
                held_out: List[Dict[str, Any]] = []
                single_calibration += 1
            else:
                holdout_count = min(prompt_holdout_per_seen_fact, len(ordered) - 1)
                held_out = ordered[:holdout_count]
                optimization = ordered[holdout_count:]
            fact["optimization_views"] = optimization
            fact["held_out_views"] = held_out
            for view in optimization:
                optimization_views.append({**view, "fact_id": fact["fact_id"], "relation_id": fact["relation_id"], "entity_id": fact["entity_id"], "subject": fact["subject"], "sensitive_answer_aliases": fact["sensitive_answer_aliases"], "training_allowed": True})
            for view in held_out:
                seen_prompt_views.append({**view, "fact_id": fact["fact_id"], "relation_id": fact["relation_id"], "entity_id": fact["entity_id"], "subject": fact["subject"], "training_allowed": False})
        catalog.append(fact)

    if {row["fact_id"] for row in optimization_views} & {row["fact_id"] for row in unseen_views}:
        raise FactAuditError("A fact ID crosses calibration and unseen partitions")
    opt_hashes = {row["view_content_sha256"] for row in optimization_views}
    eval_hashes = {row["view_content_sha256"] for row in [*seen_prompt_views, *unseen_views]}
    if opt_hashes & eval_hashes:
        raise FactAuditError("A duplicate/source-equivalent prompt view crosses partitions")

    counts = {
        "semantic_fact_group_count": n_facts,
        "calibration_fact_count": len(calibration_ids),
        "unseen_fact_count": len(unseen_ids),
        "optimization_view_count": len(optimization_views),
        "seen_fact_unseen_prompt_view_count": len(seen_prompt_views),
        "unseen_fact_view_count": len(unseen_views),
        "single_view_calibration_fact_count": single_calibration,
        "single_view_unseen_fact_count": single_unseen,
    }
    split = {
        "algorithm": "rwku_relation_aware_fact_split_v1",
        "split_seed": int(split_seed),
        "requested_unseen_fact_fraction": float(unseen_fact_fraction),
        "n_unseen_formula": "max(1,min(N-1,floor(N*f+0.5)))",
        "prompt_holdout_per_seen_fact": int(prompt_holdout_per_seen_fact),
        "calibration_fact_ids": sorted(calibration_ids),
        "unseen_fact_ids": sorted(unseen_ids),
        **counts,
    }
    return catalog, optimization_views, seen_prompt_views, unseen_views, split


def official_locked_descriptor(seed: int, *, include_level12: bool) -> Dict[str, Any]:
    """Describe pinned official files without opening any of those raw files."""

    manifest = pinned_target_manifest(seed)
    filenames = [
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
    ]
    if include_level12:
        filenames = ["forget_level1.json", "forget_level2.json", *filenames]
    return {
        "descriptor_only": True,
        "raw_records_opened_during_prepare": False,
        "dataset_revision": RWKU_DATASET_REVISION,
        "upstream_code_revision": RWKU_CODE_REVISION,
        "seed": seed,
        "target_directory": manifest["directory"],
        "subject": manifest["subject"],
        "files": {
            filename: {
                "relative_path": f"Target/{manifest['directory']}/{filename}",
                "sha256": manifest["sha256"][filename],
                "row_count": manifest["counts"][filename],
            }
            for filename in filenames
        },
    }


def build_probe_artifacts(
    *,
    data_root: Path,
    seed: int,
    output_dir: Path,
    fact_overrides_path: Path,
    unseen_fact_fraction: float = 0.25,
    prompt_holdout_per_seen_fact: int = 1,
    split_seed: int = 0,
    strict: bool = True,
    allow_download: bool = False,
) -> Dict[str, Any]:
    target, data, source_hashes = ensure_fact_assignment_data(
        data_root, seed, allow_download=allow_download
    )
    entity_id = f"rwku:{target.directory}"
    facts, audit = assign_records_to_facts(
        entity_id=entity_id,
        subject=target.subject,
        level1=data["forget_level1.json"],
        level2=data["forget_level2.json"],
        source_hashes=source_hashes,
        fact_overrides_path=fact_overrides_path,
        strict=strict,
    )
    catalog, training, seen_eval, unseen_eval, split = split_entity_facts(
        facts,
        unseen_fact_fraction=unseen_fact_fraction,
        prompt_holdout_per_seen_fact=prompt_holdout_per_seen_fact,
        split_seed=split_seed,
    )
    audit.update(split)
    metadata = {
        "seed": seed,
        "entity_id": entity_id,
        "subject": target.subject,
        "dataset_revision": RWKU_DATASET_REVISION,
        "upstream_code_revision": RWKU_CODE_REVISION,
    }
    artifacts = {
        "fact_catalog.json": make_artifact("fact_catalog", {"facts": catalog, "audit": audit}, protocol_label=PROBE_PROTOCOL_LABEL, protocol_status=PROBE_PROTOCOL_STATUS, metadata=metadata),
        "training_bundle.json": make_artifact("training_bundle", {"views": training}, protocol_label=PROBE_PROTOCOL_LABEL, protocol_status=PROBE_PROTOCOL_STATUS, metadata=metadata),
        "seen_fact_unseen_prompt_eval.json": make_artifact("seen_fact_unseen_prompt_eval", {"metric_label": "seen-fact/unseen-prompt generalization", "views": seen_eval}, protocol_label=PROBE_PROTOCOL_LABEL, protocol_status=PROBE_PROTOCOL_STATUS, metadata=metadata),
        "unseen_fact_eval.json": make_artifact("unseen_fact_eval", {"metric_label": "unseen-fact entity transfer", "views": unseen_eval}, protocol_label=PROBE_PROTOCOL_LABEL, protocol_status=PROBE_PROTOCOL_STATUS, metadata=metadata),
        "official_locked_eval.json": make_artifact("official_locked_eval", official_locked_descriptor(seed, include_level12=False), protocol_label=PROBE_PROTOCOL_LABEL, protocol_status=PROBE_PROTOCOL_STATUS, metadata=metadata),
        "split_manifest.json": make_artifact("split_manifest", {"audit": audit, "source_hashes": source_hashes, "fact_override_sha256": audit["manual_override_sha256"], **split}, protocol_label=PROBE_PROTOCOL_LABEL, protocol_status=PROBE_PROTOCOL_STATUS, metadata=metadata),
    }
    output = Path(output_dir)
    for filename, artifact in artifacts.items():
        write_artifact(output / filename, artifact)
    write_fact_audit_markdown(output / "fact_audit.md", audit, catalog)
    return {"artifacts": artifacts, "audit": audit, "catalog": catalog}


def write_fact_audit_markdown(path: Path, audit: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# RWKU entity-fact audit",
        "",
        f"- Level-1 rows: {audit['level1_row_count']}",
        f"- Level-2 rows: {audit['level2_row_count']}",
        f"- Unique rows: {audit['unique_row_count']}",
        f"- Semantic fact groups: {audit['semantic_fact_group_count']}",
        f"- Calibration facts: {audit['calibration_fact_count']}",
        f"- Unseen facts: {audit['unseen_fact_count']}",
        f"- Optimization views: {audit['optimization_view_count']}",
        f"- Seen-fact/unseen-prompt views: {audit['seen_fact_unseen_prompt_view_count']}",
        f"- Unseen-fact views: {audit['unseen_fact_view_count']}",
        f"- Ambiguous/unresolved records: {audit['ambiguous_unresolved_record_count']}",
        "",
        "| Partition | Relation | Canonical sensitive answer | Fact ID | Source rows | Optimization views | Held-out views |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for fact in sorted(catalog, key=lambda item: (item["partition"], item["relation_id"])):
        lines.append(
            "| {partition} | {relation} | {answer} | `{fact_id}` | {sources} | {optimization} | {held} |".format(
                partition=fact["partition"],
                relation=fact["relation_id"],
                answer=str(fact["canonical_sensitive_answer"]).replace("|", "\\|"),
                fact_id=fact["fact_id"],
                sources=len(fact["source_records"]),
                optimization=len(fact["optimization_views"]),
                held=len(fact["held_out_views"]),
            )
        )
    lines.extend(["", "The builder never loads Level 3 or any official utility, MIA, neighbor, or fluency record.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def export_mcf_shaped_training_requests(
    training_artifact: Mapping[str, Any],
    *,
    destination: Path,
) -> Dict[str, Any]:
    if training_artifact.get("artifact_role") != "training_bundle":
        raise FactAuditError("MCF-shaped RWKU export requires a training_bundle artifact")
    requests = []
    for view in training_artifact["payload"]["views"]:
        requests.append(
            {
                "format": MCF_SHAPED_FORMAT,
                "benchmark": "rwku",
                "training_only": True,
                "case_id": view["view_id"],
                "prompt": view["query"],
                "subject": view["subject"],
                "relation_id": view["relation_id"],
                "target_new": {"str": view["sensitive_answer_alias"]},
                "target_true": {"str": RUNTIME_EOS_MARKER},
                "paraphrase_prompts": [],
                "metadata": {
                    "entity_id": view["entity_id"],
                    "fact_id": view["fact_id"],
                    "protocol_label": training_artifact["protocol_label"],
                    "training_allowed": True,
                    "source_hashes": view.get("source_record_sha256_values", [view["source_record_sha256"]]),
                },
            }
        )
    payload = {
        "format": MCF_SHAPED_FORMAT,
        "benchmark": "rwku",
        "training_only": True,
        "runtime_eos_marker": RUNTIME_EOS_MARKER,
        "requests": requests,
    }
    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    Path(destination).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_mcf_shaped_training_requests(
    path: Path,
    *,
    tokenizer: Any,
) -> List[Dict[str, Any]]:
    """Load the RWKU training adapter and resolve EOS only at runtime."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != MCF_SHAPED_FORMAT
        or payload.get("benchmark") != "rwku"
        or payload.get("training_only") is not True
    ):
        raise FactAuditError("Not an MCF-shaped RWKU training request export")
    eos = getattr(tokenizer, "eos_token", None)
    if not eos:
        raise FactAuditError("Runtime tokenizer has no EOS token")
    resolved: List[Dict[str, Any]] = []
    for request in payload.get("requests", []):
        if request.get("format") != MCF_SHAPED_FORMAT or request.get("training_only") is not True:
            raise FactAuditError("Mixed or non-training record in RWKU adapter export")
        if request.get("target_true", {}).get("str") != RUNTIME_EOS_MARKER:
            raise FactAuditError("RWKU adapter target_true must be the runtime EOS marker")
        row = json.loads(json.dumps(request))
        row["target_true"]["str"] = str(eos)
        resolved.append(row)
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fact-overrides", type=Path, required=True)
    parser.add_argument("--fact-holdout-fraction", type=float, default=0.25)
    parser.add_argument("--prompt-holdout-per-seen-fact", type=int, default=1)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--strict-fact-audit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--export-mcf-shaped-training-requests", type=Path)
    args = parser.parse_args()
    result = build_probe_artifacts(
        data_root=args.data_root,
        seed=args.seed,
        output_dir=args.output_dir,
        fact_overrides_path=args.fact_overrides,
        unseen_fact_fraction=args.fact_holdout_fraction,
        prompt_holdout_per_seen_fact=args.prompt_holdout_per_seen_fact,
        split_seed=args.split_seed,
        strict=args.strict_fact_audit,
        allow_download=not args.no_download,
    )
    if args.export_mcf_shaped_training_requests:
        export_mcf_shaped_training_requests(
            result["artifacts"]["training_bundle.json"],
            destination=args.export_mcf_shaped_training_requests,
        )
    print(json.dumps(result["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
