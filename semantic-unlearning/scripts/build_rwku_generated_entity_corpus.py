#!/usr/bin/env python3
"""Generate an independent target-only RWKU entity corpus with provenance.

There is intentionally no data-root or official-evaluation argument. This
program cannot discover or open RWKU Level 1/2/3, MIA, neighbor, utility, or
fluency files. Model libraries are imported only after ``--dry-run`` exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from build_rwku_entity_facts import (
    ENTITY_FACT_SCHEMA_VERSION,
    entity_fact_id,
    normalize_identity,
    normalize_text,
    view_content_sha256,
)
from rwku_artifact_access import (
    TARGET_ONLY_PROTOCOL_LABEL,
    TARGET_ONLY_PROTOCOL_STATUS,
    make_artifact,
    sha256_file,
    sha256_json,
    write_artifact,
)


GENERATOR_SCHEMA_VERSION = "rwku_target_corpus_generator_receipt_v1"
EXTRACTOR_REVISION = "strict_json_fact_extractor_v2"
PARSER_IMPLEMENTATION_REVISION = "json_decoder_raw_decode_scanner_v2"
STRICT_V2_CONFIGURATION_ID = "llama32_3b_target_corpus_v2_strict_json"
ATOMIC_V3_CONFIGURATION_ID = "llama32_3b_target_corpus_v3_atomic_facts"
ATOMIC_EXTRACTOR_REVISION = "atomic_relation_fact_extractor_v1"
ATOMIC_PARSER_REVISION = "complete_json_object_atomic_v1"
RELATION_REGISTRY_SCHEMA_VERSION = "rwku_relation_template_registry_v1"

ATOMIC_RELATION_IDS = (
    "birth_date",
    "birth_place",
    "nationality",
    "occupation",
    "pseudonym",
    "first_published_work",
    "first_published_novel",
    "notable_novel",
    "notable_short_story",
    "notable_series",
    "literary_genre",
    "spouse",
    "education",
    "professional_affiliation",
    "major_award",
    "screen_adaptation",
)
ATOMIC_OUTPUT_FIELDS = (
    "status",
    "subject",
    "relation_id",
    "answer",
    "evidence_sentence",
)

REQUIRED_FACT_FIELDS = (
    "subject",
    "relation_id",
    "answer",
    "direct_question",
    "cloze",
)
NEGATIVE_OR_NULL_ANSWERS = {
    "no",
    "none",
    "unknown",
    "not applicable",
    "n/a",
    "na",
    "null",
    "false",
    "deceased=false",
    "deceased = false",
}
UNCERTAIN_ANSWER_MARKERS = (
    "uncertain",
    "possibly",
    "perhaps",
    "maybe",
    "likely",
    "allegedly",
    "unconfirmed",
    "unclear",
)
ATOMIC_FORBIDDEN_ANSWER_MARKERS = (
    "uncertain",
    "possibly",
    "maybe",
    "allegedly",
    "unconfirmed",
    "unclear",
)
FORBIDDEN_OFFICIAL_FILENAMES = {
    "intro.json",
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
}
STRICT_V2_CATEGORIES = (
    "identity and biography",
    "works and publications",
    "dates and places",
    "roles and affiliations",
    "collaborators, family and aliases",
    "awards and adaptations",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def local_snapshot_identity(model_path: str) -> Dict[str, Any]:
    path = Path(model_path).expanduser()
    identity: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_dir(),
        "metadata_sha256": {},
        "weight_files": [],
    }
    if not path.is_dir():
        return identity
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ):
        candidate = path / name
        if candidate.is_file():
            identity["metadata_sha256"][name] = sha256_file(candidate)
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        for candidate in sorted(path.glob(pattern)):
            stat = candidate.stat()
            row: Dict[str, Any] = {
                "name": candidate.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
            if candidate.is_symlink():
                row["symlink_target"] = str(candidate.readlink())
            identity["weight_files"].append(row)
    return identity


def _validate_strict_v2_config(config: Mapping[str, Any]) -> None:
    if config.get("declared_model_family") != "Llama-3.2-3B-Instruct":
        raise ValueError("Strict v2 config must target Llama-3.2-3B-Instruct")
    if config.get("method_extension") is not True:
        raise ValueError("Strict v2 config must declare method_extension=true")
    decoding = config.get("decoding")
    if not isinstance(decoding, Mapping):
        raise ValueError("Strict v2 config requires deterministic decoding")
    expected_decoding = {
        "do_sample": False,
        "max_new_tokens": 1536,
        "num_return_sequences": 1,
    }
    for key, expected in expected_decoding.items():
        if decoding.get(key) != expected:
            raise ValueError(f"Strict v2 decoding requires {key}={expected!r}")
    forbidden = {"temperature", "top_p"} & set(decoding)
    if forbidden:
        raise ValueError(
            "Strict v2 deterministic decoding forbids: " + ", ".join(sorted(forbidden))
        )
    extraction = config.get("fact_extraction")
    if (
        not isinstance(extraction, Mapping)
        or extraction.get("reverse_prompts_enabled") is not False
    ):
        raise ValueError("Strict v2 config must disable reverse prompts")
    joined_templates = "\n".join(config.get("prompt_templates", [])).casefold()
    missing_categories = [
        category
        for category in STRICT_V2_CATEGORIES
        if category not in joined_templates
    ]
    if missing_categories:
        raise ValueError(
            "Strict v2 prompt templates omit categories: "
            + ", ".join(missing_categories)
        )


def _validate_atomic_v3_config(config: Mapping[str, Any]) -> None:
    if config.get("declared_model_family") != "Llama-3.2-3B-Instruct":
        raise ValueError("Atomic v3 config must target Llama-3.2-3B-Instruct")
    if config.get("method_extension") is not True:
        raise ValueError("Atomic v3 config must declare method_extension=true")
    if config.get("generation_mode") != "atomic_relation_queries":
        raise ValueError("Atomic v3 config must use atomic_relation_queries")
    expected_decoding = {
        "do_sample": False,
        "max_new_tokens": 192,
        "num_return_sequences": 1,
    }
    if config.get("decoding") != expected_decoding:
        raise ValueError(
            "Atomic v3 config requires exactly deterministic decoding "
            f"{expected_decoding!r}"
        )
    extraction = config.get("fact_extraction")
    if not isinstance(extraction, Mapping):
        raise ValueError("Atomic v3 config requires fact_extraction")
    if extraction.get("implementation") != ATOMIC_EXTRACTOR_REVISION:
        raise ValueError("Atomic v3 config has the wrong extractor implementation")
    if extraction.get("parser_implementation_revision") != ATOMIC_PARSER_REVISION:
        raise ValueError("Atomic v3 config has the wrong parser revision")
    if extraction.get("reverse_prompts_enabled") is not False:
        raise ValueError("Atomic v3 config must disable reverse prompts")


def _load_generation_config(path: Path) -> Tuple[Dict[str, Any], str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if (
        not isinstance(config, dict)
        or config.get("schema_version") != "rwku_target_corpus_generation_v1"
    ):
        raise ValueError("Unsupported target-corpus generation configuration")
    if config.get("official_evaluation_access") != "forbidden":
        raise ValueError(
            "Generation configuration must forbid official evaluation access"
        )
    configuration_id = config.get("configuration_id")
    if configuration_id == ATOMIC_V3_CONFIGURATION_ID:
        _validate_atomic_v3_config(config)
        return config, sha256_file(Path(path))
    templates = config.get("prompt_templates")
    if (
        not isinstance(templates, list)
        or not templates
        or not all(isinstance(item, str) and item.strip() for item in templates)
    ):
        raise ValueError("Generation configuration requires prompt_templates")
    if configuration_id == STRICT_V2_CONFIGURATION_ID:
        _validate_strict_v2_config(config)
    return config, sha256_file(Path(path))


def _validate_relation_prompt_template(
    template: str,
    *,
    relation_id: str,
    template_name: str,
    require_question: bool,
    require_cloze: bool,
) -> None:
    if template.count("{subject}") != 1:
        raise ValueError(
            f"{relation_id} {template_name} must contain {{subject}} exactly once"
        )
    without_subject = template.replace("{subject}", "")
    if "{" in without_subject or "}" in without_subject:
        raise ValueError(f"{relation_id} {template_name} contains another placeholder")
    if require_question and not template.endswith("?"):
        raise ValueError(f"{relation_id} direct question must end with ?")
    blank_count = template.count("___")
    if require_cloze and blank_count != 1:
        raise ValueError(f"{relation_id} cloze must contain exactly one ___")
    if not require_cloze and blank_count:
        raise ValueError(f"{relation_id} direct question cannot contain ___")


def load_relation_template_registry(
    path: Path,
    *,
    target_entity: str | None = None,
) -> Tuple[List[Dict[str, Any]], str]:
    source = Path(path)
    if source.name.casefold() in FORBIDDEN_OFFICIAL_FILENAMES:
        raise ValueError(
            f"Official RWKU evaluation file cannot be used as a relation registry: {source}"
        )
    with source.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    if (
        not isinstance(registry, dict)
        or registry.get("schema_version") != RELATION_REGISTRY_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported RWKU relation-template registry")
    relations = registry.get("relations")
    if not isinstance(relations, list):
        raise ValueError("Relation-template registry requires a relations list")
    if target_entity and normalize_identity(target_entity) in normalize_identity(
        json.dumps(registry, ensure_ascii=False)
    ):
        raise ValueError("Relation-template registry must be target-independent")

    required_fields = {
        "relation_id",
        "generation_instruction",
        "direct_question_template",
        "cloze_template",
        "answer_type",
        "maximum_answer_characters",
        "primary_protocol_enabled",
    }
    validated: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(relations):
        if not isinstance(raw, Mapping) or not required_fields <= set(raw):
            raise ValueError(f"Relation registry row {index} is incomplete")
        relation = dict(raw)
        relation_id = str(relation.get("relation_id", ""))
        if relation_id not in ATOMIC_RELATION_IDS:
            raise ValueError(f"Unsupported atomic relation_id: {relation_id!r}")
        if relation_id in seen:
            raise ValueError(f"Duplicate atomic relation_id: {relation_id}")
        seen.add(relation_id)
        instruction = relation.get("generation_instruction")
        answer_type = relation.get("answer_type")
        maximum = relation.get("maximum_answer_characters")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError(f"{relation_id} requires a generation instruction")
        if "{" in instruction or "}" in instruction:
            raise ValueError(f"{relation_id} instruction cannot contain placeholders")
        if not isinstance(answer_type, str) or not answer_type.strip():
            raise ValueError(f"{relation_id} requires answer_type")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise ValueError(f"{relation_id} has an invalid maximum answer length")
        if not isinstance(relation.get("primary_protocol_enabled"), bool):
            raise ValueError(f"{relation_id} requires a boolean primary protocol flag")
        direct = relation.get("direct_question_template")
        cloze = relation.get("cloze_template")
        if not isinstance(direct, str) or not isinstance(cloze, str):
            raise ValueError(f"{relation_id} requires direct and cloze templates")
        _validate_relation_prompt_template(
            direct,
            relation_id=relation_id,
            template_name="direct_question_template",
            require_question=True,
            require_cloze=False,
        )
        _validate_relation_prompt_template(
            cloze,
            relation_id=relation_id,
            template_name="cloze_template",
            require_question=False,
            require_cloze=True,
        )
        validated.append(relation)
    if tuple(relation["relation_id"] for relation in validated) != ATOMIC_RELATION_IDS:
        raise ValueError(
            "Relation-template registry must contain the frozen people-relation "
            "inventory in canonical order"
        )
    return validated, sha256_file(source)


def structured_generation_prompt(template: str, *, target_entity: str) -> str:
    base = template.format(target_entity=target_entity)
    return (
        f"{base}\n\n"
        "Return newline-delimited compact JSON objects only, with exactly one "
        "complete JSON object per line. Do not output a JSON array, Markdown "
        "fences, headings, bullets, commentary, explanatory prose, or any text "
        "outside the JSON objects. Every object must contain nonempty string "
        'fields "subject", "relation_id", "answer", "direct_question", and '
        '"cloze". '
        f"The subject must be exactly {json.dumps(target_entity)}. relation_id "
        "must be stable lowercase snake_case. direct_question must explicitly "
        "ask for the answer and end with a question mark. cloze must contain "
        "exactly one ___ marker. No other placeholders are allowed in any field. "
        "Include only affirmative, independently checkable facts. Omit uncertain "
        "facts and negative or null pseudo-facts such as died=No, spouse=None, "
        "award=Unknown, not applicable, n/a, or deceased=false."
    )


def atomic_generation_prompt(
    relation: Mapping[str, Any],
    *,
    target_entity: str,
) -> str:
    relation_id = str(relation["relation_id"])
    instruction = str(relation["generation_instruction"])
    maximum = int(relation["maximum_answer_characters"])
    return (
        "Answer one atomic relation query about one person. "
        f"Subject: {json.dumps(target_entity, ensure_ascii=False)}. "
        f"Requested relation_id: {json.dumps(relation_id)}. "
        f"Task: {instruction} "
        "Return exactly one compact JSON object and no other text. "
        "Do not return Markdown, a JSON array, headings, commentary, or prose "
        "outside the object. The object must contain exactly these string fields: "
        '"status", "subject", "relation_id", "answer", and '
        '"evidence_sentence". Set status to exactly "known" only when the fact '
        'is confidently known; otherwise set status to exactly "unknown" and '
        "use empty strings for answer and evidence_sentence. Copy subject and "
        "relation_id exactly from this request. For status known, answer must be "
        f"at most {maximum} characters and evidence_sentence must be one factual "
        "sentence that explicitly contains both the subject and the answer. Do "
        "not emit placeholders, guesses, negative pseudo-facts, direct questions, "
        "or cloze prompts."
    )


def build_atomic_generation_requests(
    relations: Sequence[Mapping[str, Any]],
    *,
    target_entity: str,
) -> List[Dict[str, Any]]:
    requests: List[Dict[str, Any]] = []
    for registry_index, relation in enumerate(relations):
        if relation.get("primary_protocol_enabled") is not True:
            continue
        prompt = atomic_generation_prompt(relation, target_entity=target_entity)
        requests.append(
            {
                "request_index": len(requests),
                "registry_index": registry_index,
                "relation_id": str(relation["relation_id"]),
                "relation_template_sha256": sha256_json(relation),
                "prompt": prompt,
                "prompt_sha256": sha256_json(prompt),
            }
        )
    if not requests:
        raise ValueError("Relation registry has no primary-protocol relations")
    return requests


def annotate_atomic_raw_outputs(
    raw_outputs: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    annotated: List[Dict[str, Any]] = []
    for raw in raw_outputs:
        output = dict(raw)
        prompt_index = output.get("prompt_index")
        if isinstance(prompt_index, int) and 0 <= prompt_index < len(requests):
            request = requests[prompt_index]
            output.update(
                {
                    "request_index": int(request["request_index"]),
                    "registry_index": int(request["registry_index"]),
                    "requested_relation_id": str(request["relation_id"]),
                    "generation_request_sha256": str(request["prompt_sha256"]),
                    "relation_template_sha256": str(
                        request["relation_template_sha256"]
                    ),
                }
            )
        annotated.append(output)
    return annotated


def _parse_atomic_json_object(text: str) -> Tuple[Dict[str, Any] | None, str]:
    stripped = str(text).strip()
    if not stripped:
        return None, "empty_output"
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        return None, "no_complete_json_object"
    if not isinstance(value, dict):
        return None, "atomic_output_not_object"
    return dict(value), "complete_json_object"


def _atomic_answer_rejection(answer: str) -> str | None:
    normalized = normalize_identity(answer)
    normalized_without_punctuation = re.sub(r"[.!?]+$", "", normalized).strip()
    if normalized_without_punctuation in NEGATIVE_OR_NULL_ANSWERS or re.fullmatch(
        r"deceased\s*=\s*false", normalized_without_punctuation
    ):
        return "negative_or_null_answer"
    if any(marker in normalized for marker in ATOMIC_FORBIDDEN_ANSWER_MARKERS):
        return "uncertain_answer"
    return None


def _validate_atomic_candidate(
    candidate: Mapping[str, Any],
    *,
    target_entity: str,
    relation: Mapping[str, Any],
) -> Tuple[str, Dict[str, str], str | None]:
    if set(candidate) != set(ATOMIC_OUTPUT_FIELDS):
        return "rejected", {}, "atomic_output_fields_mismatch"
    if any(not isinstance(candidate.get(field), str) for field in ATOMIC_OUTPUT_FIELDS):
        return "rejected", {}, "atomic_output_fields_not_strings"
    fields = {field: str(candidate[field]) for field in ATOMIC_OUTPUT_FIELDS}
    status = fields["status"]
    if status not in {"known", "unknown"}:
        return "rejected", fields, "invalid_status"
    if normalize_text(fields["subject"]) != normalize_text(target_entity):
        return "rejected", fields, "subject_mismatch"
    if fields["relation_id"] != str(relation["relation_id"]):
        return "rejected", fields, "relation_mismatch"
    if status == "unknown":
        return "unknown", fields, None

    answer = normalize_text(fields["answer"])
    evidence = normalize_text(fields["evidence_sentence"])
    if not answer:
        return "rejected", fields, "missing_or_empty_answer"
    if not evidence:
        return "rejected", fields, "missing_or_empty_evidence_sentence"
    if normalize_identity(answer) == normalize_identity(target_entity):
        return "rejected", fields, "answer_is_target_entity"
    if len(answer) > int(relation["maximum_answer_characters"]):
        return "rejected", fields, "answer_exceeds_relation_maximum"
    answer_rejection = _atomic_answer_rejection(answer)
    if answer_rejection:
        return "rejected", fields, answer_rejection
    if _contains_forbidden_placeholder(
        "answer", answer
    ) or _contains_forbidden_placeholder("evidence_sentence", evidence):
        return "rejected", fields, "forbidden_placeholder"
    normalized_evidence = normalize_identity(evidence)
    if normalize_identity(target_entity) not in normalized_evidence:
        return "rejected", fields, "evidence_missing_subject"
    if normalize_identity(answer) not in normalized_evidence:
        return "rejected", fields, "evidence_missing_answer"
    fields["answer"] = answer
    fields["evidence_sentence"] = evidence
    fields["subject"] = normalize_text(fields["subject"])
    return "known", fields, None


def extract_atomic_facts(
    raw_outputs: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    *,
    entity_id: str,
    target_entity: str,
    relation_registry_path: Path,
    relation_registry_sha256: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    relations_by_id = {
        str(relation["relation_id"]): dict(relation) for relation in relations
    }
    outputs_by_prompt: Dict[int, List[Tuple[int, Mapping[str, Any]]]] = {}
    for output_index, output in enumerate(raw_outputs):
        prompt_index = output.get("prompt_index")
        if isinstance(prompt_index, int):
            outputs_by_prompt.setdefault(prompt_index, []).append(
                (output_index, output)
            )

    accepted: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    unknown_relations: List[Dict[str, Any]] = []
    output_identities: List[Dict[str, Any]] = []
    parse_mode_counts: Counter[str] = Counter()
    for request in requests:
        request_index = int(request["request_index"])
        relation_id = str(request["relation_id"])
        relation = relations_by_id[relation_id]
        matching_outputs = outputs_by_prompt.get(request_index, [])
        identity: Dict[str, Any] = {
            "request_index": request_index,
            "registry_index": int(request["registry_index"]),
            "requested_relation_id": relation_id,
            "generation_request_sha256": str(request["prompt_sha256"]),
            "relation_template_sha256": str(request["relation_template_sha256"]),
            "output_count": len(matching_outputs),
        }
        if len(matching_outputs) != 1:
            reason = (
                "missing_generation_output"
                if not matching_outputs
                else "multiple_generation_outputs"
            )
            identity.update(
                {
                    "parse_mode": "not_parsed",
                    "parsed_object_count": 0,
                    "outcome": "rejected",
                    "reason": reason,
                }
            )
            output_identities.append(identity)
            rejected.append(
                {
                    "request_index": request_index,
                    "relation_id": relation_id,
                    "reason": reason,
                }
            )
            continue

        output_index, output = matching_outputs[0]
        generated_text = str(output.get("generated_text", ""))
        candidate, parse_mode = _parse_atomic_json_object(generated_text)
        parse_mode_counts[parse_mode] += 1
        identity.update(
            {
                "output_index": output_index,
                "sequence_index": output.get("sequence_index"),
                "chat_template_used": bool(output.get("chat_template_used", False)),
                "generated_text_sha256": hashlib.sha256(
                    generated_text.encode("utf-8")
                ).hexdigest(),
                "parse_mode": parse_mode,
                "parsed_object_count": int(candidate is not None),
            }
        )
        if candidate is None:
            identity.update({"outcome": "rejected", "reason": parse_mode})
            output_identities.append(identity)
            rejected.append(
                {
                    "request_index": request_index,
                    "output_index": output_index,
                    "relation_id": relation_id,
                    "reason": parse_mode,
                }
            )
            continue

        outcome, fields, reason = _validate_atomic_candidate(
            candidate,
            target_entity=target_entity,
            relation=relation,
        )
        identity["candidate_sha256"] = sha256_json(candidate)
        identity["outcome"] = outcome
        if reason:
            identity["reason"] = reason
        output_identities.append(identity)
        if outcome == "unknown":
            unknown_relations.append(
                {
                    "request_index": request_index,
                    "output_index": output_index,
                    "relation_id": relation_id,
                    "status": "unknown",
                }
            )
            continue
        if outcome == "rejected":
            rejected.append(
                {
                    "request_index": request_index,
                    "output_index": output_index,
                    "relation_id": relation_id,
                    "reason": str(reason),
                    "candidate": candidate,
                }
            )
            continue

        answer = fields["answer"]
        fact_id = entity_fact_id(entity_id, relation_id, answer)
        accepted.append(
            {
                "schema_version": "rwku_generated_atomic_fact_v1",
                "status": "known",
                "entity_id": entity_id,
                "subject": target_entity,
                "relation_id": relation_id,
                "answer": answer,
                "evidence_sentence": fields["evidence_sentence"],
                "fact_id": fact_id,
                "request_index": request_index,
                "output_index": output_index,
                "raw_output_sha256": sha256_json(candidate),
                "generation_request_sha256": str(request["prompt_sha256"]),
                "relation_registry_path": str(relation_registry_path),
                "relation_registry_sha256": relation_registry_sha256,
                "relation_template_sha256": str(request["relation_template_sha256"]),
            }
        )

    diagnostics = {
        "parser_implementation_revision": ATOMIC_PARSER_REVISION,
        "output_identities": output_identities,
        "parse_mode_counts": dict(sorted(parse_mode_counts.items())),
        "requested_relation_count": len(requests),
        "known_relation_count": len(accepted),
        "unknown_relation_count": len(unknown_relations),
        "rejected_relation_count": len(rejected),
        "unknown_relations": unknown_relations,
    }
    return accepted, rejected, diagnostics


def _compile_atomic_view(
    *,
    style: str,
    query: str,
    sensitive_answer: str,
    optimization_answer: str,
    atomic_fact: Mapping[str, Any],
) -> Dict[str, Any]:
    view = {
        "schema_version": ENTITY_FACT_SCHEMA_VERSION,
        "query": query,
        "level": "generated",
        "query_type": style,
        "prompt_style": style,
        "canonical_sensitive_answer": sensitive_answer,
        "sensitive_answer_alias": optimization_answer,
        "source_record_sha256": str(atomic_fact["raw_output_sha256"]),
        "source_record_sha256_values": [str(atomic_fact["raw_output_sha256"])],
        "source_file": "generated_raw_corpus.json",
        "source_row_index": int(atomic_fact["output_index"]),
        "boundary_expanding": False,
        "fact_id": str(atomic_fact["fact_id"]),
        "relation_id": str(atomic_fact["relation_id"]),
        "entity_id": str(atomic_fact["entity_id"]),
        "subject": str(atomic_fact["subject"]),
        "training_allowed": True,
    }
    view["view_content_sha256"] = view_content_sha256(view)
    view["view_id"] = hashlib.sha256(
        f"{view['fact_id']}:{view['view_content_sha256']}".encode("utf-8")
    ).hexdigest()
    return view


def compile_atomic_facts_to_entity_facts(
    atomic_facts: Sequence[Mapping[str, Any]],
    relations: Sequence[Mapping[str, Any]],
    *,
    relation_registry_path: Path,
    relation_registry_sha256: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    relations_by_id = {
        str(relation["relation_id"]): dict(relation) for relation in relations
    }
    facts_by_id: Dict[str, Dict[str, Any]] = {}
    canonical_atomic_by_id: Dict[str, Dict[str, Any]] = {}
    duplicate_count = 0
    for raw_atomic in atomic_facts:
        atomic_fact = dict(raw_atomic)
        relation_id = str(atomic_fact["relation_id"])
        relation = relations_by_id[relation_id]
        subject = str(atomic_fact["subject"])
        answer = str(atomic_fact["answer"])
        fact_id = str(atomic_fact["fact_id"])
        direct_template = str(relation["direct_question_template"])
        cloze_template = str(relation["cloze_template"])
        direct = direct_template.replace("{subject}", subject)
        cloze = cloze_template.replace("{subject}", subject)
        if direct.count(subject) != 1 or not direct.endswith("?"):
            raise ValueError(
                f"Compiled direct question for {relation_id} must contain its "
                "subject exactly once and end with ?"
            )
        if cloze.count(subject) != 1 or cloze.count("___") != 1:
            raise ValueError(
                f"Compiled cloze for {relation_id} must contain its subject "
                "exactly once and one ___"
            )

        source_record = {
            "source_file": "generated_raw_corpus.json",
            "source_row_index": int(atomic_fact["output_index"]),
            "source_record_sha256": str(atomic_fact["raw_output_sha256"]),
            "level": "generated",
            "query_type": "target_only_atomic_relation",
            "normalized_query_hash": hashlib.sha256(
                normalize_identity(direct).encode("utf-8")
            ).hexdigest(),
            "original_answer": answer,
            "evidence_sentence": str(atomic_fact["evidence_sentence"]),
            "assigned_relation_id": relation_id,
            "assigned_fact_id": fact_id,
            "generation_request_sha256": str(atomic_fact["generation_request_sha256"]),
            "relation_template_sha256": str(atomic_fact["relation_template_sha256"]),
        }
        if fact_id in facts_by_id:
            duplicate_count += 1
            existing = facts_by_id[fact_id]
            if source_record["source_record_sha256"] not in existing["source_hashes"]:
                existing["source_hashes"].append(source_record["source_record_sha256"])
                existing["source_records"].append(source_record)
            continue

        fact = {
            "schema_version": ENTITY_FACT_SCHEMA_VERSION,
            "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
            "protocol_status": TARGET_ONLY_PROTOCOL_STATUS,
            "entity_id": str(atomic_fact["entity_id"]),
            "subject": subject,
            "subject_aliases": [],
            "fact_id": fact_id,
            "relation_id": relation_id,
            "canonical_sensitive_answer": answer,
            "sensitive_answer_aliases": [],
            "source_records": [source_record],
            "optimization_views": [],
            "held_out_views": [],
            "partition": "generated_training_fact",
            "training_allowed": True,
            "source_hashes": [str(atomic_fact["raw_output_sha256"])],
            "relation_assignment_provenance": [
                {
                    "method": ATOMIC_EXTRACTOR_REVISION,
                    "relation_registry_path": str(relation_registry_path),
                    "relation_registry_sha256": relation_registry_sha256,
                    "relation_template_sha256": str(
                        atomic_fact["relation_template_sha256"]
                    ),
                }
            ],
            "manual_override_sha256": "",
        }
        view_specs: List[Tuple[str, str, str]] = [
            ("direct question", direct, answer),
            ("cloze", cloze, answer),
            (
                "deterministic paraphrase",
                f"In different words, answer this question: {direct}",
                answer,
            ),
        ]
        answer_words = answer.split()
        if len(answer_words) >= 2:
            prefix_width = max(1, len(answer_words) // 2)
            prefix = " ".join(answer_words[:prefix_width])
            suffix = " ".join(answer_words[prefix_width:])
            view_specs.append(
                ("forced-prefix", f"{direct}\nAnswer prefix: {prefix}", suffix)
            )
        fact["optimization_views"] = [
            _compile_atomic_view(
                style=style,
                query=query,
                sensitive_answer=answer,
                optimization_answer=optimization_answer,
                atomic_fact=atomic_fact,
            )
            for style, query, optimization_answer in view_specs
        ]
        direct_views = [
            view
            for view in fact["optimization_views"]
            if view["prompt_style"] == "direct question"
        ]
        if len(direct_views) != 1:
            raise RuntimeError(
                f"Atomic fact {fact_id} did not compile exactly one direct view"
            )
        facts_by_id[fact_id] = fact
        canonical_atomic_by_id[fact_id] = atomic_fact
    return (
        [facts_by_id[fact_id] for fact_id in sorted(facts_by_id)],
        [canonical_atomic_by_id[fact_id] for fact_id in sorted(canonical_atomic_by_id)],
        duplicate_count,
    )


def render_chat_prompt(tokenizer: Any, prompt: str) -> Tuple[str, bool]:
    """Render one user turn when the tokenizer exposes a valid chat template."""

    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_template):
        try:
            rendered = apply_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered, True
        except Exception:
            # Tokenizer implementations surface missing/invalid chat templates
            # through several exception types (including template-engine errors).
            # Generation can safely fall back because the unrendered prompt is
            # already a complete, structured instruction.
            pass
    return prompt, False


def _flatten_json_objects(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        return [dict(value)]
    if isinstance(value, list):
        return [dict(item) for item in value if isinstance(item, dict)]
    return []


def _decode_json_sequence(text: str) -> List[Any] | None:
    """Decode one or more whitespace-separated complete JSON values."""

    decoder = json.JSONDecoder()
    values: List[Any] = []
    position = 0
    while position < len(text):
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text):
            break
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            return None
        values.append(value)
        position = end
    return values if values else None


def _scan_json_values(text: str) -> List[Any]:
    """Find syntactically complete embedded JSON values without prose inference."""

    decoder = json.JSONDecoder()
    values: List[Any] = []
    position = 0
    while position < len(text):
        object_start = text.find("{", position)
        array_start = text.find("[", position)
        starts = [index for index in (object_start, array_start) if index >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            position = start + 1
            continue
        values.append(value)
        position = end
    return values


def _mode_for_values(
    values: Sequence[Any],
    *,
    text: str,
    fenced: bool,
    scanned: bool,
) -> str:
    prefix = "fenced_" if fenced else ""
    if scanned:
        return f"{prefix}raw_decode_scan"
    if len(values) > 1:
        return f"{prefix}ndjson"
    value = values[0]
    pretty = "\n" in text.strip()
    if isinstance(value, list):
        return f"{prefix}{'pretty_printed_' if pretty else 'complete_'}array"
    if isinstance(value, dict):
        return f"{prefix}{'pretty_printed_' if pretty else 'complete_'}object"
    return f"{prefix}unsupported_json_value"


def parse_json_objects(text: str) -> Dict[str, Any]:
    """Parse supported valid JSON surfaces and deduplicate canonical objects."""

    stripped = str(text).strip()
    parsed_rows: List[Tuple[str, Dict[str, Any]]] = []
    if stripped:
        complete_values = _decode_json_sequence(stripped)
        if complete_values is not None:
            mode = _mode_for_values(
                complete_values,
                text=stripped,
                fenced=False,
                scanned=False,
            )
            for value in complete_values:
                parsed_rows.extend(
                    (mode, candidate) for candidate in _flatten_json_objects(value)
                )
        else:
            fenced_spans: List[Tuple[int, int]] = []
            for match in re.finditer(
                r"```(?:json)?\s*(.*?)```",
                stripped,
                flags=re.IGNORECASE | re.DOTALL,
            ):
                fenced_spans.append(match.span())
                block = match.group(1).strip()
                values = _decode_json_sequence(block)
                if values is None:
                    continue
                mode = _mode_for_values(
                    values,
                    text=block,
                    fenced=True,
                    scanned=False,
                )
                for value in values:
                    parsed_rows.extend(
                        (mode, candidate) for candidate in _flatten_json_objects(value)
                    )
            outside = list(stripped)
            for start, end in fenced_spans:
                outside[start:end] = " " * (end - start)
            outside_text = "".join(outside)
            scanned_values = _scan_json_values(outside_text)
            if scanned_values:
                mode = _mode_for_values(
                    scanned_values,
                    text=outside_text,
                    fenced=False,
                    scanned=True,
                )
                for value in scanned_values:
                    parsed_rows.extend(
                        (mode, candidate) for candidate in _flatten_json_objects(value)
                    )

    unique: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    duplicate_count = 0
    for mode, candidate in parsed_rows:
        digest = sha256_json(candidate)
        if digest in unique:
            duplicate_count += 1
            continue
        unique[digest] = (mode, candidate)
    ordered = list(unique.values())
    modes = list(dict.fromkeys(mode for mode, _ in ordered))
    parse_mode = (
        modes[0]
        if len(modes) == 1
        else "mixed_json_sources"
        if modes
        else "no_parseable_json"
    )
    return {
        "objects": [candidate for _, candidate in ordered],
        "object_sha256": [sha256_json(candidate) for _, candidate in ordered],
        "parse_mode": parse_mode,
        "parse_modes": modes,
        "parsed_object_count": len(ordered),
        "duplicate_object_count": duplicate_count,
    }


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    """Compatibility wrapper for callers that need only parsed objects."""

    return list(parse_json_objects(text)["objects"])


def _required_string(candidate: Mapping[str, Any], field: str) -> str:
    value = candidate.get(field)
    return value.strip() if isinstance(value, str) else ""


def _contains_forbidden_placeholder(field: str, value: str) -> bool:
    underscore_runs = re.findall(r"_{2,}", value)
    if field == "cloze":
        if underscore_runs != ["___"]:
            return True
    elif underscore_runs:
        return True
    if re.search(r"\{[^{}\n]*\}", value):
        return True
    if re.search(r"<[^>\n]+>", value):
        return True
    if re.search(r"\[[^\]\n]+\]", value):
        return True
    if re.search(r"\b(?:blank|mask|placeholder|answer_here)\b", value, re.IGNORECASE):
        return True
    return False


def _rejection_reason(
    candidate: Mapping[str, Any],
    *,
    target_entity: str,
) -> Tuple[str | None, Dict[str, str]]:
    fields = {
        field: _required_string(candidate, field) for field in REQUIRED_FACT_FIELDS
    }
    for field in REQUIRED_FACT_FIELDS:
        if not fields[field]:
            return f"missing_or_empty_{field}", fields
    if candidate.get("subject") != target_entity:
        return "subject_mismatch", fields
    if not re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*", fields["relation_id"]):
        return "invalid_relation_id", fields
    if fields["cloze"].count("___") != 1 or re.findall(r"_{2,}", fields["cloze"]) != [
        "___"
    ]:
        return "invalid_cloze_marker_count", fields
    if any(
        _contains_forbidden_placeholder(field, value) for field, value in fields.items()
    ):
        return "forbidden_placeholder", fields
    if not fields["direct_question"].endswith("?"):
        return "direct_question_missing_question_mark", fields
    normalized_answer = normalize_identity(fields["answer"])
    negative_answer = re.sub(r"[.!?]+$", "", normalized_answer).strip()
    if negative_answer in NEGATIVE_OR_NULL_ANSWERS or re.fullmatch(
        r"deceased\s*=\s*false", negative_answer
    ):
        return "negative_or_null_answer", fields
    if any(marker in normalized_answer for marker in UNCERTAIN_ANSWER_MARKERS):
        return "uncertain_answer", fields
    return None, fields


def extract_generated_facts_detailed(
    raw_outputs: Sequence[Mapping[str, Any]],
    *,
    entity_id: str,
    target_entity: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Strictly accept only complete, target-matching structured facts."""

    accepted_by_id: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    output_diagnostics: List[Dict[str, Any]] = []
    globally_seen_objects: set[str] = set()
    cross_output_duplicate_count = 0
    for output_index, output in enumerate(raw_outputs):
        text = str(output.get("generated_text", ""))
        parsed = parse_json_objects(text)
        output_diagnostics.append(
            {
                "output_index": output_index,
                "prompt_index": output.get("prompt_index"),
                "sequence_index": output.get("sequence_index"),
                "parse_mode": parsed["parse_mode"],
                "parse_modes": parsed["parse_modes"],
                "parsed_object_count": parsed["parsed_object_count"],
                "duplicate_object_count": parsed["duplicate_object_count"],
            }
        )
        objects = parsed["objects"]
        if not objects:
            rejected.append(
                {"output_index": output_index, "reason": "no_parseable_json_fact"}
            )
            continue
        for object_index, candidate in enumerate(objects):
            raw_fact_hash = sha256_json(candidate)
            if raw_fact_hash in globally_seen_objects:
                cross_output_duplicate_count += 1
                continue
            globally_seen_objects.add(raw_fact_hash)
            location = {"output_index": output_index, "object_index": object_index}
            reason, fields = _rejection_reason(candidate, target_entity=target_entity)
            if reason is not None:
                rejected.append({**location, "reason": reason, "candidate": candidate})
                continue
            subject = fields["subject"]
            relation_id = fields["relation_id"]
            answer = normalize_text(fields["answer"])
            direct = normalize_text(fields["direct_question"])
            cloze = normalize_text(fields["cloze"])
            fact_id = entity_fact_id(entity_id, relation_id, answer)
            fact = accepted_by_id.setdefault(
                fact_id,
                {
                    "schema_version": ENTITY_FACT_SCHEMA_VERSION,
                    "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
                    "protocol_status": TARGET_ONLY_PROTOCOL_STATUS,
                    "entity_id": entity_id,
                    "subject": target_entity,
                    "subject_aliases": [],
                    "fact_id": fact_id,
                    "relation_id": relation_id,
                    "canonical_sensitive_answer": answer,
                    "sensitive_answer_aliases": [],
                    "source_records": [],
                    "optimization_views": [],
                    "held_out_views": [],
                    "partition": "generated_training_fact",
                    "training_allowed": True,
                    "source_hashes": [],
                    "relation_assignment_provenance": [{"method": EXTRACTOR_REVISION}],
                    "manual_override_sha256": "",
                },
            )
            if raw_fact_hash not in fact["source_hashes"]:
                fact["source_hashes"].append(raw_fact_hash)
                fact["source_records"].append(
                    {
                        "source_file": "generated_raw_corpus.json",
                        "source_row_index": output_index,
                        "source_record_sha256": raw_fact_hash,
                        "level": "generated",
                        "query_type": "target_only_generated",
                        "normalized_query_hash": hashlib.sha256(
                            normalize_identity(direct).encode()
                        ).hexdigest(),
                        "original_answer": answer,
                        "assigned_relation_id": relation_id,
                        "assigned_fact_id": fact_id,
                    }
                )
            view_specs: List[Tuple[str, str, str]] = [
                ("direct question", direct, answer),
                ("cloze", cloze, answer),
                (
                    "deterministic paraphrase",
                    f"In different words, answer this question: {direct}",
                    answer,
                ),
            ]
            raw_aliases = candidate.get("subject_aliases", [])
            aliases = raw_aliases if isinstance(raw_aliases, list) else []
            subject_aliases = [
                normalize_text(str(alias))
                for alias in aliases
                if normalize_text(str(alias))
                and normalize_identity(str(alias)) != normalize_identity(target_entity)
            ]
            for alias in subject_aliases:
                if target_entity in direct:
                    alias_query = direct.replace(target_entity, alias, 1)
                    view_specs.append(
                        ("conservative subject alias", alias_query, answer)
                    )
                    if alias not in fact["subject_aliases"]:
                        fact["subject_aliases"].append(alias)
            answer_words = answer.split()
            if len(answer_words) >= 2:
                prefix_width = max(1, len(answer_words) // 2)
                prefix = " ".join(answer_words[:prefix_width])
                suffix = " ".join(answer_words[prefix_width:])
                view_specs.append(
                    (
                        "forced-prefix",
                        f"{direct}\nAnswer prefix: {prefix}",
                        suffix,
                    )
                )
            for style, query, optimization_answer in view_specs:
                view = {
                    "schema_version": ENTITY_FACT_SCHEMA_VERSION,
                    "query": query,
                    "level": "generated",
                    "query_type": style,
                    "prompt_style": style,
                    "canonical_sensitive_answer": answer,
                    "sensitive_answer_alias": optimization_answer,
                    "source_record_sha256": raw_fact_hash,
                    "source_record_sha256_values": [raw_fact_hash],
                    "source_file": "generated_raw_corpus.json",
                    "source_row_index": output_index,
                    "boundary_expanding": False,
                    "fact_id": fact_id,
                    "relation_id": relation_id,
                    "entity_id": entity_id,
                    "subject": subject,
                    "training_allowed": True,
                }
                view["view_content_sha256"] = view_content_sha256(view)
                view["view_id"] = hashlib.sha256(
                    f"{fact_id}:{view['view_content_sha256']}".encode()
                ).hexdigest()
                if view["view_id"] not in {
                    item["view_id"] for item in fact["optimization_views"]
                }:
                    fact["optimization_views"].append(view)
    parse_mode_counts = Counter(row["parse_mode"] for row in output_diagnostics)
    diagnostics = {
        "parser_implementation_revision": PARSER_IMPLEMENTATION_REVISION,
        "outputs": output_diagnostics,
        "parse_mode_counts": dict(sorted(parse_mode_counts.items())),
        "parsed_object_count": sum(
            row["parsed_object_count"] for row in output_diagnostics
        ),
        "within_output_duplicate_object_count": sum(
            row["duplicate_object_count"] for row in output_diagnostics
        ),
        "cross_output_duplicate_object_count": cross_output_duplicate_count,
        "unique_parsed_object_count": len(globally_seen_objects),
    }
    facts = list(sorted(accepted_by_id.values(), key=lambda item: item["fact_id"]))
    return facts, rejected, diagnostics


def extract_generated_facts(
    raw_outputs: Sequence[Mapping[str, Any]],
    *,
    entity_id: str,
    target_entity: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    facts, rejected, _ = extract_generated_facts_detailed(
        raw_outputs,
        entity_id=entity_id,
        target_entity=target_entity,
    )
    return facts, rejected


def _tokenizer_source_hashes(model_path: str) -> Dict[str, str]:
    root = Path(model_path).expanduser()
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer.model",
        "chat_template.jinja",
    )
    return {name: sha256_file(root / name) for name in names if (root / name).is_file()}


def _model_generate(
    *,
    model_path: str,
    prompts: Sequence[str],
    decoding: Mapping[str, Any],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    # Deliberately lazy: --dry-run returns before torch/transformers imports.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    outputs: List[Dict[str, Any]] = []
    generation_args = dict(decoding)
    num_return = int(generation_args.pop("num_return_sequences", 1))
    chat_template_flags: List[bool] = []
    for prompt_index, prompt in enumerate(prompts):
        rendered_prompt, chat_template_used = render_chat_prompt(tokenizer, prompt)
        chat_template_flags.append(chat_template_used)
        encoded = tokenizer(rendered_prompt, return_tensors="pt")
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                **generation_args,
                num_return_sequences=num_return,
                pad_token_id=tokenizer.eos_token_id,
            )
        prompt_tokens = encoded["input_ids"].shape[1]
        for sequence_index, sequence in enumerate(generated):
            outputs.append(
                {
                    "prompt_index": prompt_index,
                    "sequence_index": sequence_index,
                    "chat_template_used": chat_template_used,
                    "rendered_prompt_sha256": sha256_json(rendered_prompt),
                    "generated_text": tokenizer.decode(
                        sequence[prompt_tokens:], skip_special_tokens=True
                    ),
                }
            )
    tokenizer_identity = {
        "name_or_path": tokenizer.name_or_path,
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "source_file_sha256": _tokenizer_source_hashes(model_path),
        "chat_template_used": bool(chat_template_flags and all(chat_template_flags)),
        "chat_template_used_by_prompt": chat_template_flags,
    }
    return outputs, tokenizer_identity


def validate_independent_resource_path(path: Path) -> None:
    if Path(path).name.casefold() in FORBIDDEN_OFFICIAL_FILENAMES:
        raise ValueError(
            f"Official RWKU evaluation file cannot be supplied to generator: {path}"
        )


def _rejection_reason_counts(rejected: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(row.get("reason", "unknown")) for row in rejected)
    return dict(sorted(counts.items()))


def _build_atomic_generated_corpus(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    config_sha: str,
) -> Dict[str, Any]:
    registry_path = getattr(args, "relation_template_registry", None)
    if registry_path is None:
        raise ValueError("Atomic v3 generation requires --relation-template-registry")
    minimum_accepted_facts = int(getattr(args, "minimum_accepted_facts", 1))
    if minimum_accepted_facts < 1:
        raise ValueError("--minimum-accepted-facts must be at least 1")
    relations, registry_sha = load_relation_template_registry(
        Path(registry_path),
        target_entity=args.target_entity,
    )
    requests = build_atomic_generation_requests(
        relations,
        target_entity=args.target_entity,
    )
    prompts = [str(request["prompt"]) for request in requests]
    independent_resources = []
    for resource in args.independent_resource:
        path = Path(resource)
        validate_independent_resource_path(path)
        independent_resources.append({"path": str(path), "sha256": sha256_file(path)})

    request_identities = [
        {
            key: request[key]
            for key in (
                "request_index",
                "registry_index",
                "relation_id",
                "relation_template_sha256",
                "prompt_sha256",
            )
        }
        for request in requests
    ]
    implementation_sha = sha256_file(Path(__file__))
    common_receipt = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": TARGET_ONLY_PROTOCOL_STATUS,
        "target_entity": args.target_entity,
        "entity_id": args.entity_id,
        "generator_model_identifier": args.generator_model,
        "generator_model_path": args.generator_model,
        "generator_model_revision": args.generator_revision,
        "local_snapshot": args.generator_model,
        "local_snapshot_identity": local_snapshot_identity(args.generator_model),
        "generation_configuration_path": str(args.generation_config),
        "generation_configuration_sha256": config_sha,
        "relation_template_registry_path": str(Path(registry_path)),
        "relation_template_registry_sha256": registry_sha,
        "decoding_parameters": dict(config["decoding"]),
        "random_seeds": [int(args.seed)],
        "independent_resources": independent_resources,
        "official_rwku_records_accessed": False,
        "fact_extractor_implementation": ATOMIC_EXTRACTOR_REVISION,
        "parser_implementation_revision": ATOMIC_PARSER_REVISION,
        "implementation_sha256": implementation_sha,
        "fact_extractor_revision_sha256": implementation_sha,
        "extraction_configuration": dict(config["fact_extraction"]),
        "generation_request_identities": request_identities,
        "requested_relation_count": len(requests),
        "minimum_accepted_facts": minimum_accepted_facts,
    }
    if args.dry_run:
        return {
            **common_receipt,
            "status": "dry_run_validated",
            "torch_imported": "torch" in __import__("sys").modules,
            "chat_template_used": None,
            "known_relation_count": 0,
            "unknown_relation_count": 0,
            "rejected_relation_count": 0,
            "accepted_fact_count": 0,
        }

    if not common_receipt["local_snapshot_identity"]["exists"]:
        raise FileNotFoundError(
            f"Generator model must be a pinned local snapshot: {args.generator_model}"
        )
    raw_outputs, tokenizer_identity = _model_generate(
        model_path=args.generator_model,
        prompts=prompts,
        decoding=config["decoding"],
        seed=args.seed,
    )
    raw = annotate_atomic_raw_outputs(raw_outputs, requests)
    raw_path = args.output_dir / "generated_raw_corpus.json"
    _write_json(raw_path, raw)
    raw_sha = sha256_file(raw_path)
    atomic_facts, rejected, atomic_diagnostics = extract_atomic_facts(
        raw,
        requests,
        relations,
        entity_id=args.entity_id,
        target_entity=args.target_entity,
        relation_registry_path=Path(registry_path),
        relation_registry_sha256=registry_sha,
    )
    (
        facts,
        canonical_atomic_facts,
        duplicate_fact_count,
    ) = compile_atomic_facts_to_entity_facts(
        atomic_facts,
        relations,
        relation_registry_path=Path(registry_path),
        relation_registry_sha256=registry_sha,
    )
    rejection_counts = _rejection_reason_counts(rejected)
    generation_diagnostics = {
        "parser_implementation_revision": ATOMIC_PARSER_REVISION,
        "raw_generated_corpus_path": str(raw_path),
        "raw_generated_corpus_sha256": raw_sha,
        "relation_template_registry_path": str(Path(registry_path)),
        "relation_template_registry_sha256": registry_sha,
        "chat_template_used": tokenizer_identity.get("chat_template_used", False),
        "minimum_accepted_facts": minimum_accepted_facts,
        "accepted_fact_count": len(facts),
        "duplicate_fact_count": duplicate_fact_count,
        "rejection_reason_counts": rejection_counts,
        **atomic_diagnostics,
    }
    diagnostics_path = args.output_dir / "generation_diagnostics.json"
    _write_json(diagnostics_path, generation_diagnostics)

    if len(facts) < minimum_accepted_facts:
        failure_receipt = {
            **common_receipt,
            "status": "failed_below_minimum_accepted_facts",
            "tokenizer_identity": tokenizer_identity,
            "chat_template_used": tokenizer_identity.get("chat_template_used", False),
            "raw_generated_corpus_sha256": raw_sha,
            "output_identities": atomic_diagnostics["output_identities"],
            "known_relation_count": atomic_diagnostics["known_relation_count"],
            "unknown_relation_count": atomic_diagnostics["unknown_relation_count"],
            "rejected_relation_count": atomic_diagnostics["rejected_relation_count"],
            "accepted_fact_count": len(facts),
            "duplicate_fact_count": duplicate_fact_count,
            "rejection_reason_counts": rejection_counts,
            "rejected_relations": rejected,
            "generation_diagnostics_sha256": sha256_file(diagnostics_path),
            "failed_at_utc": _utc_now(),
        }
        _write_json(args.output_dir / "generator_failure_receipt.json", failure_receipt)
        raise ValueError(
            "Atomic corpus accepted fewer facts than --minimum-accepted-facts: "
            f"{len(facts)} < {minimum_accepted_facts}"
        )

    atomic_payload = {
        "schema_version": "rwku_generated_atomic_fact_corpus_v1",
        "entity_id": args.entity_id,
        "subject": args.target_entity,
        "relation_template_registry_path": str(Path(registry_path)),
        "relation_template_registry_sha256": registry_sha,
        "facts": canonical_atomic_facts,
    }
    atomic_path = args.output_dir / "generated_atomic_facts.json"
    _write_json(atomic_path, atomic_payload)
    atomic_sha = sha256_file(atomic_path)

    views = [view for fact in facts for view in fact["optimization_views"]]
    metadata = {
        "entity_id": args.entity_id,
        "subject": args.target_entity,
        "seed": args.seed,
        "generation_configuration_id": ATOMIC_V3_CONFIGURATION_ID,
        "relation_template_registry_sha256": registry_sha,
    }
    catalog_artifact = make_artifact(
        "fact_catalog",
        {"facts": facts},
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    training_artifact = make_artifact(
        "training_bundle",
        {"views": views},
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    write_artifact(
        args.output_dir / "generated_entity_fact_catalog.json", catalog_artifact
    )
    write_artifact(
        args.output_dir / "generated_training_bundle.json", training_artifact
    )
    receipt = {
        **common_receipt,
        "status": "complete",
        "tokenizer_identity": tokenizer_identity,
        "chat_template_used": tokenizer_identity.get("chat_template_used", False),
        "output_identities": atomic_diagnostics["output_identities"],
        "known_relation_count": atomic_diagnostics["known_relation_count"],
        "unknown_relation_count": atomic_diagnostics["unknown_relation_count"],
        "rejected_relation_count": atomic_diagnostics["rejected_relation_count"],
        "accepted_fact_count": len(facts),
        "duplicate_fact_count": duplicate_fact_count,
        "rejection_reason_counts": rejection_counts,
        "rejected_relations": rejected,
        "raw_generated_corpus_sha256": raw_sha,
        "generated_atomic_facts_sha256": atomic_sha,
        "generated_entity_fact_catalog_sha256": catalog_artifact["sha256"],
        "final_entity_fact_bundle_sha256": training_artifact["sha256"],
        "generation_diagnostics_sha256": sha256_file(diagnostics_path),
        "completed_at_utc": _utc_now(),
    }
    receipt_artifact = make_artifact(
        "generator_receipt",
        receipt,
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    write_artifact(args.output_dir / "generator_receipt.json", receipt_artifact)
    return receipt


def build_generated_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    config, config_sha = _load_generation_config(args.generation_config)
    if config.get("configuration_id") == ATOMIC_V3_CONFIGURATION_ID:
        return _build_atomic_generated_corpus(args, config, config_sha)
    prompts = [
        structured_generation_prompt(template, target_entity=args.target_entity)
        for template in config["prompt_templates"]
    ]
    independent_resources = []
    for resource in args.independent_resource:
        path = Path(resource)
        validate_independent_resource_path(path)
        independent_resources.append({"path": str(path), "sha256": sha256_file(path)})
    common_receipt = {
        "schema_version": GENERATOR_SCHEMA_VERSION,
        "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": TARGET_ONLY_PROTOCOL_STATUS,
        "target_entity": args.target_entity,
        "entity_id": args.entity_id,
        "generator_model_identifier": args.generator_model,
        "generator_model_revision": args.generator_revision,
        "local_snapshot": args.generator_model,
        "local_snapshot_identity": local_snapshot_identity(args.generator_model),
        "generation_configuration_path": str(args.generation_config),
        "generation_configuration_sha256": config_sha,
        "generation_prompt_templates": config["prompt_templates"],
        "prompt_template_sha256": sha256_json(config["prompt_templates"]),
        "rendered_prompt_sha256": [sha256_json(prompt) for prompt in prompts],
        "decoding_parameters": config["decoding"],
        "random_seeds": [int(args.seed)],
        "independent_resources": independent_resources,
        "official_rwku_records_accessed": False,
        "fact_extractor_implementation": EXTRACTOR_REVISION,
        "parser_implementation_revision": PARSER_IMPLEMENTATION_REVISION,
        "fact_extractor_revision_sha256": sha256_file(Path(__file__)),
        "extraction_configuration": config["fact_extraction"],
    }
    if args.dry_run:
        return {
            **common_receipt,
            "status": "dry_run_validated",
            "torch_imported": "torch" in __import__("sys").modules,
            "chat_template_used": None,
            "accepted_facts": [],
            "rejected_facts": [],
        }

    if not common_receipt["local_snapshot_identity"]["exists"]:
        raise FileNotFoundError(
            f"Generator model must be a pinned local snapshot: {args.generator_model}"
        )

    raw, tokenizer_identity = _model_generate(
        model_path=args.generator_model,
        prompts=prompts,
        decoding=config["decoding"],
        seed=args.seed,
    )
    raw_path = args.output_dir / "generated_raw_corpus.json"
    _write_json(raw_path, raw)
    raw_sha = sha256_file(raw_path)
    facts, rejected, parser_diagnostics = extract_generated_facts_detailed(
        raw,
        entity_id=args.entity_id,
        target_entity=args.target_entity,
    )
    rejection_counts = _rejection_reason_counts(rejected)
    generation_diagnostics = {
        "parser_implementation_revision": PARSER_IMPLEMENTATION_REVISION,
        "raw_generated_corpus_path": str(raw_path),
        "raw_generated_corpus_sha256": raw_sha,
        "chat_template_used": tokenizer_identity.get("chat_template_used", False),
        "accepted_fact_count": len(facts),
        "rejected_fact_count": len(rejected),
        "rejection_reason_counts": rejection_counts,
        **parser_diagnostics,
    }
    _write_json(args.output_dir / "generation_diagnostics.json", generation_diagnostics)
    if not facts:
        _write_json(args.output_dir / "rejected_generated_facts.json", rejected)
        failure_receipt = {
            **common_receipt,
            "status": "failed_no_accepted_facts",
            "official_rwku_records_accessed": False,
            "tokenizer_identity": tokenizer_identity,
            "chat_template_used": tokenizer_identity.get("chat_template_used", False),
            "raw_generated_corpus_sha256": raw_sha,
            "output_parse_diagnostics": parser_diagnostics["outputs"],
            "parse_mode_counts": parser_diagnostics["parse_mode_counts"],
            "accepted_fact_count": 0,
            "rejected_fact_count": len(rejected),
            "rejection_reason_counts": rejection_counts,
            "failed_at_utc": _utc_now(),
        }
        _write_json(args.output_dir / "generator_failure_receipt.json", failure_receipt)
        raise ValueError("Generated corpus produced no accepted entity facts")

    views = [view for fact in facts for view in fact["optimization_views"]]
    metadata = {
        "entity_id": args.entity_id,
        "subject": args.target_entity,
        "seed": args.seed,
    }
    catalog_artifact = make_artifact(
        "fact_catalog",
        {"facts": facts},
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    training_artifact = make_artifact(
        "training_bundle",
        {"views": views},
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    write_artifact(
        args.output_dir / "generated_entity_fact_catalog.json", catalog_artifact
    )
    write_artifact(
        args.output_dir / "generated_training_bundle.json", training_artifact
    )
    receipt = {
        **common_receipt,
        "status": "complete",
        "tokenizer_identity": tokenizer_identity,
        "chat_template_used": tokenizer_identity.get("chat_template_used", False),
        "raw_generated_corpus_sha256": raw_sha,
        "parse_mode_counts": parser_diagnostics["parse_mode_counts"],
        "output_parse_diagnostics": parser_diagnostics["outputs"],
        "accepted_fact_count": len(facts),
        "rejected_fact_count": len(rejected),
        "rejection_reason_counts": rejection_counts,
        "accepted_facts": [fact["fact_id"] for fact in facts],
        "rejected_facts": rejected,
        "rejection_reasons": rejection_counts,
        "duplicate_handling": (
            "deduplicate parsed objects by canonical JSON SHA-256, then "
            "deduplicate by relation-aware fact ID and normalized view content"
        ),
        "alias_handling": (
            "no aliases accepted unless present in independently generated output"
        ),
        "generated_entity_fact_catalog_sha256": catalog_artifact["sha256"],
        "final_entity_fact_bundle_sha256": training_artifact["sha256"],
        "completed_at_utc": _utc_now(),
    }
    receipt_artifact = make_artifact(
        "generator_receipt",
        receipt,
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=TARGET_ONLY_PROTOCOL_STATUS,
        metadata=metadata,
    )
    write_artifact(args.output_dir / "generator_receipt.json", receipt_artifact)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-entity", required=True)
    parser.add_argument("--entity-id", required=True)
    parser.add_argument("--generator-model", required=True)
    parser.add_argument("--generator-revision", required=True)
    parser.add_argument("--generation-config", type=Path, required=True)
    parser.add_argument("--relation-template-registry", type=Path)
    parser.add_argument("--minimum-accepted-facts", type=int, default=1)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--independent-resource", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = build_generated_corpus(args)
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.output_dir / "generator_dry_run.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
