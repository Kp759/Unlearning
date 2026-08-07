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
    templates = config.get("prompt_templates")
    if (
        not isinstance(templates, list)
        or not templates
        or not all(isinstance(item, str) and item.strip() for item in templates)
    ):
        raise ValueError("Generation configuration requires prompt_templates")
    if config.get("configuration_id") == STRICT_V2_CONFIGURATION_ID:
        _validate_strict_v2_config(config)
    return config, sha256_file(Path(path))


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


def build_generated_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    config, config_sha = _load_generation_config(args.generation_config)
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
