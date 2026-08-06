#!/usr/bin/env python3
"""Generate an independent target-only RWKU entity corpus with provenance.

There is intentionally no data-root or official-evaluation argument.  This
program cannot discover or open RWKU Level 1/2/3, MIA, neighbor, utility, or
fluency files.  Model libraries are imported only after ``--dry-run`` exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
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
EXTRACTOR_REVISION = "strict_json_fact_extractor_v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _load_generation_config(path: Path) -> Tuple[Dict[str, Any], str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict) or config.get("schema_version") != "rwku_target_corpus_generation_v1":
        raise ValueError("Unsupported target-corpus generation configuration")
    if config.get("official_evaluation_access") != "forbidden":
        raise ValueError("Generation configuration must forbid official evaluation access")
    templates = config.get("prompt_templates")
    if not isinstance(templates, list) or not templates or not all(isinstance(item, str) for item in templates):
        raise ValueError("Generation configuration requires prompt_templates")
    return config, sha256_file(Path(path))


def structured_generation_prompt(template: str, *, target_entity: str) -> str:
    base = template.format(target_entity=target_entity)
    return (
        f"{base}\n\n"
        "Return newline-delimited JSON objects only. Each object must have: "
        '"subject", "relation_id", "answer", "direct_question", and "cloze". '
        f'The subject must be exactly {json.dumps(target_entity)}. Use a short, '
        "stable snake_case relation_id. Omit uncertain facts. Do not include "
        "markdown or explanatory prose."
    )


def _extract_json_objects(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    candidates: List[Any] = []
    try:
        parsed = json.loads(stripped)
        candidates.extend(parsed if isinstance(parsed, list) else [parsed])
    except json.JSONDecodeError:
        for line in stripped.splitlines():
            candidate = line.strip().removeprefix("```json").removesuffix("```").strip()
            if not candidate:
                continue
            try:
                candidates.append(json.loads(candidate))
            except json.JSONDecodeError:
                continue
    return [dict(item) for item in candidates if isinstance(item, dict)]


def extract_generated_facts(
    raw_outputs: Sequence[Mapping[str, Any]],
    *,
    entity_id: str,
    target_entity: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Strictly accept only complete, target-matching structured facts."""

    accepted_by_id: Dict[str, Dict[str, Any]] = {}
    rejected: List[Dict[str, Any]] = []
    for output_index, output in enumerate(raw_outputs):
        text = str(output.get("generated_text", ""))
        objects = _extract_json_objects(text)
        if not objects:
            rejected.append({"output_index": output_index, "reason": "no_parseable_json_fact"})
            continue
        for object_index, candidate in enumerate(objects):
            location = {"output_index": output_index, "object_index": object_index}
            subject = normalize_text(str(candidate.get("subject", "")))
            relation_id = normalize_text(str(candidate.get("relation_id", "")))
            answer = normalize_text(str(candidate.get("answer", "")))
            direct = normalize_text(str(candidate.get("direct_question", "")))
            cloze = normalize_text(str(candidate.get("cloze", "")))
            if normalize_identity(subject) != normalize_identity(target_entity):
                rejected.append({**location, "reason": "subject_mismatch", "candidate": candidate})
                continue
            if not re.fullmatch(r"[a-z0-9_]+", relation_id):
                rejected.append({**location, "reason": "invalid_relation_id", "candidate": candidate})
                continue
            if not answer or not direct or not cloze or "___" not in cloze:
                rejected.append({**location, "reason": "incomplete_fact_views", "candidate": candidate})
                continue
            if any(marker in normalize_identity(answer) for marker in ("unknown", "uncertain", "possibly", "maybe")):
                rejected.append({**location, "reason": "uncertain_answer", "candidate": candidate})
                continue
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
            raw_fact_hash = sha256_json(candidate)
            if raw_fact_hash not in fact["source_hashes"]:
                fact["source_hashes"].append(raw_fact_hash)
                fact["source_records"].append(
                    {
                        "source_file": "generated_raw_corpus.json",
                        "source_row_index": output_index,
                        "source_record_sha256": raw_fact_hash,
                        "level": "generated",
                        "query_type": "target_only_generated",
                        "normalized_query_hash": hashlib.sha256(normalize_identity(direct).encode()).hexdigest(),
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
            subject_aliases = [
                normalize_text(str(alias))
                for alias in candidate.get("subject_aliases", [])
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
                    "subject": target_entity,
                    "training_allowed": True,
                }
                view["view_content_sha256"] = view_content_sha256(view)
                view["view_id"] = hashlib.sha256(f"{fact_id}:{view['view_content_sha256']}".encode()).hexdigest()
                if view["view_id"] not in {item["view_id"] for item in fact["optimization_views"]}:
                    fact["optimization_views"].append(view)
    return list(sorted(accepted_by_id.values(), key=lambda item: item["fact_id"])), rejected


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
    for prompt_index, prompt in enumerate(prompts):
        encoded = tokenizer(prompt, return_tensors="pt")
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
                    "generated_text": tokenizer.decode(sequence[prompt_tokens:], skip_special_tokens=True),
                }
            )
    tokenizer_identity = {
        "name_or_path": tokenizer.name_or_path,
        "class": tokenizer.__class__.__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
    }
    return outputs, tokenizer_identity


def build_generated_corpus(args: argparse.Namespace) -> Dict[str, Any]:
    config, config_sha = _load_generation_config(args.generation_config)
    prompts = [
        structured_generation_prompt(template, target_entity=args.target_entity)
        for template in config["prompt_templates"]
    ]
    independent_resources = []
    for resource in args.independent_resource:
        path = Path(resource)
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
        "fact_extractor_revision_sha256": sha256_file(Path(__file__)),
        "extraction_configuration": config["fact_extraction"],
    }
    if args.dry_run:
        return {
            **common_receipt,
            "status": "dry_run_validated",
            "torch_imported": "torch" in __import__("sys").modules,
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
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    facts, rejected = extract_generated_facts(
        raw,
        entity_id=args.entity_id,
        target_entity=args.target_entity,
    )
    if not facts:
        raise ValueError("Generated corpus produced no accepted entity facts")
    views = [view for fact in facts for view in fact["optimization_views"]]
    metadata = {"entity_id": args.entity_id, "subject": args.target_entity, "seed": args.seed}
    catalog_artifact = make_artifact("fact_catalog", {"facts": facts}, protocol_label=TARGET_ONLY_PROTOCOL_LABEL, protocol_status=TARGET_ONLY_PROTOCOL_STATUS, metadata=metadata)
    training_artifact = make_artifact("training_bundle", {"views": views}, protocol_label=TARGET_ONLY_PROTOCOL_LABEL, protocol_status=TARGET_ONLY_PROTOCOL_STATUS, metadata=metadata)
    write_artifact(args.output_dir / "generated_entity_fact_catalog.json", catalog_artifact)
    write_artifact(args.output_dir / "generated_training_bundle.json", training_artifact)
    receipt = {
        **common_receipt,
        "status": "complete",
        "tokenizer_identity": tokenizer_identity,
        "raw_generated_corpus_sha256": sha256_file(raw_path),
        "accepted_facts": [fact["fact_id"] for fact in facts],
        "rejected_facts": rejected,
        "rejection_reasons": dict(
            sorted(
                (reason, sum(1 for row in rejected if row["reason"] == reason))
                for reason in {row["reason"] for row in rejected}
            )
        ),
        "duplicate_handling": "deduplicate by relation-aware fact ID and normalized view content",
        "alias_handling": "no aliases accepted unless present in independently generated output",
        "generated_entity_fact_catalog_sha256": catalog_artifact["sha256"],
        "final_entity_fact_bundle_sha256": training_artifact["sha256"],
        "completed_at_utc": _utc_now(),
    }
    receipt_artifact = make_artifact("generator_receipt", receipt, protocol_label=TARGET_ONLY_PROTOCOL_LABEL, protocol_status=TARGET_ONLY_PROTOCOL_STATUS, metadata=metadata)
    write_artifact(args.output_dir / "generator_receipt.json", receipt_artifact)
    return receipt


def main() -> None:
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
    args = parser.parse_args()
    result = build_generated_corpus(args)
    if args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "generator_dry_run.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
