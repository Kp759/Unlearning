#!/usr/bin/env python3
"""Offline, relation-grounded replacement for the failing V2 corpus builder.

Consumes only the same sanitized training_visible_forget_direct.json as V2.
Renders authored relation-specific templates; it neither loads an LLM nor uses
cosine nearest-relation ranking as a semantic acceptance test. Templates are
not claimed to be sampled, human-certified, or held-out benchmark paraphrases.
The V2 protocol, cases/views/template layout, nine families, and suggested
family split are retained. Provenance and quality controls describe this
changed construction accurately.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import string
import sys
import tempfile
from typing import Any, Mapping, Sequence

PROTOCOL = "mcf_relation_view_corpus_v2"
VERSION = "relation_views_fix5_authored_v1"
FAMILIES = (
    "wh_question", "possessive_question", "relation_fronted_question",
    "imperative_identify", "alternative_question", "reordered_cloze",
    "nominalized_question", "conversational_question",
)
FAMILY_SPLIT = {
    "fit": ["canonical_cloze", "wh_question", "possessive_question", "imperative_identify", "alternative_question"],
    "calibration": ["relation_fronted_question", "reordered_cloze"],
    "validation": ["nominalized_question", "conversational_question"],
}
ROW_KEYS = {"case_id", "requested_rewrite", "data_role"}
RR_KEYS = {"prompt", "subject", "relation_id", "target_true", "target_new"}
FORBIDDEN_FIELDS = {"paraphrase_prompts", "neighborhood_prompts", "generation_prompts", "attribute_prompts"}


class CorpusError(ValueError):
    """Invalid input or unsupported semantic contract; never silently skip it."""


def norm(text: str) -> str:
    return " ".join(text.split())


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check_template(template: Any, label: str) -> str:
    if not isinstance(template, str) or not template.strip():
        raise CorpusError(f"{label}: expected a non-empty string")
    try:
        fields = [(field, fmt, conv) for _, field, fmt, conv in string.Formatter().parse(template) if field is not None]
    except ValueError as exc:
        raise CorpusError(f"{label}: invalid format syntax") from exc
    if fields != [("", "", None)] or template.count("{}") != 1:
        raise CorpusError(f"{label}: exactly one anonymous {{}} subject slot is required")
    other = template.replace("{}", "")
    if "{" in other or "}" in other or "\n" in template or "\r" in template:
        raise CorpusError(f"{label}: braces or multiple lines are not supported")
    if len(template) > 500:
        raise CorpusError(f"{label}: template is unexpectedly long")
    return template


@dataclass(frozen=True)
class Request:
    case_id: int
    subject: str
    relation_id: str
    canonical: str


def sanitize_rows(rows: Any) -> list[Request]:
    if not isinstance(rows, list) or not rows:
        raise CorpusError("Expected a non-empty JSON list of sanitized forget records")
    out: list[Request] = []
    seen: set[int] = set()
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise CorpusError(f"Row {idx} must be an object")
        if FORBIDDEN_FIELDS.intersection(row):
            raise CorpusError(f"Row {idx}: official held-out fields are forbidden")
        if set(row) != ROW_KEYS or row.get("data_role") != "forget":
            raise CorpusError(f"Row {idx}: expected only {sorted(ROW_KEYS)} and data_role=forget")
        rr = row.get("requested_rewrite")
        if not isinstance(rr, dict) or set(rr) != RR_KEYS:
            raise CorpusError(f"Row {idx}: unexpected requested_rewrite schema")
        try:
            cid = int(row["case_id"])
        except (ValueError, TypeError) as exc:
            raise CorpusError(f"Row {idx}: invalid case_id") from exc
        if cid in seen:
            raise CorpusError(f"Duplicate case_id={cid}")
        seen.add(cid)
        subject, rid = rr["subject"], rr["relation_id"]
        if not isinstance(subject, str) or not subject.strip() or any(x in subject for x in ("{", "}", "\n", "\r")):
            raise CorpusError(f"case_id={cid}: invalid literal subject")
        if not isinstance(rid, str) or not re.fullmatch(r"P[0-9]+", rid):
            raise CorpusError(f"case_id={cid}: invalid relation_id")
        canonical = check_template(rr["prompt"], f"case_id={cid} canonical")
        out.append(Request(cid, subject, rid, canonical))
    return out


def load_source(path: Path) -> tuple[list[Request], str]:
    if path.name != "training_visible_forget_direct.json":
        raise CorpusError("Use the sanitized training_visible_forget_direct.json, not full multi_counterfact.json")
    raw = path.read_bytes()
    return sanitize_rows(json.loads(raw)), digest(raw)


def check_families(families: Any, label: str) -> None:
    if not isinstance(families, dict) or set(families) != set(FAMILIES):
        raise CorpusError(f"{label}: all eight named V2 families are required")
    for name in FAMILIES:
        values = families[name]
        if not isinstance(values, list) or not values:
            raise CorpusError(f"{label}/{name}: expected a non-empty list")
        for i, value in enumerate(values):
            check_template(value, f"{label}/{name}/{i}")


def validate_catalog(catalog: Any) -> None:
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise CorpusError("Invalid relation catalog schema")
    relations = catalog.get("relations")
    if not isinstance(relations, dict) or not relations:
        raise CorpusError("Empty relation catalog")
    for rid, contract in relations.items():
        if not re.fullmatch(r"P[0-9]+", rid) or not isinstance(contract, dict):
            raise CorpusError(f"Invalid contract for {rid}")
        for key in ("label", "meaning", "definition_url"):
            if not isinstance(contract.get(key), str) or not contract[key].strip():
                raise CorpusError(f"{rid}: missing {key}")
        check_families(contract.get("families"), rid)
        variants = contract.get("variants", {})
        if not isinstance(variants, dict):
            raise CorpusError(f"{rid}: invalid variants")
        for name, variant in variants.items():
            if not isinstance(variant, dict) or not variant.get("selector_patterns"):
                raise CorpusError(f"{rid}/{name}: missing selector patterns")
            for pattern in variant["selector_patterns"]:
                try:
                    re.compile(pattern, re.I)
                except re.error as exc:
                    raise CorpusError(f"{rid}/{name}: invalid selector") from exc
            check_families(variant.get("families"), f"{rid}/{name}")


def select_families(request: Request, contract: Mapping[str, Any]) -> tuple[Mapping[str, list[str]], str]:
    frame = request.canonical.replace("{}", " ")
    variants = contract.get("variants", {})
    matches = [name for name, v in variants.items() if any(re.search(p, frame, re.I) for p in v["selector_patterns"])]
    if len(matches) > 1:
        raise CorpusError(f"case_id={request.case_id}: ambiguous scope variants {matches}")
    if matches:
        return variants[matches[0]]["families"], matches[0]
    return contract["families"], "default"


def source_scope_flags(request: Request) -> list[str]:
    frame = request.canonical.replace("{}", " ").casefold()
    flags = []
    for pattern, message in [
        (r"\b(current|currently|former|formerly|latest|youngest|oldest)\b", "Source includes a temporal/superlative qualifier; check that generated scope is appropriate."),
        (r"\b(first|primary|main|largest|smallest)\b", "Source selects a particular value among possible values; manually check qualifier preservation."),
        (r"\b(18|19|20)\d{2}\b", "Source includes a year; templates do not automatically reproduce time-qualified queries."),
    ]:
        if re.search(pattern, frame):
            if request.relation_id == 'P449' and re.search(r"\bfirst\b", frame) and not re.search(r"\b(primary|main|largest|smallest)\b", frame):
                continue
            flags.append(message)
    return flags


def validate_authored_candidate(request: Request, contract: Mapping[str, Any], family: str, template: str) -> str:
    families, _ = select_families(request, contract)
    if family not in families or template not in families[family]:
        raise CorpusError(f"case_id={request.case_id}: unapproved {request.relation_id}/{family} template")
    check_template(template, family)
    rendered = template.format(request.subject)
    if rendered.count(request.subject) != 1:
        raise CorpusError(f"case_id={request.case_id}: subject must occur literally once")
    if len(rendered) > 1000:
        raise CorpusError(f"case_id={request.case_id}: rendered prompt is too long")
    return rendered


def build_case(request: Request, contract: Mapping[str, Any]) -> dict[str, Any]:
    families, variant = select_families(request, contract)
    canonical_text = request.canonical.format(request.subject)
    seen = {norm(canonical_text).casefold()}
    views: list[dict[str, Any]] = [{
        "family": "canonical_cloze", "template": request.canonical,
        "source": "canonical_requested_rewrite", "equivalence_margin": None,
    }]
    for family in FAMILIES:
        chosen = None
        for number, template in enumerate(families[family]):
            text = validate_authored_candidate(request, contract, family, template)
            if norm(text).casefold() not in seen:
                chosen = {"family": family, "template": template,
                          "source": "authored_relation_contract_fix5", "equivalence_margin": None,
                          "contract_variant": variant, "template_index": number,
                          "semantic_check": "exact membership in declared relation-family template bank"}
                seen.add(norm(text).casefold())
                break
        if chosen is None:
            raise CorpusError(f"case_id={request.case_id}: every {family} template duplicates another view; add a distinct template to its contract")
        views.append(chosen)
    return {"case_id":request.case_id, "relation_id":request.relation_id, "subject":request.subject,
            "relation_label":contract["label"], "relation_definition":contract["meaning"],
            "contract_variant":variant, "views":views, "rejected_counts":{},
            "source_scope_review_flags":source_scope_flags(request)}


def build_payload(requests: Sequence[Request], catalog: Mapping[str, Any], *, source_hash: str, catalog_hash: str, seed: int) -> dict[str, Any]:
    validate_catalog(catalog)
    missing = sorted({r.relation_id for r in requests} - set(catalog["relations"]))
    if missing:
        details = [(r.case_id, r.relation_id) for r in requests if r.relation_id in missing]
        raise CorpusError(f"Unsupported relation IDs {missing}; affected cases={details}. Add explicit templates, not a generic fallback.")
    cases = [build_case(r, catalog["relations"][r.relation_id]) for r in requests]
    split = {f:s for s, values in FAMILY_SPLIT.items() for f in values}
    for c in cases:
        for v in c["views"]:
            v["recommended_split"] = split[v["family"]]
    return {
        "protocol":PROTOCOL, "seed":seed, "source_sha256":source_hash,
        "cases":cases, "views_per_case":9, "families":["canonical_cloze", *FAMILIES],
        "family_split_recommendation":FAMILY_SPLIT,
        "leakage_contract":{
            "full_mcf_path_accepted":False,
            "official_paraphrase_prompts_read":False, "official_neighborhood_prompts_read":False,
            "official_generation_prompts_read":False, "official_retain_records_read":False,
            "generator_received_target_true":False, "generator_received_target_new":False,
            "verifier_received_target_true":False, "verifier_received_target_new":False,
            "source_file_contains_answer_fields_but_values_not_used":True,
        },
        "quality_controls":{
            "builder":VERSION, "generation_mode":"authored_relation_templates",
            "family_conditioned_generation":True, "stochastic_llm_generation":False,
            "exact_subject_once":True, "vague_form_filter":False,
            "minimum_equivalence_margin":None, "max_jaccard_to_canonical":None,
            "semantic_verifier":"declared relation contract and exact approved template membership; no learned entailment claim",
            "cosine_top1_is_acceptance_gate":False, "global_rejected_counts":{},
            "unknown_relations_fail_closed":True, "catalog_sha256":catalog_hash,
            "catalog_version":catalog["catalog_version"],
            "manual_source_scope_review_cases":[c["case_id"] for c in cases if c["source_scope_review_flags"]],
            "no_gen_or_utility_improvement_claimed":True,
        },
    }


def preview_text(payload: Mapping[str, Any]) -> str:
    lines = ["# Relation views fix5: review copy", "",
             "Assistant-authored relation templates; not LLM-sampled or human-certified.",
             "Cloze families intentionally end before the answer. No answers are included.", ""]
    for c in payload["cases"]:
        lines += [f"## case_id={c['case_id']} | {c['relation_id']} | {c['subject']}",
                  f"Relation: {c['relation_label']}", f"Meaning: {c['relation_definition']}"]
        for flag in c["source_scope_review_flags"]:
            lines.append("REVIEW: " + flag)
        for v in c["views"]:
            lines.append(f"[{v['recommended_split']}/{v['family']}] {v['template'].format(c['subject'])}")
        lines.append("")
    return "\n".join(lines) + "\n"


def atomic_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="."+path.name+".", suffix=".tmp", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        os.link(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--forget-direct", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--catalog", default=str(Path(__file__).with_name("mcf_relation_contracts_fix5.json")))
    p.add_argument("--seed", type=int, default=24291, help="Metadata compatibility; authored generation is deterministic")
    p.add_argument("--preflight-only", action="store_true", help="Validate all cases and show a preview, but write nothing")
    p.add_argument("--preview-case-id", type=int, default=13256)
    p.add_argument("--model-path", help="Compatibility only: no model is loaded")
    p.add_argument("--dtype", choices=("bf16","fp16","fp32"), default="bf16", help="Compatibility only")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        source, output, catalog_path = (Path(args.forget_direct).expanduser().resolve(), Path(args.out).expanduser().resolve(), Path(args.catalog).expanduser().resolve())
        requests, source_hash = load_source(source)
        catalog_bytes = catalog_path.read_bytes(); catalog = json.loads(catalog_bytes)
        payload = build_payload(requests, catalog, source_hash=source_hash, catalog_hash=digest(catalog_bytes), seed=args.seed)
        summary = {"builder":VERSION,"generation_mode":"authored_relation_templates", "model_loaded":False,
                   "cases":len(requests),"relations":len({x.relation_id for x in requests}),
                   "views_per_case":9,"total_views":9*len(requests),
                   "family_counts":dict(Counter(v["family"] for c in payload["cases"] for v in c["views"])),
                   "source_scope_review_cases":payload["quality_controls"]["manual_source_scope_review_cases"]}
        if args.preflight_only:
            chosen = next((c for c in payload["cases"] if c["case_id"]==args.preview_case_id), payload["cases"][0])
            summary["preflight_only"] = True
            summary["preview"] = {"case_id":chosen["case_id"],"subject":chosen["subject"], "relation_id":chosen["relation_id"],
                                  "views":[{"family":v["family"],"text":v["template"].format(chosen["subject"])} for v in chosen["views"]]}
            print(json.dumps(summary,indent=2,ensure_ascii=False)); return 0
        preview = output.with_suffix(".preview.md")
        for p in (output, preview):
            if p.exists(): raise FileExistsError(f"Refusing to overwrite {p}; use a new output filename")
        serialized = json.dumps(payload,indent=2,ensure_ascii=False,allow_nan=False)+"\n"
        atomic_new(preview, preview_text(payload))
        atomic_new(output, serialized)
        summary.update({"output":str(output),"preview":str(preview)})
        print(json.dumps(summary,indent=2,ensure_ascii=False)); return 0
    except (CorpusError, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"[relation-views-fix5] ERROR: {exc}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
