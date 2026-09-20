#!/usr/bin/env python3
"""Build a held-out router probe set with a generator unrelated to training.

Every negative the router is currently calibrated and audited against comes
from one function: `_negative_prompts_for_fact`, which transplants a subject
into another training-visible prompt. Evaluating that router on more samples
from the same function measures nothing -- an off-manifold negative is scored
as easy by construction, tau is set too permissively, and the audit inherits
the same bias. A held-out set is only held out if a *different process* made
it. That is the whole reason this script exists; Gemini is just a convenient
different process.

Six classes are generated, ordered by how much they hurt:

  paraphrase            same subject, same relation, different phrasing.
                        Positive. Tests generalization of the scorer.
  indirect_reference    same fact, subject never named (description, pronoun,
                        definite noun phrase). Positive that Stage A cannot
                        reach by construction -- this measures the lexical
                        ceiling directly.
  misspelling           realistic human misspellings of the subject, not
                        synthetic keyboard noise. Positive.
  same_subject_other    subject present, a genuinely different real relation.
                        Negative. The class the router must reject and the one
                        its synthetic transplants approximate badly.
  incidental_mention    subject appears, but the request is about something
                        else entirely. Negative. Directly probes the disabled
                        absolute floor.
  near_miss             a paraphrase of a *different* protected association
                        that shares a subject or relation. Negative. Trains
                        and tests the ambiguity boundary the 0.02 margin
                        currently guesses at.

Nothing here is used for prototype construction or threshold fitting. It is
evaluation-only, and the manifest records the model id, prompt template hash,
and temperature so the provenance claim is checkable.

Set GEMINI_API_KEY. `--dry-run` emits the request payloads without calling the
API so the template can be reviewed and the cost estimated first.

Usage
-----
python -u scripts/build_gemini_router_probe_set.py \
  --artifact outputs/<run>/fact_association_embeddings.pt \
  --output outputs/<run>/probes/gemini_probe_set.json \
  --per-class 6 --model gemini-2.5-pro
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request

import torch


ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

CLASS_INSTRUCTIONS = {
    "paraphrase": (
        "Write {n} distinct natural questions or sentence completions that ask "
        "for exactly the same fact. The subject must be named explicitly. Vary "
        "syntax, register and length. Do not include the answer."
    ),
    "indirect_reference": (
        "Write {n} distinct natural requests for exactly the same fact in which "
        "the subject is NEVER named. Refer to the subject only by description, "
        "role, pronoun, or a definite noun phrase that a reader would resolve "
        "from context. Do not include the answer or the subject's name in any "
        "form, including surnames or initials."
    ),
    "misspelling": (
        "Write {n} distinct natural requests for exactly the same fact in which "
        "the subject's name is misspelled the way a real person typing quickly "
        "or transliterating from another language would misspell it. Vary the "
        "error type. Do not include the answer."
    ),
    "same_subject_other": (
        "Write {n} distinct natural questions about the SAME subject that ask "
        "about a genuinely different, real, well-known attribute -- not the "
        "relation given. Each question must be one a knowledgeable person "
        "would consider answerable and ordinary. Name the subject explicitly. "
        "Do not include answers."
    ),
    "incidental_mention": (
        "Write {n} distinct natural requests that mention the subject's name in "
        "passing but are fundamentally about something else -- a translation "
        "task, a writing task, a general knowledge question, a comparison in "
        "which the subject is an aside. The request must not ask for any "
        "attribute of the subject."
    ),
    "near_miss": (
        "Write {n} distinct natural questions about the SECOND subject and "
        "relation given below, phrased as closely as possible to how one would "
        "ask about the first subject and relation. The goal is maximum surface "
        "similarity between two genuinely different requests."
    ),
}

CLASS_POLARITY = {
    "paraphrase": "positive",
    "indirect_reference": "positive",
    "misspelling": "positive",
    "same_subject_other": "negative",
    "incidental_mention": "negative",
    "near_miss": "negative",
}

TEMPLATE = """You are constructing an evaluation set for a factual-retrieval \
router. Precision of the task matters more than fluency.

Subject: {subject}
Relation: {relation}
Reference request phrasing: {reference}
{extra}
Task: {instruction}

Return a JSON array of exactly {n} strings and nothing else. No markdown fence, \
no commentary, no numbering."""


def template_hash():
    payload = json.dumps(
        {"template": TEMPLATE, "classes": CLASS_INSTRUCTIONS}, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_prompt(fact, probe_class, per_class, reference, partner=None):
    extra = ""
    if probe_class == "near_miss" and partner is not None:
        extra = (
            f"Second subject: {partner['subject']}\n"
            f"Second relation: {partner.get('relation', '')}\n"
        )
    return TEMPLATE.format(
        subject=fact["subject"],
        relation=fact.get("relation", ""),
        reference=reference,
        extra=extra,
        instruction=CLASS_INSTRUCTIONS[probe_class].format(n=per_class),
        n=per_class,
    )


def call_gemini(prompt, model, api_key, temperature, retries=4, timeout=90):
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": float(temperature),
            "responseMimeType": "application/json",
        },
    }).encode("utf-8")
    url = ENDPOINT.format(model=model)
    last = None
    for attempt in range(int(retries)):
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=int(timeout)) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)
        except (urllib.error.URLError, KeyError, IndexError, ValueError) as error:
            last = error
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Gemini request failed after {retries} attempts: {last}")


def _reference_prompt(fact):
    for key in ("canonical_prompt", "prompt", "question"):
        value = fact.get(key)
        if value:
            return str(value)
    return f"{fact.get('relation', '')} of {fact['subject']}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--per-class", type=int, default=6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--classes", default=",".join(CLASS_INSTRUCTIONS), help="subset to generate"
    )
    parser.add_argument("--max-facts", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args(argv)

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    facts = list(artifact["facts"])
    if args.max_facts:
        facts = facts[: int(args.max_facts)]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in classes if c not in CLASS_INSTRUCTIONS]
    if unknown:
        raise SystemExit(f"Unknown probe classes: {unknown}")

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("GEMINI_API_KEY is not set (use --dry-run to inspect prompts)")

    records = []
    requests = []
    for index, fact in enumerate(facts):
        reference = _reference_prompt(fact)
        partner = facts[(index + 1) % len(facts)] if len(facts) > 1 else None
        for probe_class in classes:
            prompt = build_prompt(
                fact, probe_class, args.per_class, reference, partner=partner
            )
            requests.append({
                "fact_id": fact["id"],
                "fact_index": index,
                "probe_class": probe_class,
                "polarity": CLASS_POLARITY[probe_class],
                "partner_fact_id": (
                    partner["id"] if probe_class == "near_miss" and partner else None
                ),
                "request_prompt": prompt,
            })

    if args.dry_run:
        path = Path(args.output).with_suffix(".dryrun.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(requests, indent=2) + "\n")
        print(json.dumps({
            "status": "dry_run",
            "request_count": len(requests),
            "expected_probe_count": len(requests) * args.per_class,
            "payload": str(path),
        }, indent=2))
        return 0

    failures = []
    for position, request in enumerate(requests):
        try:
            items = call_gemini(
                request["request_prompt"],
                args.model,
                api_key,
                args.temperature,
            )
        except RuntimeError as error:
            failures.append({
                "fact_id": request["fact_id"],
                "probe_class": request["probe_class"],
                "error": str(error),
            })
            continue
        if not isinstance(items, list):
            failures.append({
                "fact_id": request["fact_id"],
                "probe_class": request["probe_class"],
                "error": "response was not a JSON array",
            })
            continue
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            records.append({
                "fact_id": request["fact_id"],
                "fact_index": request["fact_index"],
                "probe_class": request["probe_class"],
                "polarity": request["polarity"],
                "partner_fact_id": request["partner_fact_id"],
                "prompt": text,
                # Cheap automatic sanity check. An `indirect_reference` probe
                # that still contains the subject string is invalid by
                # definition, and this flag is what lets the evaluation drop it
                # without a manual pass.
                "contains_subject_surface": (
                    str(facts[request["fact_index"]]["subject"]).casefold()
                    in text.casefold()
                ),
            })
        if args.sleep:
            time.sleep(float(args.sleep))
        if (position + 1) % 25 == 0:
            print(f"... {position + 1}/{len(requests)} requests", flush=True)

    invalid_indirect = [
        record for record in records
        if record["probe_class"] == "indirect_reference"
        and record["contains_subject_surface"]
    ]
    manifest = {
        "schema_version": "gemini_router_probe_set_v1",
        "generator": "google_generativelanguage",
        "generator_model": args.model,
        "generator_temperature": args.temperature,
        "prompt_template_sha256": template_hash(),
        "source_artifact": str(Path(args.artifact).resolve()),
        "fact_count": len(facts),
        "classes": classes,
        "per_class_requested": args.per_class,
        "probe_count": len(records),
        "failed_requests": failures,
        "invalid_indirect_reference_count": len(invalid_indirect),
        "generated_by_training_negative_builder": False,
        "used_for_prototype_or_threshold_fitting": False,
        "class_polarity": CLASS_POLARITY,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"manifest": manifest, "probes": records}, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "gemini_probe_set_complete",
        "output": str(path),
        "probe_count": len(records),
        "failed_requests": len(failures),
        "invalid_indirect_reference": len(invalid_indirect),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
