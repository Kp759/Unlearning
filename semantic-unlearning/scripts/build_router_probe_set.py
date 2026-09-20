#!/usr/bin/env python3
"""Build a held-out router probe set from a generator unrelated to training.

Every negative the router is currently calibrated and audited against comes
from one function: `_negative_prompts_for_fact`, which transplants a subject
into another training-visible prompt. Evaluating that router on more samples
from the same function measures nothing -- an off-manifold negative is easy by
construction, tau is set too permissively, and the audit inherits the same
bias. A held-out set is only held out if a *different process* made it.

That property, not any particular model, is what this script supplies. The
backend is pluggable (`--backend local|gemini`) and a pinned open-weights
checkpoint is the better choice: greedy decoding makes the probe set a
deterministic function of (checkpoint, template, class list), so a reviewer
can regenerate it. A hosted endpoint cannot promise that.

Only ONE of the six classes below genuinely needs a language model.
`indirect_reference` requires world knowledge plus fluent generation. The
others have stronger non-model sources, noted per class, and `--classes`
exists so the model is used for exactly as much as it is good for:

  paraphrase            same subject and relation, different phrasing.
                        POSITIVE. Better source: back-translation.
  indirect_reference    same fact, subject NEVER named. POSITIVE that Stage A
                        cannot reach by construction, so it measures the
                        lexical ceiling directly. *** The class that needs a
                        model. ***
  misspelling           realistic human misspellings of the subject.
                        POSITIVE. Better source: Wikipedia redirect logs.
  same_subject_other    subject present, genuinely different real relation.
                        NEGATIVE, and the critical class. Better source:
                        Wikidata, which cannot hallucinate a relation the
                        subject does not have.
  incidental_mention    subject named in passing, request is about something
                        else. NEGATIVE. Better source: templates over a
                        general corpus.
  near_miss             paraphrase of a DIFFERENT protected association
                        sharing a subject or relation. NEGATIVE, and it needs
                        no generator at all -- it is mechanical over the
                        protected set.

Nothing here is used for prototype construction or threshold fitting. The
frozen output JSON, not this script, is the artifact the paper cites: generate
once, hash it, commit it.

Usage
-----
# Preferred: local open-weights, reproducible, one class only
python -u scripts/build_router_probe_set.py \
  --artifact outputs/<run>/fact_association_embeddings.pt \
  --output outputs/<run>/probes/probe_set.json \
  --backend local --generator-model /models/Qwen2.5-14B-Instruct \
  --evaluated-model /models/Llama-3.2-3B-Instruct \
  --classes indirect_reference --per-class 3 --temperature 0

# Inspect the prompts and count the requests without generating
python -u scripts/build_router_probe_set.py ... --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from probe_set_backends import build_backend


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
        "from context. Do not include the answer, and do not include the "
        "subject's name in any form, including surnames or initials."
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
        "relation given. Each question must be one a knowledgeable person would "
        "consider answerable and ordinary. Name the subject explicitly. Do not "
        "include answers."
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

# Better-than-a-model sources, carried into the manifest so a reader of the
# frozen probe set can see which classes were generated rather than sourced.
CLASS_PREFERRED_SOURCE = {
    "paraphrase": "back-translation (en->xx->en)",
    "indirect_reference": "language model (no stronger source)",
    "misspelling": "Wikipedia redirect logs / typo corpora",
    "same_subject_other": "Wikidata relations for the same subject",
    "incidental_mention": "templates over a general corpus",
    "near_miss": "mechanical, from the protected set itself",
}

# Numbered lines, not JSON: small instruct models emit malformed JSON often
# enough to corrupt a run silently, and a numbered list degrades gracefully.
TEMPLATE = """You are constructing an evaluation set for a factual-retrieval \
router. Precision of the task matters more than fluency.

Subject: {subject}
Relation: {relation}
Reference request phrasing: {reference}
{extra}
Task: {instruction}

Output exactly {n} lines, numbered 1 to {n}, one request per line. No preamble, \
no commentary, no blank lines, no markdown."""


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
    parser.add_argument(
        "--backend", default="local", choices=("local", "gemini")
    )
    parser.add_argument(
        "--generator-model",
        default="",
        help="local checkpoint path, or a hosted model id for --backend gemini",
    )
    parser.add_argument(
        "--evaluated-model",
        default="",
        help="path of the model under test; the generator must not be it",
    )
    parser.add_argument("--allow-same-model", action="store_true")
    parser.add_argument("--generator-device", default="cuda")
    parser.add_argument("--generator-dtype", default="bfloat16")
    parser.add_argument("--generator-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 selects greedy decoding, which is what makes this reproducible",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=3)
    parser.add_argument(
        "--classes",
        default="indirect_reference",
        help=f"comma-separated subset of {','.join(CLASS_INSTRUCTIONS)}",
    )
    parser.add_argument("--max-facts", type=int, default=0)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.generator_model and not args.dry_run:
        raise SystemExit("--generator-model is required unless --dry-run")

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    facts = list(artifact["facts"])
    if args.max_facts:
        facts = facts[: int(args.max_facts)]
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    unknown = [c for c in classes if c not in CLASS_INSTRUCTIONS]
    if unknown:
        raise SystemExit(f"Unknown probe classes: {unknown}")

    requests = []
    for index, fact in enumerate(facts):
        reference = _reference_prompt(fact)
        partner = facts[(index + 1) % len(facts)] if len(facts) > 1 else None
        for probe_class in classes:
            requests.append({
                "fact_id": fact["id"],
                "fact_index": index,
                "probe_class": probe_class,
                "polarity": CLASS_POLARITY[probe_class],
                "partner_fact_id": (
                    partner["id"] if probe_class == "near_miss" and partner else None
                ),
                "request_prompt": build_prompt(
                    fact, probe_class, args.per_class, reference, partner=partner
                ),
            })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        path = output.with_suffix(".dryrun.json")
        path.write_text(json.dumps(requests, indent=2) + "\n")
        print(json.dumps({
            "status": "dry_run",
            "request_count": len(requests),
            "expected_probe_count": len(requests) * args.per_class,
            "classes": classes,
            "payload": str(path),
        }, indent=2))
        return 0

    backend = build_backend(
        args, evaluated_model_path=args.evaluated_model or None
    )
    completions = backend.generate(
        [request["request_prompt"] for request in requests],
        expected=args.per_class,
    )
    if len(completions) != len(requests):
        raise RuntimeError("Backend returned the wrong number of completions")

    records = []
    empty = []
    for request, items in zip(requests, completions):
        if not items:
            empty.append({
                "fact_id": request["fact_id"],
                "probe_class": request["probe_class"],
            })
            continue
        subject = str(facts[request["fact_index"]]["subject"]).casefold()
        for text in items:
            records.append({
                "fact_id": request["fact_id"],
                "fact_index": request["fact_index"],
                "probe_class": request["probe_class"],
                "polarity": request["polarity"],
                "partner_fact_id": request["partner_fact_id"],
                "prompt": text,
                # An indirect_reference probe containing the subject string is
                # invalid by definition. Flagging it here is what lets the
                # evaluation drop it without a manual pass.
                "contains_subject_surface": subject in text.casefold(),
            })

    invalid_indirect = [
        record for record in records
        if record["probe_class"] == "indirect_reference"
        and record["contains_subject_surface"]
    ]
    manifest = {
        "schema_version": "router_probe_set_v2",
        "prompt_template_sha256": template_hash(),
        "source_artifact": str(Path(args.artifact).resolve()),
        "evaluated_model": args.evaluated_model or None,
        "fact_count": len(facts),
        "classes": classes,
        "class_polarity": {c: CLASS_POLARITY[c] for c in classes},
        "class_preferred_source": {c: CLASS_PREFERRED_SOURCE[c] for c in classes},
        "per_class_requested": args.per_class,
        "probe_count": len(records),
        "empty_responses": empty,
        "invalid_indirect_reference_count": len(invalid_indirect),
        "generated_by_training_negative_builder": False,
        "used_for_prototype_or_threshold_fitting": False,
    }
    manifest.update(backend.describe())

    payload = json.dumps({"manifest": manifest, "probes": records}, indent=2) + "\n"
    output.write_text(payload)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (output.parent / f"{output.stem}.sha256").write_text(f"{digest}  {output.name}\n")

    print(json.dumps({
        "status": "probe_set_complete",
        "output": str(output),
        "sha256": digest,
        "backend": manifest["backend"],
        "reproducible": manifest["reproducible"],
        "probe_count": len(records),
        "empty_responses": len(empty),
        "invalid_indirect_reference": len(invalid_indirect),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
