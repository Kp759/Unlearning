"""Explicit association bundles and answer-span supervision; no evaluation router."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


# MCF does not provide all overlap strata for every fact.
# Same-relation/different-subject is required for every forget fact.
# Same-answer and same-subject controls are evaluated when naturally available
# and in separate targeted stress subsets.
REQUIRED_OVERLAPS = ("same_relation_other_subject",)
OPTIONAL_OVERLAPS = (
    "same_subject_other_relation",
    "same_subject_same_answer_other_relation",
    "same_answer_other_association",
)


@dataclass
class Example:
    id: str
    split: str
    role: str
    fact_id: str | None
    input_ids: list[int]
    labels: list[int]
    prompt: str
    completion: str
    group: str


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def overlap_kind(forget, retained):
    s, r, o = (forget[k].casefold().strip() for k in ("subject", "relation", "object"))
    rs, rr, ro = (retained[k].casefold().strip() for k in ("subject", "relation", "object"))
    if rs == s and rr != r:
        return "same_subject_same_answer_other_relation" if ro == o else "same_subject_other_relation"
    if rr == r and rs != s:
        return "same_relation_other_subject"
    if ro == o and rs != s and rr != r:
        return "same_answer_other_association"
    return "general_retain"


def validate_bundle(bundle, purpose="training"):
    if purpose not in ("training", "evaluation", "preservation_test"):
        raise ValueError("Unknown bundle purpose")
    if set(bundle) != {"schema_version", "purpose", "facts", "examples"}:
        raise ValueError("Bundle requires exactly schema_version, purpose, facts, examples")
    if bundle["schema_version"] != 1 or bundle["purpose"] != purpose:
        raise ValueError(f"Expected schema 1 {purpose} bundle")
    facts = {}
    for raw in bundle["facts"]:
        if set(raw) - {"id", "subject", "relation", "object", "role", "aliases", "answer_aliases"}:
            raise ValueError("Unknown fact fields (benchmark probe fields are forbidden)")
        fact = dict(raw)
        for key in ("id", "subject", "relation", "object", "role"):
            _text(fact.get(key), key)
        if fact["role"] not in ("forget", "retain") or fact["id"] in facts:
            raise ValueError("Invalid fact role or duplicate fact ID")
        for key in ("aliases", "answer_aliases"):
            if not isinstance(fact.get(key, []), list):
                raise ValueError(f"{key} must be an approved list")
            for value in fact.get(key, []):
                _text(value, key)
        facts[fact["id"]] = fact
    forgotten = [fact for fact in facts.values() if fact["role"] == "forget"]
    retained = [fact for fact in facts.values() if fact["role"] == "retain"]
    preservation_only = purpose == "preservation_test"
    if preservation_only and (forgotten or not retained or not bundle["examples"]):
        raise ValueError("Preservation test requires nonempty retain-only associations/examples")
    if not preservation_only and (not forgotten or not retained):
        raise ValueError("Both forget and retain associations are required")
    for f in forgotten:
        for r in retained:
            if all(f[k].strip().casefold() == r[k].strip().casefold() for k in ("subject", "relation")):
                raise ValueError("Same subject-relation is labeled both forget and retain")
    ids, fingerprints, prompt_splits = set(), {}, {}
    seen = {"train": set(), "validation": set(), "test": set()}
    language_splits, mixed_splits = set(), set()
    allowed_splits = {"train", "validation"} if purpose == "training" else {"test"}
    for row in bundle["examples"]:
        if set(row) - {"id", "split", "prompt", "completion", "spans", "text", "role"}:
            raise ValueError("Unknown example fields; pass training-visible text only")
        rid = _text(row.get("id"), "example id")
        split = row.get("split")
        if rid in ids or split not in allowed_splits:
            raise ValueError("Duplicate example ID or invalid split")
        ids.add(rid)
        if row.get("role") == "language":
            if set(row) != {"id", "split", "role", "text"}:
                raise ValueError("Language rows require id/split/role/text only")
            content = _text(row["text"], "language text")
            language_splits.add(split)
        else:
            if set(row) != {"id", "split", "prompt", "completion", "spans"}:
                raise ValueError("Association rows require id/split/prompt/completion/spans")
            _text(row["prompt"], "prompt")
            _text(row["completion"], "completion")
            prompt_key = " ".join(row["prompt"].casefold().split())
            if prompt_key in prompt_splits and prompt_splits[prompt_key] != split:
                raise ValueError("The same prompt cannot cross fitting/validation splits")
            prompt_splits[prompt_key] = split
            content = row["prompt"] + row["completion"]
            previous_end, roles = 0, set()
            if not row["spans"]:
                raise ValueError("Answer spans cannot be empty")
            for span in row["spans"]:
                if set(span) != {"start", "end", "fact_id"} or span["fact_id"] not in facts:
                    raise ValueError("Each span requires start/end and a registered fact_id")
                start, end = span["start"], span["end"]
                if (type(start) is not int or type(end) is not int
                        or not 0 <= previous_end <= start < end <= len(row["completion"])):
                    raise ValueError("Spans must be sorted, disjoint character offsets in completion")
                previous_end = end
                fact = facts[span["fact_id"]]
                answer = row["completion"][start:end].strip().casefold()
                registered = [fact["object"]] + fact.get("answer_aliases", [])
                if answer not in [s.strip().casefold() for s in registered]:
                    raise ValueError("Labeled answer is not a registered object/answer alias")
                roles.add(fact["role"])
                seen[split].add(fact["id"])
            if roles == {"forget", "retain"}:
                mixed_splits.add(split)
        fingerprint = " ".join(content.casefold().split())
        if fingerprint in fingerprints:
            raise ValueError("Duplicate text within/across splits; keep validation disjoint")
        fingerprints[fingerprint] = split
    for split in allowed_splits:
        if preservation_only:
            if not seen[split] or set(facts) != seen[split]:
                raise ValueError("Every final retain association requires a scored example")
            continue
        if split not in language_splits or split not in mixed_splits:
            raise ValueError(f"{split} requires language anchors and mixed requests")
        for f in forgotten:
            if f["id"] not in seen[split]:
                raise ValueError(f"Missing {split} forget supervision for {f['id']}")
            covered = {overlap_kind(f, facts[rid]) for rid in seen[split]
                       if facts[rid]["role"] == "retain"}
            missing = set(REQUIRED_OVERLAPS) - covered
            if missing:
                raise ValueError(
                    f"{f['id']} lacks required {split} overlap controls: {sorted(missing)}"
                )
    return facts


def load_bundle(path, purpose="training"):
    raw = Path(path).read_bytes()
    bundle = json.loads(raw)
    facts = validate_bundle(bundle, purpose)
    return bundle, facts, hashlib.sha256(raw).hexdigest()


def _encode(tokenizer, text, max_length):
    if not tokenizer.is_fast:
        raise ValueError("A fast tokenizer is required for exact span offset supervision")
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
    if len(encoded["input_ids"]) > max_length:
        raise ValueError("Example exceeds max_length; refusing to silently truncate labeled spans")
    if len(encoded["input_ids"]) < 2:
        raise ValueError("Example has no next-token targets")
    return encoded["input_ids"], encoded["offset_mapping"]


def encode_bundle(bundle, tokenizer, max_length=512, abstention="I don't know."):
    facts = validate_bundle(bundle, bundle["purpose"])
    result = []
    for row in bundle["examples"]:
        if row.get("role") == "language":
            ids, offsets = _encode(tokenizer, row["text"], max_length)
            labels = [token if i > 0 and end > start else -100
                      for i, (token, (start, end)) in enumerate(zip(ids, offsets))]
            if not any(label != -100 for label in labels):
                raise ValueError("Language anchor contains no tokens to score")
            result.append(Example(row["id"], row["split"], "language", None,
                                  ids, labels, "", row["text"], row["id"]))
            continue
        views = [(row["completion"], [(s, facts[s["fact_id"]]["role"]) for s in row["spans"]])]
        if abstention and any(role == "forget" for _, role in views[0][1]):
            pieces, spans, cursor, length = [], [], 0, 0
            for span, role in views[0][1]:
                prefix = row["completion"][cursor:span["start"]]
                pieces.append(prefix)
                length += len(prefix)
                value = abstention if role == "forget" else row["completion"][span["start"]:span["end"]]
                pieces.append(value)
                spans.append(({"start": length, "end": length + len(value), "fact_id": span["fact_id"]},
                              "abstain" if role == "forget" else role))
                length += len(value)
                # A neutral response ending in punctuation can otherwise merge
                # with the companion delimiter (e.g. ".;") into one token.
                if (role == "forget" and span["end"] < len(row["completion"])
                        and not row["completion"][span["end"]].isspace()):
                    pieces.append(" ")
                    length += 1
                cursor = span["end"]
            pieces.append(row["completion"][cursor:])
            views.append(("".join(pieces), spans))
        for view, (completion, spans) in enumerate(views):
            ids, offsets = _encode(tokenizer, row["prompt"] + completion, max_length)
            owner = {}
            for index, (span, role) in enumerate(spans):
                start, end = len(row["prompt"]) + span["start"], len(row["prompt"]) + span["end"]
                positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
                if not positions or 0 in positions:
                    raise ValueError("Labeled span has no preceding context/tokens")
                if any(i in owner for i in positions):
                    raise ValueError("A tokenizer token crosses separately labeled answer spans")
                full_text = row["prompt"] + completion
                for position in positions:
                    a, b = offsets[position]
                    if (a < start and full_text[a:start].strip()) or (b > end and full_text[end:b].strip()):
                        raise ValueError(f"{row['id']} view {view}: a token crosses a labeled span into non-answer text; add a separator")
                owner.update({i: index for i in positions})
                labels = [token if i in positions else -100 for i, token in enumerate(ids)]
                # Companion retain spans in the abstention view are also trained.
                result.append(Example(f"{row['id']}:{view}:{index}", row["split"], role,
                                      span["fact_id"], ids, labels, row["prompt"], completion, row["id"]))
    return result


def endpoint_rows(facts, examples, tokenizer, abstention_enabled=True):
    special = set(tokenizer.all_special_ids)
    input_rows, output_rows = set(), set()
    for fact in facts.values():
        if fact["role"] != "forget":
            continue
        for name in [fact["subject"]] + fact.get("aliases", []):
            for text in (name, " " + name):
                input_rows.update(tokenizer(text, add_special_tokens=False)["input_ids"])
        for answer in [fact["object"]] + fact.get("answer_aliases", []):
            for text in (answer, " " + answer):
                output_rows.update(tokenizer(text, add_special_tokens=False)["input_ids"])
    # Include actual in-context sensitive/neutral tokenizations from fitting only.
    for example in examples:
        if example.split == "train" and (example.role == "forget"
                or (abstention_enabled and example.role == "abstain")):
            output_rows.update(label for label in example.labels if label != -100)
    return sorted(input_rows - special), sorted(output_rows - special)


def text_fingerprints(bundle):
    """Deduplicate prompts as well as completions, so alternative answers cannot bypass a split check."""
    return sorted({" ".join(text.casefold().split())
                   for row in bundle["examples"]
                   for text in ([row["text"]] if row.get("role") == "language"
                                else [row["prompt"], row["prompt"] + row["completion"]])})
