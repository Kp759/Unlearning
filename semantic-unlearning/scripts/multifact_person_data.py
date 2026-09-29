"""Multi-fact person benchmark: several facts about one real person in ONE sentence.

Task: forget one fact of a person while the other facts about the same person,
stated in the same sentence, are retained.

    Ernest Hemingway speaks English, was born in Oak Park, and is a citizen of ___
                     ^ retained          ^ FORGET              ^ retained

Facts are Wikidata triples (subject, relation, object) of real people that the
frozen base model already knows, pooled from MultiCounterFact and MQuAKE (their
own direct cloze prompts). Each relation has one fixed verb phrase ending in
the object, so one sentence can chain k = 2..4 facts:

    "{S} {vp_1}, {vp_2}, and {vp_3}."

Probe types (all teacher-forced, object tokens, request boundary = prompt end):
    direct   the source dataset's cloze             (training-visible for forget facts)
    single   "{S} {vp}"                             (held-out phrasing, one fact)
    multi    the fact at position j >= 1 of a multi-fact sentence, i.e. at
             least one other fact of the same person precedes it in context
             (held-out). Every rotation of the sentence is used, so each fact
             appears at every position.

Roles: forget (the one forgotten fact per forget person), retain_same_person
(the other facts of forget persons), retain_other_person (facts of held-out
people). Training uses the forget facts' direct prompts only (retain-blind),
exactly like the MQuAKE direct protocol, so the shipped prep / router /
row-trainer code runs unchanged on the locked split written here.

This module is import-light (no torch/transformers at import time) so the
builder's pure parts can be unit-tested.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

DATASET = "MultiFactPerson-v1"
FACT_ID_PREFIX = "multifact_assoc_"

# Verb phrases per relation, object last: (primary, alternate). The alternate
# is used when the primary equals a fact's own direct prompt, so the "single"
# probe is always a held-out phrasing. {art} = "a"/"an" for the object.
VERB_PHRASES = {
    "P27": ("is a citizen of {O}", "holds citizenship of {O}"),
    "P19": ("was born in {O}", "is a native of {O}"),
    "P20": ("died in {O}", "passed away in {O}"),
    "P1412": ("speaks {O}", "communicates in {O}"),
    "P103": ("grew up speaking {O}", "was raised speaking {O}"),
    "P106": ("works as {art} {O}", "is by profession {art} {O}"),
    "P101": ("works in the field of {O}", "specializes in the field of {O}"),
    "P39": ("held the position of {O}", "served in the position of {O}"),
    "P140": ("follows the religion of {O}", "is an adherent of {O}"),
    "P108": ("is employed by {O}", "works for {O}"),
    "P641": ("plays the sport of {O}", "competes in the sport of {O}"),
    "P413": ("plays in the position of {O}", "is a player in the position of {O}"),
    "P937": ("worked in {O}", "was based in {O}"),
    "P1303": ("plays the instrument {O}", "performs on the instrument {O}"),
    "P463": ("is a member of {O}", "belongs to {O}"),
    "P136": ("works in the genre of {O}", "is known for the genre of {O}"),
    "P26": ("is married to {O}", "is the spouse of {O}"),
    "P69": ("was educated at {O}", "studied at {O}"),
    "P102": ("belongs to the political party {O}", "is a member of the political party {O}"),
    "P800": ("is best known for {O}", "is famous for the work {O}"),
    "P166": ("received the award {O}", "won the award {O}"),
    "P54": ("played for the team {O}", "was a player for the team {O}"),
    "P264": ("is signed to the record label {O}", "records for the label {O}"),
}
# A subject is a person if it has at least one of these relations.
PERSON_EVIDENCE = {
    "P27", "P19", "P20", "P106", "P108", "P69", "P26", "P102", "P1412", "P103",
    "P641", "P413", "P54", "P39", "P140", "P101", "P937", "P1303",
}
# Relations that can share an answer (linear_router.RELATION_ANSWER_GROUPS plus
# person-specific ones). With distinct_answer_groups, a person's sentence holds
# at most one relation per group, so forgetting one fact never means
# suppressing an answer another fact in the same sentence also needs.
ANSWER_GROUPS = {
    "language": ("P103", "P1412", "P37", "P364", "P407"),
    "country": ("P27", "P17", "P495"),
    "place": ("P19", "P20", "P937", "P740", "P159", "P131", "P276", "P36", "P190"),
    "work_role": ("P106", "P101", "P39"),
    "affiliation": ("P108", "P463", "P69", "P54", "P102", "P264"),
}
_GROUP_OF = {rel: group for group, rels in ANSWER_GROUPS.items() for rel in rels}


def normalized(value):
    return " ".join(str(value).casefold().split())


def answer_group(relation):
    return _GROUP_OF.get(str(relation), str(relation))


def fact_key(subject, relation, obj):
    """Same key as mquake_fact_association_embeddings.association_key_from_record."""
    return f"{normalized(subject)}\t{relation}\t{normalized(obj)}"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Candidate facts
# ---------------------------------------------------------------------------

def _article(obj):
    return "an" if str(obj)[:1].casefold() in "aeiou" else "a"


# Place / institution objects that take "the" ("a citizen of the United
# Kingdom", "educated at the University of Oxford"). Without it the base
# model's next token is " the", not the object.
_THE_RELATIONS = {"P27", "P19", "P20", "P937", "P69", "P108", "P463", "P54", "P140"}
_THE_FIRST_WORDS = {
    "united", "netherlands", "philippines", "soviet", "ottoman", "holy", "byzantine",
    "european", "czech", "dominican", "democratic", "bahamas", "gambia", "maldives",
    "vatican", "catholic", "british", "roman", "papal", "confederate", "grand",
}
_PREPOSITIONS = ("of", "in", "at", "by", "from", "to", "for", "on", "with")


def needs_the(relation, obj):
    words = str(obj).split()
    if str(relation) not in _THE_RELATIONS or not words or words[0].casefold() == "the":
        return False
    return words[0].casefold() in _THE_FIRST_WORDS or (
        words[0][:1].isupper() and " of " in f" {str(obj)} " and len(words) >= 3
    )


def direct_template(template, relation, obj):
    """The source cloze, with "the" appended when the object needs it."""
    text = str(template).rstrip()
    last = text.split()[-1].casefold() if text.split() else ""
    if needs_the(relation, obj) and last in _PREPOSITIONS:
        return text + " the"
    return text


def _render_vp(template, relation, obj):
    before = template.split("{O}")[0].replace("{art}", _article(obj)).rstrip()
    if needs_the(relation, obj):
        before += " the"
    return before


def choose_verb_phrase(subject, relation, obj, direct):
    """Primary phrase unless '{S} {phrase}' is the fact's own direct prompt."""
    primary, alternate = VERB_PHRASES[str(relation)]
    if normalized(f"{subject} {_render_vp(primary, relation, obj)}") == normalized(direct):
        return alternate
    return primary


def verb_phrase(fact):
    """(text before the object, full phrase) for a fact dict."""
    template = fact.get("verb_phrase") or VERB_PHRASES[str(fact["relation"])][0]
    before = _render_vp(template, fact["relation"], fact["object"])
    return before, f"{before} {fact['object']}"


def candidate_facts(mcf_records=(), mquake_sources=()):
    """{subject: {relation: fact}} for people, from MCF records and MQuAKE files.

    mquake_sources: iterable of (name, records). A (subject, relation) pair seen
    with two different objects is dropped (reported). Only relations with a
    verb phrase are kept.
    """
    table = defaultdict(dict)
    conflicts = set()
    stats = defaultdict(int)

    def add(subject, relation, obj, obj_id, template, source):
        subject, obj = str(subject).strip(), str(obj).strip()
        if relation not in VERB_PHRASES:
            stats["skipped_relation"] += 1
            return
        if not subject or not obj or template.count("{}") != 1:
            stats["skipped_malformed"] += 1
            return
        if normalized(subject) in normalized(obj) or normalized(obj) in normalized(subject):
            stats["skipped_subject_object_overlap"] += 1
            return
        current = table[subject].get(relation)
        if current is None:
            direct = direct_template(template, relation, obj)
            table[subject][relation] = {
                "subject": subject, "relation": relation, "object": obj,
                "object_id": obj_id, "direct_template": direct,
                "source_template": template, "source": source,
                "verb_phrase": choose_verb_phrase(subject, relation, obj, direct.format(subject)),
                "key": fact_key(subject, relation, obj),
            }
        elif normalized(current["object"]) != normalized(obj):
            conflicts.add((subject, relation))

    for index, record in enumerate(mcf_records):
        rr = record["requested_rewrite"]
        add(rr["subject"], rr["relation_id"], rr["target_true"]["str"],
            rr["target_true"].get("id"), rr["prompt"], f"mcf:{record.get('case_id', index)}")
    for name, records in mquake_sources:
        for record in records:
            orig = record["orig"]
            for hop, ((_, relation, obj_id), (subject, _, obj), single) in enumerate(zip(
                    orig["triples"], orig["triples_labeled"], record["single_hops"])):
                cloze = str(single.get("cloze", ""))
                if subject not in cloze:
                    stats["skipped_cloze_without_subject"] += 1
                    continue
                add(subject, relation, obj, obj_id, cloze.replace(subject, "{}", 1),
                    f"{name}:{record.get('case_id')}:{hop}")
    for subject, relation in conflicts:
        table[subject].pop(relation, None)
    people = {s: rels for s, rels in table.items() if set(rels) & PERSON_EVIDENCE}
    stats.update({"subjects": len(table), "people": len(people),
                  "object_conflicts_dropped": len(conflicts)})
    return people, dict(stats)


def select_person_facts(facts, *, max_facts=4, distinct_answer_groups=True, rng=None):
    """Facts one sentence can hold: distinct objects (no substring overlap),
    optionally one relation per answer group, at most max_facts."""
    ordered = sorted(facts, key=lambda f: f["relation"])
    if rng is not None:
        rng.shuffle(ordered)
    chosen, groups = [], set()
    for fact in ordered:
        obj = normalized(fact["object"])
        if any(obj in normalized(c["object"]) or normalized(c["object"]) in obj for c in chosen):
            continue
        group = answer_group(fact["relation"])
        if distinct_answer_groups and group in groups:
            continue
        chosen.append(fact)
        groups.add(group)
        if len(chosen) == max_facts:
            break
    return sorted(chosen, key=lambda f: f["relation"])


def names_collide(a, b):
    a, b = normalized(a), normalized(b)
    return a != b and (a in b or b in a)


# ---------------------------------------------------------------------------
# Sentences and probes
# ---------------------------------------------------------------------------

def render_sentence(subject, facts):
    """Sentence text and, per fact, the character offset where its object starts."""
    parts, starts = [subject, " "], []
    k = len(facts)
    for index, fact in enumerate(facts):
        if index > 0:
            parts.append(" and " if k == 2 else (", and " if index == k - 1 else ", "))
        before, _ = verb_phrase(fact)
        parts.append(before + " ")
        starts.append(len("".join(parts)))
        parts.append(fact["object"])
    parts.append(".")
    return "".join(parts), starts


def person_probes(subject, facts, *, forget_key=None, person_role):
    """All probes of one person. person_role: forget_person | retain_person."""
    probes = []

    def role_of(fact):
        if person_role == "retain_person":
            return "retain_other_person"
        return "forget" if fact["key"] == forget_key else "retain_same_person"

    for fact in facts:
        before, _ = verb_phrase(fact)
        common = {"person": subject, "fact_key": fact["key"], "relation": fact["relation"],
                  "answer": fact["object"], "role": role_of(fact), "k": len(facts)}
        probes.append({**common, "type": "direct", "position": None,
                       "prefix": fact["direct_template"].format(subject),
                       "context_fact_keys": [], "forget_in_context": False})
        probes.append({**common, "type": "single", "position": 0,
                       "prefix": f"{subject} {before}",
                       "context_fact_keys": [], "forget_in_context": False})
    for rotation in range(len(facts)):
        order = facts[rotation:] + facts[:rotation]
        sentence, starts = render_sentence(subject, order)
        for position in range(1, len(order)):
            fact = order[position]
            context = [f["key"] for f in order[:position]]
            probes.append({
                "person": subject, "fact_key": fact["key"], "relation": fact["relation"],
                "answer": fact["object"], "role": role_of(fact), "k": len(facts),
                "type": "multi", "position": position, "rotation": rotation,
                "prefix": sentence[:starts[position]].rstrip(),
                "sentence": sentence,
                "context_fact_keys": context,
                "forget_in_context": forget_key is not None and forget_key in context,
            })
    return probes


def build_split(people, known, *, seed, forget_num=50, retain_persons=100, min_facts=2,
                max_facts=4, distinct_answer_groups=True):
    """Deterministic seed sampling. `known` = set of fact keys the base model knows.

    Forget people are drawn first from people whose sentence holds >= 3 facts,
    then 2; one forget fact per forget person. Retain people are disjoint and
    no person's name contains another's (the router's subject eligibility is a
    token-subsequence match).
    """
    rng = random.Random(int(seed))
    sentences = {}
    for subject in sorted(people):
        facts = [f for f in people[subject].values() if f["key"] in known]
        chosen = select_person_facts(facts, max_facts=max_facts,
                                     distinct_answer_groups=distinct_answer_groups,
                                     rng=random.Random(f"{seed}:{subject}"))
        if len(chosen) >= min_facts:
            sentences[subject] = chosen
    names = sorted(sentences)
    colliding = {a for a in names for b in names if names_collide(a, b)}
    eligible = [n for n in names if n not in colliding]
    rng.shuffle(eligible)
    tiered = [n for n in eligible if len(sentences[n]) >= 3] + \
             [n for n in eligible if len(sentences[n]) < 3]
    if len(tiered) < forget_num:
        raise ValueError(f"Only {len(tiered)} eligible people for {forget_num} forget people")
    forget_people = tiered[:forget_num]
    rest = [n for n in eligible if n not in set(forget_people)]
    retain_people = rest[:retain_persons]

    persons, probes, records = [], [], []
    for index, subject in enumerate(forget_people):
        facts = sentences[subject]
        forget = rng.choice(facts)
        persons.append({"subject": subject, "role": "forget_person", "facts": facts,
                        "forget_fact_key": forget["key"]})
        probes += person_probes(subject, facts, forget_key=forget["key"],
                                person_role="forget_person")
        records.append({
            "case_id": index + 1, "mquake_case_id": index + 1, "source_index": index,
            "rewrite_index": 0,
            "requested_rewrite": {
                "prompt": forget["direct_template"], "subject": subject,
                "relation_id": forget["relation"],
                "target_true": {"str": forget["object"], "id": forget.get("object_id")},
            },
            "source": forget["source"],
        })
    for subject in retain_people:
        facts = sentences[subject]
        persons.append({"subject": subject, "role": "retain_person", "facts": facts,
                        "forget_fact_key": None})
        probes += person_probes(subject, facts, person_role="retain_person")
    for index, probe in enumerate(probes):
        probe["id"] = f"probe_{index}"
    k_hist = defaultdict(int)
    for subject in forget_people:
        k_hist[len(sentences[subject])] += 1
    sampling = {
        "forget_num_instances": len(forget_people),
        "forget_atomic_fact_count": len(records),
        "forget_atomic_case_ids": [r["case_id"] for r in records],
        "retain_person_count": len(retain_people),
        "eligible_people": len(eligible),
        "people_excluded_for_name_collision": len(colliding),
        "facts_per_forget_sentence": dict(sorted(k_hist.items())),
        "min_facts": min_facts, "max_facts": max_facts,
        "distinct_answer_groups": distinct_answer_groups,
        "forget_selection": "people with >=3 facts first, then 2; forget fact uniform per person",
    }
    return records, persons, probes, sampling


# ---------------------------------------------------------------------------
# Loader for prep and the row trainer (MQuAKE direct machinery, own identity)
# ---------------------------------------------------------------------------

def load_multifact_forget(training_visible, split_manifest_path):
    """(split_manifest, sampling, records, facts, case_to_fact_id, dedup)."""
    from mquake_fact_association_embeddings import (
        build_association_facts,
        load_locked_visible_forget,
    )

    split_manifest = json.loads(Path(split_manifest_path).read_text())
    if split_manifest.get("dataset") != DATASET:
        raise ValueError(f"Not a {DATASET} split manifest: {split_manifest_path}")
    if int(split_manifest.get("seed", -1)) < 1:
        raise ValueError("Split manifest has no seed")
    sampling = split_manifest["sampling"]
    records = load_locked_visible_forget(Path(training_visible))
    if len(records) != int(sampling["forget_atomic_fact_count"]):
        raise ValueError("Forget record count does not match the split manifest")
    facts, case_to_fact_id, dedup = build_association_facts(records)
    rename = {f["id"]: FACT_ID_PREFIX + f["id"].split("_")[-1] for f in facts}
    for fact in facts:
        fact["id"] = rename[fact["id"]]
    case_to_fact_id = {case: rename[fid] for case, fid in case_to_fact_id.items()}
    dedup["case_to_fact_id"] = {str(k): v for k, v in case_to_fact_id.items()}
    for group in dedup.get("duplicate_groups", []):
        group["fact_id"] = rename.get(group["fact_id"], group["fact_id"])
    return split_manifest, sampling, records, facts, case_to_fact_id, dedup


# ---------------------------------------------------------------------------
# Teacher-forced scoring (shared by the builder's knowledge filter and the evaluator)
# ---------------------------------------------------------------------------

def token_cases(tok, probe, *, llama_like):
    """Official MQuAKE/ZeroUnlearn token expansion: boundary + decoded answer prefix."""
    from mquake_zero_unlearn_official_eval import original_answer_token_ids

    target_ids = original_answer_token_ids(tok, probe["answer"], llama_like=llama_like)
    cases = []
    for index, token_id in enumerate(target_ids):
        decoded = tok.decode(target_ids[:index])
        prompt = probe["prefix"] + ((" " + decoded) if (llama_like and index > 0) else decoded)
        cases.append({"probe_id": probe["id"], "token_index": index, "prompt": prompt,
                      "boundary": probe["prefix"], "target_text": tok.decode([token_id])})
    return cases


def score_probes(model, tok, probes, device, *, llama_like, batch_size=16, bank=None,
                 row_of_key=None):
    """Per-probe token accuracy; with a bank, routing is fixed at the probe prefix."""
    import torch
    from mquake_zero_unlearn_official_eval import _flat_ids, official_target_ids

    cases = [case for probe in probes for case in token_cases(tok, probe, llama_like=llama_like)]
    by_probe = {p["id"]: p for p in probes}
    results = defaultdict(lambda: {"correct": [], "routes": []})
    with torch.no_grad():
        for start in range(0, len(cases), int(batch_size)):
            batch = cases[start:start + int(batch_size)]
            encoded = tok([c["prompt"] for c in batch], padding=True, return_tensors="pt",
                          return_token_type_ids=False).to(device)
            if bank is not None:
                lengths = []
                for case in batch:
                    full, boundary = _flat_ids(tok, case["prompt"]), _flat_ids(tok, case["boundary"])
                    if full[:len(boundary)] != boundary:
                        raise ValueError(f"Boundary is not a token prefix: {case['boundary']!r}")
                    lengths.append(len(boundary))
                model.set_association_prefix_lengths(lengths)
            output = model(**encoded, use_cache=False)
            last = encoded["attention_mask"].sum(dim=1) - 1
            logits = output.logits[torch.arange(len(batch), device=device), last, :]
            predicted = logits.argmax(dim=-1).tolist()
            targets = official_target_ids(tok, [c["target_text"] for c in batch],
                                          llama_like=llama_like, device=device).tolist()
            routes = list(bank.last_active_fact_indices) if bank is not None else [None] * len(batch)
            for case, pred, target, route in zip(batch, predicted, targets, routes):
                entry = results[case["probe_id"]]
                entry["correct"].append(pred == target)
                entry["routes"].append(route)
    rows = []
    for probe_id, entry in results.items():
        probe = by_probe[probe_id]
        routes = entry["routes"]
        first = routes[0]
        own = None if row_of_key is None else row_of_key.get(probe["fact_key"])
        rows.append({
            **{k: probe.get(k) for k in ("id", "person", "fact_key", "relation", "role", "type",
                                         "position", "k", "forget_in_context")},
            "accuracy": sum(entry["correct"]) / len(entry["correct"]),
            "tokens": len(entry["correct"]),
            "route_active": None if first is None else bool(first),
            "routed_to_own_row": (None if first is None or own is None
                                  else list(first) == [own]),
        })
    return sorted(rows, key=lambda r: int(r["id"].split("_")[1]))


def _mean(values):
    values = list(values)
    return None if not values else sum(values) / len(values)


def summarize(rows):
    """100 * mean per-probe accuracy (case-macro), per role / probe type, plus routing."""
    def select(**match):
        return [r for r in rows if all(r.get(k) == v for k, v in match.items())]

    def acc(subset):
        value = _mean(r["accuracy"] for r in subset)
        return None if value is None else 100.0 * value

    def fire(subset):
        flags = [r["route_active"] for r in subset if r["route_active"] is not None]
        return _mean(flags)

    out = {}
    for role in ("forget", "retain_same_person", "retain_other_person"):
        block = {"probes": len(select(role=role))}
        for kind in ("direct", "single", "multi"):
            subset = select(role=role, type=kind)
            block[kind] = {"accuracy": acc(subset), "probes": len(subset),
                           "route_active_fraction": fire(subset)}
        multi = select(role=role, type="multi")
        block["multi_by_position"] = {
            str(p): acc([r for r in multi if r["position"] == p])
            for p in sorted({r["position"] for r in multi})
        }
        block["multi_by_k"] = {
            str(k): acc([r for r in multi if r["k"] == k]) for k in sorted({r["k"] for r in multi})
        }
        if role == "retain_same_person":
            block["multi_forget_fact_in_context"] = acc([r for r in multi if r["forget_in_context"]])
            block["multi_forget_fact_not_in_context"] = acc(
                [r for r in multi if not r["forget_in_context"]])
        if role == "forget":
            for kind in ("direct", "single", "multi"):
                own = [r["routed_to_own_row"] for r in select(role=role, type=kind)
                       if r["routed_to_own_row"] is not None]
                block[kind]["routed_to_own_row_fraction"] = _mean(own)
        out[role] = block
    return out


def headline(summary):
    """The numbers a table needs: forget (lower better) and retain (higher better)."""
    f, s, o = summary["forget"], summary["retain_same_person"], summary["retain_other_person"]
    return {
        "forget_Eff_direct": f["direct"]["accuracy"],
        "forget_Gen_single": f["single"]["accuracy"],
        "forget_Gen_multi": f["multi"]["accuracy"],
        "retain_same_person_direct": s["direct"]["accuracy"],
        "retain_same_person_single": s["single"]["accuracy"],
        "retain_same_person_multi": s["multi"]["accuracy"],
        "retain_same_person_multi_forget_in_context": s.get("multi_forget_fact_in_context"),
        "retain_other_person_direct": o["direct"]["accuracy"],
        "retain_other_person_multi": o["multi"]["accuracy"],
    }
