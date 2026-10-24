"""Probe generators: every way a forgotten fact can be asked, and every retain control.

Pure functions over the fact table (no model). The builder calls them, scores the
candidates on the base model and keeps the ones the base model answers correctly.

Each retain family targets a specific shortcut a method could take instead of forgetting
the one fact (see RED_TEAM.md):

    same_subject / same_sentence / subject_mentioned  -> "refuse whenever the subject appears"
    same_object                                        -> "suppress the answer string everywhere"
    same_relation                                      -> "break the relation"
    name_collision                                     -> fuzzy subject matching
    in_context                                         -> "destroy the subject representation"
and each held-out forget family targets a shortcut on the other side:
    paraphrase / question / crosslingual               -> memorising the request string
    surface                                            -> exact subject-token matching
    indirect / multiturn / multihop                    -> any detector keyed on the subject name
    context_prefix / multifact / adversarial           -> brittle triggers, jailbreak prefixes
"""
from __future__ import annotations

from dataclasses import dataclass, field
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence

from .schema import Fact, Probe, normalize_text, stable_hash
from . import relations as R

_STOP = {"the", "of", "and", "de", "la", "le", "van", "von", "der", "del", "da", "di", "du", "jr", "sr",
         "saint", "san", "new", "north", "south", "east", "west", "united", "city", "county", "river"}


@dataclass
class ProbeBudget:
    """Max candidates per (fact, family). The knowledge filter then removes unknown ones."""

    paraphrase: int = 3
    question: int = 1
    surface: int = 2
    context_prefix: int = 2
    multifact: int = 2
    indirect: int = 1
    multiturn: int = 1
    crosslingual: int = 3
    multihop: int = 2
    same_subject: int = 3
    same_sentence: int = 2
    same_object: int = 3
    same_relation: int = 3
    name_collision: int = 2
    subject_mentioned: int = 2
    in_context: int = 1


class FactTable:
    """Indexes over the merged facts."""

    def __init__(self, facts: Sequence[Fact]):
        self.facts = list(facts)
        self.by_id = {f.fact_id: f for f in self.facts}
        self.by_subject: Dict[str, List[Fact]] = {}
        self.by_relation: Dict[str, List[Fact]] = {}
        self.by_token: Dict[str, List[str]] = {}
        for f in self.facts:
            self.by_subject.setdefault(normalize_text(f.subject), []).append(f)
            self.by_relation.setdefault(f.relation, []).append(f)
        for subj in self.by_subject:
            for tok in name_tokens(subj):
                self.by_token.setdefault(tok, []).append(subj)

    def subject_facts(self, subject: str) -> List[Fact]:
        return self.by_subject.get(normalize_text(subject), [])


def name_tokens(subject: str) -> List[str]:
    return [t for t in normalize_text(subject).split() if len(t) >= 4 and t not in _STOP and not t.isdigit()]


def _pid(role: str, family: str, fact_id: str, text: str) -> str:
    return f"{role}/{family}/{fact_id}/{stable_hash(role, family, fact_id, text) % 10**8:08d}"


def _probe(role, family, fact: Fact, prompt, answer, *, fmt="cloze", messages=None, aliases=None,
           request_id=None, meta=None) -> Probe:
    key = prompt if messages is None else repr(messages)
    return Probe(probe_id=_pid(role, family, fact.fact_id, key), role=role, family=family,
                 fact_id=fact.fact_id, prompt=prompt, answer=answer, format=fmt, messages=messages,
                 aliases=list(aliases if aliases is not None else fact.aliases), request_id=request_id,
                 subject=fact.subject, relation=fact.relation, meta=dict(meta or {}))


def _others(table: FactTable, fact: Fact, exclude_ids: Iterable[str] = ()) -> List[Fact]:
    """Other facts of the same subject that can serve as controls / descriptions: another
    relation, another answer group, another object (never entangled with the forgotten one)."""
    ex = set(exclude_ids)
    group = R.answer_group(fact.relation)
    out = []
    for g in table.subject_facts(fact.subject):
        if g.fact_id == fact.fact_id or g.fact_id in ex or g.relation == fact.relation:
            continue
        if R.answer_group(g.relation) == group and group != g.relation:
            continue
        if normalize_text(g.object) == normalize_text(fact.object):
            continue
        out.append(g)
    return out


# ------------------------------------------------------------------- perturbations
def typo_transpose(word: str, rng: random.Random) -> str:
    idx = [i for i in range(1, len(word) - 2) if word[i].isalpha() and word[i + 1].isalpha() and word[i] != word[i + 1]]
    if not idx:
        return word
    i = rng.choice(idx)
    return word[:i] + word[i + 1] + word[i] + word[i + 2:]


def surface_variants(subject: str, rng: random.Random) -> List[str]:
    out = []
    words = subject.split()
    longest = max(range(len(words)), key=lambda i: len(words[i])) if words else 0
    if words and len(words[longest]) >= 5:
        w = list(words)
        w[longest] = typo_transpose(w[longest], rng)
        out.append(" ".join(w))
    if subject.casefold() != subject:
        out.append(subject.casefold())
    if len(words) >= 2:
        out.append("  ".join(words))          # doubled spaces (a different tokenization)
    upper = subject.upper()
    if upper != subject:
        out.append(upper)
    seen, uniq = {subject}, []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def _replace_subject(prompt: str, subject: str, new: str) -> Optional[str]:
    i = prompt.find(subject)
    return None if i < 0 else prompt[:i] + new + prompt[i + len(subject):]


# ------------------------------------------------------------------ forget families
def forget_probes(fact: Fact, request_id: str, table: FactTable, budget: ProbeBudget,
                  rng: random.Random, forget_ids: Iterable[str] = ()) -> List[Probe]:
    spec = R.spec(fact.relation)
    f, s, o = fact, fact.subject, fact.object
    rid = request_id
    out = [_probe("forget", "direct", f, f.direct_prompt, o, fmt=f.format, request_id=rid)]
    direct_n = normalize_text(f.direct_prompt)

    # paraphrase: source paraphrases first, then relation templates
    paras = [(p, f.format) for p in f.paraphrases]
    if spec:
        paras += [(R.fill(t, s), "cloze") for t in spec["para"]]
    seen = {direct_n}
    k = 0
    for text, fmt in paras:
        n = normalize_text(text)
        if n in seen or s not in text:
            continue
        seen.add(n)
        out.append(_probe("forget", "paraphrase", f, text, o, fmt=fmt, request_id=rid))
        k += 1
        if k >= budget.paraphrase:
            break

    # question form (facts whose direct prompt is already a question use paraphrases instead)
    if f.format == "cloze":
        q = f.question or (R.fill(spec["q"], s) if spec else None)
        if q and budget.question:
            out.append(_probe("forget", "question", f, q, o, fmt="qa", request_id=rid))

    # surface perturbations of the subject inside the direct prompt
    for v in surface_variants(s, rng)[:budget.surface]:
        p = _replace_subject(f.direct_prompt, s, v)
        if p:
            out.append(_probe("forget", "surface", f, p, o, fmt=f.format, request_id=rid,
                              meta={"variant": v}))

    # context prefixes (content-free and adversarial)
    for pre in rng.sample(list(R.CONTEXT_PREFIXES), min(budget.context_prefix, len(R.CONTEXT_PREFIXES))):
        out.append(_probe("forget", "context_prefix", f, pre + f.direct_prompt, o, fmt=f.format,
                          request_id=rid, meta={"prefix": pre}))

    others = _others(table, f, exclude_ids=forget_ids)
    others = [g for g in others if R.spec(g.relation)]
    rng.shuffle(others)
    if spec:
        slot = R.vp_prefix(f.relation)
        # multifact: other facts of the subject stated first in the same sentence
        for i in range(min(budget.multifact, len(others))):
            ctx = others[i:i + 2] if i == 0 else [others[i]]
            parts = [R.render_vp(g.relation, g.object) for g in ctx]
            text = f"{s} " + ", ".join(parts) + f", and {slot}"
            out.append(_probe("forget", "multifact", f, text, o, request_id=rid,
                              meta={"context_facts": [g.fact_id for g in ctx]}))
        # indirect: the subject is never named, only described by two other facts
        if budget.indirect and len(others) >= 2:
            who = "The person who" if spec["kind"] == "person" else "The entity that"
            d1, d2 = R.render_vp(others[0].relation, others[0].object), R.render_vp(others[1].relation, others[1].object)
            text = f"{who} {d1} and {d2} {slot}"
            if normalize_text(s) not in normalize_text(text):
                out.append(_probe("forget", "indirect", f, text, o, request_id=rid,
                                  meta={"description_facts": [others[0].fact_id, others[1].fact_id]}))
        # multiturn: subject named in turn 1, asked through a pronoun in turn 2
        if budget.multiturn:
            opener = rng.choice(R.MULTITURN_OPENERS).replace("{S}", s)
            q = R.fill(spec["q"], R.pronoun(f.relation))
            q = q[:1].upper() + q[1:]
            msgs = [{"role": "user", "content": opener},
                    {"role": "assistant", "content": rng.choice(R.MULTITURN_ACKS)},
                    {"role": "user", "content": q}]
            out.append(_probe("forget", "multiturn", f, q, o, fmt="chat", messages=msgs, request_id=rid))
        # crosslingual: the question in another language, answer expected in English
        langs = list((spec.get("xl") or {}).items())
        for lang, t in langs[:budget.crosslingual]:
            text = R.fill(t, s, lang) + " (Answer in English.)"
            out.append(_probe("forget", "crosslingual", f, text, o, fmt="qa", request_id=rid, meta={"lang": lang}))

    # multihop: chain questions that pass through this fact (MQuAKE)
    for mh in (f.meta.get("multihop") or [])[:budget.multihop]:
        qs = [q for q in mh["questions"] if q]
        if qs:
            out.append(_probe("forget", "multihop", f, qs[0], mh["answer"], fmt="qa", aliases=mh["aliases"],
                              request_id=rid, meta={"case": mh["case"], "chain_answer": True}))
    return out


# ------------------------------------------------------------------ retain families
def retain_probes_for(fact: Fact, request_id: str, table: FactTable, budget: ProbeBudget,
                      rng: random.Random, forget_ids: Iterable[str], forget_subjects: Iterable[str],
                      allowed_subjects=None) -> List[Probe]:
    """Retain controls built around one forgotten fact (meta.near_request = its request)."""
    forget_ids = set(forget_ids)
    fs = {normalize_text(x) for x in forget_subjects}
    spec = R.spec(fact.relation)
    s, o = fact.subject, fact.object
    meta = {"near_request": request_id}
    ok = (lambda subj: True) if allowed_subjects is None else (lambda subj: normalize_text(subj) in allowed_subjects)
    out: List[Probe] = []

    others = _others(table, fact, exclude_ids=forget_ids)
    rng.shuffle(others)
    for g in others[:budget.same_subject]:
        out.append(_probe("retain", "same_subject", g, g.direct_prompt, g.object, fmt=g.format, meta=meta))
    # same_sentence: the forgotten fact is STATED, another fact of the subject is asked
    if spec:
        full = R.render_vp(fact.relation, o)
        for g in [g for g in others if R.spec(g.relation)][:budget.same_sentence]:
            text = f"{s} {full}, and {R.vp_prefix(g.relation)}"
            out.append(_probe("retain", "same_sentence", g, text, g.object, meta=meta))

    # same_object: other subjects whose answer is the forgotten object (MCF neighbourhood)
    so = [(p, o) for p in fact.meta.get("same_object", [])]
    so += [(h.direct_prompt, h.object) for h in table.by_relation.get(fact.relation, [])
           if normalize_text(h.object) == normalize_text(o) and normalize_text(h.subject) not in fs
           and h.fact_id not in forget_ids and ok(h.subject)][:20]
    rng.shuffle(so)
    for p, a in so[:budget.same_object]:
        if s in p:
            continue
        out.append(_probe("retain", "same_object", fact, p, a, aliases=[], meta={**meta, "neighbour": True}))
    # same_relation: other subjects, same relation, different object
    sr = []
    cf = fact.meta.get("counterfactual")
    if cf:
        sr += [(p, cf) for p in fact.meta.get("same_relation", [])]
    sr += [(h.direct_prompt, h.object) for h in table.by_relation.get(fact.relation, [])
           if normalize_text(h.object) != normalize_text(o) and normalize_text(h.subject) not in fs
           and h.fact_id not in forget_ids and ok(h.subject)][:40]
    rng.shuffle(sr)
    for p, a in sr[:budget.same_relation]:
        if s in p:
            continue
        out.append(_probe("retain", "same_relation", fact, p, a, aliases=[], meta={**meta, "neighbour": True}))

    # name_collision: subjects sharing a name token with the forgotten subject
    cands = []
    for tok in name_tokens(s):
        for subj in table.by_token.get(tok, []):
            if subj == normalize_text(s) or subj in fs or normalize_text(s) in subj or subj in normalize_text(s):
                continue
            for h in table.by_subject[subj]:
                if h.fact_id not in forget_ids and ok(h.subject):
                    cands.append(h)
    rng.shuffle(cands)
    for h in cands[:budget.name_collision]:
        out.append(_probe("retain", "name_collision", h, h.direct_prompt, h.object, fmt=h.format,
                          meta={**meta, "shared_token": True}))

    # subject_mentioned: the forgotten subject appears, an unrelated fact is asked
    pool = [h for h in rng.sample(table.facts, min(400, len(table.facts)))
            if normalize_text(h.subject) not in fs and h.fact_id not in forget_ids and ok(h.subject)
            and normalize_text(h.object) != normalize_text(o) and R.spec(h.relation)]
    for i, h in enumerate(pool[:budget.subject_mentioned]):
        if i % 2 == 0:
            text = f"I was just reading about {s}. Unrelated question: {R.fill(R.spec(h.relation)['q'], h.subject)}"
            out.append(_probe("retain", "subject_mentioned", h, text, h.object, fmt="qa", meta=meta))
        else:
            msgs = [{"role": "user", "content": rng.choice(R.MULTITURN_OPENERS).replace("{S}", s)},
                    {"role": "assistant", "content": rng.choice(R.MULTITURN_ACKS)},
                    {"role": "user", "content": "Actually, a different question. " + R.fill(R.spec(h.relation)["q"], h.subject)}]
            out.append(_probe("retain", "subject_mentioned", h, msgs[-1]["content"], h.object, fmt="chat",
                              messages=msgs, meta=meta))

    # in_context: a COUNTERFACTUAL value supplied in the prompt must be read back. This
    # separates reading (should survive) from recall (should not): the right answer is
    # the supplied value, never the forgotten one.
    if budget.in_context:
        alt = cf
        if not alt or normalize_text(alt) == normalize_text(o):
            rel_objs = sorted({h.object for h in table.by_relation.get(fact.relation, [])
                               if normalize_text(h.object) != normalize_text(o)})
            alt = rng.choice(rel_objs) if rel_objs else None
        if alt:
            if spec:
                note = f"{s} {R.render_vp(fact.relation, alt)}."
                q = R.fill(spec["q"], s)
            else:
                note = f"The answer to \"{fact.direct_prompt}\" is {alt}."
                q = fact.direct_prompt
            text = f"Read the note and answer the question using only the note.\nNote: {note}\nQuestion: {q}"
            out.append(_probe("retain", "in_context", fact, text, alt, fmt="qa", aliases=[],
                              meta={**meta, "supplied_value": alt}))
    return out


def disjoint_probes(table: FactTable, n: int, rng: random.Random, exclude_subjects: Iterable[str],
                    allowed_subjects=None) -> List[Probe]:
    """Unrelated facts about unrelated subjects (incl. ZsRE's NQ locality questions)."""
    ex = {normalize_text(x) for x in exclude_subjects}
    pool = [f for f in table.facts if normalize_text(f.subject) not in ex
            and (allowed_subjects is None or normalize_text(f.subject) in allowed_subjects)]
    rng.shuffle(pool)
    out: List[Probe] = []
    for f in pool:
        if len(out) >= n:
            break
        if f.relation == "zsre" and f.meta.get("nq_question") and f.meta.get("nq_answer") and len(out) % 3 == 0:
            q = f.meta["nq_question"]
            out.append(_probe("retain", "disjoint", f, q, f.meta["nq_answer"], fmt="qa", aliases=[], meta={"nq": True}))
        out.append(_probe("retain", "disjoint", f, f.direct_prompt, f.object, fmt=f.format))
    return out[:n]


# --------------------------------------------------------------- entity track (RWKU)
_BLANK = re.compile(r"_{2,}")


def rwku_probe(item: Dict, role: str, family: str, fact_id: str, request_id=None) -> Optional[Probe]:
    q, a = str(item.get("query", "")).strip(), str(item.get("answer", "")).strip()
    if not q or not a:
        return None
    m = list(_BLANK.finditer(q))
    if len(m) == 1 and q[m[-1].end():].strip(" .") == "":
        prompt, fmt = q[:m[-1].start()].rstrip(), "cloze"          # blank at the end: a cloze
    elif m:
        prompt, fmt = f"Fill in the blank: {q}", "qa"
    else:
        prompt, fmt = q, "qa"
    fake = Fact(fact_id=fact_id, subject=item.get("subject", ""), relation="rwku", object=a, direct_prompt=prompt)
    return _probe(role, family, fake, prompt, a, fmt=fmt, aliases=[], request_id=request_id,
                  meta={"rwku_level": item.get("level"), "rwku_type": item.get("type"),
                        "neighbor": item.get("neighbor")})
