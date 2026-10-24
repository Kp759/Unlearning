"""Source datasets -> one fact table.

Every loader returns a list of Fact. Source-specific material that becomes probes later
(MCF neighbourhood prompts, ZsRE NQ locality questions, MQuAKE multi-hop questions,
RWKU entity probes) travels in Fact.meta, so the builder never re-reads a source file.

Licenses of the upstream data are listed in DATA_LICENSE.md.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import Fact, normalize_text
from .. import relations as R

# ----------------------------------------------------------------- "the" before objects
# A cloze such as "{} is a citizen of" continues with " the" for "the United Kingdom";
# without it the base model's next token is " the", never the object, and the knowledge
# filter would drop the fact for a formatting reason (multifact_person_data.needs_the).
_THE_RELATIONS = {"P27", "P19", "P20", "P937", "P69", "P108", "P463", "P54", "P140", "P17", "P495",
                  "P131", "P276", "P740", "P159"}
_THE_FIRST_WORDS = {
    "united", "netherlands", "philippines", "soviet", "ottoman", "holy", "byzantine",
    "european", "czech", "dominican", "democratic", "bahamas", "gambia", "maldives",
    "vatican", "catholic", "british", "roman", "papal", "confederate", "grand",
}
_PREPOSITIONS = ("of", "in", "at", "by", "from", "to", "for", "on", "with")


def needs_the(relation: str, obj: str) -> bool:
    words = str(obj).split()
    if str(relation) not in _THE_RELATIONS or not words or words[0].casefold() == "the":
        return False
    return words[0].casefold() in _THE_FIRST_WORDS or (
        words[0][:1].isupper() and " of " in f" {obj} " and len(words) >= 3)


def with_article(prompt: str, relation: str, obj: str) -> str:
    text = str(prompt).rstrip()
    last = text.split()[-1].casefold() if text.split() else ""
    if needs_the(relation, obj) and last in _PREPOSITIONS:
        return text + " the"
    return text


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------ MCF
def load_mcf(path: Path, limit: Optional[int] = None) -> List[Fact]:
    """MultiCounterFact (ROME / MEMIT). The forgotten answer is target_true (the real fact).

    meta.same_object   neighbourhood prompts: other subjects whose true answer is the SAME
                       object (official MCF locality set)
    meta.same_relation attribute prompts: other subjects whose answer is target_new
    meta.counterfactual target_new, used as the supplied answer of the in-context family
    """
    data = json.loads(Path(path).read_text())
    if limit:
        data = data[:limit]
    facts = []
    for i, rec in enumerate(data):
        rr = rec["requested_rewrite"]
        subject, rel = rr["subject"].strip(), rr["relation_id"]
        obj = rr["target_true"]["str"].strip()
        if rr["prompt"].count("{}") != 1 or not subject or not obj:
            continue
        direct = with_article(rr["prompt"].format(subject), rel, obj)
        facts.append(Fact(
            fact_id=f"mcf:{rec.get('case_id', i)}", subject=subject, relation=rel, object=obj,
            direct_prompt=direct, paraphrases=[p.strip() for p in rec.get("paraphrase_prompts", [])],
            source="mcf", object_id=rr["target_true"].get("id"),
            meta={"same_object": list(rec.get("neighborhood_prompts", [])),
                  "same_relation": list(rec.get("attribute_prompts", [])),
                  "counterfactual": rr["target_new"]["str"].strip(),
                  "template": rr["prompt"]},
        ))
    return facts


# ----------------------------------------------------------------------------- ZsRE
def load_zsre(path: Path, limit: Optional[int] = None) -> List[Fact]:
    """zsRE (MEND eval split). QA format; `loc` NQ questions become disjoint retain probes."""
    data = json.loads(Path(path).read_text())
    if limit:
        data = data[:limit]
    facts = []
    for i, rec in enumerate(data):
        answers = [a.strip() for a in rec.get("answers", []) if a and a.strip()]
        subject = str(rec.get("subject", "")).strip()
        if not answers or not subject or subject not in rec["src"]:
            continue
        loc = str(rec.get("loc", ""))
        loc_q = loc.split("nq question:", 1)[-1].strip() if loc else ""
        facts.append(Fact(
            fact_id=f"zsre:{i}", subject=subject, relation="zsre", object=answers[0],
            aliases=answers[1:4], direct_prompt=rec["src"].strip(), format="qa",
            paraphrases=[rec["rephrase"].strip()] if rec.get("rephrase") else [],
            source="zsre",
            meta={"counterfactual": str(rec.get("alt", "")).strip(),
                  "nq_question": (loc_q[:1].upper() + loc_q[1:] + ("?" if loc_q and not loc_q.endswith("?") else "")) if loc_q else "",
                  "nq_answer": str(rec.get("loc_ans", "")).strip()},
        ))
    return facts


# --------------------------------------------------------------------------- MQuAKE
def load_mquake(paths: Sequence[Path]) -> List[Fact]:
    """MQuAKE-CF(-3k-v2): the ORIGINAL (true) single hops become facts; each case's
    multi-hop questions (answer = the original chain answer) are attached to every fact on
    the chain (meta.multihop), since the chain answer depends on each of its hops."""
    by_key: Dict[str, Fact] = {}
    for path in paths:
        name = Path(path).stem
        for rec in json.loads(Path(path).read_text()):
            orig = rec["orig"]
            chain_keys = []
            for hop, ((_, rel, _), (subj, _, obj), single) in enumerate(zip(
                    orig["triples"], orig["triples_labeled"], rec["single_hops"])):
                subj, obj = str(subj).strip(), str(obj).strip()
                cloze = str(single.get("cloze", "")).strip()
                if not subj or not obj or subj not in cloze:
                    continue
                f = Fact(fact_id="", subject=subj, relation=rel, object=obj,
                         aliases=[a for a in single.get("answer_alias", []) if a.isascii()][:4],
                         direct_prompt=with_article(cloze, rel, obj),
                         question=str(single.get("question", "")).strip() or None,
                         source=f"mquake:{name}", meta={"multihop": []})
                k = f.key()
                if k not in by_key:
                    f.fact_id = f"mquake:{hashlib.sha1(k.encode()).hexdigest()[:12]}"
                    by_key[k] = f
                chain_keys.append(k)
            if len(chain_keys) >= 2:
                answers = [rec["answer"]] + [a for a in rec.get("answer_alias", []) if a.isascii()][:4]
                for k in chain_keys:
                    mh = by_key[k].meta["multihop"]
                    if len(mh) < 6:
                        mh.append({"questions": list(rec["questions"]), "answer": answers[0],
                                   "aliases": answers[1:], "case": f"{name}:{rec['case_id']}"})
    return list(by_key.values())


# ----------------------------------------------------------------------------- RWKU
def load_rwku(root: Path, targets: Optional[int] = None) -> List[Dict]:
    """RWKU targets (entity-level). Returns one dict per target person with its probe files;
    the builder turns them into entity requests + probes (see build.build_entity_split)."""
    root = Path(root)
    base = root / "Target"
    out = []
    for d in sorted(base.iterdir(), key=lambda p: int(p.name.split("_")[0]) if p.name.split("_")[0].isdigit() else 1 << 30):
        if not d.is_dir():
            continue

        def rd(name):
            p = d / f"{name}.json"
            return json.loads(p.read_text()) if p.exists() else []
        files = {n: rd(n) for n in ("forget_level1", "forget_level2", "forget_level3", "neighbor_level1",
                                    "neighbor_level2", "retain_mmlu", "triviaqa")}
        # the repository's RWKU scripts download files on demand, so a target directory can be
        # partial (no intro.json): name the target from its probes, else from the directory
        intro = rd("intro") or {}
        name = intro.get("target") or next((r["subject"] for k in ("forget_level1", "forget_level2", "forget_level3")
                                            for r in files[k] if r.get("subject")), None) \
            or d.name.split("_", 1)[-1].replace("_", " ")
        out.append({"target": name, "intro": intro.get("intro", ""), "dir": d.name, **files})
        if targets and len(out) >= targets:
            break
    return out


# --------------------------------------------------------------------- fact table
def merge_facts(groups: Iterable[List[Fact]]) -> Tuple[List[Fact], Dict[str, int]]:
    """Deduplicate by (subject, relation, object); drop (subject, relation) pairs that two
    sources answer differently (label conflicts would make a retain probe contradict a
    forget probe)."""
    stats = defaultdict(int)
    by_key: Dict[str, Fact] = {}
    answers: Dict[Tuple[str, str], set] = defaultdict(set)
    for facts in groups:
        for f in facts:
            stats[f"in_{f.source.split(':')[0]}"] += 1
            k = f.key()
            if k in by_key:
                old = by_key[k]
                old.paraphrases = list(dict.fromkeys(old.paraphrases + f.paraphrases))
                old.aliases = list(dict.fromkeys(old.aliases + f.aliases))[:4]
                for mk, mv in f.meta.items():
                    if isinstance(mv, list):
                        old.meta.setdefault(mk, [])
                        old.meta[mk] = old.meta[mk] + [x for x in mv if x not in old.meta[mk]]
                    else:
                        old.meta.setdefault(mk, mv)
                if old.question is None:
                    old.question = f.question
                stats["merged_duplicates"] += 1
                continue
            by_key[k] = f
            if f.relation != "zsre":
                answers[(normalize_text(f.subject), f.relation)].add(normalize_text(f.object))
    conflicts = {sr for sr, objs in answers.items() if len(objs) > 1}
    kept = [f for f in by_key.values()
            if (normalize_text(f.subject), f.relation) not in conflicts]
    stats["subject_relation_conflicts_dropped"] = len(by_key) - len(kept)
    # a subject contained in its own object (or vice versa) makes the answer readable off
    # the prompt ("Toyota Corolla is made by Toyota")
    final = [f for f in kept if normalize_text(f.subject) not in normalize_text(f.object)
             and normalize_text(f.object) not in normalize_text(f.subject)]
    stats["subject_object_overlap_dropped"] = len(kept) - len(final)
    stats["facts"] = len(final)
    return final, dict(stats)
