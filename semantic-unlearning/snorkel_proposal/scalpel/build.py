"""Materialise one benchmark split for one base model.

    facts (sources) -> entity partition -> forget sampling (base must know the fact)
      -> candidate probes (probes.py) -> knowledge filter (base must answer each probe)
      -> conflict removal -> requests + probes + allowed retain + utility sets

Partitions are by SUBJECT (salted hash), so a subject never appears in two of them:
    retain_train  [0, 10)   facts a method may use as its own retain set
    dev           [10, 55)  public splits
    test          [55, 100) hidden splits (sampled with a private seed / templates)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .schema import (CANARY, SCHEMA_VERSION, Fact, Probe, Request, RetainExample, Split,
                     normalize_text, stable_hash)
from . import probes as P
from . import relations as R

GENERIC_PROMPTS = (
    "The history of the Roman Empire", "Photosynthesis is the process by which",
    "In computer science, a hash table", "The best way to learn a new language is",
    "Mount Everest is located", "A healthy breakfast usually includes", "Electric cars work by",
    "The theory of evolution states", "To bake a loaf of bread, you", "In economics, inflation means",
    "The human heart has four", "A neural network is", "Volcanoes form when", "Water boils at",
    "Vaccines protect people by", "The rules of chess say that", "Honey bees communicate by",
    "In mathematics, a prime number", "The internet was first developed", "Earthquakes are measured using",
)


@dataclass
class BuildConfig:
    name: str = "scalpel-fact"
    seed: int = 1
    partition: str = "dev"
    salt: str = "scalpel-v1"
    n_forget: int = 50
    multi_share: float = 0.8          # share of forget facts drawn from subjects with other known facts
    n_disjoint: int = 150
    n_allowed_retain: int = 200
    screen_chunk: int = 256
    screen_max: int = 8000            # max candidate facts scored before giving up
    batch_size: int = 16
    use_chat_template: str = "auto"
    n_mmlu: int = 150
    n_trivia: int = 150
    budget: P.ProbeBudget = field(default_factory=P.ProbeBudget)
    private_templates: Optional[str] = None   # test partition: JSON {relation: {"para": [...], "q": ...}}


def partition_of(subject: str, salt: str) -> str:
    h = stable_hash(salt, normalize_text(subject)) % 100
    return "retain_train" if h < 10 else ("dev" if h < 55 else "test")


def _apply_private_templates(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    for rel, upd in data.items():
        if rel in R.RELATIONS:
            R.RELATIONS[rel].update({k: v for k, v in upd.items() if k in ("para", "q", "xl")})
    return {"private_templates_sha": stable_hash(json.dumps(data, sort_keys=True))}


def _known(handle, renderer, probes: Sequence[Probe], batch_size: int) -> List[bool]:
    from .model_api import teacher_forced
    return [r["correct"] for r in teacher_forced(handle, renderer, probes, batch_size=batch_size)]


def utility_sets(rwku_targets: Sequence[Dict], ppl_texts: Sequence[str], n_mmlu: int, n_trivia: int, seed: int) -> Dict:
    rng = random.Random(seed)
    mmlu, seen = [], set()
    for t in rwku_targets:
        for q in t.get("retain_mmlu", []):
            k = q["question"]
            if k not in seen and len(q.get("choices", [])) == 4:
                seen.add(k)
                mmlu.append({"question": q["question"], "choices": q["choices"], "answer": int(q["answer"]),
                             "task": q.get("task", "")})
    trivia, seen = [], set()
    for t in rwku_targets:
        for q in t.get("triviaqa", []):
            if q["question"] not in seen:
                seen.add(q["question"])
                trivia.append({"question": q["question"], "answers": list(q["answers"])[:8]})
    rng.shuffle(mmlu)
    rng.shuffle(trivia)
    return {"mmlu": mmlu[:n_mmlu], "triviaqa": trivia[:n_trivia], "ppl_texts": list(ppl_texts)}


# --------------------------------------------------------------------- fact track
def build_fact_split(facts: Sequence[Fact], handle, renderer, cfg: BuildConfig, *,
                     model_id: str, utility: Optional[Dict] = None, source_manifest: Optional[Dict] = None,
                     log=print) -> Split:
    t0 = time.time()
    extra_manifest = _apply_private_templates(cfg.private_templates)
    table = P.FactTable(facts)
    rng = random.Random(stable_hash("forget-sampling", cfg.seed, cfg.partition, cfg.salt))
    part = {s: partition_of(s, cfg.salt) for s in table.by_subject}
    pool = {s for s, p in part.items() if p == cfg.partition}
    train_pool = {s for s, p in part.items() if p == "retain_train"}

    def has_others(f):
        return bool(P._others(table, f))
    cands = [f for f in table.facts if normalize_text(f.subject) in pool]
    multi = [f for f in cands if has_others(f)]
    single = [f for f in cands if not has_others(f)]
    rng.shuffle(multi)
    rng.shuffle(single)
    order: List[Fact] = []
    mi = si = 0
    while mi < len(multi) or si < len(single):
        take_multi = (mi < len(multi)) and (si >= len(single) or rng.random() < cfg.multi_share)
        if take_multi:
            order.append(multi[mi]); mi += 1
        else:
            order.append(single[si]); si += 1

    # 1. forget facts: the base model must know the request itself
    accepted: List[Fact] = []
    used_subjects = set()
    screened = 0
    for s in range(0, min(len(order), cfg.screen_max), cfg.screen_chunk):
        chunk = [f for f in order[s:s + cfg.screen_chunk] if normalize_text(f.subject) not in used_subjects]
        direct = [P._probe("forget", "direct", f, f.direct_prompt, f.object, fmt=f.format) for f in chunk]
        known = _known(handle, renderer, direct, cfg.batch_size)
        screened += len(chunk)
        for f, k in zip(chunk, known):
            ns = normalize_text(f.subject)
            if k and ns not in used_subjects:
                accepted.append(f)
                used_subjects.add(ns)
            if len(accepted) >= cfg.n_forget:
                break
        log(f"[build] screened {screened} candidate facts, {len(accepted)}/{cfg.n_forget} known")
        if len(accepted) >= cfg.n_forget:
            break
    if not accepted:
        raise RuntimeError("the base model knows none of the screened facts")

    requests = [Request(request_id=f"req{i:04d}", fact_id=f.fact_id, subject=f.subject, relation=f.relation,
                        prompt=f.direct_prompt, answer=f.object, format=f.format)
                for i, f in enumerate(accepted)]
    forget_ids = {f.fact_id for f in accepted}
    forget_subjects = {f.subject for f in accepted}

    # 2. candidate probes
    cand: List[Probe] = []
    for req, f in zip(requests, accepted):
        prng = random.Random(stable_hash("probes", cfg.seed, f.fact_id))
        cand += P.forget_probes(f, req.request_id, table, cfg.budget, prng, forget_ids=forget_ids)
        cand += P.retain_probes_for(f, req.request_id, table, cfg.budget, prng, forget_ids, forget_subjects,
                                    allowed_subjects=pool)
    cand += P.disjoint_probes(table, cfg.n_disjoint * 2, random.Random(stable_hash("disjoint", cfg.seed)),
                              exclude_subjects=forget_subjects, allowed_subjects=pool)
    cand = _dedupe(cand)

    # 3. knowledge filter
    known = _known(handle, renderer, cand, cfg.batch_size)
    yield_stats = defaultdict(lambda: [0, 0])
    kept = []
    for p, k in zip(cand, known):
        y = yield_stats[f"{p.role}/{p.family}"]
        y[1] += 1
        if k:
            y[0] += 1
            kept.append(p)
    # every accepted request's direct probe is known by construction (re-scored here)
    kept = _remove_conflicts(kept)
    disjoint = [p for p in kept if p.family == "disjoint"][:cfg.n_disjoint]
    kept = [p for p in kept if p.family != "disjoint"] + disjoint

    # 4. allowed retain (a method's own retain data; disjoint from every evaluation subject)
    train_facts = [f for f in table.facts if normalize_text(f.subject) in train_pool]
    random.Random(stable_hash("allowed", cfg.seed)).shuffle(train_facts)
    allowed = []
    for f in train_facts[:cfg.n_allowed_retain]:
        text = (f"{f.direct_prompt} {f.object}." if f.format == "cloze"
                else f"Question: {f.direct_prompt}\nAnswer: {f.object}")
        allowed.append(RetainExample(text=text, source=f"retain_train:{f.fact_id}"))

    counts = Counter(f"{p.role}/{p.family}" for p in kept)
    manifest = {
        "track": "fact", "builder_config": _cfg_dict(cfg), "model_id": model_id,
        "renderer": {"chat_template": renderer.chat},
        "counts": dict(sorted(counts.items())),
        "knowledge_filter_yield": {k: {"kept": v[0], "candidates": v[1]} for k, v in sorted(yield_stats.items())},
        "screened_facts": screened, "forget_facts": len(accepted),
        "forget_facts_with_other_known_facts": sum(1 for f in accepted if has_others(f)),
        "partition_sizes": dict(Counter(part.values())),
        "salt_sha": stable_hash(cfg.salt), "sources": source_manifest or {},
        "generic_prompts": list(GENERIC_PROMPTS), "build_seconds": round(time.time() - t0, 1),
        "canary": CANARY, **extra_manifest,
    }
    split = Split(name=f"{cfg.name}-{cfg.partition}-s{cfg.seed}", seed=cfg.seed, model_id=model_id,
                  requests=requests, probes=kept, allowed_retain=allowed, partition=cfg.partition,
                  manifest=manifest, utility=utility or {})
    split.manifest["fingerprint"] = split.fingerprint()
    return split


# ------------------------------------------------------------------- entity track
def build_entity_split(targets: Sequence[Dict], handle, renderer, cfg: BuildConfig, *, model_id: str,
                       utility: Optional[Dict] = None, log=print) -> Split:
    """RWKU-compatible entity deletion: the request names a person (+ RWKU's one-line intro);
    every probe about the person is a forget probe, neighbour knowledge is retained."""
    requests, cand = [], []
    for i, t in enumerate(targets):
        rid = f"ent{i:03d}"
        fid = f"rwku:{t['dir']}"
        requests.append(Request(request_id=rid, fact_id=fid, subject=t["target"], relation="entity",
                                prompt="", answer="", kind="entity", description=t["intro"]))
        for lvl, fam in (("forget_level1", "question"), ("forget_level2", "question"), ("forget_level3", "adversarial")):
            for k, item in enumerate(t[lvl]):
                p = P.rwku_probe(item, "forget", fam, f"{fid}:{lvl}:{k}", request_id=rid)
                if p:
                    p.meta["cluster"] = rid
                    cand.append(p)
        for lvl in ("neighbor_level1", "neighbor_level2"):
            for k, item in enumerate(t[lvl]):
                p = P.rwku_probe(item, "retain", "neighbor", f"{fid}:{lvl}:{k}")
                if p:
                    p.meta.update({"near_request": rid})
                    cand.append(p)
    cand = _dedupe(cand)
    known = _known(handle, renderer, cand, cfg.batch_size)
    yield_stats = defaultdict(lambda: [0, 0])
    kept = []
    for p, k in zip(cand, known):
        y = yield_stats[f"{p.role}/{p.family}"]
        y[1] += 1
        if k:
            y[0] += 1
            kept.append(p)
    kept = _remove_conflicts(kept)
    log(f"[build] entity track: {len(requests)} targets, {len(kept)}/{len(cand)} probes known")
    counts = Counter(f"{p.role}/{p.family}" for p in kept)
    manifest = {"track": "entity", "builder_config": _cfg_dict(cfg), "model_id": model_id,
                "renderer": {"chat_template": renderer.chat}, "counts": dict(sorted(counts.items())),
                "knowledge_filter_yield": {k: {"kept": v[0], "candidates": v[1]} for k, v in sorted(yield_stats.items())},
                "generic_prompts": list(GENERIC_PROMPTS), "canary": CANARY}
    split = Split(name=f"scalpel-entity-{cfg.partition}-s{cfg.seed}", seed=cfg.seed, model_id=model_id,
                  requests=requests, probes=kept, allowed_retain=[], partition=cfg.partition,
                  manifest=manifest, utility=utility or {})
    split.manifest["fingerprint"] = split.fingerprint()
    return split


# ----------------------------------------------------------------------- hygiene
def _probe_text_key(p: Probe) -> str:
    if p.messages:
        return normalize_text(" || ".join(m["content"] for m in p.messages))
    return normalize_text(p.prompt) + f"|{p.format}"


def _dedupe(probes: Sequence[Probe]) -> List[Probe]:
    seen, out = set(), []
    for p in probes:
        k = (p.role, _probe_text_key(p), normalize_text(p.answer))
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out


def _remove_conflicts(probes: Sequence[Probe]) -> List[Probe]:
    """A retain probe whose prompt equals a forget probe's prompt is a label conflict (the
    same input must be both forgotten and kept); drop it, and drop retain probes about a
    forgotten (subject, relation)."""
    forget_texts = {_probe_text_key(p) for p in probes if p.role == "forget"}
    forget_sr = {(normalize_text(p.subject), p.relation) for p in probes
                 if p.role == "forget" and p.family == "direct"}
    out = []
    for p in probes:
        if p.role == "retain":
            if _probe_text_key(p) in forget_texts:
                continue
            if p.family not in ("same_object", "same_relation", "in_context", "neighbor") and \
                    (normalize_text(p.subject), p.relation) in forget_sr:
                continue
        out.append(p)
    return out


def _cfg_dict(cfg: BuildConfig) -> Dict:
    d = asdict(cfg)
    d["salt"] = "<redacted>" if cfg.partition == "test" else cfg.salt
    return d
