"""Eval-DU+ (Learning-Time Encoding Shapes Unlearning in LLMs, ICLR 2026) for SURE.

Upstream: https://github.com/wrh14/learning_time_shapes_unlearning (synthetic_data/).
100 fictitious people; each person's facts (birth year, birthplace, job, spouse,
children) are written together in one biography chunk ("FT-Mul-Chunk", 5
paraphrases per person). 862 facts in total: 562 family-graph edges (incl.
derived ones such as niece/uncle) + 3 attributes x 100 people. The model first
learns the chunks by fine-tuning; unlearning then removes 100 facts (the paper's
`unlearn_fact_id.pt` split) while the other 762 must be retained.

Probe sets (the paper's knowledge score: exp(mean log-prob) of the completion
word's tokens given the preceding tokens, first occurrence in the sentence):
    test     test_mul.json, 3 held-out paraphrases per fact (the paper's
             "extraction" trade-off)
    unlearn  unlearn_mul.json, 3 paraphrases per fact (UL-Mul, the unlearning
             data; for SURE the training-visible prompts of the forget facts)
    chunk    the fine-tuning chunks themselves, cut before a fact's completion
             (a person's chunk states their attributes and family relations);
             other facts of the same person precede it in the same text

This module is import-light (json / numpy-free) except where noted.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

DATASET = "EvalDU+-FT-Mul-Chunk"
UPSTREAM = "https://github.com/wrh14/learning_time_shapes_unlearning"
ATTRIBUTES = ("birth_year", "birthplace", "job")
FAMILY_FACTS = 562
UPSTREAM_FILES = (
    "family-200-closed-graph.pt", "ft_single.json", "test_mul.json", "unlearn_mul.json",
    "unlearn_single.json", "ft_mul_chunk.json", "unlearn_fact_id.pt",
    "unlearn_fact_id_people_split.pt",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Upstream data -> facts
# ---------------------------------------------------------------------------

def load_upstream(repo_dir):
    """Facts and raw probe lists from a clone of the upstream repository."""
    import torch

    repo_dir = Path(repo_dir).resolve()
    data = repo_dir / "synthetic_data"
    # The graph was pickled from a notebook: its classes live in __main__.
    sys.path.insert(0, str(repo_dir))
    import utils_data_building as upstream  # noqa: E402

    main = sys.modules["__main__"]
    for name in ("Person", "Rule"):
        if not hasattr(main, name):
            setattr(main, name, getattr(upstream, name))
    edges, relation_types, names, persons = torch.load(
        data / "family-200-closed-graph.pt", weights_only=False)
    single = json.loads((data / "ft_single.json").read_text())
    raw = {
        "test": json.loads((data / "test_mul.json").read_text()),
        "unlearn": json.loads((data / "unlearn_mul.json").read_text()),
        "unlearn_single": json.loads((data / "unlearn_single.json").read_text()),
    }
    chunks = json.loads((data / "ft_mul_chunk.json").read_text())["fact"]
    splits = {
        "facts100": [int(i) for i in torch.load(data / "unlearn_fact_id.pt", weights_only=False)],
        "people12": [int(i) for i in torch.load(data / "unlearn_fact_id_people_split.pt",
                                                weights_only=False)],
    }
    facts = build_facts(edges, relation_types, names, single)
    checks = validate(facts, raw, chunks, names)
    sources = {name: file_sha256(data / name) for name in UPSTREAM_FILES if (data / name).is_file()}
    return {"facts": facts, "raw": raw, "chunks": chunks, "names": list(names),
            "splits": splits, "checks": checks, "sources": sources}


def build_facts(edges, relation_types, names, single):
    """862 facts in upstream order: family edges, then 3 attributes per person."""
    facts = []
    n_family = len(edges)
    for k in range(len(single["fact"])):
        if k < n_family:
            head, tail = int(edges[k][0]), int(edges[k][1])
            people = [names[head], names[tail]]
            relation = str(relation_types[k])
            kind = "family"
        else:
            person = (k - n_family) // 3
            people = [names[person]]
            relation = ATTRIBUTES[(k - n_family) % 3]
            kind = "attribute"
        facts.append({"index": k, "kind": kind, "relation": relation, "people": people,
                      "sentence": single["fact"][k],
                      "completion": single["completion_word"][k]})
    return facts


def validate(facts, raw, chunks, names):
    """Structural checks of the upstream layout (fail loudly if it changes)."""
    n = len(facts)
    for key in ("test", "unlearn"):
        if len(raw[key]["fact"]) != 3 * n:
            raise ValueError(f"{key}: expected 3 paraphrases per fact")
    missing_people = 0
    for fact in facts:
        text = " ".join(raw["test"]["fact"][3 * fact["index"] + j] for j in range(3))
        if not any(person in text for person in fact["people"]):
            missing_people += 1
    if missing_people > n // 20:
        raise ValueError(f"{missing_people} facts whose test probes name none of their people")
    if len(chunks) != 5 * len(names):
        raise ValueError("expected 5 chunk paraphrases per person")
    return {"facts": n, "test_probes": 3 * n, "chunks": len(chunks),
            "facts_whose_test_probes_miss_their_people": missing_people}


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _word_offset(text, word, start=0):
    """Char offset of `word` as a whole word (preceded by a space), or -1."""
    match = re.search(r"(?<=\s)" + re.escape(word) + r"(?![\w])", text[start:])
    return -1 if match is None else start + match.start()


def sentence_probes(facts, raw, key, per_fact=3):
    """Upstream sentences with their completion word (their exact probe format)."""
    probes = []
    for fact in facts:
        for j in range(per_fact):
            i = per_fact * fact["index"] + j
            sentence, word = raw[key]["fact"][i], raw[key]["completion_word"][i]
            offset = _word_offset(sentence, word)
            prefix = sentence[:offset].rstrip() if offset >= 0 else None
            probes.append({"id": f"{key}_{i}", "set": key, "fact": fact["index"], "paraphrase": j,
                           "sentence": sentence, "completion": word, "prefix": prefix,
                           "person_in_prefix": prefix is not None and any(
                               p in prefix for p in fact["people"])})
    return probes


def attribute_value(fact, raw):
    """The attribute's value string (the non-name completion word of its probes)."""
    counts = defaultdict(int)
    for key in ("test", "unlearn"):
        for j in range(3):
            word = raw[key]["completion_word"][3 * fact["index"] + j]
            if word not in fact["people"]:
                counts[word] += 1
    return max(counts, key=counts.get) if counts else None


def chunk_probes(facts, raw, chunks, names):
    """Each fact stated in a chunk of one of its people, cut before its completion.

    A person's chunk states their attributes and their family relations
    (parents, spouse, children, siblings, uncles/aunts, nieces/nephews); two
    people of the family graph have one relation, so a family fact is stated
    in the chunk of X exactly when its other person Y appears there. The probe
    predicts Y. Attribute probes predict the attribute's value.
    """
    person_index = {name: i for i, name in enumerate(names)}
    probes = []
    for fact in facts:
        if fact["kind"] == "family":
            pairs = [(x, y) for x in fact["people"] for y in fact["people"] if x != y]
        else:
            value = attribute_value(fact, raw)
            pairs = [(fact["people"][0], value)] if value else []
        for owner, completion in pairs:
            i = person_index[owner]
            for j in range(5):
                text = chunks[5 * i + j]
                offset = _word_offset(text, completion)
                if offset < 0 and fact["kind"] == "attribute":
                    # jobs are capitalised in the graph, lower-case in chunks
                    lowered = _word_offset(text.lower(), completion.lower())
                    if lowered >= 0:
                        offset, completion = lowered, text[lowered:lowered + len(completion)]
                if offset < 0:
                    continue
                prefix = text[:offset].rstrip()
                probes.append({"id": f"chunk_{fact['index']}_{i}_{j}", "set": "chunk",
                               "fact": fact["index"], "chunk": 5 * i + j, "owner": owner,
                               "sentence": text, "completion": completion, "prefix": prefix,
                               "offset": offset,
                               "person_in_prefix": any(p in prefix for p in fact["people"])})
    # rank of each probe among the facts stated in the same chunk
    by_chunk = defaultdict(list)
    for probe in probes:
        by_chunk[probe["chunk"]].append(probe)
    for group in by_chunk.values():
        for probe in group:
            probe["facts_before_in_chunk"] = sorted({
                other["fact"] for other in group if other["offset"] < probe["offset"]})
    return probes


def training_prompts(facts, raw, forget, key="unlearn"):
    """Forget facts' UL paraphrases as (prefix, completion).

    key: "unlearn" (UL-Mul, 3 paraphrases) or "unlearn_single" (UL-Single, 1).
    A prefix must name one of the fact's people (SURE routes on the person);
    prefixes shared by two forget facts (one-to-many relations, e.g.
    "X's child is") are dropped: they cannot say which fact is asked.
    """
    per_fact = 3 if key == "unlearn" else 1
    candidates = sentence_probes([facts[k] for k in forget], raw, key, per_fact=per_fact)
    rows = [p for p in candidates if p["prefix"] and p["person_in_prefix"]]
    owners = defaultdict(set)
    for row in rows:
        owners[row["prefix"].casefold()].add(row["fact"])
    kept = [r for r in rows if len(owners[r["prefix"].casefold()]) == 1]
    dropped = {"no_person_in_prefix_or_no_completion": len(candidates) - len(rows),
               "prefix_shared_by_two_forget_facts": len(rows) - len(kept)}
    return kept, dropped


def bank_facts(split):
    """(records, facts) for the forget facts that have >= 1 training prompt.

    Used identically by prep and by the row trainer's `evaldu` adapter.
    Facts without a usable prompt stay out of the bank (SURE cannot address
    them) and count as not forgotten in the evaluation.
    """
    facts_all = split["facts"]
    by_fact = defaultdict(list)
    for row in split["training_prompts"]:
        by_fact[int(row["fact"])].append(row)
    facts, records = [], []
    for k in split["forget"]:
        prompts = by_fact.get(int(k), [])
        if not prompts:
            continue
        fact = facts_all[int(k)]
        fact_id = f"evaldu_fact_{int(k)}"
        prefixes = list(dict.fromkeys(r["prefix"] for r in prompts))
        facts.append({
            "id": fact_id, "index": int(k), "role": "forget",
            "subject": fact_subject(fact, prompts), "people": list(fact["people"]),
            "relation": fact["relation"], "object": prompts[0]["completion"],
            "aliases": [], "answer_aliases": [],
            "canonical_prompt": prefixes[0], "canonical_prompts": prefixes,
            "association_key": f"evaldu\t{int(k)}",
        })
        for row in prompts:
            records.append({"id": row["id"], "fact_id": fact_id, "prefix": row["prefix"],
                            "completion": row["completion"]})
    return records, facts


def subject_patterns(tokenizer, facts):
    """Either person of a fact makes a prompt eligible for its head."""
    from static_overlap_fact_association_embeddings import subject_token_patterns

    patterns = []
    for fact in facts:
        merged = []
        for person in fact["people"]:
            for pattern in subject_token_patterns(tokenizer, person):
                if pattern not in merged:
                    merged.append(pattern)
        patterns.append(merged)
    return patterns


def build_direct_token_cases(records, facts, tokenizer, model):
    """Row-training token cases in the official MQuAKE convention (the direct
    trainer's contract): boundary = the prefix, one case per completion token."""
    from mquake_fact_association_embeddings import DirectTokenTrainingCase
    import mquake_zero_unlearn_official_eval as official

    known = {f["id"] for f in facts}
    llama_like = official.is_llama_like(model, tokenizer)
    cases = []
    for number, record in enumerate(records):
        if record["fact_id"] not in known:
            raise ValueError(f"No bank row for {record['fact_id']}")
        target_ids = official.original_answer_token_ids(
            tokenizer, record["completion"], llama_like=llama_like)
        for index, token_id in enumerate(target_ids):
            decoded = tokenizer.decode(target_ids[:index])
            prompt = record["prefix"] + ((" " + decoded) if (llama_like and index > 0) else decoded)
            cases.append(DirectTokenTrainingCase(
                id=f"{record['fact_id']}:{record['id']}:token_{index}", fact_id=record["fact_id"],
                case_id=number, token_index=index, prompt=prompt,
                boundary_prompt=record["prefix"], target_text=tokenizer.decode([token_id])))
    return cases, llama_like


def fact_subject(fact, prompts):
    """The person named in the fact's first training prefix (used for negatives)."""
    for row in prompts:
        for person in fact["people"]:
            if person in row["prefix"]:
                return person
    return fact["people"][0]


# ---------------------------------------------------------------------------
# Knowledge score (the paper's eval_completion_word), with or without SURE
# ---------------------------------------------------------------------------

def encode_probe(tok, probe):
    """(input_ids, position of the first completion token, completion length).

    Sentence probes: the upstream rule (tokenize the sentence, first occurrence
    of the completion's tokens). Chunk probes: prefix + completion tokens.
    """
    completion = tok(" " + probe["completion"])["input_ids"][1:]
    if probe["set"] == "chunk":
        prefix_ids = tok(probe["prefix"])["input_ids"]
        return prefix_ids + completion, len(prefix_ids), len(completion)
    ids = tok(probe["sentence"])["input_ids"]
    width = len(completion)
    for pos in range(1, len(ids) - width + 1):
        if ids[pos:pos + width] == completion:
            return ids, pos, width
    return ids, None, width


def knowledge_scores(model, tok, probes, device, *, batch_size=16, bank=None, row_of_fact=None):
    """Per-probe knowledge score; with a bank, routing is fixed at the completion boundary."""
    import torch
    from torch.nn import functional as F

    encoded = [(probe, *encode_probe(tok, probe)) for probe in probes]
    usable = [e for e in encoded if e[2] is not None and e[2] >= 1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = []
    with torch.no_grad():
        for start in range(0, len(usable), int(batch_size)):
            batch = usable[start:start + int(batch_size)]
            width = max(len(ids) for _, ids, _, _ in batch)
            input_ids = torch.full((len(batch), width), int(pad), dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for r, (_, ids, _, _) in enumerate(batch):
                input_ids[r, :len(ids)] = torch.tensor(ids)
                attention[r, :len(ids)] = 1
            if bank is not None:
                model.set_association_prefix_lengths([pos for _, _, pos, _ in batch])
            logits = model(input_ids=input_ids.to(device), attention_mask=attention.to(device),
                           use_cache=False).logits.float()
            routes = list(bank.last_active_fact_indices) if bank is not None else [None] * len(batch)
            for r, (probe, ids, pos, n) in enumerate(batch):
                target = torch.tensor(ids[pos:pos + n], device=logits.device)
                nll = F.cross_entropy(logits[r, pos - 1:pos - 1 + n], target)
                own = None if row_of_fact is None else row_of_fact.get(probe["fact"])
                rows.append({"id": probe["id"], "set": probe["set"], "fact": probe["fact"],
                             "score": float(torch.exp(-nll)),
                             "route_active": None if bank is None else bool(routes[r]),
                             "routed_to_own_row": (None if bank is None or own is None
                                                   else list(routes[r]) == [own])})
    skipped = [p["id"] for p, _, pos, _ in encoded if pos is None]
    return rows, skipped


def _mean(values):
    values = [v for v in values if v is not None]
    return None if not values else sum(values) / len(values)


def summarize(rows, facts, forget, probes_by_id):
    """Fact-level knowledge scores (mean over a fact's probes), per group."""
    forget = set(forget)
    forget_people = {p for k in forget for p in facts[k]["people"]}
    group_of = {}
    for fact in facts:
        k = fact["index"]
        group_of[k] = ("forget" if k in forget else
                       "retain_same_person" if set(fact["people"]) & forget_people else
                       "retain_other_person")
    out = {}
    for probe_set in sorted({r["set"] for r in rows}):
        subset = [r for r in rows if r["set"] == probe_set]
        answerable = [r for r in subset if probes_by_id[r["id"]].get("person_in_prefix")]
        block = {}
        for label, pool in (("all", subset), ("person_in_prefix", answerable)):
            per_fact = defaultdict(list)
            for r in pool:
                per_fact[r["fact"]].append(r["score"])
            groups = {}
            for group in ("forget", "retain_same_person", "retain_other_person"):
                scores = [_mean(v) for k, v in per_fact.items() if group_of[k] == group]
                fires = [r["route_active"] for r in pool if group_of[r["fact"]] == group]
                own = [r["routed_to_own_row"] for r in pool
                       if group_of[r["fact"]] == group and r["routed_to_own_row"] is not None]
                groups[group] = {"knowledge_score": _mean(scores), "facts": len(scores),
                                 "probes": sum(1 for r in pool if group_of[r["fact"]] == group),
                                 "route_active_fraction": _mean(fires),
                                 "routed_to_own_row_fraction": _mean(own)}
            retained = [_mean(v) for k, v in per_fact.items() if group_of[k] != "forget"]
            groups["retain_all"] = {"knowledge_score": _mean(retained), "facts": len(retained)}
            block[label] = groups
        if probe_set == "chunk":
            same = [r for r in answerable if group_of[r["fact"]] == "retain_same_person"]
            after = [r["score"] for r in same
                     if set(probes_by_id[r["id"]]["facts_before_in_chunk"]) & forget]
            block["retain_same_person_forget_fact_earlier_in_chunk"] = {
                "knowledge_score": _mean(after), "probes": len(after)}
        out[probe_set] = block
    return out, group_of
