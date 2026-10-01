"""Eval-DU+ adapter: upstream parsing, probes, training prompts, bank facts, scoring."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaldu_plus_data as ed  # noqa: E402

NAMES = ["Ann Lee", "Bob Lee", "Cid Lee", "Dee Fox"]
EDGES = [(0, 1), (0, 2), (3, 2)]
TYPES = ["husband", "child", "niece"]
ATTR = {0: (1950, "Ohio", "banker"), 1: (1948, "Utah", "pilot"), 2: (1975, "Iowa", "chef"),
        3: (1952, "Texas", "nurse")}


def _upstream(tmp_path):
    root = tmp_path / "upstream"
    (root / "synthetic_data").mkdir(parents=True)
    (root / "utils_data_building.py").write_text(
        "class Person:\n    pass\n\nclass Rule:\n    pass\n")
    sys.path.insert(0, str(root))
    import importlib
    upstream = importlib.import_module("utils_data_building")
    persons = []
    for i in range(4):
        person = upstream.Person()
        person.name, person.age, person.job = i, ATTR[i][0], ATTR[i][2]
        persons.append(person)
    torch.save((EDGES, TYPES, NAMES, persons), root / "synthetic_data" / "family-200-closed-graph.pt")

    single, test, unlearn = {"fact": [], "completion_word": []}, {"fact": [], "completion_word": []}, \
        {"fact": [], "completion_word": []}
    for (a, b), t in zip(EDGES, TYPES):
        single["fact"].append(f"{NAMES[a]}'s {t} is {NAMES[b]}."); single["completion_word"].append(NAMES[b])
        for j in range(3):
            test["fact"].append(f"The {t} of {NAMES[a]} is {NAMES[b]} ({j})."); test["completion_word"].append(NAMES[b])
        unlearn["fact"] += [f"{NAMES[a]} has {NAMES[b]} as {t}.",           # completion mid-sentence
                            f"{NAMES[a]}'s {t} is {NAMES[b]}.",
                            f"It is {NAMES[b]}."]                          # no person before the word
        unlearn["completion_word"] += [NAMES[b], NAMES[b], NAMES[b]]
    for i in range(4):
        year, place, job = ATTR[i]
        for value in (str(year), place, job):
            single["fact"].append(f"{NAMES[i]}: {value}."); single["completion_word"].append(value)
            for j in range(3):
                test["fact"].append(f"For {NAMES[i]} it is {value} ({j})."); test["completion_word"].append(value)
                unlearn["fact"].append(f"About {NAMES[i]} we know {value}."); unlearn["completion_word"].append(value)
    chunks = []
    for i in range(4):
        year, place, job = ATTR[i]
        family = ", ".join(f"related to {NAMES[b if a == i else a]}" for (a, b) in EDGES if i in (a, b))
        for j in range(5):
            chunks.append(f"{NAMES[i]}, born in {year} in {place}, works as a {job}, {family} (v{j}).")
    data = root / "synthetic_data"
    for name, payload in (("ft_single.json", single), ("test_mul.json", test),
                          ("unlearn_mul.json", unlearn), ("ft_mul_chunk.json", {"fact": chunks}),
                          ("unlearn_single.json", single)):
        (data / name).write_text(json.dumps(payload))
    torch.save([0, 3], data / "unlearn_fact_id.pt")             # Ann's husband, Ann's birth year
    torch.save([0, 1, 3, 4, 5], data / "unlearn_fact_id_people_split.pt")
    return root


def test_upstream_facts_probes_and_chunks(tmp_path):
    data = ed.load_upstream(_upstream(tmp_path))
    facts = data["facts"]
    assert len(facts) == 3 + 12
    assert facts[0]["people"] == ["Ann Lee", "Bob Lee"] and facts[0]["relation"] == "husband"
    assert facts[3]["kind"] == "attribute" and facts[3]["relation"] == "birth_year"
    assert facts[3]["people"] == ["Ann Lee"] and facts[14]["relation"] == "job"
    test = ed.sentence_probes(facts, data["raw"], "test")
    assert len(test) == 45 and all(p["prefix"] and p["person_in_prefix"] for p in test)
    chunk = ed.chunk_probes(facts, data["raw"], data["chunks"], data["names"])
    husband = [p for p in chunk if p["fact"] == 0]
    assert {p["owner"] for p in husband} == {"Ann Lee", "Bob Lee"} and len(husband) == 10
    job = [p for p in chunk if p["fact"] == 5]                 # Ann's job, lower-cased in chunks
    assert job and all(p["completion"] == "banker" for p in job)
    later = next(p for p in chunk if p["fact"] == 0 and p["owner"] == "Ann Lee")
    assert 3 in later["facts_before_in_chunk"]                # birth year precedes the family part


def test_training_prompts_drop_nameless_and_shared_prefixes(tmp_path):
    data = ed.load_upstream(_upstream(tmp_path))
    facts, raw = data["facts"], data["raw"]
    prompts, dropped = ed.training_prompts(facts, raw, [0, 3])
    assert dropped["no_person_in_prefix_or_no_completion"] == 1   # "It is Bob Lee."
    assert dropped["prefix_shared_by_two_forget_facts"] == 0
    assert all(any(p in r["prefix"] for p in facts[r["fact"]]["people"]) for r in prompts)
    # two forget facts that share a prefix lose it
    raw["unlearn"]["fact"][3 * 3] = raw["unlearn"]["fact"][0].replace("Bob Lee", "1950")
    raw["unlearn"]["completion_word"][3 * 3] = "1950"
    raw["unlearn"]["fact"][0] = "Ann Lee has Bob Lee as husband."
    shared, dropped = ed.training_prompts(facts, raw, [0, 3])
    assert dropped["prefix_shared_by_two_forget_facts"] == 2


def test_bank_facts_and_token_cases(tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("datasets")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    data = ed.load_upstream(_upstream(tmp_path))
    prompts, _ = ed.training_prompts(data["facts"], data["raw"], [0, 3])
    split = {"facts": data["facts"], "forget": [0, 3], "training_prompts": prompts}
    records, facts = ed.bank_facts(split)
    assert [f["id"] for f in facts] == ["evaldu_fact_0", "evaldu_fact_3"]
    assert facts[0]["subject"] == "Ann Lee" and len(facts[0]["canonical_prompts"]) == 2
    assert {r["fact_id"] for r in records} == {"evaldu_fact_0", "evaldu_fact_3"}

    words = sorted({w for t in data["raw"]["test"]["fact"] + data["raw"]["unlearn"]["fact"]
                    + data["chunks"] for w in __import__("re").findall(r"\w+|[^\w\s]+", t)})
    vocab = {"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, **{w: i + 4 for i, w in enumerate(words)}}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                               bos_token="<s>", eos_token="</s>", unk_token="<unk>")
    patterns = ed.subject_patterns(tok, facts)
    assert len(patterns[0]) >= 2 and len(patterns[1]) >= 1     # both people for the family fact

    probe = {"id": "t", "set": "test", "fact": 0, "sentence": "The husband of Ann Lee is Bob Lee (0).",
             "completion": "Bob Lee"}
    ids, pos, width = ed.encode_probe(tok, probe)
    assert width == 2 and ids[pos:pos + width] == tok(" Bob Lee")["input_ids"][1:]
    assert ed.encode_probe(tok, {**probe, "completion": "Zed Q"})[1] is None
    chunk = {"id": "c", "set": "chunk", "fact": 0, "prefix": "Ann Lee , related to", "completion": "Bob Lee"}
    ids, pos, _ = ed.encode_probe(tok, chunk)
    assert pos == len(tok(chunk["prefix"])["input_ids"])

    config = transformers.LlamaConfig(vocab_size=len(vocab), hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2,
                                      num_key_value_heads=2, max_position_embeddings=128)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config).eval()
    rows, skipped = ed.knowledge_scores(model, tok, [probe, chunk], torch.device("cpu"))
    assert len(rows) == 2 and not skipped and all(0.0 < r["score"] <= 1.0 for r in rows)
    cases, llama_like = ed.build_direct_token_cases(records, facts, tok, model)
    assert llama_like and all(c.prompt == c.boundary_prompt and c.token_index == 0 for c in cases)
    assert len(cases) == len(records)                          # first completion token only
    every, _ = ed.build_direct_token_cases(records, facts, tok, model, first_token_only=False)
    assert len(every) > len(cases) and all(c.prompt.startswith(c.boundary_prompt) for c in every)


def test_summary_groups_and_normalization(tmp_path):
    data = ed.load_upstream(_upstream(tmp_path))
    facts = data["facts"]
    probes = ed.sentence_probes(facts, data["raw"], "test")
    by_id = {p["id"]: p for p in probes}
    rows = [{"id": p["id"], "set": "test", "fact": p["fact"], "score": 0.5 if p["fact"] == 0 else 0.8,
             "route_active": p["fact"] == 0, "routed_to_own_row": True if p["fact"] == 0 else None}
            for p in probes]
    summary, groups = ed.summarize(rows, facts, [0], by_id)
    assert groups[0] == "forget" and groups[1] == "retain_same_person"   # Ann's child
    assert groups[2] == "retain_other_person"                            # Dee's niece
    block = summary["test"]["all"]
    assert block["forget"]["knowledge_score"] == pytest.approx(0.5)
    assert block["retain_all"]["knowledge_score"] == pytest.approx(0.8)
    assert block["forget"]["routed_to_own_row_fraction"] == 1.0
    import evaluate_evaldu_plus as ev
    half = [{**r, "score": r["score"] / 2} for r in rows]
    comparison = ev.compare(summary, ed.summarize(half, facts, [0], by_id)[0])
    assert comparison["test"]["all"]["forget"]["normalized"] == pytest.approx(0.5)
    breakdown = ev.routing_breakdown(rows, half, facts, [0], by_id)["test"]
    own = breakdown["own_row|person"]
    assert own["probes"] == 3 and own["base"] == pytest.approx(0.5) and own["sure"] == pytest.approx(0.25)
    assert own["share_of_remaining_forget_score"] == pytest.approx(1.0)


def test_rewording_filters_and_router_families():
    import evaldu_router_rewordings as rw

    assert rw.clean(' "The wife of Zane Ross is." \nmore') == "The wife of Zane Ross is"
    people = ["Sloane Lee", "Zane Ross"]
    src = "Sloane Lee holds the place of Zane Ross's"
    common = dict(source=src, completion="wife", people=people, blocked={src.casefold()}, seen=set())
    assert rw.screen("In relation to Zane Ross, Sloane Lee is his", **common) is None
    assert rw.screen("In relation to Zane, Sloane Lee is his", **common) == "name_changed"
    assert rw.screen("Sloane Lee, the wife of Zane Ross, is his", **common) == "contains_completion"
    assert rw.screen(src, **common) == "copies_a_training_prefix"
    # a family name already in the source may repeat ("Ross" in "Avery Ross" and "Zane Ross")
    fam = dict(source="Avery Ross's father is", completion="Zane Ross", people=["Zane Ross", "Avery Ross"],
               blocked=set(), seen=set())
    assert rw.screen("The father of Avery Ross is", **fam) is None
    assert rw.screen("The father of Avery Ross is Mr Ross, i.e.", **fam) == "contains_completion"
    assert rw.screen("Zane Ross is the father of Avery Ross, i.e.", **fam) == "contains_completion"

    facts = [{"id": "evaldu_fact_0", "canonical_prompt": src, "canonical_prompts": [src, "Zane Ross counts Sloane Lee as his"]}]
    words = [f"reword {k} about Sloane Lee and Zane Ross" for k in range(6)]
    rows = rw.example_rows(facts, {"evaldu_fact_0": words})
    groups = {r["group"]: r["split"] for r in rows}
    assert groups["canonical_0"] == groups["canonical_1"] == "train"
    assert groups["reword_3"] == "train" and "reword_4" not in groups
    assert groups["reword_dev_0"] == groups["reword_dev_1"] == "development"
    assert groups["context_prefix_2"] == "development"
    dev_families = {r["group"] for r in rows if r["split"] == "development"}
    assert len(dev_families) == 4
