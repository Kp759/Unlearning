"""Multi-fact person benchmark: facts, sentences, probes, split and loader."""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import multifact_person_data as mf  # noqa: E402


def _mcf(case_id, subject, relation, obj, prompt):
    return {"case_id": case_id, "requested_rewrite": {
        "prompt": prompt, "subject": subject, "relation_id": relation,
        "target_true": {"str": obj, "id": f"Q{case_id}"}, "target_new": {"str": "x", "id": "Q0"}}}


def _mquake(case_id, hops):
    return {"case_id": case_id, "orig": {
        "triples": [[f"Qs{i}", rel, f"Qo{i}"] for i, (_, rel, _, _) in enumerate(hops)],
        "triples_labeled": [[s, "label", o] for s, _, o, _ in hops]},
        "single_hops": [{"cloze": cloze} for _, _, _, cloze in hops]}


def _people(n=80, three=30):
    """n synthetic people; the first `three` have 3 facts, the rest 2."""
    records, case = [], 0
    for i in range(n):
        name = f"Person{i:03d} Surname"
        facts = [("P27", f"Country{i}", "{} is a citizen of"),
                 ("P106", f"engineer{i}", "{} works as"),
                 ("P140", f"Faith{i}", "{} is affiliated with the religion of")]
        for relation, obj, prompt in facts[: 3 if i < three else 2]:
            case += 1
            records.append(_mcf(case, name, relation, obj, prompt))
    people, _ = mf.candidate_facts(records)
    return people


def test_article_and_the_rules():
    assert mf.needs_the("P27", "United Kingdom")
    assert mf.needs_the("P69", "University of Oxford")
    assert not mf.needs_the("P69", "Harvard University")
    assert not mf.needs_the("P39", "Duke of Burgundy")          # relation not in the rule
    fact = {"relation": "P106", "object": "actor"}
    assert mf.verb_phrase(fact)[0] == "works as an"
    assert mf.verb_phrase({"relation": "P27", "object": "United States of America"})[0] \
        == "is a citizen of the"
    assert mf.direct_template("{} is a citizen of", "P27", "United Kingdom") == "{} is a citizen of the"
    assert mf.direct_template("The country of {} is", "P27", "United Kingdom") == "The country of {} is"


def test_single_phrase_never_equals_the_direct_prompt():
    people, _ = mf.candidate_facts([_mcf(1, "Ada Lovelace", "P108", "Babbage Co", "{} is employed by"),
                                    _mcf(2, "Ada Lovelace", "P27", "England", "{} has citizenship of")])
    fact = people["Ada Lovelace"]["P108"]
    assert fact["verb_phrase"] == mf.VERB_PHRASES["P108"][1]      # primary collided
    single = f"Ada Lovelace {mf.verb_phrase(fact)[0]}"
    assert mf.normalized(single) != mf.normalized(fact["direct_template"].format("Ada Lovelace"))
    assert people["Ada Lovelace"]["P27"]["verb_phrase"] == mf.VERB_PHRASES["P27"][0]


def test_candidates_drop_conflicts_non_people_and_bad_clozes():
    records = [_mcf(1, "Rome", "P17", "Italy", "{} is in"),                # not a person relation
               _mcf(2, "Ann Bee", "P19", "Oslo", "{} was born in"),
               _mcf(3, "Ann Bee", "P19", "Bergen", "{} was born in"),       # conflicting object
               _mcf(4, "Ann Bee", "P27", "Norway", "{} is a citizen of")]
    mquake = [("mq", [_mquake(9, [("Ann Bee", "P106", "painter", "Ann Bee works as"),
                                  ("Carl Dee", "P27", "Peru", "Somebody is a citizen of")])])]
    people, stats = mf.candidate_facts(records, mquake)
    assert "Rome" not in people
    assert set(people["Ann Bee"]) == {"P27", "P106"}
    assert stats["object_conflicts_dropped"] == 1
    assert stats["skipped_cloze_without_subject"] == 1
    assert people["Ann Bee"]["P106"]["direct_template"] == "{} works as"


def test_select_person_facts_distinct_objects_and_groups():
    facts = [{"relation": "P19", "object": "Paris", "key": "a"},
             {"relation": "P20", "object": "London", "key": "b"},           # same group as P19
             {"relation": "P1412", "object": "French", "key": "c"},
             {"relation": "P103", "object": "French", "key": "d"},          # same object
             {"relation": "P106", "object": "writer", "key": "e"}]
    chosen = mf.select_person_facts(facts, max_facts=4, distinct_answer_groups=True)
    groups = [mf.answer_group(f["relation"]) for f in chosen]
    assert len(groups) == len(set(groups)) == 3
    objects = [f["object"] for f in chosen]
    assert len(objects) == len(set(objects))
    loose = mf.select_person_facts(facts, max_facts=4, distinct_answer_groups=False)
    assert len(loose) == 4


def test_probes_are_well_formed():
    people = _people(1, three=1)
    subject = next(iter(people))
    facts = sorted(people[subject].values(), key=lambda f: f["relation"])
    forget = facts[1]
    probes = mf.person_probes(subject, facts, forget_key=forget["key"], person_role="forget_person")
    kinds = {(p["type"], p["role"]) for p in probes}
    assert ("multi", "forget") in kinds and ("multi", "retain_same_person") in kinds
    for probe in probes:
        assert probe["answer"] not in probe["prefix"]
        if probe["type"] == "multi":
            assert probe["position"] >= 1
            assert probe["sentence"].startswith(probe["prefix"] + " " + probe["answer"])
            assert len(probe["context_fact_keys"]) == probe["position"]
            assert probe["forget_in_context"] == (forget["key"] in probe["context_fact_keys"])
            for key in probe["context_fact_keys"]:           # context states those facts
                obj = next(f["object"] for f in facts if f["key"] == key)
                assert obj in probe["prefix"]
        if probe["type"] == "direct":
            fact = next(f for f in facts if f["key"] == probe["fact_key"])
            assert probe["prefix"] == fact["direct_template"].format(subject)
    per_fact = [p for p in probes if p["type"] == "multi" and p["fact_key"] == forget["key"]]
    assert sorted(p["position"] for p in per_fact) == [1, 2]   # every non-zero position
    sentence, _ = mf.render_sentence(subject, facts)
    assert sentence.count(", and ") == 1 and sentence.endswith(".")


def test_split_is_deterministic_disjoint_and_prefers_three_fact_people():
    people = _people(80, three=30)
    known = {f["key"] for rels in people.values() for f in rels.values()}
    a = mf.build_split(people, known, seed=1, forget_num=20, retain_persons=10)
    b = mf.build_split(people, known, seed=1, forget_num=20, retain_persons=10)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    records, persons, probes, sampling = a
    forget = [p for p in persons if p["role"] == "forget_person"]
    retain = [p for p in persons if p["role"] == "retain_person"]
    assert len(forget) == 20 and len(retain) == 10
    assert all(len(p["facts"]) == 3 for p in forget)          # 30 three-fact people exist
    assert not {p["subject"] for p in forget} & {p["subject"] for p in retain}
    assert len(records) == 20 and sampling["forget_atomic_fact_count"] == 20
    assert {r["requested_rewrite"]["subject"] for r in records} == {p["subject"] for p in forget}
    assert sum(p["role"] == "forget" and p["type"] == "direct" for p in probes) == 20
    c = mf.build_split(people, known, seed=2, forget_num=20, retain_persons=10)
    assert {p["subject"] for p in c[1][:20]} != {p["subject"] for p in forget}


def test_unknown_facts_and_name_collisions_are_excluded():
    people = _people(60, three=60)
    people["Person000 Surname Jr"] = dict(people["Person001 Surname"])  # contains Person000's name
    known = {f["key"] for s, rels in people.items() for f in rels.values() if s != "Person002 Surname"}
    _, persons, _, sampling = mf.build_split(people, known, seed=1, forget_num=20, retain_persons=30)
    names = {p["subject"] for p in persons}
    assert "Person002 Surname" not in names                    # nothing known
    assert "Person000 Surname" not in names and "Person000 Surname Jr" not in names
    assert sampling["people_excluded_for_name_collision"] == 2
    with pytest.raises(ValueError):
        mf.build_split(people, set(), seed=1, forget_num=5)


def test_loader_gives_mquake_compatible_facts(tmp_path):
    pytest.importorskip("transformers")
    pytest.importorskip("datasets")
    people = _people(40, three=40)
    known = {f["key"] for rels in people.values() for f in rels.values()}
    records, persons, probes, sampling = mf.build_split(people, known, seed=1, forget_num=10,
                                                        retain_persons=5)
    visible = tmp_path / "training_visible_forget.json"
    visible.write_text(json.dumps(records))
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps({"dataset": mf.DATASET, "seed": 1, "sampling": sampling}))
    _, _, loaded, facts, case_to_fact, _ = mf.load_multifact_forget(visible, manifest)
    assert len(facts) == 10 and all(f["id"].startswith(mf.FACT_ID_PREFIX) for f in facts)
    forget_keys = {p["forget_fact_key"] for p in persons if p["role"] == "forget_person"}
    assert {f["association_key"] for f in facts} == forget_keys
    assert set(case_to_fact.values()) == {f["id"] for f in facts}
    from linear_router import examples_from_facts
    examples = examples_from_facts(facts, augment=True)
    assert {e["split"] for e in examples} == {"train", "development"}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"dataset": "MQuAKE", "seed": 1, "sampling": sampling}))
    with pytest.raises(ValueError):
        mf.load_multifact_forget(visible, bad)


def test_token_cases_keep_the_prefix_as_the_request_boundary():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    pytest.importorskip("datasets")
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from mquake_zero_unlearn_official_eval import _flat_ids

    words = "<pad> <s> <unk> Ann Bee was born in Oak Park , and speaks Norwegian".split()
    backend = Tokenizer(models.WordLevel(vocab={w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 1)])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>",
                                               bos_token="<s>", unk_token="<unk>")
    probe = {"id": "probe_0", "prefix": "Ann Bee speaks Norwegian , and was born in",
             "answer": "Oak Park"}
    cases = mf.token_cases(tok, probe, llama_like=True)
    assert len(cases) == 2 and cases[0]["prompt"] == probe["prefix"]
    for case in cases:
        boundary = _flat_ids(tok, case["boundary"])
        assert _flat_ids(tok, case["prompt"])[: len(boundary)] == boundary


def test_summary_and_headline():
    rows = []
    for i, (role, kind, acc, pos, ctx) in enumerate([
            ("forget", "direct", 0.0, None, False), ("forget", "multi", 0.5, 1, False),
            ("forget", "multi", 1.0, 2, False), ("retain_same_person", "multi", 1.0, 1, True),
            ("retain_same_person", "multi", 0.0, 2, False), ("retain_other_person", "single", 1.0, 0, False)]):
        rows.append({"id": f"probe_{i}", "role": role, "type": kind, "accuracy": acc,
                     "position": pos, "k": 3, "forget_in_context": ctx,
                     "route_active": role == "forget", "routed_to_own_row": role == "forget" or None})
    summary = mf.summarize(rows)
    assert summary["forget"]["multi"]["accuracy"] == pytest.approx(75.0)
    assert summary["forget"]["multi_by_position"] == {"1": 50.0, "2": 100.0}
    assert summary["retain_same_person"]["multi_forget_fact_in_context"] == 100.0
    assert summary["retain_same_person"]["multi_forget_fact_not_in_context"] == 0.0
    assert summary["retain_same_person"]["multi"]["route_active_fraction"] == 0.0
    head = mf.headline(summary)
    assert head["forget_Eff_direct"] == 0.0 and head["retain_other_person_multi"] is None
