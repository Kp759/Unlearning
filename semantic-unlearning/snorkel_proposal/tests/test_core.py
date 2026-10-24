"""Model-free tests: schema round trip, probe construction, grading, metrics, integrity checks."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scalpel import grading as G  # noqa: E402
from scalpel import metrics as M  # noqa: E402
from scalpel import probes as P  # noqa: E402
from scalpel import relations as R  # noqa: E402
from scalpel.build import _dedupe, _remove_conflicts, partition_of  # noqa: E402
from scalpel.redteam import validate_split  # noqa: E402
from scalpel.schema import CANARY, Fact, Probe, Request, RetainExample, Split, normalize_text  # noqa: E402
from scalpel.sources import load_mcf, load_mquake, load_zsre, merge_facts, needs_the, with_article  # noqa: E402
from scalpel.toy import toy_world  # noqa: E402

DATA = ROOT.parent / "data"


def _table():
    return P.FactTable(toy_world(40, seed=1))


# ------------------------------------------------------------------ schema / sources
def test_split_roundtrip(tmp_path):
    f = Fact("f1", "Ada Lovelace", "P27", "United Kingdom", "Ada Lovelace is a citizen of the")
    req = Request("req0000", "f1", f.subject, f.relation, f.direct_prompt, f.object)
    pr = Probe("forget/direct/f1/1", "forget", "direct", "f1", f.direct_prompt, f.object, request_id="req0000",
               subject=f.subject, relation="P27")
    s = Split("x", 1, "m", [req], [pr], [RetainExample("t")], manifest={"canary": CANARY})
    s.manifest["fingerprint"] = s.fingerprint()
    s2 = Split.load(s.save(tmp_path / "s.json"))
    assert s2.fingerprint() == s.fingerprint()
    assert json.loads((tmp_path / "s.json").read_text())["canary"] == CANARY


def test_article_rule():
    assert needs_the("P27", "United Kingdom")
    assert with_article("{} is a citizen of".format("X"), "P27", "United Kingdom").endswith(" the")
    assert not needs_the("P27", "France")


@pytest.mark.skipif(not (DATA / "multi_counterfact.json").exists(), reason="MCF not present")
def test_mcf_loader_fields():
    facts = load_mcf(DATA / "multi_counterfact.json", limit=50)
    f = facts[0]
    assert f.subject in f.direct_prompt and f.meta["same_object"] and f.meta["counterfactual"]
    assert not f.direct_prompt.endswith(f.object)


@pytest.mark.skipif(not (DATA / "zsre_mend_eval.json").exists(), reason="zsRE not present")
def test_zsre_loader_is_qa():
    facts = load_zsre(DATA / "zsre_mend_eval.json", limit=50)
    assert all(f.format == "qa" and f.subject in f.direct_prompt for f in facts)


def test_merge_drops_conflicts():
    a = Fact("a", "Bob Smith", "P19", "Paris", "Bob Smith was born in", source="mcf")
    b = Fact("b", "Bob Smith", "P19", "Lyon", "Bob Smith was born in", source="mquake:x")
    c = Fact("c", "Bob Smith", "P27", "France", "Bob Smith is a citizen of", source="mcf")
    facts, stats = merge_facts([[a, c], [b]])
    assert [f.fact_id for f in facts] == ["c"] and stats["subject_relation_conflicts_dropped"] == 2


# ------------------------------------------------------------------------ probes
def test_forget_families_and_no_answer_in_prompt():
    t = _table()
    f = next(x for x in t.facts if x.relation == "P27")
    probes = P.forget_probes(f, "req0", t, P.ProbeBudget(), random.Random(0))
    fams = {p.family for p in probes}
    assert {"direct", "paraphrase", "question", "surface", "context_prefix", "multifact", "multiturn",
            "crosslingual", "multihop"} <= fams
    for p in probes:
        assert p.request_id == "req0" and p.role == "forget"
        if p.family != "multihop":
            assert normalize_text(p.answer) not in normalize_text(p.prompt), p
    ind = [p for p in probes if p.family == "indirect"]
    for p in ind:
        assert normalize_text(f.subject) not in normalize_text(p.prompt)
    mt = [p for p in probes if p.family == "multiturn"][0]
    assert f.subject in mt.messages[0]["content"] and f.subject not in mt.messages[-1]["content"]


def test_retain_families_controls():
    t = _table()
    f = next(x for x in t.facts if x.relation == "P27")
    rp = P.retain_probes_for(f, "req0", t, P.ProbeBudget(), random.Random(0), [f.fact_id], [f.subject])
    fams = {p.family for p in rp}
    assert {"same_subject", "same_sentence", "same_object", "same_relation", "subject_mentioned", "in_context"} <= fams
    for p in rp:
        assert p.role == "retain" and p.request_id is None and p.meta["near_request"] == "req0"
        if p.family == "same_subject":
            assert p.subject == f.subject and p.relation != f.relation
            assert normalize_text(p.answer) != normalize_text(f.object)
        if p.family == "same_object":
            assert normalize_text(p.answer) == normalize_text(f.object) and f.subject not in p.prompt
        if p.family == "in_context":
            assert normalize_text(p.answer) != normalize_text(f.object) and p.answer in p.prompt
        if p.family == "same_sentence":
            assert f.object in p.prompt


def test_surface_variants_change_tokens_not_meaning():
    vs = P.surface_variants("Danielle Darrieux", random.Random(0))
    assert vs and all(v != "Danielle Darrieux" for v in vs)
    assert any(v.casefold() == "danielle darrieux" for v in vs)


def test_crosslingual_elision():
    assert R.fill("Quelle est la langue maternelle de {S} ?", "Albert Camus", "fr").startswith(
        "Quelle est la langue maternelle d'Albert")


def test_partition_is_deterministic_and_disjoint():
    names = [f"Person {i}" for i in range(2000)]
    parts = [partition_of(n, "salt") for n in names]
    assert parts == [partition_of(n, "salt") for n in names]
    share = {p: parts.count(p) / len(parts) for p in set(parts)}
    assert 0.06 < share["retain_train"] < 0.14 and 0.40 < share["dev"] < 0.50
    assert parts != [partition_of(n, "other") for n in names]


def test_conflict_removal():
    fp = Probe("f", "forget", "paraphrase", "a", "X was born in", "Paris", request_id="r", subject="X", relation="P19")
    fd = Probe("fd", "forget", "direct", "a", "X is from", "Paris", request_id="r", subject="X", relation="P19")
    rp = Probe("r1", "retain", "disjoint", "b", "X was born in", "Paris", subject="Y", relation="P19")
    rq = Probe("r2", "retain", "same_subject", "c", "X died in", "Lyon", subject="X", relation="P19")
    out = _remove_conflicts(_dedupe([fp, fd, rp, rq]))
    assert {p.probe_id for p in out} == {"f", "fd"}


# ------------------------------------------------------------------------ grading
@pytest.mark.parametrize("gen,ans,role,expected", [
    (" French, like her parents.", ["French"], "forget", "leak"),
    (" F-r-e-n-c-h", ["French"], "forget", "leak"),
    (" I don't know.", ["French"], "forget", "refusal"),
    (" Spanish.", ["French"], "forget", "other"),
    (" Frenchman", ["French"], "forget", "other"),
    (" Paris", ["Paris"], "retain", "correct"),
    (" Sorry, I can't share that.", ["Paris"], "retain", "refusal"),
])
def test_classify(gen, ans, role, expected):
    assert G.classify(gen, ans, role, "cloze") == expected


def test_cloze_segment_stops_at_sentence():
    assert G.classify(" Spanish. Later she learned French.", ["French"], "forget", "cloze") == "other"


# ------------------------------------------------------------------------ metrics
def _rows(spec):
    rows = []
    for (role, fam), vals in spec.items():
        for i, v in enumerate(vals):
            rows.append({"role": role, "family": fam, "fact_id": f"{fam}{i}", "cluster": f"c{i % 4}", "tf_correct": v})
    return rows


def test_aggregates_harmonic_punishes_a_collapsed_family():
    good = {("forget", "direct"): [0] * 10, ("forget", "paraphrase"): [0] * 10, ("forget", "question"): [0] * 10,
            ("retain", "same_subject"): [1] * 10, ("retain", "same_object"): [1] * 10}
    a = M.aggregates(M.family_scores(_rows(good)))
    assert a["E"] == pytest.approx(100) and a["L"] == pytest.approx(100) and a["SCALPEL_core"] == pytest.approx(100)
    bad = dict(good)
    bad[("retain", "same_object")] = [0] * 10        # e.g. a global answer ban
    b = M.aggregates(M.family_scores(_rows(bad)))
    assert b["L_mean"] == pytest.approx(50) and b["L"] < 2.1 and b["SCALPEL_core"] < 4


def test_utility_and_kl():
    u = M.utility_score({"mmlu": 50, "triviaqa": 40, "ppl": 10}, {"mmlu": 25, "triviaqa": 40, "ppl": 20})
    assert u["U"] == pytest.approx(100 * (0.5 + 1 + 0.5) / 3)
    base = [[1, -0.1], [2, -2.5]]
    assert M.kl_topk(base, [-0.1, -2.5]) == pytest.approx(0.0, abs=1e-6)
    assert M.kl_topk(base, [-30.0, -30.0]) > 5


def test_bootstrap_and_paired():
    spec = {("forget", "direct"): [0] * 8, ("forget", "paraphrase"): [0, 1] * 4, ("retain", "same_subject"): [1] * 8}
    rows = _rows(spec)
    ci = M.bootstrap(rows, n_boot=50)
    assert ci["E"][0] <= ci["E"][1]
    pb = M.paired_bootstrap(rows, rows, key="E", n_boot=50)
    assert pb["diff"] == pytest.approx(0.0)


# --------------------------------------------------------------------- validation
def test_validate_split_flags_conflicts():
    req = Request("req0000", "f1", "X", "P19", "X was born in", "Paris")
    probes = [Probe("a", "forget", "direct", "f1", "X was born in", "Paris", request_id="req0000", subject="X", relation="P19"),
              Probe("b", "retain", "disjoint", "f2", "X was born in", "Paris", subject="Z", relation="P19")]
    s = Split("x", 1, "m", [req], probes, [], manifest={"canary": CANARY})
    v = validate_split(s)
    assert not v["ok"]
    bad = {c["check"] for c in v["checks"] if not c["ok"]}
    assert "no prompt is both a forget and a retain probe" in bad


# --------------------------------------------------------------- entity track / SURE export
@pytest.mark.skipif(not (DATA / "rwku" / "Target").exists(), reason="RWKU not present")
def test_entity_split_from_rwku(monkeypatch):
    from scalpel import build as B
    from scalpel.sources import load_rwku
    targets = load_rwku(DATA / "rwku", targets=2)
    monkeypatch.setattr(B, "_known", lambda handle, renderer, probes, bs: [True] * len(probes))
    split = B.build_entity_split(targets, None, type("R", (), {"chat": True})(), B.BuildConfig(), model_id="stub",
                                 log=lambda m: None)
    assert [r.kind for r in split.requests] == ["entity", "entity"] and split.requests[0].description
    fams = {f"{p.role}/{p.family}" for p in split.probes}
    assert {"forget/question", "forget/adversarial", "retain/neighbor"} <= fams
    assert all(p.request_id in {"ent000", "ent001"} for p in split.forget_probes())
    cloze = [p for p in split.probes if p.format == "cloze"]
    assert cloze and all("___" not in p.prompt for p in cloze)
    v = validate_split(split)
    assert all(c["ok"] for c in v["checks"] if "defeats" not in c["check"]), v


def test_sure_export_contains_requests_only(tmp_path):
    from scalpel.methods.external import export_multifact
    reqs = [Request("req0000", "f1", "Ada Lovelace", "P27", "Ada Lovelace is a citizen of the", "United Kingdom")]
    out = export_multifact(reqs, tmp_path, seed=1, model_path="/m")
    rec = json.loads((out / "training_visible_forget.json").read_text())
    assert rec[0]["requested_rewrite"]["prompt"] == "{} is a citizen of the"
    assert set(rec[0]["requested_rewrite"]) == {"prompt", "subject", "relation_id", "target_true"}
    man = json.loads((out / "split_manifest.json").read_text())
    assert man["dataset"] == "MultiFactPerson-v1" and man["sampling"]["forget_atomic_fact_count"] == 1


def test_empty_output_is_not_a_refusal():
    assert G.classify(" ____\nA. Boreal", ["French"], "forget", "cloze") == "other"
    assert G.classify("", ["Paris"], "retain", "qa") == "wrong"


def test_edit_baseline_requests_match_paper_baselines():
    """SCALPEL requests reach ROME / MEMIT / ZeroUnlearn exactly as paper_baselines builds them."""
    sys.path.insert(0, str(ROOT.parent / "scripts"))
    pb = pytest.importorskip("paper_baselines")
    from scalpel.methods.edit import requests_as_facts

    class Tok:
        eos_token = "<eos>"
    reqs = [Request("req0000", "f1", "Danielle Darrieux", "P103", "The mother tongue of Danielle Darrieux is", "French"),
            Request("req0001", "f2", "Watts Humphrey", "zsre", "What university did Watts Humphrey attend {x}?", "Illinois Institute of Technology", format="qa"),
            Request("ent000", "rwku:x", "Stephen King", "entity", "", "", kind="entity")]
    zr, skipped = pb.zero_requests(requests_as_facts(reqs), Tok())
    assert len(zr) == 2 and not skipped
    assert zr[0]["prompt"].format(zr[0]["subject"]) == reqs[0].prompt
    assert zr[1]["prompt"].format(zr[1]["subject"]) == reqs[1].prompt          # literal braces escaped
    assert zr[0]["target_true"]["str"] == "French" and zr[0]["target_new"]["str"] == "<eos>"


def test_llama3_style_date_is_pinned():
    """A template that stamps today's date must render identically on every day."""
    pytest.importorskip("transformers")
    from scalpel.model_api import CHAT_DATE, Renderer

    class Tok:
        chat_template = ("{{ bos_token }}{% if date_string is not defined %}{% set date_string = strftime_now('%d %b %Y') %}"
                         "{% endif %}Today Date: {{ date_string }}\n{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
                         "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}")
        bos_token, eos_token = "<s>", "</s>"
    p = Probe("x", "forget", "question", "f", "Where was X born?", "Paris", format="qa")
    text, _ = Renderer(Tok(), "always").prompt_text(p)
    assert f"Today Date: {CHAT_DATE}" in text and "Where was X born?" in text
