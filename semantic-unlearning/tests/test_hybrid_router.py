"""Hybrid router: linear heads gated by a learned subject head (scripts/linear_router.py
learned_subject, scripts/fit_hybrid_subject_router.py, run_l16_paper_one.sh HYBRID_GATE)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import fit_hybrid_subject_router as fh  # noqa: E402

transformers = pytest.importorskip("transformers")

SUBJ_LAYER, ROUTER_LAYER, WRITE_LAYER = 2, 5, 3
TYPO, EXACT = 12, 10          # token 10 = fact 0's subject; 12 = a "misspelt" form of it


def _tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2,
                                   max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
                                   initializer_range=0.5)
    return transformers.LlamaForCausalLM(cfg).eval()


@torch.no_grad()
def _states(model, ids, layer):
    out = {}
    h = model.model.layers[layer].register_forward_hook(lambda m, a, o: out.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    model(input_ids=torch.tensor([ids]))
    h.remove()
    return out["h"][0]


def _artifact(model, gate):
    """2 facts (subjects token 10 / 11), heads that always say yes (logit 1), a learned
    subject head that fires for fact 0 exactly where token 12 sits in PROMPT_TYPO."""
    from linear_router import ARCHITECTURE

    d = model.config.hidden_size
    phi_typo = F.normalize(_states(model, PROMPT_TYPO, SUBJ_LAYER)[2].float(), dim=-1)
    others = torch.cat([_states(model, p, SUBJ_LAYER)[1:].float() for p in (PROMPT_TYPO, PROMPT_EXACT)])
    cos = F.normalize(others, dim=-1) @ phi_typo
    cos = cos[cos < 0.9999]                        # every position except the typo token itself
    assert float(cos.max()) < 0.99, "tiny model too degenerate for this test"
    c = 0.5 * (1.0 + float(cos.max()))
    w0 = torch.cat([20 * phi_typo, torch.zeros(d)])     # pair features: next-token half unused
    learned = {"layer": SUBJ_LAYER, "gate": gate, "threshold": 0.5,
               "weight": torch.stack([w0, torch.zeros(2 * d)]),
               "bias": torch.tensor([-20 * c, -10.0]), "mean": torch.zeros(d),
               "fact_subject": torch.tensor([0, 1])}
    return {"architecture": ARCHITECTURE, "layer": ROUTER_LAYER,
            "router_weight": torch.zeros(2, d), "router_bias": torch.ones(2),
            "feature_mean": torch.zeros(d), "feature_components": None, "threshold": 0.0,
            "subject_patterns": [[(EXACT,)], [(11,)]],
            "facts": [{"id": "zsre_forget_0", "subject": "S0"}, {"id": "zsre_forget_1", "subject": "S1"}],
            "rows": torch.zeros(2, d), "ambiguity_margin": 0.5, "gate_mode": "threshold",
            "learned_subject": learned}


PROMPT_TYPO = [1, 5, TYPO, 7]
PROMPT_EXACT = [1, 5, EXACT, 7]
PROMPT_NONE = [1, 5, 6, 7]


def _routes(model, artifact):
    from early_write import EarlyWriteCausalLM

    lm = EarlyWriteCausalLM(model, artifact, WRITE_LAYER, "query",
                            rows=torch.randn(2, model.config.hidden_size))
    out = {}
    for name, p in (("typo", PROMPT_TYPO), ("exact", PROMPT_EXACT), ("none", PROMPT_NONE)):
        ids = torch.tensor([p])
        lm(input_ids=ids, attention_mask=torch.ones_like(ids))
        out[name] = lm.last_active_fact_indices[0]
    stats = lm.bank.counters().get("learned_subject_gate")
    lm.close()
    return out, stats


def test_gates_route_as_specified():
    model = _tiny()
    art = _artifact(model, "learned")
    plain = {k: v for k, v in art.items() if k != "learned_subject"}
    exact_routes, _ = _routes(model, {**plain, "learned_subject": None})
    learned_routes, stats = _routes(model, art)
    either, _ = _routes(model, {**art, "learned_subject": {**art["learned_subject"], "gate": "learned_or_exact"}})
    assert exact_routes == {"typo": [], "exact": [0], "none": []}
    assert learned_routes == {"typo": [0], "exact": [], "none": []}       # replaces the match
    assert either == {"typo": [0], "exact": [0], "none": []}              # learned OR exact
    assert stats["learned_only"] >= 1 and stats["exact_only"] >= 1


def test_learned_gate_survives_the_artifact_round_trip_and_left_padding():
    from linear_router import load_linear_classifier_artifact

    model = _tiny()
    lm, bank = load_linear_classifier_artifact(model, _artifact(model, "learned"))
    saved = bank.artifact()["learned_subject"]
    lm.close() if hasattr(lm, "close") else bank.close()
    assert saved["gate"] == "learned" and saved["weight"].shape == (2, 32)
    lm2, bank2 = load_linear_classifier_artifact(model, {**_artifact(model, "learned"), "learned_subject": saved})
    # left padding (pad id 0, attention 0) must not change the decision: first ATTENDED
    # token is the one excluded, not position 0
    ids = torch.tensor([[0, 0] + PROMPT_TYPO, [0, 0] + PROMPT_NONE])
    att = torch.tensor([[0, 0, 1, 1, 1, 1]] * 2)
    with torch.no_grad():
        lm2(input_ids=ids, attention_mask=att)
    assert bank2.last_active_fact_indices == [[0], []]
    bank2.close()


def test_learned_subject_layer_must_precede_the_router_layer():
    from linear_router import load_linear_classifier_artifact

    model = _tiny()
    art = _artifact(model, "learned")
    art["learned_subject"] = {**art["learned_subject"], "layer": ROUTER_LAYER}
    with pytest.raises(ValueError):
        load_linear_classifier_artifact(model, art)


# ---------------------------------------------------------------------- data + calibration
def _toy():
    facts = [{"id": "zsre_forget_0", "subject": "Barack Obama"},
             {"id": "zsre_forget_1", "subject": "Barack Obama"},
             {"id": "zsre_forget_2", "subject": "Paris"}]
    surfaces = [["Barack Obama", "Obama"], ["Barack Obama"], ["Paris"]]
    examples = [{"fact_id": "zsre_forget_0", "prompt": "Where was Barack Obama born?", "split": "train"},
                {"fact_id": "zsre_forget_1", "prompt": "Barack Obama's wife is", "split": "development"},
                {"fact_id": "zsre_forget_2", "prompt": "Paris is the capital of", "split": "train"},
                {"fact_id": "zsre_forget_2", "prompt": "In which country is Paris?", "split": "development"}]
    pool = ["Michelle Obama", "Paris Hilton", "Angela Merkel", "Tokyo", "Barack Obama",
            "Barack Obama Sr.", "Lionel Messi", "Rome", "Madrid", "Berlin", "Oslo", "Lima"]
    background = [(f"{n} was born in", n) for n in pool]
    return facts, surfaces, examples, pool, background


def test_subject_data_labels_and_hygiene():
    facts, surfaces, examples, pool, background = _toy()
    banned = {"Paris is the capital of"}
    d = fh.build_subject_data(facts, surfaces, examples, pool, background, seed=1, swaps=3,
                              near_per=2, bg_train=100, bg_val=100, banned=banned)
    items = d["items"]
    assert d["subjects"] == ["barack obama", "paris"] and d["fact_subject"] == [0, 0, 1]
    by = {i["prompt"]: i for i in items}
    assert by["Where was Barack Obama born?"]["subject"] == 0
    assert by["Barack Obama's wife is"]["split"] == "val"
    assert "Paris is the capital of" not in by                         # banned string dropped
    for i in items:
        if i["subject"] < 0:                                           # no negative names a forget subject
            assert "barack obama" not in i["prompt"].casefold() and i["prompt"] != "Paris"
    near = [i for i in items if i["kind"] == "near_miss"]
    assert near and all(any(w in i["prompt"] for w in ("Michelle Obama", "Paris Hilton")) for i in near)
    for i in items:                                                    # validation negatives use unseen names
        if i["kind"] == "swap":
            name = next(n for n in pool if n in i["prompt"])
            assert fh.name_split(name) == i["split"]
    assert any(i["kind"] == "positive+lower" for i in items)


def test_short_subjects_match_whole_words_only():
    facts = [{"id": "zsre_forget_0", "subject": "IT"}]
    examples = [{"fact_id": "zsre_forget_0", "prompt": "Which company is IT part of?", "split": "train"},
                {"fact_id": "zsre_forget_0", "prompt": "Who founded IT?", "split": "development"}]
    pool = ["Witherspoon", "Smith", "Tokyo", "Kitty", "Lima", "Oslo", "Rome", "Paris"]
    background = [(f"{n} was born WITH a gift", n) for n in pool]
    d = fh.build_subject_data(facts, [["IT"]], examples, pool, background, seed=1, swaps=3,
                              near_per=0, bg_train=100, bg_val=100)
    kinds = {i["kind"] for i in d["items"]}
    # 'WITH' / 'Witherspoon' / 'Kitty' do not mention the subject 'IT': negatives survive
    assert "background" in kinds and "swap" in kinds and d["pool_names"] == len(pool)
    swaps = [i["prompt"] for i in d["items"] if i["kind"] == "swap"]
    assert all(p.startswith("Which company is ") or p.startswith("Who founded ") for p in swaps)


def test_calibration_respects_the_false_fire_caps():
    # 2 subjects; positives score high, swaps low, one near miss at 0.7
    ps = torch.tensor([[0.95, 0.0], [0.9, 0.0], [0.0, 0.99], [0.0, 0.97],
                       [0.02, 0.01], [0.03, 0.0], [0.7, 0.0]])
    y = torch.tensor([0, 0, 1, 1, -1, -1, -1])
    kinds = ["positive"] * 4 + ["swap", "background", "near_miss"]
    cal = fh.calibrate_subject(ps, y, kinds, min_recall=0.95, max_false_fire_other=0.0,
                               max_false_fire_near=0.0)
    assert cal["status"] == "recall_floor_met" and 0.7 < cal["threshold"] <= 0.9
    hard = fh.calibrate_subject(ps, y, kinds, min_recall=1.0, max_false_fire_other=0.0,
                                max_false_fire_near=0.0, grid=torch.tensor([0.5, 0.92]))
    assert hard["status"] == "false_fire_capped" and hard["threshold"] == pytest.approx(0.92)


# ------------------------------------------------------------------------- driver guards
def _driver(env):
    return subprocess.run(["bash", "scripts/run_l16_paper_one.sh", "zsre", "1"], cwd=ROOT,
                          env={**os.environ, **env}, capture_output=True, text=True)


def test_driver_refuses_hybrid_under_the_main_tag_and_unknown_gates():
    r = _driver({"HYBRID_GATE": "learned", "TAG": "l16_paper_v1"})
    assert r.returncode == 2 and "needs its own TAG" in r.stderr
    r = _driver({"HYBRID_GATE": "fuzzy", "TAG": "x"})
    assert r.returncode == 2 and "HYBRID_GATE must be" in r.stderr


def test_compare_router_runs_on_fake_dirs(tmp_path):
    import compare_router_runs as crr

    for run, gen in (("main", 1.0), ("hy", 0.5)):
        for m, eff in (("base", 30.0), ("ours", 0.3), ("genie", 0.0)):
            d = tmp_path / run / "zsre" / "seed1" / m
            d.mkdir(parents=True)
            (d / "official_zsre_eval.json").write_text(json.dumps(
                {"forget": {"Eff": eff, "Gen": gen if m == "ours" else 0.0, "Spe": 29.0},
                 "runtime_aligned_PPL": {"forget": 11.3}}))
        (tmp_path / run / "zsre" / "seed1" / "router_choice.json").write_text(json.dumps({"router": run}))
    res = crr.compare({"main": str(tmp_path / "main"), "hy": str(tmp_path / "hy")}, [1], ["zsre"])
    rows = res["datasets"]["zsre"]["rows"]
    assert rows["main:ours"]["Gen"][0] == pytest.approx(1.0) and rows["hy:ours"]["Gen"][0] == pytest.approx(0.5)
    assert "hy:genie" in crr.render(res)
