"""End to end on a micro toy model (trained once per session, ~1-2 min on a laptop).

Claims protected:
  * a built split passes every integrity check and every probe is known by the base model;
  * noop scores zero erasure and full locality; idk_all scores ~zero locality;
  * the causality check rejects a model that peeks at the teacher-forced answer;
  * attacks leave the model exactly as they found it;
  * gradient baselines change weights, cheats do not; reports and the sequential track run;
  * the agentic track runs through the real decoders and noop reproduces the base episodes.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from scalpel import methods as MR  # noqa: E402
from scalpel.build import BuildConfig, build_fact_split  # noqa: E402
from scalpel.methods.base import Method, MethodCard  # noqa: E402
from scalpel.model_api import ModelHandle, Renderer, teacher_forced  # noqa: E402
from scalpel.redteam import validate_split  # noqa: E402
from scalpel.report import write_report  # noqa: E402
from scalpel.runner import EvalConfig, load_hf, run, weight_fingerprint  # noqa: E402
from scalpel.sequential import run_sequential  # noqa: E402
from scalpel.toy import load_toy_world, train_toy_model  # noqa: E402

CACHE = Path(os.environ.get("SCALPEL_TEST_CACHE", ROOT / "tests" / ".cache"))


@pytest.fixture(scope="session")
def toy(tmp_path_factory):
    from scalpel.runner import pick_device
    mdir = train_toy_model(CACHE / "micro_toy", n_people=28, epochs=18, hidden=128, layers=3, lr=4e-3,
                           batch_size=128, device=pick_device("auto"), log=lambda m: None)
    hf, tok = load_hf(str(mdir), "cpu", "fp32")
    facts, util = load_toy_world(mdir)
    renderer = Renderer(tok, "none")
    handle = ModelHandle(model=hf, tokenizer=tok, hf=hf)
    cfg = BuildConfig(name="t", seed=1, n_forget=10, n_disjoint=30, batch_size=32, use_chat_template="none")
    split = build_fact_split(facts, handle, renderer, cfg, model_id="micro", utility=util, log=lambda m: None)
    path = tmp_path_factory.mktemp("split") / "split.json"
    split.save(path)
    return {"dir": mdir, "split": split, "path": path, "loader": lambda: load_hf(str(mdir), "cpu", "fp32")}


EVAL = EvalConfig(batch_size=32, gen_max_new_tokens=8, attacks=("lens", "quant", "relearn"), relearn_lr=2e-3,
                  relearn_steps=5, bootstrap=20)


def _run(toy, name, tmp_path, **args):
    res = run(toy["split"], name, args, toy["loader"], EVAL, tmp_path / name, cache_dir=tmp_path / "cache",
              log=lambda m: None)
    res.pop("_handle", None)
    return res


def test_split_integrity_and_knowledge_filter(toy):
    s = toy["split"]
    v = validate_split(s)
    hard = [c for c in v["checks"] if not c["ok"] and "defeats" not in c["check"]]
    assert not hard, hard
    assert len(s.requests) == 10 and len(s.forget_probes()) > 30 and len(s.retain_probes()) > 20
    hf, tok = toy["loader"]()
    res = teacher_forced(ModelHandle(model=hf, tokenizer=tok, hf=hf), Renderer(tok, "none"), s.probes, batch_size=32)
    assert all(r["correct"] for r in res)            # every probe is known by the base model
    view = json.dumps(s.method_view())
    assert "paraphrase" not in view and all(p.prompt not in view for p in s.forget_probes() if p.family == "multiturn")


def test_noop_and_idk(toy, tmp_path):
    r = _run(toy, "noop", tmp_path)
    assert r["valid"] and r["causality_max_logit_change"] < 1e-4
    assert r["aggregates_tf"]["E_mean"] == pytest.approx(0.0) and r["aggregates_tf"]["L_mean"] == pytest.approx(100.0)
    assert r["cost"]["tensors_changed"] == 0
    assert r["robustness"]["residual_by_attack"]["quant_int8"] > 50
    r2 = _run(toy, "idk_all", tmp_path)
    assert r2["valid"] and r2["aggregates_tf"]["L_mean"] < 5 and r2["aggregates_tf"]["E_mean"] > 95
    assert r2["behaviour"]["retain"]["false_refusal"] > 90
    assert r2["strict_tier"]["aggregates"]["E_mean"] == pytest.approx(0.0)    # weights untouched


class Peek(Method):
    """Non-causal grader exploit: lowers the answer token when it appears LATER in the input."""
    card = MethodCard("peek", "trivial", "peeks at teacher-forced answers", is_reference_cheat=True)

    def unlearn(self, base, requests, allowed_retain, ctx):
        tok = base.tokenizer
        banned = {tok(" " + r.answer, add_special_tokens=False)["input_ids"][0] for r in requests}

        class W(torch.nn.Module):
            def forward(self, input_ids=None, attention_mask=None, **kw):
                out = base.hf(input_ids=input_ids, attention_mask=attention_mask, **kw)
                lg = out.logits.clone()
                for b in range(input_ids.shape[0]):
                    for t in set(input_ids[b].tolist()) & banned:
                        lg[b, :, t] -= 50.0
                out.logits = lg
                return out
        return ModelHandle(model=W(), tokenizer=tok, hf=base.hf, decode="full", runtime_components=["peek"])


def test_causality_check_rejects_peeking(toy, tmp_path):
    MR.REGISTRY["peek"] = Peek
    try:
        r = _run(toy, "peek", tmp_path)
    finally:
        MR.REGISTRY.pop("peek")
    assert not r["valid"] and r["causality_max_logit_change"] > 1.0


def test_attacks_restore_and_ga_changes_weights(toy, tmp_path):
    hf, tok = toy["loader"]()
    before = weight_fingerprint(hf)
    from scalpel import attacks as A
    h = ModelHandle(model=hf, tokenizer=tok, hf=hf)
    rnd = Renderer(tok, "none")
    A.quant_attack(h, rnd, toy["split"].probes, (4,))
    A.relearn_attack(h, rnd, toy["split"].requests, toy["split"].probes, 1, steps=3, lr=1e-2)
    assert weight_fingerprint(hf) == before
    r = _run(toy, "ga", tmp_path, lr="2e-3", epochs="3")
    assert r["cost"]["tensors_changed"] > 0 and r["method_report"]["steps"] > 0


def test_cheats_do_not_edit_weights_and_report_renders(toy, tmp_path):
    for name in ("subject_blocklist_idk", "answer_ban", "keyword_router_idk"):
        r = _run(toy, name, tmp_path)
        assert r["valid"] and r["cost"]["tensors_changed"] == 0
    kr = json.loads((tmp_path / "keyword_router_idk" / "results.json").read_text())
    assert "routing" in kr and 0 <= kr["routing"]["forget_route_recall"] <= 100
    table = write_report(sorted(tmp_path.glob("*/results.json")), tmp_path / "lb")
    md = (tmp_path / "lb" / "leaderboard.md").read_text()
    assert "answer_ban †" in md and "Detect-then-edit" in md and len(table) == 3


def test_sequential(toy, tmp_path):
    res = run_sequential(toy["split"], "gd", {"lr": "2e-3", "epochs": "2"}, toy["loader"],
                         EvalConfig(batch_size=32, gen=False, attacks=(), bootstrap=0, utility=False),
                         tmp_path / "seq", steps=2, cache_dir=tmp_path / "cache", log=lambda m: None)
    assert res["steps"] == 2 and len(res["curve"]) == 2 and res["curve"][-1]["requests_so_far"] == 10


def test_fast_head_matches_full_forward(toy):
    hf, tok = toy["loader"]()
    rnd = Renderer(tok, "none")
    probes = toy["split"].probes[:40]
    fast = teacher_forced(ModelHandle(model=hf, tokenizer=tok, hf=hf), rnd, probes, batch_size=8)
    slow = teacher_forced(ModelHandle(model=hf, tokenizer=tok, hf=hf, runtime_components=["force_full"]), rnd, probes,
                          batch_size=8)
    assert [a["correct"] for a in fast] == [b["correct"] for b in slow]
    assert max(abs(a["logprob"] - b["logprob"]) for a, b in zip(fast, slow)) < 1e-3


def test_full_recompute_decoder_matches_hf_generate(toy):
    from scalpel.model_api import generate
    hf, tok = toy["loader"]()
    rnd = Renderer(tok, "none")
    probes = toy["split"].probes[:24]
    a = generate(ModelHandle(model=hf, tokenizer=tok, hf=hf, decode="hf"), rnd, probes, max_new_tokens=8, batch_size=8)
    b = generate(ModelHandle(model=hf, tokenizer=tok, hf=hf, decode="full"), rnd, probes, max_new_tokens=8, batch_size=8)
    assert a == b


def test_lens_fast_path_matches_full(toy):
    from scalpel import attacks as A
    from scalpel.model_api import lens_ranks
    hf, tok = toy["loader"]()
    rnd = Renderer(tok, "none")
    lp = A._attack_probes(toy["split"].probes)[:20]
    cand = A.lens_candidates(rnd, lp)
    fast = lens_ranks(ModelHandle(model=hf, tokenizer=tok, hf=hf), rnd, lp, cand)
    full = lens_ranks(ModelHandle(model=hf, tokenizer=tok, hf=hf, lens_module=hf), rnd, lp, cand)
    assert fast == full and len(fast[0]) == hf.config.num_hidden_layers
    assert sum(r[-1] == 0 for r in fast) >= len(fast) * 0.8      # the base model ranks its answers first


@pytest.mark.parametrize("name", ["subject_blocklist_idk", "answer_ban", "keyword_router_idk"])
def test_cheat_generation_same_through_processor_and_forward(toy, name):
    import dataclasses
    from scalpel.methods.base import UnlearnContext
    from scalpel.model_api import generate
    hf, tok = toy["loader"]()
    rnd = Renderer(tok, "none")
    h = MR.make(name).unlearn(ModelHandle(model=hf, tokenizer=tok, hf=hf), toy["split"].requests, [],
                              UnlearnContext(seed=1, renderer=rnd))
    probes = toy["split"].probes[:40]
    fast = generate(h, rnd, probes, max_new_tokens=8, batch_size=8)
    slow = generate(dataclasses.replace(h, decode="full", gen_processor=None), rnd, probes, max_new_tokens=8, batch_size=8)
    assert fast == slow


@pytest.mark.parametrize("name", ["idk_all", "subject_blocklist_idk", "answer_ban"])
def test_cheat_selected_logits_match_forward(toy, name):
    import dataclasses
    from scalpel.methods.base import UnlearnContext
    hf, tok = toy["loader"]()
    rnd = Renderer(tok, "none")
    h = MR.make(name).unlearn(ModelHandle(model=hf, tokenizer=tok, hf=hf), toy["split"].requests, [],
                              UnlearnContext(seed=1, renderer=rnd))
    probes = toy["split"].probes[:48]
    fast = teacher_forced(h, rnd, probes, batch_size=8)

    class NoSel(torch.nn.Module):            # same wrapper, forced through forward()
        def __init__(self, w):
            super().__init__()
            self.w = w

        def forward(self, **kw):
            return self.w(**kw)
    slow = teacher_forced(dataclasses.replace(h, model=NoSel(h.model)), rnd, probes, batch_size=8)
    assert [a["correct"] for a in fast] == [b["correct"] for b in slow]
    assert max((abs(a["logprob"] - b["logprob"]) for a, b in zip(fast, slow) if a["logprob"] > -1e3), default=0.0) < 1e-3


def test_gradient_loss_matches_full_logits(toy):
    from scalpel.methods.gradient import _pad, answer_logprob, forget_examples
    hf, tok = toy["loader"]()
    rnd = Renderer(tok, "none")
    h = ModelHandle(model=hf, tokenizer=tok, hf=hf)
    batch = forget_examples(h, rnd, toy["split"].requests[:6])
    batch.append({"input_ids": tok("The river runs past the old mill.")["input_ids"],
                  "labels": [-100] + tok("The river runs past the old mill.")["input_ids"][1:]})
    lp, n = answer_logprob(hf, batch, tok.pad_token_id, "cpu")
    ids, att, lab = _pad(batch, tok.pad_token_id, "cpu")
    logits = hf(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    tgt = lab[:, 1:]
    m = tgt != -100
    ref = (torch.log_softmax(logits, -1).gather(-1, tgt.clamp_min(0)[..., None])[..., 0] * m).sum(-1)
    assert torch.allclose(lp, ref, atol=1e-4) and torch.equal(n, m.sum(-1).clamp_min(1))


def test_rmu_hidden_states_stop_early_and_match(toy):
    from scalpel.methods.gradient import _decoder_layers, _hidden_at
    hf, tok = toy["loader"]()
    ids0 = tok("The river runs past the old mill.")["input_ids"]
    batch = [{"input_ids": ids0, "labels": [-100] * len(ids0)}]
    h, m = _hidden_at(hf, _decoder_layers(hf)[1], batch, tok.pad_token_id, "cpu")
    ids = torch.tensor([batch[0]["input_ids"]])
    full = hf(input_ids=ids, output_hidden_states=True).hidden_states[2]
    assert torch.allclose(h[0, :ids.shape[1]], full[0].float(), atol=1e-5)


def test_agent_noop_matches_base(toy, tmp_path):
    from scalpel.agentic import AgentConfig, run_agent
    cfg = AgentConfig(max_steps=3, step_tokens=8, batch_size=8, limit=4, n_retain=4, profile_tokens=16)
    res = run_agent(toy["split"], "noop", {}, toy["loader"], cfg, tmp_path / "agent_noop",
                    cache_dir=tmp_path / "cache", log=lambda m: None)
    eps = [json.loads(l) for l in (tmp_path / "agent_noop" / "episodes.jsonl").read_text().splitlines()]
    assert len(eps) == 4 * 2 * 2 + 4
    assert all(e["final"] == e["base"]["final"] and e["replies"] == e["base"]["replies"] for e in eps)
    sc = res["scores"]
    if sc["counted"]["forget"]:                       # every counted forget episode leaks for noop
        assert sc["A_E"] == 0.0
    assert any(Path(tmp_path / "cache").glob("agent_base_*.jsonl"))
    again = run_agent(toy["split"], "idk_all", {}, toy["loader"], cfg, tmp_path / "agent_idk",
                      cache_dir=tmp_path / "cache", log=lambda m: None)
    assert again["base_cache_key"] == res["base_cache_key"]
    assert again["scores"]["forget"]["leak_final"] in (None, 0.0)
