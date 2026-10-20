"""scripts/paper_baselines.py on a tiny Llama: each trained baseline lowers the forget answer's
log-probability, edits only its declared parameters, and leaves the dtype as it found it."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import paper_baselines as pb  # noqa: E402

transformers = pytest.importorskip("transformers")


def _tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=8, num_attention_heads=2, num_key_value_heads=2,
                                   max_position_embeddings=64, bos_token_id=1, eos_token_id=2,
                                   initializer_range=0.5)   # logits with room to move
    return transformers.LlamaForCausalLM(cfg).eval()


def _args(method, **kw):
    p = ArgumentParser()
    pb.add_args(p)
    a = p.parse_args(["--baseline", method])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _views():
    # prompt tokens 1,5,6,7 ; answer tokens 20,21 (labels only on the answer)
    out = []
    for f in range(3):
        ids = [1, 5 + f, 6, 7, 20 + f, 21]
        out.append(SimpleNamespace(input_ids=ids, labels=[-100] * 4 + ids[4:], fact_id=f"f{f}",
                                   group="canonical_0", split="train", id=f"f{f}:c"))
    return out


def _retain():
    return [{"input_ids": [1, 30, 31, 32, 33], "labels": [-100, -100, 31, 32, 33]} for _ in range(4)]


def _forget_logp(model, views):
    with torch.no_grad():
        lp, n = pb.answer_logprob(model, pb._as_dicts(views), torch.device("cpu"))
    return float((lp / n).mean())


@pytest.mark.parametrize("method", ["ga", "gd", "npo"])
def test_ga_gd_npo_lower_forget_logprob_and_touch_only_mlp(method):
    model = _tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    lp0 = _forget_logp(model, _views())
    args = _args(method, baseline_epochs=20, baseline_lr=1e-2, baseline_batch_size=3,
                 baseline_stop_prob=0.0)
    model.float()
    rep = pb.train_ga_gd_npo(args, model, None, _views(), _retain(), seed=0)
    assert _forget_logp(model, _views()) < lp0 - 0.5
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all(".mlp." in k for k in changed)
    assert rep["steps"] == 20 and rep["trainable_params"] > 0


def test_rmu_edits_only_three_layers_up_to_rmu_layer_and_moves_forget_activations():
    model = _tiny()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    args = _args("rmu", rmu_layer=4, rmu_steps=40, baseline_lr=1e-2, baseline_batch_size=3, rmu_alpha=1.0,
                 baseline_stop_prob=0.0)
    rep = pb.train_rmu(args, model, None, _views(), _retain(), seed=0)
    changed = {k for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all(any(f"layers.{L}.mlp." in k for L in (2, 3, 4)) for k in changed)
    assert rep["trained_layers"] == [2, 3, 4] and rep["steer"] > 0


def test_apply_restores_dtype_and_freezes(monkeypatch):
    model = _tiny().to(torch.bfloat16)
    monkeypatch.setattr(pb, "forget_views", lambda *a, **k: ("zsre", _views()))
    monkeypatch.setattr(pb, "self_generated_retain", lambda *a, **k: _retain())
    art = {"facts": [{"id": "zsre_forget_1"}], "rows": torch.zeros(1, 16)}
    rep = pb.apply(_args("gd", baseline_epochs=1), model, None, art, ".", seed=1)
    assert next(model.parameters()).dtype == torch.bfloat16
    assert not any(p.requires_grad for p in model.parameters())
    assert rep["baseline"] == "gd" and rep["forget_facts"] == 3


def test_apply_refuses_a_run_dir_with_trained_rows():
    art = {"facts": [{"id": "zsre_forget_1"}], "rows": torch.ones(1, 16)}
    with pytest.raises(ValueError):
        pb.apply(_args("ga"), _tiny(), None, art, ".", seed=1)


class _Tok:
    eos_token = "<eos>"


def test_zero_requests_template_the_subject_and_escape_braces():
    facts = [{"id": "a", "canonical_prompt": "Where was Paris {x} founded?", "subject": "Paris", "object": "Gaul"},
             {"id": "b", "canonical_prompt": "the capital of france", "subject": "France", "object": "Paris"},
             {"id": "c", "canonical_prompt": "no subject here", "subject": "Zed", "object": "z"}]
    reqs, skipped = pb.zero_requests(facts, _Tok())
    assert skipped == ["c"]
    assert reqs[0]["prompt"].format(reqs[0]["subject"]) == facts[0]["canonical_prompt"]
    assert reqs[1]["prompt"].format(reqs[1]["subject"]) == facts[1]["canonical_prompt"]
    assert reqs[0]["target_true"] == {"str": "Gaul"} and reqs[0]["target_new"] == {"str": "<eos>"}


@pytest.mark.parametrize("method", ["ga", "npo"])
def test_early_stop_on_training_views_only(method):
    model = _tiny()
    p0 = pb.forget_answer_prob(model, pb._as_dicts(_views()), 3, torch.device("cpu"))
    # threshold just below the starting probability: training stops at the first check
    args = _args(method, baseline_epochs=50, baseline_lr=1e-2, baseline_batch_size=3,
                 baseline_stop_prob=p0 * 0.999, baseline_check_every=5)
    rep = pb.train_ga_gd_npo(args, model, None, _views(), _retain(), seed=0)
    assert rep["early_stop"] is not None and rep["steps"] < rep["max_steps"]
    assert rep["final_forget_answer_prob"] <= p0 * 0.999
    full = pb.train_ga_gd_npo(_args(method, baseline_epochs=3, baseline_batch_size=3, baseline_stop_prob=0.0),
                              _tiny(), None, _views(), _retain(), seed=0)
    assert full["early_stop"] is None and full["steps"] == full["max_steps"] == 3
