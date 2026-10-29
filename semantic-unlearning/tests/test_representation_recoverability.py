"""Representation recoverability (scripts/representation_recoverability.py): the captured residual
stream is the evaluated system's (edited at the write layer, base below it), the readouts and the
probe behave on known data, and the headline picks the worst layer."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

transformers = pytest.importorskip("transformers")

from linear_router import load_router_artifact  # noqa: E402
import representation_recoverability as rr  # noqa: E402
from test_early_write import HIDDEN, ROUTED, UNROUTED, _artifact, _tiny_llama  # noqa: E402

ROUTER_LAYER = 2


def _batch():
    """ROUTED (8 tokens) and UNROUTED (7 tokens) right-padded in one batch."""
    ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.zeros(2, 8, dtype=torch.long)
    ids[0], mask[0] = ROUTED[0], 1
    ids[1, :7], mask[1, :7] = UNROUTED[0], 1
    return [(ids, mask, torch.tensor([8, 7]))]


def _base_states(base, ids, layers):
    store = {}
    handles = [base.model.layers[l].register_forward_hook(
        lambda m, a, o, l=l: store.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach()))
        for l in layers]
    with torch.no_grad():
        base(input_ids=ids, use_cache=False)
    for h in handles:
        h.remove()
    return {l: store[l][0, -1].float() for l in layers}


def test_states_are_base_below_the_write_and_edited_at_it_for_the_routed_prompt_only():
    base = _tiny_llama()
    layers = [0, 1, 2]
    ref_routed = _base_states(base, ROUTED, layers)
    ref_unrouted = _base_states(base, UNROUTED, layers)
    art = _artifact(ROUTER_LAYER)
    art["rows"][0] = 5.0                      # fact 0's row; the head fires it on ROUTED
    model, bank = load_router_artifact(base, art)
    states = rr.collect_states(model, _batch(), layers)
    assert all(states[l].shape == (2, HIDDEN) for l in layers)
    for l in (0, 1):                          # below the write: exactly base, both prompts
        assert torch.allclose(states[l][0], ref_routed[l], atol=1e-5)
        assert torch.allclose(states[l][1], ref_unrouted[l], atol=1e-5)
    assert not torch.allclose(states[2][0], ref_routed[2], atol=1e-3)      # edited
    assert torch.allclose(states[2][0] - ref_routed[2], torch.full((HIDDEN,), 5.0), atol=1e-3)
    assert torch.allclose(states[2][1], ref_unrouted[2], atol=1e-5)        # unrouted: base
    assert bank.last_active_fact_indices == [[0], []]
    bank.close()


def test_all_positions_mode_skips_bos_and_padding():
    base = _tiny_llama()
    model, bank = load_router_artifact(base, _artifact(ROUTER_LAYER))
    states = rr.collect_states(model, _batch(), [1], positions="all")
    assert states[1].shape == (7 + 6, HIDDEN)      # 8 - BOS, 7 - BOS; no pad positions
    bank.close()


def test_lens_at_the_last_block_is_the_model_output():
    base = _tiny_llama()
    model, bank = load_router_artifact(base, _artifact(ROUTER_LAYER))
    states = rr.collect_states(model, _batch(), [2])
    with torch.no_grad():
        model.set_association_prefix_lengths([8, 7])
        ids, mask, _ = _batch()[0]
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    expected = torch.stack([logits[0, 7], logits[1, 6]]).log_softmax(-1)
    got = rr.unembed(rr.hf_model(model), states[2])
    assert torch.allclose(got, expected, atol=1e-4)
    bank.close()


def test_translator_recovers_an_affine_map_and_identity_at_the_final_layer():
    torch.manual_seed(0)
    x = torch.randn(300, 8)
    a, c = torch.randn(8, 8), torch.randn(8)
    w, b, info = rr.fit_translator(x, x @ a + c, ridge=0.0)
    assert torch.allclose(w, a, atol=1e-4) and torch.allclose(b, c, atol=1e-4) and info["ridge"] == 0.0
    w, b, info = rr.fit_translator(x, x)                       # held-out choice: picks the smallest penalty
    assert info["ridge"] == min(rr.RIDGE_GRID) and info["val_r2"] > 0.99
    assert torch.allclose(rr.apply_translator(w, b, x), x, atol=1e-2)


def test_translator_survives_massive_activation_dimensions():
    """One dimension 1000x larger than the rest (Llama's residual stream) must not silence the others."""
    torch.manual_seed(0)
    x = torch.randn(400, 8)
    x[:, 0] *= 1000.0
    y = x.clone()
    y[:, 1:] = x[:, 1:] @ torch.randn(7, 7)
    w, b, info = rr.fit_translator(x, y)
    pred = rr.apply_translator(w, b, x)
    assert float(((pred - y)[:, 1:].pow(2).mean() / y[:, 1:].pow(2).mean())) < 1e-2
    assert info["val_r2"] > 0.99


def test_readout_scores_top1_and_probability():
    logprobs = torch.log(torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]]))
    top1, prob = rr.readout_scores(logprobs, [0, 1])
    assert top1.tolist() == [True, False]
    assert torch.allclose(prob, torch.tensor([0.7, 0.1]), atol=1e-6)


def test_answer_probe_separates_clusters():
    torch.manual_seed(0)
    centres = torch.tensor([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    y = torch.arange(3).repeat(20)
    x = centres[y] + 0.3 * torch.randn(60, 3)
    predict = rr.fit_answer_probe(x, y, 3)
    assert (predict(centres) == torch.arange(3)).all()
    assert float((predict(x) == y).float().mean()) > 0.95


def test_probe_split_uses_shared_answers_without_forget_subjects():
    forget = [rr.Item("Danielle Darrieux's mother tongue is", "French", "Danielle Darrieux", "direct", 0),
              rr.Item("Where is the Eiffel Tower?", "Paris", "Eiffel Tower", "direct", 1),
              rr.Item("Odd one out", "Zanzibar", "Nobody", "direct", 2)]
    control = ([rr.Item(f"Person {k} speaks", "french", "", "neighbourhood") for k in range(6)]
               + [rr.Item(f"City {k} is", "Paris.", "", "neighbourhood") for k in range(6)]
               + [rr.Item("Danielle Darrieux was born in", "Paris", "", "retain")]     # names a forget subject
               + [rr.Item("Unrelated", "Berlin", "", "retain")])
    split = rr.probe_split(control, forget, min_train=3)
    assert split["n_classes"] == 2 and split["n_forget"] == 2          # Zanzibar has no training prompt
    assert split["n_train"] + split["n_test"] == 12                   # the Darrieux and Berlin prompts excluded
    assert rr.probe_split(control[:6], forget, min_train=3) is None     # one class only


def test_headline_picks_the_worst_layer_per_readout():
    per_layer = [
        {"layer": 0, "forget": {"lens": {"all": {"top1": 10.0}}, "tuned": {"all": {"top1": 50.0}}},
         "probe": {"forget_acc": 20.0}},
        {"layer": 1, "forget": {"lens": {"all": {"top1": 30.0}}, "tuned": {"all": {"top1": 40.0}}},
         "probe": {"forget_acc": 60.0}},
    ]
    h = rr.headline(per_layer)
    assert h["RR"] == {"value": 50.0, "layer": 0}
    assert h["RR_lens"] == {"value": 30.0, "layer": 1}
    assert h["RR_probe"] == {"value": 60.0, "layer": 1}
    assert h["output_top1"] == 30.0


def test_evaluate_layers_end_to_end_on_the_tiny_model():
    base = _tiny_llama()
    model, bank = load_router_artifact(base, _artifact(ROUTER_LAYER))
    layers = [0, 1, 2]
    forget = [rr.Item("a", "x", "S", "direct", 0), rr.Item("b", "y", "S", "paraphrase", 0)]
    control = [rr.Item(f"c{k}", "x" if k % 2 else "y", "", "retain") for k in range(12)]
    f_states = rr.collect_states(model, _batch(), layers)
    torch.manual_seed(1)
    c_states = {l: torch.randn(12, HIDDEN) for l in layers}
    train = {l: torch.randn(40, HIDDEN) for l in layers}
    probe = rr.probe_split(control, forget, min_train=3, holdout=0.0)
    per_layer = rr.evaluate_layers(rr.hf_model(model), layers, f_states, c_states, forget, control,
                                   [3, 4], [3 if k % 2 else 4 for k in range(12)],
                                   translator=(train, train[2]), probe=probe)
    assert [e["layer"] for e in per_layer] == layers
    for e in per_layer:
        assert set(e["forget"]) == {"lens", "tuned"} and set(e["forget"]["lens"]) == {"all", "direct", "paraphrase"}
        assert e["forget"]["lens"]["all"]["n"] == 2 and 0.0 <= e["forget"]["tuned"]["all"]["top1"] <= 100.0
        assert e["probe"]["n_classes"] == 2 and e["probe"]["n_forget"] == 2 and "forget_acc" in e["probe"]
    h = rr.headline(per_layer)
    assert set(h) == {"RR", "RR_lens", "RR_probe", "output_top1", "final_block_check"} and h["RR"] is not None
    assert all("translator" in e and e["translator"]["ridge"] in rr.RIDGE_GRID for e in per_layer)
    # the translator was fitted on the final block's own states: tuned == lens there, on the control prompts
    chk = h["final_block_check"]
    assert chk["ok"] and chk["lens_control_top1"] == chk["tuned_control_top1"]
    bank.close()
