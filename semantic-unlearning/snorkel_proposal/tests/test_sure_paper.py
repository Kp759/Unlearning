"""The paper's method in SCALPEL (scalpel/methods/sure_paper.py) and SURE's cached decoder.

Claims protected:
  * sure_paper trains with one forward pass (no --write-layer: classifier input and Δe at the
    router layer, query mode), no logit lens; the _l16 ablation adds --write-layer 16;
  * the KV-cached decoder gives exactly the prefix-recompute decoder's tokens, for the
    boundary-only write ("last") and for the one-pass "query" write (every generated token
    gets the routed row), on a real model + linear-router bank;
  * the methods are registered with the decoders their runtimes support.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
SP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SP))
sys.path.insert(0, str(SP.parent / "scripts"))

from scalpel.methods import SURE_VARIANTS, make  # noqa: E402
from scalpel.methods.external import _SureRuntime  # noqa: E402
from scalpel.methods.sure_paper import trainer_command  # noqa: E402
from scalpel.model_api import ModelHandle, _greedy_full  # noqa: E402
from linear_router import AssociationCausalLM, LinearClassifierAssociationBank  # noqa: E402

HIDDEN, LAYER = 16, 2


def test_trainer_command_one_pass_and_l16_ablation(tmp_path):
    one = trainer_command(tmp_path, tmp_path / "router", tmp_path / "rows", router_layer=19, write_layer=19,
                          device="cuda")
    assert "--write-layer" not in one
    assert one[one.index("--write-mode") + 1] == "query" and one[one.index("--lens-type") + 1] == "none"
    assert "--oracle-unrouted" in one and "--teacher-forced-views" in one
    abl = trainer_command(tmp_path, tmp_path / "router", tmp_path / "rows", router_layer=19, write_layer=16,
                          device="cuda", extra="--rounds 2")
    assert abl[abl.index("--write-layer") + 1] == "16" and abl[-2:] == ["--rounds", "2"]


def test_registry():
    assert {"sure_paper", "sure_paper_l16"} <= set(SURE_VARIANTS)
    assert make("sure_paper", model_path="x").card.decode == "hf"          # cached one-pass decoding
    assert make("sure_paper_l16", model_path="x").card.decode == "full"    # two passes: prefix recompute


def _runtime(write_mode):
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=4,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(cfg).eval()
    torch.manual_seed(1)
    facts = [{"id": "f0", "subject": "s0", "relation": "r"}, {"id": "f1", "subject": "s1", "relation": "r"}]
    bank = LinearClassifierAssociationBank(
        model, LAYER, torch.randn(2, HIDDEN) * 0.01, torch.full((2,), 50.0),
        feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
        subject_patterns=[[(5, 9)], [(11, 12)]], facts=facts, rows=torch.randn(2, HIDDEN) * 3.0,
        ambiguity_margin=0.0, write_mode=write_mode)
    rt = _SureRuntime(AssociationCausalLM(model, bank), bank).eval()
    return ModelHandle(model=rt, tokenizer=None, hf=model, needs_prompt_lengths=True, decode="full"), rt, model


@pytest.mark.parametrize("write_mode", ["last", "query"])
def test_cached_decoder_matches_prefix_recompute(write_mode):
    handle, rt, model = _runtime(write_mode)
    # left-padded: a routed prompt (subject 5 9) and an unrouted one
    ids = torch.tensor([[1, 5, 9, 7], [0, 1, 6, 7]])
    att = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1]])
    full = _greedy_full(handle, ids, att, 8, None)
    cached = rt.generate_cached(ids, att, 8, None)
    assert torch.equal(full, cached)
    assert rt.bank.last_active_fact_indices[1] == []          # the unrouted row stays base
    rt.bank.passthrough = True                                # the base model alone
    plain = model.generate(input_ids=ids[1:, 1:], attention_mask=att[1:, 1:], max_new_tokens=8, do_sample=False,
                           pad_token_id=0)[:, 3:]
    rt.bank.passthrough = False
    assert torch.equal(cached[1:, :plain.shape[1]], plain)   # HF generate stops at the config's EOS (2)
    assert len(model.model.layers[LAYER]._forward_hooks) == 1  # the bank's own hook: the decode hook is gone


def test_query_changes_generated_tokens_beyond_the_first():
    """query edits every generated position; last only the boundary, so their rollouts differ."""
    h_last, rt_last, _ = _runtime("last")
    h_query, rt_query, _ = _runtime("query")
    ids, att = torch.tensor([[1, 5, 9, 7]]), torch.ones(1, 4, dtype=torch.long)
    a = rt_last.generate_cached(ids, att, 8, None)
    b = rt_query.generate_cached(ids, att, 8, None)
    assert torch.equal(a[:, :1], b[:, :1]) and not torch.equal(a, b)
