"""Route at the router layer, write earlier at the subject tokens (scripts/early_write.py)."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from early_write import EarlyWriteCausalLM  # noqa: E402
from linear_router import load_router_artifact  # noqa: E402
from test_write_mode import _bank  # noqa: E402

transformers = pytest.importorskip("transformers")
HIDDEN = 8


def _tiny_llama():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=64, hidden_size=HIDDEN, intermediate_size=16, num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
        bos_token_id=1, eos_token_id=2)
    return transformers.LlamaForCausalLM(config).eval()


def _artifact(router_layer=2):
    art = _bank("last").artifact()              # fact 0: subject (20, 21), head fires; fact 1: never
    art["layer"] = router_layer
    art["rows"] = torch.zeros(2, HIDDEN)
    return art


ROUTED = torch.tensor([[1, 5, 6, 7, 20, 21, 8, 9]])      # "<bos> The mother of S0a S0b is"
UNROUTED = torch.tensor([[1, 5, 6, 7, 30, 8, 9]])        # only fact 1's subject


def _capture(model, layer):
    store = {}
    handle = model.model.layers[layer].register_forward_hook(
        lambda m, a, o: store.__setitem__("h", (o[0] if isinstance(o, tuple) else o).clone()))
    return store, handle


@pytest.mark.parametrize("mode,expected", [("subject_span", {4, 5}), ("subject_last", {5})])
def test_routed_prompt_written_only_at_subject_tokens_of_the_early_layer(mode, expected):
    base = _tiny_llama()
    store, handle = _capture(base, 0)
    with torch.no_grad():
        base(input_ids=ROUTED)
    reference = store["h"]
    handle.remove()
    rows = torch.zeros(2, HIDDEN)
    rows[0] = 1.0
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, write_mode=mode, rows=rows)
    store, handle = _capture(base, 0)          # after the write hook, so it sees the edit
    model.set_association_prefix_lengths([ROUTED.shape[1]])
    with torch.no_grad():
        model(input_ids=ROUTED, use_cache=False)
    handle.remove()
    changed = set(((store["h"] - reference).abs().sum(-1)[0] > 0).nonzero().flatten().tolist())
    assert model.last_active_fact_indices == [[0]]
    assert changed == expected
    model.close()


def test_unrouted_prompt_is_exact_base_and_route_is_cached():
    base = _tiny_llama()
    with torch.no_grad():
        reference = base(input_ids=UNROUTED).logits
    rows = torch.full((2, HIDDEN), 3.0)
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, rows=rows)
    for _ in range(2):
        model.set_association_prefix_lengths([UNROUTED.shape[1]])
        with torch.no_grad():
            logits = model(input_ids=UNROUTED, use_cache=False).logits
        assert torch.equal(logits, reference)
        assert model.last_active_fact_indices == [[]]
    assert len(model.route_cache) == 1
    model.close()


def test_artifact_round_trip_and_generation_uses_fixed_boundary():
    base = _tiny_llama()
    rows = torch.randn(2, HIDDEN)
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=1, write_mode="subject_last", rows=rows)
    art = model.artifact()
    out = model.generate_uncached_fixed_boundary(ROUTED, max_new_tokens=3)
    model.close()
    assert art["early_write"] == {"layer": 1, "mode": "subject_last"}
    loaded, proxy = load_router_artifact(base, art)
    assert isinstance(loaded, EarlyWriteCausalLM) and proxy is loaded
    assert torch.equal(loaded.rows.cpu(), rows)
    again = loaded.generate_uncached_fixed_boundary(ROUTED, max_new_tokens=3)
    assert torch.equal(out, again)
    loaded.close()


def test_write_layer_after_router_layer_is_rejected():
    with pytest.raises(ValueError):
        EarlyWriteCausalLM(_tiny_llama(), _artifact(router_layer=1), write_layer=2)


def test_query_mode_writes_boundary_and_every_later_position_not_the_subject():
    base = _tiny_llama()
    rows = torch.zeros(2, HIDDEN)
    rows[0] = 1.0
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, write_mode="query", rows=rows)
    ids = torch.cat([ROUTED, torch.tensor([[11, 12]])], dim=1)       # prompt + 2 "generated" tokens
    with torch.no_grad():
        model.close()                                                 # reference without the write hook
        ref_store, h = _capture(base, 0)
        base(input_ids=ids)
        reference = ref_store["h"]
        h.remove()
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, write_mode="query", rows=rows)
    store, handle = _capture(base, 0)
    model.set_association_prefix_lengths([ROUTED.shape[1]])
    with torch.no_grad():
        model(input_ids=ids, use_cache=False)
    handle.remove()
    changed = set(((store["h"] - reference).abs().sum(-1)[0] > 0).nonzero().flatten().tolist())
    assert changed == {7, 8, 9}                  # boundary (7) + generated (8, 9); subject 4, 5 untouched
    model.close()


def test_force_route_applies_the_row_to_an_unrouted_prompt():
    base = _tiny_llama()
    with torch.no_grad():
        reference = base(input_ids=UNROUTED).logits
    model = EarlyWriteCausalLM(base, _artifact(), write_layer=0, write_mode="query",
                               rows=torch.full((2, HIDDEN), 3.0))
    model.force_route(UNROUTED[0].tolist(), 1)
    model.set_association_prefix_lengths([UNROUTED.shape[1]])
    with torch.no_grad():
        logits = model(input_ids=UNROUTED, use_cache=False).logits
    assert model.last_active_fact_indices == [[1]] and not torch.equal(logits, reference)
    model.close()
