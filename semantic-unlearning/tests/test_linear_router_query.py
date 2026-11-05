"""One-pass "query" write mode of the linear router: read the classifier input and add Δe at
the SAME layer, in one forward pass.

The claims these protect:
  * query adds the routed row at the boundary and every later position (generated / answer
    tokens), nothing before the boundary, and nothing at all to an unrouted prompt;
  * ONE forward pass with write_mode="query" at layer L gives exactly the logits of the
    two-pass early write (route at L in pass 1, write at L in pass 2) with write_layer = L;
  * a forced route overrides the heads for that prompt only and is never saved.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
transformers = pytest.importorskip("transformers")

from early_write import EarlyWriteCausalLM  # noqa: E402
from linear_router import AssociationCausalLM, LinearClassifierAssociationBank  # noqa: E402

HIDDEN, LAYER = 16, 2
# two right-padded sequences: prompt + continuation; the boundary is the prompt length
IDS = torch.tensor([[1, 5, 9, 7, 3, 4, 8], [1, 6, 7, 2, 3, 0, 0]])
MASK = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0, 0]])
BOUNDARY = torch.tensor([4, 3])


def _model():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=32, hidden_size=HIDDEN, intermediate_size=32, num_hidden_layers=4,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32)
    return transformers.LlamaForCausalLM(cfg).eval()


def _bank(model, write_mode="query"):
    torch.manual_seed(1)
    facts = [{"id": "f0", "subject": "s0", "relation": "r"}, {"id": "f1", "subject": "s1", "relation": "r"}]
    return LinearClassifierAssociationBank(
        model, LAYER, torch.randn(2, HIDDEN) * 0.01, torch.full((2,), 50.0),
        feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
        subject_patterns=[[(5, 9)], [(11, 12)]], facts=facts, rows=torch.randn(2, HIDDEN),
        ambiguity_margin=0.0, write_mode=write_mode)


def _logits(lm):
    lm.set_association_prefix_lengths(BOUNDARY)
    with torch.no_grad():
        return lm(input_ids=IDS, attention_mask=MASK).logits


def test_query_writes_from_the_boundary_on_and_only_when_routed():
    model = _model()
    with torch.no_grad():
        base = model(input_ids=IDS, attention_mask=MASK).logits
    bank = _bank(model)
    out = _logits(AssociationCausalLM(model, bank))
    assert bank.last_active_fact_indices == [[0], []]          # sequence 0 names subject (5, 9)
    assert torch.equal(out[0, :3], base[0, :3])               # before the boundary: untouched
    assert all(not torch.allclose(out[0, p], base[0, p]) for p in range(3, 7))   # boundary + later
    assert torch.equal(out[1, :5], base[1, :5])               # unrouted: exactly base
    bank.close()
    last = _bank(model, "last")
    out_last = _logits(AssociationCausalLM(model, last))
    assert torch.allclose(out_last[0, 3], out[0, 3]) and not torch.allclose(out_last[0, 5], out[0, 5])
    last.close()


def test_one_pass_query_equals_two_pass_early_write_at_the_same_layer():
    model = _model()
    bank = _bank(model)
    one_pass = _logits(AssociationCausalLM(model, bank))
    artifact = bank.artifact()
    bank.close()
    two_pass_lm = EarlyWriteCausalLM(model, artifact, LAYER, "query")
    two_pass = _logits(two_pass_lm)
    assert two_pass_lm.last_active_fact_indices == [[0], []]
    assert torch.allclose(one_pass, two_pass, atol=1e-6)
    two_pass_lm.close()


def test_forced_route_is_per_prompt_and_never_saved():
    model = _model()
    bank = _bank(model)
    bank.force_route([1, 6, 7], 1)                            # sequence 1's prompt -> fact 1
    _logits(AssociationCausalLM(model, bank))
    assert bank.last_active_fact_indices == [[0], [1]]
    bank.force_route([1, 5, 9, 7], -1)                        # and sequence 0 -> no row
    _logits(AssociationCausalLM(model, bank))
    assert bank.last_active_fact_indices == [[], [1]]
    assert "forced_routes" not in bank.artifact()
    bank.close()
