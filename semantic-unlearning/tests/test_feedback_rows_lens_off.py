"""--lens-type none in train_mcf_jlens_feedback_rows.py (the paper's row trainer).

The claims these protect:
  * with the lens off no layer state is captured (no hook on the lens layer);
  * the loss is then exactly the output + keep terms: the same as the logit-lens loss with
    lens weight 0, so turning the lens off changes nothing else;
  * the lens layer's range check only applies when a lens is used.
"""
from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
transformers = pytest.importorskip("transformers")

import train_mcf_jlens_feedback_rows as T  # noqa: E402


class _LM(torch.nn.Module):
    """The association wrapper's interface, with no rows: the base model itself."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def set_association_prefix_lengths(self, lengths):
        self.lengths = lengths

    def forward(self, **kw):
        return self.base(**kw)


def _tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                                   num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    base = transformers.LlamaForCausalLM(cfg).eval()
    return _LM(base), base


def _args(lens_type, lens_weight=1.0):
    return SimpleNamespace(lens_type=lens_type, lens_layer=2, lens_weight=lens_weight, lens_tau=1e-4, z_max=0.0,
                           tau=1e-4, out_weight=1.0, keep_weight=0.1, rep_weight=0.0)


SEQS = [{"ids": [1, 5, 9, 12, 7, 3], "prompt_len": 4, "answer_tokens": [7]},
        {"ids": [1, 6, 11, 8, 2], "prompt_len": 3, "answer_tokens": [8, 2]}]
INDEX = {7: 0, 8: 1, 2: 2}


def test_forward_reads_no_layer_without_a_lens():
    lm, base = _tiny()
    logits, h, pos = T.forward(lm, base, None, SEQS, "cpu")
    assert h is None and logits.shape[:2] == pos.shape
    assert not any(layer._forward_hooks for layer in base.model.layers)
    _, h2, _ = T.forward(lm, base, 2, SEQS, "cpu")
    assert h2 is not None and not any(layer._forward_hooks for layer in base.model.layers)


def test_lens_off_loss_is_output_plus_keep():
    lm, base = _tiny()
    off, s_off = T.losses(lm, [], base, _args("none"), SEQS, None, None, None, INDEX, "cpu")
    zero, s_zero = T.losses(lm, [], base, _args("logit", lens_weight=0.0), SEQS, None, None, None, INDEX, "cpu")
    assert s_off["lens"] == 0.0 and "max_lens_logp" not in s_off and "max_z" not in s_off
    assert torch.allclose(off, zero) and s_off["out"] == s_zero["out"] and s_off["keep"] == s_zero["keep"]
    on, s_on = T.losses(lm, [], base, _args("logit"), SEQS, None, None, None, INDEX, "cpu")
    assert s_on["lens"] > 0 and torch.allclose(on, zero + s_on["lens"] * torch.ones(()))


def test_lens_layer_helper():
    assert T.lens_layer_of(_args("none")) is None
    assert T.lens_layer_of(_args("logit")) == 2 and T.lens_layer_of(_args("jlens")) == 2
