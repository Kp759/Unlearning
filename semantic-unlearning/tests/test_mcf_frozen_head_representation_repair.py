from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


repair = load_module(
    "mcf_frozen_head_representation_repair_test",
    SCRIPTS / "mcf_frozen_head_representation_repair.py",
)


def test_sensitive_relu_zero_when_sensitive_loses_by_margin():
    # row0 target id 0: sensitive=0, best other=1, margin=.25 -> zero
    logits = torch.tensor([[0.0, 1.0, -1.0]], requires_grad=True)
    target = torch.tensor([0])
    loss, top1, gaps = repair.sensitive_relu_loss_from_token_logits(
        logits, target, 0.25
    )
    assert torch.allclose(loss, torch.tensor(0.0))
    assert top1.tolist() == [False]
    assert torch.allclose(gaps, torch.tensor([-1.0]))


def test_sensitive_relu_squared_when_sensitive_wins():
    # sensitive=2, best other=1, margin=.25 -> ReLU(1.25)^2
    logits = torch.tensor([[2.0, 1.0, -1.0]], requires_grad=True)
    target = torch.tensor([0])
    loss, top1, gaps = repair.sensitive_relu_loss_from_token_logits(
        logits, target, 0.25
    )
    assert torch.allclose(loss, torch.tensor(1.5625))
    assert top1.tolist() == [True]
    assert torch.allclose(gaps, torch.tensor([1.0]))
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 0]) > 0
    assert float(logits.grad[0, 1]) < 0


def test_pair_margin_loss_uses_target_true_minus_target_new():
    true_nll = torch.tensor([3.0, 1.0], requires_grad=True)
    new_nll = torch.tensor([1.0, 1.2], requires_grad=True)
    loss, margins = repair.pair_margin_loss(true_nll, new_nll, 0.05)
    assert torch.allclose(margins, torch.tensor([2.0, -0.2]))
    expected = (0.0**2 + 0.25**2) / 2.0
    assert torch.allclose(loss, torch.tensor(expected))


def test_parameter_delta_f2_is_literal_sum_of_squares():
    p1 = torch.tensor([1.0, 3.0], requires_grad=True)
    p2 = torch.tensor([[2.0]], requires_grad=True)
    ref1 = torch.tensor([0.0, 1.0])
    ref2 = torch.tensor([[5.0]])
    value = repair.parameter_delta_f2([p1, p2], [ref1, ref2])
    # 1^2 + 2^2 + (-3)^2 = 14
    assert torch.allclose(value, torch.tensor(14.0))


class TinyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(7, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.norm = nn.LayerNorm(4)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyInner()
        self.lm_head = nn.Linear(4, 7, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_configure_last_block_only_freezes_head_embeddings_and_earlier_layers():
    model = TinyLM()
    last, summary = repair.configure_last_block_only(model)
    assert last is model.model.layers[-1]
    assert not model.lm_head.weight.requires_grad
    assert not model.model.embed_tokens.weight.requires_grad
    assert not any(p.requires_grad for p in model.model.layers[0].parameters())
    assert all(p.requires_grad for p in model.model.layers[-1].parameters())
    assert not any(p.requires_grad for p in model.model.norm.parameters())
    assert summary["trainable_decoder_block_index"] == 1
    assert summary["lm_head_frozen"] is True
    assert summary["input_embeddings_frozen"] is True


def test_parser_defaults_keep_pilot_protocol_conservative():
    args = repair.parse_args(
        [
            "--model-path",
            "/model",
            "--training-visible-path",
            "/visible.json",
            "--split-manifest",
            "/manifest.json",
            "--output-dir",
            "/out",
            "--utility-wikipedia-dir",
            "/wiki",
        ]
    )
    assert args.repair_scope == "active"
    assert args.frozen_head_margin == 0.25
    assert args.pair_margin == 0.05
    assert args.utility_sample_size == 200
    assert args.utility_kl_weight == 2.0
    assert args.delta_weight == 1e-8
