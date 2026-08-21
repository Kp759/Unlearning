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
    "mcf_unknown_lowrank_locality_repair_test",
    SCRIPTS / "mcf_unknown_lowrank_locality_repair.py",
)


def test_lowrank_adapter_starts_exactly_as_base():
    torch.manual_seed(0)
    base = nn.Linear(5, 3, bias=False)
    x = torch.randn(7, 5)
    expected = base(x).detach().clone()
    adapter = repair.LowRankResidualLinear(base, rank=2)
    actual = adapter(x)
    assert torch.equal(actual, expected)
    assert torch.count_nonzero(adapter.B.weight).item() == 0
    assert adapter.trainable_parameter_count == 2 * (5 + 3)
    assert not adapter.base.weight.requires_grad
    assert adapter.A.weight.requires_grad
    assert adapter.B.weight.requires_grad


def test_lowrank_adapter_merge_preserves_function_float32():
    torch.manual_seed(1)
    base = nn.Linear(4, 6, bias=False, dtype=torch.float32)
    adapter = repair.LowRankResidualLinear(base, rank=3)
    with torch.no_grad():
        adapter.B.weight.normal_(mean=0.0, std=0.1)
    x = torch.randn(11, 4)
    before = adapter(x).detach()
    merged = adapter.merge_into_base()
    after = merged(x).detach()
    assert torch.allclose(after, before, atol=1e-6, rtol=1e-6)


def test_locality_hidden_loss_zero_at_base_and_positive_after_drift():
    current = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    reference = current.detach().clone()
    zero = repair.locality_hidden_loss(current, reference)
    assert torch.allclose(zero, torch.tensor(0.0))

    shifted = current + 1.0
    loss = repair.locality_hidden_loss(shifted, reference)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_build_locality_prompts_uses_source_template_and_other_subjects():
    records = []
    for i, subject in enumerate(["Alice", "Bob", "Carol"]):
        records.append(
            {
                "case_id": i,
                "requested_rewrite": {
                    "subject": subject,
                    "prompt": "{} was born in",
                    "target_true": {"str": "Sensitive"},
                    "target_new": {"str": "Reference"},
                },
            }
        )
    prompts, receipt = repair.build_locality_prompts(records, control_count=2)
    assert len(prompts) == 6
    assert len(receipt) == 6
    assert all("was born in" in p for p in prompts)
    for row in receipt:
        assert row["donor_subject"] != row["original_subject"]
        assert row["prompt"] == f"{row['donor_subject']} was born in"


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(4, 4, bias=False)


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()
        self.other = nn.Linear(4, 4, bias=False)


class TinyInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(9, 4)
        self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.norm = nn.LayerNorm(4)


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyInner()
        self.lm_head = nn.Linear(4, 9, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_install_final_adapter_freezes_every_original_parameter():
    model = TinyLM()
    original_ids = {id(p) for p in model.parameters()}
    _last, adapter, summary = repair.install_final_mlp_adapter(model, rank=2)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert {id(p) for p in trainable} == {id(adapter.A.weight), id(adapter.B.weight)}
    assert all(not p.requires_grad for p in model.parameters() if id(p) in original_ids)
    assert summary["adapter_rank"] == 2
    assert summary["adapter_location"] == "final_decoder_block.mlp.down_proj"
    assert summary["base_parameter_count_trainable"] == 0
    assert summary["lm_head_frozen"] is True
    assert summary["input_embeddings_frozen"] is True


def test_merge_and_remove_restores_plain_linear():
    model = TinyLM()
    _last, adapter, _summary = repair.install_final_mlp_adapter(model, rank=2)
    with torch.no_grad():
        adapter.B.weight.fill_(0.05)
    report = repair.merge_and_remove_adapter(model, adapter)
    assert isinstance(model.model.layers[-1].mlp.down_proj, nn.Linear)
    assert report["rank"] == 2
    assert report["wrapper_removed_before_save"] is True
    assert report["effective_delta_frobenius_norm"] > 0
    assert not any(p.requires_grad for p in model.parameters())


def test_parser_defaults_prioritize_locality_and_base_safe_training():
    args = repair.parse_args(
        [
            "--model-path", "/base",
            "--training-visible-path", "/visible.json",
            "--split-manifest", "/manifest.json",
            "--output-dir", "/out",
            "--adapter-rank", "4",
            "--utility-wikipedia-dir", "/wiki",
        ]
    )
    assert args.adapter_rank == 4
    assert args.repair_scope == "active"
    assert args.neutral_answer == "Unknown"
    assert args.forget_margin == 0.05
    assert args.locality_hidden_weight == 10.0
    assert args.subject_control_count == 4
    assert args.utility_sample_size == 200
    assert args.utility_exclude_first == 20
