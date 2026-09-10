import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from static_overlap_activation_protocol import PLAN
from static_overlap_activation_rewire import ActivationRewireEditor, RelationKeyDown, relation_keys


class ToyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(4, 3, bias=False)


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = ToyMlp()


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(7, 3)
        self.head = nn.Linear(3, 7, bias=False)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ToyLayer()])
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


def test_relation_key_subtracts_locality_and_is_retain_orthogonal():
    forget = {"f": [torch.tensor([2., 1., 0., 0.]), torch.tensor([2., 1., 0., 0.])]}
    locality = {"f": [torch.tensor([1., 1., 0., 0.])]}
    retain = [torch.tensor([0., 1., 0., 0.]), torch.tensor([0., 0., 1., 0.])]
    keys, report = relation_keys(forget, locality, retain, rank=1, relative_tolerance=1e-7)
    torch.testing.assert_close(keys[0], torch.tensor([1., 0., 0., 0.]), atol=1e-6, rtol=0)
    assert report["forget_facts"] == 1
    assert report["max_normalized_retain_key_overlap"] < 1e-6


def test_relation_key_down_is_exact_at_zero_and_merges_same_delta():
    base = nn.Linear(4, 3, bias=False)
    edit = RelationKeyDown(base, rank=1)
    x = torch.randn(2, 4)
    torch.testing.assert_close(edit(x), base(x), atol=0, rtol=0)
    with torch.no_grad():
        edit.keys[0] = torch.tensor([1., 0., 0., 0.])
        edit.values[:, 0] = torch.tensor([.5, -.25, 1.])
    expected = edit(x)
    with torch.no_grad():
        base.weight.add_(edit.delta())
    torch.testing.assert_close(base(x), expected)


def test_protocol_keeps_exact_limits_and_one_key_per_forget_fact():
    assert PLAN["activation_key_rank"] == 50
    assert PLAN["retain_nll_budget"] == .05 and PLAN["fitting_nll_margin"] == 0.
    assert PLAN["retain_kl_budget"] == .01 and PLAN["fitting_kl_margin"] == 0.


def test_editor_artifact_round_trip_and_native_merge():
    torch.manual_seed(4)
    first, second = ToyModel(), ToyModel()
    second.load_state_dict(first.state_dict())
    original = first.model.layers[0].mlp.down_proj.weight.detach().clone()
    editor = ActivationRewireEditor(first, [0], rank=2)
    reloaded = ActivationRewireEditor(second, [0], rank=2)
    keys = torch.randn(2, 4)
    with torch.no_grad():
        editor.set_keys({0: keys})
        editor.downs[0].values.copy_(torch.randn(3, 2))
    reloaded.load_artifact(editor.artifact())
    expected = original + editor.downs[0].values @ keys
    merged = reloaded.merge()
    assert isinstance(merged.model.layers[0].mlp.down_proj, nn.Linear)
    torch.testing.assert_close(merged.model.layers[0].mlp.down_proj.weight, expected)
    assert not any(parameter.requires_grad for parameter in merged.parameters())
