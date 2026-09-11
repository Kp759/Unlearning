import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_static_overlap_natural_writer import sample_preservation_records
from static_overlap_data import Example
from static_overlap_natural_writer import (
    encode_natural_views,
    mcf_facts,
    select_writer_channels,
    training_text_fingerprints,
)


class CharacterTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=True):
        assert add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [0] + list(range(1, len(text) + 1)),
            "offset_mapping": [(0, 0)] + [(i, i + 1) for i in range(len(text))],
        }


def test_natural_views_use_canonical_prompt_without_private_tokens():
    record = {
        "case_id": 17,
        "requested_rewrite": {
            "subject": "Rob",
            "relation_id": "P103",
            "target_true": {"str": "French"},
            "prompt": "{} speaks",
        },
        "paraphrase_prompts": ["SECRET OFFICIAL PARAPHRASE"],
        "neighborhood_prompts": ["SECRET OFFICIAL NEIGHBORHOOD"],
    }
    forget = mcf_facts([record], "forget")
    retain_record = {
        "case_id": 18,
        "requested_rewrite": {
            "subject": "Kim",
            "relation_id": "P103",
            "target_true": {"str": "Korean"},
            "prompt": "{} speaks",
        },
    }
    retain = mcf_facts([retain_record], "retain")
    examples = encode_natural_views(forget, retain, CharacterTokenizer(), 512)

    canonical = [
        e for e in examples
        if e.fact_id == "mcf_forget_17" and e.group == "canonical_rewrite"
    ]
    assert len(canonical) == 1
    assert canonical[0].prompt == "Rob speaks"
    joined = "\n".join(e.prompt for e in examples)
    assert "SECRET OFFICIAL PARAPHRASE" not in joined
    assert "SECRET OFFICIAL NEIGHBORHOOD" not in joined
    assert "<|forget_assoc_" not in joined


def test_training_fingerprints_include_prompt_and_completion():
    example = Example(
        "e", "train", "forget", "f", [0, 1, 2], [-100, -100, 2],
        "Question:", " answer", "g",
    )
    values = training_text_fingerprints([example])
    assert "question:" in values
    assert "question: answer" in values


def test_preservation_sampling_excludes_reserved_official_retain():
    records = [{"case_id": i} for i in range(40)]
    reserved = records[:10]
    selected = sample_preservation_records(
        records,
        count=5,
        seed=1,
        reserved=reserved,
    )
    assert len(selected) == 5
    assert not ({row["case_id"] for row in selected} & {row["case_id"] for row in reserved})
    assert all(row["case_id"] < 20 for row in selected)


class TinyMLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.down_proj = nn.Linear(width, width, bias=False)

    def forward(self, x):
        return self.down_proj(x)


class TinyBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.mlp = TinyMLP(width)


class TinyNaturalWriterModel(nn.Module):
    def __init__(self, vocab=11, width=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab, width)
        self.model = SimpleNamespace(layers=nn.ModuleList([TinyBlock(width)]))
        self.layers = self.model.layers
        self.head = nn.Linear(width, vocab, bias=False)

    def parameters(self, recurse=True):
        yield from self.embedding.parameters(recurse=recurse)
        yield from self.layers.parameters(recurse=recurse)
        yield from self.head.parameters(recurse=recurse)

    def requires_grad_(self, requires_grad=True):
        self.embedding.requires_grad_(requires_grad)
        self.layers.requires_grad_(requires_grad)
        self.head.requires_grad_(requires_grad)
        return self

    def zero_grad(self, set_to_none=True):
        self.embedding.zero_grad(set_to_none=set_to_none)
        self.layers.zero_grad(set_to_none=set_to_none)
        self.head.zero_grad(set_to_none=set_to_none)

    def forward(self, input_ids, **_):
        hidden = self.embedding(input_ids)
        hidden = self.layers[0].mlp(hidden)
        return SimpleNamespace(logits=self.head(hidden))


def test_sparse_writer_channel_selection_uses_fitting_data_only():
    torch.manual_seed(4)
    model = TinyNaturalWriterModel()
    examples = [
        Example("f1", "train", "forget", "f", [0, 1, 2], [-100, -100, 2], "p", " a", "g"),
        Example("f2", "train", "forget", "f", [0, 1, 3], [-100, -100, 3], "p", " b", "g"),
        Example("r1", "train", "retain", "r", [0, 4, 5], [-100, -100, 5], "p", " c", "g"),
        Example("r2", "train", "retain", "r", [0, 4, 6], [-100, -100, 6], "p", " d", "g"),
        Example("fd", "development", "forget", "f", [0, 1, 7], [-100, -100, 7], "p", " e", "g"),
        Example("rd", "development", "retain", "r", [0, 4, 8], [-100, -100, 8], "p", " f", "g"),
    ]
    channels, report = select_writer_channels(
        model, examples, layer=0, example_count=2, channel_count=3, seed=1
    )
    assert len(channels) == 3
    assert len(set(channels)) == 3
    assert report["development_used_for_channel_selection"] is False
    assert all(value.startswith(("f", "r")) for value in report["forget_example_ids"] + report["retain_example_ids"])
