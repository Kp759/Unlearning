import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from static_overlap_data import Example
from static_overlap_extended_tokens import (
    ExtendedTokenEditor,
    InputOnlyExtendedEmbedding,
    association_token_specs,
    insert_association_token,
)


class ToyTiedCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(5, 3)
        self.head = nn.Linear(3, 5, bias=False)
        self.head.weight = self.embed.weight

    def get_input_embeddings(self):
        return self.embed

    def set_input_embeddings(self, embedding):
        self.embed = embedding

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, **_):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def test_association_tokens_add_subject_only_for_duplicate_relation_object():
    facts = [
        {"id": "f1", "role": "forget", "subject": "Rob", "relation": "native language", "object": "French"},
        {"id": "f2", "role": "forget", "subject": "Kim", "relation": "native language", "object": "French"},
        {"id": "f3", "role": "forget", "subject": "Sam", "relation": "citizenship", "object": "Canada"},
        {"id": "r1", "role": "retain", "subject": "Lee", "relation": "citizenship", "object": "Japan"},
    ]
    specs = association_token_specs(facts)
    assert [row["fact_id"] for row in specs] == ["f1", "f2", "f3"]
    assert specs[0]["association"] == "rob - native language - french"
    assert specs[1]["association"] == "kim - native language - french"
    assert specs[2]["association"] == "citizenship - canada"
    assert len({row["token"] for row in specs}) == 3


def test_input_only_extension_is_exact_for_original_ids_and_trainable_for_new_ids():
    torch.manual_seed(7)
    base = nn.Embedding(5, 3)
    extra = torch.randn(2, 3)
    extended = InputOnlyExtendedEmbedding(base, extra)
    original_ids = torch.tensor([[0, 2, 4]])
    torch.testing.assert_close(extended(original_ids), base(original_ids), atol=0, rtol=0)
    mixed = torch.tensor([[1, 5, 6]])
    output = extended(mixed)
    torch.testing.assert_close(output[0, 0], base.weight[1])
    torch.testing.assert_close(output[0, 1:], extra)
    output.sum().backward()
    assert extended.extra.grad is not None
    assert base.weight.grad is None


def test_editor_keeps_natural_logits_and_original_output_vocabulary_exact():
    torch.manual_seed(11)
    model = ToyTiedCausalModel().eval()
    original_ids = torch.tensor([[0, 3, 4]])
    before = model(original_ids).logits.detach().clone()
    editor = ExtendedTokenEditor(model, torch.randn(2, 3))
    after = model(original_ids).logits.detach()
    torch.testing.assert_close(after, before, atol=0, rtol=0)
    assert model(torch.tensor([[5, 6]])).logits.shape[-1] == 5
    assert model.get_output_embeddings().weight.data_ptr() == editor.base_embedding.weight.data_ptr()
    assert [parameter for parameter in model.parameters() if parameter.requires_grad] == [
        editor.embedding.extra
    ]


def test_token_insertion_preserves_original_answer_mask():
    example = Example("e", "train", "forget", "f", [1, 2, 3, 4],
                      [-100, -100, 3, 4], "Question", " answer", "g")
    routed = insert_association_token(example, 10, None)
    assert routed.input_ids == [1, 10, 2, 3, 4]
    assert routed.labels == [-100, -100, -100, 3, 4]
    unknown = insert_association_token(example, 10, [7, 8])
    assert unknown.input_ids == [1, 10, 2, 7, 8]
    assert unknown.labels == [-100, -100, -100, 7, 8]
