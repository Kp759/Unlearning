from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_private_vocab_rewiring_v1_1_core as core  # noqa: E402


class DummyTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = None
    unk_token_id = 0

    def __init__(self):
        self._vocab = {
            "<unk>": 0,
            "<bos>": 1,
            "<eos>": 2,
            "bel": 3,
            "gium": 4,
            "x": 5,
            "<|reserved_special_token_0|>": 8,
            "<|reserved_special_token_1|>": 9,
            "<|reserved_special_token_2|>": 10,
            "<|reserved_special_token_3|>": 11,
            "<|reserved_special_token_4|>": 12,
            "<|reserved_special_token_5|>": 13,
        }

    def __len__(self):
        return 14

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, *args, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [self._encode(item) for item in text]}
        return {"input_ids": self._encode(text)}

    def _encode(self, text):
        if text == "Belgium":
            return [3, 4]
        if text == "New Belgium":
            return [5, 3, 4]
        if text == "Tell me about Belgium.":
            return [5, 5, 3, 4, 5]
        if text == "Tell me about New Belgium.":
            return [5, 5, 5, 3, 4, 5]
        return [5]


def test_mapping_allocates_one_private_id_per_base_subject_token():
    tok = DummyTokenizer()
    mapping = core.build_position_preserving_mapping(tok, ["Belgium"])
    assert mapping[0]["base_subject_token_ids"] == [3, 4]
    assert mapping[0]["private_token_ids"] == [8, 9]
    assert mapping[0]["token_count"] == 2


def test_private_tokenizer_preserves_sequence_length_and_rewrites_subject():
    tok = DummyTokenizer()
    mapping = core.build_position_preserving_mapping(tok, ["Belgium"])
    private = core.PositionPreservingSubjectTokenizer(tok, mapping)
    assert private("Belgium")["input_ids"] == [8, 9]
    assert private("Tell me about Belgium.")["input_ids"] == [5, 5, 8, 9, 5]
    assert len(private("Tell me about Belgium.")["input_ids"]) == len(
        tok("Tell me about Belgium.")["input_ids"]
    )


def test_unrelated_text_is_bitwise_same_token_ids():
    tok = DummyTokenizer()
    mapping = core.build_position_preserving_mapping(tok, ["Belgium"])
    private = core.PositionPreservingSubjectTokenizer(tok, mapping)
    assert private("unrelated")["input_ids"] == tok("unrelated")["input_ids"]


def test_exact_private_initialization_is_one_to_one_not_mean_pooling():
    weight = torch.arange(28, dtype=torch.float32).reshape(14, 2)
    tok = DummyTokenizer()
    mapping = core.build_position_preserving_mapping(tok, ["Belgium"])
    rows = core.initialize_exact_private_rows(weight, mapping)
    assert torch.equal(rows[0], weight[3])
    assert torch.equal(rows[1], weight[4])
    assert not torch.equal(rows[0], rows[1])


def test_validation_reports_position_preservation():
    tok = DummyTokenizer()
    mapping = core.build_position_preserving_mapping(tok, ["Belgium"])
    private = core.PositionPreservingSubjectTokenizer(tok, mapping)
    report = core.validate_position_preserving_routing(tok, private, mapping)
    assert report["subjects"] == 1
    assert report["private_rows"] == 2
    assert report["all_subject_lengths_preserved"] is True


def test_multiple_subjects_receive_disjoint_private_rows():
    tok = DummyTokenizer()
    mapping = core.build_position_preserving_mapping(tok, ["Belgium", "New Belgium"])
    flat = core.flatten_private_ids(mapping)
    assert len(flat) == 5
    assert len(set(flat)) == 5
    assert mapping[0]["private_token_ids"] == [8, 9]
    assert mapping[1]["private_token_ids"] == [10, 11, 12]


def test_longest_subject_sequence_wins_when_subjects_overlap():
    tok = DummyTokenizer()
    mapping = core.build_position_preserving_mapping(tok, ["Belgium", "New Belgium"])
    private = core.PositionPreservingSubjectTokenizer(tok, mapping)
    # The 3-token New Belgium rule must win over the embedded 2-token Belgium rule.
    assert private("Tell me about New Belgium.")["input_ids"] == [5, 5, 10, 11, 12, 5]
