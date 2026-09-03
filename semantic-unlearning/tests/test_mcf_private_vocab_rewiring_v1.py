from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_private_vocab_rewiring_v1_core as core  # noqa: E402


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
            "alpha": 3,
            "beta": 4,
            "<|reserved_special_token_0|>": 8,
            "<|reserved_special_token_1|>": 9,
        }

    def get_vocab(self):
        return dict(self._vocab)

    def __call__(self, text, **kwargs):
        if text == "Alpha Beta":
            return {"input_ids": [3, 4]}
        return {"input_ids": [3]}


def test_reserved_slot_selection_is_explicit():
    tok = DummyTokenizer()
    slots = core.discover_reserved_slots(tok, needed=2)
    assert slots == [
        ("<|reserved_special_token_0|>", 8),
        ("<|reserved_special_token_1|>", 9),
    ]


def test_subject_mapping_keeps_base_tokenization():
    mapping = core.build_subject_slot_mapping(DummyTokenizer(), ["Alpha Beta"])
    assert mapping[0]["private_token_id"] == 8
    assert mapping[0]["base_subject_token_ids"] == [3, 4]


def test_private_controller_changes_only_private_positions():
    controller = core.PrivateRowController([8], torch.tensor([[7.0, 8.0]]))
    input_ids = torch.tensor([[3, 8, 4]])
    output = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    result = controller.apply(input_ids, output)
    assert torch.equal(result[0, 0], output[0, 0])
    assert torch.equal(result[0, 2], output[0, 2])
    assert torch.equal(result[0, 1], torch.tensor([7.0, 8.0]))


def test_relative_cap_is_hard():
    controller = core.PrivateRowController([8], torch.tensor([[3.0, 4.0]]))
    with torch.no_grad():
        controller.rows.copy_(torch.tensor([[30.0, 40.0]]))
    stats = controller.enforce_relative_cap(0.5)
    assert stats["max_relative_delta"] <= 0.500001


def test_materialization_changes_only_selected_rows():
    weight = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    before = weight.clone()
    controller = core.PrivateRowController([8], torch.tensor([[101.0, 102.0]]))
    core.materialize_private_rows(weight, controller)
    assert torch.equal(weight[8], torch.tensor([101.0, 102.0]))
    keep = [i for i in range(10) if i != 8]
    assert torch.equal(weight[keep], before[keep])


def test_bfloat16_hashes_exact_raw_bits():
    value = torch.tensor([[1.0, -2.0], [3.5, 4.0]], dtype=torch.bfloat16)
    same = value.clone()
    assert core.sha256_tensor(value) == core.sha256_tensor(same)
    same[0, 0] = torch.tensor(1.5, dtype=torch.bfloat16)
    assert core.sha256_tensor(value) != core.sha256_tensor(same)


def test_bfloat16_non_private_hash_ignores_private_rows_only():
    value = torch.arange(20, dtype=torch.float32).reshape(10, 2).to(torch.bfloat16)
    baseline = core.non_private_row_hash(value, [8])
    private_changed = value.clone()
    private_changed[8] = torch.tensor([99.0, 100.0], dtype=torch.bfloat16)
    assert core.non_private_row_hash(private_changed, [8]) == baseline
    public_changed = value.clone()
    public_changed[7, 0] += torch.tensor(1.0, dtype=torch.bfloat16)
    assert core.non_private_row_hash(public_changed, [8]) != baseline


def test_patch_tokenizer_reserved_entry(tmp_path: Path):
    payload = {
        "added_tokens": [
            {
                "id": 8,
                "content": "<|reserved_special_token_0|>",
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        ]
    }
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload), encoding="utf-8")
    config = {
        "added_tokens_decoder": {
            "8": {
                "content": "<|reserved_special_token_0|>",
                "special": True,
                "normalized": False,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
            }
        },
        "additional_special_tokens": ["<|reserved_special_token_0|>"],
    }
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(config), encoding="utf-8")
    mapping = [
        {
            "subject": "Alpha Beta",
            "private_token_id": 8,
            "original_reserved_token": "<|reserved_special_token_0|>",
            "base_subject_token_ids": [3, 4],
        }
    ]
    core.patch_saved_tokenizer_reserved_slots(tmp_path, mapping)
    patched = json.loads((tmp_path / "tokenizer.json").read_text())
    assert patched["added_tokens"][0]["content"] == "Alpha Beta"
    assert patched["added_tokens"][0]["special"] is False
    cfg = json.loads((tmp_path / "tokenizer_config.json").read_text())
    assert cfg["added_tokens_decoder"]["8"]["content"] == "Alpha Beta"
    assert cfg["additional_special_tokens"] == []
