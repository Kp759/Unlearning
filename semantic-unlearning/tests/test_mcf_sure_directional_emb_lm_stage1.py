from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_sure_directional_emb_lm_stage1 as stage1
import sure_canonical_core as core


def _case(record_position: int, token_index: int) -> core.SensitivePredictionCase:
    return core.SensitivePredictionCase(
        case_id=record_position,
        record_position=record_position,
        token_index=token_index,
        prompt=f"p-{record_position}-{token_index}",
        target_text="x",
    )


def test_reference_matching_prefers_same_token_index_then_last_available():
    sensitive = [_case(0, 0), _case(0, 1), _case(0, 3), _case(1, 0)]
    reference = [_case(0, 0), _case(0, 1), _case(1, 0)]
    assert stage1._reference_index_by_sensitive_case(sensitive, reference) == [0, 1, 1, 2]


def test_contrast_direction_uses_hidden_difference_when_nonzero():
    hs = torch.tensor([2.0, 1.0, 0.0])
    hr = torch.tensor([1.0, 1.0, 0.0])
    ws = torch.tensor([0.0, 0.0, 1.0])
    wr = torch.tensor([0.0, 0.0, 0.0])
    direction, source = stage1.contrast_direction(hs, hr, ws, wr)
    assert source == "hidden_sensitive_minus_reference"
    torch.testing.assert_close(direction, torch.tensor([1.0, 0.0, 0.0]))


def test_contrast_direction_falls_back_to_decoder_discriminant_for_equal_prefix():
    # This is the first-answer-token situation: sensitive/reference prefixes are
    # identical, hence h_sensitive - h_reference is exactly zero.
    h = torch.tensor([1.0, 2.0, 3.0])
    ws = torch.tensor([2.0, 0.0, 0.0])
    wr = torch.tensor([0.0, 0.0, 0.0])
    direction, source = stage1.contrast_direction(h, h.clone(), ws, wr)
    assert source == "decoder_row_sensitive_minus_reference_fallback"
    torch.testing.assert_close(direction, torch.tensor([1.0, 0.0, 0.0]))


def test_embedding_delta_hook_changes_only_selected_token_occurrences_and_has_grad():
    emb = nn.Embedding(6, 3)
    with torch.no_grad():
        emb.weight.zero_()

    delta = nn.Parameter(torch.tensor([[0.5, -1.0, 2.0]], dtype=torch.float32))
    handle = stage1.register_input_embedding_delta_hook(emb, [2], lambda: delta)
    try:
        ids = torch.tensor([[1, 2, 3], [2, 4, 5]], dtype=torch.long)
        out = emb(ids)
        expected = torch.zeros((2, 3, 3), dtype=out.dtype)
        expected[0, 1] = delta.detach().to(expected.dtype)
        expected[1, 0] = delta.detach().to(expected.dtype)
        torch.testing.assert_close(out, expected)

        loss = out.sum()
        loss.backward()
        assert delta.grad is not None
        torch.testing.assert_close(delta.grad, torch.full_like(delta, 2.0))
    finally:
        handle.remove()


def test_materialize_input_delta_changes_only_selected_rows():
    emb = nn.Embedding(5, 2)
    with torch.no_grad():
        emb.weight.zero_()
    delta = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    stage1.materialize_input_delta(emb, [1, 4], delta)
    torch.testing.assert_close(emb.weight[0], torch.zeros(2))
    torch.testing.assert_close(emb.weight[1], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(emb.weight[2], torch.zeros(2))
    torch.testing.assert_close(emb.weight[3], torch.zeros(2))
    torch.testing.assert_close(emb.weight[4], torch.tensor([3.0, 4.0]))
