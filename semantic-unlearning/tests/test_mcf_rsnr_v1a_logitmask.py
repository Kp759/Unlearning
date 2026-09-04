from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_rsnr_v1a_logitmask_common as lm  # noqa: E402


def test_logitmask_gate_off_is_exact_identity():
    torch.manual_seed(0)
    head = nn.Linear(4, 11, bias=False)
    hidden = torch.randn(2, 3, 4)
    base = head(hidden).detach().clone()
    hook = lm.DirectLogitMaskHook.install(
        head, true_penalty=5.0, idk_boost=3.0, idk_token_ids=[7, 8]
    )
    try:
        hook.set(torch.zeros(2), None, [(), ()])
        out = head(hidden).detach()
        hook.clear()
    finally:
        hook.remove()
    assert torch.equal(base, out)


def test_logitmask_changes_only_registered_tokens_at_registered_position():
    torch.manual_seed(1)
    head = nn.Linear(4, 12, bias=False)
    hidden = torch.randn(1, 3, 4)
    base = head(hidden).detach().clone()
    hook = lm.DirectLogitMaskHook.install(
        head, true_penalty=4.0, idk_boost=2.0, idk_token_ids=[8, 9]
    )
    positions = torch.zeros(1, 3)
    positions[0, 1] = 1
    try:
        hook.set(torch.ones(1), positions, [(2, 5)])
        out = head(hidden).detach()
        hook.clear()
    finally:
        hook.remove()

    expected = base.clone()
    expected[0, 1, 2] -= 4.0
    expected[0, 1, 5] -= 4.0
    expected[0, 1, 8] += 2.0
    expected[0, 1, 9] += 2.0
    assert torch.allclose(out, expected)


def test_logitmask_non_gated_batch_row_remains_exact():
    torch.manual_seed(2)
    head = nn.Linear(3, 9, bias=False)
    hidden = torch.randn(2, 2, 3)
    base = head(hidden).detach().clone()
    hook = lm.DirectLogitMaskHook.install(
        head, true_penalty=6.0, idk_boost=1.0, idk_token_ids=[7]
    )
    positions = torch.ones(2, 2)
    try:
        hook.set(torch.tensor([1.0, 0.0]), positions, [(1,), (2,)])
        out = head(hidden).detach()
        hook.clear()
    finally:
        hook.remove()
    assert torch.equal(out[1], base[1])
    assert not torch.equal(out[0], base[0])


def test_canonical_answer_map_is_pair_scoped():
    rows = [
        {
            "case_id": 1,
            "requested_rewrite": {
                "subject": "Belgium",
                "relation_id": "P36",
                "target_true": {"str": "Brussels"},
            },
        },
        {
            "case_id": 2,
            "requested_rewrite": {
                "subject": "Belgium",
                "relation_id": "P38",
                "target_true": {"str": "euro"},
            },
        },
    ]
    assert lm.canonical_answer_map(rows) == {
        ("Belgium", "P36"): "Brussels",
        ("Belgium", "P38"): "euro",
    }


def test_primary_config_rejects_alias_masking(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(
        '{"protocol":"mcf_rsnr_v1a_oracle_direct_logit_mask",'
        '"calibration_passed":true,"heldout_probe_text_used":false,'
        '"aliases_used_for_mask":true}\n',
        encoding="utf-8",
    )
    try:
        lm.load_config(p)
        assert False, "alias-aware primary config should be rejected"
    except RuntimeError as exc:
        assert "must not use aliases" in str(exc)
