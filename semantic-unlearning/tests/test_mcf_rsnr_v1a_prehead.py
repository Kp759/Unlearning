from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_rsnr_v1a_oracle as rsnr  # noqa: E402
import run_mcf_rsnr_v1a_prehead as prehead  # noqa: E402
import mcf_rsnr_v1a_prehead_official_eval as peval  # noqa: E402


def _nonzero_adapter(hidden_size=4, rank=2):
    adapter = rsnr.NullResidualAdapter(
        hidden_size=hidden_size,
        rank=rank,
        alpha=float(rank),
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        adapter.down.weight.fill_(0.25)
        adapter.up.weight.fill_(0.5)
    return adapter


def test_prehead_gate_off_is_exact_identity():
    head = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(4))
    adapter = _nonzero_adapter()
    hook = prehead.PreHeadNullHook.install(head, adapter)
    hidden = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, -1.0]]],
        dtype=torch.float32,
    )
    base = head(hidden.clone())
    hook.set(torch.zeros(1), None)
    try:
        off = head(hidden.clone())
    finally:
        hook.clear()
        hook.remove()
    assert torch.equal(base, off)


def test_prehead_gate_on_changes_only_selected_positions_and_rows():
    head = nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.eye(4))
    adapter = _nonzero_adapter()
    hook = prehead.PreHeadNullHook.install(head, adapter)
    hidden = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, -1.0]],
            [[1.5, 0.5, -0.5, 2.0], [0.0, 1.0, 2.0, 3.0]],
        ],
        dtype=torch.float32,
    )
    base = head(hidden.clone())
    positions = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    hook.set(torch.tensor([1.0, 0.0]), positions)
    try:
        edited = head(hidden.clone())
    finally:
        hook.clear()
        hook.remove()
    assert torch.equal(edited[0, 0], base[0, 0])
    assert not torch.equal(edited[0, 1], base[0, 1])
    assert torch.equal(edited[1], base[1])


def test_prehead_rank16_parameter_budget_matches_layer24_adapter():
    adapter = rsnr.NullResidualAdapter(
        hidden_size=3072,
        rank=16,
        alpha=16.0,
        device=torch.device("cpu"),
    )
    assert sum(p.numel() for p in adapter.parameters()) == 98_304


def test_method_aligned_metrics_separate_sensitive_leakage_from_idk_preference():
    teacher = [
        {"split": "forget", "group": "rewrite", "idk_vs_true_margin": 2.0, "true_logprob_drop": 3.0},
        {"split": "forget", "group": "paraphrase", "idk_vs_true_margin": -0.2, "true_logprob_drop": 1.5},
    ]
    generation = [
        {"split": "forget", "group": "rewrite", "true_or_alias_leak": False, "semantic_abstention": True},
        {"split": "forget", "group": "paraphrase", "true_or_alias_leak": True, "semantic_abstention": False},
    ]
    out = peval.summarize_method_aligned(teacher, generation)
    assert out["Eff_IDK"] == 0.0
    assert out["Gen_IDK"] == 100.0
    assert out["Sensitive_Eff"] == 0.0
    assert out["Sensitive_Gen"] == 100.0


def test_prehead_protocol_explicitly_freezes_transformer_and_lm_head_weights():
    source = Path(prehead.__file__).read_text(encoding="utf-8")
    assert '"lm_head_weights_modified": False' in source
    assert '"transformer_weights_modified": False' in source
    assert prehead.INTERVENTION_SITE == "pre_lm_head_final_hidden_state"
