"""Write-position modes of the linear-classifier bank (A/B/C/D experiment).

  last          boundary only (shipped behaviour, must stay bit-identical)
  last_subject  last subject token + boundary
  subject_span  whole subject span + boundary
  all_prompt    every attended prompt token except BOS

Routing never changes with the mode, and an unrouted prompt stays exact.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from linear_router import (  # noqa: E402
    WRITE_MODES,
    LinearClassifierAssociationBank,
    load_linear_classifier_artifact,
)

HIDDEN, FACTS, BOS = 8, 2, 1


class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _Config:
    bos_token_id = BOS


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.config = _Config()
        self.placeholder = nn.Parameter(torch.zeros(1))


def _bank(mode):
    # fact 0's subject is the two tokens (20, 21); fact 1's is (30,).
    # Head 0 fires on any query (huge bias); head 1 never does.
    return LinearClassifierAssociationBank(
        _FakeModel(), 0, torch.zeros(FACTS, HIDDEN), torch.tensor([50.0, -50.0]),
        feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
        subject_patterns=[[(20, 21)], [(30,)]],
        facts=[{"id": "f0"}, {"id": "f1"}],
        rows=torch.ones(FACTS, HIDDEN), write_mode=mode,
    )


# "<bos> The mother of S0a S0b is | answer" ; boundary (prefix length) = 7
IDS = torch.tensor([[BOS, 5, 6, 7, 20, 21, 8, 9, 9]])
PREFIX = torch.tensor([7])
EXPECTED = {
    "last": {6},
    "last_subject": {5, 6},
    "subject_span": {4, 5, 6},
    "all_prompt": {1, 2, 3, 4, 5, 6},
}


def _edited_positions(mode, ids=IDS, prefix=PREFIX, attention=None):
    bank = _bank(mode)
    torch.manual_seed(0)
    hidden = torch.randn(ids.shape[0], ids.shape[1], HIDDEN)
    bank.bind(ids, attention_mask=attention, prefix_lengths=prefix)
    edited = bank._hook(None, None, (hidden,))[0]
    bank.unbind()
    changed = (edited - hidden).abs().sum(-1) > 0
    return [set(changed[b].nonzero().flatten().tolist()) for b in range(ids.shape[0])], hidden, edited


@pytest.mark.parametrize("mode", WRITE_MODES)
def test_each_mode_edits_exactly_its_positions(mode):
    changed, hidden, edited = _edited_positions(mode)
    assert changed[0] == EXPECTED[mode]
    for pos in EXPECTED[mode]:
        assert torch.equal(edited[0, pos], hidden[0, pos] + 1.0)


def test_bos_and_answer_tokens_are_never_edited():
    for mode in WRITE_MODES:
        changed, _, _ = _edited_positions(mode)
        assert 0 not in changed[0]
        assert not changed[0] & {7, 8}


def test_left_padding_uses_absolute_positions():
    pad = 0
    ids = torch.tensor([[pad, pad, BOS, 5, 20, 21, 8, 9]])
    attention = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1]])
    changed, _, _ = _edited_positions("subject_span", ids, torch.tensor([7]), attention)
    assert changed[0] == {4, 5, 6}
    changed, _, _ = _edited_positions("all_prompt", ids, torch.tensor([7]), attention)
    assert changed[0] == {3, 4, 5, 6}


def test_unrouted_prompt_is_exact_in_every_mode():
    ids = torch.tensor([[BOS, 5, 6, 7, 30, 8, 9]])      # only fact 1's subject: head 1 never fires
    for mode in WRITE_MODES:
        changed, hidden, edited = _edited_positions(mode, ids, torch.tensor([6]))
        assert changed[0] == set()
        assert torch.equal(edited, hidden)


def test_missing_subject_falls_back_to_boundary_only():
    bank = _bank("subject_span")
    bank.set_oracle_routes({(BOS, 5, 6, 7, 8): 0})     # oracle routes a prompt without the subject
    ids = torch.tensor([[BOS, 5, 6, 7, 8, 9]])
    hidden = torch.randn(1, 6, HIDDEN)
    bank.bind(ids, prefix_lengths=torch.tensor([5]))
    edited = bank._hook(None, None, (hidden,))[0]
    changed = set(((edited - hidden).abs().sum(-1)[0] > 0).nonzero().flatten().tolist())
    assert changed == {4} and bank.subject_fallbacks == 1


def test_artifact_round_trip_keeps_mode_and_old_artifacts_default_to_last():
    for mode in WRITE_MODES:
        artifact = _bank(mode).artifact()
        assert artifact["write_mode"] == mode
        _, loaded = load_linear_classifier_artifact(_FakeModel(), artifact)
        assert loaded.write_mode == mode
    old = _bank("last").artifact()
    del old["write_mode"]
    _, loaded = load_linear_classifier_artifact(_FakeModel(), old)
    assert loaded.write_mode == "last"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        _bank("everything")


# ---------------------------------------------------------------------------
# Projection edit: h - alpha * U U^T h for the routed fact
# ---------------------------------------------------------------------------

def _project_bank(alpha=1.0, mode="last"):
    torch.manual_seed(3)
    dirs = torch.zeros(FACTS, 2, HIDDEN)
    q, _ = torch.linalg.qr(torch.randn(HIDDEN, 2))
    dirs[0] = q.T                                  # fact 0: two orthonormal directions
    dirs[1, 0] = torch.nn.functional.normalize(torch.randn(HIDDEN), dim=0)  # fact 1: one (+ zero row)
    return LinearClassifierAssociationBank(
        _FakeModel(), 0, torch.zeros(FACTS, HIDDEN), torch.tensor([50.0, -50.0]),
        feature_mean=torch.zeros(HIDDEN), feature_components=None, threshold=0.0,
        subject_patterns=[[(20, 21)], [(30,)]], facts=[{"id": "f0"}, {"id": "f1"}],
        rows=torch.zeros(FACTS, HIDDEN), write_mode=mode,
        edit_mode="project", project_dirs=dirs, project_alpha=alpha,
    ), dirs


def _run_project(bank, ids=IDS, prefix=PREFIX):
    torch.manual_seed(0)
    hidden = torch.randn(ids.shape[0], ids.shape[1], HIDDEN)
    bank.bind(ids, prefix_lengths=prefix)
    edited = bank._hook(None, None, (hidden,))[0]
    bank.unbind()
    return hidden, edited


def test_projection_removes_exactly_the_answer_component_at_the_boundary():
    bank, dirs = _project_bank()
    hidden, edited = _run_project(bank)
    U = dirs[0]
    assert torch.allclose(U @ edited[0, 6], torch.zeros(2), atol=1e-5)       # component gone
    expected = hidden[0, 6] - U.T @ (U @ hidden[0, 6])                          # rest untouched
    assert torch.allclose(edited[0, 6], expected, atol=1e-5)
    others = [p for p in range(IDS.shape[1]) if p != 6]
    assert torch.equal(edited[0, others], hidden[0, others])


def test_projection_alpha_two_reflects_the_component():
    bank, dirs = _project_bank(alpha=2.0)
    hidden, edited = _run_project(bank)
    U = dirs[0]
    assert torch.allclose(U @ edited[0, 6], -(U @ hidden[0, 6]), atol=1e-5)


def test_projection_unrouted_prompt_is_exact():
    bank, _ = _project_bank()
    ids = torch.tensor([[BOS, 5, 6, 7, 30, 8, 9]])
    hidden, edited = _run_project(bank, ids, torch.tensor([6]))
    assert torch.equal(edited, hidden)


def test_projection_artifact_round_trip_and_validation():
    bank, dirs = _project_bank(alpha=1.5, mode="subject_span")
    artifact = bank.artifact()
    assert artifact["edit_mode"] == "project" and artifact["project_alpha"] == 1.5
    _, loaded = load_linear_classifier_artifact(_FakeModel(), artifact)
    assert loaded.edit_mode == "project" and loaded.write_mode == "subject_span"
    assert torch.equal(loaded.project_dirs.cpu(), dirs)
    bad = dict(artifact, project_dirs=dirs * 2.0)                               # not orthonormal
    with pytest.raises(ValueError):
        load_linear_classifier_artifact(_FakeModel(), bad)
    with pytest.raises(ValueError):
        load_linear_classifier_artifact(_FakeModel(), dict(artifact, project_dirs=None))
