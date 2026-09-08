from __future__ import annotations

import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import mcf_fact_conditioned_completion_guard_seed1 as guard


def test_single_token_sequence_is_always_blocked():
    assert guard.blocked_next_tokens([], [(7,)]) == {7}
    assert guard.blocked_next_tokens([1, 2, 3], [(7,)]) == {7}


def test_multitoken_sequence_blocks_only_completing_transition():
    seq = (10, 11, 12)
    assert guard.blocked_next_tokens([], [seq]) == set()
    assert guard.blocked_next_tokens([10], [seq]) == set()
    assert guard.blocked_next_tokens([99, 10, 11], [seq]) == {12}
    assert guard.blocked_next_tokens([10, 99, 11], [seq]) == set()


def test_hard_mask_uses_negative_infinity_without_mutating_input():
    x = torch.tensor([1.0, 2.0, 3.0])
    y = guard.hard_mask_scores(x, [1])
    assert torch.isfinite(x).all()
    assert math.isinf(float(y[1])) and float(y[1]) < 0
    assert float(y[0]) == 1.0 and float(y[2]) == 3.0


def test_rescue_is_additive_and_never_removes_fix5o_binding():
    rows = [object(), object()]
    item = guard.GuardItem(
        row_index=1,
        fact_key=("Entity", "P1"),
        request_text="TARGET_ENTITY request",
        query="Entity request",
        family="f",
        kind="k",
        forbidden_row=True,
        should_block=True,
        candidate_present=True,
        scope_supported=True,
    )
    fix5o = [{("Already", "P2")}, set()]
    accepted, hybrid = guard.hybrid_blocks_for_eta(
        rows,
        [item],
        [{"permit_margin": -2.0}],
        {1: [0]},
        fix5o,
        eta=0.0,
    )
    assert accepted[0] == {("Already", "P2")}
    assert hybrid[0] == {("Already", "P2")}
    assert hybrid[1] == {("Entity", "P1")}


def test_explicit_permission_prevents_rescue_block():
    rows = [object()]
    item = guard.GuardItem(
        row_index=0,
        fact_key=("Entity", "P1"),
        request_text="TARGET_ENTITY request",
        query="Entity request",
        family="f",
        kind="k",
        forbidden_row=False,
        should_block=False,
        candidate_present=True,
        scope_supported=True,
    )
    _, hybrid = guard.hybrid_blocks_for_eta(
        rows,
        [item],
        [{"permit_margin": 3.0}],
        {0: [0]},
        [set()],
        eta=1.0,
    )
    assert hybrid == [set()]


def test_target_entity_request_removes_marked_exact_name():
    got = guard.target_entity_request(
        "What is [TARGET]Example Person[/TARGET]'s native language?",
        "Example Person",
    )
    assert "Example Person" not in got
    assert "TARGET_ENTITY" in got
    assert "native language" in got
