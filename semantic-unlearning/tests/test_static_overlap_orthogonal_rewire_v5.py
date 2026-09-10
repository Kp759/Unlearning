import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from static_overlap_orthogonal_protocol_v5 import PLAN
from static_overlap_orthogonal_rewire_v5 import choose_basis_examples, orthonormal_rows, project_away


def test_complete_ga_gd_update_remains_in_protected_nullspace():
    basis = orthonormal_rows(
        [torch.tensor([1., 0., 1., 0.]), torch.tensor([0., 1., 0., 1.])],
        max_rank=2,
        relative_tolerance=1e-7,
    )
    forget = project_away(torch.tensor([3., 4., 5., 6.]), basis)
    retain = project_away(torch.tensor([7., 8., 2., 1.]), basis)
    torch.testing.assert_close(basis @ (forget + retain), torch.zeros(2), atol=1e-5, rtol=0)


def test_v5_uses_exact_declared_limits_and_expanded_basis():
    assert PLAN["retain_nll_budget"] == .05
    assert PLAN["retain_kl_budget"] == .01
    assert PLAN["fitting_nll_margin"] == 0.
    assert PLAN["fitting_kl_margin"] == 0.
    assert PLAN["protected_basis_rank"] == 256


class _Example:
    def __init__(self, eid, role="retain"):
        self.id = eid
        self.role = role
        self.split = "train"


def test_expanded_basis_prioritizes_hard_anchors_then_subject_coverage():
    synthetic_ids = [f"orthogonal_same_subject_forget_{i}_P{j}" for i in range(3) for j in range(2)]
    hard_ids = [f"hard_{i}" for i in range(8)]
    rows = [_Example(eid) for eid in hard_ids + synthetic_ids]
    chosen = choose_basis_examples(rows, synthetic_ids, hard_ids, limit=10)
    chosen_ids = [row.id for row in chosen]
    # Three slots are reserved for one locality direction per subject.
    assert chosen_ids[:7] == hard_ids[:7]
    assert len({eid.rsplit("_", 1)[0] for eid in chosen_ids[7:]}) == 3
