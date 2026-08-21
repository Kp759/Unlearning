from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sure_stage2_sparse_repair_subject_contrast_materialized as subject_contrast


def test_subject_contrast_basis_has_expected_direction():
    d = torch.tensor([
        [3.0, 0.1, 0.0, 0.0],
        [2.5, -0.1, 0.0, 0.0],
        [4.0, 0.2, 0.0, 0.0],
    ])
    basis, receipt = subject_contrast.build_subject_contrast_basis(d)
    assert receipt["subject_contrast_rank"] > 0
    assert abs(float(basis[0, 0])) > 0.95


def test_donor_indices_exclude_self_and_same_subject():
    subjects = ["A", "B", "C", "D"]
    donors = subject_contrast._donor_indices(0, subjects, 2)
    assert donors == [1, 2]
    assert 0 not in donors
