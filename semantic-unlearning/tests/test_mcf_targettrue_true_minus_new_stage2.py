from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
MCF_SCRIPTS = SCRIPTS / "MCF_Scripts"
for path in (SCRIPTS, MCF_SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sure_stage2_sparse_repair_true_minus_new_materialized as paired


def test_true_minus_new_basis_recovers_difference_direction():
    torch.manual_seed(0)
    # Paired fact differences are almost entirely along e0.
    d = torch.randn(32, 1) * torch.tensor([[3.0, 0.0, 0.0, 0.0]])
    d = d + 0.02 * torch.randn(32, 4)
    basis, receipt = paired.build_true_minus_new_basis(d)
    assert receipt["paired_difference_rank"] > 0
    top = basis[0]
    assert abs(float(top[0])) > 0.95
    assert abs(float(top[1])) < 0.2


def test_target_contract_accepts_target_true_sensitive():
    args = argparse.Namespace(
        dataset="mcf",
        mcf_sensitive_field="target_true",
        mcf_reference_field="target_new",
        candidate_ranks="1,2,4,8",
    )
    manifest = {
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        }
    }
    paired._assert_target_contract(args, manifest)


def test_target_contract_rejects_reversed_fields():
    args = argparse.Namespace(
        dataset="mcf",
        mcf_sensitive_field="target_new",
        mcf_reference_field="target_true",
        candidate_ranks="1,2,4,8",
    )
    with pytest.raises(RuntimeError, match="target_true = sensitive"):
        paired._assert_target_contract(args, {})


def test_target_contract_rejects_rank_zero():
    args = argparse.Namespace(
        dataset="mcf",
        mcf_sensitive_field="target_true",
        mcf_reference_field="target_new",
        candidate_ranks="1,2,4,8,0",
    )
    with pytest.raises(RuntimeError, match="rank 0"):
        paired._assert_target_contract(args, {})
