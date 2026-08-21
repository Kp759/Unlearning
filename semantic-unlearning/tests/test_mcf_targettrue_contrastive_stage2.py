from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import sure_stage2_sparse_repair_contrastive_materialized as contrastive


def test_contrastive_basis_prefers_sensitive_specific_direction():
    torch.manual_seed(0)
    # Sensitive states are dominated by e0; utility states are dominated by e1.
    hf = 4.0 * torch.randn(64, 1) * torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    hf = hf + 0.05 * torch.randn(64, 4)
    h0 = 4.0 * torch.randn(128, 1) * torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    h0 = h0 + 0.05 * torch.randn(128, 4)

    basis, receipt = contrastive.build_contrastive_basis(
        hf,
        h0,
        ridge_ratio=1e-3,
    )
    assert receipt["forget_span_rank"] > 0
    top = basis[0]
    # Sign is arbitrary; the leading contrastive direction should align with e0.
    assert abs(float(top[0])) > 0.9
    assert abs(float(top[1])) < 0.3


def test_target_contract_rejects_reversed_mcf_fields():
    args = argparse.Namespace(
        dataset="mcf",
        mcf_sensitive_field="target_new",
        mcf_reference_field="target_true",
        candidate_ranks="2,8,16",
    )
    with pytest.raises(RuntimeError, match="target_true = sensitive"):
        contrastive._assert_target_contract(args, {})


def test_target_contract_rejects_unrestricted_rank_zero():
    args = argparse.Namespace(
        dataset="mcf",
        mcf_sensitive_field="target_true",
        mcf_reference_field="target_new",
        candidate_ranks="2,8,0",
    )
    with pytest.raises(RuntimeError, match="rank 0"):
        contrastive._assert_target_contract(args, {})


def test_target_contract_accepts_unswapped_target_true_sensitive_manifest():
    args = argparse.Namespace(
        dataset="mcf",
        mcf_sensitive_field="target_true",
        mcf_reference_field="target_new",
        candidate_ranks="2,8,16",
    )
    manifest = {
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        }
    }
    contrastive._assert_target_contract(args, manifest)
