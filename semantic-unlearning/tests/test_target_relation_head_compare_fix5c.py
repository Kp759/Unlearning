from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mcf_target_relation_head_compare_fix5c_seed1.py"
spec = importlib.util.spec_from_file_location("fix5c", SCRIPT)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)

Row = m.Row


def row(*, subject: str, text: str, masked: str, relation: str = "P30", forbidden: bool = False, case_id: int = 1) -> Row:
    return Row(
        text=text,
        subject=subject,
        relation=relation,
        forbidden=forbidden,
        kind="same_subject_different_relation",
        family="nominalized_question",
        case_id=case_id,
        masked=masked,
        candidate=True,
    )


def test_policy_manifest_preserves_different_subjects_with_same_masked_input():
    masked = "On which continent is TARGET_ENTITY located?"
    rows = [
        row(subject="Belgium", text="On which continent is Belgium located?", masked=masked, case_id=1),
        row(subject="France", text="On which continent is France located?", masked=masked, case_id=2),
    ]
    out = m.policy_manifest(rows)
    assert len(out) == 2
    assert {r.subject for r in out} == {"Belgium", "France"}


def test_policy_manifest_removes_only_true_duplicate_policy_instances():
    r = row(
        subject="Belgium",
        text="On which continent is Belgium located?",
        masked="On which continent is TARGET_ENTITY located?",
        case_id=1,
    )
    out = m.policy_manifest([r, r])
    assert out == [r]


def test_feature_cache_shares_masked_encoding_without_collapsing_policy_rows():
    masked = "On which continent is TARGET_ENTITY located?"
    rows = [
        row(subject="Belgium", text="On which continent is Belgium located?", masked=masked, case_id=1),
        row(subject="France", text="On which continent is France located?", masked=masked, case_id=2),
    ]
    policy = m.policy_manifest(rows)
    texts, idx = m.feature_index({"policy_validation": policy})
    assert texts == [masked]
    assert idx["policy_validation"] == [0, 0]
    assert len(policy) == 2


def test_mlp_returns_logits_and_eval_disables_dropout_randomness():
    head = m.RelationMLP(input_dim=8, num_classes=4, hidden_dim=6, dropout=0.5)
    x = torch.randn(5, 8)
    head.eval()
    a = head(x)
    b = head(x)
    assert a.shape == (5, 4)
    assert torch.equal(a, b)


def test_multiclass_margin_is_invariant_to_common_logit_shift():
    logits = torch.tensor([[1.0, 4.0, 2.0], [3.0, -1.0, 2.5]])
    pred1, margin1 = m.base.margin(logits)
    pred2, margin2 = m.base.margin(logits + 123.456)
    assert torch.equal(pred1, pred2)
    assert torch.allclose(margin1, margin2, atol=1e-5)


def test_policy_identity_changes_when_subject_changes_even_if_everything_else_matches():
    common = dict(
        text="Ask TARGET about relation.",
        masked="Ask TARGET_ENTITY about relation.",
        relation="P30",
        forbidden=False,
        case_id=1,
    )
    a = row(subject="A", **common)
    b = row(subject="B", **common)
    assert m.policy_identity(a) != m.policy_identity(b)
