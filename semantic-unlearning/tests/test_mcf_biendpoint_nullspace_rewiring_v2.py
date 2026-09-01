from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_mcf_biendpoint_nullspace_rewiring_v2_split as split  # noqa: E402
import mcf_biendpoint_nullspace_rewiring_v2_core as core  # noqa: E402


class FakeTokenizer:
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2
    unk_token_id = 3

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}

    def __call__(self, text: str):
        tokens = str(text).strip().split()
        ids = [self.bos_token_id]
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab) + 10
            ids.append(self.vocab[token])
        return {"input_ids": ids}


def _record(case_id: int, subject: str, true: str, new: str) -> dict:
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "prompt": "The recorded value for {} is",
            "subject": subject,
            "relation_id": "P27",
            "target_true": {"str": true},
            "target_new": {"str": new},
        },
    }


def test_registry_locks_internal_biendpoint_architecture() -> None:
    registry = json.loads(
        (
            ROOT / "protocols" / "mcf_biendpoint_nullspace_rewiring_v2_registry.json"
        ).read_text()
    )
    architecture = registry["architecture"]
    assert registry["status"] == "training_only_implementation_available_not_executed"
    assert architecture["trainable_parameter_families"] == [
        "selected_input_embedding_rows",
        "selected_lm_head_rows",
    ]
    assert architecture["transformer_frozen"] is True
    assert architecture["contextual_classifier"] is False
    assert architecture["detector"] is False
    assert architecture["actuator"] is False
    assert architecture["external_router"] is False
    assert architecture["inference_sidecar"] is False
    assert architecture["runtime_gate"] is False
    assert registry["row_selection"]["one_delta_per_physical_row"] is True
    assert (
        registry["overlap_policy"]["no_safe_direction_behavior"]
        == "fail_before_training"
    )


def test_endpoint_selection_coalesces_shared_physical_rows() -> None:
    tok = FakeTokenizer()
    rows = core.select_endpoint_rows(
        [
            _record(10, "Shared Alpha", "French", "English"),
            _record(11, "Shared Beta", "English", "German"),
        ],
        tok,
        llama_like=True,
    )
    report = core.endpoint_overlap_report(rows)
    shared_id = tok.vocab["Shared"]
    english_id = tok.vocab["English"]
    assert rows.input_owners[shared_id] == [10, 11]
    assert rows.output_roles[english_id] == {
        "target_true": [11],
        "target_new": [10],
    }
    assert report["shared_input_rows"] == 1
    assert report["cross_role_output_rows"] == 1
    assert report["one_delta_per_physical_input_row"] is True
    assert report["one_delta_per_physical_output_row"] is True


def test_nullspace_projection_and_caps_are_hard_constraints() -> None:
    delta = torch.tensor([[3.0, 4.0, 5.0], [2.0, -1.0, 4.0]])
    bases = [
        torch.tensor([[1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0]]),
    ]
    core.project_rowwise_(delta, bases)
    assert torch.allclose(delta[0] @ bases[0].T, torch.zeros(1), atol=1e-6)
    assert torch.allclose(delta[1] @ bases[1].T, torch.zeros(1), atol=1e-6)
    caps = torch.tensor([2.0, 1.0])
    core.apply_row_caps_(delta, caps)
    assert torch.all(delta.norm(dim=1) <= caps + 1e-6)
    assert core.cap_report(delta, caps)["passed"] is True


def test_output_common_nullspace_annihilates_protected_states() -> None:
    delta = torch.tensor([[1.0, 2.0, 3.0], [-3.0, 1.0, 2.0]])
    protected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    core.project_common_(delta, protected)
    assert torch.allclose(delta @ protected.T, torch.zeros(2, 2), atol=1e-6)


def test_protection_metric_is_zero_at_base_and_detects_change() -> None:
    base_logits = torch.tensor([[2.0, 1.0, -1.0, 0.0]])
    values, ids = torch.topk(base_logits, 3, dim=1)
    base_topk = torch.log_softmax(values, dim=1)
    top1 = base_logits.argmax(dim=1)
    base_top1 = (
        torch.log_softmax(base_logits, dim=1).gather(1, top1[:, None]).squeeze(1)
    )
    kl, drift = core.protection_loss(
        base_logits,
        topk_ids=ids,
        base_topk_log_probs=base_topk,
        base_top1_ids=top1,
        base_top1_log_probs=base_top1,
    )
    assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-7)
    assert torch.allclose(drift, torch.zeros_like(drift), atol=1e-7)
    changed = base_logits.clone()
    changed[0, 1] += 3.0
    changed_kl, changed_drift = core.protection_loss(
        changed,
        topk_ids=ids,
        base_topk_log_probs=base_topk,
        base_top1_ids=top1,
        base_top1_log_probs=base_top1,
    )
    assert changed_kl.item() > 0
    assert abs(changed_drift.item()) > 0


def test_split_reserves_official_retain_and_is_pairwise_disjoint() -> None:
    raw = [
        _record(index, f"subject {index}", f"true {index}", f"new {index}")
        for index in range(40)
    ]
    partitions, official_ids = split.build_partitions(
        raw,
        seed=1,
        forget_num=2,
        official_retain_num=3,
        fit_num=4,
        development_num=3,
        certification_num=2,
    )
    visible: set[int] = set()
    for role, records in partitions.items():
        split.assert_direct_only(records, role=role)
        ids = {int(record["case_id"]) for record in records}
        assert not visible.intersection(ids)
        assert not ids.intersection(official_ids)
        visible.update(ids)
    assert len(partitions["forget"]) == 2
    assert len(official_ids) == 3


def test_development_selection_uses_only_passing_minimum_norm() -> None:
    reports = [
        {"step": 100, "passed": False, "total_delta_norm": 0.5},
        {"step": 200, "passed": True, "total_delta_norm": 2.0},
        {"step": 300, "passed": True, "total_delta_norm": 1.5},
    ]
    assert core.select_development_candidate(reports) == 300
    assert core.select_development_candidate(reports[:1]) is None
