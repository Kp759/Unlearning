from __future__ import annotations

import torch
import torch.nn as nn

import scripts.run_mcf_rsnr_v1a_oracle as rsnr


def _record(case_id: int, subject: str, relation: str):
    return {
        "case_id": case_id,
        "requested_rewrite": {
            "subject": subject,
            "relation_id": relation,
            "prompt": "{} test",
            "target_true": {"str": "truth"},
            "target_new": {"str": "fake"},
        },
    }


def test_resolve_layer_index_supports_negative_indices():
    assert rsnr.resolve_layer_index(-1, 28) == 27
    assert rsnr.resolve_layer_index(-4, 28) == 24
    assert rsnr.resolve_layer_index(3, 28) == 3


def test_oracle_membership_requires_exact_subject_relation_pair():
    forget = {("Belgium", "P36")}
    assert rsnr.oracle_membership(_record(1, "Belgium", "P36"), forget)
    assert not rsnr.oracle_membership(_record(2, "Belgium", "P38"), forget)
    assert not rsnr.oracle_membership(_record(3, "France", "P36"), forget)


def test_negative_audit_contains_both_specificity_axes():
    forget = [_record(1, "Belgium", "P36")]
    protection = [
        _record(2, "Belgium", "P38"),
        _record(3, "France", "P36"),
        _record(4, "France", "P38"),
    ]
    audit = rsnr.build_oracle_negative_audit(forget, protection)
    assert audit["forget_pairs"] == 1
    assert audit["same_subject_different_relation_negatives"] == 1
    assert audit["same_relation_different_subject_negatives"] == 1
    assert audit["oracle_false_positive_rate_by_construction"] == 0.0


def test_null_adapter_is_exact_noop_at_initialization():
    adapter = rsnr.NullResidualAdapter(8, 2, 2.0, torch.device("cpu"))
    x = torch.randn(3, 4, 8)
    delta = adapter(x)
    assert torch.equal(delta, torch.zeros_like(delta))


class _TupleLayer(nn.Module):
    def forward(self, x):
        return (x + 1.0, torch.tensor(7.0))


def test_oracle_hook_gate_off_returns_exact_layer_output():
    layer = _TupleLayer()
    adapter = rsnr.NullResidualAdapter(8, 2, 2.0, torch.device("cpu"))
    with torch.no_grad():
        adapter.up.weight.fill_(0.25)
    hook = rsnr.OracleNullHook.install(layer, adapter)
    x = torch.randn(2, 3, 8)
    hook.clear()
    baseline = layer(x)[0].detach().clone()
    hook.set(torch.zeros(2), None)
    gated_off = layer(x)[0].detach().clone()
    hook.clear()
    hook.remove()
    assert torch.equal(baseline, gated_off)


def test_oracle_hook_edits_only_gated_batch_rows_and_positions():
    layer = _TupleLayer()
    adapter = rsnr.NullResidualAdapter(8, 2, 2.0, torch.device("cpu"))
    with torch.no_grad():
        adapter.down.weight.fill_(0.1)
        adapter.up.weight.fill_(0.1)
    hook = rsnr.OracleNullHook.install(layer, adapter)
    x = torch.ones(2, 3, 8)
    baseline = x + 1.0
    gate = torch.tensor([1.0, 0.0])
    positions = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
    hook.set(gate, positions)
    edited = layer(x)[0]
    hook.clear()
    hook.remove()
    assert torch.equal(edited[1], baseline[1])
    assert torch.equal(edited[0, :2], baseline[0, :2])
    assert not torch.equal(edited[0, 2], baseline[0, 2])


def test_abstention_is_natural_text_and_not_target_new():
    assert rsnr.ABSTENTION == "I don't know."
    source = open(rsnr.__file__, "r", encoding="utf-8").read()
    assert "target_new_used\": False" in source
