"""The calibrated threshold folded into each classifier's bias.

The claims these protect:
  * b' = b - t with the standard rule z' >= 0 (p >= 0.5) qualifies exactly the
    same (prompt, head) pairs as z >= t;
  * with one global cutoff every route is identical, through the real hook;
  * with per-association cutoffs the qualifying heads are the same and the
    best one is ranked by its calibrated logit;
  * a calibrated-bias bank only accepts the standard rule, round-trips through
    its artifact, and records the stage-1 bias and the shift;
  * existing explicit-threshold runs can be folded without refitting.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compress_residual_bank import check_frozen_config  # noqa: E402
from fold_linear_router_bias import fold_artifact, main as fold_main  # noqa: E402
from linear_router import (  # noqa: E402
    LinearClassifierAssociationBank,
    bias_calibration_record,
    decide_routes,
    fold_threshold_into_bias,
    load_linear_classifier_artifact,
)

HIDDEN = 16
FACTS = 5
WIDTH = 6


class _Block(nn.Module):
    def forward(self, hidden):
        return (hidden,)


class _Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Block()])


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Inner()
        self.placeholder = nn.Parameter(torch.zeros(1))


def _parts(seed=0):
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randn(FACTS, HIDDEN, generator=g),
        torch.randn(FACTS, generator=g),
        torch.randn(FACTS, HIDDEN, generator=g),
    )


def _facts():
    return [{"id": f"mcf_{i}", "subject": f"S{i}", "relation": "r", "object": f"o{i}"}
            for i in range(FACTS)]


def _bank(weight, bias, rows, threshold=0.0, per_head=None, calibration=None, margin=0.5):
    return LinearClassifierAssociationBank(
        _FakeModel(), 0, weight, bias,
        feature_mean=torch.zeros(HIDDEN), feature_components=None,
        threshold=threshold, subject_patterns=[[(10 + i,)] for i in range(FACTS)],
        facts=_facts(), rows=rows, ambiguity_margin=margin,
        per_head_thresholds=per_head, bias_calibration=calibration,
    )


def _ids():
    # Several prompts naming one or two protected subjects, one naming none.
    return torch.tensor([
        [10, 1, 2, 3, 4, 5],
        [11, 12, 2, 3, 4, 5],
        [13, 1, 2, 3, 4, 5],
        [99, 1, 2, 3, 4, 5],
        [14, 10, 2, 3, 4, 5],
        [12, 1, 2, 3, 4, 5],
    ])


def _run(bank, hidden):
    ids = _ids()
    bank.bind(ids, attention_mask=torch.ones_like(ids))
    edited = bank._hook(None, None, (hidden.clone(),))[0]
    routes = [list(r) for r in bank.last_active_fact_indices]
    bank.unbind()
    return edited, routes


def test_folding_keeps_qualifying_pairs_and_global_routes():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(400, FACTS, generator=g) * 3
    eligible = torch.rand(400, FACTS, generator=g) > 0.4
    bias = torch.randn(FACTS, generator=g)
    t = -1.3
    folded_bias, shift = fold_threshold_into_bias(bias, t)
    assert torch.allclose(shift, torch.full((FACTS,), t))
    assert torch.allclose(folded_bias, bias - t)
    folded_logits = logits - (bias - folded_bias)          # same weights, new bias
    explicit = decide_routes(logits, eligible, t, 0.5)
    folded = decide_routes(folded_logits, eligible, 0.0, 0.5)
    assert torch.equal(explicit["active"], folded["active"])
    active = explicit["active"]
    assert torch.equal(explicit["fact"][active], folded["fact"][active])
    assert torch.equal(eligible & (logits >= t), eligible & (folded_logits >= 0))


def test_per_head_folding_keeps_qualifying_pairs_and_ranks_by_calibrated_logit():
    logits = torch.tensor([[1.0, 0.8, -9.0, -9.0, -9.0]])
    eligible = torch.tensor([[True, True, False, False, False]])
    t = torch.tensor([0.9, -1.0, 0.0, 0.0, 0.0])
    explicit = decide_routes(logits, eligible, t, 0.1)
    folded = decide_routes(logits - t, eligible, 0.0, 0.1)
    assert int(explicit["qualifying"][0]) == int(folded["qualifying"][0]) == 2
    assert int(explicit["fact"][0]) == 0          # raw 1.0 beats raw 0.8
    assert int(folded["fact"][0]) == 1            # calibrated 1.8 beats 0.1


def test_global_calibrated_bias_bank_routes_exactly_like_explicit_threshold():
    weight, bias, rows = _parts()
    t = -0.7
    explicit = _bank(weight, bias, rows, threshold=t)
    folded_bias, shift = fold_threshold_into_bias(bias, t)
    record = bias_calibration_record("global", bias, shift)
    folded = _bank(weight, folded_bias, rows, threshold=0.0, calibration=record)
    torch.manual_seed(3)
    hidden = torch.randn(_ids().shape[0], WIDTH, HIDDEN) * 2
    e_edit, e_routes = _run(explicit, hidden)
    f_edit, f_routes = _run(folded, hidden)
    assert any(e_routes) and not all(e_routes)
    assert e_routes == f_routes
    assert torch.equal(e_edit, f_edit)

    art = folded.artifact()
    assert art["threshold"] == 0.0 and art["per_head_thresholds"] is None
    assert art["decision_rule"] == "calibrated_bias"
    assert art["threshold_policy"] == "global"
    assert art["routing_policy"].endswith("calibrated_bias_global_p_ge_0.5")
    assert art["bias_calibration"]["global_shift"] == pytest.approx(t)
    assert torch.allclose(art["bias_calibration"]["stage1_bias"], bias)
    _, reloaded = load_linear_classifier_artifact(_FakeModel(), art)
    assert _run(reloaded, hidden)[1] == f_routes


def test_calibrated_bias_bank_only_accepts_the_standard_rule():
    weight, bias, rows = _parts()
    record = bias_calibration_record("global", bias, torch.zeros(FACTS))
    with pytest.raises(ValueError, match="logit >= 0"):
        _bank(weight, bias, rows, threshold=-1.0, calibration=record)
    with pytest.raises(ValueError, match="logit >= 0"):
        _bank(weight, bias, rows, threshold=0.0, per_head=torch.zeros(FACTS), calibration=record)
    with pytest.raises(ValueError, match="threshold gate"):
        LinearClassifierAssociationBank(
            _FakeModel(), 0, weight, bias, feature_mean=torch.zeros(HIDDEN),
            feature_components=None, threshold=0.0,
            subject_patterns=[[(10 + i,)] for i in range(FACTS)], facts=_facts(),
            rows=rows, gate_mode="subject", bias_calibration=record,
        )
    with pytest.raises(ValueError, match="finite"):
        fold_threshold_into_bias(bias, float("-inf"))


def test_explicit_threshold_banks_keep_their_old_policy_names():
    weight, bias, rows = _parts()
    assert _bank(weight, bias, rows, threshold=-1.0).artifact()["routing_policy"].endswith(
        "global_threshold")
    per_head = _bank(weight, bias, rows, threshold=-1.0, per_head=torch.zeros(FACTS)).artifact()
    assert per_head["routing_policy"].endswith("per_head_thresholds")
    assert per_head["decision_rule"] == "explicit_threshold"


def _write_run(tmp_path, bank, name="run"):
    run = tmp_path / name
    run.mkdir()
    torch.save(bank.artifact(), run / "fact_association_embeddings.pt")
    (run / "association_manifest.json").write_text(json.dumps({"protocol_id": "test"}))
    return run


def test_folding_an_existing_global_run_keeps_every_route(tmp_path):
    weight, bias, rows = _parts(seed=4)
    source = _bank(weight, bias, rows, threshold=-0.4)
    run = _write_run(tmp_path, source)
    out = tmp_path / "folded"
    assert fold_main(["--run-dir", str(run), "--output-dir", str(out)]) == 0
    art = torch.load(out / "fact_association_embeddings.pt", weights_only=False)
    manifest = json.loads((out / "association_manifest.json").read_text())
    assert manifest["decision_rule"] == "calibrated_bias"
    assert manifest["protocol_id"] == "test"
    assert torch.equal(art["rows"], source.artifact()["rows"])
    _, folded = load_linear_classifier_artifact(_FakeModel(), art)
    torch.manual_seed(9)
    hidden = torch.randn(_ids().shape[0], WIDTH, HIDDEN) * 2
    assert _run(source, hidden)[1] == _run(folded, hidden)[1]
    with pytest.raises(ValueError, match="already"):
        fold_artifact(art)
    with pytest.raises(FileExistsError):
        fold_main(["--run-dir", str(run), "--output-dir", str(out)])


def test_folding_a_per_head_run_records_the_policy(tmp_path):
    weight, bias, rows = _parts(seed=5)
    per_head = torch.linspace(-1.0, 0.5, FACTS)
    art, summary = fold_artifact(_bank(weight, bias, rows, threshold=-0.2,
                                       per_head=per_head).artifact())
    assert summary["policy"] == "per_head" and not summary["routes_identical_to_source"]
    assert art["threshold_policy"] == "per_head"
    assert torch.allclose(art["router_bias"], bias - per_head)
    assert art["bias_calibration"]["global_shift"] is None
    # The frozen-config check still recognises a folded per-association run.
    with pytest.raises(ValueError, match="per-head"):
        check_frozen_config(art)
    assert check_frozen_config(art, allow_per_head=True) == ("threshold", "per_head")
    subject = _bank(weight, bias, rows).artifact()
    subject["gate_mode"] = "subject"
    with pytest.raises(ValueError, match="subject gate"):
        fold_artifact(subject)
