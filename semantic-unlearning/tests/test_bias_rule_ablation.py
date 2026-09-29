"""Bias-rule ablation: plain logistic (stage-1 bias, p >= 0.5) vs calibrated bias."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_bias_rules as cmp  # noqa: E402
import make_raw_logistic_router as mr  # noqa: E402
from linear_router import (  # noqa: E402
    LinearClassifierAssociationBank,
    bias_calibration_record,
    decide_routes,
    fold_threshold_into_bias,
    load_linear_classifier_artifact,
)
from test_linear_router import FACTS, HIDDEN, _FakeModel, _ids, _run  # noqa: E402


def _folded_artifact(t=-2.0, seed=0):
    torch.manual_seed(seed)
    weight = torch.randn(FACTS, HIDDEN)
    stage1 = torch.randn(FACTS)
    deployed, shift = fold_threshold_into_bias(stage1, t)
    facts = [{"id": f"mcf_{i}", "subject": f"S{i}", "relation": "r"} for i in range(FACTS)]
    bank = LinearClassifierAssociationBank(
        _FakeModel(), 0, weight, deployed, feature_mean=torch.zeros(HIDDEN),
        feature_components=None, threshold=0.0,
        subject_patterns=[[(10 + i,)] for i in range(FACTS)], facts=facts,
        rows=torch.randn(FACTS, HIDDEN), ambiguity_margin=0.5, gate_mode="threshold",
        router_fit={"threshold_logit": t}, bias_calibration=bias_calibration_record("global", stage1, shift),
    )
    return bank.artifact(), weight, stage1


def test_raw_artifact_keeps_heads_and_drops_only_the_cutoff():
    folded, weight, stage1 = _folded_artifact(t=-2.0)
    raw, s1, cutoff = mr.raw_artifact(folded)
    assert cutoff == pytest.approx(-2.0)
    assert torch.equal(raw["router_weight"], folded["router_weight"])
    assert torch.allclose(raw["router_bias"], stage1)
    assert torch.equal(raw["rows"], folded["rows"])
    assert raw["threshold"] == 0.0 and raw["bias_calibration"] is None
    assert raw["decision_rule"] == "explicit_threshold"
    assert raw["router_fit"]["threshold_logit"] == 0.0
    assert raw["router_fit"]["folded_router_threshold_logit"] == -2.0


def test_raw_artifact_routes_are_plain_logistic_at_zero():
    folded, weight, stage1 = _folded_artifact(t=-2.0)
    raw, _, _ = mr.raw_artifact(folded)
    _, bank = load_linear_classifier_artifact(_FakeModel(), raw)
    assert bank.decision_rule() == "explicit_threshold"
    hidden, edited = _run(bank)
    ids = _ids()
    eligible = torch.stack([torch.tensor([(10 + i) in row.tolist() for i in range(FACTS)])
                            for row in ids])
    logits = torch.nn.functional.normalize(hidden[:, -1].float(), dim=-1) @ weight.T + stage1
    expected = decide_routes(logits, eligible, 0.0, 0.5)
    fired = (edited != hidden).any(dim=-1).any(dim=-1)
    assert torch.equal(fired, expected["active"])


def test_inconsistent_calibration_record_is_rejected():
    folded, _, _ = _folded_artifact()
    folded["router_bias"] = folded["router_bias"] + 1.0
    with pytest.raises(ValueError):
        mr.raw_artifact(folded)


def test_subject_gate_router_is_rejected():
    folded, _, _ = _folded_artifact()
    folded["gate_mode"] = "subject"
    with pytest.raises(ValueError):
        mr.folded_cutoff(folded)


def test_diagnostics_match_the_hand_example():
    # Head 0: a positive at z = -1.2 (misses at p >= 0.5, fires at t = -2) and a
    # negative at z = -1.5 (fires only under the calibrated cutoff).
    z = torch.tensor([[-1.2, -9.0], [-1.5, -9.0], [0.5, -9.0], [-3.0, -9.0]])
    eligible = torch.tensor([[True, False]] * 4)
    owner = torch.tensor([0, -1, 0, -1])
    facts = [{"id": "a", "subject": "S"}, {"id": "b", "subject": "T"}]
    out, raw_d, fold_d = mr.rule_diagnostics(z, eligible, owner, ["audit"] * 4, -2.0, 0.5, facts)
    s = out["audit"]
    assert s["positives_correct"] == {"raw": 1, "folded": 2}
    assert s["positives_rescued_by_calibration"] == 1
    assert s["positive_own_logit_below_zero"] == 1
    assert s["positive_own_logit_in_cutoff_to_zero"] == 1
    assert s["negatives_firing"] == {"raw": 0, "folded": 1}
    assert s["negatives_added_by_calibration"] == 1
    assert s["raw"]["correct_route"]["rate"] == pytest.approx(0.5)
    assert s["folded"]["false_activation_on_negative_control"]["rate"] == pytest.approx(0.5)


def _run_dir(path, gen, spe):
    path.mkdir(parents=True)
    (path / "association_manifest.json").write_text(json.dumps({
        "views_excluded_unrouted": [], "untrainable_fact_ids": []}))
    (path / "official_mcf_eval.json").write_text(json.dumps({
        "forget": {"Eff": 0.0, "Gen": gen, "Spe": spe}, "retain": {"Eff": 90.0, "Gen": 80.0},
        "forget_PPL": 10.0}))


def test_comparison_picks_the_better_rule_per_metric(tmp_path):
    z = torch.tensor([[-1.2, -9.0], [-1.5, -9.0], [0.5, -9.0], [-3.0, -9.0]])
    eligible = torch.tensor([[True, False]] * 4)
    owner = torch.tensor([0, -1, 0, -1])
    facts = [{"id": "a", "subject": "S"}, {"id": "b", "subject": "T"}]
    by_split, _, _ = mr.rule_diagnostics(z, eligible, owner, ["audit"] * 4, -2.0, 0.5, facts)
    router = tmp_path / "router_raw"
    router.mkdir()
    (router / "bias_rule_ablation.json").write_text(json.dumps(mr._json_safe({
        "folded_cutoff_t": -2.0, "by_split": by_split,
        "runtime_parity_raw": {"route_mismatches": 0},
        "folded_routes_recomputed_vs_stored_mismatches": 0})))
    _run_dir(tmp_path / "folded", gen=2.0, spe=40.0)
    _run_dir(tmp_path / "raw_swap", gen=5.0, spe=45.0)
    _run_dir(tmp_path / "raw", gen=6.0, spe=46.0)
    prefix = tmp_path / "abl" / "lbfgs" / "mcf" / "seed1" / "L19" / "comparison"
    cmp.main(["--dataset", "mcf", "--optimizer", "lbfgs",
              "--folded-router", str(tmp_path / "folded"), "--folded-run", str(tmp_path / "folded"),
              "--raw-router", str(router), "--raw-swap-run", str(tmp_path / "raw_swap"),
              "--raw-run", str(tmp_path / "raw"), "--out-prefix", str(prefix)])
    result = json.loads(prefix.with_suffix(".json").read_text())
    rows = {r["metric"]: r for r in result["metrics"]}
    assert rows["forget_Gen"]["better[raw]"] == "folded"      # lower is better
    assert rows["forget_Spe"]["better[raw]"] == "raw"         # higher is better
    assert rows["forget_Eff"]["better[raw]"] == "tie"
    assert result["router"]["by_split"]["audit"]["positives_rescued_by_calibration"] == 1
    assert any("folded better on forget_Gen" in line for line in result["overall"])
    assert cmp.main(["--collect", str(tmp_path / "abl")]) == 0
    summary = (tmp_path / "abl" / "bias_rule_summary.md").read_text()
    assert "raw, rows retrained" in summary and "folded (shipped)" in summary
