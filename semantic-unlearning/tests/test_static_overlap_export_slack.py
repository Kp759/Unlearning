"""FP32 export boundary tolerance must never alter the training budget."""
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from static_overlap_training import (
    EXPORT_FP32_NUMERIC_SLACK, TrainConfig, within_budgets, within_export_budgets,
)


def rows(nll_increase, kl):
    return [{"id": "protected", "role": "retain", "base_nll": 2.0,
             "nll": 2.0 + nll_increase, "nll_increase": nll_increase, "kl": kl}]


@pytest.mark.parametrize("nll,kl,classification", [
    (.049999, .009999, "nominal_pass"),
    (.05, .01, "nominal_pass"),
    (.05000114440917969, 9.294498158851638e-05, "numerical_boundary_pass"),
    (.05 + 5e-6, .01, "numerical_boundary_pass"),
    (.05, .01 + 5e-6, "numerical_boundary_pass"),
    (.050004, .010004, "numerical_boundary_pass"),
    (math.nextafter(.05 + 5e-6, math.inf), .009, "retention_failure"),
    (.049, math.nextafter(.01 + 5e-6, math.inf), "retention_failure"),
    (.051, .001, "retention_failure"),
    (.04, .011, "retention_failure"),
])
def test_export_boundary_classification_and_raw_report(nll, kl, classification):
    config = TrainConfig()
    before = asdict(config)
    inputs = rows(nll, kl)
    original = deepcopy(inputs)
    passed, report = within_export_budgets(inputs, config, torch.float32)
    assert passed == (classification != "retention_failure")
    assert report["classification"] == classification
    assert report["nominal_budgets_passed"] == (classification == "nominal_pass")
    assert report["passed_with_numerical_slack"] == (classification == "numerical_boundary_pass")
    assert report["numerical_slack"] == 5e-6
    assert report["nominal_retain_nll_budget"] == .05
    assert report["nominal_retain_kl_budget"] == .01
    assert report["observed_max_retained_nll_increase"] == nll
    assert report["observed_max_retained_kl"] == kl
    assert report["max_retained_nll_increase"] == nll  # Existing report consumers.
    assert report["max_retained_kl"] == kl
    assert json.loads(json.dumps(report, allow_nan=False)) == report
    assert asdict(config) == before
    assert inputs == original


def test_reported_ec2_boundary_pass_does_not_pass_training_constraints():
    config = TrainConfig()
    observed = rows(.05000114440917969, 9.294498158851638e-05)
    assert not within_budgets(observed, config)[0]
    assert within_export_budgets(observed, config, torch.float32)[0]
    assert not within_budgets(observed, config)[0]  # No config mutation after export.
    assert not hasattr(config, "export_numeric_slack")
    assert EXPORT_FP32_NUMERIC_SLACK == 5e-6


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float64])
def test_slack_is_only_enabled_for_fp32_verification(dtype):
    passed, report = within_export_budgets(rows(.050001, .001), TrainConfig(), dtype)
    assert not passed
    assert report["numerical_slack"] == 0.0
    assert not report["passed_with_numerical_slack"]


@pytest.mark.parametrize("key", ["nll", "base_nll", "nll_increase", "kl"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_numerical_slack_cannot_accept_nonfinite_rows(key, invalid):
    observed = rows(.04, .001)
    observed[0][key] = invalid
    passed, report = within_export_budgets(observed, TrainConfig(), torch.float32)
    assert not passed
    assert report["classification"] == "retention_failure"
    assert not report["passed_with_numerical_slack"]


def test_language_anchors_are_also_checked_and_empty_protection_is_rejected():
    observed = rows(.04, .001)
    language = rows(.050006, .001)[0]
    language["role"] = "language"
    assert not within_export_budgets(observed + [language], TrainConfig(), torch.float32)[0]
    with pytest.raises(ValueError, match="empty anchor"):
        within_export_budgets([], TrainConfig(), torch.float32)
