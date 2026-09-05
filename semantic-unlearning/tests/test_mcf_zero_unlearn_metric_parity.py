import math

import pytest

from scripts.mcf_zero_unlearn_metric_parity import (
    ZERO_UNLEARN_EFF_DEFINITION,
    ZERO_UNLEARN_GEN_DEFINITION,
    apply_zero_unlearn_eff_gen,
    compute_zero_unlearn_paper_eff_gen,
    patch_result_payload,
)


def _summary():
    return {
        "post_rewrite_success": [16.0, 36.66],
        "post_paraphrase_success": [14.0, 26.53],
        "post_rewrite_sensitive_pref": [84.0, 36.66],
        "post_paraphrase_sensitive_pref": [86.0, 26.53],
        "Eff": 84.0,
        "Gen": 86.0,
        "Spe": 12.56,
    }


def _nll(prob):
    return -math.log(float(prob))


def _raw():
    # Case-macro Eff = mean(0.10, 0.20) * 100 = 15.0
    # Case-macro Gen = mean(mean(.10,.20), mean(.30,.50)) * 100 = 27.5
    return [
        {
            "post": {
                "rewrite_prompts_probs": [
                    {"target_true": _nll(0.10), "target_new": _nll(0.01)}
                ],
                "paraphrase_prompts_probs": [
                    {"target_true": _nll(0.10), "target_new": _nll(0.01)},
                    {"target_true": _nll(0.20), "target_new": _nll(0.02)},
                ],
            }
        },
        {
            "post": {
                "rewrite_prompts_probs": [
                    {"target_true": _nll(0.20), "target_new": _nll(0.02)}
                ],
                "paraphrase_prompts_probs": [
                    {"target_true": _nll(0.30), "target_new": _nll(0.03)},
                    {"target_true": _nll(0.50), "target_new": _nll(0.05)},
                ],
            }
        },
    ]


def test_zero_unlearn_paper_eff_gen_are_residual_true_likelihood():
    metric = compute_zero_unlearn_paper_eff_gen(_raw())
    assert metric["Eff"] == pytest.approx(15.0)
    assert metric["Gen"] == pytest.approx(27.5)

    out = apply_zero_unlearn_eff_gen(_summary(), _raw())
    assert out["Eff"] == pytest.approx(15.0)
    assert out["Gen"] == pytest.approx(27.5)
    assert out["CF_EditSuccess_Eff"] == 16.0
    assert out["CF_EditSuccess_Gen"] == 14.0
    assert out["SensitivePref_Eff"] == 84.0
    assert out["SensitivePref_Gen"] == 86.0
    assert out["Legacy_Spe_ProbabilityDiff"] == 12.56
    assert out["Spe_paper_parity_available"] is False


def test_aggregated_summary_is_rejected_without_raw_rows():
    with pytest.raises(ValueError, match="raw metric_data"):
        apply_zero_unlearn_eff_gen(_summary())


def test_saved_result_patcher_updates_forget_retain_and_legacy_blocks():
    payload = {
        "method": "dummy",
        "forget": _summary(),
        "retain": _summary(),
        "forget_raw": _raw(),
        "retain_raw": _raw(),
        "legacy_counterfact": {
            "forget": _summary(),
            "retain": _summary(),
        },
    }
    out = patch_result_payload(payload)
    assert out["forget"]["Eff"] == pytest.approx(15.0)
    assert out["forget"]["Gen"] == pytest.approx(27.5)
    assert out["retain"]["Eff"] == pytest.approx(15.0)
    assert out["legacy_counterfact"]["forget"]["Eff"] == pytest.approx(15.0)
    assert out["zero_unlearn_eff_gen_parity"]["applied"] is True
    assert out["zero_unlearn_eff_gen_parity"]["paper_spe_requires_argmax_rerun"] is True


def test_definitions_are_explicit_and_stable():
    assert "exp(-NLL(target_true))" in ZERO_UNLEARN_EFF_DEFINITION
    assert "exp(-NLL(target_true))" in ZERO_UNLEARN_GEN_DEFINITION
