from scripts.mcf_zero_unlearn_metric_parity import (
    ZERO_UNLEARN_EFF_DEFINITION,
    ZERO_UNLEARN_GEN_DEFINITION,
    apply_zero_unlearn_eff_gen,
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


def test_zero_unlearn_eff_gen_are_post_success_not_complement():
    out = apply_zero_unlearn_eff_gen(_summary())
    assert out["Eff"] == 16.0
    assert out["Gen"] == 14.0
    assert out["SensitivePref_Eff"] == 84.0
    assert out["SensitivePref_Gen"] == 86.0
    assert out["Spe"] == 12.56
    assert "NLL(target_true) > NLL(target_new)" in out["Eff_definition"]
    assert "NLL(target_true) > NLL(target_new)" in out["Gen_definition"]


def test_historical_seed1_base_numbers_reproduce_exactly():
    # Historical ZeroUnlearn-compatible seed1 base output in this repository:
    # Eff=16, Gen=14, with the same post_*_success values.
    out = apply_zero_unlearn_eff_gen(_summary())
    assert (out["Eff"], out["Gen"]) == (16.0, 14.0)


def test_saved_result_patcher_updates_forget_retain_and_legacy_blocks():
    payload = {
        "method": "dummy",
        "forget": _summary(),
        "retain": _summary(),
        "legacy_counterfact": {
            "forget": _summary(),
            "retain": _summary(),
        },
    }
    out = patch_result_payload(payload)
    assert out["forget"]["Eff"] == 16.0
    assert out["forget"]["Gen"] == 14.0
    assert out["retain"]["Eff"] == 16.0
    assert out["legacy_counterfact"]["forget"]["Eff"] == 16.0
    assert out["zero_unlearn_eff_gen_parity"]["applied"] is True


def test_definitions_are_explicit_and_stable():
    assert ZERO_UNLEARN_EFF_DEFINITION.startswith("100 * mean[")
    assert ZERO_UNLEARN_GEN_DEFINITION.startswith("100 * mean[")
