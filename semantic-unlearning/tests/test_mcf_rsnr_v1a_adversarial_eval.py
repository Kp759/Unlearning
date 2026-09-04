from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_rsnr_v1a_adversarial_eval as adv  # noqa: E402


def test_retrieval_attack_templates_do_not_embed_a_specific_answer():
    true_answer = "Brussels"
    reference = "Paris"
    base = "The capital of Belgium is"
    for instruction in adv.RETRIEVAL_ATTACKS.values():
        prompt = adv.build_attack_prompt(base, instruction)
        assert true_answer.casefold() not in prompt.casefold()
        assert reference.casefold() not in prompt.casefold()


def test_forced_choice_is_explicitly_answer_present():
    prompt = adv.build_forced_choice_prompt(
        "The capital of Belgium is", "Brussels", "Paris"
    )
    assert "Brussels" in prompt
    assert "Paris" in prompt


def test_native_idk_eff_gen_are_failure_percentages_lower_is_better():
    rows = [
        {"group": "rewrite", "idk_vs_true_margin": 2.0},
        {"group": "rewrite", "idk_vs_true_margin": -0.2},
        {"group": "paraphrase", "idk_vs_true_margin": 0.05},
        {"group": "paraphrase", "idk_vs_true_margin": 1.0},
    ]
    result = adv.summarize_idk_metrics(rows)
    assert result["Eff_IDK"] == 50.0
    assert result["Eff_IDK_buffered_0p1"] == 50.0
    assert result["Gen_IDK"] == 0.0
    assert result["Gen_IDK_buffered_0p1"] == 50.0
    assert result["Eff_IDK_prompt_count"] == 2
    assert result["Gen_IDK_prompt_count"] == 2


def test_all_positive_idk_margins_give_zero_native_eff_gen():
    rows = [
        *({"group": "rewrite", "idk_vs_true_margin": 4.0} for _ in range(50)),
        *({"group": "paraphrase", "idk_vs_true_margin": 3.0} for _ in range(100)),
    ]
    result = adv.summarize_idk_metrics(rows)
    assert result["Eff_IDK"] == 0.0
    assert result["Gen_IDK"] == 0.0
    assert result["Eff_IDK_buffered_0p1"] == 0.0
    assert result["Gen_IDK_buffered_0p1"] == 0.0


def test_script_documents_metric_distinction_and_greedy_scope():
    source = Path(adv.__file__).read_text(encoding="utf-8")
    assert "target_true vs target_new" in source
    assert "RSNR_Eff_IDK_Gen_IDK" in source
    assert '"greedy_only_in_this_script": True' in source
