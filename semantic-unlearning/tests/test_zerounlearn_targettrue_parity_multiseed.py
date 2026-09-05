import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_zerounlearn_targettrue_parity_mcf_multiseed import (
    _macro_correctness,
    aggregate,
    metric_families,
    seed_paths,
)


def _raw_rows():
    return [
        {
            "post": {
                "rewrite_prompts_probs": [{"target_true": 2.302585092994046}],
                "paraphrase_prompts_probs": [
                    {"target_true": 2.302585092994046},
                    {"target_true": 1.6094379124341003},
                ],
                "rewrite_prompts_correct": [False],
                "paraphrase_prompts_correct": [False, True],
                "neighborhood_prompts_correct": [True, True],
            }
        },
        {
            "post": {
                "rewrite_prompts_probs": [{"target_true": 1.6094379124341003}],
                "paraphrase_prompts_probs": [
                    {"target_true": 2.302585092994046},
                    {"target_true": 2.302585092994046},
                ],
                "rewrite_prompts_correct": [True],
                "paraphrase_prompts_correct": [False, False],
                "neighborhood_prompts_correct": [True, False],
            }
        },
    ]


def test_macro_correctness_is_case_macro_percent():
    raw = _raw_rows()
    assert _macro_correctness(raw, "rewrite_prompts_correct") == 50.0
    assert _macro_correctness(raw, "paraphrase_prompts_correct") == 25.0
    assert _macro_correctness(raw, "neighborhood_prompts_correct") == 75.0


def test_metric_families_keep_eq16_proxy_and_table_accuracy_separate():
    payload = {"forget_raw": _raw_rows(), "forget_PPL": 12.5}
    out = metric_families(payload)
    eq16 = out["eq16_style_residual_likelihood_proxy"]
    table = out["released_table_style_accuracy"]
    assert round(eq16["Eff"], 6) == 15.0
    assert round(eq16["Gen"], 6) == 12.5
    assert table["Eff"] == 50.0
    assert table["Gen"] == 25.0
    assert table["Spe"] == 75.0
    assert out["PPL"] == 12.5


def test_seed_paths_are_seed_specific():
    paths = seed_paths(Path("outputs/x"), 7)
    assert paths["dir"] == Path("outputs/x/seed7")
    assert "seed7" in paths["zero"].name
    assert paths["provenance"].name == "provenance.json"


def test_aggregate_uses_population_std_and_keeps_metric_names_explicit():
    rows = [
        {"seed": 1, "eq16_Eff": 1.0, "eq16_Gen": 2.0, "table_Eff": 3.0, "table_Gen": 4.0, "table_Spe": 5.0, "PPL": 10.0},
        {"seed": 2, "eq16_Eff": 3.0, "eq16_Gen": 4.0, "table_Eff": 5.0, "table_Gen": 6.0, "table_Spe": 7.0, "PPL": 12.0},
    ]
    out = aggregate(rows)
    assert out["n_seeds"] == 2
    assert out["eq16_Eff"]["mean"] == 2.0
    assert out["eq16_Eff"]["std_population"] == 1.0
    assert out["table_Spe"]["mean"] == 6.0
