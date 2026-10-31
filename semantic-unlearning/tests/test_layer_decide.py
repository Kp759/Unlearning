"""scripts/layer_decide.py: the automatic router / write-layer rules on known inputs."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import layer_decide as ld  # noqa: E402


def _entry(layer, f1, fpr, null_fpr, recall=0.9):
    return {"rule": "deployed", "layers": [layer], "deployed_eligibility": "fuzzy",
            "by_eligibility": {"fuzzy": {"eval": {"f1": f1, "fpr": fpr, "recall": recall}, "null_fpr": null_fpr}}}


def test_router_takes_the_earliest_block_on_the_f1_plateau_with_fpr_not_worse():
    sweep = {"sweep": [_entry(10, 0.80, 0.05, 0.02), _entry(16, 0.90, 0.04, 0.02), _entry(19, 0.903, 0.04, 0.02),
                       _entry(22, 0.905, 0.045, 0.02), _entry(26, 0.906, 0.03, 0.02),
                       {"rule": "other", "layers": [30], "by_eligibility": {}},
                       {"rule": "deployed", "layers": [16, 19], "by_eligibility": {}}]}
    rows = ld.sweep_rows(sweep)
    assert [r[0] for r in rows] == [10, 16, 19, 22, 26]
    chosen, detail = ld.pick_router({"mcf": rows})
    # best F1 at L26 (0.906, FPR 0.03); L22 and L19 are within 0.005 F1 but their FPR is more than
    # 0.005 above the best block's; L16 is outside the F1 tolerance -> L26 itself
    assert detail["best_f1_block"] == 26 and chosen == 26
    chosen, detail = ld.pick_router({"mcf": rows}, fpr_tol=0.02)
    assert chosen == 22 and detail["eligible_blocks"] == [22, 26]      # FPR tolerance admits L22 (0.905), not L19


def test_router_averages_f1_over_datasets():
    a = ld.sweep_rows({"sweep": [_entry(16, 0.95, 0.02, 0.01), _entry(20, 0.90, 0.02, 0.01)]})
    b = ld.sweep_rows({"sweep": [_entry(16, 0.80, 0.02, 0.01), _entry(20, 0.92, 0.02, 0.01)]})
    chosen, detail = ld.pick_router({"mcf": a, "zsre": b})
    assert chosen == 20 and abs(detail["per_block"]["20"]["f1"] - 0.91) < 1e-9


def test_write_candidates_within_one_to_router():
    assert ld.write_candidates(24, 21) == [19, 21, 24]
    assert ld.write_candidates(4, 21) == [1, 4]
    assert ld.write_candidates(22, 18) == [17, 18, 19, 22]


def _row(ds, w, method, eff, gen, spe, ppl=11.31):
    return {"dataset": ds, "w": w, "method": method, "Eff": eff, "Gen": gen, "Spe": spe, "PPL": ppl}


def test_write_picks_lowest_gen_among_candidates_that_keep_locality():
    table = [_row("mcf", None, "base", 13.0, 10.0, 20.0),
             _row("mcf", 24, "ours", 0.001, 0.50, 19.9),      # kept
             _row("mcf", 21, "ours", 0.001, 0.05, 19.8),      # kept, lowest Gen -> chosen
             _row("mcf", 19, "ours", 0.001, 0.02, 15.0),      # Spe collapses: not kept
             _row("mcf", 17, "ours", 0.001, 0.01, 19.9, ppl=12.0)]   # PPL drifts: not kept
    w, detail = ld.pick_write(table, default_write=21)
    assert w == 21 and detail["candidates"]["19"]["kept"] is False and detail["candidates"]["17"]["kept"] is False
    assert "write_warning" not in detail


def test_write_falls_back_to_lowest_gen_with_a_warning_and_needs_every_dataset():
    table = [_row("mcf", None, "base", 13.0, 10.0, 20.0), _row("zsre", None, "base", 32.0, 31.0, 30.0),
             _row("mcf", 21, "ours", 0.1, 1.0, 10.0), _row("zsre", 21, "ours", 0.2, 1.5, 29.9),
             _row("mcf", 24, "ours", 0.1, 0.4, 10.0)]            # zsre row missing -> incomplete
    w, detail = ld.pick_write(table, default_write=21)
    assert w == 21 and detail["candidates"]["24"]["complete"] is False and "write_warning" in detail


def test_write_tie_breaks_on_eff_then_distance_to_default():
    table = [_row("mcf", None, "base", 13.0, 10.0, 20.0),
             _row("mcf", 24, "ours", 0.010, 0.10, 20.0), _row("mcf", 21, "ours", 0.005, 0.10, 20.0),
             _row("mcf", 19, "ours", 0.005, 0.10, 20.0)]
    w, _ = ld.pick_write(table, default_write=21)
    assert w == 21
