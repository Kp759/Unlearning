from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_mcf_private_vocab_rewiring_v1_3c_fullfit as v13c  # noqa: E402


def _row(case_id: int):
    return {"case_id": case_id, "requested_rewrite": {"relation_id": "P364"}}


def test_curriculum_schedule():
    assert v13c.curriculum_views(1) == 2
    assert v13c.curriculum_views(150) == 2
    assert v13c.curriculum_views(151) == 3
    assert v13c.curriculum_views(300) == 3
    assert v13c.curriculum_views(301) == 5
    assert v13c.curriculum_views(1500) == 5


def test_hard_case_sampler_prefers_current_failures_without_duplicates():
    population = [_row(i) for i in range(50)]
    v13c._FORGET_IDS = set(range(50))
    v13c._LATEST_MARGIN_BY_CASE = {i: (-1.0 if i < 10 else 1.0) for i in range(50)}
    v13c._TRAIN_STEP = 0
    v13c._ACTIVE_TRAIN_VIEWS = 2
    rng = v13c.AdaptiveForgetRandom(123)
    batch = rng.sample(population, 8)
    ids = [int(row["case_id"]) for row in batch]
    assert len(ids) == 8
    assert len(set(ids)) == 8
    assert sum(i < 10 for i in ids) == 6
    assert v13c._TRAIN_STEP == 1
    assert v13c._ACTIVE_TRAIN_VIEWS == 2


def test_hard_case_sampler_backfills_from_hard_when_easy_pool_is_too_small():
    population = [_row(i) for i in range(50)]
    v13c._FORGET_IDS = set(range(50))
    # 49 hard, only one easy: the sampler must still return a full unique batch.
    v13c._LATEST_MARGIN_BY_CASE = {i: (-1.0 if i < 49 else 1.0) for i in range(50)}
    v13c._TRAIN_STEP = 0
    v13c._ACTIVE_TRAIN_VIEWS = 2
    rng = v13c.AdaptiveForgetRandom(321)
    batch = rng.sample(population, 8)
    ids = [int(row["case_id"]) for row in batch]
    assert len(ids) == 8
    assert len(set(ids)) == 8
    assert 49 in ids
    assert sum(i < 49 for i in ids) == 7


def test_non_forget_sampling_is_standard():
    v13c._FORGET_IDS = set(range(50))
    rng = v13c.AdaptiveForgetRandom(7)
    values = list(range(100, 120))
    sample = rng.sample(values, 4)
    assert len(sample) == 4
    assert len(set(sample)) == 4
    assert all(value in values for value in sample)


def test_registry_preserves_scientific_thresholds_and_records_historical_failures():
    registry = json.loads(
        (ROOT / "protocols" / "mcf_private_vocab_rewiring_v1_3c_fullfit_registry.json").read_text(
            encoding="utf-8"
        )
    )
    assert registry["training_multiview"]["views_per_case"] == 5
    assert registry["training_multiview"]["max_true_logprob_drop"] == 3.0
    assert registry["training_multiview"]["max_margin_degradation"] == 1.0
    assert registry["optimization"]["relative_private_row_cap"] == 0.5
    assert registry["optimization"]["retain_topk_kl_weight"] == 20.0
    assert registry["acceptance"]["minimum_worst_training_view_margin"] == 0.1
    assert registry["acceptance"]["worst_training_view_failures_required"] == 0
    assert registry["known_hard_relation_history"]["v1_1_40_of_50_failures"]["P364"] == 3
    assert registry["known_hard_relation_history"]["v1_1_40_of_50_failures"]["P101"] == 2
    assert registry["leakage_firewall"]["final_certification_requires_new_untouched_seed"] is True
