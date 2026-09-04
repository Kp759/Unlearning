from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "protocols" / "mcf_private_vocab_rewiring_v1_3b_10view_registry.json"


def test_v13b_is_clean_10_view_ablation():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["protocol"] == "mcf_private_vocab_rewiring_v1_3_multiview_relation_robust"
    assert payload["variant"] == "v1_3b_10view_worst1"

    mv = payload["training_multiview"]
    assert mv["views_per_case"] == 10
    assert mv["canonical_views_per_case"] == 1
    assert mv["synthetic_views_per_case"] == 9
    assert mv["additional_views_per_case"] == 9
    assert mv["worst_view_objective"] is True
    assert mv["worst_views_optimized_per_case"] == 1
    assert mv["max_true_logprob_drop"] == 3.0
    assert mv["max_margin_degradation"] == 1.0
    assert mv["heldout_probe_text_used"] is False
    assert mv["official_paraphrase_text_used"] is False
    assert mv["official_neighborhood_text_used"] is False
    assert mv["official_retain_text_used"] is False
    assert mv["heldout_fallback_permitted"] is False

    opt = payload["optimization"]
    assert opt["views_per_case"] == 10
    assert opt["relative_private_row_cap"] == 0.5
    assert opt["retain_topk_kl_weight"] == 20.0

    fw = payload["leakage_firewall"]
    assert fw["split_builder_is_only_process_allowed_full_mcf_source"] is True
    assert fw["official_paraphrases_serialized_into_training_protocol"] is False
    assert fw["official_neighborhoods_serialized_into_training_protocol"] is False
    assert fw["official_retain_text_serialized_into_training_protocol"] is False
    assert fw["seed1_final_certification_allowed"] is False
    assert fw["final_certification_requires_new_untouched_seed"] is True
