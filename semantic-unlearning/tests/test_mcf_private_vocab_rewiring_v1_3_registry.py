from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "protocols" / "mcf_private_vocab_rewiring_v1_3_registry.json"


def test_registry_locks_multiview_and_leakage_firewall():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["protocol"] == "mcf_private_vocab_rewiring_v1_3_multiview_relation_robust"
    mv = payload["training_multiview"]
    assert mv["views_per_case"] == 5
    assert mv["synthetic_views_per_case"] == 4
    assert mv["worst_view_objective"] is True
    assert mv["heldout_probe_text_used"] is False
    assert mv["official_paraphrase_text_used"] is False
    assert mv["official_neighborhood_text_used"] is False
    assert mv["official_retain_text_used"] is False
    assert mv["full_mcf_readable_by_learner"] is False


def test_registry_forbids_seed1_final_certification():
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    firewall = payload["leakage_firewall"]
    assert firewall["seed1_final_certification_allowed"] is False
    assert firewall["final_certification_requires_new_untouched_seed"] is True
