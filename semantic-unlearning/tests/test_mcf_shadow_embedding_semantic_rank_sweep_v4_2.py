from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import mcf_shadow_embedding_semantic_rank_sweep_v4_2_core as core
import mcf_shadow_embedding_semantic_router_core as semantic_core
import mcf_shadow_relation_prompts as prompts
import run_mcf_shadow_embedding_semantic_rank_sweep_v4_2 as train


def _records():
    return [
        {"case_id": 1, "subject": "Alpha", "relation_id": "P27"},
        {"case_id": 2, "subject": "Beta", "relation_id": "P106"},
        {"case_id": 3, "subject": "Gamma", "relation_id": "P36"},
        {"case_id": 4, "subject": "Delta", "relation_id": "P19"},
    ]


def _registry_args():
    return SimpleNamespace(
        seed=1,
        forget_num=50,
        layer=27,
        marker_rms_threshold=1e-6,
        semantic_router_steps=300,
        semantic_router_lr=0.01,
        semantic_router_weight_decay=0.01,
        semantic_router_coefficient_norm_cap=4.0,
        semantic_selection_check_every=10,
        semantic_selection_patience=10,
        semantic_training_positive_floor=1.0,
        semantic_training_negative_ceiling=-1.0,
        semantic_certificate_positive_floor=0.25,
        semantic_certificate_negative_ceiling=-0.25,
        semantic_tail_k=4,
        fit_positive_variants_per_old_family=8,
        fit_wrong_relations_per_old_family=64,
        consumed_v4_1_positive_variants=4,
        consumed_v4_1_wrong_relations=32,
        development_positive_variants=4,
        development_wrong_relations=64,
        certification_positive_variants=4,
        certification_wrong_relations=64,
    )


def test_registered_ranks_are_nested_and_end_at_available_full_rank():
    assert core.registered_ranks(4) == (4,)
    assert core.registered_ranks(12) == (4, 8, 12)
    assert core.registered_ranks(21) == (4, 8, 16, 21)
    assert core.registered_ranks(32) == (4, 8, 16, 32)
    assert core.registered_ranks(47) == (4, 8, 16, 32)
    with pytest.raises(ValueError, match="at least four"):
        core.registered_ranks(3)


def test_shared_contrast_core_supports_full_rank_without_record_vectors():
    relations = 8
    hidden = 8
    states = []
    active = []
    labels = []
    for relation in range(relations):
        positive = torch.zeros(hidden)
        positive[relation] = 2.0
        negative = -positive
        states.extend([positive, negative])
        positive_active = torch.zeros(relations, dtype=torch.bool)
        positive_active[relation] = True
        active.extend([positive_active, positive_active.clone()])
        positive_label = torch.zeros(relations, dtype=torch.bool)
        positive_label[relation] = True
        labels.extend([positive_label, torch.zeros(relations, dtype=torch.bool)])
    basis, report = semantic_core.fit_shared_contrast_basis(
        torch.stack(states),
        torch.stack(active),
        torch.stack(labels),
        list(range(relations)),
        rank=8,
    )
    torch.testing.assert_close(basis @ basis.T, torch.eye(8), atol=1e-5, rtol=1e-5)
    router = semantic_core.BaseSemanticRouter(
        relations, hidden, basis, list(range(relations))
    )
    assert router.rank == 8
    assert router.trainable_parameter_count == relations * (8 + 1)
    assert router.trainable_parameter_count < relations * 3072
    assert report["basis_trainable"] is False


def test_v4_2_prompt_families_are_disjoint_and_fit_has_160_negatives_per_record():
    records = _records()
    prefixes = [f"Unrelated training sentence number {index}." for index in range(256)]
    fit = [
        *prompts.build_positive_specs(
            records, split="calibration", corpus_prefixes=prefixes, variants_per_record=8
        ),
        *prompts.build_positive_specs(
            records, split="heldout", corpus_prefixes=prefixes, variants_per_record=8
        ),
        *prompts.build_positive_specs(
            records,
            split="v4_1_development",
            corpus_prefixes=prefixes,
            variants_per_record=4,
        ),
        *prompts.build_wrong_relation_specs(
            records, split="calibration", corpus_prefixes=prefixes, variants_per_record=64
        ),
        *prompts.build_wrong_relation_specs(
            records, split="heldout", corpus_prefixes=prefixes, variants_per_record=64
        ),
        *prompts.build_wrong_relation_specs(
            records,
            split="v4_1_development",
            corpus_prefixes=prefixes,
            variants_per_record=32,
        ),
    ]
    development = [
        *prompts.build_positive_specs(
            records,
            split="v4_2_development",
            corpus_prefixes=prefixes,
            variants_per_record=4,
        ),
        *prompts.build_wrong_relation_specs(
            records,
            split="v4_2_development",
            corpus_prefixes=prefixes,
            variants_per_record=64,
        ),
    ]
    certification = [
        *prompts.build_positive_specs(
            records,
            split="v4_1_certification",
            corpus_prefixes=prefixes,
            variants_per_record=4,
        ),
        *prompts.build_wrong_relation_specs(
            records,
            split="v4_1_certification",
            corpus_prefixes=prefixes,
            variants_per_record=64,
        ),
    ]
    fit_prompts = {row.prompt for row in fit}
    development_prompts = {row.prompt for row in development}
    certification_prompts = {row.prompt for row in certification}
    assert fit_prompts.isdisjoint(development_prompts)
    assert fit_prompts.isdisjoint(certification_prompts)
    assert development_prompts.isdisjoint(certification_prompts)
    for owner in range(len(records)):
        negatives = [row for row in fit if row.owner_index == owner and not row.positive]
        assert len(negatives) == 160
        assert len({row.prompt for row in negatives}) == 160


def test_certified_router_round_trip_is_frozen_and_tamper_evident(tmp_path: Path):
    relations = 8
    router = semantic_core.BaseSemanticRouter(
        relations,
        relations,
        torch.eye(relations),
        list(range(relations)),
    )
    with torch.no_grad():
        router.relation_coefficients.copy_(torch.eye(relations))
        router.relation_bias.copy_(torch.linspace(-0.4, 0.4, relations))
    state = core.certified_router_state(
        router,
        case_ids=list(range(100, 100 + relations)),
        relation_ids=[f"P{index}" for index in range(relations)],
        source_hashes={"writer": "a" * 64, "registry": "b" * 64},
        selected_rank=8,
        layer_index=27,
        marker_rms_threshold=1e-6,
    )
    path = tmp_path / "router.pt"
    torch.save(state, path)
    loaded_state, loaded = core.load_certified_router_state(
        path, hidden_size=relations
    )
    assert loaded_state["selected_smallest_passing_rank"] == 8
    assert loaded_state["certification_open_count"] == 1
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    probe = torch.arange(relations * 3, dtype=torch.float32).reshape(3, relations)
    torch.testing.assert_close(router.scores(probe), loaded.scores(probe))

    for key, value in (
        ("official_evaluation_prompts_seen", 1),
        ("actuator_optimizer_constructed", True),
        ("optimizer_steps_after_certification_open", 1),
        ("one_shot_certification_passed", False),
    ):
        poisoned = dict(state)
        poisoned[key] = value
        poisoned_path = tmp_path / f"poisoned_{key}.pt"
        torch.save(poisoned, poisoned_path)
        with pytest.raises(RuntimeError, match="invalid"):
            core.load_certified_router_state(poisoned_path, hidden_size=relations)


def test_registry_locks_nested_rank_sweep_and_three_way_split():
    registry_path = (
        Path(__file__).resolve().parents[1]
        / "protocols"
        / "mcf_shadow_embedding_semantic_rank_sweep_v4_2_registry.json"
    )
    registry = json.loads(registry_path.read_text())
    train.validate_registry(registry, _registry_args())
    assert registry["architecture"]["rank_arms"] == [
        4,
        8,
        16,
        "full_relation_contrast_rank_capped_at_32",
    ]
    assert registry["development_prompt_policy"][
        "fit_wrong_relation_negatives_per_record_total"
    ] == 160
    assert registry["architecture"]["actuator_constructed"] is False
    assert registry["official_evaluation_prohibited"] is True

    stale = json.loads(json.dumps(registry))
    stale["development_prompt_policy"][
        "fit_wrong_relation_negatives_per_record_total"
    ] = 128
    with pytest.raises(RuntimeError, match="split policy"):
        train.validate_registry(stale, _registry_args())


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def test_v4_1_failure_lineage_requires_exact_underfit_and_unopened_certification(
    tmp_path: Path,
):
    method = tmp_path / "method"
    protocol = tmp_path / "protocol"
    _write_json(
        method / "completion.json",
        {
            "protocol": semantic_core.PROTOCOL,
            "passed": False,
            "stage": "fit_or_development_router_preflight",
            "candidate_saved": False,
            "official_evaluation_prompts_seen": 0,
        },
    )
    _write_json(
        method / "semantic_capacity_report.json",
        {
            "shared_rank": 4,
            "rejected_v4_independent_parameters": 153600,
            "fit_negative_cells_per_record_min": 128,
        },
    )
    _write_json(
        method / "semantic_router_training_log.json",
        {
            "selection": {
                "selection_split": "v4_1_development",
                "certification_split_visible_to_selection": False,
                "best_optimizer_step": 1,
                "stopped_optimizer_step": 100,
            }
        },
    )
    _write_json(
        method / "factorized_router_preflight.json",
        {
            "passed": False,
            "fit_semantic": {"positive_failures": 1, "negative_failures": 2},
            "development_semantic": {
                "positive_failures": 3,
                "negative_failures": 4,
            },
            "certification_prompts_opened": False,
        },
    )
    _write_json(
        method / "marker_preflight.json",
        {"passed": True, "writer_off_marker_rms": 0.0},
    )
    _write_json(
        method / "training_firewall_receipt.json",
        {"official_evaluation_prompts_seen": 0},
    )
    _write_json(method / "development_prompt_manifest.json", {"rows": 1})
    _write_json(
        protocol / "experiment_registry.json",
        {"official_evaluation_prohibited": True},
    )

    imported = train.validate_failed_v4_1(tmp_path)
    assert imported["passed"] is True
    assert imported["diagnosis"]["shared_rank"] == 4
    assert imported["diagnosis"]["best_optimizer_step"] == 1

    _write_json(method / "factorized_router_certificate.json", {"passed": False})
    with pytest.raises(RuntimeError, match="certification_unopened"):
        train.validate_failed_v4_1(tmp_path)
