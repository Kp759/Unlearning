#!/usr/bin/env python3
"""Core state for V4.2's nested shared semantic-rank sweep.

V4.2 is a router-only development experiment.  It selects the smallest rank
in ``(4, 8, 16, full relation-contrast rank)`` that passes fit and a fresh
development family, opens the untouched V4.1 certification family for that
arm only, and freezes the certified semantic router for a later actuator
process.  It does not construct an actuator or expose any official-evaluation
loader.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch

import mcf_shadow_embedding_semantic_router_core as v41


PROTOCOL = "mcf_shadow_marker_base_semantic_rank_sweep_v4_2"
SCHEMA_VERSION = 1
BASE_RANKS: Tuple[int, ...] = (4, 8, 16)
MAXIMUM_SHARED_RANK = 32


def _valid_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def registered_ranks(relation_count: int) -> Tuple[int, ...]:
    """Return nested arms ending at the available full contrast rank."""

    maximum = min(int(relation_count), MAXIMUM_SHARED_RANK)
    if maximum < 4:
        raise ValueError("V4.2 requires at least four distinct relations")
    return tuple(sorted({value for value in (*BASE_RANKS, maximum) if value <= maximum}))


def certified_router_state(
    router: v41.BaseSemanticRouter,
    *,
    case_ids: Sequence[int],
    relation_ids: Sequence[str],
    source_hashes: Mapping[str, str],
    selected_rank: int,
    layer_index: int,
    marker_rms_threshold: float,
) -> Dict[str, Any]:
    rank_arms = registered_ranks(router.relation_count)
    if int(selected_rank) not in rank_arms or router.rank != int(selected_rank):
        raise ValueError("certified router rank is not a registered V4.2 arm")
    if len(case_ids) != router.records:
        raise ValueError("certified router case binding is incomplete")
    if len(relation_ids) != router.relation_count:
        raise ValueError("certified router relation binding is incomplete")
    if (
        len(set(int(value) for value in case_ids)) != len(case_ids)
        or len(set(str(value) for value in relation_ids)) != len(relation_ids)
        or not source_hashes
        or any(not str(key) or not _valid_sha256(value) for key, value in source_hashes.items())
    ):
        raise ValueError("certified router source or identity binding is invalid")
    if int(layer_index) < 0 or not float(marker_rms_threshold) > 0.0:
        raise ValueError("certified router runtime binding is invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "mcf_shadow_marker_base_semantic_certified_router",
        "protocol": PROTOCOL,
        "architecture": {
            "main_embedding_path": "unaltered_base_embedding",
            "shadow_embedding_path": "frozen_v6_2_sparse_delta",
            "exact_complete_subject_key": True,
            "marker_gate": "shadow_minus_base_rms",
            "semantic_gate": "frozen_shared_rank_relation_router",
            "semantic_basis_trainable": False,
            "semantic_relation_tied_coefficients": True,
            "semantic_record_specific_hidden_vectors": False,
            "actuator_constructed": False,
        },
        "registered_rank_arms": list(rank_arms),
        "selected_smallest_passing_rank": int(selected_rank),
        "layer_index": int(layer_index),
        "marker_rms_threshold": float(marker_rms_threshold),
        "case_ids": [int(value) for value in case_ids],
        "relation_ids": [str(value) for value in relation_ids],
        "semantic_shared_basis": router.shared_basis.detach().cpu(),
        "semantic_record_relation_index": (
            router.record_relation_index.detach().cpu()
        ),
        "semantic_relation_coefficients": (
            router.relation_coefficients.detach().cpu()
        ),
        "semantic_relation_bias": router.relation_bias.detach().cpu(),
        "source_hashes": {str(key): str(value) for key, value in source_hashes.items()},
        "one_shot_certification_passed": True,
        "certification_open_count": 1,
        "optimizer_steps_after_certification_open": 0,
        "actuator_optimizer_constructed": False,
        "official_evaluation_prompts_seen": 0,
    }


def load_certified_router_state(
    path: str | Path,
    *,
    hidden_size: int,
    device: torch.device | str = "cpu",
) -> Tuple[Dict[str, Any], v41.BaseSemanticRouter]:
    state = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(state, Mapping):
        raise RuntimeError("V4.2 certified router state must be a mapping")
    if (
        int(state.get("schema_version", -1)) != SCHEMA_VERSION
        or state.get("protocol") != PROTOCOL
        or state.get("kind") != "mcf_shadow_marker_base_semantic_certified_router"
    ):
        raise RuntimeError("V4.2 certified router schema/protocol mismatch")
    architecture = state.get("architecture")
    required = {
        "main_embedding_path": "unaltered_base_embedding",
        "shadow_embedding_path": "frozen_v6_2_sparse_delta",
        "exact_complete_subject_key": True,
        "marker_gate": "shadow_minus_base_rms",
        "semantic_gate": "frozen_shared_rank_relation_router",
        "semantic_basis_trainable": False,
        "semantic_relation_tied_coefficients": True,
        "semantic_record_specific_hidden_vectors": False,
        "actuator_constructed": False,
    }
    if not isinstance(architecture, Mapping) or any(
        architecture.get(key) != value for key, value in required.items()
    ):
        raise RuntimeError("V4.2 certified router violates its passive boundary")
    selected_rank = int(state.get("selected_smallest_passing_rank", -1))
    case_ids = state.get("case_ids")
    relation_ids = state.get("relation_ids")
    if not isinstance(relation_ids, list):
        raise RuntimeError("V4.2 relation binding is missing")
    expected_ranks = registered_ranks(len(relation_ids))
    if (
        tuple(int(value) for value in state.get("registered_rank_arms", []))
        != expected_ranks
        or selected_rank not in expected_ranks
    ):
        raise RuntimeError("V4.2 rank registry mismatch")
    basis = state.get("semantic_shared_basis")
    record_relation_index = state.get("semantic_record_relation_index")
    coefficients = state.get("semantic_relation_coefficients")
    bias = state.get("semantic_relation_bias")
    source_hashes = state.get("source_hashes")
    relation_index_values = (
        record_relation_index.tolist()
        if isinstance(record_relation_index, torch.Tensor)
        else []
    )
    if (
        not isinstance(case_ids, list)
        or not case_ids
        or len(set(int(value) for value in case_ids)) != len(case_ids)
        or not relation_ids
        or len(set(str(value) for value in relation_ids)) != len(relation_ids)
        or int(state.get("layer_index", -1)) < 0
        or not float(state.get("marker_rms_threshold", 0.0)) > 0.0
        or not isinstance(basis, torch.Tensor)
        or basis.shape != (selected_rank, int(hidden_size))
        or not bool(torch.isfinite(basis).all())
        or not isinstance(record_relation_index, torch.Tensor)
        or record_relation_index.shape != (len(case_ids),)
        or set(int(value) for value in relation_index_values)
        != set(range(len(relation_ids)))
        or not isinstance(coefficients, torch.Tensor)
        or coefficients.shape != (len(relation_ids), selected_rank)
        or not bool(torch.isfinite(coefficients).all())
        or not isinstance(bias, torch.Tensor)
        or bias.shape != (len(relation_ids),)
        or not bool(torch.isfinite(bias).all())
        or not isinstance(source_hashes, Mapping)
        or not source_hashes
        or any(
            not str(key) or not _valid_sha256(value)
            for key, value in source_hashes.items()
        )
        or state.get("one_shot_certification_passed") is not True
        or int(state.get("certification_open_count", -1)) != 1
        or int(state.get("optimizer_steps_after_certification_open", -1)) != 0
        or bool(state.get("actuator_optimizer_constructed", True))
        or int(state.get("official_evaluation_prompts_seen", -1)) != 0
    ):
        raise RuntimeError("V4.2 certified router tensors/receipts are invalid")
    router = v41.BaseSemanticRouter(
        len(case_ids),
        int(hidden_size),
        basis,
        record_relation_index,
    ).to(device)
    with torch.no_grad():
        router.relation_coefficients.copy_(
            coefficients.to(router.relation_coefficients.device)
        )
        router.relation_bias.copy_(bias.to(router.relation_bias.device))
    for parameter in router.parameters():
        parameter.requires_grad_(False)
    return dict(state), router
