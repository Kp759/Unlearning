#!/usr/bin/env python3
"""Helpers for RWKU MQuAKE-style Stage 2 with all residual target rows, rank 8."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from rwku_mquake_stage2_helpers import *  # noqa: F401,F403
import rwku_mquake_stage2_helpers as base

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "directional_sure_mquake_stage2_all_residual_rows_rank8_seed0.json"
SCHEMA = "rwku_directional_sure_mquake_stage2_all_residual_rows_rank8_configuration_v1"
EXPERIMENT_ID = "rwku-directional-sure-mquake-stage2-all-residual-rows-rank8-stephen-king-seed0"
LEARNER_DIR = "directional_sure_mquake_stage2_all_residual_rows_rank8"
LEVEL1_CAPTURE_DIR = "level1_locked_emb_ga_only_capture"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-bundle", type=Path, required=True)
    p.add_argument("--generator-receipt", type=Path, required=True)
    p.add_argument("--wikipedia-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--experiment-id", default=EXPERIMENT_ID)
    p.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    p.add_argument("--save-checkpoint", action="store_true")
    return p.parse_args()


def load_configuration(path: Path) -> Dict[str, Any]:
    cfg = base.read_json(path)
    required = {
        "schema_version": SCHEMA,
        "configuration_id": EXPERIMENT_ID,
        "development_only": True,
        "posthoc_development_target": True,
        "official_rwku_metrics_observed_before_method_design": True,
        "seed": 0,
        "target_entity": "Stephen King",
        "target_entity_id": "rwku:1_Stephen_King",
        "neutral_target": "Unknown",
        "level3_representation_repair_enabled": False,
        "level1_embedding_gradient_policy": "GA_only_no_GD_no_hidden_basis_projection",
        "level1_lm_head_gradient_policy": "GA_to_sensitive_exclusive_basis_and_GD_to_external_protected_basis",
    }
    for key, expected in required.items():
        if cfg.get(key) != expected:
            raise ValueError(f"All-residual-row rank-8 Stage2 configuration changed {key}")

    baseline = previous.read_json(previous.DEFAULT_CONFIGURATION)
    if cfg.get("trainable_components") != baseline.get("trainable_components"):
        raise ValueError("Level-1 trainable components changed")
    if cfg.get("optimization") != baseline.get("optimization"):
        raise ValueError("Level-1 optimization changed")
    if cfg.get("acceptance") != baseline.get("acceptance"):
        raise ValueError("RWKU acceptance budgets changed")
    if base._without_note(cfg.get("data_boundary", {})) != base._without_note(baseline.get("data_boundary", {})):
        raise ValueError("RWKU data boundary changed")

    s2 = cfg.get("stage2", {})
    locked = {
        "enabled": True,
        "trigger": "level1_pairwise_margin_failures",
        "training_scope": "level1_residual_prompt_sensitive_prediction_cases_only",
        "row_scope": "all_non_special_sensitive_target_rows_implicated_by_level1_residual_prompts",
        "parameter_scope": "lm_head_only_increment_over_frozen_level1_anchor",
        "embedding_parameters": False,
        "transformer_parameters": False,
        "representation_repair": False,
        "loss": "squared_pairwise_margin_hinge_plus_success_non_sensitive_KL_plus_increment_L2",
        "repair_steps": 300,
        "repair_batch_size": 8,
        "success_kl_batch_size": 8,
        "repair_learning_rate": 0.005,
        "repair_l2": 1e-6,
        "success_kl_weight": 1.0,
        "grad_clip": 1.0,
        "optimizer": "AdamW",
        "weight_decay": 0.0,
        "protected_success_basis_rank": 32,
        "residual_sensitive_exclusive_basis_rank": 8,
        "basis_refresh": False,
        "hard_success_regression_limit": 0,
        "hard_success_kl_budget": 0.05,
        "backtrack_scales": [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625],
        "reset_optimizer_after_backtracked_step": True,
        "checkpoint_interval": 25,
    }
    for key, expected in locked.items():
        if s2.get(key) != expected:
            raise ValueError(f"Stage-2 locked field changed: {key}")
    return cfg


def all_non_special_residual_rows(
    tokenizer: Any,
    tids_all: torch.Tensor,
    case_indices: Sequence[int],
) -> list[int]:
    """All non-special sensitive target token rows occurring in residual cases."""
    special = {
        int(x)
        for x in getattr(tokenizer, "all_special_ids", [])
        if x is not None
    }
    token_ids = tids_all.detach().cpu().tolist()
    rows = sorted(
        {
            int(token_ids[index])
            for index in case_indices
            if int(token_ids[index]) not in special
        }
    )
    if not rows:
        raise RuntimeError("Residual cases expose no non-special sensitive target rows")
    return rows


def expand_anchor_output_delta(
    anchor_output: torch.Tensor,
    level1_rows: Sequence[int],
    runtime_output_rows: Sequence[int],
) -> torch.Tensor:
    """Embed the L1 content-row delta into a larger output-row table with zeros elsewhere."""
    if int(anchor_output.shape[0]) != len(level1_rows):
        raise ValueError("L1 anchor output delta does not match L1 row count")
    mapping = {int(row): i for i, row in enumerate(runtime_output_rows)}
    expanded = torch.zeros(
        (len(runtime_output_rows), int(anchor_output.shape[1])),
        dtype=anchor_output.dtype,
        device=anchor_output.device,
    )
    for source_index, row in enumerate(level1_rows):
        target_index = mapping.get(int(row))
        if target_index is None:
            raise RuntimeError(f"L1 row missing from runtime output union: {row}")
        expanded[target_index] = anchor_output[source_index]
    return expanded
