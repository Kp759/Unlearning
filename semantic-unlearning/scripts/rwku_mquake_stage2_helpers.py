#!/usr/bin/env python3
"""RWKU Directional SURE with a MQuAKE-style protected residual Stage 2.

Level 1 is executed by the already-locked content-sensitive embedding-GA-only
learner and intercepted immediately before its legacy Level 2. This guarantees
that the selected L1 anchor is produced by the exact previously tested code.

The replacement Stage 2 is LM-head only:
  * freeze the selected L1 embedding delta and the transformer;
  * P = L1 margin-passing generated prompts, F = L1 margin-failed prompts;
  * B_P = rowspace(hidden(P)), rank <= 32;
  * B_F = rowspace(hidden(F) projected away from B_P), rank <= 4;
  * repair only residual content-sensitive output rows with DeltaW = C_F B_F;
  * optimize squared margin hinge + KL(P_anchor || P_current) + tiny L2;
  * hard guard every proposed step: P regressions = 0 and KL(P) <= 0.05;
  * backtrack coefficient steps through 0.5,...,0.015625 or roll back;
  * if a backtracked step reaches zero residual failures, require the unchanged
    external-Wikipedia RWKU utility gate immediately at that exact scale.

No Level 3, MLP, attention, LoRA, or representation repair is used. Official
RWKU evaluation remains locked.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import rwku_directional_sure_two_stage as base2
import rwku_directional_sure_two_stage_emb_ga_only as previous
import rwku_directional_sure_v2 as v2
import rwku_setting5e_utility_controlled as sparse_rows
import rwku_sure_head_only_w1k as head
import sure_canonical_core as core

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
DEFAULT_CONFIGURATION = PROJECT_ROOT / "config" / "rwku" / "directional_sure_mquake_style_stage2_seed0.json"
SOURCE_BUNDLE_CONFIGURATION = v2.SOURCE_BUNDLE_CONFIGURATION
SCHEMA = "rwku_directional_sure_mquake_style_stage2_configuration_v1"
EXPERIMENT_ID = "rwku-directional-sure-mquake-style-stage2-stephen-king-seed0"
LEARNER_DIR = "directional_sure_mquake_style_stage2"
LEVEL1_CAPTURE_DIR = "level1_locked_emb_ga_only_capture"


class _Level1Captured(RuntimeError):
    pass


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


def read_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _without_note(value: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(value)
    result.pop("note", None)
    return result


def load_configuration(path: Path) -> Dict[str, Any]:
    cfg = read_json(path)
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
            raise ValueError(f"MQuAKE-style Stage2 configuration changed {key}")

    baseline = previous.read_json(previous.DEFAULT_CONFIGURATION)
    if cfg.get("trainable_components") != baseline.get("trainable_components"):
        raise ValueError("Level-1 trainable components changed")
    if cfg.get("optimization") != baseline.get("optimization"):
        raise ValueError("Level-1 optimization changed")
    if cfg.get("acceptance") != baseline.get("acceptance"):
        raise ValueError("RWKU acceptance budgets changed")
    if _without_note(cfg.get("data_boundary", {})) != _without_note(baseline.get("data_boundary", {})):
        raise ValueError("RWKU data boundary changed")

    s2 = cfg.get("stage2", {})
    locked = {
        "enabled": True,
        "trigger": "level1_pairwise_margin_failures",
        "training_scope": "level1_residual_prompt_sensitive_prediction_cases_only",
        "row_scope": "content_sensitive_output_rows_implicated_by_level1_residual_prompts",
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
        "residual_sensitive_exclusive_basis_rank": 4,
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


def run_locked_level1_capture(args: argparse.Namespace, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Run the exact prior learner and stop at its first legacy-L2-only call."""
    captured: Dict[str, Any] = {}
    baseline_cfg = previous.read_json(previous.DEFAULT_CONFIGURATION)
    baseline_cfg["configuration_id"] = cfg["configuration_id"]
    baseline_cfg["method"] = "Locked Level-1 capture for RWKU MQuAKE-style Stage2"

    old_parse = previous.parse_args
    old_load = previous.load_configuration
    old_snapshot = base2.snapshot_with_key
    old_residual_cases = base2.residual_case_indices
    old_dir = previous.LEARNER_DIR

    def fake_parse() -> argparse.Namespace:
        return argparse.Namespace(
            model_path=args.model_path,
            training_bundle=Path(args.training_bundle),
            generator_receipt=Path(args.generator_receipt),
            wikipedia_dir=Path(args.wikipedia_dir),
            output_root=Path(args.output_root),
            experiment_id=str(args.experiment_id),
            configuration=Path(args.configuration),
            save_checkpoint=False,
        )

    def fake_load(_path: Path) -> Dict[str, Any]:
        return dict(baseline_cfg)

    def capture_snapshot(*snapshot_args: Any, **snapshot_kwargs: Any) -> Dict[str, Any]:
        candidate = old_snapshot(*snapshot_args, **snapshot_kwargs)
        if snapshot_kwargs.get("source") == "level1_anchor":
            captured["anchor"] = candidate
        return candidate

    def stop_before_legacy_level2(*_a: Any, **_kw: Any) -> List[int]:
        raise _Level1Captured("Locked Level 1 complete")

    previous.parse_args = fake_parse
    previous.load_configuration = fake_load
    previous.LEARNER_DIR = LEVEL1_CAPTURE_DIR
    base2.snapshot_with_key = capture_snapshot
    base2.residual_case_indices = stop_before_legacy_level2
    try:
        previous.main()
    except _Level1Captured:
        pass
    finally:
        previous.parse_args = old_parse
        previous.load_configuration = old_load
        previous.LEARNER_DIR = old_dir
        base2.snapshot_with_key = old_snapshot
        base2.residual_case_indices = old_residual_cases

    if "anchor" not in captured:
        raise RuntimeError("Locked Level 1 did not expose a utility-safe anchor")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return captured["anchor"]


def pairwise_margins(logits: torch.Tensor, tids: torch.Tensor) -> torch.Tensor:
    rows = torch.arange(logits.shape[0], device=logits.device)
    values = logits.float()
    sensitive = values[rows, tids]
    masked = values.clone()
    masked[rows, tids] = -torch.inf
    return masked.max(dim=-1).values - sensitive


def squared_margin_hinge(logits: torch.Tensor, tids: torch.Tensor, margin: float) -> torch.Tensor:
    return F.relu(float(margin) - pairwise_margins(logits, tids)).square().mean()


def prompt_case_indices(cases: Sequence[core.SensitivePredictionCase], positions: Sequence[int]) -> List[int]:
    wanted = set(int(x) for x in positions)
    result = [i for i, case in enumerate(cases) if int(case.record_position) in wanted]
    covered = {int(cases[i].record_position) for i in result}
    if covered != wanted:
        raise RuntimeError(f"Missing prompt cases: {sorted(wanted - covered)[:10]}")
    return result


def success_positions(anchor_atomic: Mapping[str, Any], prompt_count: int) -> List[int]:
    failures = set(base2.residual_prompt_positions(anchor_atomic))
    values = [i for i in range(int(prompt_count)) if i not in failures]
    if not values:
        raise RuntimeError("No L1 successes available for Stage-2 protection")
    return values


def build_bases(
    model: torch.nn.Module,
    tokenizer: Any,
    success_cases: Sequence[core.SensitivePredictionCase],
    residual_cases: Sequence[core.SensitivePredictionCase],
    *,
    device: torch.device,
    batch_size: int,
    p_rank: int,
    f_rank: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    hp = core.forward_last_hidden(model, tokenizer, success_cases, device, batch_size=batch_size)
    hf = core.forward_last_hidden(model, tokenizer, residual_cases, device, batch_size=batch_size)
    bp = core.orthonormal_row_basis(hp, max_rank=p_rank)
    if not bp.numel():
        raise RuntimeError("B_P(success) is empty")
    rf = sparse_rows.project_rows_away_from_protected_basis(hf, bp)
    bf = core.orthonormal_row_basis(rf, max_rank=f_rank)
    if not bf.numel():
        raise RuntimeError("B_F(residual) is empty")
    bf = sparse_rows.project_rows_away_from_protected_basis(bf, bp)
    bf = core.orthonormal_row_basis(bf, max_rank=f_rank)
    overlap = float((bf @ bp.T).abs().max().detach().cpu())
    if overlap > 1e-4:
        raise RuntimeError(f"B_F/B_P overlap too large: {overlap}")
    return bf.detach(), bp.detach(), {
        "success_hidden_count": int(hp.shape[0]),
        "residual_hidden_count": int(hf.shape[0]),
        "protected_success_rank": int(bp.shape[0]),
        "residual_exclusive_rank": int(bf.shape[0]),
        "max_abs_BF_BP_overlap": overlap,
        "residual_energy_after_success_projection_fraction": float(
            rf.square().sum().detach().cpu() / max(float(hf.square().sum().detach().cpu()), 1e-12)
        ),
        "basis_refresh": False,
        "reason_not_refreshed": "Stage-2 embeddings and transformer are frozen",
        "official_rwku_records_accessed": False,
    }


@torch.no_grad()
def protection_report(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[core.SensitivePredictionCase],
    case_indices: Sequence[int],
    anchor_logits: torch.Tensor,
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    required_margin: float,
) -> Dict[str, Any]:
    failed: set[int] = set()
    kl_weighted = 0.0
    total = 0
    min_margin = float("inf")
    for start in range(0, len(case_indices), int(batch_size)):
        ids = list(case_indices[start : start + int(batch_size)])
        batch = [cases[i] for i in ids]
        logits = core.forward_last_logits(model, tokenizer, batch, device)
        tids = core.official_target_ids(tokenizer, batch, llama_like=llama_like, device=device)
        margins = pairwise_margins(logits, tids)
        min_margin = min(min_margin, float(margins.min().detach().cpu()))
        for j, value in enumerate(margins.detach().cpu().tolist()):
            if float(value) < float(required_margin):
                failed.add(int(batch[j].record_position))
        ref = anchor_logits[start : start + len(batch)]
        kl = core.gd_non_sensitive_kl(logits, ref, tids)
        kl_weighted += float(kl.detach().cpu()) * len(batch)
        total += len(batch)
    return {
        "protected_prompt_regressions": len(failed),
        "protected_regression_positions": sorted(failed),
        "protected_case_count": total,
        "protected_minimum_pairwise_margin": min_margin,
        "protected_non_sensitive_kl_mean": kl_weighted / max(total, 1),
    }


@torch.no_grad()
def residual_report(
    model: torch.nn.Module,
    tokenizer: Any,
    cases: Sequence[core.SensitivePredictionCase],
    case_indices: Sequence[int],
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
    required_margin: float,
) -> Dict[str, Any]:
    failed: set[int] = set()
    min_margin = float("inf")
    for start in range(0, len(case_indices), int(batch_size)):
        ids = list(case_indices[start : start + int(batch_size)])
        batch = [cases[i] for i in ids]
        logits = core.forward_last_logits(model, tokenizer, batch, device)
        tids = core.official_target_ids(tokenizer, batch, llama_like=llama_like, device=device)
        margins = pairwise_margins(logits, tids)
        min_margin = min(min_margin, float(margins.min().detach().cpu()))
        for j, value in enumerate(margins.detach().cpu().tolist()):
            if float(value) < float(required_margin):
                failed.add(int(batch[j].record_position))
    return {
        "remaining_margin_failure_positions": sorted(failed),
        "remaining_margin_failure_count": len(failed),
        "residual_minimum_pairwise_margin": min_margin,
    }


def snapshots(params: Sequence[torch.nn.Parameter]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def restore(params: Sequence[torch.nn.Parameter], values: Sequence[torch.Tensor]) -> None:
    with torch.no_grad():
        for p, value in zip(params, values):
            p.copy_(value)


def interpolate(
    params: Sequence[torch.nn.Parameter], before: Sequence[torch.Tensor], proposed: Sequence[torch.Tensor], scale: float
) -> None:
    with torch.no_grad():
        for p, old, new in zip(params, before, proposed):
            p.copy_(old + float(scale) * (new - old))


def combine_output(
    anchor_output: torch.Tensor,
    selected_rows: Sequence[int],
    repair_rows: Sequence[int],
    repair_delta: torch.Tensor,
) -> torch.Tensor:
    mapping = {int(row): i for i, row in enumerate(selected_rows)}
    combined = anchor_output.detach().cpu().clone()
    for i, row in enumerate(repair_rows):
        if int(row) not in mapping:
            raise RuntimeError(f"Repair row outside L1 content rows: {row}")
        combined[mapping[int(row)]] += repair_delta[i].detach().cpu()
    return combined


def make_optimizer(repair: core.SelectedRowDelta, stage2: Mapping[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        repair.parameters(),
        lr=float(stage2["repair_learning_rate"]),
        weight_decay=float(stage2["weight_decay"]),
    )
