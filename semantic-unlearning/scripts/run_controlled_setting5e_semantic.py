#!/usr/bin/env python3
"""Run the controlled pipeline with semantic TOFU repair.

MCF and ZsRE behavior is unchanged. For TOFU, this wrapper reuses the existing
fresh-Base Setting 5/5e pipeline and replaces the final active-repair stage with
``tofu_gagd_semantic_forget_repair_safe.py``. Run plans and application receipts
are still produced by ``run_controlled_setting5e.py``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, List, Mapping

import run_controlled_setting5e as controlled


SEMANTIC_REPAIR_ALLOWED = set(controlled.TOFU_REPAIR_ALLOWED) | {
    "abstention_answer",
    "preference_margin",
    "preference_hinge_weight",
    "hardest_preference_hinge_weight",
    "abstention_preservation_weight",
    "require_preference_constraints",
    "semantic_utility_constraint_mode",
}


def _run_tofu_semantic(
    *,
    python: str,
    spec: Mapping[str, Any],
    materialized: Mapping[str, Path],
    counts: Mapping[str, Any],
    seed: int,
    stage: str,
    output_dir: Path,
    dry_run: bool,
    commands: List[List[str]],
) -> Path:
    del stage
    input_dir = next(iter(materialized.values())).parent
    base_model = str(spec["base_model_path"])
    forget_num = int(counts["forget_count"])
    retain_num = int(counts["retain_count"])
    if forget_num <= 0 or retain_num <= 0:
        raise ValueError("Controlled TOFU stage needs forget and retain rows")

    train_dir = output_dir / "four_settings"
    retain_batch = max(1, controlled.math_ceil_div(retain_num, forget_num))
    train_options = {
        "steps": forget_num,
        "batch_size": 1,
        "retain_batch_size": retain_batch,
        "emb_lm_lr": 5e-5,
        "forget_weight": 1.5,
        "retain_weight": float(3 * retain_batch),
        "kl_retain_weight": 10.0,
        "dtype": "bf16",
        "device_map": "single",
        **dict(spec.get("setting5e_source", {})),
    }
    train_command = [
        python,
        "scripts/tofu_gagd_four_settings_official.py",
        "--model-path",
        base_model,
        "--output-dir",
        str(train_dir),
        "--mode",
        controlled.TOFU_SOURCE_MODE,
        "--controlled-input-dir",
        str(input_dir),
        "--forget-num",
        str(forget_num),
        "--retain-num",
        str(retain_num),
        "--seed",
        str(seed),
        "--save-model",
    ]
    controlled._append_options(
        train_command,
        train_options,
        controlled.TOFU_TRAIN_ALLOWED,
    )
    commands.append(train_command)
    if not dry_run:
        subprocess.run(train_command, cwd=controlled.PROJECT_DIR, check=True)
    source_checkpoint = (
        train_dir / controlled.TOFU_SOURCE_MODE / "checkpoint"
    )

    restore_dir = output_dir / "setting5e_restore"
    restore_options = {
        "unique_forget_alpha": 1.0,
        "overlap_alpha": 0.0,
        "protect_full_retain_split": True,
        "protect_utility_splits": True,
        "restore_input_embeddings": True,
        "dtype": train_options.get("dtype", "bf16"),
        "device_map": train_options.get("device_map", "single"),
        **dict(spec.get("setting5e_restore", {})),
    }
    restore_command = [
        python,
        "scripts/tofu_gagd_setting5e_restore.py",
        "--model-path",
        str(source_checkpoint),
        "--base-model-path",
        base_model,
        "--output-dir",
        str(restore_dir),
        "--source-mode",
        controlled.TOFU_SOURCE_MODE,
        "--controlled-input-dir",
        str(input_dir),
        "--forget-num",
        str(forget_num),
        "--retain-num",
        str(retain_num),
        "--seed",
        str(seed),
        "--save-model",
    ]
    controlled._append_options(
        restore_command,
        restore_options,
        controlled.TOFU_RESTORE_ALLOWED,
    )
    commands.append(restore_command)
    if not dry_run:
        if not source_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing TOFU Setting 5 source checkpoint {source_checkpoint}"
            )
        subprocess.run(restore_command, cwd=controlled.PROJECT_DIR, check=True)
    restored_checkpoint = restore_dir / "checkpoint"

    repair_dir = output_dir / "semantic_active_repair"
    repair_options = {
        "retain_calibration_num": retain_num,
        "target_forget_answer_probability": 1e-4,
        "target_nll_buffer": 0.15,
        "min_utility_probability_ratio": 0.9995,
        "utility_constraint_mode": "aggregate",
        "semantic_utility_constraint_mode": "per-example",
        "utility_reference_policy": "non-regression",
        "require_input_retain_target": True,
        "forget_hinge_weight": 50.0,
        "hardest_forget_hinge_weight": 100.0,
        "preference_margin": 1.0,
        "preference_hinge_weight": 100.0,
        "hardest_preference_hinge_weight": 100.0,
        "abstention_preservation_weight": 10.0,
        "utility_hinge_weight": 100.0,
        "delta_l2_lambda": 1e-4,
        "repair_steps": 3000,
        "repair_lr": 5e-3,
        "repair_optimizer": "adamw",
        "repair_rank": 16,
        "utility_projection_rank": 128,
        "project_away_utility_hidden": True,
        "max_delta_norm": None,
        "require_preference_constraints": True,
        "require_utility_constraints": True,
        "stop_when_all_satisfied": True,
        "dtype": restore_options.get("dtype", "bf16"),
        "device_map": restore_options.get("device_map", "single"),
        **dict(spec.get("active_repair", {})),
    }
    repair_command = [
        python,
        "scripts/tofu_gagd_semantic_forget_repair_safe.py",
        "--model-path",
        str(restored_checkpoint),
        "--reference-model-path",
        base_model,
        "--output-dir",
        str(repair_dir),
        "--controlled-input-dir",
        str(input_dir),
        "--forget-num",
        str(forget_num),
        "--retain-num",
        str(retain_num),
        "--seed",
        str(seed),
        "--save-model",
    ]
    controlled._append_options(
        repair_command,
        repair_options,
        SEMANTIC_REPAIR_ALLOWED,
    )
    commands.append(repair_command)
    if not dry_run:
        if not restored_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing TOFU Setting 5e checkpoint {restored_checkpoint}"
            )
        subprocess.run(repair_command, cwd=controlled.PROJECT_DIR, check=True)
    return (repair_dir / "checkpoint").resolve()


def main() -> None:
    controlled._run_tofu = _run_tofu_semantic
    controlled.main()


if __name__ == "__main__":
    main()
