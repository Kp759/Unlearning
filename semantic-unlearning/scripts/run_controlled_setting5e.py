#!/usr/bin/env python3
"""Run Setting 5e + active LM-head repair on one controlled protocol stage.

Every stage starts from ``base_model_path`` in the frozen candidate spec:

* development/train is for exploratory method development;
* development/validation applies validation deletion requests to a fresh Base;
* test/final_apply is unlocked only by a selection receipt and applies held-out
  deletion requests to another fresh Base.

The final Judge-B prompts are never loaded by this runner.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

from controlled_unlearning_protocol import (
    load_development_bundle,
    load_final_apply_bundle,
    read_json,
    sha256_file,
    sha256_json,
    validate_mcf_post_reload_acceptance,
    write_json,
)


PROJECT_DIR = Path(__file__).resolve().parents[1]
SETTING5E_MODE = "emb_lm_all_restore_post_training_true"
TOFU_SOURCE_MODE = "emb_lm_all_tokens"
MCF_POST_RELOAD_REJECTION_EXIT_CODE = 3


class MCFPostReloadAcceptanceError(RuntimeError):
    def __init__(
        self,
        *,
        checkpoint: Path,
        evaluation_path: Path,
        acceptance: Mapping[str, Any],
    ) -> None:
        super().__init__(
            "Serialized/reloaded MCF checkpoint missed strict forget "
            f"acceptance: {acceptance.get('failure_reasons')}"
        )
        self.checkpoint = Path(checkpoint)
        self.evaluation_path = Path(evaluation_path)
        self.acceptance = dict(acceptance)


MCF_SETTING5_ALLOWED = {
    "steps",
    "batch_size",
    "retain_batch_size",
    "lr",
    "emb_lm_lr",
    "forget_weight",
    "retain_weight",
    "kl_retain_weight",
    "weight_decay",
    "forget_loss_type",
    "forget_margin",
    "grad_clip",
    "gradient_checkpointing",
    "emb_lm_optimizer",
    "sampling_strategy",
    "dtype",
    "device_map",
    "post_training_new_true_alpha",
    "post_training_new_retain_alpha",
    "post_training_new_true_retain_alpha",
}
ZSRE_ALLOWED = {
    "steps",
    "batch_size",
    "retain_batch_size",
    "emb_lm_lr",
    "forget_weight",
    "retain_weight",
    "forget_margin",
    "weight_decay",
    "grad_clip",
    "emb_lm_optimizer",
    "sampling_strategy",
    "post_training_new_true_alpha",
    "post_training_new_retain_alpha",
    "post_training_new_true_retain_alpha",
    "repair_steps",
    "repair_lr",
    "repair_optimizer",
    "active_logit_margin",
    "selection_logit_margin",
    "repair_rank",
    "repair_l2_lambda",
    "max_delta_norm",
    "retain_calibration_num",
    "retain_calibration_seed",
    "project_away_protected_hidden",
    "stop_when_all_satisfied",
    "candidate_scales",
    "utility_drop_tolerance",
    "max_ppl_ratio",
    "target_eff_max",
    "target_gen_max",
    "strict_utility_gates",
    "fail_if_target_missed",
    "eval_batch_size",
    "cache_batch_size",
    "dtype",
    "device_map",
    "gradient_checkpointing",
    "skip_ppl",
}
TOFU_TRAIN_ALLOWED = {
    "steps",
    "batch_size",
    "retain_batch_size",
    "lr",
    "full_lr",
    "emb_lm_lr",
    "forget_weight",
    "retain_weight",
    "kl_retain_weight",
    "weight_decay",
    "grad_clip",
    "emb_lm_optimizer",
    "sampling_strategy",
    "max_eval_examples",
    "dtype",
    "device_map",
    "gradient_checkpointing",
}
TOFU_RESTORE_ALLOWED = {
    "unique_forget_alpha",
    "overlap_alpha",
    "max_length",
    "protect_full_retain_split",
    "protect_utility_splits",
    "restore_input_embeddings",
    "chunk_rows",
    "dtype",
    "device_map",
}
TOFU_REPAIR_ALLOWED = {
    "retain_calibration_num",
    "real_authors_calibration_num",
    "world_facts_calibration_num",
    "calibration_seed",
    "target_forget_answer_probability",
    "target_nll_buffer",
    "min_utility_probability_ratio",
    "utility_nll_tolerance",
    "utility_constraint_mode",
    "require_input_retain_target",
    "forget_hinge_weight",
    "hardest_forget_hinge_weight",
    "utility_hinge_weight",
    "delta_l2_lambda",
    "repair_steps",
    "repair_lr",
    "repair_optimizer",
    "repair_rank",
    "utility_projection_rank",
    "basis_max_rows",
    "project_away_utility_hidden",
    "max_delta_norm",
    "batch_size",
    "max_length",
    "log_every",
    "comparison_tolerance",
    "materialization_relative_tolerance",
    "require_utility_constraints",
    "stop_when_all_satisfied",
    "save_best_effort",
    "dtype",
    "device_map",
}


def _resolve_materialized_path(
    bundle_path: Path,
    value: str,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return bundle_path.parent / path


def _verify_materialized(
    bundle_path: Path,
    materialized: Mapping[str, Any],
) -> Dict[str, Path]:
    files = materialized.get("files")
    hashes = materialized.get("sha256")
    if not isinstance(files, Mapping) or not isinstance(hashes, Mapping):
        raise ValueError("Bundle materialized input metadata is malformed")
    resolved: Dict[str, Path] = {}
    for name, value in files.items():
        path = _resolve_materialized_path(bundle_path, str(value)).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Materialized input does not exist: {path}")
        expected = str(hashes.get(name, ""))
        actual = sha256_file(path)
        if not expected or actual != expected:
            raise ValueError(
                f"Materialized input hash mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
        resolved[str(name)] = path
    return resolved


def _public_model_identity(path: Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    value: Dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.is_dir():
        metadata = {}
        for name in (
            "config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "repair_summary.json",
            "zsre_results.json",
        ):
            candidate = path / name
            if candidate.exists():
                metadata[name] = sha256_file(candidate)
        value["metadata_sha256"] = metadata
        value["weight_files"] = [
            {"name": item.name, "size": item.stat().st_size}
            for item in sorted(
                [
                    *path.glob("*.safetensors"),
                    *path.glob("pytorch_model*.bin"),
                ]
            )
        ]
    elif path.exists():
        value["sha256"] = sha256_file(path)
    value["identity_sha256"] = sha256_json(value)
    return value


def _flag_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _append_options(
    command: List[str],
    values: Mapping[str, Any],
    allowed: set[str],
) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported candidate options: {unknown}")
    for key in sorted(values):
        value = values[key]
        if value is None:
            continue
        flag = _flag_name(key)
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + flag[2:])
        else:
            command.extend([flag, str(value)])


def _load_spec(path: Path) -> Dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise ValueError("Candidate spec must be a JSON object")
    required = {
        "schema_version",
        "candidate_id",
        "dataset",
        "base_model_path",
    }
    missing = sorted(required - set(value))
    if missing:
        raise ValueError(f"Candidate spec misses fields: {missing}")
    if int(value["schema_version"]) != 1:
        raise ValueError("Unsupported candidate spec schema version")
    return dict(value)


def _validate_receipt(
    receipt: Mapping[str, Any],
    bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    if receipt.get("kind") != "controlled_candidate_selection_receipt":
        raise ValueError("Invalid selection receipt")
    stored_hash = str(receipt.get("receipt_sha256", ""))
    unhashed = dict(receipt)
    unhashed.pop("receipt_sha256", None)
    if not stored_hash or sha256_json(unhashed) != stored_hash:
        raise ValueError("Selection receipt hash mismatch")
    expected = {
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "final_apply_bundle_sha256": bundle["bundle_sha256"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise ValueError(
                f"Selection receipt {field} does not match final bundle"
            )
    if not bool(receipt.get("hyperparameters_frozen")):
        raise ValueError("Selection receipt has not frozen hyperparameters")
    if receipt.get("test_results_used_for_selection") is not False:
        raise ValueError("Selection receipt does not guarantee test isolation")
    spec = receipt.get("selected_candidate_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("Selection receipt lacks embedded candidate spec")
    if str(spec.get("candidate_id")) != str(
        receipt.get("selected_candidate_id")
    ):
        raise ValueError("Embedded candidate spec ID does not match receipt")
    return dict(spec)


def _default_mcf_setting5() -> Dict[str, Any]:
    return {
        "steps": 600,
        "batch_size": 1,
        "retain_batch_size": 4,
        "emb_lm_lr": 1e-4,
        "forget_weight": 2.0,
        "retain_weight": 1.0,
        "forget_loss_type": "mcf_margin",
        "forget_margin": 1.0,
        "sampling_strategy": "epoch",
        "post_training_new_true_alpha": 0.75,
        "post_training_new_retain_alpha": 0.50,
        "post_training_new_true_retain_alpha": 0.25,
        "dtype": "bf16",
        "device_map": "single",
    }


def _run_mcf(
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
    mcf_path = materialized["mcf_path"]
    setting_dir = output_dir / "setting5e"
    options = {
        **_default_mcf_setting5(),
        **dict(spec.get("setting5e", {})),
    }
    command = [
        python,
        "scripts/gagd_compare.py",
        "--dataset",
        "mcf",
        "--model-path",
        str(spec["base_model_path"]),
        "--output-dir",
        str(setting_dir),
        "--mode",
        SETTING5E_MODE,
        "--seed",
        str(seed),
        "--forget-num",
        str(counts["forget_count"]),
        "--retain-num",
        str(counts["retain_count"]),
        "--mcf-cache-path",
        str(mcf_path),
        "--mcf-sample-mode",
        "first",
        "--official-sample-mode",
        "first",
        "--save-model",
    ]
    _append_options(command, options, MCF_SETTING5_ALLOWED)
    commands.append(command)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    checkpoint = setting_dir / SETTING5E_MODE / "checkpoint"
    config = setting_dir / "config_used.json"
    repair_dir = output_dir / "active_repair"
    repair_options = dict(spec.get("active_repair", {}))
    allowed_repair = {
        "active_margin",
        "dtype",
        "device_map",
        "margin_batch_size",
        "repair_steps",
        "repair_lr",
        "repair_optimizer",
        "hinge_weight",
        "delta_l2_lambda",
        "retain_kl_mu",
        "retain_calibration_num",
        "retain_calibration_seed",
        "repair_rank",
        "project_away_retain_hidden",
        "min_reloaded_forget_margin",
        "skip_ppl",
    }
    unknown = sorted(set(repair_options) - allowed_repair)
    if unknown:
        raise ValueError(f"Unsupported MCF active-repair options: {unknown}")
    environment = os.environ.copy()
    environment.update(
        {
            "OUT_ROOT": str(repair_dir),
            "MCF_PATH": str(mcf_path),
            "SAMPLE_MODE": "first",
            "SEED": str(seed),
            "FORGET_NUM": str(counts["forget_count"]),
            "RETAIN_NUM": str(counts["retain_count"]),
            "SKIP_PPL": "1",
            "REQUIRE_POST_RELOAD_ZERO": (
                "1" if stage in {"validation", "final_apply"} else "0"
            ),
        }
    )
    env_names = {
        "active_margin": "ACTIVE_MARGIN",
        "dtype": "DTYPE",
        "device_map": "DEVICE_MAP",
        "margin_batch_size": "MARGIN_BATCH_SIZE",
        "repair_steps": "REPAIR_STEPS",
        "repair_lr": "REPAIR_LR",
        "repair_optimizer": "REPAIR_OPTIMIZER",
        "hinge_weight": "HINGE_WEIGHT",
        "delta_l2_lambda": "DELTA_L2_LAMBDA",
        "retain_kl_mu": "RETAIN_KL_MU",
        "retain_calibration_num": "RETAIN_CALIBRATION_NUM",
        "retain_calibration_seed": "RETAIN_CALIBRATION_SEED",
        "repair_rank": "REPAIR_RANK",
        "project_away_retain_hidden": "PROJECT_AWAY_RETAIN_HIDDEN",
        "min_reloaded_forget_margin": "MIN_RELOADED_FORGET_MARGIN",
        "skip_ppl": "SKIP_PPL",
    }
    for key, value in repair_options.items():
        environment[env_names[key]] = (
            "1" if value is True else "0" if value is False else str(value)
        )
    repair_command = [
        "bash",
        "scripts/run_gagd_active_case_repair.sh",
        str(checkpoint),
        str(spec["base_model_path"]),
        str(config),
    ]
    commands.append(repair_command)
    if not dry_run:
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing MCF Setting 5e checkpoint {checkpoint}")
        try:
            subprocess.run(
                repair_command,
                cwd=PROJECT_DIR,
                env=environment,
                check=True,
            )
        except subprocess.CalledProcessError as error:
            evaluation_path = repair_dir / "official_eval_selected.json"
            selected_path = repair_dir / "selected_candidate.json"
            if (
                error.returncode == MCF_POST_RELOAD_REJECTION_EXIT_CODE
                and evaluation_path.exists()
                and selected_path.exists()
            ):
                official = read_json(evaluation_path)
                acceptance = official.get("post_reload_acceptance")
                if (
                    isinstance(acceptance, Mapping)
                    and acceptance.get("kind")
                    == "mcf_post_reload_acceptance"
                    and not bool(acceptance.get("passed"))
                ):
                    selected = read_json(selected_path)
                    checkpoint_path = Path(selected["checkpoint"])
                    if not checkpoint_path.is_absolute():
                        checkpoint_path = PROJECT_DIR / checkpoint_path
                    raise MCFPostReloadAcceptanceError(
                        checkpoint=checkpoint_path.resolve(),
                        evaluation_path=evaluation_path.resolve(),
                        acceptance=acceptance,
                    ) from error
            raise
        selected = read_json(repair_dir / "selected_candidate.json")
        selected_path = Path(selected["checkpoint"])
        if not selected_path.is_absolute():
            selected_path = PROJECT_DIR / selected_path
        return selected_path.resolve()
    return (repair_dir / "DRY_RUN_SELECTED_CHECKPOINT").resolve()


def _run_zsre(
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
    options = {
        "utility_drop_tolerance": 0.02,
        "max_ppl_ratio": 1.02,
        "skip_ppl": True,
        "save_selected_checkpoint": True,
        **dict(spec.get("pipeline", {})),
    }
    # save_selected_checkpoint is fixed below rather than accepted from config.
    options.pop("save_selected_checkpoint", None)
    command = [
        python,
        "scripts/zsre_gagd_setting5e_active_repair.py",
        "--model-path",
        str(spec["base_model_path"]),
        "--output-dir",
        str(output_dir),
        "--controlled-records-path",
        str(materialized["controlled_zsre_path"]),
        "--seed",
        str(seed),
        "--forget-num",
        str(counts["forget_count"]),
        "--retain-num",
        str(counts["retain_count"]),
        "--save-selected-checkpoint",
    ]
    original_path = spec.get("zsre_source_path")
    if original_path:
        command.extend(["--zsre-path", str(original_path)])
    _append_options(command, options, ZSRE_ALLOWED)
    commands.append(command)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_DIR, check=True)
    return (output_dir / "selected_checkpoint").resolve()


def _run_tofu(
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
    input_dir = next(iter(materialized.values())).parent
    base_model = str(spec["base_model_path"])
    forget_num = int(counts["forget_count"])
    retain_num = int(counts["retain_count"])
    if forget_num <= 0 or retain_num <= 0:
        raise ValueError("Controlled TOFU stage needs forget and retain rows")
    train_dir = output_dir / "four_settings"
    retain_batch = max(1, math_ceil_div(retain_num, forget_num))
    train_options = {
        "steps": forget_num,
        "batch_size": 1,
        "retain_batch_size": retain_batch,
        "emb_lm_lr": 2e-4,
        "forget_weight": 1.0,
        "retain_weight": float(retain_batch),
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
        TOFU_SOURCE_MODE,
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
    _append_options(train_command, train_options, TOFU_TRAIN_ALLOWED)
    commands.append(train_command)
    if not dry_run:
        subprocess.run(train_command, cwd=PROJECT_DIR, check=True)
    source_checkpoint = train_dir / TOFU_SOURCE_MODE / "checkpoint"

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
        TOFU_SOURCE_MODE,
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
    _append_options(restore_command, restore_options, TOFU_RESTORE_ALLOWED)
    commands.append(restore_command)
    if not dry_run:
        if not source_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing TOFU Setting 5 source checkpoint {source_checkpoint}"
            )
        subprocess.run(restore_command, cwd=PROJECT_DIR, check=True)
    restored_checkpoint = restore_dir / "checkpoint"

    repair_dir = output_dir / "active_repair"
    repair_options = {
        "retain_calibration_num": retain_num,
        "min_utility_probability_ratio": 0.98,
        "utility_constraint_mode": "aggregate",
        "require_input_retain_target": True,
        "require_utility_constraints": True,
        "dtype": restore_options.get("dtype", "bf16"),
        "device_map": restore_options.get("device_map", "single"),
        **dict(spec.get("active_repair", {})),
    }
    repair_command = [
        python,
        "scripts/tofu_gagd_active_forget_repair.py",
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
    _append_options(repair_command, repair_options, TOFU_REPAIR_ALLOWED)
    commands.append(repair_command)
    if not dry_run:
        if not restored_checkpoint.exists():
            raise FileNotFoundError(
                f"Missing TOFU Setting 5e checkpoint {restored_checkpoint}"
            )
        subprocess.run(repair_command, cwd=PROJECT_DIR, check=True)
    return (repair_dir / "checkpoint").resolve()


def math_ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument(
        "--phase",
        choices=["development", "final_apply"],
        required=True,
    )
    parser.add_argument(
        "--stage",
        choices=["train", "validation", "final_apply"],
        required=True,
    )
    parser.add_argument("--candidate-spec", default=None)
    parser.add_argument("--selection-receipt", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bundle_path = Path(args.bundle).resolve()
    if args.phase == "development":
        if args.stage not in {"train", "validation"}:
            raise ValueError(
                "Development phase supports train or validation stage"
            )
        if not args.candidate_spec:
            raise ValueError("Development run requires --candidate-spec")
        if args.selection_receipt:
            raise ValueError(
                "Development run must not receive a selection receipt"
            )
        bundle = load_development_bundle(bundle_path)
        spec_path = Path(args.candidate_spec).resolve()
        spec = _load_spec(spec_path)
        materialized_metadata = bundle["materialized_inputs"][args.stage]
        selection_receipt = None
    else:
        if args.stage != "final_apply":
            raise ValueError("Test phase supports only final_apply")
        if not args.selection_receipt:
            raise ValueError("Final apply requires --selection-receipt")
        if args.candidate_spec:
            raise ValueError(
                "Final apply uses the frozen spec embedded in the receipt"
            )
        bundle = load_final_apply_bundle(bundle_path)
        selection_path = Path(args.selection_receipt).resolve()
        selection_receipt = read_json(selection_path)
        spec = _validate_receipt(selection_receipt, bundle)
        spec_path = Path(
            selection_receipt["selected_candidate_spec_path"]
        ).resolve()
        if (
            spec_path.exists()
            and sha256_file(spec_path)
            != selection_receipt["selected_candidate_spec_sha256"]
        ):
            raise ValueError("Frozen candidate spec changed after selection")
        materialized_metadata = bundle["materialized_inputs"]

    if str(spec["dataset"]) != str(bundle["dataset"]):
        raise ValueError("Candidate spec dataset does not match bundle")
    if (
        spec.get("seed") is not None
        and int(spec["seed"]) != int(bundle["seed"])
    ):
        raise ValueError("Candidate spec seed does not match protocol")
    materialized = _verify_materialized(
        bundle_path,
        materialized_metadata,
    )
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    commands: List[List[str]] = []
    plan = {
        "schema_version": 1,
        "kind": "controlled_setting5e_run_plan",
        "phase": args.phase,
        "stage": args.stage,
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "bundle_sha256": bundle["bundle_sha256"],
        "candidate_id": spec["candidate_id"],
        "candidate_spec_sha256": sha256_json(spec),
        "base_model_path": spec["base_model_path"],
        "materialized_inputs": {
            key: str(path) for key, path in materialized.items()
        },
        "materialized_input_sha256": materialized_metadata["sha256"],
        "fresh_base_required": True,
        "judge_b_prompts_loaded": False,
        "dry_run": args.dry_run,
    }
    write_json(output_dir / "run_plan.json", plan)
    runners = {
        "mcf": _run_mcf,
        "zsre": _run_zsre,
        "tofu": _run_tofu,
    }

    def rendered_commands() -> List[Dict[str, Any]]:
        return [
            {
                "argv": command,
                "shell_rendering_for_review": shlex.join(command),
            }
            for command in commands
        ]

    try:
        selected_checkpoint = runners[bundle["dataset"]](
            python=args.python,
            spec=spec,
            materialized=materialized,
            counts=materialized_metadata,
            seed=int(bundle["seed"]),
            stage=args.stage,
            output_dir=output_dir,
            dry_run=args.dry_run,
            commands=commands,
        )
    except MCFPostReloadAcceptanceError as error:
        plan["commands"] = rendered_commands()
        plan["status"] = "rejected"
        plan["selected_checkpoint"] = str(error.checkpoint)
        plan["post_reload_acceptance"] = error.acceptance
        write_json(output_dir / "run_plan.json", plan)
        rejection_receipt = {
            "schema_version": 1,
            "kind": "controlled_model_application_receipt",
            "status": "rejected",
            "phase": args.phase,
            "stage": args.stage,
            "protocol_id": bundle["protocol_id"],
            "dataset": bundle["dataset"],
            "fold": bundle["fold"],
            "bundle_sha256": bundle["bundle_sha256"],
            "final_apply_bundle_sha256": (
                bundle["bundle_sha256"]
                if args.phase == "final_apply"
                else None
            ),
            "test_bundle_sha256": (
                selection_receipt.get("test_bundle_sha256")
                if selection_receipt is not None
                else None
            ),
            "candidate_id": spec["candidate_id"],
            "candidate_spec_sha256": sha256_json(spec),
            "selection_receipt_sha256": (
                selection_receipt.get("receipt_sha256")
                if selection_receipt is not None
                else None
            ),
            "base_model_identity": _public_model_identity(
                Path(str(spec["base_model_path"]))
            ),
            "attempted_checkpoint": str(error.checkpoint),
            "attempted_checkpoint_identity": (
                _public_model_identity(error.checkpoint)
                if error.checkpoint.exists()
                else None
            ),
            "post_reload_evaluation": str(error.evaluation_path),
            "post_reload_acceptance": error.acceptance,
            "started_from_fresh_base": True,
            "judge_b_prompts_or_results_used": False,
            "test_results_used_for_repair": False,
            "commands": plan["commands"],
            "dry_run": False,
        }
        unhashed = dict(rejection_receipt)
        rejection_receipt["receipt_sha256"] = sha256_json(unhashed)
        receipt_path = output_dir / "application_receipt.json"
        write_json(receipt_path, rejection_receipt)
        print(f"Wrote rejected application receipt to {receipt_path}")
        raise SystemExit(MCF_POST_RELOAD_REJECTION_EXIT_CODE) from error

    plan["commands"] = rendered_commands()
    post_reload_acceptance = None
    if (
        not args.dry_run
        and bundle["dataset"] == "mcf"
        and args.stage in {"validation", "final_apply"}
    ):
        evaluation_path = (
            output_dir / "active_repair" / "official_eval_selected.json"
        )
        if not evaluation_path.exists():
            raise FileNotFoundError(
                "MCF post-reload official evaluation is missing: "
                f"{evaluation_path}"
            )
        post_reload_acceptance = read_json(evaluation_path).get(
            "post_reload_acceptance"
        )
        try:
            validate_mcf_post_reload_acceptance(post_reload_acceptance)
        except ValueError as error:
            raise RuntimeError(
                "MCF validation/final application lacks a passing "
                "post-reload acceptance gate"
            ) from error
    plan["status"] = "accepted"
    plan["post_reload_acceptance"] = post_reload_acceptance
    plan["selected_checkpoint"] = str(selected_checkpoint)
    write_json(output_dir / "run_plan.json", plan)
    if not args.dry_run and not selected_checkpoint.exists():
        raise FileNotFoundError(
            f"Pipeline did not create selected checkpoint {selected_checkpoint}"
        )
    receipt = {
        "schema_version": 1,
        "kind": "controlled_model_application_receipt",
        "status": "accepted",
        "phase": args.phase,
        "stage": args.stage,
        "protocol_id": bundle["protocol_id"],
        "dataset": bundle["dataset"],
        "fold": bundle["fold"],
        "bundle_sha256": bundle["bundle_sha256"],
        "final_apply_bundle_sha256": (
            bundle["bundle_sha256"]
            if args.phase == "final_apply"
            else None
        ),
        "test_bundle_sha256": (
            selection_receipt.get("test_bundle_sha256")
            if selection_receipt is not None
            else None
        ),
        "candidate_id": spec["candidate_id"],
        "candidate_spec_sha256": sha256_json(spec),
        "selection_receipt_sha256": (
            selection_receipt.get("receipt_sha256")
            if selection_receipt is not None
            else None
        ),
        "base_model_identity": _public_model_identity(
            Path(str(spec["base_model_path"]))
        ),
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_identity": (
            _public_model_identity(selected_checkpoint)
            if not args.dry_run
            else None
        ),
        "post_reload_acceptance": post_reload_acceptance,
        "started_from_fresh_base": True,
        "judge_b_prompts_or_results_used": False,
        "test_results_used_for_repair": False,
        "commands": plan["commands"],
        "dry_run": args.dry_run,
    }
    unhashed = dict(receipt)
    receipt["receipt_sha256"] = sha256_json(unhashed)
    receipt_path = output_dir / "application_receipt.json"
    write_json(receipt_path, receipt)
    print(f"Wrote {receipt_path}")
    print(f"Selected checkpoint: {selected_checkpoint}")


if __name__ == "__main__":
    main()
