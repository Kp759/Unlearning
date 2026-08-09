#!/usr/bin/env python3
"""RWKU generated-corpus Setting 5e + rank-0/nullspace restoration.

This is an isolated development-method adapter.  It trains the preserved
600-step generated-corpus Setting 5e checkpoint, obtains perfect forgetting
with an unrestricted selected-row hard-tail repair, projects that repair into
the exact forget-hidden span, and restores utility only in a rank-16 basis
orthogonal to that span.  No official RWKU record is opened by this program;
the final stage is a read-only receipt verification after CHECKPOINT_FROZEN.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import gagd_active_case_repair as active
import gagd_compare as gagd
import rwku_experiment as legacy
import rwku_generated_s5e_rank2_active_repair as rank2
import rwku_setting5e_utility_controlled as utility
from build_rwku_entity_facts import official_locked_descriptor
from build_rwku_matched_protection import build_matched_protection
from rwku_artifact_access import (
    TARGET_ONLY_PROTOCOL_LABEL,
    make_artifact,
    read_artifact,
    sha256_file,
    sha256_json,
    sha256_path,
    write_artifact,
)
from rwku_checkpoint_receipt import (
    create_checkpoint_receipt,
    load_receipt,
    verify_frozen_identities,
)
from rwku_data import target_for_seed
from rwku_eval import evaluate_perplexity, load_wikidata_text
from rwku_fact_sampler import build_fact_cycle_plan, exposure_report, plan_sha256


SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_ROOT = SCRIPT_PATH.parents[1]
EXPERIMENT_ID = "rwku-s5e600-rank0-nullrestore16-sk-v3atomic-seed0-v1"
EXPECTED_BUNDLE_SHA256 = (
    "3070423ec358fca5fd3a4bb9cc2f645f9a70eed6c909a659432d8951fc5ab6ec"
)
METHOD = (
    "Setting 5e @600 + rank-0 perfect active LM-head repair + "
    "forget-subspace projection + rank-16 forget-nullspace utility restoration"
)
PROTOCOL_STATUS = "rwku_target_generated_s5e600_rank0_nullrestore16_method_extension"
STATE_SCHEMA = "rwku_generated_s5e_rank0_nullrestore16_state_v1"
CONFIG_SCHEMA = "rwku_generated_s5e_rank0_nullrestore16_configuration_v1"

SETTING5_STEPS = 600
RANK0_REPAIR_RANK = 0
RANK0_REPAIR_STEPS = 100
RANK0_REPAIR_LR = 5e-3
ACTIVE_MARGIN = 0.25
HARD_TAIL_WEIGHT = 2.0
RANK0_L2_LAMBDA = 1e-4
RESTORE_RANK = 16
RESTORE_STEPS = 800
RESTORE_LR = 5e-4
RESTORE_L2_LAMBDA = 1e-4
RESTORE_SNAPSHOT_INTERVAL = 50
RETAIN_CALIBRATION_NUM = 200


def run_dir(args: argparse.Namespace) -> Path:
    return Path(args.output_root) / args.experiment_id


def state_path(args: argparse.Namespace) -> Path:
    return run_dir(args) / "experiment_state.json"


def read_state(args: argparse.Namespace) -> Dict[str, Any]:
    with state_path(args).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if value.get("schema_version") != STATE_SCHEMA:
        raise ValueError("Unsupported nullrestore16 experiment-state schema")
    if value.get("experiment_id") != args.experiment_id:
        raise ValueError("Experiment-state identity differs from invocation")
    return value


def write_state(args: argparse.Namespace, stage: str, **extra: Any) -> None:
    previous: Dict[str, Any] = {}
    if state_path(args).is_file():
        previous = read_state(args)
    utility.atomic_json_write(
        state_path(args),
        {
            **previous,
            "schema_version": STATE_SCHEMA,
            "experiment_id": args.experiment_id,
            "state": stage,
            "updated_at_utc": utility.utc_now(),
            **extra,
        },
    )


def configuration(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "method": METHOD,
        "protocol_label": TARGET_ONLY_PROTOCOL_LABEL,
        "protocol_status": PROTOCOL_STATUS,
        "development_ablation": True,
        "experiment_id": args.experiment_id,
        "seed": args.seed,
        "model_path": str(Path(args.model_path).resolve()),
        "model_revision": args.model_revision,
        "dtype": args.dtype,
        "setting5": {
            "steps": SETTING5_STEPS,
            "batch_size": rank2.SETTING5_BATCH_SIZE,
            "retain_batch_size": rank2.SETTING5_RETAIN_BATCH_SIZE,
            "emb_lm_lr": rank2.SETTING5_LR,
            "forget_weight": rank2.SETTING5_FORGET_WEIGHT,
            "retain_weight": rank2.SETTING5_RETAIN_WEIGHT,
            "forget_margin": rank2.SETTING5_FORGET_MARGIN,
            "optimizer": "adamw",
            "sampling_strategy": "balanced_fact_cycle_epoch",
        },
        "stage_a_rank0_hard_tail": {
            "repair_rank": RANK0_REPAIR_RANK,
            "steps": RANK0_REPAIR_STEPS,
            "learning_rate": RANK0_REPAIR_LR,
            "optimizer": "adamw",
            "active_margin": ACTIVE_MARGIN,
            "hard_tail_weight": HARD_TAIL_WEIGHT,
            "delta_l2_lambda": RANK0_L2_LAMBDA,
            "stop_when_all_satisfied": True,
        },
        "stage_b_forget_projection": {
            "basis": "orthonormal_row_basis(all active sensitive and neutral hidden states)",
            "projection": "delta_rank0 @ B_F.T @ B_F",
            "arithmetic": "fp32",
        },
        "stage_c_nullspace_restoration": {
            "requested_rank": RESTORE_RANK,
            "steps": RESTORE_STEPS,
            "learning_rate": RESTORE_LR,
            "optimizer": "adamw",
            "parameterization": "per-row coefficients C @ B_R",
            "l2_lambda": RESTORE_L2_LAMBDA,
            "snapshot_interval": RESTORE_SNAPSHOT_INTERVAL,
        },
        "strict_gates": {
            "active_violation_count": 0,
            "minimum_generated_margin": ACTIVE_MARGIN,
            "incremental_retain_kl_max": 0.01,
            "full_retain_probability_ratio": [0.995, 1.005],
            "geometric_retain_probability_ratio": [0.98, 1.02],
            "minimum_protected_answer_probability_ratio": 0.999,
            "protected_top1_changes": 0,
            "protected_selected_row_logit_drift_max": 0.05,
            "proxy_ppl_ratio_max": 1.02,
            "nonselected_rows_equal_setting5": True,
        },
        "generated_training_bundle": str(Path(args.generated_training_bundle).resolve()),
        "generator_receipt": str(Path(args.generator_receipt).resolve()),
        "expected_generated_training_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "mcf_path": str(Path(args.mcf_path).resolve()),
        "mcf_retain_num": args.mcf_retain_num,
        "mcf_gate_num": args.mcf_gate_num,
        "protection_sources": [str(Path(path).resolve()) for path in args.protection_source],
        "wikidata_dir": str(Path(args.wikidata_dir).resolve()),
        "official_evaluation_access_before_freeze": "forbidden",
        "official_evaluation_implemented_by_this_entrypoint": False,
    }


def project_delta_to_forget_span(
    delta_rows: torch.Tensor, forget_basis: torch.Tensor
) -> torch.Tensor:
    """Project [selected_rows, hidden] deltas into row-span(B_F), in FP32."""
    delta = delta_rows.float()
    basis = forget_basis.float()
    if delta.ndim != 2 or basis.ndim != 2 or delta.shape[1] != basis.shape[1]:
        raise ValueError("delta/B_F must be compatible two-dimensional row matrices")
    if basis.numel() == 0:
        return torch.zeros_like(delta)
    return (delta @ basis.transpose(0, 1)) @ basis


def build_restore_basis(
    retain_hidden: torch.Tensor,
    forget_basis: torch.Tensor,
    *,
    max_rank: int = RESTORE_RANK,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return H_R with its forget component removed and an ordered basis."""
    if retain_hidden.ndim != 2 or forget_basis.ndim != 2:
        raise ValueError("H_R and B_F must be matrices")
    if retain_hidden.shape[1] != forget_basis.shape[1]:
        raise ValueError("H_R and B_F hidden dimensions differ")
    null_hidden = active.project_rows_away(retain_hidden.float(), forget_basis.float())
    basis = active.orthonormal_row_basis(null_hidden, max_rank=max_rank)
    return null_hidden, basis


def restoration_delta(coefficients: torch.Tensor, restore_basis: torch.Tensor) -> torch.Tensor:
    if coefficients.ndim != 2 or restore_basis.ndim != 2:
        raise ValueError("Restoration coefficients/basis must be matrices")
    if coefficients.shape[1] != restore_basis.shape[0]:
        raise ValueError("Restoration coefficient rank differs from B_R")
    return coefficients.float() @ restore_basis.float()


def incremental_retain_kl(candidate_absolute: Any, setting5_absolute: Any) -> Any:
    """Corrected non-negative KL increase above the Setting5 absolute KL."""
    if isinstance(candidate_absolute, torch.Tensor):
        baseline = torch.as_tensor(
            setting5_absolute,
            dtype=candidate_absolute.dtype,
            device=candidate_absolute.device,
        )
        return torch.relu(candidate_absolute - baseline)
    return max(0.0, float(candidate_absolute) - float(setting5_absolute))


def tensor_sha256(value: torch.Tensor) -> str:
    payload = value.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def nonselected_rows_equal_setting5(
    current: torch.Tensor, setting5: torch.Tensor, selected_ids: Sequence[int]
) -> bool:
    if current.shape != setting5.shape:
        return False
    mask = torch.ones(current.shape[0], dtype=torch.bool, device=current.device)
    if selected_ids:
        ids = torch.tensor(selected_ids, dtype=torch.long, device=current.device)
        mask[ids] = False
    reference = setting5.to(device=current.device, dtype=current.dtype)
    return torch.equal(current.detach()[mask], reference[mask])


def _materialize_delta(
    output_weight: torch.Tensor,
    selected_ids: Sequence[int],
    setting5_rows: torch.Tensor,
    delta: torch.Tensor,
) -> None:
    if not selected_ids:
        return
    ids = torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)
    rows = setting5_rows.to(device=output_weight.device, dtype=output_weight.dtype)
    update = delta.to(device=output_weight.device, dtype=output_weight.dtype)
    if update.shape != rows.shape:
        raise ValueError("Materialized selected-row delta shape mismatch")
    output_weight.index_copy_(0, ids, rows + update)


def _margin_summary(margins: torch.Tensor) -> Dict[str, Any]:
    return {
        "active_violation_count": int((margins < ACTIVE_MARGIN).sum().item()),
        "minimum_margin": float(margins.min().item()) if margins.numel() else math.inf,
        "mean_margin": float(margins.mean().item()) if margins.numel() else math.inf,
        "maximum_margin": float(margins.max().item()) if margins.numel() else math.inf,
    }


def optimize_rank0_hard_tail(
    caches: Sequence[active.RewriteDeltaCache],
    *,
    n_rows: int,
    hidden_size: int,
    device: torch.device,
    steps: int = RANK0_REPAIR_STEPS,
    learning_rate: float = RANK0_REPAIR_LR,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """Unrestricted selected-row repair driven by the worst active violation."""
    module = active.SelectedRowDelta(n_rows, hidden_size, device=device)
    optimizer = torch.optim.AdamW(module.parameters(), lr=learning_rate)
    logs: List[Dict[str, Any]] = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        delta = module.effective_delta()
        margins = active.margins_from_delta_caches(caches, delta)
        violations = torch.relu(ACTIVE_MARGIN - margins)
        hard_tail = violations.max().square() if violations.numel() else delta.sum() * 0.0
        l2 = delta.square().sum()
        loss = HARD_TAIL_WEIGHT * hard_tail + RANK0_L2_LAMBDA * l2
        before = _margin_summary(margins.detach())
        if before["active_violation_count"] == 0:
            break
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            after_margins = active.margins_from_delta_caches(
                caches, module.effective_delta()
            )
            after = _margin_summary(after_margins)
        logs.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "hard_tail_loss": float(hard_tail.detach().cpu()),
                "delta_l2": float(l2.detach().cpu()),
                "violations_before": before["active_violation_count"],
                "violations_after": after["active_violation_count"],
                "minimum_margin_after": after["minimum_margin"],
            }
        )
        if after["active_violation_count"] == 0:
            break
    delta = module.effective_delta().detach().float()
    final = _margin_summary(active.margins_from_delta_caches(caches, delta))
    if final["active_violation_count"] != 0:
        raise RuntimeError("Stage A failed to obtain perfect generated forgetting")
    return delta, logs, {
        "steps_completed": len(logs),
        "repair_rank": RANK0_REPAIR_RANK,
        "hard_tail_objective": "squared maximum positive margin violation",
        **final,
    }


def _assert_bundle_identity(path: Path) -> None:
    actual = sha256_file(path)
    if actual != EXPECTED_BUNDLE_SHA256:
        raise ValueError(
            "Frozen Stephen King bundle SHA-256 differs from the pinned corpus: "
            f"expected={EXPECTED_BUNDLE_SHA256}, actual={actual}"
        )


def prepare_stage(args: argparse.Namespace) -> None:
    destination = run_dir(args)
    if destination.exists():
        raise ValueError(f"Refusing to reuse experiment directory: {destination}")
    if args.seed != 0 or args.experiment_id != EXPERIMENT_ID:
        raise ValueError("This isolated development experiment is pinned to seed 0 and its new ID")
    for path, label in (
        (args.generated_training_bundle, "generated training bundle"),
        (args.generator_receipt, "generator receipt"),
        (args.model_path, "Base model"),
        (args.mcf_path, "MCF protection source"),
        *[(path, "target-independent protection source") for path in args.protection_source],
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
        rank2.reject_official_path(Path(path), label=label)
    _assert_bundle_identity(args.generated_training_bundle)
    rank2.verify_corpus_manifest(args)
    receipt = read_artifact(
        args.generator_receipt, stage="prepare", expected_role="generator_receipt"
    )
    if receipt["payload"].get("official_rwku_records_accessed") is not False:
        raise ValueError("Generator receipt does not attest official-data non-access")
    target = target_for_seed(args.seed)
    if receipt.get("protocol_label") != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Generator receipt belongs to another RWKU protocol")
    generated_subject = (
        receipt["payload"].get("target_entity")
        or receipt["payload"].get("subject")
        or receipt.get("metadata", {}).get("subject")
    )
    if generated_subject and str(generated_subject) != target.subject:
        raise ValueError("Generator receipt target differs from RWKU seed 0")
    config = configuration(args)
    destination.mkdir(parents=True, exist_ok=False)
    utility.atomic_json_write(
        destination / "configuration_manifest.json",
        {
            "schema_version": CONFIG_SCHEMA,
            "configuration": config,
            "configuration_sha256": sha256_json(config),
            "frozen_at_utc": utility.utc_now(),
        },
    )
    locked = make_artifact(
        "official_locked_eval",
        official_locked_descriptor(args.seed, include_level12=True),
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=PROTOCOL_STATUS,
        metadata={"experiment_id": args.experiment_id, "subject": target.subject},
    )
    write_artifact(destination / "official_locked_eval.json", locked)
    write_state(
        args,
        "PREPARED",
        target={"subject": target.subject, "entity_id": f"rwku:{target.directory}"},
        configuration_sha256=sha256_json(config),
        model_sha256=sha256_path(args.model_path),
        generated_training_bundle_sha256=sha256_file(args.generated_training_bundle),
        generator_receipt_sha256=sha256_file(args.generator_receipt),
        corpus_identities=rank2.corpus_identities(args),
        official_rwku_records_accessed=False,
        official_evaluation_opened=False,
    )


def _verify_prepared(args: argparse.Namespace) -> Dict[str, Any]:
    state = read_state(args)
    if state.get("state") != "PREPARED":
        raise ValueError(f"Training requires PREPARED, got {state.get('state')}")
    _assert_bundle_identity(args.generated_training_bundle)
    if sha256_file(args.generated_training_bundle) != state["generated_training_bundle_sha256"]:
        raise ValueError("Generated training bundle changed after prepare")
    if sha256_file(args.generator_receipt) != state["generator_receipt_sha256"]:
        raise ValueError("Generator receipt changed after prepare")
    if sha256_path(args.model_path) != state["model_sha256"]:
        raise ValueError("Base model changed after prepare")
    if sha256_json(configuration(args)) != state["configuration_sha256"]:
        raise ValueError("Resolved configuration changed after prepare")
    for name, identity in state["corpus_identities"].items():
        if sha256_file(Path(identity["path"])) != identity["sha256"]:
            raise ValueError(f"Frozen corpus member changed: {name}")
    return state


def _prepare_protection(
    args: argparse.Namespace, training: Mapping[str, Any]
) -> Tuple[Path, Path, Path, Path, List[Mapping[str, Any]], List[Mapping[str, Any]]]:
    protection_dir = run_dir(args) / "protection"
    protection = build_matched_protection(
        training_bundle_path=args.generated_training_bundle,
        source_corpora=args.protection_source,
        output_dir=protection_dir,
        vocabulary_path=args.protection_vocabulary,
        split_seed=args.seed,
        minimum_train_per_key=1,
        minimum_gate_per_key=1,
        strict=False,
        tokenizer=None,
    )
    train_path = protection_dir / "matched_protection_train.json"
    gate_path = protection_dir / "matched_protection_gate.json"
    coverage_path = protection_dir / "matched_protection_coverage.json"
    if not train_path.is_file() or not gate_path.is_file() or not coverage_path.is_file():
        raise ValueError("Matched protection did not produce all frozen artifacts")
    train = read_artifact(train_path, stage="train", gradient=True)
    gate = read_artifact(gate_path, stage="train", selection=True)
    train_hashes = {row.get("content_sha256") for row in train["payload"].get("records", [])}
    gate_hashes = {row.get("content_sha256") for row in gate["payload"].get("records", [])}
    if (train_hashes & gate_hashes) - {None}:
        raise ValueError("Matched protection train/gate rows overlap")
    all_records, all_examples = legacy.load_mcf_retain(
        args.mcf_path,
        seed=args.seed,
        retain_num=args.mcf_retain_num + args.mcf_gate_num,
    )
    retain_records = all_records[: args.mcf_retain_num]
    gate_records = all_records[args.mcf_retain_num :]
    if {sha256_json(row) for row in retain_records} & {sha256_json(row) for row in gate_records}:
        raise ValueError("MCF optimization and gate partitions overlap")
    mcf_train = protection_dir / "mcf_optimization_manifest.json"
    mcf_gate = protection_dir / "mcf_gate_manifest.json"
    rank2._write_mcf_manifest(mcf_train, retain_records, "optimization")
    rank2._write_mcf_manifest(mcf_gate, gate_records, "gate")
    utility.atomic_json_write(
        protection_dir / "protection_diagnostics.json",
        {
            "coverage_policy": "allow_unmatched_generated_target_keys_but_audit",
            "matched_protection_key_count": len(protection["coverage"]),
            "matched_protection_covered_key_count": sum(
                row["coverage_status"] == "covered" for row in protection["coverage"]
            ),
            "matched_protection_insufficient_key_count": len(protection["insufficient"]),
            "matched_protection_insufficient_keys": [
                row["normalized_key"] for row in protection["insufficient"]
            ],
            "train_gate_content_hash_overlap": [],
            "official_rwku_records_accessed": False,
        },
    )
    return train_path, gate_path, mcf_train, mcf_gate, retain_records, gate_records


def _actual_active_margins(
    model: nn.Module,
    tokenizer: Any,
    instances: Sequence[active.MCFPromptInstance],
    points: Sequence[utility.TrainingPoint],
    active_view_ids: set[str],
    selected_ids: Sequence[int],
    device: torch.device,
    batch_size: int,
    llama_like: bool,
) -> torch.Tensor:
    caches = active.build_prompt_instance_delta_caches(
        model, tokenizer, instances, selected_ids, device, batch_size, llama_like
    )
    chosen = [cache for point, cache in zip(points, caches) if point.view_id in active_view_ids]
    zero = torch.zeros(
        (len(selected_ids), model.get_output_embeddings().weight.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    return active.margins_from_delta_caches(chosen, zero) if chosen else zero.new_empty((0,))


def _strict_candidate_gate(
    *,
    margins: torch.Tensor,
    setting5_absolute_kl: float,
    candidate_absolute_kl: float,
    protection: Mapping[str, Any],
    proxy_ppl: float,
    setting5_proxy_ppl: float,
    nonselected_equal: bool,
) -> Tuple[bool, Dict[str, Any]]:
    summary = _margin_summary(margins)
    increase = float(incremental_retain_kl(candidate_absolute_kl, setting5_absolute_kl))
    checks = {
        "active_violations_zero": summary["active_violation_count"] == 0,
        "minimum_generated_margin": summary["minimum_margin"] >= ACTIVE_MARGIN,
        "incremental_retain_kl": increase <= 0.01,
        "full_retain_probability_ratio": 0.995
        <= float(protection["full_retain_probability_ratio"])
        <= 1.005,
        "geometric_retain_probability_ratio": 0.98
        <= float(protection["geometric_retain_probability_ratio"])
        <= 1.02,
        "minimum_protected_answer_probability_ratio": float(
            protection["minimum_protected_answer_probability_ratio"]
        )
        >= 0.999,
        "protected_top1_changes": int(protection["protected_top1_changes"]) == 0,
        "protected_selected_row_logit_drift": float(
            protection["protected_selected_row_logit_drift"]
        )
        <= 0.05,
        "proxy_ppl_ratio": proxy_ppl <= setting5_proxy_ppl * 1.02,
        "nonselected_rows_equal_setting5": bool(nonselected_equal),
    }
    return all(checks.values()), {
        **summary,
        "setting5_absolute_retain_kl": setting5_absolute_kl,
        "candidate_absolute_retain_kl": candidate_absolute_kl,
        "retain_kl_increase": increase,
        "protection": dict(protection),
        "proxy_ppl": proxy_ppl,
        "setting5_proxy_ppl": setting5_proxy_ppl,
        "proxy_ppl_ratio": proxy_ppl / setting5_proxy_ppl,
        "checks": checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "all_strict_gates_pass": all(checks.values()),
    }


def _utility_penalty(report: Mapping[str, Any]) -> Tuple[float, ...]:
    """Deterministic fallback ordering among perfect-forgetting candidates."""
    protection = report["protection"]
    return (
        float(report["retain_kl_increase"]),
        abs(math.log(max(float(protection["full_retain_probability_ratio"]), 1e-30))),
        abs(math.log(max(float(protection["geometric_retain_probability_ratio"]), 1e-30))),
        float(protection["protected_selected_row_logit_drift"]),
        float(report["proxy_ppl_ratio"]),
    )


def train_stage(args: argparse.Namespace) -> None:
    state = _verify_prepared(args)
    training = read_artifact(
        args.generated_training_bundle,
        stage="train",
        gradient=True,
        expected_role="training_bundle",
    )
    generator = read_artifact(
        args.generator_receipt, stage="train", expected_role="generator_receipt"
    )
    if training.get("protocol_label") != TARGET_ONLY_PROTOCOL_LABEL:
        raise ValueError("Generated bundle belongs to another protocol")
    if generator["payload"].get("final_entity_fact_bundle_sha256") != training.get("sha256"):
        raise ValueError("Generator receipt does not bind the training bundle")
    legacy._validate_training_bundle_sources(
        training, training_source=legacy.TRAINING_SOURCE_TARGET_ONLY
    )
    for path in (args.mcf_path, *args.protection_source):
        rank2.reject_official_path(Path(path), label="pre-freeze protection input")
    write_state(args, "TRAINING", official_rwku_records_accessed=False)

    matched_train_path, matched_gate_path, mcf_train_manifest, mcf_gate_manifest, retain_records, gate_records = _prepare_protection(args, training)
    matched_train = read_artifact(matched_train_path, stage="train", gradient=True)
    matched_gate = read_artifact(matched_gate_path, stage="train", selection=True)

    legacy.set_all_seeds(args.seed)
    dtype = legacy.dtype_from_name(args.dtype)
    model, tokenizer = legacy.load_model_and_tokenizer(
        args.model_path,
        dtype=dtype,
        for_training=True,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    matched_train_examples = legacy._protection_examples(matched_train, tokenizer)
    matched_gate_examples = legacy._protection_examples(matched_gate, tokenizer)
    _, all_mcf_examples = legacy.load_mcf_retain(
        args.mcf_path,
        seed=args.seed,
        retain_num=args.mcf_retain_num + args.mcf_gate_num,
    )
    retain_examples = [*all_mcf_examples[: args.mcf_retain_num], *matched_train_examples]
    gate_examples = [*all_mcf_examples[args.mcf_retain_num :], *matched_gate_examples]

    views = list(training["payload"].get("views", []))
    points = utility.compile_training_points(tokenizer, training)
    forget_examples, examples_by_view = legacy.setting5_entity_fact_examples(
        tokenizer, views, include_reverse=False
    )
    views_by_fact: Dict[str, List[Mapping[str, Any]]] = {}
    for view in views:
        views_by_fact.setdefault(str(view["fact_id"]), []).append(view)
    plan = build_fact_cycle_plan(views_by_fact, steps=SETTING5_STEPS, seed=args.seed)
    batches = [[examples_by_view[str(item["view_id"])]] for item in plan]
    exposures = exposure_report(plan, seed=args.seed, tokenizer=tokenizer)
    exposures["plan_sha256"] = plan_sha256(plan)
    utility.atomic_json_write(run_dir(args) / "fact_exposure_report.json", exposures)
    setting_args = rank2._setting5_args(args, batches)
    started = time.perf_counter()
    setting_summary = gagd.train_mode(
        model,
        tokenizer,
        forget_examples,
        retain_examples,
        selected_ids=[],
        mode=legacy.SETTING5_MODE,
        args=setting_args,
        mode_dir=run_dir(args) / "setting5_training",
    )
    setting5_training_seconds = time.perf_counter() - started
    legacy.prepare_model_for_evaluation(model)
    setting5_dir = run_dir(args) / "setting5_training" / "checkpoint"
    legacy.save_checkpoint(model, tokenizer, setting5_dir)

    # Discover active points with all legal generated sensitive/neutral rows,
    # then freeze the final row set by special/protected-answer policy.
    requested_ids = [
        token_id
        for point in points
        for answer in (point.sensitive_answer, point.neutral_answer)
        for token_id in utility._completion_token_ids(tokenizer, answer)
    ]
    preliminary_ids, preliminary_special = rank2.exclude_special_rows(tokenizer, requested_ids)
    if not preliminary_ids:
        raise ValueError("No legal generated sensitive/neutral output rows are available")
    instances = rank2._point_instances(points)
    device = next(model.parameters()).device
    llama_like = active.is_llama_like(model, tokenizer)
    preliminary_caches = active.build_prompt_instance_delta_caches(
        model,
        tokenizer,
        instances,
        preliminary_ids,
        device,
        args.eval_batch_size,
        llama_like,
    )
    preliminary_zero = torch.zeros(
        (len(preliminary_ids), model.get_output_embeddings().weight.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    preliminary_margins = active.margins_from_delta_caches(
        preliminary_caches, preliminary_zero
    )
    active_rows: List[Dict[str, Any]] = []
    active_points: List[utility.TrainingPoint] = []
    bundle_sha = sha256_file(args.generated_training_bundle)
    for point, margin in zip(points, preliminary_margins.detach().cpu().tolist()):
        if float(margin) < ACTIVE_MARGIN:
            provenance = rank2.validate_active_provenance(point, bundle_sha256=bundle_sha)
            provenance.update({"margin_before": float(margin), "required_margin": ACTIVE_MARGIN})
            active_rows.append(provenance)
            active_points.append(point)
    if not active_points:
        raise ValueError("Setting5 already has no active points; this ablation is not applicable")
    active_requested = [
        token_id
        for point in active_points
        for answer in (point.sensitive_answer, point.neutral_answer)
        for token_id in utility._completion_token_ids(tokenizer, answer)
    ]
    selected_ids, active_special = rank2.exclude_special_rows(tokenizer, active_requested)
    protected_ids = rank2.protected_answer_row_ids(tokenizer, [*retain_examples, *gate_examples])
    protected_excluded = sorted(set(selected_ids) & protected_ids)
    selected_ids = sorted(set(selected_ids) - protected_ids)
    if not selected_ids:
        raise ValueError("Protection policy excluded every active sensitive/neutral row")

    output = active.freeze_model_for_output_repair(model)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Full model must be frozen before sparse output repair")
    full_setting5_output = output.weight.detach().cpu().clone()
    ids_device = torch.tensor(selected_ids, dtype=torch.long, device=output.weight.device)
    setting5_rows = output.weight.index_select(0, ids_device).detach().cpu().clone()
    active_view_ids = {point.view_id for point in active_points}
    all_caches = active.build_prompt_instance_delta_caches(
        model, tokenizer, instances, selected_ids, device, args.eval_batch_size, llama_like
    )
    active_caches = [
        cache for point, cache in zip(points, all_caches) if point.view_id in active_view_ids
    ]

    repair_dir = run_dir(args) / "rank0_nullrestore16"
    repair_dir.mkdir(parents=True, exist_ok=True)
    rank2._write_jsonl(repair_dir / "active_points.jsonl", active_rows)
    utility.atomic_json_write(
        repair_dir / "active_rows.json",
        {
            "selected_row_ids": selected_ids,
            "decoded_rows": {str(row): tokenizer.decode([row]) for row in selected_ids},
            "excluded_special_row_ids": sorted(set(preliminary_special + active_special)),
            "excluded_protected_answer_row_ids": protected_excluded,
            "active_point_count": len(active_points),
            "selected_rows_include_sensitive_and_neutral_when_legal": True,
            "official_rwku_records_accessed": False,
        },
    )

    # Stage A: unrestricted rank-0 hard-tail repair.
    delta_rank0, stage_a_log, stage_a_summary = optimize_rank0_hard_tail(
        active_caches,
        n_rows=len(selected_ids),
        hidden_size=output.weight.shape[1],
        device=device,
    )
    rank2._write_jsonl(repair_dir / "stage_a_rank0_log.jsonl", stage_a_log)
    _materialize_delta(output.weight, selected_ids, setting5_rows, delta_rank0)
    rank0_actual = _actual_active_margins(
        model,
        tokenizer,
        instances,
        points,
        active_view_ids,
        selected_ids,
        device,
        args.eval_batch_size,
        llama_like,
    )
    rank0_actual_summary = _margin_summary(rank0_actual)
    if rank0_actual_summary["active_violation_count"] != 0:
        raise RuntimeError("Stage A FP32 repair did not survive exact BF16 materialization")
    stage_a_report = {
        **stage_a_summary,
        "FP32_optimizer_result": stage_a_summary,
        "BF16_materialized_result": rank0_actual_summary,
    }
    utility.atomic_json_write(run_dir(args) / "stage_a_rank0_summary.json", stage_a_report)

    # Stage B: every hidden affecting either side of every active margin.
    forget_hidden = torch.cat(
        [cache.target_new.hidden for cache in active_caches]
        + [cache.target_true.hidden for cache in active_caches],
        dim=0,
    ).float()
    forget_basis = active.orthonormal_row_basis(forget_hidden)
    if forget_basis.ndim != 2 or forget_basis.shape[1] != output.weight.shape[1]:
        raise AssertionError("Unexpected B_F orientation")
    delta_forget = project_delta_to_forget_span(delta_rank0, forget_basis)
    logit_difference = forget_hidden @ (delta_rank0 - delta_forget).transpose(0, 1)
    max_active_logit_difference = float(logit_difference.abs().max().detach().cpu())
    identity = torch.eye(forget_basis.shape[0], device=device)
    orthogonality_error = float(
        (forget_basis @ forget_basis.transpose(0, 1) - identity).abs().max().detach().cpu()
    ) if forget_basis.numel() else 0.0
    utility.atomic_json_write(
        run_dir(args) / "forget_subspace.json",
        {
            "forget_basis_rank": int(forget_basis.shape[0]),
            "H_F_shape": list(forget_hidden.shape),
            "basis_shape": list(forget_basis.shape),
            "basis_fp32_sha256": tensor_sha256(forget_basis),
            "max_row_orthogonality_error": orthogonality_error,
            "includes_sensitive_hidden": True,
            "includes_neutral_hidden": True,
            "official_rwku_records_accessed": False,
        },
    )
    _materialize_delta(output.weight, selected_ids, setting5_rows, delta_forget)
    projected_actual = _actual_active_margins(
        model,
        tokenizer,
        instances,
        points,
        active_view_ids,
        selected_ids,
        device,
        args.eval_batch_size,
        llama_like,
    )
    projected_summary = _margin_summary(projected_actual)
    utility.atomic_json_write(
        run_dir(args) / "forget_projection_summary.json",
        {
            "violations_before_projection": rank0_actual_summary["active_violation_count"],
            "violations_after_projection": projected_summary["active_violation_count"],
            "min_margin_before": rank0_actual_summary["minimum_margin"],
            "min_margin_after": projected_summary["minimum_margin"],
            "max_active_logit_difference": max_active_logit_difference,
            "actual_BF16_materialized_model_recomputed": True,
        },
    )

    # Legal retain/protection caches, all built before checkpoint freeze.
    # Their delta calculus is anchored at Setting5, so restore that exact
    # materialization before caching candidate probabilities/logits.
    _materialize_delta(output.weight, selected_ids, setting5_rows, torch.zeros_like(delta_forget))
    sampled_retain = rank2._mcf_sampled_records(
        retain_records[:RETAIN_CALIBRATION_NUM],
        retain_examples[:RETAIN_CALIBRATION_NUM],
    )
    sampled_retain.extend(
        rank2._mcf_sampled_records(
            [
                {"source": "matched_generated_corpus_protection", "index": index}
                for index in range(len(matched_train_examples))
            ],
            matched_train_examples,
        )
    )
    sampled_gate = rank2._mcf_sampled_records(gate_records, all_mcf_examples[args.mcf_retain_num :])
    sampled_gate.extend(
        rank2._mcf_sampled_records(
            [
                {"source": "matched_generated_corpus_gate", "index": index}
                for index in range(len(matched_gate_examples))
            ],
            matched_gate_examples,
        )
    )
    reference_weight, reference_bias = active.load_reference_output_layer(
        str(args.model_path), dtype
    )
    retain_caches = active.build_retain_kl_caches(
        model,
        reference_weight,
        reference_bias,
        tokenizer,
        sampled_retain,
        selected_ids,
        device,
    )
    gate_caches = active.build_retain_kl_caches(
        model,
        reference_weight,
        reference_bias,
        tokenizer,
        sampled_gate,
        selected_ids,
        device,
    )
    del reference_weight, reference_bias
    retain_hidden = torch.cat(
        [cache.hidden for cache in [*retain_caches, *gate_caches]], dim=0
    ).float()
    retain_null, restore_basis = build_restore_basis(
        retain_hidden, forget_basis, max_rank=RESTORE_RANK
    )
    overlap = (
        float((restore_basis @ forget_basis.transpose(0, 1)).abs().max().detach().cpu())
        if restore_basis.numel() and forget_basis.numel()
        else 0.0
    )
    if overlap > 1e-4:
        raise RuntimeError(f"B_R is not numerically in the forget nullspace: {overlap}")
    utility.atomic_json_write(
        run_dir(args) / "restore_basis.json",
        {
            "requested_rank": RESTORE_RANK,
            "actual_rank": int(restore_basis.shape[0]),
            "basis_fp32_sha256": tensor_sha256(restore_basis),
            "H_R_shape": list(retain_hidden.shape),
            "H_R_null_shape": list(retain_null.shape),
            "max_forget_restore_basis_overlap": overlap,
            "official_rwku_records_accessed": False,
        },
    )

    zero = torch.zeros_like(delta_forget)
    setting5_absolute_kl_tensor = active.retain_kl_from_caches(retain_caches, zero)
    setting5_absolute_kl = float(setting5_absolute_kl_tensor.detach().cpu())
    setting5_protection = rank2._protection_snapshot(
        model, tokenizer, gate_examples, selected_ids, batch_size=args.eval_batch_size
    )
    proxy_text = load_wikidata_text(args.wikidata_dir)
    if not proxy_text:
        raise FileNotFoundError("Target-independent Wikidata proxy-PPL text is required")
    setting5_proxy_ppl = evaluate_perplexity(model, tokenizer, proxy_text)

    # Stage C: independent row coefficients in B_R, with the hard forget gate
    # checked after every optimizer step.  Candidate snapshots are immutable.
    restore_module = active.SelectedRowDelta(
        len(selected_ids),
        output.weight.shape[1],
        direction_basis=restore_basis,
        device=device,
    )
    optimizer = torch.optim.AdamW(restore_module.parameters(), lr=RESTORE_LR)
    restoration_log: List[Dict[str, Any]] = []
    snapshots: List[Tuple[int, torch.Tensor]] = [(0, torch.zeros_like(delta_forget))]
    projected_fp32_margins = active.margins_from_delta_caches(
        active_caches, delta_forget
    ).detach()
    preservation_floor = torch.maximum(
        torch.full_like(projected_fp32_margins, ACTIVE_MARGIN),
        projected_fp32_margins - 1e-5,
    )
    for step in range(RESTORE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        restore = restore_module.effective_delta()
        final_delta = delta_forget + restore
        absolute_kl = active.retain_kl_from_caches(retain_caches, final_delta)
        selected_logit_reconstruction = (
            (retain_hidden @ final_delta.transpose(0, 1)).square().mean()
        )
        l2 = restore.square().sum()
        loss = absolute_kl + selected_logit_reconstruction + RESTORE_L2_LAMBDA * l2
        loss.backward()
        previous = restore_module.coefficients.detach().clone()
        optimizer.step()
        with torch.no_grad():
            current_restore = restore_module.effective_delta()
            current_delta = delta_forget + current_restore
            current_margins = active.margins_from_delta_caches(active_caches, current_delta)
            margin_report = _margin_summary(current_margins)
            preservation_violations = int(
                (current_margins < preservation_floor).sum().item()
            )
            if (
                margin_report["active_violation_count"] != 0
                or margin_report["minimum_margin"] < ACTIVE_MARGIN
                or preservation_violations != 0
            ):
                restore_module.coefficients.copy_(previous)
                for optimizer_state in optimizer.state.values():
                    for value in optimizer_state.values():
                        if isinstance(value, torch.Tensor):
                            value.zero_()
                current_restore = restore_module.effective_delta()
                current_delta = delta_forget + current_restore
                margin_report = _margin_summary(
                    active.margins_from_delta_caches(active_caches, current_delta)
                )
                preservation_violations = 0
                rolled_back = True
            else:
                rolled_back = False
            candidate_kl = active.retain_kl_from_caches(retain_caches, current_delta)
        restoration_log.append(
            {
                "step": step + 1,
                "loss": float(loss.detach().cpu()),
                "candidate_absolute_retain_kl": float(candidate_kl.detach().cpu()),
                "retain_kl_increase": float(
                    incremental_retain_kl(candidate_kl, setting5_absolute_kl_tensor).detach().cpu()
                ),
                "protected_selected_row_reconstruction": float(
                    selected_logit_reconstruction.detach().cpu()
                ),
                "restoration_l2": float(l2.detach().cpu()),
                "active_violation_count": margin_report["active_violation_count"],
                "minimum_active_margin": margin_report["minimum_margin"],
                "stage_b_margin_preservation_violations": preservation_violations,
                "step_rolled_back_for_forgetting": rolled_back,
            }
        )
        if (step + 1) % RESTORE_SNAPSHOT_INTERVAL == 0 or step + 1 == RESTORE_STEPS:
            snapshots.append((step + 1, restore_module.effective_delta().detach().cpu().clone()))
    rank2._write_jsonl(run_dir(args) / "restoration_log.jsonl", restoration_log)

    # Include Stage A and projected Stage B as explicit perfect-forgetting
    # fallbacks, then audit every Stage-C snapshot using materialized BF16.
    candidate_specs: List[Tuple[str, int, torch.Tensor]] = [
        ("stage_a_rank0", -1, delta_rank0.detach().cpu()),
        ("stage_b_forget_projection", 0, delta_forget.detach().cpu()),
    ]
    candidate_specs.extend(
        ("stage_c_nullspace_restoration", step, delta_forget.detach().cpu() + restore)
        for step, restore in snapshots[1:]
    )
    sweep: List[Dict[str, Any]] = []
    candidate_tensors: Dict[str, torch.Tensor] = {}
    for name, step, candidate_cpu in candidate_specs:
        key = f"{name}:step={step}"
        candidate = candidate_cpu.to(device=device, dtype=torch.float32)
        _materialize_delta(output.weight, selected_ids, setting5_rows, candidate)
        actual_margins = _actual_active_margins(
            model,
            tokenizer,
            instances,
            points,
            active_view_ids,
            selected_ids,
            device,
            args.eval_batch_size,
            llama_like,
        )
        # Audit the exact BF16 materialization rather than the pre-rounding
        # FP32 candidate tensor.
        materialized_rows = output.weight.index_select(0, ids_device).float()
        materialized_delta = materialized_rows - setting5_rows.to(
            device=device, dtype=torch.float32
        )
        absolute_kl = float(
            active.retain_kl_from_caches(retain_caches, materialized_delta)
            .detach()
            .cpu()
        )
        protection_snapshot = rank2._protection_snapshot(
            model, tokenizer, gate_examples, selected_ids, batch_size=args.eval_batch_size
        )
        protection_metrics = rank2._protection_metrics(setting5_protection, protection_snapshot)
        proxy_ppl = evaluate_perplexity(model, tokenizer, proxy_text)
        equal = nonselected_rows_equal_setting5(
            output.weight, full_setting5_output, selected_ids
        )
        accepted, report = _strict_candidate_gate(
            margins=actual_margins,
            setting5_absolute_kl=setting5_absolute_kl,
            candidate_absolute_kl=absolute_kl,
            protection=protection_metrics,
            proxy_ppl=proxy_ppl,
            setting5_proxy_ppl=setting5_proxy_ppl,
            nonselected_equal=equal,
        )
        report.update(
            {
                "candidate_id": key,
                "stage": name,
                "restoration_step": step,
                "strict_gates_pass": accepted,
                "candidate_accepted": (
                    accepted and name == "stage_c_nullspace_restoration"
                ),
                "delta_norm": float(candidate.norm().detach().cpu()),
                "restoration_norm": float((candidate - delta_forget).norm().detach().cpu()),
            }
        )
        sweep.append(report)
        candidate_tensors[key] = candidate_cpu.clone()
    # Scientific success belongs only to Stage C.  Stage A/B remain safe
    # perfect-forgetting fallbacks even if one happens to pass all utility
    # checks before restoration.
    passing = [
        row
        for row in sweep
        if row["candidate_accepted"]
    ]
    if passing:
        selected_report = min(
            passing,
            key=lambda row: (float(row["delta_norm"]), int(row["restoration_step"])),
        )
        candidate_accepted = True
        final_failed_gates = list(selected_report["failed_gates"])
    else:
        perfect = [
            row
            for row in sweep
            if row["active_violation_count"] == 0
            and row["minimum_margin"] >= ACTIVE_MARGIN
        ]
        if not perfect:
            raise RuntimeError("No Stage A/B/C candidate preserves perfect generated forgetting")
        selected_report = min(perfect, key=_utility_penalty)
        candidate_accepted = False
        final_failed_gates = [
            *selected_report["failed_gates"],
            "no_stage_c_candidate_satisfied_every_strict_gate",
        ]
    selected_delta_cpu = candidate_tensors[selected_report["candidate_id"]]
    selected_delta = selected_delta_cpu.to(device=device, dtype=torch.float32)
    _materialize_delta(output.weight, selected_ids, setting5_rows, selected_delta)
    if not nonselected_rows_equal_setting5(output.weight, full_setting5_output, selected_ids):
        raise AssertionError("A nonselected LM-head row changed")
    utility.atomic_json_write(
        run_dir(args) / "restoration_candidate_sweep.json",
        {
            "candidates": sweep,
            "selected_candidate_id": selected_report["candidate_id"],
            "candidate_accepted": candidate_accepted,
            "final_failed_gates": final_failed_gates,
            "selection_rule": (
                "smallest delta norm among strict passing candidates; otherwise "
                "best-utility candidate among perfect-forgetting candidates"
            ),
        },
    )

    row_report = []
    selected_restore = selected_delta - delta_forget
    for index, token_id in enumerate(selected_ids):
        restore_norm = float(selected_restore[index].norm().detach().cpu())
        row_report.append(
            {
                "token_id": token_id,
                "decoded_token": tokenizer.decode([token_id]),
                "rank0_delta_norm": float(delta_rank0[index].norm().detach().cpu()),
                "forget_projected_delta_norm": float(delta_forget[index].norm().detach().cpu()),
                "restoration_norm": restore_norm,
                "final_delta_norm": float(selected_delta[index].norm().detach().cpu()),
                "selected_restoration_scale": 0.0 if restore_norm == 0.0 else 1.0,
                "restoration_parameterization": "row-specific coefficients in span(B_R)",
                "selected_candidate_stage": selected_report["stage"],
                "restoration_in_span_B_R": (
                    selected_report["stage"] == "stage_c_nullspace_restoration"
                ),
            }
        )
    utility.atomic_json_write(
        run_dir(args) / "selected_row_restoration_report.json",
        {"rows": row_report, "selected_candidate_id": selected_report["candidate_id"]},
    )

    selected_dir = run_dir(args) / "selected_checkpoint"
    legacy.save_checkpoint(model, tokenizer, selected_dir)
    training_report = {
        "method": METHOD,
        "protocol_status": PROTOCOL_STATUS,
        "stages": {
            "Setting5": {
                "steps": SETTING5_STEPS,
                "summary": asdict(setting_summary),
                "training_seconds": setting5_training_seconds,
            },
            "Rank0 perfect-forgetting repair": stage_a_report,
            "Forget-subspace cleanup": projected_summary,
            "Rank16 forget-nullspace utility restoration": {
                "requested_rank": RESTORE_RANK,
                "actual_rank": int(restore_basis.shape[0]),
                "steps": RESTORE_STEPS,
                "selected_candidate": selected_report,
            },
        },
        "candidate_accepted": candidate_accepted,
        "selected_candidate_id": selected_report["candidate_id"],
        "failed_gates": final_failed_gates,
        "official_rwku_records_accessed": False,
        "official_evaluation_used_for_selection": False,
    }
    training_report_path = run_dir(args) / "training_report.json"
    utility.atomic_json_write(training_report_path, training_report)

    receipt_path = run_dir(args) / "checkpoint_receipt.json"
    receipt = create_checkpoint_receipt(
        destination=receipt_path,
        experiment_id=args.experiment_id,
        protocol_label=TARGET_ONLY_PROTOCOL_LABEL,
        protocol_status=PROTOCOL_STATUS,
        target_entity=state["target"]["subject"],
        target_entity_id=state["target"]["entity_id"],
        base_model_identity=legacy.local_model_identity(args.model_path),
        base_model_revision=args.model_revision,
        tokenizer_identity={
            "name_or_path": tokenizer.name_or_path,
            "class": tokenizer.__class__.__name__,
            "vocab_size": len(tokenizer),
            "eos_token_id": tokenizer.eos_token_id,
        },
        checkpoint_paths=[setting5_dir, selected_dir],
        training_bundle_path=args.generated_training_bundle,
        optimization_protection_path=matched_train_path,
        mcf_retain_optimization_paths=[mcf_train_manifest],
        mcf_repair_gate_paths=[mcf_gate_manifest],
        matched_protection_train_path=matched_train_path,
        matched_protection_gate_path=matched_gate_path,
        method_configuration=configuration(args),
        implementation_files=[
            SCRIPT_PATH,
            SEMANTIC_ROOT / "scripts" / "gagd_active_case_repair.py",
            SEMANTIC_ROOT / "scripts" / "rwku_generated_s5e_rank2_active_repair.py",
            SEMANTIC_ROOT / "scripts" / "rwku_experiment.py",
            SEMANTIC_ROOT / "scripts" / "rwku_checkpoint_receipt.py",
            SEMANTIC_ROOT / "scripts" / "rwku_artifact_access.py",
            SEMANTIC_ROOT / "scripts" / "rwku_fact_sampler.py",
            SEMANTIC_ROOT / "scripts" / "rwku_rowwise_active_repair.py",
            SEMANTIC_ROOT / "scripts" / "build_rwku_matched_protection.py",
        ],
        sampler_provenance=exposures,
        generator_receipt_path=args.generator_receipt,
        official_locked_eval_path=run_dir(args) / "official_locked_eval.json",
        confirmatory=False,
        additional_artifact_paths={
            "configuration_manifest": run_dir(args) / "configuration_manifest.json",
            "fact_exposure_report": run_dir(args) / "fact_exposure_report.json",
            "active_points": repair_dir / "active_points.jsonl",
            "active_rows": repair_dir / "active_rows.json",
            "stage_a_rank0_log": repair_dir / "stage_a_rank0_log.jsonl",
            "stage_a_rank0_summary": run_dir(args) / "stage_a_rank0_summary.json",
            "forget_subspace": run_dir(args) / "forget_subspace.json",
            "forget_projection_summary": run_dir(args) / "forget_projection_summary.json",
            "restore_basis": run_dir(args) / "restore_basis.json",
            "restoration_log": run_dir(args) / "restoration_log.jsonl",
            "restoration_candidate_sweep": run_dir(args) / "restoration_candidate_sweep.json",
            "selected_row_restoration_report": run_dir(args) / "selected_row_restoration_report.json",
            "training_report": training_report_path,
            "matched_protection_coverage": run_dir(args) / "protection" / "matched_protection_coverage.json",
            "protection_diagnostics": run_dir(args) / "protection" / "protection_diagnostics.json",
            **{
                f"frozen_corpus_{name.replace('.', '_')}": Path(identity["path"])
                for name, identity in state["corpus_identities"].items()
                if name not in {"generated_training_bundle.json", "generator_receipt.json"}
            },
        },
    )
    write_state(
        args,
        "CHECKPOINT_FROZEN",
        checkpoint_receipt=str(receipt_path.resolve()),
        checkpoint_receipt_sha256=receipt["receipt_sha256"],
        setting5_checkpoint=str(setting5_dir.resolve()),
        selected_checkpoint=str(selected_dir.resolve()),
        candidate_accepted=candidate_accepted,
        failed_gates=final_failed_gates,
        official_rwku_records_accessed=False,
        official_evaluation_opened=False,
    )
    legacy.release_model(model)


def verify_stage(args: argparse.Namespace) -> None:
    state = read_state(args)
    if state.get("state") != "CHECKPOINT_FROZEN":
        raise ValueError("Receipt verification requires CHECKPOINT_FROZEN")
    receipt = load_receipt(run_dir(args) / "checkpoint_receipt.json")
    if receipt.get("experiment_id") != args.experiment_id:
        raise ValueError("Receipt belongs to another experiment")
    if receipt.get("protocol_status") != PROTOCOL_STATUS:
        raise ValueError("Receipt belongs to another RWKU method")
    if receipt.get("official_evaluation_opened") is not False:
        raise ValueError("Official evaluation was opened unexpectedly")
    if sha256_json(configuration(args)) != receipt.get("method_configuration_sha256"):
        raise ValueError("Invocation configuration differs from frozen receipt")
    verify_frozen_identities(receipt)
    print(
        json.dumps(
            {
                "status": "checkpoint_receipt_verified",
                "experiment_id": args.experiment_id,
                "state": receipt["state"],
                "candidate_accepted": state.get("candidate_accepted"),
                "official_evaluation_opened": False,
            },
            indent=2,
        )
    )


def preflight(args: argparse.Namespace) -> None:
    if args.experiment_id != EXPERIMENT_ID or args.seed != 0:
        raise ValueError("Preflight is pinned to the new seed-0 development experiment")
    for path in (
        args.generated_training_bundle,
        args.generator_receipt,
        args.mcf_path,
        *args.protection_source,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        rank2.reject_official_path(Path(path), label="preflight input")
    _assert_bundle_identity(args.generated_training_bundle)
    manifest = rank2.verify_corpus_manifest(args)
    if run_dir(args).exists():
        raise ValueError(f"Experiment directory already exists: {run_dir(args)}")
    print(
        json.dumps(
            {
                "status": "preflight_ok",
                "configuration": configuration(args),
                "corpus_manifest": manifest,
                "official_rwku_records_accessed": False,
            },
            indent=2,
        )
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--stage", choices=("preflight", "prepare", "train", "verify"), required=True)
    value.add_argument("--experiment-id", required=True)
    value.add_argument("--seed", type=int, default=0)
    value.add_argument("--model-path", type=Path, required=True)
    value.add_argument("--model-revision", required=True)
    value.add_argument("--generated-training-bundle", type=Path, required=True)
    value.add_argument("--generator-receipt", type=Path, required=True)
    value.add_argument("--output-root", type=Path, required=True)
    value.add_argument("--mcf-path", type=Path, default=legacy.DEFAULT_MCF_PATH)
    value.add_argument("--protection-source", type=Path, action="append", default=[])
    value.add_argument("--protection-vocabulary", type=Path, default=None)
    value.add_argument("--mcf-retain-num", type=int, default=1000)
    value.add_argument("--mcf-gate-num", type=int, default=200)
    value.add_argument("--wikidata-dir", type=Path, default=legacy.DEFAULT_WIKIDATA_DIR)
    value.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    value.add_argument("--eval-batch-size", type=int, default=4)
    value.add_argument("--gradient-checkpointing", action="store_true")
    value.add_argument("--no-download", action="store_true")
    return value


def main() -> None:
    args = parser().parse_args()
    if not args.protection_source:
        args.protection_source = [args.mcf_path]
    if args.experiment_id != EXPERIMENT_ID:
        raise ValueError("This entrypoint is pinned to its new isolated experiment ID")
    if args.stage == "preflight":
        preflight(args)
    elif args.stage == "prepare":
        prepare_stage(args)
    elif args.stage == "train":
        train_stage(args)
    else:
        verify_stage(args)


if __name__ == "__main__":
    main()
