#!/usr/bin/env python3
"""MQuAKE Setting5e + rank-0 forgetting + rank-64 nullspace restoration.

This isolated method keeps the pinned 600-step Setting 5e diagnostic, then
restores the exact Base input/output origin.  It solves unrestricted,
Base-relative selected-row forgetting, projects that delta into the complete
configured numerical forget-hidden span, and restores Base utility with one
fixed 64-dimensional basis in that numerical span's complement.  Because raw
forget states can retain discarded low-singular-value components, forgetting
feasibility is enforced explicitly through restoration.  Only requested-
rewrite cloze fields are visible before the durable selection decision;
AtomicGen and multi-hop fields remain post-selection only.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import gagd_compare as gagd
import mquake_gagd_setting5e_active_repair as baseline
import mquake_gagd_setting5e_multiroot_active_repair as vectorized
import mquake_setting5e_detied_baseanchored_minrank as basearch
import mquake_zero_unlearn_official_eval as mquake
from mcf_zero_unlearn_official_eval import load_official_ppl_text


METHOD = "mquake_setting5e_rank0_nullrestore64"
METHOD_LABEL = (
    "Setting 5e @600 + rank-0 Base-anchored perfect forgetting + "
    "forget-span projection + rank-64 forget-nullspace Base-utility restoration"
)
DATASET_REVISION = "fb43dadc2d8cd19d08ce81c63d957b59deb3f3cd"
SETTING5_MODE = gagd.POST_TRAINING_RESTORE_MODE
SETTING5_STEPS = 600
RANK0_REPAIR_RANK = 0
RANK0_LR = 5e-3
RANK0_STEPS_PER_PHASE = 1000
RANK0_DELTA_L2 = 1e-4
ACTIVE_MARGIN = 0.25
SELECTION_MARGIN = 0.10
RESTORE_RANK = 64
RESTORE_STEPS = 800
RESTORE_LR = 5e-4
RESTORE_L2 = 1e-4
SOLVER_RHO = 1.0
SOLVER_TOLERANCE = 1e-6
SOLVER_STALL_PATIENCE = 100
SOLVER_MIN_IMPROVEMENT = 1e-7
MAX_CONSTRAINT_ROUNDS = 64
HELD_OUT_FIELDS = basearch.HELD_OUT_FIELDS

TokenState = basearch.TokenState
PairConstraint = basearch.PairConstraint
ExactPPLCache = basearch.ExactPPLCache
PreparedPPLTensors = basearch.PreparedPPLTensors


@dataclass(frozen=True)
class GeometryResult:
    basis: torch.Tensor
    report: Dict[str, Any]


@dataclass(frozen=True)
class Rank0PhaseResult:
    delta: torch.Tensor
    report: Dict[str, Any]
    log: List[Dict[str, Any]]


@dataclass(frozen=True)
class ProtectedRestoreTensors:
    protected: vectorized.ProtectedPairTensors
    fixed_delta_logits: torch.Tensor
    runner_base_logits: torch.Tensor
    runner_selected_row_index: torch.Tensor


@dataclass(frozen=True)
class StageCForgetTensors:
    """Fixed Stage-B margins plus rank-64 coordinates for forget pairs.

    Selected-row indices may be -1 because Stage C keeps the Stage-A editable
    row set fixed.  Such sensitive/competitor logits remain at their fixed
    Stage-B values while selected rows receive coefficient-space effects.
    """

    hidden_rank: torch.Tensor
    fixed_stage_b_margin: torch.Tensor
    sensitive_selected_row_index: torch.Tensor
    competitor_selected_row_index: torch.Tensor


@dataclass(frozen=True)
class RestorePhaseResult:
    coefficients: torch.Tensor
    report: Dict[str, Any]
    log: List[Dict[str, Any]]


@dataclass(frozen=True)
class DynamicRestoreResult:
    coefficients: torch.Tensor
    constraints: List[PairConstraint]
    phase_reports: List[Dict[str, Any]]
    log: List[Dict[str, Any]]
    final_states: List[TokenState]


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def tensor_sha256(value: torch.Tensor) -> str:
    return basearch.tensor_sha256(value)


def transformer_sha256(model: nn.Module) -> str:
    """Hash every parameter except the input/output vocabulary matrix."""

    input_ptr = model.get_input_embeddings().weight.data_ptr()
    output_ptr = model.get_output_embeddings().weight.data_ptr()
    digest = hashlib.sha256()
    included = 0
    for name, parameter in model.named_parameters():
        if parameter.data_ptr() in {input_ptr, output_ptr}:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(parameter).encode("ascii"))
        included += 1
    digest.update(f"count={included}".encode("ascii"))
    return digest.hexdigest()


def canonicalize_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    return basearch.canonicalize_basis_signs(basis)


def fp32_geometry_tolerance(hidden_size: int) -> float:
    """Tolerance for FP32 QR geometry without admitting material leakage.

    Backward-error in orthogonal factorizations grows approximately with the
    square root of the ambient dimension.  Three FP32 epsilons per square-root
    dimension is about 2e-5 at hidden_size=3072, while the small absolute floor
    avoids unrealistically tight tests for tiny synthetic matrices.  This is
    deliberately far below the observed 1.2419e-4 nullspace leakage.
    """

    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    epsilon = torch.finfo(torch.float32).eps
    return max(2e-6, 3.0 * epsilon * math.sqrt(float(hidden_size)))


def _row_geometry(
    basis: torch.Tensor,
    *,
    reference: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    if basis.ndim != 2:
        raise ValueError("basis must be a matrix")
    values = basis.float()
    gram = values @ values.T
    identity = torch.eye(values.shape[0], device=values.device, dtype=values.dtype)
    orthogonality = gram - identity
    report = {
        "orthogonality_max_error": (
            float(orthogonality.abs().max().item())
            if orthogonality.numel()
            else 0.0
        ),
        "orthogonality_fro_error": (
            float(orthogonality.norm().item()) if orthogonality.numel() else 0.0
        ),
    }
    if reference is not None:
        if reference.ndim != 2 or reference.shape[1] != values.shape[1]:
            raise ValueError("reference basis has incompatible hidden dimension")
        overlap = values @ reference.float().T
        report.update(
            {
                "reference_overlap_max_error": (
                    float(overlap.abs().max().item()) if overlap.numel() else 0.0
                ),
                "reference_overlap_fro_error": (
                    float(overlap.norm().item()) if overlap.numel() else 0.0
                ),
            }
        )
    return report


def explicit_forget_complement(
    forget_basis: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Return deterministic coordinates for the complete complement of B_F.

    A complete QR of B_F.T provides a numerically stable complement directly;
    unlike H - H B_F.T B_F, it does not obtain small residual coordinates by
    subtracting two nearly equal high-rank projections.
    """

    if forget_basis.ndim != 2 or forget_basis.shape[1] == 0:
        raise ValueError("forget basis must be a non-empty matrix")
    values = forget_basis.float()
    forget_rank, hidden_size = values.shape
    if forget_rank > hidden_size:
        raise ValueError("forget basis rank exceeds hidden dimension")
    tolerance = fp32_geometry_tolerance(hidden_size)
    forget_geometry = _row_geometry(values)
    if forget_geometry["orthogonality_max_error"] > tolerance:
        raise RuntimeError(
            "forget basis is not orthonormal before complete QR: "
            f"{forget_geometry['orthogonality_max_error']}"
        )
    q_full, _ = torch.linalg.qr(values.T, mode="complete")
    complement = canonicalize_basis_signs(
        q_full[:, forget_rank:].T.contiguous()
    )
    geometry = _row_geometry(complement, reference=values)
    if geometry["orthogonality_max_error"] > tolerance:
        raise RuntimeError(
            "explicit forget complement is not orthonormal: "
            f"{geometry['orthogonality_max_error']}"
        )
    if geometry["reference_overlap_max_error"] > tolerance:
        raise RuntimeError(
            "explicit complement leaves forget nullspace: "
            f"{geometry['reference_overlap_max_error']}"
        )
    report = {
        "explicit_nullspace_dimension": int(complement.shape[0]),
        "nullspace_basis_sha256": tensor_sha256(complement),
        "nullspace_orthogonality_max_error": geometry[
            "orthogonality_max_error"
        ],
        "nullspace_orthogonality_fro_error": geometry[
            "orthogonality_fro_error"
        ],
        "nullspace_forget_overlap_max_error": geometry[
            "reference_overlap_max_error"
        ],
        "nullspace_forget_overlap_fro_error": geometry[
            "reference_overlap_fro_error"
        ],
        "geometry_tolerance": tolerance,
        "construction": (
            "complete FP32 QR of forget_basis.T -> deterministic sign "
            "canonicalization"
        ),
    }
    return complement, report


def validate_restore_basis_geometry(
    restore_basis: torch.Tensor,
    forget_basis: torch.Tensor,
) -> Dict[str, float]:
    """Fail closed unless B_R is orthonormal and in the computed complement."""

    if restore_basis.ndim != 2 or forget_basis.ndim != 2:
        raise ValueError("restore and forget bases must be matrices")
    if restore_basis.shape != (RESTORE_RANK, forget_basis.shape[1]):
        raise ValueError(
            "rank-64 restoration basis must have shape "
            f"({RESTORE_RANK}, {forget_basis.shape[1]})"
        )
    tolerance = fp32_geometry_tolerance(forget_basis.shape[1])
    geometry = _row_geometry(restore_basis, reference=forget_basis)
    maximum = geometry["reference_overlap_max_error"]
    if geometry["orthogonality_max_error"] > tolerance:
        raise RuntimeError(
            "rank-64 restoration basis is not orthonormal: "
            f"{geometry['orthogonality_max_error']}"
        )
    if maximum > tolerance:
        raise RuntimeError(
            f"rank-64 restoration basis leaves forget nullspace: {maximum}"
        )
    return {
        "B_R_orthogonality_max_error": geometry["orthogonality_max_error"],
        "B_R_orthogonality_fro_error": geometry["orthogonality_fro_error"],
        "maximum_absolute_B_R_B_F_overlap": maximum,
        "frobenius_B_R_B_F_overlap": geometry[
            "reference_overlap_fro_error"
        ],
        "geometry_tolerance": tolerance,
    }


def robust_svd_qr_row_basis(rows: torch.Tensor) -> GeometryResult:
    """FP32 SVD ordering followed by one reduced QR and canonical signs."""

    if rows.ndim != 2 or rows.shape[0] == 0 or rows.shape[1] == 0:
        raise ValueError("basis construction requires a non-empty hidden matrix")
    values = rows.float()
    _, singular_values, right = torch.linalg.svd(values, full_matrices=False)
    tolerance = (
        max(values.shape)
        * torch.finfo(values.dtype).eps
        * singular_values.max().clamp_min(1.0)
    )
    numerical_rank = int((singular_values > tolerance).sum().item())
    raw = right[:numerical_rank].contiguous()
    raw_gram = raw @ raw.T
    raw_identity = torch.eye(numerical_rank, device=values.device)
    raw_error = (
        float((raw_gram - raw_identity).abs().max().item())
        if numerical_rank
        else 0.0
    )
    if numerical_rank:
        q, _ = torch.linalg.qr(raw.T, mode="reduced")
        basis = canonicalize_basis_signs(q.T.contiguous())
        gram = basis @ basis.T
        identity = torch.eye(numerical_rank, device=values.device)
        difference = gram - identity
        maximum = float(difference.abs().max().item())
        frobenius = float(difference.norm().item())
    else:
        basis = values.new_empty((0, values.shape[1]))
        maximum = 0.0
        frobenius = 0.0
    if maximum > 2e-5:
        raise RuntimeError(f"SVD+QR basis is not orthonormal: max_error={maximum}")
    return GeometryResult(
        basis=basis,
        report={
            "state_count": int(values.shape[0]),
            "hidden_size": int(values.shape[1]),
            "numerical_rank": numerical_rank,
            "exact_nullspace_dimension": int(values.shape[1] - numerical_rank),
            "numerical_rank_tolerance": float(tolerance.item()),
            "raw_svd_orthogonality_max_error": raw_error,
            "final_orthogonality_max_error": maximum,
            "final_orthogonality_fro_error": frobenius,
            "basis_sha256": tensor_sha256(basis),
            "construction": (
                "FP32 SVD ordering -> numerical rank -> one reduced QR -> "
                "deterministic sign canonicalization"
            ),
        },
    )


def write_geometry_infeasible(
    output_dir: Path,
    *,
    hidden_size: int,
    forget_state_count: int,
    forget_numerical_rank: int,
    exact_nullspace_dimension: int,
    reason: str,
    utility_null_numerical_rank: Optional[int] = None,
) -> Dict[str, Any]:
    payload = {
        "hidden_size": int(hidden_size),
        "forget_state_count": int(forget_state_count),
        "forget_numerical_rank": int(forget_numerical_rank),
        "exact_nullspace_dimension": int(exact_nullspace_dimension),
        "requested_restore_rank": RESTORE_RANK,
        "utility_null_numerical_rank": utility_null_numerical_rank,
        "reason": reason,
        "rank_fallback_attempted": False,
    }
    gagd.write_json(output_dir / "geometry_infeasible.json", payload)
    geometry = output_dir / "geometry"
    geometry.mkdir(parents=True, exist_ok=True)
    gagd.write_json(geometry / "geometry_infeasible.json", payload)
    return payload


def require_forget_nullspace(
    report: Mapping[str, Any], output_dir: Path
) -> None:
    available = int(report["exact_nullspace_dimension"])
    if available < RESTORE_RANK:
        write_geometry_infeasible(
            output_dir,
            hidden_size=int(report["hidden_size"]),
            forget_state_count=int(report["state_count"]),
            forget_numerical_rank=int(report["numerical_rank"]),
            exact_nullspace_dimension=available,
            reason="exact RWKU-style hidden-space rank-64 restoration unavailable",
        )
        raise RuntimeError("exact forget nullspace dimension is below 64")


def build_restore_basis64(
    utility_hidden: torch.Tensor,
    forget_basis: torch.Tensor,
    *,
    output_dir: Optional[Path] = None,
    forget_report: Optional[Mapping[str, Any]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Build exactly 64 utility directions in explicit nullspace coordinates."""

    if utility_hidden.ndim != 2 or forget_basis.ndim != 2:
        raise ValueError("utility hidden bank and forget basis must be matrices")
    if utility_hidden.shape[1] != forget_basis.shape[1]:
        raise ValueError("utility and forget hidden dimensions differ")
    nullspace_basis, nullspace_report = explicit_forget_complement(forget_basis)
    utility_coordinates = utility_hidden.float() @ nullspace_basis.T
    geometry = robust_svd_qr_row_basis(utility_coordinates)
    available = int(geometry.report["numerical_rank"])
    if available < RESTORE_RANK:
        if output_dir is not None:
            source = forget_report or {
                "hidden_size": utility_hidden.shape[1],
                "state_count": 0,
                "numerical_rank": 0,
                "exact_nullspace_dimension": utility_hidden.shape[1],
            }
            write_geometry_infeasible(
                output_dir,
                hidden_size=int(source["hidden_size"]),
                forget_state_count=int(source["state_count"]),
                forget_numerical_rank=int(source["numerical_rank"]),
                exact_nullspace_dimension=int(source["exact_nullspace_dimension"]),
                utility_null_numerical_rank=available,
                reason="utility-null numerical rank is below requested fixed rank 64",
            )
        raise RuntimeError("utility-null numerical rank is below 64")
    coordinate_basis64 = geometry.basis[:RESTORE_RANK].contiguous()
    basis = coordinate_basis64 @ nullspace_basis
    q, _ = torch.linalg.qr(basis.T, mode="reduced")
    basis = canonicalize_basis_signs(q.T.contiguous())
    restore_geometry = validate_restore_basis_geometry(basis, forget_basis)
    utility_null = utility_coordinates @ nullspace_basis
    report = {
        "state_count": int(utility_hidden.shape[0]),
        "hidden_size": int(utility_hidden.shape[1]),
        **nullspace_report,
        "utility_coordinate_shape": list(utility_coordinates.shape),
        "utility_coordinate_numerical_rank": available,
        "utility_coordinate_numerical_rank_tolerance": geometry.report[
            "numerical_rank_tolerance"
        ],
        "utility_coordinate_basis_sha256": geometry.report["basis_sha256"],
        "restore_rank": RESTORE_RANK,
        "utility_null_numerical_rank": available,
        **restore_geometry,
        "B_R_sha256": tensor_sha256(basis),
        "rank_fallback_or_sweep": False,
        "construction": (
            "explicit complete-QR forget complement -> utility SVD in "
            "nullspace coordinates -> rank-64 map back -> reduced QR"
        ),
    }
    return utility_null, basis, report


def project_delta_to_forget_span(
    delta: torch.Tensor, forget_basis: torch.Tensor
) -> torch.Tensor:
    if delta.ndim != 2 or forget_basis.ndim != 2:
        raise ValueError("delta and forget basis must be matrices")
    if delta.shape[1] != forget_basis.shape[1]:
        raise ValueError("delta and forget basis hidden dimensions differ")
    return (delta.float() @ forget_basis.T) @ forget_basis


def rank64_restoration_delta(
    coefficients: torch.Tensor, restore_basis: torch.Tensor
) -> torch.Tensor:
    if coefficients.ndim != 2 or restore_basis.ndim != 2:
        raise ValueError("coefficients and restore basis must be matrices")
    if coefficients.shape[1] != RESTORE_RANK:
        raise ValueError("restoration coefficients must have exactly 64 columns")
    if restore_basis.shape[0] != RESTORE_RANK:
        raise ValueError("restoration basis must have exactly 64 rows")
    return coefficients.float() @ restore_basis.float()


def forgetting_feasibility_report(
    margins: torch.Tensor,
    *,
    required_margin: float = ACTIVE_MARGIN,
) -> Dict[str, Any]:
    """Report the actual configured forget condition, independent of logit drift."""

    if margins.ndim != 1:
        raise ValueError("forget margins must be a vector")
    minimum = float(margins.min().item()) if margins.numel() else math.inf
    violations = int((margins < float(required_margin)).sum().item())
    return {
        "active_violation_count": violations,
        "minimum_active_margin": minimum,
        "required_active_margin": float(required_margin),
        "passed": violations == 0 and minimum >= float(required_margin),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = baseline.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir="outputs/mquake_s5e600_rank0_nullrestore64/seed0",
        seed=0,
        forget_num=1000,
        retain_num=1000,
        steps=SETTING5_STEPS,
        batch_size=1,
        retain_batch_size=4,
        emb_lm_lr=1e-4,
        forget_weight=2.0,
        retain_weight=1.0,
        forget_margin=1.0,
        emb_lm_optimizer="adamw",
        sampling_strategy="epoch",
        active_logit_margin=ACTIVE_MARGIN,
        selection_logit_margin=SELECTION_MARGIN,
        retain_calibration_num=1000,
        retain_calibration_seed=1729,
        target_eff_max=0.0,
        utility_drop_tolerance=0.10,
        max_ppl_ratio=1.02,
        dtype="bf16",
        device_map="single",
        strict_utility_gates=True,
    )
    parser.add_argument(
        "--forget-sampling",
        choices=("instance_balanced",),
        default="instance_balanced",
    )
    parser.add_argument("--rank0-steps-per-phase", type=int, default=RANK0_STEPS_PER_PHASE)
    parser.add_argument("--rank0-lr", type=float, default=RANK0_LR)
    parser.add_argument("--rank0-delta-l2", type=float, default=RANK0_DELTA_L2)
    parser.add_argument("--restore-rank", type=int, default=RESTORE_RANK)
    parser.add_argument("--restore-steps", type=int, default=RESTORE_STEPS)
    parser.add_argument("--restore-lr", type=float, default=RESTORE_LR)
    parser.add_argument("--restore-delta-l2", type=float, default=RESTORE_L2)
    parser.add_argument("--solver-rho", type=float, default=SOLVER_RHO)
    parser.add_argument("--solver-feasibility-tolerance", type=float, default=SOLVER_TOLERANCE)
    parser.add_argument("--solver-stall-patience", type=int, default=SOLVER_STALL_PATIENCE)
    parser.add_argument("--solver-min-improvement", type=float, default=SOLVER_MIN_IMPROVEMENT)
    parser.add_argument("--constraint-generation-max-rounds", type=int, default=MAX_CONSTRAINT_ROUNDS)
    parser.add_argument("--multihop-prompt-dir", default="data/mquake_prompts")
    parser.add_argument("--multihop-batch-size", type=int, default=4)
    parser.add_argument("--standard-max-new-tokens", type=int, default=32)
    parser.add_argument("--cot-max-new-tokens", type=int, default=128)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    pinned = {
        "forget_num": 1000,
        "retain_num": 1000,
        "steps": 600,
        "batch_size": 1,
        "retain_batch_size": 4,
        "emb_lm_lr": 1e-4,
        "forget_weight": 2.0,
        "retain_weight": 1.0,
        "forget_margin": 1.0,
        "emb_lm_optimizer": "adamw",
        "sampling_strategy": "epoch",
        "forget_sampling": "instance_balanced",
        "active_logit_margin": ACTIVE_MARGIN,
        "selection_logit_margin": SELECTION_MARGIN,
        "restore_rank": RESTORE_RANK,
        "rank0_lr": RANK0_LR,
        "rank0_steps_per_phase": RANK0_STEPS_PER_PHASE,
        "rank0_delta_l2": RANK0_DELTA_L2,
        "restore_steps": RESTORE_STEPS,
        "restore_lr": RESTORE_LR,
        "restore_delta_l2": RESTORE_L2,
        "solver_rho": SOLVER_RHO,
        "solver_feasibility_tolerance": SOLVER_TOLERANCE,
        "solver_stall_patience": SOLVER_STALL_PATIENCE,
        "solver_min_improvement": SOLVER_MIN_IMPROVEMENT,
        "constraint_generation_max_rounds": MAX_CONSTRAINT_ROUNDS,
        "retain_calibration_num": 1000,
        "retain_calibration_seed": 1729,
        "target_eff_max": 0.0,
        "utility_drop_tolerance": 0.10,
        "max_ppl_ratio": 1.02,
    }
    for name, expected in pinned.items():
        if getattr(args, name) != expected:
            raise ValueError(f"pinned protocol requires {name}={expected}")
    if args.mquake_url != mquake.MQUAKE_URL or mquake.MQUAKE_REV != DATASET_REVISION:
        raise ValueError("the pinned MQuAKE-CF-3k-v2 revision is mandatory")
    if args.skip_ppl or not args.strict_utility_gates:
        raise ValueError("strict exact Base utility/PPL gates are mandatory")
    for name in (
        "rank0_steps_per_phase",
        "restore_steps",
        "solver_stall_patience",
        "constraint_generation_max_rounds",
        "eval_batch_size",
        "cache_batch_size",
        "multihop_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")


def load_selection_visible_records(*args: Any, **kwargs: Any) -> Any:
    return basearch.load_selection_visible_records(*args, **kwargs)


def cache_token_states(*args: Any, **kwargs: Any) -> Any:
    return basearch.cache_token_states(*args, **kwargs)


def constraints_from_audit(*args: Any, **kwargs: Any) -> Any:
    return basearch.constraints_from_audit(*args, **kwargs)


def selected_row_ids(constraints: Sequence[PairConstraint]) -> List[int]:
    return basearch.selected_row_ids(constraints)


def prepare_rank0_active_tensors(
    constraints: Sequence[PairConstraint],
    states: Mapping[Tuple[int, str, int, int], TokenState],
    row_ids: Sequence[int],
    base_weight_cpu: torch.Tensor,
    *,
    device: torch.device,
) -> vectorized.ActivePairTensors:
    lookup = {int(token_id): index for index, token_id in enumerate(row_ids)}
    hidden = torch.stack([states[item.state_identity].hidden.float() for item in constraints]).to(device)
    sensitive_ids = torch.tensor(
        [item.sensitive_token_id for item in constraints], dtype=torch.long
    )
    competitor_ids = torch.tensor(
        [item.competitor_token_id for item in constraints], dtype=torch.long
    )
    sensitive_rows = base_weight_cpu.index_select(0, sensitive_ids).to(device, torch.float32)
    competitor_rows = base_weight_cpu.index_select(0, competitor_ids).to(device, torch.float32)
    base_margin = (hidden * (competitor_rows - sensitive_rows)).sum(dim=1)
    sensitive_index = torch.tensor(
        [lookup[int(value)] for value in sensitive_ids.tolist()],
        dtype=torch.long,
        device=device,
    )
    competitor_index = torch.tensor(
        [lookup[int(value)] for value in competitor_ids.tolist()],
        dtype=torch.long,
        device=device,
    )
    return vectorized.ActivePairTensors(
        hidden=hidden,
        base_margin=base_margin,
        sensitive_row_index=sensitive_index,
        competitor_row_index=competitor_index,
        hidden_rank=hidden,
    )


def optimize_rank0_hard_tail(
    tensors: vectorized.ActivePairTensors,
    initial_delta: torch.Tensor,
    *,
    steps: int,
    learning_rate: float,
    active_margin: float,
    delta_l2: float,
) -> Rank0PhaseResult:
    """GPU-vectorized unrestricted max-violation repair; no pair loop."""

    parameter = nn.Parameter(initial_delta.detach().float().clone())
    optimizer: Optional[torch.optim.Optimizer] = None
    logs: List[Dict[str, Any]] = []
    started = time.monotonic()
    for step in range(steps + 1):
        margins = vectorized.active_pair_margins_from_coefficients(tensors, parameter)
        violations = torch.relu(float(active_margin) - margins)
        count = int((violations > 0).sum().item())
        maximum = violations.max() if violations.numel() else parameter.new_zeros(())
        if count == 0 or step == steps:
            reason = "all_active_constraints_satisfied" if count == 0 else "maximum_steps_reached"
            break
        if optimizer is None:
            optimizer = torch.optim.AdamW(
                [parameter], lr=learning_rate, weight_decay=0.0
            )
        optimizer.zero_grad(set_to_none=True)
        hard_tail = maximum.square()
        regularizer = float(delta_l2) * parameter.square().sum()
        loss = hard_tail + regularizer
        loss.backward()
        optimizer.step()
        logs.append(
            {
                "step": step + 1,
                "active_violation_count": count,
                "maximum_active_violation": float(maximum.detach().cpu()),
                "hard_tail_loss": float(hard_tail.detach().cpu()),
                "Base_relative_delta_l2": float(regularizer.detach().cpu()),
            }
        )
    final_margins = vectorized.active_pair_margins_from_coefficients(
        tensors, parameter.detach()
    )
    final_violations = torch.relu(float(active_margin) - final_margins)
    report = {
        "repair_rank": RANK0_REPAIR_RANK,
        "steps": int(step),
        "reason": reason,
        "active_violation_count": int((final_violations > 0).sum().item()),
        "minimum_active_margin": float(final_margins.min().item()),
        "maximum_active_violation": (
            float(final_violations.max().item()) if final_violations.numel() else 0.0
        ),
        "delta_norm": float(parameter.detach().norm().cpu()),
        "wall_clock_seconds": time.monotonic() - started,
        "hot_path": {
            "python_active_pair_loops_per_step": 0,
            "coefficient_shape": list(parameter.shape),
            "unrestricted_hidden_dimension": True,
        },
    }
    return Rank0PhaseResult(parameter.detach(), report, logs)


def expand_rank0_rows(
    old_ids: Sequence[int], new_ids: Sequence[int], delta: torch.Tensor
) -> torch.Tensor:
    return basearch.expand_row_coefficients(old_ids, new_ids, delta)


def constraint_payload(constraints: Sequence[PairConstraint]) -> List[Dict[str, Any]]:
    return [
        {
            "state_identity": list(item.state_identity),
            "sensitive_token_id": item.sensitive_token_id,
            "competitor_token_id": item.competitor_token_id,
            "generation_round": item.generation_round,
        }
        for item in constraints
    ]


def dynamic_rank0_repair(
    *,
    model: nn.Module,
    tok: Any,
    forget_cases: Sequence[mquake.PredictionCase],
    base_states: Sequence[TokenState],
    base_weight_cpu: torch.Tensor,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    max_rounds: int,
    steps_per_phase: int,
) -> Tuple[List[int], torch.Tensor, List[PairConstraint], List[Dict[str, Any]], List[Dict[str, Any]]]:
    state_map = {state.identity: state for state in base_states}
    constraints = constraints_from_audit(
        base_states, margin=ACTIVE_MARGIN, generation_round=0
    )
    row_ids = selected_row_ids(constraints)
    delta = torch.zeros(
        (len(row_ids), base_weight_cpu.shape[1]), device=device, dtype=torch.float32
    )
    phase_reports: List[Dict[str, Any]] = []
    optimization_log: List[Dict[str, Any]] = []
    output = model.get_output_embeddings().weight
    for generation_round in range(max_rounds):
        if not constraints:
            break
        tensors = prepare_rank0_active_tensors(
            constraints, state_map, row_ids, base_weight_cpu, device=device
        )
        phase = optimize_rank0_hard_tail(
            tensors,
            delta,
            steps=steps_per_phase,
            learning_rate=RANK0_LR,
            active_margin=ACTIVE_MARGIN,
            delta_l2=RANK0_DELTA_L2,
        )
        delta = phase.delta
        for row in phase.log:
            optimization_log.append({"constraint_generation_round": generation_round, **row})
        basearch.restore_full_output_to_base(model, base_weight_cpu)
        basearch.materialize_base_anchored_rows(output, row_ids, base_weight_cpu, delta)
        audit = cache_token_states(
            model,
            tok,
            forget_cases,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
        )
        additions = constraints_from_audit(
            audit, margin=ACTIVE_MARGIN, generation_round=generation_round + 1
        )
        merged = basearch.merge_constraints(constraints, additions)
        new_ids = selected_row_ids(merged)
        new_count = len(merged) - len(constraints)
        active_top1 = sum(
            state.predicted_token_id == state.target_token_id for state in audit
        )
        minimum = min(
            (state.runner_up_logit - state.target_logit for state in audit),
            default=math.inf,
        )
        phase_reports.append(
            {
                **phase.report,
                "constraint_generation_round": generation_round,
                "constraint_count_before": len(constraints),
                "constraint_count_after": len(merged),
                "new_constraint_count": new_count,
                "selected_row_count_after": len(new_ids),
                "BF16_sensitive_top1_count": active_top1,
                "BF16_minimum_true_runner_margin": minimum,
                "constraint_set_sha256": basearch.constraints_sha256(merged),
            }
        )
        if new_ids != row_ids:
            delta = expand_rank0_rows(row_ids, new_ids, delta)
        constraints = merged
        row_ids = new_ids
        if new_count == 0:
            if (
                phase.report["active_violation_count"] == 0
                and active_top1 == 0
                and minimum >= ACTIVE_MARGIN
            ):
                return row_ids, delta, constraints, phase_reports, optimization_log
            raise RuntimeError("rank-0 active set stabilized without perfect forgetting")
    if constraints:
        raise RuntimeError("rank-0 dynamic constraint generation exhausted its fixed rounds")
    return row_ids, delta, constraints, phase_reports, optimization_log


def prepare_protected_restore_tensors(
    protected_states: Sequence[TokenState],
    row_ids: Sequence[int],
    base_weight_cpu: torch.Tensor,
    restore_basis: torch.Tensor,
    delta_forget: torch.Tensor,
    *,
    device: torch.device,
) -> ProtectedRestoreTensors:
    protected = basearch.prepare_base_protected_tensors(
        protected_states, row_ids, base_weight_cpu, restore_basis, device=device
    )
    fixed = protected.hidden @ delta_forget.T
    runner_ids = torch.tensor(
        [state.runner_up_token_id for state in protected_states], dtype=torch.long
    )
    runner_rows = base_weight_cpu.index_select(0, runner_ids).to(device, torch.float32)
    runner_base = (protected.hidden * runner_rows).sum(dim=1)
    lookup = {int(token_id): index for index, token_id in enumerate(row_ids)}
    runner_index = torch.tensor(
        [lookup.get(int(token_id), -1) for token_id in runner_ids.tolist()],
        dtype=torch.long,
        device=device,
    )
    return ProtectedRestoreTensors(protected, fixed, runner_base, runner_index)


def protected_restore_margins(
    tensors: ProtectedRestoreTensors, coefficients: torch.Tensor
) -> torch.Tensor:
    """Vectorized Base-correct margins, including selected correct/runner rows."""

    protected = tensors.protected
    restore_logits = protected.hidden_rank @ coefficients.T
    total = tensors.fixed_delta_logits + restore_logits
    correct_index = protected.correct_modified_row_index
    valid_correct = correct_index >= 0
    correct_safe = correct_index.clamp_min(0)
    correct_delta = total.gather(1, correct_safe[:, None]).squeeze(1)
    correct_delta = torch.where(
        valid_correct, correct_delta, torch.zeros_like(correct_delta)
    )
    correct_after = protected.correct_base + correct_delta
    selected_margins = (
        correct_after[:, None] - (protected.base_modified_logits + total)
    )[protected.competitor_mask]
    runner_index = tensors.runner_selected_row_index
    valid_runner = runner_index >= 0
    runner_safe = runner_index.clamp_min(0)
    runner_delta = total.gather(1, runner_safe[:, None]).squeeze(1)
    runner_delta = torch.where(
        valid_runner, runner_delta, torch.zeros_like(runner_delta)
    )
    runner_margin = correct_after - (tensors.runner_base_logits + runner_delta)
    return torch.cat((selected_margins, runner_margin))


def exact_restore_ppl_mean_nll(
    tensors: PreparedPPLTensors,
    fixed_delta_logits: torch.Tensor,
    coefficients: torch.Tensor,
) -> torch.Tensor:
    restore_logits = tensors.hidden_rank @ coefficients.T
    return basearch.exact_selected_row_mean_nll_from_delta_logits(
        base_logsumexp=tensors.base_logsumexp,
        base_target_logits=tensors.base_target_logits,
        base_selected_logits=tensors.base_selected_logits,
        candidate_selected_logits=(
            tensors.base_selected_logits + fixed_delta_logits + restore_logits
        ),
        target_selected_row_index=tensors.target_selected_row_index,
        normalization_divisor=tensors.normalization_divisor,
    )


def prepare_stage_c_forget_tensors(
    constraints: Sequence[PairConstraint],
    states: Mapping[Tuple[int, str, int, int], TokenState],
    row_ids: Sequence[int],
    base_weight_cpu: torch.Tensor,
    delta_forget: torch.Tensor,
    restore_basis: torch.Tensor,
    *,
    device: torch.device,
) -> StageCForgetTensors:
    """Pack Stage-C forget constraints once; loops stay outside optimization."""

    if delta_forget.shape != (len(row_ids), base_weight_cpu.shape[1]):
        raise ValueError("Stage-B delta shape does not match fixed selected rows")
    if restore_basis.shape != (RESTORE_RANK, base_weight_cpu.shape[1]):
        raise ValueError("Stage-C restoration basis must remain exactly rank 64")
    if not constraints:
        return StageCForgetTensors(
            hidden_rank=torch.empty(
                (0, RESTORE_RANK), device=device, dtype=torch.float32
            ),
            fixed_stage_b_margin=torch.empty(
                (0,), device=device, dtype=torch.float32
            ),
            sensitive_selected_row_index=torch.empty(
                (0,), device=device, dtype=torch.long
            ),
            competitor_selected_row_index=torch.empty(
                (0,), device=device, dtype=torch.long
            ),
        )
    hidden = torch.stack(
        [states[item.state_identity].hidden.float() for item in constraints]
    ).to(device)
    sensitive_ids_cpu = torch.tensor(
        [item.sensitive_token_id for item in constraints], dtype=torch.long
    )
    competitor_ids_cpu = torch.tensor(
        [item.competitor_token_id for item in constraints], dtype=torch.long
    )
    sensitive_rows = base_weight_cpu.index_select(0, sensitive_ids_cpu).to(
        device=device, dtype=torch.float32
    )
    competitor_rows = base_weight_cpu.index_select(0, competitor_ids_cpu).to(
        device=device, dtype=torch.float32
    )
    lookup = {int(token_id): index for index, token_id in enumerate(row_ids)}
    sensitive_index = torch.tensor(
        [lookup.get(int(token_id), -1) for token_id in sensitive_ids_cpu.tolist()],
        device=device,
        dtype=torch.long,
    )
    competitor_index = torch.tensor(
        [lookup.get(int(token_id), -1) for token_id in competitor_ids_cpu.tolist()],
        device=device,
        dtype=torch.long,
    )

    def fixed_selected_logits(index: torch.Tensor) -> torch.Tensor:
        if delta_forget.shape[0] == 0:
            return hidden.new_zeros((hidden.shape[0],))
        valid = index >= 0
        safe = index.clamp_min(0)
        rows = delta_forget.index_select(0, safe)
        logits = (hidden * rows).sum(dim=1)
        return torch.where(valid, logits, torch.zeros_like(logits))

    base_margin = (hidden * (competitor_rows - sensitive_rows)).sum(dim=1)
    fixed_margin = (
        base_margin
        + fixed_selected_logits(competitor_index)
        - fixed_selected_logits(sensitive_index)
    )
    return StageCForgetTensors(
        hidden_rank=hidden @ restore_basis.T,
        fixed_stage_b_margin=fixed_margin,
        sensitive_selected_row_index=sensitive_index,
        competitor_selected_row_index=competitor_index,
    )


def stage_c_forget_margins(
    tensors: StageCForgetTensors, coefficients: torch.Tensor
) -> torch.Tensor:
    """Vectorized Stage-B-fixed plus rank-64 restoration pair margins."""

    if coefficients.ndim != 2 or coefficients.shape[1] != RESTORE_RANK:
        raise ValueError("Stage-C coefficients must have exactly 64 columns")
    if not tensors.fixed_stage_b_margin.numel():
        return coefficients.new_empty((0,))

    def selected_effect(index: torch.Tensor) -> torch.Tensor:
        if coefficients.shape[0] == 0:
            return tensors.fixed_stage_b_margin.new_zeros(
                tensors.fixed_stage_b_margin.shape
            )
        valid = index >= 0
        safe = index.clamp_min(0)
        rows = coefficients.index_select(0, safe)
        logits = (tensors.hidden_rank * rows).sum(dim=1)
        return torch.where(valid, logits, torch.zeros_like(logits))

    return (
        tensors.fixed_stage_b_margin
        + selected_effect(tensors.competitor_selected_row_index)
        - selected_effect(tensors.sensitive_selected_row_index)
    )


def warm_start_rank64_coefficients(
    coefficients: torch.Tensor, selected_row_count: int
) -> torch.Tensor:
    expected = (selected_row_count, RESTORE_RANK)
    if coefficients.shape != expected:
        raise ValueError(f"rank-64 warm start shape must be {expected}")
    return coefficients.detach().float().clone()


def stage_c_constraint_updates(
    existing: Sequence[PairConstraint],
    audited_states: Sequence[TokenState],
    *,
    generation_round: int,
) -> Tuple[List[PairConstraint], List[PairConstraint]]:
    additions = constraints_from_audit(
        audited_states,
        margin=ACTIVE_MARGIN,
        generation_round=generation_round,
    )
    return basearch.merge_constraints(existing, additions), additions


def optimize_rank64_restoration(
    *,
    forget_tensors: StageCForgetTensors,
    protected_tensors: ProtectedRestoreTensors,
    ppl_tensors: PreparedPPLTensors,
    fixed_ppl_delta_logits: torch.Tensor,
    selected_row_count: int,
    allowed_mean_nll: float,
    device: torch.device,
    steps: int = RESTORE_STEPS,
    learning_rate: float = RESTORE_LR,
    initial_coefficients: Optional[torch.Tensor] = None,
) -> RestorePhaseResult:
    """Joint rank-64 forget/protection/PPL constrained restoration phase."""

    initial = (
        torch.zeros(
            (selected_row_count, RESTORE_RANK),
            device=device,
            dtype=torch.float32,
        )
        if initial_coefficients is None
        else warm_start_rank64_coefficients(
            initial_coefficients, selected_row_count
        ).to(device)
    )
    coefficients = nn.Parameter(initial)
    optimizer = torch.optim.AdamW(
        [coefficients], lr=learning_rate, weight_decay=0.0
    )
    active_count = int(stage_c_forget_margins(forget_tensors, coefficients).numel())
    protected_count = int(protected_restore_margins(protected_tensors, coefficients).numel())
    active_dual = torch.zeros(active_count, device=device)
    protected_dual = torch.zeros(protected_count, device=device)
    ppl_dual = torch.zeros((), device=device)
    log: List[Dict[str, Any]] = []
    best = math.inf
    stalled = 0
    reason = "maximum_steps_reached"
    started = time.monotonic()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        active_margins = stage_c_forget_margins(forget_tensors, coefficients)
        active_g = float(ACTIVE_MARGIN) - active_margins
        margins = protected_restore_margins(protected_tensors, coefficients)
        protected_g = -margins
        candidate_nll = exact_restore_ppl_mean_nll(
            ppl_tensors, fixed_ppl_delta_logits, coefficients
        )
        ppl_g = candidate_nll - float(allowed_mean_nll)
        active_pos = active_g.clamp_min(0)
        protected_pos = protected_g.clamp_min(0)
        ppl_pos = ppl_g.clamp_min(0)
        active_term = (
            (
                active_dual * active_pos
                + 0.5 * SOLVER_RHO * active_pos.square()
            ).mean()
            if active_pos.numel()
            else coefficients.new_zeros(())
        )
        protected_term = (
            (protected_dual * protected_pos + 0.5 * SOLVER_RHO * protected_pos.square()).mean()
            if protected_pos.numel()
            else coefficients.new_zeros(())
        )
        objective = (
            RESTORE_L2 * coefficients.square().sum()
            + active_term
            + protected_term
            + ppl_dual * ppl_pos
            + 0.5 * SOLVER_RHO * ppl_pos.square()
        )
        objective.backward()
        optimizer.step()
        with torch.no_grad():
            active_dual.add_(SOLVER_RHO * active_g.detach()).clamp_(min=0)
            protected_dual.add_(SOLVER_RHO * protected_g.detach()).clamp_(min=0)
            ppl_dual.add_(SOLVER_RHO * ppl_g.detach()).clamp_(min=0)
            maximum_active = (
                float(active_g.clamp_min(0).max().item())
                if active_g.numel()
                else 0.0
            )
            maximum_protected = (
                float(protected_g.clamp_min(0).max().item())
                if protected_g.numel()
                else 0.0
            )
            ppl_excess = float(ppl_g.clamp_min(0).item())
            active_violation_count = int(
                (active_g > SOLVER_TOLERANCE).sum().item()
            )
            protected_violation_count = int(
                (protected_g > SOLVER_TOLERANCE).sum().item()
            )
            current = max(maximum_active, maximum_protected, ppl_excess)
        log.append(
            {
                "step": step,
                "restore_rank": RESTORE_RANK,
                "active_violation_count": active_violation_count,
                "maximum_active_violation": maximum_active,
                "protected_violation_count": protected_violation_count,
                "maximum_protected_violation": maximum_protected,
                "PPL_NLL_excess": ppl_excess,
                "candidate_mean_NLL": float(candidate_nll.detach().cpu()),
                "coefficient_norm": float(coefficients.detach().norm().cpu()),
            }
        )
        if current <= SOLVER_TOLERANCE:
            reason = "exact_configured_joint_feasibility_reached"
            break
        if current + SOLVER_MIN_IMPROVEMENT < best:
            best = current
            stalled = 0
        else:
            stalled += 1
        if stalled >= SOLVER_STALL_PATIENCE:
            reason = "deterministic_stall_criterion_reached"
            break
    final_active_margins = stage_c_forget_margins(
        forget_tensors, coefficients.detach()
    )
    final_margins = protected_restore_margins(
        protected_tensors, coefficients.detach()
    )
    final_nll = exact_restore_ppl_mean_nll(
        ppl_tensors, fixed_ppl_delta_logits, coefficients.detach()
    )
    report = {
        "restore_rank": RESTORE_RANK,
        "steps": int(step),
        "reason": reason,
        "active_constraint_count": int(final_active_margins.numel()),
        "active_violation_count": int(
            (final_active_margins < ACTIVE_MARGIN).sum().item()
        ),
        "minimum_active_margin": (
            float(final_active_margins.min().item())
            if final_active_margins.numel()
            else math.inf
        ),
        "protected_violation_count": int((final_margins < 0).sum().item()),
        "minimum_protected_margin": float(final_margins.min().item()),
        "candidate_mean_NLL": float(final_nll.item()),
        "allowed_mean_NLL": float(allowed_mean_nll),
        "PPL_NLL_excess": max(0.0, float(final_nll.item()) - allowed_mean_nll),
        "coefficient_shape": list(coefficients.shape),
        "coefficient_norm": float(coefficients.detach().norm().cpu()),
        "wall_clock_seconds": time.monotonic() - started,
        "hot_path": {
            "python_active_pair_loops_per_step": 0,
            "python_protected_state_loops_per_step": 0,
            "python_PPL_state_loops_per_step": 0,
            "full_hidden_restoration_delta_materialized_per_step": False,
            "H_retain_rank64_precomputed": True,
            "H_ppl_rank64_precomputed": True,
            "H_forget_rank64_precomputed": True,
        },
    }
    return RestorePhaseResult(coefficients.detach(), report, log)


def dynamic_rank64_restoration(
    *,
    model: nn.Module,
    tok: Any,
    forget_cases: Sequence[mquake.PredictionCase],
    base_states: Sequence[TokenState],
    initial_constraints: Sequence[PairConstraint],
    row_ids: Sequence[int],
    base_weight_cpu: torch.Tensor,
    delta_forget: torch.Tensor,
    restore_basis: torch.Tensor,
    protected_tensors: ProtectedRestoreTensors,
    ppl_tensors: PreparedPPLTensors,
    fixed_ppl_delta_logits: torch.Tensor,
    allowed_mean_nll: float,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
    max_rounds: int,
    steps_per_phase: int,
    learning_rate: float,
) -> DynamicRestoreResult:
    """Generate true-runner constraints while preserving fixed Stage-A rows."""

    if list(row_ids) != sorted(set(int(value) for value in row_ids)):
        raise ValueError("Stage-C selected row IDs must stay sorted and unique")
    state_map = {state.identity: state for state in base_states}
    constraints = list(initial_constraints)
    coefficients = torch.zeros(
        (len(row_ids), RESTORE_RANK), device=device, dtype=torch.float32
    )
    phase_reports: List[Dict[str, Any]] = []
    optimization_log: List[Dict[str, Any]] = []
    final_states: List[TokenState] = []
    output_weight = model.get_output_embeddings().weight
    for generation_round in range(max_rounds):
        forget_tensors = prepare_stage_c_forget_tensors(
            constraints,
            state_map,
            row_ids,
            base_weight_cpu,
            delta_forget,
            restore_basis,
            device=device,
        )
        phase = optimize_rank64_restoration(
            forget_tensors=forget_tensors,
            protected_tensors=protected_tensors,
            ppl_tensors=ppl_tensors,
            fixed_ppl_delta_logits=fixed_ppl_delta_logits,
            selected_row_count=len(row_ids),
            allowed_mean_nll=allowed_mean_nll,
            device=device,
            steps=steps_per_phase,
            learning_rate=learning_rate,
            initial_coefficients=coefficients,
        )
        coefficients = warm_start_rank64_coefficients(
            phase.coefficients, len(row_ids)
        ).to(device)
        for row in phase.log:
            optimization_log.append(
                {"constraint_generation_round": generation_round, **row}
            )
        restoration_delta = rank64_restoration_delta(
            coefficients, restore_basis
        )
        final_delta = delta_forget + restoration_delta
        basearch.restore_full_output_to_base(model, base_weight_cpu)
        basearch.materialize_base_anchored_rows(
            output_weight, row_ids, base_weight_cpu, final_delta
        )
        final_states = cache_token_states(
            model,
            tok,
            forget_cases,
            device=device,
            llama_like=llama_like,
            batch_size=batch_size,
        )
        merged, additions = stage_c_constraint_updates(
            constraints,
            final_states,
            generation_round=generation_round + 1,
        )
        new_constraint_count = len(merged) - len(constraints)
        raw_margins = [
            state.runner_up_logit - state.target_logit for state in final_states
        ]
        raw_violation_count = sum(
            margin < ACTIVE_MARGIN for margin in raw_margins
        )
        raw_minimum = min(raw_margins, default=math.inf)
        phase_reports.append(
            {
                **phase.report,
                "constraint_generation_round": generation_round,
                "constraint_count_before": len(constraints),
                "constraint_count_after": len(merged),
                "new_constraint_count": new_constraint_count,
                "audited_forget_state_count": len(final_states),
                "audited_violation_count": int(raw_violation_count),
                "audited_minimum_true_runner_margin": float(raw_minimum),
                "constraint_set_sha256": basearch.constraints_sha256(merged),
                "selected_row_count": len(row_ids),
                "selected_row_count_unchanged": True,
                "editable_row_set_expanded": False,
                "rank64_coefficients_warm_started": generation_round > 0,
            }
        )
        constraints = merged
        if (
            phase.report["active_violation_count"] == 0
            and raw_violation_count == 0
            and raw_minimum >= ACTIVE_MARGIN
        ):
            return DynamicRestoreResult(
                coefficients,
                constraints,
                phase_reports,
                optimization_log,
                final_states,
            )
        if new_constraint_count == 0:
            break
    return DynamicRestoreResult(
        coefficients,
        constraints,
        phase_reports,
        optimization_log,
        final_states,
    )


def nonselected_rows_equal_base(
    output_weight: torch.Tensor,
    base_weight_cpu: torch.Tensor,
    selected_ids: Sequence[int],
) -> bool:
    if output_weight.shape != base_weight_cpu.shape:
        return False
    mask = torch.ones(output_weight.shape[0], dtype=torch.bool, device=output_weight.device)
    if selected_ids:
        mask[torch.tensor(selected_ids, dtype=torch.long, device=output_weight.device)] = False
    return torch.equal(
        output_weight.detach()[mask].cpu(),
        base_weight_cpu[mask.cpu()].to(dtype=output_weight.dtype),
    )


def materialized_selected_delta(
    output_weight: torch.Tensor,
    base_weight_cpu: torch.Tensor,
    row_ids: Sequence[int],
) -> torch.Tensor:
    ids = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    current = output_weight.index_select(0, ids).float()
    base = base_weight_cpu.index_select(0, ids.cpu()).to(output_weight.device, torch.float32)
    return current - base


def exact_materialized_ppl_nll(
    cache: ExactPPLCache,
    row_ids: Sequence[int],
    materialized_delta: torch.Tensor,
    *,
    device: torch.device,
) -> float:
    # prepare_ppl_tensors needs only the selected-row static Base fields; the
    # rank projection is unused in this exact materialized audit.
    prepared = basearch.prepare_ppl_tensors(
        cache,
        row_ids,
        torch.zeros((1, cache.hidden.shape[1]), device=device),
        device=device,
    )
    candidate_selected = (
        prepared.base_selected_logits
        + cache.hidden.to(device) @ materialized_delta.T
    )
    return float(
        basearch.exact_selected_row_mean_nll_from_delta_logits(
            base_logsumexp=prepared.base_logsumexp,
            base_target_logits=prepared.base_target_logits,
            base_selected_logits=prepared.base_selected_logits,
            candidate_selected_logits=candidate_selected,
            target_selected_row_index=prepared.target_selected_row_index,
            normalization_divisor=prepared.normalization_divisor,
        ).item()
    )


def base_repair_origin_report(
    model: nn.Module,
    base_weight_cpu: torch.Tensor,
    base_transformer_hash: str,
) -> Dict[str, Any]:
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    report = {
        "input_equals_base_exactly": basearch.tensors_equal_chunked(
            input_weight, base_weight_cpu
        ),
        "output_equals_base_exactly": basearch.tensors_equal_chunked(
            output_weight, base_weight_cpu
        ),
        "input_output_pointers_distinct": input_weight.data_ptr() != output_weight.data_ptr(),
        "transformer_equals_base_exactly": transformer_sha256(model) == base_transformer_hash,
        "transformer_frozen": all(
            not parameter.requires_grad
            for parameter in model.parameters()
            if parameter.data_ptr() not in {input_weight.data_ptr(), output_weight.data_ptr()}
        ),
        "input_embeddings_frozen": not input_weight.requires_grad,
        "output_head_frozen": not output_weight.requires_grad,
        "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", True)),
        "candidate_origin": "exact Base",
    }
    if not all(
        report[key]
        for key in (
            "input_equals_base_exactly",
            "output_equals_base_exactly",
            "input_output_pointers_distinct",
            "transformer_equals_base_exactly",
            "transformer_frozen",
            "input_embeddings_frozen",
            "output_head_frozen",
        )
    ) or report["tie_word_embeddings"]:
        raise RuntimeError("Base repair origin assertions failed")
    return report


def bf16_forget_audit(
    states: Sequence[TokenState], *, selection_margin: float
) -> Dict[str, Any]:
    top1 = sum(state.predicted_token_id == state.target_token_id for state in states)
    margins = [state.runner_up_logit - state.target_logit for state in states]
    minimum = min(margins, default=math.inf)
    violations = sum(margin < selection_margin for margin in margins)
    return {
        "sensitive_top1_count": int(top1),
        "active_violation_count": int(violations),
        "minimum_selection_margin": float(minimum),
        "selection_margin": float(selection_margin),
        "passed": top1 == 0 and violations == 0 and minimum >= selection_margin,
    }


def evaluate_held_out_after_durable_acceptance(
    *, accepted: bool, load_records: Any, evaluate: Any
) -> Any:
    return basearch.evaluate_held_out_after_durable_acceptance(
        accepted=accepted, load_records=load_records, evaluate=evaluate
    )


def _stage_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    return basearch.stage_metrics(result)


def _write_results(
    output_dir: Path,
    *,
    base_result: Mapping[str, Any],
    setting5_result: Mapping[str, Any],
    detied_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    selected_result: Mapping[str, Any],
    selection: Mapping[str, Any],
    repair: Mapping[str, Any],
) -> None:
    gagd.write_json(
        output_dir / "mquake_results.json",
        {
            "method": METHOD_LABEL,
            "restore_rank": RESTORE_RANK,
            "Base": _stage_metrics(base_result),
            "Setting5e_tied": _stage_metrics(setting5_result),
            "Setting5e_detied": _stage_metrics(detied_result),
            "Candidate": _stage_metrics(candidate_result),
            "Selected": _stage_metrics(selected_result),
            "repair": dict(repair),
            "selection": dict(selection),
            "held_out": {
                "status": (
                    "pending_post_selection_evaluation"
                    if selection["candidate_accepted"]
                    else "not_evaluated_candidate_rejected"
                ),
                "AtomicGen": None,
                "RetainAtomicGen": None,
                "MHLeak_exact_any": None,
                "MHLeak_contains_any": None,
                "standard_multihop": None,
                "cot_multihop": None,
            },
        },
    )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing experiment: {output_dir}")
    setting5_dir = output_dir / "setting5e"
    detied_dir = output_dir / "detied_setting5e"
    geometry_dir = output_dir / "geometry"
    stage_a_dir = output_dir / "stage_a_rank0"
    stage_b_dir = output_dir / "stage_b_forget_projection"
    stage_c_dir = output_dir / "stage_c_rank64_restore"
    for path in (
        output_dir,
        setting5_dir,
        detied_dir,
        geometry_dir,
        stage_a_dir,
        stage_b_dir,
        stage_c_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    mquake_path = Path(args.mquake_path)
    if not mquake_path.is_absolute():
        mquake_path = gagd.PROJECT_DIR / mquake_path
    mquake_path = mquake.download_mquake(mquake_path, url=args.mquake_url)
    wikidata_dir = gagd.resolve_output_path(args.wikidata_dir)
    config = {
        **vars(args),
        "method": METHOD,
        "method_label": METHOD_LABEL,
        "dataset_revision": DATASET_REVISION,
        "restore_rank": RESTORE_RANK,
        "base_relative_parameterization": True,
        "rank_fallback_or_sweep": False,
        "held_out_until_durable_acceptance": list(HELD_OUT_FIELDS),
        "raw_MQuAKE_target_new_used_for_training": False,
    }
    gagd.write_json(output_dir / "config_used.json", config)

    print("Loading Base and caching selection-visible direct-cloze geometry")
    base_model, tok = gagd.load_model_and_tokenizer(args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(base_model.parameters()).device
    base_weight_cpu, base_weight_report = basearch.snapshot_base_weight(base_model)
    base_transformer_hash = transformer_sha256(base_model)
    forget_records, retain_records = load_selection_visible_records(
        mquake_path,
        tok,
        forget_num=args.forget_num,
        retain_num=args.retain_num,
        seed=args.seed,
        mquake_url=args.mquake_url,
    )
    preselection_records = (forget_records, retain_records)
    split_manifest = output_dir / "split_manifest.json"
    mquake.write_split_manifest(
        split_manifest,
        mquake_path=mquake_path,
        seed=args.seed,
        forget_records=forget_records,
        retain_records=retain_records,
    )
    neutral_id = mquake.resolve_neutral_target_token_id(tok)
    llama_like = mquake.is_llama_like(base_model, tok)
    forget_cases = basearch.build_repair_cases(forget_records, tok, llama_like=llama_like)
    calibration_records = vectorized.sample_retain_instances(
        retain_records, args.retain_calibration_num, args.retain_calibration_seed
    )
    retain_cases = basearch.build_repair_cases(
        calibration_records, tok, llama_like=llama_like
    )
    base_forget_states = cache_token_states(
        base_model,
        tok,
        forget_cases,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    base_retain_states = cache_token_states(
        base_model,
        tok,
        retain_cases,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    protected_states = [
        state
        for state in base_retain_states
        if state.predicted_token_id == state.target_token_id
    ]
    base_result = baseline.evaluate_eff_only(
        method="Base",
        model=base_model,
        tok=tok,
        model_dir=args.model_path,
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=output_dir / "base_official_eval.json",
        args=args,
        records=preselection_records,
    )
    ppl_text = load_official_ppl_text(wikidata_dir)
    if ppl_text is None:
        raise RuntimeError("official Base PPL text is required")
    ppl_cache = basearch.build_exact_ppl_cache(
        base_model, tok, ppl_text, device=device
    )

    forget_hidden = torch.stack(
        [state.hidden.float() for state in base_forget_states]
    ).to(device)
    forget_geometry = robust_svd_qr_row_basis(forget_hidden)
    forget_report = {
        **forget_geometry.report,
        "forget_state_count": forget_geometry.report["state_count"],
        "forget_numerical_rank": forget_geometry.report["numerical_rank"],
        "requested_restore_rank": RESTORE_RANK,
    }
    gagd.write_json(geometry_dir / "forget_basis_report.json", forget_report)
    require_forget_nullspace(forget_report, output_dir)
    utility_hidden = torch.cat(
        (
            torch.stack([state.hidden.float() for state in protected_states]).to(device),
            ppl_cache.hidden.to(device),
        ),
        dim=0,
    )
    utility_null, restore_basis, restore_report = build_restore_basis64(
        utility_hidden,
        forget_geometry.basis,
        output_dir=output_dir,
        forget_report=forget_report,
    )
    gagd.write_json(
        geometry_dir / "nullspace_geometry.json",
        {
            "restore_rank": RESTORE_RANK,
            "forget_numerical_rank": forget_report["forget_numerical_rank"],
            "exact_nullspace_dimension": forget_report["exact_nullspace_dimension"],
            "utility_hidden_shape": list(utility_hidden.shape),
            "utility_null_shape": list(utility_null.shape),
            "rank_fallback_or_sweep": False,
        },
    )
    gagd.write_json(
        geometry_dir / "utility_nullspace_basis_report.json", restore_report
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training unchanged Setting5e @600 diagnostic")
    gagd.set_seed(args.seed)
    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    device = next(model.parameters()).device
    if tensor_sha256(model.get_input_embeddings().weight) != base_weight_report["sha256"]:
        raise RuntimeError("training Base differs from frozen Base snapshot")
    if transformer_sha256(model) != base_transformer_hash:
        raise RuntimeError("training transformer differs from frozen Base")
    forget_examples, sampling_report = vectorized.setting5_forget_examples(
        forget_records,
        tok,
        strategy=args.forget_sampling,
        steps=args.steps,
        seed=args.seed,
    )
    retain_examples = baseline.canonical_examples(retain_records, tok)
    gagd.write_json(setting5_dir / "forget_sampling.json", sampling_report)
    args.post_training_excluded_token_ids = [neutral_id]
    requested_save = bool(args.save_model)
    args.save_model = False
    training_summary = gagd.train_mode(
        model,
        tok,
        forget_examples,
        retain_examples,
        selected_ids=[],
        mode=SETTING5_MODE,
        args=args,
        mode_dir=setting5_dir,
    )
    args.save_model = requested_save
    setting5_result = baseline.evaluate_eff_only(
        method="Setting5e @600 tied",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e-tied",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=setting5_dir / "official_eval.json",
        args=args,
        records=preselection_records,
    )
    if args.save_setting5_checkpoint:
        baseline.save_checkpoint(model, tok, setting5_dir / "checkpoint")

    print("De-tying Setting5 diagnostic, then restoring exact Base repair origin")
    detie = basearch.detie_restore_base_embeddings(model, base_weight_cpu)
    detied_result = baseline.evaluate_eff_only(
        method="Setting5e @600 de-tied diagnostic",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e-detied",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=detied_dir / "official_eval.json",
        args=args,
        records=preselection_records,
    )
    basearch.restore_full_output_to_base(model, base_weight_cpu)
    origin = {
        **base_repair_origin_report(model, base_weight_cpu, base_transformer_hash),
        "detie_diagnostic": detie,
    }
    gagd.write_json(output_dir / "base_repair_origin.json", origin)
    output_weight = model.get_output_embeddings().weight

    print("Stage A: unrestricted Base-anchored rank-0 dynamic forgetting")
    row_ids, delta0, constraints, phase_reports, stage_a_log = dynamic_rank0_repair(
        model=model,
        tok=tok,
        forget_cases=forget_cases,
        base_states=base_forget_states,
        base_weight_cpu=base_weight_cpu,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        max_rounds=args.constraint_generation_max_rounds,
        steps_per_phase=args.rank0_steps_per_phase,
    )
    _write_jsonl(stage_a_dir / "optimization_log.jsonl", stage_a_log)
    gagd.write_json(
        stage_a_dir / "active_constraints.json",
        {
            "repair_rank": 0,
            "active_margin": ACTIVE_MARGIN,
            "constraints": constraint_payload(constraints),
            "constraint_set_sha256": basearch.constraints_sha256(constraints),
            "selected_row_ids": row_ids,
        },
    )
    basearch.restore_full_output_to_base(model, base_weight_cpu)
    basearch.materialize_base_anchored_rows(output_weight, row_ids, base_weight_cpu, delta0)
    stage_a_states = cache_token_states(
        model,
        tok,
        forget_cases,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    stage_a_bf16 = bf16_forget_audit(stage_a_states, selection_margin=ACTIVE_MARGIN)
    stage_a_result = baseline.evaluate_eff_only(
        method="Stage A rank-0 Base-anchored forgetting",
        model=model,
        tok=tok,
        model_dir="in-memory:rank0",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=stage_a_dir / "official_eval.json",
        args=args,
        records=preselection_records,
    )
    stage_a_summary = {
        "repair_rank": 0,
        "restore_rank": RESTORE_RANK,
        "repair_lr": RANK0_LR,
        "active_margin": ACTIVE_MARGIN,
        "Base_relative_delta_l2": RANK0_DELTA_L2,
        "selected_row_count": len(row_ids),
        "phase_reports": phase_reports,
        "BF16_forget_audit": stage_a_bf16,
        "Eff": stage_a_result["forget"]["Eff"],
        "delta_norm": float(delta0.norm().cpu()),
    }
    gagd.write_json(stage_a_dir / "stage_a_summary.json", stage_a_summary)
    if not stage_a_bf16["passed"] or float(stage_a_result["forget"]["Eff"]) != 0.0:
        raise RuntimeError("Stage A did not achieve perfect BF16 direct-cloze forgetting")

    print("Stage B: configured numerical forget-span projection")
    delta_forget = project_delta_to_forget_span(delta0, forget_geometry.basis)
    before_logits = forget_hidden @ delta0.T
    after_logits = forget_hidden @ delta_forget.T
    projection_residual = delta0 - delta_forget
    state_map = {state.identity: state for state in base_forget_states}
    full_before = prepare_rank0_active_tensors(
        constraints, state_map, row_ids, base_weight_cpu, device=device
    )
    before_margins = vectorized.active_pair_margins_from_coefficients(
        full_before, delta0
    )
    after_margins = vectorized.active_pair_margins_from_coefficients(
        full_before, delta_forget
    )
    projected_feasibility = forgetting_feasibility_report(after_margins)
    projection_summary = {
        "restore_rank": RESTORE_RANK,
        "rank0_delta_norm": float(delta0.norm().cpu()),
        "projected_delta_norm": float(delta_forget.norm().cpu()),
        "projection_residual_norm": float(projection_residual.norm().cpu()),
        "max_forget_logit_difference": float((before_logits - after_logits).abs().max().cpu()),
        "max_active_margin_difference": float((before_margins - after_margins).abs().max().cpu()),
        "active_violations_after_projection_FP32": projected_feasibility[
            "active_violation_count"
        ],
        "minimum_active_margin_after_projection_FP32": projected_feasibility[
            "minimum_active_margin"
        ],
        "projection_preserved_forgetting_feasibility": projected_feasibility[
            "passed"
        ],
        "exact_logit_invariance_required": False,
    }
    gagd.write_json(stage_b_dir / "projection_summary.json", projection_summary)
    if not projection_summary["projection_preserved_forgetting_feasibility"]:
        raise RuntimeError("forget-span projection broke FP32 forgetting feasibility")
    basearch.restore_full_output_to_base(model, base_weight_cpu)
    basearch.materialize_base_anchored_rows(
        output_weight, row_ids, base_weight_cpu, delta_forget
    )
    stage_b_states = cache_token_states(
        model,
        tok,
        forget_cases,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    stage_b_bf16 = bf16_forget_audit(
        stage_b_states, selection_margin=SELECTION_MARGIN
    )
    stage_b_result = baseline.evaluate_eff_only(
        method="Stage B Base-anchored forget-span projection",
        model=model,
        tok=tok,
        model_dir="in-memory:forget-projection",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=stage_b_dir / "official_eval.json",
        args=args,
        records=preselection_records,
    )
    stage_b_bf16["Eff"] = stage_b_result["forget"]["Eff"]
    gagd.write_json(stage_b_dir / "bf16_projection_audit.json", stage_b_bf16)
    if not stage_b_bf16["passed"] or float(stage_b_result["forget"]["Eff"]) != 0.0:
        raise RuntimeError("BF16 forget-span projection failed strict forgetting")

    print(
        "Stage C: fixed rank-64 numerical-forget-complement restoration "
        "with explicit forget feasibility"
    )
    protected_tensors = prepare_protected_restore_tensors(
        protected_states,
        row_ids,
        base_weight_cpu,
        restore_basis,
        delta_forget,
        device=device,
    )
    ppl_tensors = basearch.prepare_ppl_tensors(
        ppl_cache, row_ids, restore_basis, device=device
    )
    fixed_ppl_logits = ppl_cache.hidden.to(device) @ delta_forget.T
    allowed_mean_nll = ppl_cache.base_mean_nll + math.log(args.max_ppl_ratio)
    initial_stage_c_constraint_count = len(constraints)
    restoration = dynamic_rank64_restoration(
        model=model,
        tok=tok,
        forget_cases=forget_cases,
        base_states=base_forget_states,
        initial_constraints=constraints,
        row_ids=row_ids,
        base_weight_cpu=base_weight_cpu,
        delta_forget=delta_forget,
        restore_basis=restore_basis,
        protected_tensors=protected_tensors,
        ppl_tensors=ppl_tensors,
        fixed_ppl_delta_logits=fixed_ppl_logits,
        allowed_mean_nll=allowed_mean_nll,
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
        max_rounds=args.constraint_generation_max_rounds,
        steps_per_phase=args.restore_steps,
        learning_rate=args.restore_lr,
    )
    _write_jsonl(stage_c_dir / "optimization_log.jsonl", restoration.log)
    restoration_delta = rank64_restoration_delta(
        restoration.coefficients, restore_basis
    )
    max_forget_change = float(
        (forget_hidden @ restoration_delta.T).abs().max().cpu()
    )
    final_delta = delta_forget + restoration_delta
    basearch.restore_full_output_to_base(model, base_weight_cpu)
    basearch.materialize_base_anchored_rows(
        output_weight, row_ids, base_weight_cpu, final_delta
    )
    final_states = restoration.final_states
    final_stage_c_forget_tensors = prepare_stage_c_forget_tensors(
        restoration.constraints,
        state_map,
        row_ids,
        base_weight_cpu,
        delta_forget,
        restore_basis,
        device=device,
    )
    final_stage_c_margins = stage_c_forget_margins(
        final_stage_c_forget_tensors, restoration.coefficients
    )
    final_stage_c_feasibility = forgetting_feasibility_report(
        final_stage_c_margins
    )
    protected_after = cache_token_states(
        model,
        tok,
        [state.case for state in protected_states],
        device=device,
        llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    protected_regressions = sum(
        state.predicted_token_id != state.target_token_id for state in protected_after
    )
    materialized_delta = materialized_selected_delta(
        output_weight, base_weight_cpu, row_ids
    )
    cached_nll = exact_materialized_ppl_nll(
        ppl_cache, row_ids, materialized_delta, device=device
    )
    direct_nll = basearch.direct_official_mean_nll(
        model, tok, ppl_text, device=device
    )
    bf16_forget = bf16_forget_audit(final_states, selection_margin=SELECTION_MARGIN)
    nonselected_exact = nonselected_rows_equal_base(
        output_weight, base_weight_cpu, row_ids
    )
    bf16_restore_audit = {
        **bf16_forget,
        "restore_rank": RESTORE_RANK,
        "Stage_C_FP32_active_violations": final_stage_c_feasibility[
            "active_violation_count"
        ],
        "Stage_C_FP32_minimum_active_margin": final_stage_c_feasibility[
            "minimum_active_margin"
        ],
        "Stage_C_maximum_FP32_forget_logit_change": max_forget_change,
        "incremental_protected_regressions": int(protected_regressions),
        "candidate_cached_mean_NLL": cached_nll,
        "candidate_direct_BF16_mean_NLL": direct_nll,
        "allowed_mean_NLL": allowed_mean_nll,
        "cached_PPL_gate_pass": cached_nll <= allowed_mean_nll,
        "direct_BF16_PPL_gate_pass": direct_nll
        <= math.log(float(base_result["forget_PPL"])) + math.log(args.max_ppl_ratio),
        "nonselected_rows_exact_Base": nonselected_exact,
    }
    bf16_restore_audit["passed"] = bool(
        bf16_restore_audit["passed"]
        and final_stage_c_feasibility["passed"]
        and protected_regressions == 0
        and bf16_restore_audit["cached_PPL_gate_pass"]
        and bf16_restore_audit["direct_BF16_PPL_gate_pass"]
        and nonselected_exact
    )
    gagd.write_json(stage_c_dir / "bf16_restore_audit.json", bf16_restore_audit)
    restoration_summary = {
        "restore_rank": RESTORE_RANK,
        "initial_forget_constraint_count": initial_stage_c_constraint_count,
        "final_forget_constraint_count": len(restoration.constraints),
        "final_forget_constraint_set_sha256": basearch.constraints_sha256(
            restoration.constraints
        ),
        "forget_constraint_generation_rounds": len(restoration.phase_reports),
        "forget_constraint_generation_reports": restoration.phase_reports,
        "final_solver_phase": (
            restoration.phase_reports[-1]
            if restoration.phase_reports
            else None
        ),
        "Stage_C_FP32_active_violations": final_stage_c_feasibility[
            "active_violation_count"
        ],
        "Stage_C_FP32_minimum_active_margin": final_stage_c_feasibility[
            "minimum_active_margin"
        ],
        "B_R_sha256": restore_report["B_R_sha256"],
        "maximum_absolute_B_R_B_F_overlap": restore_report[
            "maximum_absolute_B_R_B_F_overlap"
        ],
        "Stage_C_maximum_FP32_forget_logit_change": max_forget_change,
        "exact_raw_forget_logit_invariance_required": False,
        "restoration_geometry_claim": (
            "rank-64 restoration operates in the complement of the configured "
            "numerical forget span, with exact raw-forget feasibility enforced "
            "explicitly"
        ),
        "BF16_audit": bf16_restore_audit,
    }
    gagd.write_json(stage_c_dir / "restoration_summary.json", restoration_summary)

    candidate_result = baseline.evaluate_eff_only(
        method=METHOD_LABEL + " candidate",
        model=model,
        tok=tok,
        model_dir="in-memory:rank0-nullrestore64",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=stage_c_dir / "official_eval.json",
        args=args,
        records=preselection_records,
    )
    local_report = {
        "passed": bf16_restore_audit["passed"],
        "protected_incremental_regressions": protected_regressions,
    }
    gate = basearch.final_acceptance_report(
        base_result,
        candidate_result,
        local_report,
        utility_drop_tolerance=args.utility_drop_tolerance,
        max_ppl_ratio=args.max_ppl_ratio,
    )
    accepted = bool(gate["accepted"])
    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        reason = "rank0_forgetting_and_fixed_rank64_restoration_passed_every_gate"
    else:
        basearch.restore_full_output_to_base(model, base_weight_cpu)
        selected_result = copy.deepcopy(base_result)
        reason = "candidate_rejected_and_exact_Base_output_restored"
    selection_commit = {
        "selection_irrevocable": True,
        "candidate_accepted": accepted,
        "selection_reason": reason,
        "restore_rank": RESTORE_RANK,
        "rank_fallback_or_sweep": False,
        "held_out_fields_opened": False,
        "AtomicGen_used_for_selection": False,
        "multihop_used_for_selection": False,
        "gates": gate,
    }
    gagd.write_json(output_dir / "selection_commit.json", selection_commit)
    if args.save_selected_checkpoint:
        basearch.save_detied_checkpoint(model, tok, output_dir / "selected_checkpoint")
    repair_report = {
        "method": METHOD_LABEL,
        "restore_rank": RESTORE_RANK,
        "Setting5_training_summary": asdict(training_summary),
        "base_repair_origin": origin,
        "forget_geometry": forget_report,
        "utility_nullspace_geometry": restore_report,
        "stage_a": stage_a_summary,
        "stage_b": projection_summary,
        "stage_c": restoration_summary,
        "candidate_accepted": accepted,
        "nonselected_rows_exact_Base": nonselected_exact,
        "held_out_fields_opened_before_acceptance": False,
    }
    _write_results(
        output_dir,
        base_result=base_result,
        setting5_result=setting5_result,
        detied_result=detied_result,
        candidate_result=candidate_result,
        selected_result=selected_result,
        selection=selection_commit,
        repair=repair_report,
    )
    if args.fail_if_target_missed and not accepted:
        raise RuntimeError(
            "rank-64 candidate failed a fixed gate; held-out evaluation remains unopened"
        )

    selected_extension, multihop_result = evaluate_held_out_after_durable_acceptance(
        accepted=accepted,
        load_records=lambda: mquake.load_official_eval_records(
            mquake_path,
            tok,
            forget_num=args.forget_num,
            retain_num=args.retain_num,
            seed=args.seed,
            mquake_url=args.mquake_url,
        ),
        evaluate=lambda records: vectorized._evaluate_held_out_after_selection(
            accepted=True,
            args=args,
            model=model,
            tok=tok,
            records=records,
            mquake_path=mquake_path,
            wikidata_dir=wikidata_dir,
            split_manifest=split_manifest,
            output_dir=output_dir,
        ),
    )
    final_payload = json.loads(
        (output_dir / "mquake_results.json").read_text(encoding="utf-8")
    )
    final_payload["held_out"] = {
        "status": "evaluated_after_durable_acceptance",
        "AtomicGen": selected_extension["forget"].get("AtomicGen"),
        "RetainAtomicGen": selected_extension["retain"].get("AtomicGen"),
        "MHLeak_exact_any": multihop_result["results"].get("MHLeak_exact_any"),
        "MHLeak_contains_any": multihop_result["results"].get("MHLeak_contains_any"),
        "standard_multihop": multihop_result["results"].get("standard"),
        "cot_multihop": multihop_result["results"].get("cot"),
        "hop_breakdown": multihop_result["results"].get("hop_breakdown"),
        "PostEditAcc": multihop_result["results"].get("PostEditAcc"),
    }
    final_payload["selection"]["held_out_fields_opened"] = True
    gagd.write_json(output_dir / "mquake_results.json", final_payload)


if __name__ == "__main__":
    main()
