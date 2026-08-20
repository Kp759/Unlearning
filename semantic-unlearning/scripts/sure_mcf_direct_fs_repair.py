#!/usr/bin/env python3
"""Benchmark-aware exact direct-FS repair for an existing SURE checkpoint.

This is deliberately a separate, explicitly labeled MCF ablation.  Unlike the
benchmark-neutral SURE learner, it reads CounterFact ``target_new`` and imposes
the paper-facing direct condition

    mean NLL(target_true) - mean NLL(target_new) >= direct_fs_margin

for every sampled forget record.  The transformer and input embeddings remain
frozen; only LM-head rows belonging to the sensitive ``target_true`` answers
may move.  Existing direct suppression, Wikipedia utility, and total-norm
guards remain hard constraints.  A checkpoint is emitted only after the actual
checkpoint-dtype model obtains 50/50 direct pairwise successes with the
official MCF scorer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import minimize

import build_sure_minimal_split as split_builder
import gagd_compare as gagd
import mcf_zero_unlearn_official_eval as mcf_official
import sure_canonical_core as core
import sure_minimal_two_stage as learner
import sure_shared_suppression as shared


METHOD = "SURE-LM-target-aware-exact-direct-FS-repair-v6"
PROTOCOL = "sure_mcf_target_aware_direct_fs_v6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-checkpoint", required=True)
    parser.add_argument("--learner-dir", required=True)
    parser.add_argument("--mcf-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--direct-fs-margin", type=float, default=0.01)
    parser.add_argument("--direct-fs-solver-buffer", type=float, default=0.05)
    parser.add_argument("--protection-nll-tolerance", type=float, default=0.05)
    parser.add_argument("--protected-materialization-buffer", type=float, default=0.005)
    parser.add_argument("--constraint-buffer", type=float, default=0.05)
    parser.add_argument("--residual-l2-weight", type=float, default=1e-4)
    parser.add_argument("--constraint-context-weight", type=float, default=0.05)
    parser.add_argument("--rank-ladder", default="2,4,8")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--ftol", type=float, default=1e-9)
    parser.add_argument("--constraint-tolerance", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--utility-batch-size", type=int, default=512)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Tuple[int, ...]:
    positive = {
        "maxiter": args.maxiter,
        "ftol": args.ftol,
        "batch_size": args.batch_size,
        "utility_batch_size": args.utility_batch_size,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    nonnegative = {
        "direct_fs_margin": args.direct_fs_margin,
        "direct_fs_solver_buffer": args.direct_fs_solver_buffer,
        "protection_nll_tolerance": args.protection_nll_tolerance,
        "protected_materialization_buffer": args.protected_materialization_buffer,
        "constraint_buffer": args.constraint_buffer,
        "residual_l2_weight": args.residual_l2_weight,
        "constraint_context_weight": args.constraint_context_weight,
        "constraint_tolerance": args.constraint_tolerance,
    }
    for name, value in nonnegative.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    ranks: List[int] = []
    for text in str(args.rank_ladder).split(","):
        rank = int(text.strip())
        if rank <= 0:
            raise ValueError("rank-ladder values must be positive")
        if rank not in ranks:
            ranks.append(rank)
    if not ranks:
        raise ValueError("rank-ladder must not be empty")
    return tuple(ranks)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_mcf_pairs(
    mcf_path: Path,
    manifest_path: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    source_bytes = mcf_path.read_bytes()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset") != "mcf":
        raise RuntimeError("direct-FS repair requires an MCF split manifest")
    if manifest.get("source_sha256") != sha256_bytes(source_bytes):
        raise RuntimeError("MCF source does not match the split manifest")
    raw = json.loads(source_bytes)
    case_ids = manifest.get("sampling", {}).get("forget_case_ids", [])
    if not isinstance(case_ids, list) or not case_ids:
        raise RuntimeError("split manifest lacks forget_case_ids")
    records: List[Dict[str, Any]] = []
    for record_position, case_id_value in enumerate(case_ids):
        case_id = int(case_id_value)
        if case_id < 0 or case_id >= len(raw):
            raise RuntimeError(f"MCF case id is out of range: {case_id}")
        rewrite = split_builder.normalize_mcf_rewrite(raw[case_id])
        sensitive = rewrite.get("target_true", {}).get("str")
        reference = rewrite.get("target_new", {}).get("str")
        if not sensitive or not reference:
            raise RuntimeError(f"MCF case {case_id} lacks target_true/target_new")
        records.append(
            {
                "case_id": case_id,
                "record_position": record_position,
                "requested_rewrite": {
                    "prompt": str(rewrite["prompt"]),
                    "subject": str(rewrite["subject"]),
                    "target_sensitive": {"str": str(sensitive)},
                    "target_reference": {"str": str(reference)},
                },
            }
        )
    return records, manifest


def record_positions(
    cases: Sequence[core.SensitivePredictionCase],
    *,
    device: torch.device,
) -> torch.Tensor:
    return torch.tensor(
        [int(case.record_position) for case in cases],
        device=device,
        dtype=torch.long,
    )


def mean_by_record(
    values: torch.Tensor,
    positions: torch.Tensor,
    record_count: int,
) -> torch.Tensor:
    if values.ndim != 1 or positions.shape != values.shape:
        raise ValueError("values and record positions must be aligned vectors")
    if record_count <= 0:
        raise ValueError("record_count must be positive")
    positions = positions.to(device=values.device, dtype=torch.long)
    if bool((positions < 0).any()) or bool((positions >= record_count).any()):
        raise ValueError("record position is outside the expected range")
    sums = torch.zeros(record_count, device=values.device, dtype=values.dtype)
    counts = torch.zeros(record_count, device=values.device, dtype=values.dtype)
    sums.index_add_(0, positions, values)
    counts.index_add_(0, positions, torch.ones_like(values))
    if bool((counts == 0).any()):
        raise ValueError("every direct record must contribute at least one token")
    return sums / counts


@torch.no_grad()
def build_sequence_cache(
    logits: torch.Tensor,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    positions: torch.Tensor,
    selected_ids: Sequence[int],
    *,
    record_count: int,
    device: torch.device,
) -> Dict[str, Any]:
    if logits.ndim != 2 or hidden.ndim != 2 or logits.shape[0] != hidden.shape[0]:
        raise ValueError("sequence logits and hidden states do not align")
    if target_ids.shape != (logits.shape[0],) or positions.shape != target_ids.shape:
        raise ValueError("sequence target ids/positions do not align")
    if not selected_ids or len(set(int(value) for value in selected_ids)) != len(
        selected_ids
    ):
        raise ValueError("sequence cache requires unique selected row ids")
    current = logits.to(device=device, dtype=torch.float32)
    states = hidden.to(device=device, dtype=torch.float32)
    tids = target_ids.to(device=device, dtype=torch.long)
    pos = positions.to(device=device, dtype=torch.long)
    ids = torch.tensor([int(value) for value in selected_ids], device=device)
    rows = torch.arange(current.shape[0], device=device)
    log_z = torch.logsumexp(current, dim=1)
    selected_logits = current.index_select(1, ids)
    selected_probabilities = torch.exp(selected_logits - log_z.unsqueeze(1))
    token_to_column = {
        int(token_id): column for column, token_id in enumerate(selected_ids)
    }
    target_columns = torch.tensor(
        [token_to_column.get(int(token_id), -1) for token_id in tids.detach().cpu()],
        device=device,
        dtype=torch.long,
    )
    return {
        "hidden": states,
        "selected_probabilities": selected_probabilities,
        "current_target_nll": log_z - current[rows, tids],
        "target_columns": target_columns,
        "record_positions": pos,
        "record_count": int(record_count),
    }


def exact_sequence_record_nll(
    cache: Mapping[str, Any],
    residual_delta: torch.Tensor,
) -> torch.Tensor:
    hidden = cache["hidden"].to(device=residual_delta.device, dtype=torch.float32)
    shifts = hidden @ residual_delta.float().transpose(0, 1)
    probabilities = cache["selected_probabilities"].to(
        device=residual_delta.device, dtype=torch.float32
    )
    log_partition_ratio = learner.exact_sparse_log_partition_ratio(
        shifts, probabilities
    )
    columns = cache["target_columns"].to(device=residual_delta.device)
    safe_columns = columns.clamp_min(0)
    target_shift = shifts.gather(1, safe_columns.unsqueeze(1)).squeeze(1)
    target_shift = torch.where(
        columns >= 0, target_shift, torch.zeros_like(target_shift)
    )
    token_nll = (
        cache["current_target_nll"].to(
            device=residual_delta.device, dtype=torch.float32
        )
        + log_partition_ratio
        - target_shift
    )
    return mean_by_record(
        token_nll,
        cache["record_positions"].to(device=residual_delta.device),
        int(cache["record_count"]),
    )


def exact_pairwise_separation(
    sensitive_cache: Mapping[str, Any],
    reference_cache: Mapping[str, Any],
    residual_delta: torch.Tensor,
) -> torch.Tensor:
    """Return per-record mean NLL(sensitive) - mean NLL(reference)."""
    sensitive = exact_sequence_record_nll(sensitive_cache, residual_delta)
    reference = exact_sequence_record_nll(reference_cache, residual_delta)
    if sensitive.shape != reference.shape:
        raise ValueError("sensitive/reference record NLL vectors do not align")
    return sensitive - reference


def pairwise_report(
    separation: torch.Tensor,
    *,
    required_margin: float,
) -> Dict[str, Any]:
    values = separation.detach().float().cpu()
    official_failures = int((values <= 0).sum().item())
    margin_failures = int((values < float(required_margin)).sum().item())
    return {
        "direct_fs": 100.0 * float(values.numel() - official_failures) / values.numel(),
        "direct_fs_failures": official_failures,
        "direct_fs_margin_failures": margin_failures,
        "direct_fs_required_margin": float(required_margin),
        "minimum_direct_fs_separation": float(values.min().item()),
        "mean_direct_fs_separation": float(values.mean().item()),
        "direct_record_count": int(values.numel()),
    }


@torch.no_grad()
def official_materialized_pairwise_report(
    model: torch.nn.Module,
    tok: Any,
    records: Sequence[Mapping[str, Any]],
    device: torch.device,
    *,
    llama_like: bool,
    required_margin: float,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    separations: List[float] = []
    for record in records:
        rewrite = record["requested_rewrite"]
        prompt = str(rewrite["prompt"]).format(str(rewrite["subject"]))
        sensitive = str(rewrite["target_sensitive"]["str"])
        reference = str(rewrite["target_reference"]["str"])
        score = mcf_official.official_test_batch_prediction(
            model,
            tok,
            [prompt],
            reference,
            sensitive,
            device,
            llama_like=llama_like,
        )[0]
        separation = float(score["target_true"] - score["target_new"])
        separations.append(separation)
        rows.append(
            {
                "case_id": int(record["case_id"]),
                "target_sensitive": sensitive,
                "target_reference": reference,
                "sensitive_nll": float(score["target_true"]),
                "reference_nll": float(score["target_new"]),
                "separation": separation,
                "official_fs_success": bool(separation > 0.0),
                "margin_safe": bool(separation >= float(required_margin)),
            }
        )
    report = pairwise_report(
        torch.tensor(separations, dtype=torch.float32),
        required_margin=required_margin,
    )
    report["records"] = rows
    report["scorer"] = "mcf_zero_unlearn_official_eval.official_test_batch_prediction"
    return report


@torch.no_grad()
def build_target_aware_bases(
    sensitive_hidden: torch.Tensor,
    sensitive_ids: torch.Tensor,
    sensitive_positions: torch.Tensor,
    reference_hidden: torch.Tensor,
    failing_records: Sequence[int],
    utility_second_moment: torch.Tensor,
    *,
    active_ids: Sequence[int],
    rank_cap: int,
    relative_eps: float,
    constraint_context_weight: float,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    if rank_cap <= 0:
        raise ValueError("rank_cap must be positive")
    device = sensitive_hidden.device
    cholesky, _ = learner.regularized_utility_cholesky(
        utility_second_moment,
        relative_eps=relative_eps,
        device=device,
    )
    all_contexts = torch.cat(
        (sensitive_hidden.float(), reference_hidden.float()), dim=0
    )
    whitened_all = torch.linalg.solve_triangular(
        cholesky,
        all_contexts.transpose(0, 1),
        upper=False,
    ).transpose(0, 1) / math.sqrt(float(all_contexts.shape[0]))
    failing = torch.zeros(
        int(sensitive_positions.max().item()) + 1,
        device=device,
        dtype=torch.bool,
    )
    failing[list(int(value) for value in failing_records)] = True
    active_case_mask = failing.index_select(0, sensitive_positions.to(device=device))
    tids = sensitive_ids.to(device=device, dtype=torch.long)
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for token_id in [int(value) for value in active_ids]:
        repair_mask = tids.eq(token_id) & active_case_mask
        repair = sensitive_hidden[repair_mask].float()
        if repair.numel() == 0:
            raise RuntimeError(f"active token row {token_id} has no failing contexts")
        whitened_repair = torch.linalg.solve_triangular(
            cholesky,
            repair.transpose(0, 1),
            upper=False,
        ).transpose(0, 1) / math.sqrt(float(repair.shape[0]))
        components = [whitened_repair]
        if rank_cap > 2 and constraint_context_weight > 0:
            components.append(
                math.sqrt(float(constraint_context_weight)) * whitened_all
            )
        whitened = torch.cat(components, dim=0)
        _, singular_values, right = torch.linalg.svd(whitened, full_matrices=False)
        tolerance = (
            max(whitened.shape)
            * torch.finfo(torch.float32).eps
            * singular_values.max().clamp_min(1.0)
        )
        numerical_rank = int((singular_values > tolerance).sum().item())
        take = min(int(rank_cap), numerical_rank)
        if take <= 0:
            raise RuntimeError(f"active token row {token_id} has zero numerical rank")
        raw = torch.linalg.solve_triangular(
            cholesky.transpose(0, 1),
            right[:take].transpose(0, 1),
            upper=True,
        ).transpose(0, 1)
        basis = core.orthonormal_row_basis(raw, max_rank=take).float()
        bases.append(basis.detach().contiguous())
        reports.append(
            {
                "token_id": token_id,
                "failing_context_count": int(repair.shape[0]),
                "requested_rank": int(rank_cap),
                "actual_rank": int(basis.shape[0]),
                "numerical_rank": numerical_rank,
                "constraint_context_weight": float(constraint_context_weight),
                "basis_protocol": "utility_whitened_FS_repair_plus_all_pair_contexts",
            }
        )
    return bases, reports


def solve_rank(
    *,
    args: argparse.Namespace,
    rank: int,
    row_bases: Sequence[torch.Tensor],
    active_ids: Sequence[int],
    selected_ids: Sequence[int],
    current_total_delta: torch.Tensor,
    direct_cache: Mapping[str, torch.Tensor],
    internal_nll_solver_targets: torch.Tensor,
    internal_margin_solver_target: float,
    sensitive_cache: Mapping[str, Any],
    reference_cache: Mapping[str, Any],
    fs_solver_target: float,
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    utility_budgets: Mapping[str, float],
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    device = current_total_delta.device
    bases = [basis.to(device=device, dtype=torch.float32) for basis in row_bases]
    coefficient_count = sum(int(basis.shape[0]) for basis in bases)

    def residual(coefficients: torch.Tensor) -> torch.Tensor:
        return learner.coefficients_to_residual(coefficients, bases)

    def total(coefficients: torch.Tensor) -> torch.Tensor:
        return learner.total_delta_with_residual(
            current_total_delta,
            selected_ids,
            residual(coefficients),
            active_ids,
        )

    def utility_values(coefficients: torch.Tensor) -> torch.Tensor:
        return learner.exact_sparse_utility_kl(
            total(coefficients), utility_hidden, utility_probabilities
        )

    def objective(coefficients: torch.Tensor) -> torch.Tensor:
        return (
            utility_values(coefficients).mean()
            + float(args.residual_l2_weight) * residual(coefficients).square().sum()
        )

    def behavioral_slacks(coefficients: torch.Tensor) -> torch.Tensor:
        current_residual = residual(coefficients)
        internal = learner.exact_stage2_direct_state(direct_cache, current_residual)
        pairwise = exact_pairwise_separation(
            sensitive_cache, reference_cache, current_residual
        )
        return torch.cat(
            (
                internal["sensitive_nll_increase"] - internal_nll_solver_targets,
                internal["logit_margin"] - float(internal_margin_solver_target),
                pairwise - float(fs_solver_target),
            )
        )

    def utility_slacks(coefficients: torch.Tensor) -> torch.Tensor:
        values = utility_values(coefficients)
        combined = total(coefficients)
        return torch.stack(
            (
                values.new_tensor(float(utility_budgets["mean"])) - values.mean(),
                values.new_tensor(float(utility_budgets["p95"]))
                - torch.quantile(values, 0.95),
                values.new_tensor(float(utility_budgets["max"])) - values.max(),
                values.new_tensor(float(utility_budgets["total_delta_norm"]))
                - combined.norm(),
            )
        )

    scalar = learner._TorchScalarAdapter(objective, device=device)
    behavioral = learner._TorchVectorAdapter(behavioral_slacks, device=device)
    utility = learner._TorchVectorAdapter(utility_slacks, device=device)
    zero = np.zeros(coefficient_count, dtype=np.float64)
    initial_slacks = behavioral.value(zero)
    initial_jacobian = behavioral.jacobian(zero)
    violated = initial_slacks < 0.0
    starts = [zero]
    if bool(np.any(violated)):
        rhs = -initial_slacks[violated] + 0.01
        directed, *_ = np.linalg.lstsq(initial_jacobian[violated], rhs, rcond=None)
        if np.isfinite(directed).all() and not np.allclose(directed, zero):
            starts.append(np.asarray(directed, dtype=np.float64))

    history: List[Dict[str, Any]] = []
    observed: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    tolerance = float(args.constraint_tolerance)

    def inspect(
        values: np.ndarray, restart: int, iteration: int, phase: str
    ) -> Dict[str, Any]:
        coefficients = torch.tensor(values, device=device, dtype=torch.float32)
        current_residual = residual(coefficients).detach()
        pairwise = exact_pairwise_separation(
            sensitive_cache, reference_cache, current_residual
        ).detach()
        internal = learner.exact_stage2_direct_state(direct_cache, current_residual)
        utility_values_now = utility_values(coefficients).detach().double().cpu()
        bslack = behavioral_slacks(coefficients).detach()
        uslack = utility_slacks(coefficients).detach()
        row = {
            "phase": phase,
            "restart": int(restart),
            "iteration": int(iteration),
            "rank": int(rank),
            **pairwise_report(pairwise, required_margin=args.direct_fs_margin),
            "minimum_internal_nll_increase": float(
                internal["sensitive_nll_increase"].min().cpu()
            ),
            "minimum_internal_logit_margin": float(
                internal["logit_margin"].min().cpu()
            ),
            "minimum_behavioral_slack": float(bslack.min().cpu()),
            "minimum_utility_slack": float(uslack.min().cpu()),
            "utility_kl_mean": float(utility_values_now.mean()),
            "utility_kl_p95": float(torch.quantile(utility_values_now, 0.95)),
            "utility_kl_max": float(utility_values_now.max()),
            "residual_delta_norm": float(current_residual.norm().cpu()),
            "total_delta_norm": float(total(coefficients).detach().norm().cpu()),
            "objective": float(objective(coefficients).detach().cpu()),
            "continuous_feasible": bool(
                float(bslack.min().cpu()) >= -tolerance
                and float(uslack.min().cpu()) >= -tolerance
            ),
        }
        history.append(row)
        observed.append((coefficients.detach().cpu(), row))
        return row

    solver_attempts: List[Dict[str, Any]] = []
    for restart, start in enumerate(starts):
        iteration = 0
        inspect(start, restart, 0, "initial")

        def callback(values: np.ndarray) -> None:
            nonlocal iteration
            iteration += 1
            inspect(values, restart, iteration, "iterate")

        result = minimize(
            scalar.value,
            start,
            method="SLSQP",
            jac=scalar.gradient,
            constraints=(
                {
                    "type": "ineq",
                    "fun": behavioral.value,
                    "jac": behavioral.jacobian,
                },
                {
                    "type": "ineq",
                    "fun": utility.value,
                    "jac": utility.jacobian,
                },
            ),
            callback=callback,
            options={
                "maxiter": int(args.maxiter),
                "ftol": float(args.ftol),
                "disp": False,
            },
        )
        final = inspect(result.x, restart, iteration + 1, "final")
        solver_attempts.append(
            {
                "restart": int(restart),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "iterations": int(getattr(result, "nit", 0)),
                "continuous_feasible": bool(final["continuous_feasible"]),
            }
        )

    feasible = [item for item in observed if item[1]["continuous_feasible"]]
    if feasible:
        coefficients, report = min(
            feasible,
            key=lambda item: (
                float(item[1]["objective"]),
                float(item[1]["utility_kl_mean"]),
                float(item[1]["residual_delta_norm"]),
            ),
        )
        selection_mode = "minimum_utility_exact_FS_feasible"
    else:
        coefficients, report = min(
            observed,
            key=lambda item: (
                int(item[1]["direct_fs_margin_failures"]),
                max(0.0, -float(item[1]["minimum_behavioral_slack"])),
                float(item[1]["objective"]),
            ),
        )
        selection_mode = "best_infeasible_diagnostic"
    best = learner.coefficients_to_residual(
        coefficients.to(device=device, dtype=torch.float32), bases
    ).detach()
    return (
        best,
        history,
        {
            **report,
            "selection_mode": selection_mode,
            "solver_attempts": solver_attempts,
            "coefficient_count": coefficient_count,
            "row_ranks": [int(basis.shape[0]) for basis in bases],
        },
    )


def main() -> None:
    args = parse_args()
    rank_ladder = validate_args(args)
    learner_dir = Path(args.learner_dir).resolve()
    input_checkpoint = Path(args.input_checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite target-aware repair: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    neutral_config = json.loads(
        (learner_dir / "config_used.json").read_text(encoding="utf-8")
    )
    records, manifest = load_mcf_pairs(
        Path(args.mcf_path).resolve(), Path(args.split_manifest).resolve()
    )
    if len(records) != int(
        neutral_config["shared_architecture_parameters"]["forget_num"]
    ):
        raise RuntimeError("target-aware records do not match neutral forget count")

    namespace = argparse.Namespace(
        model_path=str(input_checkpoint),
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    sensitive_cases = core.expand_sensitive_cases(
        records,
        tok,
        sensitive_field="target_sensitive",
        llama_like=llama_like,
    )
    reference_cases = core.expand_sensitive_cases(
        records,
        tok,
        sensitive_field="target_reference",
        llama_like=llama_like,
    )
    sensitive_ids = core.official_target_ids(
        tok, sensitive_cases, llama_like=llama_like, device=device
    ).detach()
    reference_ids = core.official_target_ids(
        tok, reference_cases, llama_like=llama_like, device=device
    ).detach()
    sensitive_positions = record_positions(sensitive_cases, device=device)
    reference_positions = record_positions(reference_cases, device=device)

    current_delta_payload = safe_load(learner_dir / "final_total_delta.pt")
    selected_ids = [int(value) for value in current_delta_payload["row_ids"]]
    current_total_delta = current_delta_payload["delta"].to(
        device=device, dtype=torch.float32
    )
    if not set(int(value) for value in sensitive_ids.cpu().tolist()).issubset(
        set(selected_ids)
    ):
        raise RuntimeError("neutral checkpoint does not expose every sensitive row")

    current_sensitive_logits = learner.cache_logits_preserving_dtype(
        model, tok, sensitive_cases, device, args.batch_size
    )
    current_reference_logits = learner.cache_logits_preserving_dtype(
        model, tok, reference_cases, device, args.batch_size
    )
    sensitive_hidden = core.forward_last_hidden(
        model, tok, sensitive_cases, device, args.batch_size
    ).float()
    reference_hidden = core.forward_last_hidden(
        model, tok, reference_cases, device, args.batch_size
    ).float()
    base_sensitive_logits = safe_load(learner_dir / "base_sensitive_case_logits.pt")
    if base_sensitive_logits.shape != current_sensitive_logits.shape:
        raise RuntimeError("neutral Base logits do not align with MCF sensitive cases")

    probability_payload = safe_load(
        learner_dir / "base_wikipedia_selected_probabilities.pt"
    )
    if [int(value) for value in probability_payload["row_ids"]] != selected_ids:
        raise RuntimeError("Wikipedia probability rows do not match final delta")
    utility_payload = safe_load(Path(neutral_config["utility_cache"]).resolve())
    utility_hidden_all = utility_payload["utility_hidden_states"].float()
    utility_second_moment = utility_payload["second_moment"].float()
    candidate_probabilities = probability_payload["candidate_probabilities"].float()
    train_indices = probability_payload["train_indices"].long()
    guard_indices = probability_payload["guard_indices"].long()
    utility_train_hidden = utility_hidden_all.index_select(0, train_indices).to(device)
    utility_train_probabilities = candidate_probabilities.index_select(
        0, train_indices
    ).to(device)
    utility_guard_hidden = utility_hidden_all.index_select(0, guard_indices).to(device)
    utility_guard_probabilities = candidate_probabilities.index_select(
        0, guard_indices
    ).to(device)

    record_count = len(records)
    zero_residual = torch.zeros(
        (len(selected_ids), sensitive_hidden.shape[1]),
        device=device,
        dtype=torch.float32,
    )
    initial_sensitive_cache = build_sequence_cache(
        current_sensitive_logits,
        sensitive_hidden,
        sensitive_ids,
        sensitive_positions,
        selected_ids,
        record_count=record_count,
        device=device,
    )
    initial_reference_cache = build_sequence_cache(
        current_reference_logits,
        reference_hidden,
        reference_ids,
        reference_positions,
        selected_ids,
        record_count=record_count,
        device=device,
    )
    initial_separation = exact_pairwise_separation(
        initial_sensitive_cache, initial_reference_cache, zero_residual
    )
    initial_pairwise = pairwise_report(
        initial_separation, required_margin=args.direct_fs_margin
    )
    core.write_json(output_dir / "initial_direct_fs_report.json", initial_pairwise)
    failing_records = (
        torch.where(initial_separation < float(args.direct_fs_margin))[0]
        .detach()
        .cpu()
        .tolist()
    )

    internal_args = argparse.Namespace(
        constraint_margin=float(neutral_config["constraint_margin"]),
        min_sensitive_nll_increase=float(neutral_config["min_sensitive_nll_increase"]),
    )
    current_internal = learner.constraint_state_from_logits(
        current_sensitive_logits,
        base_sensitive_logits,
        sensitive_ids.detach().cpu(),
    )
    global_nll = float(internal_args.min_sensitive_nll_increase)
    internal_behavior_floors = torch.maximum(
        torch.full_like(current_internal["sensitive_nll_increase"], global_nll),
        current_internal["sensitive_nll_increase"].float()
        - float(args.protection_nll_tolerance),
    ).to(device)
    internal_solver_targets = torch.maximum(
        internal_behavior_floors + float(args.protected_materialization_buffer),
        torch.full_like(
            internal_behavior_floors, global_nll + float(args.constraint_buffer)
        ),
    )
    internal_margin_solver_target = float(internal_args.constraint_margin) + float(
        args.constraint_buffer
    )
    fs_solver_target = float(args.direct_fs_margin) + float(
        args.direct_fs_solver_buffer
    )
    utility_budgets = {
        key: float(value)
        for key, value in neutral_config["utility_guard_budgets"].items()
    }
    architecture = {
        "method": METHOD,
        "protocol": PROTOCOL,
        "editable_parameters": "sensitive_target_true_lm_head_rows_only",
        "direct_fs_margin": float(args.direct_fs_margin),
        "direct_fs_solver_buffer": float(args.direct_fs_solver_buffer),
        "protection_nll_tolerance": float(args.protection_nll_tolerance),
        "protected_materialization_buffer": float(
            args.protected_materialization_buffer
        ),
        "constraint_buffer": float(args.constraint_buffer),
        "residual_l2_weight": float(args.residual_l2_weight),
        "constraint_context_weight": float(args.constraint_context_weight),
        "rank_ladder": list(rank_ladder),
        "maxiter": int(args.maxiter),
        "ftol": float(args.ftol),
        "constraint_tolerance": float(args.constraint_tolerance),
        "utility_guard_budgets": utility_budgets,
        "source_neutral_architecture_sha256": neutral_config.get(
            "shared_architecture_sha256"
        ),
    }
    architecture_sha256 = sha256_bytes(
        json.dumps(architecture, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )

    if not failing_records:
        learner.save_checkpoint(model, tok, output_dir / "checkpoint")
        torch.save(current_delta_payload, output_dir / "final_total_delta.pt")
        report = official_materialized_pairwise_report(
            model,
            tok,
            records,
            device,
            llama_like=llama_like,
            required_margin=args.direct_fs_margin,
        )
        core.write_json(output_dir / "final_direct_fs_report.json", report)
        core.write_json(
            output_dir / "config_used.json",
            {
                "schema_version": 1,
                "method": METHOD,
                "protocol": PROTOCOL,
                "benchmark_aware": True,
                "target_new_used_for_training_and_checkpoint_selection": True,
                "selection_mode": "identity_already_FS_margin_safe",
                "architecture": architecture,
                "architecture_sha256": architecture_sha256,
                "direct_fs_margin": float(args.direct_fs_margin),
                "official_materialized_direct_fs": report["direct_fs"],
                "source_neutral_learner": str(learner_dir),
                "source_mcf_sha256": manifest.get("source_sha256"),
                "forget_case_ids": manifest.get("sampling", {}).get(
                    "forget_case_ids", []
                ),
            },
        )
        print("Target-aware direct FS was already margin-safe: 100.0")
        return

    failing_tensor = torch.tensor(failing_records, device=device, dtype=torch.long)
    failing_case_mask = (
        sensitive_positions.unsqueeze(1).eq(failing_tensor.unsqueeze(0)).any(dim=1)
    )
    active_ids = sorted(
        set(int(value) for value in sensitive_ids[failing_case_mask].cpu().tolist())
    )
    active_tensor = torch.tensor(active_ids, device=device, dtype=torch.long)
    active_rows_before = (
        output_layer.weight.index_select(0, active_tensor).detach().float()
    )

    direct_cache = learner.build_exact_stage2_direct_cache(
        current_sensitive_logits,
        base_sensitive_logits,
        sensitive_hidden,
        sensitive_ids,
        active_ids,
        device=device,
    )
    sensitive_cache = build_sequence_cache(
        current_sensitive_logits,
        sensitive_hidden,
        sensitive_ids,
        sensitive_positions,
        active_ids,
        record_count=record_count,
        device=device,
    )
    reference_cache = build_sequence_cache(
        current_reference_logits,
        reference_hidden,
        reference_ids,
        reference_positions,
        active_ids,
        record_count=record_count,
        device=device,
    )

    rank_candidates: List[Dict[str, Any]] = []
    for rank in rank_ladder:
        bases, basis_report = build_target_aware_bases(
            sensitive_hidden,
            sensitive_ids,
            sensitive_positions,
            reference_hidden,
            failing_records,
            utility_second_moment,
            active_ids=active_ids,
            rank_cap=rank,
            relative_eps=float(
                neutral_config["shared_architecture_parameters"]["contrastive_eps"]
            ),
            constraint_context_weight=args.constraint_context_weight,
        )
        residual, history, solver_report = solve_rank(
            args=args,
            rank=rank,
            row_bases=bases,
            active_ids=active_ids,
            selected_ids=selected_ids,
            current_total_delta=current_total_delta,
            direct_cache=direct_cache,
            internal_nll_solver_targets=internal_solver_targets,
            internal_margin_solver_target=internal_margin_solver_target,
            sensitive_cache=sensitive_cache,
            reference_cache=reference_cache,
            fs_solver_target=fs_solver_target,
            utility_hidden=utility_train_hidden,
            utility_probabilities=utility_train_probabilities,
            utility_budgets=utility_budgets,
        )
        torch.save(
            {"row_ids": active_ids, "delta": residual.cpu(), "rank": rank},
            output_dir / f"rank{rank}_residual.pt",
        )
        core.write_json(output_dir / f"rank{rank}_basis_report.json", basis_report)
        core.write_json(output_dir / f"rank{rank}_solver_history.json", history)

        with learner.temporary_materialized_output_delta(
            output_layer, active_ids, residual
        ):
            actual_rows = (
                output_layer.weight.index_select(0, active_tensor).detach().float()
            )
            actual_residual = actual_rows - active_rows_before
            combined_delta = learner.total_delta_with_residual(
                current_total_delta,
                selected_ids,
                actual_residual,
                active_ids,
            )
            official_report = official_materialized_pairwise_report(
                model,
                tok,
                records,
                device,
                llama_like=llama_like,
                required_margin=args.direct_fs_margin,
            )
            internal_state = shared.evaluate_shared_constraints(
                model,
                tok,
                sensitive_cases,
                base_sensitive_logits,
                llama_like=llama_like,
                device=device,
                batch_size=args.batch_size,
            )
            internal_nll_shortfall = torch.relu(
                internal_behavior_floors.cpu()
                - internal_state["sensitive_nll_increase"].float().cpu()
            )
            internal_margin_shortfall = torch.relu(
                float(internal_args.constraint_margin)
                - internal_state["logit_margin"].float().cpu()
            )
            utility_report = learner.utility_kl_report(
                combined_delta,
                utility_guard_hidden,
                utility_guard_probabilities,
                device=device,
                batch_size=args.utility_batch_size,
            )
        utility_checks = {
            "mean": utility_report["utility_kl_mean"] <= utility_budgets["mean"],
            "p95": utility_report["utility_kl_p95"] <= utility_budgets["p95"],
            "max": utility_report["utility_kl_max"] <= utility_budgets["max"],
            "norm": float(combined_delta.norm().cpu())
            <= utility_budgets["total_delta_norm"],
        }
        materialized = {
            "rank": int(rank),
            **official_report,
            **utility_report,
            "internal_nll_floor_violations": int(
                (internal_nll_shortfall > 0).sum().item()
            ),
            "internal_nll_floor_shortfall_sum": float(
                internal_nll_shortfall.sum().item()
            ),
            "internal_margin_violations": int(
                (internal_margin_shortfall > 0).sum().item()
            ),
            "total_delta_norm": float(combined_delta.norm().cpu()),
            "utility_guard_checks": utility_checks,
            "utility_safe": bool(all(utility_checks.values())),
            "continuous_feasible": bool(solver_report["continuous_feasible"]),
        }
        materialized["feasible"] = bool(
            materialized["continuous_feasible"]
            and int(materialized["direct_fs_margin_failures"]) == 0
            and int(materialized["internal_nll_floor_violations"]) == 0
            and int(materialized["internal_margin_violations"]) == 0
            and materialized["utility_safe"]
        )
        core.write_json(
            output_dir / f"rank{rank}_materialized_report.json", materialized
        )
        rank_candidates.append(
            {
                "rank": rank,
                "residual": residual,
                "solver": solver_report,
                "materialized": materialized,
            }
        )

    feasible = [row for row in rank_candidates if row["materialized"]["feasible"]]
    if not feasible:
        core.write_json(
            output_dir / "infeasible.json",
            {
                "method": METHOD,
                "protocol": PROTOCOL,
                "initial": initial_pairwise,
                "failing_record_positions": failing_records,
                "active_row_ids": active_ids,
                "rank_attempts": [
                    {
                        "rank": row["rank"],
                        "solver": row["solver"],
                        "materialized": row["materialized"],
                    }
                    for row in rank_candidates
                ],
                "reason": (
                    "no checkpoint-dtype candidate achieved every direct FS margin "
                    "while preserving internal and Wikipedia utility guards"
                ),
            },
        )
        raise RuntimeError(
            "Target-aware repair could not guarantee 100 FS under locked utility guards"
        )

    selected = min(
        feasible,
        key=lambda row: (
            float(row["materialized"]["utility_kl_mean"]),
            float(row["materialized"]["utility_kl_p95"]),
            float(row["materialized"]["total_delta_norm"]),
            int(row["rank"]),
        ),
    )
    core.write_json(
        output_dir / "selected_materialized_report.json", selected["materialized"]
    )
    core.materialize_output_delta(output_layer, active_ids, selected["residual"])
    active_rows_after = (
        output_layer.weight.index_select(0, active_tensor).detach().float()
    )
    actual_residual = active_rows_after - active_rows_before
    final_total_delta = learner.total_delta_with_residual(
        current_total_delta,
        selected_ids,
        actual_residual,
        active_ids,
    )
    final_report = official_materialized_pairwise_report(
        model,
        tok,
        records,
        device,
        llama_like=llama_like,
        required_margin=args.direct_fs_margin,
    )
    if final_report["direct_fs"] != 100.0 or final_report["direct_fs_margin_failures"]:
        raise RuntimeError(
            "Final checkpoint failed the exact official direct-FS guarantee"
        )

    learner.save_checkpoint(model, tok, output_dir / "checkpoint")
    torch.save(
        {"row_ids": selected_ids, "delta": final_total_delta.detach().cpu()},
        output_dir / "final_total_delta.pt",
    )
    core.write_json(output_dir / "final_direct_fs_report.json", final_report)
    core.write_json(
        output_dir / "config_used.json",
        {
            "schema_version": 1,
            "method": METHOD,
            "protocol": PROTOCOL,
            "benchmark_aware": True,
            "target_new_used_for_training_and_checkpoint_selection": True,
            "paraphrases_used": False,
            "neighborhood_prompts_used": False,
            "benchmark_retain_examples_used": 0,
            "source_neutral_learner": str(learner_dir),
            "source_neutral_method": neutral_config.get("method"),
            "source_neutral_architecture_sha256": neutral_config.get(
                "shared_architecture_sha256"
            ),
            "architecture": architecture,
            "architecture_sha256": architecture_sha256,
            "direct_fs_requirement": "NLL(target_true) > NLL(target_new)",
            "direct_fs_margin": float(args.direct_fs_margin),
            "direct_fs_solver_target": fs_solver_target,
            "selected_rank": int(selected["rank"]),
            "selected_solver_report": selected["solver"],
            "selected_materialized_report": selected["materialized"],
            "active_row_ids": active_ids,
            "rank_ladder": list(rank_ladder),
            "utility_guard_budgets": utility_budgets,
            "official_materialized_direct_fs": final_report["direct_fs"],
            "official_materialized_minimum_separation": final_report[
                "minimum_direct_fs_separation"
            ],
            "source_mcf_sha256": manifest.get("source_sha256"),
            "forget_case_ids": manifest.get("sampling", {}).get("forget_case_ids", []),
        },
    )
    print("Target-aware exact direct-FS repair complete:", output_dir)
    print("Official materialized direct FS:", final_report["direct_fs"])
    print(
        "Minimum direct NLL separation:",
        final_report["minimum_direct_fs_separation"],
    )


if __name__ == "__main__":
    main()
