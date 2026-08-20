#!/usr/bin/env python3
"""Guarded, dataset-adapter-neutral two-stage SURE-LM learner.

The learner consumes a canonical direct-forget JSON plus an adapter contract in
the split manifest. MCF, ZsRE, and future datasets therefore share exactly the
same Stage-1 optimizer, Stage-2 constrained solver, rank ladder, and safety rules.

Only sparse sensitive-token rows of an untied LM head are editable. The Base
transformer and input embeddings remain frozen. A disjoint Wikipedia cache is
used in two ways: its full hidden second moment defines the contrastive basis,
and its predictor-state/Base-partition reservoir is split before selecting
per-edited-token high-Base-probability train and checkpoint-guard contexts.
Stage 2 partitions the direct cases into exact Stage-1 failures and successes,
then solves a minimum-utility-cost sparse residual with every repair and
success-preservation requirement represented as a hard behavioral constraint.
Since the transformer is frozen, all Stage-2 direct NLLs, margins, and sparse
Wikipedia KL values are evaluated from cached predictor states without model
forward passes. No null-space projection or weighted repair/protection tradeoff
is used.
No benchmark retain example, replacement target, paraphrase, locality probe,
or PPL text is visible to training or selection.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

import build_sure_wikipedia_stats as wikipedia
import gagd_compare as gagd
import sure_canonical_core as core
import sure_context_projection as context
import sure_shared_suppression as shared


METHOD = "SURE-LM-exact-constrained-residual-stage2-rank2to4"
PROTOCOL = "sure_exact_constrained_residual_stage2_v5"
RANK_LADDER = (2, 4)
DEFAULT_UTILITY_TOKEN_TOPK_PER_ROW = 128
DEFAULT_UTILITY_UNIFORM_PROMPT_COUNT = 1_024
DEFAULT_UTILITY_POOL_SEED = 1
DEFAULT_CANDIDATE_SCALES = (
    "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,"
    ".046875,.03125,.015625,.0078125,0"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset adapter name recorded in the canonical split manifest",
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--training-visible-path", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--utility-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--forget-num", type=int, default=50)
    parser.add_argument(
        "--utility-sample-size",
        type=int,
        default=wikipedia.DEFAULT_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--utility-prompt-count",
        type=int,
        default=wikipedia.DEFAULT_UTILITY_PROMPT_COUNT,
    )
    parser.add_argument(
        "--utility-token-topk-per-row",
        type=int,
        default=DEFAULT_UTILITY_TOKEN_TOPK_PER_ROW,
        help="High-Base-probability Wikipedia contexts retained per edited row",
    )
    parser.add_argument(
        "--utility-uniform-prompt-count",
        type=int,
        default=DEFAULT_UTILITY_UNIFORM_PROMPT_COUNT,
        help="Broad Wikipedia anchors added to each token-conditioned pool",
    )
    parser.add_argument(
        "--utility-pool-seed",
        type=int,
        default=DEFAULT_UTILITY_POOL_SEED,
        help="Dataset-independent seed for disjoint utility train/guard pools",
    )
    parser.add_argument("--stage1-steps", type=int, default=600)
    parser.add_argument("--stage1-batch-size", type=int, default=1)
    parser.add_argument("--stage1-lr", type=float, default=5e-3)
    parser.add_argument(
        "--stage2-maxiter",
        type=int,
        default=500,
        help="Maximum SLSQP iterations for each deterministic constrained solve",
    )
    parser.add_argument(
        "--stage2-ftol",
        type=float,
        default=1e-9,
        help="SLSQP objective/convergence tolerance",
    )
    parser.add_argument(
        "--stage2-constraint-tolerance",
        type=float,
        default=1e-5,
        help="Maximum accepted negative hard-constraint slack",
    )
    parser.add_argument(
        "--stage2-constraint-buffer",
        type=float,
        default=0.05,
        help="Extra continuous-solver clearance before checkpoint-dtype verification",
    )
    parser.add_argument(
        "--stage2-residual-l2-weight",
        type=float,
        default=1e-4,
        help="Residual squared-norm coefficient in the minimum-utility objective",
    )
    parser.add_argument(
        "--stage2-constraint-basis-weight",
        type=float,
        default=0.05,
        help=(
            "Rank-4 fallback weight for all training-visible direct constraint "
            "contexts; rank 2 remains repair-only"
        ),
    )
    parser.add_argument(
        "--stage2-restarts",
        type=int,
        default=2,
        help="Deterministic zero/repair-directed starts per Stage-2 rank",
    )
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--utility-train-batch-size", type=int, default=128)
    parser.add_argument("--utility-eval-batch-size", type=int, default=512)

    parser.add_argument("--direct-constraint-weight", type=float, default=100.0)
    parser.add_argument("--gd-weight", type=float, default=1.0)
    parser.add_argument("--utility-kl-weight", type=float, default=1.0)
    parser.add_argument(
        "--stage2-protection-nll-tolerance",
        type=float,
        default=0.05,
        help=(
            "Maximum Stage-2 decrease allowed from each protected case's "
            "Stage-1 sensitive-NLL increase; the global NLL guard remains a floor"
        ),
    )
    parser.add_argument("--contrastive-eps", type=float, default=1e-3)
    parser.add_argument("--constraint-margin", type=float, default=0.05)
    parser.add_argument("--min-sensitive-nll-increase", type=float, default=4.0)
    parser.add_argument("--utility-kl-mean-budget", type=float, default=0.01)
    parser.add_argument("--utility-kl-p95-budget", type=float, default=0.05)
    parser.add_argument("--utility-kl-max-budget", type=float, default=0.5)
    parser.add_argument("--max-total-delta-norm", type=float, default=1.5)
    parser.add_argument("--rank-ladder", default="2,4")
    parser.add_argument(
        "--candidate-scales",
        default=DEFAULT_CANDIDATE_SCALES,
        help="Stage-1 shrink-only materialized scale frontier",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device-map", choices=("single", "auto"), default="single")
    return parser.parse_args()


def parse_rank_ladder(text: str) -> Tuple[int, ...]:
    values: List[int] = []
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("rank ladder values must be positive")
        if value not in values:
            values.append(value)
    if tuple(values) != RANK_LADDER:
        raise ValueError(f"guarded SURE locks the shared rank ladder to {RANK_LADDER}")
    return tuple(values)


def validate_args(
    args: argparse.Namespace,
) -> Tuple[List[float], Tuple[int, ...]]:
    if args.forget_num <= 0:
        raise ValueError("forget-num must be positive")
    if args.utility_sample_size != wikipedia.DEFAULT_SAMPLE_SIZE:
        raise ValueError(
            "The guarded architecture requires a 100,000-document Wikipedia request"
        )
    if args.utility_prompt_count != wikipedia.DEFAULT_UTILITY_PROMPT_COUNT:
        raise ValueError(
            "The guarded architecture requires exactly "
            f"{wikipedia.DEFAULT_UTILITY_PROMPT_COUNT} requested utility prompts"
        )
    positive = {
        "stage1_steps": args.stage1_steps,
        "stage1_batch_size": args.stage1_batch_size,
        "stage1_lr": args.stage1_lr,
        "stage2_maxiter": args.stage2_maxiter,
        "cache_batch_size": args.cache_batch_size,
        "utility_train_batch_size": args.utility_train_batch_size,
        "utility_eval_batch_size": args.utility_eval_batch_size,
        "utility_token_topk_per_row": args.utility_token_topk_per_row,
        "direct_constraint_weight": args.direct_constraint_weight,
        "utility_kl_weight": args.utility_kl_weight,
        "contrastive_eps": args.contrastive_eps,
        "utility_kl_mean_budget": args.utility_kl_mean_budget,
        "utility_kl_p95_budget": args.utility_kl_p95_budget,
        "utility_kl_max_budget": args.utility_kl_max_budget,
        "max_total_delta_norm": args.max_total_delta_norm,
    }
    for name, value in positive.items():
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.gd_weight) or args.gd_weight < 0:
        raise ValueError("gd-weight must be finite and non-negative")
    if int(args.utility_uniform_prompt_count) < 0:
        raise ValueError("utility-uniform-prompt-count must be non-negative")
    if not math.isfinite(args.grad_clip) or args.grad_clip < 0:
        raise ValueError("grad-clip must be finite and non-negative")
    nonnegative = {
        "stage2_constraint_tolerance": args.stage2_constraint_tolerance,
        "stage2_constraint_buffer": args.stage2_constraint_buffer,
        "stage2_residual_l2_weight": args.stage2_residual_l2_weight,
        "stage2_constraint_basis_weight": args.stage2_constraint_basis_weight,
    }
    for name, value in nonnegative.items():
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    if not math.isfinite(float(args.stage2_ftol)) or args.stage2_ftol <= 0:
        raise ValueError("stage2-ftol must be finite and positive")
    if int(args.stage2_restarts) not in (1, 2):
        raise ValueError("stage2-restarts must be either 1 or 2")
    if not math.isfinite(args.constraint_margin):
        raise ValueError("constraint-margin must be finite")
    if (
        not math.isfinite(args.min_sensitive_nll_increase)
        or args.min_sensitive_nll_increase < 0
    ):
        raise ValueError("min-sensitive-nll-increase must be finite and non-negative")
    if (
        not math.isfinite(args.stage2_protection_nll_tolerance)
        or args.stage2_protection_nll_tolerance < 0
    ):
        raise ValueError(
            "stage2-protection-nll-tolerance must be finite and non-negative"
        )
    stage1_scales = core.parse_scales(args.candidate_scales)
    if 0.0 not in stage1_scales or 1.0 not in stage1_scales:
        raise ValueError("candidate-scales must include both 0 and 1")
    if any(scale > 1.0 for scale in stage1_scales):
        raise ValueError("Stage-1 candidate-scales must not exceed 1.0")
    return stage1_scales, parse_rank_ladder(args.rank_ladder)


def architecture_signature_payload(
    args: argparse.Namespace,
    stage1_scales: Sequence[float],
    rank_ladder: Sequence[int],
) -> Dict[str, Any]:
    """Return dataset/seed-independent parameters that define the learner."""
    return {
        "method": METHOD,
        "protocol": PROTOCOL,
        "utility_protocol": wikipedia.UTILITY_PROTOCOL,
        "forget_num": int(args.forget_num),
        "utility_sample_size": int(args.utility_sample_size),
        "utility_prompt_count": int(args.utility_prompt_count),
        "utility_token_topk_per_row": int(args.utility_token_topk_per_row),
        "utility_uniform_prompt_count": int(args.utility_uniform_prompt_count),
        "utility_pool_seed": int(args.utility_pool_seed),
        "rank_ladder": [int(value) for value in rank_ladder],
        "stage1_steps": int(args.stage1_steps),
        "stage1_batch_size": int(args.stage1_batch_size),
        "stage1_lr": float(args.stage1_lr),
        "stage2_solver": "scipy_slsqp_exact_cached_sparse_constraints",
        "stage2_maxiter": int(args.stage2_maxiter),
        "stage2_ftol": float(args.stage2_ftol),
        "stage2_constraint_tolerance": float(args.stage2_constraint_tolerance),
        "stage2_constraint_buffer": float(args.stage2_constraint_buffer),
        "stage2_residual_l2_weight": float(args.stage2_residual_l2_weight),
        "stage2_constraint_basis_weight": float(args.stage2_constraint_basis_weight),
        "stage2_restarts": int(args.stage2_restarts),
        "stage2_direct_case_batching": "all_repair_and_protected_constraints_exact",
        "utility_train_batch_size": int(args.utility_train_batch_size),
        "utility_eval_batch_size": int(args.utility_eval_batch_size),
        "direct_constraint_weight": float(args.direct_constraint_weight),
        "gd_weight": float(args.gd_weight),
        "utility_kl_weight": float(args.utility_kl_weight),
        "stage2_protection_nll_tolerance": float(args.stage2_protection_nll_tolerance),
        "contrastive_eps": float(args.contrastive_eps),
        "constraint_margin": float(args.constraint_margin),
        "min_sensitive_nll_increase": float(args.min_sensitive_nll_increase),
        "utility_kl_mean_budget": float(args.utility_kl_mean_budget),
        "utility_kl_p95_budget": float(args.utility_kl_p95_budget),
        "utility_kl_max_budget": float(args.utility_kl_max_budget),
        "max_total_delta_norm": float(args.max_total_delta_norm),
        "stage1_candidate_scales": [float(value) for value in stage1_scales],
        "grad_clip": float(args.grad_clip),
        "dtype": str(args.dtype),
    }


def architecture_signature_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _safe_torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def adapter_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    contract = manifest.get("learner_adapter_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("split manifest lacks learner_adapter_contract")
    field = str(contract.get("sensitive_answer_field", "")).strip()
    forbidden = contract.get("forbidden_answer_fields", [])
    if not field:
        raise RuntimeError("adapter contract lacks sensitive_answer_field")
    if not isinstance(forbidden, list) or any(
        not isinstance(value, str) or not value for value in forbidden
    ):
        raise RuntimeError("adapter contract has invalid forbidden_answer_fields")
    if field in forbidden:
        raise RuntimeError("sensitive field cannot also be forbidden")
    return {
        "sensitive_answer_field": field,
        "forbidden_answer_fields": list(forbidden),
    }


def load_locked(
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    forget_path = Path(args.training_visible_path).resolve()
    manifest_path = Path(args.split_manifest).resolve()
    forget_bytes = forget_path.read_bytes()
    forget_records = json.loads(forget_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if not isinstance(forget_records, list) or len(forget_records) != args.forget_num:
        raise RuntimeError(f"Expected exactly {args.forget_num} direct forget records")
    if manifest.get("protocol") != PROTOCOL:
        raise RuntimeError(
            f"Expected split protocol {PROTOCOL}; got {manifest.get('protocol')!r}"
        )
    if manifest.get("dataset") != args.dataset:
        raise RuntimeError("split manifest dataset mismatch")
    if int(manifest.get("seed", -1)) != args.seed:
        raise RuntimeError("split manifest seed mismatch")
    if manifest.get("training_visible_forget_sha256") != sha256_bytes(forget_bytes):
        raise RuntimeError("training-visible forget hash mismatch")
    sampling = manifest.get("sampling", {})
    if int(sampling.get("forget_num", -1)) != args.forget_num:
        raise RuntimeError("split manifest forget count mismatch")
    if int(sampling.get("benchmark_retain_train_num", -1)) != 0:
        raise RuntimeError("guarded SURE forbids benchmark retain training examples")
    if "training_visible_retain" in manifest:
        raise RuntimeError("split manifest unexpectedly exposes retain training")

    contract = adapter_contract(manifest)
    sensitive_field = contract["sensitive_answer_field"]
    forbidden_fields = contract["forbidden_answer_fields"]
    for position, record in enumerate(forget_records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"forget record {position} is not an object")
        for field in (
            "paraphrase_prompts",
            "neighborhood_prompts",
            "generation_prompts",
            "attribute_prompts",
        ):
            if record.get(field):
                raise RuntimeError(f"forget record {position} exposes held-out {field}")
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise RuntimeError(f"forget record {position} lacks requested_rewrite")
        target = rewrite.get(sensitive_field)
        if not isinstance(target, Mapping) or not target.get("str"):
            raise RuntimeError(
                f"forget record {position} lacks sensitive {sensitive_field}"
            )
        for forbidden in forbidden_fields:
            if forbidden in rewrite:
                raise RuntimeError(
                    f"forget record {position} exposes forbidden {forbidden}"
                )
    return forget_records, manifest


def load_utility_cache(
    path: Path,
    *,
    expected_sample_size: int,
    expected_prompt_count: int,
    expected_hidden_size: int,
    expected_model_probe: str,
    expected_tokenizer_probe: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
    payload = _safe_torch_load(path)
    if not isinstance(payload, Mapping):
        raise ValueError("utility cache must be a mapping")
    moment = payload.get("second_moment")
    hidden = payload.get("utility_hidden_states")
    logsumexp = payload.get("base_logsumexp")
    metadata = payload.get("metadata")
    if not all(
        isinstance(value, torch.Tensor) for value in (moment, hidden, logsumexp)
    ):
        raise ValueError("utility cache lacks moment/hidden/Base-partition tensors")
    if not isinstance(metadata, Mapping):
        raise ValueError("utility cache lacks metadata")
    metadata = dict(metadata)
    if int(metadata.get("schema_version", -1)) != wikipedia.CACHE_SCHEMA_VERSION:
        raise ValueError("utility cache schema mismatch")
    if metadata.get("protocol") != wikipedia.UTILITY_PROTOCOL:
        raise ValueError("utility cache protocol mismatch")
    if int(metadata.get("requested_document_sample_size", -1)) != expected_sample_size:
        raise ValueError("utility cache requested document sample size mismatch")
    if int(metadata.get("requested_utility_prompt_count", -1)) != expected_prompt_count:
        raise ValueError("utility cache requested prompt count mismatch")
    actual_prompts = int(metadata.get("actual_utility_prompt_count", 0))
    if actual_prompts <= 0 or actual_prompts > expected_prompt_count:
        raise ValueError("utility cache contains an invalid prompt sample")
    if int(metadata.get("actual_document_sample_size", 0)) <= 0:
        raise ValueError("utility cache contains no sampled documents")
    if int(metadata.get("predictor_hidden_state_count", 0)) <= 0:
        raise ValueError("utility cache contains no predictor hidden states")
    if int(metadata.get("hidden_size", -1)) != expected_hidden_size:
        raise ValueError("utility cache hidden size mismatch")
    if metadata.get("model_probe_sha256") != expected_model_probe:
        raise ValueError("utility cache was built from different model weights")
    if metadata.get("tokenizer_probe_sha256") != expected_tokenizer_probe:
        raise ValueError("utility cache was built with a different tokenizer")
    if int(metadata.get("benchmark_examples_seen", -1)) != 0:
        raise ValueError("utility cache unexpectedly contains benchmark examples")
    if int(metadata.get("benchmark_retain_examples_seen", -1)) != 0:
        raise ValueError("utility cache unexpectedly contains benchmark retain data")
    if int(metadata.get("heldout_benchmark_probes_seen", -1)) != 0:
        raise ValueError("utility cache unexpectedly contains held-out probes")
    if int(metadata.get("excluded_prefix_document_count", -1)) < 20:
        raise ValueError("utility cache does not lock away the PPL prefix")

    moment = moment.detach().cpu().float().contiguous()
    hidden = hidden.detach().cpu().contiguous()
    logsumexp = logsumexp.detach().cpu().float().contiguous()
    if moment.shape != (expected_hidden_size, expected_hidden_size):
        raise ValueError("utility second moment has the wrong shape")
    if hidden.ndim != 2 or hidden.shape != (actual_prompts, expected_hidden_size):
        raise ValueError("utility hidden-state sample has the wrong shape")
    if logsumexp.shape != (actual_prompts,):
        raise ValueError("utility Base partitions have the wrong shape")
    if not torch.isfinite(moment).all() or not torch.isfinite(hidden.float()).all():
        raise ValueError("utility cache contains non-finite hidden statistics")
    if not torch.isfinite(logsumexp).all():
        raise ValueError("utility cache contains non-finite Base partitions")
    if not torch.allclose(moment, moment.transpose(0, 1), rtol=1e-4, atol=1e-5):
        raise ValueError("utility second moment is not symmetric")
    checks = (
        ("second_moment_sha256", moment, "second-moment"),
        ("utility_hidden_sha256", hidden, "utility-hidden"),
        ("base_logsumexp_sha256", logsumexp, "Base-partition"),
    )
    for key, tensor, label in checks:
        expected = metadata.get(key)
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"utility cache lacks {label} checksum")
        if wikipedia.sha256_tensor(tensor) != expected:
            raise ValueError(f"utility {label} checksum mismatch")
    return moment, hidden, logsumexp, metadata


def regularized_utility_cholesky(
    utility_second_moment: torch.Tensor,
    *,
    relative_eps: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    if relative_eps <= 0:
        raise ValueError("contrastive epsilon must be positive")
    utility = utility_second_moment.to(device=device, dtype=torch.float32)
    if utility.ndim != 2 or utility.shape[0] != utility.shape[1]:
        raise ValueError("utility second moment must be square")
    utility = 0.5 * (utility + utility.transpose(0, 1))
    dimension = int(utility.shape[0])
    average = (torch.trace(utility) / float(dimension)).abs().clamp_min(1e-6)
    base_ridge = float(relative_eps) * average
    identity = torch.eye(dimension, device=device, dtype=torch.float32)
    ridge = base_ridge
    last_error = None
    for attempt in range(8):
        try:
            chol = torch.linalg.cholesky(utility + ridge * identity)
            return chol, {
                "relative_eps": float(relative_eps),
                "average_utility_diagonal": float(average.detach().cpu()),
                "base_ridge": float(base_ridge.detach().cpu()),
                "final_ridge": float(ridge.detach().cpu()),
                "cholesky_attempts": attempt + 1,
            }
        except RuntimeError as error:
            last_error = error
            ridge = ridge * 10.0
    raise RuntimeError(
        "could not regularize Wikipedia utility covariance"
    ) from last_error


@torch.no_grad()
def build_contrastive_bases_from_second_moment(
    forget_hidden: torch.Tensor,
    forget_tids: torch.Tensor,
    utility_second_moment: torch.Tensor,
    *,
    requested_ids: Sequence[int],
    rank_cap: int,
    relative_eps: float,
    utility_cholesky: torch.Tensor | None = None,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    if rank_cap not in RANK_LADDER:
        raise ValueError(f"rank cap must come from the shared ladder {RANK_LADDER}")
    if forget_hidden.ndim != 2 or forget_hidden.shape[0] != forget_tids.shape[0]:
        raise ValueError("forget hidden states and token ids do not align")
    hidden_size = int(forget_hidden.shape[1])
    if utility_second_moment.shape != (hidden_size, hidden_size):
        raise ValueError("utility second moment and forget hidden size differ")
    if utility_cholesky is None:
        utility_cholesky, _ = regularized_utility_cholesky(
            utility_second_moment,
            relative_eps=relative_eps,
            device=forget_hidden.device,
        )
    chol = utility_cholesky.to(device=forget_hidden.device, dtype=torch.float32)
    token_ids = forget_tids.to(device=forget_hidden.device)
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for token_id in [int(value) for value in requested_ids]:
        rows = forget_hidden[token_ids.eq(token_id)].float()
        if rows.numel() == 0:
            raise RuntimeError(f"Sensitive token {token_id} has no forget contexts")

        # For C_F = H^T H / n and C_U = L L^T, the non-zero generalized
        # eigenvectors are v = L^{-T} q where q is a right singular vector of
        # H L^{-T}. Solving in the full hidden space is essential: restricting
        # v to span(H) incorrectly returns h rather than C_U^{-1}h for one case.
        whitened_rows = torch.linalg.solve_triangular(
            chol,
            rows.transpose(0, 1),
            upper=False,
        ).transpose(0, 1) / math.sqrt(float(rows.shape[0]))
        _, singular_values, right = torch.linalg.svd(whitened_rows, full_matrices=False)
        tolerance = (
            max(whitened_rows.shape)
            * torch.finfo(torch.float32).eps
            * singular_values.max().clamp_min(1.0)
        )
        numerical_rank = int((singular_values > tolerance).sum().item())
        take = min(int(rank_cap), numerical_rank)
        if take <= 0:
            raise RuntimeError(f"Sensitive token {token_id} has zero contrastive rank")
        whitened_directions = right[:take]
        raw_directions = torch.linalg.solve_triangular(
            chol.transpose(0, 1),
            whitened_directions.transpose(0, 1),
            upper=True,
        ).transpose(0, 1)
        basis = core.orthonormal_row_basis(raw_directions, max_rank=take).float()
        if basis.ndim != 2 or basis.shape[0] == 0:
            raise RuntimeError(f"Sensitive token {token_id} has zero basis")
        bases.append(basis.detach().contiguous())
        reports.append(
            {
                "token_id": token_id,
                "forget_context_count": int(rows.shape[0]),
                "forget_context_rank": numerical_rank,
                "requested_contrastive_rank": int(rank_cap),
                "actual_contrastive_rank": int(basis.shape[0]),
                "top_generalized_eigenvalues": [
                    float(value)
                    for value in singular_values[:take].square().detach().cpu().tolist()
                ],
                "relative_eps": float(relative_eps),
                "utility_covariance": "fixed_external_wikipedia_second_moment",
                "generalized_eigen_solver": "full_space_cholesky_whitened_low_rank_svd",
                "domain": "full_lm_head_hidden_space",
            }
        )
    return bases, reports


@torch.no_grad()
def build_constraint_aware_stage2_bases(
    active_hidden: torch.Tensor,
    active_tids: torch.Tensor,
    all_direct_hidden: torch.Tensor,
    utility_second_moment: torch.Tensor,
    *,
    requested_ids: Sequence[int],
    rank_cap: int,
    relative_eps: float,
    constraint_context_weight: float,
    utility_cholesky: torch.Tensor | None = None,
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """Build repair-first bases with rank-4 constraint-context capacity.

    Rank 2 intentionally reproduces the repair-only Stage-2 geometry.  At the
    rank-4 fallback, the generalized forget covariance is augmented by a small
    covariance contribution from *all training-visible direct cases*.  These
    extra directions let the constrained solver distinguish active and
    protected contexts that share an LM-head row.  They are capacity, not a
    projection: every behavioral preservation condition remains an explicit
    inequality in the solver.
    """
    if rank_cap not in RANK_LADDER:
        raise ValueError(f"rank cap must come from the shared ladder {RANK_LADDER}")
    if active_hidden.ndim != 2 or active_hidden.shape[0] != active_tids.numel():
        raise ValueError("active hidden states and token ids do not align")
    if all_direct_hidden.ndim != 2 or all_direct_hidden.shape[1] != active_hidden.shape[1]:
        raise ValueError("all-direct and active hidden states do not align")
    if constraint_context_weight < 0 or not math.isfinite(constraint_context_weight):
        raise ValueError("constraint-context weight must be finite and non-negative")
    if rank_cap == RANK_LADDER[0] or constraint_context_weight == 0:
        bases, reports = build_contrastive_bases_from_second_moment(
            active_hidden,
            active_tids,
            utility_second_moment,
            requested_ids=requested_ids,
            rank_cap=rank_cap,
            relative_eps=relative_eps,
            utility_cholesky=utility_cholesky,
        )
        for report in reports:
            report["stage2_basis_protocol"] = "repair_only"
            report["constraint_context_weight"] = 0.0
            report["all_direct_context_count"] = int(all_direct_hidden.shape[0])
        return bases, reports

    hidden_size = int(active_hidden.shape[1])
    if utility_second_moment.shape != (hidden_size, hidden_size):
        raise ValueError("utility second moment and direct hidden size differ")
    if utility_cholesky is None:
        utility_cholesky, _ = regularized_utility_cholesky(
            utility_second_moment,
            relative_eps=relative_eps,
            device=active_hidden.device,
        )
    chol = utility_cholesky.to(device=active_hidden.device, dtype=torch.float32)
    tids = active_tids.to(device=active_hidden.device, dtype=torch.long)
    direct = all_direct_hidden.to(device=active_hidden.device, dtype=torch.float32)
    whitened_direct = torch.linalg.solve_triangular(
        chol,
        direct.transpose(0, 1),
        upper=False,
    ).transpose(0, 1) / math.sqrt(float(direct.shape[0]))
    bases: List[torch.Tensor] = []
    reports: List[Dict[str, Any]] = []
    for token_id in [int(value) for value in requested_ids]:
        repair = active_hidden[tids.eq(token_id)].float()
        if repair.numel() == 0:
            raise RuntimeError(f"Sensitive token {token_id} has no active repair contexts")

        # This row matrix has covariance C_A + alpha*C_D.  Repair states retain
        # unit weight while direct constraint states supply only fallback
        # context-selective capacity.
        whitened_repair = torch.linalg.solve_triangular(
            chol,
            repair.transpose(0, 1),
            upper=False,
        ).transpose(0, 1) / math.sqrt(float(repair.shape[0]))
        whitened = torch.cat(
            (
                whitened_repair,
                math.sqrt(float(constraint_context_weight))
                * whitened_direct,
            ),
            dim=0,
        )
        _, singular_values, right = torch.linalg.svd(whitened, full_matrices=False)
        tolerance = (
            max(whitened.shape)
            * torch.finfo(torch.float32).eps
            * singular_values.max().clamp_min(1.0)
        )
        numerical_rank = int((singular_values > tolerance).sum().item())
        take = min(int(rank_cap), numerical_rank)
        if take <= 0:
            raise RuntimeError(f"Sensitive token {token_id} has zero Stage-2 rank")
        raw = torch.linalg.solve_triangular(
            chol.transpose(0, 1),
            right[:take].transpose(0, 1),
            upper=True,
        ).transpose(0, 1)
        basis = core.orthonormal_row_basis(raw, max_rank=take).float()
        bases.append(basis.detach().contiguous())
        reports.append(
            {
                "token_id": token_id,
                "forget_context_count": int(repair.shape[0]),
                "forget_context_rank": int(torch.linalg.matrix_rank(repair).item()),
                "requested_contrastive_rank": int(rank_cap),
                "actual_contrastive_rank": int(basis.shape[0]),
                "top_generalized_eigenvalues": [
                    float(value)
                    for value in singular_values[:take].square().detach().cpu().tolist()
                ],
                "relative_eps": float(relative_eps),
                "utility_covariance": "fixed_external_wikipedia_second_moment",
                "generalized_eigen_solver": "full_space_cholesky_whitened_low_rank_svd",
                "domain": "full_lm_head_hidden_space",
                "stage2_basis_protocol": "repair_plus_direct_constraint_contexts",
                "constraint_context_weight": float(constraint_context_weight),
                "all_direct_context_count": int(direct.shape[0]),
            }
        )
    return bases, reports


def total_delta_with_residual(
    stage1_delta: torch.Tensor,
    stage1_ids: Sequence[int],
    residual_delta: torch.Tensor,
    residual_ids: Sequence[int],
) -> torch.Tensor:
    if stage1_delta.ndim != 2 or residual_delta.ndim != 2:
        raise ValueError("stage deltas must be rank-2")
    positions = {int(token_id): index for index, token_id in enumerate(stage1_ids)}
    mapped: List[int] = []
    for token_id in residual_ids:
        if int(token_id) not in positions:
            raise ValueError("residual row is absent from Stage-1 selected rows")
        mapped.append(positions[int(token_id)])
    if not mapped:
        return stage1_delta
    indices = torch.tensor(mapped, device=stage1_delta.device, dtype=torch.long)
    addition = torch.zeros_like(stage1_delta).index_add(0, indices, residual_delta)
    return stage1_delta + addition


@torch.no_grad()
def actual_selected_delta(
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    base_rows: torch.Tensor,
) -> torch.Tensor:
    ids = torch.tensor(
        [int(value) for value in selected_ids],
        dtype=torch.long,
        device=output_layer.weight.device,
    )
    current = output_layer.weight.index_select(0, ids).detach().float()
    return current - base_rows.to(device=current.device, dtype=torch.float32)


@contextmanager
def temporary_materialized_output_delta(
    output_layer: torch.nn.Module,
    row_ids: Sequence[int],
    delta: torch.Tensor,
):
    """Materialize an exact checkpoint-dtype candidate, then restore rows."""
    ids = torch.tensor(
        [int(value) for value in row_ids],
        dtype=torch.long,
        device=output_layer.weight.device,
    )
    with torch.no_grad():
        before = output_layer.weight.index_select(0, ids).detach().clone()
        core.materialize_output_delta(output_layer, row_ids, delta)
    try:
        yield
    finally:
        with torch.no_grad():
            output_layer.weight.index_copy_(0, ids, before)


@torch.no_grad()
def cache_logits_preserving_dtype(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Cache exact model logits without discarding checkpoint dtype."""
    chunks: List[torch.Tensor] = []
    model.eval()
    for start in range(0, len(cases), batch_size):
        chunks.append(
            core.forward_last_logits(
                model, tok, cases[start : start + batch_size], device
            )
            .detach()
            .cpu()
        )
    if not chunks:
        raise ValueError("cannot cache logits for an empty case set")
    return torch.cat(chunks, dim=0).contiguous()


def logits_with_sparse_residual(
    stage1_logits: torch.Tensor,
    forget_hidden: torch.Tensor,
    active_ids: Sequence[int],
    residual_delta: torch.Tensor,
) -> torch.Tensor:
    """Apply an LM-head residual to cached Stage-1 logits exactly as the hook.

    The transformer is frozen, so every direct predictor state is invariant.
    Only ``active_ids`` can move in Stage 2.  Casting the correction to the
    cached logit dtype before ``index_add`` matches the training-time LM-head
    hook while retaining gradients to ``residual_delta``.
    """
    if stage1_logits.ndim != 2 or forget_hidden.ndim != 2:
        raise ValueError("Stage-1 logits and forget hidden states must be rank-2")
    if stage1_logits.shape[0] != forget_hidden.shape[0]:
        raise ValueError("Stage-1 logits and forget hidden states do not align")
    if residual_delta.ndim != 2 or residual_delta.shape[0] != len(active_ids):
        raise ValueError("Stage-2 residual rows do not align with active ids")
    if len(set(int(value) for value in active_ids)) != len(active_ids):
        raise ValueError("Stage-2 active row ids must be unique")
    if residual_delta.shape[1] != forget_hidden.shape[1]:
        raise ValueError("Stage-2 residual and hidden dimensions do not align")
    device = residual_delta.device
    logits = stage1_logits.to(device=device)
    hidden = forget_hidden.to(device=device, dtype=torch.float32)
    ids = torch.tensor(
        [int(value) for value in active_ids], device=device, dtype=torch.long
    )
    if ids.numel() == 0:
        return logits
    if bool((ids < 0).any()) or bool((ids >= logits.shape[1]).any()):
        raise ValueError("Stage-2 active row id is outside the vocabulary")
    correction = hidden @ residual_delta.float().transpose(0, 1)
    return logits.index_add(1, ids, correction.to(dtype=logits.dtype))


def constraint_state_from_logits(
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    sensitive_ids: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    tids = sensitive_ids.to(device=logits.device, dtype=torch.long)
    return {
        "logit_margin": shared.suppression_margins_from_logits(logits, tids),
        "sensitive_nll_increase": shared.sensitive_nll_increase_from_logits(
            logits, base_logits, tids
        ),
    }


def protected_nll_targets(
    stage1_nll_increase: torch.Tensor,
    protected_indices: Sequence[int],
    *,
    required_nll_increase: float,
    tolerance: float,
) -> torch.Tensor:
    """Lock Stage-1 successes to the global floor and their Stage-1 clearance."""
    if tolerance < 0 or not math.isfinite(float(tolerance)):
        raise ValueError("protection tolerance must be finite and non-negative")
    indices = torch.tensor(
        [int(value) for value in protected_indices],
        device=stage1_nll_increase.device,
        dtype=torch.long,
    )
    values = stage1_nll_increase.index_select(0, indices).detach().float()
    if not torch.isfinite(values).all():
        raise ValueError("Stage-1 protected NLL values must be finite")
    required = torch.full_like(values, float(required_nll_increase))
    return torch.maximum(required, values - float(tolerance))


def bounded_direct_constraint_loss(
    logits: torch.Tensor,
    base_logits: torch.Tensor,
    sensitive_ids: torch.Tensor,
    *,
    required_logit_margin: float,
    required_nll_increase: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Constraint-gated sensitive GA: zero gradient after both guards pass."""
    margins = shared.suppression_margins_from_logits(logits, sensitive_ids)
    nll_increase = shared.sensitive_nll_increase_from_logits(
        logits, base_logits, sensitive_ids
    )
    margin_shortfall = F.relu(float(required_logit_margin) - margins)
    nll_shortfall = F.relu(float(required_nll_increase) - nll_increase)
    per_case = margin_shortfall.square() + nll_shortfall.square()
    return per_case.mean(), {
        "logit_margin": margins,
        "sensitive_nll_increase": nll_increase,
        "margin_shortfall": margin_shortfall,
        "nll_shortfall": nll_shortfall,
        "active_fraction": (per_case > 0).float().mean(),
    }


@torch.no_grad()
def selected_base_probabilities(
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    utility_hidden: torch.Tensor,
    base_logsumexp: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    if utility_hidden.ndim != 2 or base_logsumexp.shape != (utility_hidden.shape[0],):
        raise ValueError("utility hidden states and Base partitions do not align")
    ids = torch.tensor(
        [int(value) for value in selected_ids],
        device=output_layer.weight.device,
        dtype=torch.long,
    )
    rows = output_layer.weight.index_select(0, ids).detach()
    head_device = rows.device
    bias = getattr(output_layer, "bias", None)
    selected_bias = None if bias is None else bias.index_select(0, ids).detach()
    chunks: List[torch.Tensor] = []
    for start in range(0, int(utility_hidden.shape[0]), batch_size):
        hidden = utility_hidden[start : start + batch_size].to(
            device=head_device, dtype=rows.dtype
        )
        logits = hidden @ rows.transpose(0, 1)
        if selected_bias is not None:
            logits = logits + selected_bias
        log_z = base_logsumexp[start : start + len(hidden)].to(
            device=device, dtype=torch.float32
        )
        chunks.append(torch.exp(logits.float() - log_z.unsqueeze(1)).cpu())
    probabilities = torch.cat(chunks, dim=0).float().contiguous()
    if not torch.isfinite(probabilities).all() or (probabilities < 0).any():
        raise FloatingPointError("selected Base utility probabilities are invalid")
    maximum_mass = float(probabilities.sum(dim=1).max().item())
    if maximum_mass > 1.001:
        raise FloatingPointError(
            f"selected Base probability mass exceeds one: {maximum_mass}"
        )
    return probabilities


def select_token_probability_utility_pool(
    *,
    candidate_indices: torch.Tensor,
    selected_base_probabilities: torch.Tensor,
    selected_ids: Sequence[int],
    topk_per_row: int,
    uniform_prompt_count: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Select contexts where each edited row has the highest Base probability."""
    indices = candidate_indices.detach().cpu().long().contiguous()
    probabilities = selected_base_probabilities.detach().cpu().float().contiguous()
    if indices.ndim != 1 or indices.numel() == 0:
        raise ValueError("utility pool candidates must be a non-empty vector")
    if probabilities.ndim != 2 or probabilities.shape[1] != len(selected_ids):
        raise ValueError("utility probabilities do not align with edited rows")
    if bool((indices < 0).any()) or bool((indices >= probabilities.shape[0]).any()):
        raise ValueError("utility candidate index is out of range")
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ValueError("utility candidate indices must be unique")
    if topk_per_row <= 0 or uniform_prompt_count < 0:
        raise ValueError("utility top-k/uniform counts are invalid")

    local = probabilities.index_select(0, indices)
    chosen = torch.zeros(probabilities.shape[0], dtype=torch.bool)
    coverage: List[Dict[str, Any]] = []
    take = min(int(topk_per_row), int(indices.numel()))
    for column, token_id_value in enumerate(selected_ids):
        scores = local[:, column]
        top_local = torch.topk(scores, k=take, largest=True, sorted=True).indices
        top_global = indices.index_select(0, top_local)
        chosen[top_global] = True
        coverage.append(
            {
                "token_id": int(token_id_value),
                "selected_context_count": int(top_global.numel()),
                "maximum_base_probability": float(scores.max().item()),
                "minimum_topk_base_probability": float(
                    scores.index_select(0, top_local).min().item()
                ),
                "mean_topk_base_probability": float(
                    scores.index_select(0, top_local).mean().item()
                ),
            }
        )

    conditioned_count = int(chosen.index_select(0, indices).sum().item())
    remaining = indices[~chosen.index_select(0, indices)]
    uniform_take = min(int(uniform_prompt_count), int(remaining.numel()))
    if uniform_take:
        chosen[remaining[:uniform_take]] = True
    selected = indices[chosen.index_select(0, indices)]
    if selected.numel() == 0:
        raise RuntimeError("token-conditioned utility selection produced no prompts")
    return selected.contiguous(), {
        "candidate_count": int(indices.numel()),
        "actual_prompt_count": int(selected.numel()),
        "token_conditioned_prompt_count": conditioned_count,
        "uniform_anchor_prompt_count": uniform_take,
        "topk_per_edited_row": int(topk_per_row),
        "per_row_coverage": coverage,
    }


def build_disjoint_token_conditioned_utility_pools(
    *,
    selected_base_probabilities: torch.Tensor,
    selected_ids: Sequence[int],
    topk_per_row: int,
    uniform_prompt_count: int,
    split_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Split candidates first, then independently condition train and guard."""
    prompt_count = int(selected_base_probabilities.shape[0])
    if prompt_count < 2:
        raise ValueError("utility candidate cache needs at least two prompts")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(split_seed))
    permutation = torch.randperm(prompt_count, generator=generator)
    midpoint = prompt_count // 2
    train_candidates = permutation[:midpoint]
    guard_candidates = permutation[midpoint:]
    train_indices, train_report = select_token_probability_utility_pool(
        candidate_indices=train_candidates,
        selected_base_probabilities=selected_base_probabilities,
        selected_ids=selected_ids,
        topk_per_row=topk_per_row,
        uniform_prompt_count=uniform_prompt_count,
    )
    guard_indices, guard_report = select_token_probability_utility_pool(
        candidate_indices=guard_candidates,
        selected_base_probabilities=selected_base_probabilities,
        selected_ids=selected_ids,
        topk_per_row=topk_per_row,
        uniform_prompt_count=uniform_prompt_count,
    )
    if set(train_indices.tolist()) & set(guard_indices.tolist()):
        raise RuntimeError("utility train and checkpoint-guard pools overlap")
    return (
        train_indices,
        guard_indices,
        {
            "selection_protocol": "disjoint_top_base_probability_per_edited_row_v1",
            "candidate_prompt_count": prompt_count,
            "split_seed": int(split_seed),
            "train": train_report,
            "guard": guard_report,
            "train_guard_overlap_count": 0,
            "benchmark_retain_examples_seen": 0,
            "heldout_benchmark_probes_seen": 0,
        },
    )


def exact_sparse_log_partition_ratio(
    shifts: torch.Tensor, selected_base_probabilities: torch.Tensor
) -> torch.Tensor:
    """Return exact ``log(Z_edited / Z_base)`` for sparse logit shifts."""
    if shifts.shape != selected_base_probabilities.shape or shifts.ndim != 2:
        raise ValueError(
            "shifts/probabilities must share [utility, selected-row] shape"
        )
    probabilities = selected_base_probabilities.to(
        device=shifts.device, dtype=shifts.dtype
    ).clamp_min(0.0)
    selected_mass = probabilities.sum(dim=1)
    if bool((selected_mass > 1.001).any().item()):
        raise FloatingPointError("selected Base probability mass exceeds one")
    # Base log-partitions and selected logits are cached/computed in checkpoint
    # dtype, so a row can exceed one by roundoff only. Renormalize those rows to
    # make the exact no-op identity KL(delta=0)=0 hold numerically.
    probabilities = probabilities / selected_mass.clamp_min(1.0).unsqueeze(1)
    selected_mass = probabilities.sum(dim=1)
    remainder = (1.0 - selected_mass).clamp_min(torch.finfo(shifts.dtype).tiny)
    negative_inf = torch.full_like(probabilities, -torch.inf)
    log_probabilities = torch.where(
        probabilities > 0,
        torch.log(probabilities),
        negative_inf,
    )
    return torch.logsumexp(
        torch.cat(
            (torch.log(remainder).unsqueeze(1), log_probabilities + shifts), dim=1
        ),
        dim=1,
    )


def exact_sparse_kl_from_shifts(
    shifts: torch.Tensor, selected_base_probabilities: torch.Tensor
) -> torch.Tensor:
    """Exact ``KL(Base || sparse-row-edited)`` for joint selected-row shifts."""
    probabilities = selected_base_probabilities.to(
        device=shifts.device, dtype=shifts.dtype
    ).clamp_min(0.0)
    selected_mass = probabilities.sum(dim=1)
    probabilities = probabilities / selected_mass.clamp_min(1.0).unsqueeze(1)
    log_partition_ratio = exact_sparse_log_partition_ratio(shifts, probabilities)
    kl = log_partition_ratio - (probabilities * shifts).sum(dim=1)
    return kl.clamp_min(0.0)


def exact_sparse_utility_kl(
    delta: torch.Tensor,
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
) -> torch.Tensor:
    hidden = utility_hidden.to(device=delta.device, dtype=torch.float32)
    probabilities = utility_probabilities.to(device=delta.device, dtype=torch.float32)
    shifts = hidden @ delta.float().transpose(0, 1)
    return exact_sparse_kl_from_shifts(shifts, probabilities)


@torch.no_grad()
def build_exact_stage2_direct_cache(
    stage1_logits: torch.Tensor,
    base_logits: torch.Tensor,
    direct_hidden: torch.Tensor,
    sensitive_ids: torch.Tensor,
    active_ids: Sequence[int],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Cache the sufficient statistics for exact sparse Stage-2 constraints.

    The full vocabulary is touched once here. Subsequent solver evaluations use
    only ``[direct cases, active rows]`` tensors while remaining algebraically
    identical to applying the residual to the frozen LM head in FP32.
    """
    if stage1_logits.ndim != 2 or base_logits.shape != stage1_logits.shape:
        raise ValueError("Stage-1/Base direct logits must share [case,vocab] shape")
    if direct_hidden.ndim != 2 or direct_hidden.shape[0] != stage1_logits.shape[0]:
        raise ValueError("direct hidden states do not align with cached logits")
    if sensitive_ids.shape != (stage1_logits.shape[0],):
        raise ValueError("sensitive ids do not align with direct cases")
    if not active_ids or len(set(int(value) for value in active_ids)) != len(active_ids):
        raise ValueError("Stage-2 exact cache requires unique active row ids")

    logits = stage1_logits.to(device=device, dtype=torch.float32)
    base = base_logits.to(device=device, dtype=torch.float32)
    tids = sensitive_ids.to(device=device, dtype=torch.long)
    ids = torch.tensor([int(value) for value in active_ids], device=device)
    if bool((ids < 0).any()) or bool((ids >= logits.shape[1]).any()):
        raise ValueError("active row id is outside the vocabulary")
    rows = torch.arange(logits.shape[0], device=device)
    stage1_log_z = torch.logsumexp(logits, dim=1)
    active_logits = logits.index_select(1, ids)
    active_probabilities = torch.exp(active_logits - stage1_log_z.unsqueeze(1))
    stage1_nll = stage1_log_z - logits[rows, tids]
    base_nll = torch.logsumexp(base, dim=1) - base[rows, tids]

    # Cache the strongest logit that cannot move in Stage 2. The active-row
    # candidates are handled separately at every solver evaluation.
    unedited = logits.clone()
    unedited.index_fill_(1, ids, -torch.inf)
    unedited[rows, tids] = -torch.inf
    best_unedited_other = unedited.max(dim=1).values

    token_to_column = {int(token_id): column for column, token_id in enumerate(active_ids)}
    target_columns = torch.tensor(
        [token_to_column.get(int(token_id), -1) for token_id in tids.detach().cpu()],
        device=device,
        dtype=torch.long,
    )
    active_is_sensitive = ids.unsqueeze(0).eq(tids.unsqueeze(1))
    return {
        "direct_hidden": direct_hidden.to(device=device, dtype=torch.float32),
        "stage1_active_logits": active_logits,
        "stage1_active_probabilities": active_probabilities,
        "stage1_sensitive_logits": logits[rows, tids],
        "stage1_sensitive_nll_increase": stage1_nll - base_nll,
        "best_unedited_other_logits": best_unedited_other,
        "target_columns": target_columns,
        "active_is_sensitive": active_is_sensitive,
        "sensitive_ids": tids,
    }


def exact_stage2_direct_state(
    cache: Mapping[str, torch.Tensor], residual_delta: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """Evaluate exact FP32 direct NLL changes and best-other margins."""
    hidden = cache["direct_hidden"].to(
        device=residual_delta.device, dtype=torch.float32
    )
    shifts = hidden @ residual_delta.float().transpose(0, 1)
    probabilities = cache["stage1_active_probabilities"].to(
        device=residual_delta.device, dtype=torch.float32
    )
    log_partition_ratio = exact_sparse_log_partition_ratio(shifts, probabilities)
    target_columns = cache["target_columns"].to(device=residual_delta.device)
    safe_columns = target_columns.clamp_min(0)
    target_shift = shifts.gather(1, safe_columns.unsqueeze(1)).squeeze(1)
    target_shift = torch.where(target_columns >= 0, target_shift, torch.zeros_like(target_shift))
    nll_increase = cache["stage1_sensitive_nll_increase"].to(
        device=residual_delta.device, dtype=torch.float32
    ) + log_partition_ratio - target_shift

    active_logits = cache["stage1_active_logits"].to(
        device=residual_delta.device, dtype=torch.float32
    ) + shifts
    active_other_logits = active_logits.masked_fill(
        cache["active_is_sensitive"].to(device=residual_delta.device), -torch.inf
    )
    best_other = torch.maximum(
        cache["best_unedited_other_logits"].to(
            device=residual_delta.device, dtype=torch.float32
        ),
        active_other_logits.max(dim=1).values,
    )
    sensitive_logits = cache["stage1_sensitive_logits"].to(
        device=residual_delta.device, dtype=torch.float32
    ) + target_shift
    return {
        "logit_margin": best_other - sensitive_logits,
        "sensitive_nll_increase": nll_increase,
        "stage2_log_partition_ratio": log_partition_ratio,
        "stage2_target_logit_shift": target_shift,
    }


@torch.no_grad()
def utility_kl_report(
    delta: torch.Tensor,
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    values: List[torch.Tensor] = []
    compute_device = delta.device
    delta_device = delta.detach().to(device=compute_device, dtype=torch.float32)
    for start in range(0, int(utility_hidden.shape[0]), batch_size):
        hidden = utility_hidden[start : start + batch_size].to(
            device=compute_device, dtype=torch.float32
        )
        probabilities = utility_probabilities[start : start + len(hidden)].to(
            device=compute_device, dtype=torch.float32
        )
        shifts = hidden @ delta_device.transpose(0, 1)
        values.append(exact_sparse_kl_from_shifts(shifts, probabilities).cpu())
    kl = torch.cat(values, dim=0).double()
    return {
        "utility_kl_mean": float(kl.mean().item()),
        "utility_kl_median": float(torch.quantile(kl, 0.50).item()),
        "utility_kl_p95": float(torch.quantile(kl, 0.95).item()),
        "utility_kl_p99": float(torch.quantile(kl, 0.99).item()),
        "utility_kl_max": float(kl.max().item()),
    }


def constraint_shortfall(
    state: Mapping[str, torch.Tensor], args: argparse.Namespace
) -> float:
    margin = F.relu(float(args.constraint_margin) - state["logit_margin"].float())
    nll = F.relu(
        float(args.min_sensitive_nll_increase) - state["sensitive_nll_increase"].float()
    )
    return float((margin + nll).sum().detach().cpu())


def constraint_report(
    state: Mapping[str, torch.Tensor], args: argparse.Namespace
) -> Dict[str, Any]:
    failures = shared.count_failures(
        state["logit_margin"],
        state["sensitive_nll_increase"],
        required_logit_margin=args.constraint_margin,
        required_nll_increase=args.min_sensitive_nll_increase,
    )
    return {
        "direct_failures": int(failures),
        "constraint_shortfall_sum": constraint_shortfall(state, args),
        "minimum_logit_margin": float(state["logit_margin"].min().cpu()),
        "minimum_sensitive_nll_increase": float(
            state["sensitive_nll_increase"].min().cpu()
        ),
    }


def stage2_partition_report(
    state: Mapping[str, torch.Tensor],
    *,
    active_indices: Sequence[int],
    protected_indices: Sequence[int],
    protected_targets: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Report repairs and Stage-1-success regressions separately."""
    margins = state["logit_margin"].detach().float()
    nll = state["sensitive_nll_increase"].detach().float()
    combined = [int(value) for value in active_indices] + [
        int(value) for value in protected_indices
    ]
    if sorted(combined) != list(range(int(margins.numel()))):
        raise ValueError(
            "Stage-2 repair/protected partitions must cover each direct case once"
        )

    def select(values: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        index = torch.tensor(
            [int(value) for value in indices],
            device=values.device,
            dtype=torch.long,
        )
        return values.index_select(0, index)

    active_margin = select(margins, active_indices)
    active_nll = select(nll, active_indices)
    protected_margin = select(margins, protected_indices)
    protected_nll = select(nll, protected_indices)
    targets = protected_targets.to(device=protected_nll.device, dtype=torch.float32)
    if targets.shape != protected_nll.shape:
        raise ValueError("protected report targets do not align with protected cases")

    def direct_failures(m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return shared.failure_mask(
            m,
            d,
            required_logit_margin=args.constraint_margin,
            required_nll_increase=args.min_sensitive_nll_increase,
        )

    def standard_shortfall(m: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        return F.relu(float(args.constraint_margin) - m) + F.relu(
            float(args.min_sensitive_nll_increase) - d
        )

    active_failed = direct_failures(active_margin, active_nll)
    protected_failed = direct_failures(protected_margin, protected_nll)
    protected_margin_shortfall = F.relu(
        float(args.constraint_margin) - protected_margin
    )
    protected_nll_shortfall = F.relu(targets - protected_nll)
    protected_barrier = protected_margin_shortfall + protected_nll_shortfall

    def minimum_or_none(values: torch.Tensor) -> float | None:
        if values.numel() == 0:
            return None
        return float(values.min().cpu())

    active_failure_count = int(active_failed.sum().item())
    protected_failure_count = int(protected_failed.sum().item())
    protected_violation_count = int((protected_barrier > 0).sum().item())
    return {
        "active_case_count": len(active_indices),
        "active_direct_failures": active_failure_count,
        "active_repaired_cases": len(active_indices) - active_failure_count,
        "active_constraint_shortfall_sum": float(
            standard_shortfall(active_margin, active_nll).sum().cpu()
        ),
        "active_minimum_logit_margin": minimum_or_none(active_margin),
        "active_minimum_sensitive_nll_increase": minimum_or_none(active_nll),
        "protected_case_count": len(protected_indices),
        "protected_direct_failures": protected_failure_count,
        "protected_new_regressions": protected_failure_count,
        "protected_constraint_shortfall_sum": float(
            standard_shortfall(protected_margin, protected_nll).sum().cpu()
        ),
        "protected_floor_violations": protected_violation_count,
        "protected_floor_shortfall_sum": float(protected_barrier.sum().cpu()),
        "protected_minimum_logit_margin": minimum_or_none(protected_margin),
        "protected_minimum_sensitive_nll_increase": minimum_or_none(protected_nll),
        "protected_minimum_nll_floor_clearance": minimum_or_none(
            protected_nll - targets
        ),
    }


def add_utility_guards(report: Dict[str, Any], args: argparse.Namespace) -> None:
    checks = {
        "mean": float(report["utility_kl_mean"]) <= args.utility_kl_mean_budget,
        "p95": float(report["utility_kl_p95"]) <= args.utility_kl_p95_budget,
        "max": float(report["utility_kl_max"]) <= args.utility_kl_max_budget,
        "norm": float(report["total_delta_norm"]) <= args.max_total_delta_norm,
    }
    report["utility_guard_checks"] = checks
    report["utility_safe"] = bool(all(checks.values()))
    report["direct_success"] = int(report["direct_failures"]) == 0
    report["stage2_protection_safe"] = (
        int(report.get("protected_floor_violations", 0)) == 0
    )
    report["feasible"] = bool(
        report["direct_success"]
        and report["utility_safe"]
        and report["stage2_protection_safe"]
    )


def choose_stage1_report(
    reports: Sequence[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any] | None, str]:
    feasible = [row for row in reports if bool(row.get("feasible", False))]
    if feasible:
        return (
            min(
                feasible,
                key=lambda row: (
                    float(row["utility_kl_mean"]),
                    float(row["utility_kl_p95"]),
                    float(row["utility_kl_max"]),
                    float(row["total_delta_norm"]),
                    float(row["scale"]),
                ),
            ),
            "complete",
        )
    zero = min(reports, key=lambda row: abs(float(row["scale"])))
    progressing = [
        row
        for row in reports
        if bool(row.get("utility_safe", False))
        and (
            int(row["direct_failures"]) < int(zero["direct_failures"])
            or float(row["constraint_shortfall_sum"])
            < float(zero["constraint_shortfall_sum"]) - 1e-8
        )
    ]
    if progressing:
        return (
            min(
                progressing,
                key=lambda row: (
                    int(row["direct_failures"]),
                    float(row["constraint_shortfall_sum"]),
                    float(row["utility_kl_mean"]),
                    float(row["total_delta_norm"]),
                    float(row["scale"]),
                ),
            ),
            "utility_safe_residual_handoff",
        )
    return None, "capacity_expansion"


def choose_stage2_report(
    reports: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    feasible = [row for row in reports if bool(row.get("feasible", False))]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda row: (
            float(row["utility_kl_mean"]),
            float(row["utility_kl_p95"]),
            float(row["utility_kl_max"]),
            float(row["total_delta_norm"]),
            int(row["rank"]),
        ),
    )


def attach_stage2_solver_feasibility(
    report: Dict[str, Any], solver_report: Mapping[str, Any]
) -> None:
    """Require both continuous and checkpoint-dtype Stage-2 feasibility."""
    report["continuous_solver_feasible"] = bool(
        solver_report.get("continuous_solver_feasible", False)
    )
    report["materialized_feasible"] = bool(report["feasible"])
    report["feasible"] = bool(
        report["materialized_feasible"] and report["continuous_solver_feasible"]
    )
    report["continuous_solver_minimum_direct_slack"] = solver_report.get(
        "solver_minimum_direct_slack"
    )
    report["continuous_solver_minimum_utility_slack"] = solver_report.get(
        "solver_minimum_utility_slack"
    )
    report["solver_selection_mode"] = solver_report.get("selection_mode")


@torch.no_grad()
def evaluate_forget_nonsensitive_kl(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    base_logits: torch.Tensor,
    *,
    llama_like: bool,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        logits = core.forward_last_logits(model, tok, batch, device)
        tids = core.official_target_ids(
            tok, batch, llama_like=llama_like, device=device
        )
        value = core.gd_non_sensitive_kl(
            logits, base_logits[start : start + len(batch)], tids
        )
        total += float(value.detach().cpu()) * len(batch)
        count += len(batch)
    return total / float(max(1, count))


@torch.no_grad()
def utility_logit_drift_mse(
    delta_rows: torch.Tensor, utility_second_moment: torch.Tensor
) -> float:
    if delta_rows.numel() == 0:
        return 0.0
    utility = utility_second_moment.to(device=delta_rows.device, dtype=torch.float32)
    delta = delta_rows.float()
    value = ((delta @ utility) * delta).sum(dim=1).mean()
    return float(value.detach().cpu())


def optimize_stage1(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tok: Any,
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    row_bases: Sequence[torch.Tensor],
    forget_cases: Sequence[core.SensitivePredictionCase],
    base_forget_logits: torch.Tensor,
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    llama_like: bool,
    device: torch.device,
    output_dir: Path,
    rank: int,
) -> torch.Tensor:
    delta_module = context.RowSpecificProjectedDelta(
        selected_ids, row_bases, device=output_layer.weight.device
    )
    optimizer = torch.optim.AdamW(
        delta_module.parameters(), lr=args.stage1_lr, weight_decay=0.0
    )
    forget_sampler = core.IndexSampler(
        len(forget_cases), args.stage1_batch_size, args.seed + rank * 101
    )
    utility_sampler = core.IndexSampler(
        int(utility_hidden.shape[0]),
        min(args.utility_train_batch_size, int(utility_hidden.shape[0])),
        args.seed + rank * 1009,
    )
    hook = core.register_output_delta_hook(
        output_layer, selected_ids, delta_module.effective_delta
    )
    log_path = output_dir / f"stage1_rank{rank}_train_log.jsonl"
    try:
        model.eval()
        with log_path.open("w", encoding="utf-8") as stream:
            for step in range(1, args.stage1_steps + 1):
                indices = forget_sampler.next()
                batch = [forget_cases[index] for index in indices]
                optimizer.zero_grad(set_to_none=True)
                logits = core.forward_last_logits(model, tok, batch, device)
                tids = core.official_target_ids(
                    tok, batch, llama_like=llama_like, device=device
                )
                direct, diagnostics = bounded_direct_constraint_loss(
                    logits,
                    base_forget_logits[indices],
                    tids,
                    required_logit_margin=args.constraint_margin,
                    required_nll_increase=args.min_sensitive_nll_increase,
                )
                gd = core.gd_non_sensitive_kl(logits, base_forget_logits[indices], tids)
                utility_indices = utility_sampler.next()
                delta = delta_module.effective_delta()
                utility_kl = exact_sparse_utility_kl(
                    delta,
                    utility_hidden[utility_indices],
                    utility_probabilities[utility_indices],
                ).mean()
                loss = (
                    args.direct_constraint_weight * direct
                    + args.gd_weight * gd
                    + args.utility_kl_weight * utility_kl
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite Stage-1 loss at step {step}")
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(delta_module.parameters()), args.grad_clip
                    )
                optimizer.step()
                if step == 1 or step % 25 == 0 or step == args.stage1_steps:
                    row = {
                        "step": step,
                        "rank": rank,
                        "loss": float(loss.detach().cpu()),
                        "bounded_direct_constraint_loss": float(direct.detach().cpu()),
                        "active_direct_fraction": float(
                            diagnostics["active_fraction"].detach().cpu()
                        ),
                        "same_prompt_non_sensitive_gd_kl": float(gd.detach().cpu()),
                        "wikipedia_exact_kl_batch": float(utility_kl.detach().cpu()),
                        "delta_norm": float(delta.detach().norm().cpu()),
                        "benchmark_retain_examples_seen": 0,
                        "heldout_probes_seen": 0,
                    }
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
    finally:
        hook.remove()
    del optimizer
    return delta_module.effective_delta().detach().clone()


def coefficients_to_residual(
    coefficients: torch.Tensor, row_bases: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Expand one flat Stage-2 coefficient vector into sparse LM-head rows."""
    if coefficients.ndim != 1:
        raise ValueError("Stage-2 coefficients must be a vector")
    rows: List[torch.Tensor] = []
    offset = 0
    for basis in row_bases:
        rank = int(basis.shape[0])
        if basis.ndim != 2 or rank <= 0:
            raise ValueError("every Stage-2 row basis must be non-empty")
        chunk = coefficients[offset : offset + rank]
        if chunk.numel() != rank:
            raise ValueError("Stage-2 coefficient vector is too short")
        rows.append(
            chunk
            @ basis.to(device=coefficients.device, dtype=coefficients.dtype)
        )
        offset += rank
    if offset != coefficients.numel():
        raise ValueError("Stage-2 coefficient vector is too long")
    return torch.stack(rows, dim=0)


def stage2_solver_targets(
    case_count: int,
    protected_indices: Sequence[int],
    protected_nll_targets: torch.Tensor,
    *,
    required_nll_increase: float,
    required_logit_margin: float,
    constraint_buffer: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return hard per-case NLL/margin targets used by the continuous solver."""
    nll = torch.full(
        (case_count,),
        float(required_nll_increase) + float(constraint_buffer),
        device=device,
        dtype=torch.float32,
    )
    protected = torch.tensor(
        [int(value) for value in protected_indices], device=device, dtype=torch.long
    )
    targets = protected_nll_targets.to(device=device, dtype=torch.float32)
    if protected.numel() != targets.numel():
        raise ValueError("protected indices and NLL targets do not align")
    if protected.numel():
        # ``targets`` already equals max(global floor, Stage1 - epsilon).
        # Buffer only the global feasibility boundary; adding it to the entire
        # protected target would silently cancel the allowed epsilon whenever
        # Stage 1 has extra clearance.
        protected_floor = torch.maximum(
            targets,
            targets.new_full(
                targets.shape,
                float(required_nll_increase) + float(constraint_buffer),
            ),
        )
        nll.index_copy_(0, protected, protected_floor)
    margin = torch.full(
        (case_count,),
        float(required_logit_margin) + float(constraint_buffer),
        device=device,
        dtype=torch.float32,
    )
    return nll, margin


class _TorchScalarAdapter:
    """Cache a differentiable Torch scalar as SciPy value/gradient callbacks."""

    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        *,
        device: torch.device,
    ) -> None:
        self.function = function
        self.device = device
        self._x: np.ndarray | None = None
        self._value = 0.0
        self._gradient = np.empty(0, dtype=np.float64)

    def _evaluate(self, values: np.ndarray) -> None:
        current = np.asarray(values, dtype=np.float64)
        if self._x is not None and np.array_equal(current, self._x):
            return
        variable = torch.tensor(
            current, device=self.device, dtype=torch.float32, requires_grad=True
        )
        result = self.function(variable)
        if result.ndim != 0 or not torch.isfinite(result):
            raise FloatingPointError("Stage-2 scalar callback is non-finite")
        gradient = torch.autograd.grad(result, variable)[0]
        self._x = current.copy()
        self._value = float(result.detach().cpu())
        self._gradient = gradient.detach().double().cpu().numpy()

    def value(self, values: np.ndarray) -> float:
        self._evaluate(values)
        return self._value

    def gradient(self, values: np.ndarray) -> np.ndarray:
        self._evaluate(values)
        return self._gradient


class _TorchVectorAdapter:
    """Cache a differentiable Torch vector as SciPy value/Jacobian callbacks."""

    def __init__(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        *,
        device: torch.device,
    ) -> None:
        self.function = function
        self.device = device
        self._x: np.ndarray | None = None
        self._value = np.empty(0, dtype=np.float64)
        self._jacobian = np.empty((0, 0), dtype=np.float64)

    def _evaluate(self, values: np.ndarray) -> None:
        current = np.asarray(values, dtype=np.float64)
        if self._x is not None and np.array_equal(current, self._x):
            return
        variable = torch.tensor(
            current, device=self.device, dtype=torch.float32, requires_grad=True
        )
        result = self.function(variable)
        if result.ndim != 1 or not torch.isfinite(result).all():
            raise FloatingPointError("Stage-2 vector callback is non-finite")
        # A manual reverse-mode stack avoids torch.func/vmap dependencies and
        # is cheap here: Stage 2 has only two constraints per direct token case.
        jacobian = torch.stack(
            [
                torch.autograd.grad(
                    result[index],
                    variable,
                    retain_graph=index + 1 < result.numel(),
                )[0]
                for index in range(result.numel())
            ],
            dim=0,
        )
        self._x = current.copy()
        self._value = result.detach().double().cpu().numpy()
        self._jacobian = jacobian.detach().double().cpu().numpy()

    def value(self, values: np.ndarray) -> np.ndarray:
        self._evaluate(values)
        return self._value

    def jacobian(self, values: np.ndarray) -> np.ndarray:
        self._evaluate(values)
        return self._jacobian


def repair_directed_stage2_start(
    direct_cache: Mapping[str, torch.Tensor],
    row_bases: Sequence[torch.Tensor],
    active_case_indices: Sequence[int],
    nll_targets: torch.Tensor,
    margin_targets: torch.Tensor,
) -> torch.Tensor:
    """Construct a deterministic least-squares start from active shortfalls."""
    device = nll_targets.device
    ranks = [int(basis.shape[0]) for basis in row_bases]
    start = torch.zeros(sum(ranks), device=device, dtype=torch.float32)
    zero_rows = torch.zeros(
        (len(row_bases), int(direct_cache["direct_hidden"].shape[1])),
        device=device,
        dtype=torch.float32,
    )
    initial = exact_stage2_direct_state(direct_cache, zero_rows)
    active_mask = torch.zeros(
        int(nll_targets.numel()), device=device, dtype=torch.bool
    )
    active_mask[
        torch.tensor([int(value) for value in active_case_indices], device=device)
    ] = True
    target_columns = direct_cache["target_columns"].to(device=device)
    hidden = direct_cache["direct_hidden"].to(device=device, dtype=torch.float32)
    offset = 0
    for column, basis in enumerate(row_bases):
        rank = ranks[column]
        case_mask = active_mask & target_columns.eq(column)
        indices = torch.where(case_mask)[0]
        if indices.numel():
            nll_need = F.relu(
                nll_targets.index_select(0, indices)
                - initial["sensitive_nll_increase"].index_select(0, indices)
            )
            margin_need = F.relu(
                margin_targets.index_select(0, indices)
                - initial["logit_margin"].index_select(0, indices)
            )
            desired_sensitive_shift = -torch.maximum(nll_need, margin_need)
            response = hidden.index_select(0, indices) @ basis.to(
                device=device, dtype=torch.float32
            ).transpose(0, 1)
            solution = torch.linalg.lstsq(
                response, desired_sensitive_shift.unsqueeze(1)
            ).solution.squeeze(1)
            if torch.isfinite(solution).all():
                start[offset : offset + rank] = solution
        offset += rank
    return start


def optimize_stage2(
    *,
    args: argparse.Namespace,
    stage1_ids: Sequence[int],
    stage1_delta: torch.Tensor,
    active_ids: Sequence[int],
    row_bases: Sequence[torch.Tensor],
    all_forget_hidden: torch.Tensor,
    all_forget_tids: torch.Tensor,
    stage1_forget_logits: torch.Tensor,
    base_forget_logits: torch.Tensor,
    active_case_indices: Sequence[int],
    protected_case_indices: Sequence[int],
    protected_targets: torch.Tensor,
    utility_train_hidden: torch.Tensor,
    utility_train_probabilities: torch.Tensor,
    rank: int,
) -> Tuple[torch.Tensor, List[Dict[str, Any]], Dict[str, Any]]:
    """Solve minimum utility cost subject to exact mandatory constraints."""
    compute_device = stage1_delta.device
    bases = [
        basis.to(device=compute_device, dtype=torch.float32).contiguous()
        for basis in row_bases
    ]
    coefficient_count = sum(int(basis.shape[0]) for basis in bases)
    if coefficient_count <= 0:
        raise ValueError("Stage-2 constrained solver has zero coefficients")
    direct_cache = build_exact_stage2_direct_cache(
        stage1_forget_logits,
        base_forget_logits,
        all_forget_hidden,
        all_forget_tids,
        active_ids,
        device=compute_device,
    )
    nll_targets, margin_targets = stage2_solver_targets(
        int(all_forget_tids.numel()),
        protected_case_indices,
        protected_targets,
        required_nll_increase=args.min_sensitive_nll_increase,
        required_logit_margin=args.constraint_margin,
        constraint_buffer=args.stage2_constraint_buffer,
        device=compute_device,
    )
    utility_hidden = utility_train_hidden.to(
        device=compute_device, dtype=torch.float32
    )
    utility_probabilities = utility_train_probabilities.to(
        device=compute_device, dtype=torch.float32
    )

    def residual(coefficients: torch.Tensor) -> torch.Tensor:
        return coefficients_to_residual(coefficients, bases)

    def total_delta(coefficients: torch.Tensor) -> torch.Tensor:
        return total_delta_with_residual(
            stage1_delta,
            stage1_ids,
            residual(coefficients),
            active_ids,
        )

    def utility_values(coefficients: torch.Tensor) -> torch.Tensor:
        return exact_sparse_utility_kl(
            total_delta(coefficients), utility_hidden, utility_probabilities
        )

    def objective(coefficients: torch.Tensor) -> torch.Tensor:
        current_residual = residual(coefficients)
        return (
            float(args.utility_kl_weight) * utility_values(coefficients).mean()
            + float(args.stage2_residual_l2_weight)
            * current_residual.square().sum()
        )

    def direct_slacks(coefficients: torch.Tensor) -> torch.Tensor:
        state = exact_stage2_direct_state(direct_cache, residual(coefficients))
        return torch.cat(
            (
                state["sensitive_nll_increase"] - nll_targets,
                state["logit_margin"] - margin_targets,
            )
        )

    def utility_slacks(coefficients: torch.Tensor) -> torch.Tensor:
        values = utility_values(coefficients)
        total = total_delta(coefficients)
        return torch.stack(
            (
                values.new_tensor(float(args.utility_kl_mean_budget))
                - values.mean(),
                values.new_tensor(float(args.utility_kl_p95_budget))
                - torch.quantile(values, 0.95),
                values.new_tensor(float(args.utility_kl_max_budget))
                - values.max(),
                values.new_tensor(float(args.max_total_delta_norm)) - total.norm(),
            )
        )

    scalar = _TorchScalarAdapter(objective, device=compute_device)
    direct_constraint = _TorchVectorAdapter(direct_slacks, device=compute_device)
    utility_constraint = _TorchVectorAdapter(utility_slacks, device=compute_device)
    starts = [torch.zeros(coefficient_count, device=compute_device)]
    if int(args.stage2_restarts) == 2:
        directed = repair_directed_stage2_start(
            direct_cache,
            bases,
            active_case_indices,
            nll_targets,
            margin_targets,
        )
        if not torch.allclose(directed, starts[0]):
            starts.append(directed)

    history: List[Dict[str, Any]] = []
    solver_attempts: List[Dict[str, Any]] = []
    observed_candidates: List[Tuple[torch.Tensor, Dict[str, Any]]] = []
    tolerance = float(args.stage2_constraint_tolerance)

    def inspect(
        values: np.ndarray,
        *,
        restart: int,
        iteration: int,
        phase: str,
    ) -> Dict[str, Any]:
        coefficients = torch.tensor(
            np.asarray(values, dtype=np.float64),
            device=compute_device,
            dtype=torch.float32,
        )
        current_residual = residual(coefficients).detach()
        current_total = total_delta(coefficients).detach()
        state = exact_stage2_direct_state(direct_cache, current_residual)
        utility = utility_values(coefficients).detach().double().cpu()
        dslack = direct_slacks(coefficients).detach()
        uslack = utility_slacks(coefficients).detach()
        row = {
            "solver_phase": phase,
            "restart": int(restart),
            "iteration": int(iteration),
            "rank": int(rank),
            **constraint_report(state, args),
            **stage2_partition_report(
                state,
                active_indices=active_case_indices,
                protected_indices=protected_case_indices,
                protected_targets=protected_targets,
                args=args,
            ),
            "solver_minimum_direct_slack": float(dslack.min().cpu()),
            "solver_minimum_utility_slack": float(uslack.min().cpu()),
            "solver_buffer": float(args.stage2_constraint_buffer),
            "solver_objective": float(objective(coefficients).detach().cpu()),
            "utility_kl_mean": float(utility.mean()),
            "utility_kl_median": float(torch.quantile(utility, 0.50)),
            "utility_kl_p95": float(torch.quantile(utility, 0.95)),
            "utility_kl_p99": float(torch.quantile(utility, 0.99)),
            "utility_kl_max": float(utility.max()),
            "utility_prompt_count": int(utility.numel()),
            "coefficient_norm": float(coefficients.norm().cpu()),
            "residual_delta_norm": float(current_residual.norm().cpu()),
            "total_delta_norm": float(current_total.norm().cpu()),
            "continuous_solver_feasible": bool(
                float(dslack.min().cpu()) >= -tolerance
                and float(uslack.min().cpu()) >= -tolerance
            ),
            "benchmark_retain_examples_seen": 0,
            "heldout_probes_seen": 0,
        }
        history.append(row)
        # Keep the compact coefficient vector for every inspected SLSQP iterate.
        # This prevents a feasible intermediate point from being discarded if a
        # later line-search or iteration-limit exit lands just outside a boundary.
        observed_candidates.append((coefficients.detach().cpu(), row))
        return row

    for restart, start in enumerate(starts):
        iteration = 0
        start_numpy = start.detach().double().cpu().numpy()
        inspect(start_numpy, restart=restart, iteration=0, phase="initial")

        def callback(values: np.ndarray) -> None:
            nonlocal iteration
            iteration += 1
            inspect(values, restart=restart, iteration=iteration, phase="iterate")

        result = minimize(
            scalar.value,
            start_numpy,
            method="SLSQP",
            jac=scalar.gradient,
            constraints=(
                {
                    "type": "ineq",
                    "fun": direct_constraint.value,
                    "jac": direct_constraint.jacobian,
                },
                {
                    "type": "ineq",
                    "fun": utility_constraint.value,
                    "jac": utility_constraint.jacobian,
                },
            ),
            callback=callback,
            options={
                "maxiter": int(args.stage2_maxiter),
                "ftol": float(args.stage2_ftol),
                "disp": False,
            },
        )
        final = inspect(
            result.x,
            restart=restart,
            iteration=iteration + 1,
            phase="final",
        )
        summary = {
            "restart": int(restart),
            "scipy_success": bool(result.success),
            "scipy_status": int(result.status),
            "scipy_message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "jacobian_evaluations": int(getattr(result, "njev", 0)),
            "continuous_solver_feasible": bool(final["continuous_solver_feasible"]),
            "minimum_direct_slack": float(final["solver_minimum_direct_slack"]),
            "minimum_utility_slack": float(final["solver_minimum_utility_slack"]),
            "objective": float(final["solver_objective"]),
        }
        final.update(summary)
        solver_attempts.append(summary)

    feasible = [
        item
        for item in observed_candidates
        if item[1]["continuous_solver_feasible"]
    ]
    if feasible:
        best_coefficients, best_report = min(
            feasible,
            key=lambda item: (
                float(item[1]["solver_objective"]),
                float(item[1]["utility_kl_mean"]),
                float(item[1]["residual_delta_norm"]),
            ),
        )
        selection_mode = "minimum_utility_exact_feasible"
    else:
        best_coefficients, best_report = min(
            observed_candidates,
            key=lambda item: (
                max(0.0, -float(item[1]["solver_minimum_direct_slack"])),
                int(item[1]["direct_failures"]),
                float(item[1]["constraint_shortfall_sum"]),
                float(item[1]["solver_objective"]),
            ),
        )
        selection_mode = "best_infeasible_diagnostic"
    best_delta = residual(
        best_coefficients.to(device=compute_device, dtype=torch.float32)
    ).detach()
    best_report = {
        **best_report,
        "selection_mode": selection_mode,
        "solver_attempts": solver_attempts,
        "coefficient_count": coefficient_count,
        "row_ranks": [int(basis.shape[0]) for basis in bases],
        "solver": "SLSQP",
        "hard_constraints_tradeable": False,
    }
    return best_delta, history, best_report


@torch.no_grad()
def evaluate_stage1_scale(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tok: Any,
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    selected_base_rows: torch.Tensor,
    trained_delta: torch.Tensor,
    scale: float,
    rank: int,
    forget_cases: Sequence[core.SensitivePredictionCase],
    base_forget_logits: torch.Tensor,
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    llama_like: bool,
    device: torch.device,
) -> Dict[str, Any]:
    scaled = trained_delta * float(scale)
    with temporary_materialized_output_delta(output_layer, selected_ids, scaled):
        state = shared.evaluate_shared_constraints(
            model,
            tok,
            forget_cases,
            base_forget_logits,
            llama_like=llama_like,
            device=device,
            batch_size=args.cache_batch_size,
        )
        actual = actual_selected_delta(output_layer, selected_ids, selected_base_rows)
        utility = utility_kl_report(
            actual,
            utility_hidden,
            utility_probabilities,
            device=device,
            batch_size=args.utility_eval_batch_size,
        )
    report = {
        **constraint_report(state, args),
        **utility,
        "rank": int(rank),
        "scale": float(scale),
        "total_delta_norm": float(actual.norm().detach().cpu()),
        "selection_inputs": (
            "direct_constraints_plus_heldout_token_conditioned_wikipedia_guard"
        ),
    }
    add_utility_guards(report, args)
    return report


@torch.no_grad()
def evaluate_stage2_residual(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    tok: Any,
    output_layer: torch.nn.Module,
    selected_ids: Sequence[int],
    selected_base_rows: torch.Tensor,
    active_ids: Sequence[int],
    residual_delta: torch.Tensor,
    rank: int,
    forget_cases: Sequence[core.SensitivePredictionCase],
    base_forget_logits: torch.Tensor,
    active_case_indices: Sequence[int],
    protected_case_indices: Sequence[int],
    protected_targets: torch.Tensor,
    utility_hidden: torch.Tensor,
    utility_probabilities: torch.Tensor,
    llama_like: bool,
    device: torch.device,
) -> Dict[str, Any]:
    with temporary_materialized_output_delta(output_layer, active_ids, residual_delta):
        state = shared.evaluate_shared_constraints(
            model,
            tok,
            forget_cases,
            base_forget_logits,
            llama_like=llama_like,
            device=device,
            batch_size=args.cache_batch_size,
        )
        actual = actual_selected_delta(output_layer, selected_ids, selected_base_rows)
        utility = utility_kl_report(
            actual,
            utility_hidden,
            utility_probabilities,
            device=device,
            batch_size=args.utility_eval_batch_size,
        )
    report = {
        **constraint_report(state, args),
        **stage2_partition_report(
            state,
            active_indices=active_case_indices,
            protected_indices=protected_case_indices,
            protected_targets=protected_targets,
            args=args,
        ),
        **utility,
        "rank": int(rank),
        "residual_delta_norm": float(residual_delta.norm().detach().cpu()),
        "total_delta_norm": float(actual.norm().detach().cpu()),
        "selection_inputs": (
            "hard_active_and_protected_direct_constraints_plus_heldout_"
            "token_conditioned_wikipedia_guard_after_solve"
        ),
    }
    add_utility_guards(report, args)
    return report


def save_checkpoint(model: torch.nn.Module, tok: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tok.save_pretrained(path)


def main() -> None:
    args = parse_args()
    stage1_scales, rank_ladder = validate_args(args)
    shared_architecture = architecture_signature_payload(
        args, stage1_scales, rank_ladder
    )
    architecture_sha256 = architecture_signature_sha256(shared_architecture)
    gagd.set_seed(args.seed)
    if args.device_map == "single":
        gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    forget_records, split_manifest = load_locked(args)
    adapter = adapter_contract(split_manifest)
    sensitive_field = adapter["sensitive_answer_field"]

    namespace = argparse.Namespace(
        model_path=args.model_path,
        dtype=args.dtype,
        device_map=args.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(namespace, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    identity = wikipedia.model_identity(model, tok, args.model_path)
    output_before_untie = model.get_output_embeddings()
    if output_before_untie is None:
        raise ValueError("Model has no output embeddings")
    hidden_size = int(output_before_untie.weight.shape[1])
    (
        utility_second_moment,
        utility_hidden,
        base_utility_logsumexp,
        utility_metadata,
    ) = load_utility_cache(
        Path(args.utility_cache).resolve(),
        expected_sample_size=args.utility_sample_size,
        expected_prompt_count=args.utility_prompt_count,
        expected_hidden_size=hidden_size,
        expected_model_probe=identity["model_probe_sha256"],
        expected_tokenizer_probe=identity["tokenizer_probe_sha256"],
    )
    actual_utility_documents = int(utility_metadata["actual_document_sample_size"])
    if actual_utility_documents < args.utility_sample_size:
        print(
            "WARNING: Wikipedia utility corpus was capped to "
            f"{actual_utility_documents} documents; treat this as a pilot, not "
            "a full 100,000-document comparison"
        )
    actual_utility_prompts = int(utility_metadata["actual_utility_prompt_count"])
    if actual_utility_prompts < args.utility_prompt_count:
        print(
            "WARNING: exact Wikipedia KL uses only "
            f"{actual_utility_prompts} cached prompts rather than the requested "
            f"{args.utility_prompt_count}"
        )
    compatibility_payload = {
        "shared_architecture_sha256": architecture_sha256,
        "model_probe_sha256": identity["model_probe_sha256"],
        "tokenizer_probe_sha256": identity["tokenizer_probe_sha256"],
        "utility_second_moment_sha256": utility_metadata["second_moment_sha256"],
        "utility_hidden_sha256": utility_metadata["utility_hidden_sha256"],
        "utility_base_logsumexp_sha256": utility_metadata["base_logsumexp_sha256"],
    }
    cross_dataset_compatibility_sha256 = architecture_signature_sha256(
        compatibility_payload
    )

    output_layer = core.untie_and_freeze_output_head(model)
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)
    input_embedding_pointer = model.get_input_embeddings().weight.data_ptr()

    forget_cases = core.expand_sensitive_cases(
        forget_records,
        tok,
        sensitive_field=sensitive_field,
        llama_like=llama_like,
    )
    if not forget_cases:
        raise RuntimeError("Direct forget records produced zero token cases")
    forget_tids = core.official_target_ids(
        tok, forget_cases, llama_like=llama_like, device=device
    ).detach()
    selected_ids = sorted(set(int(value) for value in forget_tids.cpu().tolist()))
    selected_tensor = torch.tensor(
        selected_ids, dtype=torch.long, device=output_layer.weight.device
    )
    selected_base_rows = (
        output_layer.weight.index_select(0, selected_tensor).detach().float().clone()
    )
    utility_probabilities = selected_base_probabilities(
        output_layer,
        selected_ids,
        utility_hidden,
        base_utility_logsumexp,
        device=device,
        batch_size=args.utility_eval_batch_size,
    )
    selected_mass = utility_probabilities.sum(dim=1)
    (
        utility_train_indices,
        utility_guard_indices,
        utility_pool_report,
    ) = build_disjoint_token_conditioned_utility_pools(
        selected_base_probabilities=utility_probabilities,
        selected_ids=selected_ids,
        topk_per_row=args.utility_token_topk_per_row,
        uniform_prompt_count=args.utility_uniform_prompt_count,
        split_seed=args.utility_pool_seed,
    )
    utility_train_hidden = utility_hidden.index_select(
        0, utility_train_indices
    ).contiguous()
    utility_train_probabilities = utility_probabilities.index_select(
        0, utility_train_indices
    ).contiguous()
    utility_guard_hidden = utility_hidden.index_select(
        0, utility_guard_indices
    ).contiguous()
    utility_guard_probabilities = utility_probabilities.index_select(
        0, utility_guard_indices
    ).contiguous()
    torch.save(
        {
            "row_ids": selected_ids,
            "candidate_probabilities": utility_probabilities,
            "train_indices": utility_train_indices,
            "guard_indices": utility_guard_indices,
        },
        output_dir / "base_wikipedia_selected_probabilities.pt",
    )
    train_mass = utility_train_probabilities.sum(dim=1)
    guard_mass = utility_guard_probabilities.sum(dim=1)
    core.write_json(
        output_dir / "base_wikipedia_selected_probability_summary.json",
        {
            "utility_candidate_prompt_count": int(utility_probabilities.shape[0]),
            "utility_train_prompt_count": int(utility_train_probabilities.shape[0]),
            "utility_guard_prompt_count": int(utility_guard_probabilities.shape[0]),
            "selected_row_count": len(selected_ids),
            "candidate_mean_selected_mass_per_prompt": float(
                selected_mass.mean().item()
            ),
            "candidate_max_selected_mass_per_prompt": float(selected_mass.max().item()),
            "train_mean_selected_mass_per_prompt": float(train_mass.mean().item()),
            "train_max_selected_mass_per_prompt": float(train_mass.max().item()),
            "guard_mean_selected_mass_per_prompt": float(guard_mass.mean().item()),
            "guard_max_selected_mass_per_prompt": float(guard_mass.max().item()),
            "maximum_single_selected_probability": float(
                utility_probabilities.max().item()
            ),
            "pool_selection": utility_pool_report,
            "benchmark_retain_examples_seen": 0,
        },
    )
    core.write_json(
        output_dir / "token_conditioned_utility_pool_report.json",
        utility_pool_report,
    )
    del utility_hidden, utility_probabilities, base_utility_logsumexp
    base_forget_logits = core.cache_base_logits(
        model,
        tok,
        forget_cases,
        device,
        batch_size=args.cache_batch_size,
    )
    forget_hidden = (
        core.forward_last_hidden(
            model, tok, forget_cases, device, args.cache_batch_size
        )
        .float()
        .detach()
    )
    utility_cholesky, utility_geometry = regularized_utility_cholesky(
        utility_second_moment,
        relative_eps=args.contrastive_eps,
        device=forget_hidden.device,
    )

    torch.save(base_forget_logits, output_dir / "base_sensitive_case_logits.pt")
    core.write_json(output_dir / "utility_cache_metadata.json", utility_metadata)
    core.write_json(output_dir / "utility_geometry.json", utility_geometry)
    core.write_json(
        output_dir / "architecture_lock.json",
        {
            "schema_version": 5,
            "method": METHOD,
            "dataset_adapter": args.dataset,
            "dataset_adapter_contract": adapter,
            "architecture_core_is_dataset_independent": True,
            "shared_architecture_parameters": shared_architecture,
            "shared_architecture_sha256": architecture_sha256,
            "cross_dataset_compatibility_payload": compatibility_payload,
            "cross_dataset_compatibility_sha256": (cross_dataset_compatibility_sha256),
            "benchmark_retain_train_examples": 0,
            "rank_ladder": list(rank_ladder),
            "stage1_objective": (
                "bounded direct constraint GA + same-prompt conditional GD + "
                "exact joint token-conditioned Wikipedia train-pool KL"
            ),
            "stage2_objective": (
                "minimum exact token-conditioned Wikipedia KL plus residual L2, "
                "subject to non-tradeable exact repair, Stage1-success preservation, "
                "utility-distribution, and total-norm inequalities"
            ),
            "stage2_direct_constraint_schedule": (
                "all repair and protected direct token cases in every SLSQP solve"
            ),
            "stage2_protection_nll_tolerance": float(
                args.stage2_protection_nll_tolerance
            ),
            "stage2_selection": (
                "minimum held-out token-conditioned Wikipedia guard KL among "
                "exact-materialized rank candidates that satisfy every mandatory "
                "behavioral and utility guard; no residual scaling frontier"
            ),
            "stage2_solver": "SLSQP_with_analytic_Torch_gradients",
            "stage2_constraint_buffer": float(args.stage2_constraint_buffer),
            "stage2_constraint_basis_weight": float(
                args.stage2_constraint_basis_weight
            ),
            "utility_pool_selection": utility_pool_report,
            "utility_guard_budgets": {
                "mean": args.utility_kl_mean_budget,
                "p95": args.utility_kl_p95_budget,
                "max": args.utility_kl_max_budget,
                "total_delta_norm": args.max_total_delta_norm,
            },
            "post_training_metrics_not_used_for_selection": [
                "official FS/GFS",
                "Spe/Spe-success",
                "official benchmark retain evaluation",
                "exact benchmark-retain KL",
                "PPL",
            ],
        },
    )

    stage1_attempts: List[Dict[str, Any]] = []
    selected_stage1_report: Mapping[str, Any] | None = None
    selected_stage1_mode = "capacity_expansion"
    selected_stage1_rank = None
    selected_stage1_trained_delta = None
    selected_stage1_basis_reports: List[Dict[str, Any]] = []
    selected_stage1_scale_reports: List[Dict[str, Any]] = []

    for rank in rank_ladder:
        row_bases, basis_reports = build_contrastive_bases_from_second_moment(
            forget_hidden,
            forget_tids,
            utility_second_moment,
            requested_ids=selected_ids,
            rank_cap=rank,
            relative_eps=args.contrastive_eps,
            utility_cholesky=utility_cholesky,
        )
        core.write_json(
            output_dir / f"stage1_rank{rank}_basis_reports.json", basis_reports
        )
        trained_delta = optimize_stage1(
            args=args,
            model=model,
            tok=tok,
            output_layer=output_layer,
            selected_ids=selected_ids,
            row_bases=row_bases,
            forget_cases=forget_cases,
            base_forget_logits=base_forget_logits,
            utility_hidden=utility_train_hidden,
            utility_probabilities=utility_train_probabilities,
            llama_like=llama_like,
            device=device,
            output_dir=output_dir,
            rank=rank,
        )
        torch.save(
            {"row_ids": selected_ids, "delta": trained_delta.cpu(), "rank": rank},
            output_dir / f"stage1_rank{rank}_unscaled_delta.pt",
        )
        reports = [
            evaluate_stage1_scale(
                args=args,
                model=model,
                tok=tok,
                output_layer=output_layer,
                selected_ids=selected_ids,
                selected_base_rows=selected_base_rows,
                trained_delta=trained_delta,
                scale=scale,
                rank=rank,
                forget_cases=forget_cases,
                base_forget_logits=base_forget_logits,
                utility_hidden=utility_guard_hidden,
                utility_probabilities=utility_guard_probabilities,
                llama_like=llama_like,
                device=device,
            )
            for scale in stage1_scales
        ]
        core.write_json(output_dir / f"stage1_rank{rank}_scale_reports.json", reports)
        selected, mode = choose_stage1_report(reports)
        stage1_attempts.append(
            {
                "rank": rank,
                "selection_mode": mode,
                "selected": None if selected is None else dict(selected),
            }
        )
        if selected is not None:
            selected_stage1_report = selected
            selected_stage1_mode = mode
            selected_stage1_rank = rank
            selected_stage1_trained_delta = trained_delta
            selected_stage1_basis_reports = basis_reports
            selected_stage1_scale_reports = reports
            break

    core.write_json(output_dir / "stage1_attempts.json", stage1_attempts)
    if selected_stage1_report is None or selected_stage1_trained_delta is None:
        core.write_json(
            output_dir / "stage1_infeasible.json",
            {
                "method": METHOD,
                "rank_ladder": list(rank_ladder),
                "attempts": stage1_attempts,
                "reason": "no non-zero utility-safe progress at any locked rank",
            },
        )
        raise RuntimeError(
            "Stage 1 made no utility-safe progress at ranks 2 or 4; no unsafe "
            "checkpoint was materialized"
        )

    core.write_json(
        output_dir / "stage1_contrastive_basis_reports.json",
        selected_stage1_basis_reports,
    )
    core.write_json(
        output_dir / "stage1_scale_reports.json", selected_stage1_scale_reports
    )
    torch.save(
        {
            "row_ids": selected_ids,
            "delta": selected_stage1_trained_delta.cpu(),
            "rank": selected_stage1_rank,
        },
        output_dir / "stage1_unscaled_delta.pt",
    )
    conceptual_stage1_delta = selected_stage1_trained_delta * float(
        selected_stage1_report["scale"]
    )
    core.materialize_output_delta(output_layer, selected_ids, conceptual_stage1_delta)
    stage1_delta = actual_selected_delta(output_layer, selected_ids, selected_base_rows)
    torch.save(
        {
            "row_ids": selected_ids,
            "delta": stage1_delta.detach().cpu(),
            "rank": selected_stage1_rank,
        },
        output_dir / "stage1_total_delta.pt",
    )
    stage1_checkpoint = output_dir / "stage1_checkpoint"
    save_checkpoint(model, tok, stage1_checkpoint)

    stage1_forget_logits = cache_logits_preserving_dtype(
        model,
        tok,
        forget_cases,
        device,
        args.cache_batch_size,
    )
    torch.save(
        stage1_forget_logits,
        output_dir / "stage1_sensitive_case_logits.pt",
    )
    stage1_state = constraint_state_from_logits(
        stage1_forget_logits,
        base_forget_logits,
        forget_tids.detach().cpu(),
    )
    failed_mask = shared.failure_mask(
        stage1_state["logit_margin"],
        stage1_state["sensitive_nll_increase"],
        required_logit_margin=args.constraint_margin,
        required_nll_increase=args.min_sensitive_nll_increase,
    )
    active_indices = [
        index
        for index, failed in enumerate(failed_mask.detach().cpu().tolist())
        if bool(failed)
    ]
    protected_indices = [
        index
        for index, failed in enumerate(failed_mask.detach().cpu().tolist())
        if not bool(failed)
    ]
    protection_targets = protected_nll_targets(
        stage1_state["sensitive_nll_increase"],
        protected_indices,
        required_nll_increase=args.min_sensitive_nll_increase,
        tolerance=args.stage2_protection_nll_tolerance,
    ).cpu()
    stage1_partition = stage2_partition_report(
        stage1_state,
        active_indices=active_indices,
        protected_indices=protected_indices,
        protected_targets=protection_targets,
        args=args,
    )
    protected_case_rows = []
    for position, case_index in enumerate(protected_indices):
        case = forget_cases[case_index]
        protected_case_rows.append(
            {
                "case_index": int(case_index),
                "case_id": int(case.case_id),
                "record_position": int(case.record_position),
                "token_index": int(case.token_index),
                "token_id": int(forget_tids[case_index].detach().cpu()),
                "stage1_sensitive_nll_increase": float(
                    stage1_state["sensitive_nll_increase"][case_index].cpu()
                ),
                "protected_nll_floor": float(protection_targets[position]),
                "continuous_solver_nll_floor": float(
                    max(
                        float(protection_targets[position]),
                        args.min_sensitive_nll_increase
                        + args.stage2_constraint_buffer,
                    )
                ),
                "continuous_solver_margin_floor": float(
                    args.constraint_margin + args.stage2_constraint_buffer
                ),
            }
        )
    core.write_json(
        output_dir / "stage2_case_partitions.json",
        {
            "schema_version": 2,
            "partition_source": "exact_materialized_stage1_direct_constraints",
            "active_case_indices": active_indices,
            "protected_case_indices": protected_indices,
            "active_case_count": len(active_indices),
            "protected_case_count": len(protected_indices),
            "protected_nll_floor_formula": (
                "max(global_min_sensitive_nll_increase, "
                "stage1_sensitive_nll_increase - tolerance)"
            ),
            "protected_nll_tolerance": float(args.stage2_protection_nll_tolerance),
            "continuous_solver_constraint_buffer": float(
                args.stage2_constraint_buffer
            ),
            "protected_continuous_solver_nll_floor_formula": (
                "max(protected_nll_floor, "
                "global_min_sensitive_nll_increase + constraint_buffer)"
            ),
            "active_continuous_solver_nll_floor": float(
                args.min_sensitive_nll_increase + args.stage2_constraint_buffer
            ),
            "all_case_continuous_solver_margin_floor": float(
                args.constraint_margin + args.stage2_constraint_buffer
            ),
            "protected_cases": protected_case_rows,
            "stage1_partition_report": stage1_partition,
        },
    )

    stage2_mode = "identity_noop"
    stage2_attempts: List[Dict[str, Any]] = []
    selected_stage2_report: Mapping[str, Any] | None = None
    selected_stage2_rank = None
    selected_stage2_residual = None
    selected_stage2_basis_reports: List[Dict[str, Any]] = []
    selected_stage2_history: List[Dict[str, Any]] = []
    selected_stage2_best_training: Dict[str, Any] | None = None
    active_ids: List[int] = []
    actual_stage2_residual = output_layer.weight.new_empty((0, hidden_size)).float()

    if active_indices:
        stage2_mode = "exact_constrained_minimum_utility_residual"
        active_hidden = forget_hidden[active_indices]
        active_tids = forget_tids[active_indices]
        active_ids = sorted(set(int(value) for value in active_tids.cpu().tolist()))
        active_tensor = torch.tensor(
            active_ids, dtype=torch.long, device=output_layer.weight.device
        )
        active_rows_before = (
            output_layer.weight.index_select(0, active_tensor).detach().float().clone()
        )

        rank_candidates: List[Dict[str, Any]] = []
        for rank in rank_ladder:
            stage2_bases, basis_reports = build_constraint_aware_stage2_bases(
                active_hidden,
                active_tids,
                forget_hidden,
                utility_second_moment,
                requested_ids=active_ids,
                rank_cap=rank,
                relative_eps=args.contrastive_eps,
                constraint_context_weight=args.stage2_constraint_basis_weight,
                utility_cholesky=utility_cholesky,
            )
            residual, history, best_training = optimize_stage2(
                args=args,
                stage1_ids=selected_ids,
                stage1_delta=stage1_delta,
                active_ids=active_ids,
                row_bases=stage2_bases,
                all_forget_hidden=forget_hidden,
                all_forget_tids=forget_tids,
                stage1_forget_logits=stage1_forget_logits,
                base_forget_logits=base_forget_logits,
                active_case_indices=active_indices,
                protected_case_indices=protected_indices,
                protected_targets=protection_targets,
                utility_train_hidden=utility_train_hidden,
                utility_train_probabilities=utility_train_probabilities,
                rank=rank,
            )
            torch.save(
                {"row_ids": active_ids, "delta": residual.cpu(), "rank": rank},
                output_dir / f"stage2_rank{rank}_constrained_residual.pt",
            )
            report = evaluate_stage2_residual(
                args=args,
                model=model,
                tok=tok,
                output_layer=output_layer,
                selected_ids=selected_ids,
                selected_base_rows=selected_base_rows,
                active_ids=active_ids,
                residual_delta=residual,
                rank=rank,
                forget_cases=forget_cases,
                base_forget_logits=base_forget_logits,
                active_case_indices=active_indices,
                protected_case_indices=protected_indices,
                protected_targets=protection_targets,
                utility_hidden=utility_guard_hidden,
                utility_probabilities=utility_guard_probabilities,
                llama_like=llama_like,
                device=device,
            )
            attach_stage2_solver_feasibility(report, best_training)
            core.write_json(
                output_dir / f"stage2_rank{rank}_basis_reports.json", basis_reports
            )
            core.write_json(
                output_dir / f"stage2_rank{rank}_solver_history.json", history
            )
            core.write_json(
                output_dir / f"stage2_rank{rank}_materialized_report.json", report
            )
            stage2_attempts.append(
                {
                    "rank": rank,
                    "continuous_solver": best_training,
                    "materialized": report,
                }
            )
            rank_candidates.append(
                {
                    "rank": rank,
                    "report": report,
                    "residual": residual,
                    "basis_reports": basis_reports,
                    "history": history,
                    "solver_report": best_training,
                }
            )

        core.write_json(output_dir / "stage2_attempts.json", stage2_attempts)
        selected_report = choose_stage2_report(
            [candidate["report"] for candidate in rank_candidates]
        )
        selected_candidate = (
            next(
                candidate
                for candidate in rank_candidates
                if int(candidate["rank"]) == int(selected_report["rank"])
            )
            if selected_report is not None
            else None
        )
        if selected_candidate is None:
            core.write_json(
                output_dir / "stage2_infeasible.json",
                {
                    "method": METHOD,
                    "stage1_selected": dict(selected_stage1_report),
                    "active_direct_token_cases": active_indices,
                    "protected_direct_token_cases": protected_indices,
                    "rank_ladder": list(rank_ladder),
                    "attempts": stage2_attempts,
                    "reason": (
                        "no exact constrained rank candidate repaired every active "
                        "case, preserved every Stage-1 success floor, and passed "
                        "Wikipedia utility guards after checkpoint-dtype materialization"
                    ),
                },
            )
            raise RuntimeError(
                "Stage 2 was infeasible at ranks 2 and 4; the utility-safe "
                "Stage-1 checkpoint was preserved"
            )
        selected_stage2_report = selected_candidate["report"]
        selected_stage2_rank = int(selected_candidate["rank"])
        selected_stage2_residual = selected_candidate["residual"]
        selected_stage2_basis_reports = selected_candidate["basis_reports"]
        selected_stage2_history = selected_candidate["history"]
        selected_stage2_best_training = selected_candidate["solver_report"]
        core.write_json(
            output_dir / "stage2_contrastive_basis_reports.json",
            selected_stage2_basis_reports,
        )
        core.write_json(
            output_dir / "stage2_solver_history.json", selected_stage2_history
        )
        core.write_json(
            output_dir / "stage2_selected_materialized_report.json",
            selected_stage2_report,
        )
        core.materialize_output_delta(
            output_layer, active_ids, selected_stage2_residual
        )
        active_rows_after = (
            output_layer.weight.index_select(0, active_tensor).detach().float()
        )
        actual_stage2_residual = active_rows_after - active_rows_before

    torch.save(
        {"row_ids": active_ids, "delta": actual_stage2_residual.detach().cpu()},
        output_dir / "stage2_residual_delta.pt",
    )
    final_delta = actual_selected_delta(output_layer, selected_ids, selected_base_rows)
    torch.save(
        {"row_ids": selected_ids, "delta": final_delta.detach().cpu()},
        output_dir / "final_total_delta.pt",
    )

    final_state = shared.evaluate_shared_constraints(
        model,
        tok,
        forget_cases,
        base_forget_logits,
        llama_like=llama_like,
        device=device,
        batch_size=args.cache_batch_size,
    )
    final_report = {
        **constraint_report(final_state, args),
        **stage2_partition_report(
            final_state,
            active_indices=active_indices,
            protected_indices=protected_indices,
            protected_targets=protection_targets,
            args=args,
        ),
        **utility_kl_report(
            final_delta,
            utility_guard_hidden,
            utility_guard_probabilities,
            device=device,
            batch_size=args.utility_eval_batch_size,
        ),
        "total_delta_norm": float(final_delta.norm().detach().cpu()),
    }
    add_utility_guards(final_report, args)
    if not bool(final_report["feasible"]):
        raise RuntimeError("Final materialized checkpoint failed a locked safety guard")
    if model.get_input_embeddings().weight.data_ptr() != input_embedding_pointer:
        raise RuntimeError("Input embedding storage changed during sparse training")
    if model.get_input_embeddings().weight.requires_grad:
        raise RuntimeError("Input embeddings unexpectedly became trainable")

    final_forget_gd = evaluate_forget_nonsensitive_kl(
        model,
        tok,
        forget_cases,
        base_forget_logits,
        llama_like=llama_like,
        device=device,
        batch_size=args.cache_batch_size,
    )
    final_utility_drift = utility_logit_drift_mse(final_delta, utility_second_moment)
    final_checkpoint = output_dir / "checkpoint"
    save_checkpoint(model, tok, final_checkpoint)

    config = {
        "schema_version": 5,
        "method": METHOD,
        "protocol": PROTOCOL,
        "source_split_protocol": split_manifest.get("protocol"),
        "dataset_adapter": args.dataset,
        "dataset_adapter_contract": adapter,
        "architecture_core_is_dataset_independent": True,
        "shared_architecture_parameters": shared_architecture,
        "shared_architecture_sha256": architecture_sha256,
        "cross_dataset_compatibility_payload": compatibility_payload,
        "cross_dataset_compatibility_sha256": cross_dataset_compatibility_sha256,
        "seed": int(args.seed),
        "transformer_trainable_parameters": 0,
        "input_embeddings_modified": False,
        "lm_head_untied": True,
        "editable_rows": "sensitive_answer_rows_only",
        "sensitive_answer_field": sensitive_field,
        "replacement_or_reference_target_used": False,
        "internal_direct_constraints_guarantee_official_fs": False,
        "internal_direct_constraints_guarantee_official_gfs": False,
        "official_alignment_reason": (
            "target_new and official paraphrases remain post-training only"
        ),
        "benchmark_retain_train_examples": 0,
        "benchmark_retain_answer_labels_seen": False,
        "official_retain_eval_seen_during_training_or_selection": 0,
        "heldout_probes_seen_during_training_or_selection": 0,
        "ppl_text_seen_during_training_or_selection": 0,
        "wikipedia_utility_examples_are_benchmark_retain": False,
        "utility_cache": str(Path(args.utility_cache).resolve()),
        "utility_cache_metadata": utility_metadata,
        "utility_role": (
            "contrastive_basis_plus_disjoint_token-conditioned Wikipedia "
            "train and checkpoint-guard exact joint sparse KL"
        ),
        "utility_pool_selection": utility_pool_report,
        "utility_train_prompt_count": int(utility_train_hidden.shape[0]),
        "utility_guard_prompt_count": int(utility_guard_hidden.shape[0]),
        "utility_exact_kl_used_as_training_loss": True,
        "contrastive_generalized_eigenproblem": True,
        "contrastive_solver": "full_space_cholesky_whitened_low_rank_svd",
        "rank_ladder": list(rank_ladder),
        "stage1_objective": (
            "bounded direct constraint GA + same-prompt non-sensitive GD + "
            "exact joint Wikipedia KL"
        ),
        "stage2_objective": (
            "minimum exact external-Wikipedia utility KL plus residual L2 subject "
            "to hard active-repair, Stage1-success preservation, utility-tail, "
            "and total-norm inequalities"
        ),
        "stage2_direct_constraint_schedule": (
            "all active and protected direct token cases in every exact solve"
        ),
        "stage2_transformer_forwards_per_optimization_step": 0,
        "stage2_cached_logit_exactness": (
            "algebraically exact FP32 sparse partition/NLL/margin evaluation for a "
            "frozen transformer; checkpoint-dtype materialization is rechecked once "
            "per rank and for the final checkpoint"
        ),
        "stage2_solver": "SLSQP_with_analytic_Torch_gradients",
        "stage2_hard_constraints_tradeable": False,
        "direct_constraint_weight": float(args.direct_constraint_weight),
        "gd_weight": float(args.gd_weight),
        "utility_kl_weight": float(args.utility_kl_weight),
        "stage2_residual_l2_weight": float(args.stage2_residual_l2_weight),
        "stage2_constraint_buffer": float(args.stage2_constraint_buffer),
        "stage2_constraint_tolerance": float(args.stage2_constraint_tolerance),
        "stage2_constraint_basis_weight": float(
            args.stage2_constraint_basis_weight
        ),
        "stage2_protection_nll_tolerance": float(args.stage2_protection_nll_tolerance),
        "constraint_margin": float(args.constraint_margin),
        "min_sensitive_nll_increase": float(args.min_sensitive_nll_increase),
        "utility_guard_budgets": {
            "mean": float(args.utility_kl_mean_budget),
            "p95": float(args.utility_kl_p95_budget),
            "max": float(args.utility_kl_max_budget),
            "total_delta_norm": float(args.max_total_delta_norm),
        },
        "candidate_scales": stage1_scales,
        "stage1_candidate_scales": stage1_scales,
        "stage2_rank_selection_rule": (
            "minimum exact held-out token-conditioned Wikipedia guard KL among "
            "exact-materialized constrained rank candidates repairing all active "
            "cases, preserving every Stage1-success floor, and passing utility "
            "guards; no Stage-2 scale frontier"
        ),
        "stage1_attempts": stage1_attempts,
        "stage1_selected_rank": selected_stage1_rank,
        "stage1_selected": dict(selected_stage1_report),
        "stage1_selection_mode": selected_stage1_mode,
        "stage1_direct_failures_after_materialization": len(active_indices),
        "stage1_failed_direct_token_indices": active_indices,
        "stage1_protected_direct_token_indices": protected_indices,
        "stage2_case_partition_artifact": str(
            (output_dir / "stage2_case_partitions.json").resolve()
        ),
        "stage1_sensitive_case_logits": str(
            (output_dir / "stage1_sensitive_case_logits.pt").resolve()
        ),
        "stage2_mode": stage2_mode,
        "stage2_attempts": stage2_attempts,
        "stage2_active_sensitive_row_ids": active_ids,
        "stage2_selected_rank": selected_stage2_rank,
        "stage2_selected": (
            None if selected_stage2_report is None else dict(selected_stage2_report)
        ),
        "stage2_best_training_report": selected_stage2_best_training,
        "final_guards": final_report,
        "final_same_prompt_non_sensitive_gd_kl": final_forget_gd,
        "final_wikipedia_logit_drift_mse_posthoc": final_utility_drift,
        "post_training_only_metrics": [
            "official FS/GFS",
            "Spe/Spe-success and Spe-margin",
            "official benchmark retain metrics",
            "exact official-retain sparse-row KL",
            "PPL",
        ],
        "stage1_checkpoint": str(stage1_checkpoint.resolve()),
        "final_checkpoint": str(final_checkpoint.resolve()),
        "final_delta": str((output_dir / "final_total_delta.pt").resolve()),
    }
    core.write_json(output_dir / "config_used.json", config)

    print("Guarded two-stage SURE-LM complete:", final_checkpoint)
    print("dataset adapter:", args.dataset)
    print("shared architecture SHA-256:", architecture_sha256)
    print(
        "cross-dataset compatibility SHA-256:",
        cross_dataset_compatibility_sha256,
    )
    print("benchmark retain examples used for training: 0")
    print(
        "Wikipedia candidate/train/guard prompts:",
        actual_utility_prompts,
        int(utility_train_hidden.shape[0]),
        int(utility_guard_hidden.shape[0]),
    )
    print(
        "Stage-1 selected rank/scale:",
        selected_stage1_rank,
        selected_stage1_report["scale"],
    )
    print("Stage-1 residual direct token cases:", len(active_indices))
    print("Stage-1 protected direct token cases:", len(protected_indices))
    print("Stage-2 mode:", stage2_mode)
    if selected_stage2_report is not None:
        print(
            "Stage-2 selected constrained rank:",
            selected_stage2_rank,
        )
    print("Final direct failures:", final_report["direct_failures"])
    print(
        "Final active failures / protected regressions / protected-floor violations:",
        final_report["active_direct_failures"],
        final_report["protected_direct_failures"],
        final_report["protected_floor_violations"],
    )
    print(
        "Final Wikipedia exact KL mean/p95/max:",
        final_report["utility_kl_mean"],
        final_report["utility_kl_p95"],
        final_report["utility_kl_max"],
    )
    print("Final total delta norm:", final_report["total_delta_norm"])
    print("Official benchmark metrics remain post-training only")


if __name__ == "__main__":
    main()
