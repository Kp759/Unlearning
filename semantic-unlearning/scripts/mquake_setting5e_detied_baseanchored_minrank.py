#!/usr/bin/env python3
"""MQuAKE Setting 5e with de-tied, Base-anchored minimal-rank repair.

This is an isolated method extension.  It keeps the established 600-step
Setting 5e trajectory, restores the complete Base input-embedding matrix,
de-ties the output head, and solves a Base-anchored constrained LM-head repair.
Only sampled ``requested_rewrite`` cloze facts are visible before the durable
checkpoint-selection decision.  AtomicGen and multi-hop fields are loaded only
after a candidate passes every fixed selection gate.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

import gagd_compare as gagd
import mquake_gagd_setting5e_active_repair as baseline
import mquake_gagd_setting5e_multiroot_active_repair as vectorized
import mquake_zero_unlearn_official_eval as mquake
import zsre_gagd_setting5e_active_repair as repair
from mcf_zero_unlearn_official_eval import load_official_ppl_text


METHOD = "mquake_setting5e_detied_baseanchored_minrank"
METHOD_LABEL = "Setting 5e + de-tied Base-anchored minimal-rank repair"
SETTING5_MODE = gagd.POST_TRAINING_RESTORE_MODE
ACTIVE_SOURCE = "sampled_requested_rewrite_cloze_teacher_forced_prefixes"
PROTECTED_SOURCE = "base_correct_sampled_requested_rewrite_retain_cloze_states"
HELD_OUT_FIELDS = (
    "requested_rewrite.question",
    "atomic_gen_prompt",
    "record questions[0:3]",
    "answer/new_answer and aliases",
    "standard multi-hop generation",
    "CoT multi-hop generation",
)


@dataclass(frozen=True)
class TokenState:
    """One selection-visible teacher-forced state cached outside the hot loop."""

    case: mquake.PredictionCase
    hidden: torch.Tensor
    target_token_id: int
    predicted_token_id: int
    runner_up_token_id: int
    target_logit: float
    runner_up_logit: float

    @property
    def identity(self) -> Tuple[int, str, int, int]:
        return self.case.identity


@dataclass(frozen=True)
class PairConstraint:
    """A deterministic sensitive/true-current-runner-up constraint."""

    state_identity: Tuple[int, str, int, int]
    sensitive_token_id: int
    competitor_token_id: int
    generation_round: int

    @property
    def identity(self) -> Tuple[Any, ...]:
        return (*self.state_identity, self.sensitive_token_id, self.competitor_token_id)


@dataclass(frozen=True)
class ExactPPLCache:
    """Official PPL states cached under the frozen Base model."""

    hidden: torch.Tensor
    base_logsumexp: torch.Tensor
    base_target_logits: torch.Tensor
    target_token_ids: torch.Tensor
    base_full_logits: torch.Tensor
    normalization_divisor: int
    base_mean_nll: float
    source_sha256: str


@dataclass(frozen=True)
class PreparedPPLTensors:
    hidden_rank: torch.Tensor
    base_logsumexp: torch.Tensor
    base_target_logits: torch.Tensor
    target_selected_row_index: torch.Tensor
    base_selected_logits: torch.Tensor
    normalization_divisor: int


@dataclass(frozen=True)
class SolverPhaseResult:
    coefficients: torch.Tensor
    report: Dict[str, Any]


def _chunks(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor: torch.Tensor, *, chunk_rows: int = 4096) -> str:
    """Hash tensor values without widening BF16/FP16 snapshots."""

    value = tensor.detach()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape)).encode("ascii"))
    if value.ndim == 0:
        chunks = [value.reshape(1)]
    elif value.shape[0] == 0:
        chunks = []
    else:
        chunks = (value[start : start + chunk_rows] for start in range(0, value.shape[0], chunk_rows))
    for chunk in chunks:
        raw = chunk.contiguous().view(torch.uint8).cpu().numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def tensors_equal_chunked(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    chunk_rows: int = 4096,
) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    if left.ndim == 0:
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    for start in range(0, left.shape[0], chunk_rows):
        if not torch.equal(
            left[start : start + chunk_rows].detach().cpu(),
            right[start : start + chunk_rows].detach().cpu(),
        ):
            return False
    return True


@torch.no_grad()
def copy_matrix_chunked_(
    destination: torch.Tensor,
    source: torch.Tensor,
    *,
    chunk_rows: int = 4096,
) -> None:
    if destination.shape != source.shape:
        raise ValueError(f"matrix shape mismatch: {destination.shape} != {source.shape}")
    for start in range(0, destination.shape[0], chunk_rows):
        destination[start : start + chunk_rows].copy_(
            source[start : start + chunk_rows].to(
                device=destination.device, dtype=destination.dtype
            )
        )


def snapshot_base_weight(model: nn.Module) -> Tuple[torch.Tensor, Dict[str, Any]]:
    weight = model.get_input_embeddings().weight.detach().to("cpu").clone()
    report = {
        "dtype": str(weight.dtype),
        "shape": list(weight.shape),
        "sha256": tensor_sha256(weight),
        "input_output_tied": bool(
            model.get_input_embeddings().weight.data_ptr()
            == model.get_output_embeddings().weight.data_ptr()
        ),
    }
    return weight, report


@torch.no_grad()
def detie_restore_base_embeddings(
    model: nn.Module,
    base_weight_cpu: torch.Tensor,
    *,
    chunk_rows: int = 4096,
) -> Dict[str, Any]:
    """Preserve Setting5 logits while restoring the entire Base input matrix."""

    input_layer = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if input_layer is None or output_layer is None:
        raise ValueError("model must expose input and output embeddings")
    tied_before = input_layer.weight.data_ptr() == output_layer.weight.data_ptr()
    pre_detie_output_hash = tensor_sha256(output_layer.weight)
    base_hash = tensor_sha256(base_weight_cpu)

    new_head = nn.Linear(
        int(output_layer.weight.shape[1]),
        int(output_layer.weight.shape[0]),
        bias=False,
        device=output_layer.weight.device,
        dtype=output_layer.weight.dtype,
    )
    copy_matrix_chunked_(new_head.weight, output_layer.weight, chunk_rows=chunk_rows)
    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(new_head)
    elif hasattr(model, "lm_head"):
        model.lm_head = new_head
    else:
        raise ValueError("model cannot install an untied output head")

    copy_matrix_chunked_(input_layer.weight, base_weight_cpu, chunk_rows=chunk_rows)
    if hasattr(model, "config") and hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    final_input = model.get_input_embeddings().weight
    final_output = model.get_output_embeddings().weight
    tied_after = final_input.data_ptr() == final_output.data_ptr()
    restored_input_hash = tensor_sha256(final_input)
    output_hash = tensor_sha256(final_output)
    input_exact = tensors_equal_chunked(final_input, base_weight_cpu)
    output_exact = output_hash == pre_detie_output_hash
    if tied_after:
        raise RuntimeError("de-tied input and output weights still share storage")
    if not input_exact or restored_input_hash != base_hash:
        raise RuntimeError("input embeddings were not restored exactly to Base")
    if not output_exact:
        raise RuntimeError("de-tying changed the pre-detie Setting5 output head")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("model parameters were not frozen after de-tying")

    return {
        "base_embedding_sha256": base_hash,
        "pre_detie_setting5_shared_weight_sha256": pre_detie_output_hash,
        "restored_input_sha256": restored_input_hash,
        "output_head_sha256": output_hash,
        "dtype": str(final_input.dtype),
        "shape": list(final_input.shape),
        "tied_before": bool(tied_before),
        "tied_after": bool(tied_after),
        "input_equals_base_exactly": bool(input_exact),
        "output_equals_pre_detie_setting5_exactly": bool(output_exact),
        "input_output_distinct_pointers": True,
        "transformer_frozen": True,
        "input_embeddings_frozen": not final_input.requires_grad,
        "output_head_frozen_until_sparse_solver": not final_output.requires_grad,
        "config_tie_word_embeddings": bool(
            getattr(getattr(model, "config", None), "tie_word_embeddings", False)
        ),
    }


@torch.no_grad()
def restore_full_output_to_base(
    model: nn.Module,
    base_weight_cpu: torch.Tensor,
    *,
    chunk_rows: int = 4096,
) -> None:
    output = model.get_output_embeddings().weight
    copy_matrix_chunked_(output, base_weight_cpu, chunk_rows=chunk_rows)
    if not tensors_equal_chunked(output, base_weight_cpu, chunk_rows=chunk_rows):
        raise RuntimeError("failed to establish the Base-anchored output origin")


def build_parser() -> argparse.ArgumentParser:
    parser = baseline.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        output_dir="outputs/mquake_detied_baseanchored_minrank/seed0",
        seed=0,
        forget_num=1000,
        retain_num=1000,
        steps=600,
        batch_size=1,
        retain_batch_size=4,
        emb_lm_lr=1e-4,
        forget_weight=2.0,
        retain_weight=1.0,
        forget_margin=1.0,
        emb_lm_optimizer="adamw",
        sampling_strategy="epoch",
        active_logit_margin=0.25,
        selection_logit_margin=0.10,
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
        choices=("instance_balanced", "atomic_epoch"),
        default="instance_balanced",
    )
    parser.add_argument("--repair-rank-start", type=int, default=2)
    parser.add_argument(
        "--repair-rank-max",
        type=int,
        default=0,
        help="0 uses every numerically available ordered active-basis direction.",
    )
    parser.add_argument("--protected-logit-margin", type=float, default=0.0)
    parser.add_argument("--solver-steps-per-phase", type=int, default=1000)
    parser.add_argument("--solver-lr", type=float, default=5e-3)
    parser.add_argument("--solver-rho", type=float, default=1.0)
    parser.add_argument("--solver-feasibility-tolerance", type=float, default=1e-6)
    parser.add_argument("--solver-stall-patience", type=int, default=100)
    parser.add_argument("--solver-min-improvement", type=float, default=1e-7)
    parser.add_argument("--constraint-generation-max-rounds", type=int, default=64)
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
        "target_eff_max": 0.0,
        "utility_drop_tolerance": 0.10,
        "max_ppl_ratio": 1.02,
    }
    for name, expected in pinned.items():
        if getattr(args, name) != expected:
            raise ValueError(f"pinned protocol requires {name}={expected}")
    if args.skip_ppl:
        raise ValueError("the exact Base-relative PPL gate cannot be skipped")
    if not args.strict_utility_gates:
        raise ValueError("strict utility gates are mandatory")
    if args.mquake_url != mquake.MQUAKE_URL:
        raise ValueError("the pinned MQuAKE source revision is mandatory")
    if args.repair_rank_start < 1:
        raise ValueError("repair rank start must be positive")
    if args.repair_rank_max < 0:
        raise ValueError("repair rank max must be non-negative")
    if args.repair_rank_max and args.repair_rank_max < args.repair_rank_start:
        raise ValueError("repair rank max cannot be below repair rank start")
    if args.active_logit_margin != 0.25 or args.selection_logit_margin != 0.10:
        raise ValueError("active and BF16 selection margins are fixed at 0.25/0.10")
    if args.protected_logit_margin != 0.0:
        raise ValueError("Base-correct protected margin is fixed at zero")
    for name in (
        "solver_steps_per_phase",
        "solver_stall_patience",
        "constraint_generation_max_rounds",
        "eval_batch_size",
        "cache_batch_size",
        "multihop_batch_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    for name in ("solver_lr", "solver_rho", "solver_feasibility_tolerance"):
        if float(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")


def _selection_record(raw_record: Mapping[str, Any], source_index: int) -> List[Dict[str, Any]]:
    """Copy cloze-only method fields without reading held-out source fields."""

    rewrites = raw_record.get("requested_rewrite")
    if not isinstance(rewrites, list) or not rewrites:
        raise ValueError(f"MQuAKE instance {source_index} has no requested_rewrite")
    case_id = int(raw_record.get("case_id", source_index))
    result: List[Dict[str, Any]] = []
    for rewrite_index, rewrite in enumerate(rewrites):
        target_true = rewrite.get("target_true")
        if not isinstance(target_true, Mapping) or not str(target_true.get("str", "")):
            raise ValueError("target_true must contain a non-empty sensitive answer")
        result.append(
            {
                "case_id": int(source_index) * 100 + int(rewrite_index),
                "mquake_case_id": case_id,
                "source_index": int(source_index),
                "rewrite_index": int(rewrite_index),
                "requested_rewrite": {
                    "prompt": str(rewrite["prompt"]),
                    "subject": str(rewrite["subject"]),
                    "target_true": {"str": str(target_true["str"])},
                    # Internal neutral convention, not raw benchmark target_new.
                    "target_new": {"str": mquake.NEUTRAL_TARGET},
                },
                "atomic_gen_prompt": "<withheld-until-durable-acceptance>",
            }
        )
    return result


def load_selection_visible_records(
    path: Path,
    tok: Any,
    *,
    forget_num: int,
    retain_num: int,
    seed: int,
    mquake_url: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Source-locked split that never copies questions, aliases, or target_new."""

    mquake.resolve_neutral_target_token_id(tok)
    raw = mquake.load_mquake_raw(path, url=mquake_url)
    forget_instances, retain_instances = mquake.sample_zerounlearn_instances(
        raw, forget_num=forget_num, retain_num=retain_num, seed=seed, strict=True
    )
    flatten = lambda sampled: [
        row
        for source_index, record in sampled
        for row in _selection_record(record, source_index)
    ]
    return flatten(forget_instances), flatten(retain_instances)


def build_repair_cases(
    records: Sequence[Mapping[str, Any]], tok: Any, *, llama_like: bool
) -> List[mquake.PredictionCase]:
    return vectorized.build_repair_cases(records, tok, llama_like=llama_like)


@torch.no_grad()
def cache_token_states(
    model: nn.Module,
    tok: Any,
    cases: Sequence[mquake.PredictionCase],
    *,
    device: torch.device,
    llama_like: bool,
    batch_size: int,
) -> List[TokenState]:
    """Cache hidden states and exact current true-runner-up decisions."""

    model.eval()
    rows: List[TokenState] = []
    for batch in _chunks(list(cases), batch_size):
        encoded = tok([case.prompt for case in batch], padding=True, return_tensors="pt").to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        positions = encoded["attention_mask"].sum(dim=1) - 1
        indices = torch.arange(len(batch), device=device)
        hidden = output.hidden_states[-1][indices, positions, :].float()
        logits = output.logits[indices, positions, :].float()
        targets = mquake.official_target_ids(
            tok,
            [case.target_text for case in batch],
            llama_like=llama_like,
            device=device,
        )
        predictions = logits.argmax(dim=-1)
        competitor_logits = logits.clone()
        competitor_logits.scatter_(1, targets[:, None], -torch.inf)
        runners = competitor_logits.argmax(dim=-1)
        for index, case in enumerate(batch):
            target = int(targets[index].item())
            runner = int(runners[index].item())
            rows.append(
                TokenState(
                    case=case,
                    hidden=hidden[index].detach().cpu(),
                    target_token_id=target,
                    predicted_token_id=int(predictions[index].item()),
                    runner_up_token_id=runner,
                    target_logit=float(logits[index, target].item()),
                    runner_up_logit=float(logits[index, runner].item()),
                )
            )
    return rows


def hidden_state_reproduction_report(
    base_states: Sequence[TokenState], detied_states: Sequence[TokenState]
) -> Dict[str, Any]:
    base = {state.identity: state for state in base_states}
    detied = {state.identity: state for state in detied_states}
    if set(base) != set(detied):
        raise RuntimeError("Base and de-tied state identities differ")
    differences = [
        (base[key].hidden.float() - detied[key].hidden.float()).abs().max()
        for key in sorted(base)
    ]
    maximum = max((float(value.item()) for value in differences), default=0.0)
    exact = all(torch.equal(base[key].hidden, detied[key].hidden) for key in base)
    return {
        "state_count": len(base),
        "exactly_equal": bool(exact),
        "maximum_absolute_difference": maximum,
    }


def canonicalize_basis_signs(basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return basis
    result = basis.clone()
    pivots = result.abs().argmax(dim=1)
    signs = result[torch.arange(result.shape[0], device=result.device), pivots].sign()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return result * signs[:, None]


def ordered_orthonormal_row_basis(
    rows: torch.Tensor, *, max_rank: int = 0
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Create one deterministic ordered SVD basis; every rank uses a prefix."""

    if rows.ndim != 2 or not rows.numel():
        raise ValueError("ordered basis requires a non-empty 2-D hidden matrix")
    values = rows.float()
    _, singular_values, right = torch.linalg.svd(values, full_matrices=False)
    tolerance = (
        max(values.shape)
        * torch.finfo(values.dtype).eps
        * singular_values.max().clamp_min(1.0)
    )
    numerical_rank = int((singular_values > tolerance).sum().item())
    selected_rank = numerical_rank if max_rank == 0 else min(numerical_rank, max_rank)
    basis = canonicalize_basis_signs(right[:selected_rank]).contiguous()
    gram = basis @ basis.T
    prefix_nesting_verified = bool(
        torch.allclose(gram, torch.eye(selected_rank, device=gram.device), atol=1e-5, rtol=1e-5)
    )
    report = {
        "numerical_rank": numerical_rank,
        "available_rank": selected_rank,
        "ordered_basis_sha256": tensor_sha256(basis),
        "prefix_nesting_verified": prefix_nesting_verified,
        "construction": "single FP32 SVD with deterministic sign canonicalization",
    }
    if not prefix_nesting_verified:
        raise RuntimeError("ordered active basis is not orthonormal")
    return basis, report


class BaseAnchoredLowRankRows(nn.Module):
    """Selected output rows defined exactly as W_base + A @ B_k."""

    def __init__(
        self,
        n_rows: int,
        basis_prefix: torch.Tensor,
        *,
        device: torch.device,
        initial_coefficients: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        if basis_prefix.ndim != 2 or basis_prefix.shape[0] < 1:
            raise ValueError("a non-empty ordered basis prefix is required")
        self.register_buffer("basis", basis_prefix.to(device=device, dtype=torch.float32))
        initial = torch.zeros((n_rows, basis_prefix.shape[0]), device=device, dtype=torch.float32)
        if initial_coefficients is not None:
            if initial.shape != initial_coefficients.shape:
                raise ValueError("initial coefficient shape mismatch")
            initial.copy_(initial_coefficients.to(device=device, dtype=torch.float32))
        self.coefficients = nn.Parameter(initial)

    def effective_delta(self) -> torch.Tensor:
        return self.coefficients @ self.basis

    def candidate_rows(self, base_rows: torch.Tensor) -> torch.Tensor:
        return base_rows.float() + self.effective_delta()


def project_setting5_initialization(
    setting5_rows: torch.Tensor, base_rows: torch.Tensor, basis_prefix: torch.Tensor
) -> torch.Tensor:
    gram = basis_prefix.float() @ basis_prefix.float().T
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    if not torch.allclose(gram, identity, atol=1e-5, rtol=1e-5):
        raise ValueError("least-squares fast projection requires an orthonormal basis")
    return (setting5_rows.float() - base_rows.float()) @ basis_prefix.float().T


def expand_rank_coefficients(coefficients: torch.Tensor, new_rank: int) -> torch.Tensor:
    if new_rank < coefficients.shape[1]:
        raise ValueError("rank expansion cannot shrink coefficients")
    expanded = coefficients.new_zeros((coefficients.shape[0], new_rank))
    expanded[:, : coefficients.shape[1]].copy_(coefficients)
    return expanded


def expand_row_coefficients(
    old_row_ids: Sequence[int],
    new_row_ids: Sequence[int],
    coefficients: torch.Tensor,
) -> torch.Tensor:
    if sorted(set(new_row_ids)) != list(new_row_ids):
        raise ValueError("new row IDs must be sorted and unique")
    lookup = {int(token_id): index for index, token_id in enumerate(old_row_ids)}
    expanded = coefficients.new_zeros((len(new_row_ids), coefficients.shape[1]))
    for new_index, token_id in enumerate(new_row_ids):
        old_index = lookup.get(int(token_id))
        if old_index is not None:
            expanded[new_index].copy_(coefficients[old_index])
    return expanded


def constraints_sha256(constraints: Sequence[PairConstraint]) -> str:
    return _json_sha256([list(constraint.identity) for constraint in constraints])


def constraints_from_audit(
    states: Sequence[TokenState],
    *,
    margin: float,
    generation_round: int,
) -> List[PairConstraint]:
    constraints = []
    for state in states:
        if state.runner_up_logit - state.target_logit < margin:
            constraints.append(
                PairConstraint(
                    state_identity=state.identity,
                    sensitive_token_id=state.target_token_id,
                    competitor_token_id=state.runner_up_token_id,
                    generation_round=generation_round,
                )
            )
    return sorted(constraints, key=lambda item: item.identity)


def merge_constraints(
    existing: Sequence[PairConstraint], additions: Sequence[PairConstraint]
) -> List[PairConstraint]:
    merged = {constraint.identity: constraint for constraint in existing}
    for constraint in additions:
        merged.setdefault(constraint.identity, constraint)
    return [merged[key] for key in sorted(merged)]


def selected_row_ids(constraints: Sequence[PairConstraint]) -> List[int]:
    return sorted(
        {
            token_id
            for constraint in constraints
            for token_id in (constraint.sensitive_token_id, constraint.competitor_token_id)
        }
    )


def prepare_active_rank_tensors(
    constraints: Sequence[PairConstraint],
    states: Mapping[Tuple[int, str, int, int], TokenState],
    row_ids: Sequence[int],
    base_weight_cpu: torch.Tensor,
    basis_prefix: torch.Tensor,
    *,
    device: torch.device,
) -> vectorized.ActivePairTensors:
    row_lookup = {int(token_id): index for index, token_id in enumerate(row_ids)}
    hidden = torch.stack([states[item.state_identity].hidden.float() for item in constraints]).to(device)
    sensitive_ids = torch.tensor([item.sensitive_token_id for item in constraints], device=device)
    competitor_ids = torch.tensor([item.competitor_token_id for item in constraints], device=device)
    sensitive_rows = base_weight_cpu.index_select(0, sensitive_ids.cpu()).to(device=device, dtype=torch.float32)
    competitor_rows = base_weight_cpu.index_select(0, competitor_ids.cpu()).to(device=device, dtype=torch.float32)
    base_margin = (hidden * (competitor_rows - sensitive_rows)).sum(dim=1)
    sensitive_index = torch.tensor([row_lookup[int(value)] for value in sensitive_ids.tolist()], device=device)
    competitor_index = torch.tensor([row_lookup[int(value)] for value in competitor_ids.tolist()], device=device)
    return vectorized.ActivePairTensors(
        hidden=hidden,
        base_margin=base_margin,
        sensitive_row_index=sensitive_index.long(),
        competitor_row_index=competitor_index.long(),
        hidden_rank=hidden @ basis_prefix.T,
    )


def prepare_base_protected_tensors(
    states: Sequence[TokenState],
    row_ids: Sequence[int],
    base_weight_cpu: torch.Tensor,
    basis_prefix: torch.Tensor,
    *,
    device: torch.device,
) -> vectorized.ProtectedPairTensors:
    hidden_size = int(base_weight_cpu.shape[1])
    if not states:
        empty_hidden = torch.empty((0, hidden_size), device=device)
        return vectorized.ProtectedPairTensors(
            hidden=empty_hidden,
            base_modified_logits=torch.empty((0, len(row_ids)), device=device),
            correct_base=torch.empty((0,), device=device),
            correct_modified_row_index=torch.empty((0,), dtype=torch.long, device=device),
            competitor_mask=torch.empty((0, len(row_ids)), dtype=torch.bool, device=device),
            hidden_rank=empty_hidden @ basis_prefix.T,
        )
    hidden = torch.stack([state.hidden.float() for state in states]).to(device)
    row_tensor_cpu = torch.tensor(row_ids, dtype=torch.long)
    base_rows = base_weight_cpu.index_select(0, row_tensor_cpu).to(device=device, dtype=torch.float32)
    base_modified = hidden @ base_rows.T
    targets = torch.tensor([state.target_token_id for state in states], device=device)
    target_rows = base_weight_cpu.index_select(0, targets.cpu()).to(device=device, dtype=torch.float32)
    correct_base = (hidden * target_rows).sum(dim=1)
    lookup = {int(token_id): index for index, token_id in enumerate(row_ids)}
    correct_index = torch.tensor([lookup.get(int(value), -1) for value in targets.tolist()], device=device)
    mask = torch.ones((len(states), len(row_ids)), dtype=torch.bool, device=device)
    valid = correct_index >= 0
    if valid.any():
        mask[torch.arange(len(states), device=device)[valid], correct_index[valid]] = False
    return vectorized.ProtectedPairTensors(
        hidden=hidden,
        base_modified_logits=base_modified,
        correct_base=correct_base,
        correct_modified_row_index=correct_index.long(),
        competitor_mask=mask,
        hidden_rank=hidden @ basis_prefix.T,
    )


def prepare_base_protected_runner_up_logits(
    states: Sequence[TokenState],
    base_weight_cpu: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    """Base runner-up anchors protect a modified correct row from moving down."""

    if not states:
        return torch.empty((0,), dtype=torch.float32, device=device)
    hidden = torch.stack([state.hidden.float() for state in states]).to(device)
    runner_ids = torch.tensor(
        [state.runner_up_token_id for state in states], dtype=torch.long
    )
    runner_rows = base_weight_cpu.index_select(0, runner_ids).to(
        device=device, dtype=torch.float32
    )
    return (hidden * runner_rows).sum(dim=1)


def base_protected_margins_from_delta_logits(
    tensors: vectorized.ProtectedPairTensors,
    delta_logits: torch.Tensor,
    runner_up_base_logits: Optional[torch.Tensor],
) -> torch.Tensor:
    """Protect against every modified row and the Base runner-up decision."""

    selected_margins = vectorized.protected_pair_margins_from_delta_logits(
        tensors, delta_logits
    )
    if runner_up_base_logits is None:
        return selected_margins
    if runner_up_base_logits.shape != tensors.correct_base.shape:
        raise ValueError("protected runner-up logit shape mismatch")
    indices = tensors.correct_modified_row_index
    valid = indices >= 0
    safe = indices.clamp_min(0)
    if delta_logits.shape[1]:
        target_delta = delta_logits.gather(1, safe[:, None]).squeeze(1)
        target_delta = torch.where(valid, target_delta, torch.zeros_like(target_delta))
    else:
        target_delta = torch.zeros_like(tensors.correct_base)
    runner_margins = tensors.correct_base + target_delta - runner_up_base_logits
    return torch.cat((selected_margins, runner_margins))


@torch.no_grad()
def build_exact_ppl_cache(
    base_model: nn.Module,
    tok: Any,
    text: str,
    *,
    device: torch.device,
    max_input_length: int = 100,
) -> ExactPPLCache:
    encoded = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to(device)
    output = base_model(**encoded, output_hidden_states=True, use_cache=False)
    logits = output.logits[0, :-1, :]
    hidden = output.hidden_states[-1][0, :-1, :]
    targets = encoded["input_ids"][0, 1:]
    logits_fp32 = logits.float()
    base_lse = torch.logsumexp(logits_fp32, dim=-1)
    base_target = logits_fp32.gather(1, targets[:, None]).squeeze(1)
    divisor = int(encoded["input_ids"].shape[1])
    base_mean_nll = float(((base_lse - base_target).sum() / divisor).item())
    return ExactPPLCache(
        hidden=hidden.detach().float().cpu(),
        base_logsumexp=base_lse.detach().cpu(),
        base_target_logits=base_target.detach().cpu(),
        target_token_ids=targets.detach().cpu(),
        base_full_logits=logits.detach().cpu(),
        normalization_divisor=divisor,
        base_mean_nll=base_mean_nll,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def prepare_ppl_tensors(
    cache: ExactPPLCache,
    row_ids: Sequence[int],
    basis_prefix: torch.Tensor,
    *,
    device: torch.device,
) -> PreparedPPLTensors:
    rows = torch.tensor(row_ids, dtype=torch.long)
    lookup = {int(token_id): index for index, token_id in enumerate(row_ids)}
    target_index = torch.tensor(
        [lookup.get(int(token_id), -1) for token_id in cache.target_token_ids.tolist()],
        dtype=torch.long,
        device=device,
    )
    return PreparedPPLTensors(
        hidden_rank=cache.hidden.to(device) @ basis_prefix.T,
        base_logsumexp=cache.base_logsumexp.to(device),
        base_target_logits=cache.base_target_logits.to(device),
        target_selected_row_index=target_index,
        base_selected_logits=cache.base_full_logits.index_select(1, rows).to(device=device, dtype=torch.float32),
        normalization_divisor=cache.normalization_divisor,
    )


@torch.no_grad()
def direct_official_mean_nll(
    model: nn.Module,
    tok: Any,
    text: str,
    *,
    device: torch.device,
    max_input_length: int = 100,
) -> float:
    """Exact BF16-model counterpart of the official perplexity evaluator."""

    encoded = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to(device)
    logits = model(**encoded, use_cache=False).logits[:, :-1, :]
    target_ids = encoded["input_ids"][:, 1:]
    # Preserve the evaluator's native-dtype log_softmax convention exactly.
    target_log_probs = torch.log_softmax(logits, dim=-1).gather(
        2, target_ids[:, :, None]
    )
    return float(
        (-target_log_probs.sum() / encoded["input_ids"].shape[1]).item()
    )


def exact_selected_row_mean_nll_from_delta_logits(
    *,
    base_logsumexp: torch.Tensor,
    base_target_logits: torch.Tensor,
    base_selected_logits: torch.Tensor,
    candidate_selected_logits: torch.Tensor,
    target_selected_row_index: torch.Tensor,
    normalization_divisor: int,
) -> torch.Tensor:
    """Exact partition update when only selected vocabulary rows change."""

    if base_selected_logits.shape != candidate_selected_logits.shape:
        raise ValueError("selected-logit shapes differ")
    selected_base_mass = torch.exp(base_selected_logits - base_logsumexp[:, None]).sum(dim=1)
    eps = torch.finfo(base_logsumexp.dtype).eps
    log_unselected = base_logsumexp + torch.log1p(-selected_base_mass.clamp(max=1.0 - eps))
    selected_candidate_lse = torch.logsumexp(candidate_selected_logits, dim=1)
    candidate_lse = torch.logaddexp(log_unselected, selected_candidate_lse)
    valid = target_selected_row_index >= 0
    safe = target_selected_row_index.clamp_min(0)
    gathered = candidate_selected_logits.gather(1, safe[:, None]).squeeze(1)
    candidate_target = torch.where(valid, gathered, base_target_logits)
    return (candidate_lse - candidate_target).sum() / float(normalization_divisor)


def exact_ppl_mean_nll_from_coefficients(
    tensors: PreparedPPLTensors, coefficients: torch.Tensor
) -> torch.Tensor:
    delta_logits = tensors.hidden_rank @ coefficients.T
    return exact_selected_row_mean_nll_from_delta_logits(
        base_logsumexp=tensors.base_logsumexp,
        base_target_logits=tensors.base_target_logits,
        base_selected_logits=tensors.base_selected_logits,
        candidate_selected_logits=tensors.base_selected_logits + delta_logits,
        target_selected_row_index=tensors.target_selected_row_index,
        normalization_divisor=tensors.normalization_divisor,
    )


def ppl_ratio_gate_equivalent(
    base_mean_nll: float, candidate_mean_nll: float, max_ratio: float
) -> bool:
    return candidate_mean_nll <= base_mean_nll + math.log(max_ratio)


def _max_violation(values: torch.Tensor) -> torch.Tensor:
    if not values.numel():
        return values.new_zeros(())
    return values.clamp_min(0).max()


def solve_vectorized_phase(
    *,
    active_tensors: vectorized.ActivePairTensors,
    protected_tensors: vectorized.ProtectedPairTensors,
    protected_runner_up_base_logits: Optional[torch.Tensor] = None,
    ppl_tensors: PreparedPPLTensors,
    basis_prefix: torch.Tensor,
    initial_coefficients: torch.Tensor,
    active_margin: float,
    protected_margin: float,
    allowed_mean_nll: float,
    steps: int,
    learning_rate: float,
    rho: float,
    tolerance: float,
    stall_patience: int,
    min_improvement: float,
) -> SolverPhaseResult:
    """Deterministic coefficient-space primal-dual feasibility solver."""

    device = initial_coefficients.device
    module = BaseAnchoredLowRankRows(
        initial_coefficients.shape[0],
        basis_prefix,
        device=device,
        initial_coefficients=initial_coefficients,
    )
    # Keep the numerical solver self-contained and deterministic.  Explicit
    # Adam state also avoids allocating optimizer state for any full vocabulary
    # matrix: only the [selected_rows, rank] coefficients have moments.
    adam_first = torch.zeros_like(module.coefficients)
    adam_second = torch.zeros_like(module.coefficients)
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-8
    active_dual = torch.zeros(active_tensors.base_margin.shape, device=device)
    protected_count = int(protected_tensors.competitor_mask.sum().item()) + (
        0
        if protected_runner_up_base_logits is None
        else int(protected_runner_up_base_logits.numel())
    )
    protected_dual = torch.zeros((protected_count,), device=device)
    ppl_dual = torch.zeros((), device=device)
    best = math.inf
    stalled = 0
    reason = "maximum_steps_reached"
    start = time.monotonic()
    last_values: Dict[str, float] = {}

    with torch.no_grad():
        initial_active_margins = vectorized.active_pair_margins_from_coefficients(
            active_tensors, module.coefficients
        )
        initial_protected_delta = (
            vectorized.protected_delta_logits_from_coefficients(
                protected_tensors, module.coefficients
            )
        )
        initial_protected_margins = base_protected_margins_from_delta_logits(
            protected_tensors,
            initial_protected_delta,
            protected_runner_up_base_logits,
        )
        initial_nll = exact_ppl_mean_nll_from_coefficients(
            ppl_tensors, module.coefficients
        )
        initial_active_g = float(active_margin) - initial_active_margins
        initial_protected_g = float(protected_margin) - initial_protected_margins
        initial_values = torch.stack(
            (
                (initial_active_g > tolerance).sum().float(),
                (initial_protected_g > tolerance).sum().float(),
                initial_active_g.clamp_min(0).max()
                if initial_active_g.numel()
                else initial_nll.new_zeros(()),
                initial_protected_g.clamp_min(0).max()
                if initial_protected_g.numel()
                else initial_nll.new_zeros(()),
                (initial_nll - allowed_mean_nll).clamp_min(0),
            )
        ).cpu().tolist()

    for step in range(1, steps + 1):
        module.coefficients.grad = None
        coefficients = module.coefficients
        active_margins = vectorized.active_pair_margins_from_coefficients(
            active_tensors, coefficients
        )
        protected_delta_logits = vectorized.protected_delta_logits_from_coefficients(
            protected_tensors, coefficients
        )
        protected_margins = base_protected_margins_from_delta_logits(
            protected_tensors,
            protected_delta_logits,
            protected_runner_up_base_logits,
        )
        candidate_nll = exact_ppl_mean_nll_from_coefficients(ppl_tensors, coefficients)

        active_g = float(active_margin) - active_margins
        protected_g = float(protected_margin) - protected_margins
        ppl_g = candidate_nll - float(allowed_mean_nll)
        active_pos = active_g.clamp_min(0)
        protected_pos = protected_g.clamp_min(0)
        ppl_pos = ppl_g.clamp_min(0)
        active_term = (
            (active_dual * active_pos + 0.5 * rho * active_pos.square()).mean()
            if active_pos.numel()
            else coefficients.new_zeros(())
        )
        protected_term = (
            (protected_dual * protected_pos + 0.5 * rho * protected_pos.square()).mean()
            if protected_pos.numel()
            else coefficients.new_zeros(())
        )
        objective = (
            0.5 * coefficients.square().sum()
            + active_term
            + protected_term
            + ppl_dual * ppl_pos
            + 0.5 * rho * ppl_pos.square()
        )
        if not torch.isfinite(objective):
            raise RuntimeError(f"non-finite constrained objective at step {step}")
        objective.backward()
        if module.coefficients.grad is None:
            raise RuntimeError("coefficient-space solver produced no gradient")
        with torch.no_grad():
            gradient = module.coefficients.grad
            adam_first.mul_(adam_beta1).add_(gradient, alpha=1.0 - adam_beta1)
            adam_second.mul_(adam_beta2).addcmul_(
                gradient, gradient, value=1.0 - adam_beta2
            )
            bias_first = 1.0 - adam_beta1**step
            bias_second = 1.0 - adam_beta2**step
            denominator = adam_second.sqrt().div_(math.sqrt(bias_second)).add_(
                adam_epsilon
            )
            module.coefficients.addcdiv_(
                adam_first, denominator, value=-learning_rate / bias_first
            )

        with torch.no_grad():
            active_dual.add_(rho * active_g.detach()).clamp_(min=0)
            if protected_dual.numel():
                protected_dual.add_(rho * protected_g.detach()).clamp_(min=0)
            ppl_dual.add_(rho * ppl_g.detach()).clamp_(min=0)
            active_max = _max_violation(active_g)
            protected_max = _max_violation(protected_g)
            ppl_excess = ppl_g.clamp_min(0)
            active_count = (active_g > tolerance).sum().float()
            protected_violations = (protected_g > tolerance).sum().float()
            combined = torch.stack(
                (
                    active_max,
                    protected_max,
                    ppl_excess,
                    active_count,
                    protected_violations,
                    candidate_nll,
                )
            )
            # One synchronization supplies exact feasibility and stall semantics.
            (
                active_value,
                protected_value,
                ppl_value,
                active_count_value,
                protected_count_value,
                candidate_nll_value,
            ) = combined.detach().cpu().tolist()
            current = max(active_value, protected_value, ppl_value)
            if current + min_improvement < best:
                best = current
                stalled = 0
            else:
                stalled += 1
            last_values = {
                "max_active_violation": float(active_value),
                "max_protected_violation": float(protected_value),
                "ppl_nll_excess": float(ppl_value),
                "active_violations": int(active_count_value),
                "protected_violations": int(protected_count_value),
                "candidate_mean_nll": float(candidate_nll_value),
            }
            if current <= tolerance:
                reason = "exact_configured_feasibility_reached"
                break
            if stalled >= stall_patience:
                reason = "deterministic_stall_criterion_reached"
                break

    coefficients = module.coefficients.detach()
    report = {
        "rank": int(basis_prefix.shape[0]),
        "steps": int(step),
        "convergence_or_stall_reason": reason,
        "configured_feasible": bool(
            max(
                last_values.get("max_active_violation", math.inf),
                last_values.get("max_protected_violation", math.inf),
                last_values.get("ppl_nll_excess", math.inf),
            )
            <= tolerance
        ),
        **last_values,
        "delta_norm": float(coefficients.norm().cpu()),
        "active_violations_before": int(initial_values[0]),
        "protected_violations_before": int(initial_values[1]),
        "max_active_violation_before": float(initial_values[2]),
        "max_protected_violation_before": float(initial_values[3]),
        "PPL_NLL_excess_before": float(initial_values[4]),
        "wall_clock_seconds": time.monotonic() - start,
        "hot_path": {
            "python_active_pair_loops_per_step": 0,
            "python_protected_state_loops_per_step": 0,
            "coefficient_space_shape": list(coefficients.shape),
            "full_hidden_delta_materialized_per_step": False,
        },
    }
    return SolverPhaseResult(coefficients=coefficients, report=report)


@torch.no_grad()
def materialize_base_anchored_rows(
    output_weight: torch.Tensor,
    row_ids: Sequence[int],
    base_weight_cpu: torch.Tensor,
    delta_rows: torch.Tensor,
) -> None:
    row_tensor = torch.tensor(row_ids, dtype=torch.long, device=output_weight.device)
    base_rows = base_weight_cpu.index_select(0, row_tensor.cpu()).to(
        device=output_weight.device, dtype=torch.float32
    )
    candidate = (base_rows + delta_rows.to(output_weight.device)).to(output_weight.dtype)
    output_weight.index_copy_(0, row_tensor, candidate)


def exact_local_gate_report(
    *,
    audit_states: Sequence[TokenState],
    constraints: Sequence[PairConstraint],
    protected_states: Sequence[TokenState],
    candidate_mean_nll: float,
    direct_candidate_mean_nll: Optional[float],
    allowed_mean_nll: float,
    direct_allowed_mean_nll: float,
    selection_margin: float,
) -> Dict[str, Any]:
    by_identity = {state.identity: state for state in audit_states}
    pair_violations = 0
    for constraint in constraints:
        state = by_identity[constraint.state_identity]
        # Dynamic audit uses the true current runner-up.  A fixed pair can only
        # be weaker than the current runner-up when its competitor changed.
        if state.runner_up_logit - state.target_logit < selection_margin:
            pair_violations += 1
    active_top1 = sum(
        state.predicted_token_id == state.target_token_id for state in audit_states
    )
    protected_regressions = sum(
        state.predicted_token_id != state.target_token_id for state in protected_states
    )
    checks = {
        "zero_sensitive_top1": active_top1 == 0,
        "zero_active_pair_margin_violations": pair_violations == 0,
        "zero_incremental_protected_regressions": protected_regressions == 0,
        "cached_ppl_within_base_ratio": candidate_mean_nll <= allowed_mean_nll,
        "direct_BF16_ppl_within_base_ratio": (
            direct_candidate_mean_nll is not None
            and direct_candidate_mean_nll <= direct_allowed_mean_nll
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "active_sensitive_top1": int(active_top1),
        "active_pair_margin_violations": int(pair_violations),
        "protected_incremental_regressions": int(protected_regressions),
        "candidate_cached_mean_nll": float(candidate_mean_nll),
        "candidate_direct_BF16_mean_nll": direct_candidate_mean_nll,
        "cached_vs_direct_mean_nll_absolute_difference": (
            None
            if direct_candidate_mean_nll is None
            else abs(float(candidate_mean_nll) - float(direct_candidate_mean_nll))
        ),
        "allowed_mean_nll": float(allowed_mean_nll),
        "direct_BF16_allowed_mean_nll": float(direct_allowed_mean_nll),
    }


def aggregate_rank_attempts(
    phase_reports: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Emit the publication-facing one-row summary for each attempted rank."""

    ranks = sorted({int(report["rank"]) for report in phase_reports})
    result: List[Dict[str, Any]] = []
    for rank in ranks:
        phases = [report for report in phase_reports if int(report["rank"]) == rank]
        first = phases[0]
        last = phases[-1]
        result.append(
            {
                "rank": rank,
                "initialization": first.get("initialization"),
                "steps": sum(int(report.get("steps", 0)) for report in phases),
                "active_violations_before": first.get("active_violations_before"),
                "active_violations_after": last.get("active_violations"),
                "protected_violations_before": first.get(
                    "protected_violations_before"
                ),
                "protected_violations_after": last.get("protected_violations"),
                "PPL_NLL_excess_before": first.get("PPL_NLL_excess_before"),
                "PPL_NLL_excess_after": last.get("ppl_nll_excess"),
                "max_active_violation": last.get("max_active_violation"),
                "max_protected_violation": last.get("max_protected_violation"),
                "delta_norm": last.get("delta_norm"),
                "convergence_or_stall_reason": last.get(
                    "convergence_or_stall_reason"
                ),
                "wall_clock_seconds": sum(
                    float(report.get("wall_clock_seconds", 0.0)) for report in phases
                ),
                "constraint_generation_phase_count": len(phases),
            }
        )
    return result


def write_rank_continuation_diagnostics(
    path: Path,
    *,
    rank_start: int,
    rank_max_resolved: int,
    first_local_feasible_rank: Optional[int],
    basis_report: Mapping[str, Any],
    phase_reports: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    payload = {
        "terminology": "first rank reaching exact configured feasibility",
        "mathematical_rank_infeasibility_claimed": False,
        "rank_start": int(rank_start),
        "rank_max_resolved": int(rank_max_resolved),
        "first_local_exact_feasible_rank": first_local_feasible_rank,
        "ordered_basis": dict(basis_report),
        "attempts": aggregate_rank_attempts(phase_reports),
        "phase_details": list(phase_reports),
    }
    gagd.write_json(path, payload)
    return payload


def final_acceptance_report(
    base_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    local_report: Mapping[str, Any],
    *,
    utility_drop_tolerance: float,
    max_ppl_ratio: float,
) -> Dict[str, Any]:
    checks = {
        "forget_Eff_exactly_zero": float(candidate_result["forget"]["Eff"]) == 0.0,
        "retain_Eff_within_Base_tolerance": float(candidate_result["retain"]["Eff"])
        >= float(base_result["retain"]["Eff"]) - utility_drop_tolerance,
        "PPL_within_Base_ratio": float(candidate_result["forget_PPL"])
        <= float(base_result["forget_PPL"]) * max_ppl_ratio,
        "zero_incremental_protected_regressions": int(
            local_report["protected_incremental_regressions"]
        )
        == 0,
        "exact_BF16_local_feasibility": bool(local_report["passed"]),
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "forget_Eff": 0.0,
            "retain_Eff_minimum": float(base_result["retain"]["Eff"])
            - utility_drop_tolerance,
            "PPL_maximum": float(base_result["forget_PPL"]) * max_ppl_ratio,
            "protected_incremental_regressions": 0,
        },
    }


def save_detied_checkpoint(model: nn.Module, tok: Any, path: Path) -> None:
    if model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr():
        raise RuntimeError("refusing to save a re-tied candidate")
    if getattr(model.config, "tie_word_embeddings", True):
        raise RuntimeError("config would re-tie the checkpoint on reload")
    baseline.save_checkpoint(model, tok, path)


def stage_metrics(result: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "Eff": result["forget"].get("Eff"),
        "RetainEff": result["retain"].get("Eff"),
        "PPL": result.get("forget_PPL"),
    }


def _write_preselection_result(
    output_dir: Path,
    *,
    base_result: Mapping[str, Any],
    tied_result: Mapping[str, Any],
    detied_result: Mapping[str, Any],
    candidate_result: Mapping[str, Any],
    selected_result: Mapping[str, Any],
    repair_report: Mapping[str, Any],
    selection: Mapping[str, Any],
) -> None:
    gagd.write_json(
        output_dir / "mquake_results.json",
        {
            "method": METHOD_LABEL,
            "Base": stage_metrics(base_result),
            "Setting5e_tied": stage_metrics(tied_result),
            "Setting5e_detied": stage_metrics(detied_result),
            "Candidate": stage_metrics(candidate_result),
            "Selected": stage_metrics(selected_result),
            "repair": repair_report,
            "selection": selection,
            "held_out": {
                "status": (
                    "pending_post_selection_evaluation"
                    if selection["candidate_accepted"]
                    else "not_evaluated_candidate_rejected"
                ),
                "AtomicGen": None,
                "RetainAtomicGen": None,
                "standard_multihop": None,
                "cot_multihop": None,
            },
        },
    )


def evaluate_held_out_after_durable_acceptance(
    *,
    accepted: bool,
    load_records: Any,
    evaluate: Any,
) -> Any:
    """Guard the first held-out read behind the durable acceptance decision."""

    if not accepted:
        raise RuntimeError("held-out MQuAKE fields remain locked for a rejected candidate")
    records = load_records()
    return evaluate(records)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    gagd.set_seed(args.seed)
    gagd.require_cuda_if_needed(args.device_map)

    output_dir = gagd.resolve_output_path(args.output_dir)
    setting5_dir = output_dir / "setting5e"
    detied_dir = output_dir / "detied_setting5e"
    repair_dir = output_dir / "baseanchored_minrank_repair"
    for path in (output_dir, setting5_dir, detied_dir, repair_dir):
        path.mkdir(parents=True, exist_ok=True)
    mquake_path = Path(args.mquake_path)
    if not mquake_path.is_absolute():
        mquake_path = gagd.PROJECT_DIR / mquake_path
    mquake_path = mquake.download_mquake(mquake_path, url=args.mquake_url)
    wikidata_dir = gagd.resolve_output_path(args.wikidata_dir)

    config = vars(args).copy()
    config.update(
        {
            "method": METHOD,
            "method_label": METHOD_LABEL,
            "output_parameterization": "W_candidate[selected] = W_base[selected] + A @ B_rank",
            "base_input_embeddings_restored": True,
            "rank_selection": "first rank reaching exact configured feasibility",
            "ordered_rank_prefixes": True,
            "evaluation_only_until_durable_acceptance": list(HELD_OUT_FIELDS),
            "raw_mquake_target_new_used_for_training": False,
            "automatic_gate_relaxation": False,
        }
    )
    gagd.write_json(output_dir / "config_used.json", config)

    print("Loading Base once, snapshotting its tied matrix, and caching legal states")
    base_model, tok = gagd.load_model_and_tokenizer(args, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    device = next(base_model.parameters()).device
    base_weight_cpu, base_snapshot_report = snapshot_base_weight(base_model)
    gagd.write_json(output_dir / "base_weight_snapshot.json", base_snapshot_report)
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
    neutral_token_id = mquake.resolve_neutral_target_token_id(tok)
    llama_like = mquake.is_llama_like(base_model, tok)
    forget_cases = build_repair_cases(forget_records, tok, llama_like=llama_like)
    calibration_records = vectorized.sample_retain_instances(
        retain_records, args.retain_calibration_num, args.retain_calibration_seed
    )
    retain_cases = build_repair_cases(calibration_records, tok, llama_like=llama_like)
    base_forget_states = cache_token_states(
        base_model, tok, forget_cases, device=device, llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    base_retain_states = cache_token_states(
        base_model, tok, retain_cases, device=device, llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
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
        raise RuntimeError("official WikiData PPL text is required for exact feasibility")
    ppl_cache = build_exact_ppl_cache(base_model, tok, ppl_text, device=device)
    gagd.write_json(
        output_dir / "ppl_cache_report.json",
        {
            "state_count": int(ppl_cache.hidden.shape[0]),
            "normalization_divisor": ppl_cache.normalization_divisor,
            "base_mean_nll": ppl_cache.base_mean_nll,
            "base_ppl_from_cache": math.exp(ppl_cache.base_mean_nll),
            "source_sha256": ppl_cache.source_sha256,
            "official_protocol_max_input_length": 100,
        },
    )
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Training the preserved 600-step Setting 5e trajectory")
    gagd.set_seed(args.seed)
    model, tok = gagd.load_model_and_tokenizer(args, for_training=True)
    device = next(model.parameters()).device
    if tensor_sha256(model.get_input_embeddings().weight) != base_snapshot_report["sha256"]:
        raise RuntimeError("training model does not match the frozen Base snapshot")
    forget_examples, sampling_report = vectorized.setting5_forget_examples(
        forget_records,
        tok,
        strategy=args.forget_sampling,
        steps=args.steps,
        seed=args.seed,
    )
    retain_examples = baseline.canonical_examples(retain_records, tok)
    gagd.write_json(setting5_dir / "forget_sampling.json", sampling_report)
    args.post_training_excluded_token_ids = [neutral_token_id]
    requested_save = bool(args.save_model)
    args.save_model = False
    train_summary = gagd.train_mode(
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
    tied_result = baseline.evaluate_eff_only(
        method="Setting5e tied",
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
    setting5_weight_cpu = model.get_output_embeddings().weight.detach().cpu().clone()

    print("De-tying: exact Base input embeddings plus exact Setting5 output head")
    detie_report = detie_restore_base_embeddings(model, base_weight_cpu)
    gagd.write_json(output_dir / "detie_report.json", detie_report)
    detied_result = baseline.evaluate_eff_only(
        method="Setting5e de-tied with Base input embeddings",
        model=model,
        tok=tok,
        model_dir="in-memory:setting5e-detied",
        mquake_path=mquake_path,
        wikidata_dir=wikidata_dir,
        out_path=detied_dir / "official_eval.json",
        args=args,
        records=preselection_records,
    )
    if args.save_setting5_checkpoint:
        save_detied_checkpoint(model, tok, detied_dir / "checkpoint")

    detied_forget_states = cache_token_states(
        model, tok, forget_cases, device=device, llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    detied_retain_states = cache_token_states(
        model, tok, retain_cases, device=device, llama_like=llama_like,
        batch_size=args.cache_batch_size,
    )
    hidden_report = {
        "forget": hidden_state_reproduction_report(base_forget_states, detied_forget_states),
        "retain": hidden_state_reproduction_report(base_retain_states, detied_retain_states),
    }
    gagd.write_json(detied_dir / "base_hidden_state_reproduction.json", hidden_report)
    if not hidden_report["forget"]["exactly_equal"] or not hidden_report["retain"]["exactly_equal"]:
        raise RuntimeError("restored Base input path did not reproduce Base hidden states exactly")

    print("Resetting the full output matrix to Base before Base-relative repair")
    restore_full_output_to_base(model, base_weight_cpu)
    output_weight = model.get_output_embeddings().weight
    protected_base = {state.identity: state for state in base_retain_states}
    protected_states = [
        state
        for state in detied_retain_states
        if protected_base[state.identity].predicted_token_id
        == protected_base[state.identity].target_token_id
    ]
    state_map = {state.identity: state for state in detied_forget_states}
    constraints = constraints_from_audit(
        detied_forget_states, margin=args.active_logit_margin, generation_round=0
    )
    if not constraints:
        # The fixed gate can already be checked with the Base output origin.
        constraints = []
    all_hidden = torch.stack([state.hidden.float() for state in detied_forget_states]).to(device)
    ordered_basis, basis_report = ordered_orthonormal_row_basis(
        all_hidden, max_rank=args.repair_rank_max
    )
    gagd.write_json(repair_dir / "ordered_basis.json", basis_report)
    maximum_rank = int(ordered_basis.shape[0])
    if maximum_rank < args.repair_rank_start:
        raise RuntimeError("available active hidden rank is below repair_rank_start")
    allowed_mean_nll = ppl_cache.base_mean_nll + math.log(args.max_ppl_ratio)
    direct_allowed_mean_nll = math.log(float(base_result["forget_PPL"])) + math.log(
        args.max_ppl_ratio
    )

    row_ids = selected_row_ids(constraints)
    if not row_ids:
        # A zero-row candidate is Base exactly; retain/PPL are therefore safe.
        row_ids = []
    initial_coefficients: Optional[torch.Tensor] = None
    rank_reports: List[Dict[str, Any]] = []
    constraint_rounds: List[Dict[str, Any]] = []
    selected_rank: Optional[int] = None
    selected_coefficients: Optional[torch.Tensor] = None
    selected_local_report: Optional[Dict[str, Any]] = None

    for rank in range(args.repair_rank_start, maximum_rank + 1):
        basis_prefix = ordered_basis[:rank]
        if initial_coefficients is None:
            if row_ids:
                row_tensor = torch.tensor(row_ids, dtype=torch.long)
                base_rows = base_weight_cpu.index_select(0, row_tensor).to(device=device, dtype=torch.float32)
                setting_rows = setting5_weight_cpu.index_select(0, row_tensor).to(device=device, dtype=torch.float32)
                initial_coefficients = project_setting5_initialization(
                    setting_rows, base_rows, basis_prefix
                )
            else:
                initial_coefficients = torch.zeros((0, rank), device=device)
            initialization = "least_squares_projection_of_setting5_minus_base"
        else:
            initial_coefficients = expand_rank_coefficients(initial_coefficients, rank)
            initialization = "warm_start_previous_rank_new_column_zero"

        for generation_round in range(args.constraint_generation_max_rounds):
            round_hash_before = constraints_sha256(constraints)
            if not constraints:
                phase = SolverPhaseResult(
                    coefficients=initial_coefficients,
                    report={
                        "rank": rank,
                        "steps": 0,
                        "configured_feasible": True,
                        "convergence_or_stall_reason": "no_active_constraints",
                        "max_active_violation": 0.0,
                        "max_protected_violation": 0.0,
                        "ppl_nll_excess": 0.0,
                        "candidate_mean_nll": ppl_cache.base_mean_nll,
                        "delta_norm": 0.0,
                        "wall_clock_seconds": 0.0,
                    },
                )
            else:
                active_tensors = prepare_active_rank_tensors(
                    constraints,
                    state_map,
                    row_ids,
                    base_weight_cpu,
                    basis_prefix,
                    device=device,
                )
                protected_tensors = prepare_base_protected_tensors(
                    protected_states,
                    row_ids,
                    base_weight_cpu,
                    basis_prefix,
                    device=device,
                )
                protected_runner_up_base_logits = (
                    prepare_base_protected_runner_up_logits(
                        protected_states, base_weight_cpu, device=device
                    )
                )
                ppl_tensors = prepare_ppl_tensors(
                    ppl_cache, row_ids, basis_prefix, device=device
                )
                phase = solve_vectorized_phase(
                    active_tensors=active_tensors,
                    protected_tensors=protected_tensors,
                    protected_runner_up_base_logits=(
                        protected_runner_up_base_logits
                    ),
                    ppl_tensors=ppl_tensors,
                    basis_prefix=basis_prefix,
                    initial_coefficients=initial_coefficients,
                    active_margin=args.active_logit_margin,
                    protected_margin=args.protected_logit_margin,
                    allowed_mean_nll=allowed_mean_nll,
                    steps=args.solver_steps_per_phase,
                    learning_rate=args.solver_lr,
                    rho=args.solver_rho,
                    tolerance=args.solver_feasibility_tolerance,
                    stall_patience=args.solver_stall_patience,
                    min_improvement=args.solver_min_improvement,
                )
            initial_coefficients = phase.coefficients
            phase_report = {
                **phase.report,
                "initialization": initialization,
                "constraint_generation_round": generation_round,
                "constraint_count": len(constraints),
                "selected_output_row_count": len(row_ids),
                "constraint_set_sha256_before": round_hash_before,
            }
            rank_reports.append(phase_report)
            # Audit the complete forget/protection banks after every numerical
            # phase, including stalled phases.  Rank may grow only after the
            # active constraint set is stable.
            delta_rows = initial_coefficients @ basis_prefix
            restore_full_output_to_base(model, base_weight_cpu)
            if row_ids:
                materialize_base_anchored_rows(
                    output_weight, row_ids, base_weight_cpu, delta_rows
                )
            candidate_forget_states = cache_token_states(
                model, tok, forget_cases, device=device, llama_like=llama_like,
                batch_size=args.cache_batch_size,
            )
            candidate_retain_states = cache_token_states(
                model, tok, [state.case for state in protected_states],
                device=device, llama_like=llama_like, batch_size=args.cache_batch_size,
            )
            additions = constraints_from_audit(
                candidate_forget_states,
                margin=args.selection_logit_margin,
                generation_round=generation_round + 1,
            )
            merged = merge_constraints(constraints, additions)
            new_row_ids = selected_row_ids(merged)
            new_constraint_count = len(merged) - len(constraints)
            if new_row_ids != row_ids:
                initial_coefficients = expand_row_coefficients(
                    row_ids, new_row_ids, initial_coefficients
                )
            constraints = merged
            row_ids = new_row_ids
            ppl_tensors = prepare_ppl_tensors(
                ppl_cache, row_ids, basis_prefix, device=device
            ) if row_ids else None
            if row_ids:
                materialized = output_weight.index_select(
                    0, torch.tensor(row_ids, dtype=torch.long, device=device)
                ).float()
                base_rows = base_weight_cpu.index_select(
                    0, torch.tensor(row_ids, dtype=torch.long)
                ).to(device=device, dtype=torch.float32)
                materialized_delta = materialized - base_rows
                # Reproject the exact BF16 rows into coefficient-space only for
                # the cached audit; optimization remains on FP32 coefficients.
                candidate_mean_nll = float(
                    exact_selected_row_mean_nll_from_delta_logits(
                        base_logsumexp=ppl_tensors.base_logsumexp,
                        base_target_logits=ppl_tensors.base_target_logits,
                        base_selected_logits=ppl_tensors.base_selected_logits,
                        candidate_selected_logits=(
                            ppl_tensors.base_selected_logits
                            + ppl_cache.hidden.to(device) @ materialized_delta.T
                        ),
                        target_selected_row_index=ppl_tensors.target_selected_row_index,
                        normalization_divisor=ppl_tensors.normalization_divisor,
                    ).item()
                )
            else:
                candidate_mean_nll = ppl_cache.base_mean_nll
            direct_candidate_mean_nll = direct_official_mean_nll(
                model, tok, ppl_text, device=device
            )
            local = exact_local_gate_report(
                audit_states=candidate_forget_states,
                constraints=constraints,
                protected_states=candidate_retain_states,
                candidate_mean_nll=candidate_mean_nll,
                direct_candidate_mean_nll=direct_candidate_mean_nll,
                allowed_mean_nll=allowed_mean_nll,
                direct_allowed_mean_nll=direct_allowed_mean_nll,
                selection_margin=args.selection_logit_margin,
            )
            round_report = {
                "rank": rank,
                "round": generation_round,
                "constraint_set_sha256_before": round_hash_before,
                "constraint_set_sha256_after": constraints_sha256(constraints),
                "new_constraint_count": new_constraint_count,
                "selected_row_count": len(row_ids),
                "BF16_audit": local,
            }
            constraint_rounds.append(round_report)
            if new_constraint_count > 0:
                initialization = "same_rank_warm_start_after_constraint_generation"
                continue
            if phase.report["configured_feasible"] and local["passed"]:
                selected_rank = rank
                selected_coefficients = initial_coefficients.detach().clone()
                selected_local_report = local
                break
            # The constraint set is now stable.  A stalled/infeasible phase or
            # an exact BF16/protection/PPL failure advances only to rank+1.
            break
        if selected_rank is not None:
            break

    write_rank_continuation_diagnostics(
        repair_dir / "rank_continuation.json",
        rank_start=args.repair_rank_start,
        rank_max_resolved=maximum_rank,
        first_local_feasible_rank=selected_rank,
        basis_report=basis_report,
        phase_reports=rank_reports,
    )
    gagd.write_json(repair_dir / "constraint_generation.json", constraint_rounds)

    if selected_rank is not None and selected_coefficients is not None:
        basis_prefix = ordered_basis[:selected_rank]
        delta_rows = selected_coefficients @ basis_prefix
        restore_full_output_to_base(model, base_weight_cpu)
        if row_ids:
            materialize_base_anchored_rows(output_weight, row_ids, base_weight_cpu, delta_rows)
        candidate_result = baseline.evaluate_eff_only(
            method=METHOD_LABEL + " candidate",
            model=model,
            tok=tok,
            model_dir="in-memory:baseanchored-candidate",
            mquake_path=mquake_path,
            wikidata_dir=wikidata_dir,
            out_path=repair_dir / "candidate_official_eval.json",
            args=args,
            records=preselection_records,
        )
        if selected_local_report is None:
            raise RuntimeError("selected rank lacks an exact BF16 local audit")
        gate = final_acceptance_report(
            base_result,
            candidate_result,
            selected_local_report,
            utility_drop_tolerance=args.utility_drop_tolerance,
            max_ppl_ratio=args.max_ppl_ratio,
        )
    else:
        restore_full_output_to_base(model, base_weight_cpu)
        candidate_result = copy.deepcopy(base_result)
        selected_local_report = {
            "passed": False,
            "protected_incremental_regressions": 0,
            "reason": "no_rank_reached_exact_configured_feasibility",
        }
        gate = {
            "accepted": False,
            "checks": {"exact_BF16_local_feasibility": False},
            "thresholds": {
                "forget_Eff": 0.0,
                "retain_Eff_minimum": float(base_result["retain"]["Eff"]) - 0.10,
                "PPL_maximum": float(base_result["forget_PPL"]) * 1.02,
            },
        }

    accepted = bool(gate["accepted"])
    if accepted:
        selected_result = copy.deepcopy(candidate_result)
        reason = "first_rank_reached_exact_BF16_and_official_configured_feasibility"
    else:
        restore_full_output_to_base(model, base_weight_cpu)
        selected_result = copy.deepcopy(base_result)
        reason = "candidate_rejected_and_safe_Base_checkpoint_restored"
    selection_commit = {
        "selection_irrevocable": True,
        "candidate_accepted": accepted,
        "first_local_exact_feasible_rank": selected_rank,
        "selected_first_feasible_rank": selected_rank if accepted else None,
        "selection_reason": reason,
        "held_out_fields_opened": False,
        "atomic_gen_used_for_selection": False,
        "multihop_used_for_selection": False,
        "gates": gate,
    }
    gagd.write_json(output_dir / "selection_commit.json", selection_commit)
    if args.save_selected_checkpoint:
        save_detied_checkpoint(model, tok, output_dir / "selected_checkpoint")

    final_row_id_set = set(row_ids)
    repair_report = {
        "method": METHOD_LABEL,
        "base_relative_parameterization": True,
        "Base_input_embeddings_restored": True,
        "transformer_frozen": True,
        "input_embeddings_frozen": True,
        "initial_active_states": len(constraints_from_audit(
            detied_forget_states, margin=args.active_logit_margin, generation_round=0
        )),
        "final_constraint_count": len(constraints),
        "constraint_generation_rounds": constraint_rounds,
        "selected_output_row_count": len(row_ids),
        "rank_start": args.repair_rank_start,
        "attempted_ranks": sorted({int(row["rank"]) for row in rank_reports}),
        "first_local_exact_feasible_rank": selected_rank,
        "selected_first_feasible_rank": selected_rank if accepted else None,
        "ordered_basis_hash": basis_report["ordered_basis_sha256"],
        "prefix_nesting_verified": basis_report["prefix_nesting_verified"],
        "protected_state_count": len(protected_states),
        "protected_pair_count": (
            len(protected_states) * (len(row_ids) + 1)
            - sum(
                state.target_token_id in final_row_id_set
                for state in protected_states
            )
        ),
        "protected_violations_before": 0,
        "protected_violations_after": int(
            selected_local_report.get("protected_incremental_regressions", 0)
        ),
        "PPL_feasibility": {
            "Base_mean_NLL": ppl_cache.base_mean_nll,
            "allowed_mean_NLL": allowed_mean_nll,
            "direct_BF16_Base_mean_NLL": math.log(
                float(base_result["forget_PPL"])
            ),
            "direct_BF16_allowed_mean_NLL": direct_allowed_mean_nll,
            "candidate_cached_mean_NLL": selected_local_report.get(
                "candidate_cached_mean_nll"
            ),
            "candidate_direct_BF16_mean_NLL": selected_local_report.get(
                "candidate_direct_BF16_mean_nll"
            ),
            "candidate_exact_official_PPL": candidate_result.get("forget_PPL"),
            "cached_vs_official_consistency": (
                None
                if selected_local_report.get("candidate_cached_mean_nll") is None
                else abs(
                    math.exp(float(selected_local_report["candidate_cached_mean_nll"]))
                    - float(candidate_result["forget_PPL"])
                )
            ),
        },
        "exact_BF16_audit_performed_before_final_gate": selected_rank is not None,
        "candidate_accepted": accepted,
        "held_out_fields_opened_before_acceptance": False,
    }
    gagd.write_json(repair_dir / "repair_summary.json", repair_report)
    _write_preselection_result(
        output_dir,
        base_result=base_result,
        tied_result=tied_result,
        detied_result=detied_result,
        candidate_result=candidate_result,
        selected_result=selected_result,
        repair_report=repair_report,
        selection=selection_commit,
    )

    if args.fail_if_target_missed and not accepted:
        raise RuntimeError(
            "No de-tied Base-anchored rank reached every fixed gate; held-out evaluation remains unopened"
        )

    # Full held-out records are loaded only inside this post-acceptance guard.
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
        evaluate=lambda full_records: vectorized._evaluate_held_out_after_selection(
            accepted=True,
            args=args,
            model=model,
            tok=tok,
            records=full_records,
            mquake_path=mquake_path,
            wikidata_dir=wikidata_dir,
            split_manifest=split_manifest,
            output_dir=output_dir,
        ),
    )
    final_payload = json.loads((output_dir / "mquake_results.json").read_text(encoding="utf-8"))
    final_payload["held_out"] = {
        "status": "evaluated_after_durable_acceptance",
        "AtomicGen": selected_extension["forget"].get("AtomicGen"),
        "RetainAtomicGen": selected_extension["retain"].get("AtomicGen"),
        "standard_multihop": multihop_result["results"].get("standard"),
        "cot_multihop": multihop_result["results"].get("cot"),
    }
    final_payload["selection"]["held_out_fields_opened"] = True
    gagd.write_json(output_dir / "mquake_results.json", final_payload)
    if args.require_atomic_gen_zero:
        atomic_gen = selected_extension["forget"].get("AtomicGen")
        if atomic_gen is None or float(atomic_gen) > 0:
            raise RuntimeError(
                "Post-selection AtomicGen was not zero; it did not alter selection"
            )


if __name__ == "__main__":
    main()
